"""Plan and run exact completion checks for Qwen SFT adapters."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import gmtime, monotonic, strftime
from typing import Any, Iterable

from .qwen_sft_training import DEFAULT_MODEL_ID, QWEN_SFT_TRAINING_JOB_SCHEMA
from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS


QWEN_SFT_COMPLETION_JOB_SCHEMA = "flatdisk.qwen_sft_completion_job.v1"
QWEN_SFT_COMPLETION_RESULT_SCHEMA = "flatdisk.qwen_sft_completion_result.v1"
QWEN_SFT_COMPLETION_SAMPLE_SCHEMA = "flatdisk.qwen_sft_completion_sample.v1"
DEFAULT_REQUIRED_PACKAGES = [
    "peft",
    "pillow",
    "torch",
    "torchvision",
    "transformers",
]


def plan_qwen_sft_completion_check(
    sft_training_job_input: Path,
    *,
    output_dir: Path,
    adapter_path: Path | None = None,
    model_id: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 192,
    require_existing_images: bool = True,
    require_existing_adapter: bool = True,
) -> dict[str, Any]:
    sft_job_path = _resolve_sft_training_job(sft_training_job_input)
    sft_job = json.loads(sft_job_path.read_text(encoding="utf-8"))
    source_dataset = _job_path(sft_job, "qwen_sft_training_jsonl", relative_to=sft_job_path.parent)
    source_records = _read_jsonl_if_exists(source_dataset) if source_dataset else []
    selected_records = source_records[:max_samples] if max_samples is not None else list(source_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = adapter_path or _job_path(sft_job, "adapter_output_dir", relative_to=sft_job_path.parent)
    model_id = model_id or str(sft_job.get("model_id") or DEFAULT_MODEL_ID)
    dataset_path = output_dir / "qwen_sft_completion_dataset.jsonl"
    completion_script_path = output_dir / "generate_qwen_sft_completions.py"
    completion_log_path = output_dir / "qwen_sft_completion_samples.jsonl"
    progress_log_path = output_dir / "qwen_sft_completion_progress.jsonl"
    job_path = output_dir / "qwen_sft_completion_job.json"
    _write_jsonl(dataset_path, selected_records)
    validation = _validate_completion_records(
        selected_records,
        sft_job=sft_job,
        sft_job_path=sft_job_path,
        source_dataset=source_dataset,
        adapter_path=adapter_path,
        require_existing_images=require_existing_images,
        require_existing_adapter=require_existing_adapter,
        selected_from_count=len(source_records),
    )
    launch_argv = _launch_argv(
        completion_script_path=completion_script_path,
        dataset_path=dataset_path,
        model_id=model_id,
        adapter_path=adapter_path or Path(""),
        output_dir=output_dir,
        completion_log_path=completion_log_path,
        progress_log_path=progress_log_path,
        max_new_tokens=max_new_tokens,
    )
    _write_completion_script(completion_script_path)
    job = {
        "schema": QWEN_SFT_COMPLETION_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "sft_training_job": str(sft_job_path),
        "source_qwen_sft_training_jsonl": str(source_dataset) if source_dataset else "",
        "qwen_sft_completion_jsonl": str(dataset_path),
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_path) if adapter_path else "",
        "completion_script": str(completion_script_path),
        "completion_script_sha256": _sha256_file(completion_script_path),
        "completion_log_jsonl": str(completion_log_path),
        "progress_log_jsonl": str(progress_log_path),
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "model_id": model_id,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
        "runtime": {
            "python_entrypoint": str(completion_script_path),
            "launcher": "python",
            "dependency_check": "importlib.util.find_spec without importing GPU libraries",
            "required_packages": DEFAULT_REQUIRED_PACKAGES,
        },
        "generation_args": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
        "dataset": validation["dataset"],
        "audit": {
            "policy_input_only": True,
            "evaluator_labels_excluded": True,
            "deterministic_generation": True,
            "adapter_compared_to_base_disabled": True,
            "target_source": "qwen_sft_training_jsonl assistant content",
            "require_existing_images": require_existing_images,
            "require_existing_adapter": require_existing_adapter,
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job


def run_qwen_sft_completion_check(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_sft_completion_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_sft_completion_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    command = _argv_to_command(launch_argv)
    blockers = _completion_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")

    result: dict[str, Any] = {
        "schema": QWEN_SFT_COMPLETION_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "sample_count": (job.get("dataset") or {}).get("sample_count"),
        "adapter_path": job.get("adapter_path"),
        "generation_args": job.get("generation_args"),
        "launch_command": command,
        "launch_argv": launch_argv,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "duration_s": None,
        "blockers": blockers,
        "dependency_check": _dependency_check_payload(job, enabled=check_dependencies),
    }
    _attach_logs_and_metrics(result, job, job_path=job_path)
    if blockers:
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _write_result(result_path, result)
        return result
    if dry_run:
        result["status"] = "dry_run"
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _write_result(result_path, result)
        return result

    start = monotonic()
    try:
        completed = subprocess.run(
            launch_argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        result["returncode"] = completed.returncode
        result["stdout_tail"] = _tail(completed.stdout, tail_chars)
        result["stderr_tail"] = _tail(completed.stderr, tail_chars)
        result["status"] = "complete" if completed.returncode == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        result["status"] = "failed"
        result["blockers"] = [f"completion command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_logs_and_metrics(result, job, job_path=job_path)
    _write_result(result_path, result)
    return result


def _resolve_sft_training_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_sft_training_job.json":
        return path
    candidates = [
        path / "qwen_sft_training_job.json",
        path / "qwen_sft_training" / "qwen_sft_training_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_sft_training_job.json under {job_input}")


def _resolve_sft_completion_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_sft_completion_job.json":
        return path
    candidates = [
        path / "qwen_sft_completion_job.json",
        path / "qwen_sft_completion" / "qwen_sft_completion_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_sft_completion_job.json under {job_input}")


def _validate_completion_records(
    records: list[dict[str, Any]],
    *,
    sft_job: dict[str, Any],
    sft_job_path: Path,
    source_dataset: Path | None,
    adapter_path: Path | None,
    require_existing_images: bool,
    require_existing_adapter: bool,
    selected_from_count: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if sft_job.get("schema") != QWEN_SFT_TRAINING_JOB_SCHEMA:
        blockers.append(f"unexpected SFT training job schema: {sft_job.get('schema')}")
    if source_dataset is None or not source_dataset.exists():
        blockers.append(f"missing qwen_sft_training_jsonl: {sft_job.get('qwen_sft_training_jsonl')}")
    if not records:
        blockers.append("qwen_sft_completion_jsonl contains no SFT records")
    if adapter_path is None or not str(adapter_path):
        blockers.append("missing adapter_path")
    elif require_existing_adapter and not adapter_path.exists():
        blockers.append(f"missing adapter_path: {adapter_path}")
    elif require_existing_adapter and not (adapter_path / "adapter_config.json").exists():
        blockers.append(f"adapter_path lacks adapter_config.json: {adapter_path}")

    missing_images: list[str] = []
    forbidden_hits: Counter[str] = Counter()
    malformed_count = 0
    image_reference_count = 0
    target_tool_counts: Counter[str] = Counter()
    for record in records:
        messages = record.get("messages")
        target = _assistant_target(messages)
        if not isinstance(messages, list) or not target:
            malformed_count += 1
        tool = _target_action_tool(target)
        if tool:
            target_tool_counts[tool] += 1
        image_paths = _record_image_paths(record)
        image_reference_count += len(image_paths)
        missing_images.extend(str(path) for path in image_paths if not Path(path).exists())
        for token in _forbidden_tokens([messages]):
            forbidden_hits[token] += 1
    if malformed_count:
        blockers.append(f"{malformed_count} SFT completion record(s) are missing messages or assistant target")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} SFT image reference(s) are missing")
    if forbidden_hits:
        blockers.append("SFT completion messages contain forbidden privileged token(s): " + ", ".join(sorted(forbidden_hits)))
    if image_reference_count == 0:
        warnings.append("SFT completion records contain no image references")
    if sft_job_path and not sft_job_path.exists():
        blockers.append(f"missing SFT training job: {sft_job_path}")

    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "sample_count": len(records),
            "source_sample_count": selected_from_count,
            "image_reference_count": image_reference_count,
            "missing_image_count": len(missing_images),
            "missing_images": sorted(set(missing_images)),
            "forbidden_model_token_hits": dict(sorted(forbidden_hits.items())),
            "target_action_tool_counts": dict(sorted(target_tool_counts.items())),
        },
    }


def _record_image_paths(record: dict[str, Any]) -> list[str]:
    values = record.get("image_paths")
    if not isinstance(values, list):
        values = record.get("images")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _assistant_target(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _target_action_tool(target: str) -> str | None:
    payload, _parse_status = _parse_jsonish(target)
    action, _action_status = _extract_action_from_text(target, payload)
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    return str(tool) if tool is not None else None


def _parse_jsonish(text: str) -> tuple[Any, str]:
    cleaned = _strip_code_fence(str(text or "").strip())
    if not cleaned:
        return None, "empty"
    try:
        return json.loads(cleaned), "direct"
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1]), "substring"
        except json.JSONDecodeError:
            return None, "malformed"
    return None, "malformed"


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_action(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    return action if isinstance(action, dict) else None


def _extract_action_from_text(text: str, payload: Any) -> tuple[dict[str, Any] | None, str]:
    action = _extract_action(payload)
    if action is not None:
        return action, "payload"
    action_object, status = _extract_balanced_object_after_key(text, "action")
    return action_object, status


def _extract_balanced_object_after_key(text: str, key: str) -> tuple[dict[str, Any] | None, str]:
    cleaned = _strip_code_fence(str(text or ""))
    key_index = cleaned.find(f'"{key}"')
    if key_index < 0:
        return None, "missing_action_key"
    colon_index = cleaned.find(":", key_index + len(key) + 2)
    if colon_index < 0:
        return None, "missing_action_colon"
    start = cleaned.find("{", colon_index + 1)
    if start < 0:
        return None, "missing_action_object"
    end = _balanced_object_end(cleaned, start)
    if end is None:
        return None, "truncated_action_object"
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None, "malformed_action_object"
    return payload if isinstance(payload, dict) else None, "balanced_action_object"


def _balanced_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _forbidden_tokens(payloads: Iterable[Any]) -> list[str]:
    text = json.dumps(list(payloads), sort_keys=True, default=str).lower()
    return [token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text]


def _completion_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_SFT_COMPLETION_JOB_SCHEMA:
        blockers.append(f"unexpected job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"completion job is not ready: {job.get('status')}")
    completion_script = _job_path(job, "completion_script", relative_to=job_path.parent)
    if completion_script is None or not completion_script.exists():
        blockers.append(f"missing completion_script: {job.get('completion_script')}")
    dataset_path = _job_path(job, "qwen_sft_completion_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(f"missing qwen_sft_completion_jsonl: {job.get('qwen_sft_completion_jsonl')}")
    adapter_path = _job_path(job, "adapter_path", relative_to=job_path.parent)
    if adapter_path is None or not adapter_path.exists():
        blockers.append(f"missing adapter_path: {job.get('adapter_path')}")
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required completion package(s): " + ", ".join(missing_packages))
    return blockers


def _job_path(job: dict[str, Any], key: str, *, relative_to: Path) -> Path | None:
    value = job.get(key)
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else relative_to / path if (relative_to / path).exists() else path


def _dependency_check_payload(job: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    required = [str(package) for package in job.get("required_packages", [])]
    missing = _missing_required_packages(job) if enabled else []
    return {
        "enabled": enabled,
        "required_packages": required,
        "missing_packages": missing,
    }


def _missing_required_packages(job: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for package in [str(value) for value in job.get("required_packages", [])]:
        if importlib.util.find_spec(_import_module_for_package(package)) is None:
            missing.append(package)
    return missing


def _import_module_for_package(package: str) -> str:
    return {"pillow": "PIL"}.get(package.lower(), package.replace("-", "_"))


def _read_jsonl_if_exists(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _launch_argv(
    *,
    completion_script_path: Path,
    dataset_path: Path,
    model_id: str,
    adapter_path: Path,
    output_dir: Path,
    completion_log_path: Path,
    progress_log_path: Path,
    max_new_tokens: int,
) -> list[str]:
    return [
        "python",
        str(completion_script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-dir",
        str(output_dir),
        "--completion-log",
        str(completion_log_path),
        "--progress-log",
        str(progress_log_path),
        "--max-new-tokens",
        str(max_new_tokens),
    ]


def _argv_to_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _job_launch_argv(job: dict[str, Any], *, launch_command_override: str | None) -> list[str]:
    if launch_command_override:
        return shlex.split(launch_command_override)
    launch_argv = job.get("launch_argv")
    if isinstance(launch_argv, list) and all(isinstance(part, str) and part for part in launch_argv):
        return [str(part) for part in launch_argv]
    launch_command = str(job.get("launch_command") or "")
    return shlex.split(launch_command) if launch_command else []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_completion_script(path: Path) -> None:
    path.write_text(_COMPLETION_SCRIPT, encoding="utf-8")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail(text: str, chars: int) -> str:
    return text[-chars:] if chars > 0 and len(text) > chars else text


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _jsonl_tail(path: Path | None, limit: int) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"malformed": line})
    return rows[-limit:]


def _read_jsonl_with_malformed_count(path: Path | None) -> tuple[list[dict[str, Any]], int]:
    if path is None or not path.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    malformed_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed_count += 1
    return rows, malformed_count


def _jsonl_valid_count(path: Path | None) -> int:
    rows, _malformed_count = _read_jsonl_with_malformed_count(path)
    return len(rows)


def _completion_metrics(path: Path | None) -> dict[str, Any]:
    rows, malformed_line_count = _read_jsonl_with_malformed_count(path)
    valid_rows = [row for row in rows if row.get("schema") == QWEN_SFT_COMPLETION_SAMPLE_SCHEMA]
    malformed_line_count += len(rows) - len(valid_rows)
    by_target_tool: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        tool = str(row.get("target_action_tool") or "unknown")
        by_target_tool.setdefault(tool, []).append(row)

    base_tool_matches = [bool(row.get("base_tool_match")) for row in valid_rows]
    adapter_tool_matches = [bool(row.get("adapter_tool_match")) for row in valid_rows]
    base_action_matches = [bool(row.get("base_exact_action_match")) for row in valid_rows]
    adapter_action_matches = [bool(row.get("adapter_exact_action_match")) for row in valid_rows]
    base_text_matches = [bool(row.get("base_exact_target_text_match")) for row in valid_rows]
    adapter_text_matches = [bool(row.get("adapter_exact_target_text_match")) for row in valid_rows]
    base_json_matches = [bool(row.get("base_parsed_json_match")) for row in valid_rows]
    adapter_json_matches = [bool(row.get("adapter_parsed_json_match")) for row in valid_rows]
    base_parse_count = sum(1 for row in valid_rows if row.get("base_parsed_json"))
    adapter_parse_count = sum(1 for row in valid_rows if row.get("adapter_parsed_json"))
    base_action_count = sum(1 for row in valid_rows if isinstance(row.get("base_parsed_action"), dict))
    adapter_action_count = sum(1 for row in valid_rows if isinstance(row.get("adapter_parsed_action"), dict))
    base_completion_lengths = [float(row.get("base_completion_char_count") or 0) for row in valid_rows]
    adapter_completion_lengths = [float(row.get("adapter_completion_char_count") or 0) for row in valid_rows]
    return {
        "sample_count": len(valid_rows),
        "malformed_line_count": malformed_line_count,
        "target_action_tool_counts": {tool: len(tool_rows) for tool, tool_rows in sorted(by_target_tool.items())},
        "base_action_tool_counts": _value_counts(row.get("base_action_tool") for row in valid_rows),
        "adapter_action_tool_counts": _value_counts(row.get("adapter_action_tool") for row in valid_rows),
        "base_parsed_json_count": base_parse_count,
        "adapter_parsed_json_count": adapter_parse_count,
        "base_parsed_json_rate": _rate(base_parse_count, len(valid_rows)),
        "adapter_parsed_json_rate": _rate(adapter_parse_count, len(valid_rows)),
        "base_parsed_action_count": base_action_count,
        "adapter_parsed_action_count": adapter_action_count,
        "base_parsed_action_rate": _rate(base_action_count, len(valid_rows)),
        "adapter_parsed_action_rate": _rate(adapter_action_count, len(valid_rows)),
        "base_tool_match_count": sum(base_tool_matches),
        "adapter_tool_match_count": sum(adapter_tool_matches),
        "base_tool_match_rate": _rate(sum(base_tool_matches), len(valid_rows)),
        "adapter_tool_match_rate": _rate(sum(adapter_tool_matches), len(valid_rows)),
        "base_exact_action_match_count": sum(base_action_matches),
        "adapter_exact_action_match_count": sum(adapter_action_matches),
        "base_exact_action_match_rate": _rate(sum(base_action_matches), len(valid_rows)),
        "adapter_exact_action_match_rate": _rate(sum(adapter_action_matches), len(valid_rows)),
        "base_exact_target_text_match_count": sum(base_text_matches),
        "adapter_exact_target_text_match_count": sum(adapter_text_matches),
        "base_exact_target_text_match_rate": _rate(sum(base_text_matches), len(valid_rows)),
        "adapter_exact_target_text_match_rate": _rate(sum(adapter_text_matches), len(valid_rows)),
        "base_parsed_json_match_count": sum(base_json_matches),
        "adapter_parsed_json_match_count": sum(adapter_json_matches),
        "base_parsed_json_match_rate": _rate(sum(base_json_matches), len(valid_rows)),
        "adapter_parsed_json_match_rate": _rate(sum(adapter_json_matches), len(valid_rows)),
        "adapter_improved_tool_match_count": sum(
            1 for row in valid_rows if row.get("adapter_tool_match") and not row.get("base_tool_match")
        ),
        "adapter_regressed_tool_match_count": sum(
            1 for row in valid_rows if row.get("base_tool_match") and not row.get("adapter_tool_match")
        ),
        "adapter_changed_tool_count": sum(
            1 for row in valid_rows if row.get("adapter_action_tool") != row.get("base_action_tool")
        ),
        "base_completion_char_count_mean": _mean(base_completion_lengths),
        "adapter_completion_char_count_mean": _mean(adapter_completion_lengths),
        "fenced_completion_count": sum(
            1
            for row in valid_rows
            if row.get("base_completion_was_fenced") or row.get("adapter_completion_was_fenced")
        ),
    }


def _value_counts(values: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        if value is None:
            counts["null"] += 1
        else:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 9) if values else None


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 6) if total else None


def _attach_logs_and_metrics(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    completion_log = _job_path(job, "completion_log_jsonl", relative_to=job_path.parent)
    progress_log = _job_path(job, "progress_log_jsonl", relative_to=job_path.parent)
    result["completion_log_jsonl"] = str(completion_log) if completion_log else ""
    result["completion_log_sample_count"] = _jsonl_valid_count(completion_log)
    result["completion_log_metrics"] = _completion_metrics(completion_log) if completion_log and completion_log.exists() else {}
    result["progress_log_jsonl"] = str(progress_log) if progress_log else ""
    result["progress_log_count"] = _jsonl_valid_count(progress_log)
    result["progress_log_tail"] = _jsonl_tail(progress_log, 8)


_COMPLETION_SCRIPT = '''"""Generate base-disabled and adapter-enabled Qwen SFT completions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import gmtime, strftime

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def split_messages(record: dict) -> tuple[list[dict], str]:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    assistant_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant_index = index
            break
    if assistant_index is None:
        return messages, ""
    return messages[:assistant_index], str(messages[assistant_index].get("content") or "")


def record_image_paths(record: dict) -> list[str]:
    values = record.get("image_paths")
    if not isinstance(values, list):
        values = record.get("images")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def strip_code_fence(text: str) -> tuple[str, bool]:
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped, False
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\\n".join(lines).strip(), True


def parse_jsonish(text: str) -> tuple[object | None, str, bool]:
    cleaned, fenced = strip_code_fence(text)
    if not cleaned:
        return None, "empty", fenced
    try:
        return json.loads(cleaned), "direct", fenced
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1]), "substring", fenced
        except json.JSONDecodeError:
            return None, "malformed", fenced
    return None, "malformed", fenced


def extract_action(payload: object | None) -> dict | None:
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    return action if isinstance(action, dict) else None


def balanced_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_balanced_object_after_key(text: str, key: str) -> tuple[dict | None, str]:
    cleaned, _fenced = strip_code_fence(text)
    key_index = cleaned.find(f'"{key}"')
    if key_index < 0:
        return None, "missing_action_key"
    colon_index = cleaned.find(":", key_index + len(key) + 2)
    if colon_index < 0:
        return None, "missing_action_colon"
    start = cleaned.find("{", colon_index + 1)
    if start < 0:
        return None, "missing_action_object"
    end = balanced_object_end(cleaned, start)
    if end is None:
        return None, "truncated_action_object"
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None, "malformed_action_object"
    return payload if isinstance(payload, dict) else None, "balanced_action_object"


def extract_action_from_text(text: str, payload: object | None) -> tuple[dict | None, str]:
    action = extract_action(payload)
    if action is not None:
        return action, "payload"
    return extract_balanced_object_after_key(text, "action")


def action_tool(action: dict | None) -> str | None:
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    return str(tool) if tool is not None else None


def adapter_metadata(model, adapter_path: Path) -> dict:
    peft_config = getattr(model, "peft_config", {})
    active_adapter = getattr(model, "active_adapter", None)
    if callable(active_adapter):
        try:
            active_adapter = active_adapter()
        except TypeError:
            active_adapter = str(active_adapter)
    adapter_param_count = 0
    for name, parameter in model.named_parameters():
        lower_name = name.lower()
        if "lora" in lower_name or "adapter" in lower_name:
            adapter_param_count += int(parameter.numel())
    return {
        "adapter_path": str(adapter_path),
        "active_adapter": str(active_adapter) if active_adapter is not None else "",
        "peft_adapter_names": sorted(str(key) for key in peft_config.keys()) if isinstance(peft_config, dict) else [],
        "adapter_parameter_count": adapter_param_count,
    }


def prepare_generation_inputs(processor, model, record: dict) -> tuple[dict, dict, str]:
    prompt_messages, target_text = split_messages(record)
    image_paths = record_image_paths(record)
    images = [load_image(path) for path in image_paths]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [prompt_text], "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    input_ids = inputs.get("input_ids")
    prompt_token_count = int(input_ids.shape[1]) if input_ids is not None else 0
    if hasattr(inputs, "to"):
        inputs = inputs.to(model_device(model))
    return inputs, {
        "image_count": len(images),
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "target_text_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
        "prompt_token_count": prompt_token_count,
    }, target_text


def generate_completion(processor, model, inputs: dict, *, max_new_tokens: int) -> tuple[str, int]:
    prompt_len = int(inputs["input_ids"].shape[1])
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=False,
    )
    new_ids = output_ids[0, prompt_len:]
    text = processor.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0])


def score_record(processor, model, record: dict, metadata: dict, *, max_new_tokens: int) -> dict:
    inputs, input_meta, target_text = prepare_generation_inputs(processor, model, record)
    target_payload, target_parse_status, target_fenced = parse_jsonish(target_text)
    target_action, target_action_parse_status = extract_action_from_text(target_text, target_payload)
    target_tool = action_tool(target_action)
    with torch.inference_mode():
        with model.disable_adapter():
            base_completion, base_token_count = generate_completion(
                processor, model, inputs, max_new_tokens=max_new_tokens
            )
        adapter_completion, adapter_token_count = generate_completion(
            processor, model, inputs, max_new_tokens=max_new_tokens
        )
    base_payload, base_parse_status, base_fenced = parse_jsonish(base_completion)
    adapter_payload, adapter_parse_status, adapter_fenced = parse_jsonish(adapter_completion)
    base_action, base_action_parse_status = extract_action_from_text(base_completion, base_payload)
    adapter_action, adapter_action_parse_status = extract_action_from_text(adapter_completion, adapter_payload)
    base_tool = action_tool(base_action)
    adapter_tool = action_tool(adapter_action)
    return {
        "schema": "flatdisk.qwen_sft_completion_sample.v1",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "sample_id": record.get("sample_id"),
        "source_policy_step_id": record.get("source_policy_step_id"),
        "target_text": target_text,
        "target_parse_status": target_parse_status,
        "target_action_parse_status": target_action_parse_status,
        "target_was_fenced": target_fenced,
        "target_parsed_json": target_payload,
        "target_parsed_action": target_action,
        "target_action_tool": target_tool,
        "base_completion": base_completion,
        "adapter_completion": adapter_completion,
        "base_completion_token_count": base_token_count,
        "adapter_completion_token_count": adapter_token_count,
        "base_completion_char_count": len(base_completion),
        "adapter_completion_char_count": len(adapter_completion),
        "base_completion_parse_status": base_parse_status,
        "adapter_completion_parse_status": adapter_parse_status,
        "base_action_parse_status": base_action_parse_status,
        "adapter_action_parse_status": adapter_action_parse_status,
        "base_completion_was_fenced": base_fenced,
        "adapter_completion_was_fenced": adapter_fenced,
        "base_parsed_json": base_payload,
        "adapter_parsed_json": adapter_payload,
        "base_parsed_action": base_action,
        "adapter_parsed_action": adapter_action,
        "base_action_tool": base_tool,
        "adapter_action_tool": adapter_tool,
        "base_exact_target_text_match": base_completion.strip() == target_text.strip(),
        "adapter_exact_target_text_match": adapter_completion.strip() == target_text.strip(),
        "base_parsed_json_match": base_payload == target_payload and target_payload is not None,
        "adapter_parsed_json_match": adapter_payload == target_payload and target_payload is not None,
        "base_exact_action_match": base_action == target_action and target_action is not None,
        "adapter_exact_action_match": adapter_action == target_action and target_action is not None,
        "base_tool_match": base_tool == target_tool and target_tool is not None,
        "adapter_tool_match": adapter_tool == target_tool and target_tool is not None,
        "adapter_changed_tool": adapter_tool != base_tool,
        "adapter_metadata": metadata,
        **input_meta,
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--completion-log", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.completion_log.write_text("", encoding="utf-8")
    args.progress_log.write_text("", encoding="utf-8")
    append_jsonl(args.progress_log, {
        "stage": "start",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "dataset": str(args.dataset),
        "model_id": args.model_id,
        "adapter_path": str(args.adapter_path),
        "max_new_tokens": args.max_new_tokens,
    })
    processor = AutoProcessor.from_pretrained(args.model_id)
    append_jsonl(args.progress_log, {"stage": "load_model_start", "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())})
    base_model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    metadata = adapter_metadata(model, args.adapter_path)
    append_jsonl(args.progress_log, {"stage": "load_model_complete", "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()), "adapter_metadata": metadata})
    records = read_jsonl(args.dataset)
    append_jsonl(args.progress_log, {"stage": "read_dataset_complete", "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()), "sample_count": len(records)})
    for index, record in enumerate(records):
        append_jsonl(args.progress_log, {
            "stage": "sample_start",
            "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
            "sample_index": index,
            "sample_id": record.get("sample_id"),
        })
        row = score_record(processor, model, record, metadata, max_new_tokens=args.max_new_tokens)
        append_jsonl(args.completion_log, row)
        append_jsonl(args.progress_log, {
            "stage": "sample_complete",
            "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
            "sample_index": index,
            "sample_id": record.get("sample_id"),
            "target_action_tool": row["target_action_tool"],
            "base_action_tool": row["base_action_tool"],
            "adapter_action_tool": row["adapter_action_tool"],
            "base_tool_match": row["base_tool_match"],
            "adapter_tool_match": row["adapter_tool_match"],
        })
    append_jsonl(args.progress_log, {"stage": "complete", "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()), "sample_count": len(records)})
    print(json.dumps({"status": "complete", "sample_count": len(records), "completion_log": str(args.completion_log)}))


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-training-job", type=Path, required=True, help="qwen_sft_training dir or qwen_sft_training_job.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--allow-missing-adapter", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen SFT completion check.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_sft_completion dir or qwen_sft_completion_job.json")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch_command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job = plan_qwen_sft_completion_check(
        args.sft_training_job,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        require_existing_images=not args.allow_missing_images,
        require_existing_adapter=not args.allow_missing_adapter,
    )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["sample_count"],
                "target_action_tool_counts": job["dataset"]["target_action_tool_counts"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "job_path": str(Path(job["output_dir"]) / "qwen_sft_completion_job.json"),
                "completion_script": job["completion_script"],
                "launch_command": job["launch_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and job["status"] != "ready":
        return 2
    return 0


def run_main() -> int:
    args = parse_run_args()
    result = run_qwen_sft_completion_check(
        args.job,
        result_dir=args.result_dir,
        dry_run=args.dry_run,
        check_dependencies=not args.skip_dependency_check,
        timeout_s=args.timeout_s,
        launch_command=args.launch_command,
        tail_chars=args.tail_chars,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "returncode": result["returncode"],
                "blockers": result["blockers"],
                "result_path": result["result_path"],
                "launch_command": result["launch_command"],
                "completion_log_metrics": result["completion_log_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["status"] in {"complete", "dry_run"}:
        return 0
    if result["status"] == "not_ready":
        return 2
    return 1


def module_main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"plan", "run"}:
        command = sys.argv.pop(1)
        return run_main() if command == "run" else main()
    return main()


if __name__ == "__main__":
    raise SystemExit(module_main())
