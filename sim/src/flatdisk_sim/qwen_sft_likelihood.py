"""Plan and run teacher-forced likelihood checks for Qwen SFT adapters."""

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


QWEN_SFT_LIKELIHOOD_JOB_SCHEMA = "flatdisk.qwen_sft_likelihood_job.v1"
QWEN_SFT_LIKELIHOOD_RESULT_SCHEMA = "flatdisk.qwen_sft_likelihood_result.v1"
QWEN_SFT_LIKELIHOOD_SAMPLE_SCHEMA = "flatdisk.qwen_sft_likelihood_sample.v1"
DEFAULT_REQUIRED_PACKAGES = [
    "peft",
    "pillow",
    "torch",
    "torchvision",
    "transformers",
]


def plan_qwen_sft_likelihood_check(
    sft_training_job_input: Path,
    *,
    output_dir: Path,
    adapter_path: Path | None = None,
    model_id: str | None = None,
    max_samples: int | None = None,
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
    dataset_path = output_dir / "qwen_sft_likelihood_dataset.jsonl"
    score_script_path = output_dir / "score_qwen_sft_likelihood.py"
    likelihood_log_path = output_dir / "qwen_sft_likelihood_samples.jsonl"
    progress_log_path = output_dir / "qwen_sft_likelihood_progress.jsonl"
    job_path = output_dir / "qwen_sft_likelihood_job.json"
    _write_jsonl(dataset_path, selected_records)
    validation = _validate_likelihood_records(
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
        score_script_path=score_script_path,
        dataset_path=dataset_path,
        model_id=model_id,
        adapter_path=adapter_path or Path(""),
        output_dir=output_dir,
        likelihood_log_path=likelihood_log_path,
        progress_log_path=progress_log_path,
    )
    _write_score_script(score_script_path)
    job = {
        "schema": QWEN_SFT_LIKELIHOOD_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "sft_training_job": str(sft_job_path),
        "source_qwen_sft_training_jsonl": str(source_dataset) if source_dataset else "",
        "qwen_sft_likelihood_jsonl": str(dataset_path),
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_path) if adapter_path else "",
        "score_script": str(score_script_path),
        "score_script_sha256": _sha256_file(score_script_path),
        "likelihood_log_jsonl": str(likelihood_log_path),
        "progress_log_jsonl": str(progress_log_path),
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "model_id": model_id,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
        "runtime": {
            "python_entrypoint": str(score_script_path),
            "launcher": "python",
            "dependency_check": "importlib.util.find_spec without importing GPU libraries",
            "required_packages": DEFAULT_REQUIRED_PACKAGES,
        },
        "dataset": validation["dataset"],
        "audit": {
            "policy_input_only": True,
            "evaluator_labels_excluded": True,
            "teacher_forced_assistant_target": True,
            "adapter_compared_to_base_disabled": True,
            "target_source": "qwen_sft_training_jsonl assistant content",
            "require_existing_images": require_existing_images,
            "require_existing_adapter": require_existing_adapter,
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job


def run_qwen_sft_likelihood_check(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_sft_likelihood_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_sft_likelihood_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    command = _argv_to_command(launch_argv)
    blockers = _likelihood_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")

    result: dict[str, Any] = {
        "schema": QWEN_SFT_LIKELIHOOD_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "sample_count": (job.get("dataset") or {}).get("sample_count"),
        "adapter_path": job.get("adapter_path"),
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
        result["blockers"] = [f"likelihood command timed out after {timeout_s} second(s)"]
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


def _resolve_sft_likelihood_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_sft_likelihood_job.json":
        return path
    candidates = [
        path / "qwen_sft_likelihood_job.json",
        path / "qwen_sft_likelihood" / "qwen_sft_likelihood_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_sft_likelihood_job.json under {job_input}")


def _validate_likelihood_records(
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
        blockers.append("qwen_sft_likelihood_jsonl contains no SFT records")
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
        blockers.append(f"{malformed_count} SFT likelihood record(s) are missing messages or assistant target")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} SFT image reference(s) are missing")
    if forbidden_hits:
        blockers.append("SFT likelihood messages contain forbidden privileged token(s): " + ", ".join(sorted(forbidden_hits)))
    if image_reference_count == 0:
        warnings.append("SFT likelihood records contain no image references")
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
    try:
        payload = json.loads(target)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    return str(tool) if tool is not None else None


def _forbidden_tokens(payloads: Iterable[Any]) -> list[str]:
    text = json.dumps(list(payloads), sort_keys=True, default=str).lower()
    return [token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text]


def _likelihood_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_SFT_LIKELIHOOD_JOB_SCHEMA:
        blockers.append(f"unexpected job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"likelihood job is not ready: {job.get('status')}")
    score_script = _job_path(job, "score_script", relative_to=job_path.parent)
    if score_script is None or not score_script.exists():
        blockers.append(f"missing score_script: {job.get('score_script')}")
    dataset_path = _job_path(job, "qwen_sft_likelihood_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(f"missing qwen_sft_likelihood_jsonl: {job.get('qwen_sft_likelihood_jsonl')}")
    adapter_path = _job_path(job, "adapter_path", relative_to=job_path.parent)
    if adapter_path is None or not adapter_path.exists():
        blockers.append(f"missing adapter_path: {job.get('adapter_path')}")
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required likelihood package(s): " + ", ".join(missing_packages))
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
    score_script_path: Path,
    dataset_path: Path,
    model_id: str,
    adapter_path: Path,
    output_dir: Path,
    likelihood_log_path: Path,
    progress_log_path: Path,
) -> list[str]:
    return [
        "python",
        str(score_script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-dir",
        str(output_dir),
        "--likelihood-log",
        str(likelihood_log_path),
        "--progress-log",
        str(progress_log_path),
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


def _write_score_script(path: Path) -> None:
    path.write_text(_SCORE_SCRIPT, encoding="utf-8")


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


def _likelihood_metrics(path: Path | None) -> dict[str, Any]:
    rows, malformed_line_count = _read_jsonl_with_malformed_count(path)
    valid_rows = [row for row in rows if row.get("schema") == QWEN_SFT_LIKELIHOOD_SAMPLE_SCHEMA]
    malformed_line_count += len(rows) - len(valid_rows)
    target_deltas = [float(row["target_mean_logprob_delta"]) for row in valid_rows if row.get("target_mean_logprob_delta") is not None]
    action_deltas = [float(row["action_tool_mean_logprob_delta"]) for row in valid_rows if row.get("action_tool_mean_logprob_delta") is not None]
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        tool = str(row.get("target_action_tool") or "unknown")
        by_tool.setdefault(tool, []).append(row)
    return {
        "sample_count": len(valid_rows),
        "malformed_line_count": malformed_line_count,
        "target_mean_logprob_delta_mean": _mean(target_deltas),
        "target_mean_logprob_delta_min": min(target_deltas) if target_deltas else None,
        "target_mean_logprob_delta_max": max(target_deltas) if target_deltas else None,
        "target_mean_logprob_improved_count": sum(1 for value in target_deltas if value > 0),
        "target_mean_logprob_improved_rate": _rate(sum(1 for value in target_deltas if value > 0), len(target_deltas)),
        "action_tool_mean_logprob_delta_mean": _mean(action_deltas),
        "action_tool_mean_logprob_improved_count": sum(1 for value in action_deltas if value > 0),
        "action_tool_mean_logprob_improved_rate": _rate(sum(1 for value in action_deltas if value > 0), len(action_deltas)),
        "target_action_tool_counts": {tool: len(tool_rows) for tool, tool_rows in sorted(by_tool.items())},
        "target_mean_logprob_delta_by_action_tool": {
            tool: round(_mean([float(row["target_mean_logprob_delta"]) for row in tool_rows]), 9)
            for tool, tool_rows in sorted(by_tool.items())
        },
        "action_tool_mean_logprob_delta_by_action_tool": {
            tool: _mean([
                    float(row["action_tool_mean_logprob_delta"])
                    for row in tool_rows
                    if row.get("action_tool_mean_logprob_delta") is not None
                ])
            for tool, tool_rows in sorted(by_tool.items())
        },
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 9) if values else None


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 6) if total else None


def _attach_logs_and_metrics(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    likelihood_log = _job_path(job, "likelihood_log_jsonl", relative_to=job_path.parent)
    progress_log = _job_path(job, "progress_log_jsonl", relative_to=job_path.parent)
    result["likelihood_log_jsonl"] = str(likelihood_log) if likelihood_log else ""
    result["likelihood_log_sample_count"] = _jsonl_valid_count(likelihood_log)
    result["likelihood_log_metrics"] = _likelihood_metrics(likelihood_log) if likelihood_log and likelihood_log.exists() else {}
    result["progress_log_jsonl"] = str(progress_log) if progress_log else ""
    result["progress_log_count"] = _jsonl_valid_count(progress_log)
    result["progress_log_tail"] = _jsonl_tail(progress_log, 8)


_SCORE_SCRIPT = '''"""Score teacher-forced SFT assistant target likelihood under base-disabled and adapter-enabled Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def target_action_tool(target_text: str) -> str | None:
    try:
        payload = json.loads(target_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    action = payload.get("action")
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    return str(tool) if tool is not None else None


def prepare_teacher_forced_inputs(processor, model, record: dict) -> tuple[dict, dict, str]:
    prompt_messages, target_text = split_messages(record)
    images = [load_image(path) for path in record.get("image_paths", record.get("images", []))]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_kwargs = {"text": [prompt_text], "return_tensors": "pt"}
    full_kwargs = {"text": [prompt_text + target_text], "return_tensors": "pt"}
    if images:
        prompt_kwargs["images"] = images
        full_kwargs["images"] = images
    prompt_inputs = processor(**prompt_kwargs)
    full_inputs = processor(**full_kwargs)
    prompt_input_ids = prompt_inputs.get("input_ids")
    full_input_ids = full_inputs.get("input_ids")
    target_start = int(prompt_input_ids.shape[1]) if prompt_input_ids is not None else 0
    full_token_count = int(full_input_ids.shape[1]) if full_input_ids is not None else 0
    target_token_count = max(0, full_token_count - target_start)
    if hasattr(full_inputs, "to"):
        full_inputs = full_inputs.to(model_device(model))
    return full_inputs, {
        "image_count": len(images),
        "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "target_text_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
        "input_token_count": full_token_count,
        "prompt_token_count": target_start,
        "target_start_token_index": target_start,
        "target_token_count": target_token_count,
    }, target_text


def logprobs_for_target(logits: torch.Tensor, input_ids: torch.Tensor, target_start: int, target_count: int) -> torch.Tensor:
    if target_count <= 0 or target_start <= 0:
        return torch.empty(0, dtype=torch.float32)
    target_ids = input_ids[0, target_start : target_start + target_count].detach().cpu()
    prediction_logits = logits[0, target_start - 1 : target_start + target_count - 1, :].detach().float().cpu()
    log_probs = torch.log_softmax(prediction_logits, dim=-1)
    return log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)


def decode_token(processor, token_id: int) -> str:
    try:
        return processor.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def token_char_spans(pieces: list[str]) -> list[tuple[int, int]]:
    spans = []
    cursor = 0
    for piece in pieces:
        start = cursor
        cursor += len(piece)
        spans.append((start, cursor))
    return spans


def overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def action_tool_token_indices(processor, target_ids: list[int], target_text: str, tool: str | None) -> tuple[list[int], bool]:
    if not tool:
        return [], False
    pieces = [decode_token(processor, token_id) for token_id in target_ids]
    spans = token_char_spans(pieces)
    tool_segment = f'"tool": "{tool}"'
    tool_start = target_text.find(tool_segment)
    tool_end = tool_start + len(tool_segment) if tool_start >= 0 else -1
    if tool_start < 0:
        compact_segment = f'"tool":"{tool}"'
        tool_start = target_text.find(compact_segment)
        tool_end = tool_start + len(compact_segment) if tool_start >= 0 else -1
    if tool_start < 0:
        tool_start = target_text.find(tool)
        tool_end = tool_start + len(tool) if tool_start >= 0 else -1
    if tool_start < 0:
        return [index for index, piece in enumerate(pieces) if tool in piece or piece in tool], False
    indices = [
        index
        for index, (start, end) in enumerate(spans)
        if overlap(start, end, tool_start, tool_end) or tool in pieces[index]
    ]
    if not indices:
        return [index for index, piece in enumerate(pieces) if tool in piece or piece in tool], False
    return indices, True


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_exp(value: float) -> float:
    return math.exp(min(float(value), 50.0))


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


def score_record(processor, model, record: dict, metadata: dict) -> dict:
    full_inputs, input_meta, target_text = prepare_teacher_forced_inputs(processor, model, record)
    input_ids = full_inputs.get("input_ids")
    target_start = int(input_meta["target_start_token_index"])
    target_count = int(input_meta["target_token_count"])
    with torch.inference_mode():
        with model.disable_adapter():
            base_outputs = model(**full_inputs)
        adapter_outputs = model(**full_inputs)
    base_logprobs = logprobs_for_target(base_outputs.logits, input_ids, target_start, target_count)
    adapter_logprobs = logprobs_for_target(adapter_outputs.logits, input_ids, target_start, target_count)
    target_ids = input_ids[0, target_start : target_start + target_count].detach().cpu().tolist() if target_count else []
    tool = target_action_tool(target_text)
    tool_indices, tool_span_found = action_tool_token_indices(processor, target_ids, target_text, tool)
    base_values = [float(value) for value in base_logprobs.tolist()]
    adapter_values = [float(value) for value in adapter_logprobs.tolist()]
    deltas = [adapter - base for base, adapter in zip(base_values, adapter_values, strict=True)]
    tool_base_values = [base_values[index] for index in tool_indices if index < len(base_values)]
    tool_adapter_values = [adapter_values[index] for index in tool_indices if index < len(adapter_values)]
    base_target_mean = mean(base_values)
    adapter_target_mean = mean(adapter_values)
    base_target_nll = -base_target_mean
    adapter_target_nll = -adapter_target_mean
    base_tool_mean = mean(tool_base_values)
    adapter_tool_mean = mean(tool_adapter_values)
    return {
        "schema": "flatdisk.qwen_sft_likelihood_sample.v1",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "sample_id": record.get("sample_id"),
        "source_policy_step_id": record.get("source_policy_step_id"),
        "target_action_tool": tool,
        "target_text_sha256": input_meta["target_text_sha256"],
        "target_token_count": target_count,
        "action_tool_token_count": len(tool_indices),
        "action_tool_span_found": tool_span_found,
        "base_target_logprob_sum": round(sum(base_values), 9),
        "adapter_target_logprob_sum": round(sum(adapter_values), 9),
        "target_logprob_sum_delta": round(sum(adapter_values) - sum(base_values), 9),
        "base_target_mean_logprob": round(base_target_mean, 9),
        "adapter_target_mean_logprob": round(adapter_target_mean, 9),
        "target_mean_logprob_delta": round(adapter_target_mean - base_target_mean, 9),
        "base_target_nll": round(base_target_nll, 9),
        "adapter_target_nll": round(adapter_target_nll, 9),
        "target_nll_delta": round(adapter_target_nll - base_target_nll, 9),
        "base_target_perplexity": round(safe_exp(base_target_nll), 9),
        "adapter_target_perplexity": round(safe_exp(adapter_target_nll), 9),
        "base_action_tool_mean_logprob": round(base_tool_mean, 9) if tool_indices else None,
        "adapter_action_tool_mean_logprob": round(adapter_tool_mean, 9) if tool_indices else None,
        "action_tool_mean_logprob_delta": round(adapter_tool_mean - base_tool_mean, 9) if tool_indices else None,
        "adapter_improved_target_mean_logprob": adapter_target_mean > base_target_mean,
        "adapter_improved_action_tool_mean_logprob": adapter_tool_mean > base_tool_mean if tool_indices else False,
        "target_token_mean_abs_logprob_delta": round(mean([abs(value) for value in deltas]), 9),
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
    parser.add_argument("--likelihood-log", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.likelihood_log.write_text("", encoding="utf-8")
    args.progress_log.write_text("", encoding="utf-8")
    append_jsonl(args.progress_log, {
        "stage": "start",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "dataset": str(args.dataset),
        "model_id": args.model_id,
        "adapter_path": str(args.adapter_path),
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
        row = score_record(processor, model, record, metadata)
        append_jsonl(args.likelihood_log, row)
        append_jsonl(args.progress_log, {
            "stage": "sample_complete",
            "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
            "sample_index": index,
            "sample_id": record.get("sample_id"),
            "target_mean_logprob_delta": row["target_mean_logprob_delta"],
            "action_tool_mean_logprob_delta": row["action_tool_mean_logprob_delta"],
            "target_action_tool": row["target_action_tool"],
        })
    append_jsonl(args.progress_log, {"stage": "complete", "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()), "sample_count": len(records)})
    print(json.dumps({"status": "complete", "sample_count": len(records), "likelihood_log": str(args.likelihood_log)}))


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
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--allow-missing-adapter", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen SFT likelihood check.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_sft_likelihood dir or qwen_sft_likelihood_job.json")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch_command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job = plan_qwen_sft_likelihood_check(
        args.sft_training_job,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        max_samples=args.max_samples,
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
                "job_path": str(Path(job["output_dir"]) / "qwen_sft_likelihood_job.json"),
                "score_script": job["score_script"],
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
    result = run_qwen_sft_likelihood_check(
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
                "likelihood_log_metrics": result["likelihood_log_metrics"],
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
