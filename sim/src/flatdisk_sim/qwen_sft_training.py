"""Plan and run tiny Qwen VLM SFT jobs from materialized navigation actions."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import gmtime, monotonic, strftime
from typing import Any, Iterable

from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS


QWEN_SFT_TRAINING_JOB_SCHEMA = "flatdisk.qwen_sft_training_job.v1"
QWEN_SFT_TRAINING_RESULT_SCHEMA = "flatdisk.qwen_sft_training_result.v1"
QWEN_SFT_SAMPLE_SCHEMA = "flatdisk.qwen_tool_sft_sample.v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
TARGET_MODE_ASSISTANT = "assistant"
TARGET_MODE_ACTION_ONLY = "action-only"
TARGET_MODES = (TARGET_MODE_ASSISTANT, TARGET_MODE_ACTION_ONLY)
DEFAULT_REQUIRED_PACKAGES = [
    "accelerate",
    "peft",
    "pillow",
    "torch",
    "torchvision",
    "transformers",
]


def plan_qwen_sft_training(
    input_path: Path,
    *,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_output_dir: Path | None = None,
    max_samples: int | None = None,
    max_steps: int = 20,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_mode: str = TARGET_MODE_ASSISTANT,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}: {target_mode}")
    qwen_manifest = _resolve_qwen_training_manifest(input_path)
    manifest = json.loads(qwen_manifest.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(qwen_manifest)
    source_sft_path = _resolve_manifest_path(
        manifest,
        "qwen_sft_messages_jsonl",
        default=qwen_manifest.parent / "qwen_sft_messages.jsonl",
    )
    source_records = _read_jsonl_if_exists(source_sft_path)
    selected_source_records = source_records[:max_samples] if max_samples is not None else list(source_records)
    selected_records = _records_for_target_mode(selected_source_records, target_mode=target_mode)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = adapter_output_dir or output_dir / "adapter"
    train_dataset_path = output_dir / "qwen_sft_training_dataset.jsonl"
    train_script_path = output_dir / "train_qwen_sft_lora.py"
    training_log_path = output_dir / "qwen_sft_training_log.jsonl"
    job_path = output_dir / "qwen_sft_training_job.json"
    _write_jsonl(train_dataset_path, selected_records)
    validation = _validate_sft_records(
        selected_records,
        require_existing_images=require_existing_images,
        expected_source_count=_optional_int(manifest.get("accepted_count")),
        source_sft_path=source_sft_path,
        selected_from_count=len(source_records),
    )
    launch_argv = _launch_argv(
        train_script_path=train_script_path,
        dataset_path=train_dataset_path,
        model_id=model_id,
        adapter_output_dir=adapter_output_dir,
        training_log_path=training_log_path,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    _write_train_script(train_script_path)
    train_script_sha256 = _sha256_file(train_script_path)
    job = {
        "schema": QWEN_SFT_TRAINING_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "input": str(input_path),
        "qwen_tool_training_manifest": str(qwen_manifest),
        "source_qwen_sft_messages_jsonl": str(source_sft_path),
        "qwen_sft_training_jsonl": str(train_dataset_path),
        "output_dir": str(output_dir),
        "adapter_output_dir": str(adapter_output_dir),
        "train_script": str(train_script_path),
        "train_script_sha256": train_script_sha256,
        "training_log_jsonl": str(training_log_path),
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "training_method": "offline_teacher_forced_sft_lora",
        "trainer": "custom_qwen_teacher_forced_lora",
        "model_id": model_id,
        "target_mode": target_mode,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
        "runtime": {
            "python_entrypoint": str(train_script_path),
            "launcher": "python",
            "dependency_check": "importlib.util.find_spec without importing GPU training libraries",
            "required_packages": DEFAULT_REQUIRED_PACKAGES,
        },
        "training_args": {
            "max_samples": max_samples,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_mode": target_mode,
        },
        "dataset": validation["dataset"],
        "audit": {
            "policy_input_only": True,
            "evaluator_labels_excluded": True,
            "teacher_forced_assistant_target": True,
            "target_source": (
                "qwen_sft_messages compact action content"
                if target_mode == TARGET_MODE_ACTION_ONLY
                else "qwen_sft_messages assistant content"
            ),
            "target_mode": target_mode,
            "require_existing_images": require_existing_images,
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job


def plan_qwen_sft_training_from_prompt_action_dataset(
    dataset_input: Path,
    *,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_output_dir: Path | None = None,
    max_samples: int | None = None,
    sample_offset: int = 0,
    sample_stride: int = 1,
    max_steps: int = 20,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when set")
    dataset_path = _resolve_prompt_action_dataset(dataset_input)
    source_records = _read_jsonl_if_exists(dataset_path)
    selected_source_records = _select_records(
        source_records,
        max_samples=max_samples,
        sample_offset=sample_offset,
        sample_stride=sample_stride,
    )
    selected_records = [_prompt_action_sft_record(record) for record in selected_source_records]

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = adapter_output_dir or output_dir / "adapter"
    train_dataset_path = output_dir / "qwen_sft_training_dataset.jsonl"
    train_script_path = output_dir / "train_qwen_sft_lora.py"
    training_log_path = output_dir / "qwen_sft_training_log.jsonl"
    job_path = output_dir / "qwen_sft_training_job.json"
    _write_jsonl(train_dataset_path, selected_records)
    validation = _validate_sft_records(
        selected_records,
        require_existing_images=require_existing_images,
        expected_source_count=None,
        source_sft_path=dataset_path,
        selected_from_count=len(source_records),
    )
    launch_argv = _launch_argv(
        train_script_path=train_script_path,
        dataset_path=train_dataset_path,
        model_id=model_id,
        adapter_output_dir=adapter_output_dir,
        training_log_path=training_log_path,
        max_steps=max_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    _write_train_script(train_script_path)
    train_script_sha256 = _sha256_file(train_script_path)
    job = {
        "schema": QWEN_SFT_TRAINING_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "input": str(dataset_input),
        "qwen_tool_training_manifest": "",
        "source_qwen_sft_messages_jsonl": "",
        "source_prompt_action_dataset_jsonl": str(dataset_path),
        "source_dataset_kind": "qwen_grpo_prompt_action_jsonl",
        "qwen_sft_training_jsonl": str(train_dataset_path),
        "output_dir": str(output_dir),
        "adapter_output_dir": str(adapter_output_dir),
        "train_script": str(train_script_path),
        "train_script_sha256": train_script_sha256,
        "training_log_jsonl": str(training_log_path),
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "training_method": "offline_teacher_forced_sft_lora",
        "trainer": "custom_qwen_teacher_forced_lora",
        "model_id": model_id,
        "target_mode": TARGET_MODE_ACTION_ONLY,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
        "runtime": {
            "python_entrypoint": str(train_script_path),
            "launcher": "python",
            "dependency_check": "importlib.util.find_spec without importing GPU training libraries",
            "required_packages": DEFAULT_REQUIRED_PACKAGES,
        },
        "training_args": {
            "max_samples": max_samples,
            "sample_offset": sample_offset,
            "sample_stride": sample_stride,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_mode": TARGET_MODE_ACTION_ONLY,
        },
        "dataset": validation["dataset"],
        "audit": {
            "policy_input_only": True,
            "evaluator_labels_excluded": True,
            "teacher_forced_assistant_target": True,
            "target_source": "qwen_grpo prompt/action reference action compact JSON",
            "target_mode": TARGET_MODE_ACTION_ONLY,
            "require_existing_images": require_existing_images,
            "source_training_manifest_required": False,
            "reference_actions_used_only_as_sft_targets": True,
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job


def run_qwen_sft_training_job(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_sft_training_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_sft_training_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    command = _argv_to_command(launch_argv)
    blockers = _training_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")

    result: dict[str, Any] = {
        "schema": QWEN_SFT_TRAINING_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "sample_count": (job.get("dataset") or {}).get("sample_count"),
        "adapter_output_dir": job.get("adapter_output_dir"),
        "training_log_jsonl": job.get("training_log_jsonl"),
        "training_log_tail": [],
        "launch_command": command,
        "launch_argv": launch_argv,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "duration_s": None,
        "blockers": blockers,
        "dependency_check": _dependency_check_payload(job, enabled=check_dependencies),
    }
    if blockers:
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_training_log_tail(result, job, job_path=job_path)
        _write_result(result_path, result)
        return result
    if dry_run:
        result["status"] = "dry_run"
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_training_log_tail(result, job, job_path=job_path)
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
        result["blockers"] = [f"training command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_training_log_tail(result, job, job_path=job_path)
    _write_result(result_path, result)
    return result


def _resolve_qwen_training_manifest(input_path: Path) -> Path:
    path = input_path.expanduser()
    if path.is_file() and path.name == "qwen_tool_training_manifest.json":
        return path
    candidates = [
        path / "qwen_tool_training_manifest.json",
        path / "qwen_tool_training" / "qwen_tool_training_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_tool_training_manifest.json under {input_path}")


def _resolve_manifest_path(manifest: dict[str, Any], key: str, *, default: Path) -> Path:
    value = manifest.get(key)
    if not value:
        return default
    path = Path(str(value))
    if path.exists():
        return path
    manifest_path = Path(str(manifest["_manifest_path"]))
    if path.is_absolute():
        relocated = _relocated_absolute_path(path, local_qwen_training_dir=manifest_path.parent)
        if relocated is not None and relocated.exists():
            return relocated
        local_name_candidate = manifest_path.parent / path.name
        return local_name_candidate if local_name_candidate.exists() else path
    candidates = [
        manifest_path.parent / path,
        Path(str(manifest.get("output_dir") or manifest_path.parent)) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default if default.exists() else path


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


def _relocated_absolute_path(path: Path, *, local_qwen_training_dir: Path) -> Path | None:
    parts = path.parts
    if "qwen_tool_training" not in parts:
        return None
    index = len(parts) - 1 - list(reversed(parts)).index("qwen_tool_training")
    tail = parts[index + 1 :]
    return local_qwen_training_dir / Path(*tail) if tail else local_qwen_training_dir


def _resolve_prompt_action_dataset(dataset_input: Path) -> Path:
    path = dataset_input.expanduser()
    if path.is_file():
        return path
    candidates = [
        path / "qwen_grpo_completion_eval_dataset.jsonl",
        path / "qwen_grpo_action_likelihood_dataset.jsonl",
        path / "qwen_grpo_trl_dataset.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find a Qwen prompt/action dataset JSONL under {dataset_input}")


def _select_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None,
    sample_offset: int,
    sample_stride: int,
) -> list[dict[str, Any]]:
    selected = records[sample_offset::sample_stride]
    return selected[:max_samples] if max_samples is not None else selected


def _prompt_action_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = copy.deepcopy(_prompt_messages(record))
    action = _reference_action(record)
    compact_action = _compact_action(action)
    compact_payload = {"action": compact_action} if compact_action is not None else {}
    compact_text = json.dumps(compact_payload, sort_keys=True, separators=(",", ":")) if compact_payload else ""
    messages.append({"role": "assistant", "content": compact_text})
    source_sample_id = str(record.get("sample_id") or "")
    metadata = {
        "source_dataset_kind": "qwen_grpo_prompt_action_jsonl",
        "source_sample_id": source_sample_id,
        "source_rollout_id": record.get("source_rollout_id"),
        "target_mode": TARGET_MODE_ACTION_ONLY,
    }
    return {
        "schema": QWEN_SFT_SAMPLE_SCHEMA,
        "sample_id": source_sample_id,
        "source_sample_id": source_sample_id,
        "source_policy_step_id": record.get("source_policy_step_id"),
        "source_rollout_id": record.get("source_rollout_id"),
        "messages": messages,
        "images": _record_image_paths(record),
        "image_paths": _record_image_paths(record),
        "assistant_target_json": compact_payload,
        "action_only_target_json": compact_payload,
        "target_mode": TARGET_MODE_ACTION_ONLY,
        "source_reference_action_sha256": _sha256_text(_canonical_json(action)),
        "metadata": metadata,
        "audit": {
            "target_mode": TARGET_MODE_ACTION_ONLY,
            "source_dataset_kind": "qwen_grpo_prompt_action_jsonl",
            "reference_action_sidecar_used_as_target": True,
            "reward_labels_excluded_from_messages": True,
        },
    }


def _prompt_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("prompt_messages") if record.get("prompt_messages") is not None else record.get("prompt")
    return copy.deepcopy(messages) if isinstance(messages, list) else []


def _reference_action(record: dict[str, Any]) -> dict[str, Any]:
    action = record.get("reference_action_json")
    if isinstance(action, dict):
        return copy.deepcopy(action)
    action = _parse_json_dict(record.get("reference_action_canonical"))
    if action:
        return action
    target_payload = record.get("assistant_target_json")
    if isinstance(target_payload, dict) and isinstance(target_payload.get("action"), dict):
        return copy.deepcopy(target_payload["action"])
    return {}


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _validate_sft_records(
    records: list[dict[str, Any]],
    *,
    require_existing_images: bool,
    expected_source_count: int | None,
    source_sft_path: Path,
    selected_from_count: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if not source_sft_path.exists():
        blockers.append(f"missing qwen_sft_messages_jsonl: {source_sft_path}")
    if not records:
        blockers.append("qwen_sft_training_jsonl contains no SFT records")
    if expected_source_count is not None and expected_source_count != selected_from_count:
        blockers.append(
            f"accepted_count mismatch: manifest={expected_source_count}, source_jsonl={selected_from_count}"
        )

    missing_images: list[str] = []
    forbidden_hits: Counter[str] = Counter()
    malformed_count = 0
    assistant_target_count = 0
    image_reference_count = 0
    schema_counts: Counter[str] = Counter()
    target_action_tool_counts: Counter[str] = Counter()
    target_mode_counts: Counter[str] = Counter()
    for record in records:
        schema_counts[str(record.get("schema") or "")] += 1
        target_mode_counts[str(record.get("target_mode") or TARGET_MODE_ASSISTANT)] += 1
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            malformed_count += 1
        else:
            assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
            if not assistant_messages or not str(assistant_messages[-1].get("content") or "").strip():
                malformed_count += 1
            else:
                assistant_target_count += 1
                tool = _target_action_tool(str(assistant_messages[-1].get("content") or ""))
                if tool:
                    target_action_tool_counts[tool] += 1
        image_paths = _record_image_paths(record)
        image_reference_count += len(image_paths)
        missing_images.extend(str(path) for path in image_paths if not Path(path).exists())
        for token in _forbidden_tokens([record.get("messages")]):
            forbidden_hits[token] += 1
    non_sft_schema_count = sum(count for schema, count in schema_counts.items() if schema and schema != QWEN_SFT_SAMPLE_SCHEMA)
    if non_sft_schema_count:
        blockers.append(f"{non_sft_schema_count} record(s) do not use {QWEN_SFT_SAMPLE_SCHEMA}")
    if malformed_count:
        blockers.append(f"{malformed_count} SFT record(s) are missing messages or assistant target content")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} SFT image reference(s) are missing")
    if forbidden_hits:
        blockers.append("SFT messages contain forbidden privileged token(s): " + ", ".join(sorted(forbidden_hits)))
    if image_reference_count == 0:
        warnings.append("SFT records contain no image references; this is unexpected for Qwen VLM navigation actions")

    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "sample_count": len(records),
            "source_sample_count": selected_from_count,
            "expected_source_sample_count": expected_source_count,
            "assistant_target_count": assistant_target_count,
            "image_reference_count": image_reference_count,
            "missing_image_count": len(missing_images),
            "missing_images": sorted(set(missing_images)),
            "forbidden_model_token_hits": dict(sorted(forbidden_hits.items())),
            "schema_counts": dict(sorted(schema_counts.items())),
            "target_action_tool_counts": dict(sorted(target_action_tool_counts.items())),
            "target_mode_counts": dict(sorted(target_mode_counts.items())),
        },
    }


def _records_for_target_mode(records: list[dict[str, Any]], *, target_mode: str) -> list[dict[str, Any]]:
    if target_mode == TARGET_MODE_ASSISTANT:
        return list(records)
    if target_mode == TARGET_MODE_ACTION_ONLY:
        return [_action_only_record(record) for record in records]
    raise ValueError(f"target_mode must be one of {TARGET_MODES}: {target_mode}")


def _action_only_record(record: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(record)
    original_target = _assistant_target(copied.get("messages"))
    target_payload = copied.get("assistant_target_json")
    if not isinstance(target_payload, dict):
        target_payload = _parse_assistant_target_json(copied.get("messages"))
    action = target_payload.get("action") if isinstance(target_payload, dict) else None
    compact_action = _compact_action(action)
    if compact_action is None:
        compact_text = ""
        compact_payload: dict[str, Any] = {}
    else:
        compact_payload = {"action": compact_action}
        compact_text = json.dumps(compact_payload, sort_keys=True, separators=(",", ":"))
    _replace_last_assistant_content(copied, compact_text)
    copied["assistant_target_json"] = compact_payload
    copied["action_only_target_json"] = compact_payload
    copied["target_mode"] = TARGET_MODE_ACTION_ONLY
    copied["source_assistant_target_sha256"] = _sha256_text(original_target)
    audit = copied.get("audit")
    if not isinstance(audit, dict):
        audit = {}
        copied["audit"] = audit
    audit["target_mode"] = TARGET_MODE_ACTION_ONLY
    audit["source_target_preserved_by_hash"] = True
    metadata = copied.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        copied["metadata"] = metadata
    metadata["target_mode"] = TARGET_MODE_ACTION_ONLY
    return copied


def _parse_assistant_target_json(messages: Any) -> dict[str, Any]:
    target = _assistant_target(messages)
    if not target:
        return {}
    try:
        payload = json.loads(target)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _assistant_target(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _replace_last_assistant_content(record: dict[str, Any], content: str) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = content
            return


def _compact_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict) or not action.get("tool"):
        return None
    args = action.get("args")
    return {
        "tool": action.get("tool"),
        "args": args if isinstance(args, dict) else {},
    }


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_image_paths(record: dict[str, Any]) -> list[str]:
    values = record.get("images")
    if not isinstance(values, list):
        values = record.get("image_paths")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value)]


def _forbidden_tokens(payloads: Iterable[Any]) -> list[str]:
    text = json.dumps(list(payloads), sort_keys=True, default=str).lower()
    return [token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text]


def _training_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_SFT_TRAINING_JOB_SCHEMA:
        blockers.append(f"unexpected job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"training job is not ready: {job.get('status')}")
    train_script = _job_path(job, "train_script", relative_to=job_path.parent)
    if train_script is None or not train_script.exists():
        blockers.append(f"missing train_script: {job.get('train_script')}")
    dataset_path = _job_path(job, "qwen_sft_training_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(f"missing qwen_sft_training_jsonl: {job.get('qwen_sft_training_jsonl')}")
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required training package(s): " + ", ".join(missing_packages))
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


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _launch_argv(
    *,
    train_script_path: Path,
    dataset_path: Path,
    model_id: str,
    adapter_output_dir: Path,
    training_log_path: Path,
    max_steps: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> list[str]:
    return [
        "python",
        str(train_script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--output-dir",
        str(adapter_output_dir),
        "--training-log",
        str(training_log_path),
        "--max-steps",
        str(max_steps),
        "--learning-rate",
        str(learning_rate),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
        "--lora-dropout",
        str(lora_dropout),
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


def _write_train_script(path: Path) -> None:
    path.write_text(_TRAIN_SCRIPT, encoding="utf-8")


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


def _attach_training_log_tail(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    training_log = _job_path(job, "training_log_jsonl", relative_to=job_path.parent)
    result["training_log_jsonl"] = str(training_log) if training_log else ""
    result["training_log_tail"] = _jsonl_tail(training_log, 8)


_TRAIN_SCRIPT = '''"""Run tiny teacher-forced LoRA SFT over flatdisk Qwen navigation actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import gmtime, strftime

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\\n")


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def split_messages(record: dict) -> tuple[list[dict], str]:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    if not messages:
        return [], ""
    assistant_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant_index = index
            break
    if assistant_index is None:
        return messages, ""
    prompt_messages = messages[:assistant_index]
    target = str(messages[assistant_index].get("content") or "")
    return prompt_messages, target


def prepare_inputs(processor, model, record: dict) -> tuple[dict, dict]:
    prompt_messages, target_text = split_messages(record)
    images = [load_image(path) for path in record.get("image_paths", record.get("images", []))]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + target_text
    prompt_kwargs = {"text": [prompt_text], "return_tensors": "pt"}
    full_kwargs = {"text": [full_text], "return_tensors": "pt"}
    if images:
        prompt_kwargs["images"] = images
        full_kwargs["images"] = images
    prompt_inputs = processor(**prompt_kwargs)
    full_inputs = processor(**full_kwargs)
    input_ids = full_inputs["input_ids"]
    target_start = int(prompt_inputs["input_ids"].shape[1])
    labels = input_ids.clone()
    labels[:, :target_start] = -100
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    full_inputs["labels"] = labels
    if hasattr(full_inputs, "to"):
        full_inputs = full_inputs.to(model_device(model))
    return full_inputs, {
        "sample_id": record.get("sample_id"),
        "source_policy_step_id": record.get("source_policy_step_id"),
        "image_count": len(images),
        "input_token_count": int(input_ids.shape[1]),
        "target_start_token_index": target_start,
        "target_token_count": max(0, int(input_ids.shape[1]) - target_start),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.training_log.parent.mkdir(parents=True, exist_ok=True)
    args.training_log.write_text("", encoding="utf-8")
    records = read_jsonl(args.dataset)
    append_jsonl(args.training_log, {
        "stage": "start",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "sample_count": len(records),
        "model_id": args.model_id,
    })
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    append_jsonl(args.training_log, {
        "stage": "model_ready",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "trainable_parameter_count": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
    })

    completed_steps = 0
    accumulation_index = 0
    last_loss = None
    while completed_steps < args.max_steps:
        record = records[completed_steps % len(records)]
        inputs, meta = prepare_inputs(processor, model, record)
        outputs = model(**inputs)
        loss = outputs.loss / max(1, args.gradient_accumulation_steps)
        loss.backward()
        accumulation_index += 1
        if accumulation_index >= max(1, args.gradient_accumulation_steps):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0
        completed_steps += 1
        last_loss = float(outputs.loss.detach().float().cpu().item())
        append_jsonl(args.training_log, {
            "stage": "step",
            "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
            "step": completed_steps,
            "loss": round(last_loss, 9),
            **meta,
        })
    if accumulation_index:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    append_jsonl(args.training_log, {
        "stage": "complete",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_steps": completed_steps,
        "last_loss": round(last_loss, 9) if last_loss is not None else None,
        "output_dir": str(args.output_dir),
    })


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="qwen_tool_training dir or qwen_tool_training_manifest.json")
    source.add_argument(
        "--dataset-jsonl",
        type=Path,
        help="Existing Qwen GRPO prompt/action dataset JSONL or containing directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default=TARGET_MODE_ASSISTANT)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen SFT training job.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_sft_training dir or qwen_sft_training_job.json")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch_command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset_jsonl is not None:
        job = plan_qwen_sft_training_from_prompt_action_dataset(
            args.dataset_jsonl,
            output_dir=args.output_dir,
            model_id=args.model_id,
            adapter_output_dir=args.adapter_output_dir,
            max_samples=args.max_samples,
            sample_offset=args.sample_offset,
            sample_stride=args.sample_stride,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            require_existing_images=not args.allow_missing_images,
        )
    else:
        job = plan_qwen_sft_training(
            args.input,
            output_dir=args.output_dir,
            model_id=args.model_id,
            adapter_output_dir=args.adapter_output_dir,
            max_samples=args.max_samples,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_mode=args.target_mode,
            require_existing_images=not args.allow_missing_images,
        )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["sample_count"],
                "source_sample_count": job["dataset"]["source_sample_count"],
                "target_mode": job["target_mode"],
                "target_action_tool_counts": job["dataset"]["target_action_tool_counts"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "job_path": str(Path(job["output_dir"]) / "qwen_sft_training_job.json"),
                "train_script": job["train_script"],
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
    result = run_qwen_sft_training_job(
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
