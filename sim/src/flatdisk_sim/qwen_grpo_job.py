"""Plan and run Qwen GRPO training jobs from materialized rollout groups."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
from time import gmtime, monotonic, strftime
from typing import Any

from .qwen_dpo_training import DEFAULT_MODEL_ID, DEFAULT_REQUIRED_PACKAGES
from .qwen_grpo_training import QWEN_GRPO_MANIFEST_SCHEMA
from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS


QWEN_GRPO_TRAINING_JOB_SCHEMA = "flatdisk.qwen_grpo_training_job.v1"
QWEN_GRPO_TRAINING_RESULT_SCHEMA = "flatdisk.qwen_grpo_training_result.v1"
QWEN_GRPO_TRL_SAMPLE_SCHEMA = "flatdisk.qwen_grpo_trl_prompt_sample.v1"
GRPO_RESPONSE_CONTRACT = (
    "GRPO_RESPONSE_CONTRACT\n"
    "For this training action response, output only one compact JSON object and stop immediately after it. "
    'Required form: {"action":{"tool":"<tool_name>","args":{...}}}. '
    "Do not use markdown fences. Do not include thought, grounding_audit, memory_update, save_frames, commentary, "
    "or extra keys."
)


def plan_qwen_grpo_training(
    input_path: Path,
    *,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_output_dir: Path | None = None,
    max_steps: int = 100,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    num_generations: int = 2,
    max_completion_length: int = 96,
    reward_scale: float = 1.0,
    balance_reference_tools: bool = False,
    max_balance_multiplier: int = 4,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    grpo_manifest_path = _resolve_qwen_grpo_manifest(input_path)
    manifest = json.loads(grpo_manifest_path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(grpo_manifest_path)
    groups_path = _resolve_manifest_path(manifest, "qwen_grpo_rollout_groups_jsonl")
    ppo_steps_path = _resolve_manifest_path(manifest, "qwen_ppo_step_samples_jsonl")
    group_records = _read_jsonl_if_exists(groups_path)
    ppo_step_records = _read_jsonl_if_exists(ppo_steps_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = adapter_output_dir or output_dir / "adapter"
    dataset_path = output_dir / "qwen_grpo_trl_dataset.jsonl"
    train_script_path = output_dir / "train_qwen_grpo_trl.py"
    job_path = output_dir / "qwen_grpo_training_job.json"
    completion_log_path = adapter_output_dir / "completion_samples.jsonl"
    raw_dataset_records = _trl_dataset_records(ppo_step_records)
    dataset_records = (
        _balance_dataset_records_by_reference_tool(raw_dataset_records, max_multiplier=max_balance_multiplier)
        if balance_reference_tools
        else raw_dataset_records
    )
    _write_jsonl(dataset_path, dataset_records)
    validation = _validate_grpo_job_inputs(
        manifest,
        group_records=group_records,
        ppo_step_records=ppo_step_records,
        dataset_records=dataset_records,
        groups_path=groups_path,
        ppo_steps_path=ppo_steps_path,
        require_existing_images=require_existing_images,
    )
    launch_argv = _launch_argv(
        train_script_path=train_script_path,
        dataset_path=dataset_path,
        model_id=model_id,
        adapter_output_dir=adapter_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        reward_scale=reward_scale,
    )
    _write_train_script(train_script_path)
    job = {
        "schema": QWEN_GRPO_TRAINING_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "input": str(input_path),
        "qwen_grpo_training_manifest": str(grpo_manifest_path),
        "qwen_grpo_rollout_groups_jsonl": str(groups_path) if groups_path else "",
        "qwen_ppo_step_samples_jsonl": str(ppo_steps_path) if ppo_steps_path else "",
        "qwen_grpo_trl_dataset_jsonl": str(dataset_path),
        "output_dir": str(output_dir),
        "adapter_output_dir": str(adapter_output_dir),
        "completion_log_jsonl": str(completion_log_path),
        "train_script": str(train_script_path),
        "train_script_sha256": _sha256_file(train_script_path),
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "training_method": "offline_replay_grpo",
        "trainer": "trl.GRPOTrainer",
        "model_id": model_id,
        "required_packages": DEFAULT_REQUIRED_PACKAGES,
        "runtime": {
            "python_entrypoint": str(train_script_path),
            "launcher": "accelerate",
            "dependency_check": "importlib.util.find_spec without importing GPU training libraries",
            "required_packages": DEFAULT_REQUIRED_PACKAGES,
            "trl_dataset_type": "vlm_prompt_plus_images",
        },
        "training_args": {
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "num_generations": num_generations,
            "max_completion_length": max_completion_length,
            "reward_scale": reward_scale,
            "remove_unused_columns": False,
        },
        "adapter": {
            "method": "peft_lora",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "dataset": validation["dataset"],
        "dataset_action_audit": {
            "before_balancing": _grpo_dataset_action_summary(raw_dataset_records),
            "after_balancing": _grpo_dataset_action_summary(dataset_records),
        },
        "audit": {
            "prompt_only_dataset": True,
            "reward_labels_excluded_from_messages": True,
            "reference_actions_are_sidecar_columns": True,
            "require_existing_images": require_existing_images,
            "offline_replay_reward": True,
            "online_environment_reward": False,
            "reference_tool_balancing": {
                "enabled": balance_reference_tools,
                "max_multiplier": max_balance_multiplier if balance_reference_tools else 1,
                "sample_count_before": len(raw_dataset_records),
                "sample_count_after": len(dataset_records),
            },
        },
    }
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return job


def run_qwen_grpo_training_job(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_qwen_grpo_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_grpo_training_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    blockers = _training_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")
    result: dict[str, Any] = {
        "schema": QWEN_GRPO_TRAINING_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "sample_count": (job.get("dataset") or {}).get("sample_count"),
        "adapter_output_dir": job.get("adapter_output_dir"),
        "launch_command": _argv_to_command(launch_argv),
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
        result["blockers"] = [f"training command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_completion_log_summary(result, job, job_path=job_path)
    _write_result(result_path, result)
    return result


def _attach_completion_log_summary(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    completion_log = _job_path(job, "completion_log_jsonl", relative_to=job_path.parent)
    result["completion_log_jsonl"] = str(completion_log) if completion_log else ""
    result["completion_log_sample_count"] = _count_lines(completion_log) if completion_log and completion_log.exists() else 0
    result["completion_log_metrics"] = _completion_log_metrics(completion_log) if completion_log and completion_log.exists() else {}


def _completion_log_metrics(path: Path) -> dict[str, Any]:
    records = []
    malformed_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_count += 1
    completion_texts = [str(record.get("completion_text") or "") for record in records]
    exact_reference_count = 0
    parsed_action_count = 0
    positive_non_reference_count = 0
    tool_match_count = 0
    arg_match_fractions = []
    for record in records:
        parsed_action = record.get("parsed_action") if isinstance(record.get("parsed_action"), dict) else {}
        expected_action = (
            record.get("expected_action")
            if isinstance(record.get("expected_action"), dict)
            else _reference_action_from_canonical(record.get("reference_action_canonical"))
        )
        if parsed_action:
            parsed_action_count += 1
        tool_match = (
            bool(record.get("tool_match"))
            if "tool_match" in record
            else _action_tool(parsed_action) == _action_tool(expected_action)
        )
        if parsed_action and tool_match:
            tool_match_count += 1
        arg_match_fraction = _optional_float(record.get("arg_match_fraction"))
        if arg_match_fraction is None:
            arg_match_fraction = (
                _arg_match_fraction(parsed_action, expected_action) if parsed_action and tool_match else 0.0
            )
        arg_match_fractions.append(arg_match_fraction)
        exact_reference = _canonical_json(parsed_action) == record.get("reference_action_canonical")
        if exact_reference:
            exact_reference_count += 1
        elif _optional_float(record.get("reward")) and float(record["reward"]) > 0:
            positive_non_reference_count += 1
    return {
        "sample_count": len(records),
        "malformed_line_count": malformed_count,
        "parsed_action_count": parsed_action_count,
        "exact_reference_action_count": exact_reference_count,
        "positive_non_reference_reward_count": positive_non_reference_count,
        "tool_match_count": tool_match_count,
        "parsed_action_rate": round(parsed_action_count / len(records), 6) if records else 0.0,
        "exact_reference_action_rate": round(exact_reference_count / len(records), 6) if records else 0.0,
        "tool_match_rate": round(tool_match_count / len(records), 6) if records else 0.0,
        "mean_arg_match_fraction": round(sum(arg_match_fractions) / len(arg_match_fractions), 6)
        if arg_match_fractions
        else 0.0,
        "markdown_fence_count": sum("```" in text for text in completion_texts),
        "truncated_text_count": sum(bool(record.get("completion_text_truncated")) for record in records),
        "mean_completion_chars": round(sum(len(text) for text in completion_texts) / len(completion_texts), 3)
        if completion_texts
        else 0.0,
    }


def _trl_dataset_records(ppo_step_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for record in ppo_step_records:
        reward = record.get("evaluator_reward") if isinstance(record.get("evaluator_reward"), dict) else {}
        target = record.get("assistant_target_json") if isinstance(record.get("assistant_target_json"), dict) else {}
        action = target.get("action") if isinstance(target.get("action"), dict) else {}
        source_prompt_messages = record.get("prompt_messages", [])
        prompt_messages = _append_grpo_response_contract(source_prompt_messages)
        image_paths = [str(path) for path in record.get("image_paths", []) if str(path)]
        records.append(
            {
                "schema": QWEN_GRPO_TRL_SAMPLE_SCHEMA,
                "sample_id": record.get("sample_id"),
                "source_rollout_id": record.get("source_rollout_id"),
                "terminal": bool(record.get("terminal")),
                "prompt": prompt_messages,
                "prompt_messages": prompt_messages,
                "source_prompt_messages": source_prompt_messages,
                "response_contract": GRPO_RESPONSE_CONTRACT,
                "image_paths": image_paths,
                "reference_assistant_json": target,
                "reference_action_json": action,
                "reference_action_canonical": _canonical_json(action),
                "candidate_step_reward": _optional_float(reward.get("candidate_step_reward")) or 0.0,
                "candidate_episode_reward": _optional_float(reward.get("candidate_episode_reward")) or 0.0,
                "reward_source": "offline evaluator reward sidecar; excluded from prompt and completion",
            }
        )
    return records


def _balance_dataset_records_by_reference_tool(
    records: list[dict[str, Any]], *, max_multiplier: int
) -> list[dict[str, Any]]:
    if not records:
        return []
    multiplier = max(1, int(max_multiplier or 1))
    if multiplier <= 1:
        return list(records)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_reference_tool(record)].append(record)
    if len(groups) <= 1:
        return list(records)

    target_count = max(len(group) for group in groups.values())
    desired_counts = {
        tool: min(target_count, len(tool_records) * multiplier) for tool, tool_records in groups.items()
    }
    if all(desired_counts[tool] == len(tool_records) for tool, tool_records in groups.items()):
        return list(records)

    balanced: list[dict[str, Any]] = []
    for offset in range(max(desired_counts.values())):
        for tool in sorted(groups):
            tool_records = groups[tool]
            if offset >= desired_counts[tool]:
                continue
            source = tool_records[offset % len(tool_records)]
            if offset < len(tool_records):
                balanced.append(source)
                continue
            duplicate = dict(source)
            original_sample_id = str(source.get("sample_id") or f"{tool}_{offset % len(tool_records)}")
            duplicate["balance_original_sample_id"] = original_sample_id
            duplicate["balance_copy_index"] = offset // len(tool_records)
            duplicate["sample_id"] = f"{original_sample_id}_balance_copy{duplicate['balance_copy_index']:02d}"
            balanced.append(duplicate)
    return balanced


def _grpo_dataset_action_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counts = Counter(_reference_tool(record) for record in records)
    arg_key_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rewards_by_tool: dict[str, list[float]] = defaultdict(list)
    for record in records:
        tool = _reference_tool(record)
        action = record.get("reference_action_json") if isinstance(record.get("reference_action_json"), dict) else {}
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        arg_key_counts[tool].update(str(key) for key in args)
        rewards_by_tool[tool].append(float(_optional_float(record.get("candidate_step_reward")) or 0.0))
    return {
        "sample_count": len(records),
        "reference_action_tool_counts": dict(sorted(tool_counts.items())),
        "reference_action_arg_key_counts": {
            tool: dict(sorted(counts.items())) for tool, counts in sorted(arg_key_counts.items())
        },
        "candidate_step_reward_by_tool": {
            tool: _reward_summary(values) for tool, values in sorted(rewards_by_tool.items())
        },
        "terminal_count": sum(bool(record.get("terminal")) for record in records),
        "balanced_copy_count": sum(1 for record in records if "balance_original_sample_id" in record),
    }


def _reference_tool(record: dict[str, Any]) -> str:
    action = record.get("reference_action_json") if isinstance(record.get("reference_action_json"), dict) else {}
    tool = action.get("tool")
    return str(tool) if tool else "unknown"


def _reward_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "negative_count": 0,
            "zero_count": 0,
            "positive_count": 0,
        }
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "positive_count": sum(value > 0 for value in values),
    }


def _append_grpo_response_contract(messages: Any) -> list[dict[str, Any]]:
    normalized = [dict(message) for message in messages] if isinstance(messages, list) else []
    contract_item = {"type": "text", "text": GRPO_RESPONSE_CONTRACT}
    if not normalized:
        return [{"role": "user", "content": [contract_item]}]
    last = dict(normalized[-1])
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = [*content, contract_item]
    elif content:
        last["content"] = [{"type": "text", "text": str(content)}, contract_item]
    else:
        last["content"] = [contract_item]
    normalized[-1] = last
    return normalized


def _validate_grpo_job_inputs(
    manifest: dict[str, Any],
    *,
    group_records: list[dict[str, Any]],
    ppo_step_records: list[dict[str, Any]],
    dataset_records: list[dict[str, Any]],
    groups_path: Path | None,
    ppo_steps_path: Path | None,
    require_existing_images: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if manifest.get("schema") != QWEN_GRPO_MANIFEST_SCHEMA:
        blockers.append(f"unexpected Qwen GRPO manifest schema: {manifest.get('schema')}")
    if manifest.get("status") != "ready":
        blockers.append(f"Qwen GRPO handoff is not ready: {manifest.get('status')}")
    if groups_path is None or not groups_path.exists():
        blockers.append(f"missing qwen_grpo_rollout_groups_jsonl: {groups_path}")
    if ppo_steps_path is None or not ppo_steps_path.exists():
        blockers.append(f"missing qwen_ppo_step_samples_jsonl: {ppo_steps_path}")
    if not group_records:
        blockers.append("qwen_grpo_rollout_groups_jsonl contains no groups")
    if not ppo_step_records:
        blockers.append("qwen_ppo_step_samples_jsonl contains no PPO step samples")
    if not dataset_records:
        blockers.append("GRPO TRL dataset contains no prompt samples")
    expected_ppo_steps = _optional_int(manifest.get("ppo_step_sample_count"))
    if expected_ppo_steps and expected_ppo_steps != len(ppo_step_records):
        blockers.append(f"ppo_step_sample_count mismatch: manifest={expected_ppo_steps}, jsonl={len(ppo_step_records)}")
    if _optional_int(manifest.get("trainable_group_count")) == 0:
        blockers.append("Qwen GRPO handoff has no trainable groups")
    if _optional_int(manifest.get("trainable_candidate_count")) == 0:
        blockers.append("Qwen GRPO handoff has no trainable candidates")

    missing_images = sorted(
        {
            str(path)
            for record in dataset_records
            for path in record.get("image_paths", [])
            if not Path(str(path)).exists()
        }
    )
    forbidden_hits = _forbidden_message_hits(dataset_records)
    malformed_count = sum(
        1
        for record in dataset_records
        if not isinstance(record.get("prompt"), list) or not isinstance(record.get("reference_action_json"), dict)
    )
    if malformed_count:
        blockers.append(f"{malformed_count} GRPO prompt sample(s) are malformed")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} GRPO prompt image reference(s) are missing")
    elif missing_images:
        warnings.append(f"{len(missing_images)} GRPO prompt image reference(s) are missing")
    if forbidden_hits:
        blockers.append("GRPO prompt samples contain forbidden privileged token(s): " + ", ".join(forbidden_hits))
    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "sample_count": len(dataset_records),
            "source_group_count": len(group_records),
            "source_ppo_step_count": len(ppo_step_records),
            "expected_ppo_step_sample_count": expected_ppo_steps,
            "trainable_group_count": _optional_int(manifest.get("trainable_group_count")),
            "trainable_candidate_count": _optional_int(manifest.get("trainable_candidate_count")),
            "image_reference_count": sum(
                len(record.get("image_paths", []))
                for record in dataset_records
                if isinstance(record.get("image_paths"), list)
            ),
            "missing_image_count": len(missing_images),
            "missing_images": missing_images,
            "forbidden_model_token_hits": forbidden_hits,
        },
    }


def _resolve_qwen_grpo_manifest(input_path: Path) -> Path:
    path = input_path.expanduser()
    if path.is_file() and path.name == "qwen_grpo_training_manifest.json":
        return path
    candidates = [
        path / "qwen_grpo_training_manifest.json",
        path / "qwen_grpo_training" / "qwen_grpo_training_manifest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_grpo_training_manifest.json under {input_path}")


def _resolve_qwen_grpo_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_grpo_training_job.json":
        return path
    candidates = [
        path / "qwen_grpo_training_job.json",
        path / "qwen_grpo_training" / "qwen_grpo_training_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_grpo_training_job.json under {job_input}")


def _resolve_manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = manifest.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    manifest_path = Path(str(manifest["_manifest_path"]))
    if path.is_absolute():
        relocated = _relocated_path(path, local_dir=manifest_path.parent)
        if relocated is not None and relocated.exists():
            return relocated
        return path
    candidates = [
        manifest_path.parent / path,
        Path(str(manifest.get("output_dir") or manifest_path.parent)) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _relocated_path(path: Path, *, local_dir: Path) -> Path | None:
    parts = path.parts
    if "qwen_grpo_training" not in parts:
        return None
    index = len(parts) - 1 - list(reversed(parts)).index("qwen_grpo_training")
    tail = parts[index + 1 :]
    return local_dir / Path(*tail) if tail else local_dir


def _training_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_GRPO_TRAINING_JOB_SCHEMA:
        blockers.append(f"unexpected job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"training job is not ready: {job.get('status')}")
    train_script = _job_path(job, "train_script", relative_to=job_path.parent)
    if train_script is None or not train_script.exists():
        blockers.append(f"missing train_script: {job.get('train_script')}")
    dataset_path = _job_path(job, "qwen_grpo_trl_dataset_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(f"missing qwen_grpo_trl_dataset_jsonl: {job.get('qwen_grpo_trl_dataset_jsonl')}")
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
    return {"enabled": enabled, "required_packages": required, "missing_packages": missing}


def _missing_required_packages(job: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for package in [str(value) for value in job.get("required_packages", [])]:
        if importlib.util.find_spec(_import_module_for_package(package)) is None:
            missing.append(package)
    return missing


def _import_module_for_package(package: str) -> str:
    return {"pillow": "PIL"}.get(package.lower(), package.replace("-", "_"))


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _launch_argv(
    *,
    train_script_path: Path,
    dataset_path: Path,
    model_id: str,
    adapter_output_dir: Path,
    max_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_generations: int,
    max_completion_length: int,
    reward_scale: float,
) -> list[str]:
    return [
        "accelerate",
        "launch",
        str(train_script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--output-dir",
        str(adapter_output_dir),
        "--max-steps",
        str(max_steps),
        "--per-device-train-batch-size",
        str(per_device_train_batch_size),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--learning-rate",
        str(learning_rate),
        "--num-generations",
        str(num_generations),
        "--max-completion-length",
        str(max_completion_length),
        "--reward-scale",
        str(reward_scale),
    ]


def _job_launch_argv(job: dict[str, Any], *, launch_command_override: str | None) -> list[str]:
    if launch_command_override:
        return shlex.split(launch_command_override)
    launch_argv = job.get("launch_argv")
    if isinstance(launch_argv, list) and all(isinstance(part, str) and part for part in launch_argv):
        return [str(part) for part in launch_argv]
    launch_command = str(job.get("launch_command") or "")
    return shlex.split(launch_command) if launch_command else []


def _argv_to_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def _write_train_script(path: Path) -> None:
    path.write_text(_TRAIN_SCRIPT, encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl_if_exists(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _forbidden_message_hits(records: list[dict[str, Any]]) -> list[str]:
    prompt_payload = [
        {
            "prompt": record.get("prompt"),
            "reference_assistant_json": record.get("reference_assistant_json"),
        }
        for record in records
    ]
    text = json.dumps(prompt_payload, sort_keys=True, default=str).lower()
    return sorted({token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text})


def _optional_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _reference_action_from_canonical(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _action_tool(action: Any) -> str | None:
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    return str(tool) if tool is not None else None


def _action_args(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    args = action.get("args")
    return args if isinstance(args, dict) else {}


def _arg_match_fraction(parsed_action: Any, expected_action: Any) -> float:
    expected_args = _action_args(expected_action)
    parsed_args = _action_args(parsed_action)
    if not expected_args:
        return 1.0 if parsed_args == expected_args else 0.0
    matches = sum(
        1
        for key, expected_value in expected_args.items()
        if _canonical_json(parsed_args.get(key)) == _canonical_json(expected_value)
    )
    return matches / len(expected_args)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(text: str, chars: int) -> str:
    return text[-chars:] if chars > 0 and len(text) > chars else text


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_TRAIN_SCRIPT = '''"""Run TRL GRPO over flatdisk Qwen navigation prompt samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import gmtime, strftime

from datasets import Dataset
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer


_COMPLETION_LOG_BATCH_INDEX = 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def completion_text(completion) -> str:
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, list):
                    parts.extend(str(part.get("text") or part.get("content") or "") for part in content if isinstance(part, dict))
                else:
                    parts.append(str(content or ""))
            else:
                parts.append(str(item))
        return "\\n".join(part for part in parts if part)
    if isinstance(completion, dict):
        return completion_text([completion])
    return str(completion or "")


def extract_json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def parse_action(text: str) -> dict:
    payload = extract_json_object(text)
    if isinstance(payload, dict) and isinstance(payload.get("action"), dict):
        return payload["action"]
    return {}


def parse_reference_action(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def action_tool(action) -> str | None:
    if not isinstance(action, dict):
        return None
    tool = action.get("tool")
    return str(tool) if tool is not None else None


def action_args(action) -> dict:
    if not isinstance(action, dict):
        return {}
    args = action.get("args")
    return args if isinstance(args, dict) else {}


def arg_match_fraction(parsed_action, expected_action) -> float:
    expected_args = action_args(expected_action)
    parsed_args = action_args(parsed_action)
    if not expected_args:
        return 1.0 if parsed_args == expected_args else 0.0
    matches = sum(
        1
        for key, expected_value in expected_args.items()
        if canonical_json(parsed_args.get(key)) == canonical_json(expected_value)
    )
    return matches / len(expected_args)


def action_reward_diagnostics(parsed_action, expected_action) -> dict:
    tool_match = bool(parsed_action) and action_tool(parsed_action) == action_tool(expected_action)
    return {
        "tool_match": tool_match,
        "arg_match_fraction": arg_match_fraction(parsed_action, expected_action) if tool_match else 0.0,
    }


def partial_action_reward(parsed_action, expected_action) -> float:
    diagnostics = action_reward_diagnostics(parsed_action, expected_action)
    if diagnostics["tool_match"]:
        return -0.15 + (0.10 * diagnostics["arg_match_fraction"])
    return -0.30


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "image":
                continue
            text = item.get("text") or item.get("content")
            if text:
                parts.append(str(text))
        elif item:
            parts.append(str(item))
    return "\\n".join(parts)


def conversational_text_messages(messages: list[dict]) -> list[dict]:
    return [{**message, "content": content_to_text(message.get("content"))} for message in messages]


def value_at(value, index: int):
    if isinstance(value, list):
        return value[index] if index < len(value) else None
    return value


def log_completion_batch(completions, rewards, reference_action_canonical=None, candidate_step_reward=None, metadata=None) -> None:
    path_value = os.environ.get("FLATDISK_GRPO_COMPLETION_LOG")
    if not path_value:
        return
    try:
        max_batches = int(os.environ.get("FLATDISK_GRPO_COMPLETION_LOG_MAX_BATCHES", "200") or "200")
    except ValueError:
        max_batches = 200
    global _COMPLETION_LOG_BATCH_INDEX
    if _COMPLETION_LOG_BATCH_INDEX >= max_batches:
        return
    _COMPLETION_LOG_BATCH_INDEX += 1
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    logged_at = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    with path.open("a", encoding="utf-8") as handle:
        for index, completion in enumerate(completions):
            text = completion_text(completion)
            expected = value_at(reference_action_canonical, index)
            expected_action = parse_reference_action(expected)
            parsed_action = parse_action(text)
            diagnostics = action_reward_diagnostics(parsed_action, expected_action)
            record = {
                "schema": "flatdisk.qwen_grpo_completion_sample.v1",
                "logged_at": logged_at,
                "batch_index": _COMPLETION_LOG_BATCH_INDEX,
                "completion_index": index,
                "sample_id": value_at(metadata.get("sample_id"), index),
                "source_rollout_id": value_at(metadata.get("source_rollout_id"), index),
                "reward": value_at(rewards, index),
                "candidate_step_reward": value_at(candidate_step_reward, index),
                "reference_action_canonical": expected,
                "expected_action": expected_action,
                "parsed_action": parsed_action,
                "tool_match": diagnostics["tool_match"],
                "arg_match_fraction": diagnostics["arg_match_fraction"],
                "completion_text": text[:4000],
                "completion_text_truncated": len(text) > 4000,
            }
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\\n")


def navigation_tool_reward(completions, reference_action_canonical=None, candidate_step_reward=None, reward_scale=None, **kwargs):
    scale = float(reward_scale[0] if isinstance(reward_scale, list) and reward_scale else reward_scale or 1.0)
    rewards = []
    for index, completion in enumerate(completions):
        expected = reference_action_canonical[index] if isinstance(reference_action_canonical, list) else reference_action_canonical
        step_reward = candidate_step_reward[index] if isinstance(candidate_step_reward, list) else candidate_step_reward
        parsed = parse_action(completion_text(completion))
        base_reward = float(step_reward or 0.0)
        expected_action = parse_reference_action(expected)
        if canonical_json(parsed) == expected:
            rewards.append(base_reward * scale)
        elif parsed:
            rewards.append(min(partial_action_reward(parsed, expected_action), base_reward - 0.05, -0.02) * scale)
        else:
            rewards.append(min(base_reward - 0.5, -0.5) * scale)
    log_completion_batch(completions, rewards, reference_action_canonical, candidate_step_reward, kwargs)
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_completion_log = args.output_dir / "completion_samples.jsonl"
    if not os.environ.get("FLATDISK_GRPO_COMPLETION_LOG"):
        os.environ["FLATDISK_GRPO_COMPLETION_LOG"] = str(default_completion_log)
        default_completion_log.unlink(missing_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_id, padding_side="left")
    records = read_jsonl(args.dataset)
    for record in records:
        messages = record.get("prompt_messages") or record.get("prompt") or []
        record["images"] = [load_image(path) for path in record.get("image_paths", [])]
        record["prompt"] = conversational_text_messages(messages)
        record["reward_scale"] = args.reward_scale
    dataset = Dataset.from_list(records)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=navigation_tool_reward,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="qwen_grpo_training dir or manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument(
        "--balance-reference-tools",
        action="store_true",
        help="Deterministically duplicate underrepresented reference-action tool families in the TRL dataset.",
    )
    parser.add_argument(
        "--max-balance-multiplier",
        type=int,
        default=4,
        help="Maximum per-sample duplication multiplier when --balance-reference-tools is enabled.",
    )
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job = plan_qwen_grpo_training(
        args.input,
        output_dir=args.output_dir,
        model_id=args.model_id,
        adapter_output_dir=args.adapter_output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        reward_scale=args.reward_scale,
        balance_reference_tools=args.balance_reference_tools,
        max_balance_multiplier=args.max_balance_multiplier,
        require_existing_images=not args.allow_missing_images,
    )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["sample_count"],
                "reference_action_tool_counts": job["dataset_action_audit"]["after_balancing"][
                    "reference_action_tool_counts"
                ],
                "reference_tool_balancing": job["audit"]["reference_tool_balancing"],
                "trainable_group_count": job["dataset"]["trainable_group_count"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "job_path": str(Path(job["output_dir"]) / "qwen_grpo_training_job.json"),
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


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen GRPO training job.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_grpo_training_job.json or directory")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def run_main() -> int:
    args = parse_run_args()
    result = run_qwen_grpo_training_job(
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


if __name__ == "__main__":
    raise SystemExit(main())
