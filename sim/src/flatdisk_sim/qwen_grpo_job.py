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
QWEN_GRPO_COMPLETION_EVAL_JOB_SCHEMA = "flatdisk.qwen_grpo_completion_eval_job.v1"
QWEN_GRPO_COMPLETION_EVAL_RESULT_SCHEMA = "flatdisk.qwen_grpo_completion_eval_result.v1"
QWEN_GRPO_ADAPTER_EFFECT_JOB_SCHEMA = "flatdisk.qwen_grpo_adapter_effect_job.v1"
QWEN_GRPO_ADAPTER_EFFECT_RESULT_SCHEMA = "flatdisk.qwen_grpo_adapter_effect_result.v1"
QWEN_GRPO_ACTION_LIKELIHOOD_JOB_SCHEMA = "flatdisk.qwen_grpo_action_likelihood_job.v1"
QWEN_GRPO_ACTION_LIKELIHOOD_RESULT_SCHEMA = "flatdisk.qwen_grpo_action_likelihood_result.v1"
DEFAULT_EVAL_REQUIRED_PACKAGES = ["peft", "pillow", "torch", "torchvision", "transformers"]
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
    zero_reward_exact_action_bonus: float = 0.0,
    balance_reference_tools: bool = False,
    max_balance_multiplier: int = 4,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    if zero_reward_exact_action_bonus < 0.0:
        raise ValueError("zero_reward_exact_action_bonus must be non-negative")
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
        zero_reward_exact_action_bonus=zero_reward_exact_action_bonus,
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
            "zero_reward_exact_action_bonus": zero_reward_exact_action_bonus,
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
            "exact_action_reward_shaping": {
                "enabled": zero_reward_exact_action_bonus > 0.0,
                "zero_reward_exact_action_bonus": zero_reward_exact_action_bonus,
                "applies_only_when_candidate_step_reward_is_zero": True,
                "requires_candidate_step_reward_present": True,
            },
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


def plan_qwen_grpo_completion_eval(
    training_job_input: Path,
    *,
    output_dir: Path,
    model_id: str | None = None,
    adapter_path: Path | None = None,
    max_samples: int | None = None,
    sample_offset: int = 0,
    sample_stride: int = 1,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    top_p: float = 1.0,
    zero_reward_exact_action_bonus: float | None = None,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when set")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("top_p must be in (0, 1]")

    training_job_path = _resolve_qwen_grpo_job(training_job_input)
    training_job = json.loads(training_job_path.read_text(encoding="utf-8"))
    dataset_path = _job_path(training_job, "qwen_grpo_trl_dataset_jsonl", relative_to=training_job_path.parent)
    source_records = _read_jsonl_if_exists(dataset_path)
    eval_records = _select_eval_records(
        source_records,
        max_samples=max_samples,
        sample_offset=sample_offset,
        sample_stride=sample_stride,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dataset_path = output_dir / "qwen_grpo_completion_eval_dataset.jsonl"
    eval_script_path = output_dir / "eval_qwen_grpo_completions.py"
    completion_log_path = output_dir / "completion_eval_samples.jsonl"
    _write_jsonl(eval_dataset_path, eval_records)
    _write_eval_script(eval_script_path)
    training_args = training_job.get("training_args") if isinstance(training_job.get("training_args"), dict) else {}
    resolved_model_id = model_id or str(training_job.get("model_id") or DEFAULT_MODEL_ID)
    resolved_bonus = (
        float(zero_reward_exact_action_bonus)
        if zero_reward_exact_action_bonus is not None
        else float(_optional_float(training_args.get("zero_reward_exact_action_bonus")) or 0.0)
    )
    if resolved_bonus < 0.0:
        raise ValueError("zero_reward_exact_action_bonus must be non-negative")
    launch_argv = _eval_launch_argv(
        eval_script_path=eval_script_path,
        dataset_path=eval_dataset_path,
        model_id=resolved_model_id,
        output_dir=output_dir,
        completion_log_path=completion_log_path,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        zero_reward_exact_action_bonus=resolved_bonus,
    )
    validation = _validate_completion_eval_job_inputs(
        training_job,
        dataset_path=dataset_path,
        source_records=source_records,
        eval_records=eval_records,
        require_existing_images=require_existing_images,
    )
    if adapter_path is not None and not adapter_path.exists():
        validation["blockers"].append(f"missing adapter_path: {adapter_path}")
    job = {
        "schema": QWEN_GRPO_COMPLETION_EVAL_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "source_training_job": str(training_job_path),
        "source_training_job_schema": training_job.get("schema"),
        "source_training_job_status": training_job.get("status"),
        "source_dataset_jsonl": str(dataset_path) if dataset_path else "",
        "qwen_grpo_completion_eval_dataset_jsonl": str(eval_dataset_path),
        "output_dir": str(output_dir),
        "eval_script": str(eval_script_path),
        "eval_script_sha256": _sha256_file(eval_script_path),
        "completion_log_jsonl": str(completion_log_path),
        "result_path": str(output_dir / "qwen_grpo_completion_eval_result.json"),
        "evaluation_method": "heldout_qwen_completion_eval",
        "model_id": resolved_model_id,
        "adapter_path": str(adapter_path) if adapter_path else "",
        "required_packages": DEFAULT_EVAL_REQUIRED_PACKAGES,
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "eval_args": {
            "max_samples": max_samples,
            "sample_offset": sample_offset,
            "sample_stride": sample_stride,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "zero_reward_exact_action_bonus": resolved_bonus,
        },
        "dataset": validation["dataset"],
        "audit": {
            "prompt_only_dataset": True,
            "reference_actions_are_sidecar_columns": True,
            "reward_labels_excluded_from_messages": True,
            "evaluation_uses_reference_actions_only_for_scoring": True,
            "online_environment_reward": False,
            "require_existing_images": require_existing_images,
            "adapter_optional": True,
        },
    }
    (output_dir / "qwen_grpo_completion_eval_job.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return job


def run_qwen_grpo_completion_eval_job(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_qwen_grpo_completion_eval_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_grpo_completion_eval_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    blockers = _completion_eval_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")
    result: dict[str, Any] = {
        "schema": QWEN_GRPO_COMPLETION_EVAL_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "adapter_path": job.get("adapter_path"),
        "sample_count": (job.get("dataset") or {}).get("eval_sample_count"),
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
        result["blockers"] = [f"completion eval command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_completion_log_summary(result, job, job_path=job_path)
    _write_result(result_path, result)
    return result


def plan_qwen_grpo_adapter_effect_check(
    completion_eval_job_input: Path,
    *,
    output_dir: Path,
    adapter_path: Path,
    model_id: str | None = None,
    max_samples: int | None = None,
    sample_offset: int = 0,
    sample_stride: int = 1,
    top_k: int = 5,
    delta_threshold: float = 1e-6,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when set")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if delta_threshold < 0.0:
        raise ValueError("delta_threshold must be non-negative")
    completion_eval_job_path = _resolve_qwen_grpo_completion_eval_job(completion_eval_job_input)
    completion_eval_job = json.loads(completion_eval_job_path.read_text(encoding="utf-8"))
    source_dataset_path = _job_path(
        completion_eval_job,
        "qwen_grpo_completion_eval_dataset_jsonl",
        relative_to=completion_eval_job_path.parent,
    )
    source_records = _read_jsonl_if_exists(source_dataset_path)
    effect_records = _select_eval_records(
        source_records,
        max_samples=max_samples,
        sample_offset=sample_offset,
        sample_stride=sample_stride,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "qwen_grpo_adapter_effect_dataset.jsonl"
    script_path = output_dir / "check_qwen_grpo_adapter_effect.py"
    effect_log_path = output_dir / "adapter_effect_samples.jsonl"
    _write_jsonl(dataset_path, effect_records)
    _write_adapter_effect_script(script_path)

    resolved_model_id = model_id or str(completion_eval_job.get("model_id") or DEFAULT_MODEL_ID)
    launch_argv = _adapter_effect_launch_argv(
        script_path=script_path,
        dataset_path=dataset_path,
        model_id=resolved_model_id,
        output_dir=output_dir,
        adapter_path=adapter_path,
        effect_log_path=effect_log_path,
        top_k=top_k,
        delta_threshold=delta_threshold,
    )
    validation = _validate_adapter_effect_job_inputs(
        completion_eval_job,
        dataset_path=source_dataset_path,
        source_record_count=len(source_records),
        eval_records=effect_records,
        adapter_path=adapter_path,
        require_existing_images=require_existing_images,
    )
    job = {
        "schema": QWEN_GRPO_ADAPTER_EFFECT_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "source_completion_eval_job": str(completion_eval_job_path),
        "source_completion_eval_job_schema": completion_eval_job.get("schema"),
        "source_completion_eval_job_status": completion_eval_job.get("status"),
        "source_dataset_jsonl": str(source_dataset_path) if source_dataset_path else "",
        "qwen_grpo_adapter_effect_dataset_jsonl": str(dataset_path),
        "output_dir": str(output_dir),
        "adapter_effect_script": str(script_path),
        "adapter_effect_script_sha256": _sha256_file(script_path),
        "adapter_effect_log_jsonl": str(effect_log_path),
        "result_path": str(output_dir / "qwen_grpo_adapter_effect_result.json"),
        "evaluation_method": "qwen_peft_adapter_effect_logit_check",
        "model_id": resolved_model_id,
        "adapter_path": str(adapter_path),
        "required_packages": DEFAULT_EVAL_REQUIRED_PACKAGES,
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "adapter_effect_args": {
            "max_samples": max_samples,
            "sample_offset": sample_offset,
            "sample_stride": sample_stride,
            "top_k": top_k,
            "delta_threshold": delta_threshold,
        },
        "dataset": validation["dataset"],
        "audit": {
            "prompt_only_dataset": True,
            "reference_actions_are_sidecar_columns": True,
            "reward_labels_excluded_from_messages": True,
            "reference_actions_used_only_for_reporting": True,
            "online_environment_reward": False,
            "require_existing_images": require_existing_images,
            "adapter_required": True,
            "adapter_compared_by_disable_enable_on_same_peft_model": True,
        },
    }
    (output_dir / "qwen_grpo_adapter_effect_job.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return job


def run_qwen_grpo_adapter_effect_check_job(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_qwen_grpo_adapter_effect_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_grpo_adapter_effect_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    blockers = _adapter_effect_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")
    result: dict[str, Any] = {
        "schema": QWEN_GRPO_ADAPTER_EFFECT_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "adapter_path": job.get("adapter_path"),
        "sample_count": (job.get("dataset") or {}).get("eval_sample_count"),
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
        result["blockers"] = [f"adapter effect command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_adapter_effect_log_summary(result, job, job_path=job_path)
    _write_result(result_path, result)
    return result


def plan_qwen_grpo_action_likelihood_check(
    completion_eval_job_input: Path,
    *,
    output_dir: Path,
    adapter_path: Path,
    model_id: str | None = None,
    max_samples: int | None = None,
    sample_offset: int = 0,
    sample_stride: int = 1,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if sample_stride < 1:
        raise ValueError("sample_stride must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when set")
    completion_eval_job_path = _resolve_qwen_grpo_completion_eval_job(completion_eval_job_input)
    completion_eval_job = json.loads(completion_eval_job_path.read_text(encoding="utf-8"))
    source_dataset_path = _job_path(
        completion_eval_job,
        "qwen_grpo_completion_eval_dataset_jsonl",
        relative_to=completion_eval_job_path.parent,
    )
    source_records = _read_jsonl_if_exists(source_dataset_path)
    likelihood_records = _select_eval_records(
        source_records,
        max_samples=max_samples,
        sample_offset=sample_offset,
        sample_stride=sample_stride,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "qwen_grpo_action_likelihood_dataset.jsonl"
    script_path = output_dir / "score_qwen_grpo_action_likelihood.py"
    likelihood_log_path = output_dir / "action_likelihood_samples.jsonl"
    progress_log_path = output_dir / "action_likelihood_progress.jsonl"
    _write_jsonl(dataset_path, likelihood_records)
    _write_action_likelihood_script(script_path)

    resolved_model_id = model_id or str(completion_eval_job.get("model_id") or DEFAULT_MODEL_ID)
    launch_argv = _action_likelihood_launch_argv(
        script_path=script_path,
        dataset_path=dataset_path,
        model_id=resolved_model_id,
        output_dir=output_dir,
        adapter_path=adapter_path,
        likelihood_log_path=likelihood_log_path,
        progress_log_path=progress_log_path,
    )
    validation = _validate_action_likelihood_job_inputs(
        completion_eval_job,
        dataset_path=source_dataset_path,
        source_record_count=len(source_records),
        eval_records=likelihood_records,
        adapter_path=adapter_path,
        require_existing_images=require_existing_images,
    )
    job = {
        "schema": QWEN_GRPO_ACTION_LIKELIHOOD_JOB_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not validation["blockers"] else "not_ready",
        "blockers": validation["blockers"],
        "warnings": validation["warnings"],
        "source_completion_eval_job": str(completion_eval_job_path),
        "source_completion_eval_job_schema": completion_eval_job.get("schema"),
        "source_completion_eval_job_status": completion_eval_job.get("status"),
        "source_dataset_jsonl": str(source_dataset_path) if source_dataset_path else "",
        "qwen_grpo_action_likelihood_dataset_jsonl": str(dataset_path),
        "output_dir": str(output_dir),
        "action_likelihood_script": str(script_path),
        "action_likelihood_script_sha256": _sha256_file(script_path),
        "action_likelihood_log_jsonl": str(likelihood_log_path),
        "action_likelihood_progress_jsonl": str(progress_log_path),
        "result_path": str(output_dir / "qwen_grpo_action_likelihood_result.json"),
        "evaluation_method": "qwen_peft_action_likelihood_check",
        "model_id": resolved_model_id,
        "adapter_path": str(adapter_path),
        "required_packages": DEFAULT_EVAL_REQUIRED_PACKAGES,
        "launch_argv": launch_argv,
        "launch_command": _argv_to_command(launch_argv),
        "action_likelihood_args": {
            "max_samples": max_samples,
            "sample_offset": sample_offset,
            "sample_stride": sample_stride,
            "target_format": "compact_action_json",
        },
        "dataset": validation["dataset"],
        "audit": {
            "prompt_only_dataset": True,
            "reference_actions_are_sidecar_columns": True,
            "reward_labels_excluded_from_messages": True,
            "reference_action_json_appended_only_as_teacher_forced_target": True,
            "reference_action_used_for_offline_teacher_forced_target_scoring": True,
            "online_environment_reward": False,
            "require_existing_images": require_existing_images,
            "adapter_required": True,
            "adapter_compared_by_disable_enable_on_same_peft_model": True,
        },
    }
    (output_dir / "qwen_grpo_action_likelihood_job.json").write_text(
        json.dumps(job, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return job


def run_qwen_grpo_action_likelihood_check_job(
    job_input: Path,
    *,
    result_dir: Path | None = None,
    dry_run: bool = False,
    check_dependencies: bool = True,
    timeout_s: float | None = None,
    launch_command: str | None = None,
    tail_chars: int = 4000,
) -> dict[str, Any]:
    job_path = _resolve_qwen_grpo_action_likelihood_job(job_input)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_dir = result_dir or Path(str(job.get("output_dir") or job_path.parent))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "qwen_grpo_action_likelihood_result.json"
    launch_argv = _job_launch_argv(job, launch_command_override=launch_command)
    blockers = _action_likelihood_job_blockers(job, job_path=job_path, check_dependencies=check_dependencies)
    if not launch_argv:
        blockers.append("missing launch_command")
    result: dict[str, Any] = {
        "schema": QWEN_GRPO_ACTION_LIKELIHOOD_RESULT_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "completed_at": None,
        "status": "not_ready",
        "dry_run": dry_run,
        "job_manifest": str(job_path),
        "result_path": str(result_path),
        "model_id": job.get("model_id"),
        "adapter_path": job.get("adapter_path"),
        "sample_count": (job.get("dataset") or {}).get("eval_sample_count"),
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
        _attach_action_likelihood_log_summary(result, job, job_path=job_path)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _write_result(result_path, result)
        return result
    if dry_run:
        _attach_action_likelihood_log_summary(result, job, job_path=job_path)
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
        result["blockers"] = [f"action likelihood command timed out after {timeout_s} second(s)"]
        result["stdout_tail"] = _tail(_decode_timeout_output(exc.stdout), tail_chars)
        result["stderr_tail"] = _tail(_decode_timeout_output(exc.stderr), tail_chars)
    finally:
        result["duration_s"] = round(monotonic() - start, 3)
        result["completed_at"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
        _attach_action_likelihood_log_summary(result, job, job_path=job_path)
    _write_result(result_path, result)
    return result


def _attach_completion_log_summary(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    completion_log = _job_path(job, "completion_log_jsonl", relative_to=job_path.parent)
    result["completion_log_jsonl"] = str(completion_log) if completion_log else ""
    result["completion_log_sample_count"] = _count_lines(completion_log) if completion_log and completion_log.exists() else 0
    result["completion_log_metrics"] = _completion_log_metrics(completion_log) if completion_log and completion_log.exists() else {}


def _attach_adapter_effect_log_summary(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    effect_log = _job_path(job, "adapter_effect_log_jsonl", relative_to=job_path.parent)
    result["adapter_effect_log_jsonl"] = str(effect_log) if effect_log else ""
    result["adapter_effect_log_sample_count"] = _count_lines(effect_log) if effect_log and effect_log.exists() else 0
    result["adapter_effect_log_metrics"] = (
        _adapter_effect_log_metrics(effect_log) if effect_log and effect_log.exists() else {}
    )


def _attach_action_likelihood_log_summary(result: dict[str, Any], job: dict[str, Any], *, job_path: Path) -> None:
    likelihood_log = _job_path(job, "action_likelihood_log_jsonl", relative_to=job_path.parent)
    progress_log = _job_path(job, "action_likelihood_progress_jsonl", relative_to=job_path.parent)
    result["action_likelihood_log_jsonl"] = str(likelihood_log) if likelihood_log else ""
    result["action_likelihood_log_sample_count"] = (
        _count_lines(likelihood_log) if likelihood_log and likelihood_log.exists() else 0
    )
    result["action_likelihood_log_metrics"] = (
        _action_likelihood_log_metrics(likelihood_log) if likelihood_log and likelihood_log.exists() else {}
    )
    result["action_likelihood_progress_jsonl"] = str(progress_log) if progress_log else ""
    result["action_likelihood_progress_count"] = (
        _count_lines(progress_log) if progress_log and progress_log.exists() else 0
    )
    result["action_likelihood_progress_tail"] = (
        _jsonl_tail(progress_log, limit=8) if progress_log and progress_log.exists() else []
    )


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
    expected_tool_counts: Counter[str] = Counter()
    parsed_tool_counts: Counter[str] = Counter()
    exact_by_expected_tool: Counter[str] = Counter()
    tool_match_by_expected_tool: Counter[str] = Counter()
    for record in records:
        parsed_action = record.get("parsed_action") if isinstance(record.get("parsed_action"), dict) else {}
        expected_action = (
            record.get("expected_action")
            if isinstance(record.get("expected_action"), dict)
            else _reference_action_from_canonical(record.get("reference_action_canonical"))
        )
        expected_tool = _action_tool(expected_action) or "unknown"
        parsed_tool = _action_tool(parsed_action) or "unparsed"
        expected_tool_counts[expected_tool] += 1
        parsed_tool_counts[parsed_tool] += 1
        if parsed_action:
            parsed_action_count += 1
        tool_match = (
            bool(record.get("tool_match"))
            if "tool_match" in record
            else _action_tool(parsed_action) == _action_tool(expected_action)
        )
        if parsed_action and tool_match:
            tool_match_count += 1
            tool_match_by_expected_tool[expected_tool] += 1
        arg_match_fraction = _optional_float(record.get("arg_match_fraction"))
        if arg_match_fraction is None:
            arg_match_fraction = (
                _arg_match_fraction(parsed_action, expected_action) if parsed_action and tool_match else 0.0
            )
        arg_match_fractions.append(arg_match_fraction)
        exact_reference = _canonical_json(parsed_action) == record.get("reference_action_canonical")
        if exact_reference:
            exact_reference_count += 1
            exact_by_expected_tool[expected_tool] += 1
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
        "expected_tool_counts": dict(sorted(expected_tool_counts.items())),
        "parsed_tool_counts": dict(sorted(parsed_tool_counts.items())),
        "exact_reference_action_count_by_expected_tool": dict(sorted(exact_by_expected_tool.items())),
        "tool_match_count_by_expected_tool": dict(sorted(tool_match_by_expected_tool.items())),
    }


def _adapter_effect_log_metrics(path: Path) -> dict[str, Any]:
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
    expected_tool_counts: Counter[str] = Counter()
    nonzero_by_expected_tool: Counter[str] = Counter()
    top1_changed_by_expected_tool: Counter[str] = Counter()
    max_deltas = []
    mean_deltas = []
    l2_deltas = []
    kl_adapter_from_base = []
    kl_base_from_adapter = []
    top_k_jaccards = []
    nonzero_delta_count = 0
    top1_changed_count = 0
    for record in records:
        expected_tool = str(record.get("expected_tool") or "unknown")
        expected_tool_counts[expected_tool] += 1
        max_delta = _optional_float(record.get("max_abs_logit_delta")) or 0.0
        mean_delta = _optional_float(record.get("mean_abs_logit_delta")) or 0.0
        l2_delta = _optional_float(record.get("l2_logit_delta")) or 0.0
        kl_ab = _optional_float(record.get("kl_adapter_from_base")) or 0.0
        kl_ba = _optional_float(record.get("kl_base_from_adapter")) or 0.0
        jaccard = _optional_float(record.get("top_k_jaccard")) or 0.0
        max_deltas.append(max_delta)
        mean_deltas.append(mean_delta)
        l2_deltas.append(l2_delta)
        kl_adapter_from_base.append(kl_ab)
        kl_base_from_adapter.append(kl_ba)
        top_k_jaccards.append(jaccard)
        if bool(record.get("nonzero_delta")):
            nonzero_delta_count += 1
            nonzero_by_expected_tool[expected_tool] += 1
        if bool(record.get("top1_changed")):
            top1_changed_count += 1
            top1_changed_by_expected_tool[expected_tool] += 1
    return {
        "sample_count": len(records),
        "malformed_line_count": malformed_count,
        "nonzero_delta_count": nonzero_delta_count,
        "nonzero_delta_rate": round(nonzero_delta_count / len(records), 6) if records else 0.0,
        "top1_changed_count": top1_changed_count,
        "top1_changed_rate": round(top1_changed_count / len(records), 6) if records else 0.0,
        "max_abs_logit_delta_max": round(max(max_deltas), 9) if max_deltas else 0.0,
        "max_abs_logit_delta_mean": round(sum(max_deltas) / len(max_deltas), 9) if max_deltas else 0.0,
        "mean_abs_logit_delta_mean": round(sum(mean_deltas) / len(mean_deltas), 9) if mean_deltas else 0.0,
        "l2_logit_delta_mean": round(sum(l2_deltas) / len(l2_deltas), 9) if l2_deltas else 0.0,
        "kl_adapter_from_base_mean": round(sum(kl_adapter_from_base) / len(kl_adapter_from_base), 9)
        if kl_adapter_from_base
        else 0.0,
        "kl_base_from_adapter_mean": round(sum(kl_base_from_adapter) / len(kl_base_from_adapter), 9)
        if kl_base_from_adapter
        else 0.0,
        "top_k_jaccard_mean": round(sum(top_k_jaccards) / len(top_k_jaccards), 9) if top_k_jaccards else 0.0,
        "expected_tool_counts": dict(sorted(expected_tool_counts.items())),
        "nonzero_delta_count_by_expected_tool": dict(sorted(nonzero_by_expected_tool.items())),
        "top1_changed_count_by_expected_tool": dict(sorted(top1_changed_by_expected_tool.items())),
    }


def _action_likelihood_log_metrics(path: Path) -> dict[str, Any]:
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
    expected_tool_counts: Counter[str] = Counter()
    target_delta_by_tool: dict[str, list[float]] = defaultdict(list)
    tool_delta_by_tool: dict[str, list[float]] = defaultdict(list)
    target_deltas = []
    tool_deltas = []
    target_token_counts = []
    tool_token_counts = []
    target_improved_count = 0
    tool_improved_count = 0
    tool_span_found_count = 0
    zero_tool_token_count = 0
    first_token_deltas = []
    first_token_improved_count = 0
    target_nll_deltas = []
    for record in records:
        expected_tool = str(record.get("expected_tool") or "unknown")
        expected_tool_counts[expected_tool] += 1
        target_delta = _optional_float(record.get("target_mean_logprob_delta"))
        tool_delta = _optional_float(record.get("tool_mean_logprob_delta"))
        first_token_delta = _optional_float(record.get("first_target_token_logprob_delta"))
        target_nll_delta = _optional_float(record.get("target_nll_delta"))
        target_token_count = _optional_int(record.get("target_token_count"))
        tool_token_count = _optional_int(record.get("tool_token_count"))
        target_token_counts.append(target_token_count)
        tool_token_counts.append(tool_token_count)
        if bool(record.get("tool_span_found")):
            tool_span_found_count += 1
        if target_delta is not None:
            target_deltas.append(target_delta)
            target_delta_by_tool[expected_tool].append(target_delta)
            if target_delta > 0:
                target_improved_count += 1
        if tool_delta is not None:
            tool_deltas.append(tool_delta)
            tool_delta_by_tool[expected_tool].append(tool_delta)
            if tool_delta > 0:
                tool_improved_count += 1
        if first_token_delta is not None:
            first_token_deltas.append(first_token_delta)
            if first_token_delta > 0:
                first_token_improved_count += 1
        if target_nll_delta is not None:
            target_nll_deltas.append(target_nll_delta)
        if tool_token_count == 0:
            zero_tool_token_count += 1
    return {
        "sample_count": len(records),
        "malformed_line_count": malformed_count,
        "expected_tool_counts": dict(sorted(expected_tool_counts.items())),
        "target_token_count_mean": _rounded_mean(target_token_counts),
        "tool_token_count_mean": _rounded_mean(tool_token_counts),
        "zero_tool_token_count": zero_tool_token_count,
        "tool_span_found_count": tool_span_found_count,
        "tool_span_found_rate": round(tool_span_found_count / len(records), 6) if records else 0.0,
        "target_mean_logprob_delta_mean": _rounded_mean(target_deltas, digits=9),
        "target_mean_logprob_delta_min": round(min(target_deltas), 9) if target_deltas else 0.0,
        "target_mean_logprob_delta_max": round(max(target_deltas), 9) if target_deltas else 0.0,
        "target_mean_logprob_improved_count": target_improved_count,
        "target_mean_logprob_improved_rate": round(target_improved_count / len(records), 6) if records else 0.0,
        "target_nll_delta_mean": _rounded_mean(target_nll_deltas, digits=9),
        "first_target_token_logprob_delta_mean": _rounded_mean(first_token_deltas, digits=9),
        "first_target_token_logprob_improved_count": first_token_improved_count,
        "first_target_token_logprob_improved_rate": round(first_token_improved_count / len(records), 6)
        if records
        else 0.0,
        "tool_mean_logprob_delta_mean": _rounded_mean(tool_deltas, digits=9),
        "tool_mean_logprob_delta_min": round(min(tool_deltas), 9) if tool_deltas else 0.0,
        "tool_mean_logprob_delta_max": round(max(tool_deltas), 9) if tool_deltas else 0.0,
        "tool_mean_logprob_improved_count": tool_improved_count,
        "tool_mean_logprob_improved_rate": round(tool_improved_count / len(records), 6) if records else 0.0,
        "target_mean_logprob_delta_by_expected_tool": {
            tool: _rounded_mean(values, digits=9) for tool, values in sorted(target_delta_by_tool.items())
        },
        "tool_mean_logprob_delta_by_expected_tool": {
            tool: _rounded_mean(values, digits=9) for tool, values in sorted(tool_delta_by_tool.items())
        },
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
                "candidate_step_reward_present": _optional_float(reward.get("candidate_step_reward")) is not None,
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


def _rounded_mean(values: list[float | int], *, digits: int = 6) -> float:
    return round(sum(values) / len(values), digits) if values else 0.0


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


def _select_eval_records(
    records: list[dict[str, Any]],
    *,
    max_samples: int | None,
    sample_offset: int,
    sample_stride: int,
) -> list[dict[str, Any]]:
    selected = [record for index, record in enumerate(records) if index >= sample_offset and (index - sample_offset) % sample_stride == 0]
    return selected[:max_samples] if max_samples is not None else selected


def _validate_completion_eval_job_inputs(
    training_job: dict[str, Any],
    *,
    dataset_path: Path | None,
    source_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    require_existing_images: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if training_job.get("schema") != QWEN_GRPO_TRAINING_JOB_SCHEMA:
        blockers.append(f"unexpected source training job schema: {training_job.get('schema')}")
    if training_job.get("status") != "ready":
        blockers.append(f"source training job is not ready: {training_job.get('status')}")
    if dataset_path is None or not dataset_path.exists():
        blockers.append(f"missing source qwen_grpo_trl_dataset_jsonl: {training_job.get('qwen_grpo_trl_dataset_jsonl')}")
    if not source_records:
        blockers.append("source GRPO TRL dataset contains no prompt samples")
    if not eval_records:
        blockers.append("completion eval dataset contains no prompt samples")

    missing_images = sorted(
        {
            str(path)
            for record in eval_records
            for path in record.get("image_paths", [])
            if not Path(str(path)).exists()
        }
    )
    malformed_count = sum(
        1
        for record in eval_records
        if not isinstance(record.get("prompt_messages") or record.get("prompt"), list)
        or not isinstance(record.get("reference_action_json"), dict)
        or not record.get("reference_action_canonical")
    )
    forbidden_hits = _forbidden_prompt_message_hits(eval_records)
    sidecar_leak_hits = _sidecar_prompt_leak_hits(eval_records)
    if malformed_count:
        blockers.append(f"{malformed_count} completion eval sample(s) are malformed")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} completion eval image reference(s) are missing")
    elif missing_images:
        warnings.append(f"{len(missing_images)} completion eval image reference(s) are missing")
    if forbidden_hits:
        blockers.append("completion eval prompt messages contain forbidden privileged token(s): " + ", ".join(forbidden_hits))
    if sidecar_leak_hits:
        blockers.append("completion eval prompt messages contain scoring sidecar token(s): " + ", ".join(sidecar_leak_hits))
    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "source_sample_count": len(source_records),
            "eval_sample_count": len(eval_records),
            "image_reference_count": sum(
                len(record.get("image_paths", []))
                for record in eval_records
                if isinstance(record.get("image_paths"), list)
            ),
            "missing_image_count": len(missing_images),
            "missing_images": missing_images,
            "forbidden_model_token_hits": forbidden_hits,
            "sidecar_prompt_leak_hits": sidecar_leak_hits,
            "reference_action_tool_counts": _grpo_dataset_action_summary(eval_records)["reference_action_tool_counts"],
        },
    }


def _validate_adapter_effect_job_inputs(
    completion_eval_job: dict[str, Any],
    *,
    dataset_path: Path | None,
    source_record_count: int,
    eval_records: list[dict[str, Any]],
    adapter_path: Path,
    require_existing_images: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if completion_eval_job.get("schema") != QWEN_GRPO_COMPLETION_EVAL_JOB_SCHEMA:
        blockers.append(f"unexpected source completion eval job schema: {completion_eval_job.get('schema')}")
    if completion_eval_job.get("status") != "ready":
        blockers.append(f"source completion eval job is not ready: {completion_eval_job.get('status')}")
    if dataset_path is None or not dataset_path.exists():
        blockers.append(
            "missing source qwen_grpo_completion_eval_dataset_jsonl: "
            f"{completion_eval_job.get('qwen_grpo_completion_eval_dataset_jsonl')}"
        )
    if not eval_records:
        blockers.append("adapter effect dataset contains no prompt samples")

    missing_images = sorted(
        {
            str(path)
            for record in eval_records
            for path in record.get("image_paths", [])
            if not Path(str(path)).exists()
        }
    )
    malformed_count = sum(
        1
        for record in eval_records
        if not isinstance(record.get("prompt_messages") or record.get("prompt"), list)
        or not isinstance(record.get("reference_action_json"), dict)
        or not record.get("reference_action_canonical")
    )
    forbidden_hits = _forbidden_prompt_message_hits(eval_records)
    sidecar_leak_hits = _sidecar_prompt_leak_hits(eval_records)
    adapter_blockers = _adapter_path_blockers(adapter_path)
    if malformed_count:
        blockers.append(f"{malformed_count} adapter effect sample(s) are malformed")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} adapter effect image reference(s) are missing")
    elif missing_images:
        warnings.append(f"{len(missing_images)} adapter effect image reference(s) are missing")
    if forbidden_hits:
        blockers.append("adapter effect prompt messages contain forbidden privileged token(s): " + ", ".join(forbidden_hits))
    if sidecar_leak_hits:
        blockers.append("adapter effect prompt messages contain scoring sidecar token(s): " + ", ".join(sidecar_leak_hits))
    blockers.extend(adapter_blockers)
    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "source_sample_count": source_record_count,
            "eval_sample_count": len(eval_records),
            "image_reference_count": sum(
                len(record.get("image_paths", []))
                for record in eval_records
                if isinstance(record.get("image_paths"), list)
            ),
            "missing_image_count": len(missing_images),
            "missing_images": missing_images,
            "forbidden_model_token_hits": forbidden_hits,
            "sidecar_prompt_leak_hits": sidecar_leak_hits,
            "reference_action_tool_counts": _grpo_dataset_action_summary(eval_records)["reference_action_tool_counts"],
            "adapter_path_blockers": adapter_blockers,
        },
    }


def _validate_action_likelihood_job_inputs(
    completion_eval_job: dict[str, Any],
    *,
    dataset_path: Path | None,
    source_record_count: int,
    eval_records: list[dict[str, Any]],
    adapter_path: Path,
    require_existing_images: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if completion_eval_job.get("schema") != QWEN_GRPO_COMPLETION_EVAL_JOB_SCHEMA:
        blockers.append(f"unexpected source completion eval job schema: {completion_eval_job.get('schema')}")
    if completion_eval_job.get("status") != "ready":
        blockers.append(f"source completion eval job is not ready: {completion_eval_job.get('status')}")
    if dataset_path is None or not dataset_path.exists():
        blockers.append(
            "missing source qwen_grpo_completion_eval_dataset_jsonl: "
            f"{completion_eval_job.get('qwen_grpo_completion_eval_dataset_jsonl')}"
        )
    if not eval_records:
        blockers.append("action likelihood dataset contains no prompt samples")

    missing_images = sorted(
        {
            str(path)
            for record in eval_records
            for path in record.get("image_paths", [])
            if not Path(str(path)).exists()
        }
    )
    malformed_count = sum(
        1
        for record in eval_records
        if not isinstance(record.get("prompt_messages") or record.get("prompt"), list)
        or not isinstance(record.get("reference_action_json"), dict)
        or not record.get("reference_action_canonical")
    )
    empty_target_count = sum(
        1
        for record in eval_records
        if not _compact_action_response(record.get("reference_action_json")).strip()
    )
    forbidden_hits = _forbidden_prompt_message_hits(eval_records)
    sidecar_leak_hits = _sidecar_prompt_leak_hits(eval_records)
    adapter_blockers = _adapter_path_blockers(adapter_path)
    if malformed_count:
        blockers.append(f"{malformed_count} action likelihood sample(s) are malformed")
    if empty_target_count:
        blockers.append(f"{empty_target_count} action likelihood sample(s) have empty target action JSON")
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} action likelihood image reference(s) are missing")
    elif missing_images:
        warnings.append(f"{len(missing_images)} action likelihood image reference(s) are missing")
    if forbidden_hits:
        blockers.append("action likelihood prompt messages contain forbidden privileged token(s): " + ", ".join(forbidden_hits))
    if sidecar_leak_hits:
        blockers.append("action likelihood prompt messages contain scoring sidecar token(s): " + ", ".join(sidecar_leak_hits))
    blockers.extend(adapter_blockers)
    return {
        "blockers": blockers,
        "warnings": warnings,
        "dataset": {
            "source_sample_count": source_record_count,
            "eval_sample_count": len(eval_records),
            "image_reference_count": sum(
                len(record.get("image_paths", []))
                for record in eval_records
                if isinstance(record.get("image_paths"), list)
            ),
            "missing_image_count": len(missing_images),
            "missing_images": missing_images,
            "forbidden_model_token_hits": forbidden_hits,
            "sidecar_prompt_leak_hits": sidecar_leak_hits,
            "reference_action_tool_counts": _grpo_dataset_action_summary(eval_records)["reference_action_tool_counts"],
            "adapter_path_blockers": adapter_blockers,
            "empty_target_count": empty_target_count,
        },
    }


def _adapter_path_blockers(adapter_path: Path) -> list[str]:
    if not adapter_path.exists():
        return [f"missing adapter_path: {adapter_path}"]
    if not adapter_path.is_dir():
        return [f"adapter_path must be a PEFT adapter directory: {adapter_path}"]
    blockers = []
    if not (adapter_path / "adapter_config.json").exists():
        blockers.append(f"missing adapter_config.json under adapter_path: {adapter_path}")
    if not any((adapter_path / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        blockers.append(f"missing adapter model weights under adapter_path: {adapter_path}")
    return blockers


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


def _resolve_qwen_grpo_completion_eval_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_grpo_completion_eval_job.json":
        return path
    candidates = [
        path / "qwen_grpo_completion_eval_job.json",
        path / "qwen_grpo_completion_eval" / "qwen_grpo_completion_eval_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_grpo_completion_eval_job.json under {job_input}")


def _resolve_qwen_grpo_adapter_effect_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_grpo_adapter_effect_job.json":
        return path
    candidates = [
        path / "qwen_grpo_adapter_effect_job.json",
        path / "qwen_grpo_adapter_effect" / "qwen_grpo_adapter_effect_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_grpo_adapter_effect_job.json under {job_input}")


def _resolve_qwen_grpo_action_likelihood_job(job_input: Path) -> Path:
    path = job_input.expanduser()
    if path.is_file() and path.name == "qwen_grpo_action_likelihood_job.json":
        return path
    candidates = [
        path / "qwen_grpo_action_likelihood_job.json",
        path / "qwen_grpo_action_likelihood" / "qwen_grpo_action_likelihood_job.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not find qwen_grpo_action_likelihood_job.json under {job_input}")


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


def _completion_eval_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_GRPO_COMPLETION_EVAL_JOB_SCHEMA:
        blockers.append(f"unexpected completion eval job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"completion eval job is not ready: {job.get('status')}")
    eval_script = _job_path(job, "eval_script", relative_to=job_path.parent)
    if eval_script is None or not eval_script.exists():
        blockers.append(f"missing eval_script: {job.get('eval_script')}")
    dataset_path = _job_path(job, "qwen_grpo_completion_eval_dataset_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(
            f"missing qwen_grpo_completion_eval_dataset_jsonl: {job.get('qwen_grpo_completion_eval_dataset_jsonl')}"
        )
    adapter_path_value = str(job.get("adapter_path") or "")
    if adapter_path_value and not Path(adapter_path_value).exists():
        blockers.append(f"missing adapter_path: {adapter_path_value}")
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required completion eval package(s): " + ", ".join(missing_packages))
    return blockers


def _adapter_effect_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_GRPO_ADAPTER_EFFECT_JOB_SCHEMA:
        blockers.append(f"unexpected adapter effect job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"adapter effect job is not ready: {job.get('status')}")
    script_path = _job_path(job, "adapter_effect_script", relative_to=job_path.parent)
    if script_path is None or not script_path.exists():
        blockers.append(f"missing adapter_effect_script: {job.get('adapter_effect_script')}")
    dataset_path = _job_path(job, "qwen_grpo_adapter_effect_dataset_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(
            f"missing qwen_grpo_adapter_effect_dataset_jsonl: {job.get('qwen_grpo_adapter_effect_dataset_jsonl')}"
        )
    adapter_path_value = str(job.get("adapter_path") or "")
    if not adapter_path_value:
        blockers.append("missing adapter_path")
    else:
        blockers.extend(_adapter_path_blockers(Path(adapter_path_value)))
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required adapter effect package(s): " + ", ".join(missing_packages))
    return blockers


def _action_likelihood_job_blockers(
    job: dict[str, Any],
    *,
    job_path: Path,
    check_dependencies: bool,
) -> list[str]:
    blockers: list[str] = []
    if job.get("schema") != QWEN_GRPO_ACTION_LIKELIHOOD_JOB_SCHEMA:
        blockers.append(f"unexpected action likelihood job schema: {job.get('schema')}")
    if job.get("status") != "ready":
        blockers.append(f"action likelihood job is not ready: {job.get('status')}")
    script_path = _job_path(job, "action_likelihood_script", relative_to=job_path.parent)
    if script_path is None or not script_path.exists():
        blockers.append(f"missing action_likelihood_script: {job.get('action_likelihood_script')}")
    dataset_path = _job_path(job, "qwen_grpo_action_likelihood_dataset_jsonl", relative_to=job_path.parent)
    if dataset_path is None or not dataset_path.exists():
        blockers.append(
            "missing qwen_grpo_action_likelihood_dataset_jsonl: "
            f"{job.get('qwen_grpo_action_likelihood_dataset_jsonl')}"
        )
    adapter_path_value = str(job.get("adapter_path") or "")
    if not adapter_path_value:
        blockers.append("missing adapter_path")
    else:
        blockers.extend(_adapter_path_blockers(Path(adapter_path_value)))
    if check_dependencies:
        missing_packages = _missing_required_packages(job)
        if missing_packages:
            blockers.append("missing required action likelihood package(s): " + ", ".join(missing_packages))
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
    zero_reward_exact_action_bonus: float,
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
        "--zero-reward-exact-action-bonus",
        str(zero_reward_exact_action_bonus),
    ]


def _eval_launch_argv(
    *,
    eval_script_path: Path,
    dataset_path: Path,
    model_id: str,
    output_dir: Path,
    completion_log_path: Path,
    adapter_path: Path | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    zero_reward_exact_action_bonus: float,
) -> list[str]:
    argv = [
        "python",
        str(eval_script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--output-dir",
        str(output_dir),
        "--completion-log",
        str(completion_log_path),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--zero-reward-exact-action-bonus",
        str(zero_reward_exact_action_bonus),
    ]
    if adapter_path is not None:
        argv.extend(["--adapter-path", str(adapter_path)])
    return argv


def _adapter_effect_launch_argv(
    *,
    script_path: Path,
    dataset_path: Path,
    model_id: str,
    output_dir: Path,
    adapter_path: Path,
    effect_log_path: Path,
    top_k: int,
    delta_threshold: float,
) -> list[str]:
    return [
        "python",
        str(script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-dir",
        str(output_dir),
        "--adapter-effect-log",
        str(effect_log_path),
        "--top-k",
        str(top_k),
        "--delta-threshold",
        str(delta_threshold),
    ]


def _action_likelihood_launch_argv(
    *,
    script_path: Path,
    dataset_path: Path,
    model_id: str,
    output_dir: Path,
    adapter_path: Path,
    likelihood_log_path: Path,
    progress_log_path: Path,
) -> list[str]:
    return [
        "python",
        str(script_path),
        "--dataset",
        str(dataset_path),
        "--model-id",
        model_id,
        "--adapter-path",
        str(adapter_path),
        "--output-dir",
        str(output_dir),
        "--action-likelihood-log",
        str(likelihood_log_path),
        "--progress-log",
        str(progress_log_path),
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


def _write_eval_script(path: Path) -> None:
    path.write_text(_EVAL_SCRIPT, encoding="utf-8")


def _write_adapter_effect_script(path: Path) -> None:
    path.write_text(_ADAPTER_EFFECT_SCRIPT, encoding="utf-8")


def _write_action_likelihood_script(path: Path) -> None:
    path.write_text(_ACTION_LIKELIHOOD_SCRIPT, encoding="utf-8")


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


def _jsonl_tail(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"malformed": line[:200]})
    return rows[-limit:]


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


def _forbidden_prompt_message_hits(records: list[dict[str, Any]]) -> list[str]:
    payload = [
        record.get("prompt_messages") if record.get("prompt_messages") is not None else record.get("prompt")
        for record in records
    ]
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    return sorted({token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text})


def _sidecar_prompt_leak_hits(records: list[dict[str, Any]]) -> list[str]:
    sidecar_tokens = [
        "reference_action_canonical",
        "reference_action_json",
        "reference_assistant_json",
        "candidate_step_reward",
        "candidate_episode_reward",
        "zero_reward_exact_action_bonus",
    ]
    payload = [
        record.get("prompt_messages") if record.get("prompt_messages") is not None else record.get("prompt")
        for record in records
    ]
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    return sorted({token for token in sidecar_tokens if token.lower() in text})


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


def _compact_action_response(action: Any) -> str:
    if not isinstance(action, dict) or not action:
        return ""
    tool = action.get("tool")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if tool is None:
        return ""
    return json.dumps(
        {"action": {"tool": str(tool), "args": args}},
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


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


def log_completion_batch(
    completions,
    rewards,
    reference_action_canonical=None,
    candidate_step_reward=None,
    candidate_step_reward_present=None,
    zero_reward_exact_action_bonus=None,
    metadata=None,
) -> None:
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
                "candidate_step_reward_present": value_at(candidate_step_reward_present, index),
                "zero_reward_exact_action_bonus": value_at(zero_reward_exact_action_bonus, index),
                "reference_action_canonical": expected,
                "expected_action": expected_action,
                "parsed_action": parsed_action,
                "tool_match": diagnostics["tool_match"],
                "arg_match_fraction": diagnostics["arg_match_fraction"],
                "completion_text": text[:4000],
                "completion_text_truncated": len(text) > 4000,
            }
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\\n")


def exact_action_reward(base_reward: float, reward_present: bool, zero_reward_exact_action_bonus: float) -> float:
    if reward_present and zero_reward_exact_action_bonus > 0.0 and abs(base_reward) <= 1e-12:
        return zero_reward_exact_action_bonus
    return base_reward


def navigation_tool_reward(
    completions,
    reference_action_canonical=None,
    candidate_step_reward=None,
    candidate_step_reward_present=None,
    reward_scale=None,
    zero_reward_exact_action_bonus=None,
    **kwargs,
):
    scale = float(reward_scale[0] if isinstance(reward_scale, list) and reward_scale else reward_scale or 1.0)
    bonus = float(
        zero_reward_exact_action_bonus[0]
        if isinstance(zero_reward_exact_action_bonus, list) and zero_reward_exact_action_bonus
        else zero_reward_exact_action_bonus or 0.0
    )
    rewards = []
    for index, completion in enumerate(completions):
        expected = reference_action_canonical[index] if isinstance(reference_action_canonical, list) else reference_action_canonical
        step_reward = candidate_step_reward[index] if isinstance(candidate_step_reward, list) else candidate_step_reward
        reward_present = (
            candidate_step_reward_present[index]
            if isinstance(candidate_step_reward_present, list)
            else candidate_step_reward_present
        )
        parsed = parse_action(completion_text(completion))
        base_reward = float(step_reward or 0.0)
        expected_action = parse_reference_action(expected)
        if canonical_json(parsed) == expected:
            rewards.append(exact_action_reward(base_reward, bool(reward_present), bonus) * scale)
        elif parsed:
            rewards.append(min(partial_action_reward(parsed, expected_action), base_reward - 0.05, -0.02) * scale)
        else:
            rewards.append(min(base_reward - 0.5, -0.5) * scale)
    log_completion_batch(
        completions,
        rewards,
        reference_action_canonical,
        candidate_step_reward,
        candidate_step_reward_present,
        zero_reward_exact_action_bonus,
        kwargs,
    )
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
    parser.add_argument(
        "--zero-reward-exact-action-bonus",
        type=float,
        default=0.0,
        help="Optional reward for exact reference actions whose candidate step reward is an observed zero.",
    )
    args = parser.parse_args()
    if args.zero_reward_exact_action_bonus < 0.0:
        parser.error("--zero-reward-exact-action-bonus must be non-negative")
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
        record["zero_reward_exact_action_bonus"] = args.zero_reward_exact_action_bonus
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


_EVAL_SCRIPT = '''"""Run held-out completion evaluation for flatdisk Qwen GRPO prompt samples."""

from __future__ import annotations

import argparse
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


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


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


def exact_action_reward(base_reward: float, reward_present: bool, zero_reward_exact_action_bonus: float) -> float:
    if reward_present and zero_reward_exact_action_bonus > 0.0 and abs(base_reward) <= 1e-12:
        return zero_reward_exact_action_bonus
    return base_reward


def score_completion(text: str, record: dict, zero_reward_exact_action_bonus: float) -> tuple[float, dict, dict]:
    expected = record.get("reference_action_canonical")
    expected_action = parse_reference_action(expected)
    parsed_action = parse_action(text)
    diagnostics = action_reward_diagnostics(parsed_action, expected_action)
    base_reward = float(record.get("candidate_step_reward") or 0.0)
    reward_present = bool(record.get("candidate_step_reward_present"))
    if canonical_json(parsed_action) == expected:
        reward = exact_action_reward(base_reward, reward_present, zero_reward_exact_action_bonus)
    elif parsed_action:
        reward = min(partial_action_reward(parsed_action, expected_action), base_reward - 0.05, -0.02)
    else:
        reward = min(base_reward - 0.5, -0.5)
    return reward, parsed_action, diagnostics


def prompt_messages(record: dict) -> list[dict]:
    messages = record.get("prompt_messages") if record.get("prompt_messages") is not None else record.get("prompt")
    return messages if isinstance(messages, list) else []


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def generate_completion(processor, model, record: dict, args: argparse.Namespace) -> str:
    messages = prompt_messages(record)
    images = [load_image(path) for path in record.get("image_paths", [])]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    if hasattr(inputs, "to"):
        inputs = inputs.to(model_device(model))
    generation_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0.0:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        generation_kwargs["do_sample"] = False
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def write_completion_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--completion-log", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--zero-reward-exact-action-bonus", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0.0:
        parser.error("--temperature must be non-negative")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        parser.error("--top-p must be in (0, 1]")
    if args.zero_reward_exact_action_bonus < 0.0:
        parser.error("--zero-reward-exact-action-bonus must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_id, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()
    records = read_jsonl(args.dataset)
    rows = []
    logged_at = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    for index, record in enumerate(records):
        completion = generate_completion(processor, model, record, args)
        reward, parsed_action, diagnostics = score_completion(
            completion,
            record,
            args.zero_reward_exact_action_bonus,
        )
        expected_action = parse_reference_action(record.get("reference_action_canonical"))
        rows.append(
            {
                "schema": "flatdisk.qwen_grpo_completion_eval_sample.v1",
                "logged_at": logged_at,
                "completion_index": index,
                "sample_id": record.get("sample_id"),
                "source_rollout_id": record.get("source_rollout_id"),
                "reward": reward,
                "candidate_step_reward": record.get("candidate_step_reward"),
                "candidate_step_reward_present": record.get("candidate_step_reward_present"),
                "zero_reward_exact_action_bonus": args.zero_reward_exact_action_bonus,
                "reference_action_canonical": record.get("reference_action_canonical"),
                "expected_action": expected_action,
                "parsed_action": parsed_action,
                "tool_match": diagnostics["tool_match"],
                "arg_match_fraction": diagnostics["arg_match_fraction"],
                "completion_text": completion[:4000],
                "completion_text_truncated": len(completion) > 4000,
            }
        )
    write_completion_log(args.completion_log, rows)
    print(json.dumps({"status": "complete", "sample_count": len(rows), "completion_log": str(args.completion_log)}, sort_keys=True))


if __name__ == "__main__":
    main()
'''


_ADAPTER_EFFECT_SCRIPT = '''"""Check whether a PEFT adapter changes Qwen next-token logits for GRPO eval prompts."""

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


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


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


def prompt_messages(record: dict) -> list[dict]:
    messages = record.get("prompt_messages") if record.get("prompt_messages") is not None else record.get("prompt")
    return messages if isinstance(messages, list) else []


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def decode_token(processor, token_id: int) -> str:
    try:
        return processor.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def top_tokens(processor, logits: torch.Tensor, top_k: int) -> list[dict]:
    k = min(top_k, int(logits.shape[-1]))
    values, ids = torch.topk(logits, k=k)
    rows = []
    for value, token_id in zip(values.tolist(), ids.tolist(), strict=True):
        rows.append(
            {
                "token_id": int(token_id),
                "logit": round(float(value), 9),
                "token": decode_token(processor, int(token_id)),
            }
        )
    return rows


def prepare_model_inputs(processor, model, record: dict) -> tuple[dict, dict]:
    messages = prompt_messages(record)
    images = [load_image(path) for path in record.get("image_paths", [])]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    if hasattr(inputs, "to"):
        inputs = inputs.to(model_device(model))
    input_ids = inputs.get("input_ids")
    return inputs, {
        "image_count": len(images),
        "prompt_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "input_token_count": int(input_ids.shape[1]) if input_ids is not None else 0,
    }


def adapter_metadata(model, adapter_path: Path) -> dict:
    peft_config = getattr(model, "peft_config", {})
    active_adapter = getattr(model, "active_adapter", None)
    if callable(active_adapter):
        try:
            active_adapter = active_adapter()
        except TypeError:
            active_adapter = str(active_adapter)
    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        try:
            active_adapters = active_adapters()
        except TypeError:
            active_adapters = None
    adapter_param_count = 0
    adapter_param_examples = []
    for name, parameter in model.named_parameters():
        lower_name = name.lower()
        if "lora" in lower_name or "adapter" in lower_name:
            adapter_param_count += int(parameter.numel())
            if len(adapter_param_examples) < 12:
                adapter_param_examples.append(name)
    return {
        "adapter_path": str(adapter_path),
        "active_adapter": str(active_adapter) if active_adapter is not None else "",
        "active_adapters": [str(value) for value in active_adapters] if isinstance(active_adapters, (list, tuple)) else [],
        "peft_adapter_names": sorted(str(key) for key in peft_config.keys()) if isinstance(peft_config, dict) else [],
        "adapter_parameter_count": adapter_param_count,
        "adapter_parameter_examples": adapter_param_examples,
    }


def compare_prompt_logits(processor, model, record: dict, args: argparse.Namespace) -> dict:
    inputs, input_meta = prepare_model_inputs(processor, model, record)
    with torch.inference_mode():
        with model.disable_adapter():
            base_outputs = model(**inputs)
        adapter_outputs = model(**inputs)
    base_logits = base_outputs.logits[:, -1, :].detach().float().cpu().squeeze(0)
    adapter_logits = adapter_outputs.logits[:, -1, :].detach().float().cpu().squeeze(0)
    diff = adapter_logits - base_logits
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    adapter_log_probs = torch.log_softmax(adapter_logits, dim=-1)
    base_probs = base_log_probs.exp()
    adapter_probs = adapter_log_probs.exp()
    kl_adapter_from_base = torch.sum(adapter_probs * (adapter_log_probs - base_log_probs)).item()
    kl_base_from_adapter = torch.sum(base_probs * (base_log_probs - adapter_log_probs)).item()
    base_top = top_tokens(processor, base_logits, args.top_k)
    adapter_top = top_tokens(processor, adapter_logits, args.top_k)
    base_top_ids = {row["token_id"] for row in base_top}
    adapter_top_ids = {row["token_id"] for row in adapter_top}
    top_union = base_top_ids | adapter_top_ids
    top_intersection = base_top_ids & adapter_top_ids
    max_abs_delta = float(diff.abs().max().item())
    base_top1 = base_top[0] if base_top else {"token_id": None, "token": ""}
    adapter_top1 = adapter_top[0] if adapter_top else {"token_id": None, "token": ""}
    return {
        **input_meta,
        "max_abs_logit_delta": round(max_abs_delta, 9),
        "mean_abs_logit_delta": round(float(diff.abs().mean().item()), 9),
        "l2_logit_delta": round(float(torch.linalg.vector_norm(diff).item()), 9),
        "kl_adapter_from_base": round(float(kl_adapter_from_base), 9),
        "kl_base_from_adapter": round(float(kl_base_from_adapter), 9),
        "nonzero_delta": max_abs_delta > args.delta_threshold,
        "delta_threshold": args.delta_threshold,
        "base_top1_token_id": base_top1["token_id"],
        "base_top1_token": base_top1["token"],
        "adapter_top1_token_id": adapter_top1["token_id"],
        "adapter_top1_token": adapter_top1["token"],
        "top1_changed": base_top1["token_id"] != adapter_top1["token_id"],
        "top_k": args.top_k,
        "top_k_overlap_count": len(top_intersection),
        "top_k_jaccard": round(len(top_intersection) / len(top_union), 9) if top_union else 0.0,
        "base_top_tokens": base_top,
        "adapter_top_tokens": adapter_top,
    }


def write_effect_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\\n")


def append_effect_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-effect-log", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delta-threshold", type=float, default=1e-6)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.delta_threshold < 0.0:
        parser.error("--delta-threshold must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_id, padding_side="left")
    base_model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()
    metadata = adapter_metadata(model, args.adapter_path)
    records = read_jsonl(args.dataset)
    logged_at = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    args.adapter_effect_log.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_effect_log.write_text("", encoding="utf-8")
    sample_count = 0
    for index, record in enumerate(records):
        expected_action = parse_reference_action(record.get("reference_action_canonical"))
        row = {
            "schema": "flatdisk.qwen_grpo_adapter_effect_sample.v1",
            "logged_at": logged_at,
            "sample_index": index,
            "sample_id": record.get("sample_id"),
            "source_rollout_id": record.get("source_rollout_id"),
            "expected_action": expected_action,
            "expected_tool": action_tool(expected_action) or "unknown",
            "reference_action_canonical": record.get("reference_action_canonical"),
            "adapter_metadata": metadata,
        }
        row.update(compare_prompt_logits(processor, model, record, args))
        append_effect_log(args.adapter_effect_log, row)
        sample_count += 1
    print(
        json.dumps(
            {
                "status": "complete",
                "sample_count": sample_count,
                "adapter_effect_log": str(args.adapter_effect_log),
                "adapter_metadata": metadata,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
'''


_ACTION_LIKELIHOOD_SCRIPT = '''"""Score reference action JSON likelihood under base-disabled and adapter-enabled Qwen."""

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


def compact_action_response(action) -> str:
    if not isinstance(action, dict) or not action:
        return ""
    tool = action.get("tool")
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    if tool is None:
        return ""
    return json.dumps(
        {"action": {"tool": str(tool), "args": args}},
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


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


def prompt_messages(record: dict) -> list[dict]:
    messages = record.get("prompt_messages") if record.get("prompt_messages") is not None else record.get("prompt")
    return messages if isinstance(messages, list) else []


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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


def prepare_teacher_forced_inputs(processor, model, record: dict, target_text: str) -> tuple[dict, dict]:
    messages = prompt_messages(record)
    images = [load_image(path) for path in record.get("image_paths", [])]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    }


def logprobs_for_target(logits: torch.Tensor, input_ids: torch.Tensor, target_start: int, target_count: int) -> torch.Tensor:
    if target_count <= 0 or target_start <= 0:
        return torch.empty(0, dtype=torch.float32)
    target_ids = input_ids[0, target_start : target_start + target_count].detach().cpu()
    prediction_logits = logits[0, target_start - 1 : target_start + target_count - 1, :].detach().float().cpu()
    log_probs = torch.log_softmax(prediction_logits, dim=-1)
    return log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def safe_exp(value: float) -> float:
    return math.exp(min(float(value), 50.0))


def target_tool_token_indices(processor, target_ids: list[int], target_text: str, tool: str) -> tuple[list[int], bool]:
    if not tool:
        return [], False
    pieces = [decode_token(processor, token_id) for token_id in target_ids]
    spans = token_char_spans(pieces)
    tool_segment = f'"tool":"{tool}"'
    tool_start = target_text.find(tool_segment)
    tool_end = tool_start + len(tool_segment) if tool_start >= 0 else -1
    if tool_start < 0:
        tool_start = target_text.find(tool)
        tool_end = tool_start + len(tool) if tool_start >= 0 else -1
    span_found = tool_start >= 0
    if not span_found:
        fallback = [index for index, piece in enumerate(pieces) if tool in piece or piece in tool]
        return fallback, False
    indices = [
        index
        for index, (start, end) in enumerate(spans)
        if overlap(start, end, tool_start, tool_end) or tool in pieces[index]
    ]
    if not indices:
        fallback = [index for index, piece in enumerate(pieces) if tool in piece or piece in tool]
        return fallback, False
    return indices, True


def score_record(processor, model, record: dict, args: argparse.Namespace, metadata: dict) -> dict:
    expected_action = record.get("reference_action_json") if isinstance(record.get("reference_action_json"), dict) else {}
    if not expected_action:
        expected_action = parse_reference_action(record.get("reference_action_canonical"))
    expected_tool = action_tool(expected_action) or "unknown"
    target_text = compact_action_response(expected_action)
    full_inputs, input_meta = prepare_teacher_forced_inputs(processor, model, record, target_text)
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
    target_tokens = [decode_token(processor, token_id) for token_id in target_ids]
    tool_indices, tool_span_found = target_tool_token_indices(processor, target_ids, target_text, expected_tool)
    token_rows = []
    deltas = []
    for index, (token_id, token_text, base_lp, adapter_lp) in enumerate(
        zip(target_ids, target_tokens, base_logprobs.tolist(), adapter_logprobs.tolist(), strict=True)
    ):
        delta = float(adapter_lp - base_lp)
        deltas.append(delta)
        if index in tool_indices:
            token_rows.append(
                {
                    "target_token_index": index,
                    "token_id": int(token_id),
                    "token": token_text,
                    "base_logprob": round(float(base_lp), 9),
                    "adapter_logprob": round(float(adapter_lp), 9),
                    "logprob_delta": round(delta, 9),
                }
            )
    base_values = [float(value) for value in base_logprobs.tolist()]
    adapter_values = [float(value) for value in adapter_logprobs.tolist()]
    tool_base_values = [base_values[index] for index in tool_indices if index < len(base_values)]
    tool_adapter_values = [adapter_values[index] for index in tool_indices if index < len(adapter_values)]
    target_base_sum = sum(base_values)
    target_adapter_sum = sum(adapter_values)
    tool_base_sum = sum(tool_base_values)
    tool_adapter_sum = sum(tool_adapter_values)
    base_target_mean = mean(base_values)
    adapter_target_mean = mean(adapter_values)
    base_tool_mean = mean(tool_base_values)
    adapter_tool_mean = mean(tool_adapter_values)
    base_target_nll = -base_target_mean
    adapter_target_nll = -adapter_target_mean
    first_target_token_delta = deltas[0] if deltas else 0.0
    return {
        "schema": "flatdisk.qwen_grpo_action_likelihood_sample.v1",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "sample_id": record.get("sample_id"),
        "source_rollout_id": record.get("source_rollout_id"),
        "expected_action": expected_action,
        "expected_tool": expected_tool,
        "reference_action_canonical": record.get("reference_action_canonical"),
        "target_text": target_text,
        "target_text_sha256": input_meta["target_text_sha256"],
        "target_token_count": target_count,
        "tool_token_count": len(tool_indices),
        "tool_span_found": tool_span_found,
        "base_target_logprob_sum": round(target_base_sum, 9),
        "adapter_target_logprob_sum": round(target_adapter_sum, 9),
        "target_logprob_sum_delta": round(target_adapter_sum - target_base_sum, 9),
        "base_target_mean_logprob": round(base_target_mean, 9),
        "adapter_target_mean_logprob": round(adapter_target_mean, 9),
        "target_mean_logprob_delta": round(adapter_target_mean - base_target_mean, 9),
        "base_target_nll": round(base_target_nll, 9),
        "adapter_target_nll": round(adapter_target_nll, 9),
        "target_nll_delta": round(adapter_target_nll - base_target_nll, 9),
        "base_target_perplexity": round(safe_exp(base_target_nll), 9),
        "adapter_target_perplexity": round(safe_exp(adapter_target_nll), 9),
        "base_tool_logprob_sum": round(tool_base_sum, 9),
        "adapter_tool_logprob_sum": round(tool_adapter_sum, 9),
        "tool_logprob_sum_delta": round(tool_adapter_sum - tool_base_sum, 9),
        "base_tool_mean_logprob": round(base_tool_mean, 9),
        "adapter_tool_mean_logprob": round(adapter_tool_mean, 9),
        "tool_mean_logprob_delta": round(adapter_tool_mean - base_tool_mean, 9)
        if tool_indices
        else None,
        "first_target_token_logprob_delta": round(first_target_token_delta, 9),
        "adapter_improved_target_mean_logprob": adapter_target_mean > base_target_mean,
        "adapter_improved_tool_mean_logprob": adapter_tool_mean > base_tool_mean
        if tool_indices
        else False,
        "target_token_mean_abs_logprob_delta": round(mean([abs(value) for value in deltas]), 9),
        "tool_token_scores": token_rows,
        "adapter_metadata": metadata,
        **input_meta,
    }


def append_likelihood_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\\n")
        handle.flush()


def write_progress(path: Path, stage: str, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "flatdisk.qwen_grpo_action_likelihood_progress.v1",
        "logged_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "stage": stage,
        **payload,
    }
    line = json.dumps(row, sort_keys=True, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\\n")
        handle.flush()
    print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-likelihood-log", type=Path, required=True)
    parser.add_argument("--progress-log", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.action_likelihood_log.parent.mkdir(parents=True, exist_ok=True)
    args.action_likelihood_log.write_text("", encoding="utf-8")
    args.progress_log.parent.mkdir(parents=True, exist_ok=True)
    args.progress_log.write_text("", encoding="utf-8")
    write_progress(
        args.progress_log,
        "start",
        dataset=str(args.dataset),
        model_id=args.model_id,
        adapter_path=str(args.adapter_path),
        output_dir=str(args.output_dir),
    )
    write_progress(args.progress_log, "load_processor_start")
    processor = AutoProcessor.from_pretrained(args.model_id, padding_side="left")
    write_progress(args.progress_log, "load_processor_complete")
    write_progress(args.progress_log, "load_model_start")
    base_model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    write_progress(args.progress_log, "load_model_complete")
    write_progress(args.progress_log, "load_adapter_start")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    model.eval()
    metadata = adapter_metadata(model, args.adapter_path)
    write_progress(args.progress_log, "load_adapter_complete", adapter_metadata=metadata)
    write_progress(args.progress_log, "read_dataset_start")
    records = read_jsonl(args.dataset)
    write_progress(args.progress_log, "read_dataset_complete", sample_count=len(records))
    sample_count = 0
    for index, record in enumerate(records):
        write_progress(
            args.progress_log,
            "sample_start",
            sample_index=index,
            sample_id=record.get("sample_id"),
            source_rollout_id=record.get("source_rollout_id"),
        )
        row = score_record(processor, model, record, args, metadata)
        append_likelihood_log(args.action_likelihood_log, row)
        sample_count += 1
        write_progress(
            args.progress_log,
            "sample_complete",
            sample_index=index,
            sample_id=record.get("sample_id"),
            expected_tool=row.get("expected_tool"),
            target_mean_logprob_delta=row.get("target_mean_logprob_delta"),
            tool_mean_logprob_delta=row.get("tool_mean_logprob_delta"),
            tool_span_found=row.get("tool_span_found"),
        )
    write_progress(args.progress_log, "complete", sample_count=sample_count)
    print(
        json.dumps(
            {
                "status": "complete",
                "sample_count": sample_count,
                "action_likelihood_log": str(args.action_likelihood_log),
                "adapter_metadata": metadata,
            },
            sort_keys=True,
        )
    )


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
        "--zero-reward-exact-action-bonus",
        type=float,
        default=0.0,
        help="Optional reward for exact reference actions whose candidate step reward is an observed zero.",
    )
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
    args = parser.parse_args()
    if args.zero_reward_exact_action_bonus < 0.0:
        parser.error("--zero-reward-exact-action-bonus must be non-negative")
    return args


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
        zero_reward_exact_action_bonus=args.zero_reward_exact_action_bonus,
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
                "exact_action_reward_shaping": job["audit"]["exact_action_reward_shaping"],
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


def parse_eval_plan_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan held-out Qwen GRPO completion evaluation.")
    parser.add_argument("--training-job", type=Path, required=True, help="qwen_grpo_training_job.json or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--zero-reward-exact-action-bonus", type=float, default=None)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def eval_plan_main() -> int:
    args = parse_eval_plan_args()
    job = plan_qwen_grpo_completion_eval(
        args.training_job,
        output_dir=args.output_dir,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        sample_stride=args.sample_stride,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        zero_reward_exact_action_bonus=args.zero_reward_exact_action_bonus,
        require_existing_images=not args.allow_missing_images,
    )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["eval_sample_count"],
                "reference_action_tool_counts": job["dataset"]["reference_action_tool_counts"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "sidecar_prompt_leak_hits": job["dataset"]["sidecar_prompt_leak_hits"],
                "job_path": str(Path(job["output_dir"]) / "qwen_grpo_completion_eval_job.json"),
                "eval_script": job["eval_script"],
                "launch_command": job["launch_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and job["status"] != "ready":
        return 2
    return 0


def parse_eval_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen GRPO completion eval job.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_grpo_completion_eval_job.json or directory")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def eval_run_main() -> int:
    args = parse_eval_run_args()
    result = run_qwen_grpo_completion_eval_job(
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


def parse_adapter_effect_plan_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a Qwen GRPO PEFT adapter-effect logit check.")
    parser.add_argument(
        "--completion-eval-job",
        type=Path,
        required=True,
        help="qwen_grpo_completion_eval_job.json or directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--delta-threshold", type=float, default=1e-6)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    args = parser.parse_args()
    if args.sample_offset < 0:
        parser.error("--sample-offset must be non-negative")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive when set")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.delta_threshold < 0.0:
        parser.error("--delta-threshold must be non-negative")
    return args


def adapter_effect_plan_main() -> int:
    args = parse_adapter_effect_plan_args()
    job = plan_qwen_grpo_adapter_effect_check(
        args.completion_eval_job,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        sample_stride=args.sample_stride,
        top_k=args.top_k,
        delta_threshold=args.delta_threshold,
        require_existing_images=not args.allow_missing_images,
    )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["eval_sample_count"],
                "reference_action_tool_counts": job["dataset"]["reference_action_tool_counts"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "sidecar_prompt_leak_hits": job["dataset"]["sidecar_prompt_leak_hits"],
                "adapter_path_blockers": job["dataset"]["adapter_path_blockers"],
                "job_path": str(Path(job["output_dir"]) / "qwen_grpo_adapter_effect_job.json"),
                "adapter_effect_script": job["adapter_effect_script"],
                "launch_command": job["launch_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and job["status"] != "ready":
        return 2
    return 0


def parse_adapter_effect_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen GRPO adapter-effect check.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_grpo_adapter_effect_job.json or directory")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def adapter_effect_run_main() -> int:
    args = parse_adapter_effect_run_args()
    result = run_qwen_grpo_adapter_effect_check_job(
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
                "adapter_effect_log_metrics": result.get("adapter_effect_log_metrics", {}),
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


def parse_action_likelihood_plan_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a Qwen GRPO PEFT action-likelihood check.")
    parser.add_argument(
        "--completion-eval-job",
        type=Path,
        required=True,
        help="qwen_grpo_completion_eval_job.json or directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    args = parser.parse_args()
    if args.sample_offset < 0:
        parser.error("--sample-offset must be non-negative")
    if args.sample_stride < 1:
        parser.error("--sample-stride must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive when set")
    return args


def action_likelihood_plan_main() -> int:
    args = parse_action_likelihood_plan_args()
    job = plan_qwen_grpo_action_likelihood_check(
        args.completion_eval_job,
        output_dir=args.output_dir,
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        sample_stride=args.sample_stride,
        require_existing_images=not args.allow_missing_images,
    )
    print(
        json.dumps(
            {
                "status": job["status"],
                "sample_count": job["dataset"]["eval_sample_count"],
                "reference_action_tool_counts": job["dataset"]["reference_action_tool_counts"],
                "missing_image_count": job["dataset"]["missing_image_count"],
                "forbidden_model_token_hits": job["dataset"]["forbidden_model_token_hits"],
                "sidecar_prompt_leak_hits": job["dataset"]["sidecar_prompt_leak_hits"],
                "adapter_path_blockers": job["dataset"]["adapter_path_blockers"],
                "empty_target_count": job["dataset"]["empty_target_count"],
                "job_path": str(Path(job["output_dir"]) / "qwen_grpo_action_likelihood_job.json"),
                "action_likelihood_script": job["action_likelihood_script"],
                "launch_command": job["launch_command"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and job["status"] != "ready":
        return 2
    return 0


def parse_action_likelihood_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run a planned Qwen GRPO action-likelihood check.")
    parser.add_argument("--job", type=Path, required=True, help="qwen_grpo_action_likelihood_job.json or directory")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--launch-command", default=None, help="Override launch command; intended for tests or manual recovery.")
    parser.add_argument("--tail-chars", type=int, default=4000)
    return parser.parse_args()


def action_likelihood_run_main() -> int:
    args = parse_action_likelihood_run_args()
    result = run_qwen_grpo_action_likelihood_check_job(
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
                "action_likelihood_log_metrics": result.get("action_likelihood_log_metrics", {}),
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
