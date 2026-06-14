"""Export navigation harness runs into policy-training trajectory records.

The exporter keeps the model-facing policy channel separate from evaluator-only
reward labels. Hidden THOR distance/success can be used as reward metadata, but
object metadata and target details are not copied into the policy input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .llm_harness import sanitize_memory


POLICY_STEP_SCHEMA = "flatdisk.nav_policy_step.v1"
EPISODE_ROLLOUT_SCHEMA = "flatdisk.nav_episode_rollout.v1"
ROLLOUT_GROUP_SCHEMA = "flatdisk.nav_rollout_group.v1"
TRAJECTORY_PREFERENCE_SCHEMA = "flatdisk.nav_trajectory_preference_pair.v1"
POLICY_SAMPLE_SCHEMA = "flatdisk.nav_policy_sample.v1"
EVALUATOR_LABEL_SCHEMA = "flatdisk.nav_evaluator_label.v1"
POLICY_DATASET_MANIFEST_SCHEMA = "flatdisk.nav_policy_dataset_manifest.v1"
POLICY_REVIEW_TRACE_SCHEMA = "flatdisk.nav_policy_review_trace.v1"
TRAINING_EXPORT_SCHEMA = "flatdisk.nav_training_export.v1"
FORBIDDEN_POLICY_TOKENS = ("hidden_score", "nearest_target", "object_metadata", "target_pose", "distance_m", "detections")
FORBIDDEN_REVIEW_TRACE_TOKENS = (
    "hidden_score",
    "nearest_target",
    "object_metadata",
    "target_pose",
    "distance_m",
    "detections",
    "objectid",
)
REVIEW_TOOL_RESULT_KEYS = {
    "action",
    "cost",
    "debug_overlay_contact_sheet",
    "detector",
    "detection_coverage_fraction",
    "detection_status_count",
    "duration_s",
    "elapsed_s",
    "ever_detected",
    "failure_reason",
    "final_yaw_deg",
    "forward_power",
    "frame_count",
    "goal_query",
    "grounding_audit_contact_sheet",
    "grounding_stability",
    "heading_error_deg",
    "last_command",
    "map_summary",
    "matching_mode",
    "motion_contact_sheet",
    "motor_commands_sent",
    "moved",
    "ok",
    "planner_note",
    "prompt",
    "reason",
    "route_length",
    "route_node_ids",
    "route_truncated",
    "semantic_identity",
    "servo_status",
    "started_yaw_deg",
    "status_sample_count",
    "target_detected",
    "target_yaw_deg",
    "timed_out",
    "topomap_contact_sheet",
}


@dataclass(frozen=True)
class TrainingExportResult:
    output_dir: str
    manifest_path: str
    policy_steps_jsonl: str
    episode_rollouts_jsonl: str
    rollout_groups_jsonl: str
    trajectory_preferences_jsonl: str
    policy_review_traces_jsonl: str
    policy_dataset_dir: str
    policy_dataset_manifest_path: str
    policy_samples_jsonl: str
    evaluator_labels_jsonl: str
    episode_count: int
    step_count: int
    rollout_group_count: int
    trajectory_preference_count: int
    policy_review_trace_count: int
    policy_sample_count: int
    evaluator_label_count: int

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_EXPORT_SCHEMA,
            "output_dir": self.output_dir,
            "manifest_path": self.manifest_path,
            "policy_steps_jsonl": self.policy_steps_jsonl,
            "episode_rollouts_jsonl": self.episode_rollouts_jsonl,
            "rollout_groups_jsonl": self.rollout_groups_jsonl,
            "trajectory_preferences_jsonl": self.trajectory_preferences_jsonl,
            "policy_review_traces_jsonl": self.policy_review_traces_jsonl,
            "policy_dataset_dir": self.policy_dataset_dir,
            "policy_dataset_manifest_path": self.policy_dataset_manifest_path,
            "policy_samples_jsonl": self.policy_samples_jsonl,
            "evaluator_labels_jsonl": self.evaluator_labels_jsonl,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "rollout_group_count": self.rollout_group_count,
            "trajectory_preference_count": self.trajectory_preference_count,
            "policy_review_trace_count": self.policy_review_trace_count,
            "policy_sample_count": self.policy_sample_count,
            "evaluator_label_count": self.evaluator_label_count,
        }


def export_training_data_from_summaries(
    summaries: list[dict[str, Any]],
    *,
    output_dir: Path,
    experiment_id: str,
    run_id: str | None = None,
    include_prompt_text: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_steps_path = output_dir / "policy_steps.jsonl"
    global_episodes_path = output_dir / "episode_rollouts.jsonl"
    rollout_groups_path = output_dir / "rollout_groups.jsonl"
    trajectory_preferences_path = output_dir / "trajectory_preferences.jsonl"
    review_traces_path = output_dir / "policy_review_traces.jsonl"
    episode_count = 0
    step_count = 0
    review_trace_count = 0
    episode_records: list[dict[str, Any]] = []
    policy_step_records: list[dict[str, Any]] = []
    with (
        global_steps_path.open("w", encoding="utf-8") as steps_handle,
        global_episodes_path.open("w", encoding="utf-8") as episodes_handle,
        review_traces_path.open("w", encoding="utf-8") as review_traces_handle,
    ):
        for summary in summaries:
            if not isinstance(summary.get("steps"), list):
                continue
            run_id_text = _run_record_id(summary)
            run_export_dir = output_dir / "runs" / _safe_id(run_id_text)
            run_export_dir.mkdir(parents=True, exist_ok=True)
            run_steps_path = run_export_dir / "policy_steps.jsonl"
            run_episode_path = run_export_dir / "episode_rollout.json"
            run_review_trace_path = run_export_dir / "policy_review_trace.json"
            records = _step_records(summary, include_prompt_text=include_prompt_text)
            if not records:
                continue
            review_trace = _policy_review_trace(summary, step_records=records, run_steps_path=run_steps_path)
            _assert_review_trace_safe(review_trace)
            policy_step_records.extend(records)
            episode_record = _episode_record(
                summary,
                experiment_id=experiment_id,
                research_run_id=run_id,
                step_record_count=len(records),
                run_steps_path=run_steps_path,
            )
            with run_steps_path.open("w", encoding="utf-8") as run_steps:
                for record in records:
                    text = json.dumps(record, sort_keys=True, default=str)
                    run_steps.write(text + "\n")
                    steps_handle.write(text + "\n")
            run_episode_path.write_text(json.dumps(episode_record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            episodes_handle.write(json.dumps(episode_record, sort_keys=True, default=str) + "\n")
            run_review_trace_path.write_text(json.dumps(review_trace, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            review_traces_handle.write(json.dumps(review_trace, sort_keys=True, default=str) + "\n")
            episode_records.append(episode_record)
            summary["training_export_dir"] = str(run_export_dir)
            summary["training_policy_steps_jsonl"] = str(run_steps_path)
            summary["training_episode_rollout_json"] = str(run_episode_path)
            summary["policy_review_trace_json"] = str(run_review_trace_path)
            episode_count += 1
            step_count += len(records)
            review_trace_count += 1

    rollout_groups, trajectory_preferences = _rl_rollout_records(episode_records, experiment_id=experiment_id, research_run_id=run_id)
    _write_jsonl(rollout_groups_path, rollout_groups)
    _write_jsonl(trajectory_preferences_path, trajectory_preferences)
    policy_dataset = _write_policy_dataset(
        policy_step_records,
        output_dir=output_dir / "policy_dataset_v1",
        source_training_manifest=str(output_dir / "training_manifest.json"),
    )
    manifest = TrainingExportResult(
        output_dir=str(output_dir),
        manifest_path=str(output_dir / "training_manifest.json"),
        policy_steps_jsonl=str(global_steps_path),
        episode_rollouts_jsonl=str(global_episodes_path),
        rollout_groups_jsonl=str(rollout_groups_path),
        trajectory_preferences_jsonl=str(trajectory_preferences_path),
        policy_review_traces_jsonl=str(review_traces_path),
        policy_dataset_dir=policy_dataset["output_dir"],
        policy_dataset_manifest_path=policy_dataset["manifest_path"],
        policy_samples_jsonl=policy_dataset["policy_samples_jsonl"],
        evaluator_labels_jsonl=policy_dataset["evaluator_labels_jsonl"],
        episode_count=episode_count,
        step_count=step_count,
        rollout_group_count=len(rollout_groups),
        trajectory_preference_count=len(trajectory_preferences),
        policy_review_trace_count=review_trace_count,
        policy_sample_count=int(policy_dataset["sample_count"]),
        evaluator_label_count=int(policy_dataset["label_count"]),
    )
    manifest_payload = manifest.as_json() | {
        "experiment_id": experiment_id,
        "research_run_id": run_id,
        "include_prompt_text": include_prompt_text,
        "policy_input_channel": [
            "actor prompt text",
            "camera frame paths",
            "motion/topomap contact sheet paths",
            "sanitized observation",
            "sanitized memory-visible tool results",
        ],
        "evaluator_channel": [
            "success boolean",
            "final distance",
            "per-step post-action distance",
            "candidate scalar reward",
        ],
        "rl_training_channel": [
            "rollout groups keyed by the same episode prompt for GRPO-style relative rewards",
            "trajectory preference pairs keyed by evaluator reward margin for DPO/RLHF-style filtering",
            "all reward labels remain in evaluator_reward/evaluator_preference, outside policy_input",
        ],
        "agent_review_channel": [
            "policy_review_traces_jsonl stores compact per-step actor/tool/critic traces for human and sub-agent review",
            "review traces include model-facing image/contact-sheet paths and general grounding flags",
            "hidden evaluator distances, target object metadata, and detection arrays are excluded",
        ],
        "privileged_reward_used": True,
        "privileged_reward_purpose": "training label / offline ranking only; never model prompt input",
        "forbidden_policy_fields": ["pose", "scene", "objects", "object_metadata", "nearest_target", "target_pose", "distance_m"],
        "forbidden_review_trace_fields": list(FORBIDDEN_REVIEW_TRACE_TOKENS),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_payload


def export_training_data_from_paths(
    input_paths: Iterable[Path],
    *,
    output_dir: Path,
    experiment_id: str = "manual_export",
    run_id: str | None = None,
    include_prompt_text: bool = True,
) -> dict[str, Any]:
    summaries = [_load_episode_summary(path) for path in _discover_episode_summaries(input_paths)]
    return export_training_data_from_summaries(
        summaries,
        output_dir=output_dir,
        experiment_id=experiment_id,
        run_id=run_id,
        include_prompt_text=include_prompt_text,
    )


def _step_records(summary: dict[str, Any], *, include_prompt_text: bool) -> list[dict[str, Any]]:
    policy_dir = _resolve_path(summary.get("policy_dir"), base=Path(summary.get("run_dir") or "."))
    actor_outputs = _actor_outputs_by_step(policy_dir / "harness_events.jsonl")
    records: list[dict[str, Any]] = []
    previous_memory_records: list[dict[str, Any]] = []
    previous_distance: float | None = None
    for item in summary.get("steps", []):
        if not isinstance(item, dict):
            continue
        memory = item.get("harness_memory_record")
        if not isinstance(memory, dict):
            continue
        step = int(memory.get("step", item.get("step", len(records))) or 0)
        prompt_path = policy_dir / "prompts" / f"{step:03d}_actor.txt"
        post_score = _safe_hidden_score(item.get("hidden_score_for_evaluator_only"))
        current_distance = post_score.get("distance_m")
        distance_delta = None
        if isinstance(current_distance, (int, float)) and previous_distance is not None:
            distance_delta = round(previous_distance - float(current_distance), 6)
        if isinstance(current_distance, (int, float)):
            previous_distance = float(current_distance)
        actor_prompt_text = prompt_path.read_text(encoding="utf-8") if include_prompt_text and prompt_path.exists() else None
        record = {
            "schema": POLICY_STEP_SCHEMA,
            "record_id": f"{_run_record_id(summary)}_step_{step:03d}",
            "source": _source_summary(summary),
            "episode": _episode_metadata(summary),
            "policy_input": {
                "goal": str(memory.get("goal", summary.get("prompt", ""))),
                "step": step,
                "actor_prompt_path": str(prompt_path) if prompt_path.exists() else None,
                "actor_prompt_text": actor_prompt_text,
                "image_paths": _inferred_actor_image_paths(memory, previous_memory_records, policy_dir=policy_dir),
                "observation": sanitize_memory(memory.get("observation", {})),
                "recent_memory_note": "Actor prompt already contains sanitized recent memory; this record stores the current sanitized observation and artifact paths.",
            },
            "policy_input_audit": _policy_input_audit(actor_prompt_text, memory),
            "policy_output": {
                "raw_actor_output": actor_outputs.get(step) or _canonical_actor_output(memory),
                "actor_action": sanitize_memory(memory.get("actor_action", {})),
                "actor_grounding_audit": sanitize_memory(memory.get("actor_grounding_audit", {})),
                "actor_memory_update": sanitize_memory(memory.get("actor_memory_update", {})),
                "critic": sanitize_memory(memory.get("critic", {})),
                "executed_action": sanitize_memory(memory.get("executed_action", {})),
            },
            "tool_feedback": sanitize_memory(memory.get("tool_result", {})),
            "review_tool_feedback": _review_tool_feedback(memory.get("tool_result", {})),
            "evaluator_reward": {
                "privileged": True,
                "post_action_score": post_score,
                "distance_delta_from_previous_step_m": distance_delta,
                "episode_success": bool(summary.get("success")),
                "final_distance_m": _optional_float(summary.get("final_distance_m")),
                "candidate_step_reward": _candidate_step_reward(post_score, distance_delta),
                "candidate_episode_reward": _candidate_episode_reward(summary),
                "reward_source": "hidden evaluator distance/success; not included in policy_input",
            },
        }
        records.append(record)
        previous_memory_records.append(memory)
    return records


def _episode_record(
    summary: dict[str, Any],
    *,
    experiment_id: str,
    research_run_id: str | None,
    step_record_count: int,
    run_steps_path: Path,
) -> dict[str, Any]:
    return {
        "schema": EPISODE_ROLLOUT_SCHEMA,
        "record_id": _run_record_id(summary),
        "experiment_id": experiment_id,
        "research_run_id": research_run_id,
        "source": _source_summary(summary),
        "episode": _episode_metadata(summary),
        "variant": summary.get("variant"),
        "runner": summary.get("runner"),
        "model": summary.get("model"),
        "prompt_profile": summary.get("prompt_profile"),
        "success": bool(summary.get("success")),
        "reason": summary.get("reason"),
        "step_record_count": step_record_count,
        "policy_steps_jsonl": str(run_steps_path),
        "evaluator_reward": {
            "privileged": True,
            "final_distance_m": _optional_float(summary.get("final_distance_m")),
            "success_radius_m": _optional_float(summary.get("success_radius_m")),
            "candidate_episode_reward": _candidate_episode_reward(summary),
            "reward_source": "hidden evaluator final distance/success; not included in policy_input",
        },
        "policy_input_allowlist": summary.get("policy_input_allowlist", []),
        "topomap_memory": {
            "map_dir": summary.get("topomap_memory_map_dir"),
            "use_clip": bool(summary.get("topomap_memory_use_clip")),
            "allow_semantic_terms": bool(summary.get("topomap_memory_allow_semantic_terms")),
        },
    }


def _policy_review_trace(
    summary: dict[str, Any],
    *,
    step_records: list[dict[str, Any]],
    run_steps_path: Path,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    previous_visual_servo_prompt: str | None = None
    for index, record in enumerate(step_records):
        step = _policy_review_step(record, index=index)
        actor_action = step.get("actor_action") if isinstance(step.get("actor_action"), dict) else {}
        actor_audit = step.get("actor_grounding_audit") if isinstance(step.get("actor_grounding_audit"), dict) else {}
        actor_prompt = _action_prompt(actor_action)
        if (
            previous_visual_servo_prompt
            and _action_tool(actor_action) == "visual_servo_object"
            and _normalized_prompt(actor_prompt) == _normalized_prompt(previous_visual_servo_prompt)
            and _grounding_audit_requested_prompt_change(actor_audit)
        ):
            step["review_flags"].append("actor_repeated_servo_prompt_after_grounding_audit_requested_change")
        steps.append(step)
        executed_action = step.get("executed_action") if isinstance(step.get("executed_action"), dict) else {}
        tool_result = step.get("tool_result") if isinstance(step.get("tool_result"), dict) else {}
        if _action_tool(executed_action) == "visual_servo_object" or tool_result.get("action") == "visual_servo_object":
            previous_visual_servo_prompt = str(tool_result.get("prompt") or _action_prompt(executed_action) or "").strip() or None

    review_flags = sorted({flag for step in steps for flag in step.get("review_flags", []) if isinstance(flag, str)})
    goal = str(summary.get("prompt") or "")
    if not goal and step_records:
        policy_input = step_records[0].get("policy_input") if isinstance(step_records[0].get("policy_input"), dict) else {}
        goal = str(policy_input.get("goal") or "")
    trace = {
        "schema": POLICY_REVIEW_TRACE_SCHEMA,
        "record_id": _run_record_id(summary),
        "source": {
            "trial_id": summary.get("trial_id"),
            "slot_id": summary.get("slot_id"),
            "run_dir": summary.get("run_dir"),
            "policy_dir": summary.get("policy_dir"),
        },
        "task": {
            "goal": goal,
            "episode": summary.get("episode"),
        },
        "run": {
            "variant": summary.get("variant"),
            "runner": summary.get("runner"),
            "model": summary.get("model"),
            "prompt_profile": summary.get("prompt_profile"),
            "reason": summary.get("reason"),
            "step_count": int(summary.get("step_count") or len(steps)),
        },
        "policy_steps_jsonl": str(run_steps_path),
        "step_count": len(steps),
        "review_flags": review_flags,
        "steps": steps,
        "policy_safety": {
            "model_facing_artifact": True,
            "privileged_evaluator_fields_excluded": True,
            "hidden_target_metadata_excluded": True,
        },
    }
    trace["policy_safety"]["forbidden_review_field_names_present"] = _forbidden_review_field_names(trace)
    return trace


def _policy_review_step(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    policy_input = record.get("policy_input") if isinstance(record.get("policy_input"), dict) else {}
    policy_output = record.get("policy_output") if isinstance(record.get("policy_output"), dict) else {}
    actor_action = _review_action(policy_output.get("actor_action"))
    executed_action = _review_action(policy_output.get("executed_action"))
    actor_grounding_audit = _safe_review_value(policy_output.get("actor_grounding_audit", {}))
    tool_result = _review_tool_feedback(record.get("review_tool_feedback") or record.get("tool_feedback", {}))
    review_flags = _review_step_flags(
        actor_action=actor_action,
        executed_action=executed_action,
        actor_grounding_audit=actor_grounding_audit if isinstance(actor_grounding_audit, dict) else {},
        tool_result=tool_result,
    )
    return {
        "step": int(policy_input.get("step") or index),
        "image_paths": _safe_review_value(policy_input.get("image_paths", [])),
        "actor_raw_output": _safe_review_text(policy_output.get("raw_actor_output"), limit=12000),
        "actor_action": actor_action,
        "actor_grounding_audit": actor_grounding_audit,
        "actor_memory_update": _safe_review_value(policy_output.get("actor_memory_update", {})),
        "critic": _safe_review_value(policy_output.get("critic", {})),
        "executed_action": executed_action,
        "actor_action_replaced": _canonical_json(actor_action) != _canonical_json(executed_action),
        "tool_result": tool_result,
        "review_flags": review_flags,
    }


def _review_action(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    tool = str(value.get("tool", value.get("action", ""))).strip()
    args = value.get("args", {})
    return {
        "tool": tool,
        "args": _safe_review_value(args if isinstance(args, dict) else {}),
    }


def _review_tool_feedback(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text in REVIEW_TOOL_RESULT_KEYS:
            cleaned[key_text] = _safe_review_value(item)
    return cleaned


def _review_step_flags(
    *,
    actor_action: dict[str, Any],
    executed_action: dict[str, Any],
    actor_grounding_audit: dict[str, Any],
    tool_result: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if _canonical_json(actor_action) != _canonical_json(executed_action):
        flags.append("actor_action_replaced")
    if _grounding_audit_requested_prompt_change(actor_grounding_audit):
        flags.append("actor_reported_previous_grounding_mismatch")
    executed_tool = _action_tool(executed_action)
    result_action = str(tool_result.get("action") or "")
    if executed_tool == "visual_servo_object" or result_action == "visual_servo_object":
        stability = str(tool_result.get("grounding_stability") or "")
        if stability and stability != "status_track_present":
            flags.append("visual_servo_grounding_not_stable")
            flags.append(f"visual_servo_{_safe_id(stability)}")
        if tool_result.get("target_detected") is False or tool_result.get("ever_detected") is False or stability == "no_detection":
            flags.append("visual_servo_no_detection")
    if tool_result.get("reason") == "topomap_memory_not_configured":
        flags.append("topomap_memory_not_configured")
    return sorted(set(flags))


def _grounding_audit_requested_prompt_change(value: dict[str, Any]) -> bool:
    return value.get("next_prompt_should_change") is True or value.get("previous_visual_servo_box_matches_intended_object") is False


def _action_tool(value: dict[str, Any]) -> str:
    return str(value.get("tool") or value.get("action") or "")


def _action_prompt(value: dict[str, Any]) -> str:
    args = value.get("args") if isinstance(value.get("args"), dict) else {}
    return str(args.get("prompt") or "")


def _normalized_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _safe_review_value(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_review_text(value)
    if isinstance(value, list):
        return [_safe_review_value(item) for item in value[:12]]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _review_field_name_allowed(key_text):
                cleaned[key_text] = _safe_review_value(item)
        return cleaned
    return value


def _safe_review_text(value: Any, *, limit: int = 2000) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def _review_field_name_allowed(key: str) -> bool:
    key_lower = key.lower()
    return not any(token in key_lower for token in FORBIDDEN_REVIEW_TRACE_TOKENS)


def _forbidden_review_field_names(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found.update(_forbidden_review_field_names(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if not _review_field_name_allowed(key_text):
                found.add(key_text)
            found.update(_forbidden_review_field_names(item))
    return sorted(found)


def _assert_review_trace_safe(trace: dict[str, Any]) -> None:
    forbidden = _forbidden_review_field_names(trace)
    if forbidden:
        raise ValueError(f"policy review trace contains forbidden field names: {forbidden}")


def _rl_rollout_records(
    episode_records: list[dict[str, Any]],
    *,
    experiment_id: str,
    research_run_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for episode_record in episode_records:
        episode = episode_record.get("episode") if isinstance(episode_record.get("episode"), dict) else {}
        key = (str(episode.get("name") or ""), str(episode.get("prompt") or ""))
        grouped.setdefault(key, []).append(episode_record)

    rollout_groups: list[dict[str, Any]] = []
    trajectory_preferences: list[dict[str, Any]] = []
    for group_index, ((episode_name, episode_prompt), group_records) in enumerate(sorted(grouped.items())):
        ranked = sorted(group_records, key=_episode_reward_value, reverse=True)
        group_id = _safe_id(f"{experiment_id}_{research_run_id or 'manual'}_{episode_name or 'episode'}_{group_index:03d}")
        rollouts = [_rollout_training_view(record, rank=rank + 1) for rank, record in enumerate(ranked)]
        rollout_groups.append(
            {
                "schema": ROLLOUT_GROUP_SCHEMA,
                "record_id": group_id,
                "experiment_id": experiment_id,
                "research_run_id": research_run_id,
                "episode": {
                    "name": episode_name,
                    "prompt": episode_prompt,
                },
                "training_uses": ["grpo", "ppo", "trajectory_ranking"],
                "policy_context": {
                    "goal_prompt": episode_prompt,
                    "policy_input_source": "Per-rollout policy_steps_jsonl records contain model-facing prompts, image paths, sanitized observation, memory, and tool feedback.",
                    "reward_labels_excluded_from_policy_input": True,
                },
                "rollouts": rollouts,
                "reward_ranking": [
                    {
                        "record_id": item["record_id"],
                        "rank": item["rank"],
                        "candidate_episode_reward": item["evaluator_reward"]["candidate_episode_reward"],
                    }
                    for item in rollouts
                ],
                "evaluator_reward": {
                    "privileged": True,
                    "reward_source": "hidden evaluator success/final distance; use for offline ranking or RL reward only",
                },
            }
        )
        if len(ranked) >= 2:
            chosen = ranked[0]
            rejected = ranked[-1]
            chosen_reward = _episode_reward_value(chosen)
            rejected_reward = _episode_reward_value(rejected)
            if chosen_reward > rejected_reward:
                trajectory_preferences.append(
                    _trajectory_preference_record(
                        group_id=group_id,
                        index=len(trajectory_preferences),
                        chosen=chosen,
                        rejected=rejected,
                        chosen_reward=chosen_reward,
                        rejected_reward=rejected_reward,
                        episode_name=episode_name,
                        episode_prompt=episode_prompt,
                    )
                )
    return rollout_groups, trajectory_preferences


def _write_policy_dataset(
    policy_step_records: list[dict[str, Any]],
    *,
    output_dir: Path,
    source_training_manifest: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    for record in policy_step_records:
        sample = _policy_sample_record(record)
        label = _evaluator_label_record(record, sample_id=str(sample["sample_id"]), group_id=str(sample["policy_input_hash"]))
        samples.append(sample)
        labels.append(label)
        group_counts[str(sample["policy_input_hash"])] = group_counts.get(str(sample["policy_input_hash"]), 0) + 1
    for label in labels:
        group_id = str(label["grpo"]["group_id"])
        label["grpo"]["eligible"] = group_counts.get(group_id, 0) >= 2

    samples_path = output_dir / "policy_samples.jsonl"
    labels_path = output_dir / "evaluator_labels.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"
    _write_jsonl(samples_path, samples)
    _write_jsonl(labels_path, labels)
    manifest = {
        "schema": POLICY_DATASET_MANIFEST_SCHEMA,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "source_training_manifest": source_training_manifest,
        "policy_samples_jsonl": str(samples_path),
        "evaluator_labels_jsonl": str(labels_path),
        "join_key": "sample_id",
        "sample_count": len(samples),
        "label_count": len(labels),
        "grpo_eligible_sample_count": sum(1 for label in labels if label["grpo"]["eligible"]),
        "policy_input_allowlist": [
            "goal",
            "actor_prompt_path",
            "actor_prompt_text",
            "image_paths",
            "observation",
            "tool_feedback",
        ],
        "forbidden_policy_fields": ["pose", "scene", "objects", "object_metadata", "nearest_target", "target_pose", "distance_m", "detections"],
        "privileged_label_file": True,
        "no_hardcoded_labels_or_colors_required": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _policy_sample_record(record: dict[str, Any]) -> dict[str, Any]:
    policy_input = record.get("policy_input") if isinstance(record.get("policy_input"), dict) else {}
    policy_output = record.get("policy_output") if isinstance(record.get("policy_output"), dict) else {}
    actor_action = policy_output.get("actor_action", {})
    executed_action = policy_output.get("executed_action", {})
    sample_policy_input = {
        "goal": policy_input.get("goal"),
        "actor_prompt_path": policy_input.get("actor_prompt_path"),
        "actor_prompt_text": policy_input.get("actor_prompt_text"),
        "image_paths": policy_input.get("image_paths", []),
        "observation": policy_input.get("observation", {}),
        "tool_feedback": sanitize_memory(record.get("tool_feedback", {})),
    }
    sample_id = str(record.get("record_id") or _hash_policy_input(sample_policy_input))
    return {
        "schema": POLICY_SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "source_policy_step_id": record.get("record_id"),
        "episode_rollout_id": (record.get("source") or {}).get("trial_id") if isinstance(record.get("source"), dict) else None,
        "policy_input_hash": _hash_policy_input(sample_policy_input),
        "policy_input": sample_policy_input,
        "target": {
            "action_json": actor_action,
            "executed_action_json": executed_action,
            "actor_equals_executed": _canonical_json(actor_action) == _canonical_json(executed_action),
            "target_source": "actor_action",
            "memory_update_json": policy_output.get("actor_memory_update", {}),
        },
        "sft": {
            "include_candidate": bool((record.get("policy_input_audit") or {}).get("passes_policy_input_audit", False)),
        },
    }


def _evaluator_label_record(record: dict[str, Any], *, sample_id: str, group_id: str) -> dict[str, Any]:
    reward = record.get("evaluator_reward") if isinstance(record.get("evaluator_reward"), dict) else {}
    post_action_score = reward.get("post_action_score") if isinstance(reward.get("post_action_score"), dict) else {}
    step_reward = _optional_float(reward.get("candidate_step_reward")) or 0.0
    episode_reward = _optional_float(reward.get("candidate_episode_reward")) or 0.0
    return {
        "schema": EVALUATOR_LABEL_SCHEMA,
        "sample_id": sample_id,
        "privileged": True,
        "reward": {
            "candidate_step_reward": step_reward,
            "candidate_episode_reward": episode_reward,
            "distance_delta_from_previous_step_m": reward.get("distance_delta_from_previous_step_m"),
            "post_action_distance_m": post_action_score.get("distance_m"),
            "episode_success": bool(reward.get("episode_success")),
            "final_distance_m": reward.get("final_distance_m"),
        },
        "ppo": {
            "reward": step_reward,
        },
        "grpo": {
            "group_id": group_id,
            "eligible": False,
        },
        "sft": {
            "weight": _sft_weight(episode_reward, step_reward),
        },
    }


def _hash_policy_input(policy_input: dict[str, Any]) -> str:
    encoded = _canonical_json(policy_input).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _sft_weight(episode_reward: float, step_reward: float) -> float:
    if episode_reward >= 1.0:
        return 1.0
    if step_reward > 0:
        return round(0.5 + min(step_reward, 1.0) * 0.5, 6)
    return 0.0


def _rollout_training_view(record: dict[str, Any], *, rank: int) -> dict[str, Any]:
    reward = record.get("evaluator_reward") if isinstance(record.get("evaluator_reward"), dict) else {}
    return {
        "record_id": record.get("record_id"),
        "rank": rank,
        "variant": record.get("variant"),
        "runner": record.get("runner"),
        "model": record.get("model"),
        "prompt_profile": record.get("prompt_profile"),
        "success": bool(record.get("success")),
        "reason": record.get("reason"),
        "step_record_count": int(record.get("step_record_count") or 0),
        "policy_steps_jsonl": record.get("policy_steps_jsonl"),
        "policy_input_allowlist": record.get("policy_input_allowlist", []),
        "evaluator_reward": {
            "privileged": True,
            "candidate_episode_reward": _episode_reward_value(record),
            "final_distance_m": reward.get("final_distance_m"),
            "success_radius_m": reward.get("success_radius_m"),
        },
    }


def _trajectory_preference_record(
    *,
    group_id: str,
    index: int,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    chosen_reward: float,
    rejected_reward: float,
    episode_name: str,
    episode_prompt: str,
) -> dict[str, Any]:
    return {
        "schema": TRAJECTORY_PREFERENCE_SCHEMA,
        "record_id": f"{group_id}_preference_{index:04d}",
        "rollout_group_id": group_id,
        "episode": {
            "name": episode_name,
            "prompt": episode_prompt,
        },
        "policy_context": {
            "goal_prompt": episode_prompt,
            "chosen_policy_steps_jsonl": chosen.get("policy_steps_jsonl"),
            "rejected_policy_steps_jsonl": rejected.get("policy_steps_jsonl"),
            "reward_labels_excluded_from_policy_input": True,
        },
        "chosen_rollout": {
            "record_id": chosen.get("record_id"),
            "variant": chosen.get("variant"),
            "runner": chosen.get("runner"),
            "model": chosen.get("model"),
            "success": bool(chosen.get("success")),
            "reason": chosen.get("reason"),
        },
        "rejected_rollout": {
            "record_id": rejected.get("record_id"),
            "variant": rejected.get("variant"),
            "runner": rejected.get("runner"),
            "model": rejected.get("model"),
            "success": bool(rejected.get("success")),
            "reason": rejected.get("reason"),
        },
        "evaluator_preference": {
            "privileged": True,
            "chosen_candidate_episode_reward": chosen_reward,
            "rejected_candidate_episode_reward": rejected_reward,
            "reward_margin": round(chosen_reward - rejected_reward, 6),
            "reward_source": "hidden evaluator success/final distance; not included in policy_context as model input",
        },
    }


def _episode_reward_value(record: dict[str, Any]) -> float:
    reward = record.get("evaluator_reward") if isinstance(record.get("evaluator_reward"), dict) else {}
    value = _optional_float(reward.get("candidate_episode_reward"))
    return float(value) if value is not None else -1.0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _actor_outputs_by_step(path: Path) -> dict[int, str]:
    outputs: dict[int, str] = {}
    if not path.exists():
        return outputs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "actor":
            continue
        try:
            step = int(event.get("step", -1))
        except (TypeError, ValueError):
            continue
        output = event.get("output")
        if isinstance(output, str):
            outputs[step] = output
    return outputs


def _policy_input_audit(actor_prompt_text: str | None, memory: dict[str, Any]) -> dict[str, Any]:
    text = actor_prompt_text or ""
    observation = sanitize_memory(memory.get("observation", {}))
    payload = json.dumps({"prompt": text, "observation": observation}, sort_keys=True, default=str).lower()
    found = [token for token in FORBIDDEN_POLICY_TOKENS if token in payload]
    return {
        "forbidden_tokens_present": found,
        "passes_policy_input_audit": not found,
    }


def _canonical_actor_output(memory: dict[str, Any]) -> str:
    payload = {
        "thought": str(memory.get("actor_action", {}).get("thought", "")) if isinstance(memory.get("actor_action"), dict) else "",
        "action": memory.get("actor_action", {}),
        "memory_update": memory.get("actor_memory_update", {}),
        "save_frames": memory.get("saved_frames", []),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _inferred_actor_image_paths(memory: dict[str, Any], previous_memory: list[dict[str, Any]], *, policy_dir: Path) -> list[str]:
    paths: list[str] = []
    observation = memory.get("observation", {})
    if isinstance(observation, dict) and observation.get("path"):
        paths.append(_path_text(observation["path"], policy_dir=policy_dir))
    previous_motion = _latest_prior_tool_path(previous_memory, "motion_contact_sheet", policy_dir=policy_dir)
    if previous_motion:
        paths.append(previous_motion)
    previous_topomap = _latest_prior_tool_path(previous_memory, "topomap_contact_sheet", policy_dir=policy_dir)
    if previous_topomap:
        paths.append(previous_topomap)
    return paths


def _latest_prior_tool_path(previous_memory: list[dict[str, Any]], key: str, *, policy_dir: Path) -> str | None:
    for record in reversed(previous_memory):
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        value = result.get(key)
        if value:
            return _path_text(value, policy_dir=policy_dir)
    return None


def _safe_hidden_score(score: Any) -> dict[str, Any]:
    if not isinstance(score, dict):
        return {}
    safe: dict[str, Any] = {}
    if isinstance(score.get("success"), bool):
        safe["success"] = score["success"]
    distance = _optional_float(score.get("distance_m"))
    if distance is not None:
        safe["distance_m"] = distance
    return safe


def _candidate_step_reward(post_score: dict[str, Any], distance_delta: float | None) -> float:
    if post_score.get("success") is True:
        return 1.0
    if distance_delta is None:
        return 0.0
    return round(max(-1.0, min(1.0, float(distance_delta))), 6)


def _candidate_episode_reward(summary: dict[str, Any]) -> float:
    if summary.get("success"):
        return 1.0
    distance = _optional_float(summary.get("final_distance_m"))
    if distance is None:
        return -1.0
    return round(-min(max(distance, 0.0), 5.0) / 5.0, 6)


def _source_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": summary.get("run_dir"),
        "policy_dir": summary.get("policy_dir"),
        "episode_summary": str(Path(summary.get("run_dir", "")) / "episode_summary.json") if summary.get("run_dir") else None,
        "trial_id": summary.get("trial_id"),
        "slot_id": summary.get("slot_id"),
    }


def _episode_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": summary.get("episode"),
        "scene": summary.get("scene"),
        "prompt": summary.get("prompt"),
        "success_radius_m": _optional_float(summary.get("success_radius_m")),
        "step_count": int(summary.get("step_count") or 0),
    }


def _discover_episode_summaries(input_paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for input_path in input_paths:
        path = input_path.expanduser()
        if path.is_file() and path.name == "episode_summary.json":
            found.append(path)
        elif path.is_file() and path.name == "trial_summary.json":
            summary = json.loads(path.read_text(encoding="utf-8"))
            run_dir = summary.get("run_dir")
            if run_dir and (Path(run_dir) / "episode_summary.json").exists():
                found.append(Path(run_dir) / "episode_summary.json")
        elif path.is_dir():
            found.extend(path.glob("**/episode_summary.json"))
    return sorted(set(found))


def _load_episode_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("run_dir", str(path.parent))
    return data


def _run_record_id(summary: dict[str, Any]) -> str:
    return str(summary.get("trial_id") or summary.get("episode") or _safe_id(str(summary.get("run_dir", "run"))))


def _resolve_path(value: Any, *, base: Path) -> Path:
    if not value:
        return base
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    candidate = base / path
    return candidate if candidate.exists() else path


def _path_text(value: Any, *, policy_dir: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str(policy_dir / path)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return text.strip("-")[:180] or "unnamed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Episode summary, trial summary, or directory to scan.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="manual_export")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-prompt-text", action="store_true", help="Store prompt paths but omit actor prompt text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_training_data_from_paths(
        args.input,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        include_prompt_text=not args.no_prompt_text,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
