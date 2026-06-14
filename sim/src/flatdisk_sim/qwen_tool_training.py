"""Prepare Qwen tool-use SFT records from navigation policy dataset exports."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from .llm_harness import action_to_dict, parse_prompt_action, validate_harness_action
from .training_export import FORBIDDEN_POLICY_TOKENS


QWEN_TOOL_SFT_SAMPLE_SCHEMA = "flatdisk.qwen_tool_sft_sample.v1"
QWEN_TOOL_REJECTED_SAMPLE_SCHEMA = "flatdisk.qwen_tool_sft_rejected_sample.v1"
QWEN_TOOL_ACTION_PREFERENCE_SCHEMA = "flatdisk.qwen_tool_action_preference.v1"
QWEN_TOOL_DPO_SAMPLE_SCHEMA = "flatdisk.qwen_tool_dpo_sample.v1"
QWEN_TOOL_TRAINING_MANIFEST_SCHEMA = "flatdisk.qwen_tool_training_manifest.v1"
QWEN_TOOL_AUDIT_SCHEMA = "flatdisk.qwen_tool_training_audit.v1"

DEFAULT_FORBIDDEN_MODEL_TOKENS = tuple(
    sorted(
        set(
            [
                *FORBIDDEN_POLICY_TOKENS,
                "target_pose",
                "nearest_target",
                "object_metadata",
                "hidden_score_for_evaluator_only",
                "post_action_distance_m",
                "final_distance_m",
                "candidate_step_reward",
                "candidate_episode_reward",
                "success_radius_m",
                "stdout_tail",
                "stderr_tail",
            ]
        )
    )
)


def prepare_qwen_tool_training(
    input_path: Path,
    *,
    output_dir: Path,
    image_root: Path | None = None,
    min_sft_weight: float = 0.0,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    dataset = _resolve_policy_dataset(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _read_jsonl(dataset["policy_samples_jsonl"])
    labels_by_id = {str(label.get("sample_id")): label for label in _read_jsonl(dataset["evaluator_labels_jsonl"])}

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    preference_rejection_counts: Counter[str] = Counter()
    for sample in samples:
        materialized, reject_reasons = _materialize_sample(
            sample,
            labels_by_id.get(str(sample.get("sample_id"))),
            dataset_dir=Path(dataset["dataset_dir"]),
            image_root=image_root,
            min_sft_weight=min_sft_weight,
            require_existing_images=require_existing_images,
        )
        if reject_reasons:
            reason_counts.update(reject_reasons)
            rejected.append(
                {
                    "schema": QWEN_TOOL_REJECTED_SAMPLE_SCHEMA,
                    "sample_id": sample.get("sample_id"),
                    "source_policy_sample_id": sample.get("sample_id"),
                    "reject_reasons": reject_reasons,
                    "policy_input_hash": sample.get("policy_input_hash"),
                }
            )
        else:
            accepted.append(materialized)
        preference, preference_reject_reasons = _materialize_action_preference(
            sample,
            dataset_dir=Path(dataset["dataset_dir"]),
            image_root=image_root,
            require_existing_images=require_existing_images,
        )
        if preference_reject_reasons:
            preference_rejection_counts.update(preference_reject_reasons)
        if preference is not None:
            preferences.append(preference)

    accepted_path = output_dir / "qwen_sft_messages.jsonl"
    rejected_path = output_dir / "rejected_samples.jsonl"
    preferences_path = output_dir / "qwen_action_preferences.jsonl"
    dpo_path = output_dir / "qwen_dpo_messages.jsonl"
    audit_path = output_dir / "training_audit.json"
    manifest_path = output_dir / "qwen_tool_training_manifest.json"
    dpo_preferences = [_dpo_sample_from_action_preference(preference) for preference in preferences]
    _write_jsonl(accepted_path, accepted)
    _write_jsonl(rejected_path, rejected)
    _write_jsonl(preferences_path, preferences)
    _write_jsonl(dpo_path, dpo_preferences)
    audit = {
        "schema": QWEN_TOOL_AUDIT_SCHEMA,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "action_preference_count": len(preferences),
        "dpo_preference_count": len(dpo_preferences),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "action_preference_rejection_reasons": dict(sorted(preference_rejection_counts.items())),
        "require_existing_images": require_existing_images,
        "min_sft_weight": min_sft_weight,
        "forbidden_model_tokens": list(DEFAULT_FORBIDDEN_MODEL_TOKENS),
        "policy_input_channel_only": True,
        "evaluator_labels_used_for_filtering_only": True,
        "action_preferences_use_model_facing_prompt_only": True,
        "dpo_columns": ["prompt", "chosen", "rejected", "images"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": QWEN_TOOL_TRAINING_MANIFEST_SCHEMA,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "source_policy_dataset_dir": str(dataset["dataset_dir"]),
        "source_policy_samples_jsonl": str(dataset["policy_samples_jsonl"]),
        "source_evaluator_labels_jsonl": str(dataset["evaluator_labels_jsonl"]),
        "qwen_sft_messages_jsonl": str(accepted_path),
        "rejected_samples_jsonl": str(rejected_path),
        "qwen_action_preferences_jsonl": str(preferences_path),
        "qwen_dpo_messages_jsonl": str(dpo_path),
        "training_audit_json": str(audit_path),
        "sample_count": len(samples),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "action_preference_count": len(preferences),
        "dpo_preference_count": len(dpo_preferences),
        "message_format": "qwen-vl-chat-messages",
        "preference_message_format": "trl-explicit-prompt-vlm-preference",
        "target_format": "json_object_with_thought_action_memory_update",
        "policy_input_only_in_messages": True,
        "privileged_labels_excluded_from_messages": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _resolve_policy_dataset(input_path: Path) -> dict[str, Any]:
    path = input_path.expanduser()
    if path.is_file():
        manifest_path = path
    elif (path / "dataset_manifest.json").exists():
        manifest_path = path / "dataset_manifest.json"
    elif (path / "policy_dataset_v1" / "dataset_manifest.json").exists():
        manifest_path = path / "policy_dataset_v1" / "dataset_manifest.json"
    else:
        raise FileNotFoundError(f"could not find policy_dataset_v1/dataset_manifest.json under {input_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = Path(str(manifest.get("output_dir") or manifest_path.parent))
    if not dataset_dir.exists():
        dataset_dir = manifest_path.parent
    return {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "policy_samples_jsonl": _manifest_path(manifest, "policy_samples_jsonl", default=dataset_dir / "policy_samples.jsonl"),
        "evaluator_labels_jsonl": _manifest_path(manifest, "evaluator_labels_jsonl", default=dataset_dir / "evaluator_labels.jsonl"),
    }


def _manifest_path(manifest: dict[str, Any], key: str, *, default: Path) -> Path:
    value = manifest.get(key)
    if not value:
        return default
    path = Path(str(value))
    if path.exists():
        return path
    if path.is_absolute():
        local_name_candidate = default.parent / path.name
        if local_name_candidate.exists():
            return local_name_candidate
        return default
    output_dir = Path(str(manifest.get("output_dir") or ""))
    candidate = output_dir / path
    if candidate.exists():
        return candidate
    return default if default.exists() else path


def _materialize_sample(
    sample: dict[str, Any],
    label: dict[str, Any] | None,
    *,
    dataset_dir: Path,
    image_root: Path | None,
    min_sft_weight: float,
    require_existing_images: bool,
) -> tuple[dict[str, Any], list[str]]:
    reject_reasons: list[str] = []
    if label is None:
        reject_reasons.append("missing_evaluator_label")
    target = sample.get("target") if isinstance(sample.get("target"), dict) else {}
    policy_input = sample.get("policy_input") if isinstance(sample.get("policy_input"), dict) else {}
    action = target.get("action_json") if isinstance(target.get("action_json"), dict) else {}
    if not (sample.get("sft") or {}).get("include_candidate"):
        reject_reasons.append("sft_candidate_disabled")
    weight = _label_sft_weight(label)
    if weight <= min_sft_weight:
        reject_reasons.append("non_positive_sft_weight")
    if target.get("actor_equals_executed") is not True:
        reject_reasons.append("actor_action_differs_from_executed_action")
    if not action.get("tool"):
        reject_reasons.append("missing_action_tool")
    validated_action = action_to_dict(validate_harness_action(parse_prompt_action(action)))
    if _action_contract_payload(validated_action) != _action_contract_payload(action):
        reject_reasons.append("invalid_or_unbounded_action")

    image_paths = _resolve_image_paths(policy_input.get("image_paths", []), dataset_dir=dataset_dir, image_root=image_root)
    missing_images = [str(path) for path in image_paths if not path.exists()]
    if require_existing_images and missing_images:
        reject_reasons.append("missing_image")

    user_text = _user_text(policy_input)
    assistant_payload = _assistant_payload(validated_action, target.get("memory_update_json"))
    messages = _qwen_messages(user_text, image_paths=image_paths, assistant_payload=assistant_payload)
    forbidden = _forbidden_tokens(messages)
    if forbidden:
        reject_reasons.append("forbidden_model_token")

    record = {
        "schema": QWEN_TOOL_SFT_SAMPLE_SCHEMA,
        "sample_id": sample.get("sample_id"),
        "source_policy_sample_id": sample.get("sample_id"),
        "source_policy_step_id": sample.get("source_policy_step_id"),
        "policy_input_hash": sample.get("policy_input_hash"),
        "messages": messages,
        "image_paths": [str(path) for path in image_paths],
        "assistant_target_json": assistant_payload,
        "sft_weight": weight,
        "training_weight": weight,
        "audit": {
            "forbidden_model_tokens_present": forbidden,
            "filter_reasons": sorted(set(reject_reasons)),
            "image_count": len(image_paths),
            "missing_images": missing_images,
            "actor_equals_executed": target.get("actor_equals_executed") is True,
            "privileged_scan_passed": not forbidden,
            "target_source": target.get("target_source") or "actor_action",
        },
        "metadata": {
            "episode_rollout_id": sample.get("episode_rollout_id"),
            "policy_input_hash": sample.get("policy_input_hash"),
            "label_joined": label is not None,
        },
    }
    return record, sorted(set(reject_reasons))


def _materialize_action_preference(
    sample: dict[str, Any],
    *,
    dataset_dir: Path,
    image_root: Path | None,
    require_existing_images: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    reject_reasons: list[str] = []
    target = sample.get("target") if isinstance(sample.get("target"), dict) else {}
    policy_input = sample.get("policy_input") if isinstance(sample.get("policy_input"), dict) else {}
    actor_action = target.get("action_json") if isinstance(target.get("action_json"), dict) else {}
    executed_action = target.get("executed_action_json") if isinstance(target.get("executed_action_json"), dict) else {}
    if target.get("actor_equals_executed") is True:
        return None, ["actor_action_matches_executed_action"]
    if not actor_action.get("tool"):
        reject_reasons.append("missing_actor_action_tool")
    if not executed_action.get("tool"):
        reject_reasons.append("missing_executed_action_tool")

    validated_rejected = action_to_dict(validate_harness_action(parse_prompt_action(actor_action)))
    validated_chosen = action_to_dict(validate_harness_action(parse_prompt_action(executed_action)))
    if _action_contract_payload(validated_rejected) != _action_contract_payload(actor_action):
        reject_reasons.append("invalid_or_unbounded_actor_action")
    if _action_contract_payload(validated_chosen) != _action_contract_payload(executed_action):
        reject_reasons.append("invalid_or_unbounded_executed_action")

    image_paths = _resolve_image_paths(policy_input.get("image_paths", []), dataset_dir=dataset_dir, image_root=image_root)
    missing_images = [str(path) for path in image_paths if not path.exists()]
    if require_existing_images and missing_images:
        reject_reasons.append("missing_image")

    user_text = _user_text(policy_input)
    prompt_messages = _qwen_prompt_messages(user_text, image_paths=image_paths)
    forbidden = _forbidden_tokens(prompt_messages)
    if forbidden:
        reject_reasons.append("forbidden_model_token")
    if reject_reasons:
        return None, sorted(set(reject_reasons))

    chosen_payload = _assistant_payload(validated_chosen, {})
    rejected_payload = _assistant_payload(validated_rejected, target.get("memory_update_json"))
    return (
        {
            "schema": QWEN_TOOL_ACTION_PREFERENCE_SCHEMA,
            "sample_id": f"{sample.get('sample_id')}_actor_replacement_preference",
            "source_policy_sample_id": sample.get("sample_id"),
            "source_policy_step_id": sample.get("source_policy_step_id"),
            "policy_input_hash": sample.get("policy_input_hash"),
            "preference_source": "critic_or_harness_replacement",
            "preference_type": "executed_action_preferred_over_rejected_actor_action",
            "prompt_messages": prompt_messages,
            "image_paths": [str(path) for path in image_paths],
            "chosen_assistant_target_json": chosen_payload,
            "rejected_assistant_target_json": rejected_payload,
            "chosen_action_source": "executed_action_json",
            "rejected_action_source": "actor_action_json",
            "audit": {
                "forbidden_model_tokens_present": forbidden,
                "image_count": len(image_paths),
                "missing_images": missing_images,
                "actor_equals_executed": False,
                "privileged_scan_passed": not forbidden,
                "preference_labels_excluded_from_messages": True,
                "evaluator_reward_excluded_from_messages": True,
            },
            "metadata": {
                "episode_rollout_id": sample.get("episode_rollout_id"),
                "policy_input_hash": sample.get("policy_input_hash"),
            },
        },
        [],
    )


def _dpo_sample_from_action_preference(preference: dict[str, Any]) -> dict[str, Any]:
    chosen_message = _assistant_message(preference.get("chosen_assistant_target_json"))
    rejected_message = _assistant_message(preference.get("rejected_assistant_target_json"))
    image_paths = [str(path) for path in preference.get("image_paths", []) if str(path)]
    prompt_messages = preference.get("prompt_messages") if isinstance(preference.get("prompt_messages"), list) else []
    sample_id = str(preference.get("sample_id") or "preference")
    return {
        "schema": QWEN_TOOL_DPO_SAMPLE_SCHEMA,
        "sample_id": f"{sample_id}_dpo",
        "source_preference_sample_id": preference.get("sample_id"),
        "source_policy_sample_id": preference.get("source_policy_sample_id"),
        "source_policy_step_id": preference.get("source_policy_step_id"),
        "policy_input_hash": preference.get("policy_input_hash"),
        "prompt": prompt_messages,
        "chosen": [chosen_message],
        "rejected": [rejected_message],
        "images": image_paths,
        "image_paths": image_paths,
        "chosen_action_source": preference.get("chosen_action_source"),
        "rejected_action_source": preference.get("rejected_action_source"),
        "audit": {
            "derived_from_action_preference": True,
            "dpo_columns": ["prompt", "chosen", "rejected", "images"],
            "explicit_prompt": True,
            "vlm_image_column": "images",
            "preference_labels_excluded_from_messages": True,
            "evaluator_reward_excluded_from_messages": True,
        },
        "metadata": preference.get("metadata") if isinstance(preference.get("metadata"), dict) else {},
    }


def _label_sft_weight(label: dict[str, Any] | None) -> float:
    if not isinstance(label, dict):
        return 0.0
    sft = label.get("sft") if isinstance(label.get("sft"), dict) else {}
    try:
        return float(sft.get("weight") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _resolve_image_paths(values: Any, *, dataset_dir: Path, image_root: Path | None) -> list[Path]:
    if not isinstance(values, list):
        return []
    resolved: list[Path] = []
    for value in values:
        path = Path(str(value))
        if path.exists():
            resolved.append(path)
            continue
        if path.is_absolute():
            relocated = _relocate_copied_artifact_path(path, dataset_dir=dataset_dir, image_root=image_root)
            resolved.append(relocated or path)
            continue
        if image_root is not None:
            resolved.append(image_root / path)
            continue
        candidate = dataset_dir / path
        if candidate.exists():
            resolved.append(candidate)
        else:
            resolved.append(path)
    return resolved


def _relocate_copied_artifact_path(path: Path, *, dataset_dir: Path, image_root: Path | None) -> Path | None:
    marker_suffixes = _artifact_marker_suffixes(path)
    if not marker_suffixes:
        return None
    roots: list[Path] = []
    if image_root is not None:
        roots.append(image_root)
    roots.extend(_nearby_artifact_roots(dataset_dir))
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        for suffix in marker_suffixes:
            candidate = root / suffix
            if candidate.exists():
                return candidate
    return None


def _nearby_artifact_roots(dataset_dir: Path) -> list[Path]:
    roots = [dataset_dir]
    roots.extend(list(dataset_dir.parents)[:4])
    return roots


def _artifact_marker_suffixes(path: Path) -> list[Path]:
    parts = path.parts
    suffixes: list[Path] = []
    for marker in ("trials", "training_export"):
        if marker not in parts:
            continue
        index = parts.index(marker)
        suffixes.append(Path(*parts[index:]))
    return suffixes


def _user_text(policy_input: dict[str, Any]) -> str:
    prompt = policy_input.get("actor_prompt_text")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    fallback = {
        "goal": policy_input.get("goal"),
        "observation": policy_input.get("observation", {}),
    }
    return json.dumps(fallback, sort_keys=True, default=str)


def _assistant_payload(action: dict[str, Any], memory_update: Any) -> dict[str, Any]:
    return {
        "thought": str(action.get("thought") or "")[:500],
        "action": {
            "tool": action.get("tool"),
            "args": action.get("args") if isinstance(action.get("args"), dict) else {},
        },
        "memory_update": memory_update if isinstance(memory_update, dict) else {},
    }


def _action_contract_payload(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": action.get("tool"),
        "args": action.get("args") if isinstance(action.get("args"), dict) else {},
    }


def _qwen_messages(user_text: str, *, image_paths: list[Path], assistant_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *_qwen_prompt_messages(user_text, image_paths=image_paths),
        {"role": "assistant", "content": json.dumps(assistant_payload, sort_keys=True, default=str)},
    ]


def _qwen_prompt_messages(user_text: str, *, image_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def _assistant_message(payload: Any) -> dict[str, str]:
    return {"role": "assistant", "content": json.dumps(payload if isinstance(payload, dict) else {}, sort_keys=True, default=str)}


def _forbidden_tokens(messages: list[dict[str, Any]]) -> list[str]:
    text = json.dumps(messages, sort_keys=True, default=str).lower()
    return [token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="training export dir, policy_dataset_v1 dir, or dataset_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root for relative image paths.")
    parser.add_argument("--min-sft-weight", type=float, default=0.0)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_qwen_tool_training(
        args.input,
        output_dir=args.output_dir,
        image_root=args.image_root,
        min_sft_weight=args.min_sft_weight,
        require_existing_images=not args.allow_missing_images,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
