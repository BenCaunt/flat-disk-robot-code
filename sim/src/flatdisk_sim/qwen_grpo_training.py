"""Materialize Qwen trajectory groups for future GRPO/PPO training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Iterable

from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS
from .training_export import FORBIDDEN_POLICY_TOKENS


QWEN_GRPO_MANIFEST_SCHEMA = "flatdisk.qwen_grpo_training_manifest.v1"
QWEN_GRPO_GROUP_SCHEMA = "flatdisk.qwen_grpo_rollout_group.v1"
QWEN_PPO_STEP_SAMPLE_SCHEMA = "flatdisk.qwen_ppo_step_sample.v1"


def prepare_qwen_grpo_training(
    input_paths: Path | Iterable[Path],
    *,
    output_dir: Path,
    require_existing_images: bool = True,
) -> dict[str, Any]:
    training_manifests = _discover_training_manifests(_coerce_input_paths(input_paths))
    if not training_manifests:
        raise FileNotFoundError("no training_manifest.json files found")
    source_groups: list[dict[str, Any]] = []
    missing_rollout_group_manifests: list[str] = []
    for training_manifest in training_manifests:
        manifest = json.loads(training_manifest.read_text(encoding="utf-8"))
        manifest["_manifest_path"] = str(training_manifest)
        rollout_groups_path = _resolve_manifest_path(manifest, "rollout_groups_jsonl")
        rollout_groups = _read_jsonl_if_exists(rollout_groups_path)
        if rollout_groups_path is None or not rollout_groups_path.exists():
            missing_rollout_group_manifests.append(str(training_manifest))
        for group in rollout_groups:
            source_groups.append(_with_source_manifest(group, training_manifest))
    merged_groups = _merge_rollout_groups(source_groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    groups_path = output_dir / "qwen_grpo_rollout_groups.jsonl"
    ppo_steps_path = output_dir / "qwen_ppo_step_samples.jsonl"
    manifest_path = output_dir / "qwen_grpo_training_manifest.json"
    records, ppo_step_samples, validation = _materialize_groups(
        merged_groups,
        require_existing_images=require_existing_images,
    )
    _write_jsonl(groups_path, records)
    _write_jsonl(ppo_steps_path, ppo_step_samples)
    blockers = list(validation["blockers"])
    if missing_rollout_group_manifests:
        blockers.append("missing rollout_groups_jsonl for manifest(s): " + ", ".join(missing_rollout_group_manifests))
    if not source_groups:
        blockers.append("rollout_groups_jsonl contains no rollout groups")
    if validation["trainable_group_count"] == 0:
        blockers.append("no rollout group has at least two trainable Qwen trajectory candidates")

    result = {
        "schema": QWEN_GRPO_MANIFEST_SCHEMA,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "status": "ready" if not blockers else "not_ready",
        "blockers": blockers,
        "warnings": validation["warnings"],
        "input": [str(path) for path in _coerce_input_paths(input_paths)],
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "source_mode": "merged_training_exports" if len(training_manifests) > 1 else "single_training_export",
        "source_training_manifests": [str(path) for path in training_manifests],
        "source_rollout_group_count": len(source_groups),
        "comparison_key_fields": ["episode.name", "episode.prompt"],
        "qwen_grpo_rollout_groups_jsonl": str(groups_path),
        "qwen_ppo_step_samples_jsonl": str(ppo_steps_path),
        "group_count": len(records),
        "trainable_group_count": validation["trainable_group_count"],
        "candidate_count": validation["candidate_count"],
        "trainable_candidate_count": validation["trainable_candidate_count"],
        "step_sample_count": validation["step_sample_count"],
        "ppo_step_sample_count": len(ppo_step_samples),
        "missing_image_count": validation["missing_image_count"],
        "missing_images": validation["missing_images"],
        "forbidden_qwen_message_token_hits": validation["forbidden_qwen_message_token_hits"],
        "audit": {
            "trajectory_rewards_excluded_from_messages": True,
            "requires_actor_equals_executed": True,
            "require_existing_images": require_existing_images,
            "policy_input_only": True,
        },
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _materialize_groups(
    rollout_groups: list[dict[str, Any]],
    *,
    require_existing_images: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ppo_step_samples: list[dict[str, Any]] = []
    warnings: list[str] = []
    blockers: list[str] = []
    missing_images: set[str] = set()
    forbidden_hits: Counter[str] = Counter()
    trainable_group_count = 0
    candidate_count = 0
    trainable_candidate_count = 0
    step_sample_count = 0
    for index, group in enumerate(rollout_groups):
        candidates = []
        trainable_candidates = 0
        for rollout in group.get("rollouts", []) if isinstance(group.get("rollouts"), list) else []:
            candidate = _candidate_from_rollout(
                rollout,
                require_existing_images=require_existing_images,
            )
            candidate_count += 1
            step_sample_count += len(candidate["step_samples"])
            missing_images.update(candidate["audit"]["missing_images"])
            for token in candidate["audit"]["forbidden_qwen_message_token_hits"]:
                forbidden_hits[token] += 1
            if candidate["trainable"]:
                trainable_candidates += 1
                trainable_candidate_count += 1
                ppo_step_samples.extend(_ppo_step_samples_from_candidate(candidate))
            candidates.append(candidate)
        if trainable_candidates >= 2:
            trainable_group_count += 1
        elif candidates:
            warnings.append(
                f"rollout group {group.get('record_id') or index} has fewer than two trainable candidates"
            )
        records.append(
            {
                "schema": QWEN_GRPO_GROUP_SCHEMA,
                "group_id": str(group.get("record_id") or f"group_{index:04d}"),
                "source_rollout_group_id": group.get("record_id"),
                "episode": group.get("episode", {}),
                "policy_context": group.get("policy_context", {}),
                "training_uses": ["grpo", "ppo", "trajectory_ranking"],
                "candidate_count": len(candidates),
                "trainable_candidate_count": trainable_candidates,
                "candidates": candidates,
                "audit": {
                    "trajectory_rewards_excluded_from_messages": True,
                    "reward_ranking_source": "source rollout group evaluator_reward channel",
                },
            }
        )
    if require_existing_images and missing_images:
        blockers.append(f"{len(missing_images)} Qwen GRPO image reference(s) are missing")
    if forbidden_hits:
        blockers.append(
            "Qwen GRPO messages contain forbidden privileged token(s): "
            + ", ".join(sorted(forbidden_hits))
        )
    return records, ppo_step_samples, {
        "blockers": blockers,
        "warnings": warnings,
        "trainable_group_count": trainable_group_count,
        "candidate_count": candidate_count,
        "trainable_candidate_count": trainable_candidate_count,
        "step_sample_count": step_sample_count,
        "missing_image_count": len(missing_images),
        "missing_images": sorted(missing_images),
        "forbidden_qwen_message_token_hits": dict(sorted(forbidden_hits.items())),
    }


def _candidate_from_rollout(
    rollout: dict[str, Any],
    *,
    require_existing_images: bool,
) -> dict[str, Any]:
    training_manifest_path = Path(str(rollout.get("_source_training_manifest_path") or "training_manifest.json"))
    policy_steps_path = _resolve_rollout_path(
        rollout.get("policy_steps_jsonl"),
        training_manifest_path=training_manifest_path,
    )
    steps = _read_jsonl_if_exists(policy_steps_path)
    step_samples = [_step_sample(step, training_manifest_path=training_manifest_path) for step in steps]
    missing_images = sorted(
        {
            path
            for sample in step_samples
            for path in sample["image_paths"]
            if not Path(path).exists()
        }
    )
    forbidden_hits = _forbidden_tokens(
        [
            sample["prompt_messages"]
            for sample in step_samples
        ]
        + [sample["assistant_target_json"] for sample in step_samples]
    )
    action_replaced_count = sum(1 for step in steps if _action_replaced(step))
    failed_policy_audit_count = sum(
        1
        for step in steps
        if not bool((step.get("policy_input_audit") or {}).get("passes_policy_input_audit"))
    )
    candidate_blockers: list[str] = []
    if policy_steps_path is None or not policy_steps_path.exists():
        candidate_blockers.append(f"missing policy_steps_jsonl: {rollout.get('policy_steps_jsonl')}")
    if not steps:
        candidate_blockers.append("rollout has no policy step records")
    if action_replaced_count:
        candidate_blockers.append(f"{action_replaced_count} step(s) had actor action replaced before execution")
    if failed_policy_audit_count:
        candidate_blockers.append(f"{failed_policy_audit_count} step(s) failed policy input audit")
    if forbidden_hits:
        candidate_blockers.append(
            "candidate messages contain forbidden privileged token(s): " + ", ".join(sorted(forbidden_hits))
        )
    if require_existing_images and missing_images:
        candidate_blockers.append(f"{len(missing_images)} image reference(s) are missing")
    reward = ((rollout.get("evaluator_reward") or {}).get("candidate_episode_reward"))
    return {
        "rollout_id": rollout.get("record_id"),
        "rank": rollout.get("rank"),
        "variant": rollout.get("variant"),
        "runner": rollout.get("runner"),
        "model": rollout.get("model"),
        "success": bool(rollout.get("success")),
        "reason": rollout.get("reason"),
        "source_policy_steps_jsonl": str(policy_steps_path) if policy_steps_path else "",
        "trainable": not candidate_blockers,
        "blockers": candidate_blockers,
        "step_count": len(step_samples),
        "step_samples": step_samples,
        "assistant_trajectory_json": json.dumps(
            {"trajectory": [sample["assistant_target_json"] for sample in step_samples]},
            sort_keys=True,
        ),
        "evaluator_reward": {
            "privileged": True,
            "candidate_episode_reward": reward,
            "reward_source": "hidden evaluator reward; never included in prompt_messages or assistant target",
        },
        "audit": {
            "actor_action_replaced_step_count": action_replaced_count,
            "failed_policy_audit_step_count": failed_policy_audit_count,
            "missing_images": missing_images,
            "forbidden_qwen_message_token_hits": forbidden_hits,
        },
    }


def _ppo_step_samples_from_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    step_samples = candidate["step_samples"]
    for index, sample in enumerate(step_samples):
        records.append(
            {
                "schema": QWEN_PPO_STEP_SAMPLE_SCHEMA,
                "sample_id": sample["sample_id"],
                "source_rollout_id": candidate["rollout_id"],
                "source_policy_steps_jsonl": candidate["source_policy_steps_jsonl"],
                "terminal": index == len(step_samples) - 1,
                "prompt_messages": sample["prompt_messages"],
                "assistant_target_json": sample["assistant_target_json"],
                "image_paths": sample["image_paths"],
                "evaluator_reward": sample["evaluator_reward"],
                "audit": {
                    "reward_excluded_from_messages": True,
                    "source_rollout_trainable": True,
                },
            }
        )
    return records


def _step_sample(step: dict[str, Any], *, training_manifest_path: Path) -> dict[str, Any]:
    policy_input = step.get("policy_input") if isinstance(step.get("policy_input"), dict) else {}
    policy_output = step.get("policy_output") if isinstance(step.get("policy_output"), dict) else {}
    reward = step.get("evaluator_reward") if isinstance(step.get("evaluator_reward"), dict) else {}
    image_paths = [
        str(_resolve_policy_artifact_path(path, training_manifest_path=training_manifest_path))
        for path in policy_input.get("image_paths", [])
        if str(path)
    ]
    content: list[dict[str, Any]] = [{"type": "text", "text": _prompt_text(policy_input)}]
    content.extend({"type": "image", "image": path} for path in image_paths)
    target = {
        "action": policy_output.get("actor_action", {}),
        "memory_update": policy_output.get("actor_memory_update", {}),
    }
    return {
        "sample_id": str(step.get("record_id") or ""),
        "prompt_messages": [{"role": "user", "content": content}],
        "assistant_target_json": target,
        "image_paths": image_paths,
        "evaluator_reward": {
            "privileged": True,
            "candidate_step_reward": reward.get("candidate_step_reward"),
            "candidate_episode_reward": reward.get("candidate_episode_reward"),
            "episode_success": bool(reward.get("episode_success")),
            "reward_source": "hidden evaluator reward; never included in prompt_messages or assistant target",
        },
    }


def _prompt_text(policy_input: dict[str, Any]) -> str:
    actor_prompt = policy_input.get("actor_prompt_text")
    if actor_prompt:
        return str(actor_prompt)
    return json.dumps(
        {
            "goal": policy_input.get("goal"),
            "observation": policy_input.get("observation", {}),
            "tool_feedback": policy_input.get("tool_feedback", {}),
        },
        sort_keys=True,
    )


def _action_replaced(step: dict[str, Any]) -> bool:
    output = step.get("policy_output") if isinstance(step.get("policy_output"), dict) else {}
    return _canonical_json(output.get("actor_action", {})) != _canonical_json(output.get("executed_action", {}))


def _coerce_input_paths(input_paths: Path | Iterable[Path]) -> list[Path]:
    if isinstance(input_paths, Path):
        return [input_paths]
    return [Path(path) for path in input_paths]


def _discover_training_manifests(input_paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for input_path in input_paths:
        path = input_path.expanduser()
        if path.is_file() and path.name == "training_manifest.json":
            found.append(path)
        elif path.is_dir():
            candidates = [
                path / "training_manifest.json",
                path / "training_export" / "training_manifest.json",
            ]
            found.extend(candidate for candidate in candidates if candidate.exists())
            found.extend(path.glob("**/training_manifest.json"))
    return sorted(set(found))


def _with_source_manifest(group: dict[str, Any], training_manifest_path: Path) -> dict[str, Any]:
    result = dict(group)
    result["_source_training_manifest_path"] = str(training_manifest_path)
    rollouts = []
    for rollout in group.get("rollouts", []) if isinstance(group.get("rollouts"), list) else []:
        rollout_copy = dict(rollout)
        rollout_copy["_source_training_manifest_path"] = str(training_manifest_path)
        rollouts.append(rollout_copy)
    result["rollouts"] = rollouts
    return result


def _merge_rollout_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups:
        episode = group.get("episode") if isinstance(group.get("episode"), dict) else {}
        key = (str(episode.get("name") or ""), str(episode.get("prompt") or ""))
        record = merged.setdefault(
            key,
            {
                "record_id": _safe_id("qwen_grpo_" + "_".join(key)),
                "episode": {"name": key[0], "prompt": key[1]},
                "policy_context": group.get("policy_context", {}),
                "rollouts": [],
                "source_rollout_group_ids": [],
            },
        )
        record["source_rollout_group_ids"].append(group.get("record_id"))
        record["rollouts"].extend(group.get("rollouts", []) if isinstance(group.get("rollouts"), list) else [])
    for record in merged.values():
        ranked = sorted(record["rollouts"], key=_rollout_reward, reverse=True)
        for index, rollout in enumerate(ranked, 1):
            rollout["rank"] = index
        record["rollouts"] = ranked
    return sorted(merged.values(), key=lambda item: str(item["record_id"]))


def _resolve_manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = manifest.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    manifest_path = Path(str(manifest["_manifest_path"]))
    if path.is_absolute():
        relocated = _relocated_path(path, local_training_export_dir=manifest_path.parent)
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


def _resolve_rollout_path(value: Any, *, training_manifest_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    if path.is_absolute():
        relocated = _relocated_path(path, local_training_export_dir=training_manifest_path.parent)
        if relocated is not None and relocated.exists():
            return relocated
        return path
    candidates = [
        training_manifest_path.parent / path,
        training_manifest_path.parent / "runs" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _resolve_policy_artifact_path(value: Any, *, training_manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    if path.is_absolute():
        relocated = _relocated_path(path, local_training_export_dir=training_manifest_path.parent)
        if relocated is not None:
            return relocated
    return path


def _relocated_path(path: Path, *, local_training_export_dir: Path) -> Path | None:
    parts = path.parts
    if "training_export" not in parts:
        artifact_root = local_training_export_dir.parent
        if "trials" not in parts:
            return None
        index = len(parts) - 1 - list(reversed(parts)).index("trials")
        tail = parts[index:]
        return artifact_root / Path(*tail)
    index = len(parts) - 1 - list(reversed(parts)).index("training_export")
    tail = parts[index + 1 :]
    return local_training_export_dir / Path(*tail) if tail else local_training_export_dir


def _rollout_reward(rollout: dict[str, Any]) -> float:
    reward = rollout.get("evaluator_reward") if isinstance(rollout.get("evaluator_reward"), dict) else {}
    try:
        return float(reward.get("candidate_episode_reward"))
    except (TypeError, ValueError):
        return -1.0


def _forbidden_tokens(payloads: Iterable[Any]) -> list[str]:
    text = json.dumps(list(payloads), sort_keys=True, default=str).lower()
    tokens = (*DEFAULT_FORBIDDEN_MODEL_TOKENS, *FORBIDDEN_POLICY_TOKENS)
    return sorted({token for token in tokens if token.lower() in text})


def _read_jsonl_if_exists(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _safe_id(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower())
    return "_".join(part for part in cleaned.split("_") if part)[:160] or "qwen_grpo_group"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="training_export dir, run dir, or training_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_qwen_grpo_training(
        args.input,
        output_dir=args.output_dir,
        require_existing_images=not args.allow_missing_images,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "group_count": manifest["group_count"],
                "trainable_group_count": manifest["trainable_group_count"],
                "trainable_candidate_count": manifest["trainable_candidate_count"],
                "ppo_step_sample_count": manifest["ppo_step_sample_count"],
                "missing_image_count": manifest["missing_image_count"],
                "forbidden_qwen_message_token_hits": manifest["forbidden_qwen_message_token_hits"],
                "manifest_path": manifest["manifest_path"],
                "qwen_grpo_rollout_groups_jsonl": manifest["qwen_grpo_rollout_groups_jsonl"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and manifest["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
