"""Audit navigation training exports for SFT/PPO/GRPO readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Iterable

from .research_loop import DEFAULT_WARMHUB_REPO, _safe_id
from .research_warmhub import commit_ops
from .qwen_tool_training import DEFAULT_FORBIDDEN_MODEL_TOKENS
from .training_export import FORBIDDEN_POLICY_TOKENS


TRAINING_READINESS_SCHEMA = "flatdisk.nav_training_readiness.v1"


def analyze_training_readiness(
    input_paths: Iterable[Path],
    *,
    output_dir: Path,
    analysis_id: str | None = None,
    experiment_id: str | None = None,
    about: str | None = None,
    author: str = "flatdisk-sim-training-readiness",
) -> dict[str, Any]:
    input_path_list = list(input_paths)
    manifests = [_load_training_manifest(path) for path in _discover_training_manifests(input_path_list)]
    if not manifests:
        raise FileNotFoundError("no training_manifest.json files found")
    qwen_manifests = [_load_training_manifest(path) for path in _discover_qwen_tool_training_manifests(input_path_list)]

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = _safe_id(analysis_id or _default_analysis_id(manifests))
    runs = [_summarize_manifest(manifest) for manifest in manifests]
    qwen_tool_training_runs = [_summarize_qwen_tool_training_manifest(manifest) for manifest in qwen_manifests]
    aggregate = _aggregate(runs)
    aggregate = _aggregate_with_qwen_tool_training(aggregate, qwen_tool_training_runs)
    readiness = _readiness_decision(aggregate)
    report = {
        "schema": TRAINING_READINESS_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "created_at": strftime("%Y-%m-%dT%H:%M:%SZ", gmtime()),
        "input_count": len(manifests),
        "runs": runs,
        "qwen_tool_training_runs": qwen_tool_training_runs,
        "aggregate": aggregate,
        "readiness": readiness,
        "policy_constraints": [
            "Policy samples must not contain hidden target pose, object metadata, evaluator distance, or detector arrays.",
            "Evaluator labels may be used only for offline reward, ranking, filtering, PPO, GRPO, or SFT weighting.",
            "Training candidates must preserve the real robot I/O contract: RGB/contact-sheet paths, bounded tools, IMU, memory, and tool results.",
        ],
    }
    report_path = output_dir / "training_readiness.json"
    markdown_path = output_dir / "training_readiness.md"
    ops_path = output_dir / "warmhub_ops.json"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report["warmhub_ops_path"] = str(ops_path)
    report["warmhub_ops"] = _warmhub_ops(report, about=about or _default_about(experiment_id), author=author)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    ops_path.write_text(json.dumps(report["warmhub_ops"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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


def _discover_qwen_tool_training_manifests(input_paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for input_path in input_paths:
        path = input_path.expanduser()
        if path.is_file() and path.name == "qwen_tool_training_manifest.json":
            found.append(path)
        elif path.is_dir():
            candidates = [
                path / "qwen_tool_training_manifest.json",
                path / "qwen_tool_training" / "qwen_tool_training_manifest.json",
            ]
            found.extend(candidate for candidate in candidates if candidate.exists())
            found.extend(path.glob("**/qwen_tool_training_manifest.json"))
    return sorted(set(found))


def _load_training_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["_manifest_path"] = str(path)
    return manifest


def _summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(manifest["_manifest_path"]))
    policy_dataset_manifest = _load_optional_json(_resolve_manifest_path(manifest, "policy_dataset_manifest_path"))
    policy_sample_path = _resolve_manifest_path(manifest, "policy_samples_jsonl")
    evaluator_label_path = _resolve_manifest_path(manifest, "evaluator_labels_jsonl")
    rollout_groups_path = _resolve_manifest_path(manifest, "rollout_groups_jsonl")
    preferences_path = _resolve_manifest_path(manifest, "trajectory_preferences_jsonl")
    review_traces_path = _resolve_manifest_path(manifest, "policy_review_traces_jsonl")

    policy_samples = _read_jsonl_if_exists(policy_sample_path)
    evaluator_labels = _read_jsonl_if_exists(evaluator_label_path)
    rollout_groups = _read_jsonl_if_exists(rollout_groups_path)
    preferences = _read_jsonl_if_exists(preferences_path)
    forbidden_hits = _forbidden_policy_sample_hits(policy_samples)
    rollout_group_sizes = [len(item.get("rollouts", [])) for item in rollout_groups if isinstance(item.get("rollouts"), list)]
    label_sample_ids = {str(label.get("sample_id")) for label in evaluator_labels if label.get("sample_id") is not None}
    sample_ids = {str(sample.get("sample_id")) for sample in policy_samples if sample.get("sample_id") is not None}
    missing_label_count = len(sample_ids - label_sample_ids)
    required_artifacts = {
        "training_manifest": manifest_path,
        "policy_samples_jsonl": policy_sample_path,
        "evaluator_labels_jsonl": evaluator_label_path,
        "rollout_groups_jsonl": rollout_groups_path,
        "trajectory_preferences_jsonl": preferences_path,
        "policy_review_traces_jsonl": review_traces_path,
    }
    return {
        "manifest_path": str(manifest_path),
        "experiment_id": manifest.get("experiment_id"),
        "research_run_id": manifest.get("research_run_id"),
        "episode_count": _int(manifest.get("episode_count")),
        "step_count": _int(manifest.get("step_count")),
        "policy_sample_count": len(policy_samples),
        "evaluator_label_count": len(evaluator_labels),
        "missing_evaluator_label_count": missing_label_count,
        "rollout_group_count": len(rollout_groups),
        "max_rollouts_per_group": max(rollout_group_sizes) if rollout_group_sizes else 0,
        "trajectory_preference_count": len(preferences),
        "policy_review_trace_count": _jsonl_count(review_traces_path),
        "grpo_eligible_sample_count": _int(policy_dataset_manifest.get("grpo_eligible_sample_count")) if policy_dataset_manifest else 0,
        "forbidden_policy_sample_token_hits": forbidden_hits,
        "missing_required_artifacts": [name for name, path in required_artifacts.items() if path is None or not path.exists()],
        "privileged_label_file": bool(policy_dataset_manifest.get("privileged_label_file")) if policy_dataset_manifest else False,
        "policy_dataset_manifest_path": str(_resolve_manifest_path(manifest, "policy_dataset_manifest_path") or ""),
    }


def _summarize_qwen_tool_training_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(manifest["_manifest_path"]))
    sft_path = _resolve_manifest_path(manifest, "qwen_sft_messages_jsonl")
    rejected_path = _resolve_manifest_path(manifest, "rejected_samples_jsonl")
    action_preferences_path = _resolve_manifest_path(manifest, "qwen_action_preferences_jsonl")
    dpo_path = _resolve_manifest_path(manifest, "qwen_dpo_messages_jsonl")
    audit_path = _resolve_manifest_path(manifest, "training_audit_json")
    sft_records = _read_jsonl_if_exists(sft_path)
    rejected_records = _read_jsonl_if_exists(rejected_path)
    action_preferences = _read_jsonl_if_exists(action_preferences_path)
    dpo_preferences = _read_jsonl_if_exists(dpo_path)
    required_artifacts = {
        "qwen_tool_training_manifest": manifest_path,
        "qwen_sft_messages_jsonl": sft_path,
        "qwen_rejected_samples_jsonl": rejected_path,
        "qwen_action_preferences_jsonl": action_preferences_path,
        "qwen_training_audit_json": audit_path,
    }
    if manifest.get("qwen_dpo_messages_jsonl"):
        required_artifacts["qwen_dpo_messages_jsonl"] = dpo_path
    missing_required = [name for name, path in required_artifacts.items() if path is None or not path.exists()]
    qwen_records = [*sft_records, *action_preferences, *dpo_preferences]
    return {
        "manifest_path": str(manifest_path),
        "output_dir": manifest.get("output_dir"),
        "source_policy_dataset_dir": manifest.get("source_policy_dataset_dir"),
        "qwen_sft_sample_count": len(sft_records),
        "qwen_rejected_sample_count": len(rejected_records),
        "qwen_action_preference_count": len(action_preferences),
        "qwen_dpo_preference_count": len(dpo_preferences),
        "qwen_missing_image_count": _qwen_missing_image_count(qwen_records),
        "forbidden_qwen_message_token_hits": _forbidden_qwen_message_hits(qwen_records),
        "missing_required_artifacts": missing_required,
        "manifest_accepted_count": _int(manifest.get("accepted_count")),
        "manifest_rejected_count": _int(manifest.get("rejected_count")),
        "manifest_action_preference_count": _int(manifest.get("action_preference_count")),
        "manifest_dpo_preference_count": _int(manifest.get("dpo_preference_count")),
        "training_audit_path": str(audit_path or ""),
        "training_audit": _load_optional_json(audit_path),
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_count": len(runs),
        "episode_count": sum(int(run["episode_count"]) for run in runs),
        "step_count": sum(int(run["step_count"]) for run in runs),
        "policy_sample_count": sum(int(run["policy_sample_count"]) for run in runs),
        "evaluator_label_count": sum(int(run["evaluator_label_count"]) for run in runs),
        "missing_evaluator_label_count": sum(int(run["missing_evaluator_label_count"]) for run in runs),
        "rollout_group_count": sum(int(run["rollout_group_count"]) for run in runs),
        "trajectory_preference_count": sum(int(run["trajectory_preference_count"]) for run in runs),
        "policy_review_trace_count": sum(int(run["policy_review_trace_count"]) for run in runs),
        "grpo_eligible_sample_count": sum(int(run["grpo_eligible_sample_count"]) for run in runs),
        "max_rollouts_per_group": max((int(run["max_rollouts_per_group"]) for run in runs), default=0),
        "forbidden_policy_sample_token_hits": sorted(
            {
                token
                for run in runs
                for token in run.get("forbidden_policy_sample_token_hits", [])
                if isinstance(token, str)
            }
        ),
        "missing_required_artifacts": sorted(
            {
                artifact
                for run in runs
                for artifact in run.get("missing_required_artifacts", [])
                if isinstance(artifact, str)
            }
        ),
        "all_policy_dataset_labels_privileged": all(bool(run.get("privileged_label_file")) for run in runs),
    }


def _aggregate_with_qwen_tool_training(aggregate: dict[str, Any], qwen_runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(aggregate)
    result.update(
        {
            "qwen_tool_training_manifest_count": len(qwen_runs),
            "qwen_sft_sample_count": sum(int(run["qwen_sft_sample_count"]) for run in qwen_runs),
            "qwen_rejected_sample_count": sum(int(run["qwen_rejected_sample_count"]) for run in qwen_runs),
            "qwen_action_preference_count": sum(int(run["qwen_action_preference_count"]) for run in qwen_runs),
            "qwen_dpo_preference_count": sum(int(run["qwen_dpo_preference_count"]) for run in qwen_runs),
            "qwen_missing_image_count": sum(int(run["qwen_missing_image_count"]) for run in qwen_runs),
            "forbidden_qwen_message_token_hits": sorted(
                {
                    token
                    for run in qwen_runs
                    for token in run.get("forbidden_qwen_message_token_hits", [])
                    if isinstance(token, str)
                }
            ),
        }
    )
    qwen_missing_artifacts = sorted(
        {
            artifact
            for run in qwen_runs
            for artifact in run.get("missing_required_artifacts", [])
            if isinstance(artifact, str)
        }
    )
    result["missing_required_artifacts"] = sorted(
        {
            *result.get("missing_required_artifacts", []),
            *qwen_missing_artifacts,
        }
    )
    return result


def _readiness_decision(aggregate: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    if aggregate["missing_required_artifacts"]:
        blockers.append("missing required training artifacts: " + ", ".join(aggregate["missing_required_artifacts"]))
    if aggregate["forbidden_policy_sample_token_hits"]:
        blockers.append("policy samples contain forbidden privileged token(s): " + ", ".join(aggregate["forbidden_policy_sample_token_hits"]))
    if aggregate["forbidden_qwen_message_token_hits"]:
        blockers.append("Qwen training messages contain forbidden privileged token(s): " + ", ".join(aggregate["forbidden_qwen_message_token_hits"]))
    if aggregate["qwen_missing_image_count"] > 0:
        blockers.append(f"{aggregate['qwen_missing_image_count']} Qwen training image reference(s) are missing")
    if aggregate["missing_evaluator_label_count"] > 0:
        blockers.append(f"{aggregate['missing_evaluator_label_count']} policy sample(s) lack evaluator labels")
    if not aggregate["all_policy_dataset_labels_privileged"]:
        warnings.append("one or more policy dataset manifests did not mark evaluator labels as privileged")

    qwen_materialized = aggregate["qwen_tool_training_manifest_count"] > 0
    sft_basis_count = aggregate["qwen_sft_sample_count"] if qwen_materialized else aggregate["policy_sample_count"]
    sft_ready = sft_basis_count > 0 and not blockers
    preference_basis_count = aggregate["qwen_dpo_preference_count"] or aggregate["qwen_action_preference_count"]
    preference_tuning_ready = preference_basis_count > 0 and not blockers
    ppo_ready = aggregate["evaluator_label_count"] > 0 and aggregate["step_count"] > 0 and not blockers
    grpo_ready = (
        (aggregate["trajectory_preference_count"] > 0 or aggregate["max_rollouts_per_group"] >= 2 or aggregate["grpo_eligible_sample_count"] > 0)
        and not blockers
    )
    if qwen_materialized and aggregate["qwen_sft_sample_count"] == 0 and not blockers:
        warnings.append("Qwen SFT is not ready: materialized Qwen tool-training output has no accepted SFT samples.")
    if not qwen_materialized and not blockers:
        warnings.append("Qwen tool-training materializer output was not found; SFT readiness is based on raw policy samples only.")
    if aggregate["qwen_action_preference_count"] == 0 and not blockers:
        warnings.append("Preference tuning is not yet ready: no Qwen guard-replacement action preferences were found.")
    if aggregate["qwen_action_preference_count"] > 0 and aggregate["qwen_dpo_preference_count"] == 0 and not blockers:
        warnings.append("Qwen preference tuning is using action-preference records only; rerun the materializer to emit qwen_dpo_messages.jsonl.")
    if not grpo_ready and not blockers:
        warnings.append("GRPO is not yet ready: need at least two comparable rollouts, preference pairs, or eligible grouped samples.")
    return {
        "status": "ready" if any((sft_ready, preference_tuning_ready, ppo_ready, grpo_ready)) and not blockers else "not_ready",
        "sft_ready": sft_ready,
        "preference_tuning_ready": preference_tuning_ready,
        "ppo_ready": ppo_ready,
        "grpo_ready": grpo_ready,
        "blockers": blockers,
        "warnings": warnings,
    }


def _resolve_manifest_path(manifest: dict[str, Any], key: str) -> Path | None:
    value = manifest.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path
    manifest_path = Path(str(manifest["_manifest_path"]))
    if path.is_absolute():
        relocated = _relocated_absolute_path(path, local_training_export_dir=manifest_path.parent)
        if relocated is not None:
            return relocated
        return path
    if path.exists():
        return path
    candidates = [
        manifest_path.parent / path,
        Path(str(manifest.get("output_dir") or manifest_path.parent)) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _relocated_absolute_path(path: Path, *, local_training_export_dir: Path) -> Path | None:
    parts = path.parts
    for marker in ("training_export", "qwen_tool_training"):
        if marker not in parts:
            continue
        index = len(parts) - 1 - list(reversed(parts)).index(marker)
        tail = parts[index + 1 :]
        return local_training_export_dir / Path(*tail) if tail else local_training_export_dir
    return None


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_if_exists(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _jsonl_count(path: Path | None) -> int:
    return len(_read_jsonl_if_exists(path))


def _forbidden_policy_sample_hits(samples: list[dict[str, Any]]) -> list[str]:
    text = json.dumps(samples, sort_keys=True, default=str).lower()
    return sorted({token for token in FORBIDDEN_POLICY_TOKENS if token.lower() in text})


def _qwen_missing_image_count(records: list[dict[str, Any]]) -> int:
    missing = 0
    for record in records:
        missing_paths: set[str] = set()
        audit = record.get("audit") if isinstance(record.get("audit"), dict) else {}
        audit_missing = audit.get("missing_images")
        if isinstance(audit_missing, list):
            missing_paths.update(str(path) for path in audit_missing)
        image_paths = record.get("image_paths")
        if isinstance(image_paths, list):
            missing_paths.update(str(path) for path in image_paths if not Path(str(path)).exists())
        missing += len(missing_paths)
    return missing


def _forbidden_qwen_message_hits(records: list[dict[str, Any]]) -> list[str]:
    payloads: list[Any] = []
    for record in records:
        for key in ("messages", "prompt_messages", "assistant_target_json", "chosen_assistant_target_json", "rejected_assistant_target_json"):
            value = record.get(key)
            if value:
                payloads.append(value)
    text = json.dumps(payloads, sort_keys=True, default=str).lower()
    return sorted({token for token in DEFAULT_FORBIDDEN_MODEL_TOKENS if token.lower() in text})


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _default_analysis_id(manifests: list[dict[str, Any]]) -> str:
    experiment = str(manifests[0].get("experiment_id") or "training")
    run = str(manifests[0].get("research_run_id") or "manual")
    return f"{experiment}-{run}-training-readiness"


def _default_about(experiment_id: str | None) -> str | None:
    return f"NavExperiment/{_safe_id(experiment_id)}" if experiment_id else None


def _warmhub_ops(report: dict[str, Any], *, about: str | None, author: str) -> list[dict[str, Any]]:
    readiness = report["readiness"]
    aggregate = report["aggregate"]
    note = (
        f"Training readiness {readiness['status']}: "
        f"SFT={readiness['sft_ready']}, preference_tuning={readiness['preference_tuning_ready']}, "
        f"PPO={readiness['ppo_ready']}, GRPO={readiness['grpo_ready']}. "
        f"Samples={aggregate['policy_sample_count']}, labels={aggregate['evaluator_label_count']}, "
        f"trajectory_preferences={aggregate['trajectory_preference_count']}, "
        f"qwen_action_preferences={aggregate['qwen_action_preference_count']}, "
        f"qwen_dpo_preferences={aggregate['qwen_dpo_preference_count']}. Report: {report['report_path']}"
    )
    return [
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"TrainingReadiness/{_safe_id(report['analysis_id'])}",
            "about": about,
            "data": {
                "analysisId": report["analysis_id"],
                "status": readiness["status"],
                "createdAt": report["created_at"],
                "author": author,
                "sftReady": bool(readiness["sft_ready"]),
                "preferenceTuningReady": bool(readiness["preference_tuning_ready"]),
                "ppoReady": bool(readiness["ppo_ready"]),
                "grpoReady": bool(readiness["grpo_ready"]),
                "runCount": int(aggregate["run_count"]),
                "episodeCount": int(aggregate["episode_count"]),
                "stepCount": int(aggregate["step_count"]),
                "policySampleCount": int(aggregate["policy_sample_count"]),
                "evaluatorLabelCount": int(aggregate["evaluator_label_count"]),
                "rolloutGroupCount": int(aggregate["rollout_group_count"]),
                "trajectoryPreferenceCount": int(aggregate["trajectory_preference_count"]),
                "qwenToolTrainingManifestCount": int(aggregate["qwen_tool_training_manifest_count"]),
                "qwenSftSampleCount": int(aggregate["qwen_sft_sample_count"]),
                "qwenRejectedSampleCount": int(aggregate["qwen_rejected_sample_count"]),
                "qwenActionPreferenceCount": int(aggregate["qwen_action_preference_count"]),
                "qwenDpoPreferenceCount": int(aggregate["qwen_dpo_preference_count"]),
                "qwenMissingImageCount": int(aggregate["qwen_missing_image_count"]),
                "grpoEligibleSampleCount": int(aggregate["grpo_eligible_sample_count"]),
                "missingRequiredArtifacts": aggregate["missing_required_artifacts"],
                "forbiddenPolicySampleTokenHits": aggregate["forbidden_policy_sample_token_hits"],
                "forbiddenQwenMessageTokenHits": aggregate["forbidden_qwen_message_token_hits"],
                "blockers": readiness["blockers"],
                "warnings": readiness["warnings"],
                "reportPath": report["report_path"],
                "markdownPath": report["markdown_path"],
                "confidence": 0.85,
            },
        },
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"AgentNote/{_safe_id(report['analysis_id'])}",
            "about": about,
            "data": {
                "author": author,
                "createdAt": report["created_at"],
                "note": note,
                "tags": ["open-vocab-nav", "training-readiness", readiness["status"]],
                "confidence": 0.85,
            },
        }
    ]


def _markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    aggregate = report["aggregate"]
    lines = [
        "# Navigation Training Readiness",
        "",
        f"Status: **{readiness['status']}**",
        f"Analysis ID: `{report['analysis_id']}`",
        "",
        "| Channel | Ready | Evidence |",
        "|---|---:|---|",
        f"| SFT | `{readiness['sft_ready']}` | {_sft_evidence(aggregate)} |",
        f"| Preference tuning | `{readiness['preference_tuning_ready']}` | {aggregate['qwen_dpo_preference_count']} DPO handoff records; {aggregate['qwen_action_preference_count']} Qwen guard-replacement action preferences |",
        f"| PPO | `{readiness['ppo_ready']}` | {aggregate['evaluator_label_count']} evaluator labels over {aggregate['step_count']} steps |",
        f"| GRPO | `{readiness['grpo_ready']}` | {aggregate['rollout_group_count']} rollout groups, {aggregate['trajectory_preference_count']} preferences |",
        "",
        "Blockers:",
    ]
    lines.extend(f"- {item}" for item in readiness["blockers"] or ["none"])
    lines.append("")
    lines.append("Warnings:")
    lines.extend(f"- {item}" for item in readiness["warnings"] or ["none"])
    lines.append("")
    lines.append("Policy constraints:")
    lines.extend(f"- {item}" for item in report["policy_constraints"])
    lines.append("")
    return "\n".join(lines)


def _sft_evidence(aggregate: dict[str, Any]) -> str:
    if aggregate.get("qwen_tool_training_manifest_count"):
        return f"{aggregate['qwen_sft_sample_count']} Qwen SFT samples; {aggregate['policy_sample_count']} raw policy samples"
    return f"{aggregate['policy_sample_count']} raw policy samples; no Qwen tool-training manifest found"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Research output dir, training_export dir, or training_manifest.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--about", default=None)
    parser.add_argument("--author", default="flatdisk-sim-training-readiness")
    parser.add_argument("--repo", default=DEFAULT_WARMHUB_REPO)
    parser.add_argument("--commit-warmhub", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_training_readiness(
        args.input,
        output_dir=args.output_dir,
        analysis_id=args.analysis_id,
        experiment_id=args.experiment_id,
        about=args.about,
        author=args.author,
    )
    if args.commit_warmhub:
        commit_ops(args.repo, report["warmhub_ops"], message="Log navigation training readiness")
    print(
        json.dumps(
            {
                "analysis_id": report["analysis_id"],
                "status": report["readiness"]["status"],
                "sft_ready": report["readiness"]["sft_ready"],
                "preference_tuning_ready": report["readiness"]["preference_tuning_ready"],
                "ppo_ready": report["readiness"]["ppo_ready"],
                "grpo_ready": report["readiness"]["grpo_ready"],
                "qwen_sft_sample_count": report["aggregate"]["qwen_sft_sample_count"],
                "qwen_action_preference_count": report["aggregate"]["qwen_action_preference_count"],
                "qwen_dpo_preference_count": report["aggregate"]["qwen_dpo_preference_count"],
                "report_path": report["report_path"],
                "markdown_path": report["markdown_path"],
                "warmhub_ops_path": report["warmhub_ops_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_not_ready and report["readiness"]["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
