"""Parallel research loop for open-vocabulary indoor navigation experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

from .evaluate_harness_thor import run_thor_harness_episode
from .evaluate_text_goals import ThorEpisodeSpec, _find_free_port, default_episodes
from .llm_harness import POLICY_INPUT_ALLOWLIST
from .paths import SCRATCH_ROOT
from .training_export import export_training_data_from_summaries


ROBOT_IO_CONTRACT = "real_robot_camera_imu_bounded_tools"
PRIVILEGED_EVAL_PURPOSE = "scoring/debug only; never policy input"
DEFAULT_WARMHUB_REPO = "bencaunt-2/open-vocab-nav-research-loop"
STRICT_MODEL_BASED_RUNNERS = {"codex", "qwen"}


@dataclass(frozen=True)
class PromptVariant:
    name: str
    description: str = ""
    runner: str = "qwen"
    model: str = "gpt-5.5"
    reasoning_effort: str = "low"
    prompt_profile: str = "baseline"
    actor_rules: tuple[str, ...] = ()
    critic_rules: tuple[str, ...] = ()
    critic_mode: str = "auto"
    qwen_endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
    qwen_model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    qwen_temperature: float = 0.0
    qwen_max_tokens: int = 512
    object_drive_detector: str = "florence-mlx"
    topomap_memory_map_dir: str | None = None
    topomap_memory_use_clip: bool = False
    topomap_memory_allow_semantic_terms: bool = False


@dataclass(frozen=True)
class ResearchConfig:
    experiment_id: str
    objective: str
    episodes: tuple[str, ...]
    variants: tuple[PromptVariant, ...]
    repetitions: int = 1
    parallelism: int = 1
    max_steps: int | None = None
    success_radius_m: float | None = None
    render_width: int = 320
    render_height: int = 240
    rerun: bool = False
    warmhub_repo: str = DEFAULT_WARMHUB_REPO
    strict_model_based: bool = True


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    slot_id: str
    episode_name: str
    repetition_index: int
    variant: PromptVariant


def load_config(path: Path) -> ResearchConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    variants = tuple(_parse_variant(item) for item in raw.get("variants", []))
    if not variants:
        variants = default_prompt_variants()
    episodes = tuple(str(name) for name in raw.get("episodes", default_episodes().keys()))
    return ResearchConfig(
        experiment_id=_safe_id(str(raw.get("experiment_id") or path.stem)),
        objective=str(raw.get("objective") or "Open-vocabulary indoor navigation research loop"),
        episodes=episodes,
        variants=variants,
        repetitions=max(1, int(raw.get("repetitions", 1))),
        parallelism=max(1, int(raw.get("parallelism", 1))),
        max_steps=_optional_int(raw.get("max_steps")),
        success_radius_m=_optional_float(raw.get("success_radius_m")),
        render_width=max(64, int(raw.get("render_width", 320))),
        render_height=max(64, int(raw.get("render_height", 240))),
        rerun=bool(raw.get("rerun", False)),
        warmhub_repo=str(raw.get("warmhub_repo") or DEFAULT_WARMHUB_REPO),
        strict_model_based=bool(raw.get("strict_model_based", True)),
    )


def default_prompt_variants() -> tuple[PromptVariant, ...]:
    return (
        PromptVariant(
            name="qwen_baseline",
            description="Current Qwen tool-use prompt with Florence visual servo available.",
        ),
        PromptVariant(
            name="qwen_explore_memory",
            description="Bias Qwen toward active exploration, waypoint use, and explicit scratchpad state.",
            prompt_profile="explore-memory-v1",
            actor_rules=(
                "Maintain a compact belief state in memory_update: last clear goal evidence, explored headings, failed strategies, and next exploration target.",
                "If the goal object is not visible, prefer gaining viewpoint diversity over repeating the same final-goal visual servo prompt.",
                "Use visual_servo_object on a visible landmark only when it plausibly moves the robot into a more informative area, then reassess from the new image.",
                "Balance exploration and exploitation: exploit clear target evidence, otherwise rotate or move briefly to reveal unseen space.",
            ),
            critic_rules=(
                "Warn when the actor repeats a strategy without explaining new visual evidence or memory-based reason.",
                "Approve exploratory landmark moves when they are bounded and based on visible scene structure.",
            ),
        ),
    )


def build_trial_matrix(config: ResearchConfig, *, run_id: str | None = None) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    for repetition in range(config.repetitions):
        for variant in config.variants:
            for episode_name in config.episodes:
                slot_id = _safe_id(f"{config.experiment_id}_{variant.name}_{episode_name}_r{repetition + 1}")
                trial_id = _safe_id(f"{run_id}_{variant.name}_{episode_name}_r{repetition + 1}") if run_id else slot_id
                trials.append(
                    TrialSpec(
                        trial_id=trial_id,
                        slot_id=slot_id,
                        episode_name=episode_name,
                        repetition_index=repetition,
                        variant=variant,
                    )
                )
    return trials


def _variant_no_hardcoded_labels_or_colors(variant: PromptVariant) -> bool:
    return variant.runner in STRICT_MODEL_BASED_RUNNERS and not variant.topomap_memory_allow_semantic_terms


def _config_no_hardcoded_labels_or_colors(config: ResearchConfig) -> bool:
    return all(_variant_no_hardcoded_labels_or_colors(variant) for variant in config.variants)


def validate_research_config(config: ResearchConfig) -> None:
    if not config.strict_model_based:
        return
    errors: list[str] = []
    for variant in config.variants:
        if variant.runner not in STRICT_MODEL_BASED_RUNNERS:
            errors.append(
                f"variant {variant.name!r} uses runner {variant.runner!r}; strict research runs allow only {sorted(STRICT_MODEL_BASED_RUNNERS)}"
            )
        if variant.topomap_memory_allow_semantic_terms:
            errors.append(
                f"variant {variant.name!r} enables topomap_memory_allow_semantic_terms; strict runs must use model embeddings/contact sheets"
            )
    if errors:
        raise ValueError("strict_model_based research config rejected non-general policy path: " + "; ".join(errors))


def _summary_no_hardcoded_labels_or_colors(summary: dict[str, Any], *, fallback_variant: PromptVariant | None = None) -> bool:
    prompt_audit = summary.get("prompt_audit") if isinstance(summary.get("prompt_audit"), dict) else {}
    prompt_ok = bool(prompt_audit.get("no_hardcoded_labels_or_colors", True))
    allow_semantic = summary.get("topomap_memory_allow_semantic_terms")
    if allow_semantic is None and fallback_variant is not None:
        allow_semantic = fallback_variant.topomap_memory_allow_semantic_terms
    runner = str(summary.get("runner") or (fallback_variant.runner if fallback_variant is not None else ""))
    return runner in STRICT_MODEL_BASED_RUNNERS and prompt_ok and not bool(allow_semantic)


def _aggregate_no_hardcoded_labels_or_colors(manifest: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    if records:
        return all(bool(record.get("no_hardcoded_labels_or_colors")) for record in records)
    return bool(manifest.get("no_hardcoded_labels_or_colors"))


def run_research_loop(
    config: ResearchConfig,
    *,
    config_path: Path | None,
    output_root: Path,
    dry_run: bool,
    commit_warmhub: bool,
    init_warmhub_repo: bool,
    preflight_endpoints: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    validate_research_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_id = _safe_id(f"{config.experiment_id}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
    git_commit = _git_commit()
    trials = build_trial_matrix(config, run_id=run_id)
    manifest = {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "objective": config.objective,
        "config_path": str(config_path) if config_path is not None else None,
        "started_at": started_at,
        "git_commit": git_commit,
        "dry_run": dry_run,
        "preflight_endpoints": preflight_endpoints,
        "preflight_only": preflight_only,
        "robot_io_contract": ROBOT_IO_CONTRACT,
        "policy_input_allowlist": POLICY_INPUT_ALLOWLIST,
        "privileged_eval_used": True,
        "privileged_eval_purpose": PRIVILEGED_EVAL_PURPOSE,
        "strict_model_based": config.strict_model_based,
        "no_hardcoded_labels_or_colors": _config_no_hardcoded_labels_or_colors(config),
        "warmhub_repo": config.warmhub_repo,
        "trial_count": len(trials),
        "trials": [_trial_manifest(trial) for trial in trials],
        "config": _config_to_dict(config),
    }
    (output_root / "research_loop_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not dry_run:
        runnable_trials = trials
        if preflight_endpoints or preflight_only:
            runnable_trials, failures = _preflight_trials(config, trials, output_root=output_root)
        if not preflight_only:
            summaries, execution_failures = _execute_trials(config, runnable_trials, output_root=output_root, git_commit=git_commit)
            failures.extend(execution_failures)

    aggregate = _aggregate(config, manifest, summaries, failures, output_root=output_root)
    if summaries:
        aggregate["training_export"] = export_training_data_from_summaries(
            summaries,
            output_dir=output_root / "training_export",
            experiment_id=config.experiment_id,
            run_id=run_id,
        )
        (output_root / "research_loop_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_research_report(aggregate, output_root)
    write_warmhub_bundle(config, aggregate, output_root)
    if commit_warmhub:
        commit_warmhub_bundle(output_root, repo=config.warmhub_repo, init_repo=init_warmhub_repo)
    return aggregate


def write_research_report(aggregate: dict[str, Any], output_root: Path) -> Path:
    path = output_root / "research_loop_report.md"
    lines = [
        "# Open-Vocabulary Navigation Research Loop",
        "",
        f"Experiment: `{aggregate['experiment_id']}`",
        f"Dry run: `{aggregate['dry_run']}`",
        f"Robot I/O contract: `{ROBOT_IO_CONTRACT}`",
        "Policy input allowlist: " + ", ".join(POLICY_INPUT_ALLOWLIST),
        "Privileged evaluator data is used only for scoring/debugging.",
        "",
        "| Trial | Variant | Episode | Runner | Success | Reason | Steps | Best distance (m) | Final distance (m) | Final regression (m) | Output |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate["summaries"]:
        lines.append(
            f"| {row['trial_id']} | {row['variant']} | {row['episode']} | {row['runner']} | "
            f"{row['success']} | {row['reason']} | {row['step_count']} | "
            f"{_format_float(row.get('best_distance_m'))} | "
            f"{_format_float(row.get('final_distance_m'))} | "
            f"{_format_float(row.get('final_to_best_regression_m'))} | "
            f"{row.get('run_dir') or ''} |"
        )
    if not aggregate["summaries"]:
        lines.append("| dry-run | - | - | - | - | no simulator execution | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "Warmhub bundle:",
            f"- Shapes: `{output_root / 'warmhub_shapes.json'}`",
            f"- Commit ops: `{output_root / 'warmhub_ops.json'}`",
            f"- Agent startup: `{output_root / 'AGENTS.warmhub.md'}`",
        ]
    )
    training_export = aggregate.get("training_export")
    if isinstance(training_export, dict):
        lines.extend(
            [
                "",
                "Training export:",
                f"- Manifest: `{training_export.get('manifest_path')}`",
                f"- Policy steps: `{training_export.get('policy_steps_jsonl')}`",
                f"- Episode rollouts: `{training_export.get('episode_rollouts_jsonl')}`",
                f"- Rollout groups: `{training_export.get('rollout_groups_jsonl')}`",
                f"- Trajectory preferences: `{training_export.get('trajectory_preferences_jsonl')}`",
                f"- Policy review traces: `{training_export.get('policy_review_traces_jsonl')}`",
                f"- Policy samples: `{training_export.get('policy_samples_jsonl')}`",
                f"- Evaluator labels: `{training_export.get('evaluator_labels_jsonl')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate["report"] = str(path)
    return path


def write_warmhub_bundle(config: ResearchConfig, aggregate: dict[str, Any], output_root: Path) -> None:
    shapes = warmhub_shapes()
    (output_root / "warmhub_shapes.json").write_text(json.dumps(shapes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "warmhub_schema_commands.sh").write_text(_warmhub_schema_commands(config.warmhub_repo, shapes), encoding="utf-8")
    ops = warmhub_ops(config, aggregate)
    (output_root / "warmhub_ops.json").write_text(json.dumps(ops, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "AGENTS.warmhub.md").write_text(_warmhub_agents_md(config.warmhub_repo), encoding="utf-8")


def warmhub_shapes() -> dict[str, dict[str, Any]]:
    return {
        "NavExperiment": {
            "description": "Research objective or sweep for open-vocabulary indoor navigation.",
            "fields": {
                "objective": "string",
                "createdAt": "string",
                "gitCommit?": "string",
                "robotIoContract": "string",
                "policyInputAllowlist": ["string"],
                "strictModelBased": "boolean",
                "noHardcodedLabelsOrColors": "boolean",
            },
        },
        "PromptVariant": {
            "description": "Prompt/tool policy variant evaluated by the research loop.",
            "fields": {
                "description?": "string",
                "runner": "string",
                "model": "string",
                "promptProfile": "string",
                "actorRules": ["string"],
                "criticRules": ["string"],
                "criticMode": "string",
                "qwenModel?": "string",
                "objectDriveDetector": "string",
                "topomapMemoryMapDir?": "string",
                "topomapMemoryUseClip": "boolean",
                "topomapMemoryAllowSemanticTerms": "boolean",
                "noHardcodedLabelsOrColors": "boolean",
            },
        },
        "NavEpisodeSpec": {
            "description": "Simulator or robot navigation episode definition.",
            "fields": {
                "scene": "string",
                "prompt": "string",
                "targetTypes": ["string"],
                "successRadiusM": "number",
                "maxSteps": "number",
                "policyInputAllowlist": ["string"],
                "privilegedEvalUsed": "boolean",
                "privilegedEvalPurpose": "string",
            },
        },
        "NavEvalRun": {
            "description": "One evaluated navigation run.",
            "fields": {
                "experiment": {"type": "wref", "shape": "NavExperiment"},
                "variant": {"type": "wref", "shape": "PromptVariant"},
                "episodeSpec": {"type": "wref", "shape": "NavEpisodeSpec"},
                "trialId": "string",
                "runner": "string",
                "model": "string",
                "criticMode": "string",
                "gitCommit?": "string",
                "outputDir": "string",
                "success": "boolean",
                "finalDistanceM?": "number",
                "bestDistanceM?": "number",
                "bestDistanceStep?": "number",
                "bestDistanceImprovementM?": "number",
                "finalDistanceImprovementM?": "number",
                "finalToBestRegressionM?": "number",
                "reachedSuccessRadiusEver": "boolean",
                "reason": "string",
                "stepCount": "number",
                "wallClockS": "number",
                "policyInputAllowlist": ["string"],
                "privilegedEvalUsed": "boolean",
                "privilegedEvalPurpose": "string",
                "robotIoContract": "string",
                "noHardcodedLabelsOrColors": "boolean",
            },
        },
        "NavArtifact": {
            "description": "Artifact produced by a navigation run.",
            "fields": {
                "run": {"type": "wref", "shape": "NavEvalRun"},
                "artifactType": "string",
                "path": "string",
                "sha256?": "string",
                "privileged": "boolean",
                "description?": "string",
            },
        },
        "RunAssessment": {
            "description": "Assertion summarizing run metrics and evaluator confidence.",
            "fields": {
                "success": "boolean",
                "finalDistanceM?": "number",
                "bestDistanceM?": "number",
                "bestDistanceStep?": "number",
                "bestDistanceImprovementM?": "number",
                "finalDistanceImprovementM?": "number",
                "finalToBestRegressionM?": "number",
                "reachedSuccessRadiusEver": "boolean",
                "reason": "string",
                "stepCount": "number",
                "evaluator": "string",
                "confidence": "number",
                "privilegedEvalUsed": "boolean",
                "notes?": "string",
            },
        },
        "PromotionDecision": {
            "description": "Assertion recording whether a candidate navigation variant should be promoted over a baseline.",
            "fields": {
                "decisionId": "string",
                "status": {"type": "string", "enum": ["promote", "reject"]},
                "promote": "boolean",
                "createdAt": "string",
                "author": "string",
                "baselineVariants": ["string"],
                "candidateVariants": ["string"],
                "baselineTrialCount": "number",
                "candidateTrialCount": "number",
                "successRateDelta": "number",
                "meanBestDistanceImprovementM?": "number",
                "meanFinalDistanceRegressionM?": "number",
                "blockers": ["string"],
                "warnings": ["string"],
                "reasons": ["string"],
                "reportPath": "string",
                "markdownPath": "string",
                "confidence": "number",
            },
        },
        "TrainingReadiness": {
            "description": "Assertion recording whether navigation training exports are ready for SFT, PPO, or GRPO use.",
            "fields": {
                "analysisId": "string",
                "status": {"type": "string", "enum": ["ready", "not_ready"]},
                "createdAt": "string",
                "author": "string",
                "sftReady": "boolean",
                "ppoReady": "boolean",
                "grpoReady": "boolean",
                "runCount": "number",
                "episodeCount": "number",
                "stepCount": "number",
                "policySampleCount": "number",
                "evaluatorLabelCount": "number",
                "rolloutGroupCount": "number",
                "trajectoryPreferenceCount": "number",
                "grpoEligibleSampleCount": "number",
                "missingRequiredArtifacts": ["string"],
                "forbiddenPolicySampleTokenHits": ["string"],
                "blockers": ["string"],
                "warnings": ["string"],
                "reportPath": "string",
                "markdownPath": "string",
                "confidence": "number",
            },
        },
        "FailureObservation": {
            "description": "Assertion recording a failure or suspicious behavior for agent reuse.",
            "fields": {
                "category": "string",
                "severity": "string",
                "symptom": "string",
                "suspectedCause?": "string",
                "evidenceArtifacts": ["string"],
                "nextAction?": "string",
                "confidence": "number",
            },
        },
        "AgentNote": {
            "description": "Durable scratchpad note for agents working on the research loop.",
            "fields": {
                "author": "string",
                "createdAt": "string",
                "note": "string",
                "tags": ["string"],
                "confidence": "number",
            },
        },
        "AgentTask": {
            "description": "Coordination task for a sub-agent or background worker.",
            "fields": {
                "objective": "string",
                "status": {"type": "string", "enum": ["planned", "running", "complete", "blocked", "failed"]},
                "owner": "string",
                "createdAt": "string",
                "updatedAt?": "string",
                "priority?": "string",
                "tags": ["string"],
                "relatedExperiment?": {"type": "wref", "shape": "NavExperiment"},
                "notes?": "string",
            },
        },
        "SubAgentResult": {
            "description": "Assertion recording the outcome of one delegated agent task.",
            "fields": {
                "agent": "string",
                "status": {"type": "string", "enum": ["complete", "blocked", "failed"]},
                "summary": "string",
                "changedFiles": ["string"],
                "evidenceArtifacts": ["string"],
                "nextActions": ["string"],
                "createdAt": "string",
                "confidence": "number",
            },
        },
    }


def warmhub_ops(config: ResearchConfig, aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    experiment_ref = f"NavExperiment/{config.experiment_id}"
    ops: list[dict[str, Any]] = [
        {
            "operation": "add",
            "kind": "thing",
            "name": experiment_ref,
            "data": {
                "objective": config.objective,
                "createdAt": aggregate["started_at"],
                "gitCommit": aggregate.get("git_commit"),
                "robotIoContract": ROBOT_IO_CONTRACT,
                "policyInputAllowlist": POLICY_INPUT_ALLOWLIST,
                "strictModelBased": config.strict_model_based,
                "noHardcodedLabelsOrColors": bool(aggregate.get("no_hardcoded_labels_or_colors")),
            },
        }
    ]
    for variant in config.variants:
        ops.append(
            {
                "operation": "add",
                "kind": "thing",
                "name": f"PromptVariant/{_variant_id(config, variant)}",
                "data": _prompt_variant_data(variant),
            }
        )
    episode_map = default_episodes()
    for name in config.episodes:
        spec = _configured_episode(episode_map[name], config)
        ops.append(
            {
                "operation": "add",
                "kind": "thing",
                "name": f"NavEpisodeSpec/{_episode_id(spec)}",
                "data": {
                    "scene": spec.scene,
                    "prompt": spec.prompt,
                    "targetTypes": list(spec.target_types),
                    "successRadiusM": spec.success_radius_m,
                    "maxSteps": spec.max_steps,
                    "policyInputAllowlist": POLICY_INPUT_ALLOWLIST,
                    "privilegedEvalUsed": True,
                    "privilegedEvalPurpose": PRIVILEGED_EVAL_PURPOSE,
                },
            }
        )
    for summary in aggregate["summaries"]:
        run_ref = f"NavEvalRun/{summary['trial_id']}"
        ops.append(_run_op(config, summary, experiment_ref, run_ref))
        ops.extend(_artifact_ops(summary, run_ref))
        ops.append(_assessment_op(summary, run_ref))
        if not summary.get("success"):
            ops.append(_failure_op(summary, run_ref))
    for failure in aggregate.get("failed_trials", []):
        run_ref = f"NavEvalRun/{failure['trial_id']}"
        ops.append(_run_op(config, failure, experiment_ref, run_ref))
        ops.extend(_artifact_ops(failure, run_ref))
        ops.append(_assessment_op(failure, run_ref))
        ops.append(_failure_op(failure, run_ref))
    ops.append(
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"AgentNote/{config.experiment_id}-research-loop-summary",
            "about": experiment_ref,
            "data": {
                "author": "flatdisk-sim-research-loop",
                "createdAt": aggregate["started_at"],
                "note": (
                    f"Research loop emitted {aggregate['trial_count']} planned trials, "
                    f"{aggregate['completed_trial_count']} completed, "
                    f"{aggregate['success_count']} successful."
                ),
                "tags": ["open-vocab-nav", "research-loop", "qwen", "florence"],
                "confidence": 0.8,
            },
        }
    )
    return [_drop_none(op) for op in ops]


def commit_warmhub_bundle(output_root: Path, *, repo: str, init_repo: bool) -> None:
    if init_repo:
        view = subprocess.run(["wh", "repo", "view", repo, "--json"], text=True, capture_output=True, check=False)
        if view.returncode != 0:
            subprocess.run(
                [
                    "wh",
                    "repo",
                    "create",
                    repo,
                    "--visibility",
                    "private",
                    "--description",
                    "Open-vocabulary indoor robot navigation research memory",
                ],
                text=True,
                check=True,
            )
    shapes = json.loads((output_root / "warmhub_shapes.json").read_text(encoding="utf-8"))
    for shape_name, spec in shapes.items():
        view = subprocess.run(["wh", "shape", "view", shape_name, "--repo", repo, "--json"], text=True, capture_output=True, check=False)
        fields_json = json.dumps(spec["fields"], sort_keys=True)
        description = spec["description"]
        if view.returncode == 0:
            current = _shape_view_data(view.stdout)
            if current.get("fields") == spec["fields"] and current.get("description") == description:
                continue
            subprocess.run(
                [
                    "wh",
                    "shape",
                    "revise",
                    shape_name,
                    "--repo",
                    repo,
                    "--fields",
                    fields_json,
                    "--description",
                    description,
                ],
                text=True,
                check=True,
            )
            continue
        subprocess.run(
            [
                "wh",
                "shape",
                "create",
                shape_name,
                "--repo",
                repo,
                "--fields",
                fields_json,
                "--description",
                description,
            ],
            text=True,
            check=True,
        )
    subprocess.run(
        [
            "wh",
            "commit",
            "submit",
            "--repo",
            repo,
            "--file",
            str(output_root / "warmhub_ops.json"),
            "--skip-existing",
            "--message",
            "Log open-vocabulary navigation research loop results",
        ],
        text=True,
        check=True,
    )


def _shape_view_data(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    version = payload.get("version")
    if isinstance(version, dict) and isinstance(version.get("data"), dict):
        return version["data"]
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "open_vocab_nav_research_loop")
    parser.add_argument("--dry-run", action="store_true", help="Write trial matrix and Warmhub artifacts without simulator execution.")
    parser.add_argument("--episode", action="append", default=[], help="Run only this episode name from the config. Repeatable.")
    parser.add_argument("--variant", action="append", default=[], help="Run only this variant name from the config. Repeatable.")
    parser.add_argument("--parallelism", type=int, default=None, help="Override config parallelism.")
    parser.add_argument("--warmhub-repo", default=None, help="Override the Warmhub repo configured in the research config.")
    parser.add_argument("--commit-warmhub", action="store_true", help="Submit generated shapes/results with the wh CLI.")
    parser.add_argument("--init-warmhub-repo", action="store_true", help="Create the Warmhub repo if it does not exist.")
    parser.add_argument("--preflight-endpoints", action="store_true", help="Check Qwen endpoints before launching THOR trials.")
    parser.add_argument("--preflight-only", action="store_true", help="Run endpoint/topomap preflight checks and stop before launching THOR trials.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.episode:
        unknown = sorted(set(args.episode) - set(config.episodes))
        if unknown:
            raise SystemExit(f"unknown --episode value(s): {', '.join(unknown)}")
        config = replace(config, episodes=tuple(args.episode))
    if args.variant:
        by_name = {variant.name: variant for variant in config.variants}
        unknown = sorted(set(args.variant) - set(by_name))
        if unknown:
            raise SystemExit(f"unknown --variant value(s): {', '.join(unknown)}")
        config = replace(config, variants=tuple(by_name[name] for name in args.variant))
    if args.parallelism is not None:
        config = replace(config, parallelism=max(1, args.parallelism))
    if args.warmhub_repo:
        config = replace(config, warmhub_repo=args.warmhub_repo)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / stamp
    aggregate = run_research_loop(
        config,
        config_path=args.config,
        output_root=output_root,
        dry_run=args.dry_run,
        commit_warmhub=args.commit_warmhub,
        init_warmhub_repo=args.init_warmhub_repo,
        preflight_endpoints=args.preflight_endpoints,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return 0 if aggregate["failure_count"] == 0 and aggregate["success_count"] == aggregate["completed_trial_count"] else 2


def _parse_variant(raw: dict[str, Any]) -> PromptVariant:
    return PromptVariant(
        name=_safe_id(str(raw["name"])),
        description=str(raw.get("description", "")),
        runner=str(raw.get("runner", "qwen")),
        model=str(raw.get("model", "gpt-5.5")),
        reasoning_effort=str(raw.get("reasoning_effort", "low")),
        prompt_profile=str(raw.get("prompt_profile") or raw.get("name") or "baseline"),
        actor_rules=tuple(str(rule) for rule in raw.get("actor_rules", [])),
        critic_rules=tuple(str(rule) for rule in raw.get("critic_rules", [])),
        critic_mode=str(raw.get("critic_mode", "auto")),
        qwen_endpoint=str(raw.get("qwen_endpoint", "http://127.0.0.1:8080/v1/chat/completions")),
        qwen_model=str(raw.get("qwen_model", "mlx-community/Qwen3-VL-8B-Instruct-4bit")),
        qwen_temperature=float(raw.get("qwen_temperature", 0.0)),
        qwen_max_tokens=int(raw.get("qwen_max_tokens", 512)),
        object_drive_detector=str(raw.get("object_drive_detector", "florence-mlx")),
        topomap_memory_map_dir=str(raw["topomap_memory_map_dir"]) if raw.get("topomap_memory_map_dir") else None,
        topomap_memory_use_clip=bool(raw.get("topomap_memory_use_clip", False)),
        topomap_memory_allow_semantic_terms=bool(raw.get("topomap_memory_allow_semantic_terms", False)),
    )


def _prompt_variant_data(variant: PromptVariant) -> dict[str, Any]:
    data: dict[str, Any] = {
        "description": variant.description,
        "runner": variant.runner,
        "model": variant.model,
        "promptProfile": variant.prompt_profile,
        "actorRules": list(variant.actor_rules),
        "criticRules": list(variant.critic_rules),
        "criticMode": variant.critic_mode,
        "qwenModel": variant.qwen_model,
        "objectDriveDetector": variant.object_drive_detector,
        "noHardcodedLabelsOrColors": _variant_no_hardcoded_labels_or_colors(variant),
        "topomapMemoryUseClip": variant.topomap_memory_use_clip,
        "topomapMemoryAllowSemanticTerms": variant.topomap_memory_allow_semantic_terms,
    }
    if variant.topomap_memory_map_dir or variant.topomap_memory_use_clip or variant.topomap_memory_allow_semantic_terms:
        data.update(
            {
                "topomapMemoryMapDir": variant.topomap_memory_map_dir,
            }
        )
    return data


def _execute_trials(
    config: ResearchConfig,
    trials: list[TrialSpec],
    *,
    output_root: Path,
    git_commit: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del git_commit
    episode_map = default_episodes()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.parallelism) as executor:
        futures = {
            executor.submit(_run_one_trial, config, trial, episode_map, output_root): trial
            for trial in trials
        }
        for future in as_completed(futures):
            trial = futures[future]
            try:
                summaries.append(future.result())
            except Exception as exc:  # noqa: BLE001 - failed trials are research data.
                failures.append(_failed_trial_summary(config, trial, output_root=output_root, error=exc))
    summaries.sort(key=lambda item: item["trial_id"])
    failures.sort(key=lambda item: item["trial_id"])
    return summaries, failures


def _preflight_trials(
    config: ResearchConfig,
    trials: list[TrialSpec],
    *,
    output_root: Path,
) -> tuple[list[TrialSpec], list[dict[str, Any]]]:
    endpoint_errors: dict[tuple[str, str], str | None] = {}
    topomap_errors: dict[tuple[str, bool], str | None] = {}
    runnable: list[TrialSpec] = []
    failures: list[dict[str, Any]] = []
    for trial in trials:
        errors: list[str] = []
        if trial.variant.runner != "qwen":
            pass
        else:
            key = (trial.variant.qwen_endpoint, trial.variant.qwen_model)
            if key not in endpoint_errors:
                endpoint_errors[key] = _qwen_endpoint_error(trial.variant.qwen_endpoint)
            endpoint_error = endpoint_errors[key]
            if endpoint_error is not None:
                errors.append(f"qwen endpoint preflight failed for {trial.variant.qwen_endpoint}: {endpoint_error}")
        topomap_map_dir = _resolved_topomap_memory_map_dir(trial)
        if topomap_map_dir:
            key = (topomap_map_dir, trial.variant.topomap_memory_use_clip)
            if key not in topomap_errors:
                topomap_errors[key] = _topomap_memory_error(topomap_map_dir, use_clip=trial.variant.topomap_memory_use_clip)
            topomap_error = topomap_errors[key]
            if topomap_error is not None:
                errors.append(f"topomap memory preflight failed for {topomap_map_dir}: {topomap_error}")
        if not errors:
            runnable.append(trial)
            continue
        failures.append(
            _failed_trial_summary(
                config,
                trial,
                output_root=output_root,
                error=RuntimeError("; ".join(errors)),
            )
        )
    return runnable, failures


def _topomap_memory_error(map_dir: str | None, *, use_clip: bool) -> str | None:
    if not map_dir:
        return None
    try:
        from .semantic_topomap import SemanticTopomap

        topomap = SemanticTopomap.load(Path(map_dir))
    except Exception as exc:  # noqa: BLE001 - converted to structured preflight failure.
        return str(exc)
    if use_clip and topomap.clip_image_embeddings is None:
        return "topomap_memory_use_clip is true but the map has no clip_image_embeddings"
    return None


def _qwen_endpoint_error(endpoint: str, *, timeout_s: float = 3.0) -> str | None:
    models_url = _models_url_for_chat_endpoint(endpoint)
    request = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - local/user-configured endpoint.
            if response.status >= 400:
                return f"HTTP {response.status} from {models_url}"
            response.read(1024)
            return None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"HTTP {exc.code} from {models_url}: {body[:240]}"
    except urllib.error.URLError as exc:
        return str(exc.reason)
    except TimeoutError:
        return f"timeout after {timeout_s:.1f}s"


def _models_url_for_chat_endpoint(endpoint: str) -> str:
    text = endpoint.rstrip("/")
    suffix = "/chat/completions"
    if text.endswith(suffix):
        return text[: -len(suffix)] + "/models"
    if text.endswith("/v1"):
        return text + "/models"
    return text + "/models"


def _run_one_trial(
    config: ResearchConfig,
    trial: TrialSpec,
    episode_map: dict[str, ThorEpisodeSpec],
    output_root: Path,
) -> dict[str, Any]:
    spec = _configured_episode(episode_map[trial.episode_name], config)
    trial_root = output_root / "trials" / trial.trial_id
    topomap_map_dir = _resolved_topomap_memory_map_dir(trial)
    summary = run_thor_harness_episode(
        spec,
        output_root=trial_root,
        port=_find_free_port(),
        model=trial.variant.model,
        reasoning_effort=trial.variant.reasoning_effort,
        live_codex=False,
        runner=trial.variant.runner,
        qwen_endpoint=trial.variant.qwen_endpoint,
        qwen_model=trial.variant.qwen_model,
        qwen_temperature=trial.variant.qwen_temperature,
        qwen_max_tokens=trial.variant.qwen_max_tokens,
        object_drive_detector=trial.variant.object_drive_detector,
        topomap_memory_map_dir=Path(topomap_map_dir) if topomap_map_dir else None,
        topomap_memory_use_clip=trial.variant.topomap_memory_use_clip,
        topomap_memory_allow_semantic_terms=trial.variant.topomap_memory_allow_semantic_terms,
        prompt_profile=trial.variant.prompt_profile,
        actor_rules=trial.variant.actor_rules,
        critic_rules=trial.variant.critic_rules,
        critic_mode=trial.variant.critic_mode,
        render_width=config.render_width,
        render_height=config.render_height,
        rerun=config.rerun,
        max_steps=config.max_steps,
        success_radius_m=config.success_radius_m,
    )
    summary.update(
        {
            "trial_id": trial.trial_id,
            "slot_id": trial.slot_id,
            "variant": trial.variant.name,
            "variant_description": trial.variant.description,
            "repetition_index": trial.repetition_index,
            "topomap_memory_map_dir": topomap_map_dir,
            "topomap_memory_use_clip": trial.variant.topomap_memory_use_clip,
            "topomap_memory_allow_semantic_terms": trial.variant.topomap_memory_allow_semantic_terms,
            "configured_critic_mode": trial.variant.critic_mode,
            "robot_io_contract": ROBOT_IO_CONTRACT,
            "no_hardcoded_labels_or_colors": _summary_no_hardcoded_labels_or_colors(summary, fallback_variant=trial.variant),
        }
    )
    (trial_root / "trial_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _failed_trial_summary(
    config: ResearchConfig,
    trial: TrialSpec,
    *,
    output_root: Path,
    error: BaseException,
) -> dict[str, Any]:
    trial_root = output_root / "trials" / trial.trial_id
    topomap_map_dir = _resolved_topomap_memory_map_dir(trial)
    return {
        "trial_id": trial.trial_id,
        "slot_id": trial.slot_id,
        "variant": trial.variant.name,
        "variant_description": trial.variant.description,
        "episode": trial.episode_name,
        "repetition_index": trial.repetition_index,
        "runner": trial.variant.runner,
        "model": trial.variant.model,
        "qwen_model": trial.variant.qwen_model,
        "object_drive_detector": trial.variant.object_drive_detector,
        "topomap_memory_map_dir": topomap_map_dir,
        "topomap_memory_use_clip": trial.variant.topomap_memory_use_clip,
        "topomap_memory_allow_semantic_terms": trial.variant.topomap_memory_allow_semantic_terms,
        "success": False,
        "final_distance_m": None,
        "best_distance_m": None,
        "best_distance_step": None,
        "best_distance_improvement_m": None,
        "final_distance_improvement_m": None,
        "final_to_best_regression_m": None,
        "reached_success_radius_ever": False,
        "reason": "trial_exception",
        "error": str(error),
        "step_count": 0,
        "wall_clock_s": 0.0,
        "run_dir": str(trial_root),
        "policy_dir": None,
        "evaluator_only_dir": None,
        "camera_contact_sheet": None,
        "progress_contact_sheet": None,
        "rerun_path": None,
        "robot_io_contract": ROBOT_IO_CONTRACT,
        "policy_input_allowlist": POLICY_INPUT_ALLOWLIST,
        "privileged_eval_used": True,
        "privileged_eval_purpose": PRIVILEGED_EVAL_PURPOSE,
        "no_hardcoded_labels_or_colors": _variant_no_hardcoded_labels_or_colors(trial.variant),
        "config_max_steps": config.max_steps,
        "config_success_radius_m": config.success_radius_m,
    }


def _aggregate(
    config: ResearchConfig,
    manifest: dict[str, Any],
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    output_root: Path,
) -> dict[str, Any]:
    aggregate = {
        **{
            key: manifest[key]
            for key in ("experiment_id", "run_id", "objective", "started_at", "git_commit", "dry_run", "strict_model_based")
        },
        "output_root": str(output_root),
        "trial_count": manifest["trial_count"],
        "completed_trial_count": len(summaries),
        "success_count": sum(1 for summary in summaries if summary.get("success")),
        "failure_count": len(failures),
        "warmhub_repo": config.warmhub_repo,
        "robot_io_contract": ROBOT_IO_CONTRACT,
        "policy_input_allowlist": POLICY_INPUT_ALLOWLIST,
        "privileged_eval_used": True,
        "privileged_eval_purpose": PRIVILEGED_EVAL_PURPOSE,
        "no_hardcoded_labels_or_colors": _aggregate_no_hardcoded_labels_or_colors(manifest, [*summaries, *failures]),
        "summaries": summaries,
        "failed_trials": failures,
    }
    distances = [float(summary["final_distance_m"]) for summary in summaries if summary.get("final_distance_m") is not None]
    best_distances = [float(summary["best_distance_m"]) for summary in summaries if summary.get("best_distance_m") is not None]
    aggregate["mean_final_distance_m"] = round(sum(distances) / len(distances), 3) if distances else None
    aggregate["mean_best_distance_m"] = round(sum(best_distances) / len(best_distances), 3) if best_distances else None
    (output_root / "research_loop_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def _trial_manifest(trial: TrialSpec) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "slot_id": trial.slot_id,
        "episode": trial.episode_name,
        "repetition_index": trial.repetition_index,
        "variant": asdict(trial.variant),
    }


def _config_to_dict(config: ResearchConfig) -> dict[str, Any]:
    data = asdict(config)
    data["variants"] = [asdict(variant) for variant in config.variants]
    data["episodes"] = list(config.episodes)
    return data


def _configured_episode(spec: ThorEpisodeSpec, config: ResearchConfig) -> ThorEpisodeSpec:
    if config.max_steps is not None:
        spec = replace(spec, max_steps=config.max_steps)
    if config.success_radius_m is not None:
        spec = replace(spec, success_radius_m=config.success_radius_m)
    return spec


def _resolved_topomap_memory_map_dir(trial: TrialSpec) -> str | None:
    value = trial.variant.topomap_memory_map_dir
    if not value:
        return None
    replacements = {
        "{episode}": trial.episode_name,
        "{episode_name}": trial.episode_name,
        "{variant}": trial.variant.name,
        "{repetition}": str(trial.repetition_index + 1),
    }
    for token, replacement_value in replacements.items():
        value = value.replace(token, replacement_value)
    return value


def _run_op(config: ResearchConfig, summary: dict[str, Any], experiment_ref: str, run_ref: str) -> dict[str, Any]:
    variant_name = str(summary["variant"])
    variant_ref = f"PromptVariant/{_safe_id(f'{config.experiment_id}_{variant_name}')}"
    return {
        "operation": "add",
        "kind": "thing",
        "name": run_ref,
        "data": {
            "experiment": experiment_ref,
            "variant": variant_ref,
            "episodeSpec": f"NavEpisodeSpec/{_safe_id(summary['episode'])}",
            "trialId": summary["trial_id"],
            "runner": summary["runner"],
            "model": summary["model"],
            "criticMode": summary.get("critic_mode") or summary.get("configured_critic_mode") or "",
            "gitCommit": summary.get("git_commit"),
            "outputDir": summary.get("run_dir") or "",
            "success": bool(summary.get("success")),
            "finalDistanceM": summary.get("final_distance_m"),
            "bestDistanceM": summary.get("best_distance_m"),
            "bestDistanceStep": summary.get("best_distance_step"),
            "bestDistanceImprovementM": summary.get("best_distance_improvement_m"),
            "finalDistanceImprovementM": summary.get("final_distance_improvement_m"),
            "finalToBestRegressionM": summary.get("final_to_best_regression_m"),
            "reachedSuccessRadiusEver": bool(summary.get("reached_success_radius_ever")),
            "reason": summary.get("reason") or "",
            "stepCount": int(summary.get("step_count") or 0),
            "wallClockS": float(summary.get("wall_clock_s") or 0.0),
            "policyInputAllowlist": POLICY_INPUT_ALLOWLIST,
            "privilegedEvalUsed": True,
            "privilegedEvalPurpose": PRIVILEGED_EVAL_PURPOSE,
            "robotIoContract": ROBOT_IO_CONTRACT,
            "noHardcodedLabelsOrColors": bool(summary.get("no_hardcoded_labels_or_colors")),
        },
    }


def _artifact_ops(summary: dict[str, Any], run_ref: str) -> list[dict[str, Any]]:
    run_dir = summary.get("run_dir")
    policy_dir = summary.get("policy_dir")
    artifacts = [
        ("episode_summary", Path(run_dir) / "episode_summary.json" if run_dir else None, False),
        ("memory_jsonl", Path(policy_dir) / "memory.jsonl" if policy_dir else None, False),
        ("prompts_dir", Path(policy_dir) / "prompts" if policy_dir else None, False),
        ("topomap_memory_manifest", Path(policy_dir) / "topomap_memory" / "topomap_memory_manifest.json" if policy_dir else None, False),
        ("topomap_memory_query_jsonl", Path(policy_dir) / "topomap_memory" / "query_log.jsonl" if policy_dir else None, False),
        ("topomap_memory_contact_sheets", Path(policy_dir) / "topomap_memory" if policy_dir else None, False),
        ("training_export_dir", summary.get("training_export_dir"), True),
        ("training_policy_steps_jsonl", summary.get("training_policy_steps_jsonl"), True),
        ("training_episode_rollout_json", summary.get("training_episode_rollout_json"), True),
        ("policy_review_trace_json", summary.get("policy_review_trace_json"), False),
        ("camera_contact_sheet", summary.get("camera_contact_sheet"), False),
        ("progress_contact_sheet", summary.get("progress_contact_sheet"), True),
        ("rerun", summary.get("rerun_path"), False),
        ("hidden_evaluator_dir", summary.get("evaluator_only_dir"), True),
    ]
    ops: list[dict[str, Any]] = []
    for artifact_type, path_value, privileged in artifacts:
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        ops.append(
            {
                "operation": "add",
                "kind": "thing",
                "name": f"NavArtifact/{_safe_id(summary['trial_id'] + '_' + artifact_type)}",
                "data": {
                    "run": run_ref,
                    "artifactType": artifact_type,
                    "path": str(path),
                    "sha256": _sha256(path) if path.is_file() else None,
                    "privileged": privileged,
                    "description": f"{artifact_type} artifact for {summary['trial_id']}",
                },
            }
        )
    return ops


def _assessment_op(summary: dict[str, Any], run_ref: str) -> dict[str, Any]:
    return {
        "operation": "add",
        "kind": "assertion",
        "name": f"RunAssessment/{_safe_id(summary['trial_id'])}",
        "about": run_ref,
        "data": {
            "success": bool(summary.get("success")),
            "finalDistanceM": summary.get("final_distance_m"),
            "bestDistanceM": summary.get("best_distance_m"),
            "bestDistanceStep": summary.get("best_distance_step"),
            "bestDistanceImprovementM": summary.get("best_distance_improvement_m"),
            "finalDistanceImprovementM": summary.get("final_distance_improvement_m"),
            "finalToBestRegressionM": summary.get("final_to_best_regression_m"),
            "reachedSuccessRadiusEver": bool(summary.get("reached_success_radius_ever")),
            "reason": summary.get("reason") or "",
            "stepCount": int(summary.get("step_count") or 0),
            "evaluator": "ai2thor-hidden-distance",
            "confidence": 0.9,
            "privilegedEvalUsed": True,
            "notes": "Evaluator uses hidden THOR target distance outside the model-facing policy directory.",
        },
    }


def _failure_op(summary: dict[str, Any], run_ref: str) -> dict[str, Any]:
    evidence = [
        path
        for path in (
            summary.get("camera_contact_sheet"),
            summary.get("policy_review_trace_json"),
            summary.get("progress_contact_sheet"),
            str(Path(summary["policy_dir"]) / "memory.jsonl") if summary.get("policy_dir") else None,
        )
        if path
    ]
    return {
        "operation": "add",
        "kind": "assertion",
        "name": f"FailureObservation/{_safe_id(summary['trial_id'])}",
        "about": run_ref,
        "data": {
            "category": "trial_exception" if summary.get("reason") == "trial_exception" else "navigation_failure",
            "severity": "high" if summary.get("reason") == "trial_exception" else "medium",
            "symptom": str(summary.get("error") or summary.get("reason") or "unsuccessful run"),
            "suspectedCause": (
                "Infrastructure, model endpoint, simulator, or tool execution failure before a scored episode completed."
                if summary.get("reason") == "trial_exception"
                else "Review transcript/contact sheets; do not assume semantic detector failure without visual evidence."
            ),
            "evidenceArtifacts": evidence,
            "nextAction": (
                "Inspect trial logs and environment setup, then rerun before comparing policy behavior."
                if summary.get("reason") == "trial_exception"
                else "Compare prompt variant behavior and inspect whether exploration or visible-landmark servoing was used appropriately."
            ),
            "confidence": 0.6,
        },
    }


def _warmhub_schema_commands(repo: str, shapes: dict[str, dict[str, Any]]) -> str:
    repo_q = shlex.quote(repo)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"wh repo view {repo_q} >/dev/null 2>&1 || wh repo create {repo_q} --visibility private --description 'Open-vocabulary indoor robot navigation research memory'",
    ]
    for name, spec in shapes.items():
        fields = json.dumps(spec["fields"], sort_keys=True)
        description = spec["description"]
        name_q = shlex.quote(name)
        lines.append(
            f"wh shape view {name_q} --repo {repo_q} >/dev/null 2>&1 || "
            f"wh shape create {name_q} --repo {repo_q} --fields {shlex.quote(fields)} --description {shlex.quote(description)}"
        )
    lines.append(f"wh commit submit --repo {repo_q} --file warmhub_ops.json --skip-existing --message 'Log open-vocabulary navigation research loop results'")
    return "\n".join(lines) + "\n"


def _warmhub_agents_md(repo: str) -> str:
    return f"""# Warmhub Startup For Navigation Research Agents

Run this at startup or after context compaction:

```bash
export WARMHUB_REPO={repo}
wh prime
wh repo describe --json
wh thing query --shape NavEvalRun --limit 20 --json
wh thing query --shape PromptVariant --limit 20 --json
wh thing query --shape AgentTask --where status=running --limit 20 --json
wh thing query --shape AgentTask --where status=planned --limit 20 --json
wh assertion list --shape FailureObservation --limit 20 --json
wh thing search "current open vocabulary navigation failures qwen florence exploration" --mode hybrid --json
```

Write one `NavEvalRun` per simulator or robot run, attach `NavArtifact` records
for summaries/contact sheets/prompts/memory, and add `FailureObservation`
assertions for failures or suspicious success cases. Policy inputs must remain
camera frames, previous motion strips, IMU yaw, bounded tool results, and model
memory. Hidden simulator data is allowed only for evaluator/debug artifacts.

Use `flatdisk-sim-research-warmhub note`, `task-start`, and `task-finish` to
write coordination records without hand-authoring Warmhub operations.
"""


def _variant_id(config: ResearchConfig, variant: PromptVariant) -> str:
    return _safe_id(f"{config.experiment_id}_{variant.name}")


def _episode_id(spec: ThorEpisodeSpec) -> str:
    return _safe_id(spec.name)


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return text.strip("-")[:180] or "unnamed"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _git_commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
