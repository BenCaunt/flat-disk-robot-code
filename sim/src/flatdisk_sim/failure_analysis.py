"""Analyze navigation policy review traces into next-iteration guidance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .research_loop import DEFAULT_WARMHUB_REPO, _safe_id
from .research_warmhub import commit_ops


FAILURE_ANALYSIS_SCHEMA = "flatdisk.nav_failure_analysis.v1"


def analyze_failure_traces(
    input_paths: Iterable[Path],
    *,
    output_dir: Path,
    analysis_id: str | None = None,
    experiment_id: str | None = None,
    about: str | None = None,
    author: str = "flatdisk-sim-failure-analysis",
) -> dict[str, Any]:
    trace_paths = _discover_trace_paths(input_paths)
    raw_trace_records = [(trace, path) for path in trace_paths for trace in _load_trace_records(path)]
    trace_records, duplicate_records = _dedupe_trace_records(raw_trace_records)
    traces = [trace for trace, _path in trace_records]
    if not traces:
        raise FileNotFoundError("no policy_review_trace.json or policy_review_traces.jsonl files found")

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = _safe_id(analysis_id or _default_analysis_id(traces))
    runs = [_summarize_trace(trace, source_path=path) for trace, path in trace_records]
    aggregate_flags: Counter[str] = Counter()
    aggregate_actor_tools: Counter[str] = Counter()
    aggregate_executed_tools: Counter[str] = Counter()
    for run in runs:
        aggregate_flags.update(run["review_flag_counts"])
        aggregate_actor_tools.update(run["actor_tool_counts"])
        aggregate_executed_tools.update(run["executed_tool_counts"])
    recommendations = _recommendations(
        aggregate_flags=aggregate_flags,
        aggregate_actor_tools=aggregate_actor_tools,
        aggregate_executed_tools=aggregate_executed_tools,
        runs=runs,
    )
    candidate_variant = _candidate_variant(recommendations)
    report = {
        "schema": FAILURE_ANALYSIS_SCHEMA,
        "analysis_id": analysis_id,
        "experiment_id": experiment_id,
        "input_trace_record_count": len(raw_trace_records),
        "trace_count": len(traces),
        "duplicate_trace_count": len(duplicate_records),
        "duplicate_trace_records": duplicate_records,
        "run_count": len(runs),
        "runs": runs,
        "aggregate": {
            "review_flag_counts": dict(sorted(aggregate_flags.items())),
            "actor_tool_counts": dict(sorted(aggregate_actor_tools.items())),
            "executed_tool_counts": dict(sorted(aggregate_executed_tools.items())),
        },
        "recommendations": recommendations,
        "candidate_variant": candidate_variant,
        "policy_constraints": [
            "Do not use hidden simulator target pose, object metadata, or distance as policy input.",
            "Use review traces, contact-sheet paths, actor/tool JSON, and general grounding flags as model-facing evidence.",
            "Keep variants general: no static scene-specific object/color scripts.",
        ],
    }
    report_path = output_dir / "failure_analysis.json"
    markdown_path = output_dir / "failure_analysis.md"
    ops_path = output_dir / "warmhub_ops.json"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report["warmhub_ops_path"] = str(ops_path)
    report["warmhub_ops"] = _warmhub_ops(report, about=about or _default_about(experiment_id, traces), author=author)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    ops_path.write_text(json.dumps(report["warmhub_ops"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _discover_trace_paths(input_paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for input_path in input_paths:
        path = input_path.expanduser()
        if path.is_file() and path.name in {"policy_review_trace.json", "policy_review_traces.jsonl"}:
            found.append(path)
        elif path.is_dir():
            candidates = [
                path / "training_export" / "policy_review_traces.jsonl",
                path / "policy_review_trace.json",
            ]
            found.extend(candidate for candidate in candidates if candidate.exists())
            found.extend(path.glob("**/policy_review_trace.json"))
            found.extend(path.glob("**/policy_review_traces.jsonl"))
    return sorted(set(found))


def _dedupe_trace_records(trace_records: list[tuple[dict[str, Any], Path]]) -> tuple[list[tuple[dict[str, Any], Path]], list[dict[str, Any]]]:
    deduped: list[tuple[dict[str, Any], Path]] = []
    seen: dict[str, Path] = {}
    duplicates: list[dict[str, Any]] = []
    for trace, path in trace_records:
        key = _trace_identity(trace)
        if key in seen:
            duplicates.append(
                {
                    "identity": key,
                    "kept_source_path": str(seen[key]),
                    "duplicate_source_path": str(path),
                }
            )
            continue
        seen[key] = path
        deduped.append((trace, path))
    return deduped, duplicates


def _trace_identity(trace: dict[str, Any]) -> str:
    record_id = trace.get("record_id")
    if isinstance(record_id, str) and record_id.strip():
        return f"record_id:{record_id.strip()}"
    digest = hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _load_trace_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = [json.loads(path.read_text(encoding="utf-8"))]
    flattened: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record.get("traces"), list):
            flattened.extend(item for item in record["traces"] if isinstance(item, dict))
        else:
            flattened.append(record)
    return flattened


def _summarize_trace(trace: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    flag_counts: Counter[str] = Counter()
    actor_tools: Counter[str] = Counter()
    executed_tools: Counter[str] = Counter()
    servo_prompts: Counter[str] = Counter()
    unstable_servo_prompts: Counter[str] = Counter()
    no_detection_prompts: Counter[str] = Counter()
    replacement_steps: list[int] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("step") or 0)
        for flag in step.get("review_flags", []):
            if isinstance(flag, str):
                flag_counts[flag] += 1
        actor_action = step.get("actor_action") if isinstance(step.get("actor_action"), dict) else {}
        executed_action = step.get("executed_action") if isinstance(step.get("executed_action"), dict) else {}
        actor_tool = str(actor_action.get("tool") or "")
        executed_tool = str(executed_action.get("tool") or "")
        if actor_tool:
            actor_tools[actor_tool] += 1
        if executed_tool:
            executed_tools[executed_tool] += 1
        if step.get("actor_action_replaced"):
            replacement_steps.append(step_index)
        tool_result = step.get("tool_result") if isinstance(step.get("tool_result"), dict) else {}
        if actor_tool == "visual_servo_object" or tool_result.get("action") == "visual_servo_object":
            prompt = _action_prompt(actor_action) or str(tool_result.get("prompt") or "")
            if prompt:
                servo_prompts[prompt] += 1
                if "visual_servo_grounding_not_stable" in step.get("review_flags", []):
                    unstable_servo_prompts[prompt] += 1
                if "visual_servo_no_detection" in step.get("review_flags", []):
                    no_detection_prompts[prompt] += 1
    return {
        "record_id": trace.get("record_id"),
        "source_path": str(source_path),
        "goal": (trace.get("task") or {}).get("goal") if isinstance(trace.get("task"), dict) else None,
        "episode": (trace.get("task") or {}).get("episode") if isinstance(trace.get("task"), dict) else None,
        "reason": (trace.get("run") or {}).get("reason") if isinstance(trace.get("run"), dict) else None,
        "step_count": int(trace.get("step_count") or len(steps)),
        "review_flag_counts": dict(sorted(flag_counts.items())),
        "actor_tool_counts": dict(sorted(actor_tools.items())),
        "executed_tool_counts": dict(sorted(executed_tools.items())),
        "actor_action_replaced_steps": replacement_steps,
        "visual_servo_prompts": dict(sorted(servo_prompts.items())),
        "unstable_visual_servo_prompts": dict(sorted(unstable_servo_prompts.items())),
        "no_detection_visual_servo_prompts": dict(sorted(no_detection_prompts.items())),
        "policy_safety": trace.get("policy_safety", {}),
    }

def _recommendations(
    *,
    aggregate_flags: Counter[str],
    aggregate_actor_tools: Counter[str],
    aggregate_executed_tools: Counter[str],
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    if aggregate_flags.get("actor_repeated_servo_prompt_after_grounding_audit_requested_change", 0) > 0:
        recommendations.append(
            {
                "id": "grounding_audit_must_bind_next_action",
                "priority": "high",
                "evidence": "Actor repeated a visual_servo_object prompt after its own grounding_audit said the previous box mismatched or the prompt should change.",
                "general_rule": (
                    "When grounding_audit.previous_visual_servo_box_matches_intended_object is false or "
                    "grounding_audit.next_prompt_should_change is true, the next action must not reuse the same "
                    "visual_servo_object prompt. Choose a viewpoint change, topomap query, or a visually distinct "
                    "currently visible landmark phrase."
                ),
            }
        )
    if aggregate_flags.get("visual_servo_no_detection", 0) > 0 or aggregate_flags.get("visual_servo_grounding_not_stable", 0) > 0:
        recommendations.append(
            {
                "id": "treat_unstable_servo_as_weak_control",
                "priority": "high",
                "evidence": "Visual servo calls produced sparse or no detector grounding.",
                "general_rule": (
                    "Treat no_detection, sparse_detection_coverage, or insufficient_status_history as weak control evidence, "
                    "even if the robot moved. Update memory with failed_servo_prompts and prefer viewpoint diversity before retrying."
                ),
            }
        )
    if aggregate_flags.get("actor_action_replaced", 0) > 0:
        recommendations.append(
            {
                "id": "train_or_prompt_against_guard_replacements",
                "priority": "medium",
                "evidence": "The consistency guard replaced actor actions with wait.",
                "general_rule": (
                    "Use replaced steps as negative SFT/RL examples. The target behavior is to make Qwen internalize the guard: "
                    "change prompt/tool/viewpoint before the harness has to replace the action."
                ),
            }
        )
    if aggregate_actor_tools.get("query_topomap_memory", 0) == 0 and aggregate_flags.get("visual_servo_grounding_not_stable", 0) >= 2:
        recommendations.append(
            {
                "id": "query_memory_after_repeated_grounding_failures",
                "priority": "medium",
                "evidence": "Repeated grounding failures occurred without a topomap memory query.",
                "general_rule": (
                    "After repeated unstable visual-servo attempts from similar views, query image memory or take a bounded exploratory turn "
                    "instead of refining the same phrase repeatedly."
                ),
            }
        )
    if any(run.get("actor_tool_counts", {}).get("visual_servo_object", 0) > run.get("executed_tool_counts", {}).get("visual_servo_object", 0) for run in runs):
        recommendations.append(
            {
                "id": "separate_actor_intent_from_executed_motion",
                "priority": "medium",
                "evidence": "Actor selected more servo actions than were actually executed.",
                "general_rule": (
                    "Review traces should be used in training with both actor_action and executed_action. Do not train Qwen to imitate actions "
                    "that the harness rejected; prefer executed actions or explicitly rejected negative examples."
                ),
            }
        )
    return recommendations


def _candidate_variant(recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = [item["id"] for item in recommendations]
    if "grounding_audit_must_bind_next_action" in source_ids:
        rules = [
            "Use action_history_summary as explicit control evidence; if it reports same_prompt_repeat_is_contradicted_by_prior_audit, do not call visual_servo_object with that prompt.",
            "Before any visual_servo_object call, write grounding_audit.evidence tying the prompt to a currently visible instance in the latest RGB frame.",
            "When a previous audit requested a prompt change, select a visibly distinct waypoint phrase, check_object_grounding, query_topomap_memory, or a bounded viewpoint-changing action.",
            "After a critic rejection, obey replacement_action when present; otherwise choose a bounded information-gathering action and update memory with the rejected prompt.",
            "Do not turn a failed final-goal phrase into a longer synonym list; change viewpoint or use a different visible landmark selected from the latest RGB frame.",
        ]
        critic_rules = [
            "Reject visual_servo_object when action_history_summary.same_prompt_repeat_is_contradicted_by_prior_audit names the same prompt.",
            "Reject visual_servo_object when grounding_audit.previous_visual_servo_box_matches_intended_object is false and the action repeats the previous prompt.",
            "Reject stop unless memory_update.arrival_evidence cites latest RGB evidence, not detector status alone.",
            "Approve check_object_grounding, query_topomap_memory, or bounded viewpoint changes after repeated unstable servo attempts.",
        ]
        return {
            "name": "qwen_grounding_audit_critic_v1",
            "description": "Trace-derived critic-enabled recovery variant for repeated contradicted phrase grounding.",
            "prompt_profile": "grounding-audit-critic-action-history-v1",
            "actor_rules": rules,
            "critic_rules": critic_rules,
            "critic_mode": "same-model",
            "source_recommendation_ids": source_ids,
            "no_static_object_or_color_examples": True,
            "topomap_memory_allow_semantic_terms": False,
        }
    rules = [
        "Maintain failed_servo_prompts and failed_viewpoints in memory_update; remove an entry only after a later tool result reports stable grounding.",
        "If the previous grounding_audit says the box mismatched or next_prompt_should_change is true, do not reuse the same visual_servo_object prompt on the next step.",
        "After two rejected or no-motion steps from similar views, choose a viewpoint-changing tool or query_topomap_memory before another visual_servo_object call.",
        "Use non-target visible landmarks only as navigation waypoints; state why the landmark move should improve the next observation and reassess immediately afterward.",
        "Treat sparse_detection_coverage and no_detection as weak control evidence, not proof that the robot is moving toward the goal.",
    ]
    critic_rules = [
        "Reject a repeated visual_servo_object prompt when the previous grounding_audit says the prompt should change or the box mismatched.",
        "Warn when the actor narrates a strategy change but the emitted action reuses the same tool arguments.",
        "Approve bounded viewpoint changes or memory queries after repeated unstable phrase grounding.",
    ]
    return {
        "name": "qwen_grounding_recovery_v1",
        "description": "General recovery variant derived from policy-review traces with repeated unstable phrase grounding.",
        "prompt_profile": "grounding-recovery-v1",
        "actor_rules": rules,
        "critic_rules": critic_rules,
        "critic_mode": "none",
        "source_recommendation_ids": source_ids,
        "no_static_object_or_color_examples": True,
        "topomap_memory_allow_semantic_terms": False,
    }


def _warmhub_ops(report: dict[str, Any], *, about: str | None, author: str) -> list[dict[str, Any]]:
    note = _warmhub_note(report)
    return [
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"AgentNote/{_safe_id(report['analysis_id'])}",
            "about": about or "NavExperiment/open_vocab_nav_qwen_strategy_runpod_linux_v1",
            "data": {
                "author": author,
                "createdAt": _created_at_placeholder(),
                "note": note,
                "tags": ["open-vocab-nav", "failure-analysis", "policy-review-trace", "prompt-design"],
                "confidence": 0.82,
            },
        }
    ]


def _warmhub_note(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    top_flags = ", ".join(f"{key}={value}" for key, value in aggregate["review_flag_counts"].items()) or "none"
    recommendation_ids = ", ".join(item["id"] for item in report["recommendations"]) or "none"
    variant = report["candidate_variant"]
    dedupe_note = ""
    if report.get("duplicate_trace_count"):
        dedupe_note = (
            f"Deduped {report['input_trace_record_count']} input trace record(s) "
            f"to {report['trace_count']} unique trace(s). "
        )
    return (
        f"Failure analysis {report['analysis_id']} reviewed {report['run_count']} policy-review trace(s). "
        f"{dedupe_note}Top flags: {top_flags}. Recommended next checks: {recommendation_ids}. "
        f"Candidate general variant {variant['name']} should bind grounding_audit to the next action, avoid repeated failed servo prompts, "
        "and use viewpoint changes or image memory after repeated unstable grounding."
    )


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Navigation Failure Analysis",
        "",
        f"Analysis: `{report['analysis_id']}`",
        f"Traces: {report['trace_count']}",
        "",
        "## Aggregate Flags",
    ]
    for key, value in report["aggregate"]["review_flag_counts"].items():
        lines.append(f"- `{key}`: {value}")
    if report.get("duplicate_trace_count"):
        lines.extend(
            [
                "",
                "## Deduplication",
                (
                    f"Input trace records: `{report.get('input_trace_record_count')}`, "
                    f"unique traces: `{report.get('trace_count')}`, "
                    f"duplicates collapsed: `{report.get('duplicate_trace_count')}`"
                ),
            ]
        )
    lines.extend(["", "## Recommendations"])
    for item in report["recommendations"]:
        lines.append(f"- `{item['id']}` ({item['priority']}): {item['general_rule']}")
    lines.extend(["", "## Candidate Variant", "", f"`{report['candidate_variant']['name']}`"])
    for rule in report["candidate_variant"]["actor_rules"]:
        lines.append(f"- {rule}")
    return "\n".join(lines) + "\n"


def _action_prompt(action: dict[str, Any]) -> str:
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    return str(args.get("prompt") or "")


def _default_analysis_id(traces: list[dict[str, Any]]) -> str:
    first = traces[0]
    record_id = first.get("record_id") if isinstance(first, dict) else None
    return f"{record_id or 'policy-review'}-failure-analysis"


def _default_about(experiment_id: str | None, traces: list[dict[str, Any]]) -> str | None:
    if experiment_id:
        return f"NavExperiment/{_safe_id(experiment_id)}"
    first = traces[0]
    record_id = first.get("record_id") if isinstance(first, dict) else None
    if record_id:
        return f"NavEvalRun/{_safe_id(str(record_id))}"
    return None


def _created_at_placeholder() -> str:
    from time import gmtime, strftime

    return strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Review trace file, training_export dir, or research output dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--about", default=None)
    parser.add_argument("--author", default="flatdisk-sim-failure-analysis")
    parser.add_argument("--repo", default=DEFAULT_WARMHUB_REPO)
    parser.add_argument("--commit-warmhub", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_failure_traces(
        args.input,
        output_dir=args.output_dir,
        analysis_id=args.analysis_id,
        experiment_id=args.experiment_id,
        about=args.about,
        author=args.author,
    )
    if args.commit_warmhub:
        commit_ops(args.repo, report["warmhub_ops"], message="Log navigation failure analysis")
    print(json.dumps({key: report[key] for key in ("analysis_id", "trace_count", "report_path", "markdown_path", "warmhub_ops_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
