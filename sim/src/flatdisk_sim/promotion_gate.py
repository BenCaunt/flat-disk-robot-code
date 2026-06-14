"""Compare navigation runs and gate candidate promotion against a baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Iterable

from .research_loop import DEFAULT_WARMHUB_REPO, _safe_id
from .research_warmhub import commit_ops


PROMOTION_GATE_SCHEMA = "flatdisk.nav_promotion_gate.v1"


def evaluate_promotion(
    *,
    baseline_inputs: Iterable[Path],
    candidate_inputs: Iterable[Path],
    output_dir: Path,
    baseline_variants: Iterable[str] = (),
    candidate_variants: Iterable[str] = (),
    decision_id: str | None = None,
    experiment_id: str | None = None,
    about: str | None = None,
    min_best_improvement_m: float = 0.05,
    max_final_regression_m: float = 0.10,
    require_prompt_audit_pass: bool = False,
    author: str = "flatdisk-sim-promotion-gate",
) -> dict[str, Any]:
    baseline_variant_filter = _variant_filter_set(baseline_variants)
    candidate_variant_filter = _variant_filter_set(candidate_variants)
    baseline_records = _filter_records_by_variant(load_nav_summaries(baseline_inputs), baseline_variant_filter)
    candidate_records = _filter_records_by_variant(load_nav_summaries(candidate_inputs), candidate_variant_filter)
    if not baseline_records:
        raise FileNotFoundError("no baseline navigation summaries found")
    if not candidate_records:
        raise FileNotFoundError("no candidate navigation summaries found")

    output_dir.mkdir(parents=True, exist_ok=True)
    compared_at = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    decision_id = _safe_id(decision_id or _default_decision_id(baseline_records, candidate_records, compared_at=compared_at))
    baseline = summarize_records(baseline_records)
    candidate = summarize_records(candidate_records)
    decision = _promotion_decision(
        baseline,
        candidate,
        min_best_improvement_m=min_best_improvement_m,
        max_final_regression_m=max_final_regression_m,
        require_prompt_audit_pass=require_prompt_audit_pass,
    )
    report = {
        "schema": PROMOTION_GATE_SCHEMA,
        "decision_id": decision_id,
        "experiment_id": experiment_id,
        "compared_at": compared_at,
        "baseline": baseline,
        "candidate": candidate,
        "thresholds": {
            "min_best_improvement_m": min_best_improvement_m,
            "max_final_regression_m": max_final_regression_m,
            "require_prompt_audit_pass": require_prompt_audit_pass,
        },
        "filters": {
            "baseline_variants": sorted(baseline_variant_filter),
            "candidate_variants": sorted(candidate_variant_filter),
        },
        "decision": decision,
        "policy_constraints": [
            "Promotion uses privileged evaluator metrics only outside the policy loop.",
            "Candidate runs must preserve the real robot I/O contract and general model-based policy inputs.",
            "Hidden target distance, simulator object metadata, and evaluator labels are never model-facing prompt inputs.",
        ],
    }
    report_path = output_dir / "promotion_decision.json"
    markdown_path = output_dir / "promotion_decision.md"
    ops_path = output_dir / "warmhub_ops.json"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report["warmhub_ops_path"] = str(ops_path)
    report["warmhub_ops"] = _warmhub_ops(report, about=about or _default_about(experiment_id), author=author)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    ops_path.write_text(json.dumps(report["warmhub_ops"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_nav_summaries(input_paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _discover_summary_paths(input_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate_context = _aggregate_context(payload, source_path=path)
        if isinstance(payload.get("summaries"), list):
            for summary in payload["summaries"]:
                if isinstance(summary, dict):
                    records.append(_with_context(summary, aggregate_context, source_path=path))
        elif isinstance(payload, dict):
            records.append(_with_context(payload, aggregate_context, source_path=path))
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = len(records)
    success_count = sum(1 for record in records if record.get("success"))
    best_distances = [_optional_float(record.get("best_distance_m")) for record in records]
    final_distances = [_optional_float(record.get("final_distance_m")) for record in records]
    regressions = [_final_to_best_regression(record) for record in records]
    best_distances = [value for value in best_distances if value is not None]
    final_distances = [value for value in final_distances if value is not None]
    regressions = [value for value in regressions if value is not None]
    prompt_audit_pass_count = sum(1 for record in records if _prompt_audit_passed(record))
    no_hardcoded_count = sum(1 for record in records if _record_no_hardcoded_labels_or_colors(record))
    return {
        "completed_trial_count": completed,
        "success_count": success_count,
        "success_rate": round(success_count / completed, 6) if completed else 0.0,
        "mean_best_distance_m": _mean(best_distances),
        "mean_final_distance_m": _mean(final_distances),
        "mean_final_to_best_regression_m": _mean(regressions),
        "prompt_audit_pass_count": prompt_audit_pass_count,
        "prompt_audit_pass_rate": round(prompt_audit_pass_count / completed, 6) if completed else 0.0,
        "no_hardcoded_labels_or_colors_count": no_hardcoded_count,
        "no_hardcoded_labels_or_colors_rate": round(no_hardcoded_count / completed, 6) if completed else 0.0,
        "variants": sorted({str(record.get("variant") or "") for record in records if record.get("variant")}),
        "episodes": sorted({str(record.get("episode") or "") for record in records if record.get("episode")}),
        "git_commits": sorted({str(record.get("git_commit") or "") for record in records if record.get("git_commit")}),
        "sources": sorted({str(record.get("source_summary_path") or "") for record in records if record.get("source_summary_path")}),
        "runs": [_compact_run_record(record) for record in records],
    }


def _filter_records_by_variant(records: list[dict[str, Any]], variants: Iterable[str]) -> list[dict[str, Any]]:
    allowed = _variant_filter_set(variants)
    if not allowed:
        return records
    return [record for record in records if str(record.get("variant") or "") in allowed]


def _variant_filter_set(variants: Iterable[str]) -> set[str]:
    return {str(variant).strip() for variant in variants if str(variant).strip()}


def _promotion_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    min_best_improvement_m: float,
    max_final_regression_m: float,
    require_prompt_audit_pass: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    blockers: list[str] = []
    warnings: list[str] = []

    if candidate["completed_trial_count"] < baseline["completed_trial_count"]:
        warnings.append("candidate has fewer completed trials than baseline")
    if set(candidate["episodes"]) != set(baseline["episodes"]):
        warnings.append("candidate episode set differs from baseline")
    if candidate["no_hardcoded_labels_or_colors_rate"] < 1.0:
        blockers.append("candidate violates no_hardcoded_labels_or_colors on at least one run")
    if require_prompt_audit_pass and candidate["prompt_audit_pass_rate"] < 1.0:
        blockers.append("candidate prompt audit did not pass for every run")

    baseline_success = float(baseline["success_rate"])
    candidate_success = float(candidate["success_rate"])
    success_delta = round(candidate_success - baseline_success, 6)
    baseline_best = _optional_float(baseline.get("mean_best_distance_m"))
    candidate_best = _optional_float(candidate.get("mean_best_distance_m"))
    baseline_final = _optional_float(baseline.get("mean_final_distance_m"))
    candidate_final = _optional_float(candidate.get("mean_final_distance_m"))
    best_improvement = _distance_improvement(baseline_best, candidate_best)
    final_regression = _distance_regression(baseline_final, candidate_final)

    primary_improved = False
    if success_delta > 0:
        primary_improved = True
        reasons.append(f"success rate improved by {success_delta:.3f}")
    elif success_delta < 0:
        blockers.append(f"success rate regressed by {-success_delta:.3f}")
    elif best_improvement is not None and best_improvement >= min_best_improvement_m:
        primary_improved = True
        reasons.append(f"mean best distance improved by {best_improvement:.3f} m")
    else:
        if best_improvement is None:
            blockers.append("best-distance metric is missing for baseline or candidate")
        else:
            blockers.append(
                f"mean best-distance improvement {best_improvement:.3f} m is below required {min_best_improvement_m:.3f} m"
            )

    if final_regression is not None and final_regression > max_final_regression_m:
        blockers.append(f"mean final distance regressed by {final_regression:.3f} m")

    promote = primary_improved and not blockers
    return {
        "promote": promote,
        "status": "promote" if promote else "reject",
        "primary_improved": primary_improved,
        "success_rate_delta": success_delta,
        "mean_best_distance_improvement_m": best_improvement,
        "mean_final_distance_regression_m": final_regression,
        "reasons": reasons,
        "blockers": blockers,
        "warnings": warnings,
    }


def _discover_summary_paths(input_paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for input_path in input_paths:
        path = input_path.expanduser()
        if path.is_file() and path.name in {"research_loop_summary.json", "trial_summary.json", "episode_summary.json"}:
            found.append(path)
        elif path.is_dir():
            direct = path / "research_loop_summary.json"
            if direct.exists():
                found.append(direct)
            else:
                found.extend(path.glob("**/research_loop_summary.json"))
                found.extend(path.glob("**/trial_summary.json"))
                found.extend(path.glob("**/episode_summary.json"))
    return sorted(set(found))


def _aggregate_context(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    return {
        "source_summary_path": str(source_path),
        "aggregate_experiment_id": payload.get("experiment_id"),
        "aggregate_run_id": payload.get("run_id"),
        "aggregate_git_commit": payload.get("git_commit"),
        "aggregate_no_hardcoded_labels_or_colors": payload.get("no_hardcoded_labels_or_colors"),
    }


def _with_context(summary: dict[str, Any], context: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    record = dict(summary)
    record.setdefault("source_summary_path", str(source_path))
    if context.get("aggregate_git_commit") and not record.get("git_commit"):
        record["git_commit"] = context["aggregate_git_commit"]
    if context.get("aggregate_experiment_id") and not record.get("experiment_id"):
        record["experiment_id"] = context["aggregate_experiment_id"]
    if context.get("aggregate_run_id") and not record.get("run_id"):
        record["run_id"] = context["aggregate_run_id"]
    if record.get("no_hardcoded_labels_or_colors") is None:
        prompt_audit = record.get("prompt_audit") if isinstance(record.get("prompt_audit"), dict) else {}
        if "no_hardcoded_labels_or_colors" in prompt_audit:
            record["no_hardcoded_labels_or_colors"] = bool(prompt_audit["no_hardcoded_labels_or_colors"])
        elif context.get("aggregate_no_hardcoded_labels_or_colors") is not None:
            record["no_hardcoded_labels_or_colors"] = context["aggregate_no_hardcoded_labels_or_colors"]
    return record


def _compact_run_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": record.get("trial_id"),
        "variant": record.get("variant"),
        "episode": record.get("episode"),
        "success": bool(record.get("success")),
        "reason": record.get("reason"),
        "best_distance_m": _optional_float(record.get("best_distance_m")),
        "best_distance_step": record.get("best_distance_step"),
        "final_distance_m": _optional_float(record.get("final_distance_m")),
        "final_to_best_regression_m": _final_to_best_regression(record),
        "prompt_audit_passed": _prompt_audit_passed(record),
        "no_hardcoded_labels_or_colors": _record_no_hardcoded_labels_or_colors(record),
        "git_commit": record.get("git_commit"),
        "source_summary_path": record.get("source_summary_path"),
    }


def _prompt_audit_passed(record: dict[str, Any]) -> bool:
    prompt_audit = record.get("prompt_audit")
    if isinstance(prompt_audit, dict):
        return bool(prompt_audit.get("prompt_audit_passed"))
    return bool(record.get("prompt_audit_passed", False))


def _record_no_hardcoded_labels_or_colors(record: dict[str, Any]) -> bool:
    if record.get("no_hardcoded_labels_or_colors") is not None:
        return bool(record["no_hardcoded_labels_or_colors"])
    prompt_audit = record.get("prompt_audit")
    if isinstance(prompt_audit, dict) and prompt_audit.get("no_hardcoded_labels_or_colors") is not None:
        return bool(prompt_audit["no_hardcoded_labels_or_colors"])
    return False


def _final_to_best_regression(record: dict[str, Any]) -> float | None:
    explicit = _optional_float(record.get("final_to_best_regression_m"))
    if explicit is not None:
        return explicit
    best = _optional_float(record.get("best_distance_m"))
    final = _optional_float(record.get("final_distance_m"))
    if best is None or final is None:
        return None
    return round(max(0.0, final - best), 6)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _distance_improvement(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return round(baseline - candidate, 6)


def _distance_regression(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return round(max(0.0, candidate - baseline), 6)


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _default_decision_id(baseline_records: list[dict[str, Any]], candidate_records: list[dict[str, Any]], *, compared_at: str) -> str:
    baseline_token = baseline_records[0].get("variant") or baseline_records[0].get("trial_id") or "baseline"
    candidate_token = candidate_records[0].get("variant") or candidate_records[0].get("trial_id") or "candidate"
    return f"promotion_{baseline_token}_vs_{candidate_token}_{compared_at.replace(':', '').replace('-', '')}"


def _default_about(experiment_id: str | None) -> str | None:
    return f"NavExperiment/{_safe_id(experiment_id)}" if experiment_id else None


def _warmhub_ops(report: dict[str, Any], *, about: str | None, author: str) -> list[dict[str, Any]]:
    decision = report["decision"]
    structured = {
        "operation": "add",
        "kind": "assertion",
        "name": f"PromotionDecision/{_safe_id(report['decision_id'])}",
        "about": about,
        "data": {
            "decisionId": report["decision_id"],
            "status": decision["status"],
            "promote": bool(decision["promote"]),
            "createdAt": report["compared_at"],
            "author": author,
            "baselineVariants": report["baseline"].get("variants", []),
            "candidateVariants": report["candidate"].get("variants", []),
            "baselineTrialCount": int(report["baseline"].get("completed_trial_count") or 0),
            "candidateTrialCount": int(report["candidate"].get("completed_trial_count") or 0),
            "successRateDelta": decision["success_rate_delta"],
            "meanBestDistanceImprovementM": decision["mean_best_distance_improvement_m"],
            "meanFinalDistanceRegressionM": decision["mean_final_distance_regression_m"],
            "blockers": decision["blockers"],
            "warnings": decision["warnings"],
            "reasons": decision["reasons"],
            "reportPath": report["report_path"],
            "markdownPath": report["markdown_path"],
            "confidence": 0.85,
        },
    }
    note_data = {
        "author": author,
        "createdAt": report["compared_at"],
        "note": (
            f"Promotion gate {decision['status']}: "
            f"best-distance delta={decision['mean_best_distance_improvement_m']}, "
            f"final-distance regression={decision['mean_final_distance_regression_m']}. "
            f"Report: {report['report_path']}"
        ),
        "tags": ["open-vocab-nav", "promotion-gate", decision["status"]],
        "confidence": 0.85,
    }
    return [
        structured,
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"AgentNote/{_safe_id(report['decision_id'])}",
            "about": about,
            "data": note_data,
        },
    ]


def _markdown_report(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    decision = report["decision"]
    lines = [
        "# Navigation Promotion Decision",
        "",
        f"Decision: **{decision['status']}**",
        f"Decision ID: `{report['decision_id']}`",
        "",
        "| Set | Trials | Success rate | Mean best (m) | Mean final (m) | Generality rate | Prompt audit rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_row("Baseline", baseline),
        _metric_row("Candidate", candidate),
        "",
        "Thresholds:",
        f"- Minimum best-distance improvement: `{report['thresholds']['min_best_improvement_m']}` m",
        f"- Maximum allowed final-distance regression: `{report['thresholds']['max_final_regression_m']}` m",
        f"- Require every prompt audit pass: `{report['thresholds']['require_prompt_audit_pass']}`",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in decision["reasons"] or ["none"])
    lines.append("")
    lines.append("Blockers:")
    lines.extend(f"- {blocker}" for blocker in decision["blockers"] or ["none"])
    if decision["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in decision["warnings"])
    lines.append("")
    lines.append("Policy constraints:")
    lines.extend(f"- {constraint}" for constraint in report["policy_constraints"])
    lines.append("")
    return "\n".join(lines)


def _metric_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {summary['completed_trial_count']} | {_format_optional(summary['success_rate'])} | "
        f"{_format_optional(summary['mean_best_distance_m'])} | {_format_optional(summary['mean_final_distance_m'])} | "
        f"{_format_optional(summary['no_hardcoded_labels_or_colors_rate'])} | "
        f"{_format_optional(summary['prompt_audit_pass_rate'])} |"
    )


def _format_optional(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True, help="Baseline research summary, trial summary, or directory.")
    parser.add_argument("--candidate", type=Path, action="append", required=True, help="Candidate research summary, trial summary, or directory.")
    parser.add_argument("--baseline-variant", action="append", default=[], help="Only include baseline records with this variant name. Repeatable.")
    parser.add_argument("--candidate-variant", action="append", default=[], help="Only include candidate records with this variant name. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-id", default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--about", default=None)
    parser.add_argument("--min-best-improvement-m", type=float, default=0.05)
    parser.add_argument("--max-final-regression-m", type=float, default=0.10)
    parser.add_argument("--require-prompt-audit-pass", action="store_true")
    parser.add_argument("--author", default="flatdisk-sim-promotion-gate")
    parser.add_argument("--repo", default=DEFAULT_WARMHUB_REPO)
    parser.add_argument("--commit-warmhub", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_promotion(
        baseline_inputs=args.baseline,
        candidate_inputs=args.candidate,
        output_dir=args.output_dir,
        baseline_variants=args.baseline_variant,
        candidate_variants=args.candidate_variant,
        decision_id=args.decision_id,
        experiment_id=args.experiment_id,
        about=args.about,
        min_best_improvement_m=args.min_best_improvement_m,
        max_final_regression_m=args.max_final_regression_m,
        require_prompt_audit_pass=args.require_prompt_audit_pass,
        author=args.author,
    )
    if args.commit_warmhub:
        commit_ops(args.repo, report["warmhub_ops"], message="Log navigation promotion decision")
    print(
        json.dumps(
            {
                "decision_id": report["decision_id"],
                "status": report["decision"]["status"],
                "promote": report["decision"]["promote"],
                "report_path": report["report_path"],
                "markdown_path": report["markdown_path"],
                "warmhub_ops_path": report["warmhub_ops_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_reject and not report["decision"]["promote"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
