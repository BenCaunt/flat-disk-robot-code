from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.promotion_gate import evaluate_promotion, load_nav_summaries, main


def _run_summary(
    *,
    variant: str,
    episode: str = "bathroom_toilet",
    success: bool = False,
    best_distance_m: float,
    final_distance_m: float,
    no_hardcoded: bool = True,
    prompt_audit_passed: bool = False,
) -> dict:
    return {
        "trial_id": f"{variant}_{episode}_r1",
        "variant": variant,
        "episode": episode,
        "success": success,
        "reason": "harness_mode_complete_before_hidden_success",
        "best_distance_m": best_distance_m,
        "best_distance_step": 3,
        "final_distance_m": final_distance_m,
        "prompt_audit": {
            "prompt_audit_passed": prompt_audit_passed,
            "no_hardcoded_labels_or_colors": no_hardcoded,
        },
    }


def _write_research_summary(path: Path, *, run_id: str, summaries: list[dict], no_hardcoded: bool = True) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp",
                "run_id": run_id,
                "git_commit": "abc123",
                "completed_trial_count": len(summaries),
                "success_count": sum(1 for summary in summaries if summary["success"]),
                "no_hardcoded_labels_or_colors": no_hardcoded,
                "summaries": summaries,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_evaluate_promotion_promotes_best_distance_improvement_without_prompt_audit_requirement(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline" / "research_loop_summary.json"
    candidate_path = tmp_path / "candidate" / "research_loop_summary.json"
    _write_research_summary(
        baseline_path,
        run_id="baseline-run",
        summaries=[_run_summary(variant="qwen_baseline", best_distance_m=1.0, final_distance_m=1.0)],
    )
    _write_research_summary(
        candidate_path,
        run_id="candidate-run",
        summaries=[_run_summary(variant="qwen_candidate", best_distance_m=0.9, final_distance_m=1.05)],
    )

    report = evaluate_promotion(
        baseline_inputs=[baseline_path.parent],
        candidate_inputs=[candidate_path.parent],
        output_dir=tmp_path / "gate",
        decision_id="candidate-vs-baseline",
    )

    assert report["decision"]["promote"] is True
    assert report["decision"]["status"] == "promote"
    assert report["decision"]["mean_best_distance_improvement_m"] == 0.1
    assert report["decision"]["mean_final_distance_regression_m"] == 0.05
    assert report["candidate"]["prompt_audit_pass_rate"] == 0.0
    assert report["candidate"]["no_hardcoded_labels_or_colors_rate"] == 1.0
    assert (tmp_path / "gate" / "promotion_decision.json").exists()
    assert (tmp_path / "gate" / "promotion_decision.md").exists()
    ops = json.loads((tmp_path / "gate" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert ops[0]["name"] == "AgentNote/candidate-vs-baseline"
    assert "promotion-gate" in ops[0]["data"]["tags"]


def test_evaluate_promotion_rejects_metric_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline" / "research_loop_summary.json"
    candidate_path = tmp_path / "candidate" / "research_loop_summary.json"
    _write_research_summary(
        baseline_path,
        run_id="baseline-run",
        summaries=[_run_summary(variant="qwen_baseline", best_distance_m=0.5, final_distance_m=0.6)],
    )
    _write_research_summary(
        candidate_path,
        run_id="candidate-run",
        summaries=[_run_summary(variant="qwen_candidate", best_distance_m=0.55, final_distance_m=0.75)],
    )

    report = evaluate_promotion(
        baseline_inputs=[baseline_path],
        candidate_inputs=[candidate_path],
        output_dir=tmp_path / "gate",
        decision_id="regression",
    )

    assert report["decision"]["promote"] is False
    assert report["decision"]["status"] == "reject"
    assert "mean best-distance improvement -0.050 m is below required 0.050 m" in report["decision"]["blockers"]
    assert "mean final distance regressed by 0.150 m" in report["decision"]["blockers"]


def test_evaluate_promotion_rejects_hardcoded_candidate_even_when_metric_improves(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline" / "research_loop_summary.json"
    candidate_path = tmp_path / "candidate" / "research_loop_summary.json"
    _write_research_summary(
        baseline_path,
        run_id="baseline-run",
        summaries=[_run_summary(variant="qwen_baseline", best_distance_m=1.0, final_distance_m=1.0)],
    )
    _write_research_summary(
        candidate_path,
        run_id="candidate-run",
        summaries=[_run_summary(variant="scripted_candidate", best_distance_m=0.5, final_distance_m=0.6, no_hardcoded=False)],
        no_hardcoded=False,
    )

    report = evaluate_promotion(
        baseline_inputs=[baseline_path],
        candidate_inputs=[candidate_path],
        output_dir=tmp_path / "gate",
        decision_id="hardcoded",
    )

    assert report["decision"]["promote"] is False
    assert report["candidate"]["no_hardcoded_labels_or_colors_rate"] == 0.0
    assert "candidate violates no_hardcoded_labels_or_colors on at least one run" in report["decision"]["blockers"]


def test_load_nav_summaries_derives_generality_from_episode_prompt_audit(tmp_path: Path) -> None:
    summary = _run_summary(variant="qwen_baseline", best_distance_m=0.8, final_distance_m=0.9)
    summary.pop("no_hardcoded_labels_or_colors", None)
    path = tmp_path / "episode_summary.json"
    path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    records = load_nav_summaries([path])

    assert records[0]["no_hardcoded_labels_or_colors"] is True


def test_evaluate_promotion_filters_records_by_variant(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep" / "research_loop_summary.json"
    _write_research_summary(
        sweep_path,
        run_id="sweep-run",
        summaries=[
            _run_summary(variant="qwen_baseline", best_distance_m=1.0, final_distance_m=1.0),
            _run_summary(variant="qwen_bad_candidate", best_distance_m=1.5, final_distance_m=1.5),
            _run_summary(variant="qwen_good_candidate", best_distance_m=0.8, final_distance_m=0.9),
        ],
    )

    report = evaluate_promotion(
        baseline_inputs=[sweep_path],
        candidate_inputs=[sweep_path],
        baseline_variants=["qwen_baseline"],
        candidate_variants=["qwen_good_candidate"],
        output_dir=tmp_path / "gate",
        decision_id="filtered",
    )

    assert report["decision"]["promote"] is True
    assert report["baseline"]["variants"] == ["qwen_baseline"]
    assert report["candidate"]["variants"] == ["qwen_good_candidate"]
    assert report["filters"]["baseline_variants"] == ["qwen_baseline"]
    assert report["filters"]["candidate_variants"] == ["qwen_good_candidate"]


def test_cli_writes_report_and_can_fail_on_reject(tmp_path: Path, monkeypatch) -> None:
    baseline_path = tmp_path / "baseline" / "research_loop_summary.json"
    candidate_path = tmp_path / "candidate" / "research_loop_summary.json"
    _write_research_summary(
        baseline_path,
        run_id="baseline-run",
        summaries=[_run_summary(variant="qwen_baseline", best_distance_m=0.5, final_distance_m=0.5)],
    )
    _write_research_summary(
        candidate_path,
        run_id="candidate-run",
        summaries=[_run_summary(variant="qwen_candidate", best_distance_m=0.6, final_distance_m=0.6)],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-nav-promotion-gate",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--baseline-variant",
            "qwen_baseline",
            "--candidate-variant",
            "qwen_candidate",
            "--output-dir",
            str(tmp_path / "gate"),
            "--decision-id",
            "cli-reject",
            "--fail-on-reject",
        ],
    )

    assert main() == 2
    report = json.loads((tmp_path / "gate" / "promotion_decision.json").read_text(encoding="utf-8"))
    assert report["decision"]["status"] == "reject"
