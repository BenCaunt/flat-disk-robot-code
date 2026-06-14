from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.failure_analysis import analyze_failure_traces


def _trace(record_id: str = "trial_failed") -> dict:
    return {
        "schema": "flatdisk.nav_policy_review_trace.v1",
        "record_id": record_id,
        "task": {"goal": "Drive to the target.", "episode": "bathroom_toilet"},
        "run": {"reason": "harness_mode_complete_before_hidden_success"},
        "step_count": 4,
        "policy_safety": {
            "model_facing_artifact": True,
            "privileged_evaluator_fields_excluded": True,
            "hidden_target_metadata_excluded": True,
            "forbidden_review_field_names_present": [],
        },
        "steps": [
            {
                "step": 0,
                "actor_action": {"tool": "visual_servo_object", "args": {"prompt": "visible target"}},
                "executed_action": {"tool": "visual_servo_object", "args": {"prompt": "visible target"}},
                "actor_action_replaced": False,
                "review_flags": ["visual_servo_grounding_not_stable", "visual_servo_sparse_detection_coverage"],
                "tool_result": {
                    "action": "visual_servo_object",
                    "prompt": "visible target",
                    "grounding_stability": "sparse_detection_coverage",
                    "target_detected": True,
                    "moved": True,
                },
            },
            {
                "step": 1,
                "actor_action": {"tool": "visual_servo_object", "args": {"prompt": "visible target"}},
                "executed_action": {"tool": "wait", "args": {"duration_s": 0.2}},
                "actor_action_replaced": True,
                "review_flags": [
                    "actor_action_replaced",
                    "actor_reported_previous_grounding_mismatch",
                    "actor_repeated_servo_prompt_after_grounding_audit_requested_change",
                ],
                "tool_result": {"duration_s": 0.2},
            },
            {
                "step": 2,
                "actor_action": {"tool": "visual_servo_object", "args": {"prompt": "more specific visible target"}},
                "executed_action": {"tool": "visual_servo_object", "args": {"prompt": "more specific visible target"}},
                "actor_action_replaced": False,
                "review_flags": ["visual_servo_grounding_not_stable", "visual_servo_no_detection"],
                "tool_result": {
                    "action": "visual_servo_object",
                    "prompt": "more specific visible target",
                    "grounding_stability": "no_detection",
                    "target_detected": False,
                    "moved": False,
                },
            },
            {
                "step": 3,
                "actor_action": {"tool": "turn_by_angle", "args": {"degrees": -15}},
                "executed_action": {"tool": "turn_by_angle", "args": {"degrees": -15}},
                "actor_action_replaced": False,
                "review_flags": ["actor_reported_previous_grounding_mismatch"],
                "tool_result": {"action": "turn_to_angle"},
            },
        ],
    }


def _write_trace(path: Path, *, record_id: str = "trial_failed") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_trace(record_id)) + "\n", encoding="utf-8")


def test_analyze_failure_traces_writes_report_and_warmhub_ops(tmp_path: Path) -> None:
    trace_path = tmp_path / "run" / "training_export" / "policy_review_traces.jsonl"
    _write_trace(trace_path)

    report = analyze_failure_traces(
        [tmp_path / "run"],
        output_dir=tmp_path / "analysis",
        experiment_id="exp",
        analysis_id="analysis-001",
        author="tester",
    )

    assert report["schema"] == "flatdisk.nav_failure_analysis.v1"
    assert report["trace_count"] == 1
    assert report["aggregate"]["review_flag_counts"]["actor_action_replaced"] == 1
    assert report["aggregate"]["review_flag_counts"]["visual_servo_no_detection"] == 1
    assert report["runs"][0]["actor_action_replaced_steps"] == [1]
    assert report["runs"][0]["policy_safety"]["forbidden_review_field_names_present"] == []
    assert {item["id"] for item in report["recommendations"]} >= {
        "grounding_audit_must_bind_next_action",
        "treat_unstable_servo_as_weak_control",
        "train_or_prompt_against_guard_replacements",
    }
    assert report["candidate_variant"]["name"] == "qwen_grounding_recovery_v1"
    assert report["candidate_variant"]["no_static_object_or_color_examples"] is True
    assert (tmp_path / "analysis" / "failure_analysis.json").exists()
    assert (tmp_path / "analysis" / "failure_analysis.md").exists()
    ops = json.loads((tmp_path / "analysis" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert ops[0]["name"] == "AgentNote/analysis-001"
    assert ops[0]["about"] == "NavExperiment/exp"
    assert "qwen_grounding_recovery_v1" in ops[0]["data"]["note"]


def test_analyze_failure_traces_keeps_multiline_jsonl_runs_separate(tmp_path: Path) -> None:
    trace_path = tmp_path / "run" / "training_export" / "policy_review_traces.jsonl"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(
        "\n".join(json.dumps(_trace(record_id)) for record_id in ("trial-a", "trial-b")) + "\n",
        encoding="utf-8",
    )

    report = analyze_failure_traces(
        [tmp_path / "run"],
        output_dir=tmp_path / "analysis",
        analysis_id="analysis-001",
    )

    assert report["trace_count"] == 2
    assert report["run_count"] == 2
    assert [run["record_id"] for run in report["runs"]] == ["trial-a", "trial-b"]


def test_analyze_failure_traces_deduplicates_aggregate_and_per_run_trace(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    aggregate_path = run_dir / "training_export" / "policy_review_traces.jsonl"
    individual_path = run_dir / "training_export" / "runs" / "trial-a" / "policy_review_trace.json"
    aggregate_path.parent.mkdir(parents=True)
    individual_path.parent.mkdir(parents=True)
    trace = _trace("trial-a")
    aggregate_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    individual_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    report = analyze_failure_traces(
        [run_dir],
        output_dir=tmp_path / "analysis",
        analysis_id="analysis-001",
    )

    assert report["input_trace_record_count"] == 2
    assert report["trace_count"] == 1
    assert report["duplicate_trace_count"] == 1
    assert report["duplicate_trace_records"][0]["identity"] == "record_id:trial-a"
    assert report["run_count"] == 1
    ops = json.loads((tmp_path / "analysis" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert "Deduped 2 input trace record(s) to 1 unique trace(s)." in ops[0]["data"]["note"]
