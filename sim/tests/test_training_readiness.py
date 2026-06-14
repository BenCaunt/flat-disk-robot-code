from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.training_export import export_training_data_from_summaries
from flatdisk_sim.training_readiness import analyze_training_readiness, main


def _summary_fixture(
    tmp_path: Path,
    *,
    trial_id: str,
    final_distance_m: float,
    success: bool = False,
) -> dict:
    run_dir = tmp_path / trial_id
    policy_dir = run_dir / "policy"
    prompts_dir = policy_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "000_actor.txt").write_text("STATIC_HARNESS_CONTEXT\nDYNAMIC_TASK_STATE\n", encoding="utf-8")
    (policy_dir / "harness_events.jsonl").write_text(
        json.dumps(
            {
                "event": "actor",
                "step": 0,
                "output": json.dumps(
                    {
                        "thought": "move",
                        "action": {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}},
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "trial_id": trial_id,
        "slot_id": "slot_001",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "episode": "living_room_sofa",
        "scene": "FloorPlan201",
        "prompt": "Drive to the sofa.",
        "runner": "qwen",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "success": success,
        "reason": "hidden_evaluator_goal_reached" if success else "max_steps_exhausted",
        "final_distance_m": final_distance_m,
        "best_distance_m": final_distance_m,
        "best_distance_step": 0,
        "success_radius_m": 0.1,
        "step_count": 1,
        "steps": [
            {
                "step": 0,
                "harness_memory_record": {
                    "step": 0,
                    "goal": "Drive to the sofa.",
                    "observation": {
                        "path": "frames/0001.jpg",
                        "yaw_deg": 0.0,
                        "frame_seq": 1,
                        "brightness_center": 0.5,
                    },
                    "actor_action": {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"},
                    "actor_memory_update": {"belief": "open space ahead"},
                    "critic": {"verdict": "approve", "reason": "bounded"},
                    "executed_action": {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"},
                    "tool_result": {"ok": True},
                    "saved_frames": [],
                },
                "hidden_score_for_evaluator_only": {
                    "success": success,
                    "distance_m": final_distance_m,
                    "nearest_target": {"objectType": "Sofa", "objectId": "hidden"},
                },
            }
        ],
    }


def test_training_readiness_marks_sft_ppo_and_grpo_ready_for_ranked_rollouts(tmp_path: Path) -> None:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5, success=False)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )

    report = analyze_training_readiness(
        [tmp_path / "run"],
        output_dir=tmp_path / "readiness",
        analysis_id="ready",
        experiment_id="exp",
    )

    assert report["readiness"]["status"] == "ready"
    assert report["readiness"]["sft_ready"] is True
    assert report["readiness"]["ppo_ready"] is True
    assert report["readiness"]["grpo_ready"] is True
    assert report["aggregate"]["policy_sample_count"] == 2
    assert report["aggregate"]["trajectory_preference_count"] == 1
    assert (tmp_path / "readiness" / "training_readiness.json").exists()
    assert (tmp_path / "readiness" / "training_readiness.md").exists()
    ops = json.loads((tmp_path / "readiness" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert ops[0]["name"] == "AgentNote/ready"
    assert "training-readiness" in ops[0]["data"]["tags"]


def test_training_readiness_relocates_runpod_absolute_manifest_paths(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path, trial_id="trial_001", final_distance_m=0.2, success=True)
    export = export_training_data_from_summaries(
        [summary],
        output_dir=tmp_path / "copied_artifact" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    manifest_path = Path(export["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "policy_dataset_manifest_path",
        "policy_samples_jsonl",
        "evaluator_labels_jsonl",
        "rollout_groups_jsonl",
        "trajectory_preferences_jsonl",
        "policy_review_traces_jsonl",
    ):
        local_path = Path(str(manifest[key]))
        manifest[key] = str(Path("/workspace/outputs/run/training_export") / local_path.relative_to(manifest_path.parent))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = analyze_training_readiness(
        [tmp_path / "copied_artifact"],
        output_dir=tmp_path / "readiness",
        analysis_id="relocated",
    )

    assert report["aggregate"]["policy_sample_count"] == 1
    assert report["aggregate"]["evaluator_label_count"] == 1
    assert report["aggregate"]["missing_required_artifacts"] == []
    assert report["readiness"]["sft_ready"] is True


def test_training_readiness_blocks_forbidden_policy_sample_tokens(tmp_path: Path) -> None:
    summary = _summary_fixture(tmp_path, trial_id="trial_001", final_distance_m=1.25)
    export = export_training_data_from_summaries(
        [summary],
        output_dir=tmp_path / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    samples_path = Path(export["policy_samples_jsonl"])
    sample = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
    sample["policy_input"]["observation"]["distance_m"] = 1.23
    samples_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    report = analyze_training_readiness(
        [tmp_path / "training_export"],
        output_dir=tmp_path / "readiness",
        analysis_id="blocked",
    )

    assert report["readiness"]["status"] == "not_ready"
    assert report["readiness"]["sft_ready"] is False
    assert "distance_m" in report["aggregate"]["forbidden_policy_sample_token_hits"]
    assert any("forbidden privileged token" in blocker for blocker in report["readiness"]["blockers"])


def test_training_readiness_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "flatdisk.nav_training_export.v1",
                "experiment_id": "exp",
                "research_run_id": "run",
                "episode_count": 0,
                "step_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-nav-training-readiness",
            "--input",
            str(manifest),
            "--output-dir",
            str(tmp_path / "readiness"),
            "--analysis-id",
            "cli-blocked",
            "--fail-on-not-ready",
        ],
    )

    assert main() == 2
    report = json.loads((tmp_path / "readiness" / "training_readiness.json").read_text(encoding="utf-8"))
    assert report["readiness"]["status"] == "not_ready"
