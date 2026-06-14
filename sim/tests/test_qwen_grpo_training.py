from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.qwen_grpo_training import main, prepare_qwen_grpo_training
from flatdisk_sim.training_export import export_training_data_from_summaries
from flatdisk_sim.training_readiness import analyze_training_readiness


def _summary_fixture(
    tmp_path: Path,
    *,
    trial_id: str,
    final_distance_m: float,
    success: bool = False,
    actor_action: dict | None = None,
    executed_action: dict | None = None,
) -> dict:
    run_dir = tmp_path / trial_id
    policy_dir = run_dir / "policy"
    prompts_dir = policy_dir / "prompts"
    frames_dir = policy_dir / "frames"
    prompts_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    (frames_dir / "0001.jpg").write_bytes(b"fake image")
    (prompts_dir / "000_actor.txt").write_text("STATIC_HARNESS_CONTEXT\nDYNAMIC_TASK_STATE\n", encoding="utf-8")
    actor_action = actor_action or {
        "tool": "drive_straight",
        "args": {"power_percent": 20, "duration_s": 0.5},
        "thought": "move",
    }
    executed_action = executed_action or actor_action
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
                    "actor_action": actor_action,
                    "actor_memory_update": {"belief": "open space ahead"},
                    "critic": {"verdict": "approve", "reason": "bounded"},
                    "executed_action": executed_action,
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


def test_prepare_qwen_grpo_training_writes_trainable_rollout_groups(tmp_path: Path) -> None:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )

    assert manifest["schema"] == "flatdisk.qwen_grpo_training_manifest.v1"
    assert manifest["status"] == "ready"
    assert manifest["group_count"] == 1
    assert manifest["trainable_group_count"] == 1
    assert manifest["candidate_count"] == 2
    assert manifest["trainable_candidate_count"] == 2
    assert manifest["step_sample_count"] == 2
    assert manifest["ppo_step_sample_count"] == 2
    assert manifest["missing_image_count"] == 0
    assert manifest["forbidden_qwen_message_token_hits"] == {}
    groups_path = tmp_path / "run" / "qwen_grpo_training" / "qwen_grpo_rollout_groups.jsonl"
    group = json.loads(groups_path.read_text(encoding="utf-8").splitlines()[0])
    assert group["schema"] == "flatdisk.qwen_grpo_rollout_group.v1"
    assert group["trainable_candidate_count"] == 2
    assert group["candidates"][0]["trainable"] is True
    assert group["candidates"][0]["evaluator_reward"]["privileged"] is True
    assert group["candidates"][0]["step_samples"][0]["prompt_messages"][0]["role"] == "user"
    assert group["candidates"][0]["step_samples"][0]["assistant_target_json"]["action"]["tool"] == "drive_straight"
    prompt_text = json.dumps(group["candidates"][0]["step_samples"][0]["prompt_messages"]).lower()
    assert "distance_m" not in prompt_text
    assert "nearest_target" not in prompt_text
    ppo_step = json.loads(
        (tmp_path / "run" / "qwen_grpo_training" / "qwen_ppo_step_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert ppo_step["schema"] == "flatdisk.qwen_ppo_step_sample.v1"
    assert ppo_step["evaluator_reward"]["privileged"] is True
    assert ppo_step["audit"]["reward_excluded_from_messages"] is True

    readiness = analyze_training_readiness(
        [tmp_path / "run"],
        output_dir=tmp_path / "readiness",
        analysis_id="grpo-ready",
        experiment_id="exp",
    )
    assert readiness["aggregate"]["qwen_grpo_training_manifest_count"] == 1
    assert readiness["aggregate"]["qwen_grpo_trainable_group_count"] == 1
    assert readiness["aggregate"]["qwen_grpo_trainable_candidate_count"] == 2
    assert readiness["aggregate"]["qwen_ppo_step_sample_count"] == 2
    assert readiness["readiness"]["grpo_ready"] is True


def test_prepare_qwen_grpo_training_merges_multiple_exports_by_episode_prompt(tmp_path: Path) -> None:
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)],
        output_dir=tmp_path / "run_a" / "training_export",
        experiment_id="exp",
        run_id="run-a",
    )
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)],
        output_dir=tmp_path / "run_b" / "training_export",
        experiment_id="exp",
        run_id="run-b",
    )

    manifest = prepare_qwen_grpo_training(
        [tmp_path / "run_a", tmp_path / "run_b"],
        output_dir=tmp_path / "merged" / "qwen_grpo_training",
    )

    assert manifest["source_mode"] == "merged_training_exports"
    assert len(manifest["source_training_manifests"]) == 2
    assert manifest["source_rollout_group_count"] == 2
    assert manifest["group_count"] == 1
    assert manifest["candidate_count"] == 2
    assert manifest["trainable_group_count"] == 1
    assert manifest["ppo_step_sample_count"] == 2


def test_prepare_qwen_grpo_training_requires_actor_equal_executed(tmp_path: Path) -> None:
    actor = {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"}
    replaced = {"tool": "wait", "args": {"duration_s": 0.2}, "thought": "replacement"}
    better = _summary_fixture(
        tmp_path,
        trial_id="trial_better",
        final_distance_m=0.2,
        success=True,
        actor_action=actor,
        executed_action=replaced,
    )
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )

    assert manifest["status"] == "not_ready"
    assert manifest["trainable_group_count"] == 0
    assert manifest["trainable_candidate_count"] == 1
    assert any("at least two trainable" in blocker for blocker in manifest["blockers"])
    group = json.loads(
        (tmp_path / "run" / "qwen_grpo_training" / "qwen_grpo_rollout_groups.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    blocked = next(candidate for candidate in group["candidates"] if candidate["rollout_id"] == "trial_better")
    assert any("actor action replaced" in blocker for blocker in blocked["blockers"])


def test_prepare_qwen_grpo_training_can_allow_missing_images(tmp_path: Path) -> None:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    for image_path in tmp_path.glob("trial_*/policy/frames/0001.jpg"):
        image_path.unlink()

    manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
        require_existing_images=False,
    )

    assert manifest["status"] == "ready"
    assert manifest["missing_image_count"] == 2
    assert manifest["trainable_group_count"] == 1


def test_prepare_qwen_grpo_training_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_single", final_distance_m=1.2)],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-prepare-qwen-grpo-training",
            "--input",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "run" / "qwen_grpo_training"),
            "--fail-on-not-ready",
        ],
    )

    assert main() == 2
