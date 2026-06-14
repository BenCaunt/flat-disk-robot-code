from __future__ import annotations

import json

from flatdisk_sim.training_export import export_training_data_from_summaries


def _summary_fixture(
    tmp_path,
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
        "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        "success": success,
        "reason": "hidden_evaluator_goal_reached" if success else "max_steps_exhausted",
        "final_distance_m": final_distance_m,
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
                        "detections": [{"name": "legacy", "confidence": 1.0}],
                    },
                    "actor_action": {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"},
                    "actor_memory_update": {"belief": "open space ahead"},
                    "critic": {"verdict": "approve", "reason": "bounded"},
                    "executed_action": {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"},
                    "tool_result": {"motion_contact_sheet": "motion_frames/strip.jpg"},
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


def test_training_export_writes_policy_steps_without_hidden_object_metadata(tmp_path) -> None:
    summary = _summary_fixture(tmp_path, trial_id="trial_001", final_distance_m=1.25)

    manifest = export_training_data_from_summaries(
        [summary],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    assert manifest["episode_count"] == 1
    assert manifest["step_count"] == 1
    assert summary["training_policy_steps_jsonl"].endswith("policy_steps.jsonl")
    record = json.loads((tmp_path / "training" / "policy_steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["schema"] == "flatdisk.nav_policy_step.v1"
    assert record["policy_input"]["actor_prompt_text"].startswith("STATIC_HARNESS_CONTEXT")
    assert record["policy_input"]["observation"] == {
        "brightness_center": 0.5,
        "frame_seq": 1,
        "path": "frames/0001.jpg",
        "yaw_deg": 0.0,
    }
    assert record["policy_input_audit"]["passes_policy_input_audit"] is True
    assert record["evaluator_reward"]["post_action_score"] == {"distance_m": 1.25, "success": False}
    assert "nearest_target" not in json.dumps(record)
    assert "objectId" not in json.dumps(record)
    assert manifest["policy_sample_count"] == 1
    assert manifest["evaluator_label_count"] == 1


def test_policy_dataset_splits_model_inputs_from_evaluator_labels(tmp_path) -> None:
    summary = _summary_fixture(tmp_path, trial_id="trial_001", final_distance_m=1.25)

    manifest = export_training_data_from_summaries(
        [summary],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    dataset_manifest = json.loads((tmp_path / "training" / "policy_dataset_v1" / "dataset_manifest.json").read_text(encoding="utf-8"))
    sample = json.loads((tmp_path / "training" / "policy_dataset_v1" / "policy_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    label = json.loads((tmp_path / "training" / "policy_dataset_v1" / "evaluator_labels.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert manifest["policy_dataset_manifest_path"] == str(tmp_path / "training" / "policy_dataset_v1" / "dataset_manifest.json")
    assert dataset_manifest["join_key"] == "sample_id"
    assert dataset_manifest["privileged_label_file"] is True
    assert sample["schema"] == "flatdisk.nav_policy_sample.v1"
    assert label["schema"] == "flatdisk.nav_evaluator_label.v1"
    assert sample["sample_id"] == label["sample_id"]
    assert sample["sft"]["include_candidate"] is True
    assert sample["target"]["actor_equals_executed"] is True
    assert sample["target"]["executed_action_json"]["tool"] == "drive_straight"
    sample_text = json.dumps(sample).lower()
    for token in ("hidden_score", "nearest_target", "object_metadata", "target_pose", "distance_m", "objects", "pose", "scene", "success_radius", "detections"):
        assert token not in sample_text
    label_text = json.dumps(label).lower()
    for token in ("actor_prompt", "image_paths", "action_json", "memory_update_json", "tool_feedback", "observation"):
        assert token not in label_text
    assert label["reward"]["post_action_distance_m"] == 1.25
    assert label["grpo"]["eligible"] is False


def test_training_export_writes_policy_review_trace_for_agent_review(tmp_path) -> None:
    summary = _summary_fixture(tmp_path, trial_id="trial_001", final_distance_m=1.25)
    memory = summary["steps"][0]["harness_memory_record"]
    visual_servo_action = {
        "tool": "visual_servo_object",
        "args": {"prompt": "armchair", "duration_s": 1.5},
        "thought": "use a visible landmark",
    }
    memory["actor_action"] = visual_servo_action
    memory["executed_action"] = visual_servo_action
    memory["actor_grounding_audit"] = {
        "previous_visual_servo_box_matches_intended_object": False,
        "evidence": "box was on a different visible object",
        "next_prompt_should_change": True,
    }
    memory["tool_result"] = {
        "action": "visual_servo_object",
        "prompt": "armchair",
        "target_detected": False,
        "ever_detected": False,
        "grounding_stability": "no_detection",
        "detection_coverage_fraction": 0.0,
        "motion_contact_sheet": "motion_frames/strip.jpg",
        "grounding_audit_contact_sheet": "motion_frames/audit.jpg",
        "detections": [{"name": "legacy"}],
        "nearest_target": {"objectId": "hidden"},
    }

    manifest = export_training_data_from_summaries(
        [summary],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    trace = json.loads((tmp_path / "training" / "runs" / "trial_001" / "policy_review_trace.json").read_text(encoding="utf-8"))
    assert manifest["policy_review_trace_count"] == 1
    assert manifest["policy_review_traces_jsonl"] == str(tmp_path / "training" / "policy_review_traces.jsonl")
    assert summary["policy_review_trace_json"].endswith("policy_review_trace.json")
    assert trace["schema"] == "flatdisk.nav_policy_review_trace.v1"
    assert trace["steps"][0]["actor_action"]["tool"] == "visual_servo_object"
    assert trace["steps"][0]["actor_grounding_audit"]["next_prompt_should_change"] is True
    assert trace["steps"][0]["tool_result"]["target_detected"] is False
    assert "actor_reported_previous_grounding_mismatch" in trace["steps"][0]["review_flags"]
    assert "visual_servo_no_detection" in trace["steps"][0]["review_flags"]
    assert trace["policy_safety"]["forbidden_review_field_names_present"] == []
    trace_text = json.dumps(trace).lower()
    for token in ("nearest_target", "objectid", "detections", "distance_m"):
        assert token not in trace_text


def test_training_export_writes_rollout_groups_and_preference_pairs(tmp_path) -> None:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5, success=False)

    manifest = export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    assert manifest["rollout_group_count"] == 1
    assert manifest["trajectory_preference_count"] == 1
    group = json.loads((tmp_path / "training" / "rollout_groups.jsonl").read_text(encoding="utf-8").splitlines()[0])
    preference = json.loads((tmp_path / "training" / "trajectory_preferences.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert group["schema"] == "flatdisk.nav_rollout_group.v1"
    assert [rollout["record_id"] for rollout in group["rollouts"]] == ["trial_better", "trial_worse"]
    assert group["policy_context"]["reward_labels_excluded_from_policy_input"] is True
    assert group["rollouts"][0]["evaluator_reward"]["candidate_episode_reward"] > group["rollouts"][1]["evaluator_reward"]["candidate_episode_reward"]
    assert preference["schema"] == "flatdisk.nav_trajectory_preference_pair.v1"
    assert preference["chosen_rollout"]["record_id"] == "trial_better"
    assert preference["rejected_rollout"]["record_id"] == "trial_worse"
    assert preference["evaluator_preference"]["reward_margin"] > 0
    policy_context_text = json.dumps(preference["policy_context"]).lower()
    assert "distance_m" not in policy_context_text
    assert "nearest_target" not in policy_context_text
    assert "object_metadata" not in policy_context_text
