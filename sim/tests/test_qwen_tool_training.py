from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.qwen_tool_training import prepare_qwen_tool_training
from flatdisk_sim.training_export import export_training_data_from_summaries


def _summary_fixture(
    tmp_path: Path,
    *,
    trial_id: str,
    success: bool = True,
    actor_action: dict | None = None,
    executed_action: dict | None = None,
    actor_prompt_extra: str = "",
) -> dict:
    run_dir = tmp_path / trial_id
    policy_dir = run_dir / "policy"
    prompts_dir = policy_dir / "prompts"
    frames_dir = policy_dir / "frames"
    prompts_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    (frames_dir / "0001.jpg").write_bytes(b"fake image")
    (prompts_dir / "000_actor.txt").write_text(
        "STATIC_HARNESS_CONTEXT\nDYNAMIC_TASK_STATE\nUse the latest RGB frame and return one JSON action.\n"
        + actor_prompt_extra,
        encoding="utf-8",
    )
    actor_action = actor_action or {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"}
    executed_action = executed_action or {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"}
    (policy_dir / "harness_events.jsonl").write_text(
        json.dumps(
            {
                "event": "actor",
                "step": 0,
                "output": json.dumps(
                    {
                        "thought": actor_action.get("thought", ""),
                        "action": {"tool": actor_action["tool"], "args": actor_action.get("args", {})},
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
        "final_distance_m": 0.05 if success else 1.25,
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
                    "critic": {"verdict": "approve", "reason": "bounded", "replacement": None},
                    "executed_action": executed_action,
                    "tool_result": {"ok": True},
                    "saved_frames": [],
                },
                "hidden_score_for_evaluator_only": {
                    "success": success,
                    "distance_m": 0.05 if success else 1.25,
                    "nearest_target": {"objectType": "Sofa", "objectId": "hidden"},
                },
            }
        ],
    }


def test_prepare_qwen_tool_training_writes_policy_only_sft_messages(tmp_path: Path) -> None:
    export = export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_001")],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_tool_training(
        Path(export["policy_dataset_dir"]),
        output_dir=tmp_path / "qwen_training",
    )

    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 0
    assert manifest["action_preference_count"] == 0
    assert manifest["dpo_preference_count"] == 0
    record = json.loads((tmp_path / "qwen_training" / "qwen_sft_messages.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["schema"] == "flatdisk.qwen_tool_sft_sample.v1"
    assert record["source_policy_step_id"] == "trial_001_step_000"
    assert record["sft_weight"] == 1.0
    assert record["training_weight"] == 1.0
    assert record["audit"]["filter_reasons"] == []
    assert record["audit"]["image_count"] == 1
    assert record["audit"]["privileged_scan_passed"] is True
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][0]["content"][0]["type"] == "image"
    assert record["messages"][1]["role"] == "assistant"
    assistant = json.loads(record["messages"][1]["content"])
    assert assistant["action"]["tool"] == "drive_straight"
    assert assistant["memory_update"] == {"belief": "open space ahead"}
    messages_text = json.dumps(record["messages"]).lower()
    for token in ("hidden_score", "nearest_target", "object_metadata", "target_pose", "distance_m", "candidate_step_reward", "final_distance_m"):
        assert token not in messages_text


def test_prepare_qwen_tool_training_rejects_replaced_actor_actions(tmp_path: Path) -> None:
    actor = {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"}
    executed = {"tool": "wait", "args": {"duration_s": 0.2}, "thought": "replacement"}
    export = export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_001", actor_action=actor, executed_action=executed)],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_tool_training(
        Path(export["policy_dataset_dir"]),
        output_dir=tmp_path / "qwen_training",
    )

    assert manifest["accepted_count"] == 0
    assert manifest["rejected_count"] == 1
    assert manifest["action_preference_count"] == 1
    assert manifest["dpo_preference_count"] == 1
    rejected = json.loads((tmp_path / "qwen_training" / "rejected_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "actor_action_differs_from_executed_action" in rejected["reject_reasons"]
    preference = json.loads((tmp_path / "qwen_training" / "qwen_action_preferences.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert preference["schema"] == "flatdisk.qwen_tool_action_preference.v1"
    assert preference["preference_source"] == "critic_or_harness_replacement"
    assert preference["preference_type"] == "executed_action_preferred_over_rejected_actor_action"
    assert preference["prompt_messages"][0]["role"] == "user"
    assert preference["chosen_assistant_target_json"]["action"]["tool"] == "wait"
    assert preference["rejected_assistant_target_json"]["action"]["tool"] == "drive_straight"
    assert preference["audit"]["preference_labels_excluded_from_messages"] is True
    assert preference["audit"]["evaluator_reward_excluded_from_messages"] is True
    messages_text = json.dumps(preference["prompt_messages"]).lower()
    for token in ("hidden_score", "nearest_target", "object_metadata", "target_pose", "distance_m", "candidate_step_reward", "final_distance_m"):
        assert token not in messages_text
    dpo = json.loads((tmp_path / "qwen_training" / "qwen_dpo_messages.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert dpo["schema"] == "flatdisk.qwen_tool_dpo_sample.v1"
    assert dpo["source_preference_sample_id"] == preference["sample_id"]
    assert dpo["prompt"] == preference["prompt_messages"]
    assert dpo["chosen"][0]["role"] == "assistant"
    assert dpo["rejected"][0]["role"] == "assistant"
    assert json.loads(dpo["chosen"][0]["content"])["action"]["tool"] == "wait"
    assert json.loads(dpo["rejected"][0]["content"])["action"]["tool"] == "drive_straight"
    assert dpo["images"] == preference["image_paths"]
    assert dpo["image_paths"] == preference["image_paths"]
    assert all(Path(path).exists() for path in dpo["images"])
    assert dpo["audit"]["dpo_columns"] == ["prompt", "chosen", "rejected", "images"]
    assert dpo["audit"]["evaluator_reward_excluded_from_messages"] is True
    dpo_messages_text = json.dumps([dpo["prompt"], dpo["chosen"], dpo["rejected"]]).lower()
    for token in ("hidden_score", "nearest_target", "object_metadata", "target_pose", "distance_m", "candidate_step_reward", "final_distance_m"):
        assert token not in dpo_messages_text


def test_prepare_qwen_tool_training_allows_safety_contract_terms(tmp_path: Path) -> None:
    export = export_training_data_from_summaries(
        [
            _summary_fixture(
                tmp_path,
                trial_id="trial_001",
                actor_prompt_extra=(
                    "Do not use THOR object metadata. Treat last_detection and target_detected "
                    "as model-facing tool feedback, not hidden evaluator state. "
                    "A debug_overlay_contact_sheet is detector evidence, not evaluator state.\n"
                ),
            )
        ],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_tool_training(
        Path(export["policy_dataset_dir"]),
        output_dir=tmp_path / "qwen_training",
    )

    assert manifest["accepted_count"] == 1
    audit = json.loads((tmp_path / "qwen_training" / "training_audit.json").read_text(encoding="utf-8"))
    assert "thor" not in audit["forbidden_model_tokens"]
    assert "last_detection" not in audit["forbidden_model_tokens"]
    assert "target_detected" not in audit["forbidden_model_tokens"]
    assert "debug_overlay" not in audit["forbidden_model_tokens"]


def test_prepare_qwen_tool_training_relocates_copied_runpod_manifest_paths(tmp_path: Path) -> None:
    export = export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_001")],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )
    dataset_dir = Path(export["policy_dataset_dir"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_dir"] = "/workspace/outputs/copied/training_export/policy_dataset_v1"
    manifest["policy_samples_jsonl"] = "/workspace/outputs/copied/training_export/policy_dataset_v1/policy_samples.jsonl"
    manifest["evaluator_labels_jsonl"] = "/workspace/outputs/copied/training_export/policy_dataset_v1/evaluator_labels.jsonl"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = prepare_qwen_tool_training(
        dataset_dir,
        output_dir=tmp_path / "qwen_training",
    )

    assert result["accepted_count"] == 1
    assert result["source_policy_dataset_dir"] == str(dataset_dir)
    assert result["source_policy_samples_jsonl"] == str(dataset_dir / "policy_samples.jsonl")
    assert result["source_evaluator_labels_jsonl"] == str(dataset_dir / "evaluator_labels.jsonl")


def test_prepare_qwen_tool_training_relocates_copied_runpod_image_paths(tmp_path: Path) -> None:
    export = export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_001")],
        output_dir=tmp_path / "training",
        experiment_id="exp",
        run_id="run",
    )
    dataset_dir = Path(export["policy_dataset_dir"])
    local_frame = tmp_path / "trials" / "trial_001" / "policy" / "frames" / "0001.jpg"
    local_frame.parent.mkdir(parents=True)
    local_frame.write_bytes(b"copied runpod image")
    remote_frame = "/workspace/outputs/copied/trials/trial_001/policy/frames/0001.jpg"

    samples_path = dataset_dir / "policy_samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines()]
    samples[0]["policy_input"]["image_paths"] = [remote_frame]
    samples_path.write_text("\n".join(json.dumps(sample, sort_keys=True) for sample in samples) + "\n", encoding="utf-8")

    result = prepare_qwen_tool_training(
        dataset_dir,
        output_dir=tmp_path / "qwen_training",
    )

    assert result["accepted_count"] == 1
    record = json.loads((tmp_path / "qwen_training" / "qwen_sft_messages.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["image_paths"] == [str(local_frame)]
    assert record["audit"]["missing_images"] == []
