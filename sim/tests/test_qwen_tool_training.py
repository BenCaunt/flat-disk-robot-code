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
) -> dict:
    run_dir = tmp_path / trial_id
    policy_dir = run_dir / "policy"
    prompts_dir = policy_dir / "prompts"
    frames_dir = policy_dir / "frames"
    prompts_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    (frames_dir / "0001.jpg").write_bytes(b"fake image")
    (prompts_dir / "000_actor.txt").write_text(
        "STATIC_HARNESS_CONTEXT\nDYNAMIC_TASK_STATE\nUse the latest RGB frame and return one JSON action.\n",
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
    rejected = json.loads((tmp_path / "qwen_training" / "rejected_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "actor_action_differs_from_executed_action" in rejected["reject_reasons"]
