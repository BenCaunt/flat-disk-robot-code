from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.harness_run_summary import summarize_run


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_summary_warns_on_camera_sequence_drop_and_turn_timeout(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "harness_events.jsonl", [{"event": "user_goal", "goal": "drive to the chair"}])
    _write_jsonl(
        tmp_path / "memory.jsonl",
        [
            {
                "step": 0,
                "observation": {"frame_seq": 1000, "yaw_deg": 0.0, "path": "frames/a.jpg"},
                "executed_action": {"tool": "turn_by_angle", "args": {"degrees": 30}},
                "tool_result": {
                    "action": "turn_to_angle",
                    "started_yaw_deg": 0.0,
                    "final_yaw_deg": 0.0,
                    "target_yaw_deg": 30.0,
                    "heading_error_deg": 30.0,
                    "timed_out": True,
                },
            },
            {
                "step": 1,
                "observation": {"frame_seq": 20, "yaw_deg": 0.0, "path": "frames/b.jpg"},
                "executed_action": {"tool": "wait", "args": {}},
                "tool_result": {"action": "wait"},
            },
        ],
    )

    summary = summarize_run(tmp_path)

    assert "camera frame sequence dropped from 1000 to 20" in summary
    assert "turn tool timed out" in summary
