from __future__ import annotations

from pathlib import Path

import pytest

from flatdisk_sim.agent_candidates.fast_wall_clock import (
    FastWallClockActor,
    FastWallClockCritic,
    action_json,
    choose_fast_action,
)
from flatdisk_sim.llm_harness import (
    HarnessConfig,
    HarnessSession,
    build_actor_prompt,
    parse_actor_action,
)
from fakes import FakeHarnessTools


@pytest.mark.parametrize(
    "episode",
    [
        ("fast_living_room_sofa", "living_room", "Drive to the sofa in the living room."),
        ("fast_bedroom_bed", "bedroom", "Drive to the bed in the bedroom."),
        ("fast_bathroom_toilet", "bathroom", "Drive to the toilet in the bathroom."),
    ],
)
def test_fast_candidate_solves_fake_harness_in_fewer_steps(tmp_path: Path, episode: tuple[str, str, str]) -> None:
    name, environment, prompt = episode
    run_dir = tmp_path / name
    tools = FakeHarnessTools(run_dir=run_dir, environment=environment)
    session = HarnessSession(
        config=HarnessConfig(run_dir=run_dir, max_steps=6),
        tools=tools,
        actor=FastWallClockActor(),
        critic=FastWallClockCritic(),
    )

    try:
        session.start_goal(prompt)
        records = []
        while session.mode == "auto":
            record = session.run_auto_step()
            if record is None:
                break
            records.append(record)
    finally:
        hidden_score = tools.hidden_score()
        session.close()

    executed_tools = [record["executed_action"]["tool"] for record in records]
    assert hidden_score["success"] is True
    assert len(records) <= 6
    assert "observe" not in executed_tools
    assert "wait" not in executed_tools
    assert all(record["critic"]["verdict"] != "reject" for record in records)


def test_fast_actor_prefers_motion_over_redundant_observe(tmp_path: Path) -> None:
    tools = FakeHarnessTools(run_dir=tmp_path, environment="living_room")
    observation = tools.observe(label="initial").summary()
    prompt = build_actor_prompt(
        goal="Drive to the sofa in the living room.",
        mode="auto",
        step=0,
        memory_path=Path("memory.jsonl"),
        observation=observation,
        recent_memory=[],
    )
    actor = FastWallClockActor()

    action = parse_actor_action(actor.run(prompt, role="actor", image_paths=[Path(observation["path"])]))

    assert action.tool in {"drive_straight", "turn_by_angle"}
    assert action.tool != "observe"


def test_fast_action_uses_only_policy_surface_fields() -> None:
    action = choose_fast_action(
        {
            "goal": "Drive to the sofa in the living room.",
            "observation": {
                "yaw_deg": 0.0,
                "brightness_center": 0.45,
                "detections": [{"confidence": 0.4, "area_fraction": 0.02, "center_offset": 0.0}],
                "hidden_distance_m": 0.01,
                "target_pose": {"x": 1.0},
            },
            "recent_memory": [],
        },
        image_paths=[],
    )

    assert action_json(action)["tool"] == "drive_straight"


def test_fast_action_stops_once_shared_safety_gate_can_accept_stop() -> None:
    memory = [
        {
            "executed_action": {"tool": "drive_straight", "args": {"power_percent": 24.0, "duration_s": 0.9}},
            "observation": {"brightness_center": 0.40 + index * 0.02, "detections": []},
        }
        for index in range(6)
    ]

    action = choose_fast_action(
        {
            "goal": "Drive to the sofa in the living room.",
            "observation": {
                "yaw_deg": 0.0,
                "brightness_center": 0.58,
                "detections": [{"confidence": 0.5, "area_fraction": 0.03, "center_offset": 0.0}],
            },
            "recent_memory": memory,
        },
        image_paths=[],
    )

    assert action_json(action)["tool"] == "stop"
