from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from flatdisk_sim.agent_candidates.memory_recovery import (
    FORBIDDEN_HISTORY_FIELDS,
    POLICY_INPUT_ALLOWLIST,
    MemoryRecoveryConfig,
    MemoryRecoveryPolicy,
    sanitize_history,
)
from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.vision import analyze_image


def test_memory_candidate_sanitizes_privileged_history_fields() -> None:
    history = [
        {
            "action": "drive_straight",
            "power_percent": 23.0,
            "duration_s": 0.8,
            "reason": "allowed",
            "hidden_score_for_evaluator_only": {"distance_m": 0.4},
            "pose": {"x": 1.0},
            "scene": "FloorPlan999",
            "nearest_target": "Sofa|1",
        }
    ]

    safe = sanitize_history(history)

    assert safe == [{"action": "drive_straight", "power_percent": 23.0, "duration_s": 0.8, "reason": "allowed"}]
    assert FORBIDDEN_HISTORY_FIELDS.isdisjoint(safe[0])
    assert "current RGB camera frame" in POLICY_INPUT_ALLOWLIST
    assert "IMU yaw" in POLICY_INPUT_ALLOWLIST


def test_repeated_stale_drive_triggers_escape_turn(tmp_path: Path) -> None:
    policy = MemoryRecoveryPolicy()
    obs = _plain_observation(tmp_path, "plain_a.jpg", yaw_deg=0.0)
    history = [_drive_record(), _drive_record(), _drive_record()]

    policy.choose_action(obs, prompt="Drive to the sofa in the living room.", history=[])
    action = policy.choose_action(obs, prompt="Drive to the sofa in the living room.", history=history)

    assert action.action == "turn_by_angle"
    assert abs(action.degrees) >= 30.0
    assert "stale_repeated_drive_escape" in action.reason


def test_turn_oscillation_continues_last_scan_direction(tmp_path: Path) -> None:
    policy = MemoryRecoveryPolicy()
    obs = _plain_observation(tmp_path, "plain_b.jpg", yaw_deg=0.0)
    history = [
        {"action": "turn_by_angle", "degrees": 28.0, "reason": "scan right"},
        {"action": "turn_by_angle", "degrees": -28.0, "reason": "scan left"},
        {"action": "turn_by_angle", "degrees": 28.0, "reason": "scan right"},
        {"action": "turn_by_angle", "degrees": -28.0, "reason": "scan left"},
    ]

    action = policy.choose_action(obs, prompt="Drive to the sofa in the living room.", history=history)

    assert action.action == "turn_by_angle"
    assert action.degrees < 0.0
    assert "turn_oscillation" in action.reason


def test_policy_returns_to_remembered_camera_imu_sighting(tmp_path: Path) -> None:
    policy = MemoryRecoveryPolicy()
    sighting = _toilet_observation(tmp_path, "toilet_left.jpg", yaw_deg=90.0, x0=18, x1=58)
    current = _plain_observation(tmp_path, "plain_c.jpg", yaw_deg=20.0)
    history = [_drive_record(), _drive_record(), _drive_record()]

    first_action = policy.choose_action(sighting, prompt="Drive to the toilet in the bathroom.", history=[])
    recovery = policy.choose_action(current, prompt="Drive to the toilet in the bathroom.", history=history)

    assert first_action.action == "turn_by_angle"
    assert recovery.action == "turn_by_angle"
    assert recovery.degrees > 0.0
    assert "return_to_best_sighting" in recovery.reason
    memory = policy.public_memory()
    assert memory
    assert set(memory[0]).isdisjoint(FORBIDDEN_HISTORY_FIELDS)
    assert "yaw_deg" in memory[0]
    assert "score" in memory[0]


def test_scan_loop_uses_open_center_probe_without_encoders(tmp_path: Path) -> None:
    policy = MemoryRecoveryPolicy(config=MemoryRecoveryConfig(min_probe_turns=3))
    obs = _plain_observation(tmp_path, "bright_floor.jpg", yaw_deg=0.0, color=(170, 170, 170))
    history = [
        {"action": "turn_by_angle", "degrees": 32.0, "reason": "scan"},
        {"action": "turn_by_angle", "degrees": 32.0, "reason": "scan"},
        {"action": "turn_by_angle", "degrees": 32.0, "reason": "scan"},
    ]

    action = policy.choose_action(obs, prompt="Drive to the table.", history=history)

    assert action.action == "drive_straight"
    assert action.power_percent == 21.0
    assert action.duration_s == 0.62
    assert "open_center_probe_after_scan_loop" in action.reason


def _drive_record() -> dict[str, Any]:
    return {"action": "drive_straight", "power_percent": 23.0, "duration_s": 0.8, "reason": "approach"}


def _plain_observation(tmp_path: Path, name: str, *, yaw_deg: float, color: tuple[int, int, int] = (90, 90, 90)) -> Observation:
    image = Image.new("RGB", (160, 120), color)
    return _write_observation(tmp_path / name, image, yaw_deg=yaw_deg)


def _toilet_observation(tmp_path: Path, name: str, *, yaw_deg: float, x0: int, x1: int) -> Observation:
    image = Image.new("RGB", (160, 120), (72, 72, 72))
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, 44, x1, 104), fill=(244, 246, 246))
    draw.rectangle((x0 + 8, 28, x1 - 8, 48), fill=(244, 246, 246))
    return _write_observation(tmp_path / name, image, yaw_deg=yaw_deg)


def _write_observation(path: Path, image: Image.Image, *, yaw_deg: float) -> Observation:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)
    return Observation(path=path, yaw_deg=yaw_deg, frame_seq=1, analysis=analyze_image(image))
