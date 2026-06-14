from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from flatdisk_sim.agent_candidates.minimalist_policy import (
    FORBIDDEN_PRIVILEGED_INPUTS,
    RealBotMinimalistPolicy,
    camera_imu_cue,
    sanitize_action_history,
    sensor_contract,
)
from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.vision import Detection, FrameAnalysis, analyze_image_path


def test_minimalist_history_strips_privileged_fields() -> None:
    history = [
        {
            "action": "drive_straight",
            "duration_s": 0.6,
            "power_percent": 20.0,
            "reason": "prior move",
            "distance_m": 0.1,
            "pose": {"x": 1.0},
            "nearest_target": "Sofa|1|2",
        }
    ]

    safe = sanitize_action_history(history)

    assert safe == [
        {
            "action": "drive_straight",
            "duration_s": 0.6,
            "power_percent": 20.0,
            "reason": "prior move",
        }
    ]
    assert FORBIDDEN_PRIVILEGED_INPUTS.isdisjoint(safe[0])


def test_minimalist_turns_toward_off_center_goal_detection(tmp_path: Path) -> None:
    obs = _detected_toilet_observation(tmp_path, center_offset=0.48, area_fraction=0.02)
    policy = RealBotMinimalistPolicy()

    action = policy.choose_action(obs, prompt="Drive to the toilet in the bathroom.", history=[])

    assert action.action == "turn_by_angle"
    assert 12.0 <= action.degrees <= 35.0
    assert "center_strong_visual_cue" in action.reason


def test_minimalist_stop_requires_repeated_close_centered_evidence(tmp_path: Path) -> None:
    policy = RealBotMinimalistPolicy(allow_stop=True, stop_window=3)
    obs = _detected_toilet_observation(tmp_path, center_offset=0.02, area_fraction=0.045)

    first = policy.choose_action(obs, prompt="Drive to the toilet.", history=[])
    second = policy.choose_action(obs, prompt="Drive to the toilet.", history=[])
    third = policy.choose_action(obs, prompt="Drive to the toilet.", history=[])

    assert first.action == "drive_straight"
    assert second.action == "drive_straight"
    assert third.action == "stop"
    assert third.success is True
    assert third.reason == "stable_close_centered_visual_evidence"


def test_minimalist_stop_can_be_disabled_for_hidden_evaluator_runs(tmp_path: Path) -> None:
    policy = RealBotMinimalistPolicy(allow_stop=False, stop_window=3)
    obs = _detected_toilet_observation(tmp_path, center_offset=0.0, area_fraction=0.05)

    actions = [policy.choose_action(obs, prompt="Drive to the toilet.", history=[]) for _ in range(4)]

    assert {action.action for action in actions} <= {"drive_straight", "turn_by_angle"}
    assert actions[-1].action != "stop"


def test_minimalist_ignores_hidden_history_when_choosing_action(tmp_path: Path) -> None:
    obs = _detected_toilet_observation(tmp_path, center_offset=-0.42, area_fraction=0.018)
    hidden_history = [
        {
            "action": "drive_straight",
            "duration_s": 0.6,
            "power_percent": 20.0,
            "hidden_score": {"success": True},
            "target_pose": [1, 2, 3],
            "scene": "FloorPlan999",
        }
    ]
    clean_history = sanitize_action_history(hidden_history)

    hidden_action = RealBotMinimalistPolicy().choose_action(
        obs,
        prompt="Drive to the toilet.",
        history=hidden_history,
    )
    clean_action = RealBotMinimalistPolicy().choose_action(
        obs,
        prompt="Drive to the toilet.",
        history=clean_history,
    )

    assert hidden_action == clean_action


def test_minimalist_camera_cue_works_without_target_detector_for_seat(tmp_path: Path) -> None:
    path = tmp_path / "seat_right.jpg"
    image = Image.new("RGB", (320, 240), (44, 48, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 150, 320, 240), fill=(205, 198, 184))
    draw.rectangle((198, 70, 306, 142), fill=(104, 112, 122))
    draw.line((198, 95, 306, 95), fill=(62, 68, 75), width=4)
    draw.line((198, 122, 306, 122), fill=(70, 76, 82), width=4)
    image.save(path)
    obs = Observation(path=path, yaw_deg=0.0, frame_seq=1, analysis=analyze_image_path(path))

    cue = camera_imu_cue(obs, "Drive to the sofa in the living room.")
    action = RealBotMinimalistPolicy().choose_action(obs, prompt="Drive to the sofa in the living room.", history=[])

    assert cue.family == "seat"
    assert cue.score > 0.20
    assert cue.offset > 0.12
    assert action.action == "turn_by_angle"
    assert action.degrees > 0.0


def test_minimalist_sensor_contract_documents_real_robot_boundary() -> None:
    contract = sensor_contract()

    assert "IMU yaw" in contract["allowed"]
    assert "latest RGB camera frame" in contract["allowed"]
    assert "timed bounded" in contract["motion"]
    assert set(contract["forbidden"]) >= {"distance_m", "pose", "target_pose", "wheel_encoders", "odometry"}


def _detected_toilet_observation(tmp_path: Path, *, center_offset: float, area_fraction: float) -> Observation:
    path = tmp_path / f"toilet_{center_offset}_{area_fraction}.jpg"
    image = Image.new("RGB", (320, 240), (38, 42, 46))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 150, 320, 240), fill=(218, 220, 214))
    center_x = int(160 + center_offset * 160)
    half_w = max(12, int((area_fraction * 320 * 240) ** 0.5))
    bbox = (
        max(10, center_x - half_w),
        88,
        min(310, center_x + half_w),
        88 + max(24, half_w),
    )
    draw.rounded_rectangle(bbox, radius=6, fill=(244, 246, 246), outline=(220, 224, 224), width=3)
    image.save(path)
    detection = Detection(
        name="toilet",
        area_fraction=area_fraction,
        center_offset=center_offset,
        bbox=bbox,
        confidence=0.78,
    )
    analysis = FrameAnalysis(detections=(detection,), brightness_center=0.52)
    return Observation(path=path, yaw_deg=0.0, frame_seq=1, analysis=analysis)
