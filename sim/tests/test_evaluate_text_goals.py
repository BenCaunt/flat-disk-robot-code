from __future__ import annotations

from flatdisk_sim.text_goal_policy_core import PolicyAction, policy_history_record, validate_action


def test_policy_history_record_omits_hidden_evaluator_fields() -> None:
    record = policy_history_record(
        PolicyAction(
            action="drive_straight",
            power_percent=20.0,
            duration_s=0.7,
            reason="approach visible target",
        )
    )

    assert set(record) == {"action", "degrees", "power_percent", "duration_s", "reason"}
    forbidden = {"distance_m", "nearest_target", "hidden_score", "pose", "objects", "scene"}
    assert forbidden.isdisjoint(record)


def test_policy_stop_is_replaced_with_navigation_by_default() -> None:
    action = validate_action({"action": "stop", "reason": "target looks close"})

    assert action.action == "drive_straight"
    assert action.power_percent == 20.0
    assert action.duration_s == 0.6
    assert "target looks close" in action.reason


def test_policy_drive_commands_are_clamped_to_progress_bounds() -> None:
    action = validate_action({"action": "drive_straight", "power_percent": 12.0, "duration_s": 0.25})

    assert action.action == "drive_straight"
    assert action.power_percent == 20.0
    assert action.duration_s == 0.6
