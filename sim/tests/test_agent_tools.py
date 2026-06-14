from __future__ import annotations

from flatdisk_sim.agent_tools import _parse_object_drive_status


def test_parse_object_drive_status_no_detection() -> None:
    summary = _parse_object_drive_status(
        "object-drive namespace=ns connect=tcp prompt='sofa'\n"
        "object-drive armed=True pred=2 track=0 imu_pred=0 filter=off "
        "pub=0 cmd=none heading_error=nan det=none pending=False lost=2\n",
        returncode=0,
    )

    assert summary["servo_status"] == "no_detection"
    assert summary["target_detected"] is False
    assert summary["ever_detected"] is False
    assert summary["moved"] is False
    assert summary["motor_commands_sent"] == 0
    assert summary["last_detection"] == "none"
    assert summary["last_detection_label"] is None
    assert summary["semantic_identity"] == "none"
    assert summary["failure_reason"] == "no_detection"


def test_parse_object_drive_status_moved_with_detection() -> None:
    summary = _parse_object_drive_status(
        "object-drive armed=True pred=4 track=1 imu_pred=0 filter=off "
        "pub=3 cmd=19.0/17.0% heading_error=4.2deg det=sofa:florence-mlx:0.82 pending=False lost=0\n",
        returncode=0,
    )

    assert summary["servo_status"] == "moved"
    assert summary["target_detected"] is True
    assert summary["ever_detected"] is True
    assert summary["moved"] is True
    assert summary["motor_commands_sent"] == 3
    assert summary["last_command"] == "19.0/17.0%"
    assert summary["last_detection"] == "sofa:florence-mlx:0.82"
    assert summary["last_detection_label"] == "sofa"
    assert summary["last_detection_source"] == "florence-mlx"
    assert summary["last_detection_score"] == 0.82
    assert summary["semantic_identity"] == "unverified_phrase_grounding"
    assert "does not prove" in summary["planner_note"]
    assert summary["failure_reason"] is None
