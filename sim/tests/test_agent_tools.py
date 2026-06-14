from __future__ import annotations

from pathlib import Path
import sys

from flatdisk_sim.agent_tools import (
    _detector_doctor_command,
    _detector_doctor_timeout_s,
    _grounding_geometry_warning,
    _object_drive_command,
    _object_drive_timeout_s,
    _parse_object_drive_status,
    _sample_evenly,
)


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
        "object-drive armed=True pred=2 track=0 imu_pred=0 filter=off "
        "pub=0 cmd=none heading_error=nan det=none pending=False lost=1\n"
        "object-drive armed=True pred=3 track=0 imu_pred=0 filter=off "
        "pub=1 cmd=19.0/17.0% heading_error=4.2deg det=sofa:florence-mlx:0.82 pending=False lost=0\n"
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
    assert summary["status_sample_count"] == 3
    assert summary["detection_status_count"] == 2
    assert summary["detection_coverage_fraction"] == 0.667
    assert summary["grounding_stability"] == "status_track_present"
    assert summary["semantic_identity"] == "unverified_phrase_grounding"
    assert "does not prove" in summary["planner_note"]
    assert summary["failure_reason"] is None


def test_parse_object_drive_status_preserves_phrase_detection() -> None:
    summary = _parse_object_drive_status(
        "object-drive armed=True pred=1 track=0 imu_pred=0 filter=off "
        "pub=22 cmd=12/22% heading_error=-17.7deg "
        "det=the white toilet on the left:florence-transformers:1.00 pending=False lost=0\n",
        returncode=0,
    )

    assert summary["last_detection"] == "the white toilet on the left:florence-transformers:1.00"
    assert summary["last_detection_label"] == "the white toilet on the left"
    assert summary["last_detection_source"] == "florence-transformers"
    assert summary["last_detection_score"] == 1.0


def test_parse_object_drive_status_flags_sparse_grounding() -> None:
    summary = _parse_object_drive_status(
        "object-drive armed=True pred=0 track=0 imu_pred=0 filter=off "
        "pub=0 cmd=none heading_error=nan det=none pending=True lost=0\n"
        "object-drive armed=True pred=0 track=0 imu_pred=0 filter=off "
        "pub=0 cmd=none heading_error=nan det=none pending=True lost=0\n"
        "object-drive armed=True pred=1 track=0 imu_pred=0 filter=off "
        "pub=12 cmd=8/24% heading_error=-12.0deg det=the white toilet:florence-transformers:1.00 pending=False lost=0\n",
        returncode=0,
    )

    assert summary["moved"] is True
    assert summary["grounding_stability"] == "sparse_detection_coverage"
    assert summary["detection_coverage_fraction"] == 0.333
    assert "sparse or unstable" in summary["planner_note"]


def test_transformers_object_drive_command_runs_with_optional_detector_deps() -> None:
    cmd = _object_drive_command(detector="florence-transformers")

    assert cmd[:4] == ["uv", "run", "--project", "sim"]
    assert cmd.count("--with") >= 3
    assert "torch" in cmd
    assert "transformers" in cmd
    assert "timm" in cmd
    assert "einops" in cmd
    assert cmd[-2] == "python"
    assert cmd[-1].endswith("object_drive_zenoh.py")


def test_grounding_dino_object_drive_command_includes_torchvision() -> None:
    cmd = _object_drive_command(detector="grounding-dino")

    assert "torch" in cmd
    assert "torchvision" in cmd
    assert "transformers" in cmd
    assert "einops" in cmd
    assert cmd[-2] == "python"
    assert cmd[-1].endswith("object_drive_zenoh.py")


def test_grounding_dino_detector_doctor_command_includes_torchvision() -> None:
    cmd = _detector_doctor_command(detector="grounding-dino")

    assert cmd[:4] == ["uv", "run", "--project", "sim"]
    assert "torch" in cmd
    assert "torchvision" in cmd
    assert "transformers" in cmd
    assert cmd[-1] == "flatdisk-sim-detector-doctor"


def test_detector_doctor_command_override(monkeypatch) -> None:
    monkeypatch.setenv("FLATDISK_DETECTOR_DOCTOR_COMMAND", "/tmp/doctor --flag")

    assert _detector_doctor_command(detector="grounding-dino") == ["/tmp/doctor", "--flag"]


def test_grounding_geometry_warning_flags_edge_touching_boxes() -> None:
    warning = _grounding_geometry_warning(
        {
            "bbox_touches_image_edge": True,
            "bbox_edge_contact": ["left", "top"],
            "bbox_area_fraction": 0.14,
            "bbox_width_fraction": 0.2,
            "bbox_height_fraction": 0.6,
        }
    )

    assert warning is not None
    assert "selected_box_touches_left,top" in warning
    assert "inspect overlay" in warning


def test_grounding_geometry_warning_allows_centered_boxes() -> None:
    assert (
        _grounding_geometry_warning(
            {
                "bbox_touches_image_edge": False,
                "bbox_edge_contact": [],
                "bbox_area_fraction": 0.14,
                "bbox_width_fraction": 0.4,
                "bbox_height_fraction": 0.35,
            }
        )
        is None
    )


def test_transformers_object_drive_command_uses_prepared_python(monkeypatch) -> None:
    monkeypatch.setenv("FLATDISK_OBJECT_DRIVE_PYTHON", "/tmp/object-drive/bin/python")

    cmd = _object_drive_command(detector="florence-transformers")

    assert cmd[0] == "/tmp/object-drive/bin/python"
    assert "--with" not in cmd
    assert cmd[-1].endswith("object_drive_zenoh.py")


def test_mlx_object_drive_command_keeps_current_python() -> None:
    cmd = _object_drive_command(detector="florence-mlx")

    assert cmd[0] == sys.executable
    assert cmd[-1].endswith("object_drive_zenoh.py")


def test_object_drive_timeout_allows_cold_model_load() -> None:
    assert _object_drive_timeout_s(2.0) >= 300.0


def test_detector_doctor_timeout_allows_cold_model_load() -> None:
    assert _detector_doctor_timeout_s() >= 420.0


def test_sample_evenly_keeps_first_middle_and_last_paths() -> None:
    paths = [Path(f"frame_{index}") for index in range(10)]

    assert _sample_evenly(paths, count=5) == [
        Path("frame_0"),
        Path("frame_2"),
        Path("frame_4"),
        Path("frame_7"),
        Path("frame_9"),
    ]


def test_object_drive_timeout_env_override_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("FLATDISK_OBJECT_DRIVE_TIMEOUT_S", "30")

    assert _object_drive_timeout_s(40.0) == 45.0


def test_detector_doctor_timeout_env_override_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("FLATDISK_DETECTOR_DOCTOR_TIMEOUT_S", "2")

    assert _detector_doctor_timeout_s() == 5.0
