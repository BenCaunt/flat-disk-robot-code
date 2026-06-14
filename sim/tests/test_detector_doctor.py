from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from flatdisk_sim.detector_doctor import run_detector_doctor
from object_drive_zenoh import Detection


class _FakeDetector:
    def detect(self, image: Image.Image, prompt: str) -> tuple[Detection, ...]:
        del image
        if "edge" in prompt:
            return (Detection((0.0, 0.0, 50.0, 60.0), "edge chair", 0.88, "fake", raw="edge raw"),)
        if "chair" not in prompt:
            return ()
        return (
            Detection((10.0, 20.0, 90.0, 110.0), "chair", 0.91, "fake", raw="chair raw"),
            Detection((100.0, 30.0, 140.0, 90.0), "lamp", 0.41, "fake", raw="lamp raw"),
        )


def _fake_factory(args: argparse.Namespace) -> _FakeDetector:
    assert args.detector == "fake-detector"
    return _FakeDetector()


def test_detector_doctor_reports_selected_boxes_and_writes_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (160, 120), "white").save(image_path)

    report = run_detector_doctor(
        image_paths=[image_path],
        prompts=["chair"],
        output_dir=tmp_path / "doctor",
        detector_name="fake-detector",
        detector_factory=_fake_factory,
    )

    assert report["schema"] == "flatdisk.open_vocab_detector_doctor.v1"
    assert report["ready_for_visual_servo"] is True
    assert report["detection_count"] == 2
    assert report["selected_detection_count"] == 1
    assert report["checks"][0]["selected_detection"]["label"] == "chair"
    assert report["checks"][0]["selected_detection"]["bbox_area_fraction"] == 0.375
    assert report["checks"][0]["selected_detection"]["bbox_center_xy_norm"] == [0.3125, 0.5417]
    assert report["checks"][0]["selected_detection"]["bbox_touches_image_edge"] is False
    assert Path(report["checks"][0]["overlay"]).exists()
    assert (tmp_path / "doctor" / "detector_doctor.md").exists()
    written = json.loads((tmp_path / "doctor" / "detector_doctor.json").read_text(encoding="utf-8"))
    assert written["ready_for_visual_servo"] is True


def test_detector_doctor_marks_no_boxes_not_ready(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (160, 120), "white").save(image_path)

    report = run_detector_doctor(
        image_paths=[image_path],
        prompts=["toilet"],
        output_dir=tmp_path / "doctor",
        detector_name="fake-detector",
        detector_factory=_fake_factory,
    )

    assert report["ready_for_visual_servo"] is False
    assert report["detection_count"] == 0
    assert "returned no usable boxes" in report["recommendation"]
    assert Path(report["checks"][0]["overlay"]).exists()


def test_detector_doctor_marks_edge_touching_selected_box_as_warning(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (160, 120), "white").save(image_path)

    report = run_detector_doctor(
        image_paths=[image_path],
        prompts=["edge chair"],
        output_dir=tmp_path / "doctor",
        detector_name="fake-detector",
        detector_factory=_fake_factory,
    )

    selected = report["checks"][0]["selected_detection"]
    assert report["ready_for_visual_servo"] is True
    assert selected["bbox_touches_image_edge"] is True
    assert selected["bbox_edge_contact"] == ["left", "top"]
    assert "touches the image edge" in report["recommendation"]
