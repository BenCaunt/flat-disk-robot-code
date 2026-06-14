"""Agent tool layer that talks to the simulator through the real robot client."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageDraw

from .paths import REPO_ROOT
from .protocol import DEFAULT_NAMESPACE
from .topomap_memory import TopomapMemoryConfig, TopomapMemoryTool
from .vision import FrameAnalysis, analyze_image_path


SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from flatdisk_robot_client import DEFAULT_CONNECT, FlatDiskRobotClient, MotionResult  # noqa: E402


OBJECT_DRIVE_SCRIPT = SCRIPTS_DIR / "object_drive_zenoh.py"
TRANSFORMERS_OBJECT_DRIVE_DETECTORS = {"florence-transformers", "grounding-dino"}
TRANSFORMERS_OBJECT_DRIVE_EXTRAS = ("torch", "transformers", "timm", "einops")
DEFAULT_OBJECT_DRIVE_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class Observation:
    path: Path
    yaw_deg: float
    frame_seq: int
    analysis: FrameAnalysis

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "yaw_deg": self.yaw_deg,
            "frame_seq": self.frame_seq,
            "brightness_center": self.analysis.brightness_center,
        }


@dataclass(frozen=True)
class AgentMotionResult:
    result: MotionResult
    frame_paths: tuple[Path, ...]
    motion_contact_sheet: Path | None = None

    def summary(self) -> dict[str, Any]:
        summary = dict(self.result.summary())
        summary["motion_frame_paths"] = [str(path) for path in self.frame_paths]
        if self.motion_contact_sheet is not None:
            summary["motion_contact_sheet"] = str(self.motion_contact_sheet)
        return summary


class AgentTools:
    def __init__(
        self,
        *,
        run_dir: Path,
        namespace: str = DEFAULT_NAMESPACE,
        connect: str = DEFAULT_CONNECT,
        reverse_yaw: bool = True,
        object_drive_detector: str = "florence-mlx",
        topomap_memory_map_dir: Path | None = None,
        topomap_memory_use_clip: bool = False,
        topomap_memory_allow_semantic_terms: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.namespace = namespace
        self.connect = connect
        self.object_drive_detector = object_drive_detector
        self.topomap_memory: TopomapMemoryTool | None = None
        self.frames_dir = run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.motion_frames_dir = run_dir / "motion_frames"
        self.motion_frames_dir.mkdir(parents=True, exist_ok=True)
        self.topomap_memory_dir = run_dir / "topomap_memory"
        self.events_path = run_dir / "events.jsonl"
        self.client = FlatDiskRobotClient(namespace=namespace, connect=connect, reverse_yaw=reverse_yaw)
        self.client.open()
        self._observation_count = 0
        self._motion_count = 0
        if topomap_memory_map_dir is not None:
            self.topomap_memory = TopomapMemoryTool(
                TopomapMemoryConfig(
                    map_dir=topomap_memory_map_dir,
                    output_dir=self.topomap_memory_dir,
                    use_clip=topomap_memory_use_clip,
                    allow_semantic_terms=topomap_memory_allow_semantic_terms,
                )
            )

    def close(self) -> None:
        self.client.close()

    def log(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "t": time.time(),
            "event": event,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def observe(self, *, label: str = "observe", timeout_s: float = 2.0) -> Observation:
        frame = self.client.latest_frame(timeout_s=timeout_s)
        imu = self.client.wait_for_imu(timeout_s=timeout_s)
        image = frame.image(rotate_180=self.client.rotate_frames_180)
        self._observation_count += 1
        path = self.frames_dir / f"{self._observation_count:04d}_{label}_seq{frame.seq}.jpg"
        image.save(path, format="JPEG", quality=90)
        analysis = analyze_image_path(path)
        obs = Observation(path=path, yaw_deg=imu.yaw_deg, frame_seq=frame.seq, analysis=analysis)
        self.log("observe", obs.summary())
        return obs

    def observe_sequence(self, *, label: str, count: int = 4, interval_s: float = 0.12) -> list[Observation]:
        observations: list[Observation] = []
        for index in range(count):
            observations.append(self.observe(label=f"{label}_{index + 1}"))
            if index + 1 < count:
                time.sleep(interval_s)
        return observations

    def turn_by_angle(self, degrees: float, *, power_percent: float = 10.0) -> AgentMotionResult:
        result = self.client.turn_by_angle(
            degrees,
            power=power_percent,
            frame_count=5,
            output_dir=self.motion_frames_dir,
        )
        recorded = self._record_motion_result(result, label="turn_by_angle")
        self.log("turn_by_angle", {"degrees": degrees, "power_percent": power_percent, "result": recorded.summary()})
        return recorded

    def drive_straight(self, power_percent: float, duration_s: float) -> AgentMotionResult:
        result = self.client.drive_straight(
            power_percent,
            duration_s,
            frame_count=5,
            max_duration_s=6.0,
            output_dir=self.motion_frames_dir,
        )
        recorded = self._record_motion_result(result, label="drive_straight")
        self.log(
            "drive_straight",
            {"power_percent": power_percent, "duration_s": duration_s, "result": recorded.summary()},
        )
        return recorded

    def visual_servo_object(
        self,
        prompt: str,
        *,
        duration_s: float = 2.0,
        detector: str | None = None,
        forward_power: float = 18.0,
    ) -> dict[str, Any]:
        """Run the visible-object servo controller as a bounded harness tool."""

        detector_name = detector or self.object_drive_detector
        self._motion_count += 1
        overlay_dir = self.motion_frames_dir / f"{self._motion_count:04d}_visual_servo_overlays"
        raw_dir = self.motion_frames_dir / f"{self._motion_count:04d}_visual_servo_raw"
        cmd = _object_drive_command(detector=detector_name) + [
            "--prompt",
            prompt,
            "--duration",
            f"{duration_s:.3f}",
            "--forward-power",
            f"{forward_power:.3f}",
            "--namespace",
            self.namespace,
            "--connect",
            self.connect,
            "--detector",
            detector_name,
            "--arm",
            "--overlay-dir",
            str(overlay_dir),
            "--overlay-every",
            "1",
            "--raw-frame-dir",
            str(raw_dir),
            "--raw-frame-every",
            "1",
        ]
        started = time.perf_counter()
        timeout_s = _object_drive_timeout_s(duration_s)
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            cwd=REPO_ROOT,
        )
        elapsed_s = time.perf_counter() - started
        frame_paths = tuple(_sample_evenly(sorted(raw_dir.glob("*.jpg")), count=5))
        debug_overlay_paths = tuple(_sample_evenly(sorted(overlay_dir.glob("*.jpg")), count=5))
        contact_sheet = None
        debug_overlay_sheet = None
        grounding_audit_sheet = None
        if frame_paths:
            contact_sheet = make_contact_sheet(list(frame_paths), self.motion_frames_dir / f"{self._motion_count:04d}_visual_servo_strip.jpg")
        if debug_overlay_paths:
            debug_overlay_sheet = make_contact_sheet(
                list(debug_overlay_paths),
                self.motion_frames_dir / f"{self._motion_count:04d}_visual_servo_debug_overlay_strip.jpg",
            )
        if frame_paths and debug_overlay_paths:
            grounding_audit_sheet = make_visual_servo_grounding_audit_sheet(
                list(frame_paths),
                list(debug_overlay_paths),
                self.motion_frames_dir / f"{self._motion_count:04d}_visual_servo_grounding_audit.jpg",
            )
        summary = {
            "action": "visual_servo_object",
            "prompt": prompt,
            "detector": detector_name,
            "duration_s": duration_s,
            "timeout_s": timeout_s,
            "forward_power": forward_power,
            "elapsed_s": elapsed_s,
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            **_parse_object_drive_status(completed.stdout, returncode=completed.returncode),
            "motion_frame_paths": [str(path) for path in frame_paths],
            "motion_contact_sheet": str(contact_sheet) if contact_sheet else None,
            "motion_frame_source": "raw_camera",
            "debug_overlay_frame_paths": [str(path) for path in debug_overlay_paths],
            "debug_overlay_contact_sheet": str(debug_overlay_sheet) if debug_overlay_sheet else None,
            "grounding_audit_contact_sheet": str(grounding_audit_sheet) if grounding_audit_sheet else None,
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
        }
        self.log("visual_servo_object", summary)
        return summary

    def query_topomap_memory(self, *, image_path: Path, goal_query: str) -> dict[str, Any]:
        if self.topomap_memory is None:
            summary = {
                "action": "query_topomap_memory",
                "ok": False,
                "reason": "topomap_memory_not_configured",
                "goal_query": goal_query,
                "topomap_contact_sheet": None,
            }
        else:
            summary = self.topomap_memory.query(image_path=image_path, goal_query=goal_query)
        self.log("query_topomap_memory", summary)
        return summary

    def stop(self) -> None:
        self.client.stop()
        self.log("stop", {"stopped": True})

    def _record_motion_result(self, result: MotionResult, *, label: str) -> AgentMotionResult:
        self._motion_count += 1
        frame_paths: list[Path] = []
        for index, frame in enumerate(result.frames, start=1):
            path = self.motion_frames_dir / f"{self._motion_count:04d}_{label}_{index:02d}_seq{frame.seq}.jpg"
            frame.image(rotate_180=self.client.rotate_frames_180).save(path, format="JPEG", quality=90)
            frame_paths.append(path)
        return AgentMotionResult(
            result=result,
            frame_paths=tuple(frame_paths),
            motion_contact_sheet=result.stitched_path,
        )


def make_contact_sheet(paths: list[Path], output_path: Path) -> Path:
    if not paths:
        raise ValueError("no paths for contact sheet")
    images = [Image.open(path).convert("RGB") for path in paths]
    thumb_w = 240
    thumbs: list[Image.Image] = []
    for image in images:
        scale = thumb_w / image.width
        thumb = image.resize((thumb_w, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    sheet = Image.new("RGB", (thumb_w * len(thumbs), max(t.height for t in thumbs)), "white")
    x = 0
    for thumb in thumbs:
        sheet.paste(thumb, (x, 0))
        x += thumb.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)
    return output_path


def make_visual_servo_grounding_audit_sheet(raw_paths: list[Path], overlay_paths: list[Path], output_path: Path) -> Path:
    if not raw_paths or not overlay_paths:
        raise ValueError("raw and overlay paths are required for grounding audit sheet")
    pairs = list(zip(raw_paths, overlay_paths))
    if not pairs:
        raise ValueError("no paired frames for grounding audit sheet")

    thumb_w = 200
    label_h = 18
    raw_thumbs: list[Image.Image] = []
    overlay_thumbs: list[Image.Image] = []
    for raw_path, overlay_path in pairs:
        raw = Image.open(raw_path).convert("RGB")
        overlay = Image.open(overlay_path).convert("RGB")
        for image, target in ((raw, raw_thumbs), (overlay, overlay_thumbs)):
            scale = thumb_w / image.width
            target.append(image.resize((thumb_w, max(1, int(image.height * scale))), Image.Resampling.LANCZOS))

    tile_h = max([thumb.height for thumb in raw_thumbs + overlay_thumbs])
    sheet = Image.new("RGB", (thumb_w * len(pairs), (tile_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (raw_thumb, overlay_thumb) in enumerate(zip(raw_thumbs, overlay_thumbs), start=1):
        x = (index - 1) * thumb_w
        draw.rectangle((x, 0, x + thumb_w, label_h), fill=(32, 32, 32))
        draw.text((x + 5, 3), f"raw t{index}", fill=(255, 255, 255))
        sheet.paste(raw_thumb, (x, label_h))
        overlay_y = tile_h + label_h
        draw.rectangle((x, overlay_y, x + thumb_w, overlay_y + label_h), fill=(96, 24, 24))
        draw.text((x + 5, overlay_y + 3), f"detector overlay t{index}", fill=(255, 255, 255))
        sheet.paste(overlay_thumb, (x, overlay_y + label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=90)
    return output_path


def _sample_evenly(paths: list[Path], *, count: int) -> list[Path]:
    if count <= 0 or len(paths) <= count:
        return list(paths)
    last_index = len(paths) - 1
    indices = [round(index * last_index / (count - 1)) for index in range(count)]
    return [paths[index] for index in indices]


def _tail_text(text: str, *, max_chars: int = 1600) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _parse_object_drive_status(stdout: str, *, returncode: int) -> dict[str, Any]:
    """Summarize object-drive logs into fields a planner can reason about."""

    status_records: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.startswith("object-drive armed="):
            continue
        status_records.append(_parse_object_drive_status_fields(line))
    last = status_records[-1] if status_records else {}
    pub_values = [_int_or_zero(record.get("pub")) for record in status_records]
    command_values = [record.get("cmd", "none") for record in status_records]
    det_values = [record.get("det", "none") for record in status_records]
    detection_status_count = sum(1 for det in det_values if det and det != "none")
    detection_coverage_fraction = detection_status_count / len(status_records) if status_records else 0.0
    final_detection = last.get("det", "none")
    final_command = last.get("cmd", "none")
    detection_label, detection_source, detection_score = _parse_detection_descriptor(final_detection)
    ever_detected = any(det and det != "none" for det in det_values)
    target_detected = bool(final_detection and final_detection != "none")
    motor_commands_sent = max(pub_values, default=0)
    moved = motor_commands_sent > 0 or any(cmd and cmd != "none" for cmd in command_values)
    if returncode != 0:
        failure_reason = "process_failed"
    elif moved:
        failure_reason = None
    elif not ever_detected:
        failure_reason = "no_detection"
    else:
        failure_reason = "no_motion_command"
    if returncode != 0:
        servo_status = "process_failed"
    elif moved:
        servo_status = "moved"
    else:
        servo_status = failure_reason or "no_motion"
    grounding_stability = _grounding_stability(
        status_count=len(status_records),
        detection_status_count=detection_status_count,
        detection_coverage_fraction=detection_coverage_fraction,
        ever_detected=ever_detected,
    )
    planner_note = _planner_note_for_grounding(target_detected=target_detected, grounding_stability=grounding_stability)
    return {
        "servo_status": servo_status,
        "target_detected": target_detected,
        "ever_detected": ever_detected,
        "status_sample_count": len(status_records),
        "detection_status_count": detection_status_count,
        "detection_coverage_fraction": round(detection_coverage_fraction, 3),
        "grounding_stability": grounding_stability,
        "moved": moved,
        "motor_commands_sent": motor_commands_sent,
        "last_command": final_command,
        "last_detection": final_detection,
        "last_detection_label": detection_label,
        "last_detection_source": detection_source,
        "last_detection_score": detection_score,
        "semantic_identity": "unverified_phrase_grounding" if target_detected else "none",
        "planner_note": planner_note,
        "prediction_count": _int_or_zero(last.get("pred")),
        "track_count": _int_or_zero(last.get("track")),
        "imu_prediction_count": _int_or_zero(last.get("imu_pred")),
        "lost_count": _int_or_zero(last.get("lost")),
        "detector_pending": _bool_or_false(last.get("pending")),
        "failure_reason": failure_reason,
    }


def _object_drive_command(*, detector: str) -> list[str]:
    override = os.environ.get("FLATDISK_OBJECT_DRIVE_COMMAND", "").strip()
    if override:
        return shlex.split(override) + [str(OBJECT_DRIVE_SCRIPT)]
    if detector in TRANSFORMERS_OBJECT_DRIVE_DETECTORS:
        python_override = os.environ.get("FLATDISK_OBJECT_DRIVE_PYTHON", "").strip()
        if python_override:
            return [python_override, str(OBJECT_DRIVE_SCRIPT)]
        cmd = ["uv", "run", "--project", "sim"]
        for package in TRANSFORMERS_OBJECT_DRIVE_EXTRAS:
            cmd.extend(["--with", package])
        cmd.extend(["python", str(OBJECT_DRIVE_SCRIPT)])
        return cmd
    return [sys.executable, str(OBJECT_DRIVE_SCRIPT)]


def _object_drive_timeout_s(duration_s: float) -> float:
    override = os.environ.get("FLATDISK_OBJECT_DRIVE_TIMEOUT_S", "").strip()
    if override:
        try:
            return max(float(duration_s) + 5.0, float(override))
        except ValueError:
            pass
    return max(DEFAULT_OBJECT_DRIVE_TIMEOUT_S, float(duration_s) + DEFAULT_OBJECT_DRIVE_TIMEOUT_S)


def _parse_object_drive_status_fields(line: str) -> dict[str, str]:
    matches = list(re.finditer(r"\b([A-Za-z_]+)=", line))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        fields[key] = line[value_start:value_end].strip()
    return fields


def _grounding_stability(
    *,
    status_count: int,
    detection_status_count: int,
    detection_coverage_fraction: float,
    ever_detected: bool,
) -> str:
    if not ever_detected:
        return "no_detection"
    if status_count < 3:
        return "insufficient_status_history"
    if detection_status_count < 2 or detection_coverage_fraction < 0.35:
        return "sparse_detection_coverage"
    return "status_track_present"


def _planner_note_for_grounding(*, target_detected: bool, grounding_stability: str) -> str:
    if not target_detected:
        return "visual_servo_object did not provide a visible phrase-grounded track."
    if grounding_stability != "status_track_present":
        return (
            "visual_servo_object moved after sparse or unstable phrase grounding; "
            "do not repeat the same prompt without checking the grounding audit and latest RGB frame."
        )
    return (
        "visual_servo_object moved toward a detector/tracker match for the prompt; "
        "this does not prove the visible object is the final goal class."
    )


def _parse_detection_descriptor(value: str | None) -> tuple[str | None, str | None, float | None]:
    text = str(value or "").strip()
    if not text or text == "none":
        return None, None, None
    label, source, score_text = (text.rsplit(":", 2) + ["", ""])[:3] if ":" in text else (text, "", "")
    try:
        score = float(score_text)
    except ValueError:
        score = None
    return label or None, source or None, score


def _int_or_zero(value: str | None) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _bool_or_false(value: str | None) -> bool:
    return str(value).strip().lower() == "true"
