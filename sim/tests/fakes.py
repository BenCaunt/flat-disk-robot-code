from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageDraw

from flatdisk_sim.vision import analyze_image_path


class FakeMotionResult:
    def __init__(
        self,
        *,
        command: str,
        ok: bool = True,
        elapsed_s: float = 0.0,
        motion_frame_paths: list[Path] | None = None,
        motion_contact_sheet: Path | None = None,
    ) -> None:
        self.command = command
        self.ok = ok
        self.elapsed_s = elapsed_s
        self.motion_frame_paths = motion_frame_paths or []
        self.motion_contact_sheet = motion_contact_sheet

    def summary(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "ok": self.ok,
            "elapsed_s": round(self.elapsed_s, 3),
            "motion_frame_paths": [str(path) for path in self.motion_frame_paths],
            "motion_contact_sheet": str(self.motion_contact_sheet) if self.motion_contact_sheet else None,
        }


class FakeHarnessTools:
    def __init__(self, *, run_dir: Path, environment: str = "studio") -> None:
        self.run_dir = run_dir
        self.environment = environment
        self.frames_dir = run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.motion_frames_dir = run_dir / "motion_frames"
        self.motion_frames_dir.mkdir(parents=True, exist_ok=True)
        self.yaw_deg = 0.0
        self.frame_seq = 0
        self.motion_seq = 0
        self._progress = 0.0
        self._closed = False

    def observe(self, *, label: str = "observe", timeout_s: float = 2.0) -> Any:
        del timeout_s
        self.frame_seq += 1
        path = self.frames_dir / f"{self.frame_seq:04d}_{label}.jpg"
        self._render_camera(path)
        analysis = analyze_image_path(path)
        return _FakeObservation(path=path, yaw_deg=self.yaw_deg, frame_seq=self.frame_seq, analysis=analysis)

    def preview_frame(self, *, label: str = "dashboard_preview", timeout_s: float = 0.25) -> dict[str, Any]:
        observation = self.observe(label=label, timeout_s=timeout_s).summary()
        return {
            "path": observation["path"],
            "frame_seq": observation["frame_seq"],
            "width": 320,
            "height": 240,
            "received_age_s": 0.0,
        }

    def preview_frame_bytes(self, *, timeout_s: float = 0.1) -> dict[str, Any]:
        del timeout_s
        self.frame_seq += 1
        path = self.frames_dir / f"{self.frame_seq:04d}_dashboard_preview_bytes.jpg"
        self._render_camera(path)
        buffer = BytesIO()
        Image.open(path).save(buffer, format="JPEG", quality=85)
        return {
            "jpeg": buffer.getvalue(),
            "frame_seq": self.frame_seq,
            "width": 320,
            "height": 240,
            "frame_received_age_s": 0.0,
            "imu_seq": self.frame_seq * 5,
            "yaw_deg": self.yaw_deg,
            "imu_received_age_s": 0.0,
        }

    def preview_telemetry(self, *, timeout_s: float = 0.05) -> dict[str, Any]:
        del timeout_s
        return {
            "frame_seq": self.frame_seq,
            "width": 320,
            "height": 240,
            "frame_received_age_s": 0.0,
            "imu_seq": self.frame_seq * 5,
            "yaw_deg": self.yaw_deg,
            "imu_received_age_s": 0.0,
        }

    def turn_by_angle(self, degrees: float, *, power_percent: float = 10.0) -> FakeMotionResult:
        del power_percent
        start = time.perf_counter()
        self.yaw_deg = ((self.yaw_deg + float(degrees) + 180.0) % 360.0) - 180.0
        return self._fake_motion_result("turn_by_angle", elapsed_s=time.perf_counter() - start)

    def drive_straight(self, power_percent: float, duration_s: float) -> FakeMotionResult:
        del power_percent
        start = time.perf_counter()
        heading_quality = max(0.45, 1.0 - abs(self.yaw_deg) / 90.0)
        self._progress = min(1.0, self._progress + 0.26 * heading_quality * (float(duration_s) / 0.7))
        return self._fake_motion_result("drive_straight", elapsed_s=time.perf_counter() - start)

    def visual_servo_object(
        self,
        prompt: str,
        *,
        duration_s: float = 2.0,
        detector: str | None = None,
        forward_power: float = 18.0,
    ) -> dict[str, Any]:
        del prompt, duration_s, detector, forward_power
        self._progress = min(1.0, self._progress + 0.2)
        summary = self._fake_motion_result("visual_servo_object", elapsed_s=0.01).summary()
        debug_overlay = self.motion_frames_dir / f"{self.motion_seq:04d}_visual_servo_debug_overlay_strip.jpg"
        grounding_audit = self.motion_frames_dir / f"{self.motion_seq:04d}_visual_servo_grounding_audit.jpg"
        if summary.get("motion_contact_sheet"):
            source = Path(str(summary["motion_contact_sheet"]))
            debug_overlay.write_bytes(source.read_bytes())
            grounding_audit.write_bytes(source.read_bytes())
        summary.update(
            {
                "action": "visual_servo_object",
                "debug_overlay_contact_sheet": str(debug_overlay),
                "grounding_audit_contact_sheet": str(grounding_audit),
                "servo_status": "moved",
                "target_detected": True,
                "ever_detected": True,
                "moved": True,
                "motor_commands_sent": 1,
                "last_command": "18.0/18.0%",
                "last_detection": "fake:florence-mlx:0.90",
                "failure_reason": None,
            }
        )
        return summary

    def query_topomap_memory(self, *, image_path: Path, goal_query: str) -> dict[str, Any]:
        return {
            "action": "query_topomap_memory",
            "ok": False,
            "reason": "topomap_memory_not_configured",
            "image_path": str(image_path),
            "goal_query": goal_query,
            "topomap_contact_sheet": None,
        }

    def check_object_grounding(self, *, image_path: Path, prompt: str, detector: str | None = None) -> dict[str, Any]:
        detector_name = detector or "fake-open-vocab"
        overlay_path = self.motion_frames_dir / f"{self.motion_seq + 1:04d}_grounding_check_overlay.jpg"
        Image.open(image_path).save(overlay_path, format="JPEG", quality=90)
        return {
            "action": "check_object_grounding",
            "ok": True,
            "prompt": prompt,
            "detector": detector_name,
            "image_path": str(image_path),
            "ready_for_visual_servo": True,
            "detection_count": 1,
            "selected_detection_count": 1,
            "selected_label": prompt,
            "selected_score": 0.91,
            "selected_bbox_xyxy": [80, 60, 240, 180],
            "selected_bbox_area_fraction": 0.5,
            "selected_bbox_center_xy_norm": [0.5, 0.5],
            "selected_bbox_width_fraction": 0.5,
            "selected_bbox_height_fraction": 0.5,
            "selected_bbox_touches_image_edge": False,
            "selected_bbox_edge_contact": [],
            "grounding_geometry_warning": None,
            "overlay_path": str(overlay_path),
            "report_path": None,
            "markdown_path": None,
            "recommendation": "usable grounding on fake frame",
        }

    def stop(self) -> FakeMotionResult:
        return FakeMotionResult(command="stop")

    def close(self) -> None:
        self._closed = True

    def hidden_score(self) -> dict[str, Any]:
        distance_m = max(0.0, 1.0 - self._progress)
        return {
            "success": self._progress >= 0.86,
            "distance_m": round(distance_m, 3),
            "environment": self.environment,
        }

    def _render_camera(self, path: Path) -> None:
        image = Image.new("RGB", (320, 240), (36, 38, 42))
        draw = ImageDraw.Draw(image)
        palette = {
            "living_room": ((51, 143, 119), (205, 193, 163)),
            "bedroom": ((97, 119, 190), (214, 192, 205)),
            "bathroom": ((86, 163, 181), (222, 232, 230)),
            "kitchen": ((188, 132, 61), (220, 220, 205)),
        }
        accent, floor = palette.get(self.environment, ((92, 158, 118), (210, 205, 190)))
        draw.rectangle((0, 145, 320, 240), fill=floor)
        draw.rectangle((0, 0, 320, 145), fill=(54, 57, 63))
        target_size = int(34 + 70 * self._progress)
        yaw_offset = int(max(-90.0, min(90.0, self.yaw_deg)) * 0.9)
        center_x = 160 - yaw_offset
        center_y = 118
        x0 = max(12, center_x - target_size)
        x1 = min(308, center_x + target_size)
        y0 = max(35, center_y - target_size // 2)
        y1 = min(180, center_y + target_size // 2)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=accent, outline=(240, 240, 235), width=3)
        draw.line((160, 240, center_x, y1), fill=(110, 110, 110), width=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="JPEG", quality=90)

    def _fake_motion_result(self, command: str, *, elapsed_s: float) -> FakeMotionResult:
        self.motion_seq += 1
        frame_paths: list[Path] = []
        for index in range(5):
            path = self.motion_frames_dir / f"{self.motion_seq:04d}_{command}_{index + 1:02d}.jpg"
            self._render_camera(path)
            frame_paths.append(path)
        strip = self.motion_frames_dir / f"{self.motion_seq:04d}_{command}_strip.jpg"
        Image.open(frame_paths[0]).save(strip, format="JPEG", quality=90)
        return FakeMotionResult(
            command=command,
            elapsed_s=elapsed_s,
            motion_frame_paths=frame_paths,
            motion_contact_sheet=strip,
        )


@dataclass(frozen=True)
class _FakeObservation:
    path: Path
    yaw_deg: float
    frame_seq: int
    analysis: Any

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "yaw_deg": self.yaw_deg,
            "frame_seq": self.frame_seq,
            "brightness_center": self.analysis.brightness_center,
        }
