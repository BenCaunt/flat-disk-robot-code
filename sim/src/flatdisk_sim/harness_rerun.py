"""Rerun logging adapter for the LLM robot-control harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODE_VALUES = ["idle", "auto", "paused", "teleop", "complete", "error"]
MODE_LABELS = ["Idle", "Auto", "Paused", "Teleop", "Complete", "Error"]
MODE_COLORS = [0x8A8F98FF, 0x2E7D6BFF, 0xD6A21DFF, 0x3B73D9FF, 0x4CAF50FF, 0xD94C4CFF]


class HarnessRerunLogger:
    def __init__(
        self,
        *,
        recording_id: str,
        save_path: Path | None = None,
        spawn: bool = False,
        rr_module: Any | None = None,
    ) -> None:
        self.save_path = save_path
        self.enabled = False
        self.rr: Any | None = rr_module
        self._injected_rr_module = rr_module is not None
        if self.rr is None:
            try:
                import rerun as rr  # type: ignore
            except ImportError:
                self.rr = None
                return
            self.rr = rr
        self.rr.init(recording_id, spawn=spawn)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.rr.save(str(save_path))
        self.enabled = True
        self._send_blueprint()
        self._log_state_configuration()

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.enabled or self.rr is None:
            return
        text = json.dumps(metadata, indent=2, sort_keys=True, default=str)
        if hasattr(self.rr, "TextDocument"):
            self.rr.log("harness/metadata", self.rr.TextDocument(text), static=True)
        elif hasattr(self.rr, "TextLog"):
            self.rr.log("harness/metadata", self.rr.TextLog(text), static=True)

    def log_state(self, step: int, mode: str) -> None:
        if not self.enabled or self.rr is None:
            return
        self.rr.set_time("step", sequence=int(step))
        if hasattr(self.rr, "StateChange"):
            self.rr.log("robot/mode", self.rr.StateChange(state=mode))
        else:
            self.log_event(step, "mode", {"mode": mode})

    def log_observation(self, step: int, observation: dict[str, Any]) -> None:
        if not self.enabled or self.rr is None:
            return
        self.rr.set_time("step", sequence=int(step))
        yaw = observation.get("yaw_deg")
        if yaw is not None and hasattr(self.rr, "Scalars"):
            self.rr.log("robot/imu/yaw_deg", self.rr.Scalars(float(yaw)))
        image_path = observation.get("path")
        if image_path and hasattr(self.rr, "Image"):
            try:
                import numpy as np
                from PIL import Image

                image = Image.open(image_path).convert("RGB")
                self.rr.log("robot/camera/rgb", self.rr.Image(np.asarray(image)))
            except Exception:  # noqa: BLE001 - image logging should not break control.
                self.log_event(step, "image_log_error", {"path": image_path})
        self.log_event(step, "observation", observation)

    def log_command(self, step: int, source: str, action: dict[str, Any]) -> None:
        if self.enabled and self.rr is not None:
            self.rr.set_time("step", sequence=int(step))
            text = json.dumps({"source": source, "action": action}, sort_keys=True, default=str)
            self._log_text("robot/commands", text)
        self.log_event(step, "command", {"source": source, "action": action})

    def log_llm(self, step: int, role: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.rr is None:
            return
        self.rr.set_time("step", sequence=int(step))
        text = json.dumps({"role": role, **payload}, sort_keys=True, default=str)
        self._log_text(f"harness/llm/{role}", text)

    def log_event(self, step: int, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.rr is None:
            return
        self.rr.set_time("step", sequence=int(step))
        text = json.dumps({"event": event, **payload}, sort_keys=True, default=str)
        self._log_text("harness/events", text)

    def _log_text(self, entity_path: str, text: str, *, static: bool = False) -> None:
        if not self.enabled or self.rr is None:
            return
        if hasattr(self.rr, "TextLog"):
            self.rr.log(entity_path, self.rr.TextLog(text), static=static)
        elif hasattr(self.rr, "TextDocument"):
            self.rr.log(entity_path, self.rr.TextDocument(text), static=static)

    def close(self) -> None:
        if not self.enabled or self.rr is None:
            return
        if hasattr(self.rr, "shutdown"):
            self.rr.shutdown()

    def _send_blueprint(self) -> None:
        if not self.enabled or self.rr is None or not hasattr(self.rr, "send_blueprint"):
            return
        try:
            import rerun.blueprint as rrb  # type: ignore

            self.rr.send_blueprint(
                rrb.Blueprint(
                    rrb.StateTimelineView(origin="/robot", name="Robot mode"),
                    rrb.Spatial2DView(origin="/robot/camera", name="Camera"),
                    rrb.TimeSeriesView(origin="/robot/imu", name="IMU yaw"),
                    rrb.TextLogView(origin="/robot/commands", name="Commands"),
                    rrb.TextLogView(origin="/harness/llm", name="LLM actor/critic"),
                    rrb.TextLogView(origin="/harness", name="Harness events"),
                    rrb.TextDocumentView(origin="/harness/metadata", name="Run metadata"),
                    auto_layout=True,
                    collapse_panels=True,
                )
            )
        except Exception:  # noqa: BLE001 - blueprint setup should not block logging.
            if self._injected_rr_module:
                try:
                    self.rr.send_blueprint(None)
                except Exception:
                    pass
            return

    def _log_state_configuration(self) -> None:
        if not self.enabled or self.rr is None or not hasattr(self.rr, "StateConfiguration"):
            return
        self.rr.log(
            "robot/mode",
            self.rr.StateConfiguration(values=MODE_VALUES, labels=MODE_LABELS, colors=MODE_COLORS),
            static=True,
        )
