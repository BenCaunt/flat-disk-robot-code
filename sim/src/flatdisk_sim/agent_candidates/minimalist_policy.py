"""RealBot Minimalist candidate for camera+IMU text-goal navigation.

This policy is intentionally deterministic and sensor-bound. It consumes only
the latest RGB image, camera-derived analysis, IMU yaw, the text goal, and its
own prior action history. It does not consume simulator pose, object metadata,
target distance, wheel encoders, collision flags, or evaluator success state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.text_goal_policy_core import PolicyAction


FORBIDDEN_PRIVILEGED_INPUTS = frozenset(
    {
        "distance_m",
        "hidden_score",
        "nearest_target",
        "object_metadata",
        "objects",
        "pose",
        "scene",
        "success_radius",
        "target_pose",
        "thor",
        "wheel_encoder",
        "wheel_encoders",
        "odometry",
    }
)


@dataclass(frozen=True)
class MinimalistCue:
    family: str
    score: float
    offset: float
    close: float
    open_center: float
    blocked_center: float
    brightness_center: float
    source: str
    signature: tuple[float, ...]


class RealBotMinimalistPolicy:
    """Fast policy candidate that favors bounded motion and auditable stops."""

    name = "minimalist_realbot"

    def __init__(self, *, allow_stop: bool = False, stop_window: int = 3) -> None:
        self.allow_stop = allow_stop
        self.stop_window = max(3, int(stop_window))
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._scan_direction = 1.0
        self._last_signature: tuple[float, ...] | None = None
        self._repeat_count = 0
        self._cue_history: list[MinimalistCue] = []
        self._best_sighting: tuple[float, float, float] | None = None

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        safe_history = sanitize_action_history(history)
        cue = camera_imu_cue(obs, prompt)
        self._update_repeat_count(cue.signature)
        self._remember_sighting(obs.yaw_deg, cue)
        self._cue_history.append(cue)
        self._cue_history = self._cue_history[-self.stop_window :]
        self._step += 1

        if self._should_stop():
            return PolicyAction(
                action="stop",
                success=True,
                reason="stable_close_centered_visual_evidence",
            )

        if cue.score >= _strong_threshold(cue.family):
            return self._approach(cue, safe_history)

        if cue.score >= _weak_threshold(cue.family) and abs(cue.offset) > _weak_turn_deadband(cue.family):
            return _turn(cue.offset * 30.0, f"weak_visual_recenter:{cue.family}:score={cue.score:.2f}")

        memory_action = self._turn_toward_best_sighting(obs.yaw_deg)
        if memory_action is not None and _recent_turn_count(safe_history, window=3) < 2:
            return memory_action

        return self._search(cue, safe_history)

    def _approach(self, cue: MinimalistCue, history: list[dict[str, Any]]) -> PolicyAction:
        deadband = _center_deadband(cue.family)
        if abs(cue.offset) > deadband:
            return _turn(cue.offset * 34.0, f"center_strong_visual_cue:{cue.family}:offset={cue.offset:.2f}")

        if _recent_drive_count(history, window=3) >= 3 and (self._repeat_count >= 1 or cue.close > 0.72):
            self._scan_direction *= -1.0
            return _turn(24.0 * self._scan_direction, f"break_repeated_drive_without_new_view:{cue.family}")

        duration = 0.6 if cue.close > 0.62 or cue.blocked_center > 0.58 else 0.75
        power = 20.0 if duration <= 0.6 else 22.0
        return _drive(power, duration, f"bounded_approach:{cue.family}:score={cue.score:.2f}:close={cue.close:.2f}")

    def _search(self, cue: MinimalistCue, history: list[dict[str, Any]]) -> PolicyAction:
        recent_turns = _recent_turn_count(history, window=3)
        recent_drives = _recent_drive_count(history, window=3)
        if recent_drives >= 2 and cue.score < _weak_threshold(cue.family):
            self._scan_direction *= -1.0
            return _turn(28.0 * self._scan_direction, f"rescan_after_blind_drives:{cue.family}")

        if self._repeat_count >= 2:
            self._scan_direction *= -1.0
            return _turn(30.0 * self._scan_direction, "escape_stale_camera_view")

        if recent_turns >= 2:
            self._scan_direction *= -1.0
            if cue.open_center > 0.48 and cue.blocked_center < 0.55:
                return _drive(20.0, 0.6, f"short_probe_after_scan:{cue.family}")
            return _turn(26.0 * self._scan_direction, f"reverse_scan_direction:{cue.family}")

        if cue.blocked_center > 0.62:
            direction = -1.0 if cue.offset > 0.0 else 1.0
            self._scan_direction = direction
            return _turn(30.0 * direction, f"avoid_center_blockage:{cue.family}")

        if self._step % 5 == 0 and cue.open_center > 0.58:
            return _drive(20.0, 0.6, f"periodic_open_center_probe:{cue.family}")

        return _turn(24.0 * self._scan_direction, f"bounded_visual_scan:{cue.family}:score={cue.score:.2f}")

    def _should_stop(self) -> bool:
        if not self.allow_stop or len(self._cue_history) < self.stop_window:
            return False
        recent = self._cue_history[-self.stop_window :]
        family = recent[-1].family
        threshold = _strong_threshold(family)
        if not all(cue.score >= threshold for cue in recent):
            return False
        if not all(abs(cue.offset) <= _center_deadband(family) for cue in recent):
            return False
        close_values = [cue.close for cue in recent]
        if min(close_values) < 0.56:
            return False
        if close_values[-1] + 0.08 < close_values[0]:
            return False
        return True

    def _remember_sighting(self, yaw_deg: float, cue: MinimalistCue) -> None:
        if cue.score < _weak_threshold(cue.family):
            return
        current = self._best_sighting
        weighted_score = cue.score + max(0.0, 0.45 - abs(cue.offset)) * 0.18
        if current is None or weighted_score > current[1]:
            self._best_sighting = (_wrap_deg(yaw_deg), weighted_score, cue.offset)

    def _turn_toward_best_sighting(self, yaw_deg: float) -> PolicyAction | None:
        if self._best_sighting is None:
            return None
        best_yaw, best_score, best_offset = self._best_sighting
        if best_score < 0.34:
            return None
        yaw_error = _wrap_deg(best_yaw - yaw_deg)
        if abs(yaw_error) < 10.0 and abs(best_offset) < 0.18:
            return None
        correction = yaw_error + best_offset * 12.0
        return _turn(correction, f"return_to_best_visual_sighting:score={best_score:.2f}")

    def _update_repeat_count(self, signature: tuple[float, ...]) -> None:
        if self._last_signature is None:
            self._repeat_count = 0
        else:
            delta = sum(abs(a - b) for a, b in zip(signature, self._last_signature)) / max(1, len(signature))
            self._repeat_count = self._repeat_count + 1 if delta < 0.025 else 0
        self._last_signature = signature


def sanitize_action_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only policy-owned action fields before using recent history."""

    allowed = {"action", "degrees", "duration_s", "power_percent", "reason"}
    safe: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        safe.append({key: item[key] for key in allowed if key in item})
    return safe


def camera_imu_cue(obs: Observation, prompt: str) -> MinimalistCue:
    """Build a visual cue from the current camera image and IMU-compatible metadata."""

    family = _goal_family(prompt)
    image = Image.open(obs.path).convert("RGB")
    width, height = image.size
    resized = image.resize((96, 72), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    value = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (value - min_channel) / np.maximum(value, 1e-4)
    gray = rgb.mean(axis=2)
    edge = _edge(gray)
    horizontal = _horizontal_edge(gray)
    y = np.linspace(0.0, 1.0, rgb.shape[0], dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, rgb.shape[1], dtype=np.float32)[None, :]

    score_map = _goal_score_map(family, rgb, value, saturation, edge, horizontal, y)
    score, offset, close = _summarize(score_map, x, y)
    source = "image_heuristic"

    detection = _matching_detection(obs, family, width, height)
    if detection is not None:
        det_score = min(1.0, float(detection.confidence) * 1.15)
        det_close = min(1.0, float(detection.area_fraction) * 28.0)
        score = max(score, det_score)
        offset = float(detection.center_offset)
        close = max(close, det_close)
        source = f"detection:{detection.name}"

    open_center = _open_center(value, saturation, edge, y, x)
    blocked_center = _blocked_center(value, edge, y, x)
    return MinimalistCue(
        family=family,
        score=max(0.0, min(1.0, score)),
        offset=max(-1.0, min(1.0, offset)),
        close=max(0.0, min(1.0, close)),
        open_center=open_center,
        blocked_center=blocked_center,
        brightness_center=float(obs.analysis.brightness_center),
        source=source,
        signature=_signature(rgb, edge),
    )


def sensor_contract() -> dict[str, Any]:
    return {
        "allowed": [
            "latest RGB camera frame",
            "camera-derived frame analysis",
            "IMU yaw",
            "text goal",
            "policy-owned action history",
        ],
        "forbidden": sorted(FORBIDDEN_PRIVILEGED_INPUTS),
        "motion": "timed bounded turns/drives; no wheel encoder dependency",
        "stop": "optional, only after repeated close centered visual evidence",
    }


def _goal_family(prompt: str) -> str:
    text = prompt.lower()
    if _has_word(text, "bed", "mattress", "pillow", "blanket"):
        return "bed"
    if _has_word(text, "sofa", "couch", "loveseat", "chair", "armchair", "recliner", "ottoman"):
        return "seat"
    if _has_word(text, "toilet", "commode"):
        return "toilet"
    if _has_word(text, "sink", "basin", "vanity"):
        return "sink"
    if _has_word(text, "table", "desk", "counter", "countertop"):
        return "table"
    return "generic"


def _has_word(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _goal_score_map(
    family: str,
    rgb: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    edge: np.ndarray,
    horizontal: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mid = ((y > 0.14) & (y < 0.72)).astype(np.float32)
    lower_mid = ((y > 0.28) & (y < 0.86)).astype(np.float32)
    upper_mid = ((y > 0.10) & (y < 0.62)).astype(np.float32)

    neutral_object = (value > 0.24) & (value < 0.78) & (saturation < 0.42)
    pale_object = (value > 0.62) & (saturation < 0.30)
    colored_fabric = (value > 0.18) & (value < 0.92) & (saturation > 0.12) & (saturation < 0.70)

    if family == "toilet":
        ceramic = pale_object | ((red > blue * 0.92) & (green > blue * 0.86) & (value > 0.50) & (saturation < 0.44))
        return (ceramic.astype(np.float32) * (horizontal * 3.4 + edge * 2.1 + 0.03) * lower_mid).astype(np.float32)
    if family == "sink":
        return (pale_object.astype(np.float32) * (horizontal * 2.6 + edge * 2.2 + 0.02) * mid).astype(np.float32)
    if family == "bed":
        broad_band = colored_fabric.astype(np.float32) * (horizontal * 3.1 + edge * 0.7 + 0.04) * upper_mid
        neutral_band = neutral_object.astype(np.float32) * horizontal * 2.0 * upper_mid
        return (broad_band + neutral_band).astype(np.float32)
    if family == "seat":
        fabric = neutral_object | ((blue > red * 0.72) & (green > red * 0.58) & (value < 0.72) & (saturation < 0.58))
        return (fabric.astype(np.float32) * (horizontal * 3.8 + edge * 0.9 + 0.05) * mid).astype(np.float32)
    if family == "table":
        dark_shape = (value > 0.08) & (value < 0.40) & (edge > 0.018)
        return ((dark_shape.astype(np.float32) * 0.20 + horizontal * 2.5) * mid).astype(np.float32)
    return ((edge * 2.4 + colored_fabric.astype(np.float32) * 0.10) * mid).astype(np.float32)


def _matching_detection(obs: Observation, family: str, width: int, height: int) -> Any | None:
    matches: list[Any] = []
    for detection in obs.analysis.detections:
        if family in {"toilet", "sink"} and detection.name != "toilet":
            continue
        if family not in {"toilet", "sink"}:
            continue
        x0, y0, x1, y1 = detection.bbox
        center_y = ((y0 + y1) * 0.5) / max(1.0, float(height))
        box_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
        if detection.confidence < 0.28 or detection.area_fraction < 0.003:
            continue
        if not 0.22 <= center_y <= 0.92:
            continue
        if box_area > width * height * 0.20:
            continue
        matches.append(detection)
    if not matches:
        return None
    return max(matches, key=lambda item: item.confidence)


def _summarize(score_map: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    total = float(score_map.sum())
    if total <= 1e-6:
        return 0.0, 0.0, 0.0
    column = score_map.sum(axis=0)
    smooth = np.convolve(column, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    score = min(1.0, float(smooth.max()) / max(1.0, score_map.shape[0] * 0.20))
    offset = float((score_map * x).sum() / total)
    close = min(1.0, float((score_map * (y > 0.54)).sum()) / max(1.0, total * 0.62))
    return score, offset, close


def _edge(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.clip(gx + gy, 0.0, 1.0)


def _horizontal_edge(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(np.diff(gray, axis=0, prepend=gray[:1, :])), 0.0, 1.0)


def _open_center(value: np.ndarray, saturation: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_floor = (np.abs(x) < 0.34) & (y > 0.56)
    if not np.any(center_floor):
        return 0.0
    low_edge = 1.0 - min(1.0, float(edge[center_floor].mean()) * 4.0)
    brightness = min(1.0, float(value[center_floor].mean()) * 1.25)
    muted = 1.0 - min(1.0, float(saturation[center_floor].mean()))
    return max(0.0, min(1.0, brightness * 0.52 + low_edge * 0.32 + muted * 0.16))


def _blocked_center(value: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_low = (np.abs(x) < 0.32) & (y > 0.46)
    if not np.any(center_low):
        return 0.0
    dark_mass = float(((value < 0.22) & center_low).sum()) / float(center_low.sum())
    edge_mass = min(1.0, float(edge[center_low].mean()) * 5.0)
    bright_plain = float(((value > 0.90) & (edge < 0.04) & center_low).sum()) / float(center_low.sum())
    return max(0.0, min(1.0, dark_mass * 0.58 + edge_mass * 0.28 + bright_plain * 0.30))


def _signature(rgb: np.ndarray, edge: np.ndarray) -> tuple[float, ...]:
    color_blocks = rgb.reshape(6, 12, 8, 12, 3).mean(axis=(1, 3, 4)).reshape(-1)
    edge_blocks = edge.reshape(6, 12, 8, 12).mean(axis=(1, 3)).reshape(-1)
    values = np.concatenate([color_blocks, edge_blocks])
    return tuple(round(float(value), 3) for value in values)


def _strong_threshold(family: str) -> float:
    if family in {"toilet", "sink"}:
        return 0.36
    if family == "bed":
        return 0.28
    if family == "seat":
        return 0.34
    return 0.40


def _weak_threshold(family: str) -> float:
    if family in {"toilet", "sink"}:
        return 0.18
    if family == "bed":
        return 0.12
    return 0.20


def _center_deadband(family: str) -> float:
    if family in {"toilet", "sink"}:
        return 0.20
    if family in {"bed", "seat"}:
        return 0.14
    return 0.16


def _weak_turn_deadband(family: str) -> float:
    if family in {"bed", "seat"}:
        return 0.08
    return 0.13


def _recent_turn_count(history: list[dict[str, Any]], *, window: int) -> int:
    return sum(1 for item in history[-window:] if item.get("action") == "turn_by_angle")


def _recent_drive_count(history: list[dict[str, Any]], *, window: int) -> int:
    return sum(1 for item in history[-window:] if item.get("action") == "drive_straight")


def _turn(degrees: float, reason: str) -> PolicyAction:
    degrees = max(-35.0, min(35.0, float(degrees)))
    if abs(degrees) < 12.0:
        degrees = math.copysign(12.0, degrees or 1.0)
    return PolicyAction(action="turn_by_angle", degrees=degrees, reason=reason[:240])


def _drive(power_percent: float, duration_s: float, reason: str) -> PolicyAction:
    return PolicyAction(
        action="drive_straight",
        power_percent=max(18.0, min(24.0, float(power_percent))),
        duration_s=max(0.45, min(0.9, float(duration_s))),
        reason=reason[:240],
    )


def _wrap_deg(angle_deg: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))))
