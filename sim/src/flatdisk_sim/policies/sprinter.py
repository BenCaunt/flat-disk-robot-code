"""Sprinter policy: deterministic camera+IMU text-goal navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.text_goal_policy_core import PolicyAction


@dataclass(frozen=True)
class _VisualCue:
    family: str
    score: float
    offset: float
    close: float
    open_center: float
    blocked_center: float
    brightness_center: float
    detected_goal: bool
    wall_like: float
    signature: tuple[float, ...]


class SprinterPolicy:
    """Fast deterministic baseline that races toward cheap visual goal cues."""

    name = "sprinter"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._scan_direction = 1.0
        self._last_signature: tuple[float, ...] | None = None
        self._repeat_count = 0
        self._memory: list[tuple[float, float, float]] = []

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        self._step += 1
        family = _goal_family(prompt)
        cue = _visual_cue(obs, family)
        self._update_repeat_count(cue.signature)
        self._remember(obs.yaw_deg, cue)

        if cue.score >= _strong_threshold(family):
            return self._approach_cue(cue, family, history)

        weak_turn_deadband = 0.07 if family in {"bed", "seat", "toilet"} else 0.18
        if cue.score >= _weak_threshold(family) and abs(cue.offset) > weak_turn_deadband:
            degrees = math.copysign(max(16.0, abs(cue.offset) * 42.0), cue.offset)
            return _turn(
                degrees,
                f"weak_{family}_cue_offset={cue.offset:.2f}_score={cue.score:.2f}",
            )

        remembered_turn = self._turn_toward_memory(obs.yaw_deg)
        if remembered_turn is not None:
            return remembered_turn

        if family == "toilet":
            return self._scan_action(cue, history, family)

        if self._repeat_count >= 2 or cue.blocked_center > 0.66:
            self._scan_direction *= -1.0
            return _turn(32.0 * self._scan_direction, "stale_or_blocked_view_escape_scan")

        if cue.open_center > 0.62 and _recent_turn_count(history) >= 2:
            return _drive(23.0, 0.75, f"open_center_after_scan_{family}")

        return self._scan_action(cue, history, family)

    def _approach_cue(self, cue: _VisualCue, family: str, history: list[dict[str, Any]]) -> PolicyAction:
        drive_streak = _recent_drive_count(history)
        stale_escape = self._repeat_count >= 1 and family not in {"bed", "seat"}
        if drive_streak >= 3 and (stale_escape or cue.close > 0.66):
            direction = cue.offset if abs(cue.offset) > 0.08 else self._scan_direction
            return _turn(direction * 32.0, f"break_repeated_{family}_drive_offset={cue.offset:.2f}")

        if family == "toilet":
            deadband = 0.24
        elif family in {"bed", "seat"}:
            deadband = 0.12
        else:
            deadband = 0.15
        if abs(cue.offset) > deadband:
            degrees = math.copysign(max(16.0, abs(cue.offset) * 38.0), cue.offset)
            return _turn(degrees, f"center_{family}_cue_score={cue.score:.2f}_offset={cue.offset:.2f}")

        duration = 0.62 if cue.close > 0.68 or cue.blocked_center > 0.58 else 0.9
        if family == "toilet" and not cue.detected_goal:
            duration = 0.62
        power = 22.0 if duration < 0.7 else 24.0
        return _drive(power, duration, f"drive_on_{family}_cue_score={cue.score:.2f}_close={cue.close:.2f}")

    def _scan_action(self, cue: _VisualCue, history: list[dict[str, Any]], family: str) -> PolicyAction:
        if _recent_drive_count(history) >= 2 and cue.score < _weak_threshold(family):
            self._scan_direction *= -1.0

        if (
            family == "toilet"
            and cue.score < _weak_threshold(family)
            and cue.open_center > 0.70
            and cue.blocked_center < 0.18
            and _recent_turn_count(history) >= 3
        ):
            return _drive(20.0, 0.6, "toilet_close_range_floor_probe_after_scan")

        if family != "toilet" and self._step % 5 == 0 and cue.open_center > 0.54:
            return _drive(22.0, 0.65, f"probing_open_space_for_{family}")

        direction = self._scan_direction
        if cue.blocked_center > 0.55:
            direction = -1.0 if cue.offset > 0.0 else 1.0
        degrees = 33.0 * direction
        return _turn(degrees, f"scan_for_{family}_score={cue.score:.2f}")

    def _remember(self, yaw_deg: float, cue: _VisualCue) -> None:
        if cue.score < 0.18:
            return
        self._memory.append((_wrap_deg(yaw_deg), cue.score, cue.offset))
        self._memory = sorted(self._memory, key=lambda item: item[1], reverse=True)[:8]

    def _turn_toward_memory(self, yaw_deg: float) -> PolicyAction | None:
        if not self._memory:
            return None
        best_yaw, best_score, best_offset = self._memory[0]
        if best_score < 0.32:
            return None
        error = _wrap_deg(best_yaw - yaw_deg)
        if abs(error) < 13.0 and abs(best_offset) > 0.16:
            return _turn(best_offset * 32.0, f"align_to_best_camera_cue_offset={best_offset:.2f}")
        if abs(error) < 13.0:
            return None
        limited = max(-35.0, min(35.0, error + best_offset * 10.0))
        return _turn(limited, f"return_to_best_camera_cue_score={best_score:.2f}")

    def _update_repeat_count(self, signature: tuple[float, ...]) -> None:
        if self._last_signature is None:
            self._repeat_count = 0
        else:
            delta = sum(abs(a - b) for a, b in zip(signature, self._last_signature)) / len(signature)
            self._repeat_count = self._repeat_count + 1 if delta < 0.035 else 0
        self._last_signature = signature


def _goal_family(prompt: str) -> str:
    text = prompt.lower()
    if _has_word(text, "bed", "mattress", "pillow", "blanket"):
        return "bed"
    if _has_word(text, "sofa", "couch", "loveseat", "chair", "armchair", "ottoman", "recliner"):
        return "seat"
    if _has_word(text, "toilet", "commode"):
        return "toilet"
    if _has_word(text, "sink", "basin", "vanity"):
        return "sink"
    if _has_word(text, "table", "desk", "counter", "countertop", "island"):
        return "table"
    if _has_word(text, "tv", "television", "monitor", "screen"):
        return "screen"
    return "generic"


def _has_word(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _visual_cue(obs: Observation, family: str) -> _VisualCue:
    source = Image.open(obs.path).convert("RGB")
    source_width, source_height = source.size
    image = source.resize((96, 72), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = rgb.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    gray = rgb.mean(axis=2)
    value = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (value - min_channel) / np.maximum(value, 1e-4)
    edge = _edge_strength(gray)
    horizontal_edge = _horizontal_edge(gray)

    score_map = _score_map(family, rgb, value, saturation, edge, horizontal_edge, y)
    score, offset, close = _summarize_score(score_map, x, y)
    if family == "seat":
        fabric_offset, fabric_score = _broad_upholstery_hint(rgb, value, saturation, horizontal_edge, y)
        if fabric_score > 0.28:
            offset = fabric_offset * 0.82 + offset * 0.18
            score = max(score, fabric_score)
    open_center = _open_center_score(value, saturation, edge, y, x)
    blocked_center = _blocked_center_score(value, edge, y, x)
    brightness_center = float(obs.analysis.brightness_center)
    wall_like = _wall_like_score(value, saturation, edge, y, x)
    detected_goal = False

    for det in _matching_goal_detections(obs.analysis.detections, family, source_width, source_height):
        detected_goal = True
        if family in {"toilet", "sink"}:
            score = max(score, min(1.0, det.confidence * 1.15))
            offset = det.center_offset
            close = max(close, min(1.0, det.area_fraction * 30.0))
        elif family == "bed":
            score = max(score, min(1.0, det.confidence * 0.8))
            offset = (offset + det.center_offset) * 0.5

    if family == "toilet" and not detected_goal:
        score *= max(0.08, 1.0 - wall_like * 0.92)
        close *= max(0.35, 1.0 - wall_like * 0.55)

    signature = _signature(rgb, edge)
    return _VisualCue(
        family=family,
        score=score,
        offset=offset,
        close=close,
        open_center=open_center,
        blocked_center=blocked_center,
        brightness_center=brightness_center,
        detected_goal=detected_goal,
        wall_like=wall_like,
        signature=signature,
    )


def _score_map(
    family: str,
    rgb: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    edge: np.ndarray,
    horizontal_edge: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    row_mid = _row_weight(y, 0.16, 0.66)
    row_low_mid = _row_weight(y, 0.30, 0.82)
    row_upper_mid = _row_weight(y, 0.10, 0.58)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    white_object = (value > 0.70) & (saturation < 0.22)
    tan_fabric = (red > 0.34) & (green > 0.28) & (blue > 0.18) & (value < 0.88) & (saturation < 0.55)
    muted_fabric = (value > 0.16) & (value < 0.74) & (saturation < 0.60)
    dark_object = (value > 0.08) & (value < 0.34) & (edge > 0.025)
    colored_cover = (saturation > 0.22) & (value > 0.28) & (value < 0.95)

    if family == "toilet":
        warm_ceramic = (
            (value > 0.52)
            & (saturation < 0.42)
            & (red > blue * 0.92)
            & (green > blue * 0.82)
        )
        ceramic = white_object | warm_ceramic
        structured_ceramic = ceramic & ((edge > 0.014) | (horizontal_edge > 0.012))
        ceramic_edges = structured_ceramic.astype(np.float32) * (0.16 + edge * 5.2 + horizontal_edge * 3.2)
        lower_shape = ceramic.astype(np.float32) * horizontal_edge * 2.4
        return ((ceramic_edges + lower_shape) * row_low_mid).astype(np.float32)
    if family == "sink":
        return ((white_object.astype(np.float32) * (edge * 5.0)) * row_mid).astype(np.float32)
    if family == "bed":
        patterned_fabric = colored_cover.astype(np.float32) * (0.35 + edge * 3.0) * row_upper_mid
        bed_band = horizontal_edge * (0.25 + saturation) * row_upper_mid * 3.2
        under_gap = dark_object.astype(np.float32) * row_mid * 0.35
        return (bed_band + under_gap).astype(np.float32)
    if family == "seat":
        sofa_fabric = (value > 0.30) & (value < 0.72) & (saturation < 0.34)
        dark_upholstery = (value > 0.13) & (value < 0.50) & (saturation < 0.48)
        seat_rows = _row_weight(y, 0.12, 0.62)
        body_rows = _row_weight(y, 0.18, 0.66)
        fabric_edge = sofa_fabric.astype(np.float32) * (horizontal_edge * 6.0 + edge * 0.9)
        fabric_mass = sofa_fabric.astype(np.float32) * 0.08
        dark_body = dark_upholstery.astype(np.float32) * (horizontal_edge * 4.2 + edge * 1.1 + 0.07) * body_rows
        seat_body = (fabric_edge + fabric_mass) * seat_rows + dark_body
        return seat_body.astype(np.float32)
    if family == "table":
        legs = dark_object.astype(np.float32) * row_low_mid * 0.45
        tops = horizontal_edge * row_mid * 3.2
        return (legs + tops).astype(np.float32)
    if family == "screen":
        screen = ((value < 0.24) & (saturation < 0.55)).astype(np.float32) * row_mid
        return (screen * (0.4 + edge * 3.2)).astype(np.float32)

    salient = ((edge * 2.8) + (colored_cover.astype(np.float32) * 0.18)) * row_mid
    return salient.astype(np.float32)


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.clip(gx + gy, 0.0, 1.0)


def _horizontal_edge(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(np.diff(gray, axis=0, prepend=gray[:1, :])), 0.0, 1.0)


def _row_weight(y: np.ndarray, low: float, high: float) -> np.ndarray:
    return ((y >= low) & (y <= high)).astype(np.float32)


def _summarize_score(score_map: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    total = float(score_map.sum())
    if total <= 1e-5:
        return 0.0, 0.0, 0.0

    column = score_map.sum(axis=0)
    smooth = np.convolve(column, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    best = float(smooth.max())
    height = score_map.shape[0]
    score = min(1.0, best / max(1.0, height * 0.22))
    offset = float((score_map * x).sum() / total)
    low_total = float((score_map * (y > 0.56)).sum())
    close = min(1.0, low_total / max(1.0, total * 0.66))
    return score, max(-1.0, min(1.0, offset)), close


def _wall_like_score(value: np.ndarray, saturation: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_upper = (np.abs(x) < 0.52) & (y > 0.12) & (y < 0.70)
    if not np.any(center_upper):
        return 0.0
    bright_plain = (value > 0.68) & (saturation < 0.24) & (edge < 0.035) & center_upper
    plain_ratio = float(bright_plain.sum()) / float(center_upper.sum())
    center_edge = float(edge[center_upper].mean())
    return max(0.0, min(1.0, plain_ratio * 1.15 + max(0.0, 0.06 - center_edge) * 3.0))


def _broad_upholstery_hint(
    rgb: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    horizontal_edge: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    rows = (y >= 0.14) & (y <= 0.62)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    neutral_fabric = (value > 0.30) & (value < 0.72) & (saturation < 0.34) & rows
    blue_gray_fabric = (
        (value > 0.16)
        & (value < 0.62)
        & (saturation < 0.55)
        & (blue >= red * 0.72)
        & (blue >= green * 0.62)
        & rows
    )
    structured = (neutral_fabric & (horizontal_edge > 0.007)) | (blue_gray_fabric & (horizontal_edge > 0.004))
    column = (
        structured.sum(axis=0).astype(np.float32)
        + blue_gray_fabric.sum(axis=0).astype(np.float32) * 0.35
        + neutral_fabric.sum(axis=0).astype(np.float32) * 0.12
    )
    if float(column.sum()) <= 1.0:
        return 0.0, 0.0
    smooth = np.convolve(column, np.ones(15, dtype=np.float32) / 15.0, mode="same")
    score = min(1.0, float(smooth.max()) / 8.0)
    column_x = np.linspace(-1.0, 1.0, len(column), dtype=np.float32)
    offset = float((column * column_x).sum() / max(1.0, float(column.sum())))
    return max(-1.0, min(1.0, offset)), score


def _open_center_score(value: np.ndarray, saturation: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_floor = (np.abs(x) < 0.36) & (y > 0.56)
    floor_like = center_floor & (value > 0.18) & (edge < 0.16)
    if not np.any(center_floor):
        return 0.0
    texture = 1.0 - min(1.0, float(edge[center_floor].mean()) * 4.0)
    brightness = min(1.0, float(value[floor_like].mean()) * 1.35) if np.any(floor_like) else 0.0
    saturation_bonus = 1.0 - min(1.0, float(saturation[center_floor].mean()))
    return max(0.0, min(1.0, brightness * 0.55 + texture * 0.30 + saturation_bonus * 0.15))


def _blocked_center_score(value: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_low = (np.abs(x) < 0.32) & (y > 0.46)
    if not np.any(center_low):
        return 0.0
    dark_mass = float(((value < 0.22) & center_low).sum()) / float(center_low.sum())
    high_edge = min(1.0, float(edge[center_low].mean()) * 5.0)
    very_bright_wall = float(((value > 0.88) & (edge < 0.04) & center_low).sum()) / float(center_low.sum())
    return max(0.0, min(1.0, dark_mass * 0.65 + high_edge * 0.25 + very_bright_wall * 0.35))


def _matching_goal_detections(detections: tuple[Any, ...], family: str, width: int, height: int) -> list[Any]:
    matches: list[Any] = []
    for det in detections:
        if family == "toilet" and det.name != "toilet":
            continue
        if family == "sink" and det.name != "toilet":
            continue
        if family not in {"toilet", "sink"}:
            continue
        x0, y0, x1, y1 = det.bbox
        center_y = ((y0 + y1) * 0.5) / max(1.0, float(height))
        if det.confidence < 0.35 or det.area_fraction < 0.004:
            continue
        if family == "toilet" and not (0.28 <= center_y <= 0.88):
            continue
        if (x1 - x0 + 1) * (y1 - y0 + 1) > width * height * 0.16:
            continue
        matches.append(det)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def _signature(rgb: np.ndarray, edge: np.ndarray) -> tuple[float, ...]:
    small = rgb.reshape(6, 12, 8, 12, 3).mean(axis=(1, 3, 4)).reshape(-1)
    edge_rows = edge.reshape(6, 12, 8, 12).mean(axis=(1, 3)).reshape(-1)
    values = np.concatenate([small, edge_rows])
    return tuple(round(float(v), 3) for v in values)


def _strong_threshold(family: str) -> float:
    if family in {"bed", "table"}:
        return 0.30
    if family in {"toilet", "sink"}:
        return 0.30
    return 0.38


def _weak_threshold(family: str) -> float:
    if family in {"toilet", "sink"}:
        return 0.18
    if family == "bed":
        return 0.12
    return 0.22


def _recent_turn_count(history: list[dict[str, Any]]) -> int:
    return sum(1 for item in history[-3:] if item.get("action") == "turn_by_angle")


def _recent_drive_count(history: list[dict[str, Any]]) -> int:
    return sum(1 for item in history[-3:] if item.get("action") == "drive_straight")


def _turn(degrees: float, reason: str) -> PolicyAction:
    return PolicyAction(action="turn_by_angle", degrees=max(-35.0, min(35.0, float(degrees))), reason=reason[:240])


def _drive(power_percent: float, duration_s: float, reason: str) -> PolicyAction:
    return PolicyAction(
        action="drive_straight",
        power_percent=max(20.0, min(24.0, float(power_percent))),
        duration_s=max(0.6, min(0.9, float(duration_s))),
        reason=reason[:240],
    )


def _wrap_deg(angle_deg: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))))
