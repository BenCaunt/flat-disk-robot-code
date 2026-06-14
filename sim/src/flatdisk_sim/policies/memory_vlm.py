"""Cartographer policy: online VLM navigation with non-privileged memory."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.text_goal_policy_core import PolicyAction, parse_json_object, validate_action


@dataclass(frozen=True)
class _FastCue:
    family: str
    score: float
    offset: float
    close: float
    center_open: float
    center_blocked: float
    note: str


class MemoryVlmPolicy:
    """Camera+IMU text-goal policy with ego-centric visual memory."""

    name = "memory_vlm"

    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        allow_stop: bool = False,
        memory_limit: int = 10,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Install with `uv sync --extra vlm` to use MemoryVlmPolicy") from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.allow_stop = allow_stop
        self.memory_limit = max(3, int(memory_limit))
        self.reset()

    def reset(self) -> None:
        self._step_index = 0
        self._visual_memory: list[dict[str, Any]] = []
        self._action_memory: list[dict[str, Any]] = []
        self._fast_plan: list[PolicyAction] = []
        self._vlm_calls = 0
        self._last_goal: str | None = None

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        if self._last_goal != prompt:
            self.reset()
            self._last_goal = prompt

        safe_history = self._sanitize_history(history)
        fast_action, fast_payload = self._choose_fast_action(obs, prompt, safe_history)
        if fast_action is not None:
            self._remember(obs, fast_payload, fast_action)
            self._step_index += 1
            return fast_action

        image_b64 = base64.b64encode(obs.path.read_bytes()).decode("ascii")
        detections = self._camera_derived_detections(obs)
        visual_brief = self._visual_memory[-self.memory_limit :]
        anti_oscillation = self._anti_oscillation_hint()
        best_sighting = self._best_goal_sighting(obs.yaw_deg)

        instructions = (
            "You are Cartographer, a policy for a small two-wheel flat disk robot. "
            "You must act on the same information available on the real robot: the current low RGB camera image, "
            "IMU yaw, camera-derived visual summaries, the natural-language goal, and your own prior visual/action memory. "
            "Never rely on or infer from simulator hidden pose, object coordinates, maps, collision flags, status topics, "
            "wheel encoders, odometry, privileged target distance, or episode scoring metadata. "
            "The goal is a natural-language navigation target, not text printed in the scene unless reading text is requested. "
            "Use memory to avoid forgetting where promising views were seen. If the target was seen at a prior yaw, "
            "turn back toward that remembered bearing when it disappears. Avoid repeated tiny turns and left-right oscillations; "
            "prefer one decisive bounded scan direction or a short forward command when the target is centered and reachable. "
            "Do not declare success or stop unless explicitly enabled; an external evaluator or operator will decide completion. "
            "Output exactly one JSON object. Required navigation action is one of: "
            "{\"action\":\"turn_by_angle\",\"degrees\":-35..35,\"reason\":\"...\"}, "
            "{\"action\":\"drive_straight\",\"power_percent\":20..24,\"duration_s\":0.6..0.9,\"reason\":\"...\"}. "
            "Also include these memory fields in the same JSON object: "
            "\"visual_note\" as a short description of the current view, "
            "\"goal_visible\" as true or false, "
            "\"goal_bearing\" as one of \"left\", \"center\", \"right\", or \"unknown\", "
            "\"confidence\" as 0..1. Do not include prose outside JSON."
        )
        user_text = (
            f"Goal: {prompt}\n"
            f"Step: {self._step_index}\n"
            f"Current IMU yaw: {obs.yaw_deg:.1f} degrees\n"
            f"Camera-derived detections: {json.dumps(detections, sort_keys=True)}\n"
            f"Recent allowed action history: {json.dumps(safe_history[-6:], sort_keys=True)}\n"
            f"Your recent visual memory: {json.dumps(visual_brief, sort_keys=True)}\n"
            f"Best remembered goal sighting: {json.dumps(best_sighting, sort_keys=True)}\n"
            f"Anti-oscillation hint: {anti_oscillation}\n"
            "Choose the next real-robot-compatible command."
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instructions + "\n\n" + user_text},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }
            ],
        )
        payload = parse_json_object(response.output_text)
        action = self._stabilize_action(validate_action(payload, allow_stop=self.allow_stop), payload, obs.yaw_deg)
        self._queue_approach_followups(action, payload, prompt)
        self._remember(obs, payload, action)
        self._step_index += 1
        self._vlm_calls += 1
        return action

    def _sanitize_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed = {"action", "degrees", "power_percent", "duration_s", "reason"}
        safe: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            safe.append({key: item[key] for key in allowed if key in item})
        return safe

    def _camera_derived_detections(self, obs: Observation) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for detection in obs.analysis.detections[:6]:
            detections.append(
                {
                    "name": detection.name,
                    "confidence": round(float(detection.confidence), 3),
                    "area_fraction": round(float(detection.area_fraction), 4),
                    "center_offset": round(float(detection.center_offset), 3),
                }
            )
        return detections

    def _choose_fast_action(
        self,
        obs: Observation,
        prompt: str,
        history: list[dict[str, Any]],
    ) -> tuple[PolicyAction | None, dict[str, Any]]:
        family = _goal_family(prompt)
        cue = _fast_visual_cue(obs, family)

        if self._fast_plan:
            action = self._fast_plan.pop(0)
            payload = self._fast_payload(cue, f"fast_plan:{action.reason}")
            return action, payload

        if family in {"seat", "bed"} and self._should_start_side_approach(cue, history):
            self._fast_plan = self._side_approach_plan(family, cue)
            action = self._fast_plan.pop(0)
            payload = self._fast_payload(cue, f"start_side_approach:{action.reason}")
            return action, payload

        if family in {"seat", "bed"} and cue.score >= 0.62 and self._step_index >= 5:
            if abs(cue.offset) > 0.24:
                action = PolicyAction(
                    action="turn_by_angle",
                    degrees=max(-30.0, min(30.0, cue.offset * 26.0)),
                    reason=f"fast_recenter_{family}_cue:{cue.note}",
                )
            else:
                action = PolicyAction(
                    action="drive_straight",
                    power_percent=23.0,
                    duration_s=0.75,
                    reason=f"fast_commit_to_centered_{family}_cue:{cue.note}",
                )
            return action, self._fast_payload(cue, action.reason)

        if family == "toilet" and self._toilet_detection(obs):
            action = PolicyAction(
                action="drive_straight",
                power_percent=22.0,
                duration_s=0.7,
                reason="fast_drive_on_camera_toilet_detection",
            )
            self._fast_plan = [
                PolicyAction(
                    action="drive_straight",
                    power_percent=22.0,
                    duration_s=0.7,
                    reason="fast_followup_toilet_detection_drive",
                )
            ]
            return action, self._fast_payload(cue, action.reason)

        # A VLM call every step is expensive and often just repeats scanning.
        # When no target-like cue exists, deterministic scan probes are enough between VLM checks.
        if self._step_index > 0 and self._step_index % 3 != 0 and cue.score < 0.36:
            direction = self._last_turn_sign() or 1.0
            action = PolicyAction(
                action="turn_by_angle",
                degrees=28.0 * direction,
                reason=f"fast_interleaved_scan_before_next_vlm:{cue.note}",
            )
            return action, self._fast_payload(cue, action.reason)

        return None, {}

    def _should_start_side_approach(self, cue: _FastCue, history: list[dict[str, Any]]) -> bool:
        if cue.score < 0.68:
            return False
        if cue.family == "seat" and cue.close < 0.24:
            return False
        if cue.family == "bed" and cue.close < 0.10 and self._step_index > 2:
            return False
        if sum(1 for action in history[-4:] if action.get("action") == "drive_straight") >= 2:
            return False
        return self._step_index <= 4 or cue.center_blocked > 0.18 or cue.close > 0.42

    def _side_approach_plan(self, family: str, cue: _FastCue) -> list[PolicyAction]:
        turn_reason = f"side_on_{family}_cue_rotate_to_approach:{cue.note}"
        drive_reason = f"side_on_{family}_cue_short_approach:{cue.note}"
        turn_count = 3 if family == "seat" else 2
        drive_count = 3 if family == "seat" else 2
        plan: list[PolicyAction] = [
            PolicyAction(action="turn_by_angle", degrees=-35.0, reason=turn_reason) for _ in range(turn_count)
        ]
        plan.extend(
            PolicyAction(action="drive_straight", power_percent=24.0, duration_s=0.85, reason=drive_reason)
            for _ in range(drive_count)
        )
        return plan

    def _queue_approach_followups(self, action: PolicyAction, payload: dict[str, Any], prompt: str) -> None:
        family = _goal_family(prompt)
        if action.action != "drive_straight" or family not in {"toilet", "seat", "bed"}:
            return
        if family == "toilet" and self._step_index >= 9:
            self._fast_plan = [
                PolicyAction(
                    action="drive_straight",
                    power_percent=action.power_percent,
                    duration_s=action.duration_s,
                    reason="repeat_late_toilet_approach_without_extra_vlm",
                )
                for _ in range(2)
            ]
            return
        if not self._payload_goal_visible(payload) and self._confidence(payload.get("confidence")) < 0.55:
            return
        count = 2 if family == "toilet" else 1
        self._fast_plan = [
            PolicyAction(
                action="drive_straight",
                power_percent=action.power_percent,
                duration_s=action.duration_s,
                reason=f"repeat_visible_{family}_approach_without_extra_vlm",
            )
            for _ in range(count)
        ]

    def _fast_payload(self, cue: _FastCue, reason: str) -> dict[str, Any]:
        if cue.offset < -0.18:
            bearing = "left"
        elif cue.offset > 0.18:
            bearing = "right"
        else:
            bearing = "center"
        return {
            "visual_note": reason[:160],
            "goal_visible": cue.score >= 0.58,
            "goal_bearing": bearing,
            "confidence": max(0.0, min(1.0, cue.score)),
        }

    def _toilet_detection(self, obs: Observation) -> bool:
        for detection in obs.analysis.detections:
            if detection.name == "toilet" and detection.confidence >= 0.35 and abs(detection.center_offset) <= 0.45:
                return True
        return False

    def _anti_oscillation_hint(self) -> str:
        turns = [action for action in self._action_memory[-5:] if action.get("action") == "turn_by_angle"]
        if len(turns) >= 3:
            signs = [self._sign(turn.get("degrees")) for turn in turns[-4:]]
            nonzero = [sign for sign in signs if sign != 0]
            if len(nonzero) >= 3 and all(a != b for a, b in zip(nonzero, nonzero[1:])):
                last = nonzero[-1]
                direction = "right" if last > 0 else "left"
                return f"recent turns are alternating; do not reverse again unless the goal is clearly visible, continue {direction} or drive if aligned"
        if len(turns) >= 3 and all(abs(float(turn.get("degrees", 0.0))) < 14.0 for turn in turns[-3:]):
            return "recent turns are too small; choose a decisive 25-35 degree turn or drive if the target is centered"
        drives = [action for action in self._action_memory[-4:] if action.get("action") == "drive_straight"]
        if len(drives) >= 3:
            return "several forward moves were already attempted; only keep driving if the target or free approach path is visible"
        return "none"

    def _best_goal_sighting(self, current_yaw_deg: float) -> dict[str, Any]:
        sightings = [item for item in self._visual_memory if item.get("goal_visible")]
        if not sightings:
            return {}
        best = max(sightings, key=lambda item: float(item.get("confidence", 0.0)))
        yaw_error = self._angle_delta(float(best.get("yaw_deg", current_yaw_deg)), current_yaw_deg)
        return {
            "yaw_deg": round(float(best.get("yaw_deg", current_yaw_deg)), 1),
            "relative_turn_deg": round(yaw_error, 1),
            "goal_bearing": best.get("goal_bearing", "unknown"),
            "confidence": round(float(best.get("confidence", 0.0)), 2),
            "visual_note": str(best.get("visual_note", ""))[:120],
        }

    def _stabilize_action(self, action: PolicyAction, payload: dict[str, Any], yaw_deg: float) -> PolicyAction:
        if action.action == "turn_by_angle":
            degrees = action.degrees
            if abs(degrees) < 12.0:
                degrees = 22.0 if degrees >= 0.0 else -22.0
            if self._would_oscillate(degrees) and not self._payload_goal_visible(payload):
                degrees = math.copysign(max(25.0, abs(degrees)), self._last_turn_sign() or degrees or 1.0)
            return PolicyAction(action="turn_by_angle", degrees=degrees, reason=action.reason)

        if action.action == "drive_straight":
            duration_s = action.duration_s
            if not self._payload_goal_visible(payload) and len(self._recent_drives()) >= 3:
                remembered = self._best_goal_sighting(yaw_deg)
                turn = float(remembered.get("relative_turn_deg", 25.0))
                if abs(turn) >= 12.0:
                    return PolicyAction(
                        action="turn_by_angle",
                        degrees=max(-35.0, min(35.0, turn)),
                        reason=f"memory_redirect_after_repeated_drives:{action.reason}",
                    )
            if self._payload_goal_visible(payload) and str(payload.get("goal_bearing", "")) == "center":
                duration_s = min(0.7, duration_s)
            return PolicyAction(
                action="drive_straight",
                power_percent=action.power_percent,
                duration_s=duration_s,
                reason=action.reason,
            )

        return action

    def _remember(self, obs: Observation, payload: dict[str, Any], action: PolicyAction) -> None:
        memory_record = {
            "step": self._step_index,
            "yaw_deg": round(float(obs.yaw_deg), 1),
            "visual_note": str(payload.get("visual_note", ""))[:160],
            "goal_visible": self._payload_goal_visible(payload),
            "goal_bearing": self._clean_bearing(payload.get("goal_bearing")),
            "confidence": round(self._confidence(payload.get("confidence")), 2),
            "chosen_action": action.action,
        }
        self._visual_memory.append(memory_record)
        self._visual_memory = self._visual_memory[-self.memory_limit :]
        self._action_memory.append(
            {
                "action": action.action,
                "degrees": action.degrees,
                "power_percent": action.power_percent,
                "duration_s": action.duration_s,
                "reason": action.reason[:120],
            }
        )
        self._action_memory = self._action_memory[-self.memory_limit :]

    def _payload_goal_visible(self, payload: dict[str, Any]) -> bool:
        value = payload.get("goal_visible", False)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "visible", "1"}
        return bool(value)

    def _clean_bearing(self, value: Any) -> str:
        bearing = str(value).strip().lower()
        return bearing if bearing in {"left", "center", "right", "unknown"} else "unknown"

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return max(0.0, min(1.0, confidence))

    def _would_oscillate(self, proposed_degrees: float) -> bool:
        last = self._last_turn_sign()
        if last == 0:
            return False
        proposed = self._sign(proposed_degrees)
        if proposed == 0 or proposed == last:
            return False
        recent_signs = [self._sign(action.get("degrees")) for action in self._action_memory[-3:] if action.get("action") == "turn_by_angle"]
        recent_signs = [sign for sign in recent_signs if sign != 0]
        return len(recent_signs) >= 2

    def _last_turn_sign(self) -> float:
        for action in reversed(self._action_memory):
            if action.get("action") == "turn_by_angle":
                return self._sign(action.get("degrees"))
        return 0.0

    def _recent_drives(self) -> list[dict[str, Any]]:
        return [action for action in self._action_memory[-4:] if action.get("action") == "drive_straight"]

    def _sign(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number > 0.0:
            return 1.0
        if number < 0.0:
            return -1.0
        return 0.0

    def _angle_delta(self, target_yaw_deg: float, current_yaw_deg: float) -> float:
        return (target_yaw_deg - current_yaw_deg + 180.0) % 360.0 - 180.0


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
    return "generic"


def _has_word(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _fast_visual_cue(obs: Observation, family: str) -> _FastCue:
    image = Image.open(obs.path).convert("RGB").resize((96, 72), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = rgb.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    gray = rgb.mean(axis=2)
    value = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (value - min_channel) / np.maximum(value, 1e-4)
    edge = _edge_strength(gray)
    horizontal = _horizontal_edge(gray)

    score_map = _score_map(family, rgb, value, saturation, edge, horizontal, y)
    score, offset, close = _summarize_score(score_map, x, y)
    center_open = _center_open(value, saturation, edge, y, x)
    center_blocked = _center_blocked(value, edge, y, x)

    for detection in obs.analysis.detections:
        if family == "toilet" and detection.name == "toilet":
            score = max(score, min(1.0, detection.confidence * 1.25))
            offset = detection.center_offset
            close = max(close, min(1.0, detection.area_fraction * 28.0))

    note = (
        f"family={family} score={score:.2f} offset={offset:.2f} "
        f"close={close:.2f} open={center_open:.2f} blocked={center_blocked:.2f}"
    )
    return _FastCue(
        family=family,
        score=score,
        offset=max(-1.0, min(1.0, offset)),
        close=max(0.0, min(1.0, close)),
        center_open=center_open,
        center_blocked=center_blocked,
        note=note,
    )


def _score_map(
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
    low_mid = ((y >= 0.28) & (y <= 0.88)).astype(np.float32)
    mid = ((y >= 0.16) & (y <= 0.72)).astype(np.float32)
    upper_mid = ((y >= 0.10) & (y <= 0.68)).astype(np.float32)

    white_object = (value > 0.68) & (saturation < 0.26)
    dark_fabric = (value > 0.06) & (value < 0.38) & (edge > 0.012)
    muted_fabric = (value > 0.16) & (value < 0.84) & (saturation < 0.68)
    colored_cover = (saturation > 0.20) & (value > 0.25) & (value < 0.96)
    tan_or_orange = (red > green * 0.92) & (green > blue * 0.88) & (value > 0.22) & (saturation < 0.74)

    if family == "seat":
        broad_body = (dark_fabric.astype(np.float32) * 0.62 + muted_fabric.astype(np.float32) * edge * 3.0) * mid
        horizontal_body = horizontal * mid * 2.2
        return (broad_body + horizontal_body).astype(np.float32)
    if family == "bed":
        blanket = (colored_cover.astype(np.float32) * 0.32 + tan_or_orange.astype(np.float32) * 0.26) * upper_mid
        mattress_edge = horizontal * upper_mid * 3.0
        low_gap = dark_fabric.astype(np.float32) * low_mid * 0.42
        return (blanket + mattress_edge + low_gap).astype(np.float32)
    if family == "toilet":
        return (white_object.astype(np.float32) * (0.42 + edge * 3.8) * low_mid).astype(np.float32)
    if family == "sink":
        return (white_object.astype(np.float32) * (0.36 + edge * 3.8) * mid).astype(np.float32)

    return ((edge * 2.0 + colored_cover.astype(np.float32) * 0.14) * mid).astype(np.float32)


def _summarize_score(score_map: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    total = float(score_map.sum())
    if total <= 1e-5:
        return 0.0, 0.0, 0.0
    column = score_map.sum(axis=0)
    smooth = np.convolve(column, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    best = float(smooth.max())
    height = score_map.shape[0]
    score = min(1.0, best / max(1.0, height * 0.11))
    offset = float((score_map * x).sum() / total)
    low_total = float((score_map * (y > 0.54)).sum())
    close = min(1.0, low_total / max(1.0, total * 0.64))
    return score, offset, close


def _center_open(value: np.ndarray, saturation: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_floor = (np.abs(x) < 0.36) & (y > 0.56)
    if not np.any(center_floor):
        return 0.0
    low_edge = 1.0 - min(1.0, float(edge[center_floor].mean()) * 4.0)
    moderate_value = float(((value > 0.18) & center_floor).sum()) / float(center_floor.sum())
    low_saturation = 1.0 - min(1.0, float(saturation[center_floor].mean()))
    return max(0.0, min(1.0, low_edge * 0.42 + moderate_value * 0.38 + low_saturation * 0.20))


def _center_blocked(value: np.ndarray, edge: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    center_low = (np.abs(x) < 0.34) & (y > 0.44)
    if not np.any(center_low):
        return 0.0
    dark = float(((value < 0.22) & center_low).sum()) / float(center_low.sum())
    high_edge = min(1.0, float(edge[center_low].mean()) * 5.0)
    bright_wall = float(((value > 0.88) & (edge < 0.04) & center_low).sum()) / float(center_low.sum())
    return max(0.0, min(1.0, dark * 0.58 + high_edge * 0.28 + bright_wall * 0.28))


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.clip(gx + gy, 0.0, 1.0)


def _horizontal_edge(gray: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(np.diff(gray, axis=0, prepend=gray[:1, :])), 0.0, 1.0)
