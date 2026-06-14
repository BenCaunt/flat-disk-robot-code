"""Persona B candidate: minimize wall-clock while staying camera/IMU-only.

This module is intentionally self-contained so it can be evaluated as a
low-conflict candidate before wiring it into the shared policy registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image

from flatdisk_sim.llm_harness import HarnessAction, action_to_dict, parse_prompt_action, validate_harness_action
from flatdisk_sim.text_goal_policy_core import clamp_float


_DYNAMIC_MARKER = "DYNAMIC_TASK_STATE\n"


@dataclass(frozen=True)
class FastVisualCue:
    score: float
    offset: float
    close: float
    blocked_center: float
    signature: tuple[float, ...]


class FastWallClockActor:
    """Harness-compatible actor that spends steps on motion, not observation."""

    name = "fast_wall_clock_actor"

    def __init__(self, *, allow_stop: bool = True, stale_drive_limit: int = 3) -> None:
        self.allow_stop = allow_stop
        self.stale_drive_limit = max(3, int(stale_drive_limit))

    def reset(self) -> None:
        pass

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        if role != "actor":
            raise ValueError(f"FastWallClockActor only handles actor prompts, got {role!r}")
        state = _extract_dynamic_state(prompt)
        action = choose_fast_action(
            state,
            image_paths=image_paths or [],
            allow_stop=self.allow_stop,
            stale_drive_limit=self.stale_drive_limit,
        )
        return json.dumps({"thought": action.thought, "action": {"tool": action.tool, "args": action.args}}, sort_keys=True)


class FastWallClockCritic:
    """Cheap local critic for the fast actor's bounded command stream."""

    name = "fast_wall_clock_critic"

    def reset(self) -> None:
        pass

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        del image_paths
        if role != "critic":
            raise ValueError(f"FastWallClockCritic only handles critic prompts, got {role!r}")
        try:
            payload = json.loads(prompt)
        except json.JSONDecodeError:
            return json.dumps(_critic_json("warn", "critic prompt was not parseable; keep bounded command"), sort_keys=True)
        action = validate_harness_action(parse_prompt_action(payload.get("candidate_action", {})))
        memory = payload.get("recent_memory", [])
        if not isinstance(memory, list):
            memory = []
        decision = _critic_decision(action, memory)
        return json.dumps(decision, sort_keys=True)


def choose_fast_action(
    state: dict[str, Any],
    *,
    image_paths: list[Path],
    allow_stop: bool = True,
    stale_drive_limit: int = 3,
) -> HarnessAction:
    """Choose one bounded action from only prompt-visible state and camera image."""

    observation = state.get("observation", {})
    if not isinstance(observation, dict):
        observation = {}
    memory = state.get("recent_memory", [])
    if not isinstance(memory, list):
        memory = []
    goal = str(state.get("goal", ""))
    family = _goal_family(goal)
    cue = _visual_cue(observation, image_paths)

    drive_count = _action_count(memory, "drive_straight")
    recent_drives = _recent_action_count(memory, "drive_straight", limit=3)
    recent_turns = _recent_action_count(memory, "turn_by_angle", limit=3)
    visual_change = _visual_change(observation, memory)

    if allow_stop and _should_stop(cue, drive_count, recent_drives, memory):
        return HarnessAction(
            "stop",
            {},
            f"fast_stop_after_bounded_progress_drives={drive_count}_close={cue.close:.2f}_score={cue.score:.2f}",
        )

    if cue.blocked_center > 0.72 and cue.score < 0.28:
        return _turn(_scan_direction(memory) * 30.0, "fast_escape_blocked_center")

    if cue.score >= _strong_threshold(family) and abs(cue.offset) > _turn_deadband(family):
        degrees = math.copysign(min(28.0, max(10.0, abs(cue.offset) * 34.0)), cue.offset)
        return _turn(degrees, f"fast_center_{family}_cue_score={cue.score:.2f}_offset={cue.offset:.2f}")

    if recent_drives >= max(3, stale_drive_limit) and visual_change < 0.018:
        return _turn(_scan_direction(memory) * 28.0, "fast_break_stale_drive_sequence")

    if recent_turns >= 2 and cue.score < 0.16:
        return _drive(22.0, 0.72, "fast_probe_after_scan")

    duration = 0.9
    if cue.score < 0.10 and drive_count == 0:
        duration = 0.72
    return _drive(24.0, duration, f"fast_bounded_forward_{family}_score={cue.score:.2f}_close={cue.close:.2f}")


def _extract_dynamic_state(prompt: str) -> dict[str, Any]:
    if _DYNAMIC_MARKER not in prompt:
        try:
            payload = json.loads(prompt)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    dynamic_text = prompt.split(_DYNAMIC_MARKER, 1)[1].strip()
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(dynamic_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _critic_decision(action: HarnessAction, memory: list[Any]) -> dict[str, Any]:
    recent = [_memory_action(record) for record in memory[-3:] if isinstance(record, dict)]
    if action.tool == "stop" and not _stop_evidence(memory):
        replacement = _drive(22.0, 0.62, "critic_replaces_early_fast_stop")
        return _critic_json("reject", "stop before enough bounded visual progress evidence", replacement)
    if action.tool == "turn_by_angle" and _same_turn_direction(action, recent[-2:]):
        replacement = _drive(22.0, 0.62, "critic_replaces_repeated_fast_turn")
        return _critic_json("reject", "third same-direction turn wastes wall-clock", replacement)
    return _critic_json("approve", "bounded fast action with no obvious loop")


def _stop_evidence(memory: list[Any]) -> bool:
    drive_count = _action_count(memory, "drive_straight")
    if drive_count >= 4:
        return True
    for record in memory[-3:]:
        if not isinstance(record, dict):
            continue
        observation = record.get("observation", {})
        if not isinstance(observation, dict):
            continue
        if _detection_close(observation) >= 0.62:
            return True
    return drive_count >= 3 and _recent_action_count(memory, "drive_straight", limit=4) >= 2


def _should_stop(cue: FastVisualCue, drive_count: int, recent_drives: int, memory: list[Any]) -> bool:
    gate_will_allow_stop = len(memory) >= 6 or _gate_visual_target(memory)
    if not gate_will_allow_stop:
        return False
    if drive_count < 6:
        return False
    if recent_drives >= 2 and cue.close >= 0.42 and cue.score >= 0.22:
        return True
    if cue.close >= 0.24 or _recent_visual_growth(memory):
        return True
    if drive_count >= 7:
        return True
    return False


def _gate_visual_target(memory: list[Any]) -> bool:
    for record in memory[-4:]:
        if not isinstance(record, dict):
            continue
        observation = record.get("observation", {})
        if not isinstance(observation, dict):
            continue
        detections = observation.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if isinstance(detection, dict) and clamp_float(detection.get("confidence", 0.0), 0.0, 1.0) >= 0.45:
                return True
    return False


def _visual_cue(observation: dict[str, Any], image_paths: list[Path]) -> FastVisualCue:
    det_score, det_offset, det_close = _detection_cue(observation)
    img_score, img_offset, img_close, blocked, signature = _image_cue(image_paths)
    score = max(det_score, img_score)
    if det_score >= img_score:
        offset = det_offset
    else:
        offset = img_offset
    close = max(det_close, img_close)
    return FastVisualCue(score=score, offset=offset, close=close, blocked_center=blocked, signature=signature)


def _detection_cue(observation: dict[str, Any]) -> tuple[float, float, float]:
    detections = observation.get("detections", [])
    if not isinstance(detections, list) or not detections:
        return 0.0, 0.0, 0.0
    best_score = 0.0
    best_offset = 0.0
    best_close = 0.0
    for det in detections:
        if not isinstance(det, dict):
            continue
        confidence = clamp_float(det.get("confidence", 0.0), 0.0, 1.0)
        area = clamp_float(det.get("area_fraction", 0.0), 0.0, 1.0)
        score = max(confidence, min(1.0, area * 22.0))
        if score > best_score:
            best_score = score
            best_offset = clamp_float(det.get("center_offset", 0.0), -1.0, 1.0)
            best_close = min(1.0, area * 42.0 + confidence * 0.25)
    return best_score, best_offset, best_close


def _detection_close(observation: dict[str, Any]) -> float:
    return _detection_cue(observation)[2]


def _image_cue(image_paths: list[Path]) -> tuple[float, float, float, float, tuple[float, ...]]:
    path = next((candidate for candidate in image_paths if candidate.exists()), None)
    if path is None:
        return 0.0, 0.0, 0.0, 0.0, ()
    try:
        image = Image.open(path).convert("RGB").resize((96, 72), Image.Resampling.BILINEAR)
    except OSError:
        return 0.0, 0.0, 0.0, 0.0, ()

    width, height = image.size
    pixels = image.load()
    mass = 0.0
    weighted_x = 0.0
    bbox = [width, height, -1, -1]
    blocked = 0
    blocked_total = 0
    signature_values: list[float] = []

    for by in range(6):
        for bx in range(8):
            total = 0.0
            count = 0
            for y in range(by * 12, (by + 1) * 12):
                for x in range(bx * 12, (bx + 1) * 12):
                    r, g, b = pixels[x, y]
                    total += (r + g + b) / 765.0
                    count += 1
            signature_values.append(round(total / max(1, count), 3))

    for y in range(8, height - 4):
        y_norm = y / max(1, height - 1)
        for x in range(2, width - 2):
            r, g, b = pixels[x, y]
            value = max(r, g, b) / 255.0
            low = min(r, g, b) / 255.0
            saturation = (value - low) / max(0.02, value)
            gray = (r + g + b) / 765.0
            left = sum(pixels[x - 1, y]) / 765.0
            up = sum(pixels[x, y - 1]) / 765.0
            edge = abs(gray - left) + abs(gray - up)
            center_x = abs((x / max(1, width - 1)) * 2.0 - 1.0)
            if center_x < 0.32 and y_norm > 0.48:
                blocked_total += 1
                if value < 0.18 or (value > 0.88 and edge < 0.035):
                    blocked += 1
            salient = ((saturation > 0.16 and value > 0.24) or edge > 0.08 or (value > 0.72 and saturation < 0.24))
            if not salient or y_norm > 0.84:
                continue
            weight = 0.18 + min(1.0, saturation + edge * 3.4)
            mass += weight
            weighted_x += weight * ((x / max(1, width - 1)) * 2.0 - 1.0)
            bbox[0] = min(bbox[0], x)
            bbox[1] = min(bbox[1], y)
            bbox[2] = max(bbox[2], x)
            bbox[3] = max(bbox[3], y)

    if mass <= 1e-5 or bbox[2] < bbox[0]:
        blocked_ratio = blocked / max(1, blocked_total)
        return 0.0, 0.0, 0.0, blocked_ratio, tuple(signature_values)
    offset = weighted_x / mass
    area = ((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)) / float(width * height)
    low_extent = max(0.0, bbox[3] / max(1, height - 1) - 0.45)
    score = min(1.0, mass / (width * height * 0.11))
    close = min(1.0, area * 2.1 + low_extent * 0.7)
    blocked_ratio = blocked / max(1, blocked_total)
    return score, max(-1.0, min(1.0, offset)), close, blocked_ratio, tuple(signature_values)


def _visual_change(observation: dict[str, Any], memory: list[Any]) -> float:
    brightness = clamp_float(observation.get("brightness_center", 0.0), 0.0, 1.0)
    previous = []
    for record in memory[-3:]:
        if not isinstance(record, dict):
            continue
        obs = record.get("observation", {})
        if isinstance(obs, dict) and "brightness_center" in obs:
            previous.append(clamp_float(obs.get("brightness_center"), 0.0, 1.0))
    if not previous:
        return 1.0
    return max(abs(brightness - value) for value in previous)


def _recent_visual_growth(memory: list[Any]) -> bool:
    closes = []
    for record in memory[-4:]:
        if not isinstance(record, dict):
            continue
        obs = record.get("observation", {})
        if isinstance(obs, dict):
            closes.append(_detection_close(obs))
    return len(closes) >= 2 and closes[-1] > closes[0] + 0.10


def _goal_family(prompt: str) -> str:
    text = prompt.lower()
    if _has_word(text, "toilet", "commode"):
        return "toilet"
    if _has_word(text, "bed", "mattress", "pillow"):
        return "bed"
    if _has_word(text, "sofa", "couch", "chair", "loveseat"):
        return "seat"
    if _has_word(text, "sink", "vanity"):
        return "sink"
    return "generic"


def _has_word(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _strong_threshold(family: str) -> float:
    return 0.24 if family in {"toilet", "bed", "seat"} else 0.30


def _turn_deadband(family: str) -> float:
    if family in {"bed", "seat"}:
        return 0.32
    if family == "toilet":
        return 0.24
    return 0.26


def _action_count(memory: list[Any], action_name: str) -> int:
    return sum(1 for record in memory if _memory_action(record).tool == action_name)


def _recent_action_count(memory: list[Any], action_name: str, *, limit: int) -> int:
    return sum(1 for record in memory[-limit:] if _memory_action(record).tool == action_name)


def _memory_action(record: Any) -> HarnessAction:
    if not isinstance(record, dict):
        return HarnessAction("wait", {"duration_s": 0.2}, "missing memory record")
    return validate_harness_action(parse_prompt_action(record.get("executed_action", record.get("actor_action", {}))))


def _same_turn_direction(action: HarnessAction, recent_actions: list[HarnessAction]) -> bool:
    direction = _turn_direction(action)
    if direction == 0 or len(recent_actions) < 2:
        return False
    return all(_turn_direction(recent) == direction for recent in recent_actions)


def _turn_direction(action: HarnessAction) -> int:
    if action.tool != "turn_by_angle":
        return 0
    degrees = float(action.args.get("degrees", 0.0))
    if degrees > 0:
        return 1
    if degrees < 0:
        return -1
    return 0


def _scan_direction(memory: list[Any]) -> float:
    recent_turns = [_memory_action(record) for record in memory[-4:]]
    for action in reversed(recent_turns):
        if action.tool == "turn_by_angle":
            return -1.0 if float(action.args.get("degrees", 0.0)) > 0.0 else 1.0
    return 1.0


def _drive(power_percent: float, duration_s: float, reason: str) -> HarnessAction:
    return HarnessAction(
        "drive_straight",
        {"power_percent": clamp_float(power_percent, 18.0, 24.0), "duration_s": clamp_float(duration_s, 0.25, 0.9)},
        reason[:500],
    )


def _turn(degrees: float, reason: str) -> HarnessAction:
    return HarnessAction(
        "turn_by_angle",
        {"degrees": clamp_float(degrees, -35.0, 35.0), "power_percent": 10.0},
        reason[:500],
    )


def _critic_json(verdict: str, reason: str, replacement: HarnessAction | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"verdict": verdict, "reason": reason, "replacement_action": None}
    if replacement is not None:
        checked = validate_harness_action(replacement)
        payload["replacement_action"] = {"tool": checked.tool, "args": checked.args}
    return payload


def action_json(action: HarnessAction) -> dict[str, Any]:
    """Small helper for tests and candidate comparison scripts."""

    return action_to_dict(validate_harness_action(action))
