"""Persona C candidate: deterministic memory and recovery text-goal policy.

This candidate intentionally stays on the real-robot observation contract:
current camera image, IMU yaw, text goal, prior actions, and its own memory.
It does not read simulator pose, object metadata, distances, collisions,
odometry, or wheel encoder state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.policies.sprinter import (
    _drive,
    _goal_family,
    _strong_threshold,
    _turn,
    _visual_cue,
    _weak_threshold,
    _wrap_deg,
)
from flatdisk_sim.text_goal_policy_core import PolicyAction


POLICY_INPUT_ALLOWLIST = (
    "current RGB camera frame",
    "camera-derived visual cues",
    "IMU yaw",
    "natural-language goal",
    "recent policy actions",
    "candidate-owned visual memory",
)

FORBIDDEN_HISTORY_FIELDS = frozenset(
    {
        "distance_m",
        "nearest_target",
        "hidden_score",
        "hidden_score_for_evaluator_only",
        "pose",
        "objects",
        "scene",
        "target_types",
        "success",
        "collision",
        "odometry",
        "encoder",
        "wheel_encoder",
    }
)


@dataclass(frozen=True)
class MemoryRecoveryConfig:
    memory_limit: int = 12
    stale_signature_delta: float = 0.028
    min_probe_turns: int = 5
    repeated_drive_count: int = 3


@dataclass(frozen=True)
class VisualMemoryRecord:
    step: int
    yaw_deg: float
    goal_family: str
    score: float
    offset: float
    open_center: float
    blocked_center: float
    close: float
    action: str
    reason: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryRecoveryPolicy:
    """Fast real-robot-compatible policy focused on recovery from stuck loops.

    The base navigation behavior is deterministic visual servoing. The Persona C
    contribution is the recovery layer: it detects repeated camera signatures,
    remembers promising camera/IMU bearings, avoids left/right oscillation, and
    probes forward after long scan-only loops when the camera shows open center.
    """

    name = "memory_recovery"

    def __init__(self, *, config: MemoryRecoveryConfig | None = None) -> None:
        self.config = config or MemoryRecoveryConfig()
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._scan_direction = 1.0
        self._last_signature: tuple[float, ...] | None = None
        self._repeat_signature_count = 0
        self._memory: list[VisualMemoryRecord] = []
        self._last_goal: str | None = None

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        if self._last_goal != prompt:
            self.reset()
            self._last_goal = prompt

        safe_history = sanitize_history(history)
        family = _goal_family(prompt)
        cue = _visual_cue(obs, family)
        stale_view = self._update_stale_count(cue.signature)

        action = self._recovery_action(
            obs=obs,
            family=family,
            cue=cue,
            history=safe_history,
            stale_view=stale_view,
        )
        if action is None:
            action = self._base_action(family=family, cue=cue, history=safe_history)

        self._remember(obs=obs, family=family, cue=cue, action=action)
        self._step += 1
        return action

    def public_memory(self) -> list[dict[str, Any]]:
        return [record.public_dict() for record in self._memory]

    def _recovery_action(
        self,
        *,
        obs: Observation,
        family: str,
        cue: Any,
        history: list[dict[str, Any]],
        stale_view: bool,
    ) -> PolicyAction | None:
        turn_streak = _recent_action_streak(history, "turn_by_angle")
        drive_streak = _recent_action_streak(history, "drive_straight")
        strong = cue.score >= _strong_threshold(family)

        oscillation = _alternating_turn_sign(history[-5:])
        if oscillation is not None and (not strong or turn_streak >= 4):
            return _turn(32.0 * oscillation, "memory_recovery_continue_scan_after_turn_oscillation")

        if drive_streak >= self.config.repeated_drive_count:
            remembered = self._turn_toward_best_memory(obs.yaw_deg, cue_offset=cue.offset)
            if remembered is not None and (not strong or stale_view):
                return remembered
            if stale_view or cue.blocked_center > 0.54 or cue.close > 0.64:
                direction = _preferred_escape_direction(cue.offset, history, self._scan_direction)
                self._scan_direction = direction
                return _turn(32.0 * direction, "memory_recovery_stale_repeated_drive_escape")

        if turn_streak >= self.config.min_probe_turns and not strong:
            if cue.open_center > 0.58 and cue.blocked_center < 0.42:
                return _drive(21.0, 0.62, "memory_recovery_open_center_probe_after_scan_loop")
            direction = _last_turn_sign(history) or self._scan_direction
            return _turn(32.0 * direction, "memory_recovery_decisive_scan_after_no_progress")

        remembered = self._turn_toward_best_memory(obs.yaw_deg, cue_offset=cue.offset)
        if remembered is not None and cue.score < _weak_threshold(family):
            return remembered

        return None

    def _base_action(self, *, family: str, cue: Any, history: list[dict[str, Any]]) -> PolicyAction:
        if cue.score >= _strong_threshold(family):
            deadband = 0.22 if family in {"toilet", "sink"} else 0.14
            if abs(cue.offset) > deadband:
                return _turn(math.copysign(max(16.0, abs(cue.offset) * 36.0), cue.offset), f"memory_center_{family}_cue")
            duration = 0.62 if cue.close > 0.66 or cue.blocked_center > 0.58 else 0.82
            return _drive(23.0, duration, f"memory_drive_on_centered_{family}_cue")

        if cue.score >= _weak_threshold(family) and abs(cue.offset) > 0.10:
            self._scan_direction = 1.0 if cue.offset >= 0.0 else -1.0
            return _turn(math.copysign(max(18.0, abs(cue.offset) * 34.0), cue.offset), f"memory_align_weak_{family}_cue")

        if _recent_action_streak(history, "drive_straight") >= 2:
            self._scan_direction *= -1.0

        if cue.blocked_center > 0.56:
            self._scan_direction = -1.0 if cue.offset > 0.0 else 1.0

        return _turn(32.0 * self._scan_direction, f"memory_scan_for_{family}")

    def _turn_toward_best_memory(self, yaw_deg: float, *, cue_offset: float) -> PolicyAction | None:
        if not self._memory:
            return None
        candidates = [record for record in self._memory if record.score >= 0.20]
        if not candidates:
            return None
        best = max(candidates, key=lambda record: (record.score, -abs(record.offset), record.step))
        yaw_error = _wrap_deg(best.yaw_deg - yaw_deg)
        correction = yaw_error + best.offset * 12.0 + cue_offset * 6.0
        if abs(correction) < 12.0:
            return None
        return _turn(correction, f"memory_recovery_return_to_best_sighting_score={best.score:.2f}")

    def _remember(self, *, obs: Observation, family: str, cue: Any, action: PolicyAction) -> None:
        if cue.score < _weak_threshold(family) and action.action != "drive_straight":
            return
        record = VisualMemoryRecord(
            step=self._step,
            yaw_deg=round(float(obs.yaw_deg), 1),
            goal_family=family,
            score=round(float(cue.score), 3),
            offset=round(float(cue.offset), 3),
            open_center=round(float(cue.open_center), 3),
            blocked_center=round(float(cue.blocked_center), 3),
            close=round(float(cue.close), 3),
            action=action.action,
            reason=action.reason[:120],
        )
        self._memory.append(record)
        limit = max(3, int(self.config.memory_limit))
        self._memory = self._memory[-limit:]

    def _update_stale_count(self, signature: tuple[float, ...]) -> bool:
        if self._last_signature is None:
            self._last_signature = signature
            self._repeat_signature_count = 0
            return False
        delta = _signature_delta(signature, self._last_signature)
        self._last_signature = signature
        if delta <= self.config.stale_signature_delta:
            self._repeat_signature_count += 1
        else:
            self._repeat_signature_count = 0
        return self._repeat_signature_count >= 1


def sanitize_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"action", "degrees", "power_percent", "duration_s", "reason"}
    safe: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        safe.append({key: item[key] for key in allowed if key in item and key not in FORBIDDEN_HISTORY_FIELDS})
    return safe


def _signature_delta(current: tuple[float, ...], previous: tuple[float, ...]) -> float:
    if not current or not previous:
        return 1.0
    size = min(len(current), len(previous))
    if size == 0:
        return 1.0
    return sum(abs(float(current[index]) - float(previous[index])) for index in range(size)) / float(size)


def _alternating_turn_sign(history: list[dict[str, Any]]) -> float | None:
    signs = [_turn_sign(item) for item in history if item.get("action") == "turn_by_angle"]
    signs = [sign for sign in signs if sign != 0.0]
    if len(signs) < 4:
        return None
    recent = signs[-4:]
    if all(left != right for left, right in zip(recent, recent[1:])):
        return recent[-1]
    return None


def _last_turn_sign(history: list[dict[str, Any]]) -> float:
    for item in reversed(history):
        if item.get("action") == "turn_by_angle":
            return _turn_sign(item)
    return 0.0


def _recent_action_streak(history: list[dict[str, Any]], action_name: str) -> int:
    count = 0
    for item in reversed(history):
        if item.get("action") != action_name:
            break
        count += 1
    return count


def _turn_sign(item: dict[str, Any]) -> float:
    try:
        degrees = float(item.get("degrees", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if degrees > 0.0:
        return 1.0
    if degrees < 0.0:
        return -1.0
    return 0.0


def _preferred_escape_direction(cue_offset: float, history: list[dict[str, Any]], fallback: float) -> float:
    if abs(cue_offset) > 0.08:
        return -1.0 if cue_offset > 0.0 else 1.0
    last = _last_turn_sign(history)
    if last:
        return -last
    return 1.0 if fallback >= 0.0 else -1.0
