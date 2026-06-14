"""Shared policy contract for camera+IMU text-goal navigation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from .agent_tools import Observation


@dataclass(frozen=True)
class PolicyAction:
    action: str
    degrees: float = 0.0
    power_percent: float = 0.0
    duration_s: float = 0.0
    success: bool = False
    reason: str = ""


class TextGoalPolicy(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        ...


def policy_history_record(action: PolicyAction) -> dict[str, Any]:
    return {
        "action": action.action,
        "degrees": action.degrees,
        "power_percent": action.power_percent,
        "duration_s": action.duration_s,
        "reason": action.reason,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("policy did not return a JSON object")
    return parsed


def validate_action(payload: dict[str, Any], *, allow_stop: bool = False) -> PolicyAction:
    action = str(payload.get("action", "")).strip()
    reason = str(payload.get("reason", ""))[:240]
    if action == "turn_by_angle":
        return PolicyAction(action=action, degrees=clamp_float(payload.get("degrees", 0.0), -35.0, 35.0), reason=reason)
    if action == "drive_straight":
        return PolicyAction(
            action=action,
            power_percent=clamp_float(payload.get("power_percent", 20.0), 20.0, 24.0),
            duration_s=clamp_float(payload.get("duration_s", 0.6), 0.6, 0.9),
            reason=reason,
        )
    if action == "stop" and allow_stop:
        return PolicyAction(action=action, success=bool(payload.get("success", False)), reason=reason)
    if action == "stop":
        return PolicyAction(
            action="drive_straight",
            power_percent=20.0,
            duration_s=0.6,
            reason=f"policy_stop_replaced_with_navigation:{reason}",
        )
    return PolicyAction(action="turn_by_angle", degrees=25.0, reason=f"invalid_policy_action:{action}")


def clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))
