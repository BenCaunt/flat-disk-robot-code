from __future__ import annotations

from pathlib import Path
from typing import Any

from flatdisk_sim import run_hardware_harness as hardware
from flatdisk_sim.run_hardware_harness import ArmedGuardTools


class _FakeInnerTools:
    def __init__(self) -> None:
        self.object_drive_detector = "florence-mlx"
        self.events: list[tuple[str, Any]] = []

    def stop(self) -> None:
        self.events.append(("stop", None))

    def log(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))

    def turn_by_angle(self, degrees: float, *, power_percent: float = 10.0) -> dict[str, Any]:
        self.events.append(("turn_by_angle", (degrees, power_percent)))
        return {"action": "turn_by_angle", "ok": True}

    def drive_straight(self, power_percent: float, duration_s: float) -> dict[str, Any]:
        self.events.append(("drive_straight", (power_percent, duration_s)))
        return {"action": "drive_straight", "ok": True}

    def visual_servo_object(
        self,
        prompt: str,
        *,
        duration_s: float = 2.0,
        detector: str | None = None,
        forward_power: float = 18.0,
    ) -> dict[str, Any]:
        self.events.append(("visual_servo_object", (prompt, duration_s, detector, forward_power)))
        return {"action": "visual_servo_object", "ok": True}

    def check_object_grounding(self, *, image_path: Path, prompt: str, detector: str | None = None) -> dict[str, Any]:
        return {"action": "check_object_grounding", "image_path": str(image_path), "prompt": prompt, "detector": detector}

    def query_topomap_memory(self, *, image_path: Path, goal_query: str) -> dict[str, Any]:
        return {"action": "query_topomap_memory", "image_path": str(image_path), "goal_query": goal_query}

    def observe(self, *, label: str = "observe", timeout_s: float = 2.0) -> dict[str, Any]:
        return {"label": label, "timeout_s": timeout_s}

    def close(self) -> None:
        self.events.append(("close", None))


def test_unarmed_guard_blocks_motor_tools() -> None:
    inner = _FakeInnerTools()
    tools = ArmedGuardTools(inner, armed=False)

    turn = tools.turn_by_angle(12.0, power_percent=9.0)
    drive = tools.drive_straight(20.0, 0.4)
    servo = tools.visual_servo_object("chair", duration_s=1.0, detector=None, forward_power=12.0)

    assert turn["failure_reason"] == "not_armed"
    assert drive["motor_commands_sent"] == 0
    assert servo["servo_status"] == "not_armed"
    assert ("turn_by_angle", (12.0, 9.0)) not in inner.events
    assert ("drive_straight", (20.0, 0.4)) not in inner.events
    assert ("visual_servo_object", ("chair", 1.0, None, 12.0)) not in inner.events
    assert [event for event, _payload in inner.events].count("blocked_motion") == 3


def test_armed_guard_delegates_motor_tools() -> None:
    inner = _FakeInnerTools()
    tools = ArmedGuardTools(inner, armed=True)

    assert tools.turn_by_angle(12.0, power_percent=9.0)["ok"] is True
    assert tools.drive_straight(20.0, 0.4)["ok"] is True
    assert tools.visual_servo_object("chair", duration_s=1.0, detector=None, forward_power=12.0)["ok"] is True

    assert ("turn_by_angle", (12.0, 9.0)) in inner.events
    assert ("drive_straight", (20.0, 0.4)) in inner.events
    assert ("visual_servo_object", ("chair", 1.0, None, 12.0)) in inner.events


def test_armed_guard_caps_forward_motor_power() -> None:
    inner = _FakeInnerTools()
    tools = ArmedGuardTools(inner, armed=True, max_forward_power_percent=10.0)

    drive = tools.drive_straight(20.0, 0.4)
    servo = tools.visual_servo_object("chair", duration_s=1.0, detector=None, forward_power=18.0)
    assert tools.turn_by_angle(12.0, power_percent=9.0)["ok"] is True

    assert drive["effective_power_percent"] == 10.0
    assert drive["requested_power_percent"] == 20.0
    assert servo["effective_power_percent"] == 10.0
    assert servo["requested_power_percent"] == 18.0
    assert ("drive_straight", (10.0, 0.4)) in inner.events
    assert ("visual_servo_object", ("chair", 1.0, None, 10.0)) in inner.events
    assert ("turn_by_angle", (12.0, 9.0)) in inner.events
    limit_events = [payload for event, payload in inner.events if event == "forward_power_limited"]
    assert limit_events == [
        {
            "action": "drive_straight",
            "requested_power_percent": 20.0,
            "effective_power_percent": 10.0,
            "max_forward_power_percent": 10.0,
        },
        {
            "action": "visual_servo_object",
            "requested_power_percent": 18.0,
            "effective_power_percent": 10.0,
            "max_forward_power_percent": 10.0,
        },
    ]


def test_armed_guard_caps_reverse_motor_power() -> None:
    inner = _FakeInnerTools()
    tools = ArmedGuardTools(inner, armed=True, max_forward_power_percent=10.0)

    drive = tools.drive_straight(-20.0, 0.4)

    assert drive["effective_power_percent"] == -10.0
    assert drive["requested_power_percent"] == -20.0
    assert drive["max_forward_power_percent"] == 10.0
    assert ("drive_straight", (-10.0, 0.4)) in inner.events
    limit_events = [payload for event, payload in inner.events if event == "forward_power_limited"]
    assert limit_events == [
        {
            "action": "drive_straight",
            "requested_power_percent": -20.0,
            "effective_power_percent": -10.0,
            "max_forward_power_percent": 10.0,
        }
    ]


def test_detector_readiness_surfaces_missing_torch(monkeypatch: Any) -> None:
    monkeypatch.setattr(hardware, "_module_available", lambda module: module != "torch")

    readiness = hardware._detector_readiness("florence-mlx")

    assert readiness["ok"] is False
    assert readiness["missing_modules"] == ["torch"]
    assert readiness["install_hint"] == "uv pip install torch"
    assert "detector_not_ready" in hardware._detector_not_ready_error(readiness)


def test_collect_failures_summarizes_process_failed_stderr(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "memory.jsonl").write_text(
        (
            '{"step": 0, "executed_action": {"tool": "visual_servo_object", '
            '"args": {"prompt": "chair"}}, "tool_result": {"ok": false, '
            '"action": "visual_servo_object", "detector": "florence-mlx", '
            '"failure_reason": "process_failed", "servo_status": "process_failed", '
            '"returncode": 1, "stderr_tail": "ImportError: torch missing"}}\n'
        ),
        encoding="utf-8",
    )

    failures = hardware._collect_failures({"goal": "chair"}, policy_dir)

    assert failures == [
        {
            "source": "tool_result",
            "step": 0,
            "tool": "visual_servo_object",
            "action": "visual_servo_object",
            "ok": False,
            "failure_reason": "process_failed",
            "servo_status": "process_failed",
            "returncode": 1,
            "detector": "florence-mlx",
            "prompt": "chair",
            "stderr_tail": "ImportError: torch missing",
        }
    ]
