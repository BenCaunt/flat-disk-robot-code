from __future__ import annotations

from flatdisk_sim.harness_rerun import HarnessRerunLogger


class FakeRerun:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def init(self, *args, **kwargs) -> None:
        self.calls.append(("init", args, kwargs))

    def save(self, *args, **kwargs) -> None:
        self.calls.append(("save", args, kwargs))

    def set_time(self, *args, **kwargs) -> None:
        self.calls.append(("set_time", args, kwargs))

    def log(self, *args, **kwargs) -> None:
        self.calls.append(("log", args, kwargs))

    class StateConfiguration:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class StateChange:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class TextLog:
        def __init__(self, text: str) -> None:
            self.text = text

    class Scalars:
        def __init__(self, value) -> None:
            self.value = value

    class TextDocument:
        def __init__(self, text: str) -> None:
            self.text = text

    def send_blueprint(self, *args, **kwargs) -> None:
        self.calls.append(("send_blueprint", args, kwargs))


def test_rerun_logger_uses_state_timeline_archetypes(tmp_path) -> None:
    fake = FakeRerun()

    logger = HarnessRerunLogger(recording_id="test", save_path=tmp_path / "run.rrd", rr_module=fake)
    logger.log_metadata({"model": "gpt-5.5", "policy_input_allowlist": ["camera frame attachment"]})
    logger.log_state(3, "teleop")
    logger.log_command(3, "actor", {"tool": "drive_straight", "args": {"duration_s": 0.5}})
    logger.log_llm(3, "actor", {"output": '{"action":{"tool":"drive_straight"}}'})
    logger.log_llm(3, "critic", {"output": '{"verdict":"approve"}'})

    log_calls = [call for call in fake.calls if call[0] == "log"]
    assert any(call[1][0] == "robot/mode" and isinstance(call[1][1], FakeRerun.StateConfiguration) for call in log_calls)
    assert any(call[1][0] == "robot/mode" and isinstance(call[1][1], FakeRerun.StateChange) for call in log_calls)
    assert any(call[1][0] == "harness/metadata" and isinstance(call[1][1], FakeRerun.TextDocument) for call in log_calls)
    assert any(call[1][0] == "robot/commands" and isinstance(call[1][1], FakeRerun.TextLog) for call in log_calls)
    assert any(call[1][0] == "harness/llm/actor" and isinstance(call[1][1], FakeRerun.TextLog) for call in log_calls)
    assert any(call[1][0] == "harness/llm/critic" and isinstance(call[1][1], FakeRerun.TextLog) for call in log_calls)
    assert any(call[0] == "send_blueprint" for call in fake.calls)
    assert any(call[0] == "set_time" and call[2] == {"sequence": 3} for call in fake.calls)
