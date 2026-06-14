from __future__ import annotations

import time

import pytest

flask = pytest.importorskip("flask")

from flatdisk_sim.harness_dashboard import create_app
from flatdisk_sim.llm_harness import DeterministicHarnessRunner, HarnessConfig, HarnessSession
from fakes import FakeHarnessTools


def test_dashboard_goal_and_teleop_api(tmp_path) -> None:
    runner = DeterministicHarnessRunner()
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=2),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=runner,
        critic=runner,
    )
    app = create_app(session=session, max_worker_steps=2, worker_interval_s=0.01)
    client = app.test_client()

    response = client.post("/api/goal", json={"goal": "Drive to the sofa."})
    assert response.status_code == 200

    for _ in range(80):
        state = client.get("/api/state").get_json()
        if state["step"] >= 1 or state["mode"] in {"complete", "error"}:
            break
        time.sleep(0.01)
    assert state["step"] >= 1
    assert state["mode"] in {"auto", "complete"}
    assert state["metadata"]["model"] == "gpt-5.5"
    assert state["metadata"]["reasoning_effort"] == "low"
    assert state["metadata"]["live_codex"] is False
    assert state["metadata"]["model_facing_paths"] == "relative"
    assert "camera frame attachment" in state["metadata"]["policy_input_allowlist"]

    response = client.post("/api/pause", json={})
    assert response.status_code == 200
    paused = response.get_json()["state"]
    assert paused["mode"] in {"paused", "complete"}

    response = client.post("/api/teleop", json={"command": "left"})
    assert response.status_code == 200
    assert response.get_json()["result"]["command"] == "turn_by_angle"

    response = client.post("/api/resume", json={})
    assert response.status_code == 200
    resumed = response.get_json()["state"]
    assert resumed["mode"] in {"auto", "complete"}

    response = client.get("/api/latest-frame")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"


def test_dashboard_html_uses_dark_sharp_agent_monitoring_surface(tmp_path) -> None:
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=DeterministicHarnessRunner(),
        critic=DeterministicHarnessRunner(),
    )
    app = create_app(session=session, max_worker_steps=1, worker_interval_s=0.01)
    client = app.test_client()

    html = client.get("/").text

    assert "color-scheme: dark" in html
    assert "--accent: #ff4fd8" in html
    assert "border-radius: 0" in html
    assert "clip-path: polygon" in html
    assert "Agent Chat" in html
    assert "safety gate step" in html
    assert "color-scheme: light" not in html
    assert "gradient" not in html.lower()
    assert "box-shadow" not in html
