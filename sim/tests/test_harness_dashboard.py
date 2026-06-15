from __future__ import annotations

import sys
import time
from urllib.parse import quote

import pytest
from PIL import Image

flask = pytest.importorskip("flask")

import flatdisk_sim.harness_dashboard as dashboard
from flatdisk_sim.harness_dashboard import _visual_servo_status_sample, create_app
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
    event_names = [event["event"] for event in state["recent_events"]]
    assert "actor_request" in event_names
    assert "actor" in event_names
    assert "critic_request" in event_names
    assert "tool_start" in event_names
    assert "tool_result" in event_names

    events_response = client.get("/api/events?limit=20")
    assert events_response.status_code == 200
    events_payload = events_response.get_json()
    assert events_payload["events_path"].endswith("harness_events.jsonl")
    assert any(event["event"] == "tool_result" for event in events_payload["events"])

    memory_response = client.get("/api/memory?limit=5")
    assert memory_response.status_code == 200
    memory_payload = memory_response.get_json()
    assert memory_payload["memory_path"].endswith("memory.jsonl")
    assert memory_payload["memory"]

    response = client.post("/api/pause", json={})
    assert response.status_code == 200
    paused = response.get_json()["state"]
    assert paused["mode"] in {"paused", "complete"}

    response = client.post("/api/teleop", json={"command": "left"})
    assert response.status_code == 200
    assert response.get_json()["result"]["command"] == "turn_by_angle"

    response = client.post("/api/teleop", json={"command": "reverse"})
    assert response.status_code == 200
    assert response.get_json()["result"]["command"] == "reverse"

    response = client.post("/api/resume", json={})
    assert response.status_code == 200
    resumed = response.get_json()["state"]
    assert resumed["mode"] in {"auto", "complete"}

    response = client.get("/api/latest-frame")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["X-Frame-Seq"]
    assert response.headers["X-Frame-Age-Ms"]
    assert response.headers["X-Imu-Seq"]

    live_response = client.get("/api/live")
    assert live_response.status_code == 200
    live_payload = live_response.get_json()
    assert live_payload["ok"] is True
    assert live_payload["telemetry"]["yaw_deg"] == pytest.approx(0.0)

    context_before_reset = state["context_generation"]
    response = client.post("/api/reset-context", json={})
    assert response.status_code == 200
    reset_state = response.get_json()["state"]
    assert reset_state["mode"] == "idle"
    assert reset_state["goal"] == ""
    assert reset_state["completion_reason"] == "context_reset"
    assert reset_state["memory_record_count"] == 0
    assert reset_state["context_generation"] > context_before_reset
    assert client.get("/api/memory?limit=5").get_json()["memory"] == []


def test_latest_frame_endpoint_uses_preview_without_recording_observation(tmp_path) -> None:
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=DeterministicHarnessRunner(),
        critic=DeterministicHarnessRunner(),
    )
    app = create_app(session=session, max_worker_steps=1, worker_interval_s=0.01)
    client = app.test_client()

    response = client.get("/api/latest-frame")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["X-Frame-Seq"] == "1"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Imu-Seq"] == "5"

    state = client.get("/api/state").get_json()
    assert state["last_observation"] is None
    assert not any(event["event"] == "observation" for event in state["recent_events"])


def test_artifact_endpoint_serves_run_images_only(tmp_path) -> None:
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=DeterministicHarnessRunner(),
        critic=DeterministicHarnessRunner(),
    )
    app = create_app(session=session, max_worker_steps=1, worker_interval_s=0.01)
    client = app.test_client()
    artifact = tmp_path / "motion_frames" / "florence_overlay.jpg"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (255, 79, 216)).save(artifact, format="JPEG")

    relative_response = client.get("/api/artifact?path=motion_frames/florence_overlay.jpg")
    assert relative_response.status_code == 200
    assert relative_response.mimetype == "image/jpeg"

    absolute_response = client.get(f"/api/artifact?path={quote(str(artifact), safe='')}")
    assert absolute_response.status_code == 200
    assert absolute_response.mimetype == "image/jpeg"

    outside = tmp_path.parent / f"{tmp_path.name}_outside.jpg"
    try:
        Image.new("RGB", (8, 6), (69, 240, 176)).save(outside, format="JPEG")
        escape_response = client.get(f"/api/artifact?path={quote(str(outside), safe='')}")
        assert escape_response.status_code == 404
    finally:
        outside.unlink(missing_ok=True)

    note = tmp_path / "motion_frames" / "not_an_image.txt"
    note.write_text("not an image", encoding="utf-8")
    assert client.get("/api/artifact?path=motion_frames/not_an_image.txt").status_code == 404


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
    assert "Reasoning Trace" in html
    assert "Tool I/O" in html
    assert "safety gate step" in html
    assert "function parseJsonDeep(value)" in html
    assert "function refreshLive()" in html
    assert "completionReason" in html
    assert "Reset Ctx" in html
    assert "resetContext" in html
    assert "/api/reset-context" in html
    assert "contextId" in html
    assert "memoryCount" in html
    assert "decision_summary" in html
    assert "completion_check" in html
    assert "Florence / Grounding Output" in html
    assert "grounding_audit_contact_sheet" in html
    assert "debug_overlay_contact_sheet" in html
    assert "motion_contact_sheet" in html
    assert "overlay_path" in html
    assert "Servo Test" in html
    assert 'href="/servo"' in html
    assert "/api/artifact?path=" in html
    assert "setInterval(refreshLive, 100)" in html
    assert "setInterval(refreshFrame, 150)" in html
    assert "color-scheme: light" not in html
    assert "gradient" not in html.lower()
    assert "box-shadow" not in html


def test_visual_servo_status_sample_parses_command_and_bbox_fields() -> None:
    fields = {
        "armed": "True",
        "pred": "3",
        "track": "5",
        "imu_pred": "2",
        "frame_seq": "42",
        "frame_age_s": "0.035",
        "filter": "klt-kalman:1.2deg",
        "pub": "7",
        "cmd": "8/12%",
        "heading_error": "-2.0deg",
        "target_yaw": "-160.8deg",
        "turn": "-2.0",
        "forward": "10.0",
        "bbox_cx_frac": "0.480",
        "bbox_cy_frac": "0.520",
        "bbox_w_frac": "0.200",
        "bbox_h_frac": "0.300",
        "det": "chair:florence:0.91",
        "det_count": "2",
        "lock_rej": "1",
        "lock_gate": "12.0deg",
        "lock_err": "3.0deg",
        "pending": "False",
        "lost": "0",
    }

    sample = _visual_servo_status_sample(fields, t=123.0)

    assert sample["t"] == 123.0
    assert sample["armed"] is True
    assert sample["prediction_count"] == 3
    assert sample["track_count"] == 5
    assert sample["imu_prediction_count"] == 2
    assert sample["motor1_percent"] == 8.0
    assert sample["motor2_percent"] == 12.0
    assert sample["heading_error_deg"] == pytest.approx(-2.0)
    assert sample["target_yaw_deg"] == pytest.approx(-160.8)
    assert sample["bbox_cx_frac"] == pytest.approx(0.48)
    assert sample["bbox_h_frac"] == pytest.approx(0.3)
    assert sample["detection"] == "chair:florence:0.91"
    assert sample["detection_count"] == 2
    assert sample["target_lock_reject_count"] == 1
    assert sample["target_lock_gate_deg"] == pytest.approx(12.0)
    assert sample["target_lock_error_deg"] == pytest.approx(3.0)
    assert sample["detector_pending"] is False


def test_visual_servo_test_page_and_fake_process_state(tmp_path, monkeypatch) -> None:
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=DeterministicHarnessRunner(),
        critic=DeterministicHarnessRunner(),
    )
    script = (
        "import time; "
        "print('object-drive armed=True pred=1 track=2 imu_pred=0 frame_seq=42 frame_age_s=0.030 "
        "filter=klt-kalman:1.2deg pub=3 cmd=8/12% heading_error=-2.0deg target_yaw=-160.8deg "
        "turn=-2.0 forward=10.0 bbox_cx_frac=0.480 bbox_cy_frac=0.520 bbox_w_frac=0.200 "
        "bbox_h_frac=0.300 det=chair:florence:0.91 det_count=2 lock_rej=1 "
        "lock_gate=12.0deg lock_err=3.0deg pending=False lost=0', flush=True); "
        "time.sleep(0.05)"
    )
    monkeypatch.setattr(dashboard, "_object_drive_command", lambda *, detector: [sys.executable, "-c", script])
    app = create_app(session=session, max_worker_steps=1, worker_interval_s=0.01)
    client = app.test_client()

    html = client.get("/servo").text
    assert "Visual Servo Test" in html
    assert "/api/visual-servo-test/start" in html
    assert "/api/visual-servo-test/stop" in html
    assert "plotCommands" in html
    assert "maxAbsOutput" in html
    assert "targetFilter" in html
    assert "targetLock" in html
    assert "targetLockMaxBearingDeg" in html
    assert "modelBearingNoiseDeg" in html
    assert 'id="headingDeadbandDeg" type="number" min="0" max="45" step="0.25" value="0"' in html
    assert 'id="stopWhenLost" type="checkbox" checked' in html

    state_response = client.get("/api/visual-servo-test/state")
    assert state_response.status_code == 200
    assert state_response.get_json()["state"]["running"] is False

    start_response = client.post("/api/visual-servo-test/start", json={"prompt": "chair", "duration_s": 2.0})
    assert start_response.status_code == 200
    config = start_response.get_json()["state"]["config"]
    assert config["heading_deadband_deg"] == 0.0
    assert config["stop_when_lost"] is True
    assert "--stop-when-lost" in config["command"]

    sample = None
    for _ in range(50):
        state = client.get("/api/visual-servo-test/state").get_json()["state"]
        sample = state["latest_sample"]
        if sample:
            break
        time.sleep(0.01)
    assert sample is not None
    assert sample["frame_seq"] == 42
    assert sample["motor1_percent"] == 8.0
    assert sample["motor2_percent"] == 12.0
    assert sample["detection"] == "chair:florence:0.91"
    assert sample["detection_count"] == 2
    assert sample["target_lock_reject_count"] == 1
    assert sample["target_lock_error_deg"] == pytest.approx(3.0)

    stop_response = client.post("/api/visual-servo-test/stop", json={})
    assert stop_response.status_code == 200


def test_visual_servo_test_rejects_empty_prompt(tmp_path) -> None:
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=DeterministicHarnessRunner(),
        critic=DeterministicHarnessRunner(),
    )
    app = create_app(session=session, max_worker_steps=1, worker_interval_s=0.01)
    client = app.test_client()

    response = client.post("/api/visual-servo-test/start", json={"prompt": ""})

    assert response.status_code == 400
    assert "prompt cannot be empty" in response.get_json()["error"]
