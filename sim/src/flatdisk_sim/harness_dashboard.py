"""Dashboard for slash-goal LLM control of the flat disk robot."""

from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any

from PIL import Image, ImageDraw

from .agent_tools import AgentTools, _float_or_none, _int_or_none, _object_drive_command, _parse_object_drive_status_fields
from .env import default_zenoh_connect
from .harness_rerun import HarnessRerunLogger
from .llm_harness import (
    CodexExecRunner,
    DeterministicHarnessRunner,
    HarnessConfig,
    HarnessSession,
    NoopCriticRunner,
    OpenAICompatibleVisionRunner,
    SafetyCriticRunner,
    ScriptedOpenVocabRunner,
)
from .paths import REPO_ROOT, SCRATCH_ROOT
from .protocol import DEFAULT_NAMESPACE
from .run_hardware_harness import ArmedGuardTools, forward_power_limit_rule


def create_app(
    *,
    session: HarnessSession,
    max_worker_steps: int = 24,
    worker_interval_s: float = 0.25,
) -> Any:
    try:
        from flask import Flask, abort, jsonify, request, send_file
    except ImportError as exc:  # pragma: no cover - covered by importorskip tests.
        raise RuntimeError("Flask is required for the harness dashboard") from exc

    app = Flask(__name__)
    worker_lock = threading.Lock()
    worker_thread: threading.Thread | None = None
    visual_servo_controller = VisualServoTestController(session=session)
    atexit.register(visual_servo_controller.stop)

    def worker() -> None:
        session.set_worker_active(True)
        try:
            while True:
                status = session.status()
                if status["mode"] == "auto":
                    if status["step"] >= max_worker_steps:
                        session.request_stop()
                        break
                    session.run_auto_step()
                elif status["mode"] in {"paused", "teleop"}:
                    time.sleep(0.15)
                else:
                    break
                time.sleep(worker_interval_s)
        finally:
            session.set_worker_active(False)

    def ensure_worker() -> None:
        nonlocal worker_thread
        with worker_lock:
            if worker_thread is not None and worker_thread.is_alive():
                return
            worker_thread = threading.Thread(target=worker, name="flatdisk-harness-worker", daemon=True)
            worker_thread.start()

    @app.get("/")
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/servo")
    def visual_servo_test() -> str:
        return SERVO_TEST_HTML

    @app.get("/api/state")
    def api_state() -> Any:
        return jsonify(session.status())

    @app.get("/api/visual-servo-test/state")
    def api_visual_servo_test_state() -> Any:
        return jsonify({"ok": True, "state": visual_servo_controller.state()})

    @app.post("/api/visual-servo-test/start")
    def api_visual_servo_test_start() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        try:
            state = visual_servo_controller.start(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "state": visual_servo_controller.state()}), 400
        return jsonify({"ok": True, "state": state})

    @app.post("/api/visual-servo-test/stop")
    def api_visual_servo_test_stop() -> Any:
        state = visual_servo_controller.stop(force_robot_stop=True)
        return jsonify({"ok": True, "state": state})

    @app.get("/api/events")
    def api_events() -> Any:
        limit = _bounded_int(request.args.get("limit"), default=200, minimum=1, maximum=500)
        return jsonify({"events": session.read_events_tail(limit), "events_path": str(session.events_path)})

    @app.get("/api/memory")
    def api_memory() -> Any:
        limit = _bounded_int(request.args.get("limit"), default=60, minimum=1, maximum=200)
        return jsonify({"memory": session.read_memory_tail(limit), "memory_path": str(session.memory_path)})

    @app.get("/api/live")
    def api_live() -> Any:
        try:
            return jsonify({"ok": True, "telemetry": session.preview_telemetry(timeout_s=0.04)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc), "telemetry": {}}), 503

    @app.get("/api/latest-frame")
    def api_latest_frame() -> Any:
        try:
            camera = session.preview_frame_bytes(timeout_s=0.08)
            jpeg = camera.get("jpeg")
            if isinstance(jpeg, bytes):
                response = app.response_class(jpeg, mimetype="image/jpeg")
                _add_live_headers(response, camera)
                return response
        except Exception:
            pass
        frame_summary: dict[str, Any] = {}
        try:
            frame_summary = session.preview_frame(timeout_s=0.25)
        except Exception:
            status = session.status()
            observation = status.get("last_observation") if isinstance(status.get("last_observation"), dict) else {}
            path_text = observation.get("path") or status.get("latest_frame_path")
            if path_text:
                frame_summary["path"] = path_text
            if observation.get("frame_seq") is not None:
                frame_summary["frame_seq"] = observation.get("frame_seq")
        path_text = frame_summary.get("path")
        if not path_text:
            path_text = str(_placeholder_image(session.run_dir))
        path = _resolve_dashboard_artifact_path(path_text, run_dir=session.run_dir)
        if not path.exists():
            path = _placeholder_image(session.run_dir)
        response = send_file(path.resolve(), mimetype="image/jpeg")
        _add_live_headers(response, frame_summary | {"path": str(path)})
        return response

    @app.get("/api/artifact")
    def api_artifact() -> Any:
        path_text = request.args.get("path", "")
        path = _safe_dashboard_artifact_path(path_text, run_dir=session.run_dir)
        if path is None:
            abort(404)
        return send_file(path, mimetype=_dashboard_artifact_mimetype(path) or "application/octet-stream")

    @app.post("/api/goal")
    def api_goal() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        reset_context = bool(payload.get("reset_context", True))
        visual_servo_controller.stop()
        session.start_goal(str(payload.get("goal", "")), reset_context=reset_context)
        ensure_worker()
        return jsonify({"ok": True, "state": session.status()})

    @app.post("/api/reset-context")
    def api_reset_context() -> Any:
        visual_servo_controller.stop(force_robot_stop=True)
        session.reset_context()
        return jsonify({"ok": True, "state": session.status()})

    @app.post("/api/pause")
    def api_pause() -> Any:
        session.pause()
        return jsonify({"ok": True, "state": session.status()})

    @app.post("/api/resume")
    def api_resume() -> Any:
        session.resume()
        ensure_worker()
        return jsonify({"ok": True, "state": session.status()})

    @app.post("/api/stop")
    def api_stop() -> Any:
        visual_servo_controller.stop(force_robot_stop=True)
        session.request_stop()
        return jsonify({"ok": True, "state": session.status()})

    @app.post("/api/teleop")
    def api_teleop() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        visual_servo_controller.stop(force_robot_stop=True)
        result = session.teleop(str(payload.get("command", "stop")), payload.get("value"))
        return jsonify({"ok": True, "result": result, "state": session.status()})

    return app


class VisualServoTestController:
    """Run object_drive_zenoh.py from the dashboard and expose parsed live status."""

    def __init__(self, *, session: HarnessSession) -> None:
        self.session = session
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._returncode: int | None = None
        self._config: dict[str, Any] = {}
        self._samples: list[dict[str, Any]] = []
        self._logs: list[dict[str, Any]] = []
        self._stderr_tail: list[str] = []
        self._run_id = 0
        self._started_t: float | None = None
        self._stopped_t: float | None = None
        self._overlay_dir: Path | None = None
        self._raw_dir: Path | None = None

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.stop()
        raw_tools = _raw_agent_tools(self.session.tools)
        client = getattr(raw_tools, "client", None)
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("visual servo prompt cannot be empty")
        detector = str(payload.get("detector") or getattr(raw_tools, "object_drive_detector", "florence-mlx")).strip()
        if detector not in {"florence-mlx", "florence-transformers", "grounding-dino"}:
            raise ValueError(f"unknown detector: {detector}")

        cap = _forward_power_cap(self.session.tools)
        default_forward = min(10.0, cap) if cap is not None else 10.0
        requested_forward = _bounded_float(payload.get("forward_power"), default=default_forward, minimum=0.0, maximum=60.0)
        forward_power = _limit_forward_power(requested_forward, cap)
        default_turn = float(getattr(client, "max_turn_percent", 10.0))
        max_turn_percent = _bounded_float(
            payload.get("max_turn_percent"),
            default=default_turn,
            minimum=0.0,
            maximum=100.0,
        )
        min_turn_percent = _bounded_float(
            payload.get("min_turn_percent"),
            default=float(getattr(client, "min_turn_percent", 1.5)),
            minimum=0.0,
            maximum=100.0,
        )
        min_turn_percent = min(min_turn_percent, max_turn_percent)
        max_abs_default = max(12.0, min(60.0, forward_power + max_turn_percent))
        max_abs_output = _bounded_float(payload.get("max_abs_output"), default=max_abs_default, minimum=0.0, maximum=100.0)
        heading_kp = _bounded_float(payload.get("heading_kp"), default=float(getattr(client, "heading_kp", 8.0)), minimum=0.0, maximum=100.0)
        heading_deadband_deg = _bounded_float(payload.get("heading_deadband_deg"), default=0.0, minimum=0.0, maximum=45.0)
        duration_s = _bounded_float(payload.get("duration_s"), default=30.0, minimum=0.5, maximum=1800.0)
        target_filter = _bool_payload(payload.get("target_filter"), default=True)
        target_lock = _bool_payload(payload.get("target_lock"), default=True)
        target_lock_max_bearing_deg = _bounded_float(
            payload.get("target_lock_max_bearing_deg"),
            default=12.0,
            minimum=0.0,
            maximum=90.0,
        )
        imu_heading_noise_deg = _bounded_float(payload.get("imu_heading_noise_deg"), default=2.0, minimum=0.0, maximum=45.0)
        model_bearing_noise_deg = _bounded_float(payload.get("model_bearing_noise_deg"), default=4.0, minimum=0.1, maximum=90.0)
        track_bearing_noise_deg = _bounded_float(payload.get("track_bearing_noise_deg"), default=8.0, minimum=0.1, maximum=90.0)
        target_process_noise_deg_s = _bounded_float(
            payload.get("target_process_noise_deg_s"),
            default=10.0,
            minimum=0.1,
            maximum=180.0,
        )
        stop_when_lost = _bool_payload(payload.get("stop_when_lost"), default=True)
        armed = bool(getattr(self.session.tools, "armed", False))

        run_name = time.strftime("%Y%m%d_%H%M%S")
        base_dir = self.session.run_dir / "visual_servo_test" / run_name
        overlay_dir = base_dir / "overlays"
        raw_dir = base_dir / "raw"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        cmd = _object_drive_command(detector=detector) + [
            "--prompt",
            prompt,
            "--duration",
            f"{duration_s:.3f}",
            "--forward-power",
            f"{forward_power:.3f}",
            "--namespace",
            str(getattr(raw_tools, "namespace", DEFAULT_NAMESPACE)),
            "--connect",
            str(getattr(raw_tools, "connect", default_zenoh_connect())),
            "--detector",
            detector,
            "--heading-kp",
            f"{heading_kp:.3f}",
            "--max-turn-percent",
            f"{max_turn_percent:.3f}",
            "--min-turn-percent",
            f"{min_turn_percent:.3f}",
            "--heading-deadband-deg",
            f"{heading_deadband_deg:.3f}",
            "--max-abs-output",
            f"{max_abs_output:.3f}",
            "--overlay-dir",
            str(overlay_dir),
            "--overlay-every",
            "1",
            "--raw-frame-dir",
            str(raw_dir),
            "--raw-frame-every",
            "1",
            "--target-filter" if target_filter else "--no-target-filter",
            "--target-lock" if target_lock else "--no-target-lock",
            "--target-lock-max-bearing-deg",
            f"{target_lock_max_bearing_deg:.3f}",
            "--imu-heading-noise-deg",
            f"{imu_heading_noise_deg:.3f}",
            "--model-bearing-noise-deg",
            f"{model_bearing_noise_deg:.3f}",
            "--track-bearing-noise-deg",
            f"{track_bearing_noise_deg:.3f}",
            "--target-process-noise-deg-s",
            f"{target_process_noise_deg_s:.3f}",
        ]
        if armed:
            cmd.append("--arm")
        cmd.append("--stop-when-lost" if stop_when_lost else "--no-stop-when-lost")
        if bool(getattr(client, "reverse_yaw", True)):
            cmd.append("--reverse-yaw")
        else:
            cmd.append("--no-reverse-yaw")

        try:
            proc = subprocess.Popen(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=REPO_ROOT,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise ValueError(f"failed to start visual servo process: {exc}") from exc

        config = {
            "prompt": prompt,
            "detector": detector,
            "duration_s": duration_s,
            "requested_forward_power": requested_forward,
            "forward_power": forward_power,
            "max_forward_power_percent": cap,
            "heading_kp": heading_kp,
            "max_turn_percent": max_turn_percent,
            "min_turn_percent": min_turn_percent,
            "heading_deadband_deg": heading_deadband_deg,
            "max_abs_output": max_abs_output,
            "target_filter": target_filter,
            "target_lock": target_lock,
            "target_lock_max_bearing_deg": target_lock_max_bearing_deg,
            "imu_heading_noise_deg": imu_heading_noise_deg,
            "model_bearing_noise_deg": model_bearing_noise_deg,
            "track_bearing_noise_deg": track_bearing_noise_deg,
            "target_process_noise_deg_s": target_process_noise_deg_s,
            "stop_when_lost": stop_when_lost,
            "armed": armed,
            "overlay_dir": _run_relative_path(overlay_dir, run_dir=self.session.run_dir),
            "raw_dir": _run_relative_path(raw_dir, run_dir=self.session.run_dir),
            "command": cmd,
        }
        with self._lock:
            self._run_id += 1
            run_id = self._run_id
            self._proc = proc
            self._returncode = None
            self._config = config
            self._samples = []
            self._logs = []
            self._stderr_tail = []
            self._started_t = time.time()
            self._stopped_t = None
            self._overlay_dir = overlay_dir
            self._raw_dir = raw_dir

        if proc.stdout is not None:
            threading.Thread(
                target=self._read_stream,
                args=(proc, run_id, proc.stdout, "stdout"),
                name="visual-servo-test-stdout",
                daemon=True,
            ).start()
        if proc.stderr is not None:
            threading.Thread(
                target=self._read_stream,
                args=(proc, run_id, proc.stderr, "stderr"),
                name="visual-servo-test-stderr",
                daemon=True,
            ).start()
        threading.Thread(
            target=self._wait_for_process,
            args=(proc, run_id),
            name="visual-servo-test-waiter",
            daemon=True,
        ).start()
        return self.state()

    def stop(self, *, force_robot_stop: bool = False) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            running = proc is not None and proc.poll() is None
        if proc is None:
            if force_robot_stop:
                self._stop_robot()
            return self.state()
        if running:
            self._stop_robot()
            _terminate_process_group(proc)
            self._stop_robot()
        elif force_robot_stop:
            self._stop_robot()
        with self._lock:
            if self._proc is proc:
                self._returncode = proc.poll()
                self._stopped_t = time.time()
        return self.state()

    def state(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            returncode = proc.poll() if proc is not None else self._returncode
            running = proc is not None and returncode is None
            overlay_path = _latest_image_path(self._overlay_dir)
            raw_path = _latest_image_path(self._raw_dir)
            return {
                "running": running,
                "returncode": returncode,
                "started_t": self._started_t,
                "stopped_t": self._stopped_t,
                "armed": bool(getattr(self.session.tools, "armed", False)),
                "config": dict(self._config),
                "latest_sample": dict(self._samples[-1]) if self._samples else None,
                "samples": list(self._samples[-240:]),
                "logs": list(self._logs[-200:]),
                "stderr_tail": "\n".join(self._stderr_tail[-80:]),
                "latest_overlay_path": _run_relative_path(overlay_path, run_dir=self.session.run_dir) if overlay_path else None,
                "latest_raw_path": _run_relative_path(raw_path, run_dir=self.session.run_dir) if raw_path else None,
            }

    def _read_stream(self, proc: subprocess.Popen[str], run_id: int, stream: Any, source: str) -> None:
        try:
            for raw_line in stream:
                line = raw_line.rstrip("\n")
                self._append_log(run_id, source, line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _wait_for_process(self, proc: subprocess.Popen[str], run_id: int) -> None:
        returncode = proc.wait()
        with self._lock:
            if self._proc is proc and self._run_id == run_id:
                self._returncode = returncode
                self._stopped_t = time.time()

    def _append_log(self, run_id: int, source: str, line: str) -> None:
        now = time.time()
        with self._lock:
            if run_id != self._run_id:
                return
            self._logs.append({"t": now, "source": source, "line": line})
            self._logs = self._logs[-1200:]
            if source == "stderr":
                self._stderr_tail.append(line)
                self._stderr_tail = self._stderr_tail[-200:]
            if source == "stdout" and line.startswith("object-drive armed="):
                fields = _parse_object_drive_status_fields(line)
                sample = _visual_servo_status_sample(fields, t=now)
                sample["line"] = line
                self._samples.append(sample)
                self._samples = self._samples[-2000:]

    def _stop_robot(self) -> None:
        try:
            self.session.tools.stop()
        except Exception as exc:
            with self._lock:
                self._logs.append({"t": time.time(), "source": "dashboard", "line": f"robot stop failed: {exc}"})
                self._logs = self._logs[-1200:]


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _visual_servo_status_sample(fields: dict[str, str], *, t: float | None = None) -> dict[str, Any]:
    motor1, motor2 = _parse_motor_pair(fields.get("cmd"))
    return {
        "t": time.time() if t is None else t,
        "armed": _bool_payload(fields.get("armed"), default=False),
        "prediction_count": _int_or_none(fields.get("pred")),
        "track_count": _int_or_none(fields.get("track")),
        "imu_prediction_count": _int_or_none(fields.get("imu_pred")),
        "frame_seq": _int_or_none(fields.get("frame_seq")),
        "frame_age_s": _float_or_none(fields.get("frame_age_s")),
        "filter": fields.get("filter"),
        "publish_count": _int_or_none(fields.get("pub")),
        "command": fields.get("cmd"),
        "motor1_percent": motor1,
        "motor2_percent": motor2,
        "heading_error_deg": _float_or_none(fields.get("heading_error")),
        "target_yaw_deg": _float_or_none(fields.get("target_yaw")),
        "turn_percent": _float_or_none(fields.get("turn")),
        "forward_percent": _float_or_none(fields.get("forward")),
        "bbox_cx_frac": _float_or_none(fields.get("bbox_cx_frac")),
        "bbox_cy_frac": _float_or_none(fields.get("bbox_cy_frac")),
        "bbox_w_frac": _float_or_none(fields.get("bbox_w_frac")),
        "bbox_h_frac": _float_or_none(fields.get("bbox_h_frac")),
        "detection": fields.get("det"),
        "detection_count": _int_or_none(fields.get("det_count")),
        "target_lock_reject_count": _int_or_none(fields.get("lock_rej")),
        "target_lock_gate_deg": _float_or_none(fields.get("lock_gate")),
        "target_lock_error_deg": _float_or_none(fields.get("lock_err")),
        "detector_pending": _bool_payload(fields.get("pending"), default=False),
        "lost_count": _int_or_none(fields.get("lost")),
    }


def _parse_motor_pair(value: str | None) -> tuple[float | None, float | None]:
    if not value or value == "none":
        return None, None
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)%?\s*$", value)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _raw_agent_tools(tools: Any) -> Any:
    return getattr(tools, "inner", tools)


def _forward_power_cap(tools: Any) -> float | None:
    cap = getattr(tools, "max_forward_power_percent", None)
    if cap is None:
        return None
    try:
        return abs(float(cap))
    except (TypeError, ValueError):
        return None


def _limit_forward_power(power: float, cap: float | None) -> float:
    if cap is None:
        return power
    return max(0.0, min(float(cap), float(power)))


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bool_payload(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _latest_image_path(directory: Path | None) -> Path | None:
    if directory is None:
        return None
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(directory.glob(pattern))
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _run_relative_path(path: Path, *, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except (OSError, ValueError):
        return str(path)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _resolve_dashboard_artifact_path(path_value: Any, *, run_dir: Path) -> Path:
    path = Path(str(path_value))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, run_dir / path, path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _safe_dashboard_artifact_path(path_value: Any, *, run_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    candidates = [path] if path.is_absolute() else [run_dir / path, Path.cwd() / path, path]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            run_root = run_dir.resolve()
            resolved.relative_to(run_root)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and _dashboard_artifact_mimetype(resolved) is not None:
            return resolved
    return None


def _dashboard_artifact_mimetype(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return None


def _add_live_headers(response: Any, telemetry: dict[str, Any]) -> None:
    if telemetry.get("frame_seq") is not None:
        response.headers["X-Frame-Seq"] = str(telemetry["frame_seq"])
    if telemetry.get("frame_received_age_s") is not None:
        response.headers["X-Frame-Age-Ms"] = f"{float(telemetry['frame_received_age_s']) * 1000.0:.0f}"
    if telemetry.get("imu_seq") is not None:
        response.headers["X-Imu-Seq"] = str(telemetry["imu_seq"])
    if telemetry.get("yaw_deg") is not None:
        response.headers["X-Yaw-Deg"] = f"{float(telemetry['yaw_deg']):.3f}"
    if telemetry.get("imu_received_age_s") is not None:
        response.headers["X-Imu-Age-Ms"] = f"{float(telemetry['imu_received_age_s']) * 1000.0:.0f}"
    if telemetry.get("path") is not None:
        response.headers["X-Frame-Path"] = str(telemetry["path"])
    response.headers["Cache-Control"] = "no-store"


def build_session(args: argparse.Namespace) -> HarnessSession:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / stamp
    runner_name = "codex" if args.live_codex else args.runner
    critic_mode = _resolve_critic_mode(runner_name, args.critic_mode)
    model_name = args.qwen_model if runner_name == "qwen" else args.model
    config = HarnessConfig(
        run_dir=run_dir,
        model=model_name,
        reasoning_effort=args.reasoning_effort,
        actor_rules=tuple(rule for rule in [forward_power_limit_rule(args.max_forward_power_percent)] if rule),
        critic_mode=critic_mode,
        max_steps=args.max_steps,
        sleep_scale=1.0,
        rerun_enabled=args.rerun,
    )
    raw_tools = AgentTools(
        run_dir=run_dir,
        namespace=args.namespace,
        connect=args.connect,
        reverse_yaw=not args.no_reverse_yaw,
        object_drive_detector=args.object_drive_detector,
        topomap_memory_map_dir=args.topomap_memory_map_dir,
        topomap_memory_use_clip=args.topomap_memory_use_clip,
        topomap_memory_allow_semantic_terms=args.topomap_memory_allow_semantic_terms,
    )
    tools = ArmedGuardTools(
        raw_tools,
        armed=args.arm,
        max_forward_power_percent=args.max_forward_power_percent,
    )
    if runner_name == "codex":
        actor = CodexExecRunner(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            codex_binary=args.codex_binary,
            cwd=run_dir,
        )
    elif runner_name == "qwen":
        actor = OpenAICompatibleVisionRunner(
            model=args.qwen_model,
            endpoint=args.qwen_endpoint,
            temperature=args.qwen_temperature,
            max_tokens=args.qwen_max_tokens,
        )
    elif runner_name == "scripted-open-vocab":
        actor = ScriptedOpenVocabRunner(visual_servo_detector=args.object_drive_detector)
    elif runner_name == "fast-wall-clock":
        from .agent_candidates.fast_wall_clock import FastWallClockActor

        actor = FastWallClockActor()
    elif runner_name == "fast-demo":
        from .agent_candidates.fast_wall_clock import FastWallClockActor

        actor = FastWallClockActor(allow_stop=False, stale_drive_limit=5)
    else:
        actor = DeterministicHarnessRunner()
    if critic_mode == "none":
        critic = NoopCriticRunner()
    elif critic_mode == "same-model":
        critic = actor
    elif runner_name in {"fast-wall-clock", "fast-demo"}:
        from .agent_candidates.fast_wall_clock import FastWallClockCritic

        critic = FastWallClockCritic()
    else:
        critic = SafetyCriticRunner()
    rerun_logger = None
    if args.rerun:
        rerun_logger = HarnessRerunLogger(
            recording_id=f"flatdisk_llm_harness_{stamp}",
            save_path=run_dir / "harness.rrd",
            spawn=args.rerun_spawn,
        )
    return HarnessSession(config=config, tools=tools, actor=actor, critic=critic, rerun_logger=rerun_logger)


def _resolve_critic_mode(runner: str, mode: str) -> str:
    mode = str(mode or "auto").strip().lower()
    if mode not in {"auto", "none", "safety", "same-model"}:
        raise ValueError(f"unknown critic mode: {mode}")
    if mode != "auto":
        return mode
    if runner == "qwen":
        return "none"
    if runner == "codex":
        return "same-model"
    return "safety"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "llm_harness_dashboard")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--live-codex", action="store_true", help="Use codex exec; kept as an alias for --runner codex.")
    parser.add_argument(
        "--runner",
        choices=("deterministic", "scripted-open-vocab", "fast-wall-clock", "fast-demo", "qwen", "codex"),
        default="qwen",
        help="Actor runner. qwen/codex are model-based; deterministic/scripted/fast runners are explicit smoke-test paths.",
    )
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--qwen-model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-max-tokens", type=int, default=512)
    parser.add_argument(
        "--critic-mode",
        choices=("auto", "none", "safety", "same-model"),
        default="auto",
        help="Critic selection. auto uses no critic for Qwen, same-model for Codex, and safety for scripted baselines.",
    )
    parser.add_argument(
        "--object-drive-detector",
        choices=("florence-mlx", "florence-transformers", "grounding-dino"),
        default="florence-mlx",
    )
    parser.add_argument("--topomap-memory-map-dir", type=Path, default=None)
    parser.add_argument("--topomap-memory-use-clip", action="store_true")
    parser.add_argument("--topomap-memory-allow-semantic-terms", action="store_true")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--connect", default=default_zenoh_connect())
    parser.add_argument("--arm", action="store_true", help="Allow dashboard actions to publish motor commands.")
    parser.add_argument(
        "--max-forward-power-percent",
        type=float,
        default=10.0,
        help="Maximum forward/reverse motor percent allowed for drive_straight, reverse, and visual_servo_object.",
    )
    parser.add_argument("--no-reverse-yaw", action="store_true")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun-spawn", action="store_true")
    args = parser.parse_args()
    if args.max_forward_power_percent <= 0:
        parser.error("--max-forward-power-percent must be positive")
    return args


def main() -> int:
    args = parse_args()
    app = create_app(session=build_session(args), max_worker_steps=args.max_steps)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


def _placeholder_image(run_dir: Path) -> Path:
    path = run_dir / "dashboard_placeholder.jpg"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), (9, 7, 13))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 150, 320, 240), fill=(16, 13, 22))
    draw.polygon(((96, 72), (232, 72), (218, 142), (82, 142)), fill=(23, 17, 31), outline=(255, 79, 216))
    draw.line((160, 240, 156, 142), fill=(165, 108, 255), width=2)
    draw.text((112, 168), "awaiting goal", fill=(245, 238, 248))
    image.save(path, format="JPEG", quality=90)
    return path


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flat Disk Harness</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #09070d;
      --surface: #100d16;
      --surface-2: #17111f;
      --line: #34243f;
      --line-strong: #5b3b70;
      --ink: #f5eef8;
      --muted: #a89ab2;
      --accent: #ff4fd8;
      --accent-2: #a56cff;
      --green: #45f0b0;
      --amber: #f7c85f;
      --blue: #8ba6ff;
      --red: #ff667a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; }
    .shell {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    header {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      font-size: 14px;
      font-weight: 780;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .brand::before {
      content: "";
      width: 13px;
      height: 27px;
      background: var(--accent);
      clip-path: polygon(0 0, 100% 0, 46% 100%, 0 100%);
      display: inline-block;
    }
    .goalbar {
      display: grid;
      grid-template-columns: 1fr auto auto auto auto auto auto;
      gap: 8px;
      min-width: 0;
    }
    input {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      padding: 10px 12px;
      background: #07050b;
      color: var(--ink);
      outline: none;
    }
    input:focus { border-color: var(--accent); outline: 2px solid rgba(255, 79, 216, 0.22); outline-offset: 0; }
    button, .navbutton {
      border: 1px solid var(--line);
      background: #0b0810;
      color: var(--ink);
      padding: 10px 12px;
      min-width: 42px;
      min-height: 40px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    }
    button:hover, .navbutton:hover { border-color: var(--accent); color: #fff; }
    button.primary { background: var(--accent); border-color: var(--accent); color: #08050b; font-weight: 760; }
    button.warn { background: transparent; border-color: var(--accent-2); color: #d9c5ff; }
    button.stop { background: transparent; border-color: var(--red); color: #ff9aa8; }
    .mode {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      background: #07050b;
      color: var(--muted);
      font-size: 13px;
      font-weight: 680;
      text-transform: uppercase;
      clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    }
    .dot { width: 9px; height: 9px; background: var(--muted); }
    .mode.auto .dot { background: var(--green); }
    .mode.paused .dot { background: var(--amber); }
    .mode.teleop .dot { background: var(--blue); }
    .mode.error .dot { background: var(--red); }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.15fr) minmax(320px, 0.85fr);
      gap: 18px;
      padding: 18px;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    section {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 0;
      min-height: 0;
    }
    .left {
      display: grid;
      grid-template-rows: minmax(280px, 0.78fr) auto minmax(230px, 0.48fr);
      gap: 14px;
      min-height: 0;
    }
    .camera {
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 280px;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 11px 13px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .section-head span:first-child {
      color: var(--ink);
    }
    .camera img {
      width: 100%;
      height: 100%;
      min-height: 0;
      object-fit: cover;
      display: block;
      background: #050408;
    }
    .teleop {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(5, minmax(70px, 1fr));
      gap: 8px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0;
      padding: 0;
    }
    .metric {
      border: 0;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      min-width: 0;
      background: transparent;
    }
    .metric:nth-child(4n) { border-right: 0; }
    .metric b {
      display: block;
      font-size: 17px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      overflow-wrap: anywhere;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .right {
      display: grid;
      grid-template-rows: minmax(260px, 0.95fr) minmax(230px, 0.78fr) minmax(170px, 0.48fr);
      gap: 14px;
      min-height: 0;
    }
    .log, .tools, .memory {
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .scroll {
      overflow: auto;
      padding: 12px;
      min-height: 0;
    }
    .event {
      border-left: 3px solid var(--line-strong);
      border-bottom: 1px solid var(--line);
      padding: 10px 0 10px 10px;
      font-size: 13px;
    }
    .event:last-child { border-bottom: 0; }
    .event.actor { border-left-color: var(--accent); }
    .event.actor_request, .event.critic_request { border-left-color: var(--amber); }
    .event.critic { border-left-color: var(--accent-2); }
    .event.safety_gate { border-left-color: var(--red); }
    .event.observation { border-left-color: var(--blue); }
    .event.tool_start { border-left-color: var(--amber); }
    .event.tool_result { border-left-color: var(--green); }
    .event.tool_error, .event.actor_error, .event.error { border-left-color: var(--red); }
    .event.command, .event.teleop { border-left-color: var(--green); }
    .event .who {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 740;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .event .who span:first-child {
      color: var(--ink);
    }
	    pre {
	      margin: 6px 0 0;
	      white-space: pre-wrap;
	      overflow-wrap: anywhere;
	      font-size: 12px;
	      line-height: 1.38;
	      color: #e7ddeb;
	      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
	    }
	    .tool-visual {
	      margin-top: 9px;
	      padding-top: 9px;
	      border-top: 1px solid var(--line);
	    }
	    .tool-visual-title {
	      color: var(--ink);
	      font-size: 11px;
	      font-weight: 760;
	      text-transform: uppercase;
	      letter-spacing: 0.06em;
	    }
	    .tool-summary {
	      display: grid;
	      grid-template-columns: repeat(2, minmax(0, 1fr));
	      gap: 5px;
	      margin-top: 7px;
	    }
	    .tool-chip {
	      min-width: 0;
	      border: 1px solid var(--line);
	      background: #07050b;
	      padding: 6px 7px;
	    }
	    .tool-chip span {
	      display: block;
	      color: var(--muted);
	      font-size: 10px;
	      font-weight: 720;
	      text-transform: uppercase;
	      letter-spacing: 0.05em;
	    }
	    .tool-chip b {
	      display: block;
	      margin-top: 2px;
	      color: var(--ink);
	      font-size: 12px;
	      font-weight: 620;
	      overflow-wrap: anywhere;
	    }
	    .artifact-grid {
	      display: grid;
	      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
	      gap: 8px;
	      margin-top: 8px;
	    }
	    .artifact {
	      display: block;
	      min-width: 0;
	      border: 1px solid var(--line);
	      background: #050408;
	      color: var(--ink);
	      text-decoration: none;
	    }
	    .artifact-label {
	      display: block;
	      padding: 6px 7px;
	      border-bottom: 1px solid var(--line);
	      color: var(--muted);
	      font-size: 10px;
	      font-weight: 720;
	      text-transform: uppercase;
	      letter-spacing: 0.05em;
	      overflow-wrap: anywhere;
	    }
	    .artifact img {
	      width: 100%;
	      max-height: 230px;
	      object-fit: contain;
	      display: block;
	      background: #050408;
	    }
	    @media (max-width: 980px) {
      .shell {
        min-height: 100vh;
        height: auto;
        overflow: auto;
      }
      header { grid-template-columns: 1fr; }
      .goalbar { grid-template-columns: 1fr 1fr 1fr; }
      .goalbar input { grid-column: 1 / -1; }
      main {
        grid-template-columns: 1fr;
        height: auto;
        overflow: visible;
      }
      .left { grid-template-rows: auto auto auto; }
      .right { grid-template-rows: auto auto; }
      .camera { height: 420px; }
      .log, .tools, .memory { min-height: 280px; }
      .scroll { max-height: 420px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(4n) { border-right: 1px solid var(--line); }
      .metric:nth-child(2n) { border-right: 0; }
    }
    @media (max-width: 560px) {
      main { padding: 10px; gap: 10px; }
      .camera { height: 300px; }
      .scroll { max-height: 360px; }
      .teleop { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .goalbar { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">Flat Disk Harness</div>
      <div class="goalbar">
        <input id="goal" value="Drive to the sofa in the living room." aria-label="Goal">
        <button class="primary" id="go">Go</button>
        <button class="warn" id="pause">Pause</button>
        <button id="resume">Resume</button>
        <button id="resetContext">Reset Ctx</button>
        <a class="navbutton" href="/servo">Servo Test</a>
        <button class="stop" id="stop">Stop</button>
      </div>
      <div class="mode idle" id="mode"><span class="dot"></span><span>idle</span></div>
    </header>
    <main>
      <div class="left">
        <section class="camera">
          <div class="section-head"><span>Camera</span><span id="frameSeq">frame --</span></div>
          <img id="camera" src="/api/latest-frame" alt="Robot camera">
        </section>
        <section>
          <div class="teleop">
            <button data-teleop="left">Left</button>
            <button data-teleop="forward">Forward</button>
            <button data-teleop="right">Right</button>
            <button data-teleop="reverse">Reverse</button>
            <button class="stop" data-teleop="stop">Stop</button>
          </div>
        </section>
        <section>
          <div class="section-head"><span>Run</span><span id="worker">worker idle</span></div>
          <div class="metrics">
            <div class="metric"><b id="step">0</b><span>Step</span></div>
            <div class="metric"><b id="yaw">--</b><span>Yaw</span></div>
            <div class="metric"><b id="currentMode">--</b><span>Mode</span></div>
            <div class="metric"><b id="runnerName">--</b><span>Runner</span></div>
            <div class="metric"><b id="modelName">--</b><span>Model</span></div>
            <div class="metric"><b id="reasoning">--</b><span>Reasoning</span></div>
            <div class="metric"><b id="rerunStatus">--</b><span>Rerun</span></div>
            <div class="metric"><b id="schemaStatus">--</b><span>Schema</span></div>
            <div class="metric"><b id="boundary">--</b><span>Policy input</span></div>
            <div class="metric"><b id="contextId">--</b><span>Context</span></div>
            <div class="metric"><b id="memoryCount">--</b><span>Memory</span></div>
            <div class="metric"><b id="runDir">--</b><span>Run dir</span></div>
          </div>
        </section>
      </div>
      <div class="right">
        <section class="log">
          <div class="section-head"><span>Reasoning Trace</span><span id="traceCount">0 events</span></div>
          <div class="scroll" id="traceEvents"></div>
        </section>
        <section class="tools">
          <div class="section-head"><span>Tool I/O</span><span id="toolCount">0 events</span></div>
          <div class="scroll" id="toolEvents"></div>
        </section>
        <section class="memory">
          <div class="section-head"><span>Memory</span><span id="memoryPath">memory.jsonl</span></div>
          <div class="scroll" id="memory"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const state = { frameBusy: false, liveBusy: false, frameObjectUrl: "", liveFrameSeq: "", liveImuSeq: "" };
    const q = (id) => document.getElementById(id);
    async function post(path, body = {}) {
      const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function parseJsonDeep(value) {
      if (typeof value === "string") {
        const trimmed = value.trim();
        const looksJson = (trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"));
        if (looksJson) {
          try { return parseJsonDeep(JSON.parse(trimmed)); } catch { return value; }
        }
        return value;
      }
      if (Array.isArray(value)) return value.map((item) => parseJsonDeep(item));
      if (value && typeof value === "object") {
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, parseJsonDeep(item)]));
      }
      return value;
    }
    function compact(value) {
      if (value === undefined || value === null) return "";
      const normalized = parseJsonDeep(value);
      if (typeof normalized === "string") return normalized;
      return JSON.stringify(normalized, null, 2);
    }
    function parseMaybeJson(value) {
      return parseJsonDeep(value);
    }
    function actionLine(action) {
      if (!action) return "";
      const args = action.args ? JSON.stringify(action.args) : "{}";
      return `${action.tool || "action"} ${args}`;
    }
    function eventClass(event) {
      return String(event.event || "event").replace(/[^a-z0-9_-]/gi, "_");
    }
    function eventTitle(event) {
      if (event.event === "actor_request") return `actor request step ${event.step ?? "--"}`;
      if (event.event === "actor") return `actor step ${event.step ?? "--"}`;
      if (event.event === "actor_parse_retry") return `actor retry step ${event.step ?? "--"}`;
      if (event.event === "actor_error") return `actor error step ${event.step ?? "--"}`;
      if (event.event === "critic_request") return `critic request step ${event.step ?? "--"}`;
      if (event.event === "critic") return `critic step ${event.step ?? "--"}`;
      if (event.event === "safety_gate") return `safety gate step ${event.step ?? "--"}`;
      if (event.event === "tool_preflight") return `tool preflight step ${event.step ?? "--"}`;
      if (event.event === "tool_start") return `tool input step ${event.step ?? "--"}`;
      if (event.event === "tool_result") return `tool output step ${event.step ?? "--"}`;
      if (event.event === "tool_error") return `tool error step ${event.step ?? "--"}`;
      if (event.event === "observation") return `camera/imu frame ${event.frame_seq ?? "--"}`;
      if (event.event === "teleop") return `teleop ${event.command || ""}`;
      return event.event || "event";
    }
	    function eventText(event) {
	      if (event.event === "actor_request" || event.event === "critic_request") {
	        return compact({
          runner: event.runner,
          prompt_path: event.prompt_path,
          prompt_chars: event.prompt_chars,
          image_paths: event.image_paths,
          prompt_preview: event.prompt_preview,
        });
      }
      if (event.event === "actor") {
        const output = parseMaybeJson(event.output);
        const action = event.action || output.action;
        const trace = event.runner_trace || {};
        return compact({
          thought: event.thought || output.thought || "",
          decision_summary: event.decision_summary || output.decision_summary,
          completion_check: event.completion_check || output.completion_check,
          action,
          grounding_audit: event.grounding_audit || output.grounding_audit,
          memory_update: event.memory_update || output.memory_update,
          qwen_reasoning_content: trace.reasoning_content,
          elapsed_s: trace.elapsed_s,
          finish_reason: trace.finish_reason,
          usage: trace.usage,
          raw_output: output,
        });
      }
      if (event.event === "actor_parse_retry" || event.event === "actor_error") {
        return compact({
          error: event.error,
          prompt_path: event.prompt_path,
          previous_output: event.previous_output,
          output: event.output,
          image_paths: event.image_paths,
        });
      }
      if (event.event === "tool_start") {
        return compact({
          source: event.source,
          action: event.action,
          execution_context: event.execution_context,
        });
      }
      if (event.event === "tool_preflight") {
        return compact({
          source: event.source,
          action: event.action,
          actor_frame_seq: event.actor_frame_seq,
          actor_obs_age_s_at_tool_start: event.actor_obs_age_s_at_tool_start,
          tool_preflight_frame_seq: event.tool_preflight_frame_seq,
          tool_preflight_frame_age_s: event.tool_preflight_frame_age_s,
          tool_preflight_frame_delta_from_actor: event.tool_preflight_frame_delta_from_actor,
          tool_preflight_yaw_deg: event.tool_preflight_yaw_deg,
          tool_preflight_imu_seq: event.tool_preflight_imu_seq,
          tool_preflight_imu_age_s: event.tool_preflight_imu_age_s,
        });
      }
      if (event.event === "tool_result") {
        return compact({
          source: event.source,
          action: event.action,
          result: event.result,
        });
      }
      if (event.event === "tool_error") {
        return compact({
          source: event.source,
          action: event.action,
          error: event.error,
        });
      }
      if (event.event === "critic") {
        const model = event.model_decision || {};
        const finalDecision = event.decision || {};
        const trace = event.runner_trace || {};
        return compact({
          model_verdict: model.verdict,
          final_verdict: finalDecision.verdict,
          reason: finalDecision.reason || model.reason,
          selected_action: event.selected_action,
          qwen_reasoning_content: trace.reasoning_content,
          finish_reason: trace.finish_reason,
          usage: trace.usage,
        });
      }
      if (event.event === "safety_gate") {
        return compact({
          model: event.model_decision,
          safety: event.safety_decision,
        });
      }
      if (event.event === "observation") return compact({ yaw_deg: event.yaw_deg, frame_seq: event.frame_seq, detections: event.detections });
	      if (event.event === "command") return compact(event.action);
	      return compact(event);
	    }
	    function toolResultFromEvent(event) {
	      const result = parseMaybeJson(event.result);
	      return result && typeof result === "object" && !Array.isArray(result) ? result : {};
	    }
	    function toolNameFromInputs(result, actionInput) {
	      if (result && typeof result === "object" && result.action) return String(result.action);
	      const action = parseMaybeJson(actionInput);
	      if (action && typeof action === "object" && action.tool) return String(action.tool);
	      return "";
	    }
	    function inlineValue(value) {
	      const parsed = parseMaybeJson(value);
	      if (parsed === undefined || parsed === null || parsed === "") return "";
	      if (typeof parsed === "object") return JSON.stringify(parsed);
	      return String(parsed);
	    }
	    function normalizedArtifactPath(value) {
	      if (typeof value !== "string") return "";
	      const trimmed = value.trim();
	      if (!trimmed || trimmed === "None" || trimmed === "null") return "";
	      return trimmed;
	    }
	    function artifactUrl(path) {
	      return `/api/artifact?path=${encodeURIComponent(String(path))}`;
	    }
	    function artifactItems(result) {
	      const specs = [
	        ["grounding_audit_contact_sheet", "Raw + detector audit"],
	        ["debug_overlay_contact_sheet", "Detector / tracker strip"],
	        ["motion_contact_sheet", "Raw motion strip"],
	        ["overlay_path", "Grounding overlay"],
	        ["topomap_contact_sheet", "Topomap contact sheet"],
	      ];
	      return specs.map(([key, label]) => ({ key, label, path: normalizedArtifactPath(result[key]) })).filter((item) => item.path);
	    }
	    function toolSummaryRows(result, actionName) {
	      const rows = [];
	      const add = (label, value) => {
	        const text = inlineValue(value);
	        if (text) rows.push([label, text]);
	      };
	      if (actionName) add("tool", actionName);
	      add("prompt", result.prompt);
	      add("detector", result.detector);
	      add("servo", result.servo_status || result.status);
	      add("grounding", result.grounding_stability);
	      add("semantic", result.semantic_identity);
	      add("detected", result.target_detected);
	      add("last detection", result.last_detection || result.selected_label);
	      add("last command", result.last_command);
	      add("ready", result.ready_for_visual_servo);
	      add("selected score", result.selected_score);
	      add("geometry", result.grounding_geometry_warning);
	      return rows;
	    }
	    function toolVisualHtmlFromResult(resultInput, actionInput) {
	      const result = parseMaybeJson(resultInput);
	      if (!result || typeof result !== "object" || Array.isArray(result)) return "";
	      const actionName = toolNameFromInputs(result, actionInput);
	      const artifacts = artifactItems(result);
	      const detectorLike = ["visual_servo_object", "check_object_grounding", "query_topomap_memory"].includes(actionName);
	      if (!detectorLike && artifacts.length === 0) return "";
	      const summary = toolSummaryRows(result, actionName).slice(0, 12).map(([label, value]) => `
	        <div class="tool-chip"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`).join("");
	      const artifactHtml = artifacts.map((item) => `
	        <a class="artifact" href="${escapeHtml(artifactUrl(item.path))}" target="_blank" rel="noopener">
	          <span class="artifact-label">${escapeHtml(item.label)}</span>
	          <img src="${escapeHtml(artifactUrl(item.path))}" alt="${escapeHtml(item.label)}" loading="lazy">
	        </a>`).join("");
	      return `
	        <div class="tool-visual">
	          <div class="tool-visual-title">Florence / Grounding Output</div>
	          ${summary ? `<div class="tool-summary">${summary}</div>` : ""}
	          ${artifactHtml ? `<div class="artifact-grid">${artifactHtml}</div>` : ""}
	        </div>`;
	    }
	    function toolVisualHtml(event) {
	      if (event.event !== "tool_result") return "";
	      return toolVisualHtmlFromResult(toolResultFromEvent(event), event.action);
	    }
	    function isTraceEvent(event) {
	      return ["actor_request", "actor", "actor_parse_retry", "actor_error", "critic_request", "critic", "safety_gate", "error"].includes(event.event);
	    }
    function isToolEvent(event) {
      return ["tool_preflight", "tool_start", "tool_result", "tool_error", "teleop"].includes(event.event);
    }
    function renderEventList(targetId, countId, events) {
      q(countId).textContent = `${events.length} events`;
	      q(targetId).innerHTML = events.slice().reverse().map((event) => `
	        <div class="event ${eventClass(event)}">
	          <div class="who"><span>${escapeHtml(eventTitle(event))}</span><span>${new Date((event.t || 0) * 1000).toLocaleTimeString()}</span></div>
	          <pre>${escapeHtml(eventText(event))}</pre>
	          ${toolVisualHtml(event)}
	        </div>`).join("");
	    }
	    function renderMemory(records) {
	      q("memory").innerHTML = records.slice().reverse().map((record) => `
	        <div class="event">
	          <div class="who"><span>step ${record.step ?? "--"}</span><span>${record.mode || ""}</span></div>
	          <pre>${escapeHtml(compact({
	            action: record.executed_action,
	            decision_summary: record.actor_decision_summary,
	            completion_check: record.actor_completion_check,
	            critic: record.critic,
	            result: record.tool_result,
	          }))}</pre>
	          ${toolVisualHtmlFromResult(record.tool_result, record.executed_action)}
	        </div>`).join("");
	    }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    async function refresh() {
      const res = await fetch("/api/state");
      const data = await res.json();
      const mode = data.mode || "idle";
      const completionReason = data.completion_reason || "";
      const modeText = completionReason ? `${mode} / ${completionReason}` : mode;
      q("mode").className = `mode ${mode}`;
      q("mode").lastElementChild.textContent = modeText;
      q("step").textContent = data.step ?? 0;
      q("currentMode").textContent = modeText;
      q("worker").textContent = data.worker_active ? "worker active" : "worker idle";
      q("runDir").textContent = (data.run_dir || "").split("/").slice(-2).join("/");
      q("memoryPath").textContent = (data.memory_path || "").split("/").slice(-1)[0] || "memory.jsonl";
      const meta = data.metadata || {};
      q("runnerName").textContent = runnerLabel(meta);
      q("modelName").textContent = meta.model || "--";
      q("reasoning").textContent = meta.reasoning_effort || "--";
      q("rerunStatus").textContent = meta.rerun_path ? "rrd" : (meta.rerun_enabled ? "pending" : "off");
      q("schemaStatus").textContent = meta.codex_schema_dir ? "schemas" : (meta.live_codex ? "pending" : "n/a");
      q("boundary").textContent = Array.isArray(meta.policy_input_allowlist) ? meta.policy_input_allowlist.slice(0, 3).join(", ") : "--";
      q("contextId").textContent = `ctx ${data.context_generation ?? 0}`;
      q("memoryCount").textContent = `${data.memory_record_count ?? 0} rec`;
      const obs = data.last_observation || {};
      if (!state.liveImuSeq) q("yaw").textContent = obs.yaw_deg === undefined ? "--" : `${Number(obs.yaw_deg).toFixed(1)} deg`;
      if (!state.liveFrameSeq) q("frameSeq").textContent = obs.frame_seq === undefined ? "frame --" : `frame ${obs.frame_seq}`;
      const events = data.recent_events || [];
      renderEventList("traceEvents", "traceCount", events.filter(isTraceEvent));
      renderEventList("toolEvents", "toolCount", events.filter(isToolEvent));
      renderMemory(data.recent_memory || []);
    }
    async function refreshFrame() {
      if (state.frameBusy) return;
      state.frameBusy = true;
      try {
        const res = await fetch(`/api/latest-frame?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) return;
        const blob = await res.blob();
        const nextUrl = URL.createObjectURL(blob);
        const previousUrl = state.frameObjectUrl;
        q("camera").onload = () => { if (previousUrl) URL.revokeObjectURL(previousUrl); };
        q("camera").src = nextUrl;
        state.frameObjectUrl = nextUrl;
        const frameSeq = res.headers.get("X-Frame-Seq");
        const frameAgeMs = res.headers.get("X-Frame-Age-Ms");
        if (frameSeq) {
          state.liveFrameSeq = frameSeq;
          q("frameSeq").textContent = frameAgeMs ? `frame ${frameSeq} / ${frameAgeMs} ms` : `frame ${frameSeq}`;
        }
        const yawDeg = res.headers.get("X-Yaw-Deg");
        const imuSeq = res.headers.get("X-Imu-Seq");
        const imuAgeMs = res.headers.get("X-Imu-Age-Ms");
        if (yawDeg && imuSeq) {
          state.liveImuSeq = imuSeq;
          q("yaw").textContent = imuAgeMs ? `${Number(yawDeg).toFixed(1)} deg / ${imuAgeMs} ms` : `${Number(yawDeg).toFixed(1)} deg`;
        }
      } catch {
      } finally {
        state.frameBusy = false;
      }
    }
    async function refreshLive() {
      if (state.liveBusy) return;
      state.liveBusy = true;
      try {
        const res = await fetch(`/api/live?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) return;
        const payload = await res.json();
        const telemetry = payload.telemetry || {};
        if (telemetry.frame_seq !== undefined) {
          state.liveFrameSeq = String(telemetry.frame_seq);
          const ageMs = telemetry.frame_received_age_s === undefined ? null : Math.round(Number(telemetry.frame_received_age_s) * 1000);
          q("frameSeq").textContent = ageMs === null ? `frame ${telemetry.frame_seq}` : `frame ${telemetry.frame_seq} / ${ageMs} ms`;
        }
        if (telemetry.yaw_deg !== undefined && telemetry.imu_seq !== undefined) {
          state.liveImuSeq = String(telemetry.imu_seq);
          const ageMs = telemetry.imu_received_age_s === undefined ? null : Math.round(Number(telemetry.imu_received_age_s) * 1000);
          q("yaw").textContent = ageMs === null ? `${Number(telemetry.yaw_deg).toFixed(1)} deg` : `${Number(telemetry.yaw_deg).toFixed(1)} deg / ${ageMs} ms`;
        }
      } catch {
      } finally {
        state.liveBusy = false;
      }
    }
    function runnerLabel(meta) {
      if (meta.actor_runner === "OpenAICompatibleVisionRunner") return "qwen";
      if (meta.live_codex) return "codex";
      if (meta.actor_runner) return meta.actor_runner.replace(/Runner$/, "");
      return "local";
    }
    q("go").onclick = () => post("/api/goal", { goal: q("goal").value, reset_context: true }).then(refresh);
    q("pause").onclick = () => post("/api/pause").then(refresh);
    q("resume").onclick = () => post("/api/resume").then(refresh);
    q("resetContext").onclick = () => post("/api/reset-context").then(refresh);
    q("stop").onclick = () => post("/api/stop").then(refresh);
    document.querySelectorAll("[data-teleop]").forEach((button) => {
      button.onclick = () => post("/api/teleop", { command: button.dataset.teleop }).then(refresh);
    });
    refresh();
    refreshLive();
    refreshFrame();
    setInterval(refresh, 900);
    setInterval(refreshLive, 100);
    setInterval(refreshFrame, 150);
  </script>
</body>
</html>
"""


SERVO_TEST_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visual Servo Test</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #09070d;
      --surface: #100d16;
      --surface-2: #17111f;
      --line: #34243f;
      --line-strong: #5b3b70;
      --ink: #f5eef8;
      --muted: #a89ab2;
      --accent: #ff4fd8;
      --accent-2: #a56cff;
      --green: #45f0b0;
      --amber: #f7c85f;
      --blue: #8ba6ff;
      --red: #ff667a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; }
    .shell {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    header {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 14px 18px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      font-size: 14px;
      font-weight: 780;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .brand::before {
      content: "";
      width: 13px;
      height: 27px;
      background: var(--accent);
      clip-path: polygon(0 0, 100% 0, 46% 100%, 0 100%);
      display: inline-block;
    }
    .top-actions {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    button, .navbutton {
      border: 1px solid var(--line);
      background: #0b0810;
      color: var(--ink);
      padding: 10px 12px;
      min-width: 42px;
      min-height: 40px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    }
    button:hover, .navbutton:hover { border-color: var(--accent); color: #fff; }
    button.primary { background: var(--accent); border-color: var(--accent); color: #08050b; font-weight: 760; }
    button.stop { background: transparent; border-color: var(--red); color: #ff9aa8; }
    .mode {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      background: #07050b;
      color: var(--muted);
      font-size: 13px;
      font-weight: 680;
      text-transform: uppercase;
      clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    }
    .dot { width: 9px; height: 9px; background: var(--muted); }
    .mode.running .dot { background: var(--green); }
    .mode.stopped .dot { background: var(--amber); }
    .mode.error .dot { background: var(--red); }
    main {
      display: grid;
      grid-template-columns: minmax(420px, 1.15fr) minmax(360px, 0.85fr);
      gap: 18px;
      padding: 18px;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    section {
      background: var(--surface);
      border: 1px solid var(--line);
      min-height: 0;
    }
    .left {
      display: grid;
      grid-template-rows: minmax(330px, 1fr) 220px 220px;
      gap: 14px;
      min-height: 0;
    }
    .camera {
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    .image-wrap {
      position: relative;
      min-height: 0;
      background: #050408;
    }
    .image-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
      background: #050408;
    }
    .right {
      display: grid;
      grid-template-rows: auto minmax(210px, 0.72fr) minmax(220px, 0.78fr);
      gap: 14px;
      min-height: 0;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid var(--line);
      padding: 11px 13px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 720;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .section-head span:first-child { color: var(--ink); }
    .controls {
      padding: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    .field.wide { grid-column: 1 / -1; }
    label span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 720;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    input, select {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      padding: 9px 10px;
      background: #07050b;
      color: var(--ink);
      outline: none;
    }
    input:focus, select:focus {
      border-color: var(--accent);
      outline: 2px solid rgba(255, 79, 216, 0.22);
      outline-offset: 0;
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 40px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      background: #07050b;
    }
    .check-row input { width: 18px; min-width: 18px; height: 18px; }
    .button-row {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0;
      overflow: auto;
    }
    .metric {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 10px 11px;
      min-width: 0;
    }
    .metric:nth-child(3n) { border-right: 0; }
    .metric b {
      display: block;
      font-size: 16px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-top: 3px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .plot {
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      background: #07050b;
    }
    .log {
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }
    .scroll {
      overflow: auto;
      min-height: 0;
      padding: 12px;
    }
    .logline {
      border-left: 3px solid var(--line-strong);
      border-bottom: 1px solid var(--line);
      padding: 8px 0 8px 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
      color: #e7ddeb;
    }
    .logline.stderr { border-left-color: var(--red); }
    .logline.stdout { border-left-color: var(--green); }
    .muted { color: var(--muted); }
    @media (max-width: 980px) {
      .shell { height: auto; min-height: 100vh; overflow: auto; }
      header { grid-template-columns: 1fr; }
      .top-actions { justify-content: stretch; flex-wrap: wrap; }
      main { grid-template-columns: 1fr; height: auto; overflow: visible; }
      .left, .right { grid-template-rows: auto; }
      .camera { height: 420px; }
      .plot { height: 220px; }
      .log { min-height: 280px; }
    }
    @media (max-width: 560px) {
      main { padding: 10px; gap: 10px; }
      .camera { height: 310px; }
      .controls, .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric:nth-child(3n) { border-right: 1px solid var(--line); }
      .metric:nth-child(2n) { border-right: 0; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">Visual Servo Test</div>
      <div class="mode stopped" id="mode"><span class="dot"></span><span>stopped</span></div>
      <div class="top-actions">
        <a class="navbutton" href="/">Agent Dashboard</a>
        <button class="stop" id="topStop">Stop</button>
      </div>
    </header>
    <main>
      <div class="left">
        <section class="camera">
          <div class="section-head"><span>Camera / Bbox</span><span id="imageLabel">live frame</span></div>
          <div class="image-wrap"><img id="servoImage" src="/api/latest-frame" alt="Visual servo camera"></div>
        </section>
        <section class="plot">
          <div class="section-head"><span>Commands</span><span>m1 m2 forward turn heading</span></div>
          <canvas id="plotCommands"></canvas>
        </section>
        <section class="plot">
          <div class="section-head"><span>Bbox</span><span>center and size</span></div>
          <canvas id="plotBbox"></canvas>
        </section>
      </div>
      <div class="right">
        <section>
          <div class="section-head"><span>Controls</span><span id="armedLabel">armed --</span></div>
          <div class="controls">
            <label class="field wide"><span>Prompt</span><input id="prompt" value="chair"></label>
            <label class="field"><span>Duration s</span><input id="durationS" type="number" min="0.5" max="1800" step="0.5" value="30"></label>
            <label class="field"><span>Detector</span><select id="detector"><option>florence-mlx</option><option>florence-transformers</option><option>grounding-dino</option></select></label>
            <label class="field"><span>Forward %</span><input id="forwardPower" type="number" min="0" max="60" step="0.5" value="10"></label>
            <label class="field"><span>Heading kp</span><input id="headingKp" type="number" min="0" max="100" step="0.5" value="8"></label>
            <label class="field"><span>Max turn %</span><input id="maxTurnPercent" type="number" min="0" max="100" step="0.5" value="10"></label>
            <label class="field"><span>Min turn %</span><input id="minTurnPercent" type="number" min="0" max="100" step="0.5" value="1.5"></label>
            <label class="field"><span>Deadband deg</span><input id="headingDeadbandDeg" type="number" min="0" max="45" step="0.25" value="0"></label>
            <label class="field"><span>Max output %</span><input id="maxAbsOutput" type="number" min="0" max="100" step="0.5" value="20"></label>
            <label class="check-row"><input id="targetFilter" type="checkbox" checked><span>Target filter</span></label>
            <label class="check-row"><input id="targetLock" type="checkbox" checked><span>Target lock</span></label>
            <label class="field"><span>Lock gate deg</span><input id="targetLockMaxBearingDeg" type="number" min="0" max="90" step="0.5" value="12"></label>
            <label class="field"><span>IMU noise deg</span><input id="imuHeadingNoiseDeg" type="number" min="0" max="45" step="0.5" value="2"></label>
            <label class="field"><span>Model noise deg</span><input id="modelBearingNoiseDeg" type="number" min="0.1" max="90" step="0.5" value="4"></label>
            <label class="field"><span>Track noise deg</span><input id="trackBearingNoiseDeg" type="number" min="0.1" max="90" step="0.5" value="8"></label>
            <label class="field"><span>Process deg/s</span><input id="targetProcessNoiseDegS" type="number" min="0.1" max="180" step="0.5" value="10"></label>
            <label class="check-row"><input id="stopWhenLost" type="checkbox" checked><span>Stop when lost</span></label>
            <div class="button-row">
              <button class="primary" id="start">Start</button>
              <button class="stop" id="stop">Stop</button>
            </div>
          </div>
        </section>
        <section>
          <div class="section-head"><span>Live Data</span><span id="sampleCount">0 samples</span></div>
          <div class="metrics">
            <div class="metric"><b id="cmd">--</b><span>Wheel speeds</span></div>
            <div class="metric"><b id="forward">--</b><span>Forward</span></div>
            <div class="metric"><b id="turn">--</b><span>Turn</span></div>
            <div class="metric"><b id="headingError">--</b><span>Heading error</span></div>
            <div class="metric"><b id="targetYaw">--</b><span>Target yaw</span></div>
            <div class="metric"><b id="yaw">--</b><span>IMU yaw</span></div>
            <div class="metric"><b id="frameSeq">--</b><span>Frame</span></div>
            <div class="metric"><b id="frameAge">--</b><span>Frame age</span></div>
            <div class="metric"><b id="imuAge">--</b><span>IMU age</span></div>
            <div class="metric"><b id="bboxCenter">--</b><span>Bbox center</span></div>
            <div class="metric"><b id="bboxSize">--</b><span>Bbox size</span></div>
            <div class="metric"><b id="detection">--</b><span>Detection</span></div>
            <div class="metric"><b id="filter">--</b><span>Filter</span></div>
            <div class="metric"><b id="pub">--</b><span>Publishes</span></div>
            <div class="metric"><b id="lost">--</b><span>Lost</span></div>
            <div class="metric"><b id="detCount">--</b><span>Candidates</span></div>
            <div class="metric"><b id="lockReject">--</b><span>Lock rejects</span></div>
            <div class="metric"><b id="lockError">--</b><span>Lock error</span></div>
          </div>
        </section>
        <section class="log">
          <div class="section-head"><span>Servo Logs</span><span id="returncode">return --</span></div>
          <div class="scroll" id="logs"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const q = (id) => document.getElementById(id);
    const colors = ["#45f0b0", "#ff4fd8", "#8ba6ff", "#f7c85f", "#ff667a"];
    let latestOverlayPath = "";
    let liveBusy = false;
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    function num(id) {
      const value = Number(q(id).value);
      return Number.isFinite(value) ? value : undefined;
    }
    function fmt(value, digits = 1, suffix = "") {
      if (value === undefined || value === null || value === "") return "--";
      const n = Number(value);
      if (!Number.isFinite(n)) return String(value);
      return `${n.toFixed(digits)}${suffix}`;
    }
    function post(path, body = {}) {
      return fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      });
    }
    function payload() {
      return {
        prompt: q("prompt").value,
        detector: q("detector").value,
        duration_s: num("durationS"),
        forward_power: num("forwardPower"),
        heading_kp: num("headingKp"),
        max_turn_percent: num("maxTurnPercent"),
        min_turn_percent: num("minTurnPercent"),
        heading_deadband_deg: num("headingDeadbandDeg"),
        max_abs_output: num("maxAbsOutput"),
        target_filter: q("targetFilter").checked,
        target_lock: q("targetLock").checked,
        target_lock_max_bearing_deg: num("targetLockMaxBearingDeg"),
        imu_heading_noise_deg: num("imuHeadingNoiseDeg"),
        model_bearing_noise_deg: num("modelBearingNoiseDeg"),
        track_bearing_noise_deg: num("trackBearingNoiseDeg"),
        target_process_noise_deg_s: num("targetProcessNoiseDegS"),
        stop_when_lost: q("stopWhenLost").checked,
      };
    }
    function artifactUrl(path) {
      return `/api/artifact?path=${encodeURIComponent(String(path))}`;
    }
    function setMode(state) {
      const running = Boolean(state.running);
      const errored = state.returncode !== null && state.returncode !== 0;
      q("mode").className = `mode ${running ? "running" : (errored ? "error" : "stopped")}`;
      q("mode").lastElementChild.textContent = running ? "running" : (errored ? `exit ${state.returncode}` : "stopped");
      q("returncode").textContent = state.returncode === null || state.returncode === undefined ? "return --" : `return ${state.returncode}`;
      q("armedLabel").textContent = state.armed ? "armed" : "detect only";
    }
    function renderMetrics(state) {
      const sample = state.latest_sample || {};
      q("sampleCount").textContent = `${(state.samples || []).length} samples`;
      q("cmd").textContent = sample.command || "--";
      q("forward").textContent = fmt(sample.forward_percent, 1, "%");
      q("turn").textContent = fmt(sample.turn_percent, 1, "%");
      q("headingError").textContent = fmt(sample.heading_error_deg, 1, " deg");
      q("targetYaw").textContent = fmt(sample.target_yaw_deg, 1, " deg");
      q("frameSeq").textContent = sample.frame_seq ?? "--";
      q("frameAge").textContent = fmt(sample.frame_age_s, 3, " s");
      q("bboxCenter").textContent = sample.bbox_cx_frac === null || sample.bbox_cx_frac === undefined ? "--" : `${fmt(sample.bbox_cx_frac, 3)}, ${fmt(sample.bbox_cy_frac, 3)}`;
      q("bboxSize").textContent = sample.bbox_w_frac === null || sample.bbox_w_frac === undefined ? "--" : `${fmt(sample.bbox_w_frac, 3)} x ${fmt(sample.bbox_h_frac, 3)}`;
      q("detection").textContent = sample.detection || "--";
      q("filter").textContent = sample.filter || "--";
      q("pub").textContent = sample.publish_count ?? "--";
      q("lost").textContent = sample.lost_count ?? "--";
      q("detCount").textContent = sample.detection_count ?? "--";
      q("lockReject").textContent = sample.target_lock_reject_count ?? "--";
      q("lockError").textContent = sample.target_lock_error_deg === null || sample.target_lock_error_deg === undefined ? "--" : `${fmt(sample.target_lock_error_deg, 1, " deg")} / ${fmt(sample.target_lock_gate_deg, 1, " deg")}`;
    }
    function renderLogs(logs) {
      q("logs").innerHTML = (logs || []).slice().reverse().map((entry) => `
        <div class="logline ${escapeHtml(entry.source || "")}">
          <span class="muted">${new Date((entry.t || 0) * 1000).toLocaleTimeString()} ${escapeHtml(entry.source || "")}</span><br>
          ${escapeHtml(entry.line || "")}
        </div>`).join("");
    }
    function updateImage(state) {
      const overlayPath = state.latest_overlay_path || "";
      if (overlayPath) {
        latestOverlayPath = overlayPath;
        q("servoImage").src = `${artifactUrl(overlayPath)}&ts=${Date.now()}`;
        q("imageLabel").textContent = "servo overlay";
        return;
      }
      if (!latestOverlayPath) {
        q("servoImage").src = `/api/latest-frame?ts=${Date.now()}`;
        q("imageLabel").textContent = "live frame";
      }
    }
    function drawPlot(id, samples, specs, fixedMin, fixedMax) {
      const canvas = q(id);
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));
      if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#07050b";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#34243f";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i += 1) {
        const y = (height * i) / 4;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      const visible = (samples || []).slice(-120);
      const values = [];
      visible.forEach((sample) => specs.forEach((spec) => {
        const v = Number(sample[spec.key]);
        if (Number.isFinite(v)) values.push(v);
      }));
      const minValue = values.length ? Math.min(...values, fixedMin) : fixedMin;
      const maxValue = values.length ? Math.max(...values, fixedMax) : fixedMax;
      const span = Math.max(0.001, maxValue - minValue);
      specs.forEach((spec, index) => {
        ctx.strokeStyle = colors[index % colors.length];
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        visible.forEach((sample, i) => {
          const v = Number(sample[spec.key]);
          if (!Number.isFinite(v)) return;
          const x = visible.length <= 1 ? width : (i / (visible.length - 1)) * width;
          const y = height - ((v - minValue) / span) * height;
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      });
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
      specs.forEach((spec, index) => {
        ctx.fillStyle = colors[index % colors.length];
        ctx.fillText(spec.label, 10 + index * 74, 16);
      });
    }
    function renderPlots(samples) {
      drawPlot("plotCommands", samples, [
        { key: "motor1_percent", label: "m1" },
        { key: "motor2_percent", label: "m2" },
        { key: "forward_percent", label: "fwd" },
        { key: "turn_percent", label: "turn" },
        { key: "heading_error_deg", label: "err" },
      ], -25, 25);
      drawPlot("plotBbox", samples, [
        { key: "bbox_cx_frac", label: "cx" },
        { key: "bbox_cy_frac", label: "cy" },
        { key: "bbox_w_frac", label: "w" },
        { key: "bbox_h_frac", label: "h" },
      ], 0, 1);
    }
    async function refreshServo() {
      const res = await fetch(`/api/visual-servo-test/state?ts=${Date.now()}`, { cache: "no-store" });
      const payload = await res.json();
      const state = payload.state || {};
      setMode(state);
      renderMetrics(state);
      renderLogs(state.logs || []);
      renderPlots(state.samples || []);
      updateImage(state);
      const config = state.config || {};
      if (config.max_forward_power_percent !== null && config.max_forward_power_percent !== undefined) {
        q("forwardPower").max = String(config.max_forward_power_percent);
      }
    }
    async function refreshLive() {
      if (liveBusy) return;
      liveBusy = true;
      try {
        const res = await fetch(`/api/live?ts=${Date.now()}`, { cache: "no-store" });
        if (!res.ok) return;
        const payload = await res.json();
        const telemetry = payload.telemetry || {};
        q("yaw").textContent = telemetry.yaw_deg === undefined ? "--" : fmt(telemetry.yaw_deg, 1, " deg");
        q("imuAge").textContent = telemetry.imu_received_age_s === undefined ? "--" : fmt(Number(telemetry.imu_received_age_s) * 1000, 0, " ms");
      } finally {
        liveBusy = false;
      }
    }
    q("start").onclick = () => post("/api/visual-servo-test/start", payload()).then(refreshServo).catch((err) => alert(err.message));
    q("stop").onclick = () => post("/api/visual-servo-test/stop").then(refreshServo);
    q("topStop").onclick = () => post("/api/visual-servo-test/stop").then(refreshServo);
    refreshServo();
    refreshLive();
    setInterval(refreshServo, 300);
    setInterval(refreshLive, 120);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
