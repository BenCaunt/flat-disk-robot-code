"""Dashboard for slash-goal LLM control of the flat disk robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
from typing import Any

from PIL import Image, ImageDraw

from .agent_tools import AgentTools
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
from .paths import SCRATCH_ROOT
from .protocol import DEFAULT_NAMESPACE


def create_app(
    *,
    session: HarnessSession,
    max_worker_steps: int = 24,
    worker_interval_s: float = 0.25,
) -> Any:
    try:
        from flask import Flask, jsonify, request, send_file
    except ImportError as exc:  # pragma: no cover - covered by importorskip tests.
        raise RuntimeError("Flask is required for the harness dashboard") from exc

    app = Flask(__name__)
    worker_lock = threading.Lock()
    worker_thread: threading.Thread | None = None

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

    @app.get("/api/state")
    def api_state() -> Any:
        return jsonify(session.status())

    @app.get("/api/latest-frame")
    def api_latest_frame() -> Any:
        status = session.status()
        path_text = status.get("latest_frame_path")
        if not path_text:
            path_text = str(_placeholder_image(session.run_dir))
        path = Path(path_text)
        if not path.exists():
            path = _placeholder_image(session.run_dir)
        return send_file(path, mimetype="image/jpeg")

    @app.post("/api/goal")
    def api_goal() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        session.start_goal(str(payload.get("goal", "")))
        ensure_worker()
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
        session.request_stop()
        return jsonify({"ok": True, "state": session.status()})

    @app.post("/api/teleop")
    def api_teleop() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        result = session.teleop(str(payload.get("command", "stop")), payload.get("value"))
        return jsonify({"ok": True, "result": result, "state": session.status()})

    return app


def build_session(args: argparse.Namespace) -> HarnessSession:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / stamp
    runner_name = "codex" if args.live_codex else args.runner
    critic_mode = _resolve_critic_mode(runner_name, args.critic_mode)
    config = HarnessConfig(
        run_dir=run_dir,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        critic_mode=critic_mode,
        max_steps=args.max_steps,
        sleep_scale=1.0,
        rerun_enabled=args.rerun,
    )
    tools = AgentTools(
        run_dir=run_dir,
        namespace=args.namespace,
        connect=args.connect,
        reverse_yaw=not args.no_reverse_yaw,
        object_drive_detector=args.object_drive_detector,
        topomap_memory_map_dir=args.topomap_memory_map_dir,
        topomap_memory_use_clip=args.topomap_memory_use_clip,
        topomap_memory_allow_semantic_terms=args.topomap_memory_allow_semantic_terms,
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
    parser.add_argument("--no-reverse-yaw", action="store_true")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun-spawn", action="store_true")
    return parser.parse_args()


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
    button, input { font: inherit; }
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
      grid-template-columns: 1fr auto auto auto auto;
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
    button {
      border: 1px solid var(--line);
      background: #0b0810;
      color: var(--ink);
      padding: 10px 12px;
      min-width: 42px;
      min-height: 40px;
      cursor: pointer;
      clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
    }
    button:hover { border-color: var(--accent); color: #fff; }
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
      grid-template-columns: repeat(4, minmax(70px, 1fr));
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
      grid-template-rows: minmax(340px, 1fr) minmax(210px, 0.42fr);
      gap: 14px;
      min-height: 0;
    }
    .log, .memory {
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
    .event.critic { border-left-color: var(--accent-2); }
    .event.safety_gate { border-left-color: var(--red); }
    .event.observation { border-left-color: var(--blue); }
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
      .log, .memory { min-height: 280px; }
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
            <div class="metric"><b id="runDir">--</b><span>Run dir</span></div>
          </div>
        </section>
      </div>
      <div class="right">
        <section class="log">
          <div class="section-head"><span>Agent Chat</span><span id="eventCount">0 events</span></div>
          <div class="scroll" id="events"></div>
        </section>
        <section class="memory">
          <div class="section-head"><span>Memory</span><span id="memoryPath">memory.jsonl</span></div>
          <div class="scroll" id="memory"></div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const state = { lastFrame: "" };
    const q = (id) => document.getElementById(id);
    async function post(path, body = {}) {
      const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function compact(value) {
      if (value === undefined || value === null) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }
    function parseMaybeJson(value) {
      if (typeof value !== "string") return value;
      try { return JSON.parse(value); } catch { return value; }
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
      if (event.event === "actor") return `actor step ${event.step ?? "--"}`;
      if (event.event === "critic") return `critic step ${event.step ?? "--"}`;
      if (event.event === "safety_gate") return `safety gate step ${event.step ?? "--"}`;
      if (event.event === "observation") return `camera/imu frame ${event.frame_seq ?? "--"}`;
      if (event.event === "teleop") return `teleop ${event.command || ""}`;
      return event.event || "event";
    }
    function eventText(event) {
      if (event.event === "actor") {
        const output = parseMaybeJson(event.output);
        const action = event.action || output.action;
        const thought = output.thought || event.thought || "";
        return `${thought}\n${actionLine(action)}`.trim();
      }
      if (event.event === "critic") {
        const model = event.model_decision || {};
        const finalDecision = event.decision || {};
        return compact({
          model_verdict: model.verdict,
          final_verdict: finalDecision.verdict,
          reason: finalDecision.reason || model.reason,
          selected_action: event.selected_action,
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
    function renderEvents(events) {
      q("eventCount").textContent = `${events.length} events`;
      q("events").innerHTML = events.slice().reverse().map((event) => `
        <div class="event ${eventClass(event)}">
          <div class="who"><span>${escapeHtml(eventTitle(event))}</span><span>${new Date((event.t || 0) * 1000).toLocaleTimeString()}</span></div>
          <pre>${escapeHtml(eventText(event))}</pre>
        </div>`).join("");
    }
    function renderMemory(records) {
      q("memory").innerHTML = records.slice().reverse().map((record) => `
        <div class="event">
          <div class="who"><span>step ${record.step ?? "--"}</span><span>${record.mode || ""}</span></div>
          <pre>${escapeHtml(compact({ action: record.executed_action, critic: record.critic, result: record.tool_result }))}</pre>
        </div>`).join("");
    }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
    }
    async function refresh() {
      const res = await fetch("/api/state");
      const data = await res.json();
      const mode = data.mode || "idle";
      q("mode").className = `mode ${mode}`;
      q("mode").lastElementChild.textContent = mode;
      q("step").textContent = data.step ?? 0;
      q("currentMode").textContent = mode;
      q("worker").textContent = data.worker_active ? "worker active" : "worker idle";
      q("runDir").textContent = (data.run_dir || "").split("/").slice(-2).join("/");
      q("memoryPath").textContent = (data.memory_path || "").split("/").slice(-1)[0] || "memory.jsonl";
      const meta = data.metadata || {};
      q("runnerName").textContent = meta.live_codex ? "live codex" : "local";
      q("modelName").textContent = meta.model || "--";
      q("reasoning").textContent = meta.reasoning_effort || "--";
      q("rerunStatus").textContent = meta.rerun_path ? "rrd" : (meta.rerun_enabled ? "pending" : "off");
      q("schemaStatus").textContent = meta.codex_schema_dir ? "schemas" : (meta.live_codex ? "pending" : "n/a");
      q("boundary").textContent = Array.isArray(meta.policy_input_allowlist) ? meta.policy_input_allowlist.slice(0, 3).join(", ") : "--";
      const obs = data.last_observation || {};
      q("yaw").textContent = obs.yaw_deg === undefined ? "--" : Number(obs.yaw_deg).toFixed(1);
      q("frameSeq").textContent = obs.frame_seq === undefined ? "frame --" : `frame ${obs.frame_seq}`;
      if (data.latest_frame_path !== state.lastFrame) {
        state.lastFrame = data.latest_frame_path;
        q("camera").src = `/api/latest-frame?ts=${Date.now()}`;
      }
      renderEvents(data.recent_events || []);
      renderMemory(data.recent_memory || []);
    }
    q("go").onclick = () => post("/api/goal", { goal: q("goal").value }).then(refresh);
    q("pause").onclick = () => post("/api/pause").then(refresh);
    q("resume").onclick = () => post("/api/resume").then(refresh);
    q("stop").onclick = () => post("/api/stop").then(refresh);
    document.querySelectorAll("[data-teleop]").forEach((button) => {
      button.onclick = () => post("/api/teleop", { command: button.dataset.teleop }).then(refresh);
    });
    refresh();
    setInterval(refresh, 900);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
