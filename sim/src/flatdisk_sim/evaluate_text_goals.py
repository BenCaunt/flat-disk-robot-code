"""Evaluate text-goal navigation through the Zenoh AI2-THOR simulator.

The evaluator launches ``flatdisk-sim`` as a subprocess, drives it only through
``AgentTools`` (camera frame, IMU yaw, turn/drive/stop), and reads hidden THOR
pose/object metadata only for scoring and progress rendering.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .agent_tools import AgentTools, Observation, make_contact_sheet
from .paths import SCRATCH_ROOT
from .policy_registry import build_policy, policy_registry
from .text_goal_policy_core import PolicyAction, policy_history_record


@dataclass(frozen=True)
class ThorEpisodeSpec:
    name: str
    scene: str
    prompt: str
    target_types: tuple[str, ...]
    success_radius_m: float
    max_steps: int
    start_yaw_deg: float | None = None


def default_episodes() -> dict[str, ThorEpisodeSpec]:
    return {
        "living_room_sofa": ThorEpisodeSpec(
            name="living_room_sofa",
            scene="FloorPlan201",
            prompt="Drive to the sofa in the living room.",
            target_types=("Sofa",),
            success_radius_m=0.55,
            max_steps=18,
        ),
        "bedroom_bed": ThorEpisodeSpec(
            name="bedroom_bed",
            scene="FloorPlan301",
            prompt="Drive to the bed in the bedroom.",
            target_types=("Bed",),
            success_radius_m=0.55,
            max_steps=18,
        ),
        "bathroom_toilet": ThorEpisodeSpec(
            name="bathroom_toilet",
            scene="FloorPlan402",
            prompt="Drive to the toilet in the bathroom.",
            target_types=("Toilet",),
            success_radius_m=0.55,
            max_steps=8,
        ),
    }


def run_episode(
    spec: ThorEpisodeSpec,
    *,
    output_root: Path,
    port: int,
    policy_name: str,
    model: str,
    render_width: int,
    render_height: int,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    run_dir = output_root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    hidden_log_dir = run_dir / "hidden"
    hidden_log_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"tcp/127.0.0.1:{port}"
    namespace = f"flatdisk/eval/{int(time.time())}/{spec.name}"
    sim_proc = _start_bridge(
        scene=spec.scene,
        namespace=namespace,
        listen=endpoint,
        log_dir=hidden_log_dir,
        render_width=render_width,
        render_height=render_height,
        start_yaw_deg=spec.start_yaw_deg,
    )
    if args is None:
        args = argparse.Namespace(model=model)
    policy = build_policy(policy_name, args)
    policy.reset()
    tools: AgentTools | None = None
    steps: list[dict[str, Any]] = []
    hidden_snapshots: list[dict[str, Any]] = []
    progress_paths: list[Path] = []
    success = False
    reason = "max_steps_exhausted"
    try:
        _wait_for_hidden_log(hidden_log_dir, timeout_s=20.0)
        tools = AgentTools(run_dir=run_dir, namespace=namespace, connect=endpoint)
        _wait_for_first_observation(tools)
        history: list[dict[str, Any]] = []
        for step_index in range(spec.max_steps):
            obs = tools.observe(label=f"thor_{step_index:02d}", timeout_s=5.0)
            hidden = _latest_hidden_pose(hidden_log_dir)
            score = _score_hidden_distance(hidden, spec)
            hidden_snapshots.append(hidden)
            if score["success"]:
                success = True
                reason = "hidden_evaluator_goal_reached_before_action"
                break

            action = policy.choose_action(obs, prompt=spec.prompt, history=history)
            motion_summary = _execute_action(tools, action)
            post_hidden = _latest_hidden_pose(hidden_log_dir)
            post_score = _score_hidden_distance(post_hidden, spec)
            progress_path = _render_hidden_progress(
                hidden_snapshots + [post_hidden],
                spec,
                run_dir / "progress" / f"{step_index + 1:03d}.png",
                title=f"{spec.name}: step {step_index + 1} {action.action}",
            )
            progress_paths.append(progress_path)
            step_record = {
                "step": step_index,
                "observation": obs.summary(),
                "policy_action": action.__dict__,
                "motion_result": motion_summary,
                "hidden_score_for_evaluator_only": post_score,
            }
            steps.append(step_record)
            history.append(policy_history_record(action))
            if post_score["success"]:
                success = True
                reason = "hidden_evaluator_goal_reached"
                break
            if action.action == "stop":
                reason = "policy_stopped_without_hidden_success"
                break
    finally:
        if tools is not None:
            tools.close()
        _terminate_bridge(sim_proc)

    final_hidden = _latest_hidden_pose(hidden_log_dir)
    final_score = _score_hidden_distance(final_hidden, spec)
    contact_sheet = None
    frame_paths = [Path(step["observation"]["path"]) for step in steps[-8:]]
    if frame_paths:
        contact_sheet = make_contact_sheet(frame_paths, run_dir / "camera_contact_sheet.jpg")
    progress_sheet = None
    if progress_paths:
        progress_sheet = make_contact_sheet(progress_paths[-8:], run_dir / "progress_contact_sheet.jpg")
    summary = {
        "episode": spec.name,
        "policy": policy.name,
        "scene": spec.scene,
        "prompt": spec.prompt,
        "target_types": spec.target_types,
        "success": success,
        "reason": reason,
        "final_distance_m": final_score["distance_m"],
        "nearest_target": final_score["nearest_target"],
        "step_count": len(steps),
        "steps": steps,
        "camera_contact_sheet": str(contact_sheet) if contact_sheet else None,
        "progress_contact_sheet": str(progress_sheet) if progress_sheet else None,
        "last_progress_frame": str(progress_paths[-1]) if progress_paths else None,
        "hidden_log_dir": str(hidden_log_dir),
        "namespace": namespace,
        "run_dir": str(run_dir),
    }
    (run_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_report(summaries: list[dict[str, Any]], output_root: Path) -> Path:
    report = output_root / "report.md"
    lines = [
        "# THOR Text-Goal Navigation Evaluation",
        "",
        "Policy input allowlist: low RGB camera frame, IMU yaw, and its own action history.",
        "Evaluator-only data: hidden THOR pose and target object metadata from bridge logs.",
        "",
        "| Policy | Episode | Scene | Prompt | Success | Steps | Final distance (m) | Camera frames | Progress |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['policy']} | {summary['episode']} | {summary['scene']} | {summary['prompt']} | "
            f"{summary['success']} | {summary['step_count']} | {summary['final_distance_m']:.3f} | "
            f"{summary.get('camera_contact_sheet') or ''} | {summary.get('progress_contact_sheet') or ''} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    episodes = sorted(default_episodes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "thor_text_goal_eval")
    parser.add_argument("--episodes", nargs="+", choices=episodes, default=episodes)
    parser.add_argument("--policy", choices=sorted(policy_registry()), default="control_vlm")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-port", type=int, default=0, help="0 chooses a free local port per episode.")
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / stamp
    output_root.mkdir(parents=True, exist_ok=True)
    episode_map = default_episodes()
    summaries: list[dict[str, Any]] = []
    for index, name in enumerate(args.episodes):
        port = _find_free_port() if args.base_port == 0 else args.base_port + index
        summaries.append(
            run_episode(
                episode_map[name],
                output_root=output_root,
                port=port,
                policy_name=args.policy,
                model=args.model,
                render_width=args.render_width,
                render_height=args.render_height,
                args=args,
            )
        )
    aggregate = {
        "success_count": sum(1 for summary in summaries if summary["success"]),
        "episode_count": len(summaries),
        "summaries": summaries,
        "output_root": str(output_root),
    }
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = write_report(summaries, output_root)
    print(json.dumps({"report": str(report), **aggregate}, indent=2, sort_keys=True))
    return 0 if aggregate["success_count"] == aggregate["episode_count"] else 2


def _start_bridge(
    *,
    scene: str,
    namespace: str,
    listen: str,
    log_dir: Path,
    render_width: int,
    render_height: int,
    start_yaw_deg: float | None,
) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "flatdisk_sim.zenoh_bridge",
        "--backend",
        "ithor",
        "--namespace",
        namespace,
        "--scene",
        scene,
        "--listen",
        listen,
        "--mode",
        "peer",
        "--camera-hz",
        "5",
        "--imu-hz",
        "60",
        "--status-hz",
        "1",
        "--sim-hz",
        "20",
        "--render-width",
        str(render_width),
        "--render-height",
        str(render_height),
        "--log-dir",
        str(log_dir),
        "--quality",
        "Low",
    ]
    if start_yaw_deg is not None:
        cmd.extend(["--start-yaw-deg", str(start_yaw_deg)])
    stdout_path = log_dir / "bridge_stdout.txt"
    stderr_path = log_dir / "bridge_stderr.txt"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, start_new_session=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate_bridge(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(signal, "SIGTERM"):
            proc.send_signal(signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=8)


def _wait_for_hidden_log(log_dir: Path, *, timeout_s: float) -> Path:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() <= deadline:
        paths = sorted(log_dir.glob("bridge_*.jsonl"))
        if paths and paths[-1].stat().st_size > 0:
            return paths[-1]
        time.sleep(0.1)
    raise TimeoutError(f"no simulator hidden log appeared under {log_dir}")


def _wait_for_first_observation(tools: AgentTools) -> None:
    deadline = time.monotonic() + 25.0
    while time.monotonic() <= deadline:
        try:
            tools.observe(label="startup", timeout_s=2.0)
            return
        except TimeoutError:
            time.sleep(0.2)
    raise TimeoutError("simulator did not publish camera/IMU observations")


def _latest_hidden_pose(log_dir: Path) -> dict[str, Any]:
    path = _wait_for_hidden_log(log_dir, timeout_s=5.0)
    last: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    if last is None:
        raise RuntimeError(f"empty hidden log: {path}")
    return last["pose"]


def _score_hidden_distance(hidden_pose: dict[str, Any], spec: ThorEpisodeSpec) -> dict[str, Any]:
    px = float(hidden_pose["x"])
    pz = float(hidden_pose["z"])
    candidates: list[dict[str, Any]] = []
    for obj in hidden_pose.get("objects", []):
        if obj.get("objectType") not in spec.target_types:
            continue
        distance = _distance_to_hidden_object_xz(px, pz, obj)
        if distance is None:
            continue
        candidates.append({"distance_m": distance, "object": obj})
    if not candidates:
        return {"success": False, "distance_m": float("inf"), "nearest_target": None}
    nearest = min(candidates, key=lambda item: item["distance_m"])
    return {
        "success": nearest["distance_m"] <= spec.success_radius_m,
        "distance_m": round(float(nearest["distance_m"]), 4),
        "nearest_target": nearest["object"],
    }


def _execute_action(tools: AgentTools, action: PolicyAction) -> dict[str, Any] | None:
    if action.action == "turn_by_angle":
        return tools.turn_by_angle(action.degrees, power_percent=9.0).summary()
    if action.action == "drive_straight":
        return tools.drive_straight(action.power_percent, action.duration_s).summary()
    if action.action == "stop":
        tools.stop()
        return None
    raise ValueError(action.action)


def _render_hidden_progress(
    hidden_poses: list[dict[str, Any]],
    spec: ThorEpisodeSpec,
    output_path: Path,
    *,
    title: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 700, 460
    margin = 36
    pose_points = [(float(pose["x"]), float(pose["z"])) for pose in hidden_poses]
    targets: list[tuple[float, float, str]] = []
    for obj in hidden_poses[-1].get("objects", []):
        if obj.get("objectType") not in spec.target_types:
            continue
        pos = obj.get("position") or {}
        try:
            targets.append((float(pos["x"]), float(pos["z"]), str(obj.get("objectType"))))
        except (KeyError, TypeError, ValueError):
            pass
    xs = [p[0] for p in pose_points] + [t[0] for t in targets]
    zs = [p[1] for p in pose_points] + [t[1] for t in targets]
    if not xs:
        xs, zs = [0.0], [0.0]
    min_x, max_x = min(xs) - 1.0, max(xs) + 1.0
    min_z, max_z = min(zs) - 1.0, max(zs) + 1.0
    scale = min((width - margin * 2) / max(max_x - min_x, 1e-6), (height - margin * 2) / max(max_z - min_z, 1e-6))

    def project(x: float, z: float) -> tuple[int, int]:
        return margin + int((x - min_x) * scale), height - margin - int((z - min_z) * scale)

    image = Image.new("RGB", (width, height), (246, 246, 241))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((14, 12), title, fill=(26, 28, 30), font=font)
    if len(pose_points) >= 2:
        draw.line([project(x, z) for x, z in pose_points], fill=(218, 62, 52), width=4)
    for x, z, label in targets:
        tx, ty = project(x, z)
        radius = int(max(10, spec.success_radius_m * scale))
        draw.ellipse((tx - radius, ty - radius, tx + radius, ty + radius), outline=(20, 130, 95), width=3)
        draw.ellipse((tx - 7, ty - 7, tx + 7, ty + 7), fill=(20, 130, 95))
        draw.text((tx + 10, ty - 8), label, fill=(20, 80, 60), font=font)
    if hidden_poses:
        pose = hidden_poses[-1]
        rx, ry = project(float(pose["x"]), float(pose["z"]))
        yaw = math.radians(float(pose["yaw_deg"]))
        hx = rx + int(math.sin(yaw) * 26)
        hy = ry - int(math.cos(yaw) * 26)
        draw.ellipse((rx - 9, ry - 9, rx + 9, ry + 9), fill=(40, 90, 190), outline=(10, 35, 75), width=2)
        draw.line((rx, ry, hx, hy), fill=(10, 35, 75), width=4)
    image.save(output_path)
    return output_path


def _distance_to_hidden_object_xz(px: float, pz: float, obj: dict[str, Any]) -> float | None:
    bounds = obj.get("axisAlignedBoundingBox")
    if isinstance(bounds, dict) and isinstance(bounds.get("min"), dict) and isinstance(bounds.get("max"), dict):
        try:
            min_x = float(bounds["min"]["x"])
            max_x = float(bounds["max"]["x"])
            min_z = float(bounds["min"]["z"])
            max_z = float(bounds["max"]["z"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            dx = max(min_x - px, 0.0, px - max_x)
            dz = max(min_z - pz, 0.0, pz - max_z)
            return math.hypot(dx, dz)
    pos = obj.get("position") or {}
    try:
        return math.hypot(float(pos["x"]) - px, float(pos["z"]) - pz)
    except (KeyError, TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
