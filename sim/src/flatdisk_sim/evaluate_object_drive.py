"""Evaluate continuous object-drive control through the Zenoh THOR simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .agent_tools import make_contact_sheet
from .evaluate_text_goals import (
    ThorEpisodeSpec,
    _distance_to_hidden_object_xz,
    _find_free_port,
    _latest_hidden_pose,
    _render_hidden_progress,
    _score_hidden_distance,
    _terminate_bridge,
    _wait_for_hidden_log,
)
from .paths import REPO_ROOT, SCRATCH_ROOT
from .protocol import wrap_pi
from .thor_backend import FlatDiskThorSim, ThorSimConfig


@dataclass(frozen=True)
class ObjectDriveEpisodeSpec:
    name: str
    scene: str
    prompt: str
    target_types: tuple[str, ...]
    success_radius_m: float
    duration_s: float
    forward_power: float
    yaw_offset_deg: float
    min_start_distance_m: float = 0.8
    max_start_distance_m: float = 2.2

    def scoring_spec(self) -> ThorEpisodeSpec:
        return ThorEpisodeSpec(
            name=self.name,
            scene=self.scene,
            prompt=f"Drive toward the visible {self.prompt}.",
            target_types=self.target_types,
            success_radius_m=self.success_radius_m,
            max_steps=1,
        )


@dataclass(frozen=True)
class StartPose:
    x: float
    y: float
    z: float
    yaw_deg: float
    target_object_id: str | None
    target_type: str
    target_position: dict[str, float]
    initial_distance_m: float
    initial_bearing_deg: float

    def bridge_args(self) -> list[str]:
        return [
            "--start-x",
            str(self.x),
            "--start-y",
            str(self.y),
            "--start-z",
            str(self.z),
            "--start-yaw-deg",
            str(self.yaw_deg),
        ]


def default_episodes() -> dict[str, ObjectDriveEpisodeSpec]:
    return {
        "living_room_chair": ObjectDriveEpisodeSpec(
            name="living_room_chair",
            scene="FloorPlan201",
            prompt="chair",
            target_types=("ArmChair", "Chair"),
            success_radius_m=0.65,
            duration_s=6.0,
            forward_power=22.0,
            yaw_offset_deg=22.0,
        ),
        "living_room_closest_chair": ObjectDriveEpisodeSpec(
            name="living_room_closest_chair",
            scene="FloorPlan201",
            prompt="closest chair",
            target_types=("ArmChair", "Chair"),
            success_radius_m=0.65,
            duration_s=6.0,
            forward_power=22.0,
            yaw_offset_deg=22.0,
        ),
        "living_room_individual_chair": ObjectDriveEpisodeSpec(
            name="living_room_individual_chair",
            scene="FloorPlan201",
            prompt="closest individual chair, not the table",
            target_types=("ArmChair", "Chair"),
            success_radius_m=0.65,
            duration_s=6.0,
            forward_power=22.0,
            yaw_offset_deg=22.0,
        ),
        "bathroom_toilet": ObjectDriveEpisodeSpec(
            name="bathroom_toilet",
            scene="FloorPlan402",
            prompt="toilet",
            target_types=("Toilet",),
            success_radius_m=0.6,
            duration_s=5.0,
            forward_power=20.0,
            yaw_offset_deg=-18.0,
        ),
    }


def run_episode(
    spec: ObjectDriveEpisodeSpec,
    *,
    output_root: Path,
    port: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    hidden_log_dir = run_dir / "hidden"
    hidden_log_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"tcp/127.0.0.1:{port}"
    namespace = f"flatdisk/object-drive-eval/{int(time.time())}/{spec.name}"
    start_pose = choose_start_pose(
        spec,
        render_width=args.render_width,
        render_height=args.render_height,
        quality=args.quality,
        candidate_limit=args.start_candidate_limit,
    )
    sim_proc = _start_bridge(
        spec=spec,
        namespace=namespace,
        listen=endpoint,
        log_dir=hidden_log_dir,
        render_width=args.render_width,
        render_height=args.render_height,
        quality=args.quality,
        start_pose=start_pose,
    )
    object_proc: subprocess.Popen[str] | None = None
    progress_path: Path | None = None
    try:
        _wait_for_hidden_log(hidden_log_dir, timeout_s=25.0)
        initial_hidden = _latest_hidden_pose(hidden_log_dir)
        initial_score = _score_hidden_distance(initial_hidden, spec.scoring_spec())
        object_proc = _start_object_drive(
            spec=spec,
            namespace=namespace,
            connect=endpoint,
            run_dir=run_dir,
            args=args,
        )
        timeout_s = spec.duration_s + args.object_drive_timeout_margin_s
        try:
            object_proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _terminate_bridge(object_proc)
        time.sleep(0.5)
    finally:
        if object_proc is not None:
            _terminate_bridge(object_proc)
        _terminate_bridge(sim_proc)

    hidden_poses = _read_hidden_poses(hidden_log_dir)
    if hidden_poses:
        progress_path = _render_hidden_progress(
            hidden_poses,
            spec.scoring_spec(),
            run_dir / "progress.png",
            title=f"{spec.name}: object drive",
        )
    final_hidden = _latest_hidden_pose(hidden_log_dir)
    final_score = _score_hidden_distance(final_hidden, spec.scoring_spec())
    initial_distance = float(initial_score["distance_m"])
    final_distance = float(final_score["distance_m"])
    improvement = initial_distance - final_distance
    distance_trace = [
        _score_hidden_distance(pose, spec.scoring_spec())["distance_m"]
        for pose in hidden_poses
    ]
    success = (
        math.isfinite(initial_distance)
        and math.isfinite(final_distance)
        and improvement >= args.min_distance_improvement_m
    )
    contact_sheet = None
    if progress_path is not None:
        contact_sheet = make_contact_sheet([progress_path], run_dir / "progress_contact_sheet.jpg")
    overlay_paths = sorted((run_dir / "overlays").glob("*.jpg"))
    overlay_contact_sheet = None
    if overlay_paths:
        overlay_contact_sheet = make_contact_sheet(overlay_paths[-12:], run_dir / "robot_pov_bbox_contact_sheet.jpg")
    summary = {
        "episode": spec.name,
        "scene": spec.scene,
        "prompt": spec.prompt,
        "target_types": spec.target_types,
        "namespace": namespace,
        "start_pose": start_pose.__dict__,
        "duration_s": spec.duration_s,
        "forward_power": spec.forward_power,
        "detector": args.detector,
        "success": success,
        "initial_distance_m": initial_distance,
        "final_distance_m": final_distance,
        "distance_improvement_m": round(improvement, 4),
        "min_distance_improvement_m": args.min_distance_improvement_m,
        "nearest_target_initial": initial_score["nearest_target"],
        "nearest_target_final": final_score["nearest_target"],
        "distance_trace_m": distance_trace,
        "hidden_log_dir": str(hidden_log_dir),
        "progress": str(progress_path) if progress_path is not None else None,
        "progress_contact_sheet": str(contact_sheet) if contact_sheet is not None else None,
        "robot_pov_overlay_dir": str(run_dir / "overlays"),
        "robot_pov_overlay_count": len(overlay_paths),
        "robot_pov_bbox_contact_sheet": str(overlay_contact_sheet) if overlay_contact_sheet is not None else None,
        "object_drive_stdout": str(run_dir / "object_drive_stdout.txt"),
        "object_drive_stderr": str(run_dir / "object_drive_stderr.txt"),
        "bridge_stdout": str(hidden_log_dir / "bridge_stdout.txt"),
        "bridge_stderr": str(hidden_log_dir / "bridge_stderr.txt"),
        "rerun_path": str(run_dir / "object_drive.rrd") if (run_dir / "object_drive.rrd").exists() else None,
        "run_dir": str(run_dir),
    }
    (run_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def choose_start_pose(
    spec: ObjectDriveEpisodeSpec,
    *,
    render_width: int,
    render_height: int,
    quality: str,
    candidate_limit: int,
) -> StartPose:
    sim = FlatDiskThorSim(
        ThorSimConfig(
            backend="ithor",
            scene=spec.scene,
            width=render_width,
            height=render_height,
            quality=quality,
            random_start=False,
        )
    )
    try:
        reachable = sim.privileged_reachable_positions()
        targets = [obj for obj in sim.hidden_objects() if obj.get("objectType") in spec.target_types]
        if not targets:
            raise RuntimeError(f"no target objects {spec.target_types} in {spec.scene}")
        best: StartPose | None = None
        for obj in targets:
            center = _object_center_xz(obj)
            if center is None:
                continue
            target_x, target_z = center
            candidate_positions: list[tuple[float, dict[str, Any]]] = []
            for position in reachable:
                try:
                    x = float(position["x"])
                    z = float(position["z"])
                except (KeyError, TypeError, ValueError):
                    continue
                distance = _distance_to_hidden_object_xz(x, z, obj)
                if distance is None or distance < spec.min_start_distance_m or distance > spec.max_start_distance_m:
                    continue
                candidate_positions.append((float(distance), position))
            candidate_positions.sort(key=lambda item: item[0], reverse=True)
            if candidate_limit > 0:
                candidate_positions = candidate_positions[:candidate_limit]
            for distance, position in candidate_positions:
                try:
                    x = float(position["x"])
                    y = float(position.get("y", sim.state.y))
                    z = float(position["z"])
                except (KeyError, TypeError, ValueError):
                    continue
                yaw_to_target = math.atan2(target_x - x, target_z - z)
                for yaw_offset_deg in (spec.yaw_offset_deg, -spec.yaw_offset_deg, spec.yaw_offset_deg * 0.5, 0.0):
                    yaw_rad = wrap_pi(yaw_to_target + math.radians(yaw_offset_deg))
                    if not sim.privileged_teleport(x=x, y=y, z=z, yaw_rad=yaw_rad):
                        continue
                    visible_obj = _matching_hidden_object(sim.hidden_objects(), obj)
                    if visible_obj is None:
                        continue
                    bearing_deg = math.degrees(wrap_pi(yaw_to_target - yaw_rad))
                    if abs(bearing_deg) < 4.0 and yaw_offset_deg != 0.0:
                        continue
                    target_position = visible_obj.get("position") or {}
                    candidate = StartPose(
                        x=x,
                        y=y,
                        z=z,
                        yaw_deg=math.degrees(yaw_rad),
                        target_object_id=visible_obj.get("objectId"),
                        target_type=str(visible_obj.get("objectType")),
                        target_position={
                            "x": float(target_position.get("x", target_x)),
                            "y": float(target_position.get("y", 0.0)),
                            "z": float(target_position.get("z", target_z)),
                        },
                        initial_distance_m=round(float(distance), 4),
                        initial_bearing_deg=round(float(bearing_deg), 4),
                    )
                    if best is None or candidate.initial_distance_m > best.initial_distance_m:
                        best = candidate
        if best is None:
            raise RuntimeError(
                f"could not find a reachable off-center start pose for {spec.name} in {spec.scene}"
            )
        return best
    finally:
        sim.stop()
        sim.close()


def _matching_hidden_object(objects: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any] | None:
    reference_id = reference.get("objectId")
    for obj in objects:
        if reference_id is not None and obj.get("objectId") == reference_id:
            return obj
    reference_name = reference.get("name")
    for obj in objects:
        if reference_name is not None and obj.get("name") == reference_name:
            return obj
    return None


def _object_center_xz(obj: dict[str, Any]) -> tuple[float, float] | None:
    bounds = obj.get("axisAlignedBoundingBox")
    if isinstance(bounds, dict) and isinstance(bounds.get("center"), dict):
        try:
            return float(bounds["center"]["x"]), float(bounds["center"]["z"])
        except (KeyError, TypeError, ValueError):
            pass
    pos = obj.get("position") or {}
    try:
        return float(pos["x"]), float(pos["z"])
    except (KeyError, TypeError, ValueError):
        return None


def _start_bridge(
    *,
    spec: ObjectDriveEpisodeSpec,
    namespace: str,
    listen: str,
    log_dir: Path,
    render_width: int,
    render_height: int,
    quality: str,
    start_pose: StartPose,
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
        spec.scene,
        "--listen",
        listen,
        "--mode",
        "peer",
        "--camera-hz",
        "8",
        "--imu-hz",
        "60",
        "--status-hz",
        "5",
        "--sim-hz",
        "30",
        "--render-width",
        str(render_width),
        "--render-height",
        str(render_height),
        "--log-dir",
        str(log_dir),
        "--quality",
        quality,
        *start_pose.bridge_args(),
    ]
    stdout = (log_dir / "bridge_stdout.txt").open("w", encoding="utf-8")
    stderr = (log_dir / "bridge_stderr.txt").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, start_new_session=True)


def _start_object_drive(
    *,
    spec: ObjectDriveEpisodeSpec,
    namespace: str,
    connect: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> subprocess.Popen[str]:
    script = REPO_ROOT / "scripts" / "object_drive_zenoh.py"
    cmd = [
        sys.executable,
        str(script),
        "--prompt",
        spec.prompt,
        "--duration",
        str(spec.duration_s),
        "--forward-power",
        str(spec.forward_power),
        "--namespace",
        namespace,
        "--mode",
        "client",
        "--connect",
        connect,
        "--arm",
        "--detector",
        args.detector,
        "--control-hz",
        str(args.control_hz),
        "--detect-interval",
        str(args.detect_interval),
        "--max-track-age",
        str(args.max_track_age),
        "--max-abs-output",
        str(args.max_abs_output),
        "--heading-kp",
        str(args.heading_kp),
        "--max-turn-percent",
        str(args.max_turn_percent),
        "--max-bbox-area-fraction",
        str(args.max_bbox_area_fraction),
        "--stop-when-lost",
        "--overlay-dir",
        str(run_dir / "overlays"),
        "--overlay-every",
        str(args.overlay_every),
    ]
    if args.detector == "florence-mlx":
        cmd.extend(["--model", args.model])
    if args.detector == "florence-transformers":
        cmd.extend(["--transformers-model", args.transformers_model, "--device", args.device])
    if args.detector == "grounding-dino":
        cmd.extend(
            [
                "--grounding-dino-model",
                args.grounding_dino_model,
                "--grounding-dino-box-threshold",
                str(args.grounding_dino_box_threshold),
                "--grounding-dino-text-threshold",
                str(args.grounding_dino_text_threshold),
                "--device",
                args.device,
            ]
        )
    if args.rerun:
        cmd.extend(["--rerun-save", str(run_dir / "object_drive.rrd")])
    stdout = (run_dir / "object_drive_stdout.txt").open("w", encoding="utf-8")
    stderr = (run_dir / "object_drive_stderr.txt").open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True, start_new_session=True)


def _read_hidden_poses(log_dir: Path) -> list[dict[str, Any]]:
    path = _wait_for_hidden_log(log_dir, timeout_s=5.0)
    poses: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        pose = payload.get("pose")
        if isinstance(pose, dict):
            poses.append(pose)
    return poses


def write_report(summaries: list[dict[str, Any]], output_root: Path) -> Path:
    report = output_root / "object_drive_report.md"
    lines = [
        "# Object-Drive THOR Evaluation",
        "",
        "Policy input allowlist: Zenoh camera JPEG, IMU yaw, prompt, duration, and forward power.",
        "Evaluator-only data: hidden THOR pose/object metadata used for start-pose selection and scoring.",
        "",
        "| Episode | Scene | Prompt | Detector | Success | Initial m | Final m | Improvement m | Start Bearing | Robot POV | Progress | Rerun |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for summary in summaries:
        start_pose = summary["start_pose"]
        lines.append(
            f"| {summary['episode']} | {summary['scene']} | {summary['prompt']} | {summary['detector']} | "
            f"{summary['success']} | {summary['initial_distance_m']:.3f} | {summary['final_distance_m']:.3f} | "
            f"{summary['distance_improvement_m']:.3f} | {start_pose['initial_bearing_deg']:.1f} | "
            f"{summary.get('robot_pov_bbox_contact_sheet') or ''} | "
            f"{summary.get('progress_contact_sheet') or summary.get('progress') or ''} | {summary.get('rerun_path') or ''} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    episode_names = sorted(default_episodes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "object_drive_eval")
    parser.add_argument("--episodes", nargs="+", choices=episode_names, default=episode_names)
    parser.add_argument("--base-port", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--quality", default="Low")
    parser.add_argument(
        "--detector",
        choices=("florence-mlx", "florence-transformers", "grounding-dino"),
        default="florence-mlx",
    )
    parser.add_argument("--model", default="mlx-community/Florence-2-base-ft-4bit")
    parser.add_argument("--transformers-model", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--detect-interval", type=float, default=0.6)
    parser.add_argument("--max-track-age", type=float, default=2.5)
    parser.add_argument("--heading-kp", type=float, default=18.0)
    parser.add_argument("--max-turn-percent", type=float, default=16.0)
    parser.add_argument("--max-abs-output", type=float, default=50.0)
    parser.add_argument("--max-bbox-area-fraction", type=float, default=0.60)
    parser.add_argument("--min-distance-improvement-m", type=float, default=0.12)
    parser.add_argument(
        "--start-candidate-limit",
        type=int,
        default=80,
        help="Max farthest reachable candidate poses to teleport-test per target. 0 evaluates all.",
    )
    parser.add_argument("--object-drive-timeout-margin-s", type=float, default=20.0)
    parser.add_argument("--overlay-every", type=int, default=3)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / stamp
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = default_episodes()
    summaries: list[dict[str, Any]] = []
    for index, name in enumerate(args.episodes):
        port = _find_free_port() if args.base_port == 0 else args.base_port + index
        summaries.append(run_episode(episodes[name], output_root=output_root, port=port, args=args))
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


if __name__ == "__main__":
    raise SystemExit(main())
