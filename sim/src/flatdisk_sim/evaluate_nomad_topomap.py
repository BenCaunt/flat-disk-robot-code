"""Evaluate semantic-topomap navigation over the Zenoh robot interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from PIL import Image
import zenoh

from .nomad_policy import NoMaDPolicy, NoMaDUnavailable
from .paths import SCRATCH_ROOT
from .protocol import DEFAULT_LISTEN, DEFAULT_NAMESPACE, VIDEO_STRUCT, build_config
from .semantic_topomap import ClipEmbedder, SemanticTopomap
from .topomap_navigator import NavigatorConfig, SemanticTopomapNavigator


DEFAULT_NOMAD_WEIGHTS = Path("/Users/bencaunt/Downloads/nomad.pth")


@dataclass(frozen=True)
class HiddenPoseSample:
    t: float
    pose: dict[str, Any]
    collided: bool = False


@dataclass
class EvaluationMetrics:
    frames: int = 0
    commands_published: int = 0
    route_reached_goal: bool = False
    best_goal_distance_m: float | None = None
    final_goal_distance_m: float | None = None
    hidden_pose_samples: int = 0
    collisions: int = 0
    errors: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "commands_published": self.commands_published,
            "route_reached_goal": self.route_reached_goal,
            "best_goal_distance_m": self.best_goal_distance_m,
            "final_goal_distance_m": self.final_goal_distance_m,
            "hidden_pose_samples": self.hidden_pose_samples,
            "collisions": self.collisions,
            "errors": self.errors,
        }


class HiddenPoseLogMonitor:
    def __init__(self, log_dir: Path | None) -> None:
        self.log_dir = log_dir
        self.path: Path | None = None
        self.offset = 0
        self.latest: HiddenPoseSample | None = None

    def poll(self) -> list[HiddenPoseSample]:
        if self.log_dir is None:
            return []
        if self.path is None or not self.path.exists():
            candidates = sorted(self.log_dir.glob("bridge_*.jsonl"), key=lambda path: path.stat().st_mtime)
            if not candidates:
                return []
            self.path = candidates[-1]
            self.offset = 0
        samples: list[HiddenPoseSample] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pose = payload.get("pose")
                if isinstance(pose, dict):
                    samples.append(
                        HiddenPoseSample(
                            t=float(payload.get("t", 0.0)),
                            pose=pose,
                            collided=bool(payload.get("collided", False)),
                        )
                    )
            self.offset = handle.tell()
        if samples:
            self.latest = samples[-1]
        return samples


def parse_video_sample(data: bytes) -> tuple[int, int, int, int, bytes] | None:
    if len(data) < VIDEO_STRUCT.size:
        return None
    magic, version, _fmt, width, height, header_len, seq, esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(data)
    if magic != b"FDV1" or version != 1 or header_len > len(data):
        return None
    jpeg = data[header_len : header_len + jpeg_len]
    if len(jpeg) != jpeg_len:
        return None
    return seq, esp_us, width, height, jpeg


def image_from_jpeg(jpeg: bytes, *, rotate_180: bool) -> Image.Image:
    image = Image.open(BytesIO(jpeg)).convert("RGB")
    return image.rotate(180) if rotate_180 else image


def publish_motor_percent(session: zenoh.Session, namespace: str, command: tuple[int, int]) -> None:
    payload = json.dumps({"m1": int(command[0]), "m2": int(command[1])}, sort_keys=True).encode("utf-8")
    session.put(f"{namespace}/cmd/motors/percent", payload)


def publish_stop(session: zenoh.Session, namespace: str) -> None:
    session.put(f"{namespace}/cmd/motors/stop", b"stop")


def node_pose_xz(topomap: SemanticTopomap, node_id: str) -> tuple[float, float]:
    pose = topomap.node(node_id)["pose"]
    return float(pose["x"]), float(pose["z"])


def pose_distance_to_node(topomap: SemanticTopomap, node_id: str, hidden_pose: dict[str, Any]) -> float:
    gx, gz = node_pose_xz(topomap, node_id)
    return ((float(hidden_pose["x"]) - gx) ** 2 + (float(hidden_pose["z"]) - gz) ** 2) ** 0.5


def update_hidden_metrics(
    metrics: EvaluationMetrics,
    topomap: SemanticTopomap,
    goal_node_id: str | None,
    samples: list[HiddenPoseSample],
) -> None:
    if not samples:
        return
    metrics.hidden_pose_samples += len(samples)
    metrics.collisions += sum(1 for sample in samples if sample.collided)
    if goal_node_id is None:
        return
    update_hidden_goal_distance(metrics, topomap, goal_node_id, samples)


def update_hidden_goal_distance(
    metrics: EvaluationMetrics,
    topomap: SemanticTopomap,
    goal_node_id: str,
    samples: list[HiddenPoseSample],
) -> None:
    for sample in samples:
        try:
            distance = pose_distance_to_node(topomap, goal_node_id, sample.pose)
        except (KeyError, TypeError, ValueError):
            continue
        metrics.final_goal_distance_m = round(float(distance), 4)
        if metrics.best_goal_distance_m is None or distance < metrics.best_goal_distance_m:
            metrics.best_goal_distance_m = round(float(distance), 4)


def launch_simulator(args: argparse.Namespace, log_dir: Path) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "flatdisk_sim.zenoh_bridge",
        "--namespace",
        args.namespace,
        "--mode",
        "peer",
        "--listen",
        args.sim_listen,
        "--camera-hz",
        str(args.sim_camera_hz),
        "--imu-hz",
        str(args.sim_imu_hz),
        "--status-hz",
        "2",
        "--duration",
        str(max(1.0, args.duration + 2.0)),
        "--log-dir",
        str(log_dir),
        "--backend",
        args.backend,
        "--scene",
        args.scene,
        "--procthor-seed",
        str(args.procthor_seed),
        "--procthor-split",
        args.procthor_split,
    ]
    if args.house_json is not None:
        cmd.extend(["--house-json", str(args.house_json)])
    if args.start_x is not None:
        cmd.extend(["--start-x", str(args.start_x)])
    if args.start_y is not None:
        cmd.extend(["--start-y", str(args.start_y)])
    if args.start_z is not None:
        cmd.extend(["--start-z", str(args.start_z)])
    if args.start_yaw_deg is not None:
        cmd.extend(["--start-yaw-deg", str(args.start_yaw_deg)])
    if args.random_start:
        cmd.append("--random-start")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    topomap = SemanticTopomap.load(args.map_dir)
    clip_embedder = ClipEmbedder(args.clip_model) if args.clip else None
    nomad = None
    if not args.no_nomad:
        try:
            nomad = NoMaDPolicy(
                checkpoint=args.checkpoint,
                visualnav_repo=args.visualnav_repo,
                diffusion_policy_repo=args.diffusion_policy_repo,
                device=args.device,
                num_samples=args.num_samples,
            )
        except NoMaDUnavailable as exc:
            raise RuntimeError(f"NoMaD unavailable: {exc}") from exc

    navigator = SemanticTopomapNavigator(
        topomap,
        NavigatorConfig(
            goal=args.goal,
            reached_threshold=args.reached_threshold,
            waypoint_index=args.waypoint_index,
            waypoint_dt_s=args.waypoint_dt_s,
            max_v_mps=args.max_v_mps,
            max_w_rad_s=args.max_w_rad_s,
            wheel_base_m=args.wheel_base_m,
            max_wheel_speed_mps=args.max_wheel_speed_mps,
            max_abs_output=args.max_abs_output,
            nomad_sample_aggregation=args.nomad_sample_aggregation,
            invert_angular=args.invert_angular,
            use_nomad_distance_for_progress=args.nomad_distance_progress,
            use_visual_match_for_progress=args.visual_progress,
            use_route_window_progress=args.route_window_progress,
            route_window_lookahead=args.route_window_lookahead,
            route_window_advance_threshold=args.route_window_advance_threshold,
            route_window_advance_margin=args.route_window_advance_margin,
            route_window_stable_frames=args.route_window_stable_frames,
            route_window_max_advance=args.route_window_max_advance,
            nomad_close_threshold=args.nomad_close_threshold,
            nomad_advance_margin=args.nomad_advance_margin,
        ),
        command_policy=nomad,
        clip_embedder=clip_embedder,
    )

    log_dir = args.sim_log_dir
    sim_process: subprocess.Popen[bytes] | None = None
    if args.launch_sim:
        log_dir.mkdir(parents=True, exist_ok=True)
        sim_process = launch_simulator(args, log_dir)
        time.sleep(args.sim_startup_s)

    metrics = EvaluationMetrics()
    pose_monitor = HiddenPoseLogMonitor(log_dir)
    session = zenoh.open(build_config(args.mode, args.listen, args.connect))
    video_sub = session.declare_subscriber(f"{args.namespace}/camera/jpeg")
    status_key = args.status_key or f"{args.namespace}/nomad_topomap/eval_status"
    events: list[dict[str, Any]] = []
    stop_requested = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    try:
        started = time.monotonic()
        last_tick = 0.0
        latest_sample: bytes | None = None
        route_initialized = False
        goal_node_id: str | None = None
        while not stop_requested and time.monotonic() - started < args.duration:
            while True:
                sample = video_sub.try_recv()
                if sample is None:
                    break
                latest_sample = sample.payload.to_bytes()

            now = time.monotonic()
            if latest_sample is None or now - last_tick < 1.0 / max(args.control_hz, 0.1):
                update_hidden_metrics(metrics, topomap, goal_node_id, pose_monitor.poll())
                time.sleep(0.002)
                continue
            last_tick = now

            parsed = parse_video_sample(latest_sample)
            latest_sample = None
            if parsed is None:
                metrics.errors.append("invalid video sample")
                continue
            seq, esp_us, _width, _height, jpeg = parsed
            image = image_from_jpeg(jpeg, rotate_180=not args.no_rotate_180)
            if not route_initialized:
                route = navigator.get_sequence(image)
                goal_node_id = route.goal.node_id
                route_initialized = True
                if pose_monitor.latest is not None:
                    update_hidden_goal_distance(metrics, topomap, goal_node_id, [pose_monitor.latest])

            step = navigator.drive_to_goal(image, armed=args.arm)
            metrics.frames += 1
            if args.arm and step.command_ready:
                publish_motor_percent(session, args.namespace, step.command)
                metrics.commands_published += 1
            if step.reached_goal:
                metrics.route_reached_goal = True
                publish_stop(session, args.namespace)
                if args.stop_on_route_goal:
                    break
            if step.error:
                metrics.errors.append(step.error)
                publish_stop(session, args.namespace)
            update_hidden_metrics(metrics, topomap, goal_node_id, pose_monitor.poll())
            event = {
                "seq": seq,
                "esp_us": esp_us,
                "elapsed_s": round(now - started, 3),
                "step": step.as_json(),
                "metrics": metrics.as_json(),
            }
            events.append(event)
            session.put(status_key, json.dumps(event, sort_keys=True).encode("utf-8"))
    finally:
        publish_stop(session, args.namespace)
        try:
            video_sub.undeclare()
        except Exception:
            pass
        session.close()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if sim_process is not None:
            sim_process.terminate()
            try:
                sim_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                sim_process.kill()

    summary = {
        "schema": "flatdisk.nomad_topomap.eval.v1",
        "map_dir": str(topomap.map_dir),
        "goal": args.goal,
        "armed": args.arm,
        "status_key": status_key,
        "metrics": metrics.as_json(),
        "events": events if args.include_events else [],
        "sim_log_dir": str(log_dir) if log_dir is not None else None,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_NOMAD_WEIGHTS)
    parser.add_argument("--visualnav-repo", type=Path, default=Path("/tmp/visualnav-transformer"))
    parser.add_argument("--diffusion-policy-repo", type=Path, default=None)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--connect", default=DEFAULT_LISTEN)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=4.0)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument("--status-key", default="")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--waypoint-index", type=int, default=2)
    parser.add_argument("--reached-threshold", type=float, default=0.88)
    parser.add_argument("--max-abs-output", type=float, default=35.0)
    parser.add_argument("--max-v-mps", type=float, default=0.2)
    parser.add_argument("--max-w-rad-s", type=float, default=0.4)
    parser.add_argument("--wheel-base-m", type=float, default=0.215)
    parser.add_argument("--max-wheel-speed-mps", type=float, default=0.78)
    parser.add_argument("--waypoint-dt-s", type=float, default=1.0 / 4.0)
    parser.add_argument(
        "--nomad-sample-aggregation",
        default="medoid",
        choices=("first", "mean", "median", "medoid"),
        help="How to choose a waypoint from NoMaD's diffusion samples.",
    )
    parser.add_argument("--invert-angular", action="store_true")
    parser.add_argument(
        "--nomad-distance-progress",
        action="store_true",
        help="Use NoMaD's distance head to advance image goals. Off by default; route-window heuristic is the default.",
    )
    parser.add_argument(
        "--no-nomad-distance-progress",
        action="store_false",
        dest="nomad_distance_progress",
        help="Disable NoMaD distance-head route progress.",
    )
    parser.add_argument("--visual-progress", action="store_true")
    parser.add_argument("--route-window-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-window-lookahead", type=int, default=4)
    parser.add_argument("--route-window-advance-threshold", type=float, default=0.55)
    parser.add_argument("--route-window-advance-margin", type=float, default=0.015)
    parser.add_argument("--route-window-stable-frames", type=int, default=3)
    parser.add_argument("--route-window-max-advance", type=int, default=1)
    parser.add_argument("--nomad-close-threshold", type=float, default=3.0)
    parser.add_argument("--nomad-advance-margin", type=float, default=0.5)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--no-nomad", action="store_true")
    parser.add_argument("--no-rotate-180", action="store_true")
    parser.add_argument("--stop-on-route-goal", action="store_true")
    parser.add_argument("--sim-log-dir", type=Path, default=None)
    parser.add_argument("--launch-sim", action="store_true")
    parser.add_argument("--sim-listen", default=DEFAULT_LISTEN)
    parser.add_argument("--sim-startup-s", type=float, default=4.0)
    parser.add_argument("--sim-camera-hz", type=float, default=10.0)
    parser.add_argument("--sim-imu-hz", type=float, default=60.0)
    parser.add_argument("--backend", default="ithor", choices=("ithor", "procthor", "house-json"))
    parser.add_argument("--scene", default="FloorPlan401")
    parser.add_argument("--house-json", type=Path)
    parser.add_argument("--procthor-seed", type=int, default=42)
    parser.add_argument("--procthor-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-z", type=float, default=None)
    parser.add_argument("--start-yaw-deg", type=float, default=None)
    args = parser.parse_args()
    if args.launch_sim:
        args.mode = "client"
        args.connect = args.connect or args.sim_listen
        if args.sim_log_dir is None:
            args.sim_log_dir = SCRATCH_ROOT / "nomad_topomap_eval"
    return args


def main() -> int:
    args = parse_args()
    try:
        summary = run_evaluation(args)
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    if args.output_json is not None:
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
