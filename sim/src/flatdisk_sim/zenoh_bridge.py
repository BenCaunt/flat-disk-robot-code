"""Zenoh bridge that makes the simulator look like the real flat disk robot."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import zenoh

from .paths import SCRATCH_ROOT
from .protocol import (
    DEFAULT_LISTEN,
    DEFAULT_NAMESPACE,
    build_config,
    image_to_jpeg,
    pack_imu,
    pack_time_sync_reply,
    pack_video_jpeg,
    parse_pair_payload,
)
from .thor_backend import (
    DEFAULT_CAMERA_FAR_PLANE_M,
    DEFAULT_CAMERA_FORWARD_OFFSET_M,
    DEFAULT_CAMERA_HEIGHT_M,
    DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    DEFAULT_CAMERA_NEAR_PLANE_M,
    DEFAULT_ITHOR_SCENE,
    FlatDiskThorSim,
    ThorSimConfig,
)


class ZenohFlatDiskSimulator:
    def __init__(
        self,
        *,
        namespace: str,
        mode: str,
        listen: str,
        connect: str,
        camera_hz: float,
        imu_hz: float,
        status_hz: float,
        sim_hz: float,
        render_width: int,
        render_height: int,
        log_dir: Path,
        backend: str,
        scene: str,
        house_json: Path | None,
        procthor_seed: int,
        procthor_split: str,
        random_start: bool,
        field_of_view: float,
        field_of_view_axis: str,
        camera_height_m: float,
        camera_forward_offset_m: float,
        camera_near_plane_m: float,
        camera_far_plane_m: float,
        camera_calibration: Path | None,
        use_third_party_camera: bool,
        grid_size: float,
        rotate_step_degrees: float,
        quality: str,
        start_x: float | None,
        start_y: float | None,
        start_z: float | None,
        start_yaw_deg: float | None,
    ) -> None:
        self.namespace = namespace.strip("/")
        self.mode = mode
        self.listen = listen
        self.connect = connect
        self.camera_period = 1.0 / max(camera_hz, 0.1)
        self.imu_period = 1.0 / max(imu_hz, 0.1)
        self.status_period = 1.0 / max(status_hz, 0.1)
        self.sim_period = 1.0 / max(sim_hz, 0.1)
        self.sim = FlatDiskThorSim(
            ThorSimConfig(
                backend=backend,
                scene=scene,
                house_json=house_json,
                procthor_seed=procthor_seed,
                procthor_split=procthor_split,
                random_start=random_start,
                width=render_width,
                height=render_height,
                field_of_view=field_of_view,
                field_of_view_axis=field_of_view_axis,
                camera_height_m=camera_height_m,
                camera_forward_offset_m=camera_forward_offset_m,
                camera_near_plane_m=camera_near_plane_m,
                camera_far_plane_m=camera_far_plane_m,
                camera_calibration=camera_calibration,
                use_third_party_camera=use_third_party_camera,
                grid_size=grid_size,
                rotate_step_degrees=rotate_step_degrees,
                quality=quality,
                start_x=start_x,
                start_y=start_y,
                start_z=start_z,
                start_yaw_deg=start_yaw_deg,
            )
        )
        self.log_dir = log_dir
        self.log_path = self._make_log_path()
        self.video_seq = 0
        self.imu_seq = 0
        self.video_published = 0
        self.imu_published = 0
        self.status_published = 0
        self.motor_commands = 0
        self.motor_command_errors = 0
        self.video_errors = 0
        self.imu_errors = 0
        self._stop = False
        self._start_monotonic = time.monotonic()

    def _make_log_path(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"bridge_{stamp}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, *, duration_s: float | None = None) -> None:
        config = build_config(self.mode, self.listen, self.connect)
        session = zenoh.open(config)
        subs: list[Any] = []
        try:
            subs.append(session.declare_subscriber(f"{self.namespace}/cmd/motors/percent"))
            subs.append(session.declare_subscriber(f"{self.namespace}/cmd/motors/us"))
            subs.append(session.declare_subscriber(f"{self.namespace}/cmd/motors/stop"))
            subs.append(session.declare_subscriber(f"{self.namespace}/cmd/time_sync"))
            self._run_loop(session, subs, duration_s=duration_s)
        finally:
            self.sim.stop()
            self.sim.close()
            for sub in subs:
                try:
                    sub.undeclare()
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop = True

    def _run_loop(self, session: zenoh.Session, subs: list[Any], *, duration_s: float | None) -> None:
        print(
            f"flatdisk sim namespace={self.namespace} mode={self.mode} "
            f"listen={self.listen or '-'} connect={self.connect or '-'} log={self.log_path}",
            flush=True,
        )
        last_step = time.monotonic()
        next_camera = last_step
        next_imu = last_step
        next_status = last_step
        next_log = last_step
        while not self._stop:
            now = time.monotonic()
            if duration_s is not None and now - self._start_monotonic >= duration_s:
                break
            self._poll_commands(session, subs)
            dt = now - last_step
            if dt >= self.sim_period:
                self.sim.step(dt)
                last_step = now
            if now >= next_imu:
                self._publish_imu(session)
                next_imu = now + self.imu_period
            if now >= next_camera:
                self._publish_camera(session)
                next_camera = now + self.camera_period
            if now >= next_status:
                self._publish_status(session)
                next_status = now + self.status_period
            if now >= next_log:
                self._log_hidden_state()
                next_log = now + 0.5
            time.sleep(0.002)

    def _poll_commands(self, session: zenoh.Session, subs: list[Any]) -> None:
        percent_sub, us_sub, stop_sub, sync_sub = subs
        while True:
            sample = percent_sub.try_recv()
            if sample is None:
                break
            command = parse_pair_payload(sample.payload.to_bytes(), "m1", "m2")
            if command is None:
                self.motor_command_errors += 1
                continue
            self.sim.set_motor_percent(command[0], command[1])
            self.motor_commands += 1
        while True:
            sample = us_sub.try_recv()
            if sample is None:
                break
            command = parse_pair_payload(sample.payload.to_bytes(), "m1_us", "m2_us")
            if command is None:
                self.motor_command_errors += 1
                continue
            self.sim.set_motor_us(command[0], command[1])
            self.motor_commands += 1
        while True:
            sample = stop_sub.try_recv()
            if sample is None:
                break
            self.sim.stop()
            self.motor_commands += 1
        while True:
            sample = sync_sub.try_recv()
            if sample is None:
                break
            recv_us = self._esp_us()
            reply = pack_time_sync_reply(sample.payload.to_bytes(), recv_us, self._esp_us())
            if reply is not None:
                session.put(f"{self.namespace}/time_sync", reply)

    def _publish_camera(self, session: zenoh.Session) -> None:
        try:
            image = self.sim.render_image()
            # The physical camera is mounted upside down and existing Python
            # tools rotate decoded frames by 180 degrees. Publish the same raw
            # orientation so simulator clients see upright images through the
            # real robot client path.
            raw_image = image.rotate(180)
            jpeg = image_to_jpeg(raw_image)
            payload = pack_video_jpeg(
                jpeg=jpeg,
                width=image.width,
                height=image.height,
                seq=self.video_seq,
                esp_us=self._esp_us(),
            )
            session.put(f"{self.namespace}/camera/jpeg", payload)
            self.video_seq = (self.video_seq + 1) & 0xFFFFFFFF
            self.video_published += 1
        except Exception as exc:
            self.video_errors += 1
            print(f"camera publish failed: {exc}", flush=True)

    def _publish_imu(self, session: zenoh.Session) -> None:
        try:
            state = self.sim.state
            payload = pack_imu(
                yaw_rad=state.yaw,
                yaw_rate_rad_s=state.yaw_rate,
                linear_accel_body=state.linear_accel_body,
                seq=self.imu_seq,
                esp_us=self._esp_us(),
            )
            session.put(f"{self.namespace}/imu", payload)
            self.imu_seq = (self.imu_seq + 1) & 0xFFFFFFFF
            self.imu_published += 1
        except Exception as exc:
            self.imu_errors += 1
            print(f"imu publish failed: {exc}", flush=True)

    def _publish_status(self, session: zenoh.Session) -> None:
        status = {
            "sim": True,
            "backend": self.sim.backend_name,
            "world": self.sim.world_name,
            "scene": self.sim.scene_name,
            "camera_ready": True,
            "imu_ready": True,
            "camera_width": self.sim.config.width,
            "camera_height": self.sim.config.height,
            "camera_mount_height_m": self.sim.config.camera_height_m,
            "camera_forward_offset_m": self.sim.config.camera_forward_offset_m,
            "camera_horizontal_fov_deg": round(self.sim.camera_settings.horizontal_fov_deg, 6),
            "camera_vertical_fov_deg": round(self.sim.camera_settings.vertical_fov_deg, 6),
            "camera_near_plane_m": self.sim.config.camera_near_plane_m,
            "camera_far_plane_m": self.sim.config.camera_far_plane_m,
            "camera_calibration": str(self.sim.config.camera_calibration) if self.sim.config.camera_calibration else None,
            "camera_source": "third-party" if self.sim.config.use_third_party_camera else "agent",
            "camera_fov_source": self.sim.camera_settings.source,
            "motor1_percent": round(self.sim.state.motor1_percent, 2),
            "motor2_percent": round(self.sim.state.motor2_percent, 2),
            "motor_commands": self.motor_commands,
            "motor_command_errors": self.motor_command_errors,
            "video_published": self.video_published,
            "imu_published": self.imu_published,
            "video_errors": self.video_errors,
            "imu_errors": self.imu_errors,
            "collided": self.sim.state.collided,
            "last_action_success": self.sim.state.last_action_success,
            "last_error": self.sim.state.error_message,
            "sim_time_s": round(time.monotonic() - self._start_monotonic, 3),
        }
        session.put(f"{self.namespace}/status", json.dumps(status, sort_keys=True).encode("utf-8"))
        self.status_published += 1

    def _log_hidden_state(self) -> None:
        payload = {
            "t": round(time.monotonic() - self._start_monotonic, 3),
            "pose": self.sim.hidden_pose(),
            "motor1_percent": self.sim.state.motor1_percent,
            "motor2_percent": self.sim.state.motor2_percent,
            "collided": self.sim.state.collided,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _esp_us(self) -> int:
        return int((time.monotonic() - self._start_monotonic) * 1_000_000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--mode", default="peer", choices=("peer", "client"))
    parser.add_argument("--listen", default=DEFAULT_LISTEN)
    parser.add_argument("--connect", default="")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional run duration in seconds.")
    parser.add_argument("--camera-hz", type=float, default=10.0)
    parser.add_argument("--imu-hz", type=float, default=60.0)
    parser.add_argument("--status-hz", type=float, default=1.0)
    parser.add_argument("--sim-hz", type=float, default=20.0)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--log-dir", type=Path, default=SCRATCH_ROOT)
    parser.add_argument("--backend", default="procthor", choices=("procthor", "ithor", "house-json"))
    parser.add_argument(
        "--scene",
        default=DEFAULT_ITHOR_SCENE,
        help="AI2-THOR scene name, a group name such as bedrooms/bathrooms, or random when --backend ithor.",
    )
    parser.add_argument("--house-json", type=Path, default=None, help="ProcTHOR house JSON for --backend house-json.")
    parser.add_argument("--procthor-seed", type=int, default=42)
    parser.add_argument("--procthor-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--random-start", action="store_true", help="Teleport to a random reachable point after reset.")
    parser.add_argument(
        "--field-of-view",
        type=float,
        default=DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
        help="Camera FOV in degrees. Defaults to the flat disk camera horizontal FOV.",
    )
    parser.add_argument(
        "--field-of-view-axis",
        default="horizontal",
        choices=("horizontal", "vertical"),
        help="Axis for --field-of-view. AI2-THOR receives a derived vertical FOV.",
    )
    parser.add_argument("--camera-height-m", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    parser.add_argument("--camera-forward-offset-m", type=float, default=DEFAULT_CAMERA_FORWARD_OFFSET_M)
    parser.add_argument("--camera-near-plane-m", type=float, default=DEFAULT_CAMERA_NEAR_PLANE_M)
    parser.add_argument("--camera-far-plane-m", type=float, default=DEFAULT_CAMERA_FAR_PLANE_M)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=None,
        help="Calibration JSON from scripts/checkerboard_calibration_logger.py; overrides --field-of-view.",
    )
    parser.add_argument(
        "--use-agent-camera",
        action="store_true",
        help="Render the AI2-THOR primary agent camera instead of the attached low robot camera.",
    )
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--rotate-step-degrees", type=float, default=5.0)
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--start-z", type=float, default=None)
    parser.add_argument("--start-yaw-deg", type=float, default=None)
    parser.add_argument("--quality", default="Low")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sim = ZenohFlatDiskSimulator(
        namespace=args.namespace,
        mode=args.mode,
        listen=args.listen,
        connect=args.connect,
        camera_hz=args.camera_hz,
        imu_hz=args.imu_hz,
        status_hz=args.status_hz,
        sim_hz=args.sim_hz,
        render_width=args.render_width,
        render_height=args.render_height,
        log_dir=args.log_dir,
        backend=args.backend,
        scene=args.scene,
        house_json=args.house_json,
        procthor_seed=args.procthor_seed,
        procthor_split=args.procthor_split,
        random_start=args.random_start,
        field_of_view=args.field_of_view,
        field_of_view_axis=args.field_of_view_axis,
        camera_height_m=args.camera_height_m,
        camera_forward_offset_m=args.camera_forward_offset_m,
        camera_near_plane_m=args.camera_near_plane_m,
        camera_far_plane_m=args.camera_far_plane_m,
        camera_calibration=args.camera_calibration,
        use_third_party_camera=not args.use_agent_camera,
        grid_size=args.grid_size,
        rotate_step_degrees=args.rotate_step_degrees,
        quality=args.quality,
        start_x=args.start_x,
        start_y=args.start_y,
        start_z=args.start_z,
        start_yaw_deg=args.start_yaw_deg,
    )

    def handle_signal(_signum: int, _frame: object) -> None:
        sim.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    sim.run(duration_s=args.duration if args.duration > 0 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
