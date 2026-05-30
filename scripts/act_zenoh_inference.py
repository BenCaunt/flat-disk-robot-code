#!/usr/bin/env python3
"""Run ACT policy inference from live Zenoh camera frames and publish motor commands."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from io import BytesIO
import json
import secrets
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import zenoh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_act import (
    ActPolicy,
    TemporalEnsembler,
    choose_device,
    decode_actions,
    denormalize_model_actions,
    encode_actions,
    normalize_model_actions,
)


VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
AUTO_RERUN_SAVE = Path("__auto__")


def build_config(mode: str, listen: str, connect: str) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps([listen]))
    if connect:
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    return config


def parse_video_sample(data: bytes) -> tuple[int, int, int, bytes] | None:
    if len(data) < VIDEO_STRUCT.size:
        return None
    magic, version, _fmt, width, height, header_len, seq, _esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(data)
    if magic != b"FDV1" or version != 1 or header_len > len(data):
        return None
    jpeg = data[header_len:header_len + jpeg_len]
    if len(jpeg) != jpeg_len:
        return None
    return seq, width, height, jpeg


def preprocess_jpeg(jpeg: bytes, image_size: int, rotate_180: bool) -> torch.Tensor:
    image = Image.open(BytesIO(jpeg)).convert("RGB")
    if rotate_180:
        image = image.rotate(180)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def model_view_jpeg(jpeg: bytes, rotate_180: bool) -> bytes:
    if not rotate_180:
        return jpeg
    image = Image.open(BytesIO(jpeg)).convert("RGB").rotate(180)
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[ActPolicy, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = ActPolicy(
        image_history=int(config["image_history"]),
        past_action_steps=int(config["past_action_steps"]),
        chunk_len=int(config["chunk_len"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def clamp_action(action: np.ndarray, max_abs_output: float) -> tuple[int, int]:
    clipped = np.clip(action, -max_abs_output, max_abs_output)
    return int(round(float(clipped[0]))), int(round(float(clipped[1])))


def forward_steer(action: np.ndarray | tuple[int, int]) -> tuple[float, float]:
    motor1 = float(action[0])
    motor2 = float(action[1])
    return (motor1 + motor2) * 0.5, (motor1 - motor2) * 0.5


def unique_rerun_save_path(checkpoint: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = checkpoint.parent.name or checkpoint.stem
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in run_name)[:64]
    return Path("captures/act-inference") / f"{timestamp}-{secrets.token_hex(4)}-{safe_name}.rrd"


@dataclass
class InferenceResult:
    command: tuple[int, int]
    raw_action: np.ndarray
    chunk0_action: np.ndarray
    chunk_motor_percent: np.ndarray


class RerunActLogger:
    def __init__(self, args: argparse.Namespace, model_config: dict[str, Any]) -> None:
        self.enabled = bool(args.rerun or args.rerun_save or args.rerun_grpc)
        self.rr: Any | None = None
        if not self.enabled:
            return

        import rerun as rr

        self.rr = rr
        rr.init("flatdisk_act_inference", spawn=args.rerun and not args.rerun_no_spawn)
        if args.rerun_save is not None:
            args.rerun_save.parent.mkdir(parents=True, exist_ok=True)
            rr.save(args.rerun_save)
            print(f"ACT Rerun save path: {args.rerun_save}", flush=True)
        elif args.rerun_grpc:
            rr.connect_grpc(args.rerun_grpc)

        rr.log("act/config", rr.TextLog(json.dumps(model_config, sort_keys=True)))

    def close(self) -> None:
        if self.rr is not None:
            self.rr.disconnect()

    def log_inference(
        self,
        *,
        elapsed_s: float,
        video_seq: int | None,
        jpeg: bytes,
        result: InferenceResult,
        prediction_count: int,
        publish_count: int,
        video_drop_count: int,
        video_sample_count: int,
        latest_status: dict[str, Any] | None,
        rotate_180: bool,
    ) -> None:
        if self.rr is None:
            return
        rr = self.rr
        rr.set_time("act_time", duration=elapsed_s)
        if video_seq is not None:
            rr.set_time("video_seq", sequence=video_seq)

        rr.log("act/camera/model_view", rr.EncodedImage(contents=model_view_jpeg(jpeg, rotate_180), media_type="image/jpeg"))
        rr.log("act/counts/predictions", rr.Scalars(prediction_count))
        rr.log("act/counts/published", rr.Scalars(publish_count))
        rr.log("act/video/drops", rr.Scalars(video_drop_count))
        rr.log("act/video/samples", rr.Scalars(video_sample_count))

        raw_forward, raw_steer = forward_steer(result.raw_action)
        cmd_forward, cmd_steer = forward_steer(result.command)
        chunk0_forward, chunk0_steer = forward_steer(result.chunk0_action)
        rr.log("act/motors/raw/motor1_percent", rr.Scalars(float(result.raw_action[0])))
        rr.log("act/motors/raw/motor2_percent", rr.Scalars(float(result.raw_action[1])))
        rr.log("act/motors/command/motor1_percent", rr.Scalars(result.command[0]))
        rr.log("act/motors/command/motor2_percent", rr.Scalars(result.command[1]))
        rr.log("act/motors/chunk0/motor1_percent", rr.Scalars(float(result.chunk0_action[0])))
        rr.log("act/motors/chunk0/motor2_percent", rr.Scalars(float(result.chunk0_action[1])))
        rr.log("act/motors/raw/forward_percent", rr.Scalars(raw_forward))
        rr.log("act/motors/raw/steer_percent", rr.Scalars(raw_steer))
        rr.log("act/motors/command/forward_percent", rr.Scalars(cmd_forward))
        rr.log("act/motors/command/steer_percent", rr.Scalars(cmd_steer))
        rr.log("act/motors/chunk0/forward_percent", rr.Scalars(chunk0_forward))
        rr.log("act/motors/chunk0/steer_percent", rr.Scalars(chunk0_steer))
        rr.log("act/motors/delta/raw_m1_minus_m2", rr.Scalars(float(result.raw_action[0] - result.raw_action[1])))
        rr.log("act/motors/delta/command_m1_minus_m2", rr.Scalars(float(result.command[0] - result.command[1])))
        rr.log("act/motors/delta/chunk0_m1_minus_m2", rr.Scalars(float(result.chunk0_action[0] - result.chunk0_action[1])))

        for step, action in enumerate(result.chunk_motor_percent[: min(len(result.chunk_motor_percent), 16)]):
            rr.log(f"act/chunk/motor1_step_{step:02d}_percent", rr.Scalars(float(action[0])))
            rr.log(f"act/chunk/motor2_step_{step:02d}_percent", rr.Scalars(float(action[1])))

        if latest_status is not None:
            for key in ("motor1_percent", "motor2_percent", "rssi", "video_errors", "motor_command_errors"):
                value = latest_status.get(key)
                if isinstance(value, (int, float)):
                    rr.log(f"act/status/{key}", rr.Scalars(value))


class ActZenohRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = choose_device(args.device)
        self.model, self.model_config = load_policy(args.checkpoint, self.device)
        self.action_scale = float(self.model_config.get("action_scale", 100.0))
        self.action_representation = str(self.model_config.get("action_representation", "left_right"))
        self.action_mean = np.asarray(self.model_config.get("action_mean", [0.0, 0.0]), dtype=np.float32)
        self.action_std = np.asarray(self.model_config.get("action_std", [1.0, 1.0]), dtype=np.float32)
        self.image_size = int(self.model_config["image_size"])
        self.image_history_len = int(self.model_config["image_history"])
        self.past_action_steps = int(self.model_config["past_action_steps"])
        self.chunk_len = int(self.model_config["chunk_len"])
        self.temporal_ensembling = bool(args.temporal_ensembling)
        self.ensembler = TemporalEnsembler(
            self.chunk_len,
            action_dim=2,
            decay=float(args.temporal_ensemble_decay),
        )
        self.rerun_log = RerunActLogger(args, self.model_config)

        self.image_history: deque[torch.Tensor] = deque(maxlen=self.image_history_len)
        self.action_history: deque[np.ndarray] = deque(
            [np.zeros(2, dtype=np.float32) for _ in range(self.past_action_steps)],
            maxlen=self.past_action_steps,
        )
        self.stop = False
        self.latest_status: dict[str, Any] | None = None
        self.last_video_seq: int | None = None
        self.last_frame_ns: int | None = None
        self.prediction_count = 0
        self.publish_count = 0
        self.video_sample_count = 0
        self.video_invalid_count = 0
        self.video_drop_count = 0
        self.last_command: tuple[int, int] = (0, 0)
        self.next_report_ns = time.monotonic_ns() + 1_000_000_000

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop = True

    def publish_stop(self, session: zenoh.Session) -> None:
        session.put(f"{self.args.namespace}/cmd/motors/stop", b"stop")

    def publish_action(self, session: zenoh.Session, m1: int, m2: int) -> None:
        payload = json.dumps({"m1": m1, "m2": m2}).encode("utf-8")
        session.put(f"{self.args.namespace}/cmd/motors/percent", payload)
        self.publish_count += 1

    def publish_act_status(self, session: zenoh.Session, action: tuple[int, int], raw_action: np.ndarray) -> None:
        if not self.args.publish_status:
            return
        payload = json.dumps(
            {
                "armed": self.args.arm,
                "predictions": self.prediction_count,
                "published": self.publish_count,
                "motor1_percent": action[0],
                "motor2_percent": action[1],
                "raw_motor1_percent": float(raw_action[0]),
                "raw_motor2_percent": float(raw_action[1]),
                "raw_forward_percent": forward_steer(raw_action)[0],
                "raw_steer_percent": forward_steer(raw_action)[1],
                "command_forward_percent": forward_steer(action)[0],
                "command_steer_percent": forward_steer(action)[1],
                "action_representation": self.action_representation,
                "normalize_actions": bool(self.model_config.get("normalize_actions", False)),
                "temporal_ensembling": self.temporal_ensembling,
                "video_seq": self.last_video_seq,
                "video_drops": self.video_drop_count,
            },
            sort_keys=True,
        ).encode("utf-8")
        session.put(f"{self.args.namespace}/act/status", payload)

    def update_status(self, status_sub: Any) -> None:
        latest: dict[str, Any] | None = None
        while True:
            sample = status_sub.try_recv()
            if sample is None:
                break
            try:
                latest = json.loads(sample.payload.to_bytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if latest is None:
            return
        self.latest_status = latest
        if self.args.seed_actions_from_status:
            m1 = latest.get("motor1_percent")
            m2 = latest.get("motor2_percent")
            if isinstance(m1, (int, float)) and isinstance(m2, (int, float)):
                self.action_history.append(np.asarray([m1, m2], dtype=np.float32))

    def latest_video(self, video_sub: Any) -> tuple[int, int, int, bytes] | None:
        latest: tuple[int, int, int, bytes] | None = None
        while True:
            sample = video_sub.try_recv()
            if sample is None:
                break
            self.video_sample_count += 1
            parsed = parse_video_sample(sample.payload.to_bytes())
            if parsed is not None:
                latest = parsed
            else:
                self.video_invalid_count += 1
        return latest

    def infer(self, jpeg: bytes) -> InferenceResult:
        image = preprocess_jpeg(jpeg, self.image_size, self.args.rotate_180)
        self.image_history.append(image)
        while len(self.image_history) < self.image_history_len:
            self.image_history.appendleft(image)

        images = torch.stack(list(self.image_history)).unsqueeze(0).to(self.device)
        past_motor_actions = np.stack(list(self.action_history), axis=0)
        past_actions_np = encode_actions(past_motor_actions, self.action_representation) / self.action_scale
        past_actions_np = normalize_model_actions(past_actions_np, self.action_mean, self.action_std)
        past_actions = torch.from_numpy(past_actions_np).float().unsqueeze(0).to(self.device)

        with torch.inference_mode():
            chunk = self.model(images, past_actions).squeeze(0).detach().cpu().numpy()
        chunk_model_percent = denormalize_model_actions(chunk, self.action_mean, self.action_std) * self.action_scale
        chunk_motor_percent = decode_actions(chunk_model_percent, self.action_representation)
        if self.temporal_ensembling:
            action_norm = self.ensembler.add_prediction(chunk)
        else:
            action_norm = chunk[0]
        model_action = denormalize_model_actions(action_norm, self.action_mean, self.action_std) * self.action_scale
        raw_action = decode_actions(model_action, self.action_representation)
        command = clamp_action(raw_action, self.args.max_abs_output)
        self.action_history.append(np.asarray(command, dtype=np.float32))
        self.prediction_count += 1
        self.last_command = command
        return InferenceResult(
            command=command,
            raw_action=raw_action,
            chunk0_action=chunk_motor_percent[0],
            chunk_motor_percent=chunk_motor_percent,
        )

    def report(self) -> None:
        now_ns = time.monotonic_ns()
        if now_ns < self.next_report_ns:
            return
        status = self.latest_status or {}
        age_ms = float("nan")
        if self.last_frame_ns is not None:
            age_ms = (now_ns - self.last_frame_ns) / 1_000_000.0
        print(
            "act "
            f"armed={self.args.arm} pred={self.prediction_count} pub={self.publish_count} "
            f"cmd={self.last_command[0]}/{self.last_command[1]}% "
            f"video_samples={self.video_sample_count} invalid={self.video_invalid_count} "
            f"video_seq={self.last_video_seq} drops={self.video_drop_count} frame_age_ms={age_ms:.0f} "
            f"device={status.get('motor1_percent', '?')}/{status.get('motor2_percent', '?')}%",
            flush=True,
        )
        self.next_report_ns = now_ns + 1_000_000_000

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)

        session = zenoh.open(build_config(self.args.mode, self.args.listen, self.args.connect))
        video_sub = session.declare_subscriber(f"{self.args.namespace}/camera/jpeg")
        status_sub = session.declare_subscriber(f"{self.args.namespace}/status")
        endpoint = self.args.connect if self.args.connect else self.args.listen
        print(
            f"ACT Zenoh inference mode={self.args.mode} endpoint={endpoint} namespace={self.args.namespace} "
            f"checkpoint={self.args.checkpoint} device={self.device} armed={self.args.arm}",
            flush=True,
        )
        if not self.args.arm:
            print("Not armed: running inference and status publishing only. Add --arm to publish motor commands.", flush=True)

        start_ns = time.monotonic_ns()
        try:
            while not self.stop:
                now_ns = time.monotonic_ns()
                if self.args.duration > 0 and (now_ns - start_ns) / 1_000_000_000.0 >= self.args.duration:
                    break

                self.update_status(status_sub)
                latest = self.latest_video(video_sub)
                if latest is not None:
                    seq, _width, _height, jpeg = latest
                    if self.last_video_seq is not None and seq > self.last_video_seq + 1:
                        self.video_drop_count += seq - self.last_video_seq - 1
                    self.last_video_seq = seq
                    self.last_frame_ns = now_ns
                    result = self.infer(jpeg)
                    command = result.command
                    if self.args.arm:
                        self.publish_action(session, *command)
                    self.publish_act_status(session, command, result.raw_action)
                    self.rerun_log.log_inference(
                        elapsed_s=(now_ns - start_ns) / 1_000_000_000.0,
                        video_seq=self.last_video_seq,
                        jpeg=jpeg,
                        result=result,
                        prediction_count=self.prediction_count,
                        publish_count=self.publish_count,
                        video_drop_count=self.video_drop_count,
                        video_sample_count=self.video_sample_count,
                        latest_status=self.latest_status,
                        rotate_180=self.args.rotate_180,
                    )

                if (
                    self.args.arm
                    and self.last_frame_ns is not None
                    and (now_ns - self.last_frame_ns) / 1_000_000_000.0 > self.args.stale_timeout
                ):
                    print("Camera stream stale; publishing stop.", flush=True)
                    self.publish_stop(session)
                    self.stop = True

                self.report()
                time.sleep(0.001)
        finally:
            try:
                if self.args.arm and self.args.stop_on_exit:
                    self.publish_stop(session)
            finally:
                video_sub.undeclare()
                status_sub.undeclare()
                session.close()
                self.rerun_log.close()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/act_vla/best.pt"))
    parser.add_argument("--namespace", default="flatdisk/xiao")
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arm", action="store_true", help="Actually publish motor commands.")
    parser.add_argument("--max-abs-output", type=float, default=10.0, help="Clamp motor percent output magnitude.")
    parser.add_argument("--stale-timeout", type=float, default=0.5, help="Stop if no camera frame arrives for this long.")
    parser.add_argument("--no-stop-on-exit", dest="stop_on_exit", action="store_false")
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.add_argument("--no-temporal-ensembling", dest="temporal_ensembling", action="store_false")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=None)
    parser.add_argument("--no-publish-status", dest="publish_status", action="store_false")
    parser.add_argument("--seed-actions-from-status", action="store_true")
    parser.add_argument("--rerun", action="store_true", help="Log ACT inference to a spawned Rerun viewer.")
    parser.add_argument(
        "--rerun-save",
        type=Path,
        nargs="?",
        const=AUTO_RERUN_SAVE,
        default=None,
        help="Save ACT inference Rerun recording. Omit the path to use a unique captures/act-inference/*.rrd name.",
    )
    parser.add_argument("--rerun-grpc", default="", help="Connect ACT inference logging to an existing Rerun gRPC endpoint.")
    parser.add_argument("--rerun-no-spawn", action="store_true", help="Initialize Rerun logging without spawning a viewer.")
    parser.set_defaults(stop_on_exit=True, rotate_180=True, temporal_ensembling=None, publish_status=True)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.rerun_save == AUTO_RERUN_SAVE:
        args.rerun_save = unique_rerun_save_path(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint["config"]
    if args.temporal_ensembling is None:
        args.temporal_ensembling = bool(checkpoint_config.get("temporal_ensembling", True))
    if args.temporal_ensemble_decay is None:
        args.temporal_ensemble_decay = float(checkpoint_config.get("temporal_ensemble_decay", 0.01))

    return ActZenohRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
