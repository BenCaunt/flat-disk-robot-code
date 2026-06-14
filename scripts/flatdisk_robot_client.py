#!/usr/bin/env python3
"""Reusable Zenoh client for IMU-guided flat-disk robot moves."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import struct
import time
from typing import Any

from PIL import Image, ImageDraw
import zenoh

from zenoh_env import default_zenoh_connect


VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
IMU_STRUCT = struct.Struct("<4sBBBBIQ14f")

DEFAULT_NAMESPACE = "flatdisk/xiao"
DEFAULT_MODE = "client"
DEFAULT_CONNECT = default_zenoh_connect()
DEFAULT_CONTROL_HZ = 20.0
DEFAULT_REVERSE_YAW = True
DEFAULT_REVERSE_CORRECTION = False
DEFAULT_HEADING_KP = 8.0
DEFAULT_MAX_TURN_PERCENT = 10.0
DEFAULT_MIN_TURN_PERCENT = 2.0
DEFAULT_HEADING_DEADBAND_DEG = 1.0
DEFAULT_ANGLE_TOLERANCE_DEG = 3.0
DEFAULT_SETTLE_TIME_S = 0.2
DEFAULT_IMU_TIMEOUT_S = 0.5
DEFAULT_SAMPLE_TIMEOUT_S = 2.0
DEFAULT_MAX_DRIVE_S = 10.0
DEFAULT_FRAME_COUNT = 5


def build_config(mode: str = DEFAULT_MODE, listen: str = "", connect: str = DEFAULT_CONNECT) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps([listen]))
    if connect:
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    return config


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def wrap_degrees(angle_deg: float) -> float:
    return math.degrees(wrap_pi(math.radians(angle_deg)))


def quat_xyzw_to_yaw_rad(x: float, y: float, z: float, w: float) -> float | None:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-6:
        return None
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True)
class ImuSample:
    seq: int
    esp_us: int
    yaw_rad: float
    raw_yaw_rad: float
    quat_accuracy: int
    accel_accuracy: int
    flags: int
    received_ns: int

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    @property
    def raw_yaw_deg(self) -> float:
        return math.degrees(self.raw_yaw_rad)


@dataclass(frozen=True)
class VideoFrame:
    seq: int
    esp_us: int
    width: int
    height: int
    jpeg: bytes
    received_ns: int

    def image(self, *, rotate_180: bool = True) -> Image.Image:
        image = Image.open(BytesIO(self.jpeg)).convert("RGB")
        return image.rotate(180) if rotate_180 else image


@dataclass(frozen=True)
class MotionResult:
    action: str
    started_yaw_deg: float
    final_yaw_deg: float
    target_yaw_deg: float | None
    elapsed_s: float
    timed_out: bool
    frames: list[VideoFrame]
    stitched_path: Path | None = None

    @property
    def heading_error_deg(self) -> float | None:
        if self.target_yaw_deg is None:
            return None
        return wrap_degrees(self.target_yaw_deg - self.final_yaw_deg)

    def summary(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "started_yaw_deg": self.started_yaw_deg,
            "final_yaw_deg": self.final_yaw_deg,
            "target_yaw_deg": self.target_yaw_deg,
            "heading_error_deg": self.heading_error_deg,
            "elapsed_s": self.elapsed_s,
            "timed_out": self.timed_out,
            "frame_count": len(self.frames),
            "frame_seqs": [frame.seq for frame in self.frames],
            "stitched_path": str(self.stitched_path) if self.stitched_path else None,
        }


class MotionFrameRecorder:
    def __init__(self) -> None:
        self._frames: list[tuple[float, VideoFrame]] = []
        self._last_seq: int | None = None

    def note(self, frame: VideoFrame | None) -> None:
        if frame is None:
            return
        if self._last_seq == frame.seq:
            return
        self._last_seq = frame.seq
        self._frames.append((time.monotonic(), frame))

    def evenly_spaced(self, count: int) -> list[VideoFrame]:
        if count <= 0 or not self._frames:
            return []
        if len(self._frames) <= count:
            return [frame for _ts, frame in self._frames]

        first_ts = self._frames[0][0]
        last_ts = self._frames[-1][0]
        if last_ts <= first_ts:
            return [frame for _ts, frame in self._frames[:count]]

        selected: list[VideoFrame] = []
        start_index = 0
        for i in range(count):
            target_ts = first_ts + (last_ts - first_ts) * i / max(count - 1, 1)
            best_index = start_index
            best_dist = abs(self._frames[best_index][0] - target_ts)
            for j in range(start_index + 1, len(self._frames)):
                dist = abs(self._frames[j][0] - target_ts)
                if dist > best_dist:
                    break
                best_index = j
                best_dist = dist
            start_index = best_index
            selected.append(self._frames[best_index][1])
        return selected


class FlatDiskRobotClient:
    """Small synchronous client for flat-disk camera, IMU, and motor topics."""

    def __init__(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        mode: str = DEFAULT_MODE,
        connect: str = DEFAULT_CONNECT,
        listen: str = "",
        reverse_yaw: bool = DEFAULT_REVERSE_YAW,
        reverse_correction: bool = DEFAULT_REVERSE_CORRECTION,
        heading_kp: float = DEFAULT_HEADING_KP,
        max_turn_percent: float = DEFAULT_MAX_TURN_PERCENT,
        min_turn_percent: float = DEFAULT_MIN_TURN_PERCENT,
        heading_deadband_deg: float = DEFAULT_HEADING_DEADBAND_DEG,
        imu_timeout_s: float = DEFAULT_IMU_TIMEOUT_S,
        control_hz: float = DEFAULT_CONTROL_HZ,
        rotate_frames_180: bool = True,
    ) -> None:
        self.namespace = namespace.strip()
        self.mode = mode
        self.connect_endpoint = connect
        self.listen_endpoint = listen
        self.reverse_yaw = reverse_yaw
        self.reverse_correction = reverse_correction
        self.heading_kp = heading_kp
        self.max_turn_percent = max_turn_percent
        self.min_turn_percent = min_turn_percent
        self.heading_deadband_deg = heading_deadband_deg
        self.imu_timeout_s = imu_timeout_s
        self.control_hz = control_hz
        self.rotate_frames_180 = rotate_frames_180

        self.session: zenoh.Session | None = None
        self.video_sub: Any | None = None
        self.imu_sub: Any | None = None
        self.last_frame: VideoFrame | None = None
        self.last_imu: ImuSample | None = None

    def __enter__(self) -> FlatDiskRobotClient:
        self.open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def open(self) -> None:
        if self.session is not None:
            return
        self.session = zenoh.open(build_config(self.mode, self.listen_endpoint, self.connect_endpoint))
        self.video_sub = self.session.declare_subscriber(f"{self.namespace}/camera/jpeg")
        self.imu_sub = self.session.declare_subscriber(f"{self.namespace}/imu")

    def close(self) -> None:
        self.stop()
        for sub in (self.video_sub, self.imu_sub):
            if sub is not None:
                try:
                    sub.undeclare()
                except Exception:
                    pass
        self.video_sub = None
        self.imu_sub = None
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self.session = None

    def poll(self) -> None:
        self.open()
        self._poll_video()
        self._poll_imu()

    def get_angle(self, *, timeout_s: float = DEFAULT_SAMPLE_TIMEOUT_S) -> float:
        return self.wait_for_imu(timeout_s=timeout_s).yaw_deg

    def latest_frame(self, *, timeout_s: float = DEFAULT_SAMPLE_TIMEOUT_S) -> VideoFrame:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() <= deadline:
            self.poll()
            if self.last_frame is not None:
                return self.last_frame
            time.sleep(0.02)
        raise TimeoutError(f"no camera frame within {timeout_s:.2f}s")

    def turn_by_angle(
        self,
        delta_deg: float,
        *,
        power: float = DEFAULT_MAX_TURN_PERCENT,
        frame_count: int = DEFAULT_FRAME_COUNT,
        timeout_s: float | None = None,
        tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
        settle_time_s: float = DEFAULT_SETTLE_TIME_S,
        output_dir: Path | None = None,
    ) -> MotionResult:
        start = self.wait_for_imu(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        return self.turn_to_angle(
            start.yaw_deg + delta_deg,
            power=power,
            frame_count=frame_count,
            timeout_s=timeout_s,
            tolerance_deg=tolerance_deg,
            settle_time_s=settle_time_s,
            output_dir=output_dir,
        )

    def turn_to_angle(
        self,
        target_deg: float,
        *,
        power: float = DEFAULT_MAX_TURN_PERCENT,
        frame_count: int = DEFAULT_FRAME_COUNT,
        timeout_s: float | None = None,
        tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
        settle_time_s: float = DEFAULT_SETTLE_TIME_S,
        output_dir: Path | None = None,
    ) -> MotionResult:
        start = self.wait_for_imu(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        if frame_count > 0:
            self.latest_frame(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        target_rad = wrap_pi(math.radians(target_deg))
        max_turn = abs(power)
        timeout_s = timeout_s if timeout_s is not None else self._angle_timeout_s(wrap_degrees(target_deg - start.yaw_deg))
        recorder = MotionFrameRecorder()
        recorder.note(self.last_frame)
        started = time.monotonic()
        settle_started: float | None = None
        timed_out = True

        try:
            while time.monotonic() - started <= timeout_s:
                self.poll()
                imu = self._fresh_imu_or_raise()
                recorder.note(self.last_frame)
                error_rad = wrap_pi(target_rad - imu.yaw_rad)
                if abs(math.degrees(error_rad)) <= tolerance_deg:
                    settle_started = settle_started if settle_started is not None else time.monotonic()
                    if time.monotonic() - settle_started >= settle_time_s:
                        timed_out = False
                        break
                else:
                    settle_started = None

                turn = self._heading_turn_percent(error_rad, max_turn=max_turn)
                self._publish_percent(turn, -turn)
                time.sleep(self._control_period_s())
        finally:
            self.stop()
            self.poll()
            recorder.note(self.last_frame)

        final = self.wait_for_imu(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        frames = recorder.evenly_spaced(frame_count)
        result = MotionResult(
            action="turn_to_angle",
            started_yaw_deg=start.yaw_deg,
            final_yaw_deg=final.yaw_deg,
            target_yaw_deg=math.degrees(target_rad),
            elapsed_s=time.monotonic() - started,
            timed_out=timed_out,
            frames=frames,
        )
        return self._with_stitched_frames(result, output_dir=output_dir)

    def drive_straight(
        self,
        power: float,
        duration_s: float,
        *,
        frame_count: int = DEFAULT_FRAME_COUNT,
        timeout_s: float | None = None,
        max_duration_s: float = DEFAULT_MAX_DRIVE_S,
        output_dir: Path | None = None,
    ) -> MotionResult:
        if duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        if duration_s > max_duration_s:
            raise ValueError(f"duration_s {duration_s:.2f}s exceeds max_duration_s {max_duration_s:.2f}s")

        start = self.wait_for_imu(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        if frame_count > 0:
            self.latest_frame(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        target_rad = start.yaw_rad
        timeout_s = timeout_s if timeout_s is not None else duration_s + 1.0
        recorder = MotionFrameRecorder()
        recorder.note(self.last_frame)
        started = time.monotonic()
        timed_out = True

        try:
            while time.monotonic() - started <= timeout_s:
                elapsed = time.monotonic() - started
                if elapsed >= duration_s:
                    timed_out = False
                    break
                self.poll()
                imu = self._fresh_imu_or_raise()
                recorder.note(self.last_frame)
                error_rad = wrap_pi(target_rad - imu.yaw_rad)
                turn = self._heading_turn_percent(error_rad, max_turn=self.max_turn_percent)
                self._publish_percent(power + turn, power - turn)
                time.sleep(self._control_period_s())
        finally:
            self.stop()
            self.poll()
            recorder.note(self.last_frame)

        final = self.wait_for_imu(timeout_s=DEFAULT_SAMPLE_TIMEOUT_S)
        frames = recorder.evenly_spaced(frame_count)
        result = MotionResult(
            action="drive_straight",
            started_yaw_deg=start.yaw_deg,
            final_yaw_deg=final.yaw_deg,
            target_yaw_deg=start.yaw_deg,
            elapsed_s=time.monotonic() - started,
            timed_out=timed_out,
            frames=frames,
        )
        return self._with_stitched_frames(result, output_dir=output_dir)

    def stop(self) -> None:
        if self.session is None:
            return
        try:
            self.session.put(f"{self.namespace}/cmd/motors/stop", b"stop")
        except Exception:
            pass

    def publish_percent(self, motor1: float, motor2: float) -> None:
        self._publish_percent(motor1, motor2)

    def wait_for_imu(self, *, timeout_s: float = DEFAULT_SAMPLE_TIMEOUT_S) -> ImuSample:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() <= deadline:
            self.poll()
            if self.last_imu is not None and self._imu_age_s(self.last_imu) <= self.imu_timeout_s:
                return self.last_imu
            time.sleep(0.02)
        raise TimeoutError(f"no fresh IMU sample within {timeout_s:.2f}s")

    def _with_stitched_frames(self, result: MotionResult, *, output_dir: Path | None) -> MotionResult:
        if output_dir is None or not result.frames:
            return result
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{stamp}_{result.action}.jpg"
        stitch_frames(result.frames, path, rotate_180=self.rotate_frames_180)
        return MotionResult(
            action=result.action,
            started_yaw_deg=result.started_yaw_deg,
            final_yaw_deg=result.final_yaw_deg,
            target_yaw_deg=result.target_yaw_deg,
            elapsed_s=result.elapsed_s,
            timed_out=result.timed_out,
            frames=result.frames,
            stitched_path=path,
        )

    def _fresh_imu_or_raise(self) -> ImuSample:
        if self.last_imu is None:
            raise TimeoutError("no IMU sample")
        if self._imu_age_s(self.last_imu) > self.imu_timeout_s:
            raise TimeoutError(f"stale IMU sample: {self._imu_age_s(self.last_imu):.2f}s old")
        return self.last_imu

    def _poll_video(self) -> None:
        if self.video_sub is None:
            return
        while True:
            sample = self.video_sub.try_recv()
            if sample is None:
                break
            parsed = self._parse_video_sample(sample.payload.to_bytes(), time.monotonic_ns())
            if parsed is not None:
                self.last_frame = parsed

    def _poll_imu(self) -> None:
        if self.imu_sub is None:
            return
        while True:
            sample = self.imu_sub.try_recv()
            if sample is None:
                break
            parsed = self._parse_imu_sample(sample.payload.to_bytes(), time.monotonic_ns())
            if parsed is not None:
                self.last_imu = parsed

    def _parse_video_sample(self, data: bytes, received_ns: int) -> VideoFrame | None:
        if len(data) < VIDEO_STRUCT.size:
            return None
        magic, version, _fmt, width, height, header_len, seq, esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(data)
        if magic != b"FDV1" or version != 1 or header_len > len(data):
            return None
        jpeg = data[header_len:header_len + jpeg_len]
        if len(jpeg) != jpeg_len:
            return None
        return VideoFrame(seq=seq, esp_us=esp_us, width=width, height=height, jpeg=jpeg, received_ns=received_ns)

    def _parse_imu_sample(self, data: bytes, received_ns: int) -> ImuSample | None:
        if len(data) < IMU_STRUCT.size:
            return None
        unpacked = IMU_STRUCT.unpack_from(data)
        magic, version, quat_accuracy, accel_accuracy, flags, seq, esp_us = unpacked[:7]
        if magic != b"FDI1" or version != 1:
            return None
        qi, qj, qk, qr = unpacked[7:11]
        raw_yaw = quat_xyzw_to_yaw_rad(qi, qj, qk, qr)
        if raw_yaw is None:
            return None
        yaw = wrap_pi(-raw_yaw) if self.reverse_yaw else raw_yaw
        return ImuSample(
            seq=seq,
            esp_us=esp_us,
            yaw_rad=yaw,
            raw_yaw_rad=raw_yaw,
            quat_accuracy=quat_accuracy,
            accel_accuracy=accel_accuracy,
            flags=flags,
            received_ns=received_ns,
        )

    def _heading_turn_percent(self, error_rad: float, *, max_turn: float) -> float:
        deadband_rad = math.radians(max(0.0, self.heading_deadband_deg))
        controlled_error = 0.0
        if abs(error_rad) > deadband_rad:
            controlled_error = math.copysign(abs(error_rad) - deadband_rad, error_rad)
        turn = self.heading_kp * controlled_error
        if 0.0 < abs(turn) < self.min_turn_percent:
            turn = math.copysign(self.min_turn_percent, turn)
        if self.reverse_correction:
            turn = -turn
        return clamp(turn, -abs(max_turn), abs(max_turn))

    def _publish_percent(self, motor1: float, motor2: float) -> None:
        self.open()
        payload = json.dumps({
            "m1": int(round(clamp(motor1, -100.0, 100.0))),
            "m2": int(round(clamp(motor2, -100.0, 100.0))),
        }).encode("utf-8")
        assert self.session is not None
        self.session.put(f"{self.namespace}/cmd/motors/percent", payload)

    def _imu_age_s(self, sample: ImuSample) -> float:
        return (time.monotonic_ns() - sample.received_ns) / 1_000_000_000.0

    def _control_period_s(self) -> float:
        return 1.0 / max(self.control_hz, 1.0)

    @staticmethod
    def _angle_timeout_s(delta_deg: float) -> float:
        return clamp(abs(delta_deg) / 30.0 + 2.0, 3.0, 12.0)


def stitch_frames(
    frames: list[VideoFrame],
    output_path: Path,
    *,
    rotate_180: bool = True,
    max_tile_width: int = 320,
) -> Path:
    if not frames:
        raise ValueError("no frames to stitch")

    images: list[Image.Image] = []
    for i, frame in enumerate(frames, start=1):
        image = frame.image(rotate_180=rotate_180)
        if image.width > max_tile_width:
            scale = max_tile_width / image.width
            image = image.resize((max_tile_width, max(1, int(round(image.height * scale)))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image.width, image.height + 24), "white")
        canvas.paste(image, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), f"{i}/{len(frames)}  seq {frame.seq}", fill=(0, 0, 0))
        images.append(canvas)

    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    stitched = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        stitched.paste(image, (x, 0))
        x += image.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path, format="JPEG", quality=90)
    return output_path


def get_angle(*, timeout_s: float = DEFAULT_SAMPLE_TIMEOUT_S, **client_kwargs: Any) -> float:
    with FlatDiskRobotClient(**client_kwargs) as robot:
        return robot.get_angle(timeout_s=timeout_s)


def latest_frame(*, timeout_s: float = DEFAULT_SAMPLE_TIMEOUT_S, **client_kwargs: Any) -> VideoFrame:
    with FlatDiskRobotClient(**client_kwargs) as robot:
        return robot.latest_frame(timeout_s=timeout_s)


def turn_to_angle(
    target_angle_deg: float,
    *,
    power: float = DEFAULT_MAX_TURN_PERCENT,
    frame_count: int = DEFAULT_FRAME_COUNT,
    timeout_s: float | None = None,
    tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
    output_dir: Path | None = None,
    **client_kwargs: Any,
) -> MotionResult:
    with FlatDiskRobotClient(**client_kwargs) as robot:
        return robot.turn_to_angle(
            target_angle_deg,
            power=power,
            frame_count=frame_count,
            timeout_s=timeout_s,
            tolerance_deg=tolerance_deg,
            output_dir=output_dir,
        )


def turn_by_angle(
    delta_angle_deg: float,
    *,
    power: float = DEFAULT_MAX_TURN_PERCENT,
    frame_count: int = DEFAULT_FRAME_COUNT,
    timeout_s: float | None = None,
    tolerance_deg: float = DEFAULT_ANGLE_TOLERANCE_DEG,
    output_dir: Path | None = None,
    **client_kwargs: Any,
) -> MotionResult:
    with FlatDiskRobotClient(**client_kwargs) as robot:
        return robot.turn_by_angle(
            delta_angle_deg,
            power=power,
            frame_count=frame_count,
            timeout_s=timeout_s,
            tolerance_deg=tolerance_deg,
            output_dir=output_dir,
        )


def drive_straight(
    power: float,
    duration_s: float,
    *,
    frame_count: int = DEFAULT_FRAME_COUNT,
    timeout_s: float | None = None,
    max_duration_s: float = DEFAULT_MAX_DRIVE_S,
    output_dir: Path | None = None,
    **client_kwargs: Any,
) -> MotionResult:
    with FlatDiskRobotClient(**client_kwargs) as robot:
        return robot.drive_straight(
            power,
            duration_s,
            frame_count=frame_count,
            timeout_s=timeout_s,
            max_duration_s=max_duration_s,
            output_dir=output_dir,
        )
