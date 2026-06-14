"""Binary and JSON wire protocol helpers matching the real flat disk robot."""

from __future__ import annotations

from io import BytesIO
import json
import math
import struct
from typing import Any

from PIL import Image
import zenoh


VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
IMU_STRUCT = struct.Struct("<4sBBBBIQ14f")
SYNC_REQ_STRUCT = struct.Struct("<4sIQ")
SYNC_REPLY_STRUCT = struct.Struct("<4sIQQQ")

DEFAULT_NAMESPACE = "flatdisk/xiao"
DEFAULT_LISTEN = "tcp/127.0.0.1:7447"
DEFAULT_CONNECT = ""

VIDEO_FORMAT_JPEG = 1
IMU_REPORT_FLAGS_ROTATION_VECTOR = 1 << 0


def build_config(mode: str = "peer", listen: str = DEFAULT_LISTEN, connect: str = DEFAULT_CONNECT) -> zenoh.Config:
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


def image_to_jpeg(image: Image.Image, *, quality: int = 82) -> bytes:
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def pack_video_jpeg(*, jpeg: bytes, width: int, height: int, seq: int, esp_us: int) -> bytes:
    header_len = VIDEO_STRUCT.size
    header = VIDEO_STRUCT.pack(
        b"FDV1",
        1,
        VIDEO_FORMAT_JPEG,
        int(width),
        int(height),
        header_len,
        int(seq) & 0xFFFFFFFF,
        int(esp_us) & 0xFFFFFFFFFFFFFFFF,
        len(jpeg),
    )
    return header + jpeg


def yaw_to_quat_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    half = yaw_rad * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def pack_imu(
    *,
    yaw_rad: float,
    yaw_rate_rad_s: float,
    linear_accel_body: tuple[float, float, float],
    seq: int,
    esp_us: int,
    reverse_yaw_client_compatible: bool = True,
) -> bytes:
    # Existing Python tools default to reverse_yaw=True, so publish the raw
    # quaternion with opposite yaw to make the client-reported yaw match sim yaw.
    raw_yaw = -yaw_rad if reverse_yaw_client_compatible else yaw_rad
    qi, qj, qk, qr = yaw_to_quat_xyzw(raw_yaw)
    lin_x, lin_y, lin_z = linear_accel_body
    values = (
        qi,
        qj,
        qk,
        qr,
        0.02,
        0.0,
        0.0,
        9.81,
        0.0,
        0.0,
        yaw_rate_rad_s,
        lin_x,
        lin_y,
        lin_z,
    )
    return IMU_STRUCT.pack(
        b"FDI1",
        1,
        3,
        3,
        IMU_REPORT_FLAGS_ROTATION_VECTOR,
        int(seq) & 0xFFFFFFFF,
        int(esp_us) & 0xFFFFFFFFFFFFFFFF,
        *values,
    )


def parse_pair_payload(payload: bytes, key1: str, key2: str) -> tuple[float, float] | None:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    if not isinstance(parsed, dict):
        return None
    try:
        return float(parsed[key1]), float(parsed[key2])
    except (KeyError, TypeError, ValueError):
        return None


def pack_time_sync_reply(payload: bytes, recv_esp_us: int, tx_esp_us: int) -> bytes | None:
    if len(payload) < SYNC_REQ_STRUCT.size:
        return None
    magic, seq, pc_send_ns = SYNC_REQ_STRUCT.unpack_from(payload)
    if magic != b"FDSQ":
        return None
    return SYNC_REPLY_STRUCT.pack(b"FDSR", seq, pc_send_ns, int(recv_esp_us), int(tx_esp_us))
