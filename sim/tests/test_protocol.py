from __future__ import annotations

import math

from flatdisk_sim.protocol import IMU_STRUCT, VIDEO_STRUCT, pack_imu, pack_video_jpeg


def test_video_packet_header_round_trip() -> None:
    payload = pack_video_jpeg(jpeg=b"abc", width=320, height=240, seq=7, esp_us=1234)
    magic, version, fmt, width, height, header_len, seq, esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(payload)
    assert magic == b"FDV1"
    assert version == 1
    assert fmt == 1
    assert width == 320
    assert height == 240
    assert header_len == VIDEO_STRUCT.size
    assert seq == 7
    assert esp_us == 1234
    assert jpeg_len == 3
    assert payload[header_len:] == b"abc"


def test_imu_packet_contains_reverse_yaw_compatible_quaternion() -> None:
    yaw = math.radians(30.0)
    payload = pack_imu(yaw_rad=yaw, yaw_rate_rad_s=0.2, linear_accel_body=(1.0, 0.0, 0.0), seq=4, esp_us=99)
    unpacked = IMU_STRUCT.unpack_from(payload)
    assert unpacked[0] == b"FDI1"
    assert unpacked[1] == 1
    assert unpacked[5] == 4
    assert unpacked[6] == 99
    qi, qj, qk, qr = unpacked[7:11]
    raw_yaw = math.atan2(2.0 * (qr * qk + qi * qj), 1.0 - 2.0 * (qj * qj + qk * qk))
    assert math.isclose(raw_yaw, -yaw, abs_tol=1e-6)
