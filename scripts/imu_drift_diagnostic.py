#!/usr/bin/env python3
"""Collect and summarize BNO085 IMU drift from the flat-disk Zenoh stream."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import struct
import time
from typing import Any

import zenoh

from flatdisk_robot_client import DEFAULT_CONNECT, DEFAULT_NAMESPACE, build_config, wrap_degrees


IMU_STRUCT = struct.Struct("<4sBBBBIQ14f")


@dataclass(frozen=True)
class ImuRecord:
    host_t_s: float
    seq: int
    esp_us: int
    quat_accuracy: int
    accel_accuracy: int
    flags: int
    qx: float
    qy: float
    qz: float
    qw: float
    quat_rad_accuracy: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    linear_accel_x: float
    linear_accel_y: float
    linear_accel_z: float
    roll_deg: float
    pitch_deg: float
    raw_yaw_deg: float
    client_yaw_deg: float
    accel_norm: float
    linear_accel_norm: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--connect", default=DEFAULT_CONNECT)
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON summary path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional raw sample CSV path.")
    parser.add_argument("--reverse-yaw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stationary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assume the robot is stationary and emit stationary-specific recommendations.",
    )
    args = parser.parse_args()

    records = collect_imu_records(
        namespace=args.namespace,
        connect=args.connect,
        mode=args.mode,
        listen=args.listen,
        duration_s=args.duration,
        reverse_yaw=args.reverse_yaw,
    )
    summary = summarize(records, stationary=args.stationary)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv, records)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if records else 1


def collect_imu_records(
    *,
    namespace: str,
    connect: str,
    mode: str,
    listen: str,
    duration_s: float,
    reverse_yaw: bool,
) -> list[ImuRecord]:
    records: list[ImuRecord] = []
    config = build_config(mode=mode, listen=listen, connect=connect)
    with zenoh.open(config) as session:
        sub = session.declare_subscriber(f"{namespace}/imu")
        deadline = time.monotonic() + max(0.0, duration_s)
        try:
            while time.monotonic() < deadline:
                sample = sub.try_recv()
                if sample is None:
                    time.sleep(0.002)
                    continue
                parsed = parse_imu_sample(sample.payload.to_bytes(), host_t_s=time.monotonic(), reverse_yaw=reverse_yaw)
                if parsed is not None:
                    records.append(parsed)
        finally:
            sub.undeclare()
    return records


def parse_imu_sample(data: bytes, *, host_t_s: float, reverse_yaw: bool) -> ImuRecord | None:
    if len(data) < IMU_STRUCT.size:
        return None
    unpacked = IMU_STRUCT.unpack_from(data)
    magic, version, quat_accuracy, accel_accuracy, flags, seq, esp_us = unpacked[:7]
    if magic != b"FDI1" or version != 1:
        return None
    (
        qx,
        qy,
        qz,
        qw,
        quat_rad_accuracy,
        accel_x,
        accel_y,
        accel_z,
        gyro_x,
        gyro_y,
        gyro_z,
        linear_accel_x,
        linear_accel_y,
        linear_accel_z,
    ) = (float(value) for value in unpacked[7:21])
    roll_deg, pitch_deg, raw_yaw_deg = quat_to_euler_deg(qx, qy, qz, qw)
    client_yaw_deg = -raw_yaw_deg if reverse_yaw else raw_yaw_deg
    return ImuRecord(
        host_t_s=host_t_s,
        seq=int(seq),
        esp_us=int(esp_us),
        quat_accuracy=int(quat_accuracy),
        accel_accuracy=int(accel_accuracy),
        flags=int(flags),
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        quat_rad_accuracy=quat_rad_accuracy,
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        linear_accel_x=linear_accel_x,
        linear_accel_y=linear_accel_y,
        linear_accel_z=linear_accel_z,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        raw_yaw_deg=raw_yaw_deg,
        client_yaw_deg=client_yaw_deg,
        accel_norm=math.sqrt(accel_x * accel_x + accel_y * accel_y + accel_z * accel_z),
        linear_accel_norm=math.sqrt(linear_accel_x * linear_accel_x + linear_accel_y * linear_accel_y + linear_accel_z * linear_accel_z),
    )


def quat_to_euler_deg(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_arg)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def summarize(records: list[ImuRecord], *, stationary: bool) -> dict[str, Any]:
    if not records:
        return {"ok": False, "error": "no IMU samples received"}

    yaw_unwrapped = unwrap_degrees([record.client_yaw_deg for record in records])
    raw_yaw_unwrapped = unwrap_degrees([record.raw_yaw_deg for record in records])
    duration_s = max(records[-1].host_t_s - records[0].host_t_s, 1e-9)
    sample_intervals = [
        (records[index].host_t_s - records[index - 1].host_t_s) * 1000.0
        for index in range(1, len(records))
    ]
    drift_deg = yaw_unwrapped[-1] - yaw_unwrapped[0]
    raw_drift_deg = raw_yaw_unwrapped[-1] - raw_yaw_unwrapped[0]
    drift_rate_deg_min = drift_deg / duration_s * 60.0
    gyro_z_deg_s = [math.degrees(record.gyro_z) for record in records]
    roll = [record.roll_deg for record in records]
    pitch = [record.pitch_deg for record in records]
    accel_norm = [record.accel_norm for record in records]
    linear_accel_norm = [record.linear_accel_norm for record in records]
    quat_acc = [record.quat_accuracy for record in records]
    accel_acc = [record.accel_accuracy for record in records]
    quat_rad_acc = [record.quat_rad_accuracy for record in records]

    summary: dict[str, Any] = {
        "ok": True,
        "sample_count": len(records),
        "duration_s": round(duration_s, 3),
        "rate_hz": round((len(records) - 1) / duration_s, 2) if len(records) > 1 else 0.0,
        "sample_interval_ms": describe(sample_intervals),
        "seq": {"first": records[0].seq, "last": records[-1].seq},
        "yaw": {
            "first_deg": round(yaw_unwrapped[0], 4),
            "last_deg": round(yaw_unwrapped[-1], 4),
            "drift_deg": round(drift_deg, 4),
            "drift_rate_deg_min": round(drift_rate_deg_min, 4),
            "raw_drift_deg": round(raw_drift_deg, 4),
            "peak_to_peak_deg": round(max(yaw_unwrapped) - min(yaw_unwrapped), 4),
        },
        "tilt": {
            "roll_deg": describe(roll),
            "pitch_deg": describe(pitch),
            "roll_peak_to_peak_deg": round(max(roll) - min(roll), 4),
            "pitch_peak_to_peak_deg": round(max(pitch) - min(pitch), 4),
        },
        "accel": {
            "norm_mps2": describe(accel_norm),
            "linear_norm_mps2": describe(linear_accel_norm),
        },
        "gyro": {
            "z_deg_s": describe(gyro_z_deg_s),
            "z_bias_implied_deg_min": round(mean(gyro_z_deg_s) * 60.0, 4),
        },
        "accuracy": {
            "quat_code_counts": counts(quat_acc),
            "accel_code_counts": counts(accel_acc),
            "quat_rad_accuracy": describe(quat_rad_acc),
        },
        "correlation": {
            "yaw_vs_roll": round(correlation(yaw_unwrapped, roll), 4),
            "yaw_vs_pitch": round(correlation(yaw_unwrapped, pitch), 4),
            "yaw_vs_gyro_z": round(correlation(yaw_unwrapped, gyro_z_deg_s), 4),
        },
    }
    summary["assessment"] = assess(summary, stationary=stationary)
    return summary


def assess(summary: dict[str, Any], *, stationary: bool) -> list[str]:
    notes: list[str] = []
    yaw = summary["yaw"]
    tilt = summary["tilt"]
    gyro = summary["gyro"]
    accuracy = summary["accuracy"]
    corr = summary["correlation"]

    if stationary and abs(yaw["drift_rate_deg_min"]) > 5.0:
        notes.append("Stationary yaw drift is high. Treat this as a sensor-fusion/calibration issue until proven otherwise.")
    elif stationary and abs(yaw["drift_rate_deg_min"]) > 1.0:
        notes.append("Stationary yaw drift is measurable. It may be enough to hurt heading-hold over multi-second moves.")
    else:
        notes.append("Stationary yaw drift rate is small for short closed-loop moves.")

    if abs(gyro["z_bias_implied_deg_min"]) > 2.0:
        notes.append("Gyro Z mean implies yaw drift. Let the BNO085 calibrate while stationary after boot, and avoid commanding motion until gyro bias settles.")
    elif gyro["z_deg_s"]["max"] == 0.0 and gyro["z_deg_s"]["min"] == 0.0:
        notes.append("Gyro samples are all zero. The flashed firmware is probably not enabling the BNO085 gyro report yet, so gyro-bias diagnosis is incomplete.")

    quat_counts = accuracy["quat_code_counts"]
    if quat_counts.get("3", 0) < summary["sample_count"] * 0.8:
        notes.append("Quaternion accuracy is often below 3. Do the BNO085 calibration motion and keep motors/magnets/wires away from the IMU.")

    if abs(tilt["roll_deg"]["mean"]) > 3.0 or abs(tilt["pitch_deg"]["mean"]) > 3.0:
        notes.append("The IMU is mounted noticeably off-level. A fixed mount correction can improve robot-frame yaw consistency, but it will not fix true gyro/magnetometer drift.")

    if abs(corr["yaw_vs_roll"]) > 0.5 or abs(corr["yaw_vs_pitch"]) > 0.5:
        notes.append("Yaw is strongly correlated with tilt. That points to mounting/extrinsic compensation or vibration/acceleration coupling.")

    if summary["accel"]["linear_norm_mps2"]["max"] == 0.0:
        notes.append("Linear acceleration samples are all zero. Enable the BNO085 linear-accel report before using this as a vibration/settling check.")
    elif summary["accel"]["linear_norm_mps2"]["p95"] > 0.5 and stationary:
        notes.append("Linear acceleration is nonzero while supposedly stationary. Check vibration, table motion, or sensor fusion still settling.")

    return notes


def unwrap_degrees(values: list[float]) -> list[float]:
    if not values:
        return []
    unwrapped = [values[0]]
    for value in values[1:]:
        delta = wrap_degrees(value - unwrapped[-1])
        unwrapped.append(unwrapped[-1] + delta)
    return unwrapped


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, max(0, round(0.95 * (len(sorted_values) - 1))))
    return {
        "mean": round(mean(values), 6),
        "stdev": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(sorted_values[0], 6),
        "p50": round(statistics.median(sorted_values), 6),
        "p95": round(sorted_values[p95_index], 6),
        "max": round(sorted_values[-1], 6),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def counts(values: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    if left_var <= 1e-12 or right_var <= 1e-12:
        return 0.0
    return numerator / math.sqrt(left_var * right_var)


def write_csv(path: Path, records: list[ImuRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else [])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


if __name__ == "__main__":
    raise SystemExit(main())
