#!/usr/bin/env python3
"""Desktop Zenoh companion for the XIAO ESP32S3 stream firmware."""

from __future__ import annotations

import argparse
import json
import signal
import struct
import time
from pathlib import Path

import zenoh


VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
IMU_STRUCT = struct.Struct("<4sBBBBIQ14f")
SYNC_REQ_STRUCT = struct.Struct("<4sIQ")
SYNC_REPLY_STRUCT = struct.Struct("<4sIQQQ")


class RateStats:
    def __init__(self) -> None:
        self.video_count = 0
        self.imu_count = 0
        self.video_bytes = 0
        self.video_drops = 0
        self.imu_drops = 0
        self.video_latency_sum_ms = 0.0
        self.imu_latency_sum_ms = 0.0
        self.video_latency_count = 0
        self.imu_latency_count = 0
        self.last_video_seq: int | None = None
        self.last_imu_seq: int | None = None
        self.window_start_ns = time.monotonic_ns()

    def reset_window(self) -> None:
        self.video_count = 0
        self.imu_count = 0
        self.video_bytes = 0
        self.video_latency_sum_ms = 0.0
        self.imu_latency_sum_ms = 0.0
        self.video_latency_count = 0
        self.imu_latency_count = 0
        self.window_start_ns = time.monotonic_ns()

    def note_video(self, seq: int, byte_count: int, latency_ms: float | None) -> None:
        if self.last_video_seq is not None and seq > self.last_video_seq + 1:
            self.video_drops += seq - self.last_video_seq - 1
        self.last_video_seq = seq
        self.video_count += 1
        self.video_bytes += byte_count
        if latency_ms is not None:
            self.video_latency_sum_ms += latency_ms
            self.video_latency_count += 1

    def note_imu(self, seq: int, latency_ms: float | None) -> None:
        if self.last_imu_seq is not None and seq > self.last_imu_seq + 1:
            self.imu_drops += seq - self.last_imu_seq - 1
        self.last_imu_seq = seq
        self.imu_count += 1
        if latency_ms is not None:
            self.imu_latency_sum_ms += latency_ms
            self.imu_latency_count += 1


def build_config(mode: str, listen: str, connect: str | None) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps([listen]))
    if connect:
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    return config


def estimate_latency_ms(esp_us: int, recv_ns: int, offset_ns: float | None) -> float | None:
    if offset_ns is None:
        return None
    sample_pc_ns = esp_us * 1000 + offset_ns
    return (recv_ns - sample_pc_ns) / 1_000_000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="flatdisk/xiao")
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--sync-rate", type=float, default=1.0)
    parser.add_argument("--save-latest", type=Path, default=Path("captures/latest.jpg"))
    parser.add_argument("--save-all", action="store_true")
    args = parser.parse_args()

    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    args.save_latest.parent.mkdir(parents=True, exist_ok=True)

    session = zenoh.open(build_config(args.mode, args.listen, args.connect))
    video_sub = session.declare_subscriber(f"{args.namespace}/camera/jpeg")
    imu_sub = session.declare_subscriber(f"{args.namespace}/imu")
    sync_sub = session.declare_subscriber(f"{args.namespace}/time_sync")
    status_sub = session.declare_subscriber(f"{args.namespace}/status")

    endpoint = args.connect if args.connect else args.listen
    print(f"Zenoh companion mode={args.mode} endpoint={endpoint}, namespace={args.namespace}")
    print("Waiting for ESP32 samples...")

    stats = RateStats()
    offset_ns: float | None = None
    last_rtt_ms: float | None = None
    sync_seq = 0
    next_sync_ns = 0
    start_ns = time.monotonic_ns()
    next_report_ns = start_ns + 1_000_000_000
    sync_period_ns = int(1_000_000_000 / max(args.sync_rate, 0.1))

    while not stop:
        now_ns = time.monotonic_ns()
        if args.duration > 0 and (now_ns - start_ns) / 1_000_000_000.0 >= args.duration:
            break

        if now_ns >= next_sync_ns:
            payload = SYNC_REQ_STRUCT.pack(b"FDSQ", sync_seq, now_ns)
            session.put(f"{args.namespace}/cmd/time_sync", payload)
            sync_seq = (sync_seq + 1) & 0xFFFFFFFF
            next_sync_ns = now_ns + sync_period_ns

        while True:
            sample = video_sub.try_recv()
            if sample is None:
                break
            recv_ns = time.monotonic_ns()
            data = sample.payload.to_bytes()
            if len(data) < VIDEO_STRUCT.size:
                continue
            magic, version, fmt, width, height, header_len, seq, esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(data)
            if magic != b"FDV1" or version != 1 or header_len > len(data):
                continue
            jpeg = data[header_len:header_len + jpeg_len]
            if len(jpeg) != jpeg_len:
                continue
            args.save_latest.write_bytes(jpeg)
            if args.save_all:
                args.save_latest.parent.joinpath(f"frame_{seq:08d}.jpg").write_bytes(jpeg)
            latency_ms = estimate_latency_ms(esp_us, recv_ns, offset_ns)
            stats.note_video(seq, len(jpeg), latency_ms)

        while True:
            sample = imu_sub.try_recv()
            if sample is None:
                break
            recv_ns = time.monotonic_ns()
            data = sample.payload.to_bytes()
            if len(data) < IMU_STRUCT.size:
                continue
            unpacked = IMU_STRUCT.unpack_from(data)
            magic = unpacked[0]
            version = unpacked[1]
            seq = unpacked[5]
            esp_us = unpacked[6]
            if magic != b"FDI1" or version != 1:
                continue
            latency_ms = estimate_latency_ms(esp_us, recv_ns, offset_ns)
            stats.note_imu(seq, latency_ms)

        while True:
            sample = sync_sub.try_recv()
            if sample is None:
                break
            recv_ns = time.monotonic_ns()
            data = sample.payload.to_bytes()
            if len(data) < SYNC_REPLY_STRUCT.size:
                continue
            magic, _seq, pc_send_ns, _esp_rx_us, esp_tx_us = SYNC_REPLY_STRUCT.unpack_from(data)
            if magic != b"FDSR":
                continue
            midpoint_ns = (pc_send_ns + recv_ns) / 2.0
            measured_offset_ns = midpoint_ns - esp_tx_us * 1000.0
            offset_ns = measured_offset_ns if offset_ns is None else 0.9 * offset_ns + 0.1 * measured_offset_ns
            last_rtt_ms = (recv_ns - pc_send_ns) / 1_000_000.0

        while True:
            sample = status_sub.try_recv()
            if sample is None:
                break
            try:
                status = json.loads(sample.payload.to_bytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if status.get("video_errors") or status.get("imu_errors"):
                print(f"device status: {status}")

        now_ns = time.monotonic_ns()
        if now_ns >= next_report_ns:
            elapsed = max((now_ns - stats.window_start_ns) / 1_000_000_000.0, 1e-6)
            video_hz = stats.video_count / elapsed
            imu_hz = stats.imu_count / elapsed
            mbps = (stats.video_bytes * 8.0) / elapsed / 1_000_000.0
            video_lat = (stats.video_latency_sum_ms / stats.video_latency_count
                         if stats.video_latency_count else None)
            imu_lat = (stats.imu_latency_sum_ms / stats.imu_latency_count
                       if stats.imu_latency_count else None)
            offset_ms = offset_ns / 1_000_000.0 if offset_ns is not None else None
            print(
                "rate "
                f"video={video_hz:5.1f}Hz imu={imu_hz:5.1f}Hz "
                f"video_bw={mbps:5.2f}Mbps "
                f"drops(v/i)={stats.video_drops}/{stats.imu_drops} "
                f"lat_ms(v/i)={video_lat if video_lat is not None else float('nan'):6.2f}/"
                f"{imu_lat if imu_lat is not None else float('nan'):6.2f} "
                f"sync_offset_ms={offset_ms if offset_ms is not None else float('nan'):9.2f} "
                f"rtt_ms={last_rtt_ms if last_rtt_ms is not None else float('nan'):6.2f}"
            )
            stats.reset_window()
            next_report_ns = now_ns + 1_000_000_000

        time.sleep(0.001)

    video_sub.undeclare()
    imu_sub.undeclare()
    sync_sub.undeclare()
    status_sub.undeclare()
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
