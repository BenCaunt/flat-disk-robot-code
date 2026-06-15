from __future__ import annotations

import sys
import time

import pytest

from flatdisk_sim.paths import REPO_ROOT


SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from flatdisk_robot_client import FlatDiskRobotClient, VideoFrame  # noqa: E402


def test_latest_frame_rejects_stale_cached_frame() -> None:
    client = FlatDiskRobotClient(connect="", frame_timeout_s=0.05)
    client.poll = lambda: None  # type: ignore[method-assign]
    client.last_frame = VideoFrame(
        seq=10,
        esp_us=0,
        width=1,
        height=1,
        jpeg=b"",
        received_ns=time.monotonic_ns() - 1_000_000_000,
    )

    with pytest.raises(TimeoutError, match="fresh camera frame"):
        client.latest_frame(timeout_s=0.01)


def test_latest_frame_accepts_fresh_cached_frame() -> None:
    client = FlatDiskRobotClient(connect="", frame_timeout_s=0.5)
    client.poll = lambda: None  # type: ignore[method-assign]
    frame = VideoFrame(
        seq=11,
        esp_us=0,
        width=1,
        height=1,
        jpeg=b"",
        received_ns=time.monotonic_ns(),
    )
    client.last_frame = frame

    assert client.latest_frame(timeout_s=0.01) is frame


def test_latest_frame_can_require_next_sequence() -> None:
    client = FlatDiskRobotClient(connect="", frame_timeout_s=0.5)
    frames = [
        VideoFrame(seq=20, esp_us=0, width=1, height=1, jpeg=b"", received_ns=time.monotonic_ns()),
        VideoFrame(seq=21, esp_us=0, width=1, height=1, jpeg=b"", received_ns=time.monotonic_ns()),
    ]

    def poll() -> None:
        if frames:
            frame = frames.pop(0)
            client.last_frame = VideoFrame(
                seq=frame.seq,
                esp_us=frame.esp_us,
                width=frame.width,
                height=frame.height,
                jpeg=frame.jpeg,
                received_ns=time.monotonic_ns(),
            )

    client.poll = poll  # type: ignore[method-assign]

    assert client.latest_frame(timeout_s=0.1, require_new=True).seq == 21
