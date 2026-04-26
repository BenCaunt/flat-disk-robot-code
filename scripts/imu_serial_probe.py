#!/usr/bin/env python3
"""Drive the IMU serial-debug firmware and print a compact transcript."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterable

import serial
from serial.tools import list_ports


DEFAULT_COMMANDS = ("direct",)
XIAO_USB_VID = 0x303A
XIAO_USB_PID = 0x1001


def find_port() -> str:
    candidates = []
    for port in list_ports.comports():
        if port.vid == XIAO_USB_VID and port.pid == XIAO_USB_PID:
            candidates.append(port.device)

    if candidates:
        return candidates[0]

    for port in list_ports.comports():
        if "usbmodem" in port.device:
            return port.device

    raise RuntimeError("No XIAO ESP32S3 serial port found")


def read_available(ser: serial.Serial, duration_s: float) -> str:
    deadline = time.monotonic() + duration_s
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunks.append(ser.read(waiting))
            deadline = max(deadline, time.monotonic() + 0.15)
        else:
            time.sleep(0.03)
    return b"".join(chunks).decode("utf-8", errors="replace")


def read_until_marker(ser: serial.Serial, marker: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    chunks: list[bytes] = []
    text = ""
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            chunk = ser.read(waiting)
            chunks.append(chunk)
            text += chunk.decode("utf-8", errors="replace")
            if marker in text:
                break
        else:
            time.sleep(0.02)

    return b"".join(chunks).decode("utf-8", errors="replace")


def run_commands(port: str, baud: int, commands: Iterable[str], boot_wait_s: float, command_wait_s: float) -> str:
    transcript: list[str] = []
    with serial.Serial(port, baudrate=baud, timeout=0.05, exclusive=True) as ser:
        # Native USB CDC on the ESP32-S3 only starts emitting reliably once the
        # host asserts the terminal line state.
        ser.dtr = True
        ser.rts = False
        time.sleep(0.2)
        transcript.append(read_available(ser, boot_wait_s))
        ser.reset_input_buffer()

        for command in commands:
            transcript.append(f"\n>>> {command}\n")
            ser.write((command + "\n").encode("ascii"))
            ser.flush()
            marker = f"<<<END {command}>>>"
            output = read_until_marker(ser, marker, command_wait_s)
            if marker not in output:
                output += f"\n# timed out waiting for {marker}\n"
            transcript.append(output)

    return "".join(transcript)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="Serial port. Defaults to auto-detecting the XIAO.")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--boot-wait", type=float, default=3.0)
    parser.add_argument("--command-wait", type=float, default=18.0)
    parser.add_argument("commands", nargs="*", default=list(DEFAULT_COMMANDS))
    args = parser.parse_args()

    port = args.port or find_port()
    print(f"# port={port} baud={args.baud}", flush=True)
    try:
        transcript = run_commands(port, args.baud, args.commands, args.boot_wait, args.command_wait)
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 2

    print(transcript)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
