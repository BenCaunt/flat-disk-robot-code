#!/usr/bin/env python3
"""Small Tkinter Zenoh motor-control GUI for the flat disk robot."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
import struct
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk
import zenoh

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:  # Keep the GUI usable when gamepad support is not installed.
    pygame = None


VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
VIDEO_PREVIEW_SIZE = (320, 240)
JOYSTICK_SIZE = 150
JOYSTICK_RADIUS = 58
JOYSTICK_CENTER = JOYSTICK_SIZE // 2
JOYSTICK_COMMAND_LIMIT = 10.0
GAMEPAD_POLL_MS = 30
GAMEPAD_DEADZONE = 0.12
GAMEPAD_AXIS_X = 2
GAMEPAD_AXIS_Y = 1
GAMEPAD_STOP_BUTTONS = {1}
ARROW_KEY_NAMES = {"Up", "Down", "Left", "Right"}


def build_config(mode: str, listen: str, connect: str) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps([listen]))
    if connect:
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    return config


class MotorGui(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.title("Flat Disk Motor Control")
        self.minsize(560, 430)

        self.session: zenoh.Session | None = None
        self.video_sub: Any | None = None
        self.status_sub: Any | None = None
        self.command_after_id: str | None = None
        self.poll_after_id: str | None = None
        self.gamepad_after_id: str | None = None
        self.gamepad: Any | None = None
        self.video_photo: ImageTk.PhotoImage | None = None
        self.video_frames = 0
        self.video_last_report_ns = time.monotonic_ns()
        self.video_last_seq: int | None = None
        self.active_arrow_keys: set[str] = set()
        self.gamepad_has_control = False
        self.gamepad_button_states: set[int] = set()

        self.namespace_var = tk.StringVar(value=args.namespace)
        self.mode_var = tk.StringVar(value=args.mode)
        self.connect_var = tk.StringVar(value=args.connect)
        self.listen_var = tk.StringVar(value=args.listen)
        self.rate_var = tk.DoubleVar(value=args.rate)
        self.command_mode_var = tk.StringVar(value="percent")
        self.armed_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Disconnected")
        self.device_status_var = tk.StringVar(value="No device status yet")
        self.video_status_var = tk.StringVar(value="No video frames yet")
        self.gamepad_status_var = tk.StringVar(value="No gamepad")
        self.m1_var = tk.DoubleVar(value=0)
        self.m2_var = tk.DoubleVar(value=0)
        self.m1_value_var = tk.StringVar(value="0")
        self.m2_value_var = tk.StringVar(value="0")

        self._build_ui()
        self._sync_slider_labels()
        self._bind_keyboard_controls()
        self._init_gamepad()
        self._schedule_gamepad_poll()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        connection = ttk.LabelFrame(outer, text="Zenoh", padding=10)
        connection.grid(row=0, column=0, sticky="ew")
        connection.columnconfigure(1, weight=1)
        connection.columnconfigure(3, weight=1)

        ttk.Label(connection, text="Namespace").grid(row=0, column=0, sticky="w")
        ttk.Entry(connection, textvariable=self.namespace_var).grid(row=0, column=1, sticky="ew", padx=(6, 12))
        ttk.Label(connection, text="Mode").grid(row=0, column=2, sticky="w")
        ttk.Combobox(connection, textvariable=self.mode_var, values=("client", "peer"),
                     width=8, state="readonly").grid(row=0, column=3, sticky="w", padx=(6, 0))

        ttk.Label(connection, text="Connect").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(connection, textvariable=self.connect_var).grid(row=1, column=1, sticky="ew", padx=(6, 12), pady=(8, 0))
        ttk.Label(connection, text="Listen").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(connection, textvariable=self.listen_var).grid(row=1, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))

        ttk.Button(connection, text="Connect", command=self.connect).grid(row=2, column=0, pady=(10, 0), sticky="ew")
        ttk.Button(connection, text="Disconnect", command=self.disconnect).grid(row=2, column=1, pady=(10, 0), sticky="w")
        ttk.Label(connection, textvariable=self.status_var).grid(row=2, column=2, columnspan=2, pady=(10, 0), sticky="w")

        controls = ttk.LabelFrame(outer, text="Motors", padding=10)
        controls.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(3, minsize=JOYSTICK_SIZE + 20)

        mode_row = ttk.Frame(controls)
        mode_row.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(mode_row, text="Percent", variable=self.command_mode_var, value="percent",
                        command=self._set_slider_mode).grid(row=0, column=0, padx=(0, 12))
        ttk.Radiobutton(mode_row, text="Pulse us", variable=self.command_mode_var, value="us",
                        command=self._set_slider_mode).grid(row=0, column=1)
        ttk.Label(mode_row, text="Rate Hz").grid(row=0, column=2, padx=(22, 6))
        ttk.Spinbox(mode_row, textvariable=self.rate_var, from_=1, to=50, increment=1, width=6).grid(row=0, column=3)

        ttk.Label(controls, text="Motor 1").grid(row=1, column=0, sticky="w", pady=(14, 0))
        self.m1_scale = ttk.Scale(controls, from_=-100, to=100, variable=self.m1_var,
                                  command=lambda _value: self._sync_slider_labels())
        self.m1_scale.grid(row=1, column=1, sticky="ew", padx=10, pady=(14, 0))
        ttk.Label(controls, textvariable=self.m1_value_var, width=8).grid(row=1, column=2, sticky="e", pady=(14, 0))

        ttk.Label(controls, text="Motor 2").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.m2_scale = ttk.Scale(controls, from_=-100, to=100, variable=self.m2_var,
                                  command=lambda _value: self._sync_slider_labels())
        self.m2_scale.grid(row=2, column=1, sticky="ew", padx=10, pady=(10, 0))
        ttk.Label(controls, textvariable=self.m2_value_var, width=8).grid(row=2, column=2, sticky="e", pady=(10, 0))

        buttons = ttk.Frame(controls)
        buttons.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        ttk.Button(buttons, text="Neutral", command=self.neutral).grid(row=0, column=0, padx=(0, 8))
        self.arm_button = ttk.Button(buttons, text="Arm", command=self.toggle_arm)
        self.arm_button.grid(row=0, column=1, padx=(0, 8))
        tk.Button(buttons, text="STOP", command=self.stop_motors, bg="#b00020", fg="white",
                  activebackground="#7f0017", activeforeground="white", padx=18).grid(row=0, column=2)

        joystick = ttk.Frame(controls)
        joystick.grid(row=0, column=3, rowspan=4, sticky="n", padx=(18, 0))
        ttk.Label(joystick, text="Joystick").grid(row=0, column=0)
        self.joystick_canvas = tk.Canvas(joystick, width=JOYSTICK_SIZE, height=JOYSTICK_SIZE,
                                         bg="#f2f2f2", highlightthickness=1, highlightbackground="#999999")
        self.joystick_canvas.grid(row=1, column=0, pady=(6, 0))
        ttk.Label(joystick, textvariable=self.gamepad_status_var, wraplength=JOYSTICK_SIZE).grid(
            row=2, column=0, sticky="ew", pady=(6, 0)
        )
        self.joystick_canvas.bind("<ButtonPress-1>", self._joystick_drag)
        self.joystick_canvas.bind("<B1-Motion>", self._joystick_drag)
        self.joystick_canvas.bind("<ButtonRelease-1>", self._joystick_release)
        self._draw_joystick(0, 0)

        video = ttk.LabelFrame(outer, text="Camera", padding=10)
        video.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        video.columnconfigure(0, weight=1)
        self.video_label = tk.Label(video, width=VIDEO_PREVIEW_SIZE[0], height=VIDEO_PREVIEW_SIZE[1],
                                    bg="#111111", fg="#dddddd", text="Waiting for video")
        self.video_label.grid(row=0, column=0)
        ttk.Label(video, textvariable=self.video_status_var).grid(row=1, column=0, sticky="w", pady=(6, 0))

        telemetry = ttk.LabelFrame(outer, text="Telemetry", padding=10)
        telemetry.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        telemetry.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        ttk.Label(telemetry, textvariable=self.device_status_var, justify="left").grid(row=0, column=0, sticky="ew")

    def _set_slider_mode(self) -> None:
        if self.command_mode_var.get() == "us":
            self.m1_scale.configure(from_=1000, to=2000)
            self.m2_scale.configure(from_=1000, to=2000)
        else:
            self._configure_percent_sliders()
        self.neutral()
        self._sync_slider_labels()

    def _configure_percent_sliders(self) -> None:
        self.m1_scale.configure(from_=-100, to=100)
        self.m2_scale.configure(from_=-100, to=100)

    def _sync_slider_labels(self) -> None:
        self.m1_value_var.set(str(int(round(self.m1_var.get()))))
        self.m2_value_var.set(str(int(round(self.m2_var.get()))))

    def _draw_joystick(self, x_norm: float, y_norm: float) -> None:
        canvas = self.joystick_canvas
        canvas.delete("all")
        c = JOYSTICK_CENTER
        r = JOYSTICK_RADIUS
        canvas.create_oval(c - r, c - r, c + r, c + r, outline="#777777", width=2)
        canvas.create_line(c, c - r, c, c + r, fill="#bbbbbb")
        canvas.create_line(c - r, c, c + r, c, fill="#bbbbbb")
        knob_x = c + x_norm * r
        knob_y = c + y_norm * r
        canvas.create_line(c, c, knob_x, knob_y, fill="#4b6f9c", width=2)
        canvas.create_oval(knob_x - 13, knob_y - 13, knob_x + 13, knob_y + 13,
                           fill="#2f6fbd", outline="#1c4f8d", width=2)

    def _center_joystick(self) -> None:
        self._draw_joystick(0, 0)

    def _bind_keyboard_controls(self) -> None:
        for key in ARROW_KEY_NAMES:
            self.bind_all(f"<KeyPress-{key}>", self._arrow_key_press, add="+")
            self.bind_all(f"<KeyRelease-{key}>", self._arrow_key_release, add="+")

    def _init_gamepad(self) -> None:
        if pygame is None:
            self.gamepad_status_var.set("Install pygame for gamepad")
            return
        try:
            pygame.init()
            pygame.joystick.init()
        except pygame.error as exc:
            self.gamepad_status_var.set(f"Gamepad init failed: {exc}")
            return
        self._refresh_gamepad()

    def _refresh_gamepad(self) -> None:
        if pygame is None:
            return
        try:
            count = pygame.joystick.get_count()
            if count <= 0:
                self.gamepad = None
                self.gamepad_has_control = False
                self.gamepad_button_states.clear()
                self.gamepad_status_var.set("No gamepad")
                return
            if self.gamepad is None or not self.gamepad.get_init():
                self.gamepad = pygame.joystick.Joystick(0)
                self.gamepad.init()
            self.gamepad_status_var.set(f"Gamepad: {self.gamepad.get_name()}")
        except pygame.error as exc:
            self.gamepad = None
            self.gamepad_status_var.set(f"Gamepad error: {exc}")

    def _schedule_gamepad_poll(self) -> None:
        self.gamepad_after_id = self.after(GAMEPAD_POLL_MS, self._gamepad_tick)

    def _gamepad_tick(self) -> None:
        try:
            self._poll_gamepad()
        finally:
            if self.winfo_exists():
                self._schedule_gamepad_poll()

    def _poll_gamepad(self) -> None:
        if pygame is None:
            return
        try:
            pygame.event.pump()
        except pygame.error:
            self._refresh_gamepad()
            return

        if self.gamepad is None:
            self._refresh_gamepad()
            return

        try:
            if not self.gamepad.get_init():
                self._refresh_gamepad()
                return
            x_norm = self._gamepad_axis(GAMEPAD_AXIS_X)
            y_norm = self._gamepad_axis(GAMEPAD_AXIS_Y)
            self._handle_gamepad_buttons()
        except pygame.error as exc:
            self.gamepad = None
            self.gamepad_has_control = False
            self.gamepad_status_var.set(f"Gamepad error: {exc}")
            return

        x_norm = self._apply_gamepad_deadzone(x_norm)
        y_norm = self._apply_gamepad_deadzone(y_norm)
        distance = (x_norm * x_norm + y_norm * y_norm) ** 0.5
        if distance > 1.0:
            x_norm /= distance
            y_norm /= distance

        is_active = abs(x_norm) > 0.0 or abs(y_norm) > 0.0
        if not is_active and not self.gamepad_has_control:
            return
        self.gamepad_has_control = is_active
        self._set_percent_command_from_axes(x_norm, y_norm)

    def _gamepad_axis(self, axis_index: int) -> float:
        if self.gamepad is None or axis_index >= self.gamepad.get_numaxes():
            return 0.0
        return float(self.gamepad.get_axis(axis_index))

    def _apply_gamepad_deadzone(self, value: float) -> float:
        if abs(value) < GAMEPAD_DEADZONE:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * min((abs(value) - GAMEPAD_DEADZONE) / (1.0 - GAMEPAD_DEADZONE), 1.0)

    def _handle_gamepad_buttons(self) -> None:
        if self.gamepad is None:
            return
        pressed: set[int] = set()
        for button in range(self.gamepad.get_numbuttons()):
            if self.gamepad.get_button(button):
                pressed.add(button)

        newly_pressed = pressed - self.gamepad_button_states
        if newly_pressed & GAMEPAD_STOP_BUTTONS:
            self.stop_motors()
        self.gamepad_button_states = pressed

    def _arrow_key_press(self, event: tk.Event) -> str | None:
        key = event.keysym
        if key not in ARROW_KEY_NAMES:
            return None
        if self._keyboard_focus_is_text_input():
            return None
        if key not in self.active_arrow_keys:
            self.active_arrow_keys.add(key)
            self._apply_arrow_keys()
        return "break"

    def _arrow_key_release(self, event: tk.Event) -> str | None:
        key = event.keysym
        if key not in ARROW_KEY_NAMES:
            return None
        was_active = key in self.active_arrow_keys
        if not was_active and self._keyboard_focus_is_text_input():
            return None
        if was_active:
            self.active_arrow_keys.remove(key)
            self._apply_arrow_keys()
        return "break"

    def _keyboard_focus_is_text_input(self) -> bool:
        focus = self.focus_get()
        if focus is None:
            return False
        return focus.winfo_class() in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox"}

    def _apply_arrow_keys(self) -> None:
        if self.command_mode_var.get() != "percent":
            self.command_mode_var.set("percent")
            self._configure_percent_sliders()

        x_norm = 0.0
        y_norm = 0.0
        if "Left" in self.active_arrow_keys:
            x_norm -= 1.0
        if "Right" in self.active_arrow_keys:
            x_norm += 1.0
        if "Up" in self.active_arrow_keys:
            y_norm -= 1.0
        if "Down" in self.active_arrow_keys:
            y_norm += 1.0

        distance = (x_norm * x_norm + y_norm * y_norm) ** 0.5
        if distance > 1.0:
            x_norm /= distance
            y_norm /= distance

        self._set_percent_command_from_axes(x_norm, y_norm)

    def _set_percent_command_from_axes(self, x_norm: float, y_norm: float) -> None:
        if self.command_mode_var.get() != "percent":
            self.command_mode_var.set("percent")
            self._configure_percent_sliders()

        forward = -y_norm * JOYSTICK_COMMAND_LIMIT
        turn = x_norm * JOYSTICK_COMMAND_LIMIT
        motor1 = max(-JOYSTICK_COMMAND_LIMIT, min(JOYSTICK_COMMAND_LIMIT, forward + turn))
        motor2 = max(-JOYSTICK_COMMAND_LIMIT, min(JOYSTICK_COMMAND_LIMIT, forward - turn))

        self.m1_var.set(motor1)
        self.m2_var.set(motor2)
        self._sync_slider_labels()
        self._draw_joystick(x_norm, y_norm)
        if self.armed_var.get():
            self._send_motor_command()

    def _joystick_drag(self, event: tk.Event) -> None:
        if self.command_mode_var.get() != "percent":
            self.command_mode_var.set("percent")
            self._configure_percent_sliders()

        dx = float(event.x - JOYSTICK_CENTER)
        dy = float(event.y - JOYSTICK_CENTER)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > JOYSTICK_RADIUS:
            scale = JOYSTICK_RADIUS / distance
            dx *= scale
            dy *= scale

        x_norm = dx / JOYSTICK_RADIUS
        y_norm = dy / JOYSTICK_RADIUS
        self._set_percent_command_from_axes(x_norm, y_norm)

    def _joystick_release(self, _event: tk.Event) -> None:
        self.neutral()
        if self.armed_var.get():
            self._send_motor_command()

    def connect(self) -> None:
        if self.session is not None:
            return
        try:
            self.session = zenoh.open(build_config(self.mode_var.get(), self.listen_var.get(), self.connect_var.get()))
            namespace = self.namespace_var.get().strip()
            self.video_sub = self.session.declare_subscriber(f"{namespace}/camera/jpeg")
            self.status_sub = self.session.declare_subscriber(f"{namespace}/status")
        except Exception as exc:
            self.session = None
            messagebox.showerror("Zenoh connection failed", str(exc))
            self.status_var.set("Disconnected")
            return

        self.status_var.set("Connected")
        self._schedule_poll()

    def disconnect(self) -> None:
        self.disarm(send_stop=True)
        if self.poll_after_id is not None:
            self.after_cancel(self.poll_after_id)
            self.poll_after_id = None
        for sub in (self.video_sub, self.status_sub):
            if sub is not None:
                try:
                    sub.undeclare()
                except Exception:
                    pass
        self.video_sub = None
        self.status_sub = None
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
        self.session = None
        self.status_var.set("Disconnected")
        self.video_label.configure(image="", text="Waiting for video")
        self.video_photo = None

    def toggle_arm(self) -> None:
        if self.armed_var.get():
            self.disarm(send_stop=True)
            return
        if self.session is None:
            self.connect()
        if self.session is None:
            return
        self.armed_var.set(True)
        self.arm_button.configure(text="Disarm")
        self._send_motor_command()
        self._schedule_command()

    def disarm(self, send_stop: bool) -> None:
        if self.command_after_id is not None:
            self.after_cancel(self.command_after_id)
            self.command_after_id = None
        self.armed_var.set(False)
        self.arm_button.configure(text="Arm")
        if send_stop:
            self._publish_stop()

    def neutral(self) -> None:
        if self.command_mode_var.get() == "us":
            self.m1_var.set(1500)
            self.m2_var.set(1500)
        else:
            self.m1_var.set(0)
            self.m2_var.set(0)
        self._sync_slider_labels()
        self._center_joystick()

    def stop_motors(self) -> None:
        self.neutral()
        self.disarm(send_stop=True)

    def _schedule_command(self) -> None:
        if not self.armed_var.get():
            return
        period_ms = int(1000 / max(self.rate_var.get(), 1.0))
        self.command_after_id = self.after(max(period_ms, 20), self._command_tick)

    def _command_tick(self) -> None:
        if self.armed_var.get():
            self._send_motor_command()
            self._schedule_command()

    def _send_motor_command(self) -> None:
        if self.session is None:
            return
        namespace = self.namespace_var.get().strip()
        m1 = int(round(self.m1_var.get()))
        m2 = int(round(self.m2_var.get()))
        try:
            if self.command_mode_var.get() == "us":
                payload = json.dumps({"m1_us": m1, "m2_us": m2}).encode("utf-8")
                self.session.put(f"{namespace}/cmd/motors/us", payload)
            else:
                payload = json.dumps({"m1": m1, "m2": m2}).encode("utf-8")
                self.session.put(f"{namespace}/cmd/motors/percent", payload)
        except Exception as exc:
            self.status_var.set(f"Send failed: {exc}")
            self.disarm(send_stop=False)

    def _publish_stop(self) -> None:
        if self.session is None:
            return
        try:
            self.session.put(f"{self.namespace_var.get().strip()}/cmd/motors/stop", b"stop")
        except Exception as exc:
            self.status_var.set(f"Stop failed: {exc}")

    def _schedule_poll(self) -> None:
        self.poll_after_id = self.after(100, self._poll_samples)

    def _poll_samples(self) -> None:
        self._poll_video()
        self._poll_status()
        if self.session is not None:
            self._schedule_poll()

    def _poll_video(self) -> None:
        if self.video_sub is None:
            return
        latest: tuple[int, int, int, bytes] | None = None
        while True:
            sample = self.video_sub.try_recv()
            if sample is None:
                break
            data = sample.payload.to_bytes()
            parsed = self._parse_video_sample(data)
            if parsed is not None:
                latest = parsed
        if latest is None:
            return
        seq, width, height, jpeg = latest
        try:
            self._render_jpeg(jpeg)
        except Exception as exc:
            self.video_status_var.set(f"Video decode failed: {exc}")
            return

        now_ns = time.monotonic_ns()
        self.video_frames += 1
        elapsed = max((now_ns - self.video_last_report_ns) / 1_000_000_000.0, 1e-6)
        hz = self.video_frames / elapsed
        drops = 0
        if self.video_last_seq is not None and seq > self.video_last_seq + 1:
            drops = seq - self.video_last_seq - 1
        self.video_last_seq = seq
        if elapsed >= 1.0:
            self.video_status_var.set(f"seq {seq}  {width}x{height}  {hz:.1f} Hz  drops +{drops}")
            self.video_frames = 0
            self.video_last_report_ns = now_ns

    def _parse_video_sample(self, data: bytes) -> tuple[int, int, int, bytes] | None:
        if len(data) < VIDEO_STRUCT.size:
            return None
        magic, version, _fmt, width, height, header_len, seq, _esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(data)
        if magic != b"FDV1" or version != 1 or header_len > len(data):
            return None
        jpeg = data[header_len:header_len + jpeg_len]
        if len(jpeg) != jpeg_len:
            return None
        return seq, width, height, jpeg

    def _render_jpeg(self, jpeg: bytes) -> None:
        image = Image.open(BytesIO(jpeg)).convert("RGB")
        image = image.rotate(180)
        image.thumbnail(VIDEO_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", VIDEO_PREVIEW_SIZE, "#111111")
        offset = ((VIDEO_PREVIEW_SIZE[0] - image.width) // 2, (VIDEO_PREVIEW_SIZE[1] - image.height) // 2)
        canvas.paste(image, offset)
        self.video_photo = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=self.video_photo, text="")

    def _poll_status(self) -> None:
        if self.status_sub is None:
            return
        latest: dict[str, Any] | None = None
        while True:
            sample = self.status_sub.try_recv()
            if sample is None:
                break
            try:
                latest = json.loads(sample.payload.to_bytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if latest is None:
            return
        self.device_status_var.set(
            "Status: "
            f"motors {latest.get('motor1_percent', '?')}/{latest.get('motor2_percent', '?')}% "
            f"us {latest.get('motor1_us', '?')}/{latest.get('motor2_us', '?')}  "
            f"rssi {latest.get('rssi', '?')}  "
            f"cmd {latest.get('motor_commands', '?')} err {latest.get('motor_command_errors', '?')}"
        )

    def close(self) -> None:
        if self.gamepad_after_id is not None:
            self.after_cancel(self.gamepad_after_id)
            self.gamepad_after_id = None
        if pygame is not None:
            try:
                pygame.joystick.quit()
            except pygame.error:
                pass
        self.disconnect()
        self.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="flatdisk/xiao")
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args()

    app = MotorGui(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
