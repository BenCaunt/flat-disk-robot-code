#!/usr/bin/env python3
"""Replay a recorded Rerun ACT log through a policy and compare motor outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_act import (
    ActPolicy,
    TemporalEnsembler,
    build_episode,
    choose_device,
    decode_actions,
    denormalize_model_actions,
    encode_actions,
    normalize_model_actions,
)


@dataclass
class ReplaySummary:
    rrd: str
    checkpoint: str
    frames: int
    duration_s: float
    output_rate_hz: float
    history_source: str
    temporal_ensembling: bool
    temporal_ensemble_decay: float
    action_representation: str
    normalize_actions: bool
    chunk_len: int
    image_history: int
    past_action_steps: int
    max_abs_output: float | None
    action_time_offset_s: float
    raw_mae_motor1_percent: float
    raw_mae_motor2_percent: float
    raw_rmse_motor1_percent: float
    raw_rmse_motor2_percent: float
    clamped_mae_motor1_percent: float
    clamped_mae_motor2_percent: float
    chunk0_mae_motor1_percent: float
    chunk0_mae_motor2_percent: float
    mean_actual_motor1_percent: float
    mean_actual_motor2_percent: float
    mean_chunk0_pred_motor1_percent: float
    mean_chunk0_pred_motor2_percent: float
    mean_raw_pred_motor1_percent: float
    mean_raw_pred_motor2_percent: float
    mean_clamped_pred_motor1_percent: float
    mean_clamped_pred_motor2_percent: float
    mean_actual_delta_motor1_minus_motor2: float
    mean_chunk0_pred_delta_motor1_minus_motor2: float
    mean_raw_pred_delta_motor1_minus_motor2: float
    mean_clamped_pred_delta_motor1_minus_motor2: float
    mean_chunk0_delta_error_percent: float
    mean_raw_delta_error_percent: float
    mean_clamped_delta_error_percent: float
    chunk0_delta_mae_percent: float
    raw_delta_mae_percent: float
    clamped_delta_mae_percent: float
    chunk0_delta_rmse_percent: float
    raw_delta_rmse_percent: float
    clamped_delta_rmse_percent: float


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


def decode_jpeg(jpeg: bytes, image_size: int, rotate_180: bool) -> torch.Tensor:
    image = Image.open(BytesIO(jpeg)).convert("RGB")
    if rotate_180:
        image = image.rotate(180)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def clamp_action(action: np.ndarray, max_abs_output: float | None) -> np.ndarray:
    if max_abs_output is None:
        return action.astype(np.float32, copy=True)
    clipped = np.clip(action, -max_abs_output, max_abs_output)
    return np.asarray([round(float(clipped[0])), round(float(clipped[1]))], dtype=np.float32)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def finite_metric(values: np.ndarray, fn: str) -> float:
    if values.size == 0:
        return float("nan")
    if fn == "mae":
        return float(np.mean(np.abs(values)))
    if fn == "rmse":
        return float(math.sqrt(float(np.mean(values * values))))
    if fn == "mean":
        return float(np.mean(values))
    raise ValueError(fn)


def summarize(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    rows: list[dict[str, float | int]],
) -> ReplaySummary:
    actual = np.asarray([[r["actual_motor1_percent"], r["actual_motor2_percent"]] for r in rows], dtype=np.float32)
    chunk0 = np.asarray([[r["chunk0_pred_motor1_percent"], r["chunk0_pred_motor2_percent"]] for r in rows], dtype=np.float32)
    raw = np.asarray([[r["raw_pred_motor1_percent"], r["raw_pred_motor2_percent"]] for r in rows], dtype=np.float32)
    clamped = np.asarray([[r["cmd_pred_motor1_percent"], r["cmd_pred_motor2_percent"]] for r in rows], dtype=np.float32)
    duration_s = float(rows[-1]["time_s"] - rows[0]["time_s"]) if len(rows) > 1 else 0.0
    output_rate_hz = (len(rows) - 1) / duration_s if duration_s > 0 else 0.0
    actual_delta = actual[:, 0] - actual[:, 1]
    chunk0_delta = chunk0[:, 0] - chunk0[:, 1]
    raw_delta = raw[:, 0] - raw[:, 1]
    clamped_delta = clamped[:, 0] - clamped[:, 1]

    return ReplaySummary(
        rrd=str(args.rrd),
        checkpoint=str(args.checkpoint),
        frames=len(rows),
        duration_s=duration_s,
        output_rate_hz=output_rate_hz,
        history_source=args.history_source,
        temporal_ensembling=bool(args.temporal_ensembling),
        temporal_ensemble_decay=float(args.temporal_ensemble_decay),
        action_representation=str(config.get("action_representation", "left_right")),
        normalize_actions=bool(config.get("normalize_actions", False)),
        chunk_len=int(config["chunk_len"]),
        image_history=int(config["image_history"]),
        past_action_steps=int(config["past_action_steps"]),
        max_abs_output=args.max_abs_output,
        action_time_offset_s=args.action_time_offset_s,
        raw_mae_motor1_percent=finite_metric(raw[:, 0] - actual[:, 0], "mae"),
        raw_mae_motor2_percent=finite_metric(raw[:, 1] - actual[:, 1], "mae"),
        raw_rmse_motor1_percent=finite_metric(raw[:, 0] - actual[:, 0], "rmse"),
        raw_rmse_motor2_percent=finite_metric(raw[:, 1] - actual[:, 1], "rmse"),
        clamped_mae_motor1_percent=finite_metric(clamped[:, 0] - actual[:, 0], "mae"),
        clamped_mae_motor2_percent=finite_metric(clamped[:, 1] - actual[:, 1], "mae"),
        chunk0_mae_motor1_percent=finite_metric(chunk0[:, 0] - actual[:, 0], "mae"),
        chunk0_mae_motor2_percent=finite_metric(chunk0[:, 1] - actual[:, 1], "mae"),
        mean_actual_motor1_percent=finite_metric(actual[:, 0], "mean"),
        mean_actual_motor2_percent=finite_metric(actual[:, 1], "mean"),
        mean_chunk0_pred_motor1_percent=finite_metric(chunk0[:, 0], "mean"),
        mean_chunk0_pred_motor2_percent=finite_metric(chunk0[:, 1], "mean"),
        mean_raw_pred_motor1_percent=finite_metric(raw[:, 0], "mean"),
        mean_raw_pred_motor2_percent=finite_metric(raw[:, 1], "mean"),
        mean_clamped_pred_motor1_percent=finite_metric(clamped[:, 0], "mean"),
        mean_clamped_pred_motor2_percent=finite_metric(clamped[:, 1], "mean"),
        mean_actual_delta_motor1_minus_motor2=finite_metric(actual_delta, "mean"),
        mean_chunk0_pred_delta_motor1_minus_motor2=finite_metric(chunk0_delta, "mean"),
        mean_raw_pred_delta_motor1_minus_motor2=finite_metric(raw_delta, "mean"),
        mean_clamped_pred_delta_motor1_minus_motor2=finite_metric(clamped_delta, "mean"),
        mean_chunk0_delta_error_percent=finite_metric(chunk0_delta - actual_delta, "mean"),
        mean_raw_delta_error_percent=finite_metric(raw_delta - actual_delta, "mean"),
        mean_clamped_delta_error_percent=finite_metric(clamped_delta - actual_delta, "mean"),
        chunk0_delta_mae_percent=finite_metric(chunk0_delta - actual_delta, "mae"),
        raw_delta_mae_percent=finite_metric(raw_delta - actual_delta, "mae"),
        clamped_delta_mae_percent=finite_metric(clamped_delta - actual_delta, "mae"),
        chunk0_delta_rmse_percent=finite_metric(chunk0_delta - actual_delta, "rmse"),
        raw_delta_rmse_percent=finite_metric(raw_delta - actual_delta, "rmse"),
        clamped_delta_rmse_percent=finite_metric(clamped_delta - actual_delta, "rmse"),
    )


def draw_plot(path: Path, rows: list[dict[str, float | int]], title: str) -> None:
    width, height = 1400, 850
    margin_l, margin_r, margin_t, margin_b = 85, 35, 70, 70
    gap = 70
    plot_h = (height - margin_t - margin_b - gap) // 2
    plot_w = width - margin_l - margin_r
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    times = np.asarray([float(r["time_s"]) for r in rows], dtype=np.float32)
    times = times - times[0]
    actual = np.asarray([[r["actual_motor1_percent"], r["actual_motor2_percent"]] for r in rows], dtype=np.float32)
    raw = np.asarray([[r["raw_pred_motor1_percent"], r["raw_pred_motor2_percent"]] for r in rows], dtype=np.float32)
    cmd = np.asarray([[r["cmd_pred_motor1_percent"], r["cmd_pred_motor2_percent"]] for r in rows], dtype=np.float32)
    y_min = float(np.floor(min(np.min(actual), np.min(raw), np.min(cmd), -1.0) / 5.0) * 5.0)
    y_max = float(np.ceil(max(np.max(actual), np.max(raw), np.max(cmd), 1.0) / 5.0) * 5.0)
    if y_max <= y_min:
        y_max = y_min + 1.0
    t_min, t_max = 0.0, float(max(times[-1], 1e-6))

    def xy(t: float, y: float, top: int) -> tuple[int, int]:
        x = margin_l + int((t - t_min) / (t_max - t_min) * plot_w)
        py = top + int((y_max - y) / (y_max - y_min) * plot_h)
        return x, py

    def polyline(series: np.ndarray, motor: int, top: int, color: str) -> None:
        points = [xy(float(t), float(y), top) for t, y in zip(times, series[:, motor], strict=True)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)

    draw.text((margin_l, 24), title, fill="black", font=font)
    legend = [
        ("actual", "#111111"),
        ("raw pred", "#1f77b4"),
        ("clamped cmd", "#d62728"),
    ]
    lx = margin_l
    for label, color in legend:
        draw.line((lx, 48, lx + 35, 48), fill=color, width=3)
        draw.text((lx + 42, 42), label, fill="black", font=font)
        lx += 150

    for motor, name in enumerate(("motor1", "motor2")):
        top = margin_t + motor * (plot_h + gap)
        bottom = top + plot_h
        draw.rectangle((margin_l, top, margin_l + plot_w, bottom), outline="#777777")
        draw.text((18, top + 8), f"{name} %", fill="black", font=font)
        for frac in np.linspace(0.0, 1.0, 5):
            y = y_min + frac * (y_max - y_min)
            _, py = xy(0.0, y, top)
            draw.line((margin_l, py, margin_l + plot_w, py), fill="#eeeeee")
            draw.text((35, py - 6), f"{y:.0f}", fill="#555555", font=font)
        for frac in np.linspace(0.0, 1.0, 6):
            t = frac * t_max
            px, _ = xy(t, y_min, top)
            draw.line((px, top, px, bottom), fill="#f2f2f2")
            draw.text((px - 14, bottom + 8), f"{t:.1f}", fill="#555555", font=font)
        polyline(actual, motor, top, "#111111")
        polyline(raw, motor, top, "#1f77b4")
        polyline(cmd, motor, top, "#d62728")

    draw.text((margin_l + plot_w // 2 - 30, height - 35), "time (s)", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_rerun(path: Path, rows: list[dict[str, float | int]], frame_jpegs: list[bytes]) -> None:
    import rerun as rr

    path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("flatdisk_act_replay", spawn=False)
    rr.save(path)
    try:
        t0 = float(rows[0]["time_s"])
        for row, jpeg in zip(rows, frame_jpegs, strict=True):
            rr.set_time("recording_time", duration=float(row["time_s"]) - t0)
            rr.log("camera/image", rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))
            for motor in (1, 2):
                rr.log(f"comparison/motor{motor}/actual_percent", rr.Scalars(row[f"actual_motor{motor}_percent"]))
                rr.log(f"comparison/motor{motor}/chunk0_pred_percent", rr.Scalars(row[f"chunk0_pred_motor{motor}_percent"]))
                rr.log(f"comparison/motor{motor}/raw_pred_percent", rr.Scalars(row[f"raw_pred_motor{motor}_percent"]))
                rr.log(f"comparison/motor{motor}/cmd_pred_percent", rr.Scalars(row[f"cmd_pred_motor{motor}_percent"]))
                rr.log(f"comparison/motor{motor}/chunk0_error_percent", rr.Scalars(row[f"chunk0_error_motor{motor}_percent"]))
                rr.log(f"comparison/motor{motor}/raw_error_percent", rr.Scalars(row[f"raw_error_motor{motor}_percent"]))
                rr.log(f"comparison/motor{motor}/cmd_error_percent", rr.Scalars(row[f"cmd_error_motor{motor}_percent"]))
            rr.log("comparison/delta/actual_m1_minus_m2", rr.Scalars(row["actual_delta_motor1_minus_motor2"]))
            rr.log("comparison/delta/chunk0_pred_m1_minus_m2", rr.Scalars(row["chunk0_pred_delta_motor1_minus_motor2"]))
            rr.log("comparison/delta/raw_pred_m1_minus_m2", rr.Scalars(row["raw_pred_delta_motor1_minus_motor2"]))
            rr.log("comparison/delta/cmd_pred_m1_minus_m2", rr.Scalars(row["cmd_pred_delta_motor1_minus_motor2"]))
            rr.log("comparison/delta/chunk0_error", rr.Scalars(row["chunk0_delta_error_percent"]))
            rr.log("comparison/delta/raw_error", rr.Scalars(row["raw_delta_error_percent"]))
            rr.log("comparison/delta/cmd_error", rr.Scalars(row["cmd_delta_error_percent"]))
    finally:
        rr.disconnect()


def replay(args: argparse.Namespace) -> tuple[ReplaySummary, Path, Path, Path | None]:
    device = choose_device(args.device)
    model, config = load_policy(args.checkpoint, device)
    action_scale = float(config.get("action_scale", 100.0))
    action_representation = str(config.get("action_representation", "left_right"))
    action_mean = np.asarray(config.get("action_mean", [0.0, 0.0]), dtype=np.float32)
    action_std = np.asarray(config.get("action_std", [1.0, 1.0]), dtype=np.float32)
    image_size = int(config["image_size"])
    image_history_len = int(config["image_history"])
    past_action_steps = int(config["past_action_steps"])
    chunk_len = int(config["chunk_len"])

    episode = build_episode(
        args.rrd,
        args.max_command_age_s,
        args.min_sample_interval_s,
        args.action_time_offset_s,
    )
    if episode is None or len(episode.frame_jpegs) == 0:
        raise RuntimeError(f"No aligned camera/command frames found in {args.rrd}")

    start = max(args.start_frame, 0)
    stop = len(episode.frame_jpegs) if args.max_frames <= 0 else min(len(episode.frame_jpegs), start + args.max_frames)
    if stop <= start:
        raise RuntimeError(f"Selected empty frame range: start={start} stop={stop}")

    image_history: deque[torch.Tensor] = deque(maxlen=image_history_len)
    action_history: deque[np.ndarray] = deque(
        [np.zeros(2, dtype=np.float32) for _ in range(past_action_steps)],
        maxlen=past_action_steps,
    )
    ensembler = TemporalEnsembler(chunk_len, action_dim=2, decay=float(args.temporal_ensemble_decay))
    rows: list[dict[str, float | int]] = []
    chunks: list[np.ndarray] = []

    for frame_index in range(start, stop):
        jpeg = episode.frame_jpegs[frame_index]
        image = decode_jpeg(jpeg, image_size, args.rotate_180)
        image_history.append(image)
        while len(image_history) < image_history_len:
            image_history.appendleft(image)

        images = torch.stack(list(image_history)).unsqueeze(0).to(device)
        past_motor_actions = np.stack(list(action_history), axis=0)
        past_actions_np = encode_actions(past_motor_actions, action_representation) / action_scale
        past_actions_np = normalize_model_actions(past_actions_np, action_mean, action_std)
        past_actions = torch.from_numpy(past_actions_np).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            chunk_norm = model(images, past_actions).squeeze(0).detach().cpu().numpy()
        chunk_model_percent = denormalize_model_actions(chunk_norm, action_mean, action_std) * action_scale
        chunk_motor_percent = decode_actions(chunk_model_percent, action_representation)
        chunks.append(chunk_motor_percent)

        if args.temporal_ensembling:
            action_norm = ensembler.add_prediction(chunk_norm)
        else:
            action_norm = chunk_norm[0]
        model_action = denormalize_model_actions(action_norm, action_mean, action_std) * action_scale
        raw_action = decode_actions(model_action, action_representation)
        chunk0_action = chunk_motor_percent[0]
        cmd_action = clamp_action(raw_action, args.max_abs_output)
        actual_action = episode.frame_actions[frame_index].astype(np.float32)

        if args.history_source == "demo":
            action_history.append(actual_action)
        elif args.history_source == "raw-predicted":
            action_history.append(raw_action.astype(np.float32))
        else:
            action_history.append(cmd_action.astype(np.float32))

        time_s = (int(episode.frame_times_ns[frame_index]) - int(episode.frame_times_ns[start])) / 1_000_000_000.0
        row: dict[str, float | int] = {
            "frame_index": frame_index,
            "time_s": time_s,
            "actual_motor1_percent": float(actual_action[0]),
            "actual_motor2_percent": float(actual_action[1]),
            "chunk0_pred_motor1_percent": float(chunk0_action[0]),
            "chunk0_pred_motor2_percent": float(chunk0_action[1]),
            "raw_pred_motor1_percent": float(raw_action[0]),
            "raw_pred_motor2_percent": float(raw_action[1]),
            "cmd_pred_motor1_percent": float(cmd_action[0]),
            "cmd_pred_motor2_percent": float(cmd_action[1]),
            "chunk0_error_motor1_percent": float(chunk0_action[0] - actual_action[0]),
            "chunk0_error_motor2_percent": float(chunk0_action[1] - actual_action[1]),
            "raw_error_motor1_percent": float(raw_action[0] - actual_action[0]),
            "raw_error_motor2_percent": float(raw_action[1] - actual_action[1]),
            "cmd_error_motor1_percent": float(cmd_action[0] - actual_action[0]),
            "cmd_error_motor2_percent": float(cmd_action[1] - actual_action[1]),
            "actual_delta_motor1_minus_motor2": float(actual_action[0] - actual_action[1]),
            "chunk0_pred_delta_motor1_minus_motor2": float(chunk0_action[0] - chunk0_action[1]),
            "raw_pred_delta_motor1_minus_motor2": float(raw_action[0] - raw_action[1]),
            "cmd_pred_delta_motor1_minus_motor2": float(cmd_action[0] - cmd_action[1]),
        }
        row["chunk0_delta_error_percent"] = (
            row["chunk0_pred_delta_motor1_minus_motor2"] - row["actual_delta_motor1_minus_motor2"]
        )
        row["raw_delta_error_percent"] = (
            row["raw_pred_delta_motor1_minus_motor2"] - row["actual_delta_motor1_minus_motor2"]
        )
        row["cmd_delta_error_percent"] = (
            row["cmd_pred_delta_motor1_minus_motor2"] - row["actual_delta_motor1_minus_motor2"]
        )
        rows.append(row)

    stem = args.output_stem or f"{args.rrd.stem}_{args.checkpoint.parent.name}_{args.history_source}"
    output_dir = args.output_dir
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    plot_path = output_dir / f"{stem}.png"
    chunk_path = output_dir / f"{stem}_chunks.npz"
    rerun_path = output_dir / f"{stem}.rrd" if args.write_rerun else None

    write_csv(csv_path, rows)
    np.savez_compressed(
        chunk_path,
        chunks=np.stack(chunks),
        chunks_motor_percent=np.stack(chunks),
        frame_indices=np.arange(start, stop),
        config=json.dumps(config),
    )
    summary = summarize(args=args, config=config, rows=rows)
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    draw_plot(plot_path, rows, f"ACT replay: {args.rrd.name} ({args.history_source} history)")
    if rerun_path is not None:
        write_rerun(rerun_path, rows, episode.frame_jpegs[start:stop])

    return summary, csv_path, plot_path, rerun_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rrd", type=Path, help="Recorded Rerun .rrd log to replay.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/act_vla_no_obstacle_fast_collect/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/act_replay"))
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--history-source", choices=("predicted", "raw-predicted", "demo"), default="predicted",
                        help="Past actions fed back to the model after each prediction.")
    parser.add_argument("--max-abs-output", type=float, default=10.0,
                        help="Clamp and round predicted output like live inference. Use --no-clamp to disable.")
    parser.add_argument("--no-clamp", dest="max_abs_output", action="store_const", const=None)
    parser.add_argument("--max-command-age-s", type=float, default=0.2)
    parser.add_argument("--min-sample-interval-s", type=float, default=0.0)
    parser.add_argument("--action-time-offset-s", type=float, default=0.0,
                        help="Pair each replay image at time t with recorded commands at t + offset.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--rotate-180", action="store_true",
                        help="Rotate replay images before inference. Recorded ACT camera frames normally already match training orientation.")
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument("--temporal-ensembling", dest="temporal_ensembling", action="store_true")
    temporal.add_argument("--no-temporal-ensembling", dest="temporal_ensembling", action="store_false")
    parser.set_defaults(temporal_ensembling=None)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=None)
    parser.add_argument("--write-rerun", action="store_true", help="Also write a Rerun .rrd comparison log.")
    args = parser.parse_args()

    if not args.rrd.exists():
        raise FileNotFoundError(f"Rerun log not found: {args.rrd}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint["config"]
    if args.temporal_ensembling is None:
        args.temporal_ensembling = bool(checkpoint_config.get("temporal_ensembling", True))
    if args.temporal_ensemble_decay is None:
        args.temporal_ensemble_decay = float(checkpoint_config.get("temporal_ensemble_decay", 0.01))

    summary, csv_path, plot_path, rerun_path = replay(args)
    print(json.dumps(asdict(summary), indent=2), flush=True)
    print(f"wrote CSV: {csv_path}", flush=True)
    print(f"wrote plot: {plot_path}", flush=True)
    if rerun_path is not None:
        print(f"wrote Rerun comparison: {rerun_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
