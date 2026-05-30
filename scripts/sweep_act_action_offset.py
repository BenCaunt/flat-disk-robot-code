#!/usr/bin/env python3
"""Sweep image-to-action time offsets for an ACT replay checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_replay(args: argparse.Namespace, offset_s: float, output_dir: Path) -> dict[str, float | int | str | bool | None]:
    stem = f"offset_{offset_s:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("replay_act_log.py")),
        str(args.rrd),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(output_dir),
        "--output-stem",
        stem,
        "--action-time-offset-s",
        str(offset_s),
        "--max-command-age-s",
        str(args.max_command_age_s),
        "--device",
        args.device,
    ]
    if args.no_temporal_ensembling:
        cmd.append("--no-temporal-ensembling")
    if args.history_source != "predicted":
        cmd.extend(["--history-source", args.history_source])
    if args.max_frames > 0:
        cmd.extend(["--max-frames", str(args.max_frames)])

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    summary_path = output_dir / f"{stem}_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["offset_s"] = offset_s
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str | bool | None]]) -> None:
    if not rows:
        return
    keys = [
        "offset_s",
        "frames",
        "duration_s",
        "output_rate_hz",
        "history_source",
        "temporal_ensembling",
        "action_representation",
        "normalize_actions",
        "raw_mae_motor1_percent",
        "raw_mae_motor2_percent",
        "raw_delta_mae_percent",
        "raw_delta_rmse_percent",
        "mean_raw_delta_error_percent",
        "clamped_delta_mae_percent",
        "mean_clamped_delta_error_percent",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rrd", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/act_vla_no_obstacle_fast_collect_norm_aug/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/act_replay/action_offset_sweep.csv"))
    parser.add_argument("--offset-start-s", type=float, default=-0.5)
    parser.add_argument("--offset-stop-s", type=float, default=0.5)
    parser.add_argument("--offset-step-s", type=float, default=0.05)
    parser.add_argument("--history-source", choices=("predicted", "raw-predicted", "demo"), default="demo")
    parser.add_argument("--no-temporal-ensembling", action="store_true")
    parser.add_argument("--max-command-age-s", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    offsets: list[float] = []
    value = args.offset_start_s
    while value <= args.offset_stop_s + args.offset_step_s * 0.5:
        offsets.append(round(value, 6))
        value += args.offset_step_s

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="act_offset_sweep_") as tmp:
        tmpdir = Path(tmp)
        rows = [run_replay(args, offset, tmpdir) for offset in offsets]

    rows.sort(key=lambda row: float(row["raw_delta_mae_percent"]))
    write_csv(args.output, rows)
    best = rows[0]
    print(
        "best "
        f"offset_s={best['offset_s']:.3f} "
        f"raw_delta_mae={best['raw_delta_mae_percent']:.3f}% "
        f"mean_raw_delta_error={best['mean_raw_delta_error_percent']:.3f}% "
        f"frames={best['frames']}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
