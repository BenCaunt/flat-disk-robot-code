#!/usr/bin/env python3
"""Rerun blueprint for Flatdisk robot captures."""

from __future__ import annotations

import argparse
from pathlib import Path

import rerun.blueprint as rrb


APPLICATION_ID = "flatdisk_xiao_zenoh"


def build_flatdisk_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(
                    origin="/camera",
                    contents="/camera/image",
                    name="Camera",
                ),
                rrb.TextLogView(
                    origin="/status",
                    contents="/status/json",
                    name="Status JSON",
                ),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/commands",
                    contents="$origin/**",
                    name="Motor Commands",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/telemetry",
                    contents="$origin/**",
                    name="Motor Telemetry",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                column_shares=[1, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    origin="/imu",
                    contents=[
                        "/imu/accel_mps2/**",
                        "/imu/gyro_radps/**",
                        "/imu/linear_accel_mps2/**",
                    ],
                    name="IMU",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/timing",
                    contents="$origin/**",
                    name="Timing",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    origin="/status",
                    contents=[
                        "/status/rssi",
                        "/status/video_errors",
                        "/status/imu_errors",
                        "/status/motor_command_errors",
                    ],
                    name="Status Scalars",
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                column_shares=[2, 1, 1],
            ),
            row_shares=[3, 2, 2],
        ),
        collapse_panels=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures/recovered/flatdisk_xiao_zenoh.rbl"),
        help="Blueprint output path.",
    )
    parser.add_argument("--application-id", default=APPLICATION_ID)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_flatdisk_blueprint().save(args.application_id, args.output)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
