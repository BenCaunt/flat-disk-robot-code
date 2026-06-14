"""Shared filesystem paths for simulator assets and scratch output."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = PACKAGE_ROOT
REPO_ROOT = SIM_ROOT.parent
ASSETS_ROOT = SIM_ROOT / "assets"
ROBOT_URDF_DIR = ASSETS_ROOT / "robot" / "flat-disk-robot"
ROBOT_URDF = ROBOT_URDF_DIR / "flat-disk-robot.urdf"
SCRATCH_ROOT = SIM_ROOT / "scratch"
