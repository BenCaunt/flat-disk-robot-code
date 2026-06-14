"""Render a still image from the AI2-THOR/ProcTHOR flat disk backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .paths import SCRATCH_ROOT
from .thor_backend import (
    DEFAULT_CAMERA_FAR_PLANE_M,
    DEFAULT_CAMERA_FORWARD_OFFSET_M,
    DEFAULT_CAMERA_HEIGHT_M,
    DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    DEFAULT_CAMERA_NEAR_PLANE_M,
    DEFAULT_ITHOR_SCENE,
    FlatDiskThorSim,
    ThorSimConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "thor_render")
    parser.add_argument("--backend", default="procthor", choices=("procthor", "ithor", "house-json"))
    parser.add_argument("--scene", default=DEFAULT_ITHOR_SCENE)
    parser.add_argument("--house-json", type=Path, default=None)
    parser.add_argument("--procthor-seed", type=int, default=42)
    parser.add_argument("--procthor-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--field-of-view",
        type=float,
        default=DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
        help="Camera FOV in degrees. Defaults to the flat disk camera horizontal FOV.",
    )
    parser.add_argument(
        "--field-of-view-axis",
        default="horizontal",
        choices=("horizontal", "vertical"),
        help="Axis for --field-of-view. AI2-THOR receives a derived vertical FOV.",
    )
    parser.add_argument("--camera-height-m", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    parser.add_argument("--camera-forward-offset-m", type=float, default=DEFAULT_CAMERA_FORWARD_OFFSET_M)
    parser.add_argument("--camera-near-plane-m", type=float, default=DEFAULT_CAMERA_NEAR_PLANE_M)
    parser.add_argument("--camera-far-plane-m", type=float, default=DEFAULT_CAMERA_FAR_PLANE_M)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=None,
        help="Calibration JSON from scripts/checkerboard_calibration_logger.py; overrides --field-of-view.",
    )
    parser.add_argument(
        "--use-agent-camera",
        action="store_true",
        help="Render the AI2-THOR primary agent camera instead of the attached low robot camera.",
    )
    parser.add_argument("--quality", default="Low")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim = FlatDiskThorSim(
        ThorSimConfig(
            backend=args.backend,
            scene=args.scene,
            house_json=args.house_json,
            procthor_seed=args.procthor_seed,
            procthor_split=args.procthor_split,
            random_start=args.random_start,
            width=args.width,
            height=args.height,
            field_of_view=args.field_of_view,
            field_of_view_axis=args.field_of_view_axis,
            camera_height_m=args.camera_height_m,
            camera_forward_offset_m=args.camera_forward_offset_m,
            camera_near_plane_m=args.camera_near_plane_m,
            camera_far_plane_m=args.camera_far_plane_m,
            camera_calibration=args.camera_calibration,
            use_third_party_camera=not args.use_agent_camera,
            quality=args.quality,
        )
    )
    try:
        image_path = args.output_dir / "camera.jpg"
        metadata_path = args.output_dir / "metadata.json"
        sim.render_image().save(image_path, format="JPEG", quality=90)
        metadata_path.write_text(json.dumps(sim.hidden_pose(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        sim.close()

    print(json.dumps({"image": str(image_path), "metadata": str(metadata_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
