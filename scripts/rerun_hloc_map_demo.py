#!/usr/bin/env python3
"""Log an hloc/pycolmap map and optional localization pose to Rerun."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pycolmap
import rerun as rr


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def qvec_wxyz_to_rotmat(qvec: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero-length quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def image_pose(image: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    cam_from_world = image.cam_from_world()
    rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    center = np.asarray(image.projection_center(), dtype=np.float64)
    return rotation, center


def camera_intrinsics(camera: pycolmap.Camera) -> tuple[float, float, float, float]:
    params = np.asarray(camera.params, dtype=np.float64)
    model = str(camera.model)
    if "SIMPLE" in model:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported camera model for demo intrinsics: {camera.model}")


def frustum_strips(
    rotation_cam_from_world: np.ndarray,
    center_world: np.ndarray,
    camera: pycolmap.Camera,
    scale: float,
) -> list[np.ndarray]:
    fx, fy, cx, cy = camera_intrinsics(camera)
    width = float(camera.width)
    height = float(camera.height)
    corners_px = np.array(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=np.float64,
    )
    corners_cam = np.column_stack(
        [
            (corners_px[:, 0] - cx) / fx * scale,
            (corners_px[:, 1] - cy) / fy * scale,
            np.full(4, scale, dtype=np.float64),
        ]
    )
    corners_world = center_world[None, :] + corners_cam @ rotation_cam_from_world
    center = center_world.reshape(1, 3)
    return [
        np.vstack([corners_world, corners_world[0]]),
        np.vstack([center, corners_world[0]]),
        np.vstack([center, corners_world[1]]),
        np.vstack([center, corners_world[2]]),
        np.vstack([center, corners_world[3]]),
    ]


def pose_from_localization_json(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = load_json(path)
    rotation = qvec_wxyz_to_rotmat(payload["qvec_wxyz_cam_from_world"])
    if "camera_center_world" in payload:
        center = np.asarray(payload["camera_center_world"], dtype=np.float64)
    else:
        tvec = np.asarray(payload["tvec_cam_from_world"], dtype=np.float64)
        center = -rotation.T @ tvec
    return rotation, center, payload


def pose_from_hloc_text(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    line = path.read_text().strip().splitlines()[0]
    fields = line.split()
    if len(fields) != 8:
        raise ValueError(f"Expected 'name qw qx qy qz tx ty tz' in {path}")
    name = fields[0]
    qvec = [float(v) for v in fields[1:5]]
    tvec = np.asarray([float(v) for v in fields[5:8]], dtype=np.float64)
    rotation = qvec_wxyz_to_rotmat(qvec)
    center = -rotation.T @ tvec
    return rotation, center, {"query": name, "qvec_wxyz_cam_from_world": qvec, "tvec_cam_from_world": tvec.tolist()}


def find_pose_file(map_dir: Path) -> Path | None:
    candidates = [
        map_dir.parent / f"{map_dir.name.replace('_map', '')}_pose_nn.json",
        map_dir / "live" / "pose.json",
        map_dir / "live" / "query-localization.txt",
    ]
    return next((path for path in candidates if path.exists()), None)


def log_image_sequence(image_dir: Path, image_names: list[str], max_images: int) -> None:
    if max_images <= 0:
        return
    step = max(1, math.ceil(len(image_names) / max_images))
    for idx, name in enumerate(image_names[::step]):
        image_path = image_dir / name
        if not image_path.exists():
            continue
        rr.set_time("map_frame", sequence=idx)
        rr.log(f"map_images/{Path(name).stem}", rr.EncodedImage(path=image_path))


def run(args: argparse.Namespace) -> int:
    map_dir = args.map_dir.resolve()
    sfm_dir = map_dir / "sfm"
    image_dir = map_dir / "images"
    if not sfm_dir.exists():
        raise RuntimeError(f"Missing reconstruction directory: {sfm_dir}")

    reconstruction = pycolmap.Reconstruction(sfm_dir)
    images = sorted(
        (image for image in reconstruction.images.values() if image.has_pose),
        key=lambda image: image.name,
    )
    if not images:
        raise RuntimeError(f"No registered images in {sfm_dir}")

    rr.init(args.app_id, spawn=args.spawn and args.output is None)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.output)
    if args.connect_grpc is not None:
        rr.connect_grpc(args.connect_grpc)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    point_items = list(reconstruction.points3D.values())
    if point_items:
        points = np.asarray([point.xyz for point in point_items], dtype=np.float32)
        colors = np.asarray([point.color for point in point_items], dtype=np.uint8)
        rr.log("world/sparse_points", rr.Points3D(points, colors=colors, radii=0.015), static=True)

    centers: list[np.ndarray] = []
    frustums: list[np.ndarray] = []
    labels: list[str] = []
    for image in images:
        camera = reconstruction.cameras[image.camera_id]
        rotation, center = image_pose(image)
        centers.append(center)
        labels.append(Path(image.name).stem)
        frustums.extend(frustum_strips(rotation, center, camera, args.frustum_scale))

    centers_arr = np.asarray(centers, dtype=np.float32)
    rr.log(
        "world/map_cameras/centers",
        rr.Points3D(centers_arr, colors=[40, 170, 255], radii=0.06, labels=labels),
        static=True,
    )
    rr.log("world/map_cameras/path", rr.LineStrips3D([centers_arr], colors=[40, 170, 255], radii=0.018), static=True)
    rr.log("world/map_cameras/frustums", rr.LineStrips3D(frustums, colors=[40, 170, 255], radii=0.01), static=True)

    pose_path = args.pose_json or find_pose_file(map_dir)
    if pose_path is not None and pose_path.exists():
        if pose_path.suffix.lower() == ".json":
            query_rotation, query_center, pose_payload = pose_from_localization_json(pose_path)
        else:
            query_rotation, query_center, pose_payload = pose_from_hloc_text(pose_path)
        query_camera = reconstruction.cameras[images[0].camera_id]
        query_frustum = frustum_strips(query_rotation, query_center, query_camera, args.frustum_scale * 1.5)
        rr.log(
            "world/query_pose/center",
            rr.Points3D([query_center], colors=[255, 64, 64], radii=0.12, labels=[pose_payload.get("query", "query")]),
            static=True,
        )
        rr.log("world/query_pose/frustum", rr.LineStrips3D(query_frustum, colors=[255, 64, 64], radii=0.025), static=True)
        rr.log(
            "world/query_pose/metadata",
            rr.TextLog(json.dumps(pose_payload, sort_keys=True)),
            static=True,
        )
        query_image = map_dir / "live" / "query_images" / str(pose_payload.get("query", "query.jpg"))
        if args.include_query_image and query_image.exists():
            rr.log("query/image", rr.EncodedImage(path=query_image), static=True)

    log_image_sequence(image_dir, [image.name for image in images], args.max_images)

    print(reconstruction.summary())
    if args.output is not None:
        print(f"Rerun recording written to {args.output}")
    elif args.spawn:
        print("Rerun viewer spawned.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=Path("captures/freiburg_test_map"))
    parser.add_argument("--pose-json", type=Path, default=None, help="Optional visual_localization.py pose JSON.")
    parser.add_argument("--output", type=Path, default=Path("captures/freiburg_test_map/rerun_demo.rrd"))
    parser.add_argument("--app-id", default="flatdisk_hloc_map_demo")
    parser.add_argument("--frustum-scale", type=float, default=0.35)
    parser.add_argument("--max-images", type=int, default=0, help="Log this many map images as 2D panels. 0 disables images.")
    parser.add_argument("--include-query-image", action="store_true", help="Also log the localized query JPEG as a 2D panel.")
    parser.add_argument("--spawn", action="store_true", help="Spawn Rerun viewer when not saving to --output.")
    parser.add_argument("--connect-grpc", default=None, help="Connect to an existing Rerun gRPC endpoint.")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
