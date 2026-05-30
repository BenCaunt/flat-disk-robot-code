#!/usr/bin/env python3
"""Collect ESP32 Zenoh camera frames and calibrate from checkerboard images."""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import cv2
import numpy as np
import zenoh


cv2.ocl.setUseOpenCL(False)

VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
CALIBRATION_SCHEMA = "flatdisk.camera_calibration.v1"


def build_config(mode: str, listen: str, connect: str | None) -> zenoh.Config:
    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps([listen]))
    if connect:
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    return config


def parse_video_sample(data: bytes) -> tuple[int, int, int, int, bytes] | None:
    if len(data) < VIDEO_STRUCT.size:
        return None
    magic, version, _fmt, width, height, header_len, seq, esp_us, jpeg_len = VIDEO_STRUCT.unpack_from(data)
    if magic != b"FDV1" or version != 1 or header_len > len(data):
        return None
    jpeg = data[header_len : header_len + jpeg_len]
    if len(jpeg) != jpeg_len:
        return None
    return seq, esp_us, width, height, jpeg


def rotate_bgr(image: np.ndarray, rotate_deg: int) -> np.ndarray:
    rotate_deg %= 360
    if rotate_deg == 0:
        return image
    if rotate_deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotate_deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("--rotate-deg must be one of 0, 90, 180, 270")


def decode_jpeg(jpeg: bytes, rotate_deg: int) -> np.ndarray | None:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return rotate_bgr(image, rotate_deg)


def detect_checkerboard(image_bgr: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            return True, corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return False, None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return True, corners.astype(np.float32)


def image_sharpness(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(root.glob(suffix))
    return sorted(set(paths))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def collect(args: argparse.Namespace) -> int:
    out_dir = args.output_dir.resolve()
    raw_dir = out_dir / "raw"
    accepted_dir = out_dir / "accepted"
    rejected_dir = out_dir / "rejected"
    overlay_dir = out_dir / "overlays"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    session = zenoh.open(build_config(args.mode, args.listen, args.connect))
    video_sub = session.declare_subscriber(f"{args.namespace}/camera/jpeg")
    pattern_size = (args.pattern_cols, args.pattern_rows)

    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Collecting checkerboard frames from {args.namespace}/camera/jpeg")
    print(f"Pattern: {args.pattern_cols}x{args.pattern_rows} inner corners")
    print(f"Output: {out_dir}")

    start_ns = time.monotonic_ns()
    last_saved_ns = 0
    accepted = 0
    rejected = 0
    seen = 0
    meta_path = out_dir / "frames.jsonl"

    with meta_path.open("a") as meta_file:
        while not stop:
            now_ns = time.monotonic_ns()
            if args.duration > 0 and (now_ns - start_ns) / 1_000_000_000.0 >= args.duration:
                break
            if args.max_images > 0 and accepted >= args.max_images:
                break

            sample = video_sub.try_recv()
            if sample is None:
                time.sleep(0.005)
                continue

            parsed = parse_video_sample(sample.payload.to_bytes())
            if parsed is None:
                continue
            seq, esp_us, width, height, jpeg = parsed
            seen += 1
            if args.process_every_n > 1 and seq % args.process_every_n != 0:
                continue
            if now_ns - last_saved_ns < int(args.min_interval * 1_000_000_000):
                continue

            image = decode_jpeg(jpeg, args.rotate_deg)
            if image is None:
                continue
            sharpness = image_sharpness(image)
            found, corners = detect_checkerboard(image, pattern_size)
            is_good = found and sharpness >= args.min_sharpness
            stem = f"frame_{seq:08d}"

            if args.save_all:
                cv2.imwrite(str(raw_dir / f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if is_good:
                accepted += 1
                last_saved_ns = now_ns
                cv2.imwrite(str(accepted_dir / f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                overlay = image.copy()
                cv2.drawChessboardCorners(overlay, pattern_size, corners, found)
                cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            elif args.save_rejected:
                rejected += 1
                cv2.imwrite(str(rejected_dir / f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            meta_file.write(
                json.dumps(
                    {
                        "seq": seq,
                        "esp_us": esp_us,
                        "width": width,
                        "height": height,
                        "rotated_width": int(image.shape[1]),
                        "rotated_height": int(image.shape[0]),
                        "sharpness": sharpness,
                        "checkerboard_found": bool(found),
                        "accepted": bool(is_good),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            meta_file.flush()

            print(
                f"seq={seq} found={found} sharpness={sharpness:.1f} "
                f"accepted={accepted} seen={seen}"
            )

    summary = {
        "accepted": accepted,
        "rejected_saved": rejected,
        "seen": seen,
        "accepted_dir": str(accepted_dir),
        "frames_jsonl": str(meta_path),
    }
    write_json(out_dir / "collection.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def calibrate_from_images(args: argparse.Namespace) -> int:
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_template = np.zeros((args.pattern_cols * args.pattern_rows, 3), np.float32)
    grid = np.mgrid[0 : args.pattern_cols, 0 : args.pattern_rows].T.reshape(-1, 2)
    object_template[:, :2] = grid * args.square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    rejected_images: list[dict[str, Any]] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths(images_dir):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected_images.append({"image": str(path), "reason": "unreadable"})
            continue
        image = rotate_bgr(image, args.rotate_deg)
        size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            rejected_images.append({"image": str(path), "reason": f"size {size} != {image_size}"})
            continue

        found, corners = detect_checkerboard(image, pattern_size)
        if not found or corners is None:
            rejected_images.append({"image": str(path), "reason": "checkerboard_not_found"})
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        used_images.append(str(path))

        if args.write_overlays:
            overlay = image.copy()
            cv2.drawChessboardCorners(overlay, pattern_size, corners, found)
            overlay_path = output_dir / "overlays" / path.name
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if image_size is None:
        raise RuntimeError(f"No readable images found in {images_dir}")
    if len(object_points) < args.min_images:
        raise RuntimeError(
            f"Only {len(object_points)} usable checkerboard images found; "
            f"need at least {args.min_images}."
        )

    flags = 0
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    dist = dist_coeffs.ravel()
    dist8 = np.zeros(8, dtype=np.float64)
    dist8[: min(len(dist), 8)] = dist[: min(len(dist), 8)]
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    k1, k2, p1, p2, k3 = (float(dist8[0]), float(dist8[1]), float(dist8[2]), float(dist8[3]), float(dist8[4]))

    per_image_errors: list[dict[str, Any]] = []
    total_sq_error = 0.0
    total_points = 0
    for image_path, objp, imgp, rvec, tvec in zip(used_images, object_points, image_points, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        n = len(projected)
        total_sq_error += error * error
        total_points += n
        per_image_errors.append({"image": image_path, "rms_px": float((error * error / n) ** 0.5)})
    total_rms = float((total_sq_error / max(total_points, 1)) ** 0.5)

    opencv_params = [fx, fy, cx, cy, k1, k2, p1, p2]
    simple_radial_params = [(fx + fy) / 2.0, cx, cy, k1]
    calibration = {
        "schema": CALIBRATION_SCHEMA,
        "created_unix": time.time(),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size": args.square_size,
        "used_image_count": len(used_images),
        "rejected_image_count": len(rejected_images),
        "rms_reprojection_error_px": float(rms),
        "total_rms_reprojection_error_px": total_rms,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.ravel().tolist(),
        "colmap": {
            "camera_model": "OPENCV",
            "camera_params": opencv_params,
            "camera_params_string": ",".join(f"{x:.12g}" for x in opencv_params),
            "camera_line": " ".join(
                [
                    "OPENCV",
                    str(image_size[0]),
                    str(image_size[1]),
                    *[f"{x:.12g}" for x in opencv_params],
                ]
            ),
            "simple_radial_params": simple_radial_params,
            "simple_radial_params_string": ",".join(f"{x:.12g}" for x in simple_radial_params),
        },
        "used_images": used_images,
        "rejected_images": rejected_images,
        "per_image_errors": per_image_errors,
    }

    calibration_path = output_dir / args.output_name
    write_json(calibration_path, calibration)
    (output_dir / "colmap_camera.txt").write_text(calibration["colmap"]["camera_line"] + "\n")
    (output_dir / "visual_localization_args.txt").write_text(
        "\n".join(
            [
                f'--camera-calibration "{calibration_path}"',
                f'--query-camera-calibration "{calibration_path}"',
                f'--camera-model OPENCV --camera-params "{calibration["colmap"]["camera_params_string"]}"',
                f'--query-camera-model OPENCV --query-camera-params "{calibration["colmap"]["camera_params_string"]}"',
            ]
        )
        + "\n"
    )

    print(f"Wrote calibration to {calibration_path}")
    print(f"RMS reprojection error: {rms:.3f} px from {len(used_images)} images")
    print(f"visual_localization.py can read: --query-camera-calibration {calibration_path}")
    return 0


def capture_calibrate(args: argparse.Namespace) -> int:
    collect_args = argparse.Namespace(**vars(args))
    collect(collect_args)
    accepted_dir = args.output_dir.resolve() / "accepted"
    calibrate_args = argparse.Namespace(**vars(args))
    calibrate_args.images_dir = accepted_dir
    return calibrate_from_images(calibrate_args)


def add_zenoh_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace", default="flatdisk/xiao")
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447")


def add_board_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pattern-cols", type=int, required=True, help="Inner checkerboard corners across columns.")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Inner checkerboard corners across rows.")
    parser.add_argument("--square-size", type=float, default=1.0, help="Checker square size in any consistent unit.")
    parser.add_argument("--rotate-deg", type=int, default=0, choices=(0, 90, 180, 270))


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--min-interval", type=float, default=0.4)
    parser.add_argument("--process-every-n", type=int, default=1)
    parser.add_argument("--min-sharpness", type=float, default=20.0)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--save-rejected", action="store_true")


def add_calibrate_args(parser: argparse.ArgumentParser, *, include_output_dir: bool = True) -> None:
    if include_output_dir:
        parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="camera_calibration.json")
    parser.add_argument("--min-images", type=int, default=10)
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--zero-tangent-dist", action="store_true")
    parser.add_argument("--fix-k3", action="store_true")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect_parser = sub.add_parser("collect", help="Log accepted checkerboard frames from Zenoh.")
    add_zenoh_args(collect_parser)
    add_board_args(collect_parser)
    add_collect_args(collect_parser)
    collect_parser.set_defaults(func=collect)

    calibrate_parser = sub.add_parser("calibrate", help="Calibrate from an image directory.")
    calibrate_parser.add_argument("--images-dir", type=Path, required=True)
    add_board_args(calibrate_parser)
    add_calibrate_args(calibrate_parser)
    calibrate_parser.set_defaults(func=calibrate_from_images)

    both_parser = sub.add_parser("capture-calibrate", help="Collect Zenoh frames, then calibrate.")
    add_zenoh_args(both_parser)
    add_board_args(both_parser)
    add_collect_args(both_parser)
    add_calibrate_args(both_parser, include_output_dir=False)
    both_parser.set_defaults(func=capture_calibrate)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
