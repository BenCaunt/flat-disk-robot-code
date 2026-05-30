#!/usr/bin/env python3
"""Build an hloc room map from video and localize ESP32 camera frames in it."""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HLOC_DEVICE", "mps")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_ROOT = PROJECT_ROOT.parent
DEFAULT_HLOC_REPO = DOCUMENTS_ROOT / "Hierarchical-Localization"
HLOC_REPO = Path(os.environ.get("HLOC_REPO", DEFAULT_HLOC_REPO)).expanduser().resolve()
if HLOC_REPO.exists():
    sys.path.insert(0, str(HLOC_REPO))

import cv2
import h5py
import numpy as np
import pycolmap
import torch
from PIL import Image

from hloc import extract_features, extractors, localize_sfm, match_features, matchers, pairs_from_retrieval, reconstruction
from hloc.utils.base_model import dynamic_load
from hloc.utils.device import get_torch_device
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import names_to_pair


VIDEO_STRUCT = struct.Struct("<4sBBHHHIQI")
MAP_SCHEMA = "flatdisk.hloc_map.v1"
POSE_SCHEMA = "flatdisk.visual_pose.v1"
GLOBAL_CONFS = ("netvlad", "dir", "openibl", "megaloc")


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def resize_long_edge(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image
    height, width = image.shape[:2]
    long_edge = max(width, height)
    if long_edge <= max_width:
        return image
    scale = max_width / long_edge
    size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def extract_video_frames(args: argparse.Namespace, image_dir: Path) -> list[str]:
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0
    sample_fps = args.fps if args.fps > 0 else src_fps
    keep_interval = 1.0 / sample_fps

    image_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    next_keep_s = max(args.start_sec, 0.0)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_s = frame_idx / src_fps
        frame_idx += 1
        if t_s < args.start_sec:
            continue
        if args.end_sec > 0 and t_s > args.end_sec:
            break
        if t_s + 1e-9 < next_keep_s:
            continue

        frame = rotate_bgr(frame, args.rotate_deg)
        frame = resize_long_edge(frame, args.max_image_width)
        name = f"frame_{len(names):06d}.jpg"
        out = image_dir / name
        if not cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]):
            raise RuntimeError(f"Could not write {out}")
        names.append(name)
        next_keep_s += keep_interval

        if args.max_frames > 0 and len(names) >= args.max_frames:
            break

    cap.release()
    if len(names) < 2:
        raise RuntimeError("Need at least 2 extracted frames to build a map.")
    return names


def write_pairs(names: list[str], path: Path, mode: str, window: int) -> None:
    pairs: list[tuple[str, str]] = []
    if mode == "exhaustive":
        for i, name0 in enumerate(names):
            for name1 in names[i + 1 :]:
                pairs.append((name0, name1))
    elif mode == "sequential":
        for i, name0 in enumerate(names):
            for j in range(i + 1, min(len(names), i + window + 1)):
                pairs.append((name0, names[j]))
    else:
        raise ValueError(f"Unsupported pair mode {mode}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n")


def camera_mode_from_args(args: argparse.Namespace) -> pycolmap.CameraMode:
    return pycolmap.CameraMode.AUTO if args.per_image_camera else pycolmap.CameraMode.SINGLE


def image_options_from_args(args: argparse.Namespace, image_dir: Path) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if getattr(args, "camera_calibration", None):
        image_size = first_image_size(image_dir)
        model, params = scaled_calibration(args.camera_calibration, image_size[0], image_size[1])
        options["camera_model"] = model
        options["camera_params"] = params
        return options
    if args.camera_model:
        options["camera_model"] = args.camera_model
    if args.camera_params:
        options["camera_params"] = args.camera_params
    return options


def first_image_size(image_dir: Path) -> tuple[int, int]:
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in sorted(image_dir.glob(suffix)):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return int(image.shape[1]), int(image.shape[0])
    raise RuntimeError(f"No readable images found in {image_dir}")


def scale_camera_params(
    model: str,
    params: list[float],
    from_size: tuple[int, int],
    to_size: tuple[int, int],
) -> list[float]:
    if from_size == to_size:
        return params
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    scaled = list(params)
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        if len(scaled) >= 4:
            scaled[0] *= sx
            scaled[1] *= sy
            scaled[2] *= sx
            scaled[3] *= sy
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        if len(scaled) >= 3:
            scaled[0] *= (sx + sy) / 2.0
            scaled[1] *= sx
            scaled[2] *= sy
    return scaled


def scaled_calibration(calibration_path: Path | str, width: int, height: int) -> tuple[str, str]:
    calibration_path = Path(calibration_path)
    data = load_json(calibration_path)
    colmap = data.get("colmap", {})
    model = str(colmap.get("camera_model", ""))
    params = colmap.get("camera_params")
    if not model or not isinstance(params, list):
        raise RuntimeError(f"{calibration_path} does not contain colmap.camera_model and colmap.camera_params")
    from_size = (int(data["image_width"]), int(data["image_height"]))
    to_size = (int(width), int(height))
    scaled = scale_camera_params(model, [float(x) for x in params], from_size, to_size)
    return model, ",".join(f"{x:.12g}" for x in scaled)


def build_map(args: argparse.Namespace) -> int:
    map_dir = args.map_dir.resolve()
    image_dir = map_dir / "images"
    hloc_dir = map_dir / "hloc"
    sfm_dir = map_dir / "sfm"
    manifest_path = map_dir / "map.json"

    if map_dir.exists() and any(map_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{map_dir} already exists and is not empty. Pass --overwrite to reuse it.")
    map_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_frames:
        names = sorted(path.name for path in image_dir.glob("*.jpg"))
        if len(names) < 2:
            raise RuntimeError(f"No reusable JPG frames found in {image_dir}")
    else:
        names = extract_video_frames(args, image_dir)

    local_conf = extract_features.confs[args.feature_conf]
    matcher_conf = match_features.confs[args.matcher_conf]
    features = extract_features.main(
        local_conf,
        image_dir,
        hloc_dir,
        as_half=True,
        image_list=names,
        overwrite=args.overwrite,
    )

    pairs_path = hloc_dir / "pairs-sfm.txt"
    write_pairs(names, pairs_path, args.pairing, args.window)
    matches_path = hloc_dir / f"matches-{args.matcher_conf}-sfm.h5"
    match_features.main(
        matcher_conf,
        pairs_path,
        features,
        matches=matches_path,
        features_ref=features,
        overwrite=args.overwrite,
    )

    rec = reconstruction.main(
        sfm_dir,
        image_dir,
        pairs_path,
        features,
        matches_path,
        camera_mode=camera_mode_from_args(args),
        skip_geometric_verification=args.skip_geometric_verification,
        image_options=image_options_from_args(args, image_dir),
        mapper_options={"num_threads": 1},
    )
    if rec is None:
        raise RuntimeError("COLMAP/pycolmap reconstruction failed.")

    global_features: Path | None = None
    if args.global_conf != "none":
        global_conf = extract_features.confs[args.global_conf]
        global_features = extract_features.main(
            global_conf,
            image_dir,
            hloc_dir,
            as_half=False,
            image_list=names,
            overwrite=args.overwrite,
        )

    registered = {img.name for img in rec.images.values()}
    registered_names = [name for name in names if name in registered]
    manifest = {
        "schema": MAP_SCHEMA,
        "created_unix": time.time(),
        "hloc_repo": str(HLOC_REPO),
        "video": str(args.video.resolve()) if args.video else None,
        "image_dir": rel(image_dir, map_dir),
        "hloc_dir": rel(hloc_dir, map_dir),
        "sfm_dir": rel(sfm_dir, map_dir),
        "features": rel(features, map_dir),
        "matches": rel(matches_path, map_dir),
        "pairs": rel(pairs_path, map_dir),
        "global_conf": None if global_features is None else args.global_conf,
        "global_features": None if global_features is None else rel(global_features, map_dir),
        "feature_conf": args.feature_conf,
        "matcher_conf": args.matcher_conf,
        "pairing": args.pairing,
        "window": args.window,
        "frames": names,
        "registered_frames": registered_names,
        "camera_calibration": str(args.camera_calibration.resolve()) if args.camera_calibration else None,
        "camera_model": args.camera_model,
        "camera_params": args.camera_params,
        "summary": rec.summary(),
    }
    write_json(manifest_path, manifest)

    print(f"Map written to {map_dir}")
    print(rec.summary())
    print(f"Registered {len(registered_names)} / {len(names)} extracted frames")
    return 0


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


def build_zenoh_config(mode: str, listen: str, connect: str | None):
    import zenoh

    config = zenoh.Config()
    config.insert_json5("mode", json.dumps(mode))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps([listen]))
    if connect:
        config.insert_json5("connect/endpoints", json.dumps([connect]))
    return config


def query_camera(width: int, height: int, args: argparse.Namespace) -> pycolmap.Camera:
    if args.query_camera_calibration:
        model, params_string = scaled_calibration(args.query_camera_calibration, width, height)
        params = [float(x) for x in params_string.replace(",", " ").split()]
        return pycolmap.Camera(
            model=model,
            width=width,
            height=height,
            params=params,
        )

    if args.query_camera_params:
        params = [float(x) for x in args.query_camera_params.replace(",", " ").split()]
        return pycolmap.Camera(
            model=args.query_camera_model,
            width=width,
            height=height,
            params=params,
        )

    focal = args.query_focal_factor * max(width, height)
    return pycolmap.Camera(
        model="SIMPLE_RADIAL",
        width=width,
        height=height,
        params=[focal, width / 2.0, height / 2.0, 0.0],
    )


def write_query_list(path: Path, name: str, camera: pycolmap.Camera) -> None:
    params = " ".join(str(float(x)) for x in camera.params)
    path.write_text(f"{name} {camera.model.name} {camera.width} {camera.height} {params}\n")


def spaced_candidates(names: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(names):
        return names
    if count == 1:
        return [names[len(names) // 2]]
    idxs = np.linspace(0, len(names) - 1, count).round().astype(int)
    return [names[i] for i in dict.fromkeys(idxs.tolist())]


def quat_wxyz_to_matrix(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_payload_from_result(
    results_path: Path,
    logs_path: Path,
    query_name: str,
    seq: int | None,
    esp_us: int | None,
    candidate_count: int,
    latency_ms: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": POSE_SCHEMA,
        "localized": False,
        "query": query_name,
        "seq": seq,
        "esp_us": esp_us,
        "candidate_count": candidate_count,
        "latency_ms": latency_ms,
        "host_time_unix": time.time(),
    }
    if not results_path.exists():
        return payload

    lines = [line.split() for line in results_path.read_text().splitlines() if line.strip()]
    row = next((line for line in lines if line[0] == Path(query_name).name), None)
    if row is None:
        return payload

    qvec = [float(x) for x in row[1:5]]
    tvec = np.array([float(x) for x in row[5:8]], dtype=np.float64)
    rotation = quat_wxyz_to_matrix(qvec)
    camera_center_world = -rotation.T @ tvec
    payload.update(
        {
            "localized": True,
            "qvec_wxyz_cam_from_world": qvec,
            "tvec_cam_from_world": tvec.tolist(),
            "camera_center_world": camera_center_world.tolist(),
            "robot_center_world_approx": camera_center_world.tolist(),
        }
    )

    if logs_path.exists():
        import pickle

        with open(logs_path, "rb") as f:
            logs = pickle.load(f)
        loc = logs.get("loc", {}).get(query_name)
        if loc:
            ret = loc.get("PnP_ret")
            payload["num_matches"] = int(loc.get("num_matches", 0))
            if ret is not None:
                payload["num_inliers"] = int(ret.get("num_inliers", 0))
                payload["inlier_ratio"] = payload["num_inliers"] / max(1, payload["num_matches"])
    return payload


class MapLocalizer:
    def __init__(self, map_dir: Path, args: argparse.Namespace) -> None:
        self.map_dir = map_dir.resolve()
        self.meta = load_json(self.map_dir / "map.json")
        if self.meta.get("schema") != MAP_SCHEMA:
            raise RuntimeError(f"Unsupported map schema in {self.map_dir / 'map.json'}")

        self.image_dir = self.map_dir / self.meta["image_dir"]
        self.sfm_dir = self.map_dir / self.meta["sfm_dir"]
        self.map_features = self.map_dir / self.meta["features"]
        self.global_features = (
            None if self.meta.get("global_features") is None else self.map_dir / self.meta["global_features"]
        )
        self.feature_conf_name = self.meta["feature_conf"]
        self.matcher_conf_name = args.matcher_conf or self.meta["matcher_conf"]
        self.global_conf_name = self.meta.get("global_conf")
        self.local_conf = extract_features.confs[self.feature_conf_name]
        self.matcher_conf = match_features.confs[self.matcher_conf_name]
        self.device = get_torch_device()
        LocalModel = dynamic_load(extractors, self.local_conf["model"]["name"])
        self.local_model = LocalModel(self.local_conf["model"]).eval().to(self.device)
        MatcherModel = dynamic_load(matchers, self.matcher_conf["model"]["name"])
        self.matcher_model = MatcherModel(self.matcher_conf["model"]).eval().to(self.device)
        self.global_model = None
        if self.global_features is not None and self.global_conf_name is not None:
            global_conf = extract_features.confs[self.global_conf_name]
            GlobalModel = dynamic_load(extractors, global_conf["model"]["name"])
            self.global_model = GlobalModel(global_conf["model"]).eval().to(self.device)
        self.live_dir = self.map_dir / "live"
        self.query_image_dir = self.live_dir / "query_images"
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.query_image_dir.mkdir(parents=True, exist_ok=True)

        self.registered_names = list(self.meta.get("registered_frames") or self.meta["frames"])
        if not self.registered_names:
            raise RuntimeError("Map has no registered frames.")

        self.query_name = "query.jpg"
        self.query_image = self.query_image_dir / self.query_name
        self.query_list = self.live_dir / "query.txt"
        self.query_features = self.live_dir / "query-features.h5"
        self.query_global = self.live_dir / "query-global.h5"
        self.retrieval = self.live_dir / "query-pairs.txt"
        self.query_matches = self.live_dir / "query-matches.h5"
        self.results = self.live_dir / "query-localization.txt"

    @torch.inference_mode()
    def write_features(
        self,
        model: torch.nn.Module,
        conf: dict[str, Any],
        image_dir: Path,
        image_name: str,
        feature_path: Path,
        as_half: bool,
    ) -> None:
        dataset = extract_features.ImageDataset(image_dir, conf["preprocessing"], [image_name])
        data = dataset[0]
        image = torch.from_numpy(data["image"][None]).to(self.device)
        pred = model({"image": image})
        pred_np = {key: value[0].detach().cpu().numpy() for key, value in pred.items()}

        pred_np["image_size"] = original_size = data["original_size"]
        uncertainty = None
        if "keypoints" in pred_np:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred_np["keypoints"] = (pred_np["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred_np:
                pred_np["scales"] *= scales.mean()
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        if as_half:
            for key, value in list(pred_np.items()):
                if value.dtype == np.float32:
                    pred_np[key] = value.astype(np.float16)

        with h5py.File(str(feature_path), "w", libver="latest") as fd:
            grp = fd.create_group(image_name)
            for key, value in pred_np.items():
                grp.create_dataset(key, data=value)
            if uncertainty is not None:
                grp["keypoints"].attrs["uncertainty"] = uncertainty

    def read_match_inputs(self, query_name: str, ref_name: str) -> dict[str, torch.Tensor]:
        data: dict[str, torch.Tensor] = {}
        for suffix, path, name in (
            ("0", self.query_features, query_name),
            ("1", self.map_features, ref_name),
        ):
            with h5py.File(str(path), "r", libver="latest") as fd:
                grp = fd[name]
                for key, value in grp.items():
                    data[key + suffix] = torch.from_numpy(value.__array__()).float().unsqueeze(0)
                width, height = tuple(grp["image_size"])
                data["image" + suffix] = torch.empty((1, 1, int(height), int(width)))
        return data

    @torch.inference_mode()
    def match_query_pairs(self) -> None:
        pairs = [line.split() for line in self.retrieval.read_text().splitlines() if line.strip()]
        with h5py.File(str(self.query_matches), "w", libver="latest") as fd:
            for query_name, ref_name in pairs:
                data = self.read_match_inputs(query_name, ref_name)
                data = {
                    key: value if key.startswith("image") else value.to(self.device, non_blocking=True)
                    for key, value in data.items()
                }
                pred = self.matcher_model(data)
                grp = fd.create_group(names_to_pair(query_name, ref_name))
                grp.create_dataset("matches0", data=pred["matches0"][0].detach().cpu().short().numpy())
                if "matching_scores0" in pred:
                    grp.create_dataset(
                        "matching_scores0",
                        data=pred["matching_scores0"][0].detach().cpu().half().numpy(),
                    )

    def write_query_image(self, jpeg: bytes, rotate_deg: int) -> tuple[int, int]:
        image = Image.open(BytesIO(jpeg)).convert("RGB")
        if rotate_deg:
            image = image.rotate(-rotate_deg, expand=True)
        image.save(self.query_image, format="JPEG", quality=95)
        return image.size

    def write_retrieval_pairs(self, args: argparse.Namespace) -> int:
        if self.global_features is not None and self.global_conf_name is not None:
            global_conf = extract_features.confs[self.global_conf_name]
            assert self.global_model is not None
            self.write_features(
                self.global_model,
                global_conf,
                self.query_image_dir,
                self.query_name,
                self.query_global,
                as_half=False,
            )
            db_count = len(list_h5_names(self.global_features))
            num_matched = args.top_k
            if num_matched <= 0 or num_matched > db_count:
                num_matched = db_count
            pairs_from_retrieval.main(
                self.query_global,
                self.retrieval,
                num_matched,
                db_descriptors=self.global_features,
            )
            return len(self.retrieval.read_text().splitlines())

        candidates = spaced_candidates(self.registered_names, args.top_k)
        self.retrieval.write_text("\n".join(f"{self.query_name} {name}" for name in candidates) + "\n")
        return len(candidates)

    def localize_jpeg(self, jpeg: bytes, args: argparse.Namespace, seq: int | None, esp_us: int | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        width, height = self.write_query_image(jpeg, args.rotate_deg)
        camera = query_camera(width, height, args)
        write_query_list(self.query_list, self.query_name, camera)

        self.write_features(
            self.local_model,
            self.local_conf,
            self.query_image_dir,
            self.query_name,
            self.query_features,
            as_half=True,
        )
        candidate_count = self.write_retrieval_pairs(args)
        self.match_query_pairs()

        if self.results.exists():
            self.results.unlink()
        logs_path = Path(f"{self.results}_logs.pkl")
        if logs_path.exists():
            logs_path.unlink()

        localize_sfm.main(
            self.sfm_dir,
            self.query_list,
            self.retrieval,
            self.query_features,
            self.query_matches,
            self.results,
            ransac_thresh=args.ransac_thresh,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return pose_payload_from_result(
            self.results,
            logs_path,
            self.query_name,
            seq,
            esp_us,
            candidate_count,
            latency_ms,
        )


def localize_image(args: argparse.Namespace) -> int:
    localizer = MapLocalizer(args.map_dir, args)
    jpeg = args.image.read_bytes()
    payload = localizer.localize_jpeg(jpeg, args, seq=None, esp_us=None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json:
        write_json(args.output_json, payload)
    return 0 if payload["localized"] else 2


def localize_zenoh(args: argparse.Namespace) -> int:
    import zenoh

    localizer = MapLocalizer(args.map_dir, args)
    session = zenoh.open(build_zenoh_config(args.mode, args.listen, args.connect))
    video_sub = session.declare_subscriber(f"{args.namespace}/camera/jpeg")
    pose_key = args.pose_key or f"{args.namespace}/pose/visual"
    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Visual localizer map={args.map_dir} namespace={args.namespace} pose_key={pose_key}")
    print("Waiting for Zenoh camera frames...")
    start_ns = time.monotonic_ns()
    last_process_ns = 0
    processed = 0
    latest_sample: bytes | None = None

    while not stop:
        now_ns = time.monotonic_ns()
        if args.duration > 0 and (now_ns - start_ns) / 1_000_000_000.0 >= args.duration:
            break

        while True:
            sample = video_sub.try_recv()
            if sample is None:
                break
            latest_sample = sample.payload.to_bytes()

        period_ns = int(1_000_000_000 / max(args.max_rate, 0.1))
        if latest_sample is None or now_ns - last_process_ns < period_ns:
            time.sleep(0.002)
            continue

        parsed = parse_video_sample(latest_sample)
        latest_sample = None
        last_process_ns = now_ns
        if parsed is None:
            continue
        seq, esp_us, width, height, jpeg = parsed
        if args.process_every_n > 1 and seq % args.process_every_n != 0:
            continue

        try:
            payload = localizer.localize_jpeg(jpeg, args, seq=seq, esp_us=esp_us)
        except Exception as exc:  # Keep a long-running localizer alive after bad frames.
            payload = {
                "schema": POSE_SCHEMA,
                "localized": False,
                "seq": seq,
                "esp_us": esp_us,
                "error": str(exc),
                "host_time_unix": time.time(),
            }

        session.put(pose_key, json.dumps(payload).encode("utf-8"))
        processed += 1
        status = "OK" if payload.get("localized") else "MISS"
        center = payload.get("camera_center_world")
        inliers = payload.get("num_inliers", 0)
        matches = payload.get("num_matches", 0)
        latency = payload.get("latency_ms", 0.0)
        print(
            f"{status} seq={seq} input={width}x{height} "
            f"latency={latency:.1f}ms inliers={inliers}/{matches} center={center}"
        )

    print(f"Processed {processed} frames.")
    return 0


def add_common_query_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rotate-deg", type=int, default=0, choices=(0, 90, 180, 270))
    parser.add_argument("--top-k", type=int, default=20, help="Reference images to localize against. <=0 means all.")
    parser.add_argument("--matcher-conf", default=None, choices=list(match_features.confs.keys()))
    parser.add_argument("--ransac-thresh", type=float, default=12.0)
    parser.add_argument("--query-camera-calibration", type=Path)
    parser.add_argument("--query-camera-model", default="SIMPLE_RADIAL")
    parser.add_argument(
        "--query-camera-params",
        default="",
        help="Explicit query camera params, e.g. '260,160,120,0' for SIMPLE_RADIAL.",
    )
    parser.add_argument(
        "--query-focal-factor",
        type=float,
        default=1.2,
        help="Fallback focal length as factor * max(width,height) when params are not provided.",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-map", help="Extract video frames and build an hloc SfM map.")
    build.add_argument("--video", type=Path, required=True)
    build.add_argument("--map-dir", type=Path, required=True)
    build.add_argument("--fps", type=float, default=2.0)
    build.add_argument("--start-sec", type=float, default=0.0)
    build.add_argument("--end-sec", type=float, default=0.0)
    build.add_argument("--max-frames", type=int, default=250)
    build.add_argument("--max-image-width", type=int, default=1280)
    build.add_argument("--jpeg-quality", type=int, default=95)
    build.add_argument("--rotate-deg", type=int, default=0, choices=(0, 90, 180, 270))
    build.add_argument("--reuse-frames", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--feature-conf", default="superpoint_aachen", choices=list(extract_features.confs.keys()))
    build.add_argument("--matcher-conf", default="superpoint+lightglue", choices=list(match_features.confs.keys()))
    build.add_argument("--global-conf", default="none", choices=["none", *GLOBAL_CONFS])
    build.add_argument("--pairing", default="sequential", choices=("sequential", "exhaustive"))
    build.add_argument("--window", type=int, default=8)
    build.add_argument("--per-image-camera", action="store_true")
    build.add_argument("--camera-calibration", type=Path)
    build.add_argument("--camera-model", default="")
    build.add_argument("--camera-params", default="")
    build.add_argument("--skip-geometric-verification", action="store_true")
    build.set_defaults(func=build_map)

    image = sub.add_parser("localize-image", help="Localize a single JPEG against a built map.")
    image.add_argument("--map-dir", type=Path, required=True)
    image.add_argument("--image", type=Path, required=True)
    image.add_argument("--output-json", type=Path)
    add_common_query_args(image)
    image.set_defaults(func=localize_image)

    live = sub.add_parser("localize-zenoh", help="Subscribe to ESP32 JPEG frames and publish visual pose.")
    live.add_argument("--map-dir", type=Path, required=True)
    live.add_argument("--namespace", default="flatdisk/xiao")
    live.add_argument("--mode", default="client")
    live.add_argument("--listen", default="")
    live.add_argument("--connect", default="tcp/127.0.0.1:7447")
    live.add_argument("--pose-key", default="")
    live.add_argument("--duration", type=float, default=0.0)
    live.add_argument("--max-rate", type=float, default=2.0)
    live.add_argument("--process-every-n", type=int, default=1)
    add_common_query_args(live)
    live.set_defaults(func=localize_zenoh)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
