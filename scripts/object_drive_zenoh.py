#!/usr/bin/env python3
"""Drive toward a visible prompted object from Zenoh camera/IMU data."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import re
import secrets
import signal
import sys
import threading
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flatdisk_robot_client import (  # noqa: E402
    DEFAULT_CONNECT,
    DEFAULT_CONTROL_HZ,
    DEFAULT_NAMESPACE,
    FlatDiskRobotClient,
    ImuSample,
    VideoFrame,
    clamp,
    wrap_pi,
)


DEFAULT_MODEL = "mlx-community/Florence-2-base-ft-4bit"
DEFAULT_HFOV_DEG = 68.0
AUTO_RERUN_SAVE = Path("__auto__")
FLORENCE_OBJECT_DETECTION_TASK = "<OPEN_VOCABULARY_DETECTION>"


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    score: float
    source: str
    raw: str = ""

    @property
    def center_x(self) -> float:
        return (self.bbox_xyxy[0] + self.bbox_xyxy[2]) * 0.5

    @property
    def center_y(self) -> float:
        return (self.bbox_xyxy[1] + self.bbox_xyxy[3]) * 0.5


@dataclass(frozen=True)
class DetectionResult:
    frame_seq: int
    received_ns: int
    image_size: tuple[int, int]
    detections: tuple[Detection, ...]
    elapsed_s: float
    error: str | None = None

    @property
    def best(self) -> Detection | None:
        if not self.detections:
            return None
        return max(self.detections, key=lambda det: det.score)


def select_detection(
    detections: tuple[Detection, ...],
    *,
    prompt: str,
    image_size: tuple[int, int],
    max_area_fraction: float,
) -> Detection | None:
    if not detections:
        return None
    width, height = image_size
    image_area = max(1.0, float(width * height))
    candidates: list[Detection] = []
    for det in detections:
        area_fraction = _bbox_area(det.bbox_xyxy) / image_area
        if 0.0 < max_area_fraction < area_fraction:
            continue
        candidates.append(det)
    if not candidates:
        return None

    prompt_words = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    prefer_closest = bool(prompt_words & {"closest", "nearest", "front", "foreground"})

    def score(det: Detection) -> float:
        x0, y0, x1, y1 = det.bbox_xyxy
        area_fraction = _bbox_area(det.bbox_xyxy) / image_area
        lower_fraction = ((y0 + y1) * 0.5) / max(float(height), 1.0)
        height_fraction = max(0.0, y1 - y0) / max(float(height), 1.0)
        value = det.score
        if prefer_closest:
            # In the low robot camera, the closest instance is usually larger
            # and lower in the frame. Keep this as a tie-breaker over the VLM
            # label score, not a replacement for the language-conditioned box.
            value += 0.35 * math.sqrt(max(0.0, area_fraction))
            value += 0.20 * lower_fraction
            value += 0.10 * height_fraction
        return value

    return max(candidates, key=score)


@dataclass(frozen=True)
class FrameState:
    frame: VideoFrame
    imu: ImuSample
    image: Image.Image
    gray: np.ndarray
    monotonic_ns: int


@dataclass(frozen=True)
class DriveCommand:
    motor1_percent: int
    motor2_percent: int
    forward_percent: float
    turn_percent: float
    heading_error_rad: float
    heading_error_deg: float
    target_yaw_deg: float
    clipped_cos: float
    forward_scale: float


class ObjectDetector(Protocol):
    def detect(self, image: Image.Image, prompt: str) -> tuple[Detection, ...]:
        ...


def clipped_cos(x_rad: float) -> float:
    """Cosine gated to zero outside [-pi, pi]."""

    if x_rad < -math.pi or x_rad > math.pi:
        return 0.0
    return math.cos(x_rad)


def positive_clipped_cos(x_rad: float) -> float:
    return max(0.0, clipped_cos(x_rad))


def bbox_center_error_rad(bbox_xyxy: tuple[float, float, float, float], width: int, hfov_deg: float) -> float:
    center_x = (bbox_xyxy[0] + bbox_xyxy[2]) * 0.5
    normalized = (center_x - width * 0.5) / max(width * 0.5, 1.0)
    return math.atan(normalized * math.tan(math.radians(hfov_deg) * 0.5))


def bbox_center_x_from_bearing_rad(bearing_rad: float, width: int, hfov_deg: float) -> float:
    half_width = max(width * 0.5, 1.0)
    half_fov = math.radians(hfov_deg) * 0.5
    projected_bearing = clamp(bearing_rad, -half_fov * 0.98, half_fov * 0.98)
    denom = max(math.tan(half_fov), 1e-6)
    normalized = math.tan(projected_bearing) / denom
    return half_width * (1.0 + normalized)


def shift_bbox_center_x(
    bbox_xyxy: tuple[float, float, float, float],
    center_x: float,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    box_width = max(1.0, x1 - x0)
    half_width = box_width * 0.5
    return _clamp_bbox((center_x - half_width, y0, center_x + half_width, y1), image_size)


def target_yaw_from_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    current_yaw_rad: float,
    hfov_deg: float,
) -> float:
    return wrap_pi(current_yaw_rad + bbox_center_error_rad(bbox_xyxy, image_width, hfov_deg))


def command_from_heading_error(
    *,
    heading_error_rad: float,
    current_yaw_rad: float,
    forward_power: float,
    heading_kp: float,
    max_turn_percent: float,
    min_turn_percent: float,
    heading_deadband_deg: float,
    max_abs_output: float,
    reverse_correction: bool,
) -> DriveCommand:
    deadband_rad = math.radians(max(0.0, heading_deadband_deg))
    controlled_error = 0.0
    if abs(heading_error_rad) > deadband_rad:
        controlled_error = math.copysign(abs(heading_error_rad) - deadband_rad, heading_error_rad)
    turn = heading_kp * controlled_error
    if 0.0 < abs(turn) < min_turn_percent:
        turn = math.copysign(min_turn_percent, turn)
    if reverse_correction:
        turn = -turn
    turn = clamp(turn, -abs(max_turn_percent), abs(max_turn_percent))
    cos_value = clipped_cos(heading_error_rad)
    forward_scale = max(0.0, cos_value)
    forward = forward_power * forward_scale
    m1 = int(round(clamp(forward + turn, -max_abs_output, max_abs_output)))
    m2 = int(round(clamp(forward - turn, -max_abs_output, max_abs_output)))
    target_yaw_rad = wrap_pi(current_yaw_rad + heading_error_rad)
    return DriveCommand(
        motor1_percent=m1,
        motor2_percent=m2,
        forward_percent=forward,
        turn_percent=turn,
        heading_error_rad=heading_error_rad,
        heading_error_deg=math.degrees(heading_error_rad),
        target_yaw_deg=math.degrees(target_yaw_rad),
        clipped_cos=cos_value,
        forward_scale=forward_scale,
    )


def command_from_bbox(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    current_yaw_rad: float,
    hfov_deg: float,
    forward_power: float,
    heading_kp: float,
    max_turn_percent: float,
    min_turn_percent: float,
    heading_deadband_deg: float,
    max_abs_output: float,
    reverse_correction: bool,
) -> DriveCommand:
    return command_from_heading_error(
        heading_error_rad=bbox_center_error_rad(bbox_xyxy, image_width, hfov_deg),
        current_yaw_rad=current_yaw_rad,
        forward_power=forward_power,
        heading_kp=heading_kp,
        max_turn_percent=max_turn_percent,
        min_turn_percent=min_turn_percent,
        heading_deadband_deg=heading_deadband_deg,
        max_abs_output=max_abs_output,
        reverse_correction=reverse_correction,
    )


class FlorenceMlxDetector:
    """Florence-2 detector through mlx-vlm.

    The Florence task prompt used here is phrase grounding. MLX-VLM exposes the
    converted model and processor; when available, the processor's Florence
    post-processor is used. The string parser is retained as a fallback because
    mlx-vlm return values have changed across releases.
    """

    def __init__(self, model_path: str, *, max_tokens: int = 256, temperature: float = 0.0) -> None:
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Florence MLX detection requires mlx-vlm. Install with `pip install -U mlx-vlm`."
            ) from exc
        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self.model_path = model_path
        try:
            self.model, self.processor = load(model_path)
        except AttributeError as exc:
            if "forced_bos_token_id" not in str(exc):
                raise
            patched = _patch_florence_forced_bos_token_id()
            if patched == 0:
                raise
            self.model, self.processor = load(model_path)
        try:
            self.config = load_config(model_path)
        except Exception:
            self.config = getattr(self.model, "config", None)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def detect(self, image: Image.Image, prompt: str) -> tuple[Detection, ...]:
        task = FLORENCE_OBJECT_DETECTION_TASK
        text_prompt = task + prompt.strip().rstrip(".") + "."
        started = time.perf_counter()
        formatted_prompt = text_prompt
        try:
            formatted_prompt = self._apply_chat_template(self.processor, self.config, text_prompt, num_images=1)
        except Exception:
            pass

        output = self._generate(
            self.model,
            self.processor,
            formatted_prompt,
            [image],
            max_tokens=self.max_tokens,
            temp=self.temperature,
            verbose=False,
        )
        elapsed_s = time.perf_counter() - started
        raw_text = _generation_to_text(output)
        parsed = _post_process_florence(self.processor, raw_text, task=task, image_size=image.size)
        detections = _detections_from_parsed(parsed, task=task, prompt=prompt, source="florence-mlx", raw=raw_text)
        if detections:
            return detections
        return _parse_florence_loc_tokens(raw_text, image.size, prompt=prompt, source="florence-mlx", elapsed_s=elapsed_s)


class FlorenceTransformersDetector:
    def __init__(self, model_path: str, *, device: str = "auto", max_tokens: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Transformers Florence detection requires torch and transformers."
            ) from exc
        if device == "auto":
            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
        self.dtype = dtype
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = _load_florence_transformers_model(AutoModelForCausalLM, model_path, dtype).to(self.device)
        except AttributeError as exc:
            if not _is_florence_transformers_compat_error(exc):
                raise
            patched = _patch_florence_transformers_compat()
            if patched == 0:
                raise
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = _load_florence_transformers_model(AutoModelForCausalLM, model_path, dtype).to(self.device)
        self.model.eval()
        self.max_tokens = max_tokens

    def detect(self, image: Image.Image, prompt: str) -> tuple[Detection, ...]:
        task = FLORENCE_OBJECT_DETECTION_TASK
        text_prompt = task + prompt.strip().rstrip(".") + "."
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.device, self.dtype)
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_tokens,
                do_sample=False,
                num_beams=3,
            )
        raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = _post_process_florence(self.processor, raw_text, task=task, image_size=image.size)
        return _detections_from_parsed(parsed, task=task, prompt=prompt, source="florence-transformers", raw=raw_text)


class GroundingDinoDetector:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Grounding DINO detection requires torch and transformers."
            ) from exc
        if device == "auto":
            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(self, image: Image.Image, prompt: str) -> tuple[Detection, ...]:
        text = grounding_dino_prompt(prompt)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )
        if not results:
            return ()
        result = results[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        detections: list[Detection] = []
        for index, box in enumerate(boxes):
            try:
                xyxy = tuple(float(v) for v in box.detach().cpu().tolist())
            except AttributeError:
                xyxy = tuple(float(v) for v in box)
            if len(xyxy) != 4:
                continue
            score = 1.0
            if index < len(scores):
                try:
                    score = float(scores[index].detach().cpu().item())
                except AttributeError:
                    score = float(scores[index])
            label = str(labels[index]) if index < len(labels) else text
            detections.append(
                Detection(
                    bbox_xyxy=_clamp_bbox(xyxy, image.size),
                    label=label,
                    score=score,
                    source="grounding-dino",
                    raw=text,
                )
            )
        return tuple(detections)


def grounding_dino_prompt(prompt: str) -> str:
    words = re.findall(r"[a-z0-9]+", prompt.lower())
    stop = {
        "closest",
        "nearest",
        "front",
        "foreground",
        "individual",
        "specific",
        "visible",
        "the",
        "a",
        "an",
        "object",
        "not",
        "table",
        "cluster",
    }
    kept = [word for word in words if word not in stop]
    if not kept:
        kept = words or ["object"]
    phrase = " ".join(kept[:4]).strip()
    return phrase.rstrip(".") + "."


def _generation_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "output_text", "generated_text"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    return str(output)


def _patch_florence_forced_bos_token_id() -> int:
    """Patch a Transformers 5 / Florence remote-code compatibility gap.

    Current mlx-vlm releases require Transformers 5.x. The Florence remote
    config loaded by the converted MLX checkpoint can still assume the older
    config object has an instance-level ``forced_bos_token_id``. The failed
    first import leaves the class in ``sys.modules``; a class default lets the
    retry proceed without editing the downloaded Hugging Face cache.
    """

    patched = 0
    for module in list(sys.modules.values()):
        cls = getattr(module, "Florence2LanguageConfig", None)
        if cls is None:
            continue
        setattr(cls, "forced_bos_token_id", None)
        patched += 1
    return patched


def _patch_transformers_tokenizer_additional_special_tokens() -> int:
    """Restore a tokenizer attribute expected by Florence remote code.

    Florence-2's remote processor reads ``tokenizer.additional_special_tokens``
    directly. Transformers 5 can route missing tokenizer fields through
    ``__getattr__`` and raise even though the data is still present in the
    special-token maps. A base-class property keeps the remote code compatible
    without mutating the downloaded model files.
    """

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return 0

    if isinstance(getattr(PreTrainedTokenizerBase, "additional_special_tokens", None), property):
        return 0

    def _as_token_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _get_additional_special_tokens(tokenizer: Any) -> list[Any]:
        for attr_name in ("_additional_special_tokens", "additional_special_tokens"):
            if attr_name in tokenizer.__dict__:
                return _as_token_list(tokenizer.__dict__[attr_name])
        for map_name in ("_special_tokens_map", "init_kwargs"):
            token_map = tokenizer.__dict__.get(map_name)
            if isinstance(token_map, dict) and "additional_special_tokens" in token_map:
                return _as_token_list(token_map["additional_special_tokens"])
        return []

    setattr(PreTrainedTokenizerBase, "additional_special_tokens", property(_get_additional_special_tokens))
    return 1


def _patch_florence_transformers_compat() -> int:
    return _patch_florence_forced_bos_token_id() + _patch_transformers_tokenizer_additional_special_tokens()


def _load_florence_transformers_model(auto_model: Any, model_path: str, dtype: Any) -> Any:
    try:
        return auto_model.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        return auto_model.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )


def _is_florence_transformers_compat_error(exc: AttributeError) -> bool:
    text = str(exc)
    return "forced_bos_token_id" in text or "additional_special_tokens" in text


def _post_process_florence(processor: Any, text: str, *, task: str, image_size: tuple[int, int]) -> Any:
    post_process = getattr(processor, "post_process_generation", None)
    if post_process is None:
        return None
    try:
        return post_process(text, task=task, image_size=image_size)
    except Exception:
        return None


def _detections_from_parsed(
    parsed: Any,
    *,
    task: str,
    prompt: str,
    source: str,
    raw: str,
) -> tuple[Detection, ...]:
    if not isinstance(parsed, dict):
        return ()
    payload = parsed.get(task) if task in parsed else parsed
    if not isinstance(payload, dict):
        return ()
    bboxes = payload.get("bboxes")
    labels = payload.get("labels") or []
    if not isinstance(bboxes, list):
        return ()
    detections: list[Detection] = []
    for i, bbox in enumerate(bboxes):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            xyxy = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        label = str(labels[i]) if i < len(labels) else prompt
        score = _label_score(label, prompt)
        detections.append(Detection(bbox_xyxy=xyxy, label=label, score=score, source=source, raw=raw))
    return tuple(detections)


_LOC_RE = re.compile(r"<loc_(\d+)>")


def _parse_florence_loc_tokens(
    text: str,
    image_size: tuple[int, int],
    *,
    prompt: str,
    source: str,
    elapsed_s: float,
) -> tuple[Detection, ...]:
    del elapsed_s
    values = [int(match.group(1)) for match in _LOC_RE.finditer(text)]
    if len(values) < 4:
        return ()
    width, height = image_size
    detections: list[Detection] = []
    for i in range(0, len(values) - 3, 4):
        x0 = values[i] / 999.0 * width
        y0 = values[i + 1] / 999.0 * height
        x1 = values[i + 2] / 999.0 * width
        y1 = values[i + 3] / 999.0 * height
        bbox = _clamp_bbox((x0, y0, x1, y1), image_size)
        if _bbox_area(bbox) <= 1.0:
            continue
        detections.append(Detection(bbox_xyxy=bbox, label=prompt, score=0.5, source=source, raw=text))
    return tuple(detections)


def _label_score(label: str, prompt: str) -> float:
    label_words = set(re.findall(r"[a-z0-9]+", label.lower()))
    prompt_words = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    if not prompt_words:
        return 0.5
    if label_words & prompt_words:
        return 1.0
    return 0.6


def _clamp_bbox(bbox: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    x0, y0, x1, y1 = bbox
    lo_x, hi_x = sorted((x0, x1))
    lo_y, hi_y = sorted((y0, y1))
    return (
        clamp(lo_x, 0.0, max(0.0, width - 1.0)),
        clamp(lo_y, 0.0, max(0.0, height - 1.0)),
        clamp(hi_x, 0.0, max(0.0, width - 1.0)),
        clamp(hi_y, 0.0, max(0.0, height - 1.0)),
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


class AsyncDetector:
    def __init__(self, detector: ObjectDetector, prompt: str) -> None:
        self.detector = detector
        self.prompt = prompt
        self._jobs: queue.Queue[FrameState | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[DetectionResult] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="object-drive-detector", daemon=True)
        self._pending = False
        self._closed = False
        self._thread.start()

    @property
    def pending(self) -> bool:
        return self._pending

    def submit_latest(self, frame_state: FrameState) -> bool:
        if self._closed or self._pending:
            return False
        try:
            self._jobs.put_nowait(frame_state)
        except queue.Full:
            return False
        self._pending = True
        return True

    def get_results(self) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            results.append(result)
            self._pending = False
        return results

    def close(self, *, timeout_s: float = 2.0) -> None:
        self._closed = True
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._pending and time.monotonic() < deadline:
            self.get_results()
            if not self._pending:
                break
            time.sleep(0.05)
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        remaining_s = max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining_s)

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            started = time.perf_counter()
            try:
                detections = self.detector.detect(job.image, self.prompt)
                result = DetectionResult(
                    frame_seq=job.frame.seq,
                    received_ns=job.frame.received_ns,
                    image_size=job.image.size,
                    detections=detections,
                    elapsed_s=time.perf_counter() - started,
                )
            except Exception as exc:
                result = DetectionResult(
                    frame_seq=job.frame.seq,
                    received_ns=job.frame.received_ns,
                    image_size=job.image.size,
                    detections=(),
                    elapsed_s=time.perf_counter() - started,
                    error=str(exc),
                )
            self._results.put(result)


class KltBoxTracker:
    def __init__(self, *, max_points: int = 80) -> None:
        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            cv2 = None
        self.cv2 = cv2
        self.max_points = max_points
        self.bbox: tuple[float, float, float, float] | None = None
        self.points: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.last_seq: int | None = None
        self.source = "none"
        self.last_success = False
        self.last_good_points = 0

    def reset(self, frame: FrameState, bbox: tuple[float, float, float, float], *, source: str) -> tuple[float, float, float, float]:
        self.bbox = _clamp_bbox(bbox, frame.image.size)
        self.prev_gray = frame.gray
        self.points = self._points_in_bbox(frame.gray, self.bbox)
        self.last_seq = frame.frame.seq
        self.source = source
        self.last_success = False
        self.last_good_points = int(len(self.points)) if self.points is not None else 0
        return self.bbox

    def replay(self, frames: list[FrameState]) -> tuple[float, float, float, float] | None:
        current: tuple[float, float, float, float] | None = self.bbox
        for frame in frames:
            current = self.track(frame)
            if current is None:
                break
        return current

    def track(self, frame: FrameState) -> tuple[float, float, float, float] | None:
        if self.bbox is None or self.prev_gray is None or self.last_seq == frame.frame.seq:
            self.last_success = False
            self.last_good_points = 0
            return self.bbox
        if self.cv2 is None or self.points is None or len(self.points) < 4:
            self.prev_gray = frame.gray
            self.last_seq = frame.frame.seq
            self.last_success = False
            self.last_good_points = 0
            return self.bbox

        next_points, status, _err = self.cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame.gray,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            self.prev_gray = frame.gray
            self.last_seq = frame.frame.seq
            self.last_success = False
            self.last_good_points = 0
            return self.bbox
        good_old = self.points[status.reshape(-1) == 1].reshape(-1, 2)
        good_new = next_points[status.reshape(-1) == 1].reshape(-1, 2)
        if len(good_new) < 4:
            self.prev_gray = frame.gray
            self.last_seq = frame.frame.seq
            self.last_success = False
            self.last_good_points = int(len(good_new))
            return self.bbox
        delta = np.median(good_new - good_old, axis=0)
        dx = float(delta[0])
        dy = float(delta[1])
        x0, y0, x1, y1 = self.bbox
        self.bbox = _clamp_bbox((x0 + dx, y0 + dy, x1 + dx, y1 + dy), frame.image.size)
        self.prev_gray = frame.gray
        self.last_seq = frame.frame.seq
        self.points = self._points_in_bbox(frame.gray, self.bbox)
        self.last_success = True
        self.last_good_points = int(len(good_new))
        return self.bbox

    def _points_in_bbox(self, gray: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
        if self.cv2 is None:
            return None
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        x0 = max(0, min(gray.shape[1] - 1, x0))
        x1 = max(0, min(gray.shape[1] - 1, x1))
        y0 = max(0, min(gray.shape[0] - 1, y0))
        y1 = max(0, min(gray.shape[0] - 1, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y0 : y1 + 1, x0 : x1 + 1] = 255
        points = self.cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_points,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
            blockSize=7,
        )
        return points


class TargetBearingKalman:
    """Constant-velocity world-yaw filter for one visible target.

    The state is target world yaw and target angular velocity. A stationary
    object should have near-zero velocity, so robot turns are handled by
    subtracting the current IMU yaw when we need the camera-relative bearing.
    """

    def __init__(self, *, process_noise_deg_s: float, initial_velocity_noise_deg_s: float = 45.0) -> None:
        self.process_noise_rad_s = math.radians(max(0.01, process_noise_deg_s))
        self.initial_velocity_noise_rad_s = math.radians(max(0.1, initial_velocity_noise_deg_s))
        self.x = np.zeros(2, dtype=np.float64)
        self.p = np.eye(2, dtype=np.float64)
        self.last_ns: int | None = None
        self.initialized = False
        self.last_measurement_source = "none"

    def reset(self, *, target_yaw_rad: float, now_ns: int, measurement_std_rad: float, source: str) -> None:
        std = max(math.radians(0.1), measurement_std_rad)
        self.x[:] = (target_yaw_rad, 0.0)
        self.p[:] = np.diag([std * std, self.initial_velocity_noise_rad_s * self.initial_velocity_noise_rad_s])
        self.last_ns = now_ns
        self.initialized = True
        self.last_measurement_source = source

    def clear(self) -> None:
        self.initialized = False
        self.last_ns = None
        self.last_measurement_source = "none"

    def predict(self, *, now_ns: int) -> None:
        if not self.initialized:
            return
        if self.last_ns is None:
            self.last_ns = now_ns
            return
        dt = clamp((now_ns - self.last_ns) / 1_000_000_000.0, 0.0, 0.5)
        if dt <= 0.0:
            return
        transition = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q_pos = self.process_noise_rad_s * dt
        q_vel = self.process_noise_rad_s
        process = np.diag([q_pos * q_pos, q_vel * q_vel])
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process
        self.last_ns = now_ns

    def correct(self, *, target_yaw_rad: float, measurement_std_rad: float, source: str) -> None:
        if not self.initialized:
            raise RuntimeError("TargetBearingKalman.correct called before reset")
        std = max(math.radians(0.1), measurement_std_rad)
        measurement = self.x[0] + wrap_pi(target_yaw_rad - self.x[0])
        residual = measurement - self.x[0]
        innovation = self.p[0, 0] + std * std
        gain = self.p[:, 0] / max(innovation, 1e-12)
        self.x = self.x + gain * residual
        identity = np.eye(2, dtype=np.float64)
        h = np.asarray([[1.0, 0.0]], dtype=np.float64)
        self.p = (identity - gain.reshape(2, 1) @ h) @ self.p
        self.last_measurement_source = source

    def bearing_error_rad(self, current_yaw_rad: float) -> float:
        if not self.initialized:
            return 0.0
        return wrap_pi(float(self.x[0]) - current_yaw_rad)

    @property
    def target_yaw_rad(self) -> float:
        return float(self.x[0])

    @property
    def uncertainty_deg(self) -> float:
        if not self.initialized:
            return float("inf")
        return math.degrees(math.sqrt(max(0.0, float(self.p[0, 0]))))


class ObjectDriveRerunLogger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.enabled = bool(args.rerun or args.rerun_save or args.rerun_grpc)
        self.rr: Any | None = None
        self.recordings: list[Any] = []
        self.save_path: Path | None = args.rerun_save
        self.step_count = 0
        self.image_count = 0
        self.bbox_count = 0
        if not self.enabled:
            return
        import rerun as rr

        self.rr = rr
        rr.init("flatdisk_object_drive", spawn=False)
        default_recording = rr.get_global_data_recording()
        if default_recording is not None and hasattr(default_recording, "set_sinks"):
            self.recordings.append(default_recording)
            sinks = []
            if args.rerun and not args.rerun_no_spawn:
                rr.spawn(recording=default_recording)
                sinks.append(rr.GrpcSink())
                print("Object-drive Rerun live viewer stream enabled", flush=True)
            if args.rerun_save is not None:
                args.rerun_save.parent.mkdir(parents=True, exist_ok=True)
                sinks.append(rr.FileSink(args.rerun_save))
                print(f"Object-drive Rerun save path: {args.rerun_save}", flush=True)
            if args.rerun_grpc:
                sinks.append(rr.GrpcSink(args.rerun_grpc))
                print(f"Object-drive Rerun gRPC stream: {args.rerun_grpc}", flush=True)
            if sinks:
                default_recording.set_sinks(*sinks)
            self._log("object_drive/config", rr.TextLog(json.dumps(_serializable_args(args), sort_keys=True)), static=True)
            return

        if args.rerun_save is not None:
            args.rerun_save.parent.mkdir(parents=True, exist_ok=True)
            rr.save(args.rerun_save, recording=default_recording)
            self.recordings.append(default_recording)
            print(f"Object-drive Rerun save path: {args.rerun_save}", flush=True)
        if args.rerun_grpc:
            grpc_recording = default_recording
            if self.recordings:
                grpc_recording = rr.new_recording("flatdisk_object_drive_grpc") if hasattr(rr, "new_recording") else default_recording
            rr.connect_grpc(args.rerun_grpc, recording=grpc_recording)
            self.recordings.append(grpc_recording)
            print(f"Object-drive Rerun gRPC stream: {args.rerun_grpc}", flush=True)
        if args.rerun and not args.rerun_no_spawn:
            live_recording = default_recording
            if self.recordings:
                live_recording = rr.new_recording("flatdisk_object_drive_live") if hasattr(rr, "new_recording") else default_recording
            rr.spawn(recording=live_recording)
            self.recordings.append(live_recording)
            print("Object-drive Rerun live viewer stream enabled", flush=True)
        if not self.recordings and default_recording is not None:
            self.recordings.append(default_recording)
        self._log("object_drive/config", rr.TextLog(json.dumps(_serializable_args(args), sort_keys=True)), static=True)

    def close(self) -> None:
        if self.rr is not None:
            for recording in self.recordings:
                try:
                    recording.flush(blocking=True)
                except TypeError:
                    try:
                        recording.flush()
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    recording.disconnect()
                except Exception:
                    pass
            if self.enabled:
                target = f" to {self.save_path}" if self.save_path is not None else ""
                print(
                    f"Object-drive Rerun logged {self.step_count} steps, "
                    f"{self.image_count} images, {self.bbox_count} boxes{target}",
                    flush=True,
                )

    def _set_time(self, *, elapsed_s: float, frame_seq: int | None) -> None:
        if self.rr is None:
            return
        rr = self.rr
        for recording in self.recordings:
            if hasattr(rr, "set_time"):
                rr.set_time("object_drive_time", duration=elapsed_s, recording=recording)
                if frame_seq is not None:
                    rr.set_time("video_seq", sequence=frame_seq, recording=recording)
            else:
                rr.set_time_seconds("object_drive_time", elapsed_s, recording=recording)
                if frame_seq is not None:
                    rr.set_time_sequence("video_seq", frame_seq, recording=recording)

    def _log(self, entity_path: str, entity: Any, *extra: Any, static: bool = False) -> None:
        if self.rr is None:
            return
        for recording in self.recordings:
            self.rr.log(entity_path, entity, *extra, static=static, recording=recording)

    def _scalar(self, value: float | int) -> Any:
        if self.rr is None:
            return value
        scalar_type = getattr(self.rr, "Scalars", None) or getattr(self.rr, "Scalar")
        return scalar_type(value)

    def log_step(
        self,
        *,
        elapsed_s: float,
        frame_state: FrameState | None,
        detection: Detection | None,
        command: DriveCommand | None,
        published: bool,
        detection_age_s: float | None,
        detector_pending: bool,
        detection_error: str | None,
        target_filter_source: str | None = None,
        target_uncertainty_deg: float | None = None,
        robot_yaw_delta_deg: float | None = None,
    ) -> None:
        if self.rr is None:
            return
        rr = self.rr
        self.step_count += 1
        self._set_time(elapsed_s=elapsed_s, frame_seq=frame_state.frame.seq if frame_state is not None else None)
        if frame_state is not None:
            self._log("object_drive/camera/image", rr.Image(np.asarray(frame_state.image)))
            self._log("object_drive/imu/yaw_deg", self._scalar(frame_state.imu.yaw_deg))
            self.image_count += 1
        if detection is not None and frame_state is not None:
            x0, y0, x1, y1 = detection.bbox_xyxy
            self._log(
                "object_drive/camera/image/bbox",
                rr.Boxes2D(
                    array=np.asarray([[x0, y0, x1, y1]], dtype=np.float32),
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[detection.label],
                    show_labels=True,
                ),
            )
            self._log("object_drive/detection/center_x", self._scalar(detection.center_x))
            self._log("object_drive/detection/score", self._scalar(detection.score))
            self._log("object_drive/detection/source", rr.TextLog(detection.source))
            self.bbox_count += 1
        if command is not None:
            self._log("object_drive/command/motor1_percent", self._scalar(command.motor1_percent))
            self._log("object_drive/command/motor2_percent", self._scalar(command.motor2_percent))
            self._log("object_drive/command/forward_percent", self._scalar(command.forward_percent))
            self._log("object_drive/command/turn_percent", self._scalar(command.turn_percent))
            self._log("object_drive/command/heading_error_deg", self._scalar(command.heading_error_deg))
            self._log("object_drive/command/target_yaw_deg", self._scalar(command.target_yaw_deg))
            self._log("object_drive/command/clipped_cos", self._scalar(command.clipped_cos))
            self._log("object_drive/command/forward_scale", self._scalar(command.forward_scale))
        self._log("object_drive/state/published", self._scalar(1 if published else 0))
        self._log("object_drive/state/detector_pending", self._scalar(1 if detector_pending else 0))
        if detection_age_s is not None:
            self._log("object_drive/detection/age_s", self._scalar(detection_age_s))
        if detection_error:
            self._log("object_drive/detection/error", rr.TextLog(detection_error))
        if target_filter_source:
            self._log("object_drive/target_filter/source", rr.TextLog(target_filter_source))
        if target_uncertainty_deg is not None and math.isfinite(target_uncertainty_deg):
            self._log("object_drive/target_filter/uncertainty_deg", self._scalar(target_uncertainty_deg))
        if robot_yaw_delta_deg is not None:
            self._log("object_drive/imu/yaw_delta_deg", self._scalar(robot_yaw_delta_deg))


class ObjectDriveRunner:
    def __init__(self, args: argparse.Namespace, detector: ObjectDetector | None = None) -> None:
        self.args = args
        self.detector = detector if detector is not None else build_detector(args)
        self.robot = FlatDiskRobotClient(
            namespace=args.namespace,
            mode=args.mode,
            connect=args.connect,
            listen=args.listen,
            reverse_yaw=args.reverse_yaw,
            reverse_correction=args.reverse_correction,
            heading_kp=args.heading_kp,
            max_turn_percent=args.max_turn_percent,
            min_turn_percent=args.min_turn_percent,
            heading_deadband_deg=args.heading_deadband_deg,
            imu_timeout_s=args.imu_timeout,
            control_hz=args.control_hz,
            rotate_frames_180=args.rotate_180,
        )
        self.rerun = ObjectDriveRerunLogger(args)
        self.stop_requested = False
        self.frame_buffer: deque[FrameState] = deque(maxlen=max(4, int(args.frame_buffer_s * args.camera_hz_hint)))
        self.tracker = KltBoxTracker()
        self.target_filter = TargetBearingKalman(process_noise_deg_s=args.target_process_noise_deg_s)
        self.active_detection: Detection | None = None
        self.active_detection_frame_ns: int | None = None
        self.last_visual_correction_ns: int | None = None
        self.latest_detection_error: str | None = None
        self.status_sub: Any | None = None
        self.latest_status: dict[str, Any] | None = None
        self.camera_hfov_deg = float(args.camera_hfov_deg)
        self.last_frame_seq: int | None = None
        self.last_imu_yaw_rad: float | None = None
        self.last_robot_yaw_delta_deg: float | None = None
        self.target_filter_source = "none"
        self.last_command: DriveCommand | None = None
        self.publish_count = 0
        self.prediction_count = 0
        self.track_count = 0
        self.imu_predict_count = 0
        self.lost_count = 0
        self.overlay_count = 0
        self.raw_frame_count = 0
        self.next_report_ns = time.monotonic_ns() + 1_000_000_000

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        detector = AsyncDetector(self.detector, self.args.prompt)
        start_ns = time.monotonic_ns()
        active_start_ns: int | None = None
        first_detection_deadline_ns = start_ns + int(self.args.initial_detection_timeout * 1_000_000_000.0)
        next_detect_ns = start_ns
        try:
            self.robot.open()
            if self.robot.session is not None:
                self.status_sub = self.robot.session.declare_subscriber(f"{self.args.namespace}/status")
            print(
                f"object-drive namespace={self.args.namespace} connect={self.args.connect or '-'} "
                f"prompt={self.args.prompt!r} duration={self.args.duration:.2f}s power={self.args.forward_power:.1f}% "
                f"detector={self.args.detector} armed={self.args.arm}",
                flush=True,
            )
            if not self.args.arm:
                print("Not armed: detecting and logging only. Add --arm to publish motor commands.", flush=True)
            self._wait_for_startup()
            while not self.stop_requested:
                now_ns = time.monotonic_ns()
                elapsed_s = (now_ns - start_ns) / 1_000_000_000.0
                active_elapsed_s = 0.0 if active_start_ns is None else (now_ns - active_start_ns) / 1_000_000_000.0

                frame_state = self._poll_frame_state(now_ns)
                self._poll_status()
                if frame_state is not None:
                    self.frame_buffer.append(frame_state)
                    self._update_robot_motion(frame_state)
                    if self.last_frame_seq is None or frame_state.frame.seq != self.last_frame_seq:
                        self._track_latest(frame_state)
                        self.last_frame_seq = frame_state.frame.seq
                    active_remaining_s = float("inf") if active_start_ns is None else self.args.duration - active_elapsed_s
                    can_submit_detection = active_remaining_s >= self.args.min_detection_remaining_s
                    if now_ns >= next_detect_ns and not detector.pending and can_submit_detection:
                        if detector.submit_latest(frame_state):
                            next_detect_ns = now_ns + int(self.args.detect_interval * 1_000_000_000.0)

                result_count = self._consume_detection_results(detector)
                if active_start_ns is None:
                    if result_count > 0:
                        active_start_ns = now_ns
                    elif now_ns >= first_detection_deadline_ns and not detector.pending:
                        active_start_ns = now_ns
                if active_start_ns is not None and (now_ns - active_start_ns) / 1_000_000_000.0 >= self.args.duration:
                    break
                command, detection_age_s = self._make_command(frame_state)
                published = False
                if command is not None and self.args.arm:
                    self.robot.publish_percent(command.motor1_percent, command.motor2_percent)
                    self.publish_count += 1
                    published = True
                elif self.args.arm and self.args.stop_when_lost:
                    self.robot.publish_percent(0.0, 0.0)
                    published = True

                self.last_command = command
                self.rerun.log_step(
                    elapsed_s=elapsed_s,
                    frame_state=frame_state,
                    detection=self.active_detection,
                    command=command,
                    published=published,
                    detection_age_s=detection_age_s,
                    detector_pending=detector.pending,
                    detection_error=self.latest_detection_error,
                    target_filter_source=self.target_filter_source if self.target_filter.initialized else None,
                    target_uncertainty_deg=self.target_filter.uncertainty_deg if self.target_filter.initialized else None,
                    robot_yaw_delta_deg=self.last_robot_yaw_delta_deg,
                )
                self._save_overlay(frame_state, command)
                self._save_raw_frame(frame_state)
                self._report(now_ns, detector_pending=detector.pending)
                time.sleep(1.0 / max(self.args.control_hz, 1.0))
        finally:
            detector.close(timeout_s=self.args.detector_shutdown_timeout)
            try:
                if self.args.stop_on_exit:
                    self.robot.stop()
            finally:
                if self.status_sub is not None:
                    try:
                        self.status_sub.undeclare()
                    except Exception:
                        pass
                self.robot.close()
                self.rerun.close()
        return 0

    def _wait_for_startup(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() <= deadline:
            try:
                self.robot.latest_frame(timeout_s=0.2)
                self.robot.wait_for_imu(timeout_s=0.2)
                return
            except TimeoutError:
                time.sleep(0.05)
        raise TimeoutError(f"no camera/IMU startup samples within {self.args.startup_timeout:.1f}s")

    def _poll_frame_state(self, now_ns: int) -> FrameState | None:
        self.robot.poll()
        frame = self.robot.last_frame
        imu = self.robot.last_imu
        if frame is None or imu is None:
            return None
        if self.robot._imu_age_s(imu) > self.args.imu_timeout:
            return None
        image = frame.image(rotate_180=self.robot.rotate_frames_180)
        gray = _image_gray(image)
        return FrameState(frame=frame, imu=imu, image=image, gray=gray, monotonic_ns=now_ns)

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
        self.latest_status = latest
        value = latest.get("camera_horizontal_fov_deg")
        if isinstance(value, (int, float)) and 1.0 <= float(value) <= 179.0:
            self.camera_hfov_deg = float(value)

    def _update_robot_motion(self, frame_state: FrameState) -> None:
        if self.last_imu_yaw_rad is None:
            self.last_robot_yaw_delta_deg = 0.0
        else:
            self.last_robot_yaw_delta_deg = math.degrees(wrap_pi(frame_state.imu.yaw_rad - self.last_imu_yaw_rad))
        self.last_imu_yaw_rad = frame_state.imu.yaw_rad

    def _measurement_std_rad(self, bearing_noise_deg: float) -> float:
        imu_noise = math.radians(max(0.0, self.args.imu_heading_noise_deg))
        bearing_noise = math.radians(max(0.1, bearing_noise_deg))
        return math.hypot(imu_noise, bearing_noise)

    def _correct_target_filter(
        self,
        frame_state: FrameState,
        bbox: tuple[float, float, float, float],
        *,
        source: str,
        bearing_noise_deg: float,
    ) -> None:
        if not self.args.target_filter:
            return
        target_yaw = target_yaw_from_bbox(
            bbox,
            image_width=frame_state.image.width,
            current_yaw_rad=frame_state.imu.yaw_rad,
            hfov_deg=self.camera_hfov_deg,
        )
        measurement_std = self._measurement_std_rad(bearing_noise_deg)
        if not self.target_filter.initialized:
            self.target_filter.reset(
                target_yaw_rad=target_yaw,
                now_ns=frame_state.monotonic_ns,
                measurement_std_rad=measurement_std,
                source=source,
            )
        else:
            self.target_filter.predict(now_ns=frame_state.monotonic_ns)
            self.target_filter.correct(target_yaw_rad=target_yaw, measurement_std_rad=measurement_std, source=source)
        self.last_visual_correction_ns = frame_state.monotonic_ns
        self.target_filter_source = source

    def _target_filter_bbox(
        self,
        frame_state: FrameState,
        reference_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float] | None:
        if not self.args.target_filter or not self.target_filter.initialized:
            return None
        bearing = self.target_filter.bearing_error_rad(frame_state.imu.yaw_rad)
        center_x = bbox_center_x_from_bearing_rad(bearing, frame_state.image.width, self.camera_hfov_deg)
        return shift_bbox_center_x(reference_bbox, center_x, frame_state.image.size)

    def _track_latest(self, frame_state: FrameState) -> None:
        if self.active_detection is None:
            return
        if self.args.target_filter and self.target_filter.initialized:
            self.target_filter.predict(now_ns=frame_state.monotonic_ns)
        bbox = self.tracker.track(frame_state)
        if bbox is None:
            return
        if self.tracker.last_success:
            self._correct_target_filter(
                frame_state,
                bbox,
                source="klt-track",
                bearing_noise_deg=self.args.track_bearing_noise_deg,
            )
            filtered_bbox = self._target_filter_bbox(frame_state, bbox)
            if filtered_bbox is not None:
                bbox = filtered_bbox
            source = "klt-kalman" if self.args.target_filter else "klt-track"
            self.last_visual_correction_ns = frame_state.monotonic_ns
            self.track_count += 1
        else:
            filtered_bbox = self._target_filter_bbox(frame_state, bbox)
            if filtered_bbox is not None:
                bbox = filtered_bbox
                source = "imu-kalman"
                self.imu_predict_count += 1
                self.target_filter_source = source
            else:
                source = "klt-hold"
        self.active_detection = Detection(
            bbox_xyxy=bbox,
            label=self.active_detection.label,
            score=max(0.01, self.active_detection.score * (0.995 if self.tracker.last_success else 0.985)),
            source=source,
            raw=self.active_detection.raw,
        )
        self.active_detection_frame_ns = frame_state.monotonic_ns

    def _consume_detection_results(self, detector: AsyncDetector) -> int:
        result_count = 0
        for result in detector.get_results():
            result_count += 1
            self.latest_detection_error = result.error
            if result.error:
                self.lost_count += 1
                continue
            self.prediction_count += 1
            best = select_detection(
                result.detections,
                prompt=self.args.prompt,
                image_size=result.image_size,
                max_area_fraction=self.args.max_bbox_area_fraction,
            )
            if best is None:
                self.lost_count += 1
                continue
            origin_index = None
            for index, frame_state in enumerate(self.frame_buffer):
                if frame_state.frame.seq == result.frame_seq:
                    origin_index = index
                    break
            if origin_index is None:
                self.active_detection = best
                self.active_detection_frame_ns = time.monotonic_ns()
                self.last_visual_correction_ns = self.active_detection_frame_ns
                self.target_filter_source = best.source
                continue
            origin = list(self.frame_buffer)[origin_index]
            self.tracker.reset(origin, best.bbox_xyxy, source=best.source)
            self._correct_target_filter(
                origin,
                best.bbox_xyxy,
                source=best.source,
                bearing_noise_deg=self.args.model_bearing_noise_deg,
            )
            current_bbox = best.bbox_xyxy
            current_source = best.source
            for replay_frame in list(self.frame_buffer)[origin_index + 1 :]:
                if self.args.target_filter and self.target_filter.initialized:
                    self.target_filter.predict(now_ns=replay_frame.monotonic_ns)
                replay_bbox = self.tracker.track(replay_frame)
                if replay_bbox is None:
                    continue
                if self.tracker.last_success:
                    self._correct_target_filter(
                        replay_frame,
                        replay_bbox,
                        source="klt-replay",
                        bearing_noise_deg=self.args.track_bearing_noise_deg,
                    )
                    current_bbox = self._target_filter_bbox(replay_frame, replay_bbox) or replay_bbox
                    current_source = "klt-kalman" if self.args.target_filter else "klt-track"
                else:
                    current_bbox = self._target_filter_bbox(replay_frame, current_bbox) or replay_bbox
                    current_source = "imu-kalman" if self.args.target_filter and self.target_filter.initialized else "klt-hold"
            self.active_detection = Detection(
                bbox_xyxy=current_bbox,
                label=best.label,
                score=best.score,
                source=current_source,
                raw=best.raw,
            )
            self.active_detection_frame_ns = (list(self.frame_buffer)[-1].monotonic_ns if self.frame_buffer else origin.monotonic_ns)
        return result_count

    def _make_command(self, frame_state: FrameState | None) -> tuple[DriveCommand | None, float | None]:
        if frame_state is None or self.active_detection is None:
            return None, None
        detection_age_s = None
        visual_anchor_ns = self.last_visual_correction_ns or self.active_detection_frame_ns
        if visual_anchor_ns is not None:
            detection_age_s = (frame_state.monotonic_ns - visual_anchor_ns) / 1_000_000_000.0
            if detection_age_s > self.args.max_track_age:
                self.active_detection = None
                self.last_visual_correction_ns = None
                self.target_filter.clear()
                self.target_filter_source = "none"
                self.lost_count += 1
                return None, detection_age_s
        if self.args.target_filter and self.target_filter.initialized:
            heading_error_rad = self.target_filter.bearing_error_rad(frame_state.imu.yaw_rad)
            command = command_from_heading_error(
                heading_error_rad=heading_error_rad,
                current_yaw_rad=frame_state.imu.yaw_rad,
                forward_power=self.args.forward_power,
                heading_kp=self.args.heading_kp,
                max_turn_percent=self.args.max_turn_percent,
                min_turn_percent=self.args.min_turn_percent,
                heading_deadband_deg=self.args.heading_deadband_deg,
                max_abs_output=self.args.max_abs_output,
                reverse_correction=self.args.reverse_correction,
            )
        else:
            command = command_from_bbox(
                bbox_xyxy=self.active_detection.bbox_xyxy,
                image_width=frame_state.image.width,
                current_yaw_rad=frame_state.imu.yaw_rad,
                hfov_deg=self.camera_hfov_deg,
                forward_power=self.args.forward_power,
                heading_kp=self.args.heading_kp,
                max_turn_percent=self.args.max_turn_percent,
                min_turn_percent=self.args.min_turn_percent,
                heading_deadband_deg=self.args.heading_deadband_deg,
                max_abs_output=self.args.max_abs_output,
                reverse_correction=self.args.reverse_correction,
            )
        return command, detection_age_s

    def _save_overlay(self, frame_state: FrameState | None, command: DriveCommand | None) -> None:
        if frame_state is None or self.args.overlay_dir is None:
            return
        every = max(1, int(self.args.overlay_every))
        if frame_state.frame.seq % every != 0:
            return
        self.overlay_count += 1
        path = self.args.overlay_dir / f"{self.overlay_count:04d}_seq{frame_state.frame.seq}.jpg"
        save_debug_overlay(path, frame_state.image, self.active_detection, command)

    def _save_raw_frame(self, frame_state: FrameState | None) -> None:
        if frame_state is None or self.args.raw_frame_dir is None:
            return
        every = max(1, int(self.args.raw_frame_every))
        if frame_state.frame.seq % every != 0:
            return
        self.raw_frame_count += 1
        path = self.args.raw_frame_dir / f"{self.raw_frame_count:04d}_seq{frame_state.frame.seq}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_state.image.save(path, format="JPEG", quality=90)

    def _report(self, now_ns: int, *, detector_pending: bool) -> None:
        if now_ns < self.next_report_ns:
            return
        cmd = self.last_command
        if cmd is None:
            cmd_text = "none"
            err_text = "nan"
        else:
            cmd_text = f"{cmd.motor1_percent}/{cmd.motor2_percent}%"
            err_text = f"{cmd.heading_error_deg:.1f}deg"
        det = self.active_detection
        det_text = "none" if det is None else f"{det.label}:{det.source}:{det.score:.2f}"
        filter_text = "off"
        if self.args.target_filter and self.target_filter.initialized:
            filter_text = f"{self.target_filter_source}:{self.target_filter.uncertainty_deg:.1f}deg"
        print(
            f"object-drive armed={self.args.arm} pred={self.prediction_count} track={self.track_count} "
            f"imu_pred={self.imu_predict_count} filter={filter_text} "
            f"pub={self.publish_count} cmd={cmd_text} heading_error={err_text} det={det_text} "
            f"pending={detector_pending} lost={self.lost_count}",
            flush=True,
        )
        self.next_report_ns = now_ns + 1_000_000_000


def _image_gray(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    return np.ascontiguousarray(arr)


def build_detector(args: argparse.Namespace) -> ObjectDetector:
    if args.detector == "florence-mlx":
        return FlorenceMlxDetector(args.model, max_tokens=args.max_tokens, temperature=args.temperature)
    if args.detector == "florence-transformers":
        return FlorenceTransformersDetector(args.transformers_model, device=args.device, max_tokens=args.max_tokens)
    if args.detector == "grounding-dino":
        return GroundingDinoDetector(
            args.grounding_dino_model,
            device=args.device,
            box_threshold=args.grounding_dino_box_threshold,
            text_threshold=args.grounding_dino_text_threshold,
        )
    raise ValueError(f"unsupported detector: {args.detector}")


def unique_rerun_save_path(prompt: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prompt.lower())[:48]
    return Path("captures/object-drive") / f"{timestamp}-{secrets.token_hex(4)}-{safe_prompt or 'object'}.rrd"


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def save_debug_overlay(path: Path, image: Image.Image, detection: Detection | None, command: DriveCommand | None) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if detection is not None:
        draw.rectangle(detection.bbox_xyxy, outline=(255, 48, 48), width=3)
        draw.text((detection.bbox_xyxy[0] + 4, detection.bbox_xyxy[1] + 4), detection.label, fill=(255, 48, 48))
    if command is not None:
        draw.text((8, 8), f"cmd {command.motor1_percent}/{command.motor2_percent}% err {command.heading_error_deg:.1f}", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="Visible object phrase, e.g. 'chair' or 'toilet'.")
    parser.add_argument("--duration", type=float, required=True, help="Total controller runtime in seconds.")
    parser.add_argument("--forward-power", type=float, required=True, help="Base forward motor percent before cosine scaling.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--mode", default="client")
    parser.add_argument("--listen", default="")
    parser.add_argument("--connect", default=DEFAULT_CONNECT)
    parser.add_argument("--arm", action="store_true", help="Actually publish motor commands.")
    parser.add_argument("--no-stop-on-exit", dest="stop_on_exit", action="store_false")
    parser.add_argument(
        "--detector",
        choices=("florence-mlx", "florence-transformers", "grounding-dino"),
        default="florence-mlx",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX Florence model repo/path.")
    parser.add_argument("--transformers-model", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="Device for transformer-backed detectors.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--detect-interval", type=float, default=0.6)
    parser.add_argument(
        "--initial-detection-timeout",
        type=float,
        default=180.0,
        help="Wait up to this many seconds for the first detector result before starting the active duration.",
    )
    parser.add_argument(
        "--detector-shutdown-timeout",
        type=float,
        default=180.0,
        help="Wait up to this many seconds for an in-flight detector job before process shutdown.",
    )
    parser.add_argument(
        "--min-detection-remaining-s",
        type=float,
        default=5.0,
        help="Only submit a new detector job if this much active servo time remains.",
    )
    parser.add_argument("--camera-hz-hint", type=float, default=10.0)
    parser.add_argument("--frame-buffer-s", type=float, default=4.0)
    parser.add_argument("--max-track-age", type=float, default=2.5)
    parser.add_argument("--target-filter", dest="target_filter", action="store_true")
    parser.add_argument("--no-target-filter", dest="target_filter", action="store_false")
    parser.add_argument(
        "--imu-heading-noise-deg",
        type=float,
        default=2.0,
        help="Assumed 1-sigma IMU yaw noise for target-bearing filtering.",
    )
    parser.add_argument(
        "--model-bearing-noise-deg",
        type=float,
        default=4.0,
        help="Assumed 1-sigma bbox bearing noise for model detections.",
    )
    parser.add_argument(
        "--track-bearing-noise-deg",
        type=float,
        default=8.0,
        help="Assumed 1-sigma bbox bearing noise for KLT updates.",
    )
    parser.add_argument(
        "--target-process-noise-deg-s",
        type=float,
        default=10.0,
        help="Target world-yaw process noise; raise for moving targets or heavy drift.",
    )
    parser.add_argument(
        "--max-bbox-area-fraction",
        type=float,
        default=0.75,
        help="Reject VLM boxes covering more than this frame fraction. Use <=0 to disable.",
    )
    parser.add_argument("--heading-kp", type=float, default=18.0)
    parser.add_argument("--max-turn-percent", type=float, default=16.0)
    parser.add_argument("--min-turn-percent", type=float, default=1.5)
    parser.add_argument("--heading-deadband-deg", type=float, default=1.0)
    parser.add_argument("--max-abs-output", type=float, default=50.0)
    parser.add_argument("--camera-hfov-deg", type=float, default=DEFAULT_HFOV_DEG)
    parser.add_argument("--imu-timeout", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--stop-when-lost", action="store_true", help="Publish 0/0 while armed and no fresh tracked bbox exists.")
    parser.add_argument("--reverse-yaw", dest="reverse_yaw", action="store_true")
    parser.add_argument("--no-reverse-yaw", dest="reverse_yaw", action="store_false")
    parser.add_argument("--reverse-correction", action="store_true")
    parser.add_argument("--no-rotate-180", dest="rotate_180", action="store_false")
    parser.add_argument("--rerun", action="store_true", help="Spawn Rerun viewer.")
    parser.add_argument(
        "--rerun-save",
        type=Path,
        nargs="?",
        const=AUTO_RERUN_SAVE,
        default=None,
        help="Save a Rerun .rrd. Omit path to use captures/object-drive/*.rrd.",
    )
    parser.add_argument("--rerun-grpc", default="", help="Connect to an existing Rerun gRPC endpoint.")
    parser.add_argument("--rerun-no-spawn", action="store_true")
    parser.add_argument("--overlay-dir", type=Path, default=None, help="Write robot POV JPEG overlays with the active bbox.")
    parser.add_argument("--overlay-every", type=int, default=3, help="Save one overlay every N camera frame sequence numbers.")
    parser.add_argument("--raw-frame-dir", type=Path, default=None, help="Write raw robot POV JPEG frames for planner motion strips.")
    parser.add_argument("--raw-frame-every", type=int, default=3, help="Save one raw frame every N camera frame sequence numbers.")
    parser.set_defaults(stop_on_exit=True, rotate_180=True, reverse_yaw=True, target_filter=True)
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.forward_power < 0.0:
        parser.error("--forward-power must be non-negative")
    if args.initial_detection_timeout < 0.0:
        parser.error("--initial-detection-timeout must be non-negative")
    if args.detector_shutdown_timeout < 0.0:
        parser.error("--detector-shutdown-timeout must be non-negative")
    if args.min_detection_remaining_s < 0.0:
        parser.error("--min-detection-remaining-s must be non-negative")
    if args.imu_heading_noise_deg < 0.0:
        parser.error("--imu-heading-noise-deg must be non-negative")
    if args.model_bearing_noise_deg <= 0.0:
        parser.error("--model-bearing-noise-deg must be positive")
    if args.track_bearing_noise_deg <= 0.0:
        parser.error("--track-bearing-noise-deg must be positive")
    if args.target_process_noise_deg_s <= 0.0:
        parser.error("--target-process-noise-deg-s must be positive")
    if args.rerun_save == AUTO_RERUN_SAVE:
        args.rerun_save = unique_rerun_save_path(args.prompt)
    return args


def main() -> int:
    args = parse_args()
    return ObjectDriveRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
