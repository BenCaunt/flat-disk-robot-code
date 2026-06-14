from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import types

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from flatdisk_robot_client import ImuSample, VideoFrame  # noqa: E402
from object_drive_zenoh import (  # noqa: E402
    Detection,
    FrameState,
    KltBoxTracker,
    TargetBearingKalman,
    _detections_from_parsed,
    _image_gray,
    _load_florence_transformers_model,
    _patch_florence_forced_bos_token_id,
    _patch_transformers_tokenizer_additional_special_tokens,
    bbox_center_x_from_bearing_rad,
    clipped_cos,
    command_from_bbox,
    grounding_dino_prompt,
    positive_clipped_cos,
    select_detection,
    target_yaw_from_bbox,
)


def test_clipped_cos_keeps_only_front_arc_for_forward_scale() -> None:
    assert clipped_cos(0.0) == 1.0
    assert abs(clipped_cos(math.pi * 0.5)) < 1e-9
    assert clipped_cos(math.pi + 0.01) == 0.0
    assert clipped_cos(-math.pi - 0.01) == 0.0
    assert positive_clipped_cos(math.pi * 0.75) == 0.0


def test_command_from_right_bbox_turns_right_and_slows_forward() -> None:
    command = command_from_bbox(
        bbox_xyxy=(470.0, 120.0, 610.0, 360.0),
        image_width=640,
        current_yaw_rad=0.0,
        hfov_deg=68.0,
        forward_power=20.0,
        heading_kp=18.0,
        max_turn_percent=16.0,
        min_turn_percent=1.5,
        heading_deadband_deg=1.0,
        max_abs_output=50.0,
        reverse_correction=False,
    )

    assert command.heading_error_deg > 0.0
    assert command.turn_percent > 0.0
    assert command.motor1_percent > command.motor2_percent
    assert 0.0 < command.forward_scale < 1.0


def test_florence_post_processed_detection_prefers_matching_label() -> None:
    parsed = {
        "<CAPTION_TO_PHRASE_GROUNDING>": {
            "bboxes": [[10, 20, 110, 180], [200, 50, 260, 120]],
            "labels": ["chair", "lamp"],
        }
    }

    detections = _detections_from_parsed(
        parsed,
        task="<CAPTION_TO_PHRASE_GROUNDING>",
        prompt="chair",
        source="test",
        raw="",
    )

    assert len(detections) == 2
    assert max(detections, key=lambda det: det.score).label == "chair"


def test_closest_prompt_prefers_lower_larger_non_oversized_box() -> None:
    detections = (
        Detection((5.0, 5.0, 310.0, 230.0), "chair", 0.95, "test"),
        Detection((120.0, 115.0, 230.0, 230.0), "chair", 0.75, "test"),
        Detection((20.0, 40.0, 85.0, 125.0), "chair", 0.80, "test"),
    )

    selected = select_detection(
        detections,
        prompt="closest chair",
        image_size=(320, 240),
        max_area_fraction=0.60,
    )

    assert selected is not None
    assert selected.bbox_xyxy == detections[1].bbox_xyxy


def test_grounding_dino_prompt_removes_closest_and_negative_qualifiers() -> None:
    assert grounding_dino_prompt("closest individual chair, not the table") == "chair."


def test_florence_forced_bos_patch_sets_class_default(monkeypatch) -> None:
    class Florence2LanguageConfig:
        pass

    module = types.SimpleNamespace(Florence2LanguageConfig=Florence2LanguageConfig)
    monkeypatch.setitem(sys.modules, "fake_florence_remote_config_for_test", module)

    patched = _patch_florence_forced_bos_token_id()

    assert patched >= 1
    assert Florence2LanguageConfig.forced_bos_token_id is None


def test_transformers_tokenizer_additional_special_tokens_patch(monkeypatch) -> None:
    class PreTrainedTokenizerBase:
        def __init__(self) -> None:
            self.init_kwargs = {"additional_special_tokens": ["<loc_0>", "<loc_1>"]}

        @property
        def special_tokens_map(self) -> dict[str, list[str]]:
            raise AssertionError("patch must not read recursive special_tokens_map properties")

    module = types.SimpleNamespace(PreTrainedTokenizerBase=PreTrainedTokenizerBase)
    monkeypatch.setitem(sys.modules, "transformers.tokenization_utils_base", module)

    patched = _patch_transformers_tokenizer_additional_special_tokens()

    assert patched == 1
    assert PreTrainedTokenizerBase().additional_special_tokens == ["<loc_0>", "<loc_1>"]


def test_florence_transformers_model_loader_requests_eager_attention() -> None:
    calls: list[dict[str, object]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(_model_path: str, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    _load_florence_transformers_model(AutoModel, "model", "dtype")

    assert calls == [{"torch_dtype": "dtype", "trust_remote_code": True, "attn_implementation": "eager"}]


def test_florence_transformers_model_loader_falls_back_without_attention_kwarg() -> None:
    calls: list[dict[str, object]] = []

    class AutoModel:
        @staticmethod
        def from_pretrained(_model_path: str, **kwargs: object) -> object:
            calls.append(kwargs)
            if "attn_implementation" in kwargs:
                raise TypeError("got an unexpected keyword argument 'attn_implementation'")
            return object()

    _load_florence_transformers_model(AutoModel, "model", "dtype")

    assert calls == [
        {"torch_dtype": "dtype", "trust_remote_code": True, "attn_implementation": "eager"},
        {"torch_dtype": "dtype", "trust_remote_code": True},
    ]


def test_klt_tracker_replays_late_detection_to_current_frame() -> None:
    tracker = KltBoxTracker()
    if tracker.cv2 is None:
        return

    frame0 = _frame_state(seq=1, square=(20, 20, 58, 58))
    frame1 = _frame_state(seq=2, square=(34, 22, 72, 60))
    tracker.reset(frame0, (20.0, 20.0, 58.0, 58.0), source="test")
    bbox = tracker.replay([frame1])

    assert bbox is not None
    assert (bbox[0] + bbox[2]) * 0.5 > 50.0
    assert tracker.last_success


def test_target_filter_projects_static_target_with_imu_yaw_change() -> None:
    target_filter = TargetBearingKalman(process_noise_deg_s=1.0)
    frame0 = _frame_state(seq=1, square=(40, 20, 80, 60), yaw_rad=0.0)
    bbox = (40.0, 20.0, 80.0, 60.0)
    target_yaw = target_yaw_from_bbox(
        bbox,
        image_width=frame0.image.width,
        current_yaw_rad=frame0.imu.yaw_rad,
        hfov_deg=68.0,
    )
    target_filter.reset(
        target_yaw_rad=target_yaw,
        now_ns=frame0.monotonic_ns,
        measurement_std_rad=math.radians(2.0),
        source="test",
    )

    frame1 = _frame_state(seq=2, square=(40, 20, 80, 60), yaw_rad=math.radians(10.0))
    target_filter.predict(now_ns=frame1.monotonic_ns)
    bearing = target_filter.bearing_error_rad(frame1.imu.yaw_rad)
    center_x = bbox_center_x_from_bearing_rad(bearing, frame1.image.width, 68.0)

    assert math.degrees(bearing) < -9.0
    assert center_x < frame1.image.width * 0.5


def _frame_state(*, seq: int, square: tuple[int, int, int, int], yaw_rad: float = 0.0) -> FrameState:
    image = Image.new("RGB", (120, 90), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle(square, fill="white")
    draw.line((square[0], square[1], square[2], square[3]), fill=(80, 80, 80), width=2)
    jpeg = b""
    now = time.monotonic_ns()
    return FrameState(
        frame=VideoFrame(seq=seq, esp_us=seq * 1000, width=image.width, height=image.height, jpeg=jpeg, received_ns=now),
        imu=ImuSample(
            seq=seq,
            esp_us=seq * 1000,
            yaw_rad=yaw_rad,
            raw_yaw_rad=yaw_rad,
            quat_accuracy=3,
            accel_accuracy=3,
            flags=1,
            received_ns=now,
        ),
        image=image,
        gray=_image_gray(image),
        monotonic_ns=now,
    )
