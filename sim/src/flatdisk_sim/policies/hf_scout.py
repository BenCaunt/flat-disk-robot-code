"""HF Scout policy: local-model hooks with a deterministic camera scout fallback.

This policy intentionally uses only the real robot observation contract:
the latest camera frame, IMU yaw, the text prompt, and its own action history.
It never reads simulator status, hidden pose, object coordinates, encoders, or
motion result internals.

Optional local model hooks are controlled by environment variables:

* ``FLATDISK_HF_SCOUT_ENABLE=1`` enables Hugging Face inference.
* ``FLATDISK_HF_SCOUT_MODEL`` points at a cached/local VLM such as
  ``HuggingFaceTB/SmolVLM-500M-Instruct`` or ``Qwen/Qwen2.5-VL-3B-Instruct``.
* ``FLATDISK_HF_SCOUT_DEPTH_MODEL`` can point at a cached/local depth model
  such as ``depth-anything/Depth-Anything-V2-Small-hf``.
* ``FLATDISK_HF_SCOUT_LOCAL_ONLY=0`` permits downloads. The default is local
  files only so competition runs remain practical and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.text_goal_policy_core import PolicyAction, parse_json_object, validate_action


@dataclass(frozen=True)
class _FrameFeatures:
    open_center: float
    blockage: float
    free_offset: float
    salient_offset: float
    salient_score: float
    brightness: float


@dataclass(frozen=True)
class _GoalCue:
    family: str
    score: float
    offset: float
    close: float
    blocked_center: float


class HfScoutPolicy:
    """A fast camera+IMU scout with optional cached Hugging Face inference."""

    name = "hf_scout"

    def __init__(
        self,
        *,
        enable_hf: bool | None = None,
        model: str | None = None,
        depth_model: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.enable_hf = _env_bool("FLATDISK_HF_SCOUT_ENABLE", False) if enable_hf is None else enable_hf
        self.model = model or os.environ.get("FLATDISK_HF_SCOUT_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
        self.depth_model = depth_model or os.environ.get("FLATDISK_HF_SCOUT_DEPTH_MODEL", "")
        self.local_files_only = (
            _env_bool("FLATDISK_HF_SCOUT_LOCAL_ONLY", True) if local_files_only is None else local_files_only
        )
        self._vlm_pipe: Any | None = None
        self._depth_pipe: Any | None = None
        self._hf_failed = False
        self._depth_failed = False
        self.reset()

    def reset(self) -> None:
        self.step_index = 0
        self._scan_direction = 1.0
        self._last_feature: _FrameFeatures | None = None
        self._last_signature: tuple[float, ...] | None = None
        self._repeat_count = 0
        self._last_target: str | None = None
        self._memory: list[tuple[float, float, float, str]] = []
        self._toilet_scan_count = 0

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        if self.enable_hf:
            action = self._choose_hf_action(obs, prompt=prompt, history=history)
            if action is not None:
                self.step_index += 1
                return action

        features = self._analyze_frame(obs.path)
        self._last_feature = features
        self._update_repeat_count(_frame_signature(obs.path))
        family = _goal_family(prompt)
        cue = _goal_cue(obs.path, family, features)
        if family != "toilet" or self._toilet_scan_count >= 5:
            self._remember(obs.yaw_deg, cue)
        action = self._choose_scout_action(
            obs,
            prompt=prompt,
            history=history,
            features=features,
            family=family,
            cue=cue,
        )
        self.step_index += 1
        return action

    def _choose_hf_action(
        self,
        obs: Observation,
        *,
        prompt: str,
        history: list[dict[str, Any]],
    ) -> PolicyAction | None:
        pipe = self._get_vlm_pipe()
        if pipe is None:
            return None

        image = Image.open(obs.path).convert("RGB")
        instruction = (
            "You control a small two-wheel flat disk robot using only this RGB camera image, IMU yaw, "
            "the text goal, and recent action history. You do not have a map, pose, simulator metadata, "
            "object coordinates, collision flags, encoders, or wheel odometry. Choose one short action. "
            "If the target is visible, steer toward it and drive. If it is not visible, scan or scout. "
            "Never stop; an external evaluator stops the run. Return exactly one JSON object using "
            "{\"action\":\"turn_by_angle\",\"degrees\":-35..35,\"reason\":\"...\"} or "
            "{\"action\":\"drive_straight\",\"power_percent\":20..24,\"duration_s\":0.6..0.9,\"reason\":\"...\"}."
        )
        message = (
            f"{instruction}\nGoal: {prompt}\n"
            f"IMU yaw: {obs.yaw_deg:.1f} degrees\n"
            f"Recent actions: {json.dumps(history[-4:], sort_keys=True)}"
        )
        try:
            result = pipe(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": message},
                        ],
                    }
                ],
                max_new_tokens=96,
            )
            text = _extract_generated_text(result)
            return validate_action(parse_json_object(text), allow_stop=False)
        except Exception:
            self._hf_failed = True
            return None

    def _choose_scout_action(
        self,
        obs: Observation,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        features: _FrameFeatures,
        family: str,
        cue: _GoalCue,
    ) -> PolicyAction:
        target = self._best_camera_detection(obs, prompt)
        if target is not None:
            self._last_target = target.name
            if abs(target.center_offset) > 0.18:
                return _validated(
                    "turn_by_angle",
                    degrees=target.center_offset * 30.0,
                    reason=f"camera_detection_centering:{target.name}:{target.confidence:.2f}",
                )
            duration = 0.62 if target.area_fraction > 0.035 else 0.82
            return _validated(
                "drive_straight",
                power_percent=23.0,
                duration_s=duration,
                reason=f"camera_detection_approach:{target.name}:{target.confidence:.2f}",
            )

        if family == "toilet" and self._toilet_scan_count < 5:
            cue_action = self._choose_goal_cue_action(
                obs,
                history=history,
                features=features,
                family=family,
                cue=cue,
            )
            if cue_action is not None and cue.score >= _strong_threshold(family):
                return cue_action
            return self._choose_toilet_search(obs, history=history, features=features)

        cue_action = self._choose_goal_cue_action(obs, history=history, features=features, family=family, cue=cue)
        if cue_action is not None:
            return cue_action

        depth_action = self._choose_depth_action(obs.path, features)
        if depth_action is not None:
            return depth_action

        recent = [str(item.get("action", "")) for item in history[-4:]]
        recent_turns = recent.count("turn_by_angle")
        recent_drives = recent.count("drive_straight")
        if family == "toilet":
            return self._choose_toilet_search(obs, history=history, features=features)

        memory_action = self._turn_toward_memory(obs.yaw_deg)
        if memory_action is not None and recent_turns < 3:
            return memory_action

        if len(recent) >= 3 and recent[-3:].count("drive_straight") == 3:
            return _validated(
                "turn_by_angle",
                degrees=30.0 * self._scan_direction,
                reason="periodic_scan_after_forward_progress",
            )

        if features.blockage > 0.64:
            self._scan_direction = 1.0 if features.free_offset >= 0.0 else -1.0
            return _validated(
                "turn_by_angle",
                degrees=32.0 * self._scan_direction,
                reason=f"visual_blockage_turn_to_free_space:{features.free_offset:.2f}",
            )

        if features.salient_score > 0.30 and abs(features.salient_offset) > 0.28 and recent_turns < 2:
            return _validated(
                "turn_by_angle",
                degrees=_min_turn(features.salient_offset * 25.0, 16.0),
                reason=f"salient_region_probe:{features.salient_score:.2f}",
            )

        if features.open_center > 0.48 or (recent_turns >= 2 and recent_drives == 0):
            return _validated(
                "drive_straight",
                power_percent=24.0,
                duration_s=0.82,
                reason=f"open_center_scout:yaw={obs.yaw_deg:.1f}",
            )

        if self.step_index % 3 == 0:
            self._scan_direction *= -1.0
        return _validated(
            "turn_by_angle",
            degrees=34.0 * self._scan_direction,
            reason=f"structured_visual_sweep_for_{family}:step={self.step_index}",
        )

    def _choose_goal_cue_action(
        self,
        obs: Observation,
        *,
        history: list[dict[str, Any]],
        features: _FrameFeatures,
        family: str,
        cue: _GoalCue,
    ) -> PolicyAction | None:
        strong = _strong_threshold(family)
        weak = _weak_threshold(family)
        recent_turns = sum(1 for item in history[-3:] if item.get("action") == "turn_by_angle")
        recent_drives = sum(1 for item in history[-4:] if item.get("action") == "drive_straight")
        drive_streak = _recent_action_streak(history, "drive_straight")

        if family in {"bed", "seat"} and drive_streak >= 2 and cue.score >= weak:
            direction = _probe_escape_direction(features, cue, self._scan_direction)
            self._scan_direction = direction
            return _validated(
                "turn_by_angle",
                degrees=32.0 * direction,
                reason=(
                    f"break_repeated_lateral_probe_{family}:"
                    f"cue={cue.score:.2f}:repeat={self._repeat_count}"
                ),
            )

        if family in {"toilet", "sink"} and drive_streak >= 3 and self._repeat_count >= 1:
            direction = _probe_escape_direction(features, cue, self._scan_direction)
            self._scan_direction = direction
            return _validated(
                "turn_by_angle",
                degrees=30.0 * direction,
                reason=f"break_repeated_ceramic_probe_{family}:repeat={self._repeat_count}",
            )

        if (
            family in {"bed", "seat"}
            and recent_drives > 0
            and drive_streak < 2
            and cue.score >= weak
            and features.open_center > 0.24
        ):
            return _validated(
                "drive_straight",
                power_percent=23.0,
                duration_s=0.82,
                reason=f"continue_lateral_probe_{family}:{cue.score:.2f}",
            )

        if (
            family in {"bed", "seat"}
            and cue.score >= weak
            and abs(cue.offset) < 0.22
            and recent_turns < 3
            and recent_drives == 0
        ):
            return _validated(
                "turn_by_angle",
                degrees=-32.0,
                reason=f"lateral_acquisition_scan_for_{family}:{cue.score:.2f}",
            )

        if family in {"bed", "seat"} and cue.score >= weak and recent_turns >= 2 and features.open_center > 0.28:
            return _validated(
                "drive_straight",
                power_percent=23.0,
                duration_s=0.82,
                reason=f"lateral_probe_after_scan_{family}:{cue.score:.2f}",
            )

        if cue.score >= strong:
            deadband = 0.13 if family in {"toilet", "sink"} else 0.18
            if abs(cue.offset) > deadband:
                return _validated(
                    "turn_by_angle",
                    degrees=_min_turn(cue.offset * 31.0, 14.0),
                    reason=f"center_{family}_visual_cue:{cue.score:.2f}:{cue.offset:.2f}",
                )
            duration = 0.62 if cue.close > 0.58 or cue.blocked_center > 0.62 else 0.88
            power = 22.0 if duration < 0.7 else 24.0
            return _validated(
                "drive_straight",
                power_percent=power,
                duration_s=duration,
                reason=f"approach_{family}_visual_cue:{cue.score:.2f}:close={cue.close:.2f}",
            )

        if cue.score >= weak and abs(cue.offset) > 0.16 and recent_turns < 3:
            return _validated(
                "turn_by_angle",
                degrees=_min_turn(cue.offset * 27.0, 16.0),
                reason=f"weak_{family}_visual_cue:{cue.score:.2f}:{cue.offset:.2f}",
            )

        if family in {"bed", "seat"} and cue.score >= weak and features.open_center > 0.38:
            return _validated(
                "drive_straight",
                power_percent=23.0,
                duration_s=0.78,
                reason=f"fast_probe_visible_{family}:{cue.score:.2f}",
            )

        return None

    def _choose_toilet_search(
        self,
        obs: Observation,
        *,
        history: list[dict[str, Any]],
        features: _FrameFeatures,
    ) -> PolicyAction:
        recent_turns = sum(1 for item in history[-4:] if item.get("action") == "turn_by_angle")
        drive_streak = _recent_action_streak(history, "drive_straight")
        if self._toilet_scan_count >= 5:
            memory_action = self._turn_toward_memory(obs.yaw_deg)
            if memory_action is not None and recent_turns < 4:
                return memory_action

        if self._toilet_scan_count < 5:
            self._toilet_scan_count += 1
            return _validated(
                "turn_by_angle",
                degrees=32.0,
                reason=f"toilet_clockwise_wall_scan:{self._toilet_scan_count}",
            )

        if drive_streak >= 3:
            direction = _probe_escape_direction(features, _GoalCue("toilet", 0.0, 0.0, 0.0, features.blockage), self._scan_direction)
            self._scan_direction = direction
            return _validated(
                "turn_by_angle",
                degrees=30.0 * direction,
                reason=f"toilet_probe_budget_escape:drives={drive_streak}",
            )

        if features.blockage > 0.68:
            return _validated("turn_by_angle", degrees=30.0, reason="toilet_blocked_continue_scan")

        if features.open_center > 0.34:
            duration = 0.62 if drive_streak >= 1 else 0.7
            return _validated(
                "drive_straight",
                power_percent=22.0,
                duration_s=duration,
                reason=f"toilet_post_scan_probe:yaw={obs.yaw_deg:.1f}",
            )

        return _validated("turn_by_angle", degrees=30.0, reason="toilet_search_no_clear_path")

    def _remember(self, yaw_deg: float, cue: _GoalCue) -> None:
        if cue.score < _weak_threshold(cue.family):
            return
        self._memory.append((_wrap_deg(yaw_deg), cue.score, cue.offset, cue.family))
        self._memory = sorted(self._memory, key=lambda item: item[1], reverse=True)[:8]

    def _turn_toward_memory(self, yaw_deg: float) -> PolicyAction | None:
        if not self._memory:
            return None
        best_yaw, best_score, best_offset, family = self._memory[0]
        if best_score < _strong_threshold(family):
            return None
        error = _wrap_deg(best_yaw - yaw_deg)
        if abs(error) < 12.0:
            return None
        return _validated(
            "turn_by_angle",
            degrees=_min_turn(max(-35.0, min(35.0, error + best_offset * 9.0)), 15.0),
            reason=f"return_to_best_{family}_cue:{best_score:.2f}",
        )

    def _update_repeat_count(self, signature: tuple[float, ...]) -> None:
        if self._last_signature is None:
            self._repeat_count = 0
        else:
            delta = sum(abs(a - b) for a, b in zip(signature, self._last_signature)) / len(signature)
            self._repeat_count = self._repeat_count + 1 if delta < 0.035 else 0
        self._last_signature = signature

    def _best_camera_detection(self, obs: Observation, prompt: str) -> Any | None:
        target_names = _target_names_from_prompt(prompt)
        candidates = []
        for name in target_names:
            detection = obs.analysis.best(name)
            if detection is not None:
                candidates.append(detection)
        if not candidates:
            return None
        best = max(candidates, key=lambda det: det.confidence)
        if best.confidence < 0.07:
            return None
        return best

    def _choose_depth_action(self, image_path: Path, features: _FrameFeatures) -> PolicyAction | None:
        pipe = self._get_depth_pipe()
        if pipe is None:
            return None
        try:
            result = pipe(str(image_path))
            depth = _extract_depth_array(result)
        except Exception:
            self._depth_failed = True
            return None
        if depth is None or depth.size == 0:
            return None
        depth = _normalize(depth)
        height, width = depth.shape
        lower = depth[height // 2 :, :]
        thirds = np.array_split(lower, 3, axis=1)
        clearances = [float(chunk.mean()) for chunk in thirds]
        best_index = int(np.argmax(clearances))
        center_clearance = clearances[1]
        if center_clearance > 0.54 and features.blockage < 0.72:
            return _validated("drive_straight", power_percent=23.0, duration_s=0.8, reason="depth_center_clear")
        if best_index != 1:
            return _validated(
                "turn_by_angle",
                degrees=(-24.0 if best_index == 0 else 24.0),
                reason=f"depth_clearance_turn:{clearances}",
            )
        return None

    def _get_vlm_pipe(self) -> Any | None:
        if not self.enable_hf or self._hf_failed:
            return None
        if self._vlm_pipe is not None:
            return self._vlm_pipe
        try:
            from transformers import pipeline

            self._vlm_pipe = pipeline(
                task="image-text-to-text",
                model=self.model,
                device_map="auto",
                model_kwargs={"local_files_only": self.local_files_only},
            )
        except Exception:
            self._hf_failed = True
            return None
        return self._vlm_pipe

    def _get_depth_pipe(self) -> Any | None:
        if not self.enable_hf or not self.depth_model or self._depth_failed:
            return None
        if self._depth_pipe is not None:
            return self._depth_pipe
        try:
            from transformers import pipeline

            self._depth_pipe = pipeline(
                task="depth-estimation",
                model=self.depth_model,
                device_map="auto",
                model_kwargs={"local_files_only": self.local_files_only},
            )
        except Exception:
            self._depth_failed = True
            return None
        return self._depth_pipe

    def _analyze_frame(self, image_path: Path) -> _FrameFeatures:
        image = Image.open(image_path).convert("RGB").resize((160, 120), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        height, width = gray.shape
        lower = gray[height // 2 :, :]
        center = lower[:, width // 3 : width * 2 // 3]
        left = lower[:, : width // 3]
        right = lower[:, width * 2 // 3 :]

        grad_x = np.abs(np.diff(lower, axis=1, prepend=lower[:, :1]))
        grad_y = np.abs(np.diff(lower, axis=0, prepend=lower[:1, :]))
        edge = grad_x + grad_y
        center_edge = edge[:, width // 3 : width * 2 // 3]
        center_brightness = float(center.mean())
        center_edge_mean = float(center_edge.mean())
        open_center = _clamp01(center_brightness * 1.2 - center_edge_mean * 2.4)

        left_score = float(left.mean() - edge[:, : width // 3].mean() * 1.4)
        center_score = float(center.mean() - center_edge.mean() * 1.4)
        right_score = float(right.mean() - edge[:, width * 2 // 3 :].mean() * 1.4)
        free_offset = float(np.argmax([left_score, center_score, right_score]) - 1)

        max_rgb = rgb.max(axis=2)
        min_rgb = rgb.min(axis=2)
        saturation = max_rgb - min_rgb
        saliency = saturation[height // 2 :, :] + edge * 1.25
        usable = saliency
        column_scores = usable.mean(axis=0)
        if float(column_scores.max()) <= 1e-6:
            salient_offset = 0.0
            salient_score = 0.0
        else:
            index = int(column_scores.argmax())
            salient_offset = (index - (width - 1) * 0.5) / max(1.0, (width - 1) * 0.5)
            salient_score = _clamp01(float(column_scores.max()) * 2.5)

        blockage = _clamp01(center_edge_mean * 4.0 + max(0.0, 0.34 - center_brightness) * 1.7)
        return _FrameFeatures(
            open_center=open_center,
            blockage=blockage,
            free_offset=free_offset,
            salient_offset=float(salient_offset),
            salient_score=salient_score,
            brightness=center_brightness,
        )


def _validated(action: str, *, reason: str, **payload: Any) -> PolicyAction:
    return validate_action({"action": action, "reason": reason, **payload}, allow_stop=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _target_names_from_prompt(prompt: str) -> tuple[str, ...]:
    text = re.sub(r"[^a-z0-9 ]+", " ", prompt.lower())
    aliases = {
        "bed": ("bed", "mattress", "pillow", "blanket"),
        "mattress": ("bed", "mattress", "pillow", "blanket"),
        "pillow": ("bed", "mattress", "pillow", "blanket"),
        "blanket": ("bed", "mattress", "pillow", "blanket"),
        "sofa": ("sofa", "couch", "loveseat", "chair", "armchair", "ottoman", "recliner"),
        "couch": ("sofa", "couch", "loveseat", "chair", "armchair", "ottoman", "recliner"),
        "loveseat": ("sofa", "couch", "loveseat", "chair", "armchair", "ottoman", "recliner"),
        "chair": ("chair", "armchair", "sofa", "couch", "ottoman", "recliner"),
        "armchair": ("armchair", "chair", "sofa", "couch", "ottoman", "recliner"),
        "table": ("table", "coffee table", "desk", "counter", "countertop"),
        "desk": ("desk", "table", "counter"),
        "counter": ("counter", "countertop", "table"),
        "sink": ("sink", "basin", "vanity"),
        "basin": ("sink", "basin", "vanity"),
        "tv": ("tv", "television", "monitor", "screen"),
        "television": ("tv", "television", "monitor", "screen"),
        "monitor": ("monitor", "screen", "tv", "television"),
        "screen": ("screen", "monitor", "tv", "television"),
        "bathroom sign": ("bathroom sign",),
        "bathroom": ("bathroom sign", "bath mat"),
        "bath mat": ("bath mat",),
        "duck": ("yellow duck",),
        "yellow duck": ("yellow duck",),
        "sock": ("blue sock",),
        "blue sock": ("blue sock",),
        "book": ("green book",),
        "green book": ("green book",),
        "ball": ("red ball",),
        "red ball": ("red ball",),
        "cube": ("purple cube",),
        "purple cube": ("purple cube",),
        "toilet": ("toilet",),
    }
    names: list[str] = []
    for key, values in aliases.items():
        if key in text:
            names.extend(values)
    return tuple(dict.fromkeys(names))


def _goal_family(prompt: str) -> str:
    text = prompt.lower()
    if _has_word(text, "bed", "mattress", "pillow", "blanket"):
        return "bed"
    if _has_word(text, "sofa", "couch", "loveseat", "chair", "armchair", "ottoman", "recliner"):
        return "seat"
    if _has_word(text, "toilet", "commode"):
        return "toilet"
    if _has_word(text, "sink", "basin", "vanity"):
        return "sink"
    if _has_word(text, "table", "desk", "counter", "countertop", "island"):
        return "table"
    if _has_word(text, "tv", "television", "monitor", "screen"):
        return "screen"
    return "generic"


def _has_word(text: str, *words: str) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _goal_cue(image_path: Path, family: str, features: _FrameFeatures) -> _GoalCue:
    image = Image.open(image_path).convert("RGB").resize((96, 72), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = rgb.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    gray = rgb.mean(axis=2)
    value = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (value - min_channel) / np.maximum(value, 1e-4)
    edge = _edge_strength(gray)
    horizontal = np.clip(np.abs(np.diff(gray, axis=0, prepend=gray[:1, :])), 0.0, 1.0)
    score_map = _score_map(family, rgb, value, saturation, edge, horizontal, y)
    score, offset, close = _summarize_score(score_map, x, y)

    return _GoalCue(
        family=family,
        score=score,
        offset=offset,
        close=close,
        blocked_center=features.blockage,
    )


def _score_map(
    family: str,
    rgb: np.ndarray[Any, Any],
    value: np.ndarray[Any, Any],
    saturation: np.ndarray[Any, Any],
    edge: np.ndarray[Any, Any],
    horizontal: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    row_mid = _row_weight(y, 0.18, 0.80)
    row_low_mid = _row_weight(y, 0.34, 0.90)
    row_upper_mid = _row_weight(y, 0.10, 0.70)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    white_object = (value > 0.68) & (saturation < 0.22) & (edge > 0.025)
    fabric = (value > 0.18) & (value < 0.86) & (saturation < 0.66)
    tan_or_gray = fabric & (red > 0.20) & (green > 0.18) & (blue > 0.14)
    dark_structure = (value > 0.05) & (value < 0.34) & (edge > 0.018)
    colored_cover = (saturation > 0.20) & (value > 0.24) & (value < 0.96)

    if family == "toilet":
        warm_ceramic = (
            (value > 0.52)
            & (saturation < 0.40)
            & (red > blue * 0.92)
            & (green > blue * 0.82)
            & ((edge > 0.012) | (horizontal > 0.010))
        )
        ceramic = white_object | warm_ceramic
        bowl_like = ceramic.astype(np.float32) * (0.28 + edge * 4.4 + horizontal * 2.6)
        plain_wall_penalty = ((value > 0.80) & (saturation < 0.16) & (edge < 0.012) & (horizontal < 0.010)).astype(
            np.float32
        )
        return np.maximum(0.0, bowl_like * row_low_mid - plain_wall_penalty * row_low_mid * 0.20).astype(np.float32)
    if family == "sink":
        sink_like = white_object.astype(np.float32) * (0.45 + edge * 4.5)
        return (sink_like * row_mid).astype(np.float32)
    if family == "bed":
        blanket_area = colored_cover.astype(np.float32) * 0.34
        bed_edges = horizontal * 4.6
        under_gap = dark_structure.astype(np.float32) * 0.45
        return ((blanket_area + bed_edges) * row_upper_mid + under_gap * row_mid).astype(np.float32)
    if family == "seat":
        upholstery = tan_or_gray.astype(np.float32) * (0.28 + edge * 3.2)
        low_mass = fabric.astype(np.float32) * row_low_mid * 0.16
        return (upholstery * row_mid + low_mass).astype(np.float32)
    if family == "table":
        legs = dark_structure.astype(np.float32) * row_low_mid * 0.55
        tops = horizontal * row_mid * 3.5
        return (legs + tops).astype(np.float32)
    if family == "screen":
        screen = ((value < 0.26) & (saturation < 0.50)).astype(np.float32) * row_mid
        return (screen * (0.35 + edge * 3.4)).astype(np.float32)
    return ((edge * 2.5 + colored_cover.astype(np.float32) * 0.16) * row_mid).astype(np.float32)


def _edge_strength(gray: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    grad_x = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    grad_y = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return np.clip(grad_x + grad_y, 0.0, 1.0)


def _row_weight(y: np.ndarray[Any, Any], low: float, high: float) -> np.ndarray[Any, Any]:
    return ((y >= low) & (y <= high)).astype(np.float32)


def _summarize_score(
    score_map: np.ndarray[Any, Any],
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
) -> tuple[float, float, float]:
    total = float(score_map.sum())
    if total <= 1e-5:
        return 0.0, 0.0, 0.0
    column = score_map.sum(axis=0)
    smooth = np.convolve(column, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    best = float(smooth.max())
    score = min(1.0, best / max(1.0, score_map.shape[0] * 0.11))
    offset = float((score_map * x).sum() / total)
    low_total = float((score_map * (y > 0.56)).sum())
    close = min(1.0, low_total / max(1.0, total * 0.64))
    return score, max(-1.0, min(1.0, offset)), close


def _strong_threshold(family: str) -> float:
    if family in {"toilet", "sink"}:
        return 0.40
    if family in {"bed", "table"}:
        return 0.42
    if family == "seat":
        return 0.34
    return 0.36


def _weak_threshold(family: str) -> float:
    if family in {"toilet", "sink"}:
        return 0.18
    if family == "seat":
        return 0.16
    return 0.20


def _min_turn(degrees: float, minimum: float) -> float:
    degrees = float(degrees)
    if abs(degrees) < minimum:
        return minimum if degrees >= 0.0 else -minimum
    return max(-35.0, min(35.0, degrees))


def _frame_signature(image_path: Path) -> tuple[float, ...]:
    image = Image.open(image_path).convert("RGB").resize((48, 36), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    edge = _edge_strength(gray)
    color_blocks = rgb.reshape(6, 6, 8, 6, 3).mean(axis=(1, 3, 4)).reshape(-1)
    edge_blocks = edge.reshape(6, 6, 8, 6).mean(axis=(1, 3)).reshape(-1)
    values = np.concatenate([color_blocks, edge_blocks])
    return tuple(round(float(value), 3) for value in values)


def _recent_action_streak(history: list[dict[str, Any]], action_name: str) -> int:
    streak = 0
    for item in reversed(history):
        if item.get("action") != action_name:
            break
        streak += 1
    return streak


def _probe_escape_direction(features: _FrameFeatures, cue: _GoalCue, scan_direction: float) -> float:
    if abs(cue.offset) > 0.12:
        return -1.0 if cue.offset > 0.0 else 1.0
    if abs(features.salient_offset) > 0.18:
        return 1.0 if features.salient_offset > 0.0 else -1.0
    if features.free_offset != 0.0:
        return 1.0 if features.free_offset > 0.0 else -1.0
    return 1.0 if scan_direction >= 0.0 else -1.0


def _wrap_deg(angle_deg: float) -> float:
    return math.degrees(math.atan2(math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))))


def _extract_generated_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        return _extract_generated_text(result[0])
    if isinstance(result, dict):
        for key in ("generated_text", "text", "answer"):
            value = result.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value:
                return _extract_generated_text(value[-1])
        message = result.get("message")
        if isinstance(message, dict):
            return _extract_generated_text(message)
        content = result.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if texts:
                return "\n".join(texts)
    return str(result)


def _extract_depth_array(result: Any) -> np.ndarray[Any, Any] | None:
    if isinstance(result, dict):
        for key in ("depth", "predicted_depth"):
            value = result.get(key)
            if isinstance(value, Image.Image):
                return np.asarray(value.convert("L"), dtype=np.float32)
            if isinstance(value, np.ndarray):
                return value.astype(np.float32)
            if hasattr(value, "detach"):
                return value.detach().cpu().numpy().astype(np.float32)
    return None


def _normalize(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    values = np.asarray(values, dtype=np.float32)
    low = float(np.nanpercentile(values, 5))
    high = float(np.nanpercentile(values, 95))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
