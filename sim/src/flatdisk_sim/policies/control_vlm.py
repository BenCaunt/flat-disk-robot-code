"""Control policy: direct online VLM action selection."""

from __future__ import annotations

import base64
import json
from typing import Any

from flatdisk_sim.agent_tools import Observation
from flatdisk_sim.text_goal_policy_core import PolicyAction, parse_json_object, validate_action


class ControlVlmPolicy:
    """VLM policy with the same observation contract as the real robot."""

    name = "control_vlm"

    def __init__(self, *, model: str = "gpt-5.5", allow_stop: bool = False) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install with `uv sync --extra vlm` to use OpenAI policy evaluation") from exc
        self.client = OpenAI()
        self.model = model
        self.allow_stop = allow_stop

    def reset(self) -> None:
        return None

    def choose_action(self, obs: Observation, *, prompt: str, history: list[dict[str, Any]]) -> PolicyAction:
        image_b64 = base64.b64encode(obs.path.read_bytes()).decode("ascii")
        recent = history[-4:]
        instructions = (
            "You are the policy for a small two-wheel flat disk robot. "
            "You are running on the same sensor contract as the real robot: one low RGB camera image and IMU yaw. "
            "You do not have a map, pose, collision flags, encoders, simulator metadata, or object coordinates. "
            "The text goal is a natural-language instruction; do not search for printed words unless the goal explicitly says to read text. "
            "Recognize ordinary objects visually, such as a sofa, bed, toilet, sink, chair, or table. "
            "Drive toward the text goal if it is visible; otherwise scan with bounded turns. "
            "Use conservative short moves. Do not declare final success or stop the trial; "
            "a separate evaluator or real-robot operator will stop the run externally. "
            "If the target looks close, choose a shorter forward move instead of stopping. "
            "Output exactly one JSON object with one of these navigation actions: "
            "{\"action\":\"turn_by_angle\",\"degrees\":-35..35,\"reason\":\"...\"}, "
            "{\"action\":\"drive_straight\",\"power_percent\":20..24,\"duration_s\":0.6..0.9,\"reason\":\"...\"}. "
            "Do not include prose outside JSON."
        )
        user_text = (
            f"Goal: {prompt}\n"
            f"Current IMU yaw: {obs.yaw_deg:.1f} degrees\n"
            f"Recent actions: {json.dumps(recent, sort_keys=True)}\n"
            "Choose the next robot command."
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instructions + "\n\n" + user_text},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }
            ],
        )
        return validate_action(parse_json_object(response.output_text), allow_stop=self.allow_stop)
