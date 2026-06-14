"""Thin LLM harness for flat disk robot text-goal control.

The actor only sees the same surface the real robot can provide today:
camera-derived observation summaries, IMU yaw, bounded tools, and its own
memory/action history. Hidden simulator state is kept behind the tool/eval
objects and is never inserted into actor or critic prompts.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import json
import mimetypes
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Protocol
import urllib.error
import urllib.request

from .text_goal_policy_core import clamp_float, parse_json_object


HARNESS_MODES = ("idle", "auto", "paused", "teleop", "complete", "error")
ALLOWED_OBSERVATION_KEYS = {"path", "yaw_deg", "frame_seq", "brightness_center"}
ALLOWED_ACTIONS = {
    "turn_by_angle",
    "drive_straight",
    "visual_servo_object",
    "check_object_grounding",
    "query_topomap_memory",
    "stop",
    "wait",
}
FORBIDDEN_PRIVILEGED_KEYS = {
    "distance_m",
    "hidden",
    "hidden_score",
    "nearest_target",
    "object_metadata",
    "objects",
    "pose",
    "scene",
    "success_radius",
    "target_pose",
    "thor",
}

TOOL_CONTRACT: dict[str, Any] = {
    "turn_by_angle": {
        "args": {"degrees": "float -35..35", "power_percent": "float 8..14 optional"},
        "effect": "Rotate in place by a bounded relative angle.",
    },
    "drive_straight": {
        "args": {"power_percent": "float 18..24", "duration_s": "float 0.25..0.9"},
        "effect": "Drive forward briefly.",
    },
    "visual_servo_object": {
        "args": {
            "prompt": "short phrase for a currently visible object, landmark, passage, or region chosen from the latest RGB frame",
            "duration_s": "float 0.5..4.0",
            "forward_power": "float 8..24 optional",
            "detector": "optional detector name; default uses the configured object-drive detector",
        },
        "effect": (
            "Run a bounded phrase-grounded visual servo toward the named visible object/landmark. "
            "This is not a search action and does not verify that the object is the final goal."
        ),
    },
    "check_object_grounding": {
        "args": {
            "prompt": "short phrase for a candidate object, landmark, passage, or region in the latest RGB frame",
            "detector": "optional detector name; default uses the configured object-drive detector",
        },
        "effect": (
            "Non-motion phrase-grounding check on the latest RGB frame. "
            "Returns whether the detector produced a usable selected box and writes a detector overlay for inspection."
        ),
    },
    "query_topomap_memory": {
        "args": {"goal_query": "optional text goal query; default is the current navigation goal"},
        "effect": (
            "Non-motion memory lookup using the latest RGB frame. "
            "When a topomap is configured, returns image-match/route hints and a contact sheet without map coordinates or object metadata."
        ),
    },
    "stop": {
        "args": {},
        "effect": "Stop all motion. Use when the goal is reached or safety requires it.",
    },
    "wait": {
        "args": {"duration_s": "float 0.05..1.0"},
        "effect": "Do nothing for a short interval.",
    },
}

POLICY_INPUT_ALLOWLIST = [
    "camera frame attachment",
    "camera-derived summary",
    "imu yaw",
    "bounded tool results",
    "non-motion phrase-grounding check result",
    "topomap image-memory tool result",
    "relative memory log",
    "previous motion strip",
    "previous raw/detector paired grounding audit strip",
    "previous detector debug overlay strip",
]
PROMPT_MEMORY_RECORD_LIMIT = 8
PROMPT_TEXT_LIMIT = 220
PROMPT_LIST_LIMIT = 4
PROMPT_TOOL_RESULT_KEYS = {
    "action",
    "cost",
    "debug_overlay_contact_sheet",
    "detector",
    "detection_coverage_fraction",
    "detection_count",
    "detection_status_count",
    "duration_s",
    "elapsed_s",
    "ever_detected",
    "failure_reason",
    "final_yaw_deg",
    "forward_power",
    "frame_count",
    "goal_candidates",
    "goal_query",
    "grounding_geometry_warning",
    "grounding_audit_contact_sheet",
    "heading_error_deg",
    "image_path",
    "last_command",
    "markdown_path",
    "map_summary",
    "matching_mode",
    "motion_contact_sheet",
    "motor_commands_sent",
    "moved",
    "ok",
    "overlay_path",
    "planner_note",
    "prompt",
    "reason",
    "ready_for_visual_servo",
    "recommendation",
    "report_path",
    "route_length",
    "route_node_ids",
    "route_truncated",
    "routes",
    "selected_bbox_xyxy",
    "selected_bbox_area_fraction",
    "selected_bbox_center_xy_norm",
    "selected_bbox_edge_contact",
    "selected_bbox_height_fraction",
    "selected_bbox_touches_image_edge",
    "selected_bbox_width_fraction",
    "selected_detection_count",
    "selected_label",
    "selected_score",
    "semantic_identity",
    "servo_status",
    "started_yaw_deg",
    "grounding_stability",
    "status_sample_count",
    "target_detected",
    "target_yaw_deg",
    "timed_out",
    "topomap_contact_sheet",
}

ACTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["thought", "action", "grounding_audit"],
    "additionalProperties": False,
    "properties": {
        "thought": {"type": "string"},
        "action": {
            "type": "object",
            "required": ["tool", "args"],
            "additionalProperties": False,
            "properties": {
                "tool": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                "args": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "degrees": {"type": ["number", "null"]},
                        "power_percent": {"type": ["number", "null"]},
                        "duration_s": {"type": ["number", "null"]},
                        "prompt": {"type": ["string", "null"]},
                        "goal_query": {"type": ["string", "null"]},
                        "detector": {"type": ["string", "null"]},
                        "forward_power": {"type": ["number", "null"]},
                    },
                },
            },
        },
        "memory_update": {
            "type": ["object", "null"],
            "additionalProperties": True,
        },
        "grounding_audit": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "previous_visual_servo_box_matches_intended_object": {"type": ["boolean", "null"]},
                "previous_check_box_matches_intended_object": {"type": ["boolean", "null"]},
                "evidence": {"type": ["string", "null"]},
                "check_overlay_evidence": {"type": ["string", "null"]},
                "next_prompt_should_change": {"type": ["boolean", "null"]},
            },
        },
        "save_frames": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "additionalProperties": True,
            },
        },
    },
}

CRITIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "reason", "replacement_action"],
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "warn", "reject"]},
        "reason": {"type": "string"},
        "replacement_action": {
            "type": ["object", "null"],
            "required": ["tool", "args"],
            "additionalProperties": False,
            "properties": {
                "tool": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
                "args": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "degrees": {"type": ["number", "null"]},
                        "power_percent": {"type": ["number", "null"]},
                        "duration_s": {"type": ["number", "null"]},
                        "prompt": {"type": ["string", "null"]},
                        "goal_query": {"type": ["string", "null"]},
                        "detector": {"type": ["string", "null"]},
                        "forward_power": {"type": ["number", "null"]},
                    },
                },
            },
        },
    },
}


class RobotTools(Protocol):
    def observe(self, *, label: str = "observe", timeout_s: float = 2.0) -> Any:
        ...

    def turn_by_angle(self, degrees: float, *, power_percent: float = 10.0) -> Any:
        ...

    def drive_straight(self, power_percent: float, duration_s: float) -> Any:
        ...

    def visual_servo_object(
        self,
        prompt: str,
        *,
        duration_s: float = 2.0,
        detector: str | None = None,
        forward_power: float = 18.0,
    ) -> Any:
        ...

    def check_object_grounding(self, *, image_path: Path, prompt: str, detector: str | None = None) -> Any:
        ...

    def query_topomap_memory(self, *, image_path: Path, goal_query: str) -> Any:
        ...

    def stop(self) -> Any:
        ...

    def close(self) -> None:
        ...


class LlmRunner(Protocol):
    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        ...


@dataclass(frozen=True)
class HarnessConfig:
    run_dir: Path
    model: str = "gpt-5.5"
    reasoning_effort: str = "low"
    prompt_profile: str = "baseline"
    actor_rules: tuple[str, ...] = field(default_factory=tuple)
    critic_rules: tuple[str, ...] = field(default_factory=tuple)
    critic_mode: str = "auto"
    max_memory_records: int = 16
    max_steps: int = 24
    sleep_scale: float = 0.0
    rerun_enabled: bool = False


@dataclass(frozen=True)
class HarnessAction:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""


@dataclass(frozen=True)
class CriticDecision:
    verdict: str
    reason: str
    replacement: HarnessAction | None = None


@dataclass
class HarnessStatus:
    mode: str = "idle"
    goal: str = ""
    step: int = 0
    last_observation: dict[str, Any] | None = None
    last_action: dict[str, Any] | None = None
    last_critic: dict[str, Any] | None = None
    latest_frame_path: str | None = None
    worker_active: bool = False
    error: str | None = None


class CodexExecRunner:
    """Runs actor/critic prompts through ``codex exec`` when live mode is used."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        reasoning_effort: str = "low",
        codex_binary: str = "codex",
        cwd: Path | None = None,
        output_schema_dir: Path | None = None,
        timeout_s: float = 90.0,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex_binary = codex_binary
        self.cwd = cwd
        self.output_schema_dir = output_schema_dir
        self.timeout_s = timeout_s

    def command(self, *, role: str = "actor", image_paths: list[Path] | None = None) -> list[str]:
        cmd = [
            self.codex_binary,
            "exec",
            "-m",
            self.model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "-c",
            'approval_policy="never"',
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--color",
            "never",
        ]
        if self.cwd is not None:
            cmd.extend(["--cd", str(self.cwd)])
        schema_path = self._output_schema_path(role)
        if schema_path is not None:
            cmd.extend(["--output-schema", str(schema_path)])
        for image_path in image_paths or []:
            cmd.extend(["--image", str(image_path)])
        cmd.append(
            "-",
        )
        return cmd

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        completed = subprocess.run(
            self.command(role=role, image_paths=image_paths),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"codex exec failed for {role}: {stderr or completed.returncode}")
        return completed.stdout.strip()

    def _output_schema_path(self, role: str) -> Path | None:
        schema = {"actor": ACTION_OUTPUT_SCHEMA, "critic": CRITIC_OUTPUT_SCHEMA}.get(role)
        if schema is None:
            return None
        schema_root = self.output_schema_dir or (self.cwd / "codex_schemas" if self.cwd is not None else None)
        if schema_root is None:
            return None
        schema_root.mkdir(parents=True, exist_ok=True)
        path = schema_root / f"{role}_output_schema.json"
        text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
        return path


class OpenAICompatibleVisionRunner:
    """Runs prompts through a local OpenAI-compatible VLM server such as vLLM Qwen3-VL."""

    def __init__(
        self,
        *,
        model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        del role
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path)}})
        data: dict[str, Any] | None = None
        last_context_body = ""
        for max_tokens in _qwen_completion_token_budgets(self.max_tokens):
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - local/user-configured endpoint.
                    data = json.loads(response.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as exc:  # pragma: no cover - live endpoint path.
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 400 and _is_qwen_context_budget_error(body) and max_tokens > 128:
                    last_context_body = body
                    continue
                raise RuntimeError(f"Qwen endpoint returned HTTP {exc.code}: {body[:1000]}") from exc
        if data is None:
            raise RuntimeError(f"Qwen endpoint returned context-window errors after token budget retries: {last_context_body[:1000]}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Qwen endpoint response has no choices: {data!r}")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content_text = message.get("content") if isinstance(message, dict) else None
        if isinstance(content_text, list):
            parts = [part.get("text", "") for part in content_text if isinstance(part, dict)]
            content_text = "\n".join(str(part) for part in parts if part)
        if not isinstance(content_text, str) or not content_text.strip():
            raise RuntimeError(f"Qwen endpoint response has empty content: {data!r}")
        return content_text.strip()


def _qwen_completion_token_budgets(max_tokens: int) -> list[int]:
    requested = max(1, int(max_tokens))
    budgets = [requested]
    for fallback in (384, 256, 128):
        if fallback < requested and fallback not in budgets:
            budgets.append(fallback)
    return budgets


def _is_qwen_context_budget_error(body: str) -> bool:
    text = body.lower()
    return "context length" in text and "maximum input length" in text and "max_tokens" not in text


class DeterministicHarnessRunner:
    """Fast local runner used for tests and UI smoke checks.

    This runner intentionally uses a goal-agnostic motion pattern so it cannot
    masquerade as a successful open-vocabulary policy.
    """

    def __init__(self) -> None:
        self._actor_calls = 0

    def reset(self) -> None:
        self._actor_calls = 0

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        del image_paths
        if role == "critic":
            return json.dumps(
                {
                    "verdict": "approve",
                    "reason": "bounded command; no repeated unsafe state detected",
                },
                sort_keys=True,
            )
        self._actor_calls += 1
        if self._actor_calls >= 10:
            action = {"tool": "stop", "args": {}}
            thought = "End the bounded smoke sequence after several generic motion probes."
        elif self._actor_calls % 3 == 1:
            sweep_index = (self._actor_calls - 1) // 3
            degrees = 18.0 if sweep_index % 2 == 0 else -18.0
            action = {"tool": "turn_by_angle", "args": {"degrees": degrees, "power_percent": 10.0}}
            thought = "Sweep the camera toward likely target bearing before committing to forward motion."
        else:
            action = {"tool": "drive_straight", "args": {"power_percent": 22.0, "duration_s": 0.7}}
            thought = "Move forward in a short bounded segment while keeping the next observation available."
        return json.dumps({"thought": thought, "action": action}, sort_keys=True)


class ScriptedOpenVocabRunner:
    """Deterministic Qwen-schema runner for long-horizon simulator smoke demos.

    This is not a research policy. It avoids static object-name branches and
    only echoes the user-provided goal phrase when exercising the visual-servo
    tool.
    """

    def __init__(self, *, visual_servo_detector: str = "florence-mlx") -> None:
        self.visual_servo_detector = visual_servo_detector
        self._actor_calls = 0

    def reset(self) -> None:
        self._actor_calls = 0

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        del image_paths
        if role == "critic":
            return json.dumps({"verdict": "approve", "reason": "scripted bounded demo action"}, sort_keys=True)
        self._actor_calls += 1
        state = _dynamic_task_state_from_prompt(prompt)
        step = int(state.get("step", self._actor_calls - 1) or 0)
        if step >= 2 and step % 4 == 2:
            servo_prompt = _navigation_goal_phrase(str(state.get("goal", "")))
            action = {
                "tool": "visual_servo_object",
                "args": {
                    "prompt": servo_prompt,
                    "duration_s": 1.2,
                    "forward_power": 16.0,
                    "detector": self.visual_servo_detector,
                },
            }
            note = "Use the visible-object servo for a short approach segment toward the user-provided goal phrase."
        elif step % 5 in {0, 1}:
            direction = -1.0 if (step // 5) % 2 else 1.0
            action = {"tool": "turn_by_angle", "args": {"degrees": direction * 24.0, "power_percent": 10.0}}
            note = "Sweep the camera in a bounded arc while preserving motion-strip evidence."
        else:
            action = {"tool": "drive_straight", "args": {"power_percent": 22.0, "duration_s": 0.75}}
            note = "Advance in a short segment after reviewing the latest camera and motion strip."
        return json.dumps(
            {
                "thought": note,
                "action": action,
                "memory_update": {
                    "observation_note": note,
                    "beliefs": [f"goal={state.get('goal', '')}", "use previous motion strip to detect progress"],
                    "next_strategy": "alternate bounded scans and short approach segments",
                },
                "save_frames": [
                    {
                        "id": f"step_{step:03d}_latest",
                        "source": "latest",
                        "frame_index": 0,
                        "note": "latest frame before the selected macro-action",
                    }
                ],
            },
            sort_keys=True,
        )


class SafetyCriticRunner:
    """Non-live critic that rejects obvious loops using only prompt-visible data."""

    def reset(self) -> None:
        pass

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        del role, image_paths
        try:
            payload = json.loads(prompt)
        except json.JSONDecodeError:
            return json.dumps(_critic_json("warn", "critic prompt was not parseable; keep bounded command"), sort_keys=True)
        action = validate_harness_action(parse_prompt_action(payload.get("candidate_action", {})))
        memory = payload.get("recent_memory", [])
        if not isinstance(memory, list):
            memory = []
        verdict = self._evaluate(action, memory)
        return json.dumps(verdict, sort_keys=True)

    def _evaluate(self, action: HarnessAction, memory: list[Any]) -> dict[str, Any]:
        return evaluate_deterministic_safety(action, memory)


class NoopCriticRunner:
    """Critic shim used when the actor should fully own the selected action."""

    def reset(self) -> None:
        pass

    def run(self, prompt: str, *, role: str, image_paths: list[Path] | None = None) -> str:
        del prompt, role, image_paths
        return json.dumps(
            {
                "verdict": "approve",
                "reason": "critic_mode=none; actor owns bounded tool selection and stop decisions",
            },
            sort_keys=True,
        )


class HarnessSession:
    def __init__(
        self,
        *,
        config: HarnessConfig,
        tools: RobotTools,
        actor: LlmRunner,
        critic: LlmRunner,
        rerun_logger: Any | None = None,
    ) -> None:
        self.config = config
        self.tools = tools
        self.actor = actor
        self.critic = critic
        self.rerun = rerun_logger
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "harness_events.jsonl"
        self.memory_path = self.run_dir / "memory.jsonl"
        self.prompt_dir = self.run_dir / "prompts"
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self._status = HarnessStatus()
        self._lock = threading.RLock()
        self._stop_requested = False
        self.log_event("session", {"model": config.model, "reasoning_effort": config.reasoning_effort})
        if self.rerun is not None:
            self._log_rerun_metadata()
            self.rerun.log_state(0, "idle")

    @property
    def mode(self) -> str:
        return self._status.mode

    def start_goal(self, goal: str) -> None:
        goal = goal.strip()
        if not goal:
            raise ValueError("goal cannot be empty")
        with self._lock:
            self._status.goal = goal
            self._status.mode = "auto"
            self._status.step = 0
            self._status.error = None
            self._reset_runners()
            self._stop_requested = False
            self.log_event("user_goal", {"goal": goal})
            self._log_mode("auto")

    def pause(self) -> None:
        with self._lock:
            if self._status.mode in {"auto", "teleop"}:
                self._status.mode = "paused"
                self.log_event("user_pause", {})
                self._log_mode("paused")

    def resume(self) -> None:
        with self._lock:
            if self._status.goal and self._status.mode in {"paused", "teleop"}:
                self._status.mode = "auto"
                self.log_event("user_resume", {})
                self._log_mode("auto")

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._status.mode = "complete"
            self.tools.stop()
            self.log_event("user_stop", {})
            self._log_mode("complete")

    def set_worker_active(self, active: bool) -> None:
        with self._lock:
            self._status.worker_active = active

    def run_auto_step(self) -> dict[str, Any] | None:
        with self._lock:
            if self._stop_requested or self._status.mode != "auto" or not self._status.goal:
                return None
            step = self._status.step
            goal = self._status.goal
        try:
            observation = self.tools.observe(label=f"llm_{step:03d}", timeout_s=3.0)
            observation_summary = sanitize_observation(observation.summary())
            prompt_observation = prompt_safe_observation(observation_summary, root=self.run_dir)
            self._record_observation(observation_summary)
            recent_memory = self.read_memory_tail()
            prompt_memory = prompt_memory_tail(recent_memory)
            previous_motion = compact_prompt_value(latest_motion_summary(recent_memory))
            prompt = build_actor_prompt(
                goal=goal,
                mode="auto",
                step=step,
                memory_path=Path(self.memory_path.name),
                observation=prompt_observation,
                recent_memory=prompt_memory,
                previous_motion=previous_motion,
                prompt_profile=self.config.prompt_profile,
                extra_rules=self.config.actor_rules,
            )
            self._write_prompt(step, "actor", prompt)
            image_paths = _actor_image_paths(observation_summary, recent_memory, root=self.run_dir)
            actor_output = ""
            actor_attempt_outputs: list[str] = []
            try:
                actor_output = self.actor.run(prompt, role="actor", image_paths=image_paths)
                actor_attempt_outputs.append(actor_output)
                try:
                    action = parse_actor_action(actor_output)
                    actor_side_effects = parse_actor_side_effects(actor_output)
                except Exception as first_exc:
                    retry_prompt = build_actor_json_repair_prompt(prompt, error=str(first_exc))
                    self._write_prompt(step, "actor_retry", retry_prompt)
                    self.log_event(
                        "actor_parse_retry",
                        {
                            "step": step,
                            "error": str(first_exc),
                            "prompt_path": str(Path("prompts") / f"{step:03d}_actor_retry.txt"),
                            "previous_output": actor_output[:1200],
                        },
                    )
                    actor_output = self.actor.run(retry_prompt, role="actor", image_paths=image_paths)
                    actor_attempt_outputs.append(actor_output)
                    action = parse_actor_action(actor_output)
                    actor_side_effects = parse_actor_side_effects(actor_output)
            except Exception as exc:
                error_payload = {
                    "error": str(exc),
                    "output": actor_output,
                    "attempt_count": len(actor_attempt_outputs),
                    "previous_output": actor_attempt_outputs[0] if len(actor_attempt_outputs) > 1 else None,
                    "prompt_path": str(Path("prompts") / f"{step:03d}_actor.txt"),
                    "image_paths": [str(path.relative_to(self.run_dir)) if _path_is_relative_to(path, self.run_dir) else path.name for path in image_paths],
                }
                self._log_llm(step, "actor_error", error_payload)
                self.log_event("actor_error", error_payload)
                if not actor_attempt_outputs:
                    raise
                action = HarnessAction("wait", {"duration_s": 0.2}, "actor JSON parse fallback")
                actor_side_effects = {
                    "grounding_audit": {},
                    "memory_update": {"actor_parse_error": str(exc)[:PROMPT_TEXT_LIMIT]},
                    "save_frames": [],
                }
                actor_output = json.dumps(
                    {
                        "thought": "actor JSON parse failed after retry; using bounded wait fallback",
                        "action": {"tool": action.tool, "args": action.args},
                        "grounding_audit": {},
                        "memory_update": actor_side_effects["memory_update"],
                    }
                )
            saved_frames = self._save_requested_frames(
                actor_side_effects["save_frames"],
                observation=observation_summary,
                recent_memory=recent_memory,
            )
            self._log_llm(
                step,
                "actor",
                {
                    "output": actor_output,
                    "parsed_action": action_to_dict(action),
                    "grounding_audit": actor_side_effects["grounding_audit"],
                    "memory_update": actor_side_effects["memory_update"],
                    "saved_frames": saved_frames,
                    "prompt_path": str(Path("prompts") / f"{step:03d}_actor.txt"),
                    "image_paths": [str(path.relative_to(self.run_dir)) if _path_is_relative_to(path, self.run_dir) else path.name for path in image_paths],
                },
            )
            self.log_event("actor", {"step": step, "output": actor_output, "action": action_to_dict(action)})

            critic_prompt = build_critic_prompt(
                goal=goal,
                step=step,
                observation=prompt_observation,
                action=action,
                actor_output=actor_output,
                recent_memory=prompt_memory,
                prompt_profile=self.config.prompt_profile,
                extra_rules=self.config.critic_rules,
            )
            self._write_prompt(step, "critic", critic_prompt)
            critic_output = self.critic.run(critic_prompt, role="critic", image_paths=image_paths)
            model_decision = parse_critic_decision(critic_output)
            safety_decision = apply_deterministic_safety_gate(model_decision, action, recent_memory)
            decision = apply_actor_consistency_guard(safety_decision, action, actor_side_effects, recent_memory)
            self._log_llm(
                step,
                "critic",
                {
                    "output": critic_output,
                    "model_decision": critic_to_dict(model_decision),
                    "final_decision": critic_to_dict(decision),
                    "candidate_action": action_to_dict(action),
                    "prompt_path": str(Path("prompts") / f"{step:03d}_critic.txt"),
                },
            )
            if safety_decision != model_decision:
                self.log_event(
                    "safety_gate",
                    {
                        "step": step,
                        "model_decision": critic_to_dict(model_decision),
                        "safety_decision": critic_to_dict(safety_decision),
                    },
                )
            if decision != safety_decision:
                self.log_event(
                    "actor_consistency_guard",
                    {
                        "step": step,
                        "model_decision": critic_to_dict(safety_decision),
                        "guard_decision": critic_to_dict(decision),
                        "actor_grounding_audit": actor_side_effects["grounding_audit"],
                    },
                )
            self._log_rerun_metadata()
            selected_action = decision.replacement if decision.verdict == "reject" and decision.replacement else action
            self.log_event(
                "critic",
                {
                    "step": step,
                    "output": critic_output,
                    "model_decision": critic_to_dict(model_decision),
                    "decision": critic_to_dict(decision),
                    "selected_action": action_to_dict(selected_action),
                },
            )
            result_summary = prompt_safe_tool_result(self.execute_action(selected_action, source="actor"), root=self.run_dir)
            memory_record = {
                "step": step,
                "goal": goal,
                "mode": "auto",
                "observation": prompt_observation,
                "actor_action": action_to_dict(action),
                "actor_grounding_audit": actor_side_effects["grounding_audit"],
                "actor_memory_update": actor_side_effects["memory_update"],
                "saved_frames": saved_frames,
                "critic": critic_to_dict(decision),
                "executed_action": action_to_dict(selected_action),
                "tool_result": result_summary,
            }
            self.append_memory(memory_record)
            with self._lock:
                self._status.step += 1
                self._status.last_action = action_to_dict(selected_action)
                self._status.last_critic = critic_to_dict(decision)
                if selected_action.tool == "stop" or self._status.step >= self.config.max_steps:
                    self._status.mode = "complete"
                    self._log_mode("complete")
            return memory_record
        except Exception as exc:  # noqa: BLE001 - errors are surfaced in UI/logs.
            with self._lock:
                self._status.mode = "error"
                self._status.error = str(exc)
                self._log_mode("error")
            self.log_event("error", {"message": str(exc)})
            raise

    def teleop(self, command: str, value: float | None = None) -> dict[str, Any]:
        mapping = {
            "forward": HarnessAction("drive_straight", {"power_percent": 20.0, "duration_s": value or 0.5}, "teleop"),
            "left": HarnessAction("turn_by_angle", {"degrees": -(value or 18.0), "power_percent": 10.0}, "teleop"),
            "right": HarnessAction("turn_by_angle", {"degrees": value or 18.0, "power_percent": 10.0}, "teleop"),
            "stop": HarnessAction("stop", {}, "teleop"),
        }
        if command not in mapping:
            raise ValueError(f"unknown teleop command: {command}")
        with self._lock:
            self._status.mode = "teleop"
            self._log_mode("teleop")
        result = self.execute_action(mapping[command], source="teleop")
        self.log_event("teleop", {"command": command, "result": result})
        return result

    def execute_action(self, action: HarnessAction, *, source: str) -> dict[str, Any]:
        action = validate_harness_action(action)
        if self.rerun is not None:
            self.rerun.log_command(self._status.step, source, action_to_dict(action))
        if action.tool == "observe":
            obs = self.tools.observe(label=f"{source}_observe_{self._status.step:03d}", timeout_s=3.0)
            summary = sanitize_observation(obs.summary())
            self._record_observation(summary)
            return {"observation": summary}
        if action.tool == "turn_by_angle":
            result = self.tools.turn_by_angle(
                action.args["degrees"],
                power_percent=action.args.get("power_percent", 10.0),
            )
            return motion_result_summary(result)
        if action.tool == "drive_straight":
            result = self.tools.drive_straight(action.args["power_percent"], action.args["duration_s"])
            return motion_result_summary(result)
        if action.tool == "visual_servo_object":
            result = self.tools.visual_servo_object(
                action.args["prompt"],
                duration_s=action.args["duration_s"],
                detector=action.args.get("detector"),
                forward_power=action.args.get("forward_power", 18.0),
            )
            return motion_result_summary(result)
        if action.tool == "check_object_grounding":
            with self._lock:
                observation = dict(self._status.last_observation or {})
            image_path_text = observation.get("path")
            if not image_path_text:
                return {
                    "action": "check_object_grounding",
                    "ok": False,
                    "reason": "no_latest_observation_frame",
                    "prompt": action.args["prompt"],
                }
            result = self.tools.check_object_grounding(
                image_path=Path(str(image_path_text)),
                prompt=action.args["prompt"],
                detector=action.args.get("detector"),
            )
            return motion_result_summary(result)
        if action.tool == "query_topomap_memory":
            with self._lock:
                observation = dict(self._status.last_observation or {})
                goal_query = str(action.args.get("goal_query") or self._status.goal or "").strip()
            image_path_text = observation.get("path")
            if not image_path_text:
                return {
                    "action": "query_topomap_memory",
                    "ok": False,
                    "reason": "no_latest_observation_frame",
                    "goal_query": goal_query,
                }
            result = self.tools.query_topomap_memory(image_path=Path(str(image_path_text)), goal_query=goal_query)
            return motion_result_summary(result)
        if action.tool == "stop":
            result = self.tools.stop()
            return motion_result_summary(result)
        if action.tool == "wait":
            duration_s = action.args["duration_s"]
            if self.config.sleep_scale > 0:
                time.sleep(duration_s * self.config.sleep_scale)
            return {"command": "wait", "duration_s": duration_s}
        raise ValueError(f"unsupported action: {action.tool}")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._status) | {
                "run_dir": str(self.run_dir),
                "events_path": str(self.events_path),
                "memory_path": str(self.memory_path),
                "metadata": self.metadata(),
                "recent_events": self.read_events_tail(80),
                "recent_memory": self.read_memory_tail(12),
            }

    def metadata(self) -> dict[str, Any]:
        schema_dir = self.run_dir / "codex_schemas"
        rerun_path = getattr(self.rerun, "save_path", None)
        return {
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "max_steps": self.config.max_steps,
            "prompt_profile": self.config.prompt_profile,
            "actor_rules": list(self.config.actor_rules),
            "critic_rules": list(self.config.critic_rules),
            "critic_mode": self.config.critic_mode,
            "rerun_enabled": self.config.rerun_enabled,
            "rerun_path": str(rerun_path) if rerun_path is not None else None,
            "codex_schema_dir": str(schema_dir) if schema_dir.exists() else None,
            "actor_runner": self.actor.__class__.__name__,
            "critic_runner": self.critic.__class__.__name__,
            "live_codex": isinstance(self.actor, CodexExecRunner),
            "tool_backend": self.tools.__class__.__name__,
            "policy_input_allowlist": POLICY_INPUT_ALLOWLIST,
            "model_facing_paths": "relative",
        }

    def log_event(self, event: str, payload: dict[str, Any]) -> None:
        record = {"t": time.time(), "event": event, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        if self.rerun is not None:
            self.rerun.log_event(self._status.step, event, payload)

    def append_memory(self, record: dict[str, Any]) -> None:
        with self.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def read_memory_tail(self, limit: int | None = None) -> list[dict[str, Any]]:
        return _read_jsonl_tail(self.memory_path, limit or self.config.max_memory_records)

    def read_events_tail(self, limit: int = 80) -> list[dict[str, Any]]:
        return _read_jsonl_tail(self.events_path, limit)

    def close(self) -> None:
        self.tools.close()
        if self.rerun is not None and hasattr(self.rerun, "close"):
            self.rerun.close()

    def _record_observation(self, observation_summary: dict[str, Any]) -> None:
        with self._lock:
            self._status.last_observation = observation_summary
            self._status.latest_frame_path = observation_summary.get("path")
        self.log_event("observation", observation_summary)
        if self.rerun is not None:
            self.rerun.log_observation(self._status.step, observation_summary)

    def _write_prompt(self, step: int, role: str, prompt: str) -> None:
        path = self.prompt_dir / f"{step:03d}_{role}.txt"
        path.write_text(prompt, encoding="utf-8")

    def _log_mode(self, mode: str) -> None:
        if self.rerun is not None:
            self.rerun.log_state(self._status.step, mode)

    def _log_rerun_metadata(self) -> None:
        if self.rerun is not None and hasattr(self.rerun, "log_metadata"):
            self.rerun.log_metadata(self.metadata())

    def _log_llm(self, step: int, role: str, payload: dict[str, Any]) -> None:
        if self.rerun is not None and hasattr(self.rerun, "log_llm"):
            self.rerun.log_llm(step, role, payload)

    def _save_requested_frames(
        self,
        requests: list[dict[str, Any]],
        *,
        observation: dict[str, Any],
        recent_memory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        sources = {
            "latest": _observation_image_paths(observation),
            "previous_motion": _latest_motion_frame_paths(recent_memory, root=self.run_dir),
        }
        memory_dir = self.run_dir / "memory_frames"
        memory_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict[str, Any]] = []
        for index, request in enumerate(requests[:6]):
            source_name = str(request.get("source", "latest")).strip().lower()
            source_paths = sources.get(source_name, sources["latest"])
            if not source_paths:
                continue
            try:
                frame_index = int(request.get("frame_index", 0))
            except (TypeError, ValueError):
                frame_index = 0
            frame_index = max(0, min(len(source_paths) - 1, frame_index))
            src = source_paths[frame_index]
            if not src.exists():
                continue
            requested_id = str(request.get("id") or f"step_{self._status.step:03d}_{index:02d}")
            safe_id = _safe_artifact_id(requested_id)
            dest = memory_dir / f"{self._status.step:03d}_{index:02d}_{safe_id}{src.suffix or '.jpg'}"
            shutil.copyfile(src, dest)
            record = {
                "id": safe_id,
                "path": _model_facing_path(dest, root=self.run_dir),
                "source": source_name,
                "frame_index": frame_index,
                "note": str(request.get("note", ""))[:240],
            }
            saved.append(record)
            self.log_event("save_frame", record)
        return saved

    def _reset_runners(self) -> None:
        seen: set[int] = set()
        for runner in (self.actor, self.critic):
            identity = id(runner)
            if identity in seen:
                continue
            seen.add(identity)
            reset = getattr(runner, "reset", None)
            if callable(reset):
                reset()


def _clean_extra_rules(extra_rules: tuple[str, ...] | list[str] | None) -> list[str]:
    cleaned: list[str] = []
    for rule in extra_rules or ():
        text = str(rule).strip()
        if text:
            cleaned.append(text[:500])
    return cleaned


def build_actor_prompt(
    *,
    goal: str,
    mode: str,
    step: int,
    memory_path: Path,
    observation: dict[str, Any],
    recent_memory: list[dict[str, Any]],
    previous_motion: dict[str, Any] | None = None,
    prompt_profile: str = "baseline",
    extra_rules: tuple[str, ...] | list[str] | None = None,
) -> str:
    rules = [
        "Use only the provided camera/IMU observation, memory, and tool results.",
        "When an image is attached, treat it as the authoritative latest RGB camera frame.",
        "Image attachments are ordered as: latest RGB frame; previous check_object_grounding detector overlay if present; previous raw/detector paired grounding audit strip if present; previous raw motion strip if present; previous detector debug overlay strip if present; topomap contact sheet if present.",
        "When a previous raw/detector paired grounding audit strip is attached, read columns left-to-right; each column shows the same moment as raw camera above and detector overlay below.",
        "When a previous raw motion strip is attached, read it left-to-right as evenly spaced frames from the last tool call.",
        "When a previous detector debug overlay strip is attached, inspect the boxes/labels yourself; if the box is on the wrong object, treat moved=true as a detector grounding failure and switch strategy or use a more precise visible phrase.",
        "After any visual_servo_object result with a grounding_audit_contact_sheet, fill grounding_audit before choosing an action: state whether the previous detector box overlapped the intended object instance.",
        "After any check_object_grounding result with an overlay attachment, fill grounding_audit.previous_check_box_matches_intended_object and check_overlay_evidence before choosing visual_servo_object.",
        "Only repeat the same visual_servo_object prompt when grounding_audit.previous_visual_servo_box_matches_intended_object is true and the latest RGB frame still supports that same target.",
        "If grounding_stability is sparse_detection_coverage, insufficient_status_history, or no_detection, treat moved=true as weak control evidence and do not claim the box consistently tracked the intended object.",
        "If the detector box is on a nearby surface, cabinet, wall, or other object instead of the intended instance, set next_prompt_should_change=true and choose a different bounded action or a more specific visible phrase.",
        "For visual_servo_object in clutter, prefer a precise visible phrase that identifies the intended instance by category plus color/material, part, or image location, not a bare category prompt.",
        "If a grounding audit or overlay shows the detector box on a nearby wrong object, do not repeat the same prompt; change viewpoint or use a more specific visible phrase selected from the latest RGB frame.",
        "check_object_grounding is non-motion; use it when the latest RGB frame has a plausible visible candidate but the previous visual servo grounding was sparse, absent, or on the wrong region.",
        "Treat check_object_grounding ready_for_visual_servo=true as detector-box existence only, not proof that the box is on the intended object.",
        "If check_object_grounding returns ready_for_visual_servo=false, grounding_geometry_warning, an edge-clipped box, or an overlay on the wrong region, change viewpoint or prompt before visual_servo_object.",
        "Treat brightness_center as a low-level camera summary that may be wrong or incomplete.",
        "Do not assume access to map coordinates, object metadata, target distance, wheel encoders, or simulator state.",
        "Use the image, IMU yaw, and recent motion history to choose exactly one bounded action.",
        "visual_servo_object can steer only toward a currently visible object or landmark; it cannot discover a hidden goal object by itself.",
        "The visual_servo_object prompt may name the final goal or any useful visible waypoint selected from the latest RGB frame.",
        "If the final goal is not clearly visible, do not call visual_servo_object with the final-goal phrase just to search; use a visible waypoint, bounded turn, or short drive to change viewpoint.",
        "After using a non-goal waypoint, re-check the latest RGB frame and previous motion strip; turn or rescan when that waypoint dominates the view or stops improving goal evidence.",
        "Treat visual_servo_object moved=true and last_detection as evidence that some phrase-grounded track moved the robot, not proof of semantic identity or goal completion.",
        "A previous visual_servo_object prompt is control history, not proof that the named object was truly visible; verify the current RGB frame yourself before repeating it.",
        "query_topomap_memory is a non-motion lookup; use it when route/image memory could help choose where to explore next, then inspect its returned contact sheet on the next step.",
        "Treat topomap route hints as memory evidence, not ground-truth completion; prefer the live camera when topomap memory disagrees with the current RGB frame.",
        "If visual_servo_object reports moved=false or failure_reason, switch strategy with a bounded turn or drive instead of waiting.",
        "The harness already captures a fresh observation before each actor call; do not request another observation as your action.",
        "Prefer short bounded movements; turn to search or center the target, drive only briefly when the path or target evidence is plausible.",
        "You may write scratchpad facts in memory_update and request durable frame copies in save_frames.",
        "For save_frames, source must be latest or previous_motion; frame_index is zero-based.",
        "Do not stop unless repeated observations show the described goal object is reached or very close, or an operator/evaluator has requested stop.",
        "Return exactly one JSON object and no prose.",
    ]
    rules.extend(_clean_extra_rules(extra_rules))
    static_prefix = {
        "role": "flat_disk_robot_actor",
        "prompt_profile": prompt_profile,
        "rules": rules,
        "tool_contract": TOOL_CONTRACT,
        "output_schema": {
            "thought": "short private control rationale",
            "action": {"tool": "one of turn_by_angle, drive_straight, visual_servo_object, check_object_grounding, query_topomap_memory, stop, wait", "args": "object"},
            "grounding_audit": "required after visual_servo_object or check_object_grounding history; object with previous_visual_servo_box_matches_intended_object, previous_check_box_matches_intended_object, evidence, check_overlay_evidence, next_prompt_should_change",
            "memory_update": "optional object with observation_note, beliefs, next_strategy, or other non-privileged scratchpad facts",
            "save_frames": "optional list of {id, source, frame_index, note} requests",
        },
        "memory_log_path": str(memory_path),
    }
    dynamic_state = {
        "goal": goal,
        "mode": mode,
        "step": step,
        "observation": sanitize_observation(observation),
        "previous_motion": compact_prompt_value(previous_motion or {}),
        "recent_memory": prompt_memory_tail(recent_memory),
    }
    return (
        "STATIC_HARNESS_CONTEXT\n"
        + json.dumps(static_prefix, indent=2, sort_keys=True, default=str)
        + "\n\nDYNAMIC_TASK_STATE\n"
        + json.dumps(dynamic_state, indent=2, sort_keys=True, default=str)
        + "\n"
    )


def build_actor_json_repair_prompt(prompt: str, *, error: str) -> str:
    repair = {
        "role": "actor_json_repair",
        "error": str(error)[:240],
        "rules": [
            "The previous actor output was invalid JSON or was truncated.",
            "Return exactly one complete JSON object and no prose.",
            "Keep thought, evidence, check_overlay_evidence, and memory_update strings concise.",
            "Choose exactly one bounded action using the same tool contract and latest attached images.",
        ],
    }
    return prompt + "\n\nACTOR_JSON_REPAIR\n" + json.dumps(repair, indent=2, sort_keys=True) + "\n"


def build_critic_prompt(
    *,
    goal: str,
    step: int,
    observation: dict[str, Any],
    action: HarnessAction,
    actor_output: str,
    recent_memory: list[dict[str, Any]],
    prompt_profile: str = "baseline",
    extra_rules: tuple[str, ...] | list[str] | None = None,
) -> str:
    rules = [
        "Approve only bounded actions that make sense from camera/IMU and memory.",
        "When an image is attached, evaluate it as the authoritative latest RGB camera frame.",
        "Image attachments are ordered as: latest RGB frame; previous check_object_grounding detector overlay if present; previous raw/detector paired grounding audit strip if present; previous raw motion strip if present; previous detector debug overlay strip if present; topomap contact sheet if present.",
        "When a previous raw/detector paired grounding audit strip is attached, compare each raw frame with its detector overlay before trusting moved=true or last_detection.",
        "When a previous detector debug overlay strip is attached, check whether the detector box is actually on the named object before trusting moved=true or last_detection.",
        "When a previous check_object_grounding overlay is attached, check whether the selected box is actually on the intended object before approving a follow-up visual_servo_object.",
        "Warn or reject claims of consistent tracking when grounding_stability is not status_track_present.",
        "Warn or reject repeating the same visual_servo_object prompt when the actor did not audit the previous detector box or the audit evidence is ambiguous.",
        "Warn when the actor treats ready_for_visual_servo=true as semantic proof without inspecting the overlay, especially when grounding_geometry_warning or edge contact is present.",
        "Treat brightness_center as a low-level camera summary that may be wrong or incomplete.",
        "Reject unsafe, looping, ungrounded, or impossible commands.",
        "Reject visual_servo_object with the final-goal phrase when the actor is using it merely to search and the latest image lacks clear final-goal evidence.",
        "Approve visual_servo_object toward a non-goal visible landmark when it is a bounded waypoint move intended to improve viewpoint.",
        "Approve check_object_grounding when it tests a plausible visible phrase before a risky or repeated visual_servo_object call.",
        "Approve query_topomap_memory when route/image memory could guide exploration; it is non-motion and should not be treated as goal completion.",
        "Reject stop unless repeated observations show the described goal object is reached or very close.",
        "Do not request or use privileged simulator state.",
        "Return exactly one JSON object and no prose.",
    ]
    rules.extend(_clean_extra_rules(extra_rules))
    payload = {
        "role": "flat_disk_robot_critic",
        "prompt_profile": prompt_profile,
        "rules": rules,
        "output_schema": {
            "verdict": "approve, warn, or reject",
            "reason": "short critical reason",
            "replacement_action": "optional action object with tool/args when rejecting",
        },
        "goal": goal,
        "step": step,
        "observation": sanitize_observation(observation),
        "recent_memory": prompt_memory_tail(recent_memory),
        "actor_output": actor_output,
        "candidate_action": action_to_dict(action),
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def parse_actor_action(text: str) -> HarnessAction:
    payload = parse_json_object(text)
    action_payload = payload.get("action", payload)
    if isinstance(action_payload, str):
        action_payload = {"tool": action_payload, "args": payload.get("args", {})}
    if not isinstance(action_payload, dict):
        raise ValueError("actor action must be a JSON object")
    tool = str(action_payload.get("tool", action_payload.get("action", ""))).strip()
    args = action_payload.get("args", {})
    if not isinstance(args, dict):
        args = {}
    thought = str(payload.get("thought", payload.get("reason", "")))[:500]
    return validate_harness_action(HarnessAction(tool=tool, args=args, thought=thought))


def parse_actor_side_effects(text: str) -> dict[str, Any]:
    payload = parse_json_object(text)
    grounding_audit = payload.get("grounding_audit")
    if not isinstance(grounding_audit, dict):
        grounding_audit = {}
    memory_update = payload.get("memory_update", {})
    if not isinstance(memory_update, dict):
        memory_update = {}
    save_frames = payload.get("save_frames", [])
    if not isinstance(save_frames, list):
        save_frames = []
    cleaned_requests: list[dict[str, Any]] = []
    for item in save_frames:
        if not isinstance(item, dict):
            continue
        cleaned_requests.append(
            {
                "id": str(item.get("id", ""))[:80],
                "source": str(item.get("source", "latest"))[:40],
                "frame_index": item.get("frame_index", 0),
                "note": str(item.get("note", ""))[:240],
            }
        )
    return {
        "grounding_audit": compact_prompt_value(grounding_audit),
        "memory_update": sanitize_memory(memory_update),
        "save_frames": cleaned_requests,
    }


def parse_critic_decision(text: str) -> CriticDecision:
    payload = parse_json_object(text)
    verdict = str(payload.get("verdict", "approve")).strip().lower()
    if verdict not in {"approve", "warn", "reject"}:
        verdict = "warn"
    reason = str(payload.get("reason", ""))[:500]
    replacement_payload = payload.get("replacement_action")
    replacement = None
    if isinstance(replacement_payload, dict):
        replacement = parse_actor_action(json.dumps({"action": replacement_payload}))
    return CriticDecision(verdict=verdict, reason=reason, replacement=replacement)


def apply_deterministic_safety_gate(
    model_decision: CriticDecision,
    action: HarnessAction,
    recent_memory: list[Any],
) -> CriticDecision:
    del action, recent_memory
    return model_decision


def apply_actor_consistency_guard(
    decision: CriticDecision,
    action: HarnessAction,
    actor_side_effects: dict[str, Any],
    recent_memory: list[Any],
) -> CriticDecision:
    if decision.verdict == "reject":
        return decision
    if action.tool != "visual_servo_object":
        return decision
    audit = actor_side_effects.get("grounding_audit")
    if not isinstance(audit, dict) or not _actor_audit_requires_prompt_change(audit):
        return decision
    previous_prompt = _latest_visual_servo_prompt(recent_memory)
    current_prompt = str(action.args.get("prompt", "")).strip()
    if not previous_prompt or _normalize_prompt(previous_prompt) != _normalize_prompt(current_prompt):
        return decision
    return CriticDecision(
        verdict="reject",
        reason="actor grounding_audit requested a changed visual-servo prompt, but action repeated the previous prompt",
        replacement=HarnessAction("wait", {"duration_s": 0.2}, "actor consistency guard"),
    )


def _actor_audit_requires_prompt_change(audit: dict[str, Any]) -> bool:
    if audit.get("next_prompt_should_change") is True:
        return True
    return audit.get("previous_visual_servo_box_matches_intended_object") is False


def _latest_visual_servo_prompt(recent_memory: list[Any]) -> str:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        action = record.get("executed_action") or record.get("actor_action")
        if not isinstance(action, dict) or action.get("tool") != "visual_servo_object":
            continue
        args = action.get("args")
        if isinstance(args, dict):
            prompt = str(args.get("prompt", "")).strip()
            if prompt:
                return prompt
    return ""


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def evaluate_deterministic_safety(action: HarnessAction, memory: list[Any]) -> dict[str, Any]:
    recent_actions = [_memory_action(record) for record in memory[-4:] if isinstance(record, dict)]
    if action.tool == "stop" and len(memory) < 6 and not _recent_visual_target(memory):
        return _critic_json(
            "reject",
            "stop is too early without repeated target evidence",
            HarnessAction("drive_straight", {"power_percent": 20.0, "duration_s": 0.5}, "critic replacement"),
        )
    if action.tool == "turn_by_angle" and _same_turn_direction(action, recent_actions[-2:]):
        return _critic_json(
            "reject",
            "third same-direction turn risks scanning loop",
            HarnessAction("drive_straight", {"power_percent": 20.0, "duration_s": 0.45}, "critic replacement"),
        )
    if action.tool == "drive_straight" and _stale_drive_sequence(memory[-5:]):
        return _critic_json(
            "reject",
            "repeated drives without visual change; rescan before continuing",
            HarnessAction("turn_by_angle", {"degrees": 24.0, "power_percent": 10.0}, "critic replacement"),
        )
    return _critic_json("approve", "bounded command; no loop or early-stop pattern detected")


def validate_harness_action(action: HarnessAction) -> HarnessAction:
    tool = action.tool.strip()
    if tool not in ALLOWED_ACTIONS:
        return HarnessAction("wait", {"duration_s": 0.2}, f"invalid action replaced: {tool}")
    args = dict(action.args)
    if tool == "turn_by_angle":
        args = {
            "degrees": clamp_float(args.get("degrees", 0.0), -35.0, 35.0),
            "power_percent": clamp_float(args.get("power_percent", 10.0), 8.0, 14.0),
        }
    elif tool == "drive_straight":
        args = {
            "power_percent": clamp_float(args.get("power_percent", 20.0), 18.0, 24.0),
            "duration_s": clamp_float(args.get("duration_s", 0.5), 0.25, 0.9),
        }
    elif tool == "visual_servo_object":
        prompt = str(args.get("prompt", "")).strip()[:160]
        if not prompt:
            return HarnessAction("wait", {"duration_s": 0.2}, "visual_servo_object missing prompt")
        detector = str(args.get("detector") or "").strip()[:80] or None
        args = {
            "prompt": prompt,
            "duration_s": clamp_float(args.get("duration_s", 2.0), 0.5, 4.0),
            "forward_power": clamp_float(args.get("forward_power", args.get("power_percent", 18.0)), 8.0, 24.0),
        }
        if detector is not None:
            args["detector"] = detector
    elif tool == "check_object_grounding":
        prompt = str(args.get("prompt", "")).strip()[:160]
        if not prompt:
            return HarnessAction("wait", {"duration_s": 0.2}, "check_object_grounding missing prompt")
        detector = str(args.get("detector") or "").strip()[:80] or None
        args = {"prompt": prompt}
        if detector is not None:
            args["detector"] = detector
    elif tool == "query_topomap_memory":
        args = {"goal_query": str(args.get("goal_query", "")).strip()[:160]}
    elif tool == "wait":
        args = {"duration_s": clamp_float(args.get("duration_s", 0.2), 0.05, 1.0)}
    else:
        args = {}
    return HarnessAction(tool=tool, args=args, thought=action.thought[:500])


def sanitize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {key: observation[key] for key in sorted(ALLOWED_OBSERVATION_KEYS) if key in observation}


def prompt_safe_observation(observation: dict[str, Any], *, root: Path) -> dict[str, Any]:
    safe = sanitize_observation(observation)
    path_text = safe.get("path")
    if not path_text:
        return safe
    safe["path"] = _model_facing_path(Path(str(path_text)), root=root)
    return safe


def prompt_safe_tool_result(value: Any, *, root: Path) -> Any:
    if isinstance(value, list):
        return [prompt_safe_tool_result(item, root=root) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if (
                key_lower.endswith("_path")
                or key_lower.endswith("_contact_sheet")
                or key_lower in {"path", "stitched_path"}
            ):
                cleaned[key_text] = _model_facing_path(Path(str(item)), root=root) if item else item
            elif key_lower.endswith("_paths"):
                if isinstance(item, list):
                    cleaned[key_text] = [_model_facing_path(Path(str(path)), root=root) for path in item]
                else:
                    cleaned[key_text] = []
            else:
                cleaned[key_text] = prompt_safe_tool_result(item, root=root)
        return cleaned
    return value


def prompt_memory_tail(records: list[dict[str, Any]], limit: int = PROMPT_MEMORY_RECORD_LIMIT) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return [compact_prompt_record(record) for record in records[-limit:] if isinstance(record, dict)]


def compact_prompt_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    if "step" in record:
        cleaned["step"] = record["step"]
    observation = record.get("observation")
    if isinstance(observation, dict):
        cleaned["observation"] = sanitize_observation(observation)
    actor_action = _compact_action_payload(record.get("actor_action"))
    executed_action = _compact_action_payload(record.get("executed_action"))
    if executed_action:
        cleaned["executed_action"] = executed_action
    if actor_action and actor_action != executed_action:
        cleaned["actor_action"] = actor_action
    actor_grounding_audit = record.get("actor_grounding_audit")
    if isinstance(actor_grounding_audit, dict) and actor_grounding_audit:
        cleaned["actor_grounding_audit"] = compact_prompt_value(actor_grounding_audit)
    memory_update = record.get("actor_memory_update")
    if isinstance(memory_update, dict):
        cleaned["actor_memory_update"] = compact_prompt_value(memory_update)
    saved_frames = record.get("saved_frames")
    if isinstance(saved_frames, list) and saved_frames:
        cleaned["saved_frames"] = compact_prompt_value(saved_frames[:PROMPT_LIST_LIMIT])
    critic = _compact_critic_payload(record.get("critic"))
    if critic:
        cleaned["critic"] = critic
    tool_result = record.get("tool_result")
    if isinstance(tool_result, dict):
        cleaned["tool_result"] = _compact_tool_result(tool_result)
    return cleaned


def compact_prompt_value(value: Any) -> Any:
    value = sanitize_memory(value)
    if isinstance(value, str):
        return value[:PROMPT_TEXT_LIMIT]
    if isinstance(value, list):
        return [compact_prompt_value(item) for item in value[:PROMPT_LIST_LIMIT]]
    if isinstance(value, dict):
        return {str(key): compact_prompt_value(item) for key, item in value.items()}
    return value


def _compact_action_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    tool = str(value.get("tool", value.get("action", ""))).strip()
    if not tool:
        return {}
    args = value.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return {"tool": tool, "args": compact_prompt_value(args)}


def _compact_critic_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {
        "verdict": str(value.get("verdict", ""))[:40],
        "reason": str(value.get("reason", ""))[:PROMPT_TEXT_LIMIT],
    }
    replacement = _compact_action_payload(value.get("replacement"))
    if replacement:
        compact["replacement"] = replacement
    return {key: item for key, item in compact.items() if item != "" and item != {}}


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    safe = sanitize_memory(result)
    for key, item in safe.items():
        key_text = str(key)
        if key_text in PROMPT_TOOL_RESULT_KEYS:
            cleaned[key_text] = compact_prompt_value(item)
    return cleaned


def sanitize_memory(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize_memory(item) for item in value]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_lower in {
                "thought",
                "stdout_tail",
                "stderr_tail",
                "debug_overlay_frame_paths",
                "last_detection",
                "last_detection_label",
                "last_detection_source",
                "last_detection_score",
                "target_detected",
                "ever_detected",
                "detections",
            }:
                continue
            if key_lower in FORBIDDEN_PRIVILEGED_KEYS or any(token in key_lower for token in FORBIDDEN_PRIVILEGED_KEYS):
                continue
            cleaned[key_text] = sanitize_memory(item)
        return cleaned
    return value


def action_to_dict(action: HarnessAction) -> dict[str, Any]:
    return {"tool": action.tool, "args": action.args, "thought": action.thought}


def critic_to_dict(decision: CriticDecision) -> dict[str, Any]:
    return {
        "verdict": decision.verdict,
        "reason": decision.reason,
        "replacement": action_to_dict(decision.replacement) if decision.replacement else None,
    }


def motion_result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {"ok": True}
    if hasattr(result, "summary"):
        summary = result.summary()
        if isinstance(summary, dict):
            return summary
    if isinstance(result, dict):
        return result
    return {"result": str(result)}


def parse_prompt_action(payload: Any) -> HarnessAction:
    if not isinstance(payload, dict):
        return HarnessAction("wait", {"duration_s": 0.2}, "missing action")
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return HarnessAction(
        tool=str(payload.get("tool", payload.get("action", ""))),
        args=args,
        thought=str(payload.get("thought", "")),
    )


def _critic_json(verdict: str, reason: str, replacement: HarnessAction | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"verdict": verdict, "reason": reason}
    if replacement is not None:
        checked = validate_harness_action(replacement)
        payload["replacement_action"] = {"tool": checked.tool, "args": checked.args}
    return payload


def _memory_action(record: dict[str, Any]) -> HarnessAction:
    return validate_harness_action(parse_prompt_action(record.get("executed_action", {})))


def _turn_direction(action: HarnessAction) -> int:
    if action.tool != "turn_by_angle":
        return 0
    degrees = float(action.args.get("degrees", 0.0))
    if degrees > 0:
        return 1
    if degrees < 0:
        return -1
    return 0


def _same_turn_direction(action: HarnessAction, recent_actions: list[HarnessAction]) -> bool:
    direction = _turn_direction(action)
    if direction == 0 or len(recent_actions) < 2:
        return False
    return all(_turn_direction(recent) == direction for recent in recent_actions)


def _recent_visual_target(memory: list[Any]) -> bool:
    for record in memory[-4:]:
        if not isinstance(record, dict):
            continue
        observation = record.get("observation", {})
        if not isinstance(observation, dict):
            continue
        detections = observation.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if isinstance(detection, dict) and float(detection.get("confidence", 0.0)) >= 0.45:
                return True
    return False


def _stale_drive_sequence(records: list[Any]) -> bool:
    if len(records) < 5:
        return False
    actions: list[HarnessAction] = []
    brightness: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            return False
        action = _memory_action(record)
        if action.tool != "drive_straight":
            return False
        actions.append(action)
        observation = record.get("observation", {})
        if not isinstance(observation, dict) or "brightness_center" not in observation:
            return False
        brightness.append(float(observation["brightness_center"]))
    del actions
    return max(brightness) - min(brightness) < 0.015


def _observation_image_paths(observation: dict[str, Any]) -> list[Path]:
    path_text = observation.get("path")
    if not path_text:
        return []
    path = Path(str(path_text))
    return [path] if path.exists() else []


def _actor_image_paths(observation: dict[str, Any], recent_memory: list[dict[str, Any]], *, root: Path) -> list[Path]:
    paths = _observation_image_paths(observation)
    grounding_check_overlay = _latest_grounding_check_overlay_path(recent_memory, root=root)
    if grounding_check_overlay is not None and grounding_check_overlay.exists() and grounding_check_overlay not in paths:
        paths.append(grounding_check_overlay)
    grounding_audit_strip = _latest_grounding_audit_contact_sheet(recent_memory, root=root)
    if grounding_audit_strip is not None and grounding_audit_strip.exists():
        paths.append(grounding_audit_strip)
    else:
        previous_strip = _latest_motion_contact_sheet(recent_memory, root=root)
        if previous_strip is not None and previous_strip.exists():
            paths.append(previous_strip)
        debug_overlay_strip = _latest_debug_overlay_contact_sheet(recent_memory, root=root)
        if debug_overlay_strip is not None and debug_overlay_strip.exists():
            paths.append(debug_overlay_strip)
    topomap_sheet = _latest_topomap_contact_sheet(recent_memory, root=root)
    if topomap_sheet is not None and topomap_sheet.exists():
        paths.append(topomap_sheet)
    return paths


def latest_motion_summary(recent_memory: list[dict[str, Any]]) -> dict[str, Any]:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        frame_paths = result.get("motion_frame_paths")
        strip = result.get("motion_contact_sheet") or result.get("stitched_path")
        if frame_paths or strip:
            return {
                "executed_action": record.get("executed_action", {}),
                "tool_result": result,
                "reading_order": (
                    "motion strips read left-to-right; grounding_audit_contact_sheet pairs raw camera "
                    "over detector overlay for the same moment in each column"
                ),
            }
    return {}


def _latest_motion_contact_sheet(recent_memory: list[dict[str, Any]], *, root: Path) -> Path | None:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        path_text = result.get("motion_contact_sheet") or result.get("stitched_path")
        if path_text:
            return _resolve_model_path(str(path_text), root=root)
    return None


def _latest_grounding_audit_contact_sheet(recent_memory: list[dict[str, Any]], *, root: Path) -> Path | None:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        path_text = result.get("grounding_audit_contact_sheet")
        if path_text:
            return _resolve_model_path(str(path_text), root=root)
    return None


def _latest_debug_overlay_contact_sheet(recent_memory: list[dict[str, Any]], *, root: Path) -> Path | None:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        path_text = result.get("debug_overlay_contact_sheet")
        if path_text:
            return _resolve_model_path(str(path_text), root=root)
    return None


def _latest_grounding_check_overlay_path(recent_memory: list[dict[str, Any]], *, root: Path) -> Path | None:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict) or result.get("action") != "check_object_grounding":
            continue
        path_text = result.get("overlay_path")
        if path_text:
            return _resolve_model_path(str(path_text), root=root)
    return None


def _latest_topomap_contact_sheet(recent_memory: list[dict[str, Any]], *, root: Path) -> Path | None:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        path_text = result.get("topomap_contact_sheet")
        if path_text:
            return _resolve_model_path(str(path_text), root=root)
    return None


def _latest_motion_frame_paths(recent_memory: list[dict[str, Any]], *, root: Path) -> list[Path]:
    for record in reversed(recent_memory):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict):
            continue
        paths = result.get("motion_frame_paths")
        if isinstance(paths, list):
            resolved = [_resolve_model_path(str(path), root=root) for path in paths]
            return [path for path in resolved if path.exists()]
    return []


def _resolve_model_path(path_text: str, *, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _model_facing_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def _safe_artifact_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    text = text.strip("._-")
    return (text or "frame")[:80]


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _navigation_goal_phrase(goal: str) -> str:
    text = re.sub(r"\s+", " ", goal.strip())
    text = re.sub(r"^(go|drive|navigate|move|head|travel)\s+(to|toward|towards)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.split(r"\s+(in|inside|near|by|at|within)\s+", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.strip(" .,:;")
    return text[:80] or "visible goal-relevant landmark"


def _dynamic_task_state_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "DYNAMIC_TASK_STATE"
    if marker not in prompt:
        return {}
    text = prompt.split(marker, 1)[1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _read_jsonl_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records
