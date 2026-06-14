from __future__ import annotations

import io
import json
import urllib.error

import pytest

from flatdisk_sim.llm_harness import (
    CodexExecRunner,
    CriticDecision,
    DeterministicHarnessRunner,
    HarnessAction,
    HarnessConfig,
    HarnessSession,
    OpenAICompatibleVisionRunner,
    SafetyCriticRunner,
    ScriptedOpenVocabRunner,
    apply_actor_consistency_guard,
    build_actor_prompt,
    build_critic_prompt,
    parse_critic_decision,
    parse_actor_action,
    parse_actor_side_effects,
    prompt_safe_tool_result,
    prompt_safe_observation,
    prompt_memory_tail,
    sanitize_memory,
    validate_harness_action,
)
from fakes import FakeHarnessTools


class _SequenceActor:
    def __init__(self, actions: list[HarnessAction]) -> None:
        self.actions = actions
        self.calls = 0

    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del prompt, role, image_paths
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return json.dumps({"thought": action.thought, "action": {"tool": action.tool, "args": action.args}})


class _AlwaysApproveCritic:
    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del prompt, role, image_paths
        return json.dumps({"verdict": "approve", "reason": "model approved repeated turn", "replacement_action": None})


class _MotionAwareActor:
    def __init__(self) -> None:
        self.calls = 0
        self.image_paths_by_call: list[list] = []

    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del prompt, role
        self.image_paths_by_call.append(list(image_paths or []))
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "thought": "drive first",
                    "action": {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}},
                    "memory_update": {"observation_note": "start"},
                }
            )
        return json.dumps(
            {
                "thought": "save prior motion",
                "action": {"tool": "turn_by_angle", "args": {"degrees": 18, "power_percent": 10}},
                "memory_update": {"observation_note": "previous strip showed forward progress"},
                "save_frames": [
                    {
                        "id": "prior_motion_mid",
                        "source": "previous_motion",
                        "frame_index": 2,
                        "note": "middle frame from previous motion strip",
                    }
                ],
            }
        )


class _ServoAuditActor:
    def __init__(self) -> None:
        self.calls = 0
        self.image_paths_by_call: list[list] = []

    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del prompt, role
        self.image_paths_by_call.append(list(image_paths or []))
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "thought": "servo first",
                    "action": {
                        "tool": "visual_servo_object",
                        "args": {"prompt": "sofa", "duration_s": 1.0, "forward_power": 18.0},
                    },
                }
            )
        return json.dumps({"thought": "audit previous overlay", "action": {"tool": "wait", "args": {"duration_s": 0.2}}})


class _GroundingCheckActor:
    def __init__(self) -> None:
        self.calls = 0
        self.image_paths_by_call: list[list] = []

    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del prompt, role
        self.image_paths_by_call.append(list(image_paths or []))
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "thought": "verify phrase on the latest image before servo",
                    "action": {"tool": "check_object_grounding", "args": {"prompt": "white toilet", "detector": "grounding-dino"}},
                }
            )
        return json.dumps({"thought": "inspect grounding overlay", "action": {"tool": "wait", "args": {"duration_s": 0.2}}})


class _MalformedActor:
    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del prompt, role, image_paths
        return '{"thought":"missing comma" "action":{"tool":"wait","args":{"duration_s":0.2}}}'


class _MalformedThenValidActor:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, prompt: str, *, role: str, image_paths=None) -> str:  # noqa: ANN001
        del role, image_paths
        self.calls += 1
        if self.calls == 1:
            return '{"thought":"truncated","action":{"tool":"wait","args":{"duration_s":0.2}'
        assert "ACTOR_JSON_REPAIR" in prompt
        return json.dumps({"thought": "repaired", "action": {"tool": "wait", "args": {"duration_s": 0.2}}})


class _RecordingRerun:
    save_path = None

    def __init__(self) -> None:
        self.llm_calls: list[tuple[int, str, dict]] = []

    def log_event(self, step: int, event: str, payload: dict) -> None:
        del step, event, payload

    def log_state(self, step: int, mode: str) -> None:
        del step, mode

    def log_observation(self, step: int, observation: dict) -> None:
        del step, observation

    def log_command(self, step: int, source: str, action: dict) -> None:
        del step, source, action

    def log_metadata(self, metadata: dict) -> None:
        del metadata

    def log_llm(self, step: int, role: str, payload: dict) -> None:
        self.llm_calls.append((step, role, payload))


def test_actor_prompt_keeps_static_context_before_dynamic_state(tmp_path) -> None:
    prompt = build_actor_prompt(
        goal="Drive to the sofa.",
        mode="auto",
        step=2,
        memory_path=tmp_path / "memory.jsonl",
        observation={
            "path": "frame.jpg",
            "yaw_deg": 12.0,
            "frame_seq": 4,
            "detections": [],
            "brightness_center": 0.4,
            "hidden_score": {"distance_m": 0.1},
        },
        recent_memory=[
            {
                "step": 1,
                "executed_action": {"tool": "drive_straight"},
                "hidden_score_for_evaluator_only": {"distance_m": 0.2},
            }
        ],
    )

    assert prompt.index("STATIC_HARNESS_CONTEXT") < prompt.index("DYNAMIC_TASK_STATE")
    assert "tool_contract" in prompt
    assert str(tmp_path / "memory.jsonl") in prompt
    assert "hidden_score" not in prompt
    assert "distance_m" not in prompt
    assert "target_pose" not in prompt


def test_actor_prompt_declares_camera_image_authoritative_and_strips_legacy_detections(tmp_path) -> None:
    prompt = build_actor_prompt(
        goal="Drive to the sofa.",
        mode="auto",
        step=0,
        memory_path=tmp_path / "memory.jsonl",
        observation={
            "path": "frames/0001.jpg",
            "yaw_deg": 0.0,
            "frame_seq": 1,
            "detections": [{"name": "toilet", "confidence": 0.31}],
            "brightness_center": 0.45,
        },
        recent_memory=[],
    )

    assert "authoritative latest RGB camera frame" in prompt
    assert "previous motion strip" in prompt
    assert "low-level camera summary that may be wrong or incomplete" in prompt
    assert "image, IMU yaw, and recent motion history" in prompt
    assert "Do not stop unless repeated observations show" in prompt
    assert '"observe"' not in prompt
    assert "do not request another observation" in prompt
    assert "cannot discover a hidden goal object by itself" in prompt
    assert "visible waypoint" in prompt
    assert "not proof of semantic identity" in prompt
    assert "visual_servo_object reports moved=false or failure_reason" in prompt
    assert "fill grounding_audit before choosing an action" in prompt
    assert "Only repeat the same visual_servo_object prompt" in prompt
    assert "grounding_stability is sparse_detection_coverage" in prompt
    assert "check_object_grounding is non-motion" in prompt
    assert "ready_for_visual_servo=false" in prompt
    assert "ready_for_visual_servo=true as detector-box existence only" in prompt
    assert "grounding_geometry_warning" in prompt
    assert "previous_check_box_matches_intended_object" in prompt
    assert "detections" not in prompt
    assert "toilet" not in prompt


def test_critic_prompt_declares_camera_image_authoritative_and_stop_requires_repeated_evidence() -> None:
    prompt = build_critic_prompt(
        goal="Drive to the sofa.",
        step=3,
        observation={"path": "frames/0004.jpg", "yaw_deg": 10.0, "frame_seq": 4, "detections": [], "brightness_center": 0.5},
        action=HarnessAction("stop", {}),
        actor_output='{"action":{"tool":"stop","args":{}},"thought":"done"}',
        recent_memory=[],
    )

    assert "authoritative latest RGB camera frame" in prompt
    assert "low-level camera summary that may be wrong or incomplete" in prompt
    assert "final-goal phrase" in prompt
    assert "non-goal visible landmark" in prompt
    assert "did not audit the previous detector box" in prompt
    assert "grounding_stability is not status_track_present" in prompt
    assert "Approve check_object_grounding" in prompt
    assert "ready_for_visual_servo=true as semantic proof" in prompt
    assert "Reject stop unless repeated observations show" in prompt


def test_actor_prompt_accepts_prompt_profile_rule_overlay(tmp_path) -> None:
    prompt = build_actor_prompt(
        goal="Drive to the sofa.",
        mode="auto",
        step=0,
        memory_path=tmp_path / "memory.jsonl",
        observation={"path": "frames/0001.jpg", "yaw_deg": 0.0, "frame_seq": 1, "brightness_center": 0.5},
        recent_memory=[],
        prompt_profile="explore-memory-v1",
        extra_rules=("Write useful scratchpad state.",),
    )

    assert '"prompt_profile": "explore-memory-v1"' in prompt
    assert "Write useful scratchpad state." in prompt


def test_sanitize_memory_removes_nested_privileged_fields() -> None:
    cleaned = sanitize_memory(
        [
            {
                "step": 1,
                "pose": {"x": 1.0},
                "thought": "stale private rationale",
                "nested": {"nearest_target": "sofa", "reason": "visible target"},
                "observation": {
                    "detections": [
                        {"name": "toilet", "confidence": 0.31},
                        {"name": "chair", "confidence": 0.7},
                    ]
                },
            }
        ]
    )

    assert cleaned == [
        {
            "step": 1,
            "nested": {"reason": "visible target"},
            "observation": {},
        }
    ]


def test_prompt_memory_tail_compacts_verbose_records() -> None:
    records = [
        {
            "step": step,
            "goal": "Drive to the sofa.",
            "observation": {
                "path": f"frames/{step:04d}.jpg",
                "yaw_deg": step,
                "frame_seq": step,
                "hidden_score": {"distance_m": 0.1},
            },
            "actor_action": {
                "tool": "drive_straight",
                "args": {"power_percent": 20, "duration_s": 0.5},
                "thought": "long repeated private rationale " * 50,
            },
            "executed_action": {
                "tool": "drive_straight",
                "args": {"power_percent": 20, "duration_s": 0.5},
                "thought": "long repeated private rationale " * 50,
            },
            "actor_memory_update": {"observation_note": "bed visible " * 80, "pose": {"x": 1}},
            "tool_result": {
                "action": "drive_straight",
                "motion_contact_sheet": f"motion_frames/{step:04d}_strip.jpg",
                "debug_overlay_contact_sheet": f"motion_frames/{step:04d}_debug_overlay_strip.jpg",
                "grounding_audit_contact_sheet": f"motion_frames/{step:04d}_grounding_audit.jpg",
                "motion_frame_paths": [f"motion_frames/{step:04d}_{index}.jpg" for index in range(5)],
                "debug_overlay_frame_paths": [f"motion_frames/{step:04d}_debug_{index}.jpg" for index in range(5)],
                "stderr_tail": "stack trace " * 200,
                "final_yaw_deg": 12.0,
            },
        }
        for step in range(12)
    ]

    compact = prompt_memory_tail(records)

    assert len(compact) == 8
    assert compact[0]["step"] == 4
    assert "goal" not in compact[0]
    assert "hidden_score" not in json.dumps(compact)
    assert "thought" not in json.dumps(compact)
    assert "stderr_tail" not in json.dumps(compact)
    assert "motion_frame_paths" not in compact[-1]["tool_result"]
    assert "debug_overlay_frame_paths" not in compact[-1]["tool_result"]
    assert compact[-1]["tool_result"]["motion_contact_sheet"].endswith("_strip.jpg")
    assert compact[-1]["tool_result"]["debug_overlay_contact_sheet"].endswith("_debug_overlay_strip.jpg")
    assert compact[-1]["tool_result"]["grounding_audit_contact_sheet"].endswith("_grounding_audit.jpg")
    assert len(compact[-1]["actor_memory_update"]["observation_note"]) <= 220


def test_actor_action_parser_accepts_tool_schema_and_clamps() -> None:
    action = parse_actor_action(
        json.dumps(
            {
                "thought": "drive",
                "action": {"tool": "drive_straight", "args": {"power_percent": 99, "duration_s": 12}},
            }
        )
    )

    assert action.tool == "drive_straight"
    assert action.args == {"power_percent": 24.0, "duration_s": 0.9}


def test_actor_action_parser_accepts_flat_action_schema() -> None:
    action = parse_actor_action(
        json.dumps(
            {
                "thought": "servo",
                "action": "visual_servo_object",
                "args": {"prompt": "sofa"},
            }
        )
    )

    assert action.tool == "visual_servo_object"
    assert action.args["prompt"] == "sofa"


def test_actor_action_parser_accepts_visual_servo_tool() -> None:
    action = parse_actor_action(
        json.dumps(
            {
                "thought": "servo",
                "action": {
                    "tool": "visual_servo_object",
                    "args": {"prompt": "red mug", "duration_s": 9, "forward_power": 99, "detector": "florence-mlx"},
                },
            }
        )
    )

    assert action.tool == "visual_servo_object"
    assert action.args == {
        "prompt": "red mug",
        "duration_s": 4.0,
        "forward_power": 24.0,
        "detector": "florence-mlx",
    }


def test_actor_action_parser_accepts_topomap_memory_tool() -> None:
    action = parse_actor_action(
        json.dumps(
            {
                "thought": "ask image memory before moving",
                "action": {"tool": "query_topomap_memory", "args": {"goal_query": "sofa"}},
            }
        )
    )

    assert action.tool == "query_topomap_memory"
    assert action.args == {"goal_query": "sofa"}


def test_actor_action_parser_accepts_grounding_check_tool() -> None:
    action = parse_actor_action(
        json.dumps(
            {
                "thought": "verify visible candidate before moving",
                "action": {"tool": "check_object_grounding", "args": {"prompt": "white toilet", "detector": "grounding-dino"}},
            }
        )
    )

    assert action.tool == "check_object_grounding"
    assert action.args == {"prompt": "white toilet", "detector": "grounding-dino"}


def test_actor_side_effect_parser_sanitizes_memory_and_frame_requests() -> None:
    side_effects = parse_actor_side_effects(
        json.dumps(
            {
                "action": {"tool": "wait", "args": {"duration_s": 0.2}},
                "grounding_audit": {
                    "previous_visual_servo_box_matches_intended_object": False,
                    "next_prompt_should_change": True,
                    "evidence": "box was on cabinet",
                },
                "memory_update": {"belief": "sofa left", "pose": {"x": 1}},
                "save_frames": [{"id": "sofa left", "source": "previous_motion", "frame_index": 3, "note": "best view"}],
            }
        )
    )

    assert side_effects["memory_update"] == {"belief": "sofa left"}
    assert side_effects["grounding_audit"]["next_prompt_should_change"] is True
    assert side_effects["save_frames"][0]["source"] == "previous_motion"


def test_actor_consistency_guard_rejects_unchanged_prompt_after_failed_audit() -> None:
    decision = apply_actor_consistency_guard(
        CriticDecision("approve", "ok"),
        HarnessAction("visual_servo_object", {"prompt": "the white toilet"}, "repeat"),
        {
            "grounding_audit": {
                "previous_visual_servo_box_matches_intended_object": False,
                "next_prompt_should_change": True,
                "evidence": "box was on cabinet",
            }
        },
        [{"executed_action": {"tool": "visual_servo_object", "args": {"prompt": "the white toilet"}}}],
    )

    assert decision.verdict == "reject"
    assert decision.replacement is not None
    assert decision.replacement.tool == "wait"
    assert "requested a changed visual-servo prompt" in decision.reason


def test_invalid_harness_action_falls_back_to_wait() -> None:
    action = validate_harness_action(HarnessAction("read_hidden_pose", {}, "bad"))

    assert action.tool == "wait"
    assert action.args["duration_s"] == 0.2


def test_actor_requested_observe_falls_back_to_wait() -> None:
    action = parse_actor_action(json.dumps({"action": "observe", "thought": "look again"}))

    assert action.tool == "wait"
    assert action.args["duration_s"] == 0.2


def test_scripted_smoke_runner_does_not_branch_on_static_object_names() -> None:
    runner = ScriptedOpenVocabRunner()
    sofa_prompt = "DYNAMIC_TASK_STATE\n" + json.dumps({"goal": "Drive to the sofa.", "step": 0})
    toilet_prompt = "DYNAMIC_TASK_STATE\n" + json.dumps({"goal": "Drive to the toilet.", "step": 0})

    sofa_action = json.loads(runner.run(sofa_prompt, role="actor"))["action"]
    runner.reset()
    toilet_action = json.loads(runner.run(toilet_prompt, role="actor"))["action"]

    assert sofa_action == toilet_action


def test_session_can_execute_topomap_memory_lookup_without_motion(tmp_path) -> None:
    actor = _SequenceActor([HarnessAction("query_topomap_memory", {"goal_query": "sofa"}, "ask memory")])
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=actor,
        critic=_AlwaysApproveCritic(),
    )

    session.start_goal("Drive to the sofa.")
    record = session.run_auto_step()
    session.close()

    assert record is not None
    assert record["executed_action"]["tool"] == "query_topomap_memory"
    assert record["tool_result"]["action"] == "query_topomap_memory"
    assert record["tool_result"]["reason"] == "topomap_memory_not_configured"


def test_session_can_execute_grounding_check_without_motion_and_attach_overlay(tmp_path) -> None:
    actor = _GroundingCheckActor()
    tools = FakeHarnessTools(run_dir=tmp_path, environment="bathroom")
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=2),
        tools=tools,
        actor=actor,
        critic=_AlwaysApproveCritic(),
    )

    session.start_goal("Drive to the toilet.")
    first = session.run_auto_step()
    second = session.run_auto_step()
    session.close()

    assert first is not None and second is not None
    assert first["executed_action"]["tool"] == "check_object_grounding"
    assert first["tool_result"]["action"] == "check_object_grounding"
    assert first["tool_result"]["ready_for_visual_servo"] is True
    assert first["tool_result"]["selected_label"] == "white toilet"
    assert first["tool_result"]["selected_bbox_touches_image_edge"] is False
    assert first["tool_result"]["grounding_geometry_warning"] is None
    assert tools.motion_seq == 0
    assert len(actor.image_paths_by_call[0]) == 1
    assert len(actor.image_paths_by_call[1]) == 2
    assert actor.image_paths_by_call[1][1].name.endswith("_grounding_check_overlay.jpg")


def test_codex_exec_runner_uses_supported_low_reasoning_config_and_images(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    runner = CodexExecRunner(model="gpt-5.5", reasoning_effort="low", cwd=tmp_path)

    command = runner.command(image_paths=[image])

    assert "--reasoning-effort" not in command
    assert command[:5] == ["codex", "exec", "-m", "gpt-5.5", "-c"]
    assert 'model_reasoning_effort="low"' in command
    assert "--ignore-user-config" in command
    assert ["--sandbox", "read-only"] == command[command.index("--sandbox") : command.index("--sandbox") + 2]
    assert ["--cd", str(tmp_path)] == command[command.index("--cd") : command.index("--cd") + 2]
    schema_path = command[command.index("--output-schema") + 1]
    assert schema_path.endswith("actor_output_schema.json")
    actor_schema = (tmp_path / "codex_schemas" / "actor_output_schema.json").read_text(encoding="utf-8")
    actor_schema_json = json.loads(actor_schema)
    assert '"required": [' in actor_schema
    assert "grounding_audit" in actor_schema_json["required"]
    assert "check_object_grounding" in actor_schema_json["properties"]["action"]["properties"]["tool"]["enum"]
    arg_properties = actor_schema_json["properties"]["action"]["properties"]["args"]["properties"]
    assert {"degrees", "power_percent", "duration_s", "prompt", "goal_query", "detector"} <= set(arg_properties)
    assert ["--image", str(image)] == command[command.index("--image") : command.index("--image") + 2]
    assert command[-1] == "-"

    critic_command = runner.command(role="critic")
    critic_schema_path = critic_command[critic_command.index("--output-schema") + 1]
    critic_schema = (tmp_path / "codex_schemas" / "critic_output_schema.json").read_text(encoding="utf-8")
    assert critic_schema_path.endswith("critic_output_schema.json")
    assert '"replacement_action"' in critic_schema


def test_safety_critic_rejects_early_stop_without_visual_target() -> None:
    prompt = build_critic_prompt(
        goal="Drive to the sofa.",
        step=1,
        observation={"path": "frame.jpg", "yaw_deg": 0.0, "frame_seq": 1, "detections": [], "brightness_center": 0.5},
        action=HarnessAction("stop", {}),
        actor_output='{"action":{"tool":"stop","args":{}},"thought":"done"}',
        recent_memory=[],
    )

    decision = parse_critic_decision(SafetyCriticRunner().run(prompt, role="critic"))

    assert decision.verdict == "reject"
    assert decision.replacement is not None
    assert decision.replacement.tool == "drive_straight"


def test_prompt_safe_observation_uses_relative_path_inside_run_dir(tmp_path) -> None:
    frame = tmp_path / "frames" / "0001.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"fake")

    safe = prompt_safe_observation({"path": str(frame), "yaw_deg": 0.0, "frame_seq": 1}, root=tmp_path)

    assert safe["path"] == "frames/0001.jpg"
    assert str(tmp_path) not in safe["path"]


def test_prompt_safe_tool_result_uses_relative_motion_paths(tmp_path) -> None:
    frame = tmp_path / "motion_frames" / "0001.jpg"
    strip = tmp_path / "motion_frames" / "strip.jpg"
    debug_strip = tmp_path / "motion_frames" / "debug_strip.jpg"
    topomap_manifest = tmp_path / "topomap_memory" / "topomap_memory_manifest.json"
    topomap_query_log = tmp_path / "topomap_memory" / "query_log.jsonl"
    frame.parent.mkdir()
    frame.write_bytes(b"fake")
    strip.write_bytes(b"fake")
    debug_strip.write_bytes(b"fake")
    topomap_manifest.parent.mkdir()
    topomap_manifest.write_text("{}", encoding="utf-8")
    topomap_query_log.write_text("{}\n", encoding="utf-8")

    safe = prompt_safe_tool_result(
        {
            "motion_frame_paths": [str(frame)],
            "motion_contact_sheet": str(strip),
            "debug_overlay_contact_sheet": str(debug_strip),
            "topomap_memory_manifest_path": str(topomap_manifest),
            "topomap_query_log_path": str(topomap_query_log),
            "elapsed_s": 0.5,
        },
        root=tmp_path,
    )

    assert safe["motion_frame_paths"] == ["motion_frames/0001.jpg"]
    assert safe["motion_contact_sheet"] == "motion_frames/strip.jpg"
    assert safe["debug_overlay_contact_sheet"] == "motion_frames/debug_strip.jpg"
    assert safe["topomap_memory_manifest_path"] == "topomap_memory/topomap_memory_manifest.json"
    assert safe["topomap_query_log_path"] == "topomap_memory/query_log.jsonl"


def test_harness_memory_keeps_model_facing_paths_relative(tmp_path) -> None:
    runner = DeterministicHarnessRunner()
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=runner,
        critic=SafetyCriticRunner(),
    )

    session.start_goal("Drive to the sofa.")
    record = session.run_auto_step()
    session.close()

    assert record is not None
    assert not record["observation"]["path"].startswith("/")
    assert not record["tool_result"]["motion_contact_sheet"].startswith("/")
    assert not any(path.startswith("/") for path in record["tool_result"]["motion_frame_paths"])
    actor_prompt = (tmp_path / "prompts" / "000_actor.txt").read_text(encoding="utf-8")
    assert str(tmp_path) not in actor_prompt
    assert '"memory_log_path": "memory.jsonl"' in actor_prompt


def test_session_attaches_previous_motion_strip_and_saves_requested_frame(tmp_path) -> None:
    actor = _MotionAwareActor()
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=2),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=actor,
        critic=_AlwaysApproveCritic(),
    )

    session.start_goal("Drive to the sofa.")
    first = session.run_auto_step()
    second = session.run_auto_step()
    session.close()

    assert first is not None and second is not None
    assert len(actor.image_paths_by_call[0]) == 1
    assert len(actor.image_paths_by_call[1]) == 2
    assert actor.image_paths_by_call[1][1].name.endswith("_strip.jpg")
    assert second["actor_memory_update"]["observation_note"] == "previous strip showed forward progress"
    assert second["saved_frames"]
    saved_path = tmp_path / second["saved_frames"][0]["path"]
    assert saved_path.exists()


def test_session_attaches_visual_servo_grounding_audit_after_latest_frame(tmp_path) -> None:
    actor = _ServoAuditActor()
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=2),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=actor,
        critic=_AlwaysApproveCritic(),
    )

    session.start_goal("Drive to the sofa.")
    first = session.run_auto_step()
    second = session.run_auto_step()
    session.close()

    assert first is not None and second is not None
    assert len(actor.image_paths_by_call[0]) == 1
    assert len(actor.image_paths_by_call[1]) == 2
    assert actor.image_paths_by_call[1][1].name.endswith("_grounding_audit.jpg")


def test_session_logs_llm_outputs_to_rerun_sink(tmp_path) -> None:
    rerun = _RecordingRerun()
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1, rerun_enabled=True),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=DeterministicHarnessRunner(),
        critic=SafetyCriticRunner(),
        rerun_logger=rerun,
    )

    session.start_goal("Drive to the sofa.")
    session.run_auto_step()
    session.close()

    roles = [role for _, role, _ in rerun.llm_calls]
    assert roles == ["actor", "critic"]
    actor_payload = rerun.llm_calls[0][2]
    critic_payload = rerun.llm_calls[1][2]
    assert actor_payload["prompt_path"] == "prompts/000_actor.txt"
    assert actor_payload["parsed_action"]["tool"] in {
        "turn_by_angle",
        "drive_straight",
        "visual_servo_object",
        "check_object_grounding",
        "query_topomap_memory",
        "stop",
        "wait",
    }
    assert not any(str(path).startswith("/") for path in actor_payload["image_paths"])
    assert "output" in actor_payload
    assert critic_payload["prompt_path"] == "prompts/000_critic.txt"
    assert critic_payload["final_decision"]["verdict"] in {"approve", "warn", "reject"}


def test_session_logs_raw_actor_output_on_parse_error(tmp_path) -> None:
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1, rerun_enabled=True),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=_MalformedActor(),
        critic=_AlwaysApproveCritic(),
        rerun_logger=_RecordingRerun(),
    )

    session.start_goal("Drive to the sofa.")
    with pytest.raises(json.JSONDecodeError):
        session.run_auto_step()
    events = session.read_events_tail(20)
    session.close()

    actor_errors = [event for event in events if event.get("event") == "actor_error"]
    assert actor_errors
    assert "missing comma" in actor_errors[-1]["output"]
    assert actor_errors[-1]["prompt_path"] == "prompts/000_actor.txt"
    assert actor_errors[-1]["attempt_count"] == 2


def test_session_retries_malformed_actor_json_once(tmp_path) -> None:
    actor = _MalformedThenValidActor()
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=1),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=actor,
        critic=_AlwaysApproveCritic(),
    )

    session.start_goal("Drive to the sofa.")
    record = session.run_auto_step()
    events = session.read_events_tail(20)
    session.close()

    assert record is not None
    assert actor.calls == 2
    assert record["executed_action"]["tool"] == "wait"
    assert (tmp_path / "prompts" / "000_actor_retry.txt").exists()
    retries = [event for event in events if event.get("event") == "actor_parse_retry"]
    assert retries
    assert "invalid JSON" not in retries[-1]["previous_output"]


def test_safety_critic_rejects_third_same_direction_turn() -> None:
    recent_memory = [
        {"executed_action": {"tool": "turn_by_angle", "args": {"degrees": 18.0}}, "observation": {}},
        {"executed_action": {"tool": "turn_by_angle", "args": {"degrees": 12.0}}, "observation": {}},
    ]
    prompt = build_critic_prompt(
        goal="Drive to the sofa.",
        step=2,
        observation={"path": "frame.jpg", "yaw_deg": 0.0, "frame_seq": 3, "detections": [], "brightness_center": 0.5},
        action=HarnessAction("turn_by_angle", {"degrees": 16.0}),
        actor_output='{"action":{"tool":"turn_by_angle","args":{"degrees":16}},"thought":"scan"}',
        recent_memory=recent_memory,
    )

    decision = parse_critic_decision(SafetyCriticRunner().run(prompt, role="critic"))

    assert decision.verdict == "reject"
    assert decision.replacement is not None
    assert decision.replacement.tool == "drive_straight"


def test_session_respects_model_critic_when_deterministic_safety_gate_is_disabled(tmp_path) -> None:
    actor = _SequenceActor(
        [
            HarnessAction("turn_by_angle", {"degrees": 18.0, "power_percent": 10.0}, "scan right"),
            HarnessAction("turn_by_angle", {"degrees": 18.0, "power_percent": 10.0}, "scan right again"),
            HarnessAction("turn_by_angle", {"degrees": 18.0, "power_percent": 10.0}, "scan right third time"),
        ]
    )
    session = HarnessSession(
        config=HarnessConfig(run_dir=tmp_path, max_steps=3),
        tools=FakeHarnessTools(run_dir=tmp_path, environment="living_room"),
        actor=actor,
        critic=_AlwaysApproveCritic(),
    )

    session.start_goal("Drive to the sofa.")
    session.run_auto_step()
    session.run_auto_step()
    record = session.run_auto_step()
    events = session.read_events_tail(30)
    session.close()

    assert record is not None
    assert record["actor_action"]["tool"] == "turn_by_angle"
    assert record["critic"]["verdict"] == "approve"
    assert record["executed_action"]["tool"] == "turn_by_angle"
    assert not any(event.get("event") == "safety_gate" for event in events)


def test_qwen_runner_formats_openai_compatible_image_payload(tmp_path, monkeypatch) -> None:
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image")
    captured = {}

    class _Response:
        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, *_args):  # noqa: ANN001
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    runner = OpenAICompatibleVisionRunner(model="Qwen/Qwen3-VL-8B-Instruct", endpoint="http://localhost", timeout_s=3)

    assert runner.run("prompt", role="actor", image_paths=[image]) == '{"ok": true}'
    content = captured["payload"]["messages"][0]["content"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert content[0] == {"type": "text", "text": "prompt"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_qwen_runner_retries_context_budget_errors_with_smaller_completion(monkeypatch) -> None:
    payloads = []

    class _Response:
        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, *_args):  # noqa: ANN001
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"tool\": \"wait\"}"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):  # noqa: ANN001
        del timeout
        payload = json.loads(request.data.decode("utf-8"))
        payloads.append(payload)
        if len(payloads) == 1:
            body = (
                '{"error":{"message":"You passed 7681 input tokens and requested 512 output tokens. '
                "However, the model's context length is only 8192 tokens, resulting in a maximum input length of 7680 tokens.\"}}"
            ).encode("utf-8")
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    runner = OpenAICompatibleVisionRunner(endpoint="http://localhost", max_tokens=512)

    assert runner.run("prompt", role="actor") == '{"tool": "wait"}'
    assert [payload["max_tokens"] for payload in payloads] == [512, 384]
