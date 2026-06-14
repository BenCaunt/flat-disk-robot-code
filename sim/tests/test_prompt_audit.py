from __future__ import annotations

import json

from flatdisk_sim.llm_harness import build_actor_prompt
from flatdisk_sim.prompt_audit import audit_prompts


def test_prompt_audit_ignores_dynamic_goal_but_checks_static_context(tmp_path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt = build_actor_prompt(
        goal="Drive to the sofa.",
        mode="auto",
        step=0,
        memory_path=tmp_path / "memory.jsonl",
        observation={"path": "frame.jpg", "yaw_deg": 0.0, "frame_seq": 1},
        recent_memory=[],
    )
    (prompt_dir / "000_actor.txt").write_text(prompt, encoding="utf-8")

    audit = audit_prompts(prompt_dir)

    assert audit["checked_prompt_count"] == 1
    assert audit["static_context_checked_count"] == 1
    assert audit["static_context_forbidden_terms_found"] == []
    assert audit["no_hardcoded_labels_or_colors"] is True
    assert audit["prompt_audit_passed"] is True


def test_prompt_audit_flags_static_object_and_color_examples(tmp_path) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    static_context = {
        "rules": ["Use a visible waypoint such as a red mug."],
        "tool_contract": {"visual_servo_object": {"args": {"prompt": "try an armchair"}}},
    }
    prompt = (
        "STATIC_HARNESS_CONTEXT\n"
        + json.dumps(static_context)
        + "\n\nDYNAMIC_TASK_STATE\n"
        + json.dumps({"goal": "Drive to the sofa."})
        + "\n"
    )
    (prompt_dir / "000_actor.txt").write_text(prompt, encoding="utf-8")

    audit = audit_prompts(prompt_dir)

    assert "000_actor.txt:armchair" in audit["static_context_forbidden_terms_found"]
    assert "000_actor.txt:mug" in audit["static_context_forbidden_terms_found"]
    assert "000_actor.txt:red" in audit["static_context_forbidden_terms_found"]
    assert "000_actor.txt:sofa" not in audit["static_context_forbidden_terms_found"]
    assert audit["no_hardcoded_labels_or_colors"] is False
    assert audit["prompt_audit_passed"] is False
