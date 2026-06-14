from __future__ import annotations

import json

from flatdisk_sim.llm_harness import build_actor_prompt
from flatdisk_sim.prompt_audit import audit_prompts
from flatdisk_sim.strategy_sweep import generate_strategy_config
from flatdisk_sim.research_loop import load_config


def test_generate_strategy_config_produces_qwen_variants_without_semantic_terms(tmp_path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "experiment_id": "base_exp",
                "objective": "base",
                "episodes": ["living_room_sofa"],
                "variants": [
                    {
                        "name": "qwen_base",
                        "runner": "qwen",
                        "qwen_endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                        "qwen_model": "Qwen/Qwen3-VL-8B-Instruct",
                        "object_drive_detector": "florence-transformers",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    base = load_config(base_config)

    generated = generate_strategy_config(base, experiment_id="generated_exp")

    output_config = tmp_path / "generated.json"
    output_config.write_text(json.dumps(generated), encoding="utf-8")
    parsed = load_config(output_config)
    assert len(parsed.variants) == 7
    assert {variant.runner for variant in parsed.variants} == {"qwen"}
    assert all(variant.qwen_endpoint == "http://127.0.0.1:8000/v1/chat/completions" for variant in parsed.variants)
    assert all(variant.qwen_model == "Qwen/Qwen3-VL-8B-Instruct" for variant in parsed.variants)
    assert all(variant.critic_mode == "none" for variant in parsed.variants)
    assert all(not variant.topomap_memory_allow_semantic_terms for variant in parsed.variants)
    topomap = next(variant for variant in parsed.variants if variant.name == "qwen_topomap_memory")
    assert topomap.topomap_memory_use_clip is True
    assert topomap.topomap_memory_map_dir == "sim/scratch/semantic_topomaps/{episode}_clip"
    grounding = next(variant for variant in parsed.variants if variant.name == "qwen_grounding_recovery")
    assert grounding.prompt_profile == "grounding-recovery-v1"
    assert any("failed_servo_prompts" in rule for rule in grounding.actor_rules)
    dino = next(variant for variant in parsed.variants if variant.name == "qwen_grounding_dino_recovery")
    assert dino.prompt_profile == "grounding-dino-recovery-v1"
    assert dino.object_drive_detector == "grounding-dino"
    assert any("check_object_grounding" in rule for rule in dino.actor_rules)


def test_generated_strategy_prompts_pass_static_generality_audit(tmp_path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "experiment_id": "base_exp",
                "objective": "base",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_base", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    generated = generate_strategy_config(load_config(base_config), experiment_id="generated_exp")
    generated_config = tmp_path / "generated.json"
    generated_config.write_text(json.dumps(generated), encoding="utf-8")
    parsed = load_config(generated_config)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()

    for index, variant in enumerate(parsed.variants):
        prompt = build_actor_prompt(
            goal="Drive to the sofa.",
            mode="auto",
            step=0,
            memory_path=tmp_path / "memory.jsonl",
            observation={"path": "frame.jpg", "yaw_deg": 0.0, "frame_seq": 1},
            recent_memory=[],
            prompt_profile=variant.prompt_profile,
            extra_rules=variant.actor_rules,
        )
        (prompt_dir / f"{index:03d}_{variant.name}.txt").write_text(prompt, encoding="utf-8")

    audit = audit_prompts(prompt_dir)
    assert audit["checked_prompt_count"] == len(parsed.variants)
    assert audit["static_context_checked_count"] == len(parsed.variants)
    assert audit["static_context_forbidden_terms_found"] == []
    assert audit["prompt_audit_passed"] is True


def test_strategy_sweep_cli_writes_loadable_config(monkeypatch, tmp_path, capsys) -> None:
    base_config = tmp_path / "base.json"
    output_config = tmp_path / "strategy.json"
    base_config.write_text(
        json.dumps(
            {
                "experiment_id": "base_exp",
                "objective": "base",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_base", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-generate-nav-strategy-sweep",
            "--base-config",
            str(base_config),
            "--output",
            str(output_config),
            "--experiment-id",
            "generated_exp",
            "--exclude-topomap",
        ],
    )

    from flatdisk_sim import strategy_sweep

    assert strategy_sweep.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["variant_count"] == 6
    parsed = load_config(output_config)
    assert len(parsed.variants) == 6
    assert {variant.name for variant in parsed.variants} == {
        "qwen_baseline",
        "qwen_frontier_scan",
        "qwen_evidence_exploit",
        "qwen_recovery_switch",
        "qwen_grounding_recovery",
        "qwen_grounding_dino_recovery",
    }
