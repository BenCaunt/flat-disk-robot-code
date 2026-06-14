from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

from flatdisk_sim.evaluate_text_goals import ThorEpisodeSpec
from fakes import FakeHarnessTools


def test_auto_critic_mode_keeps_qwen_policy_autonomous() -> None:
    from flatdisk_sim import evaluate_harness_thor as module

    actor = object()
    qwen_critic = module._build_critic("qwen", actor, mode=module._resolve_critic_mode("qwen", "auto"))
    codex_critic = module._build_critic("codex", actor, mode=module._resolve_critic_mode("codex", "auto"))
    scripted_critic = module._build_critic(
        "scripted-open-vocab",
        actor,
        mode=module._resolve_critic_mode("scripted-open-vocab", "auto"),
    )

    assert qwen_critic.__class__.__name__ == "NoopCriticRunner"
    assert json.loads(qwen_critic.run("{}", role="critic"))["verdict"] == "approve"
    assert codex_critic is actor
    assert scripted_critic.__class__.__name__ == "SafetyCriticRunner"


def test_harness_thor_cli_defaults_to_model_based_qwen(monkeypatch) -> None:
    from flatdisk_sim import evaluate_harness_thor as module

    monkeypatch.setattr(sys, "argv", ["flatdisk-sim-evaluate-harness-thor"])

    args = module.parse_args()

    assert args.runner == "qwen"


def test_evaluator_distance_metrics_keep_best_approach_and_regression() -> None:
    from flatdisk_sim import evaluate_harness_thor as module

    metrics = module._evaluator_distance_metrics(
        initial_score={"success": False, "distance_m": 1.0},
        steps=[
            {"step": 0, "hidden_score_for_evaluator_only": {"success": False, "distance_m": 0.7}},
            {"step": 1, "hidden_score_for_evaluator_only": {"success": False, "distance_m": 0.25}},
            {"step": 2, "hidden_score_for_evaluator_only": {"success": False, "distance_m": 0.45}},
        ],
        final_score={"success": False, "distance_m": 0.5},
        success_radius_m=0.1,
    )

    assert metrics["initial_distance_m"] == 1.0
    assert metrics["best_distance_m"] == 0.25
    assert metrics["best_distance_step"] == 1
    assert metrics["best_distance_improvement_m"] == 0.75
    assert metrics["final_distance_improvement_m"] == 0.5
    assert metrics["final_to_best_regression_m"] == 0.25
    assert metrics["reached_success_radius_ever"] is False


class FakeProcess:
    def poll(self):
        return 0


def test_thor_harness_eval_keeps_hidden_logs_outside_policy_dir(tmp_path, monkeypatch) -> None:
    from flatdisk_sim import evaluate_harness_thor as module

    score_calls = {"count": 0}

    def fake_score(_hidden, _spec):
        score_calls["count"] += 1
        return {
            "success": score_calls["count"] >= 2,
            "distance_m": 0.1 if score_calls["count"] >= 2 else 1.0,
            "nearest_target": {"objectType": "Sofa"},
        }

    def fake_render(_hidden_poses, _spec, output_path: Path, *, title: str) -> Path:
        del title
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), "white").save(output_path)
        return output_path

    monkeypatch.setattr(module, "_start_bridge", lambda **_kwargs: FakeProcess())
    monkeypatch.setattr(module, "_terminate_bridge", lambda _proc: None)
    monkeypatch.setattr(module, "_wait_for_hidden_log", lambda *_args, **_kwargs: tmp_path / "hidden.jsonl")
    monkeypatch.setattr(module, "_wait_for_first_observation", lambda _tools: None)
    monkeypatch.setattr(
        module,
        "_latest_hidden_pose",
        lambda _log_dir: {
            "x": 0.0,
            "z": 0.0,
            "yaw_deg": 0.0,
            "objects": [{"objectType": "Sofa", "position": {"x": 0.1, "z": 0.1}}],
        },
    )
    monkeypatch.setattr(module, "_score_hidden_distance", fake_score)
    monkeypatch.setattr(module, "_render_hidden_progress", fake_render)
    monkeypatch.setattr(
        module,
        "AgentTools",
        lambda *, run_dir, namespace, connect, **_kwargs: FakeHarnessTools(run_dir=run_dir, environment="living_room"),
    )

    summary = module.run_thor_harness_episode(
        ThorEpisodeSpec(
            name="fake_living_room",
            scene="FloorPlan201",
            prompt="Drive to the sofa.",
            target_types=("Sofa",),
            success_radius_m=0.55,
            max_steps=4,
        ),
        output_root=tmp_path,
        port=12345,
        model="gpt-5.5",
        reasoning_effort="low",
        live_codex=False,
        runner="scripted-open-vocab",
        render_width=160,
        render_height=120,
        rerun=False,
    )

    policy_dir = Path(summary["policy_dir"])
    hidden_dir = Path(summary["evaluator_only_dir"])
    assert policy_dir.name == "policy"
    assert hidden_dir.name == "evaluator_hidden"
    assert hidden_dir.parent == policy_dir.parent
    assert not (policy_dir / "evaluator_hidden").exists()
    assert summary["model"] == "gpt-5.5"
    assert summary["actor_model"] == "gpt-5.5"
    assert summary["qwen_model"] is None
    assert summary["success"] is True
    assert summary["prompt_audit"]["forbidden_tokens_found"] == []
    assert summary["camera_contact_sheet"]
    assert summary["progress_contact_sheet"]
