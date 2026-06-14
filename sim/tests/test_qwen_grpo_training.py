from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys
from time import gmtime, strftime

from flatdisk_sim import qwen_grpo_job
from flatdisk_sim.qwen_grpo_training import main, prepare_qwen_grpo_training
from flatdisk_sim.training_export import export_training_data_from_summaries
from flatdisk_sim.training_readiness import analyze_training_readiness


def _summary_fixture(
    tmp_path: Path,
    *,
    trial_id: str,
    final_distance_m: float,
    success: bool = False,
    actor_action: dict | None = None,
    executed_action: dict | None = None,
) -> dict:
    run_dir = tmp_path / trial_id
    policy_dir = run_dir / "policy"
    prompts_dir = policy_dir / "prompts"
    frames_dir = policy_dir / "frames"
    prompts_dir.mkdir(parents=True)
    frames_dir.mkdir(parents=True)
    (frames_dir / "0001.jpg").write_bytes(b"fake image")
    (prompts_dir / "000_actor.txt").write_text("STATIC_HARNESS_CONTEXT\nDYNAMIC_TASK_STATE\n", encoding="utf-8")
    actor_action = actor_action or {
        "tool": "drive_straight",
        "args": {"power_percent": 20, "duration_s": 0.5},
        "thought": "move",
    }
    executed_action = executed_action or actor_action
    return {
        "trial_id": trial_id,
        "slot_id": "slot_001",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "episode": "living_room_sofa",
        "scene": "FloorPlan201",
        "prompt": "Drive to the sofa.",
        "runner": "qwen",
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "success": success,
        "reason": "hidden_evaluator_goal_reached" if success else "max_steps_exhausted",
        "final_distance_m": final_distance_m,
        "best_distance_m": final_distance_m,
        "best_distance_step": 0,
        "success_radius_m": 0.1,
        "step_count": 1,
        "steps": [
            {
                "step": 0,
                "harness_memory_record": {
                    "step": 0,
                    "goal": "Drive to the sofa.",
                    "observation": {
                        "path": "frames/0001.jpg",
                        "yaw_deg": 0.0,
                        "frame_seq": 1,
                        "brightness_center": 0.5,
                    },
                    "actor_action": actor_action,
                    "actor_memory_update": {"belief": "open space ahead"},
                    "critic": {"verdict": "approve", "reason": "bounded"},
                    "executed_action": executed_action,
                    "tool_result": {"ok": True},
                    "saved_frames": [],
                },
                "hidden_score_for_evaluator_only": {
                    "success": success,
                    "distance_m": final_distance_m,
                    "nearest_target": {"objectType": "Sofa", "objectId": "hidden"},
                },
            }
        ],
    }


def test_prepare_qwen_grpo_training_writes_trainable_rollout_groups(tmp_path: Path) -> None:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )

    assert manifest["schema"] == "flatdisk.qwen_grpo_training_manifest.v1"
    assert manifest["status"] == "ready"
    assert manifest["group_count"] == 1
    assert manifest["trainable_group_count"] == 1
    assert manifest["candidate_count"] == 2
    assert manifest["trainable_candidate_count"] == 2
    assert manifest["step_sample_count"] == 2
    assert manifest["ppo_step_sample_count"] == 2
    assert manifest["missing_image_count"] == 0
    assert manifest["forbidden_qwen_message_token_hits"] == {}
    groups_path = tmp_path / "run" / "qwen_grpo_training" / "qwen_grpo_rollout_groups.jsonl"
    group = json.loads(groups_path.read_text(encoding="utf-8").splitlines()[0])
    assert group["schema"] == "flatdisk.qwen_grpo_rollout_group.v1"
    assert group["trainable_candidate_count"] == 2
    assert group["candidates"][0]["trainable"] is True
    assert group["candidates"][0]["evaluator_reward"]["privileged"] is True
    assert group["candidates"][0]["step_samples"][0]["prompt_messages"][0]["role"] == "user"
    assert group["candidates"][0]["step_samples"][0]["assistant_target_json"]["action"]["tool"] == "drive_straight"
    prompt_text = json.dumps(group["candidates"][0]["step_samples"][0]["prompt_messages"]).lower()
    assert "distance_m" not in prompt_text
    assert "nearest_target" not in prompt_text
    ppo_step = json.loads(
        (tmp_path / "run" / "qwen_grpo_training" / "qwen_ppo_step_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert ppo_step["schema"] == "flatdisk.qwen_ppo_step_sample.v1"
    assert ppo_step["evaluator_reward"]["privileged"] is True
    assert ppo_step["audit"]["reward_excluded_from_messages"] is True

    readiness = analyze_training_readiness(
        [tmp_path / "run"],
        output_dir=tmp_path / "readiness",
        analysis_id="grpo-ready",
        experiment_id="exp",
    )
    assert readiness["aggregate"]["qwen_grpo_training_manifest_count"] == 1
    assert readiness["aggregate"]["qwen_grpo_trainable_group_count"] == 1
    assert readiness["aggregate"]["qwen_grpo_trainable_candidate_count"] == 2
    assert readiness["aggregate"]["qwen_ppo_step_sample_count"] == 2
    assert readiness["readiness"]["grpo_ready"] is True


def test_prepare_qwen_grpo_training_merges_multiple_exports_by_episode_prompt(tmp_path: Path) -> None:
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)],
        output_dir=tmp_path / "run_a" / "training_export",
        experiment_id="exp",
        run_id="run-a",
    )
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)],
        output_dir=tmp_path / "run_b" / "training_export",
        experiment_id="exp",
        run_id="run-b",
    )

    manifest = prepare_qwen_grpo_training(
        [tmp_path / "run_a", tmp_path / "run_b"],
        output_dir=tmp_path / "merged" / "qwen_grpo_training",
    )

    assert manifest["source_mode"] == "merged_training_exports"
    assert len(manifest["source_training_manifests"]) == 2
    assert manifest["source_rollout_group_count"] == 2
    assert manifest["group_count"] == 1
    assert manifest["candidate_count"] == 2
    assert manifest["trainable_group_count"] == 1
    assert manifest["ppo_step_sample_count"] == 2


def test_prepare_qwen_grpo_training_requires_actor_equal_executed(tmp_path: Path) -> None:
    actor = {"tool": "drive_straight", "args": {"power_percent": 20, "duration_s": 0.5}, "thought": "move"}
    replaced = {"tool": "wait", "args": {"duration_s": 0.2}, "thought": "replacement"}
    better = _summary_fixture(
        tmp_path,
        trial_id="trial_better",
        final_distance_m=0.2,
        success=True,
        actor_action=actor,
        executed_action=replaced,
    )
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )

    manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )

    assert manifest["status"] == "not_ready"
    assert manifest["trainable_group_count"] == 0
    assert manifest["trainable_candidate_count"] == 1
    assert any("at least two trainable" in blocker for blocker in manifest["blockers"])
    group = json.loads(
        (tmp_path / "run" / "qwen_grpo_training" / "qwen_grpo_rollout_groups.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    blocked = next(candidate for candidate in group["candidates"] if candidate["rollout_id"] == "trial_better")
    assert any("actor action replaced" in blocker for blocker in blocked["blockers"])


def test_prepare_qwen_grpo_training_can_allow_missing_images(tmp_path: Path) -> None:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    for image_path in tmp_path.glob("trial_*/policy/frames/0001.jpg"):
        image_path.unlink()

    manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
        require_existing_images=False,
    )

    assert manifest["status"] == "ready"
    assert manifest["missing_image_count"] == 2
    assert manifest["trainable_group_count"] == 1


def test_prepare_qwen_grpo_training_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_single", final_distance_m=1.2)],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-prepare-qwen-grpo-training",
            "--input",
            str(tmp_path / "run"),
            "--output-dir",
            str(tmp_path / "run" / "qwen_grpo_training"),
            "--fail-on-not-ready",
        ],
    )

    assert main() == 2


def _write_ready_grpo_handoff(tmp_path: Path) -> Path:
    better = _summary_fixture(tmp_path, trial_id="trial_better", final_distance_m=0.2, success=True)
    worse = _summary_fixture(tmp_path, trial_id="trial_worse", final_distance_m=2.5)
    export_training_data_from_summaries(
        [worse, better],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )
    return tmp_path / "run" / "qwen_grpo_training"


def _generated_reward_namespace() -> dict:
    script = qwen_grpo_job._TRAIN_SCRIPT
    start = script.index("def canonical_json")
    end = script.index("\ndef main")
    namespace = {
        "json": json,
        "os": os,
        "Path": Path,
        "gmtime": gmtime,
        "strftime": strftime,
        "_COMPLETION_LOG_BATCH_INDEX": 0,
    }
    exec(script[start:end], namespace)
    return namespace


def test_reference_tool_balancer_duplicates_underrepresented_tools_generically() -> None:
    records = [
        {
            "sample_id": "a",
            "reference_action_json": {"tool": "alpha_tool", "args": {"x": 1}},
            "candidate_step_reward": 0.2,
        },
        {
            "sample_id": "b",
            "reference_action_json": {"tool": "beta_tool", "args": {"y": 1}},
            "candidate_step_reward": 0.0,
        },
        {
            "sample_id": "c",
            "reference_action_json": {"tool": "beta_tool", "args": {"y": 2}},
            "candidate_step_reward": -0.1,
        },
    ]

    balanced = qwen_grpo_job._balance_dataset_records_by_reference_tool(records, max_multiplier=4)
    summary = qwen_grpo_job._grpo_dataset_action_summary(balanced)

    assert len(balanced) == 4
    assert summary["reference_action_tool_counts"] == {"alpha_tool": 2, "beta_tool": 2}
    assert summary["balanced_copy_count"] == 1
    duplicate = next(record for record in balanced if record.get("balance_original_sample_id"))
    assert duplicate["balance_original_sample_id"] == "a"
    assert duplicate["sample_id"] == "a_balance_copy01"


def test_plan_qwen_grpo_training_uses_existing_manifest_and_writes_job(tmp_path: Path) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)

    job = qwen_grpo_job.plan_qwen_grpo_training(
        grpo_dir,
        output_dir=tmp_path / "grpo_job",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_steps=7,
        num_generations=3,
    )

    assert job["schema"] == "flatdisk.qwen_grpo_training_job.v1"
    assert job["status"] == "ready"
    assert job["training_method"] == "offline_replay_grpo"
    assert job["trainer"] == "trl.GRPOTrainer"
    assert job["audit"]["online_environment_reward"] is False
    assert job["dataset"]["sample_count"] == 2
    assert job["dataset"]["source_group_count"] == 1
    assert job["dataset"]["source_ppo_step_count"] == 2
    assert job["dataset"]["image_reference_count"] == 2
    assert job["dataset"]["missing_image_count"] == 0
    assert job["dataset"]["forbidden_model_token_hits"] == []
    assert job["dataset_action_audit"]["before_balancing"]["reference_action_tool_counts"] == {"drive_straight": 2}
    assert job["dataset_action_audit"]["after_balancing"]["reference_action_tool_counts"] == {"drive_straight": 2}
    assert job["audit"]["reference_tool_balancing"] == {
        "enabled": False,
        "max_multiplier": 1,
        "sample_count_after": 2,
        "sample_count_before": 2,
    }
    assert job["training_args"]["max_steps"] == 7
    assert job["training_args"]["num_generations"] == 3
    assert job["training_args"]["max_completion_length"] == 96
    assert job["training_args"]["zero_reward_exact_action_bonus"] == 0.0
    assert job["audit"]["exact_action_reward_shaping"] == {
        "applies_only_when_candidate_step_reward_is_zero": True,
        "enabled": False,
        "requires_candidate_step_reward_present": True,
        "zero_reward_exact_action_bonus": 0.0,
    }
    assert job["adapter"]["method"] == "peft_lora"
    assert job["adapter"]["r"] == 8
    assert job["completion_log_jsonl"] == str(tmp_path / "grpo_job" / "adapter" / "completion_samples.jsonl")
    assert "trl" in job["required_packages"]
    assert "torchvision" in job["required_packages"]
    assert "accelerate launch" in job["launch_command"]
    assert "--max-completion-length 96" in job["launch_command"]
    assert job["launch_argv"][:3] == ["accelerate", "launch", str(tmp_path / "grpo_job" / "train_qwen_grpo_trl.py")]
    assert job["runtime"]["dependency_check"].startswith("importlib.util.find_spec")
    assert len(job["train_script_sha256"]) == 64

    job_path = tmp_path / "grpo_job" / "qwen_grpo_training_job.json"
    dataset_path = tmp_path / "grpo_job" / "qwen_grpo_trl_dataset.jsonl"
    train_script = tmp_path / "grpo_job" / "train_qwen_grpo_trl.py"
    assert job_path.exists()
    assert dataset_path.exists()
    assert train_script.exists()
    dataset_record = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert dataset_record["schema"] == "flatdisk.qwen_grpo_trl_prompt_sample.v1"
    assert dataset_record["prompt"] == dataset_record["prompt_messages"]
    assert dataset_record["source_prompt_messages"]
    assert "GRPO_RESPONSE_CONTRACT" in dataset_record["response_contract"]
    assert "thought" in dataset_record["response_contract"]
    assert "toilet" not in dataset_record["response_contract"].lower()
    last_content = dataset_record["prompt_messages"][-1]["content"]
    assert last_content[-1]["type"] == "text"
    assert last_content[-1]["text"] == dataset_record["response_contract"]
    assert dataset_record["image_paths"]
    assert dataset_record["reference_action_json"]["tool"] == "drive_straight"
    assert dataset_record["candidate_step_reward_present"] is True
    assert dataset_record["reward_source"].startswith("offline evaluator reward sidecar")
    script_text = train_script.read_text(encoding="utf-8")
    assert "GRPOTrainer" in script_text
    assert "AutoModelForImageTextToText" in script_text
    assert "LoraConfig" in script_text
    assert "get_peft_model" in script_text
    assert "navigation_tool_reward" in script_text
    assert "FLATDISK_GRPO_COMPLETION_LOG" in script_text
    assert "log_completion_batch" in script_text
    assert "partial_action_reward" in script_text
    assert "exact_action_reward" in script_text
    assert "arg_match_fraction" in script_text
    assert "expected_action" in script_text
    assert "candidate_step_reward_present" in script_text
    assert "zero_reward_exact_action_bonus" in script_text
    assert "base_reward - 0.05" in script_text
    assert "-0.02" in script_text
    assert "min(base_reward - 0.5, -0.5)" in script_text
    assert "conversational_text_messages" in script_text
    assert "record[\"prompt\"] = conversational_text_messages(messages)" in script_text
    assert "apply_chat_template" not in script_text


def test_plan_qwen_grpo_training_wires_zero_reward_exact_action_bonus(tmp_path: Path) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)

    job = qwen_grpo_job.plan_qwen_grpo_training(
        grpo_dir,
        output_dir=tmp_path / "grpo_job",
        zero_reward_exact_action_bonus=0.05,
    )

    assert job["training_args"]["zero_reward_exact_action_bonus"] == 0.05
    assert job["audit"]["exact_action_reward_shaping"] == {
        "applies_only_when_candidate_step_reward_is_zero": True,
        "enabled": True,
        "requires_candidate_step_reward_present": True,
        "zero_reward_exact_action_bonus": 0.05,
    }
    assert "--zero-reward-exact-action-bonus" in job["launch_argv"]
    assert "0.05" in job["launch_argv"]
    assert "--zero-reward-exact-action-bonus 0.05" in job["launch_command"]

    dataset_path = tmp_path / "grpo_job" / "qwen_grpo_trl_dataset.jsonl"
    dataset_record = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert "zero_reward_exact_action_bonus" not in json.dumps(dataset_record["prompt_messages"])
    assert "candidate_step_reward" not in json.dumps(dataset_record["prompt_messages"])
    assert "reference_action_canonical" not in json.dumps(dataset_record["prompt_messages"])
    assert "zero_reward_exact_action_bonus" not in dataset_record["response_contract"]
    assert "candidate_step_reward" not in dataset_record["response_contract"]


def test_plan_qwen_grpo_training_rejects_negative_zero_reward_exact_action_bonus(tmp_path: Path) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)

    try:
        qwen_grpo_job.plan_qwen_grpo_training(
            grpo_dir,
            output_dir=tmp_path / "grpo_job",
            zero_reward_exact_action_bonus=-0.01,
        )
    except ValueError as exc:
        assert "zero_reward_exact_action_bonus must be non-negative" in str(exc)
    else:
        raise AssertionError("expected negative zero_reward_exact_action_bonus to fail")


def test_generated_navigation_reward_applies_exact_bonus_only_to_observed_zero_rewards(
    tmp_path: Path, monkeypatch
) -> None:
    namespace = _generated_reward_namespace()
    navigation_tool_reward = namespace["navigation_tool_reward"]
    canonical_json = namespace["canonical_json"]
    completion_log = tmp_path / "completion_samples.jsonl"
    monkeypatch.setenv("FLATDISK_GRPO_COMPLETION_LOG", str(completion_log))

    alpha = {"tool": "alpha_tool", "args": {"target": "left", "power": 1}}
    alpha_wrong_args = {"tool": "alpha_tool", "args": {"target": "right", "power": 1}}
    beta = {"tool": "beta_tool", "args": {}}

    completions = [
        json.dumps({"action": alpha}),
        json.dumps({"action": alpha}),
        json.dumps({"action": alpha}),
        json.dumps({"action": alpha_wrong_args}),
        json.dumps({"action": beta}),
        "not json",
        json.dumps({"action": alpha}),
    ]
    rewards = navigation_tool_reward(
        completions,
        reference_action_canonical=[canonical_json(alpha)] * len(completions),
        candidate_step_reward=[0.0, 0.2, -0.2, 0.2, 0.2, 0.2, 0.0],
        candidate_step_reward_present=[True, True, True, True, True, True, False],
        reward_scale=[1.0] * len(completions),
        zero_reward_exact_action_bonus=[0.05] * len(completions),
        sample_id=[f"sample_{index}" for index in range(len(completions))],
    )

    assert rewards[0] == 0.05
    assert rewards[1] == 0.2
    assert rewards[2] == -0.2
    assert rewards[3] <= 0.0
    assert rewards[4] <= 0.0
    assert rewards[5] <= 0.0
    assert rewards[6] == 0.0
    assert completion_log.exists()
    rows = [json.loads(line) for line in completion_log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["zero_reward_exact_action_bonus"] == 0.05
    assert rows[0]["candidate_step_reward_present"] is True
    assert rows[6]["candidate_step_reward_present"] is False
    metrics = qwen_grpo_job._completion_log_metrics(completion_log)
    assert metrics["positive_non_reference_reward_count"] == 0


def test_plan_qwen_grpo_training_blocks_not_ready_manifest(tmp_path: Path) -> None:
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_single", final_distance_m=1.2)],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    grpo_manifest = prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )
    assert grpo_manifest["status"] == "not_ready"

    job = qwen_grpo_job.plan_qwen_grpo_training(
        tmp_path / "run" / "qwen_grpo_training",
        output_dir=tmp_path / "grpo_job",
    )

    assert job["status"] == "not_ready"
    assert any("not ready" in blocker for blocker in job["blockers"])
    assert any("no trainable groups" in blocker for blocker in job["blockers"])


def test_plan_qwen_grpo_training_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    export_training_data_from_summaries(
        [_summary_fixture(tmp_path, trial_id="trial_single", final_distance_m=1.2)],
        output_dir=tmp_path / "run" / "training_export",
        experiment_id="exp",
        run_id="run",
    )
    prepare_qwen_grpo_training(
        tmp_path / "run",
        output_dir=tmp_path / "run" / "qwen_grpo_training",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-plan-qwen-grpo-training",
            "--input",
            str(tmp_path / "run" / "qwen_grpo_training"),
            "--output-dir",
            str(tmp_path / "grpo_job"),
            "--fail-on-not-ready",
        ],
    )

    assert qwen_grpo_job.main() == 2
    job = json.loads((tmp_path / "grpo_job" / "qwen_grpo_training_job.json").read_text(encoding="utf-8"))
    assert job["status"] == "not_ready"


def test_run_qwen_grpo_training_job_dry_run_writes_result(tmp_path: Path) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)
    qwen_grpo_job.plan_qwen_grpo_training(grpo_dir, output_dir=tmp_path / "grpo_job")

    result = qwen_grpo_job.run_qwen_grpo_training_job(
        tmp_path / "grpo_job",
        dry_run=True,
        check_dependencies=False,
    )

    assert result["schema"] == "flatdisk.qwen_grpo_training_result.v1"
    assert result["status"] == "dry_run"
    assert result["returncode"] is None
    assert result["blockers"] == []
    assert result["dependency_check"]["enabled"] is False
    assert result["launch_argv"][0] == "accelerate"
    assert result["sample_count"] == 2
    result_path = tmp_path / "grpo_job" / "qwen_grpo_training_result.json"
    assert result_path.exists()


def test_run_qwen_grpo_training_job_reports_missing_dependencies(tmp_path: Path) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)
    qwen_grpo_job.plan_qwen_grpo_training(grpo_dir, output_dir=tmp_path / "grpo_job")
    job_path = tmp_path / "grpo_job" / "qwen_grpo_training_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = ["definitely_missing_flatdisk_training_package"]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = qwen_grpo_job.run_qwen_grpo_training_job(job_path, dry_run=True)

    assert result["status"] == "not_ready"
    assert result["dependency_check"]["missing_packages"] == ["definitely_missing_flatdisk_training_package"]
    assert any("missing required training package" in blocker for blocker in result["blockers"])


def test_run_qwen_grpo_training_job_executes_ready_job(tmp_path: Path) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)
    qwen_grpo_job.plan_qwen_grpo_training(grpo_dir, output_dir=tmp_path / "grpo_job")
    fake_train = tmp_path / "grpo_job" / "fake_train.py"
    job_path = tmp_path / "grpo_job" / "qwen_grpo_training_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    completion_log = Path(job["completion_log_jsonl"])
    completion_log_payload = (
        '{"completion_text":"{\\"action\\":{\\"tool\\":\\"wait\\",\\"args\\":{}}}",'
        '"parsed_action":{"args":{},"tool":"wait"},'
        '"reward":0.1,'
        '"reference_action_canonical":"{\\"args\\": {}, \\"tool\\": \\"wait\\"}",'
        '"completion_text_truncated":false}\n'
        '{"completion_text":"```json",'
        '"parsed_action":{},'
        '"reward":0.25,'
        '"reference_action_canonical":"{\\"args\\": {}, \\"tool\\": \\"wait\\"}",'
        '"completion_text_truncated":false}\n'
    )
    fake_train.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                f"path = Path({str(completion_log)!r})",
                "path.parent.mkdir(parents=True, exist_ok=True)",
                f"path.write_text({completion_log_payload!r}, encoding='utf-8')",
                "print('grpo-trained-ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    job["required_packages"] = []
    job["train_script"] = str(fake_train)
    job["launch_command"] = f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_train))}"
    job["launch_argv"] = [sys.executable, str(fake_train)]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = qwen_grpo_job.run_qwen_grpo_training_job(job_path)

    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert "grpo-trained-ok" in result["stdout_tail"]
    assert result["completion_log_jsonl"] == str(completion_log)
    assert result["completion_log_sample_count"] == 2
    assert result["completion_log_metrics"]["sample_count"] == 2
    assert result["completion_log_metrics"]["parsed_action_count"] == 1
    assert result["completion_log_metrics"]["exact_reference_action_count"] == 1
    assert result["completion_log_metrics"]["positive_non_reference_reward_count"] == 1
    assert result["completion_log_metrics"]["tool_match_count"] == 1
    assert result["completion_log_metrics"]["exact_reference_action_rate"] == 0.5
    assert result["completion_log_metrics"]["tool_match_rate"] == 0.5
    assert result["completion_log_metrics"]["mean_arg_match_fraction"] == 0.5
    assert result["completion_log_metrics"]["markdown_fence_count"] == 1


def test_run_qwen_grpo_training_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    grpo_dir = _write_ready_grpo_handoff(tmp_path)
    qwen_grpo_job.plan_qwen_grpo_training(grpo_dir, output_dir=tmp_path / "grpo_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-run-qwen-grpo-training",
            "--job",
            str(tmp_path / "grpo_job"),
            "--dry-run",
            "--skip-dependency-check",
        ],
    )

    assert qwen_grpo_job.run_main() == 0
