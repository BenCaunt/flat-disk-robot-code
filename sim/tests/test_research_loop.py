from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from flatdisk_sim import research_loop
from flatdisk_sim.research_loop import build_trial_matrix, load_config, run_research_loop, warmhub_ops


def test_research_loop_dry_run_writes_trial_matrix_and_warmhub_bundle(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "dry-run prompt sweep",
                "episodes": ["living_room_sofa"],
                "parallelism": 2,
                "variants": [
                    {"name": "qwen_baseline", "runner": "qwen"},
                    {
                        "name": "qwen_explore",
                        "runner": "qwen",
                        "critic_mode": "none",
                        "prompt_profile": "explore-v1",
                        "actor_rules": ["Write useful scratchpad state."],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=True,
        commit_warmhub=False,
        init_warmhub_repo=False,
    )

    assert aggregate["dry_run"] is True
    assert aggregate["trial_count"] == 2
    assert aggregate["completed_trial_count"] == 0
    manifest = json.loads((tmp_path / "out" / "research_loop_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"].startswith("test_open_vocab_")
    assert [trial["variant"]["name"] for trial in manifest["trials"]] == ["qwen_baseline", "qwen_explore"]
    assert manifest["trials"][1]["variant"]["critic_mode"] == "none"
    assert manifest["trials"][0]["slot_id"] == "test_open_vocab_qwen_baseline_living_room_sofa_r1"
    assert manifest["trials"][0]["trial_id"] != manifest["trials"][0]["slot_id"]
    shapes = json.loads((tmp_path / "out" / "warmhub_shapes.json").read_text(encoding="utf-8"))
    assert {"NavExperiment", "PromptVariant", "NavEvalRun", "FailureObservation"} <= set(shapes)
    assert shapes["PromptVariant"]["fields"]["noHardcodedLabelsOrColors"] == "boolean"
    assert shapes["NavEvalRun"]["fields"]["bestDistanceM?"] == "number"
    assert shapes["NavEvalRun"]["fields"]["actorModel"] == "string"
    assert shapes["NavEvalRun"]["fields"]["qwenModel?"] == "string"
    assert shapes["NavArtifact"]["fields"]["availabilityStatus"] == "string"
    assert shapes["NavArtifact"]["fields"]["directoryManifestSha256?"] == "string"
    assert shapes["RunAssessment"]["fields"]["finalToBestRegressionM?"] == "number"
    assert shapes["PromotionDecision"]["fields"]["status"]["enum"] == ["promote", "reject"]
    assert shapes["TrainingReadiness"]["fields"]["grpoReady"] == "boolean"
    ops = json.loads((tmp_path / "out" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert any(op["name"] == "NavExperiment/test_open_vocab" for op in ops)
    assert any(op["name"] == "PromptVariant/test_open_vocab_qwen_explore" for op in ops)
    variant_op = next(op for op in ops if op["name"] == "PromptVariant/test_open_vocab_qwen_explore")
    assert variant_op["data"]["criticMode"] == "none"
    assert variant_op["data"]["topomapMemoryUseClip"] is False
    assert variant_op["data"]["topomapMemoryAllowSemanticTerms"] is False
    assert (tmp_path / "out" / "AGENTS.warmhub.md").exists()


def test_research_loop_marks_semantic_topomap_terms_as_non_strict(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "semantic term debug mode",
                "strict_model_based": False,
                "episodes": ["living_room_sofa"],
                "variants": [
                    {
                        "name": "qwen_semantic_debug",
                        "runner": "qwen",
                        "topomap_memory_map_dir": "maps/{episode}",
                        "topomap_memory_allow_semantic_terms": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=True,
        commit_warmhub=False,
        init_warmhub_repo=False,
    )

    assert aggregate["no_hardcoded_labels_or_colors"] is False
    manifest = json.loads((tmp_path / "out" / "research_loop_manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_hardcoded_labels_or_colors"] is False
    ops = json.loads((tmp_path / "out" / "warmhub_ops.json").read_text(encoding="utf-8"))
    experiment = next(op for op in ops if op["name"] == "NavExperiment/test_open_vocab")
    variant = next(op for op in ops if op["name"] == "PromptVariant/test_open_vocab_qwen_semantic_debug")
    assert experiment["data"]["noHardcodedLabelsOrColors"] is False
    assert variant["data"]["noHardcodedLabelsOrColors"] is False


def test_research_loop_rejects_semantic_terms_in_strict_mode(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "semantic term debug mode",
                "episodes": ["living_room_sofa"],
                "variants": [
                    {
                        "name": "qwen_semantic_debug",
                        "runner": "qwen",
                        "topomap_memory_map_dir": "maps/{episode}",
                        "topomap_memory_allow_semantic_terms": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(ValueError, match="topomap_memory_allow_semantic_terms"):
        run_research_loop(
            config,
            config_path=config_path,
            output_root=tmp_path / "out",
            dry_run=True,
            commit_warmhub=False,
            init_warmhub_repo=False,
        )


def test_research_loop_rejects_scripted_runners_in_strict_mode(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "scripted smoke mode",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "scripted_smoke", "runner": "scripted-open-vocab"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(ValueError, match="scripted-open-vocab"):
        run_research_loop(
            config,
            config_path=config_path,
            output_root=tmp_path / "out",
            dry_run=True,
            commit_warmhub=False,
            init_warmhub_repo=False,
        )


def test_research_loop_marks_explicit_smoke_runners_as_non_strict(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "scripted smoke mode",
                "strict_model_based": False,
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "scripted_smoke", "runner": "scripted-open-vocab"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=True,
        commit_warmhub=False,
        init_warmhub_repo=False,
    )

    assert aggregate["strict_model_based"] is False
    assert aggregate["no_hardcoded_labels_or_colors"] is False
    ops = json.loads((tmp_path / "out" / "warmhub_ops.json").read_text(encoding="utf-8"))
    experiment = next(op for op in ops if op["name"] == "NavExperiment/test_open_vocab")
    variant = next(op for op in ops if op["name"] == "PromptVariant/test_open_vocab_scripted_smoke")
    assert experiment["data"]["strictModelBased"] is False
    assert variant["data"]["noHardcodedLabelsOrColors"] is False


def test_warmhub_ops_record_failed_run_artifacts_and_failure_observation(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "failed run logging",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    run_dir = tmp_path / "run"
    policy_dir = run_dir / "policy"
    topomap_dir = policy_dir / "topomap_memory"
    evaluator_dir = run_dir / "evaluator_hidden"
    prompts_dir = policy_dir / "prompts"
    for directory in (topomap_dir, evaluator_dir, prompts_dir):
        directory.mkdir(parents=True)
    (run_dir / "episode_summary.json").write_text("{}", encoding="utf-8")
    (policy_dir / "memory.jsonl").write_text("{}\n", encoding="utf-8")
    (policy_dir / "camera_contact_sheet.jpg").write_bytes(b"camera")
    (policy_dir / "policy_review_trace.json").write_text("{}", encoding="utf-8")
    (run_dir / "progress_contact_sheet.jpg").write_bytes(b"progress")
    (topomap_dir / "topomap_memory_manifest.json").write_text("{}", encoding="utf-8")
    (topomap_dir / "query_log.jsonl").write_text("{}\n", encoding="utf-8")
    (topomap_dir / "0001_route_topomap.jpg").write_bytes(b"topomap")
    summary = {
        "trial_id": "trial_failed",
        "variant": "qwen_baseline",
        "episode": "living_room_sofa",
        "runner": "qwen",
        "model": "gpt-5.5",
        "qwen_model": "Qwen/Qwen3-VL-8B-Instruct",
        "qwen_endpoint": "http://127.0.0.1:8000/v1/chat/completions",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "evaluator_only_dir": str(evaluator_dir),
        "camera_contact_sheet": str(policy_dir / "camera_contact_sheet.jpg"),
        "policy_review_trace_json": str(policy_dir / "policy_review_trace.json"),
        "progress_contact_sheet": str(run_dir / "progress_contact_sheet.jpg"),
        "success": False,
        "final_distance_m": 1.23,
        "best_distance_m": 0.42,
        "best_distance_step": 7,
        "best_distance_improvement_m": 0.81,
        "final_distance_improvement_m": 0.4,
        "final_to_best_regression_m": 0.81,
        "reached_success_radius_ever": False,
        "reason": "max_steps_exhausted",
        "step_count": 18,
        "wall_clock_s": 12.0,
    }

    ops = warmhub_ops(
        config,
        {
            "started_at": "2026-06-13T00:00:00Z",
            "git_commit": "abc123",
            "trial_count": 1,
            "completed_trial_count": 1,
            "success_count": 0,
            "summaries": [summary],
        },
    )

    names = {op["name"] for op in ops}
    assert "NavEvalRun/trial_failed" in names
    assert "RunAssessment/trial_failed" in names
    assert "FailureObservation/trial_failed" in names
    assert "NavArtifact/trial_failed_camera_contact_sheet" in names
    assert "NavArtifact/trial_failed_topomap_memory_manifest" in names
    assert "NavArtifact/trial_failed_topomap_memory_query_jsonl" in names
    assert "NavArtifact/trial_failed_topomap_memory_contact_sheets" in names
    assert "NavArtifact/trial_failed_policy_review_trace_json" in names
    run = next(op for op in ops if op["name"] == "NavEvalRun/trial_failed")
    assessment = next(op for op in ops if op["name"] == "RunAssessment/trial_failed")
    assert run["data"]["model"] == "gpt-5.5"
    assert run["data"]["actorModel"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert run["data"]["qwenModel"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert run["data"]["qwenEndpoint"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert run["data"]["bestDistanceM"] == 0.42
    assert assessment["data"]["finalToBestRegressionM"] == 0.81
    camera_artifact = next(op for op in ops if op["name"] == "NavArtifact/trial_failed_camera_contact_sheet")
    assert camera_artifact["data"]["pathKind"] == "file"
    assert camera_artifact["data"]["availabilityStatus"] == "available_at_commit_path"
    assert camera_artifact["data"]["sha256"] == hashlib.sha256(b"camera").hexdigest()
    assert camera_artifact["data"]["sizeBytes"] == len(b"camera")
    assert "writer filesystem" in camera_artifact["data"]["retrievalHint"]
    topomap_dir_artifact = next(op for op in ops if op["name"] == "NavArtifact/trial_failed_topomap_memory_contact_sheets")
    assert topomap_dir_artifact["data"]["pathKind"] == "directory"
    assert topomap_dir_artifact["data"]["directoryFileCount"] == 3
    assert topomap_dir_artifact["data"]["directoryTotalBytes"] > 0
    assert len(topomap_dir_artifact["data"]["directoryManifestSha256"]) == 64
    failure = next(op for op in ops if op["name"] == "FailureObservation/trial_failed")
    assert str(policy_dir / "policy_review_trace.json") in failure["data"]["evidenceArtifacts"]


def test_warmhub_ops_use_summary_generality_flag_for_eval_run(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "generality flag",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    ops = warmhub_ops(
        config,
        {
            "started_at": "2026-06-13T00:00:00Z",
            "git_commit": "abc123",
            "trial_count": 1,
            "completed_trial_count": 1,
            "success_count": 0,
            "no_hardcoded_labels_or_colors": False,
            "summaries": [
                {
                    "trial_id": "trial_non_strict",
                    "variant": "qwen_baseline",
                    "episode": "living_room_sofa",
                    "runner": "qwen",
                    "model": "gpt-5.5",
                    "run_dir": str(tmp_path / "run"),
                    "success": False,
                    "final_distance_m": 1.0,
                    "reason": "prompt_audit_failed",
                    "step_count": 1,
                    "wall_clock_s": 0.1,
                    "no_hardcoded_labels_or_colors": False,
                }
            ],
        },
    )

    experiment = next(op for op in ops if op["name"] == "NavExperiment/test_open_vocab")
    run = next(op for op in ops if op["name"] == "NavEvalRun/trial_non_strict")
    assert experiment["data"]["noHardcodedLabelsOrColors"] is False
    assert run["data"]["noHardcodedLabelsOrColors"] is False



def test_trial_matrix_can_preserve_slot_ids_while_adding_run_id(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "trial id uniqueness",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    default_trial = build_trial_matrix(config)[0]
    run_trial = build_trial_matrix(config, run_id="run-001")[0]

    assert default_trial.trial_id == default_trial.slot_id
    assert run_trial.slot_id == default_trial.slot_id
    assert run_trial.trial_id == "run-001_qwen_baseline_living_room_sofa_r1"


def test_research_loop_preflight_records_qwen_endpoint_failures(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "endpoint preflight",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    monkeypatch.setattr(research_loop, "_qwen_endpoint_error", lambda _endpoint: "connection refused")

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=False,
        commit_warmhub=False,
        init_warmhub_repo=False,
        preflight_endpoints=True,
    )

    assert aggregate["completed_trial_count"] == 0
    assert aggregate["failure_count"] == 1
    assert aggregate["failed_trials"][0]["reason"] == "trial_exception"
    assert "connection refused" in aggregate["failed_trials"][0]["error"]
    ops = json.loads((tmp_path / "out" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert any(op["name"].startswith("NavEvalRun/") for op in ops)
    assert any(op["name"].startswith("FailureObservation/") for op in ops)


def test_research_loop_preflight_records_missing_topomap_memory_map(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "topomap preflight",
                "episodes": ["living_room_sofa"],
                "variants": [
                    {
                        "name": "qwen_topomap",
                        "runner": "qwen",
                        "topomap_memory_map_dir": str(tmp_path / "missing_map"),
                        "topomap_memory_use_clip": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    monkeypatch.setattr(research_loop, "_qwen_endpoint_error", lambda _endpoint: None)

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=False,
        commit_warmhub=False,
        init_warmhub_repo=False,
        preflight_endpoints=True,
    )

    assert aggregate["completed_trial_count"] == 0
    assert aggregate["failure_count"] == 1
    assert aggregate["failed_trials"][0]["reason"] == "trial_exception"
    assert "topomap memory preflight failed" in aggregate["failed_trials"][0]["error"]


def test_research_loop_preflight_only_does_not_execute_runnable_trials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "preflight only",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    monkeypatch.setattr(research_loop, "_qwen_endpoint_error", lambda _endpoint: None)

    def fail_episode(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("preflight-only should not execute THOR trials")

    monkeypatch.setattr(research_loop, "run_thor_harness_episode", fail_episode)

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=False,
        commit_warmhub=False,
        init_warmhub_repo=False,
        preflight_only=True,
    )

    assert aggregate["completed_trial_count"] == 0
    assert aggregate["failure_count"] == 0
    assert aggregate["trial_count"] == 1
    manifest = json.loads((tmp_path / "out" / "research_loop_manifest.json").read_text(encoding="utf-8"))
    assert manifest["preflight_only"] is True


def test_research_loop_exports_training_records_for_completed_trials(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "training export",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    def fake_episode(spec, *, output_root, **_kwargs):  # noqa: ANN001
        run_dir = output_root / spec.name
        policy_dir = run_dir / "policy"
        prompts_dir = policy_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "000_actor.txt").write_text("STATIC_HARNESS_CONTEXT\nDYNAMIC_TASK_STATE\n", encoding="utf-8")
        (policy_dir / "harness_events.jsonl").write_text(
            json.dumps({"event": "actor", "step": 0, "output": '{"action":{"tool":"wait","args":{"duration_s":0.2}},"thought":"pause"}'})
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "episode": spec.name,
            "scene": spec.scene,
            "prompt": spec.prompt,
            "success_radius_m": spec.success_radius_m,
            "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
            "runner": "qwen",
            "success": False,
            "reason": "max_steps_exhausted",
            "final_distance_m": 2.0,
            "step_count": 1,
            "wall_clock_s": 0.1,
            "policy_dir": str(policy_dir),
            "run_dir": str(run_dir),
            "steps": [
                {
                    "step": 0,
                    "harness_memory_record": {
                        "step": 0,
                        "goal": spec.prompt,
                        "observation": {"path": "frames/0001.jpg", "yaw_deg": 0.0, "frame_seq": 1, "brightness_center": 0.5},
                        "actor_action": {"tool": "wait", "args": {"duration_s": 0.2}},
                        "actor_memory_update": {},
                        "critic": {"verdict": "approve"},
                        "executed_action": {"tool": "wait", "args": {"duration_s": 0.2}},
                        "tool_result": {"command": "wait"},
                    },
                    "hidden_score_for_evaluator_only": {"success": False, "distance_m": 2.0},
                }
            ],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "episode_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr(research_loop, "run_thor_harness_episode", fake_episode)

    aggregate = run_research_loop(
        config,
        config_path=config_path,
        output_root=tmp_path / "out",
        dry_run=False,
        commit_warmhub=False,
        init_warmhub_repo=False,
    )

    assert aggregate["completed_trial_count"] == 1
    assert aggregate["training_export"]["step_count"] == 1
    summary = aggregate["summaries"][0]
    assert Path(summary["training_policy_steps_jsonl"]).exists()
    assert Path(summary["policy_review_trace_json"]).exists()
    ops = json.loads((tmp_path / "out" / "warmhub_ops.json").read_text(encoding="utf-8"))
    assert any(op["name"].endswith("_training_policy_steps_jsonl") for op in ops)
    assert any(op["name"].endswith("_policy_review_trace_json") for op in ops)


def test_warmhub_ops_record_trial_exceptions_as_failed_runs(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "exception logging",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)

    ops = warmhub_ops(
        config,
        {
            "started_at": "2026-06-13T00:00:00Z",
            "git_commit": "abc123",
            "trial_count": 1,
            "completed_trial_count": 0,
            "success_count": 0,
            "summaries": [],
            "failed_trials": [
                {
                    "trial_id": "trial_exception",
                    "variant": "qwen_baseline",
                    "episode": "living_room_sofa",
                    "runner": "qwen",
                    "model": "gpt-5.5",
                    "run_dir": str(tmp_path / "missing"),
                    "success": False,
                    "final_distance_m": None,
                    "reason": "trial_exception",
                    "error": "Qwen endpoint refused connection",
                    "step_count": 0,
                    "wall_clock_s": 0.0,
                }
            ],
        },
    )

    run = next(op for op in ops if op["name"] == "NavEvalRun/trial_exception")
    failure = next(op for op in ops if op["name"] == "FailureObservation/trial_exception")
    assert run["data"]["success"] is False
    assert run["data"]["reason"] == "trial_exception"
    assert failure["data"]["category"] == "trial_exception"
    assert "Qwen endpoint refused connection" in failure["data"]["symptom"]
    assert not any(op["name"].startswith("NavArtifact/trial_exception") for op in ops)


def test_main_can_override_warmhub_repo(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "repo override",
                "episodes": ["living_room_sofa"],
                "variants": [{"name": "qwen_baseline", "runner": "qwen"}],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flatdisk-sim-research-loop",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--warmhub-repo",
            "example-org/example-repo",
            "--dry-run",
        ],
    )

    assert research_loop.main() == 0

    run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "research_loop_manifest.json").read_text(encoding="utf-8"))
    assert manifest["warmhub_repo"] == "example-org/example-repo"


def test_commit_warmhub_bundle_revises_existing_stale_shapes(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / "bundle"
    output_root.mkdir()
    (output_root / "warmhub_shapes.json").write_text(
        json.dumps({"NavEvalRun": {"description": "new desc", "fields": {"bestDistanceM?": "number"}}}),
        encoding="utf-8",
    )
    (output_root / "warmhub_ops.json").write_text("[]", encoding="utf-8")
    calls = []

    def fake_run(command, **_kwargs):  # noqa: ANN001
        calls.append(command)
        if command[:3] == ["wh", "shape", "view"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"data": {"description": "old desc", "fields": {"finalDistanceM?": "number"}}}),
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(research_loop.subprocess, "run", fake_run)

    research_loop.commit_warmhub_bundle(output_root, repo="bencaunt-2/open-vocab-nav-research-loop", init_repo=False)

    assert any(command[:3] == ["wh", "shape", "revise"] for command in calls)
    assert not any(command[:3] == ["wh", "shape", "create"] for command in calls)
    revise = next(command for command in calls if command[:3] == ["wh", "shape", "revise"])
    assert revise[3] == "NavEvalRun"
    assert json.loads(revise[revise.index("--fields") + 1]) == {"bestDistanceM?": "number"}
