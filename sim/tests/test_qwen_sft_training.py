from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

from flatdisk_sim.qwen_sft_training import (
    main,
    module_main,
    plan_qwen_sft_training,
    plan_qwen_sft_training_from_prompt_action_dataset,
    run_main,
    run_qwen_sft_training_job,
)


def _write_qwen_sft_fixture(tmp_path: Path, *, prompt_text: str = "Drive safely.", count: int = 2) -> Path:
    qwen_dir = tmp_path / "qwen_tool_training"
    qwen_dir.mkdir()
    image_path = qwen_dir / "frame.jpg"
    image_path.write_bytes(b"fake image")
    sft_path = qwen_dir / "qwen_sft_messages.jsonl"
    records = []
    for index in range(count):
        records.append(
            {
                "schema": "flatdisk.qwen_tool_sft_sample.v1",
                "sample_id": f"sample-{index:03d}",
                "source_policy_step_id": f"trial_001_step_{index:03d}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action": {
                                    "tool": "drive_straight",
                                    "args": {"duration_s": 0.5, "power_percent": 18.0},
                                }
                            },
                            sort_keys=True,
                        ),
                    },
                ],
                "images": [str(image_path)],
                "image_paths": [str(image_path)],
                "sft_weight": 1.0,
                "training_weight": 1.0,
                "audit": {"evaluator_labels_excluded_from_messages": True},
            }
        )
    sft_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    manifest_path = qwen_dir / "qwen_tool_training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "flatdisk.qwen_tool_training_manifest.v1",
                "output_dir": str(qwen_dir),
                "manifest_path": str(manifest_path),
                "qwen_sft_messages_jsonl": str(sft_path),
                "accepted_count": count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return qwen_dir


def test_plan_qwen_sft_training_writes_ready_job_and_teacher_forced_script(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path, count=2)

    job = plan_qwen_sft_training(
        qwen_dir,
        output_dir=tmp_path / "sft_job",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_samples=1,
        max_steps=3,
        learning_rate=1e-5,
    )

    assert job["schema"] == "flatdisk.qwen_sft_training_job.v1"
    assert job["status"] == "ready"
    assert job["dataset"]["sample_count"] == 1
    assert job["dataset"]["source_sample_count"] == 2
    assert job["dataset"]["assistant_target_count"] == 1
    assert job["dataset"]["image_reference_count"] == 1
    assert job["dataset"]["missing_image_count"] == 0
    assert job["dataset"]["forbidden_model_token_hits"] == {}
    assert job["training_args"]["max_samples"] == 1
    assert job["training_args"]["max_steps"] == 3
    assert job["training_args"]["learning_rate"] == 1e-5
    assert job["target_mode"] == "assistant"
    assert job["dataset"]["target_mode_counts"] == {"assistant": 1}
    assert job["dataset"]["target_action_tool_counts"] == {"drive_straight": 1}
    assert "trl" not in job["required_packages"]
    assert "peft" in job["required_packages"]
    assert job["launch_argv"][:2] == ["python", str(tmp_path / "sft_job" / "train_qwen_sft_lora.py")]
    assert "--training-log" in job["launch_argv"]
    assert job["runtime"]["dependency_check"].startswith("importlib.util.find_spec")
    assert (tmp_path / "sft_job" / "qwen_sft_training_job.json").exists()
    assert (tmp_path / "sft_job" / "qwen_sft_training_dataset.jsonl").exists()
    train_script = tmp_path / "sft_job" / "train_qwen_sft_lora.py"
    assert train_script.exists()
    assert len(job["train_script_sha256"]) == 64
    script_text = train_script.read_text(encoding="utf-8")
    assert "LoraConfig" in script_text
    assert "labels[:, :target_start] = -100" in script_text
    assert "apply_chat_template" in script_text
    assert "DPOTrainer" not in script_text
    assert "GRPOTrainer" not in script_text


def test_plan_qwen_sft_training_can_write_action_only_targets(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path, count=2)

    job = plan_qwen_sft_training(
        qwen_dir,
        output_dir=tmp_path / "sft_job",
        max_samples=1,
        max_steps=3,
        target_mode="action-only",
    )

    assert job["status"] == "ready"
    assert job["target_mode"] == "action-only"
    assert job["training_args"]["target_mode"] == "action-only"
    assert job["audit"]["target_source"] == "qwen_sft_messages compact action content"
    assert job["dataset"]["sample_count"] == 1
    assert job["dataset"]["target_mode_counts"] == {"action-only": 1}
    assert job["dataset"]["target_action_tool_counts"] == {"drive_straight": 1}
    rows = [
        json.loads(line)
        for line in (tmp_path / "sft_job" / "qwen_sft_training_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["target_mode"] == "action-only"
    assert row["assistant_target_json"] == {
        "action": {"args": {"duration_s": 0.5, "power_percent": 18.0}, "tool": "drive_straight"}
    }
    assert row["messages"][-1]["role"] == "assistant"
    assert row["messages"][-1]["content"] == (
        '{"action":{"args":{"duration_s":0.5,"power_percent":18.0},"tool":"drive_straight"}}'
    )
    assert len(row["source_assistant_target_sha256"]) == 64
    assert row["audit"]["target_mode"] == "action-only"


def test_plan_qwen_sft_training_from_prompt_action_dataset_writes_action_only_records(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image")
    dataset_path = tmp_path / "qwen_grpo_action_likelihood_dataset.jsonl"
    records = [
        {
            "sample_id": "sample-000",
            "source_rollout_id": "rollout-a",
            "prompt_messages": [{"role": "user", "content": [{"type": "image", "image": str(image_path)}]}],
            "image_paths": [str(image_path)],
            "reference_action_json": {"tool": "check_object_grounding", "args": {"prompt": "chair"}},
            "candidate_step_reward": 0.0,
        },
        {
            "sample_id": "sample-001",
            "source_rollout_id": "rollout-b",
            "prompt_messages": [{"role": "user", "content": [{"type": "text", "text": "turn left"}]}],
            "image_paths": [str(image_path)],
            "reference_action_json": {"tool": "turn_by_angle", "args": {"degrees": -30.0, "power_percent": 12.0}},
            "candidate_step_reward": -0.1,
        },
        {
            "sample_id": "sample-002",
            "source_rollout_id": "rollout-c",
            "prompt_messages": [{"role": "user", "content": [{"type": "text", "text": "stop"}]}],
            "image_paths": [str(image_path)],
            "reference_action_json": {"tool": "stop", "args": {}},
            "candidate_step_reward": 0.0,
        },
    ]
    dataset_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    job = plan_qwen_sft_training_from_prompt_action_dataset(
        dataset_path,
        output_dir=tmp_path / "sft_job",
        sample_offset=1,
        max_samples=2,
        max_steps=4,
    )

    assert job["status"] == "ready"
    assert job["source_dataset_kind"] == "qwen_grpo_prompt_action_jsonl"
    assert job["source_prompt_action_dataset_jsonl"] == str(dataset_path)
    assert job["qwen_tool_training_manifest"] == ""
    assert job["target_mode"] == "action-only"
    assert job["training_args"]["sample_offset"] == 1
    assert job["training_args"]["sample_stride"] == 1
    assert job["dataset"]["source_sample_count"] == 3
    assert job["dataset"]["sample_count"] == 2
    assert job["dataset"]["target_action_tool_counts"] == {"stop": 1, "turn_by_angle": 1}
    assert job["audit"]["reference_actions_used_only_as_sft_targets"] is True

    rows = [
        json.loads(line)
        for line in (tmp_path / "sft_job" / "qwen_sft_training_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["sample_id"] for row in rows] == ["sample-001", "sample-002"]
    assert rows[0]["schema"] == "flatdisk.qwen_tool_sft_sample.v1"
    assert rows[0]["messages"][-1]["role"] == "assistant"
    assert rows[0]["messages"][-1]["content"] == (
        '{"action":{"args":{"degrees":-30.0,"power_percent":12.0},"tool":"turn_by_angle"}}'
    )
    prompt_text = json.dumps(rows[0]["messages"][:-1])
    assert "candidate_step_reward" not in prompt_text
    assert "reference_action_json" not in prompt_text
    assert rows[0]["metadata"]["source_dataset_kind"] == "qwen_grpo_prompt_action_jsonl"


def test_plan_qwen_sft_training_blocks_forbidden_prompt_tokens(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path, prompt_text="Hidden distance_m should not be here.")

    job = plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job")

    assert job["status"] == "not_ready"
    assert job["dataset"]["forbidden_model_token_hits"] == {"distance_m": 2}
    assert any("forbidden privileged token" in blocker for blocker in job["blockers"])


def test_plan_qwen_sft_training_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path, prompt_text="target_pose leak")
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-plan-qwen-sft-training",
            "--input",
            str(qwen_dir),
            "--output-dir",
            str(tmp_path / "sft_job"),
            "--fail-on-not-ready",
        ],
    )

    assert main() == 2
    job = json.loads((tmp_path / "sft_job" / "qwen_sft_training_job.json").read_text(encoding="utf-8"))
    assert job["status"] == "not_ready"


def test_plan_qwen_sft_training_cli_accepts_prompt_action_dataset(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image")
    dataset_path = tmp_path / "qwen_grpo_completion_eval_dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-000",
                "prompt_messages": [{"role": "user", "content": [{"type": "text", "text": "stop"}]}],
                "image_paths": [str(image_path)],
                "reference_action_canonical": '{"args": {}, "tool": "stop"}',
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-plan-qwen-sft-training",
            "--dataset-jsonl",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "sft_job"),
            "--max-steps",
            "2",
            "--fail-on-not-ready",
        ],
    )

    assert main() == 0
    job = json.loads((tmp_path / "sft_job" / "qwen_sft_training_job.json").read_text(encoding="utf-8"))
    assert job["status"] == "ready"
    assert job["source_prompt_action_dataset_jsonl"] == str(dataset_path)
    assert job["dataset"]["target_action_tool_counts"] == {"stop": 1}


def test_run_qwen_sft_training_job_dry_run_writes_result(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job")

    result = run_qwen_sft_training_job(tmp_path / "sft_job", dry_run=True, check_dependencies=False)

    assert result["schema"] == "flatdisk.qwen_sft_training_result.v1"
    assert result["status"] == "dry_run"
    assert result["returncode"] is None
    assert result["blockers"] == []
    assert result["dependency_check"]["enabled"] is False
    assert result["launch_argv"][0] == "python"
    assert result["sample_count"] == 2
    assert result["training_log_tail"] == []
    assert (tmp_path / "sft_job" / "qwen_sft_training_result.json").exists()


def test_run_qwen_sft_training_job_reports_missing_dependencies(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job")
    job_path = tmp_path / "sft_job" / "qwen_sft_training_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = ["definitely_missing_flatdisk_training_package"]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_qwen_sft_training_job(job_path, dry_run=True)

    assert result["status"] == "not_ready"
    assert result["dependency_check"]["missing_packages"] == ["definitely_missing_flatdisk_training_package"]
    assert any("missing required training package" in blocker for blocker in result["blockers"])


def test_run_qwen_sft_training_job_executes_ready_job_and_attaches_training_log(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job")
    fake_train = tmp_path / "sft_job" / "fake_train.py"
    fake_log = tmp_path / "sft_job" / "qwen_sft_training_log.jsonl"
    fake_train.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"Path({str(fake_log)!r}).write_text(json.dumps({{'stage':'complete','loss':0.1}})+'\\n')\n"
        "print('sft-trained-ok')\n",
        encoding="utf-8",
    )
    job_path = tmp_path / "sft_job" / "qwen_sft_training_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = []
    job["train_script"] = str(fake_train)
    job["launch_command"] = f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_train))}"
    job["launch_argv"] = [sys.executable, str(fake_train)]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_qwen_sft_training_job(job_path)

    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert "sft-trained-ok" in result["stdout_tail"]
    assert result["training_log_tail"][-1]["stage"] == "complete"


def test_run_qwen_sft_training_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-run-qwen-sft-training",
            "--job",
            str(tmp_path / "sft_job"),
            "--dry-run",
            "--skip-dependency-check",
        ],
    )

    assert run_main() == 0


def test_qwen_sft_training_module_main_supports_run_subcommand(tmp_path: Path, monkeypatch) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m flatdisk_sim.qwen_sft_training",
            "run",
            "--job",
            str(tmp_path / "sft_job"),
            "--dry-run",
            "--skip-dependency-check",
        ],
    )

    assert module_main() == 0
