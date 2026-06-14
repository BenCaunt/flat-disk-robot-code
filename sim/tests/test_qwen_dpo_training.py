from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

from flatdisk_sim.qwen_dpo_training import main, plan_qwen_dpo_training, run_main, run_qwen_dpo_training_job


def _write_qwen_dpo_fixture(tmp_path: Path, *, prompt_text: str = "Drive safely.") -> Path:
    qwen_dir = tmp_path / "qwen_tool_training"
    qwen_dir.mkdir()
    image_path = qwen_dir / "frame.jpg"
    image_path.write_bytes(b"fake image")
    dpo_path = qwen_dir / "qwen_dpo_messages.jsonl"
    dpo_record = {
        "schema": "flatdisk.qwen_tool_dpo_sample.v1",
        "sample_id": "sample-001",
        "prompt": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "chosen": [
            {
                "role": "assistant",
                "content": json.dumps({"action": {"tool": "wait", "args": {"duration_s": 0.2}}}),
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": json.dumps({"action": {"tool": "visual_servo_object", "args": {"prompt": "target"}}}),
            }
        ],
        "images": [str(image_path)],
        "image_paths": [str(image_path)],
        "audit": {"evaluator_reward_excluded_from_messages": True},
    }
    dpo_path.write_text(json.dumps(dpo_record, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = qwen_dir / "qwen_tool_training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "flatdisk.qwen_tool_training_manifest.v1",
                "output_dir": str(qwen_dir),
                "manifest_path": str(manifest_path),
                "qwen_dpo_messages_jsonl": str(dpo_path),
                "dpo_preference_count": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return qwen_dir


def test_plan_qwen_dpo_training_writes_ready_job_and_trl_script(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path)

    job = plan_qwen_dpo_training(
        qwen_dir,
        output_dir=tmp_path / "dpo_job",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        max_steps=7,
    )

    assert job["schema"] == "flatdisk.qwen_dpo_training_job.v1"
    assert job["status"] == "ready"
    assert job["dataset"]["sample_count"] == 1
    assert job["dataset"]["image_reference_count"] == 1
    assert job["dataset"]["missing_image_count"] == 0
    assert job["dataset"]["forbidden_model_token_hits"] == {}
    assert job["training_args"]["max_steps"] == 7
    assert "trl" in job["required_packages"]
    assert "torchvision" in job["required_packages"]
    assert "accelerate launch" in job["launch_command"]
    assert job["launch_argv"][:3] == ["accelerate", "launch", str(tmp_path / "dpo_job" / "train_qwen_dpo_trl.py")]
    assert job["runtime"]["dependency_check"].startswith("importlib.util.find_spec")
    job_path = tmp_path / "dpo_job" / "qwen_dpo_training_job.json"
    train_script = tmp_path / "dpo_job" / "train_qwen_dpo_trl.py"
    assert job_path.exists()
    assert train_script.exists()
    assert len(job["train_script_sha256"]) == 64
    script_text = train_script.read_text(encoding="utf-8")
    assert "DPOTrainer" in script_text
    assert "AutoModelForImageTextToText" in script_text
    assert "Image.open" in script_text


def test_plan_qwen_dpo_training_blocks_forbidden_prompt_tokens(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path, prompt_text="Hidden distance_m should not be here.")

    job = plan_qwen_dpo_training(qwen_dir, output_dir=tmp_path / "dpo_job")

    assert job["status"] == "not_ready"
    assert job["dataset"]["forbidden_model_token_hits"] == {"distance_m": 1}
    assert any("forbidden privileged token" in blocker for blocker in job["blockers"])


def test_plan_qwen_dpo_training_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path, prompt_text="target_pose leak")
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-plan-qwen-dpo-training",
            "--input",
            str(qwen_dir),
            "--output-dir",
            str(tmp_path / "dpo_job"),
            "--fail-on-not-ready",
        ],
    )

    assert main() == 2
    job = json.loads(
        (tmp_path / "dpo_job" / "qwen_dpo_training_job.json").read_text(encoding="utf-8")
    )
    assert job["status"] == "not_ready"


def test_run_qwen_dpo_training_job_dry_run_writes_result(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path)
    plan_qwen_dpo_training(qwen_dir, output_dir=tmp_path / "dpo_job")

    result = run_qwen_dpo_training_job(tmp_path / "dpo_job", dry_run=True, check_dependencies=False)

    assert result["schema"] == "flatdisk.qwen_dpo_training_result.v1"
    assert result["status"] == "dry_run"
    assert result["returncode"] is None
    assert result["blockers"] == []
    assert result["dependency_check"]["enabled"] is False
    assert result["launch_argv"][0] == "accelerate"
    assert result["sample_count"] == 1
    result_path = tmp_path / "dpo_job" / "qwen_dpo_training_result.json"
    assert result_path.exists()


def test_run_qwen_dpo_training_job_reports_missing_dependencies(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path)
    plan_qwen_dpo_training(qwen_dir, output_dir=tmp_path / "dpo_job")
    job_path = tmp_path / "dpo_job" / "qwen_dpo_training_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = ["definitely_missing_flatdisk_training_package"]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_qwen_dpo_training_job(job_path, dry_run=True)

    assert result["status"] == "not_ready"
    assert result["dependency_check"]["missing_packages"] == ["definitely_missing_flatdisk_training_package"]
    assert any("missing required training package" in blocker for blocker in result["blockers"])


def test_run_qwen_dpo_training_job_executes_ready_job(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path)
    plan_qwen_dpo_training(qwen_dir, output_dir=tmp_path / "dpo_job")
    fake_train = tmp_path / "dpo_job" / "fake_train.py"
    fake_train.write_text("print('trained-ok')\n", encoding="utf-8")
    job_path = tmp_path / "dpo_job" / "qwen_dpo_training_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = []
    job["train_script"] = str(fake_train)
    job["launch_command"] = f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_train))}"
    job["launch_argv"] = [sys.executable, str(fake_train)]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_qwen_dpo_training_job(job_path)

    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert "trained-ok" in result["stdout_tail"]


def test_run_qwen_dpo_training_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    qwen_dir = _write_qwen_dpo_fixture(tmp_path)
    plan_qwen_dpo_training(qwen_dir, output_dir=tmp_path / "dpo_job")
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-run-qwen-dpo-training",
            "--job",
            str(tmp_path / "dpo_job"),
            "--dry-run",
            "--skip-dependency-check",
        ],
    )

    assert run_main() == 0
