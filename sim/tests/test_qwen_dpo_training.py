from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim.qwen_dpo_training import main, plan_qwen_dpo_training


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
    assert "accelerate launch" in job["launch_command"]
    job_path = tmp_path / "dpo_job" / "qwen_dpo_training_job.json"
    train_script = tmp_path / "dpo_job" / "train_qwen_dpo_trl.py"
    assert job_path.exists()
    assert train_script.exists()
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
