from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

from flatdisk_sim.qwen_sft_likelihood import (
    module_main,
    plan_qwen_sft_likelihood_check,
    run_main,
    run_qwen_sft_likelihood_check,
)
from flatdisk_sim.qwen_sft_training import plan_qwen_sft_training


def _write_qwen_sft_fixture(
    tmp_path: Path,
    *,
    prompt_text: str = "Drive safely.",
    assistant_content: str | None = None,
    count: int = 2,
    conflicting_images: bool = False,
) -> Path:
    qwen_dir = tmp_path / "qwen_tool_training"
    qwen_dir.mkdir()
    image_path = qwen_dir / "frame.jpg"
    image_path.write_bytes(b"fake image")
    missing_image_path = qwen_dir / "missing.jpg"
    sft_path = qwen_dir / "qwen_sft_messages.jsonl"
    assistant_content = assistant_content or json.dumps(
        {
            "thought": "move a little",
            "action": {"tool": "drive_straight", "args": {"duration_s": 0.5, "power_percent": 18.0}},
            "memory_update": {"belief": "open path"},
        },
        sort_keys=True,
    )
    records = []
    for index in range(count):
        record = {
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
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [str(missing_image_path if conflicting_images else image_path)],
            "image_paths": [str(image_path)],
            "sft_weight": 1.0,
            "training_weight": 1.0,
        }
        records.append(record)
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


def _write_sft_job_with_adapter(tmp_path: Path, **fixture_kwargs) -> Path:
    qwen_dir = _write_qwen_sft_fixture(tmp_path, **fixture_kwargs)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job", max_samples=2, max_steps=1)
    adapter_dir = tmp_path / "sft_job" / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return tmp_path / "sft_job"


def test_plan_qwen_sft_likelihood_writes_ready_job_and_score_script(tmp_path: Path) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path)

    job = plan_qwen_sft_likelihood_check(
        sft_job,
        output_dir=tmp_path / "likelihood",
        max_samples=1,
    )

    assert job["schema"] == "flatdisk.qwen_sft_likelihood_job.v1"
    assert job["status"] == "ready"
    assert job["dataset"]["sample_count"] == 1
    assert job["dataset"]["source_sample_count"] == 2
    assert job["dataset"]["target_action_tool_counts"] == {"drive_straight": 1}
    assert job["dataset"]["missing_image_count"] == 0
    assert job["dataset"]["forbidden_model_token_hits"] == {}
    assert job["launch_argv"][:2] == ["python", str(tmp_path / "likelihood" / "score_qwen_sft_likelihood.py")]
    assert "--progress-log" in job["launch_argv"]
    assert (tmp_path / "likelihood" / "qwen_sft_likelihood_job.json").exists()
    assert (tmp_path / "likelihood" / "qwen_sft_likelihood_dataset.jsonl").exists()
    score_script = tmp_path / "likelihood" / "score_qwen_sft_likelihood.py"
    assert score_script.exists()
    assert len(job["score_script_sha256"]) == 64
    script_text = score_script.read_text(encoding="utf-8")
    assert "PeftModel.from_pretrained" in script_text
    assert "model.disable_adapter()" in script_text
    assert "target_start - 1" in script_text
    assert "target_text_sha256" in script_text
    assert "get_peft_model" not in script_text


def test_plan_qwen_sft_likelihood_uses_image_paths_before_images(tmp_path: Path) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path, conflicting_images=True)

    job = plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")

    assert job["status"] == "ready"
    assert job["dataset"]["missing_image_count"] == 0


def test_plan_qwen_sft_likelihood_blocks_missing_adapter_config(tmp_path: Path) -> None:
    qwen_dir = _write_qwen_sft_fixture(tmp_path)
    plan_qwen_sft_training(qwen_dir, output_dir=tmp_path / "sft_job", max_samples=1, max_steps=1)

    job = plan_qwen_sft_likelihood_check(tmp_path / "sft_job", output_dir=tmp_path / "likelihood")

    assert job["status"] == "not_ready"
    assert any("missing adapter_path" in blocker for blocker in job["blockers"])


def test_plan_qwen_sft_likelihood_blocks_forbidden_prompt_tokens(tmp_path: Path) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path, prompt_text="Hidden distance_m should not be here.")

    job = plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")

    assert job["status"] == "not_ready"
    assert job["dataset"]["forbidden_model_token_hits"] == {"distance_m": 2}
    assert any("forbidden privileged token" in blocker for blocker in job["blockers"])


def test_qwen_sft_likelihood_plan_cli_can_fail_on_not_ready(tmp_path: Path, monkeypatch) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path, prompt_text="target_pose leak")
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m flatdisk_sim.qwen_sft_likelihood",
            "--sft-training-job",
            str(sft_job),
            "--output-dir",
            str(tmp_path / "likelihood"),
            "--fail-on-not-ready",
        ],
    )

    assert module_main() == 2


def test_run_qwen_sft_likelihood_dry_run_writes_result(tmp_path: Path) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path)
    plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")

    result = run_qwen_sft_likelihood_check(tmp_path / "likelihood", dry_run=True, check_dependencies=False)

    assert result["schema"] == "flatdisk.qwen_sft_likelihood_result.v1"
    assert result["status"] == "dry_run"
    assert result["returncode"] is None
    assert result["blockers"] == []
    assert result["dependency_check"]["enabled"] is False
    assert result["launch_argv"][0] == "python"
    assert result["sample_count"] == 2
    assert result["likelihood_log_metrics"] == {}
    assert (tmp_path / "likelihood" / "qwen_sft_likelihood_result.json").exists()


def test_run_qwen_sft_likelihood_reports_missing_dependencies(tmp_path: Path) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path)
    plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")
    job_path = tmp_path / "likelihood" / "qwen_sft_likelihood_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = ["definitely_missing_flatdisk_likelihood_package"]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_qwen_sft_likelihood_check(job_path, dry_run=True)

    assert result["status"] == "not_ready"
    assert result["dependency_check"]["missing_packages"] == ["definitely_missing_flatdisk_likelihood_package"]
    assert any("missing required likelihood package" in blocker for blocker in result["blockers"])


def test_run_qwen_sft_likelihood_fake_run_attaches_metrics_and_progress(tmp_path: Path) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path)
    plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")
    fake_run = tmp_path / "likelihood" / "fake_score.py"
    likelihood_log = tmp_path / "likelihood" / "qwen_sft_likelihood_samples.jsonl"
    progress_log = tmp_path / "likelihood" / "qwen_sft_likelihood_progress.jsonl"
    fake_run.write_text(
        "from pathlib import Path\n"
        "import json\n"
        f"ll=Path({str(likelihood_log)!r})\n"
        f"pl=Path({str(progress_log)!r})\n"
        "rows=[\n"
        " {'schema':'flatdisk.qwen_sft_likelihood_sample.v1','target_action_tool':'drive_straight','target_mean_logprob_delta':0.5,'action_tool_mean_logprob_delta':0.25},\n"
        " {'schema':'flatdisk.qwen_sft_likelihood_sample.v1','target_action_tool':'wait','target_mean_logprob_delta':-0.1,'action_tool_mean_logprob_delta':None},\n"
        "]\n"
        "ll.write_text('\\n'.join(json.dumps(r) for r in rows)+'\\nnot-json\\n')\n"
        "pl.write_text(json.dumps({'stage':'sample_complete'})+'\\n')\n"
        "print('scored-ok')\n",
        encoding="utf-8",
    )
    job_path = tmp_path / "likelihood" / "qwen_sft_likelihood_job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job["required_packages"] = []
    job["score_script"] = str(fake_run)
    job["launch_command"] = f"{shlex.quote(sys.executable)} {shlex.quote(str(fake_run))}"
    job["launch_argv"] = [sys.executable, str(fake_run)]
    job_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = run_qwen_sft_likelihood_check(job_path)

    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert "scored-ok" in result["stdout_tail"]
    assert result["likelihood_log_sample_count"] == 2
    metrics = result["likelihood_log_metrics"]
    assert metrics["sample_count"] == 2
    assert metrics["malformed_line_count"] == 1
    assert metrics["target_mean_logprob_delta_mean"] == 0.2
    assert metrics["target_mean_logprob_improved_rate"] == 0.5
    assert metrics["action_tool_mean_logprob_delta_mean"] == 0.25
    assert metrics["action_tool_mean_logprob_improved_rate"] == 1.0
    assert metrics["target_action_tool_counts"] == {"drive_straight": 1, "wait": 1}
    assert result["progress_log_count"] == 1
    assert result["progress_log_tail"][-1]["stage"] == "sample_complete"


def test_qwen_sft_likelihood_run_cli_dry_run(tmp_path: Path, monkeypatch) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path)
    plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m flatdisk_sim.qwen_sft_likelihood",
            "run",
            "--job",
            str(tmp_path / "likelihood"),
            "--dry-run",
            "--skip-dependency-check",
        ],
    )

    assert module_main() == 0


def test_qwen_sft_likelihood_run_main_dry_run(tmp_path: Path, monkeypatch) -> None:
    sft_job = _write_sft_job_with_adapter(tmp_path)
    plan_qwen_sft_likelihood_check(sft_job, output_dir=tmp_path / "likelihood")
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-run-qwen-sft-likelihood",
            "--job",
            str(tmp_path / "likelihood"),
            "--dry-run",
            "--skip-dependency-check",
        ],
    )

    assert run_main() == 0
