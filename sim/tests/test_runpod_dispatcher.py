from __future__ import annotations

import json

from flatdisk_sim import runpod_dispatcher
from flatdisk_sim.runpod_dispatcher import AgentTaskSummary, filter_tasks


def test_filter_tasks_defaults_to_trial_slices_and_applies_tags() -> None:
    tasks = [
        AgentTaskSummary(
            wref="AgentTask/plan-run-a",
            name="plan-run-a",
            status="planned",
            owner="unassigned",
            objective="Run A",
            tags=("trial-slice", "runpod", "qwen"),
        ),
        AgentTaskSummary(
            wref="AgentTask/plan-preflight",
            name="plan-preflight",
            status="planned",
            owner="unassigned",
            objective="Preflight",
            tags=("runpod", "qwen"),
        ),
        AgentTaskSummary(
            wref="AgentTask/other-run",
            name="other-run",
            status="planned",
            owner="unassigned",
            objective="Run B",
            tags=("trial-slice", "local", "qwen"),
        ),
    ]

    selected = filter_tasks(tasks, name_prefix="plan-", tags=("runpod",))

    assert [task.wref for task in selected] == ["AgentTask/plan-run-a"]


def test_filter_tasks_skips_incomplete_prerequisites_by_default() -> None:
    tasks = [
        AgentTaskSummary(
            wref="AgentTask/plan-run-ready",
            name="plan-run-ready",
            status="planned",
            owner="unassigned",
            objective="Run ready",
            tags=("trial-slice", "runpod", "qwen"),
            prerequisites=("AgentTask/plan-preflight",),
        ),
        AgentTaskSummary(
            wref="AgentTask/plan-run-blocked",
            name="plan-run-blocked",
            status="planned",
            owner="unassigned",
            objective="Run blocked",
            tags=("trial-slice", "runpod", "qwen"),
            prerequisites=("AgentTask/plan-other-preflight",),
        ),
    ]

    selected = filter_tasks(tasks, name_prefix="plan-", tags=("runpod",), completed_task_refs={"AgentTask/plan-preflight"})

    assert [task.wref for task in selected] == ["AgentTask/plan-run-ready"]


def test_main_dry_run_dispatches_selected_runpod_workers(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_dispatcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_dispatcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_dispatcher, "worktree_dirty", lambda _cwd: True)
    monkeypatch.setattr(runpod_dispatcher, "query_completed_task_refs", lambda *_args, **_kwargs: {"AgentTask/plan-preflight"})
    monkeypatch.setattr(
        runpod_dispatcher,
        "query_agent_tasks",
        lambda *_args, **_kwargs: [
            AgentTaskSummary(
                wref="AgentTask/plan-run-a",
                name="plan-run-a",
                status="planned",
                owner="unassigned",
                objective="Run A",
                tags=("trial-slice", "runpod", "qwen"),
                prerequisites=("AgentTask/plan-preflight",),
            ),
            AgentTaskSummary(
                wref="AgentTask/plan-run-b",
                name="plan-run-b",
                status="planned",
                owner="unassigned",
                objective="Run B",
                tags=("trial-slice", "runpod", "qwen"),
                prerequisites=("AgentTask/plan-missing-preflight",),
            ),
            AgentTaskSummary(
                wref="AgentTask/plan-preflight",
                name="plan-preflight",
                status="planned",
                owner="unassigned",
                objective="Preflight",
                tags=("runpod", "qwen"),
            ),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-dispatch",
            "--name-prefix",
            "plan-",
            "--tag",
            "runpod",
            "--max-workers",
            "2",
            "--agent-prefix",
            "agent",
            "--start-qwen-server",
            "--qwen-vllm-extra-args",
            "--max-model-len 8192",
            "--env",
            "WARMHUB_API_KEY=secret",
            "--dispatch-manifest",
            str(tmp_path / "dispatch.json"),
        ],
    )

    assert runpod_dispatcher.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dispatch"] is False
    assert payload["dirty_worktree"] is True
    assert payload["queried_task_count"] == 3
    assert payload["selected_task_count"] == 1
    assert payload["worker_count"] == 1
    assert payload["skipped_for_prerequisites_count"] == 1
    assert payload["skipped_for_prerequisites"][0]["task"] == "AgentTask/plan-run-b"
    assert [worker["task"] for worker in payload["workers"]] == ["AgentTask/plan-run-a"]
    command = payload["workers"][0]["runpodctl_command"]
    env = json.loads(command[command.index("--env") + 1])
    assert env["START_QWEN_SERVER"] == "1"
    assert env["QWEN_VLLM_EXTRA_ARGS"] == "--max-model-len 8192"
    assert env["WARMHUB_API_KEY"] == "<redacted>"
    assert "--no-claim" in command[command.index("--docker-args") + 1]
    assert payload["dispatch_manifest"] == str(tmp_path / "dispatch.json")
    manifest = json.loads((tmp_path / "dispatch.json").read_text(encoding="utf-8"))
    assert manifest["warmhub_repo"] == "bencaunt-2/open-vocab-nav-research-loop"
    assert manifest["git_ref"] == "abc123"
    assert manifest["selected_task_count"] == 1
    assert manifest["workers"][0]["task"] == "AgentTask/plan-run-a"
    manifest_command = manifest["workers"][0]["runpodctl_command"]
    manifest_env = json.loads(manifest_command[manifest_command.index("--env") + 1])
    assert manifest_env["WARMHUB_API_KEY"] == "<redacted>"
    assert manifest["skipped_for_prerequisites"][0]["missing_prerequisites"] == ["AgentTask/plan-missing-preflight"]


def test_main_launch_refuses_dirty_worktree(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_dispatcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_dispatcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_dispatcher, "worktree_dirty", lambda _cwd: True)
    monkeypatch.setattr(runpod_dispatcher, "query_agent_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runpod_dispatcher, "query_completed_task_refs", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-dispatch",
            "--launch",
        ],
    )

    try:
        runpod_dispatcher.main()
    except SystemExit as exc:
        assert "dirty worktree" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
