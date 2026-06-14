from __future__ import annotations

import json
from types import SimpleNamespace

from flatdisk_sim import runpod_dispatcher
from flatdisk_sim.runpod_dispatcher import AgentTaskSummary, filter_tasks, select_tasks_for_dispatch, task_stage


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


def test_select_tasks_for_dispatch_auto_uses_ready_stage_order() -> None:
    tasks = [
        AgentTaskSummary(
            wref="AgentTask/plan-promotion-gate",
            name="plan-promotion-gate",
            status="planned",
            owner="unassigned",
            objective="Run promotion gate",
            tags=("promotion-gate", "baseline-preservation"),
        ),
        AgentTaskSummary(
            wref="AgentTask/plan-failure-analysis",
            name="plan-failure-analysis",
            status="planned",
            owner="unassigned",
            objective="Analyze failures",
            tags=("failure-analysis",),
        ),
    ]

    selected, stage = select_tasks_for_dispatch(tasks, stage="auto", max_workers=1)

    assert stage == "promotion-gate"
    assert [task.wref for task in selected] == ["AgentTask/plan-promotion-gate"]
    assert task_stage(tasks[1]) == "failure-analysis"


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
    assert payload["stage_filter"] == "trial-slice"
    assert payload["selected_stage"] == "trial-slice"
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
    assert manifest["selected_stage"] == "trial-slice"
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


def test_reserve_tasks_before_launch_claims_each_task(monkeypatch) -> None:
    claim_calls = []
    commit_calls = []

    def fake_claim(repo, task, *, owner, note):  # noqa: ANN001
        claim_calls.append((repo, task, owner, note))
        return {"operation": "revise", "name": task}

    def fake_commit(repo, ops, *, message):  # noqa: ANN001
        commit_calls.append((repo, ops, message))

    monkeypatch.setattr(runpod_dispatcher, "make_task_claim_revision_op", fake_claim)
    monkeypatch.setattr(runpod_dispatcher, "commit_ops", fake_commit)

    reserved = runpod_dispatcher.reserve_tasks_before_launch(
        "bencaunt-2/open-vocab-nav-research-loop",
        [
            SimpleNamespace(task="AgentTask/plan-run-a", agent="agent-a"),
            SimpleNamespace(task="AgentTask/plan-run-b", agent="agent-b"),
        ],
    )

    assert reserved == ["AgentTask/plan-run-a", "AgentTask/plan-run-b"]
    assert claim_calls[0][:3] == ("bencaunt-2/open-vocab-nav-research-loop", "AgentTask/plan-run-a", "agent-a")
    assert "before Runpod pod launch" in claim_calls[0][3]
    assert len(commit_calls) == 2
    assert commit_calls[1][1] == [{"operation": "revise", "name": "AgentTask/plan-run-b"}]


def test_main_launch_reserves_before_creating_pods(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    events = []
    monkeypatch.setattr(runpod_dispatcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_dispatcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_dispatcher, "worktree_dirty", lambda _cwd: False)
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
            )
        ],
    )

    def fake_reserve(_repo, specs):  # noqa: ANN001
        events.append("reserve")
        return [spec.task for spec in specs]

    def fake_launch(_command):  # noqa: ANN001
        events.append("launch")
        return 0

    monkeypatch.setattr(runpod_dispatcher, "reserve_tasks_before_launch", fake_reserve)
    monkeypatch.setattr(runpod_dispatcher, "launch_with_runpodctl", fake_launch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-dispatch",
            "--name-prefix",
            "plan-",
            "--tag",
            "runpod",
            "--max-workers",
            "1",
            "--launch",
        ],
    )

    assert runpod_dispatcher.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert events == ["reserve", "launch"]
    assert payload["reserved_task_count"] == 1
    assert payload["reserved_tasks"] == ["AgentTask/plan-run-a"]
    assert payload["failed_launches"] == 0


def test_main_auto_stage_dispatches_ready_promotion_gate(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_dispatcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_dispatcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_dispatcher, "worktree_dirty", lambda _cwd: False)
    monkeypatch.setattr(
        runpod_dispatcher,
        "query_completed_task_refs",
        lambda *_args, **_kwargs: {
            "AgentTask/plan-run-a",
            "AgentTask/plan-run-b",
        },
    )
    monkeypatch.setattr(
        runpod_dispatcher,
        "query_agent_tasks",
        lambda *_args, **_kwargs: [
            AgentTaskSummary(
                wref="AgentTask/plan-promotion-gate",
                name="plan-promotion-gate",
                status="planned",
                owner="unassigned",
                objective="Gate candidates",
                tags=("promotion-gate", "baseline-preservation"),
                prerequisites=("AgentTask/plan-run-a", "AgentTask/plan-run-b"),
            ),
            AgentTaskSummary(
                wref="AgentTask/plan-failure-analysis",
                name="plan-failure-analysis",
                status="planned",
                owner="unassigned",
                objective="Analyze",
                tags=("failure-analysis",),
                prerequisites=("AgentTask/plan-promotion-gate",),
            ),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-dispatch",
            "--stage",
            "auto",
            "--name-prefix",
            "plan-",
            "--max-workers",
            "2",
            "--agent-prefix",
            "agent",
        ],
    )

    assert runpod_dispatcher.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["stage_filter"] == "auto"
    assert payload["effective_stage_filter"] == "auto"
    assert payload["selected_stage"] == "promotion-gate"
    assert payload["selected_task_count"] == 1
    assert payload["workers"][0]["task"] == "AgentTask/plan-promotion-gate"
    assert payload["skipped_for_prerequisites"][0]["task"] == "AgentTask/plan-failure-analysis"


def test_main_include_non_slice_preserves_legacy_any_stage_selection(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_dispatcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_dispatcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_dispatcher, "worktree_dirty", lambda _cwd: False)
    monkeypatch.setattr(runpod_dispatcher, "query_completed_task_refs", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        runpod_dispatcher,
        "query_agent_tasks",
        lambda *_args, **_kwargs: [
            AgentTaskSummary(
                wref="AgentTask/plan-preflight",
                name="plan-preflight",
                status="planned",
                owner="unassigned",
                objective="Preflight",
                tags=("preflight",),
            ),
            AgentTaskSummary(
                wref="AgentTask/plan-run-a",
                name="plan-run-a",
                status="planned",
                owner="unassigned",
                objective="Run",
                tags=("trial-slice",),
            ),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-dispatch",
            "--include-non-slice",
            "--name-prefix",
            "plan-",
            "--max-workers",
            "2",
        ],
    )

    assert runpod_dispatcher.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["stage_filter"] == "trial-slice"
    assert payload["effective_stage_filter"] == "any"
    assert payload["selected_stage"] == "any"
    assert [worker["task"] for worker in payload["workers"]] == ["AgentTask/plan-preflight", "AgentTask/plan-run-a"]
