from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace

from flatdisk_sim import runpod_dispatcher
from flatdisk_sim.runpod_dispatcher import (
    AgentTaskSummary,
    filter_tasks,
    make_dispatch_specs,
    query_queue_health,
    select_tasks_for_dispatch,
    task_stage,
)
from flatdisk_sim.runpod_launcher import build_runpodctl_command


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


def test_filter_tasks_matches_versioned_related_experiment_refs() -> None:
    tasks = [
        AgentTaskSummary(
            wref="AgentTask/plan-run-a",
            name="plan-run-a",
            status="planned",
            owner="unassigned",
            objective="Run A",
            tags=("trial-slice", "runpod", "qwen"),
            related_experiment="NavExperiment/open_vocab_nav_qwen_strategy_runpod_linux_v1@v1",
        ),
        AgentTaskSummary(
            wref="AgentTask/plan-run-b",
            name="plan-run-b",
            status="planned",
            owner="unassigned",
            objective="Run B",
            tags=("trial-slice", "runpod", "qwen"),
            related_experiment="NavExperiment/other_experiment@v1",
        ),
    ]

    selected = filter_tasks(
        tasks,
        tags=("runpod",),
        related_experiment="open_vocab_nav_qwen_strategy_runpod_linux_v1",
    )

    assert [task.wref for task in selected] == ["AgentTask/plan-run-a"]


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


def test_task_stage_classifies_preference_training_tasks() -> None:
    task = AgentTaskSummary(
        wref="AgentTask/plan-qwen-dpo-train-plan",
        name="plan-qwen-dpo-train-plan",
        status="planned",
        owner="unassigned",
        objective="Plan Qwen DPO training",
        tags=("preference-training", "qwen-dpo", "training-worker"),
    )

    assert task_stage(task) == "preference-training"


def test_preference_training_dispatch_can_skip_thor_xorg() -> None:
    task = AgentTaskSummary(
        wref="AgentTask/plan-qwen-dpo-train-worker",
        name="plan-qwen-dpo-train-worker",
        status="planned",
        owner="unassigned",
        objective="Run Qwen DPO training",
        tags=("gpu-training-worker", "preference-training", "qwen-dpo"),
    )
    args = SimpleNamespace(
        env=[],
        agent=None,
        agent_prefix="runpod-open-vocab-nav",
        command_index=0,
        all_commands=False,
        repo="bencaunt-2/open-vocab-nav-research-loop",
        project_dir="/workspace/flat-disk-robot-code",
        image=runpod_dispatcher.DEFAULT_IMAGE,
        gpu_id=runpod_dispatcher.DEFAULT_GPU_ID,
        gpu_count=1,
        name=[],
        cloud_type="SECURE",
        container_disk_gb=80,
        volume_gb=80,
        volume_mount_path="/workspace",
        ports="22/tcp,8888/http,8000/http",
        stop_after=None,
        terminate_after=None,
        min_cuda_version="12.8",
        log_file="/workspace/open_vocab_nav_worker.log",
        task_timeout_s=None,
        evidence_artifact=[],
        start_qwen_server=False,
        no_start_thor_xorg=True,
        qwen_model="Qwen/Qwen3-VL-8B-Instruct",
        qwen_host="127.0.0.1",
        qwen_port=8000,
        qwen_server_log="/workspace/qwen_vllm.log",
        qwen_server_timeout_s=900,
        qwen_vllm_package="vllm",
        qwen_vllm_extra_args="--max-model-len 16384",
    )

    specs = make_dispatch_specs(
        args,
        [task],
        git_url="https://github.com/BenCaunt/flat-disk-robot-code.git",
        git_ref="abc123",
    )
    command = build_runpodctl_command(specs[0])
    env = json.loads(command[command.index("--env") + 1])
    docker_args = command[command.index("--docker-args") + 1]

    assert specs[0].start_thor_xorg is False
    assert specs[0].start_qwen_server is False
    assert env["START_THOR_XORG"] == "0"
    assert "scripts/runpod_start_thor_xorg.sh" not in docker_args
    assert "scripts/runpod_start_qwen_vllm.sh" not in docker_args


def test_query_queue_health_summarizes_active_tasks(monkeypatch) -> None:
    def fake_query_agent_tasks(_repo, *, status, limit):  # noqa: ANN001
        assert limit == 100
        if status == "running":
            return [
                AgentTaskSummary(
                    wref="AgentTask/run-active-a",
                    name="run-active-a",
                    status="running",
                    owner="agent-a",
                    objective="Run active A",
                    tags=("trial-slice", "runpod"),
                    related_experiment="NavExperiment/exp@v1",
                    updated_at="2026-06-14T08:00:00Z",
                ),
                AgentTaskSummary(
                    wref="AgentTask/run-active-b",
                    name="run-active-b",
                    status="running",
                    owner="agent-b",
                    objective="Run active B",
                    tags=("trial-slice", "runpod"),
                    related_experiment="NavExperiment/exp@v1",
                    updated_at="2026-06-14T14:00:00Z",
                ),
                AgentTaskSummary(
                    wref="AgentTask/run-other",
                    name="run-other",
                    status="running",
                    owner="agent-c",
                    objective="Other experiment",
                    tags=("trial-slice",),
                    related_experiment="NavExperiment/other@v1",
                ),
            ]
        if status == "blocked":
            return [
                AgentTaskSummary(
                    wref="AgentTask/run-blocked",
                    name="run-blocked",
                    status="blocked",
                    owner="agent-d",
                    objective="Blocked run",
                    tags=("trial-slice", "runpod"),
                    prerequisites=("AgentTask/preflight",),
                    related_experiment="NavExperiment/exp@v1",
                )
            ]
        raise AssertionError(status)

    monkeypatch.setattr(runpod_dispatcher, "query_agent_tasks", fake_query_agent_tasks)
    now_s = datetime(2026, 6, 14, 15, tzinfo=timezone.utc).timestamp()

    health = query_queue_health(
        "repo/example",
        sample_limit=1,
        query_limit=100,
        related_experiment="exp",
        stale_running_after_s=4 * 60 * 60,
        now_s=now_s,
    )

    assert health["running_task_count"] == 2
    assert [task["task"] for task in health["running_tasks"]] == ["AgentTask/run-active-a"]
    assert health["running_tasks"][0]["updated_at"] == "2026-06-14T08:00:00Z"
    assert health["stale_running_task_count"] == 1
    assert health["stale_running_tasks"][0]["task"] == "AgentTask/run-active-a"
    assert health["stale_running_tasks"][0]["updated_age_s"] == 25200.0
    assert health["blocked_task_count"] == 1
    assert health["blocked_tasks"][0]["prerequisites"] == ["AgentTask/preflight"]
    assert "stale-looking running AgentTask(s)" in health["next_actions"][0]
    assert any("Inspect 2 running AgentTask(s)" in action for action in health["next_actions"])


def test_main_dry_run_dispatches_selected_runpod_workers(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_dispatcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_dispatcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_dispatcher, "worktree_dirty", lambda _cwd: True)
    monkeypatch.setattr(runpod_dispatcher, "query_completed_task_refs", lambda *_args, **_kwargs: {"AgentTask/plan-preflight"})
    def fake_query_agent_tasks(_repo, *, status, limit):  # noqa: ANN001
        if status == "planned":
            return [
                AgentTaskSummary(
                    wref="AgentTask/plan-run-a",
                    name="plan-run-a",
                    status="planned",
                    owner="unassigned",
                    objective="Run A",
                    tags=("trial-slice", "runpod", "qwen"),
                    prerequisites=("AgentTask/plan-preflight",),
                    related_experiment="NavExperiment/exp@v1",
                ),
                AgentTaskSummary(
                    wref="AgentTask/plan-run-b",
                    name="plan-run-b",
                    status="planned",
                    owner="unassigned",
                    objective="Run B",
                    tags=("trial-slice", "runpod", "qwen"),
                    prerequisites=("AgentTask/plan-missing-preflight",),
                    related_experiment="NavExperiment/exp@v1",
                ),
                AgentTaskSummary(
                    wref="AgentTask/plan-preflight",
                    name="plan-preflight",
                    status="planned",
                    owner="unassigned",
                    objective="Preflight",
                    tags=("runpod", "qwen"),
                    related_experiment="NavExperiment/exp@v1",
                ),
            ]
        if status == "running":
            return [
                AgentTaskSummary(
                    wref="AgentTask/active-run",
                    name="active-run",
                    status="running",
                    owner="runpod-agent",
                    objective="Active Runpod trial",
                    tags=("trial-slice", "runpod", "qwen"),
                    related_experiment="NavExperiment/exp@v1",
                )
            ]
        if status == "blocked":
            return [
                AgentTaskSummary(
                    wref="AgentTask/blocked-run",
                    name="blocked-run",
                    status="blocked",
                    owner="agent-blocked",
                    objective="Blocked trial",
                    tags=("trial-slice", "runpod", "qwen"),
                    related_experiment="NavExperiment/exp@v1",
                )
            ]
        return []

    monkeypatch.setattr(runpod_dispatcher, "query_agent_tasks", fake_query_agent_tasks)
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
            "--related-experiment",
            "exp",
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
    assert payload["queue_health"]["running_task_count"] == 1
    assert payload["queue_health"]["running_tasks"][0]["task"] == "AgentTask/active-run"
    assert payload["queue_health"]["blocked_task_count"] == 1
    assert payload["queue_health"]["blocked_tasks"][0]["task"] == "AgentTask/blocked-run"
    assert payload["skipped_for_prerequisites_count"] == 1
    assert payload["skipped_for_prerequisites"][0]["task"] == "AgentTask/plan-run-b"
    assert [worker["task"] for worker in payload["workers"]] == ["AgentTask/plan-run-a"]
    command = payload["workers"][0]["runpodctl_command"]
    env = json.loads(command[command.index("--env") + 1])
    assert env["START_QWEN_SERVER"] == "1"
    assert env["QWEN_VLLM_EXTRA_ARGS"] == "--max-model-len 8192"
    assert env["WARMHUB_API_KEY"] == "<redacted>"
    assert env["WH_TOKEN"] == "<redacted>"
    assert "--no-claim" in command[command.index("--docker-args") + 1]
    assert payload["dispatch_manifest"] == str(tmp_path / "dispatch.json")
    manifest = json.loads((tmp_path / "dispatch.json").read_text(encoding="utf-8"))
    assert manifest["warmhub_repo"] == "bencaunt-2/open-vocab-nav-research-loop"
    assert manifest["git_ref"] == "abc123"
    assert manifest["selected_stage"] == "trial-slice"
    assert manifest["selected_task_count"] == 1
    assert manifest["queue_health"]["running_tasks"][0]["owner"] == "runpod-agent"
    assert manifest["queue_health"]["blocked_tasks"][0]["owner"] == "agent-blocked"
    assert manifest["workers"][0]["task"] == "AgentTask/plan-run-a"
    manifest_command = manifest["workers"][0]["runpodctl_command"]
    manifest_env = json.loads(manifest_command[manifest_command.index("--env") + 1])
    assert manifest_env["WARMHUB_API_KEY"] == "<redacted>"
    assert manifest_env["WH_TOKEN"] == "<redacted>"
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

    def fake_check_runpod_auth():  # noqa: ANN202
        events.append("auth")

    monkeypatch.setattr(runpod_dispatcher, "check_runpod_auth", fake_check_runpod_auth)
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
            "--env",
            "WH_TOKEN=secret",
            "--max-workers",
            "1",
            "--launch",
        ],
    )

    assert runpod_dispatcher.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert events == ["auth", "reserve", "launch"]
    assert payload["reserved_task_count"] == 1
    assert payload["reserved_tasks"] == ["AgentTask/plan-run-a"]
    assert payload["failed_launches"] == 0


def test_main_launch_refuses_missing_worker_warmhub_auth_before_reserving(monkeypatch, tmp_path) -> None:
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

    def fake_check_runpod_auth():  # noqa: ANN202
        events.append("auth")

    def fake_reserve(_repo, _specs):  # noqa: ANN001, ANN202
        events.append("reserve")
        return []

    monkeypatch.setattr(runpod_dispatcher, "check_runpod_auth", fake_check_runpod_auth)
    monkeypatch.setattr(runpod_dispatcher, "reserve_tasks_before_launch", fake_reserve)
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

    try:
        runpod_dispatcher.main()
    except SystemExit as exc:
        assert "lack remote WarmHub auth" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    assert events == []


def test_main_launch_refuses_missing_runpod_auth_before_reserving(monkeypatch, tmp_path) -> None:
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

    def fake_check_runpod_auth():  # noqa: ANN202
        events.append("auth")
        raise SystemExit("missing runpod auth")

    def fake_reserve(_repo, _specs):  # noqa: ANN001, ANN202
        events.append("reserve")
        return []

    def fake_launch(_command):  # noqa: ANN001, ANN202
        events.append("launch")
        return 0

    monkeypatch.setattr(runpod_dispatcher, "check_runpod_auth", fake_check_runpod_auth)
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
            "--env",
            "WH_TOKEN=secret",
            "--max-workers",
            "1",
            "--launch",
        ],
    )

    try:
        runpod_dispatcher.main()
    except SystemExit as exc:
        assert "missing runpod auth" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
    assert events == ["auth"]


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
