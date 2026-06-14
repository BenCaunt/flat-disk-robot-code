from __future__ import annotations

import json
from pathlib import Path

from flatdisk_sim import runpod_launcher
from flatdisk_sim.runpod_launcher import RunpodLaunchSpec, build_runpodctl_command, compact_safe_id, parse_env_assignments, redacted_command, remote_worker_script


def test_build_runpodctl_command_contains_worker_env_and_docker_args() -> None:
    spec = RunpodLaunchSpec(
        task="AgentTask/qwen-topomap-memory-runpod-linux-v1-preflight",
        agent="agent-a",
        command_index=0,
        git_url="https://github.com/BenCaunt/flat-disk-robot-code.git",
        git_ref="abc123",
        env={"WH_INSTALL_CMD": "echo install wh"},
    )

    command = build_runpodctl_command(spec)

    assert command[:3] == ["runpodctl", "pod", "create"]
    assert "--image" in command
    assert "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404" in command
    env = json.loads(command[command.index("--env") + 1])
    assert env["TASK_ID"] == "AgentTask/qwen-topomap-memory-runpod-linux-v1-preflight"
    assert env["AGENT_NAME"] == "agent-a"
    assert env["COMMAND_INDEX"] == "0"
    assert env["GIT_URL"] == "https://github.com/BenCaunt/flat-disk-robot-code.git"
    assert env["START_THOR_XORG"] == "1"
    assert env["THOR_XORG_DISPLAY"] == "0"
    docker_args = command[command.index("--docker-args") + 1]
    assert "flatdisk-sim-research-warmhub" in docker_args
    assert "task-run-command" in docker_args
    assert "scripts/runpod_start_thor_xorg.sh" in docker_args
    assert "--complete-exit-code 2" in docker_args
    assert "git checkout abc123" in docker_args


def test_compact_safe_id_preserves_uniqueness_for_long_shared_prefixes() -> None:
    first = compact_safe_id("qwen-topomap-memory-runpod-linux-v1-run-qwen_topomap_memory_clip-bathroom_toilet", max_len=48)
    second = compact_safe_id("qwen-topomap-memory-runpod-linux-v1-run-qwen_topomap_memory_clip-bedroom_bed", max_len=48)

    assert first != second
    assert len(first) <= 48
    assert len(second) <= 48
    assert "bathroom_toilet" in first
    assert "bedroom_bed" in second


def test_start_qwen_server_adds_endpoint_env_and_bootstrap_script() -> None:
    spec = RunpodLaunchSpec(
        task="AgentTask/qwen-topomap-memory-runpod-linux-v1-preflight",
        agent="agent-a",
        git_url="https://github.com/BenCaunt/flat-disk-robot-code.git",
        start_qwen_server=True,
        qwen_server_timeout_s=1200,
        qwen_vllm_extra_args="--max-model-len 8192",
    )

    command = build_runpodctl_command(spec)
    env = json.loads(command[command.index("--env") + 1])
    docker_args = command[command.index("--docker-args") + 1]

    assert env["START_QWEN_SERVER"] == "1"
    assert env["QWEN_MODEL"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert env["QWEN_SERVER_TIMEOUT_S"] == "1200"
    assert env["QWEN_VLLM_EXTRA_ARGS"] == "--max-model-len 8192"
    assert "scripts/runpod_start_qwen_vllm.sh" in docker_args
    assert "scripts/runpod_start_thor_xorg.sh" in docker_args
    assert "finish_preclaimed_task" in docker_args
    assert "--no-claim" in docker_args
    assert "--evidence-artifact /workspace/qwen_vllm.log" in docker_args


def test_redacted_command_hides_sensitive_env_values() -> None:
    spec = RunpodLaunchSpec(
        task="AgentTask/example",
        agent="agent-a",
        git_url="https://github.com/BenCaunt/flat-disk-robot-code.git",
        env={"WARMHUB_API_KEY": "secret", "NORMAL_VALUE": "visible"},
    )

    command = redacted_command(build_runpodctl_command(spec))
    env = json.loads(command[command.index("--env") + 1])

    assert env["WARMHUB_API_KEY"] == "<redacted>"
    assert env["NORMAL_VALUE"] == "visible"


def test_parse_env_assignments_rejects_bad_values() -> None:
    assert parse_env_assignments(["A=B", "C=two=parts"]) == {"A": "B", "C": "two=parts"}
    try:
        parse_env_assignments(["missing-equals"])
    except ValueError as exc:
        assert "KEY=VALUE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_remote_worker_script_requires_wh_and_runs_selected_task_command() -> None:
    spec = RunpodLaunchSpec(
        task="AgentTask/example",
        agent="agent-a",
        command_index=2,
        git_url="https://github.com/BenCaunt/flat-disk-robot-code.git",
        evidence_artifacts=("/workspace/outputs/open_vocab_nav_research_loop",),
    )

    script = remote_worker_script(spec)

    assert "command -v wh" in script
    assert "--command-index \"$COMMAND_INDEX\"" in script
    assert "--evidence-artifact /workspace/outputs/open_vocab_nav_research_loop" in script
    assert "--log-file \"$LOG_FILE\"" in script


def test_remote_worker_script_can_run_all_task_commands() -> None:
    spec = RunpodLaunchSpec(
        task="AgentTask/fixtures",
        agent="agent-a",
        all_commands=True,
        git_url="https://github.com/BenCaunt/flat-disk-robot-code.git",
    )

    command = build_runpodctl_command(spec)
    env = json.loads(command[command.index("--env") + 1])
    script = remote_worker_script(spec)

    assert env["ALL_COMMANDS"] == "1"
    assert "--all-commands" in script
    assert "--command-index \"$COMMAND_INDEX\"" not in script


def test_main_dry_run_prints_runpodctl_command(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_launcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_launcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_launcher, "worktree_dirty", lambda _cwd: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-launch-task",
            "--task",
            "AgentTask/example",
            "--agent",
            "agent-a",
            "--env",
            "WARMHUB_API_KEY=secret",
        ],
    )

    assert runpod_launcher.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["launch"] is False
    assert payload["dirty_worktree"] is True
    command = payload["runpodctl_command"]
    assert command[:3] == ["runpodctl", "pod", "create"]
    env = json.loads(command[command.index("--env") + 1])
    assert env["WARMHUB_API_KEY"] == "<redacted>"


def test_main_launch_refuses_dirty_worktree(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runpod_launcher, "current_git_remote", lambda _cwd: "https://github.com/BenCaunt/flat-disk-robot-code.git")
    monkeypatch.setattr(runpod_launcher, "current_git_ref", lambda _cwd: "abc123")
    monkeypatch.setattr(runpod_launcher, "worktree_dirty", lambda _cwd: True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-runpod-launch-task",
            "--task",
            "AgentTask/example",
            "--agent",
            "agent-a",
            "--launch",
        ],
    )

    try:
        runpod_launcher.main()
    except SystemExit as exc:
        assert "dirty worktree" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
