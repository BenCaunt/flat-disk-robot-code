"""Launch Runpod workers for Warmhub-planned navigation research tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from .research_loop import DEFAULT_WARMHUB_REPO, _safe_id


DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
DEFAULT_GPU_ID = "NVIDIA GeForce RTX 4090"
DEFAULT_PROJECT_DIR = "/workspace/flat-disk-robot-code"
DEFAULT_LOG_FILE = "/workspace/open_vocab_nav_worker.log"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QWEN_HOST = "127.0.0.1"
DEFAULT_QWEN_PORT = 8000
DEFAULT_QWEN_SERVER_LOG = "/workspace/qwen_vllm.log"
DEFAULT_QWEN_VLLM_EXTRA_ARGS = "--max-model-len 16384"
WARMHUB_AUTH_ENV_NAMES = ("WH_TOKEN", "WARMHUB_TOKEN", "WARMHUB_API_KEY")


@dataclass(frozen=True)
class RunpodLaunchSpec:
    task: str
    agent: str
    command_index: int = 0
    all_commands: bool = False
    warmhub_repo: str = DEFAULT_WARMHUB_REPO
    git_url: str | None = None
    git_ref: str | None = None
    project_dir: str = DEFAULT_PROJECT_DIR
    image: str = DEFAULT_IMAGE
    gpu_id: str = DEFAULT_GPU_ID
    gpu_count: int = 1
    name: str | None = None
    cloud_type: str = "SECURE"
    container_disk_gb: int = 80
    volume_gb: int = 80
    volume_mount_path: str = "/workspace"
    ports: str = "22/tcp,8888/http,8000/http"
    stop_after: str | None = None
    terminate_after: str | None = None
    min_cuda_version: str | None = "12.8"
    log_file: str = DEFAULT_LOG_FILE
    task_timeout_s: float | None = None
    evidence_artifacts: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    start_qwen_server: bool = False
    start_thor_xorg: bool = True
    thor_xorg_display: int = 0
    qwen_model: str = DEFAULT_QWEN_MODEL
    qwen_host: str = DEFAULT_QWEN_HOST
    qwen_port: int = DEFAULT_QWEN_PORT
    qwen_server_log: str = DEFAULT_QWEN_SERVER_LOG
    qwen_server_timeout_s: int = 900
    qwen_vllm_package: str = "vllm"
    qwen_vllm_extra_args: str = DEFAULT_QWEN_VLLM_EXTRA_ARGS


def build_runpodctl_command(spec: RunpodLaunchSpec) -> list[str]:
    env = worker_env(spec)
    command = [
        "runpodctl",
        "pod",
        "create",
        "--image",
        spec.image,
        "--name",
        spec.name or _default_pod_name(spec),
        "--gpu-id",
        spec.gpu_id,
        "--gpu-count",
        str(max(1, spec.gpu_count)),
        "--cloud-type",
        spec.cloud_type,
        "--container-disk-in-gb",
        str(max(20, spec.container_disk_gb)),
        "--volume-in-gb",
        str(max(20, spec.volume_gb)),
        "--volume-mount-path",
        spec.volume_mount_path,
        "--ports",
        spec.ports,
        "--env",
        json.dumps(env, sort_keys=True),
        "--docker-args",
        docker_args(spec),
    ]
    if spec.min_cuda_version:
        command.extend(["--min-cuda-version", spec.min_cuda_version])
    if spec.stop_after:
        command.extend(["--stop-after", spec.stop_after])
    if spec.terminate_after:
        command.extend(["--terminate-after", spec.terminate_after])
    return command


def docker_args(spec: RunpodLaunchSpec) -> str:
    return "bash -lc " + shlex.quote(remote_worker_script(spec))


def remote_worker_script(spec: RunpodLaunchSpec) -> str:
    evidence_paths = [*spec.evidence_artifacts]
    if spec.start_qwen_server:
        evidence_paths.append(spec.qwen_server_log)
    evidence_args = " ".join(
        f"--evidence-artifact {shlex.quote(path)}" for path in evidence_paths
    )
    timeout_args = f" --timeout-s {float(spec.task_timeout_s):.3f}" if spec.task_timeout_s is not None else ""
    no_claim_args = " --no-claim" if spec.start_qwen_server else ""
    command_selection_args = "--all-commands" if spec.all_commands else '--command-index "$COMMAND_INDEX"'
    git_ref_block = ""
    if spec.git_ref:
        git_ref_block = f"""
git fetch --all --tags
git checkout {shlex.quote(spec.git_ref)}
"""
    thor_xorg_setup_block = """
if [[ "${START_THOR_XORG:-1}" == "1" ]]; then
  export DISPLAY="${DISPLAY:-:${THOR_XORG_DISPLAY:-0}}"
  chmod +x scripts/runpod_start_thor_xorg.sh
  scripts/runpod_start_thor_xorg.sh
fi
"""
    qwen_start_block = ""
    if spec.start_qwen_server:
        qwen_start_block = """
worker_task_claimed=0
worker_task_finished=0
finish_preclaimed_task() {
  code=$?
  if [[ "${worker_task_claimed}" == "1" && "${worker_task_finished}" != "1" ]]; then
    task_status="complete"
    if [[ "${code}" != "0" ]]; then
      task_status="failed"
    fi
    uv run --project sim flatdisk-sim-research-warmhub --repo "$WARMHUB_REPO" task-finish \\
      --task "$TASK_ID" \\
      --agent "$AGENT_NAME" \\
      --status "$task_status" \\
      --summary "Runpod worker exited during Qwen endpoint bootstrap or task setup with code ${code}." \\
      --evidence-artifact "$LOG_FILE" \\
      --evidence-artifact "$QWEN_SERVER_LOG" \\
      --confidence 0.6 || true
  fi
  exit "$code"
}
trap finish_preclaimed_task EXIT
uv run --project sim flatdisk-sim-research-warmhub --repo "$WARMHUB_REPO" task-claim \\
  --task "$TASK_ID" \\
  --owner "$AGENT_NAME" \\
  --note "Runpod worker claimed task before starting the local Qwen endpoint."
worker_task_claimed=1
""" + thor_xorg_setup_block + """
chmod +x scripts/runpod_start_qwen_vllm.sh
scripts/runpod_start_qwen_vllm.sh
"""
    object_drive_env_block = """
if [[ "${PREPARE_OBJECT_DRIVE_ENV:-1}" == "1" ]]; then
  chmod +x scripts/runpod_prepare_object_drive_env.sh
  source scripts/runpod_prepare_object_drive_env.sh
fi
"""
    warmhub_auth_block = """
if [[ -z "${WH_TOKEN:-}" ]]; then
  if [[ -n "${WARMHUB_TOKEN:-}" ]]; then
    export WH_TOKEN="$WARMHUB_TOKEN"
  elif [[ -n "${WARMHUB_API_KEY:-}" ]]; then
    export WH_TOKEN="$WARMHUB_API_KEY"
  fi
fi
if [[ -z "${WH_TOKEN:-}" ]]; then
  echo "[abort] missing WarmHub auth; pass --env WH_TOKEN=<pat> or an alias such as WARMHUB_TOKEN" >&2
  exit 2
fi
"""
    warmhub_smoke_block = """
if ! wh auth status >/tmp/warmhub_auth_status.txt 2>&1; then
  cat /tmp/warmhub_auth_status.txt >&2 || true
  echo "[abort] WarmHub auth check failed; verify the WH_TOKEN passed to the pod" >&2
  exit 2
fi
if ! wh repo describe "$WARMHUB_REPO" --json >/tmp/warmhub_repo_describe.json 2>/tmp/warmhub_repo_describe.err; then
  cat /tmp/warmhub_repo_describe.err >&2 || true
  echo "[abort] WarmHub repo check failed for ${WARMHUB_REPO}" >&2
  exit 2
fi
"""
    task_command = f"""uv run --project sim flatdisk-sim-research-warmhub --repo "$WARMHUB_REPO" task-run-command \\
  --task "$TASK_ID" \\
  --agent "$AGENT_NAME" \\
  {command_selection_args} \\
  --cwd {shlex.quote(spec.project_dir)} \\
  --log-file "$LOG_FILE" \\
  --complete-exit-code 0 \\
  --complete-exit-code 2 \\
  {evidence_args}{timeout_args}{no_claim_args}"""
    if spec.start_qwen_server:
        task_command_block = f"""echo "[command] running Warmhub task command"
set +e
{task_command}
task_code=$?
set -e
worker_task_finished=1
trap - EXIT
echo "[done] $(date -Iseconds)"
exit "${{task_code}}"
"""
    else:
        if spec.start_thor_xorg:
            task_command = f"{thor_xorg_setup_block}{task_command}"
        task_command_block = f"""echo "[command] running Warmhub task command"
{task_command}
echo "[done] $(date -Iseconds)"
"""
    return f"""set -Eeuo pipefail
echo "[start] $(date -Iseconds)"
python3 -m pip install --upgrade pip uv
if [[ ! -d {shlex.quote(spec.project_dir)}/.git ]]; then
  git clone "$GIT_URL" {shlex.quote(spec.project_dir)}
fi
cd {shlex.quote(spec.project_dir)}
{git_ref_block}if [[ -n "${{WH_INSTALL_CMD:-}}" ]]; then
  eval "$WH_INSTALL_CMD"
fi
if ! command -v wh >/dev/null 2>&1; then
  echo "[abort] wh CLI missing; use an image with wh or set WH_INSTALL_CMD" >&2
  exit 2
fi
{warmhub_auth_block}{warmhub_smoke_block}{qwen_start_block}{object_drive_env_block}{task_command_block}
"""


def worker_env(spec: RunpodLaunchSpec) -> dict[str, str]:
    if not spec.git_url:
        raise ValueError("git_url is required")
    env = {
        "TASK_ID": spec.task,
        "AGENT_NAME": spec.agent,
        "COMMAND_INDEX": str(spec.command_index),
        "ALL_COMMANDS": "1" if spec.all_commands else "0",
        "WARMHUB_REPO": spec.warmhub_repo,
        "GIT_URL": spec.git_url,
        "LOG_FILE": spec.log_file,
        "START_THOR_XORG": "1" if spec.start_thor_xorg else "0",
        "THOR_XORG_DISPLAY": str(spec.thor_xorg_display),
    }
    if spec.start_qwen_server:
        env.update(
            {
                "START_QWEN_SERVER": "1",
                "QWEN_MODEL": spec.qwen_model,
                "QWEN_HOST": spec.qwen_host,
                "QWEN_PORT": str(spec.qwen_port),
                "QWEN_SERVER_LOG": spec.qwen_server_log,
                "QWEN_SERVER_TIMEOUT_S": str(spec.qwen_server_timeout_s),
                "QWEN_VLLM_PACKAGE": spec.qwen_vllm_package,
            }
        )
        if spec.qwen_vllm_extra_args:
            env["QWEN_VLLM_EXTRA_ARGS"] = spec.qwen_vllm_extra_args
    env.update(normalize_warmhub_auth_env(spec.env))
    return env


def normalize_warmhub_auth_env(env: dict[str, str]) -> dict[str, str]:
    normalized = dict(env)
    if normalized.get("WH_TOKEN"):
        return normalized
    for alias in ("WARMHUB_TOKEN", "WARMHUB_API_KEY"):
        if normalized.get(alias):
            normalized["WH_TOKEN"] = normalized[alias]
            break
    return normalized


def has_warmhub_auth_env(env: dict[str, str]) -> bool:
    return any(bool(str(env.get(name) or "").strip()) for name in WARMHUB_AUTH_ENV_NAMES)


def spec_has_warmhub_auth_env(spec: RunpodLaunchSpec) -> bool:
    return has_warmhub_auth_env(spec.env)


def redacted_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next_env = False
    for item in command:
        if skip_next_env:
            redacted.append(json.dumps(redact_env(json.loads(item)), sort_keys=True))
            skip_next_env = False
            continue
        redacted.append(item)
        if item == "--env":
            skip_next_env = True
    return redacted


def redact_env(env: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in env.items():
        upper = str(key).upper()
        if any(token in upper for token in ("TOKEN", "SECRET", "PASSWORD", "KEY")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def launch_with_runpodctl(command: list[str]) -> int:
    return subprocess.run(command, text=True, check=False).returncode


def parse_env_assignments(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env expects KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--env key cannot be empty")
        result[key] = raw
    return result


def current_git_remote(cwd: Path) -> str | None:
    completed = subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def current_git_ref(cwd: Path) -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def worktree_dirty(cwd: Path) -> bool:
    completed = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, text=True, capture_output=True, check=False)
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _default_pod_name(spec: RunpodLaunchSpec) -> str:
    return f"open-vocab-nav-{compact_safe_id(spec.task.replace('AgentTask/', ''), max_len=48)}"


def compact_safe_id(value: str, *, max_len: int = 48) -> str:
    slug = _safe_id(value)
    if len(slug) <= max_len:
        return slug
    if max_len < 16:
        return hashlib.sha1(slug.encode("utf-8")).hexdigest()[:max_len]
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    tail_len = min(16, max_len - 10)
    head_len = max_len - tail_len - len(digest) - 2
    return f"{slug[:head_len]}-{slug[-tail_len:]}-{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Warmhub AgentTask id or wref.")
    parser.add_argument("--agent", required=True, help="Worker owner/agent name.")
    parser.add_argument("--command-index", type=int, default=0)
    parser.add_argument("--all-commands", action="store_true", help="Run every notes.commands entry before finishing the task.")
    parser.add_argument("--repo", default=DEFAULT_WARMHUB_REPO, help="Warmhub repo.")
    parser.add_argument("--git-url", default=None, help="Git URL the pod should clone. Defaults to origin.")
    parser.add_argument("--git-ref", default=None, help="Git ref/commit the pod should checkout. Defaults to current HEAD.")
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu-id", default=DEFAULT_GPU_ID)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--name", default=None)
    parser.add_argument("--cloud-type", default="SECURE")
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument("--volume-gb", type=int, default=80)
    parser.add_argument("--volume-mount-path", default="/workspace")
    parser.add_argument("--ports", default="22/tcp,8888/http,8000/http")
    parser.add_argument("--stop-after", default=None)
    parser.add_argument("--terminate-after", default=None)
    parser.add_argument("--min-cuda-version", default="12.8")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE)
    parser.add_argument("--task-timeout-s", type=float, default=None)
    parser.add_argument("--evidence-artifact", action="append", default=[])
    parser.add_argument("--env", action="append", default=[], help="Additional pod env as KEY=VALUE. Secrets are redacted in dry-run output.")
    parser.add_argument("--start-qwen-server", action="store_true", help="Start a local vLLM Qwen endpoint before running the task command.")
    parser.add_argument("--no-start-thor-xorg", action="store_true", help="Skip starting the AI2-THOR Xorg display before running the task command.")
    parser.add_argument("--thor-xorg-display", type=int, default=0)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-host", default=DEFAULT_QWEN_HOST)
    parser.add_argument("--qwen-port", type=int, default=DEFAULT_QWEN_PORT)
    parser.add_argument("--qwen-server-log", default=DEFAULT_QWEN_SERVER_LOG)
    parser.add_argument("--qwen-server-timeout-s", type=int, default=900)
    parser.add_argument("--qwen-vllm-package", default="vllm")
    parser.add_argument("--qwen-vllm-extra-args", default=DEFAULT_QWEN_VLLM_EXTRA_ARGS)
    parser.add_argument("--launch", action="store_true", help="Actually run runpodctl pod create. Default is dry-run.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow launch even if the local worktree is dirty.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    git_url = args.git_url or current_git_remote(cwd)
    if not git_url:
        raise SystemExit("--git-url is required when origin remote is unavailable")
    git_ref = args.git_ref or current_git_ref(cwd)
    if args.launch and worktree_dirty(cwd) and not args.allow_dirty:
        raise SystemExit("refusing to launch from a dirty worktree; commit/push the worker code or pass --allow-dirty")
    spec = RunpodLaunchSpec(
        task=args.task,
        agent=args.agent,
        command_index=args.command_index,
        all_commands=args.all_commands,
        warmhub_repo=args.repo,
        git_url=git_url,
        git_ref=git_ref,
        project_dir=args.project_dir,
        image=args.image,
        gpu_id=args.gpu_id,
        gpu_count=args.gpu_count,
        name=args.name,
        cloud_type=args.cloud_type,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
        volume_mount_path=args.volume_mount_path,
        ports=args.ports,
        stop_after=args.stop_after,
        terminate_after=args.terminate_after,
        min_cuda_version=args.min_cuda_version,
        log_file=args.log_file,
        task_timeout_s=args.task_timeout_s,
        evidence_artifacts=tuple(args.evidence_artifact),
        env=parse_env_assignments(args.env),
        start_qwen_server=args.start_qwen_server,
        start_thor_xorg=not args.no_start_thor_xorg,
        thor_xorg_display=args.thor_xorg_display,
        qwen_model=args.qwen_model,
        qwen_host=args.qwen_host,
        qwen_port=args.qwen_port,
        qwen_server_log=args.qwen_server_log,
        qwen_server_timeout_s=args.qwen_server_timeout_s,
        qwen_vllm_package=args.qwen_vllm_package,
        qwen_vllm_extra_args=args.qwen_vllm_extra_args,
    )
    if args.launch and not spec_has_warmhub_auth_env(spec):
        raise SystemExit(
            "refusing to launch Runpod pod because the remote worker lacks WarmHub auth; "
            "pass --env WH_TOKEN=<pat> or an alias such as WARMHUB_TOKEN"
        )
    command = build_runpodctl_command(spec)
    if not args.launch:
        print(
            json.dumps(
                {
                    "launch": False,
                    "dirty_worktree": worktree_dirty(cwd),
                    "runpodctl_command": redacted_command(command),
                    "shell": " ".join(shlex.quote(part) for part in redacted_command(command)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return launch_with_runpodctl(command)


if __name__ == "__main__":
    raise SystemExit(main())
