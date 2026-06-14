"""Dispatch Warmhub-planned research tasks to Runpod workers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import time
from typing import Any

from .research_loop import DEFAULT_WARMHUB_REPO
from .research_warmhub import commit_ops, make_task_claim_revision_op, _normalize_task_ref, _parse_task_notes, _read_warmhub_json
from .runpod_launcher import (
    DEFAULT_GPU_ID,
    DEFAULT_IMAGE,
    DEFAULT_LOG_FILE,
    DEFAULT_PROJECT_DIR,
    DEFAULT_QWEN_HOST,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_PORT,
    DEFAULT_QWEN_SERVER_LOG,
    DEFAULT_QWEN_VLLM_EXTRA_ARGS,
    RunpodLaunchSpec,
    build_runpodctl_command,
    compact_safe_id,
    current_git_ref,
    current_git_remote,
    launch_with_runpodctl,
    parse_env_assignments,
    redacted_command,
    worktree_dirty,
)


TASK_STAGE_ORDER = (
    "fixture",
    "preflight",
    "trial-slice",
    "sweep",
    "promotion-gate",
    "failure-analysis",
    "training-review",
    "other",
)
TASK_STAGE_CHOICES = ("auto", "any", *TASK_STAGE_ORDER)


@dataclass(frozen=True)
class AgentTaskSummary:
    wref: str
    name: str
    status: str
    owner: str
    objective: str
    tags: tuple[str, ...]
    prerequisites: tuple[str, ...] = ()
    related_experiment: str | None = None

    def prerequisites_satisfied(self, completed_task_refs: set[str]) -> bool:
        return all(_normalize_task_ref(ref) in completed_task_refs for ref in self.prerequisites)


def query_agent_tasks(repo: str, *, status: str = "planned", limit: int = 20) -> list[AgentTaskSummary]:
    payload = _read_warmhub_json(
        [
            "wh",
            "thing",
            "query",
            "--repo",
            repo,
            "--shape",
            "AgentTask",
            "--where",
            f"status={status}",
            "--limit",
            str(max(1, limit)),
            "--json",
        ]
    )
    tasks: list[AgentTaskSummary] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        data = item.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        tags = data.get("tags") or []
        notes = _parse_task_notes(data.get("notes"))
        raw_prerequisites = notes.get("prerequisites") if isinstance(notes, dict) else []
        if not isinstance(raw_prerequisites, list):
            raw_prerequisites = []
        tasks.append(
            AgentTaskSummary(
                wref=str(item.get("wref") or f"AgentTask/{item.get('name', '')}"),
                name=str(item.get("name") or ""),
                status=str(data.get("status") or ""),
                owner=str(data.get("owner") or ""),
                objective=str(data.get("objective") or ""),
                tags=tuple(str(tag) for tag in tags if str(tag)),
                prerequisites=tuple(_normalize_task_ref(str(ref)) for ref in raw_prerequisites if str(ref).strip()),
                related_experiment=str(data["relatedExperiment"]) if data.get("relatedExperiment") else None,
            )
        )
    return tasks


def query_completed_task_refs(repo: str, *, limit: int = 500) -> set[str]:
    payload = _read_warmhub_json(
        [
            "wh",
            "thing",
            "query",
            "--repo",
            repo,
            "--shape",
            "AgentTask",
            "--where",
            "status=complete",
            "--limit",
            str(max(1, limit)),
            "--json",
        ]
    )
    refs: set[str] = set()
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        refs.add(_normalize_task_ref(str(item.get("wref") or f"AgentTask/{item.get('name', '')}")))
    return refs


def filter_tasks(
    tasks: list[AgentTaskSummary],
    *,
    name_prefix: str | None = None,
    tags: tuple[str, ...] = (),
    related_experiment: str | None = None,
    include_non_slice: bool = False,
    completed_task_refs: set[str] | None = None,
    respect_prerequisites: bool = True,
) -> list[AgentTaskSummary]:
    selected: list[AgentTaskSummary] = []
    required_tags = set(tags)
    completed = completed_task_refs or set()
    for task in tasks:
        if name_prefix and not task.name.startswith(name_prefix):
            continue
        if related_experiment and not _related_experiment_matches(task.related_experiment, related_experiment):
            continue
        if required_tags and not required_tags.issubset(set(task.tags)):
            continue
        if not include_non_slice and "trial-slice" not in task.tags:
            continue
        if respect_prerequisites and not task.prerequisites_satisfied(completed):
            continue
        selected.append(task)
    return selected


def _related_experiment_matches(task_experiment: str | None, requested: str) -> bool:
    return _normalize_experiment_ref(task_experiment) == _normalize_experiment_ref(requested)


def _normalize_experiment_ref(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("NavExperiment/"):
        text = text.split("/", 1)[1]
    return text.split("@", 1)[0]


def skipped_for_prerequisites(tasks: list[AgentTaskSummary], *, completed_task_refs: set[str]) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for task in tasks:
        missing = [ref for ref in task.prerequisites if _normalize_task_ref(ref) not in completed_task_refs]
        if not missing:
            continue
        skipped.append(
            {
                "task": task.wref,
                "name": task.name,
                "missing_prerequisites": missing,
            }
        )
    return skipped


def task_stage(task: AgentTaskSummary) -> str:
    tags = set(task.tags)
    name = task.name
    if "fixture" in tags:
        return "fixture"
    if "preflight" in tags or name.endswith("-preflight"):
        return "preflight"
    if "trial-slice" in tags:
        return "trial-slice"
    if "sweep" in tags:
        return "sweep"
    if "promotion-gate" in tags:
        return "promotion-gate"
    if "failure-analysis" in tags:
        return "failure-analysis"
    if "training-export" in tags or "training-review" in tags:
        return "training-review"
    return "other"


def select_tasks_for_dispatch(
    tasks: list[AgentTaskSummary],
    *,
    stage: str,
    max_workers: int,
) -> tuple[list[AgentTaskSummary], str | None]:
    limit = max(0, max_workers)
    if stage == "any":
        return tasks[:limit], "any" if tasks else None
    if stage == "auto":
        for candidate_stage in TASK_STAGE_ORDER:
            staged = [task for task in tasks if task_stage(task) == candidate_stage]
            if staged:
                return staged[:limit], candidate_stage
        return [], None
    staged = [task for task in tasks if task_stage(task) == stage]
    return staged[:limit], stage if staged else stage


def make_dispatch_specs(args: argparse.Namespace, tasks: list[AgentTaskSummary], *, git_url: str, git_ref: str | None) -> list[RunpodLaunchSpec]:
    env = parse_env_assignments(args.env)
    specs: list[RunpodLaunchSpec] = []
    for index, task in enumerate(tasks):
        agent = args.agent or f"{args.agent_prefix}-{compact_safe_id(task.name or task.wref, max_len=48)}"
        specs.append(
            RunpodLaunchSpec(
                task=task.wref,
                agent=agent,
                command_index=args.command_index,
                all_commands=args.all_commands,
                warmhub_repo=args.repo,
                git_url=git_url,
                git_ref=git_ref,
                project_dir=args.project_dir,
                image=args.image,
                gpu_id=args.gpu_id,
                gpu_count=args.gpu_count,
                name=args.name[index] if index < len(args.name) else None,
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
                env=env,
                start_qwen_server=args.start_qwen_server,
                qwen_model=args.qwen_model,
                qwen_host=args.qwen_host,
                qwen_port=args.qwen_port,
                qwen_server_log=args.qwen_server_log,
                qwen_server_timeout_s=args.qwen_server_timeout_s,
                qwen_vllm_package=args.qwen_vllm_package,
                qwen_vllm_extra_args=args.qwen_vllm_extra_args,
            )
        )
    return specs


def dispatch_payload(specs: list[RunpodLaunchSpec], *, launch: bool, dirty_worktree: bool) -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    for spec in specs:
        command = build_runpodctl_command(spec)
        redacted = redacted_command(command)
        workers.append(
            {
                "task": spec.task,
                "agent": spec.agent,
                "pod_name": spec.name or f"open-vocab-nav-{compact_safe_id(spec.task.replace('AgentTask/', ''), max_len=48)}",
                "runpodctl_command": redacted,
                "shell": " ".join(shlex.quote(part) for part in redacted),
            }
        )
    return {
        "dispatch": launch,
        "dirty_worktree": dirty_worktree,
        "worker_count": len(workers),
        "workers": workers,
    }


def annotate_dispatch_payload(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    git_url: str,
    git_ref: str | None,
    effective_stage: str,
    selected_stage: str | None,
    queried_task_count: int,
    selected_task_count: int,
    skipped_prerequisites: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    annotated = dict(payload)
    annotated.update(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "warmhub_repo": args.repo,
            "git_url": git_url,
            "git_ref": git_ref,
            "status_filter": args.status,
            "name_prefix_filter": args.name_prefix,
            "tag_filters": list(args.tag),
            "related_experiment_filter": args.related_experiment,
            "stage_filter": args.stage,
            "effective_stage_filter": effective_stage,
            "selected_stage": selected_stage,
            "include_non_slice": bool(args.include_non_slice),
            "ignore_prerequisites": bool(args.ignore_prerequisites),
            "queried_task_count": queried_task_count,
            "selected_task_count": selected_task_count,
        }
    )
    if skipped_prerequisites is not None:
        annotated["skipped_for_prerequisites_count"] = len(skipped_prerequisites)
        annotated["skipped_for_prerequisites"] = skipped_prerequisites[:20]
    return annotated


def write_dispatch_manifest(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def reserve_tasks_before_launch(repo: str, specs: list[RunpodLaunchSpec]) -> list[str]:
    reserved: list[str] = []
    for spec in specs:
        op = make_task_claim_revision_op(
            repo,
            spec.task,
            owner=spec.agent,
            note="Reserved by flatdisk-sim-runpod-dispatch before Runpod pod launch; worker command runs with --no-claim.",
        )
        commit_ops(repo, [op], message=f"Reserve navigation research task {spec.task} for Runpod launch")
        reserved.append(spec.task)
    return reserved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_WARMHUB_REPO, help="Warmhub repo.")
    parser.add_argument("--status", default="planned", choices=("planned", "running", "complete", "blocked", "failed"))
    parser.add_argument("--limit", type=int, default=200, help="Maximum Warmhub tasks to read before local filtering.")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum selected tasks to dispatch.")
    parser.add_argument("--name-prefix", default=None, help="Only dispatch tasks whose AgentTask name starts with this prefix.")
    parser.add_argument("--tag", action="append", default=[], help="Require a tag. Repeat for AND filtering.")
    parser.add_argument("--related-experiment", default=None)
    parser.add_argument(
        "--stage",
        choices=TASK_STAGE_CHOICES,
        default="trial-slice",
        help=(
            "Task stage to dispatch. Default preserves legacy trial-slice behavior; "
            "auto selects the first ready stage in queue order."
        ),
    )
    parser.add_argument("--include-non-slice", action="store_true", help="Allow non-trial-slice tasks such as preflight or analysis.")
    parser.add_argument("--ignore-prerequisites", action="store_true", help="Dispatch matching tasks even when prerequisite AgentTasks are incomplete.")
    parser.add_argument("--agent", default=None, help="Exact agent name to use for every launched task.")
    parser.add_argument("--agent-prefix", default="runpod-open-vocab-nav")
    parser.add_argument("--command-index", type=int, default=0)
    parser.add_argument("--all-commands", action="store_true", help="Run every notes.commands entry for selected tasks.")
    parser.add_argument("--git-url", default=None, help="Git URL the pod should clone. Defaults to origin.")
    parser.add_argument("--git-ref", default=None, help="Git ref/commit the pod should checkout. Defaults to current HEAD.")
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu-id", default=DEFAULT_GPU_ID)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--name", action="append", default=[], help="Optional pod name. Repeat to set names for selected tasks in order.")
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
    parser.add_argument("--start-qwen-server", action="store_true", help="Start a local vLLM Qwen endpoint before running each task command.")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-host", default=DEFAULT_QWEN_HOST)
    parser.add_argument("--qwen-port", type=int, default=DEFAULT_QWEN_PORT)
    parser.add_argument("--qwen-server-log", default=DEFAULT_QWEN_SERVER_LOG)
    parser.add_argument("--qwen-server-timeout-s", type=int, default=900)
    parser.add_argument("--qwen-vllm-package", default="vllm")
    parser.add_argument("--qwen-vllm-extra-args", default=DEFAULT_QWEN_VLLM_EXTRA_ARGS)
    parser.add_argument(
        "--dispatch-manifest",
        type=Path,
        default=None,
        help="Write the redacted dispatch payload to this JSON file for review or Warmhub evidence.",
    )
    parser.add_argument("--launch", action="store_true", help="Actually create Runpod pods. Default is dry-run.")
    parser.add_argument(
        "--no-reserve-before-launch",
        action="store_true",
        help="Skip parent-side Warmhub AgentTask reservation before pod creation. Intended only for manual recovery.",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="Allow launch even if the local worktree is dirty.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    dirty = worktree_dirty(cwd)
    git_url = args.git_url or current_git_remote(cwd)
    if not git_url:
        raise SystemExit("--git-url is required when origin remote is unavailable")
    git_ref = args.git_ref or current_git_ref(cwd)
    if args.launch and dirty and not args.allow_dirty:
        raise SystemExit("refusing to dispatch from a dirty worktree; commit/push worker code or pass --allow-dirty")

    queried = query_agent_tasks(args.repo, status=args.status, limit=args.limit)
    completed_task_refs = query_completed_task_refs(args.repo, limit=max(args.limit, 500)) if not args.ignore_prerequisites else set()
    effective_stage = "any" if args.include_non_slice and args.stage == "trial-slice" else args.stage
    include_non_slice = bool(args.include_non_slice or effective_stage != "trial-slice")
    matching = filter_tasks(
        queried,
        name_prefix=args.name_prefix,
        tags=tuple(args.tag),
        related_experiment=args.related_experiment,
        include_non_slice=include_non_slice,
        completed_task_refs=set(),
        respect_prerequisites=False,
    )
    ready = filter_tasks(
        matching,
        include_non_slice=True,
        completed_task_refs=completed_task_refs,
        respect_prerequisites=not args.ignore_prerequisites,
    )
    selected, selected_stage = select_tasks_for_dispatch(ready, stage=effective_stage, max_workers=args.max_workers)
    specs = make_dispatch_specs(args, selected, git_url=git_url, git_ref=git_ref)
    base_payload = dispatch_payload(specs, launch=args.launch, dirty_worktree=dirty)
    prerequisite_skips = None
    if not args.ignore_prerequisites:
        prerequisite_skips = skipped_for_prerequisites(matching, completed_task_refs=completed_task_refs)
    payload = annotate_dispatch_payload(
        base_payload,
        args=args,
        git_url=git_url,
        git_ref=git_ref,
        effective_stage=effective_stage,
        selected_stage=selected_stage,
        queried_task_count=len(queried),
        selected_task_count=len(selected),
        skipped_prerequisites=prerequisite_skips,
    )
    if args.dispatch_manifest is not None:
        payload["dispatch_manifest"] = str(args.dispatch_manifest)
        write_dispatch_manifest(payload, args.dispatch_manifest)
    if not args.launch:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not args.no_reserve_before_launch:
        reserved_tasks = reserve_tasks_before_launch(args.repo, specs)
        payload["reserved_task_count"] = len(reserved_tasks)
        payload["reserved_tasks"] = reserved_tasks

    failures = 0
    for spec in specs:
        failures += 1 if launch_with_runpodctl(build_runpodctl_command(spec)) != 0 else 0
    payload["failed_launches"] = failures
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
