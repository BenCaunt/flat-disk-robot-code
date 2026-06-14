"""Warmhub helpers for open-vocabulary navigation research agents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .evaluate_text_goals import default_episodes
from .llm_harness import POLICY_INPUT_ALLOWLIST
from .research_loop import (
    DEFAULT_WARMHUB_REPO,
    ROBOT_IO_CONTRACT,
    STRICT_MODEL_BASED_RUNNERS,
    PromptVariant,
    ResearchConfig,
    _safe_id,
    load_config,
    warmhub_shapes,
)


DEFAULT_EXPERIMENT_WREF = "NavExperiment/open_vocab_nav_qwen_prompt_sweep_v1"
TASK_STATUSES = ("planned", "running", "complete", "failed", "blocked")


def ensure_schema(repo: str) -> None:
    shapes = warmhub_shapes()
    for shape_name, spec in shapes.items():
        view = subprocess.run(["wh", "shape", "view", shape_name, "--repo", repo, "--json"], text=True, capture_output=True, check=False)
        fields_json = json.dumps(spec["fields"], sort_keys=True)
        description = spec["description"]
        if view.returncode == 0:
            current = _shape_view_data(view.stdout)
            if current.get("fields") == spec["fields"] and current.get("description") == description:
                continue
            subprocess.run(
                [
                    "wh",
                    "shape",
                    "revise",
                    shape_name,
                    "--repo",
                    repo,
                    "--fields",
                    fields_json,
                    "--description",
                    description,
                ],
                text=True,
                check=True,
            )
            continue
        subprocess.run(
            [
                "wh",
                "shape",
                "create",
                shape_name,
                "--repo",
                repo,
                "--fields",
                fields_json,
                "--description",
                description,
            ],
            text=True,
            check=True,
        )


def _shape_view_data(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    version = payload.get("version")
    if isinstance(version, dict) and isinstance(version.get("data"), dict):
        return version["data"]
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def commit_ops(repo: str, ops: list[dict[str, Any]], *, message: str) -> None:
    subprocess.run(
        [
            "wh",
            "commit",
            "submit",
            "--repo",
            repo,
            "--ops",
            json.dumps(ops, sort_keys=True),
            "--skip-existing",
            "--message",
            message,
        ],
        text=True,
        check=True,
    )


def make_agent_note_ops(
    *,
    about: str,
    author: str,
    note: str,
    tags: list[str],
    confidence: float,
    name: str | None = None,
) -> list[dict[str, Any]]:
    created_at = _now()
    note_name = name or f"{_safe_id(author)}-{int(time.time())}"
    return [
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"AgentNote/{_safe_id(note_name)}",
            "about": about,
            "data": {
                "author": author,
                "createdAt": created_at,
                "note": note,
                "tags": tags,
                "confidence": confidence,
            },
        }
    ]


def make_task_start_ops(
    *,
    task_id: str,
    objective: str,
    owner: str,
    tags: list[str],
    priority: str | None,
    related_experiment: str | None,
    notes: str | None,
) -> list[dict[str, Any]]:
    created_at = _now()
    data: dict[str, Any] = {
        "objective": objective,
        "status": "running",
        "owner": owner,
        "createdAt": created_at,
        "updatedAt": created_at,
        "tags": tags,
    }
    if priority:
        data["priority"] = priority
    if related_experiment:
        data["relatedExperiment"] = related_experiment
    if notes:
        data["notes"] = notes
    return [
        {
            "operation": "add",
            "kind": "thing",
            "name": f"AgentTask/{_safe_id(task_id)}",
            "data": data,
        }
    ]


def make_task_plan_ops(
    *,
    config_path: Path,
    output_dir: Path,
    plan_id: str | None,
    owner: str,
    priority: str | None,
    related_experiment: str | None,
    tags: list[str],
    include_slice_tasks: bool,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    plan_name = _safe_id(plan_id or f"{config.experiment_id}-plan-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}")
    experiment_ref = related_experiment or f"NavExperiment/{config.experiment_id}"
    ops: list[dict[str, Any]] = []
    if related_experiment is None or related_experiment == f"NavExperiment/{config.experiment_id}":
        ops.append(
            {
                "operation": "add",
                "kind": "thing",
                "name": f"NavExperiment/{config.experiment_id}",
                "data": {
                    "objective": config.objective,
                    "createdAt": _now(),
                    "robotIoContract": ROBOT_IO_CONTRACT,
                    "policyInputAllowlist": POLICY_INPUT_ALLOWLIST,
                    "noHardcodedLabelsOrColors": all(
                        variant.runner in STRICT_MODEL_BASED_RUNNERS and not variant.topomap_memory_allow_semantic_terms
                        for variant in config.variants
                    ),
                },
            }
        )
    common_tags = sorted(set(["open-vocab-nav", "research-loop", config.experiment_id, *tags]))
    preflight_task_id = f"{plan_name}-preflight"
    fixture_task_id = f"{plan_name}-topomap-fixtures"
    fixture_ref = _task_wref(fixture_task_id)
    preflight_ref = _task_wref(preflight_task_id)
    run_task_refs: list[str] = []
    if _config_uses_topomap_memory(config):
        ops.append(
            _planned_task_op(
                task_id=fixture_task_id,
                objective=f"Build or validate CLIP-backed topomap fixtures needed by {config.experiment_id}.",
                owner=owner,
                priority=priority,
                related_experiment=experiment_ref,
                tags=[*common_tags, "topomap-memory", "fixture"],
                notes={
                    "config_path": str(config_path),
                    "expected_map_dirs": _topomap_map_dirs(config),
                    "commands": _topomap_fixture_commands(config, config_path=config_path, output_dir=output_dir),
                    "constraints": [
                        "Policy-facing map query results must not expose poses, THOR object metadata, scene metadata, or hidden evaluator state.",
                        "Prefer CLIP-backed map query for strict sweeps; semantic-term mode is debug-only unless terms came from a real-robot-compatible process.",
                    ],
                },
            )
        )
    ops.append(
        _planned_task_op(
            task_id=preflight_task_id,
            objective=f"Preflight research config {config.experiment_id} before launching simulator work.",
            owner=owner,
            priority=priority,
            related_experiment=experiment_ref,
            tags=[*common_tags, "preflight"],
            prerequisites=[fixture_ref] if _config_uses_topomap_memory(config) else [],
            notes={
                "config_path": str(config_path),
                "commands": [
                    _with_topomap_ensure_commands(
                        _research_loop_command(
                            config_path=config_path,
                            output_dir=output_dir,
                            extra_args=["--preflight-only"],
                        ),
                        _topomap_ensure_commands(config),
                    )
                ],
                "success_criteria": [
                    "Qwen endpoint is reachable or failure is recorded as structured trial_exception.",
                    "Configured topomap maps and CLIP embeddings exist for topomap-memory variants.",
                    "No THOR trial is launched until preflight output is reviewed.",
                ],
            },
        )
    )
    if include_slice_tasks:
        for variant in config.variants:
            for episode in config.episodes:
                task_id = f"{plan_name}-run-{variant.name}-{episode}"
                run_task_refs.append(_task_wref(task_id))
                ops.append(
                    _planned_task_op(
                        task_id=task_id,
                        objective=f"Run one research slice: variant={variant.name}, episode={episode}.",
                        owner=owner,
                        priority=priority,
                        related_experiment=experiment_ref,
                        tags=[*common_tags, "trial-slice", variant.name, episode],
                        prerequisites=[preflight_ref],
                        notes={
                            "config_path": str(config_path),
                            "variant": variant.name,
                            "episode": episode,
                            "commands": [
                                _with_topomap_ensure_commands(
                                    _research_loop_command(
                                        config_path=config_path,
                                        output_dir=output_dir,
                                        extra_args=[
                                            "--variant",
                                            variant.name,
                                            "--episode",
                                            episode,
                                            "--preflight-endpoints",
                                            "--parallelism",
                                            "1",
                                        ],
                                    ),
                                    _topomap_ensure_commands_for_variant_episode(config, variant=variant, episode=episode),
                                )
                            ],
                            "artifact_expectations": [
                                "research_loop_summary.json",
                                "research_loop_report.md",
                                "warmhub_ops.json",
                                "policy camera/contact sheet artifacts if the trial executes",
                                "training_export artifacts for completed trials",
                            ],
                            "policy_constraints": _policy_constraints(),
                        },
                    )
            )
    else:
        run_sweep_task_id = f"{plan_name}-run-sweep"
        run_task_refs.append(_task_wref(run_sweep_task_id))
        ops.append(
            _planned_task_op(
                task_id=run_sweep_task_id,
                objective=f"Run the full research sweep for {config.experiment_id}.",
                owner=owner,
                priority=priority,
                related_experiment=experiment_ref,
                tags=[*common_tags, "sweep"],
                prerequisites=[preflight_ref],
                notes={
                    "config_path": str(config_path),
                    "commands": [
                        _with_topomap_ensure_commands(
                            _research_loop_command(
                                config_path=config_path,
                                output_dir=output_dir,
                                extra_args=["--preflight-endpoints", "--parallelism", str(config.parallelism)],
                            ),
                            _topomap_ensure_commands(config),
                        )
                    ],
                    "policy_constraints": _policy_constraints(),
                },
            )
        )
    ops.append(
        _planned_task_op(
            task_id=f"{plan_name}-failure-analysis",
            objective=f"Analyze completed {config.experiment_id} runs and propose the next general prompt/tool/model variant.",
            owner=owner,
            priority=priority,
            related_experiment=experiment_ref,
            tags=[*common_tags, "failure-analysis", "prompt-design"],
            prerequisites=run_task_refs,
            notes={
                "commands": [
                    (
                        "uv run --project sim flatdisk-sim-analyze-nav-failures "
                        f"--input {output_dir} "
                        f"--output-dir {output_dir / 'failure_analysis' / plan_name} "
                        f"--experiment-id {config.experiment_id} "
                        f"--about {experiment_ref} "
                        "--commit-warmhub"
                    )
                ],
                "warmhub_queries": [
                    f"wh thing query --repo {DEFAULT_WARMHUB_REPO} --shape NavEvalRun --limit 20 --json",
                    f"wh assertion list --repo {DEFAULT_WARMHUB_REPO} --shape FailureObservation --limit 20 --json",
                    f"wh thing search --repo {DEFAULT_WARMHUB_REPO} \"{config.experiment_id} qwen florence topomap failure\" --mode hybrid --json",
                ],
                "analysis_questions": [
                    "Did Qwen use visible landmarks or repeat final-goal servo without visual evidence?",
                    "Did topomap memory return a plausible route, and did Qwen use or ignore it?",
                    "Are failures due to endpoint/tool infrastructure, exploration strategy, grounding, or evaluator/label mismatch?",
                    "What next variant remains general and model-based without hard-coded names/colors?",
                ],
            },
        )
    )
    ops.append(
        _planned_task_op(
            task_id=f"{plan_name}-training-review",
            objective=f"Review training_export artifacts from {config.experiment_id} for SFT/GRPO/PPO readiness.",
            owner=owner,
            priority=priority,
            related_experiment=experiment_ref,
            tags=[*common_tags, "training-export", "grpo"],
            prerequisites=run_task_refs,
            notes={
                "expected_artifacts": [
                    "training_export/policy_steps.jsonl",
                    "training_export/episode_rollouts.jsonl",
                    "training_export/training_manifest.json",
                ],
                "checks": [
                    "Policy inputs contain no hidden target/object metadata.",
                    "Evaluator rewards are separated from policy inputs.",
                    "Successful and failed trajectories are both represented for ranking/filtering.",
                    "Candidate reward shaping is documented before using it for GRPO/PPO.",
                ],
            },
        )
    )
    return ops


def make_task_finish_ops(
    *,
    task: str,
    agent: str,
    status: str,
    summary: str,
    changed_files: list[str],
    evidence_artifacts: list[str],
    next_actions: list[str],
    confidence: float,
    result_id: str | None = None,
) -> list[dict[str, Any]]:
    created_at = _now()
    task_ref = _task_wref(task)
    result_name = result_id or f"{_safe_id(task_ref)}-{_safe_id(agent)}-{int(time.time())}"
    return [
        {
            "operation": "add",
            "kind": "assertion",
            "name": f"SubAgentResult/{_safe_id(result_name)}",
            "about": task_ref,
            "data": {
                "agent": agent,
                "status": status,
                "summary": summary,
                "changedFiles": changed_files,
                "evidenceArtifacts": evidence_artifacts,
                "nextActions": next_actions,
                "createdAt": created_at,
                "confidence": confidence,
            },
        }
    ]


def make_task_claim_revision_op(
    repo: str,
    task: str,
    *,
    owner: str,
    note: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    task_ref = _task_wref(task)
    payload = _read_warmhub_json(["wh", "thing", "view", task_ref, "--repo", repo, "--json"])
    data = dict(payload.get("data") or {})
    current_status = str(data.get("status") or "")
    if current_status != "planned" and not force:
        raise RuntimeError(f"{task_ref} has status {current_status!r}; use --force to claim a non-planned task")
    if not force:
        missing = incomplete_prerequisites(repo, _task_prerequisites(data))
        if missing:
            detail = ", ".join(f"{ref}={status or 'missing'}" for ref, status in missing.items())
            raise RuntimeError(f"{task_ref} has incomplete prerequisite(s): {detail}; use --force to claim anyway")
    claimed_at = _now()
    data["status"] = "running"
    data["owner"] = owner
    data["updatedAt"] = claimed_at
    if note:
        data["notes"] = _append_task_event_note(
            data.get("notes"),
            {
                "event": "claimed",
                "owner": owner,
                "createdAt": claimed_at,
                "note": note,
            },
        )
    return {
        "operation": "revise",
        "kind": "thing",
        "name": task_ref,
        "data": data,
    }


def task_command_payload(repo: str, task: str, *, command_index: int = 0) -> dict[str, Any]:
    task_ref = _task_wref(task)
    payload = _read_warmhub_json(["wh", "thing", "view", task_ref, "--repo", repo, "--json"])
    data = dict(payload.get("data") or {})
    notes = _parse_task_notes(data.get("notes"))
    commands = _validated_task_commands(task_ref, notes)
    if command_index < 0 or command_index >= len(commands):
        raise RuntimeError(f"{task_ref} has {len(commands)} command(s); index {command_index} is out of range")
    return {
        "task": task_ref,
        "data": data,
        "notes": notes,
        "command_index": command_index,
        "command": commands[command_index],
        "commands": commands,
    }


def _validated_task_commands(task_ref: str, notes: dict[str, Any]) -> list[str]:
    commands = notes.get("commands")
    if not isinstance(commands, list) or not commands:
        raise RuntimeError(f"{task_ref} does not contain notes.commands")
    validated: list[str] = []
    for index, command in enumerate(commands):
        if not isinstance(command, str) or not command.strip():
            raise RuntimeError(f"{task_ref} command {index} is empty or not a string")
        validated.append(command)
    return validated


def run_task_command(
    repo: str,
    task: str,
    *,
    agent: str,
    command_index: int = 0,
    cwd: Path | None = None,
    log_file: Path | None = None,
    evidence_artifacts: list[str] | None = None,
    no_claim: bool = False,
    force_claim: bool = False,
    dry_run: bool = False,
    timeout_s: float | None = None,
    complete_exit_codes: list[int] | None = None,
    all_commands: bool = False,
) -> int:
    payload = task_command_payload(repo, task, command_index=command_index)
    task_ref = str(payload["task"])
    commands = [str(command) for command in payload.get("commands", [])]
    selected_commands = list(enumerate(commands)) if all_commands else [(command_index, str(payload["command"]))]
    cwd = cwd or Path.cwd()
    artifacts = [*(evidence_artifacts or [])]
    accepted_codes = sorted(set(complete_exit_codes or [0]))
    if log_file is not None:
        artifacts.append(str(log_file))
    if dry_run:
        print(
            json.dumps(
                {
                    "task": task_ref,
                    "agent": agent,
                    "command_index": command_index,
                    "command": selected_commands[0][1] if len(selected_commands) == 1 else None,
                    "command_indices": [index for index, _command in selected_commands],
                    "commands": [command for _index, command in selected_commands],
                    "all_commands": all_commands,
                    "cwd": str(cwd),
                    "log_file": str(log_file) if log_file is not None else None,
                    "would_claim": not no_claim,
                    "evidence_artifacts": artifacts,
                    "complete_exit_codes": accepted_codes,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not no_claim:
        claim_op = make_task_claim_revision_op(
            repo,
            task_ref,
            owner=agent,
            note=(
                f"Running all {len(selected_commands)} notes.commands entries"
                if all_commands
                else f"Running notes.commands[{command_index}]"
            )
            + f" from {os.uname().nodename if hasattr(os, 'uname') else 'worker'}.",
            force=force_claim,
        )
        _maybe_commit(repo, [claim_op], dry_run=False, message="Claim navigation research agent task")
    started = time.perf_counter()
    status = "complete"
    returncode = 0
    summary_detail = ""
    command_results: list[dict[str, Any]] = []
    try:
        for selected_index, command in selected_commands:
            completed = _run_shell_task_command(
                task_ref=task_ref,
                command_index=selected_index,
                command=command,
                cwd=cwd,
                log_file=log_file,
                timeout_s=timeout_s,
            )
            returncode = int(completed.returncode)
            command_results.append({"commandIndex": selected_index, "returnCode": returncode})
            if returncode not in accepted_codes:
                status = "failed"
                summary_detail = f"notes.commands[{selected_index}] exited with code {returncode}."
                break
        else:
            accepted_nonzero = [result for result in command_results if int(result["returnCode"]) != 0]
            if len(selected_commands) == 1 and accepted_nonzero:
                summary_detail = f"Command exited with accepted code {accepted_nonzero[0]['returnCode']}."
            elif len(selected_commands) == 1:
                summary_detail = "Command completed successfully."
            elif accepted_nonzero:
                detail = ", ".join(f"{result['commandIndex']}={result['returnCode']}" for result in accepted_nonzero)
                summary_detail = f"All {len(command_results)} command(s) finished; accepted nonzero exit code(s): {detail}."
            else:
                summary_detail = f"All {len(command_results)} command(s) completed successfully."
    except subprocess.TimeoutExpired:
        status = "failed"
        returncode = 124
        summary_detail = f"Command timed out after {timeout_s:.1f}s." if timeout_s is not None else "Command timed out."
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as stream:
                stream.write(f"\n[timeout] {summary_detail}\n[end] {_now()}\n")
    elapsed_s = time.perf_counter() - started
    finish_ops = make_task_finish_ops(
        task=task_ref,
        agent=agent,
        status=status,
        summary=(
            f"Ran notes.commands[{command_index}] for {task_ref}. {summary_detail} "
            f"Elapsed {elapsed_s:.1f}s."
        ),
        changed_files=[],
        evidence_artifacts=artifacts,
        next_actions=["Inspect command logs and generated research artifacts."],
        confidence=0.75 if status == "complete" else 0.6,
    )
    finish_ops.insert(0, make_task_status_revision_op(repo, task_ref, status=status))
    _maybe_commit(repo, finish_ops, dry_run=False, message="Finish navigation research agent task")
    return 0 if status == "complete" else returncode


def _run_shell_task_command(
    *,
    task_ref: str,
    command_index: int,
    command: str,
    cwd: Path,
    log_file: Path | None,
    timeout_s: float | None,
) -> subprocess.CompletedProcess[str]:
    if log_file is None:
        return subprocess.run(command, shell=True, cwd=cwd, text=True, timeout=timeout_s, check=False)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as stream:
        stream.write(
            f"[task] {task_ref}\n"
            f"[command_index] {command_index}\n"
            f"[command] {command}\n"
            f"[cwd] {cwd}\n"
            f"[start] {_now()}\n"
        )
        stream.flush()
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            text=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        stream.write(f"\n[exit] {completed.returncode}\n[end] {_now()}\n")
        return completed


def warmhub_status_snapshot(
    repo: str,
    *,
    limit: int = 20,
    related_experiment: str | None = None,
) -> dict[str, Any]:
    tasks_by_status: dict[str, list[dict[str, Any]]] = {}
    task_query_limit = max(limit, 100)
    for status in TASK_STATUSES:
        payload = _query_agent_tasks(repo, status=status, limit=task_query_limit)
        tasks_by_status[status] = _filter_related_experiment(
            [_task_item_summary(item) for item in payload.get("items", [])],
            related_experiment=related_experiment,
        )
    run_payload = _query_things(repo, shape="NavEvalRun", limit=limit)
    runs = _filter_related_experiment(
        [_run_item_summary(item) for item in run_payload.get("items", [])],
        related_experiment=related_experiment,
    )
    failure_payload = _query_assertions(repo, shape="FailureObservation", limit=limit)
    failures = [_assertion_item_summary(item) for item in failure_payload.get("items", [])]
    result_payload = _query_assertions(repo, shape="SubAgentResult", limit=limit)
    results = [_assertion_item_summary(item) for item in result_payload.get("items", [])]
    return {
        "repo": repo,
        "related_experiment": related_experiment,
        "task_counts": {status: len(tasks_by_status[status]) for status in TASK_STATUSES},
        "tasks": tasks_by_status,
        "recent_runs": runs,
        "run_counts": {
            "total": len(runs),
            "success": sum(1 for run in runs if run.get("success") is True),
            "failed": sum(1 for run in runs if run.get("success") is False),
        },
        "recent_failures": failures,
        "recent_subagent_results": results,
        "next_actions": _status_next_actions(tasks_by_status, runs, failures),
    }


def _query_agent_tasks(repo: str, *, status: str, limit: int) -> dict[str, Any]:
    return _read_warmhub_json(
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


def _query_things(repo: str, *, shape: str, limit: int) -> dict[str, Any]:
    return _read_warmhub_json(["wh", "thing", "query", "--repo", repo, "--shape", shape, "--limit", str(max(1, limit)), "--json"])


def _query_assertions(repo: str, *, shape: str, limit: int) -> dict[str, Any]:
    return _read_warmhub_json(["wh", "assertion", "list", "--repo", repo, "--shape", shape, "--limit", str(max(1, limit)), "--json"])


def _task_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    prerequisites = _task_prerequisites(data)
    return {
        "wref": item.get("wref") or f"AgentTask/{item.get('name', '')}",
        "name": item.get("name"),
        "status": data.get("status"),
        "owner": data.get("owner"),
        "priority": data.get("priority"),
        "objective": data.get("objective"),
        "related_experiment": data.get("relatedExperiment"),
        "tags": data.get("tags") if isinstance(data.get("tags"), list) else [],
        "prerequisites": prerequisites,
        "updated_at": data.get("updatedAt"),
    }


def _run_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    return {
        "wref": item.get("wref") or f"NavEvalRun/{item.get('name', '')}",
        "name": item.get("name"),
        "experiment": data.get("experiment"),
        "variant": data.get("variant"),
        "episode_spec": data.get("episodeSpec"),
        "trial_id": data.get("trialId"),
        "runner": data.get("runner"),
        "model": data.get("model"),
        "success": data.get("success"),
        "final_distance_m": data.get("finalDistanceM"),
        "reason": data.get("reason"),
        "step_count": data.get("stepCount"),
        "output_dir": data.get("outputDir"),
    }


def _assertion_item_summary(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    return {
        "wref": item.get("wref"),
        "name": item.get("name"),
        "about": item.get("aboutWref") or item.get("about"),
        "status": data.get("status"),
        "agent": data.get("agent"),
        "summary": data.get("summary") or data.get("observation") or data.get("note"),
        "reason": data.get("reason"),
        "created_at": data.get("createdAt"),
        "confidence": data.get("confidence"),
    }


def _filter_related_experiment(items: list[dict[str, Any]], *, related_experiment: str | None) -> list[dict[str, Any]]:
    if not related_experiment:
        return items
    experiment_ref = related_experiment if related_experiment.startswith("NavExperiment/") else f"NavExperiment/{related_experiment}"
    versionless = experiment_ref.split("@", 1)[0]
    return [
        item
        for item in items
        if str(item.get("related_experiment") or item.get("experiment") or "").split("@", 1)[0] == versionless
    ]


def _status_next_actions(tasks_by_status: dict[str, list[dict[str, Any]]], runs: list[dict[str, Any]], failures: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    running = tasks_by_status.get("running", [])
    planned = tasks_by_status.get("planned", [])
    failed = tasks_by_status.get("failed", [])
    completed_wrefs = {_normalize_task_ref(str(task.get("wref") or "")) for task in tasks_by_status.get("complete", [])}
    ready_planned = [task for task in planned if _prerequisites_satisfied(task.get("prerequisites", []), completed_wrefs)]
    blocked_by_prereqs = [task for task in planned if task not in ready_planned]
    if running:
        actions.append(f"Inspect {len(running)} running AgentTask(s) before dispatching more workers.")
    if failed:
        actions.append(f"Review {len(failed)} failed AgentTask(s) and their SubAgentResult evidence.")
    preflight = [task for task in ready_planned if "preflight" in set(task.get("tags", [])) or str(task.get("name", "")).endswith("-preflight")]
    fixtures = [task for task in ready_planned if "fixture" in set(task.get("tags", []))]
    trial_slices = [task for task in ready_planned if "trial-slice" in set(task.get("tags", []))]
    if preflight:
        actions.append(f"Run planned preflight task next: {preflight[0].get('wref')}.")
    elif fixtures:
        actions.append(f"Build or validate topomap fixtures next: {fixtures[0].get('wref')}.")
    elif trial_slices:
        actions.append(f"Dispatch ready planned trial slices with flatdisk-sim-runpod-dispatch; {len(trial_slices)} visible in this page.")
    elif blocked_by_prereqs:
        actions.append(f"{len(blocked_by_prereqs)} planned task(s) are waiting on prerequisites; complete their prerequisite AgentTasks first.")
    if runs and failures:
        actions.append("Run or claim a failure-analysis AgentTask to turn completed failures into the next general strategy/tool variant.")
    elif runs:
        actions.append("Review training-export readiness once successful and failed rollouts are both present.")
    elif not planned and not running:
        actions.append("No active queue found; generate or plan a new strategy sweep.")
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_WARMHUB_REPO)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ensure-schema", help="Create missing Warmhub shapes for the research loop.")

    prime = subparsers.add_parser("prime", help="Print agent startup commands for the configured Warmhub repo.")
    prime.add_argument("--json", action="store_true", help="Emit JSON instead of shell text.")

    note = subparsers.add_parser("note", help="Write an AgentNote assertion.")
    note.add_argument("--about", default=DEFAULT_EXPERIMENT_WREF)
    note.add_argument("--author", required=True)
    note.add_argument("--note", required=True)
    note.add_argument("--tag", action="append", default=[])
    note.add_argument("--confidence", type=float, default=0.75)
    note.add_argument("--name", default=None)
    note.add_argument("--dry-run", action="store_true")

    start = subparsers.add_parser("task-start", help="Create a running AgentTask.")
    start.add_argument("--task-id", required=True)
    start.add_argument("--objective", required=True)
    start.add_argument("--owner", required=True)
    start.add_argument("--tag", action="append", default=[])
    start.add_argument("--priority", default=None)
    start.add_argument("--related-experiment", default=DEFAULT_EXPERIMENT_WREF)
    start.add_argument("--notes", default=None)
    start.add_argument("--dry-run", action="store_true")

    plan = subparsers.add_parser("task-plan-config", help="Create planned AgentTask records from a research-loop config.")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, default=Path("sim/scratch/open_vocab_nav_research_loop"))
    plan.add_argument("--plan-id", default=None)
    plan.add_argument("--owner", default="unassigned")
    plan.add_argument("--priority", default="normal")
    plan.add_argument("--related-experiment", default=None)
    plan.add_argument("--tag", action="append", default=[])
    plan.add_argument("--include-slice-tasks", action="store_true", help="Create one planned task per variant/episode slice.")
    plan.add_argument("--dry-run", action="store_true")

    claim = subparsers.add_parser("task-claim", help="Claim a planned AgentTask by revising it to running.")
    claim.add_argument("--task", required=True)
    claim.add_argument("--owner", required=True)
    claim.add_argument("--note", default=None)
    claim.add_argument("--force", action="store_true", help="Claim even if the task is not currently planned.")
    claim.add_argument("--dry-run", action="store_true")

    task_list = subparsers.add_parser("task-list", help="List AgentTask records for workers.")
    task_list.add_argument("--status", choices=("planned", "running", "complete", "blocked", "failed"), default="planned")
    task_list.add_argument("--limit", type=int, default=20)
    task_list.add_argument("--ready-only", action="store_true", help="For planned tasks, only show tasks whose prerequisites are complete.")
    task_list.add_argument("--json", action="store_true", help="Emit raw Warmhub JSON.")

    status = subparsers.add_parser("status", help="Summarize Warmhub queue, recent runs, failures, and recommended next actions.")
    status.add_argument("--limit", type=int, default=20)
    status.add_argument("--related-experiment", default=None, help="Optional NavExperiment id/wref filter.")
    status.add_argument("--json", action="store_true")

    run_command = subparsers.add_parser("task-run-command", help="Claim a task, run one notes.commands entry, and finish it.")
    run_command.add_argument("--task", required=True)
    run_command.add_argument("--agent", required=True)
    run_command.add_argument("--command-index", type=int, default=0)
    run_command.add_argument("--all-commands", action="store_true", help="Run every notes.commands entry before finishing the task.")
    run_command.add_argument("--cwd", type=Path, default=Path.cwd())
    run_command.add_argument("--log-file", type=Path, default=None)
    run_command.add_argument("--evidence-artifact", action="append", default=[])
    run_command.add_argument("--no-claim", action="store_true", help="Run without claiming first.")
    run_command.add_argument("--force-claim", action="store_true", help="Claim even if the task is already non-planned.")
    run_command.add_argument("--timeout-s", type=float, default=None)
    run_command.add_argument("--complete-exit-code", type=int, action="append", default=None, help="Exit code that should finish the AgentTask as complete. Defaults to 0; repeatable.")
    run_command.add_argument("--dry-run", action="store_true")

    finish = subparsers.add_parser("task-finish", help="Write a SubAgentResult assertion about an AgentTask.")
    finish.add_argument("--task", required=True)
    finish.add_argument("--agent", required=True)
    finish.add_argument("--status", choices=("complete", "blocked", "failed"), required=True)
    finish.add_argument("--summary", required=True)
    finish.add_argument("--changed-file", action="append", default=[])
    finish.add_argument("--evidence-artifact", action="append", default=[])
    finish.add_argument("--next-action", action="append", default=[])
    finish.add_argument("--confidence", type=float, default=0.75)
    finish.add_argument("--result-id", default=None)
    finish.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "ensure-schema":
        ensure_schema(args.repo)
        return 0
    if args.command == "prime":
        payload = _prime_payload(args.repo)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("\n".join(payload["commands"]))
        return 0
    if args.command == "note":
        ops = make_agent_note_ops(
            about=args.about,
            author=args.author,
            note=args.note,
            tags=args.tag,
            confidence=args.confidence,
            name=args.name,
        )
        return _maybe_commit(args.repo, ops, dry_run=args.dry_run, message="Log navigation research agent note")
    if args.command == "task-start":
        ops = make_task_start_ops(
            task_id=args.task_id,
            objective=args.objective,
            owner=args.owner,
            tags=args.tag,
            priority=args.priority,
            related_experiment=args.related_experiment,
            notes=args.notes,
        )
        return _maybe_commit(args.repo, ops, dry_run=args.dry_run, message="Start navigation research agent task")
    if args.command == "task-plan-config":
        ops = make_task_plan_ops(
            config_path=args.config,
            output_dir=args.output_dir,
            plan_id=args.plan_id,
            owner=args.owner,
            priority=args.priority,
            related_experiment=args.related_experiment,
            tags=args.tag,
            include_slice_tasks=args.include_slice_tasks,
        )
        return _maybe_commit(args.repo, ops, dry_run=args.dry_run, message="Plan navigation research agent tasks")
    if args.command == "task-claim":
        ops = [
            make_task_claim_revision_op(
                args.repo,
                args.task,
                owner=args.owner,
                note=args.note,
                force=args.force,
            )
        ]
        return _maybe_commit(args.repo, ops, dry_run=args.dry_run, message="Claim navigation research agent task")
    if args.command == "task-list":
        return _print_task_list(args.repo, status=args.status, limit=args.limit, raw_json=args.json, ready_only=args.ready_only)
    if args.command == "status":
        snapshot = warmhub_status_snapshot(args.repo, limit=args.limit, related_experiment=args.related_experiment)
        if args.json:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            print(_format_status_text(snapshot))
        return 0
    if args.command == "task-run-command":
        return run_task_command(
            args.repo,
            args.task,
            agent=args.agent,
            command_index=args.command_index,
            all_commands=args.all_commands,
            cwd=args.cwd,
            log_file=args.log_file,
            evidence_artifacts=args.evidence_artifact,
            no_claim=args.no_claim,
            force_claim=args.force_claim,
            dry_run=args.dry_run,
            timeout_s=args.timeout_s,
            complete_exit_codes=args.complete_exit_code,
        )
    if args.command == "task-finish":
        ops = make_task_finish_ops(
            task=args.task,
            agent=args.agent,
            status=args.status,
            summary=args.summary,
            changed_files=args.changed_file,
            evidence_artifacts=args.evidence_artifact,
            next_actions=args.next_action,
            confidence=args.confidence,
            result_id=args.result_id,
        )
        if not args.dry_run:
            ops.insert(0, make_task_status_revision_op(args.repo, args.task, status=args.status))
        return _maybe_commit(args.repo, ops, dry_run=args.dry_run, message="Finish navigation research agent task")
    raise ValueError(f"unsupported command: {args.command}")


def _maybe_commit(repo: str, ops: list[dict[str, Any]], *, dry_run: bool, message: str) -> int:
    if dry_run:
        print(json.dumps(ops, indent=2, sort_keys=True))
        return 0
    ensure_schema(repo)
    commit_ops(repo, ops, message=message)
    return 0


def _planned_task_op(
    *,
    task_id: str,
    objective: str,
    owner: str,
    priority: str | None,
    related_experiment: str | None,
    tags: list[str],
    notes: dict[str, Any],
    prerequisites: list[str] | None = None,
) -> dict[str, Any]:
    created_at = _now()
    prereq_refs = sorted({_normalize_task_ref(ref) for ref in prerequisites or [] if str(ref).strip()})
    if prereq_refs:
        notes = dict(notes)
        notes["prerequisites"] = prereq_refs
        notes["prerequisitePolicy"] = "all_complete"
    data: dict[str, Any] = {
        "objective": objective,
        "status": "planned",
        "owner": owner,
        "createdAt": created_at,
        "updatedAt": created_at,
        "tags": sorted(set(tags)),
        "notes": json.dumps(notes, indent=2, sort_keys=True),
    }
    if priority:
        data["priority"] = priority
    if related_experiment:
        data["relatedExperiment"] = related_experiment
    return {
        "operation": "add",
        "kind": "thing",
        "name": f"AgentTask/{_safe_id(task_id)}",
        "data": data,
    }


def _research_loop_command(*, config_path: Path, output_dir: Path, extra_args: list[str]) -> str:
    parts = [
        "uv",
        "run",
        "--project",
        "sim",
        "flatdisk-sim-research-loop",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    return " ".join(_shell_quote(part) for part in parts)


def _shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def _policy_constraints() -> list[str]:
    return [
        "No hard-coded object names, colors, or THOR object metadata in the policy path.",
        "Qwen sees only RGB frames/contact sheets, camera-derived summaries, IMU yaw, bounded tool results, and memory.",
        "Hidden THOR target distance/object metadata may be used only for evaluator scoring, debugging artifacts, and training labels.",
        "Florence/GroundingDINO phrase grounding and topomap memory are tool evidence, not proof of semantic goal completion.",
    ]


def _config_uses_topomap_memory(config: ResearchConfig) -> bool:
    return any(bool(variant.topomap_memory_map_dir) for variant in config.variants)


def _topomap_map_dirs(config: ResearchConfig) -> list[str]:
    paths: list[str] = []
    for variant in config.variants:
        if not variant.topomap_memory_map_dir:
            continue
        for episode in config.episodes:
            value = _resolve_topomap_template(variant.topomap_memory_map_dir, variant=variant, episode=episode)
            paths.append(value)
    return sorted(set(paths))


def _topomap_fixture_commands(config: ResearchConfig, *, config_path: Path, output_dir: Path) -> list[str]:
    commands: list[str] = []
    commands.extend(_topomap_ensure_commands(config))
    validation_specs: set[tuple[str, str]] = set()
    for variant in config.variants:
        if not variant.topomap_memory_map_dir:
            continue
        for episode in config.episodes:
            validation_specs.add((variant.name, episode))
    for variant_name, episode in sorted(validation_specs):
        commands.append(
            _research_loop_command(
                config_path=config_path,
                output_dir=output_dir,
                extra_args=["--variant", variant_name, "--episode", episode, "--parallelism", "1", "--preflight-only"],
            )
        )
    return commands


def _topomap_ensure_commands(config: ResearchConfig) -> list[str]:
    commands: dict[tuple[str, str, bool], str] = {}
    for variant in config.variants:
        if not variant.topomap_memory_map_dir:
            continue
        for episode in config.episodes:
            for command in _topomap_ensure_commands_for_variant_episode(config, variant=variant, episode=episode):
                map_dir = _resolve_topomap_template(variant.topomap_memory_map_dir, variant=variant, episode=episode)
                commands[(episode, map_dir, variant.topomap_memory_use_clip)] = command
    return [commands[key] for key in sorted(commands)]


def _topomap_ensure_commands_for_variant_episode(
    config: ResearchConfig,
    *,
    variant: PromptVariant,
    episode: str,
) -> list[str]:
    if not variant.topomap_memory_map_dir:
        return []
    episode_map = default_episodes()
    spec = episode_map[episode]
    map_dir = Path(_resolve_topomap_template(variant.topomap_memory_map_dir, variant=variant, episode=episode))
    build_command = _semantic_topomap_build_command(
        output_dir=map_dir,
        scene=spec.scene,
        render_width=config.render_width,
        render_height=config.render_height,
        use_clip=variant.topomap_memory_use_clip,
    )
    required = [map_dir / "semantic_topomap.json", map_dir / "descriptors.npy"]
    if variant.topomap_memory_use_clip:
        required.append(map_dir / "clip_image_embeddings.npy")
    checks = " && ".join(f"test -f {_shell_quote(str(path))}" for path in required)
    return [f"if ! ( {checks} ); then {build_command}; fi"]


def _with_topomap_ensure_commands(command: str, ensure_commands: list[str]) -> str:
    if not ensure_commands:
        return command
    return " && ".join([*ensure_commands, command])


def _semantic_topomap_build_command(
    *,
    output_dir: Path,
    scene: str,
    render_width: int,
    render_height: int,
    use_clip: bool,
) -> str:
    parts = ["uv", "run", "--project", "sim", "--extra", "thor"]
    if use_clip:
        parts.extend(["--with", "torch", "--with", "git+https://github.com/openai/CLIP.git"])
    parts.extend(
        [
            "flatdisk-sim-build-semantic-topomap",
            "--output-dir",
            str(output_dir),
            "--backend",
            "ithor",
            "--scene",
            scene,
            "--render-width",
            str(render_width),
            "--render-height",
            str(render_height),
            "--clean",
        ]
    )
    if use_clip:
        parts.append("--clip")
    return " ".join(_shell_quote(part) for part in parts)


def _resolve_topomap_template(template: str, *, variant: PromptVariant, episode: str) -> str:
    replacements = {
        "{episode}": episode,
        "{episode_name}": episode,
        "{variant}": variant.name,
        "{repetition}": "1",
    }
    value = template
    for token, replacement_value in replacements.items():
        value = value.replace(token, replacement_value)
    return value


def make_task_status_revision_op(repo: str, task: str, *, status: str) -> dict[str, Any]:
    task_ref = _task_wref(task)
    payload = _read_warmhub_json(["wh", "thing", "view", task_ref, "--repo", repo, "--json"])
    data = dict(payload.get("data") or {})
    data["status"] = status
    data["updatedAt"] = _now()
    return {
        "operation": "revise",
        "kind": "thing",
        "name": task_ref,
        "data": data,
    }


def _prime_payload(repo: str) -> dict[str, Any]:
    return {
        "repo": repo,
        "commands": [
            f"export WARMHUB_REPO={repo}",
            "wh prime",
            "wh repo describe --json",
            "uv run --project sim flatdisk-sim-research-warmhub task-list --status planned --ready-only --limit 20",
            "wh thing query --shape AgentTask --where status=running --limit 20 --json",
            "wh thing query --shape AgentTask --where status=planned --limit 20 --json",
            "wh thing query --shape NavEvalRun --limit 20 --json",
            "wh assertion list --shape FailureObservation --limit 20 --json",
            'wh thing search "current open vocabulary navigation failures qwen florence exploration" --mode hybrid --json',
        ],
    }


def _print_task_list(repo: str, *, status: str, limit: int, raw_json: bool, ready_only: bool = False) -> int:
    query_limit = max(1, limit)
    if ready_only and status == "planned":
        query_limit = max(query_limit * 10, 100)
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
            str(query_limit),
            "--json",
        ]
    )
    if ready_only:
        completed_wrefs = _completed_task_refs(repo, limit=max(limit, 500))
        payload = dict(payload)
        payload["items"] = [
            item
            for item in payload.get("items", [])
            if not isinstance(item, dict)
            or status != "planned"
            or _prerequisites_satisfied(_task_prerequisites(item.get("data") if isinstance(item.get("data"), dict) else {}), completed_wrefs)
        ][: max(1, limit)]
    if raw_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for item in payload.get("items", [])[: max(1, limit)]:
        data = item.get("data") or {}
        print(
            "\t".join(
                [
                    str(item.get("wref") or item.get("name") or ""),
                    str(data.get("status") or ""),
                    str(data.get("owner") or ""),
                    str(data.get("priority") or ""),
                    str(data.get("objective") or ""),
                ]
            )
        )
    return 0


def _completed_task_refs(repo: str, *, limit: int = 500) -> set[str]:
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


def _task_prerequisites(data: dict[str, Any]) -> list[str]:
    notes = _parse_task_notes(data.get("notes"))
    values = notes.get("prerequisites")
    if not isinstance(values, list):
        return []
    return sorted({_normalize_task_ref(str(value)) for value in values if str(value).strip()})


def incomplete_prerequisites(repo: str, prerequisites: list[str]) -> dict[str, str | None]:
    missing: dict[str, str | None] = {}
    for ref in sorted({_normalize_task_ref(prereq) for prereq in prerequisites}):
        if not ref:
            continue
        try:
            payload = _read_warmhub_json(["wh", "thing", "view", ref, "--repo", repo, "--json"])
        except RuntimeError:
            missing[ref] = None
            continue
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        status = str(data.get("status") or "")
        if status != "complete":
            missing[ref] = status or None
    return missing


def _prerequisites_satisfied(prerequisites: Any, completed_wrefs: set[str]) -> bool:
    if not isinstance(prerequisites, list):
        return True
    return all(_normalize_task_ref(str(ref)) in completed_wrefs for ref in prerequisites if str(ref).strip())


def _format_status_text(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Warmhub repo: {snapshot['repo']}",
    ]
    if snapshot.get("related_experiment"):
        lines.append(f"Experiment filter: {snapshot['related_experiment']}")
    counts = snapshot.get("task_counts", {})
    lines.extend(
        [
            "",
            "Task counts:",
            "  "
            + ", ".join(
                f"{status}={counts.get(status, 0)}"
                for status in TASK_STATUSES
            ),
            "",
            "Recent runs:",
        ]
    )
    run_counts = snapshot.get("run_counts", {})
    lines.append(f"  total={run_counts.get('total', 0)}, success={run_counts.get('success', 0)}, failed={run_counts.get('failed', 0)}")
    for run in snapshot.get("recent_runs", [])[:5]:
        lines.append(
            "  - "
            + " | ".join(
                str(part)
                for part in [
                    run.get("trial_id") or run.get("wref"),
                    run.get("variant"),
                    run.get("success"),
                    run.get("reason"),
                    run.get("final_distance_m"),
                ]
                if part is not None
            )
        )
    lines.append("")
    lines.append("Visible planned tasks:")
    completed_wrefs = {_normalize_task_ref(str(task.get("wref") or "")) for task in snapshot.get("tasks", {}).get("complete", [])}
    planned = sorted(
        snapshot.get("tasks", {}).get("planned", []),
        key=lambda task: (
            not _prerequisites_satisfied(task.get("prerequisites", []), completed_wrefs),
            str(task.get("name") or task.get("wref") or ""),
        ),
    )
    for task in planned[:8]:
        lines.append(f"  - {task.get('wref')} [{','.join(task.get('tags', [])[:4])}] {task.get('objective')}")
    if snapshot.get("recent_failures"):
        lines.append("")
        lines.append("Recent failures:")
        for failure in snapshot["recent_failures"][:5]:
            lines.append(f"  - {failure.get('about') or failure.get('wref')}: {failure.get('summary') or failure.get('reason')}")
    lines.append("")
    lines.append("Next actions:")
    for action in snapshot.get("next_actions", []):
        lines.append(f"  - {action}")
    return "\n".join(lines)


def _read_warmhub_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"command failed: {' '.join(command)}")
    return json.loads(completed.stdout)


def _append_task_event_note(existing_notes: Any, event: dict[str, Any]) -> str:
    payload = _parse_task_notes(existing_notes)
    history = payload.get("agentEvents")
    if not isinstance(history, list):
        history = []
    history.append(event)
    payload["agentEvents"] = history
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_task_notes(existing_notes: Any) -> dict[str, Any]:
    if isinstance(existing_notes, str) and existing_notes.strip():
        try:
            payload = json.loads(existing_notes)
            if isinstance(payload, dict):
                return dict(payload)
            return {"previousNotes": existing_notes}
        except json.JSONDecodeError:
            return {"previousNotes": existing_notes}
    if isinstance(existing_notes, dict):
        return dict(existing_notes)
    return {}


def _task_wref(task: str) -> str:
    return task if task.startswith("AgentTask/") else f"AgentTask/{_safe_id(task)}"


def _normalize_task_ref(task: str) -> str:
    text = str(task).strip()
    if not text:
        return ""
    if "@" in text:
        text = text.split("@", 1)[0]
    return _task_wref(text.replace("AgentTask/", "", 1)) if text.startswith("AgentTask/") else _task_wref(text)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
