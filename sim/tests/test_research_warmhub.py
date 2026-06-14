from __future__ import annotations

import json
from pathlib import Path
import sys

from flatdisk_sim import research_warmhub
from flatdisk_sim.research_warmhub import (
    ensure_schema,
    make_agent_note_ops,
    make_task_claim_revision_op,
    make_task_finish_ops,
    make_task_plan_ops,
    make_task_start_ops,
    make_task_status_revision_op,
    run_task_command,
    task_command_payload,
)


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_ensure_schema_creates_missing_shape(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_warmhub_shapes():  # noqa: ANN202
        return {"Example": {"description": "Example shape.", "fields": {"name": "string"}}}

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(command)
        if command[:3] == ["wh", "shape", "view"]:
            return _Completed(returncode=1)
        return _Completed(returncode=0)

    monkeypatch.setattr(research_warmhub, "warmhub_shapes", fake_warmhub_shapes)
    monkeypatch.setattr(research_warmhub.subprocess, "run", fake_run)

    ensure_schema("org/repo")

    assert calls[0][:4] == ["wh", "shape", "view", "Example"]
    assert calls[1][:4] == ["wh", "shape", "create", "Example"]
    assert calls[1][calls[1].index("--fields") + 1] == '{"name": "string"}'


def test_ensure_schema_revises_stale_shape(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_warmhub_shapes():  # noqa: ANN202
        return {"Example": {"description": "New description.", "fields": {"name": "string", "score": "number"}}}

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(command)
        if command[:3] == ["wh", "shape", "view"]:
            return _Completed(
                returncode=0,
                stdout=json.dumps(
                    {
                        "version": {
                            "data": {
                                "description": "Old description.",
                                "fields": {"name": "string"},
                            }
                        }
                    }
                ),
            )
        return _Completed(returncode=0)

    monkeypatch.setattr(research_warmhub, "warmhub_shapes", fake_warmhub_shapes)
    monkeypatch.setattr(research_warmhub.subprocess, "run", fake_run)

    ensure_schema("org/repo")

    assert len(calls) == 2
    assert calls[1][:4] == ["wh", "shape", "revise", "Example"]
    assert calls[1][calls[1].index("--fields") + 1] == '{"name": "string", "score": "number"}'
    assert calls[1][calls[1].index("--description") + 1] == "New description."


def test_ensure_schema_noops_when_shape_matches(monkeypatch) -> None:
    calls: list[list[str]] = []
    fields = {"name": "string"}

    def fake_warmhub_shapes():  # noqa: ANN202
        return {"Example": {"description": "Example shape.", "fields": fields}}

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN202
        calls.append(command)
        return _Completed(
            returncode=0,
            stdout=json.dumps({"version": {"data": {"description": "Example shape.", "fields": fields}}}),
        )

    monkeypatch.setattr(research_warmhub, "warmhub_shapes", fake_warmhub_shapes)
    monkeypatch.setattr(research_warmhub.subprocess, "run", fake_run)

    ensure_schema("org/repo")

    assert len(calls) == 1
    assert calls[0][:4] == ["wh", "shape", "view", "Example"]


def test_agent_note_ops_target_experiment() -> None:
    ops = make_agent_note_ops(
        about="NavExperiment/example",
        author="agent-a",
        note="Started prompt sweep.",
        tags=["prompt", "qwen"],
        confidence=0.8,
        name="agent-a-start",
    )

    assert ops == [
        {
            "operation": "add",
            "kind": "assertion",
            "name": "AgentNote/agent-a-start",
            "about": "NavExperiment/example",
            "data": {
                "author": "agent-a",
                "createdAt": ops[0]["data"]["createdAt"],
                "note": "Started prompt sweep.",
                "tags": ["prompt", "qwen"],
                "confidence": 0.8,
            },
        }
    ]


def test_task_start_ops_create_running_task() -> None:
    ops = make_task_start_ops(
        task_id="run-qwen-sweep",
        objective="Run Qwen prompt sweep.",
        owner="agent-b",
        tags=["eval"],
        priority="high",
        related_experiment="NavExperiment/example",
        notes="Use THOR only.",
    )

    op = ops[0]
    assert op["name"] == "AgentTask/run-qwen-sweep"
    assert op["kind"] == "thing"
    assert op["data"]["status"] == "running"
    assert op["data"]["relatedExperiment"] == "NavExperiment/example"


def test_task_plan_config_creates_planned_slice_tasks(tmp_path) -> None:
    config_path = tmp_path / "research.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "test_open_vocab",
                "objective": "task planning",
                "episodes": ["living_room_sofa"],
                "variants": [
                    {"name": "qwen_baseline", "runner": "qwen"},
                    {
                        "name": "qwen_topomap",
                        "runner": "qwen",
                        "topomap_memory_map_dir": "sim/scratch/semantic_topomaps/{episode}_clip",
                        "topomap_memory_use_clip": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    ops = make_task_plan_ops(
        config_path=config_path,
        output_dir=Path("sim/scratch/open_vocab_nav_research_loop"),
        plan_id="plan-001",
        owner="unassigned",
        priority="normal",
        related_experiment=None,
        tags=["qwen"],
        include_slice_tasks=True,
    )

    names = {op["name"] for op in ops}
    assert "AgentTask/plan-001-preflight" in names
    assert "AgentTask/plan-001-topomap-fixtures" in names
    assert "AgentTask/plan-001-run-qwen_baseline-living_room_sofa" in names
    assert "AgentTask/plan-001-run-qwen_topomap-living_room_sofa" in names
    preflight = next(op for op in ops if op["name"] == "AgentTask/plan-001-preflight")
    assert preflight["data"]["status"] == "planned"
    assert preflight["data"]["relatedExperiment"] == "NavExperiment/test_open_vocab"
    preflight_notes = json.loads(preflight["data"]["notes"])
    assert preflight_notes["prerequisites"] == ["AgentTask/plan-001-topomap-fixtures"]
    assert preflight_notes["commands"][0].count("flatdisk-sim-build-semantic-topomap") == 1
    assert preflight_notes["commands"][0].startswith("if ! ( test -f sim/scratch/semantic_topomaps/living_room_sofa_clip/semantic_topomap.json")
    assert preflight_notes["commands"][0].endswith("--preflight-only")
    slice_task = next(op for op in ops if op["name"] == "AgentTask/plan-001-run-qwen_topomap-living_room_sofa")
    notes = json.loads(slice_task["data"]["notes"])
    assert notes["prerequisites"] == ["AgentTask/plan-001-preflight"]
    assert "--variant qwen_topomap" in notes["commands"][0]
    assert "--episode living_room_sofa" in notes["commands"][0]
    assert notes["commands"][0].count("flatdisk-sim-build-semantic-topomap") == 1
    assert notes["commands"][0].startswith("if ! ( test -f sim/scratch/semantic_topomaps/living_room_sofa_clip/semantic_topomap.json")
    baseline_task = next(op for op in ops if op["name"] == "AgentTask/plan-001-run-qwen_baseline-living_room_sofa")
    baseline_notes = json.loads(baseline_task["data"]["notes"])
    assert "flatdisk-sim-build-semantic-topomap" not in baseline_notes["commands"][0]
    fixture = next(op for op in ops if op["name"] == "AgentTask/plan-001-topomap-fixtures")
    fixture_notes = json.loads(fixture["data"]["notes"])
    assert fixture_notes["expected_map_dirs"] == ["sim/scratch/semantic_topomaps/living_room_sofa_clip"]
    assert "flatdisk-sim-build-semantic-topomap" in fixture_notes["commands"][0]
    assert fixture_notes["commands"][0].startswith("if ! ( test -f sim/scratch/semantic_topomaps/living_room_sofa_clip/semantic_topomap.json")
    assert "--scene FloorPlan201" in fixture_notes["commands"][0]
    assert "--clip" in fixture_notes["commands"][0]
    assert "--preflight-only" in fixture_notes["commands"][1]
    failure_analysis = next(op for op in ops if op["name"] == "AgentTask/plan-001-failure-analysis")
    analysis_notes = json.loads(failure_analysis["data"]["notes"])
    promotion_gate = next(op for op in ops if op["name"] == "AgentTask/plan-001-promotion-gate")
    promotion_notes = json.loads(promotion_gate["data"]["notes"])
    assert "AgentTask/plan-001-run-qwen_baseline-living_room_sofa" in promotion_notes["prerequisites"]
    assert "AgentTask/plan-001-run-qwen_topomap-living_room_sofa" in promotion_notes["prerequisites"]
    assert promotion_notes["baseline_variant"] == "qwen_baseline"
    assert promotion_notes["candidate_variants"] == ["qwen_topomap"]
    assert promotion_notes["accepted_exit_codes"] == [0, 2]
    assert promotion_notes["commands"][0].startswith("uv run --project sim flatdisk-sim-nav-promotion-gate")
    assert "--baseline-variant qwen_baseline" in promotion_notes["commands"][0]
    assert "--candidate-variant qwen_topomap" in promotion_notes["commands"][0]
    assert "--commit-warmhub" in promotion_notes["commands"][0]
    assert "--fail-on-reject" in promotion_notes["commands"][0]
    assert analysis_notes["prerequisites"] == ["AgentTask/plan-001-promotion-gate"]
    assert analysis_notes["commands"][0].startswith("uv run --project sim flatdisk-sim-analyze-nav-failures")
    assert "--input sim/scratch/open_vocab_nav_research_loop" in analysis_notes["commands"][0]
    assert "--commit-warmhub" in analysis_notes["commands"][0]
    training_review = next(op for op in ops if op["name"] == "AgentTask/plan-001-training-review")
    training_notes = json.loads(training_review["data"]["notes"])
    assert training_notes["commands"][0].startswith("uv run --project sim flatdisk-sim-nav-training-readiness")
    assert "--input sim/scratch/open_vocab_nav_research_loop" in training_notes["commands"][0]
    assert "--commit-warmhub" in training_notes["commands"][0]
    assert "training_readiness/training_readiness.json" in training_notes["expected_artifacts"]


def test_task_finish_ops_write_subagent_result() -> None:
    ops = make_task_finish_ops(
        task="run-qwen-sweep",
        agent="agent-b",
        status="complete",
        summary="Finished dry run.",
        changed_files=["sim/src/flatdisk_sim/research_loop.py"],
        evidence_artifacts=["sim/scratch/run/research_loop_summary.json"],
        next_actions=["Run live Qwen sweep."],
        confidence=0.9,
        result_id="run-qwen-sweep-result",
    )

    op = ops[0]
    assert op["name"] == "SubAgentResult/run-qwen-sweep-result"
    assert op["about"] == "AgentTask/run-qwen-sweep"
    assert op["data"]["status"] == "complete"
    assert op["data"]["changedFiles"] == ["sim/src/flatdisk_sim/research_loop.py"]


def test_task_claim_revision_op_preserves_planned_task_data(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = (
            '{"data":{"objective":"Run slice","status":"planned","owner":"unassigned",'
            '"createdAt":"2026-06-13T00:00:00Z","priority":"normal",'
            '"tags":["qwen"],"notes":"{\\"commands\\":[\\"run it\\"]}"}}'
        )
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    op = make_task_claim_revision_op("repo/example", "run-slice", owner="agent-c")

    assert op["operation"] == "revise"
    assert op["name"] == "AgentTask/run-slice"
    assert op["data"]["objective"] == "Run slice"
    assert op["data"]["status"] == "running"
    assert op["data"]["owner"] == "agent-c"
    assert op["data"]["priority"] == "normal"
    assert op["data"]["tags"] == ["qwen"]
    assert json.loads(op["data"]["notes"]) == {"commands": ["run it"]}
    assert "updatedAt" in op["data"]


def test_task_claim_revision_op_appends_claim_note(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"data":{"objective":"Preflight","status":"planned","owner":"unassigned","notes":"{\\"commands\\":[]}"}}'
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    op = make_task_claim_revision_op("repo/example", "preflight", owner="agent-c", note="Starting endpoint check.")

    notes = json.loads(op["data"]["notes"])
    assert notes["commands"] == []
    assert notes["agentEvents"][0]["event"] == "claimed"
    assert notes["agentEvents"][0]["owner"] == "agent-c"
    assert notes["agentEvents"][0]["note"] == "Starting endpoint check."


def test_task_claim_revision_op_rejects_incomplete_prerequisites(monkeypatch) -> None:
    responses = iter(
        [
            (
                '{"data":{"objective":"Run slice","status":"planned","owner":"unassigned",'
                '"notes":"{\\"commands\\":[\\"run it\\"],\\"prerequisites\\":[\\"AgentTask/preflight\\"]}"}}'
            ),
            '{"data":{"objective":"Preflight","status":"running","owner":"agent-a"}}',
        ]
    )

    class Completed:
        returncode = 0
        stderr = ""

        @property
        def stdout(self) -> str:
            return next(responses)

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    try:
        make_task_claim_revision_op("repo/example", "run-slice", owner="agent-c")
    except RuntimeError as exc:
        assert "incomplete prerequisite" in str(exc)
        assert "AgentTask/preflight=running" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_task_claim_revision_op_rejects_non_planned_task(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"data":{"objective":"Run slice","status":"running","owner":"agent-b"}}'
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    try:
        make_task_claim_revision_op("repo/example", "run-slice", owner="agent-c")
    except RuntimeError as exc:
        assert "use --force" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_task_claim_revision_op_force_allows_non_planned_task(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"data":{"objective":"Run slice","status":"running","owner":"stale-agent"}}'
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    op = make_task_claim_revision_op("repo/example", "run-slice", owner="agent-c", force=True)

    assert op["data"]["status"] == "running"
    assert op["data"]["owner"] == "agent-c"


def test_task_command_payload_reads_selected_notes_command(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = (
            '{"data":{"objective":"Build fixtures","status":"planned","owner":"unassigned",'
            '"notes":"{\\"commands\\":[\\"echo first\\",\\"echo second\\"],\\"expected_map_dirs\\":[\\"maps/a\\"]}"}}'
        )
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    payload = task_command_payload("repo/example", "fixtures", command_index=1)

    assert payload["task"] == "AgentTask/fixtures"
    assert payload["command"] == "echo second"
    assert payload["notes"]["expected_map_dirs"] == ["maps/a"]


def test_task_command_payload_rejects_missing_command(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"data":{"objective":"No command","status":"planned","owner":"unassigned","notes":"{}"}}'
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    try:
        task_command_payload("repo/example", "no-command")
    except RuntimeError as exc:
        assert "does not contain notes.commands" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_run_task_command_dry_run_prints_plan_without_committing(monkeypatch, capsys, tmp_path) -> None:
    class Completed:
        returncode = 0
        stdout = '{"data":{"objective":"Run command","status":"planned","owner":"unassigned","notes":"{\\"commands\\":[\\"echo ok\\"]}"}}'
        stderr = ""

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append((args, kwargs))
        return Completed()

    monkeypatch.setattr(research_warmhub.subprocess, "run", fake_run)

    code = run_task_command(
        "repo/example",
        "run-command",
        agent="agent-c",
        cwd=tmp_path,
        log_file=tmp_path / "worker.log",
        evidence_artifacts=["artifact-dir"],
        dry_run=True,
    )

    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["command"] == "echo ok"
    assert output["would_claim"] is True
    assert output["evidence_artifacts"] == ["artifact-dir", str(tmp_path / "worker.log")]
    assert len(calls) == 1
    assert calls[0][0][0][:4] == ["wh", "thing", "view", "AgentTask/run-command"]


def test_run_task_command_executes_and_writes_finish_ops(monkeypatch, tmp_path) -> None:
    commits: list[tuple[list[dict[str, object]], str]] = []

    monkeypatch.setattr(
        research_warmhub,
        "task_command_payload",
        lambda *_args, **_kwargs: {
            "task": "AgentTask/run-command",
            "command": f"{sys.executable} -c \"print('worker ok')\"",
            "command_index": 0,
            "data": {},
            "notes": {},
        },
    )
    monkeypatch.setattr(
        research_warmhub,
        "make_task_claim_revision_op",
        lambda *_args, **_kwargs: {"operation": "revise", "kind": "thing", "name": "AgentTask/run-command", "data": {"status": "running"}},
    )
    monkeypatch.setattr(
        research_warmhub,
        "make_task_status_revision_op",
        lambda *_args, **_kwargs: {"operation": "revise", "kind": "thing", "name": "AgentTask/run-command", "data": {"status": "complete"}},
    )

    def fake_maybe_commit(_repo, ops, *, dry_run, message):  # noqa: ANN001
        assert dry_run is False
        commits.append((ops, message))
        return 0

    monkeypatch.setattr(research_warmhub, "_maybe_commit", fake_maybe_commit)

    log_file = tmp_path / "worker.log"
    code = run_task_command("repo/example", "run-command", agent="agent-c", log_file=log_file, cwd=tmp_path)

    assert code == 0
    assert "worker ok" in log_file.read_text(encoding="utf-8")
    assert [message for _ops, message in commits] == [
        "Claim navigation research agent task",
        "Finish navigation research agent task",
    ]
    finish_ops = commits[1][0]
    assert finish_ops[1]["data"]["status"] == "complete"
    assert str(log_file) in finish_ops[1]["data"]["evidenceArtifacts"]


def test_run_task_command_can_execute_all_notes_commands(monkeypatch, tmp_path) -> None:
    commits: list[tuple[list[dict[str, object]], str]] = []

    commands = [
        f"{sys.executable} -c \"print('first')\"",
        f"{sys.executable} -c \"print('second')\"",
    ]
    monkeypatch.setattr(
        research_warmhub,
        "task_command_payload",
        lambda *_args, **_kwargs: {
            "task": "AgentTask/fixtures",
            "command": commands[0],
            "commands": commands,
            "command_index": 0,
            "data": {},
            "notes": {"commands": commands},
        },
    )
    monkeypatch.setattr(
        research_warmhub,
        "make_task_claim_revision_op",
        lambda *_args, **_kwargs: {"operation": "revise", "kind": "thing", "name": "AgentTask/fixtures", "data": {"status": "running"}},
    )
    monkeypatch.setattr(
        research_warmhub,
        "make_task_status_revision_op",
        lambda *_args, **kwargs: {"operation": "revise", "kind": "thing", "name": "AgentTask/fixtures", "data": {"status": kwargs["status"]}},
    )

    def fake_maybe_commit(_repo, ops, *, dry_run, message):  # noqa: ANN001
        assert dry_run is False
        commits.append((ops, message))
        return 0

    monkeypatch.setattr(research_warmhub, "_maybe_commit", fake_maybe_commit)

    log_file = tmp_path / "fixtures.log"
    code = run_task_command("repo/example", "fixtures", agent="agent-c", log_file=log_file, cwd=tmp_path, all_commands=True)

    assert code == 0
    log_text = log_file.read_text(encoding="utf-8")
    assert "[command_index] 0" in log_text
    assert "[command_index] 1" in log_text
    assert "first" in log_text
    assert "second" in log_text
    finish_ops = commits[1][0]
    assert finish_ops[0]["data"]["status"] == "complete"
    assert "All 2 command(s) completed successfully." in finish_ops[1]["data"]["summary"]


def test_run_task_command_can_treat_research_exit_two_as_complete(monkeypatch, tmp_path) -> None:
    commits: list[tuple[list[dict[str, object]], str]] = []

    monkeypatch.setattr(
        research_warmhub,
        "task_command_payload",
        lambda *_args, **_kwargs: {
            "task": "AgentTask/run-command",
            "command": f"{sys.executable} -c \"raise SystemExit(2)\"",
            "command_index": 0,
            "data": {},
            "notes": {"accepted_exit_codes": [0, 2]},
        },
    )
    monkeypatch.setattr(
        research_warmhub,
        "make_task_claim_revision_op",
        lambda *_args, **_kwargs: {"operation": "revise", "kind": "thing", "name": "AgentTask/run-command", "data": {"status": "running"}},
    )
    monkeypatch.setattr(
        research_warmhub,
        "make_task_status_revision_op",
        lambda *_args, **kwargs: {"operation": "revise", "kind": "thing", "name": "AgentTask/run-command", "data": {"status": kwargs["status"]}},
    )

    def fake_maybe_commit(_repo, ops, *, dry_run, message):  # noqa: ANN001
        assert dry_run is False
        commits.append((ops, message))
        return 0

    monkeypatch.setattr(research_warmhub, "_maybe_commit", fake_maybe_commit)

    code = run_task_command(
        "repo/example",
        "run-command",
        agent="agent-c",
        log_file=tmp_path / "worker.log",
        cwd=tmp_path,
    )

    assert code == 0
    finish_ops = commits[1][0]
    assert finish_ops[0]["data"]["status"] == "complete"
    assert finish_ops[1]["data"]["status"] == "complete"
    assert "accepted code 2" in finish_ops[1]["data"]["summary"]


def test_task_status_revision_op_preserves_existing_task_data(monkeypatch) -> None:
    class Completed:
        returncode = 0
        stdout = (
            '{"data":{"objective":"Run sweep","status":"running","owner":"agent",'
            '"createdAt":"2026-06-13T00:00:00Z","tags":["qwen"]}}'
        )
        stderr = ""

    monkeypatch.setattr(research_warmhub.subprocess, "run", lambda *_args, **_kwargs: Completed())

    op = make_task_status_revision_op("repo/example", "run-sweep", status="complete")

    assert op["operation"] == "revise"
    assert op["name"] == "AgentTask/run-sweep"
    assert op["data"]["objective"] == "Run sweep"
    assert op["data"]["status"] == "complete"
    assert op["data"]["owner"] == "agent"
    assert op["data"]["tags"] == ["qwen"]
    assert "updatedAt" in op["data"]


def test_task_list_ready_only_filters_incomplete_prerequisites(monkeypatch, capsys) -> None:
    def fake_read(command):  # noqa: ANN001
        if command[:3] == ["wh", "thing", "query"] and "AgentTask" in command:
            status = next(part.split("=", 1)[1] for part in command if isinstance(part, str) and part.startswith("status="))
            if status == "planned":
                return {
                    "items": [
                        {
                            "wref": "AgentTask/run-ready",
                            "name": "run-ready",
                            "data": {
                                "status": "planned",
                                "owner": "unassigned",
                                "objective": "Run ready",
                                "notes": '{"prerequisites":["AgentTask/preflight"]}',
                            },
                        },
                        {
                            "wref": "AgentTask/run-blocked",
                            "name": "run-blocked",
                            "data": {
                                "status": "planned",
                                "owner": "unassigned",
                                "objective": "Run blocked",
                                "notes": '{"prerequisites":["AgentTask/missing-preflight"]}',
                            },
                        },
                    ]
                }
            if status == "complete":
                return {"items": [{"wref": "AgentTask/preflight", "name": "preflight", "data": {"status": "complete"}}]}
        raise AssertionError(command)

    monkeypatch.setattr(research_warmhub, "_read_warmhub_json", fake_read)

    code = research_warmhub._print_task_list("repo/example", status="planned", limit=20, raw_json=False, ready_only=True)

    output = capsys.readouterr().out
    assert code == 0
    assert "AgentTask/run-ready" in output
    assert "AgentTask/run-blocked" not in output


def test_warmhub_status_snapshot_summarizes_queue_runs_and_next_actions(monkeypatch) -> None:
    def fake_read(command):  # noqa: ANN001
        if command[:3] == ["wh", "thing", "query"] and "AgentTask" in command:
            status = next(part.split("=", 1)[1] for part in command if isinstance(part, str) and part.startswith("status="))
            if status == "planned":
                return {
                    "items": [
                        {
                            "wref": "AgentTask/plan-preflight",
                            "name": "plan-preflight",
                            "data": {
                                "status": "planned",
                                "owner": "unassigned",
                                "objective": "Preflight",
                                "relatedExperiment": "NavExperiment/exp@v1",
                                "tags": ["preflight", "qwen"],
                            },
                        },
                        {
                            "wref": "AgentTask/plan-run",
                            "name": "plan-run",
                            "data": {
                                "status": "planned",
                                "owner": "unassigned",
                                "objective": "Run slice",
                                "relatedExperiment": "NavExperiment/exp@v1",
                                "tags": ["trial-slice", "qwen"],
                            },
                        },
                    ]
                }
            if status == "running":
                return {"items": []}
            return {"items": []}
        if command[:3] == ["wh", "thing", "query"] and "NavEvalRun" in command:
            return {
                "items": [
                    {
                        "wref": "NavEvalRun/run-a",
                        "name": "run-a",
                        "data": {
                            "experiment": "NavExperiment/exp@v1",
                            "trialId": "trial-a",
                            "variant": "PromptVariant/variant-a",
                            "success": False,
                            "reason": "max_steps_exhausted",
                            "finalDistanceM": 1.4,
                        },
                    }
                ]
            }
        if command[:3] == ["wh", "assertion", "list"] and "FailureObservation" in command:
            return {
                "items": [
                    {
                        "wref": "FailureObservation/run-a",
                        "aboutWref": "NavEvalRun/run-a",
                        "data": {"observation": "stalled", "confidence": 0.7},
                    }
                ]
            }
        if command[:3] == ["wh", "assertion", "list"] and "SubAgentResult" in command:
            return {"items": []}
        raise AssertionError(command)

    monkeypatch.setattr(research_warmhub, "_read_warmhub_json", fake_read)

    snapshot = research_warmhub.warmhub_status_snapshot("repo/example", related_experiment="exp")

    assert snapshot["task_counts"]["planned"] == 2
    assert snapshot["run_counts"] == {"total": 1, "success": 0, "failed": 1}
    assert snapshot["recent_failures"][0]["summary"] == "stalled"
    assert any("preflight" in action for action in snapshot["next_actions"])


def test_warmhub_status_snapshot_prioritizes_ready_promotion_gate(monkeypatch) -> None:
    def fake_read(command):  # noqa: ANN001
        if command[:3] == ["wh", "thing", "query"] and "AgentTask" in command:
            status = next(part.split("=", 1)[1] for part in command if isinstance(part, str) and part.startswith("status="))
            if status == "planned":
                return {
                    "items": [
                        {
                            "wref": "AgentTask/plan-promotion-gate",
                            "name": "plan-promotion-gate",
                            "data": {
                                "status": "planned",
                                "owner": "unassigned",
                                "objective": "Run gate",
                                "relatedExperiment": "NavExperiment/exp",
                                "tags": ["promotion-gate"],
                                "notes": '{"prerequisites":["AgentTask/run-a"]}',
                            },
                        },
                        {
                            "wref": "AgentTask/plan-failure-analysis",
                            "name": "plan-failure-analysis",
                            "data": {
                                "status": "planned",
                                "owner": "unassigned",
                                "objective": "Analyze",
                                "relatedExperiment": "NavExperiment/exp",
                                "tags": ["failure-analysis"],
                                "notes": '{"prerequisites":["AgentTask/plan-promotion-gate"]}',
                            },
                        },
                    ]
                }
            if status == "complete":
                return {
                    "items": [
                        {
                            "wref": "AgentTask/run-a",
                            "name": "run-a",
                            "data": {"status": "complete", "relatedExperiment": "NavExperiment/exp"},
                        }
                    ]
                }
            return {"items": []}
        if command[:3] == ["wh", "thing", "query"] and "NavEvalRun" in command:
            return {"items": []}
        if command[:3] == ["wh", "assertion", "list"]:
            return {"items": []}
        raise AssertionError(command)

    monkeypatch.setattr(research_warmhub, "_read_warmhub_json", fake_read)

    snapshot = research_warmhub.warmhub_status_snapshot("repo/example", related_experiment="exp")

    assert snapshot["next_actions"][0] == "Run planned promotion gate next: AgentTask/plan-promotion-gate."


def test_warmhub_status_cli_prints_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        research_warmhub,
        "warmhub_status_snapshot",
        lambda repo, *, limit, related_experiment: {
            "repo": repo,
            "related_experiment": related_experiment,
            "task_counts": {"planned": 1, "running": 0, "complete": 0, "failed": 0, "blocked": 0},
            "tasks": {"planned": [], "running": [], "complete": [], "failed": [], "blocked": []},
            "recent_runs": [],
            "run_counts": {"total": 0, "success": 0, "failed": 0},
            "recent_failures": [],
            "recent_subagent_results": [],
            "next_actions": ["Run planned preflight task next."],
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "flatdisk-sim-research-warmhub",
            "--repo",
            "repo/example",
            "status",
            "--related-experiment",
            "NavExperiment/exp",
            "--json",
        ],
    )

    assert research_warmhub.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo"] == "repo/example"
    assert payload["related_experiment"] == "NavExperiment/exp"
    assert payload["next_actions"] == ["Run planned preflight task next."]
