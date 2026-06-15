"""Summarize a hardware harness run into a short incident timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT, SIM_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def summarize_run(run_dir: Path) -> str:
    run_dir = run_dir.expanduser().resolve()
    memory = read_jsonl(run_dir / "memory.jsonl")
    events = read_jsonl(run_dir / "harness_events.jsonl")
    lines: list[str] = [f"# Harness Run Summary: {run_dir.name}", ""]

    goal = next((event.get("goal") for event in events if event.get("event") == "user_goal"), None)
    if goal:
        lines.append(f"Goal: {goal}")
        lines.append("")

    warnings = run_warnings(run_dir, memory)
    if warnings:
        lines.append("Key warnings:")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("Timeline:")
    for record in memory:
        step = record.get("step", "?")
        obs = record.get("observation") if isinstance(record.get("observation"), dict) else {}
        action = record.get("executed_action") if isinstance(record.get("executed_action"), dict) else {}
        result = record.get("tool_result") if isinstance(record.get("tool_result"), dict) else {}
        tool = action.get("tool", result.get("action", "?"))
        lines.append(
            f"- step {step}: obs seq={obs.get('frame_seq')} yaw={_fmt(obs.get('yaw_deg'))} "
            f"image={obs.get('path')} -> {tool} {_compact_args(action.get('args'))}"
        )
        for detail in result_details(result):
            lines.append(f"  {detail}")
        for warning in step_warnings(run_dir, record):
            lines.append(f"  warning: {warning}")

    if not memory:
        lines.append("- no memory.jsonl records found")

    return "\n".join(lines).rstrip() + "\n"


def run_warnings(run_dir: Path, memory: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    seqs: list[int] = []
    for record in memory:
        obs = record.get("observation")
        if isinstance(obs, dict) and isinstance(obs.get("frame_seq"), int):
            seqs.append(int(obs["frame_seq"]))
    for before, after in zip(seqs, seqs[1:]):
        if after + 100 < before:
            warnings.append(
                f"camera frame sequence dropped from {before} to {after}; this usually means a stale cached frame "
                "or a firmware reboot boundary entered the run"
            )
            break

    hidden = hidden_visual_servo_artifacts(run_dir)
    if hidden:
        warnings.append(
            "visual-servo raw/overlay artifacts exist outside the recorded run directory; older logs may show null "
            f"grounding sheets even though evidence exists at {hidden}"
        )
    return warnings


def result_details(result: dict[str, Any]) -> list[str]:
    action = result.get("action")
    if action == "turn_to_angle":
        return [
            "turn result: "
            f"yaw {_fmt(result.get('started_yaw_deg'))} -> {_fmt(result.get('final_yaw_deg'))}, "
            f"target={_fmt(result.get('target_yaw_deg'))}, "
            f"error={_fmt(result.get('heading_error_deg'))}, timed_out={result.get('timed_out')}"
        ]
    if action == "visual_servo_object":
        return [
            "servo result: "
            f"status={result.get('servo_status')} failure={result.get('failure_reason')} "
            f"detected={result.get('ever_detected')}/{result.get('target_detected')} "
            f"track_count={result.get('track_count')} commands={result.get('motor_commands_sent')} "
            f"last_cmd={result.get('last_command')} last_det={result.get('last_detection')}",
            f"grounding: {result.get('grounding_stability')} semantic={result.get('semantic_identity')}",
        ]
    return [f"result: {action or '?'}"]


def step_warnings(run_dir: Path, record: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    result = record.get("tool_result") if isinstance(record.get("tool_result"), dict) else {}
    if result.get("action") == "turn_to_angle" and result.get("timed_out") is True:
        warnings.append("turn tool timed out; IMU yaw did not reach the requested angle")
    if result.get("action") == "visual_servo_object":
        if result.get("moved") and result.get("semantic_identity") != "verified_phrase_grounding":
            warnings.append("robot moved on unverified phrase grounding; inspect grounding audit before trusting it")
        if result.get("motion_contact_sheet") in (None, "") and hidden_visual_servo_artifacts(run_dir):
            warnings.append("recorded summary is missing visual-servo sheets, but raw/overlay frames exist elsewhere")
    return warnings


def hidden_visual_servo_artifacts(run_dir: Path) -> Path | None:
    try:
        rel = run_dir.relative_to(SIM_ROOT / "scratch")
    except ValueError:
        return None
    alternate = REPO_ROOT / "scratch" / rel
    if alternate == run_dir or not alternate.exists():
        return None
    if any(alternate.glob("motion_frames/*_visual_servo_raw/*.jpg")) or any(
        alternate.glob("motion_frames/*_visual_servo_overlays/*.jpg")
    ):
        return alternate
    return None


def _compact_args(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    compact = {key: value[key] for key in sorted(value)}
    return json.dumps(compact, sort_keys=True)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return str(value)


def latest_dashboard_run(root: Path) -> Path | None:
    candidates = sorted(path for path in root.glob("qwen_trace_dashboard/*") if path.is_dir())
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, help="Run directory containing memory.jsonl.")
    args = parser.parse_args()
    run_dir = args.run_dir or latest_dashboard_run(SIM_ROOT / "scratch")
    if run_dir is None:
        parser.error("run_dir was not provided and no sim/scratch/qwen_trace_dashboard run exists")
    print(summarize_run(run_dir))


if __name__ == "__main__":
    main()
