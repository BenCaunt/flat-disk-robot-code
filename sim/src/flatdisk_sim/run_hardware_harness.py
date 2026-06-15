"""Run the LLM navigation harness against the physical robot over Zenoh.

This entry point intentionally does not launch THOR and does not use hidden
evaluator state. It points the same harness session used in simulator evals at
the real flat-disk robot topics. Motor-capable tools are blocked unless
``--arm`` is passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .agent_tools import AgentTools, DEFAULT_CONNECT, make_contact_sheet
from .harness_rerun import HarnessRerunLogger
from .llm_harness import (
    CodexExecRunner,
    DeterministicHarnessRunner,
    HarnessConfig,
    HarnessSession,
    NoopCriticRunner,
    OpenAICompatibleVisionRunner,
    SafetyCriticRunner,
    ScriptedOpenVocabRunner,
)
from .paths import REPO_ROOT, SCRATCH_ROOT
from .protocol import DEFAULT_NAMESPACE
from .prompt_audit import audit_prompts


@dataclass(frozen=True)
class HardwareRunResult:
    run_dir: Path
    summary_path: Path
    report_path: Path | None
    exit_code: int


@dataclass(frozen=True)
class _PowerLimitedResult:
    inner: Any
    requested: float
    effective: float
    max_power: float

    def summary(self) -> dict[str, Any]:
        summary_method = getattr(self.inner, "summary", None)
        if callable(summary_method):
            summary = summary_method()
            if isinstance(summary, dict):
                payload = dict(summary)
            else:
                payload = {"result": str(summary)}
        elif isinstance(self.inner, dict):
            payload = dict(self.inner)
        else:
            payload = {"result": str(self.inner)}
        payload.setdefault("requested_power_percent", float(self.requested))
        payload.setdefault("effective_power_percent", float(self.effective))
        payload.setdefault("max_forward_power_percent", float(self.max_power))
        return payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


DETECTOR_REQUIRED_MODULES = {
    "florence-mlx": ("mlx_vlm", "transformers", "torch", "torchvision"),
    "florence-transformers": ("torch", "transformers", "timm", "einops"),
    "grounding-dino": ("torch", "torchvision", "transformers", "timm", "einops"),
}

DETECTOR_DEFAULT_MODELS = {
    "florence-mlx": "mlx-community/Florence-2-base-ft-4bit",
    "florence-transformers": "microsoft/Florence-2-base-ft",
    "grounding-dino": "IDEA-Research/grounding-dino-tiny",
}

PIP_PACKAGE_BY_MODULE = {
    "mlx_vlm": "mlx-vlm",
    "torchvision": "torchvision",
}


class ArmedGuardTools:
    """RobotTools wrapper that blocks motion unless the run is armed."""

    def __init__(self, inner: Any, *, armed: bool, max_forward_power_percent: float | None = None) -> None:
        self.inner = inner
        self.armed = armed
        self.max_forward_power_percent = max_forward_power_percent

    def observe(self, *, label: str = "observe", timeout_s: float = 2.0) -> Any:
        return self.inner.observe(label=label, timeout_s=timeout_s)

    def preview_frame(self, *, label: str = "dashboard_preview", timeout_s: float = 0.25) -> Any:
        preview = getattr(self.inner, "preview_frame", None)
        if not callable(preview):
            return self.inner.observe(label=label, timeout_s=timeout_s).summary()
        return preview(label=label, timeout_s=timeout_s)

    def preview_frame_bytes(self, *, timeout_s: float = 0.1) -> Any:
        preview = getattr(self.inner, "preview_frame_bytes", None)
        if callable(preview):
            return preview(timeout_s=timeout_s)
        summary = self.preview_frame(timeout_s=timeout_s)
        image_path = Path(str(summary["path"]))
        return {"jpeg": image_path.read_bytes(), **summary}

    def preview_telemetry(self, *, timeout_s: float = 0.05) -> Any:
        preview = getattr(self.inner, "preview_telemetry", None)
        if callable(preview):
            return preview(timeout_s=timeout_s)
        return {}

    def turn_by_angle(self, degrees: float, *, power_percent: float = 10.0) -> Any:
        if not self.armed:
            return self._blocked_motion(
                "turn_by_angle",
                requested_degrees=degrees,
                requested_power_percent=power_percent,
            )
        return self.inner.turn_by_angle(degrees, power_percent=power_percent)

    def drive_straight(self, power_percent: float, duration_s: float) -> Any:
        effective_power = self._limit_forward_power(power_percent)
        if not self.armed:
            return self._blocked_motion(
                "drive_straight",
                requested_power_percent=power_percent,
                effective_power_percent=effective_power,
                requested_duration_s=duration_s,
                max_forward_power_percent=self.max_forward_power_percent,
            )
        self._log_forward_limit("drive_straight", requested=power_percent, effective=effective_power)
        result = self.inner.drive_straight(effective_power, duration_s)
        return self._annotate_forward_limit(result, requested=power_percent, effective=effective_power)

    def visual_servo_object(
        self,
        prompt: str,
        *,
        duration_s: float = 2.0,
        detector: str | None = None,
        forward_power: float = 18.0,
    ) -> Any:
        effective_forward_power = self._limit_forward_power(forward_power)
        if not self.armed:
            return self._blocked_motion(
                "visual_servo_object",
                prompt=prompt,
                detector=detector or self.inner.object_drive_detector,
                requested_duration_s=duration_s,
                requested_forward_power=forward_power,
                effective_forward_power=effective_forward_power,
                max_forward_power_percent=self.max_forward_power_percent,
            )
        self._log_forward_limit("visual_servo_object", requested=forward_power, effective=effective_forward_power)
        result = self.inner.visual_servo_object(
            prompt,
            duration_s=duration_s,
            detector=detector,
            forward_power=effective_forward_power,
        )
        return self._annotate_forward_limit(result, requested=forward_power, effective=effective_forward_power)

    def check_object_grounding(self, *, image_path: Path, prompt: str, detector: str | None = None) -> Any:
        return self.inner.check_object_grounding(image_path=image_path, prompt=prompt, detector=detector)

    def query_topomap_memory(self, *, image_path: Path, goal_query: str) -> Any:
        return self.inner.query_topomap_memory(image_path=image_path, goal_query=goal_query)

    def stop(self) -> Any:
        return self.inner.stop()

    def close(self) -> None:
        self.inner.close()

    def _blocked_motion(self, action: str, **details: Any) -> dict[str, Any]:
        self.inner.stop()
        summary = {
            "action": action,
            "ok": False,
            "armed": False,
            "moved": False,
            "motor_commands_sent": 0,
            "servo_status": "not_armed" if action == "visual_servo_object" else None,
            "failure_reason": "not_armed",
            "reason": "motion command blocked; rerun with --arm to publish motor commands",
            **details,
        }
        self.inner.log("blocked_motion", summary)
        return summary

    def _limit_forward_power(self, power_percent: float) -> float:
        power = float(power_percent)
        if self.max_forward_power_percent is None:
            return power
        cap = abs(float(self.max_forward_power_percent))
        return max(-cap, min(cap, power))

    def _log_forward_limit(self, action: str, *, requested: float, effective: float) -> None:
        if self.max_forward_power_percent is None or float(requested) == float(effective):
            return
        self.inner.log(
            "forward_power_limited",
            {
                "action": action,
                "requested_power_percent": float(requested),
                "effective_power_percent": float(effective),
                "max_forward_power_percent": abs(float(self.max_forward_power_percent)),
            },
        )

    def _annotate_forward_limit(self, result: Any, *, requested: float, effective: float) -> Any:
        if self.max_forward_power_percent is None:
            return result
        max_power = abs(float(self.max_forward_power_percent))
        if not isinstance(result, dict):
            return _PowerLimitedResult(result, requested=float(requested), effective=float(effective), max_power=max_power)
        result.setdefault("requested_power_percent", float(requested))
        result.setdefault("effective_power_percent", float(effective))
        result.setdefault("max_forward_power_percent", max_power)
        return result


def forward_power_limit_rule(max_forward_power_percent: float | None) -> str | None:
    if max_forward_power_percent is None:
        return None
    return (
        f"Hardware safety limit: forward/reverse drive commands are capped at "
        f"{abs(float(max_forward_power_percent)):.1f}% power. Prefer short, deliberate forward, reverse, "
        "and visual servo moves."
    )


def run_hardware_harness(args: argparse.Namespace) -> HardwareRunResult:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir / stamp
    policy_dir = run_root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    session: HarnessSession | None = None
    tools: ArmedGuardTools | None = None
    detector_readiness = _detector_readiness(
        args.object_drive_detector,
        load_check=not args.skip_detector_load_preflight,
        load_timeout_s=args.detector_load_timeout,
    )
    try:
        if not args.skip_detector_preflight and not detector_readiness["ok"]:
            summary = {
                "goal": args.goal,
                "armed": args.arm,
                "runner": args.runner,
                "critic_mode": _resolve_critic_mode(args.runner, args.critic_mode),
                "preflight_only": args.preflight_only,
                "object_drive_detector": args.object_drive_detector,
                "detector_readiness": detector_readiness,
                "detector_preflight_skipped": False,
                "namespace": args.namespace,
                "connect": args.connect,
                "max_forward_power_percent": args.max_forward_power_percent,
                "policy_dir": str(policy_dir),
                "error": _detector_not_ready_error(detector_readiness),
            }
            print(f"[hardware-harness] {summary['error']}", file=sys.stderr, flush=True)
            return _finish_run(run_root, policy_dir, summary, exit_code=1)

        raw_tools = AgentTools(
            run_dir=policy_dir,
            namespace=args.namespace,
            connect=args.connect,
            reverse_yaw=not args.no_reverse_yaw,
            reverse_correction=args.reverse_correction,
            heading_kp=args.turn_heading_kp,
            max_turn_percent=args.max_turn_percent,
            min_turn_percent=args.min_turn_percent,
            control_hz=args.control_hz,
            object_drive_detector=args.object_drive_detector,
            topomap_memory_map_dir=args.topomap_memory_map_dir,
            topomap_memory_use_clip=args.topomap_memory_use_clip,
            topomap_memory_allow_semantic_terms=args.topomap_memory_allow_semantic_terms,
        )
        tools = ArmedGuardTools(
            raw_tools,
            armed=args.arm,
            max_forward_power_percent=args.max_forward_power_percent,
        )
        preflight = raw_tools.observe(label="preflight", timeout_s=args.startup_timeout).summary()
        if args.preflight_only:
            summary = {
                "goal": args.goal,
                "armed": args.arm,
                "preflight_only": True,
                "preflight_observation": preflight,
                "object_drive_detector": args.object_drive_detector,
                "detector_readiness": detector_readiness,
                "detector_preflight_skipped": args.skip_detector_preflight,
                "namespace": args.namespace,
                "connect": args.connect,
                "max_forward_power_percent": args.max_forward_power_percent,
                "policy_dir": str(policy_dir),
            }
            return _finish_run(run_root, policy_dir, summary, exit_code=0)

        actor = _build_actor(args, policy_dir=policy_dir)
        critic_mode = _resolve_critic_mode(args.runner, args.critic_mode)
        critic = _build_critic(args.runner, actor, mode=critic_mode)
        rerun_logger = None
        if args.rerun:
            rerun_logger = HarnessRerunLogger(
                recording_id=f"flatdisk_hardware_harness_{stamp}",
                save_path=policy_dir / "hardware_harness.rrd",
                spawn=not args.rerun_no_spawn,
            )

        forward_rule = forward_power_limit_rule(args.max_forward_power_percent)
        actor_rules = tuple(args.actor_rule) + ((forward_rule,) if forward_rule else ())
        session = HarnessSession(
            config=HarnessConfig(
                run_dir=policy_dir,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                prompt_profile=args.prompt_profile,
                actor_rules=actor_rules,
                critic_rules=tuple(args.critic_rule),
                critic_mode=critic_mode,
                max_steps=args.max_steps,
                rerun_enabled=args.rerun,
            ),
            tools=tools,
            actor=actor,
            critic=critic,
            rerun_logger=rerun_logger,
        )
        session.start_goal(args.goal)

        step_records: list[dict[str, Any]] = []
        for step_index in range(args.max_steps):
            if session.mode != "auto":
                break
            print(f"[hardware-harness] step {step_index + 1}/{args.max_steps} mode={session.mode}", flush=True)
            record = session.run_auto_step()
            if record is not None:
                step_records.append(record)

        status = session.status()
        exit_code = 0 if status.get("mode") in {"complete", "auto"} else 2
        summary = {
            "goal": args.goal,
            "armed": args.arm,
            "runner": args.runner,
            "critic_mode": critic_mode,
            "qwen_endpoint": args.qwen_endpoint if args.runner == "qwen" else None,
            "qwen_model": args.qwen_model if args.runner == "qwen" else None,
            "object_drive_detector": args.object_drive_detector,
            "detector_readiness": detector_readiness,
            "detector_preflight_skipped": args.skip_detector_preflight,
            "namespace": args.namespace,
            "connect": args.connect,
            "turn_heading_kp": args.turn_heading_kp,
            "max_turn_percent": args.max_turn_percent,
            "min_turn_percent": args.min_turn_percent,
            "control_hz": args.control_hz,
            "reverse_correction": args.reverse_correction,
            "max_forward_power_percent": args.max_forward_power_percent,
            "max_steps": args.max_steps,
            "step_count": len(step_records),
            "status": status,
            "preflight_observation": preflight,
            "policy_input_allowlist": [
                "real robot camera frame",
                "IMU yaw",
                "bounded tool results",
                "motion strips",
                "harness memory",
            ],
            "policy_dir": str(policy_dir),
        }
        return _finish_run(run_root, policy_dir, summary, exit_code=exit_code)
    except Exception as exc:
        summary = {
            "goal": args.goal,
            "armed": args.arm,
            "runner": args.runner,
            "critic_mode": _resolve_critic_mode(args.runner, args.critic_mode),
            "namespace": args.namespace,
            "connect": args.connect,
            "max_forward_power_percent": args.max_forward_power_percent,
            "error": str(exc),
            "object_drive_detector": args.object_drive_detector,
            "detector_readiness": detector_readiness,
            "detector_preflight_skipped": args.skip_detector_preflight,
            "policy_dir": str(policy_dir),
        }
        return _finish_run(run_root, policy_dir, summary, exit_code=1)
    finally:
        if session is not None:
            session.close()
        elif tools is not None:
            tools.close()


def _finish_run(run_root: Path, policy_dir: Path, summary: dict[str, Any], *, exit_code: int) -> HardwareRunResult:
    prompt_audit = audit_prompts(policy_dir / "prompts")
    frame_paths = sorted((policy_dir / "frames").glob("*.jpg"))[-8:]
    contact_sheet = make_contact_sheet(frame_paths, policy_dir / "camera_contact_sheet.jpg") if frame_paths else None
    full_summary = {
        **summary,
        "prompt_audit": prompt_audit,
        "camera_contact_sheet": str(contact_sheet) if contact_sheet else None,
        "run_root": str(run_root),
    }
    failures = _collect_failures(full_summary, policy_dir)
    full_summary["failures"] = failures
    timeline = _write_timeline_html(run_root, policy_dir, full_summary)
    report = _write_report(run_root, full_summary, prompt_audit=prompt_audit, contact_sheet=contact_sheet, timeline=timeline)
    full_summary["timeline_html"] = str(timeline)
    full_summary["report"] = str(report)
    summary_path = run_root / "hardware_harness_summary.json"
    summary_path.write_text(json.dumps(full_summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    terminal_summary: dict[str, Any] = {
        "summary": str(summary_path),
        "report": str(report),
        "timeline_html": str(timeline),
        "exit_code": exit_code,
        "failure_count": len(failures),
    }
    if full_summary.get("error"):
        terminal_summary["error"] = full_summary.get("error")
    if failures:
        terminal_summary["first_failure"] = failures[0]
    print(
        json.dumps(terminal_summary, indent=2, sort_keys=True, default=str),
        flush=True,
    )
    return HardwareRunResult(run_dir=run_root, summary_path=summary_path, report_path=report, exit_code=exit_code)


def _detector_readiness(
    detector: str,
    *,
    load_check: bool = False,
    load_timeout_s: float = 180.0,
) -> dict[str, Any]:
    required_modules = DETECTOR_REQUIRED_MODULES.get(detector, ())
    missing_modules = [module for module in required_modules if not _module_available(module)]
    missing_packages = [_pip_package_for_module(module) for module in missing_modules]
    install_hint = f"uv pip install {' '.join(missing_packages)}" if missing_packages else None
    load_report: dict[str, Any] | None = None
    ok = not missing_modules
    if load_check and ok:
        load_report = _detector_load_check(detector, timeout_s=load_timeout_s)
        ok = load_report["ok"]
    return {
        "detector": detector,
        "ok": ok,
        "required_modules": list(required_modules),
        "missing_modules": missing_modules,
        "python": sys.executable,
        "install_hint": install_hint,
        "load_check": load_report,
    }


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _pip_package_for_module(module_name: str) -> str:
    return PIP_PACKAGE_BY_MODULE.get(module_name, module_name)


def _detector_load_check(detector: str, *, timeout_s: float) -> dict[str, Any]:
    code = f"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, {str(REPO_ROOT / "scripts")!r})
from object_drive_zenoh import build_detector

args = argparse.Namespace(
    detector={detector!r},
    model={DETECTOR_DEFAULT_MODELS["florence-mlx"]!r},
    transformers_model={DETECTOR_DEFAULT_MODELS["florence-transformers"]!r},
    grounding_dino_model={DETECTOR_DEFAULT_MODELS["grounding-dino"]!r},
    grounding_dino_box_threshold=0.25,
    grounding_dino_text_threshold=0.25,
    device="auto",
    max_tokens=1,
    temperature=0.0,
)
started = time.perf_counter()
detector_instance = build_detector(args)
print(json.dumps({{
    "detector": {detector!r},
    "detector_class": type(detector_instance).__name__,
    "elapsed_s": round(time.perf_counter() - started, 3),
}}))
"""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "timeout_s": timeout_s,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "failure_reason": "timeout",
            "stdout_tail": _short_text(exc.stdout),
            "stderr_tail": _short_text(exc.stderr),
        }
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "timeout_s": timeout_s,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "failure_reason": None if completed.returncode == 0 else "load_failed",
        "stdout_tail": _short_text(completed.stdout),
        "stderr_tail": _short_text(completed.stderr),
    }


def _detector_not_ready_error(readiness: dict[str, Any]) -> str:
    detector = readiness.get("detector", "unknown")
    if readiness.get("missing_modules"):
        missing = ", ".join(str(module) for module in readiness.get("missing_modules", [])) or "unknown"
        hint = readiness.get("install_hint")
        suffix = f" Install with `{hint}`." if hint else ""
        return (
            f"detector_not_ready: {detector} is missing Python module(s): {missing}."
            f"{suffix} Use --skip-detector-preflight only if object-drive runs in a different Python environment."
        )
    load_report = readiness.get("load_check") if isinstance(readiness.get("load_check"), dict) else {}
    stderr = _short_text(load_report.get("stderr_tail"), max_lines=6, max_chars=900) if load_report else None
    detail = f" stderr_tail={stderr!r}" if stderr else ""
    return (
        f"detector_not_ready: {detector} dependency imports succeeded but detector load failed"
        f" with reason={load_report.get('failure_reason') or 'unknown'} returncode={load_report.get('returncode')}."
        f"{detail} Use --skip-detector-load-preflight only if you accept catching detector load errors during visual_servo_object."
    )



def _collect_failures(summary: dict[str, Any], policy_dir: Path) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    readiness = summary.get("detector_readiness")
    detector_preflight_failed = (
        isinstance(readiness, dict) and readiness.get("ok") is False and not summary.get("detector_preflight_skipped")
    )
    if detector_preflight_failed:
        load_report = readiness.get("load_check") if isinstance(readiness.get("load_check"), dict) else {}
        missing_modules = readiness.get("missing_modules", [])
        failure_reason = "missing_dependencies" if missing_modules else load_report.get("failure_reason") or "load_failed"
        failures.append(
            {
                "source": "detector_preflight",
                "tool": "visual_servo_object",
                "detector": readiness.get("detector"),
                "failure_reason": failure_reason,
                "missing_modules": missing_modules,
                "install_hint": readiness.get("install_hint"),
                "returncode": load_report.get("returncode"),
                "message": summary.get("error"),
                "stdout_tail": load_report.get("stdout_tail"),
                "stderr_tail": load_report.get("stderr_tail"),
            }
        )
    elif summary.get("error"):
        failures.append(
            {
                "source": "run",
                "failure_reason": "error",
                "message": str(summary.get("error")),
            }
        )
    for record in _read_jsonl(policy_dir / "memory.jsonl"):
        if not isinstance(record, dict):
            continue
        result = record.get("tool_result")
        if not isinstance(result, dict) or not _is_failed_tool_result(result):
            continue
        action = record.get("executed_action") if isinstance(record.get("executed_action"), dict) else {}
        failures.append(_summarize_tool_failure(record, action, result))
    return failures


def _is_failed_tool_result(result: dict[str, Any]) -> bool:
    if result.get("ok") is False:
        return True
    if result.get("failure_reason"):
        return True
    if result.get("servo_status") in {"process_failed", "not_armed"}:
        return True
    return result.get("returncode") not in {None, 0}


def _summarize_tool_failure(record: dict[str, Any], action: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    action_args = action.get("args") if isinstance(action.get("args"), dict) else {}
    failure = {
        "source": "tool_result",
        "step": record.get("step"),
        "tool": action.get("tool") or result.get("action"),
        "action": result.get("action"),
        "ok": result.get("ok"),
        "failure_reason": result.get("failure_reason"),
        "servo_status": result.get("servo_status"),
        "returncode": result.get("returncode"),
        "detector": result.get("detector"),
        "prompt": result.get("prompt") or action_args.get("prompt"),
        "planner_note": result.get("planner_note"),
        "stdout_tail": _short_text(result.get("stdout_tail")),
        "stderr_tail": _short_text(result.get("stderr_tail")),
    }
    return {key: value for key, value in failure.items() if _has_value(value)}


def _short_text(value: Any, *, max_lines: int = 14, max_chars: int = 2400) -> str | None:
    if not value:
        return None
    text = str(value)
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[-max_lines:])
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _has_value(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _write_report(
    run_root: Path,
    summary: dict[str, Any],
    *,
    prompt_audit: dict[str, Any],
    contact_sheet: Path | None,
    timeline: Path | None = None,
) -> Path:
    report = run_root / "hardware_harness_report.md"
    status = summary.get("status") if isinstance(summary.get("status"), dict) else {}
    lines = [
        "# Hardware LLM Harness Run",
        "",
        f"- Goal: `{summary.get('goal', '')}`",
        f"- Armed: `{summary.get('armed', False)}`",
        f"- Runner: `{summary.get('runner', 'n/a')}`",
        f"- Critic mode: `{summary.get('critic_mode', 'n/a')}`",
        f"- Namespace: `{summary.get('namespace', '')}`",
        f"- Connect: `{summary.get('connect', '')}`",
        f"- Final mode: `{status.get('mode', 'preflight')}`",
        f"- Steps: `{summary.get('step_count', 0)}`",
        f"- Prompt leaks: `{', '.join(prompt_audit.get('forbidden_tokens_found', [])) or 'none'}`",
    ]
    if contact_sheet is not None:
        lines.append(f"- Camera contact sheet: `{contact_sheet}`")
    if timeline is not None:
        lines.append(f"- Timeline HTML: `{timeline}`")
    lines.extend(
        [
            "",
            "Policy inputs were limited to robot camera frames, IMU yaw, bounded tool results, motion strips, and harness memory.",
            "This run has no hidden simulator evaluator or automatic success signal; inspect artifacts and stop decisions manually.",
        ]
    )
    readiness = summary.get("detector_readiness")
    if isinstance(readiness, dict):
        missing = readiness.get("missing_modules") or []
        lines.extend(
            [
                "",
                "## Detector Preflight",
                "",
                f"- Detector: `{readiness.get('detector')}`",
                f"- OK: `{readiness.get('ok')}`",
                f"- Required modules: `{', '.join(readiness.get('required_modules') or []) or 'none'}`",
                f"- Missing modules: `{', '.join(missing) or 'none'}`",
            ]
        )
        if readiness.get("install_hint"):
            lines.append(f"- Install hint: `{readiness.get('install_hint')}`")
        load_report = readiness.get("load_check") if isinstance(readiness.get("load_check"), dict) else None
        if load_report is not None:
            lines.extend(
                [
                    f"- Load check OK: `{load_report.get('ok')}`",
                    f"- Load check returncode: `{load_report.get('returncode')}`",
                    f"- Load check elapsed_s: `{load_report.get('elapsed_s')}`",
                ]
            )
            if load_report.get("failure_reason"):
                lines.append(f"- Load check failure_reason: `{load_report.get('failure_reason')}`")
            if load_report.get("stdout_tail"):
                lines.extend(["", "Load check stdout:", "", "```text", str(load_report.get("stdout_tail")), "```"])
            if load_report.get("stderr_tail"):
                lines.extend(["", "Load check stderr:", "", "```text", str(load_report.get("stderr_tail")), "```"])
        if summary.get("detector_preflight_skipped"):
            lines.append("- Preflight skipped: `True`")
    if summary.get("error"):
        lines.extend(["", "## Run Error", "", "```text", str(summary.get("error")), "```"])
    failures = summary.get("failures") if isinstance(summary.get("failures"), list) else []
    if failures:
        lines.extend(["", "## Failures"])
        for index, failure in enumerate(failures[:8], start=1):
            if not isinstance(failure, dict):
                continue
            label = failure.get("tool") or failure.get("source") or "failure"
            step = failure.get("step")
            step_text = f" step `{step}`" if step is not None else ""
            lines.extend(["", f"### {index}. `{label}`{step_text}"])
            for key in (
                "source",
                "failure_reason",
                "servo_status",
                "returncode",
                "detector",
                "prompt",
                "install_hint",
                "missing_modules",
                "message",
            ):
                if _has_value(failure.get(key)):
                    if key == "message":
                        lines.append(f"- {key}: {failure.get(key)}")
                    else:
                        lines.append(f"- {key}: `{failure.get(key)}`")
            if failure.get("planner_note"):
                lines.append(f"- planner_note: {failure.get('planner_note')}")
            if failure.get("stdout_tail"):
                lines.extend(["", "```text", str(failure.get("stdout_tail")), "```"])
            if failure.get("stderr_tail"):
                lines.extend(["", "```text", str(failure.get("stderr_tail")), "```"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_timeline_html(run_root: Path, policy_dir: Path, summary: dict[str, Any]) -> Path:
    memory_records = _read_jsonl(policy_dir / "memory.jsonl")
    event_records = _read_jsonl(policy_dir / "events.jsonl")
    status = summary.get("status") if isinstance(summary.get("status"), dict) else {}
    readiness = summary.get("detector_readiness") if isinstance(summary.get("detector_readiness"), dict) else {}
    failures = summary.get("failures") if isinstance(summary.get("failures"), list) else []
    rows = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Hardware Harness Timeline</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f6f7f9;color:#20242a}",
        "h1{font-size:22px;margin:0 0 12px}",
        ".meta,.step{background:#fff;border:1px solid #d9dde5;border-radius:8px;padding:14px;margin:12px 0}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}",
        ".thumb{max-width:100%;border:1px solid #cbd1db;border-radius:6px;background:#111}",
        "pre{white-space:pre-wrap;word-break:break-word;background:#f0f2f5;border-radius:6px;padding:10px;font-size:12px;max-height:360px;overflow:auto}",
        ".label{font-size:12px;color:#596171;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 4px}",
        ".pill{display:inline-block;border-radius:999px;padding:2px 8px;background:#e7edf7;margin-right:6px;font-size:12px}",
        ".bad{background:#f8dddd}.ok{background:#dcf2e4}",
        "a{color:#1f5fae}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Hardware Harness Timeline</h1>",
        '<section class="meta">',
        f"<div><span class=\"pill\">goal</span>{_esc(summary.get('goal'))}</div>",
        f"<div><span class=\"pill\">armed</span>{_esc(summary.get('armed'))}</div>",
        f"<div><span class=\"pill\">connect</span>{_esc(summary.get('connect'))}</div>",
        f"<div><span class=\"pill\">namespace</span>{_esc(summary.get('namespace'))}</div>",
        f"<div><span class=\"pill\">mode</span>{_esc(status.get('mode'))}</div>",
        f"<div><span class=\"pill\">steps</span>{_esc(summary.get('step_count'))}</div>",
        f"<div><span class=\"pill {'ok' if readiness.get('ok') else 'bad'}\">detector</span>{_esc(readiness.get('detector') or summary.get('object_drive_detector'))}</div>",
        f"<div><span class=\"pill {'ok' if not failures else 'bad'}\">failures</span>{_esc(len(failures))}</div>",
        _link_line("Summary JSON", run_root / "hardware_harness_summary.json", root=run_root),
        _link_line("Report", run_root / "hardware_harness_report.md", root=run_root),
        _link_line("Memory JSONL", policy_dir / "memory.jsonl", root=run_root),
        _link_line("Events JSONL", policy_dir / "events.jsonl", root=run_root),
        _link_line("Harness Events JSONL", policy_dir / "harness_events.jsonl", root=run_root),
        _image_block("Camera Contact Sheet", summary.get("camera_contact_sheet"), root=run_root),
        "</section>",
    ]
    if readiness:
        rows.extend(
            [
                '<section class="step">',
                "<h2>Detector Preflight</h2>",
                f"<pre>{_json_block(readiness)}</pre>",
                "</section>",
            ]
        )
    if summary.get("error"):
        rows.extend(
            [
                '<section class="step">',
                "<h2>Run Error</h2>",
                f"<pre>{_esc(summary.get('error'))}</pre>",
                "</section>",
            ]
        )
    if failures:
        rows.extend(
            [
                '<section class="step">',
                "<h2>Failures</h2>",
                f"<pre>{_json_block(failures)}</pre>",
                "</section>",
            ]
        )
    for record in memory_records:
        if not isinstance(record, dict):
            continue
        step = record.get("step")
        action = record.get("executed_action") or record.get("actor_action") or {}
        tool_result = record.get("tool_result") if isinstance(record.get("tool_result"), dict) else {}
        observation = record.get("observation") if isinstance(record.get("observation"), dict) else {}
        status_class = "ok" if tool_result.get("ok", True) is not False and not tool_result.get("failure_reason") else "bad"
        rows.extend(
            [
                f'<section class="step">',
                f'<h2>Step {_esc(step)} <span class="pill {status_class}">{_esc(action.get("tool"))}</span></h2>',
                '<div class="grid">',
                _image_block("Latest Camera Frame", observation.get("path"), root=run_root, policy_dir=policy_dir),
                _image_block("Motion Contact Sheet", tool_result.get("motion_contact_sheet"), root=run_root, policy_dir=policy_dir),
                _image_block("Detector Overlay", tool_result.get("debug_overlay_contact_sheet") or tool_result.get("overlay_path"), root=run_root, policy_dir=policy_dir),
                _image_block("Grounding Audit", tool_result.get("grounding_audit_contact_sheet"), root=run_root, policy_dir=policy_dir),
                _image_block("Topomap Contact Sheet", tool_result.get("topomap_contact_sheet"), root=run_root, policy_dir=policy_dir),
                "</div>",
                '<div class="label">Executed Action</div>',
                f"<pre>{_json_block(action)}</pre>",
                '<div class="label">Tool Result</div>',
                f"<pre>{_json_block(tool_result)}</pre>",
                '<div class="label">Memory Record</div>',
                f"<pre>{_json_block(record)}</pre>",
                "</section>",
            ]
        )
    if not memory_records:
        rows.append('<section class="step"><p>No memory records were written. Check preflight/error fields in the summary.</p></section>')
    if event_records:
        rows.extend(
            [
                '<section class="step">',
                "<h2>Recent Tool Events</h2>",
                f"<pre>{_json_block(event_records[-20:])}</pre>",
                "</section>",
            ]
        )
    rows.extend(["</body>", "</html>"])
    path = run_root / "hardware_harness_timeline.html"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append({"unparsed": line})
    return records


def _json_block(value: Any) -> str:
    return html.escape(json.dumps(value, indent=2, sort_keys=True, default=str))


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _link_line(label: str, path: Path, *, root: Path) -> str:
    if not path.exists():
        return ""
    rel = _relative_artifact_path(path, root=root)
    return f'<div><a href="{_esc(rel)}">{_esc(label)}</a></div>'


def _image_block(label: str, path_value: Any, *, root: Path, policy_dir: Path | None = None) -> str:
    path = _resolve_artifact_path(path_value, root=root, policy_dir=policy_dir)
    if path is None or not path.exists():
        return ""
    rel = _relative_artifact_path(path, root=root)
    return (
        "<div>"
        f'<div class="label">{_esc(label)}</div>'
        f'<a href="{_esc(rel)}"><img class="thumb" src="{_esc(rel)}" alt="{_esc(label)}"></a>'
        f'<div><a href="{_esc(rel)}">{_esc(path.name)}</a></div>'
        "</div>"
    )


def _resolve_artifact_path(path_value: Any, *, root: Path, policy_dir: Path | None = None) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    candidates = []
    if policy_dir is not None:
        candidates.append(policy_dir / path)
    candidates.append(root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _relative_artifact_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _build_actor(args: argparse.Namespace, *, policy_dir: Path) -> Any:
    if args.runner == "codex":
        return CodexExecRunner(model=args.model, reasoning_effort=args.reasoning_effort, cwd=policy_dir)
    if args.runner == "qwen":
        return OpenAICompatibleVisionRunner(
            model=args.qwen_model,
            endpoint=args.qwen_endpoint,
            temperature=args.qwen_temperature,
            max_tokens=args.qwen_max_tokens,
        )
    if args.runner == "scripted-open-vocab":
        return ScriptedOpenVocabRunner(visual_servo_detector=args.object_drive_detector)
    if args.runner == "deterministic":
        return DeterministicHarnessRunner()
    raise ValueError(f"unknown runner: {args.runner}")


def _resolve_critic_mode(runner: str, mode: str) -> str:
    mode = str(mode or "auto").strip().lower()
    if mode not in {"auto", "none", "safety", "same-model"}:
        raise ValueError(f"unknown critic mode: {mode}")
    if mode != "auto":
        return mode
    if runner == "qwen":
        return "none"
    if runner == "codex":
        return "same-model"
    return "safety"


def _build_critic(runner: str, actor: Any, *, mode: str) -> Any:
    if mode == "none":
        return NoopCriticRunner()
    if mode == "same-model":
        return actor
    if mode != "safety":
        raise ValueError(f"unknown critic mode: {mode}")
    del runner
    return SafetyCriticRunner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True, help="Navigation goal, e.g. 'go to the blue chair'.")
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "hardware_llm_harness")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--connect", default=DEFAULT_CONNECT)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--preflight-only", action="store_true", help="Wait for one camera/IMU observation and exit.")
    parser.add_argument("--arm", action="store_true", help="Allow bounded tools to publish motor commands.")
    parser.add_argument("--no-reverse-yaw", action="store_true")
    parser.add_argument("--reverse-correction", action="store_true", help="Flip IMU heading correction sign for closed-loop turns/drives.")
    parser.add_argument("--turn-heading-kp", type=float, default=8.0, help="Closed-loop heading proportional gain.")
    parser.add_argument("--max-turn-percent", type=float, default=10.0, help="Maximum closed-loop turn motor percent.")
    parser.add_argument("--min-turn-percent", type=float, default=1.5, help="Minimum nonzero closed-loop turn motor percent.")
    parser.add_argument("--control-hz", type=float, default=20.0, help="Motor command publish rate for closed-loop robot tools.")
    parser.add_argument(
        "--max-forward-power-percent",
        type=float,
        default=10.0,
        help="Maximum forward/reverse motor percent allowed for drive_straight, reverse, and visual_servo_object.",
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--runner",
        choices=("qwen", "codex", "scripted-open-vocab", "deterministic"),
        default="qwen",
        help="Actor runner. qwen/codex are model-based; scripted/deterministic are smoke-test runners.",
    )
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--qwen-model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-max-tokens", type=int, default=512)
    parser.add_argument("--prompt-profile", default="baseline")
    parser.add_argument("--actor-rule", action="append", default=[], help="Additional actor prompt rule. Repeatable.")
    parser.add_argument("--critic-rule", action="append", default=[], help="Additional critic prompt rule. Repeatable.")
    parser.add_argument(
        "--critic-mode",
        choices=("auto", "none", "safety", "same-model"),
        default="auto",
    )
    parser.add_argument(
        "--object-drive-detector",
        choices=("florence-mlx", "florence-transformers", "grounding-dino"),
        default="florence-mlx",
    )
    parser.add_argument(
        "--skip-detector-preflight",
        action="store_true",
        help="Skip all Python dependency and detector-load checks for the object-drive detector.",
    )
    parser.add_argument(
        "--skip-detector-load-preflight",
        action="store_true",
        help="Skip detector construction smoke test while still checking required Python modules.",
    )
    parser.add_argument(
        "--detector-load-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for the detector construction smoke test.",
    )
    parser.add_argument("--topomap-memory-map-dir", type=Path, default=None)
    parser.add_argument("--topomap-memory-use-clip", action="store_true")
    parser.add_argument("--topomap-memory-allow-semantic-terms", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--rerun-no-spawn", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.turn_heading_kp < 0:
        parser.error("--turn-heading-kp must be non-negative")
    if args.max_turn_percent <= 0:
        parser.error("--max-turn-percent must be positive")
    if args.min_turn_percent < 0:
        parser.error("--min-turn-percent must be non-negative")
    if args.control_hz <= 0:
        parser.error("--control-hz must be positive")
    if args.max_forward_power_percent <= 0:
        parser.error("--max-forward-power-percent must be positive")
    if args.detector_load_timeout <= 0:
        parser.error("--detector-load-timeout must be positive")
    if args.skip_detector_preflight:
        args.skip_detector_load_preflight = True
    return args


def main() -> int:
    result = run_hardware_harness(parse_args())
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
