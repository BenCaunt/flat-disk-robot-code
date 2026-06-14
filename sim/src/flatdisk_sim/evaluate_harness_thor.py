"""Evaluate the LLM harness through the Zenoh AI2-THOR simulator.

This is the closer-to-real-robot harness evaluation path: the policy is a
``HarnessSession`` connected to ``AgentTools`` and receives only RGB camera
frames, IMU yaw, tool results, and memory. Hidden THOR pose/object data stays in
an evaluator-only directory and is used only for scoring and progress renders.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

from .agent_tools import AgentTools, make_contact_sheet
from .evaluate_text_goals import (
    ThorEpisodeSpec,
    _find_free_port,
    _latest_hidden_pose,
    _render_hidden_progress,
    _score_hidden_distance,
    _start_bridge,
    _terminate_bridge,
    _wait_for_first_observation,
    _wait_for_hidden_log,
    default_episodes,
)
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
from .paths import SCRATCH_ROOT
from .prompt_audit import audit_prompts


def run_thor_harness_episode(
    spec: ThorEpisodeSpec,
    *,
    output_root: Path,
    port: int,
    model: str,
    reasoning_effort: str,
    live_codex: bool,
    runner: str = "qwen",
    qwen_endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
    qwen_model: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    qwen_temperature: float = 0.0,
    qwen_max_tokens: int = 512,
    object_drive_detector: str = "florence-mlx",
    topomap_memory_map_dir: Path | None = None,
    topomap_memory_use_clip: bool = False,
    topomap_memory_allow_semantic_terms: bool = False,
    prompt_profile: str = "baseline",
    actor_rules: tuple[str, ...] = (),
    critic_rules: tuple[str, ...] = (),
    critic_mode: str = "auto",
    render_width: int,
    render_height: int,
    rerun: bool,
    max_steps: int | None = None,
    success_radius_m: float | None = None,
) -> dict[str, Any]:
    if max_steps is not None:
        spec = replace(spec, max_steps=max_steps)
    if success_radius_m is not None:
        spec = replace(spec, success_radius_m=success_radius_m)
    run_dir = output_root / spec.name
    policy_dir = run_dir / "policy"
    hidden_log_dir = run_dir / "evaluator_hidden"
    policy_dir.mkdir(parents=True, exist_ok=True)
    hidden_log_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"tcp/127.0.0.1:{port}"
    namespace = f"flatdisk/harness-eval/{int(time.time())}/{spec.name}"
    sim_proc = _start_bridge(
        scene=spec.scene,
        namespace=namespace,
        listen=endpoint,
        log_dir=hidden_log_dir,
        render_width=render_width,
        render_height=render_height,
        start_yaw_deg=spec.start_yaw_deg,
    )
    tools: AgentTools | None = None
    session: HarnessSession | None = None
    steps: list[dict[str, Any]] = []
    hidden_snapshots: list[dict[str, Any]] = []
    progress_paths: list[Path] = []
    success = False
    reason = "max_steps_exhausted"
    started = time.perf_counter()
    try:
        _wait_for_hidden_log(hidden_log_dir, timeout_s=20.0)
        tools = AgentTools(
            run_dir=policy_dir,
            namespace=namespace,
            connect=endpoint,
            object_drive_detector=object_drive_detector,
            topomap_memory_map_dir=topomap_memory_map_dir,
            topomap_memory_use_clip=topomap_memory_use_clip,
            topomap_memory_allow_semantic_terms=topomap_memory_allow_semantic_terms,
        )
        _wait_for_first_observation(tools)
        runner_name = "codex" if live_codex else runner
        actor = _build_actor(
            runner_name,
            model=model,
            reasoning_effort=reasoning_effort,
            policy_dir=policy_dir,
            qwen_endpoint=qwen_endpoint,
            qwen_model=qwen_model,
            qwen_temperature=qwen_temperature,
            qwen_max_tokens=qwen_max_tokens,
            object_drive_detector=object_drive_detector,
        )
        resolved_critic_mode = _resolve_critic_mode(runner_name, critic_mode)
        critic = _build_critic(runner_name, actor, mode=resolved_critic_mode)
        rerun_logger = None
        if rerun:
            rerun_logger = HarnessRerunLogger(
                recording_id=f"flatdisk_harness_thor_{spec.name}",
                save_path=policy_dir / "harness.rrd",
                spawn=False,
            )
        session = HarnessSession(
            config=HarnessConfig(
                run_dir=policy_dir,
                model=model,
                reasoning_effort=reasoning_effort,
                prompt_profile=prompt_profile,
                actor_rules=actor_rules,
                critic_rules=critic_rules,
                critic_mode=resolved_critic_mode,
                max_steps=spec.max_steps,
                rerun_enabled=rerun,
            ),
            tools=tools,
            actor=actor,
            critic=critic,
            rerun_logger=rerun_logger,
        )
        session.start_goal(spec.prompt)
        for step_index in range(spec.max_steps):
            hidden = _latest_hidden_pose(hidden_log_dir)
            pre_score = _score_hidden_distance(hidden, spec)
            hidden_snapshots.append(hidden)
            if pre_score["success"]:
                success = True
                reason = "hidden_evaluator_goal_reached_before_action"
                break

            memory_record = session.run_auto_step()
            post_hidden = _latest_hidden_pose(hidden_log_dir)
            post_score = _score_hidden_distance(post_hidden, spec)
            progress_path = _render_hidden_progress(
                hidden_snapshots + [post_hidden],
                spec,
                run_dir / "progress" / f"{step_index + 1:03d}.png",
                title=f"{spec.name}: harness step {step_index + 1}",
            )
            progress_paths.append(progress_path)
            step_record = {
                "step": step_index,
                "harness_memory_record": memory_record,
                "hidden_score_for_evaluator_only": post_score,
            }
            steps.append(step_record)
            if post_score["success"]:
                success = True
                reason = "hidden_evaluator_goal_reached"
                break
            if session.mode != "auto":
                reason = f"harness_mode_{session.mode}_before_hidden_success"
                break
    finally:
        if session is not None:
            session.close()
        elif tools is not None:
            tools.close()
        _terminate_bridge(sim_proc)
    elapsed_s = time.perf_counter() - started
    final_hidden = _latest_hidden_pose(hidden_log_dir)
    final_score = _score_hidden_distance(final_hidden, spec)
    frame_paths = sorted((policy_dir / "frames").glob("*.jpg"))[-8:]
    contact_sheet = make_contact_sheet(frame_paths, policy_dir / "camera_contact_sheet.jpg") if frame_paths else None
    progress_sheet = make_contact_sheet(progress_paths[-8:], run_dir / "progress_contact_sheet.jpg") if progress_paths else None
    prompt_audit = audit_prompts(policy_dir / "prompts")
    rerun_path = policy_dir / "harness.rrd"
    summary = {
        "episode": spec.name,
        "scene": spec.scene,
        "prompt": spec.prompt,
        "target_types": spec.target_types,
        "success_radius_m": spec.success_radius_m,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "runner": "codex" if live_codex else runner,
        "live_codex": live_codex,
        "prompt_profile": prompt_profile,
        "actor_rules": list(actor_rules),
        "critic_rules": list(critic_rules),
        "critic_mode": resolved_critic_mode,
        "topomap_memory_map_dir": str(topomap_memory_map_dir) if topomap_memory_map_dir is not None else None,
        "topomap_memory_use_clip": topomap_memory_use_clip,
        "topomap_memory_allow_semantic_terms": topomap_memory_allow_semantic_terms,
        "success": success,
        "reason": reason,
        "final_distance_m": final_score["distance_m"],
        "nearest_target": final_score["nearest_target"],
        "step_count": len(steps),
        "wall_clock_s": round(elapsed_s, 3),
        "steps": steps,
        "policy_input_allowlist": ["camera frame", "previous motion strip", "camera-derived summary", "imu yaw", "tool result", "memory"],
        "evaluator_only_dir": str(hidden_log_dir),
        "policy_dir": str(policy_dir),
        "namespace": namespace,
        "camera_contact_sheet": str(contact_sheet) if contact_sheet else None,
        "progress_contact_sheet": str(progress_sheet) if progress_sheet else None,
        "last_progress_frame": str(progress_paths[-1]) if progress_paths else None,
        "rerun_path": str(rerun_path) if rerun_path.exists() else None,
        "prompt_audit": prompt_audit,
        "run_dir": str(run_dir),
    }
    (run_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _build_actor(
    runner: str,
    *,
    model: str,
    reasoning_effort: str,
    policy_dir: Path,
    qwen_endpoint: str,
    qwen_model: str,
    qwen_temperature: float,
    qwen_max_tokens: int,
    object_drive_detector: str,
) -> Any:
    if runner == "codex":
        return CodexExecRunner(model=model, reasoning_effort=reasoning_effort, cwd=policy_dir)
    if runner == "qwen":
        return OpenAICompatibleVisionRunner(
            model=qwen_model,
            endpoint=qwen_endpoint,
            temperature=qwen_temperature,
            max_tokens=qwen_max_tokens,
        )
    if runner == "scripted-open-vocab":
        return ScriptedOpenVocabRunner(visual_servo_detector=object_drive_detector)
    if runner == "fast-wall-clock":
        from .agent_candidates.fast_wall_clock import FastWallClockActor

        return FastWallClockActor()
    if runner == "fast-demo":
        from .agent_candidates.fast_wall_clock import FastWallClockActor

        return FastWallClockActor(allow_stop=False, stale_drive_limit=5)
    if runner == "deterministic":
        return DeterministicHarnessRunner()
    raise ValueError(f"unknown runner: {runner}")


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
    if runner in {"fast-wall-clock", "fast-demo"}:
        from .agent_candidates.fast_wall_clock import FastWallClockCritic

        return FastWallClockCritic()
    if mode != "safety":
        raise ValueError(f"unknown critic mode: {mode}")
    return SafetyCriticRunner()


def write_report(summaries: list[dict[str, Any]], output_root: Path) -> Path:
    report = output_root / "harness_thor_report.md"
    lines = [
        "# THOR LLM Harness Evaluation",
        "",
        "Policy input allowlist: camera frame attachment, camera-derived summary, IMU yaw, tool results, and memory.",
        "Evaluator-only data: hidden THOR pose/object metadata stored outside the policy directory.",
        "",
        "| Episode | Scene | Success | Reason | Steps | Wall clock (s) | Success radius (m) | Final distance (m) | Prompt leaks | Camera | Progress | Rerun |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for summary in summaries:
        leaks = ", ".join(summary["prompt_audit"]["forbidden_tokens_found"]) or "none"
        lines.append(
            f"| {summary['episode']} | {summary['scene']} | {summary['success']} | {summary['reason']} | "
            f"{summary['step_count']} | {summary['wall_clock_s']:.3f} | {summary['success_radius_m']:.3f} | "
            f"{summary['final_distance_m']:.3f} | "
            f"{leaks} | {summary.get('camera_contact_sheet') or ''} | "
            f"{summary.get('progress_contact_sheet') or ''} | {summary.get('rerun_path') or ''} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    episodes = sorted(default_episodes())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "thor_llm_harness_eval")
    parser.add_argument("--episodes", nargs="+", choices=episodes, default=episodes)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--live-codex", action="store_true")
    parser.add_argument(
        "--runner",
        choices=("deterministic", "scripted-open-vocab", "fast-wall-clock", "fast-demo", "qwen", "codex"),
        default="qwen",
        help="Actor runner. qwen/codex are model-based; deterministic/scripted/fast runners are explicit smoke-test paths.",
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
        help="Critic selection. auto uses no critic for Qwen, same-model for Codex, and safety for scripted baselines.",
    )
    parser.add_argument(
        "--object-drive-detector",
        choices=("florence-mlx", "florence-transformers", "grounding-dino"),
        default="florence-mlx",
        help="Detector used by the visual_servo_object tool.",
    )
    parser.add_argument("--topomap-memory-map-dir", type=Path, default=None, help="Optional semantic topomap directory for query_topomap_memory.")
    parser.add_argument("--topomap-memory-use-clip", action="store_true", help="Use CLIP text/image embeddings for topomap goal matching.")
    parser.add_argument(
        "--topomap-memory-allow-semantic-terms",
        action="store_true",
        help="Allow saved topomap semantic terms for goal matching. Keep off for strict non-privileged policy runs.",
    )
    parser.add_argument("--base-port", type=int, default=0, help="0 chooses a free local port per episode.")
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=None, help="Override each episode's default step cap.")
    parser.add_argument(
        "--success-radius-m",
        type=float,
        default=None,
        help="Override the hidden evaluator success radius in meters.",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / stamp
    output_root.mkdir(parents=True, exist_ok=True)
    episode_map = default_episodes()
    summaries: list[dict[str, Any]] = []
    for index, name in enumerate(args.episodes):
        port = _find_free_port() if args.base_port == 0 else args.base_port + index
        summaries.append(
            run_thor_harness_episode(
                episode_map[name],
                output_root=output_root,
                port=port,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                live_codex=args.live_codex,
                runner=args.runner,
                qwen_endpoint=args.qwen_endpoint,
                qwen_model=args.qwen_model,
                qwen_temperature=args.qwen_temperature,
                qwen_max_tokens=args.qwen_max_tokens,
                object_drive_detector=args.object_drive_detector,
                topomap_memory_map_dir=args.topomap_memory_map_dir,
                topomap_memory_use_clip=args.topomap_memory_use_clip,
                topomap_memory_allow_semantic_terms=args.topomap_memory_allow_semantic_terms,
                prompt_profile=args.prompt_profile,
                actor_rules=tuple(args.actor_rule),
                critic_rules=tuple(args.critic_rule),
                critic_mode=args.critic_mode,
                render_width=args.render_width,
                render_height=args.render_height,
                rerun=args.rerun,
                max_steps=args.max_steps,
                success_radius_m=args.success_radius_m,
            )
        )
    aggregate = {
        "success_count": sum(1 for summary in summaries if summary["success"]),
        "episode_count": len(summaries),
        "mean_wall_clock_s": round(sum(summary["wall_clock_s"] for summary in summaries) / max(1, len(summaries)), 3),
        "summaries": summaries,
        "output_root": str(output_root),
    }
    (output_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = write_report(summaries, output_root)
    print(json.dumps({"report": str(report), **aggregate}, indent=2, sort_keys=True))
    return 0 if aggregate["success_count"] == aggregate["episode_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
