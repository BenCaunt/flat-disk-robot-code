"""Run a text-goal navigation policy competition in AI2-THOR."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

from .evaluate_text_goals import ThorEpisodeSpec, run_episode
from .paths import SCRATCH_ROOT
from .policy_registry import policy_registry


@dataclass(frozen=True)
class CompetitionSuite:
    name: str
    episodes: tuple[ThorEpisodeSpec, ...]


def competition_suites() -> dict[str, CompetitionSuite]:
    return {
        "dev": CompetitionSuite(
            name="dev",
            episodes=(
                ThorEpisodeSpec(
                    name="dev_bedroom_bed",
                    scene="FloorPlan301",
                    prompt="Drive to the bed in the bedroom.",
                    target_types=("Bed",),
                    success_radius_m=0.55,
                    max_steps=18,
                ),
                ThorEpisodeSpec(
                    name="dev_living_room_sofa",
                    scene="FloorPlan201",
                    prompt="Drive to the sofa in the living room.",
                    target_types=("Sofa",),
                    success_radius_m=0.55,
                    max_steps=18,
                ),
            ),
        ),
        "heldout": CompetitionSuite(
            name="heldout",
            episodes=(
                ThorEpisodeSpec(
                    name="heldout_living_room_sofa",
                    scene="FloorPlan202",
                    prompt="Drive to the sofa in the living room.",
                    target_types=("Sofa",),
                    success_radius_m=0.55,
                    max_steps=18,
                ),
                ThorEpisodeSpec(
                    name="heldout_bedroom_bed",
                    scene="FloorPlan302",
                    prompt="Drive to the bed in the bedroom.",
                    target_types=("Bed",),
                    success_radius_m=0.55,
                    max_steps=18,
                ),
                ThorEpisodeSpec(
                    name="heldout_bathroom_toilet",
                    scene="FloorPlan403",
                    prompt="Drive to the toilet in the bathroom.",
                    target_types=("Toilet",),
                    success_radius_m=0.55,
                    max_steps=18,
                ),
            ),
        ),
    }


def build_competition_suite(name: str, *, random_seed: int) -> CompetitionSuite:
    if name != "random":
        return competition_suites()[name]
    rng = random.Random(random_seed)
    living_scene = rng.choice([scene for scene in range(203, 231) if scene not in {201, 202}])
    bedroom_scene = rng.choice([scene for scene in range(303, 331) if scene not in {301, 302}])
    bathroom_scene = rng.choice([scene for scene in range(404, 431) if scene not in {402, 403}])
    return CompetitionSuite(
        name=f"random_seed_{random_seed}",
        episodes=(
            ThorEpisodeSpec(
                name=f"random_living_room_sofa_{living_scene}",
                scene=f"FloorPlan{living_scene}",
                prompt="Drive to the sofa in the living room.",
                target_types=("Sofa",),
                success_radius_m=0.55,
                max_steps=18,
            ),
            ThorEpisodeSpec(
                name=f"random_bedroom_bed_{bedroom_scene}",
                scene=f"FloorPlan{bedroom_scene}",
                prompt="Drive to the bed in the bedroom.",
                target_types=("Bed",),
                success_radius_m=0.55,
                max_steps=18,
            ),
            ThorEpisodeSpec(
                name=f"random_bathroom_toilet_{bathroom_scene}",
                scene=f"FloorPlan{bathroom_scene}",
                prompt="Drive to the toilet in the bathroom.",
                target_types=("Toilet",),
                success_radius_m=0.55,
                max_steps=18,
            ),
        ),
    )


def run_competition(args: argparse.Namespace) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir / stamp
    output_root.mkdir(parents=True, exist_ok=True)
    suite = build_competition_suite(args.suite, random_seed=args.random_seed)
    policy_summaries: list[dict[str, Any]] = []
    episode_results: list[dict[str, Any]] = []

    for policy_index, policy_name in enumerate(args.policies):
        policy_root = output_root / policy_name
        policy_root.mkdir(parents=True, exist_ok=True)
        policy_started = time.perf_counter()
        summaries: list[dict[str, Any]] = []
        for episode_index, spec in enumerate(suite.episodes):
            port = _policy_port(args.base_port, policy_index, episode_index)
            started = time.perf_counter()
            try:
                summary = run_episode(
                    spec,
                    output_root=policy_root,
                    port=port,
                    policy_name=policy_name,
                    model=args.model,
                    render_width=args.render_width,
                    render_height=args.render_height,
                    args=args,
                )
                summary["wall_clock_s"] = round(time.perf_counter() - started, 3)
            except Exception as exc:
                summary = _failed_episode_summary(policy_name, spec, started, exc, policy_root)
            summaries.append(summary)
            episode_results.append(summary)
        policy_elapsed_s = round(time.perf_counter() - policy_started, 3)
        policy_summaries.append(_policy_score(policy_name, summaries, policy_elapsed_s))

    leaderboard = sorted(policy_summaries, key=lambda item: (-item["success_count"], item["wall_clock_s"], item["mean_final_distance_m"]))
    aggregate = {
        "suite": suite.name,
        "policies": args.policies,
        "leaderboard": leaderboard,
        "episode_results": episode_results,
        "output_root": str(output_root),
    }
    report = write_competition_report(aggregate, output_root)
    aggregate["report"] = str(report)
    (output_root / "competition_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate


def write_competition_report(aggregate: dict[str, Any], output_root: Path) -> Path:
    report = output_root / "competition_report.md"
    lines = [
        "# Text-Goal Policy Competition",
        "",
        f"Suite: `{aggregate['suite']}`",
        "",
        "Scoring: rank by correctness first, then lower policy wall-clock time, then lower mean final distance.",
        "Policy input allowlist for every competitor: camera frame, IMU yaw, text goal, and recent policy actions.",
        "",
        "## Leaderboard",
        "",
        "| Rank | Policy | Successes | Episodes | Wall clock (s) | Mean final distance (m) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(aggregate["leaderboard"], start=1):
        lines.append(
            f"| {rank} | {row['policy']} | {row['success_count']} | {row['episode_count']} | "
            f"{row['wall_clock_s']:.3f} | {row['mean_final_distance_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            "| Policy | Episode | Scene | Success | Steps | Wall clock (s) | Final distance (m) | Camera frames | Progress |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in aggregate["episode_results"]:
        lines.append(
            f"| {row['policy']} | {row['episode']} | {row['scene']} | {row['success']} | "
            f"{row['step_count']} | {row['wall_clock_s']:.3f} | {row['final_distance_m']:.3f} | "
            f"{row.get('camera_contact_sheet') or ''} | {row.get('progress_contact_sheet') or ''} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "text_goal_competition")
    parser.add_argument("--suite", choices=sorted([*competition_suites(), "random"]), default="heldout")
    parser.add_argument("--random-seed", type=int, default=20260608)
    parser.add_argument("--policies", nargs="+", choices=sorted(policy_registry()), default=["control_vlm"])
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-port", type=int, default=0, help="0 chooses a free local port per episode.")
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    aggregate = run_competition(parse_args())
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    best = aggregate["leaderboard"][0] if aggregate["leaderboard"] else None
    return 0 if best and best["success_count"] > 0 else 2


def _policy_port(base_port: int, policy_index: int, episode_index: int) -> int:
    if base_port == 0:
        from .evaluate_text_goals import _find_free_port

        return _find_free_port()
    return base_port + policy_index * 100 + episode_index


def _policy_score(policy_name: str, summaries: list[dict[str, Any]], wall_clock_s: float) -> dict[str, Any]:
    distances = [float(summary["final_distance_m"]) for summary in summaries if summary["final_distance_m"] != float("inf")]
    mean_distance = sum(distances) / len(distances) if distances else float("inf")
    return {
        "policy": policy_name,
        "success_count": sum(1 for summary in summaries if summary["success"]),
        "episode_count": len(summaries),
        "wall_clock_s": wall_clock_s,
        "mean_final_distance_m": round(mean_distance, 4),
    }


def _failed_episode_summary(
    policy_name: str,
    spec: ThorEpisodeSpec,
    started: float,
    exc: Exception,
    policy_root: Path,
) -> dict[str, Any]:
    run_dir = policy_root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "policy": policy_name,
        "episode": spec.name,
        "scene": spec.scene,
        "prompt": spec.prompt,
        "target_types": spec.target_types,
        "success": False,
        "reason": f"exception:{type(exc).__name__}:{exc}",
        "final_distance_m": float("inf"),
        "nearest_target": None,
        "step_count": 0,
        "steps": [],
        "camera_contact_sheet": None,
        "progress_contact_sheet": None,
        "last_progress_frame": None,
        "hidden_log_dir": None,
        "namespace": None,
        "run_dir": str(run_dir),
        "wall_clock_s": round(time.perf_counter() - started, 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
