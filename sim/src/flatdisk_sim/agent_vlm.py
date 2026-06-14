"""Visual-agent task runner using camera frames, IMU yaw, and motor tools only."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
from typing import Any

from .agent_tools import AgentTools, Observation, make_contact_sheet
from .env import default_zenoh_connect
from .paths import SCRATCH_ROOT
from .protocol import DEFAULT_NAMESPACE


OBJECT_ALIASES = {
    "duck": "yellow duck",
    "yellow duck": "yellow duck",
    "sock": "blue sock",
    "blue sock": "blue sock",
    "book": "green book",
    "green book": "green book",
    "ball": "red ball",
    "red ball": "red ball",
    "cube": "purple cube",
    "purple cube": "purple cube",
}


class HeuristicVisualPlanner:
    def __init__(self, tools: AgentTools) -> None:
        self.tools = tools

    def drive_to_bathroom(self, *, max_steps: int) -> dict[str, Any]:
        observations: list[Observation] = []
        success = False
        reason = "max_steps_exhausted"
        for step in range(max_steps):
            obs = self.tools.observe(label=f"bathroom_step{step:02d}")
            observations.append(obs)
            bathroom = obs.analysis.best("bathroom sign")
            mat = obs.analysis.best("bath mat")
            if mat and mat.confidence > 0.25 and abs(mat.center_offset) < 0.45:
                success = True
                reason = "bath_mat_visible_close"
                break
            if bathroom and bathroom.confidence > 0.08:
                correction = bathroom.center_offset * 24.0
                if abs(correction) > 4.0:
                    self.tools.turn_by_angle(correction, power_percent=9.0)
                self.tools.drive_straight(28.0, 1.2)
                continue

            best = self._scan_for(("bathroom sign", "bath mat"), sweep_deg=30.0, samples=8)
            observations.extend(best["observations"])
            if best["score"] <= 0.02:
                self.tools.turn_by_angle(35.0, power_percent=9.0)
            else:
                offset = best["offset"]
                if abs(offset) > 0.08:
                    self.tools.turn_by_angle(offset * 26.0, power_percent=9.0)
                self.tools.drive_straight(24.0, 0.9)

        self.tools.stop()
        return self._finish("bathroom", success, reason, observations)

    def find_object(self, *, target: str, max_steps: int) -> dict[str, Any]:
        target_name = OBJECT_ALIASES.get(target.lower(), target.lower())
        observations: list[Observation] = []
        success = False
        reason = "max_steps_exhausted"
        for step in range(max_steps):
            obs = self.tools.observe(label=f"find_{target_name.replace(' ', '_')}_{step:02d}")
            observations.append(obs)
            detection = obs.analysis.best(target_name)
            if detection and detection.confidence > 0.48 and abs(detection.center_offset) < 0.6:
                success = True
                reason = f"{target_name}_visible_close"
                break
            if detection and detection.confidence > 0.08:
                correction = detection.center_offset * 28.0
                if abs(correction) > 3.5:
                    self.tools.turn_by_angle(correction, power_percent=9.0)
                self.tools.drive_straight(16.0, 0.55)
                continue

            best = self._scan_for((target_name,), sweep_deg=30.0, samples=12)
            observations.extend(best["observations"])
            if best["score"] <= 0.02:
                reason = f"{target_name}_not_seen"
                continue
            if abs(best["offset"]) > 0.08:
                self.tools.turn_by_angle(best["offset"] * 30.0, power_percent=9.0)
            self.tools.drive_straight(16.0, 0.55)

        self.tools.stop()
        return self._finish(f"find_{target_name.replace(' ', '_')}", success, reason, observations)

    def _scan_for(self, names: tuple[str, ...], *, sweep_deg: float, samples: int) -> dict[str, Any]:
        best_score = -1.0
        best_offset = 0.0
        best_index = 0
        observations: list[Observation] = []
        for index in range(samples):
            obs = self.tools.observe(label=f"scan_{index:02d}")
            observations.append(obs)
            score = obs.analysis.score(*names)
            detection_offsets = [obs.analysis.best(name).center_offset for name in names if obs.analysis.best(name)]
            offset = detection_offsets[0] if detection_offsets else 0.0
            if score > best_score:
                best_score = score
                best_offset = offset
                best_index = index
            if index + 1 < samples:
                self.tools.turn_by_angle(sweep_deg, power_percent=9.0)
        if best_score > 0.02:
            return_degrees = -(samples - 1 - best_index) * sweep_deg
            if abs(return_degrees) > 1.0:
                self.tools.turn_by_angle(return_degrees, power_percent=9.0)
        return {"score": best_score, "offset": best_offset, "index": best_index, "observations": observations}

    def _finish(self, task_name: str, success: bool, reason: str, observations: list[Observation]) -> dict[str, Any]:
        contact_sheet = None
        if observations:
            contact_sheet = make_contact_sheet([obs.path for obs in observations[-8:]], self.tools.run_dir / "contact_sheet.jpg")
        summary = {
            "task": task_name,
            "success": success,
            "reason": reason,
            "observation_count": len(observations),
            "contact_sheet": str(contact_sheet) if contact_sheet else None,
            "run_dir": str(self.tools.run_dir),
        }
        return summary


class OpenAIVisionPlanner:
    """Experimental VLM planner. It keeps the same no-pose tool contract."""

    def __init__(self, tools: AgentTools, *, model: str = "gpt-5.5") -> None:
        self.tools = tools
        self.model = model
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install with `uv sync --extra vlm` to use --planner openai") from exc
        self.client = OpenAI()

    def choose_action(self, obs: Observation, *, objective: str) -> dict[str, Any]:  # pragma: no cover - live API path
        image_b64 = base64.b64encode(obs.path.read_bytes()).decode("ascii")
        prompt = (
            "You are driving a small two-wheel differential-drive robot. "
            "You do not know pose or a map. Use only the camera image and IMU yaw. "
            "Choose one JSON action: turn_by_angle(degrees), drive_straight(power_percent,duration_s), or stop. "
            f"Objective: {objective}. IMU yaw: {obs.yaw_deg:.1f} degrees."
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }
            ],
        )
        text = response.output_text.strip()
        return json.loads(text)


def make_run_dir(task: str, root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"{stamp}_{task}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(summary: dict[str, Any], run_dir: Path) -> Path:
    report = run_dir / "report.md"
    lines = [
        f"# {summary['task']}",
        "",
        f"- Success: `{summary['success']}`",
        f"- Reason: `{summary['reason']}`",
        f"- Observations: `{summary['observation_count']}`",
        f"- Contact sheet: `{summary.get('contact_sheet')}`",
        "",
        "The agent only used camera frames, IMU yaw, and motor tools. No pose or encoder telemetry was exposed.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("bathroom", "find-object"))
    parser.add_argument("--target", default="yellow duck")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--connect", default=default_zenoh_connect())
    parser.add_argument("--run-root", type=Path, default=SCRATCH_ROOT)
    parser.add_argument("--planner", choices=("heuristic", "openai"), default="heuristic")
    parser.add_argument("--openai-model", default="gpt-5.5")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_slug = args.task if args.task == "bathroom" else f"find_{args.target.replace(' ', '_')}"
    run_dir = make_run_dir(task_slug, args.run_root)
    tools = AgentTools(run_dir=run_dir, namespace=args.namespace, connect=args.connect)
    try:
        if args.planner == "openai":
            planner = OpenAIVisionPlanner(tools, model=args.openai_model)
            objective = "drive to the bathroom" if args.task == "bathroom" else f"find {args.target} on the floor"
            observations: list[Observation] = []
            success = False
            reason = "max_steps_exhausted"
            for step in range(args.max_steps):
                obs = tools.observe(label=f"vlm_step{step:02d}")
                observations.append(obs)
                action = planner.choose_action(obs, objective=objective)
                tools.log("vlm_action", action)
                name = action.get("action")
                if name == "turn_by_angle":
                    tools.turn_by_angle(float(action.get("degrees", 0.0)))
                elif name == "drive_straight":
                    tools.drive_straight(float(action.get("power_percent", 16.0)), float(action.get("duration_s", 0.6)))
                elif name == "stop":
                    success = bool(action.get("success", False))
                    reason = str(action.get("reason", "vlm_stop"))
                    break
                else:
                    reason = f"invalid_vlm_action_{name}"
                    break
            summary = HeuristicVisualPlanner(tools)._finish(task_slug, success, reason, observations)
        else:
            planner = HeuristicVisualPlanner(tools)
            if args.task == "bathroom":
                summary = planner.drive_to_bathroom(max_steps=args.max_steps)
            else:
                summary = planner.find_object(target=args.target, max_steps=args.max_steps)
    finally:
        tools.close()

    report = write_report(summary, run_dir)
    print(json.dumps({"summary": summary, "report": str(report)}, indent=2, sort_keys=True))
    return 0 if summary["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
