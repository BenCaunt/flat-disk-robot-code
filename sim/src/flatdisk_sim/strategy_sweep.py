"""Generate general Qwen strategy-sweep configs for open-vocab navigation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .research_loop import DEFAULT_WARMHUB_REPO, ResearchConfig, _config_to_dict, _parse_variant, _safe_id, load_config


@dataclass(frozen=True)
class StrategyTemplate:
    name: str
    description: str
    prompt_profile: str
    actor_rules: tuple[str, ...]
    critic_rules: tuple[str, ...] = ()
    topomap_memory: bool = False
    object_drive_detector: str | None = None


STRATEGY_TEMPLATES: tuple[StrategyTemplate, ...] = (
    StrategyTemplate(
        name="baseline",
        description="Qwen tool-use baseline with bounded motion, memory, motion strips, and model-based phrase grounding.",
        prompt_profile="baseline",
        actor_rules=(),
    ),
    StrategyTemplate(
        name="frontier_scan",
        description="Exploration-biased strategy that tracks viewpoint coverage before committing to repeated movement.",
        prompt_profile="frontier-scan-v1",
        actor_rules=(
            "Maintain scratchpad coverage notes: recent headings, areas with weak evidence, areas with stronger evidence, and the next viewpoint to test.",
            "When goal evidence is weak, prefer viewpoint diversity using bounded turns or short drives instead of repeating the same servo phrase.",
            "After any movement that changes the view, compare the latest RGB frame with the previous motion strip before selecting the next tool.",
            "Avoid long straight-line commitments unless the latest image provides a plausible free path or clear target evidence.",
        ),
        critic_rules=(
            "Warn when the actor repeats the same search pattern without new visual evidence or a memory-based reason.",
            "Approve bounded viewpoint-seeking actions when they increase visual coverage and preserve the chance to reassess.",
        ),
    ),
    StrategyTemplate(
        name="evidence_exploit",
        description="Exploitation-biased strategy that commits quickly when live visual evidence is strong, otherwise falls back to bounded exploration.",
        prompt_profile="evidence-exploit-v1",
        actor_rules=(
            "Classify the latest visual evidence as strong, partial, or weak before choosing a tool.",
            "When evidence is strong and currently visible, use visual_servo_object with a phrase selected from the latest RGB frame.",
            "When evidence is partial, first turn or move briefly to improve centering rather than declaring success.",
            "When evidence is weak, switch to exploration rather than forcing a final-goal servo phrase.",
        ),
        critic_rules=(
            "Reject stop decisions that do not cite repeated strong live visual evidence.",
            "Warn when exploitation is attempted from weak or stale evidence.",
        ),
    ),
    StrategyTemplate(
        name="recovery_switch",
        description="Failure-recovery strategy that changes tools after stalls, failed grounding, or repeated low-information views.",
        prompt_profile="recovery-switch-v1",
        actor_rules=(
            "Track failed tool choices and stale views in memory_update so the next step does not repeat them blindly.",
            "If a phrase-grounded servo reports failure or no movement, switch to a bounded turn, short drive, or memory query on the next step.",
            "If short drives do not improve visual evidence, switch to rotation or memory lookup instead of continuing forward.",
            "Prefer reversible, bounded actions when uncertainty is high.",
        ),
        critic_rules=(
            "Warn when the actor repeats a failed tool choice without explaining what changed.",
            "Approve switching strategies after clear stalled progress or failed tool feedback.",
        ),
    ),
    StrategyTemplate(
        name="grounding_recovery",
        description="Trace-derived strategy that binds grounding audits to the next action and avoids repeated failed servo prompts.",
        prompt_profile="grounding-recovery-v1",
        actor_rules=(
            "Maintain failed_servo_prompts and failed_viewpoints in memory_update; remove an entry only after a later tool result reports stable grounding.",
            "If the previous grounding_audit says the detector box mismatched or next_prompt_should_change is true, do not reuse the same visual_servo_object prompt on the next step.",
            "After two rejected or no-motion steps from similar views, choose a viewpoint-changing tool or query_topomap_memory before another visual_servo_object call.",
            "Use non-target visible landmarks only as navigation waypoints; state why the landmark move should improve the next observation and reassess immediately afterward.",
            "Treat sparse_detection_coverage and no_detection as weak control evidence, not proof that the robot is moving toward the goal.",
        ),
        critic_rules=(
            "Reject a repeated visual_servo_object prompt when the previous grounding_audit says the prompt should change or the box mismatched.",
            "Warn when the actor narrates a strategy change but the emitted action reuses the same tool arguments.",
            "Approve bounded viewpoint changes or memory queries after repeated unstable phrase grounding.",
        ),
    ),
    StrategyTemplate(
        name="grounding_dino_recovery",
        description="Detector-backend recovery strategy that uses GroundingDINO phrase grounding while keeping the same Qwen tool loop.",
        prompt_profile="grounding-dino-recovery-v1",
        actor_rules=(
            "Use visual_servo_object only for an object instance that is currently visible in the latest RGB frame.",
            "When a plausible candidate is visible but previous detector grounding was sparse or suspicious, call check_object_grounding on a compact phrase before visual_servo_object.",
            "Treat check_object_grounding ready_for_visual_servo=true as only a detector-box hypothesis; inspect the overlay and any grounding_geometry_warning before servoing.",
            "If check_object_grounding reports ready_for_visual_servo=false, grounding_geometry_warning, an edge-clipped box, or an overlay on the wrong region, change viewpoint or choose a different visible phrase instead of servoing.",
            "Prefer compact visible phrases for the configured phrase-grounding backend; if no_detection repeats, change viewpoint instead of adding more descriptors.",
            "After every visual_servo_object call, audit the paired raw/detector strip before deciding whether to trust the movement.",
            "If close foreground or edge-cropped live evidence suggests arrival and the same-goal servo then returns no_detection, choose stop with memory_update.arrival_evidence instead of turning solely to reacquire the detector box.",
            "If the detector box is absent or on the wrong region twice from similar views, choose a viewpoint-changing tool or query_topomap_memory before another servo attempt.",
            "Record detector_backend_grounding_failures in memory_update with the prompt and viewpoint so the next step does not repeat them blindly.",
        ),
        critic_rules=(
            "Reject repeated visual_servo_object calls from the same viewpoint after two no_detection tool results.",
            "Warn when descriptor refinement substitutes for viewpoint change after the detector produced no usable box.",
            "Approve using the configured open-vocabulary detector backend when the latest RGB frame contains a visible candidate instance.",
        ),
        object_drive_detector="grounding-dino",
    ),
    StrategyTemplate(
        name="topomap_memory",
        description="Memory-first strategy that uses CLIP-backed topomap image memory as a non-motion route hint when live evidence is weak.",
        prompt_profile="topomap-memory-v2",
        actor_rules=(
            "When live visual evidence is weak or motion progress stalls, query topomap memory before another movement.",
            "Use returned contact sheets as visual memory hints only; never treat memory results as proof of completion.",
            "After memory lookup, choose a bounded action that is consistent with both the latest RGB frame and the returned image memory.",
            "If memory is unavailable or conflicts with the latest frame, fall back to camera-only exploration.",
        ),
        critic_rules=(
            "Approve memory lookup when uncertainty is high and no strong live target evidence is present.",
            "Warn if memory output is treated as ground-truth success.",
        ),
        topomap_memory=True,
    ),
)


def generate_strategy_config(
    base: ResearchConfig,
    *,
    experiment_id: str,
    objective: str | None = None,
    qwen_endpoint: str | None = None,
    qwen_model: str | None = None,
    object_drive_detector: str | None = None,
    include_topomap: bool = True,
    topomap_memory_map_dir: str = "sim/scratch/semantic_topomaps/{episode}_clip",
    topomap_memory_use_clip: bool = True,
) -> dict[str, Any]:
    inherited = _first_qwen_variant(base)
    variants: list[dict[str, Any]] = []
    for template in STRATEGY_TEMPLATES:
        if template.topomap_memory and not include_topomap:
            continue
        variant: dict[str, Any] = {
            "name": f"qwen_{template.name}",
            "description": template.description,
            "runner": "qwen",
            "prompt_profile": template.prompt_profile,
            "qwen_endpoint": qwen_endpoint or inherited.get("qwen_endpoint"),
            "qwen_model": qwen_model or inherited.get("qwen_model"),
            "qwen_temperature": inherited.get("qwen_temperature", 0.0),
            "qwen_max_tokens": _qwen_completion_budget(inherited.get("qwen_max_tokens", 512)),
            "object_drive_detector": object_drive_detector or template.object_drive_detector or inherited.get("object_drive_detector"),
            "actor_rules": list(template.actor_rules),
            "critic_rules": list(template.critic_rules),
            "critic_mode": "none",
        }
        if template.topomap_memory:
            variant.update(
                {
                    "topomap_memory_map_dir": topomap_memory_map_dir,
                    "topomap_memory_use_clip": topomap_memory_use_clip,
                    "topomap_memory_allow_semantic_terms": False,
                }
            )
        variants.append(variant)

    payload = _config_to_dict(base)
    payload.update(
        {
            "experiment_id": _safe_id(experiment_id),
            "objective": objective or f"Generated general Qwen strategy sweep based on {base.experiment_id}.",
            "warmhub_repo": base.warmhub_repo or DEFAULT_WARMHUB_REPO,
            "variants": variants,
            "strategy_sweep": {
                "schema": "flatdisk.nav_strategy_sweep_config.v1",
                "source_experiment_id": base.experiment_id,
                "strategy_count": len(variants),
                "no_static_object_or_color_examples": True,
                "semantic_topomap_terms_allowed": False,
            },
        }
    )
    return payload


def _first_qwen_variant(base: ResearchConfig) -> dict[str, Any]:
    for variant in base.variants:
        if variant.runner == "qwen":
            return asdict(variant)
    fallback = asdict(_parse_variant({"name": "qwen_baseline", "runner": "qwen"}))
    return fallback


def _qwen_completion_budget(value: Any) -> int:
    try:
        inherited = int(value)
    except (TypeError, ValueError):
        inherited = 512
    return max(1024, inherited)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--qwen-endpoint", default=None)
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--object-drive-detector", default=None)
    parser.add_argument("--exclude-topomap", action="store_true")
    parser.add_argument("--topomap-memory-map-dir", default="sim/scratch/semantic_topomaps/{episode}_clip")
    parser.add_argument("--no-topomap-clip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_config(args.base_config)
    payload = generate_strategy_config(
        base,
        experiment_id=args.experiment_id,
        objective=args.objective,
        qwen_endpoint=args.qwen_endpoint,
        qwen_model=args.qwen_model,
        object_drive_detector=args.object_drive_detector,
        include_topomap=not args.exclude_topomap,
        topomap_memory_map_dir=args.topomap_memory_map_dir,
        topomap_memory_use_clip=not args.no_topomap_clip,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variant_count": len(payload["variants"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
