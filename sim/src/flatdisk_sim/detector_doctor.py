"""Smoke-test open-vocabulary object detectors on saved robot frames."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Protocol

from PIL import Image, ImageDraw

from .paths import REPO_ROOT, SCRATCH_ROOT


SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from object_drive_zenoh import (  # noqa: E402
    Detection,
    build_detector,
    save_debug_overlay,
    select_detection,
)


DETECTOR_DOCTOR_SCHEMA = "flatdisk.open_vocab_detector_doctor.v1"
DETECTOR_CHOICES = ("florence-mlx", "florence-transformers", "grounding-dino")


class Detector(Protocol):
    def detect(self, image: Image.Image, prompt: str) -> tuple[Detection, ...]:
        ...


DetectorFactory = Callable[[argparse.Namespace], Detector]


def run_detector_doctor(
    *,
    image_paths: Iterable[Path],
    prompts: Iterable[str],
    output_dir: Path,
    detector_name: str = "florence-transformers",
    model: str = "mlx-community/Florence-2-base-ft-4bit",
    transformers_model: str = "microsoft/Florence-2-base-ft",
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny",
    grounding_dino_box_threshold: float = 0.25,
    grounding_dino_text_threshold: float = 0.25,
    device: str = "auto",
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_area_fraction: float = 0.75,
    detector_factory: DetectorFactory = build_detector,
) -> dict[str, Any]:
    image_list = [path.expanduser() for path in image_paths]
    prompt_list = [prompt.strip() for prompt in prompts if prompt.strip()]
    if not image_list:
        raise ValueError("at least one --image or --image-glob match is required")
    if not prompt_list:
        raise ValueError("at least one --prompt is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    detector_args = argparse.Namespace(
        detector=detector_name,
        model=model,
        transformers_model=transformers_model,
        grounding_dino_model=grounding_dino_model,
        grounding_dino_box_threshold=grounding_dino_box_threshold,
        grounding_dino_text_threshold=grounding_dino_text_threshold,
        device=device,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    started = time.perf_counter()
    detector = detector_factory(detector_args)
    load_elapsed_s = time.perf_counter() - started

    checks: list[dict[str, Any]] = []
    for image_path in image_list:
        image = Image.open(image_path).convert("RGB")
        for prompt in prompt_list:
            check = _run_one_check(
                detector,
                image=image,
                image_path=image_path,
                prompt=prompt,
                detector_name=detector_name,
                overlay_dir=overlay_dir,
                max_area_fraction=max_area_fraction,
            )
            checks.append(check)

    detection_count = sum(item["detection_count"] for item in checks)
    selected_count = sum(1 for item in checks if item["selected_detection"] is not None)
    report = {
        "schema": DETECTOR_DOCTOR_SCHEMA,
        "created_at": _now(),
        "detector": detector_name,
        "model": _model_for_detector(
            detector_name,
            model=model,
            transformers_model=transformers_model,
            grounding_dino_model=grounding_dino_model,
        ),
        "device": device,
        "image_count": len(image_list),
        "prompt_count": len(prompt_list),
        "check_count": len(checks),
        "detection_count": detection_count,
        "selected_detection_count": selected_count,
        "load_elapsed_s": round(load_elapsed_s, 3),
        "max_area_fraction": max_area_fraction,
        "checks": checks,
        "ready_for_visual_servo": selected_count > 0,
        "recommendation": _recommendation(checks),
    }
    report_path = output_dir / "detector_doctor.json"
    markdown_path = output_dir / "detector_doctor.md"
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return report


def _run_one_check(
    detector: Detector,
    *,
    image: Image.Image,
    image_path: Path,
    prompt: str,
    detector_name: str,
    overlay_dir: Path,
    max_area_fraction: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    try:
        detections = detector.detect(image, prompt)
    except Exception as exc:  # noqa: BLE001 - doctor should report detector failures.
        detections = ()
        error = f"{type(exc).__name__}: {exc}"
    elapsed_s = time.perf_counter() - started
    selected = select_detection(
        detections,
        prompt=prompt,
        image_size=image.size,
        max_area_fraction=max_area_fraction,
    )
    overlay_path = overlay_dir / f"{_safe_stem(image_path.stem)}__{_safe_stem(prompt)}__{detector_name}.jpg"
    _save_detection_overlay(overlay_path, image, detections, selected)
    return {
        "image": str(image_path),
        "image_size": list(image.size),
        "prompt": prompt,
        "elapsed_s": round(elapsed_s, 3),
        "error": error,
        "detection_count": len(detections),
        "detections": [_detection_data(det, image_size=image.size) for det in detections],
        "selected_detection": None if selected is None else _detection_data(selected, image_size=image.size),
        "overlay": str(overlay_path),
    }


def _save_detection_overlay(path: Path, image: Image.Image, detections: tuple[Detection, ...], selected: Detection | None) -> None:
    if len(detections) <= 1:
        save_debug_overlay(path, image, selected, None)
        return
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        color = (48, 144, 255) if detection != selected else (255, 48, 48)
        draw.rectangle(detection.bbox_xyxy, outline=color, width=3)
        label = f"{detection.label}:{detection.score:.2f}"
        draw.text((detection.bbox_xyxy[0] + 4, detection.bbox_xyxy[1] + 4), label, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=90)


def _detection_data(detection: Detection, *, image_size: tuple[int, int] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "bbox_xyxy": [round(value, 3) for value in detection.bbox_xyxy],
        "label": detection.label,
        "score": round(float(detection.score), 4),
        "source": detection.source,
        "raw": detection.raw[:300],
    }
    if image_size is not None:
        data.update(_bbox_geometry(detection.bbox_xyxy, image_size=image_size))
    return data


def _bbox_geometry(bbox_xyxy: tuple[float, float, float, float], *, image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    x0, y0, x1, y1 = bbox_xyxy
    box_width = max(0.0, x1 - x0)
    box_height = max(0.0, y1 - y0)
    edge_tolerance_px = 1.0
    edge_contact = []
    if x0 <= edge_tolerance_px:
        edge_contact.append("left")
    if y0 <= edge_tolerance_px:
        edge_contact.append("top")
    if x1 >= width - edge_tolerance_px:
        edge_contact.append("right")
    if y1 >= height - edge_tolerance_px:
        edge_contact.append("bottom")
    return {
        "bbox_area_fraction": round((box_width * box_height) / max(1.0, float(width * height)), 4),
        "bbox_center_xy_norm": [
            round(((x0 + x1) / 2.0) / max(1.0, float(width)), 4),
            round(((y0 + y1) / 2.0) / max(1.0, float(height)), 4),
        ],
        "bbox_width_fraction": round(box_width / max(1.0, float(width)), 4),
        "bbox_height_fraction": round(box_height / max(1.0, float(height)), 4),
        "bbox_touches_image_edge": bool(edge_contact),
        "bbox_edge_contact": edge_contact,
    }


def _recommendation(checks: list[dict[str, Any]]) -> str:
    selected = [item["selected_detection"] for item in checks if item["selected_detection"] is not None]
    if any(item.get("bbox_touches_image_edge") for item in selected):
        return (
            "Detector produced a selected box that touches the image edge; treat it as a partial grounding hypothesis "
            "and inspect the overlay before visual-servo use."
        )
    if selected:
        return "Detector produced at least one selected box; inspect overlays before enabling long visual-servo rollouts."
    if any(item["error"] for item in checks):
        return "Detector failed on saved frames; fix dependency/model loading before launching visual-servo rollouts."
    return "Detector returned no usable boxes on saved frames; try another open-vocabulary grounding backend or prompt family before rerunning navigation."


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Open-Vocabulary Detector Doctor",
        "",
        f"Detector: `{report['detector']}`",
        f"Model: `{report['model']}`",
        f"Ready for visual servo: `{report['ready_for_visual_servo']}`",
        f"Recommendation: {report['recommendation']}",
        "",
        "| Image | Prompt | Detections | Selected | Error | Overlay |",
        "|---|---|---:|---|---|---|",
    ]
    for item in report["checks"]:
        selected = item["selected_detection"]
        selected_text = ""
        if selected is not None:
            selected_text = f"{selected['label']} {selected['score']:.2f}"
            if selected.get("bbox_touches_image_edge"):
                selected_text += f" edge={','.join(selected.get('bbox_edge_contact') or [])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(item["image"]).name,
                    item["prompt"].replace("|", "\\|"),
                    str(item["detection_count"]),
                    selected_text,
                    (item["error"] or "").replace("|", "\\|"),
                    item["overlay"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _model_for_detector(detector_name: str, *, model: str, transformers_model: str, grounding_dino_model: str) -> str:
    if detector_name == "florence-mlx":
        return model
    if detector_name == "grounding-dino":
        return grounding_dino_model
    return transformers_model


def _safe_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:80] or "item"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _images_from_args(args: argparse.Namespace) -> list[Path]:
    paths = list(args.image or [])
    for pattern in args.image_glob or []:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    return sorted(dict.fromkeys(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--image-glob", action="append", default=[])
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH_ROOT / "detector_doctor")
    parser.add_argument("--detector", choices=DETECTOR_CHOICES, default="florence-transformers")
    parser.add_argument("--model", default="mlx-community/Florence-2-base-ft-4bit")
    parser.add_argument("--transformers-model", default="microsoft/Florence-2-base-ft")
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-area-fraction", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_detector_doctor(
        image_paths=_images_from_args(args),
        prompts=args.prompt,
        output_dir=args.output_dir,
        detector_name=args.detector,
        model=args.model,
        transformers_model=args.transformers_model,
        grounding_dino_model=args.grounding_dino_model,
        grounding_dino_box_threshold=args.grounding_dino_box_threshold,
        grounding_dino_text_threshold=args.grounding_dino_text_threshold,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_area_fraction=args.max_area_fraction,
    )
    print(json.dumps({key: report[key] for key in ("detector", "ready_for_visual_servo", "report_path", "markdown_path")}, indent=2))
    return 0 if report["ready_for_visual_servo"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
