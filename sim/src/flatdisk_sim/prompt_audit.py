"""Prompt auditing helpers shared by THOR evaluators."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


FORBIDDEN_PRIVILEGED_TOKENS = {
    "distance_m",
    "hidden_score",
    "nearest_target",
    "object_metadata",
    "objects",
    "pose",
    "scene",
    "success_radius",
    "target_pose",
}

STATIC_CONTEXT_FORBIDDEN_TERMS = {
    "armchair",
    "bed",
    "blue",
    "brown",
    "chair",
    "doorway",
    "green",
    "grey",
    "gray",
    "mug",
    "orange",
    "pink",
    "purple",
    "red",
    "sofa",
    "table",
    "toilet",
    "white",
    "yellow",
}


def audit_prompts(prompt_dir: Path) -> dict[str, Any]:
    checked = 0
    leaks: list[str] = []
    static_checked = 0
    static_forbidden: list[str] = []
    for path in sorted(prompt_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        text = raw_text.lower()
        checked += 1
        for token in FORBIDDEN_PRIVILEGED_TOKENS:
            if token in text:
                leaks.append(f"{path.name}:{token}")
        static_context = _extract_static_context(raw_text)
        if static_context is None:
            continue
        static_checked += 1
        static_text = _canonical_static_text(static_context).lower()
        for term in sorted(STATIC_CONTEXT_FORBIDDEN_TERMS):
            if re.search(rf"\b{re.escape(term)}\b", static_text):
                static_forbidden.append(f"{path.name}:{term}")
    return {
        "checked_prompt_count": checked,
        "forbidden_tokens_found": leaks,
        "static_context_checked_count": static_checked,
        "static_context_forbidden_terms_found": static_forbidden,
        "no_hardcoded_labels_or_colors": not static_forbidden,
        "prompt_audit_passed": not leaks and not static_forbidden,
    }


def _extract_static_context(prompt_text: str) -> str | None:
    marker = "STATIC_HARNESS_CONTEXT\n"
    if marker not in prompt_text:
        return None
    static_and_rest = prompt_text.split(marker, 1)[1]
    end_marker = "\n\nDYNAMIC_TASK_STATE"
    if end_marker not in static_and_rest:
        return None
    return static_and_rest.split(end_marker, 1)[0]


def _canonical_static_text(static_context: str) -> str:
    try:
        parsed = json.loads(static_context)
    except json.JSONDecodeError:
        return static_context
    return json.dumps(parsed, sort_keys=True, default=str)
