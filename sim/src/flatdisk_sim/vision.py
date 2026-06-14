"""Image-only frame summaries used by visual agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Detection:
    name: str
    area_fraction: float
    center_offset: float
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class FrameAnalysis:
    detections: tuple[Detection, ...]
    brightness_center: float

    def best(self, name: str) -> Detection | None:
        matches = [d for d in self.detections if d.name == name]
        if not matches:
            return None
        return max(matches, key=lambda det: det.confidence)

    def score(self, *names: str) -> float:
        return sum((self.best(name).confidence if self.best(name) else 0.0) for name in names)


def analyze_image_path(path: Path) -> FrameAnalysis:
    return analyze_image(Image.open(path).convert("RGB"))


def analyze_image(image: Image.Image) -> FrameAnalysis:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    _, width, _ = arr.shape
    center = arr[:, width // 3 : width * 2 // 3, :]
    brightness_center = float(center.mean() / 255.0)
    return FrameAnalysis(detections=(), brightness_center=brightness_center)
