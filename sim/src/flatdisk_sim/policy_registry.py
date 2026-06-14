"""Policy registry for text-goal navigation competitions."""

from __future__ import annotations

import argparse
from typing import Callable

from .text_goal_policy_core import TextGoalPolicy


PolicyFactory = Callable[[argparse.Namespace], TextGoalPolicy]


def build_policy(name: str, args: argparse.Namespace) -> TextGoalPolicy:
    registry = policy_registry()
    try:
        return registry[name](args)
    except KeyError as exc:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(registry)}") from exc


def policy_registry() -> dict[str, PolicyFactory]:
    from .policies.control_vlm import ControlVlmPolicy
    from .policies.hf_scout import HfScoutPolicy
    from .policies.memory_vlm import MemoryVlmPolicy
    from .policies.sprinter import SprinterPolicy

    return {
        "control_vlm": lambda args: ControlVlmPolicy(model=args.model),
        "hf_scout": lambda _args: HfScoutPolicy(),
        "memory_vlm": lambda args: MemoryVlmPolicy(model=args.model),
        "sprinter": lambda _args: SprinterPolicy(),
    }
