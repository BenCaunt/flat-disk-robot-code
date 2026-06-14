"""Dependency and artifact checks for NoMaD semantic-topomap navigation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


DEFAULT_NOMAD_WEIGHTS = Path("/Users/bencaunt/Downloads/nomad.pth")


def module_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"ok": False, "module": name, "path": None}
    return {"ok": True, "module": name, "path": spec.origin}


def path_status(path: Path, *, kind: str) -> dict[str, Any]:
    exists = path.exists()
    result: dict[str, Any] = {
        "ok": exists,
        "kind": kind,
        "path": str(path),
        "exists": exists,
    }
    if exists and path.is_file():
        result["bytes"] = path.stat().st_size
    return result


def visualnav_repo_status(path: Path) -> dict[str, Any]:
    base = path_status(path, kind="visualnav_repo")
    train_dir = path / "train" / "vint_train"
    deploy_utils = path / "deployment" / "src" / "utils.py"
    base["train_package"] = str(train_dir)
    base["deployment_utils"] = str(deploy_utils)
    base["ok"] = bool(base["exists"] and train_dir.exists())
    return base


def checkpoint_status(path: Path) -> dict[str, Any]:
    status = path_status(path, kind="nomad_checkpoint")
    if not status["ok"]:
        return status
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        status["loadable_with_torch"] = None
        status["note"] = "torch is not installed in this environment"
        return status
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        status["loadable_with_torch"] = True
        if isinstance(checkpoint, dict):
            status["top_level_keys"] = list(checkpoint.keys())[:8]
            status["tensor_count"] = sum(1 for value in checkpoint.values() if hasattr(value, "shape"))
    except Exception as exc:  # noqa: BLE001 - diagnostic command should report failures, not raise.
        status["loadable_with_torch"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    if args.visualnav_repo is not None:
        visualnav_repo = args.visualnav_repo.expanduser().resolve()
        train_dir = visualnav_repo / "train"
        if train_dir.exists() and str(train_dir) not in sys.path:
            sys.path.insert(0, str(train_dir))
    if args.diffusion_policy_repo is not None:
        diffusion_policy_repo = args.diffusion_policy_repo.expanduser().resolve()
        if diffusion_policy_repo.exists() and str(diffusion_policy_repo) not in sys.path:
            sys.path.insert(0, str(diffusion_policy_repo))

    modules = [
        "ai2thor",
        "procthor",
        "torch",
        "torchvision",
        "diffusers",
        "efficientnet_pytorch",
        "einops",
        "diffusion_policy",
        "clip",
        "zenoh",
    ]
    result = {
        "schema": "flatdisk.nomad_topomap.doctor.v1",
        "modules": {name: module_status(name) for name in modules},
        "checkpoint": checkpoint_status(args.checkpoint.expanduser().resolve()),
        "visualnav_repo": (
            None if args.visualnav_repo is None else visualnav_repo_status(args.visualnav_repo.expanduser().resolve())
        ),
        "diffusion_policy_repo": (
            None
            if args.diffusion_policy_repo is None
            else path_status(args.diffusion_policy_repo.expanduser().resolve(), kind="diffusion_policy_repo")
        ),
    }
    required_for_nomad = [
        "torch",
        "torchvision",
        "diffusers",
        "efficientnet_pytorch",
        "einops",
        "diffusion_policy",
    ]
    result["nomad_ready"] = (
        all(result["modules"][name]["ok"] for name in required_for_nomad)
        and result["checkpoint"]["ok"]
        and (result["visualnav_repo"] is not None and result["visualnav_repo"]["ok"])
    )
    result["ithor_ready"] = result["modules"]["ai2thor"]["ok"]
    result["procthor_ready"] = result["modules"]["procthor"]["ok"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_NOMAD_WEIGHTS)
    parser.add_argument("--visualnav-repo", type=Path, default=Path("/tmp/visualnav-transformer"))
    parser.add_argument("--diffusion-policy-repo", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print raw JSON only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_checks(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"iTHOR ready: {result['ithor_ready']}")
        print(f"ProcTHOR ready: {result['procthor_ready']}")
        print(f"NoMaD ready: {result['nomad_ready']}")
        missing = [name for name, status in result["modules"].items() if not status["ok"]]
        if missing:
            print("Missing modules:", ", ".join(missing))
        print(f"checkpoint: {result['checkpoint']['path']} ok={result['checkpoint']['ok']}")
        if result["visualnav_repo"] is not None:
            print(f"visualnav repo: {result['visualnav_repo']['path']} ok={result['visualnav_repo']['ok']}")
    return 0 if result["ithor_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
