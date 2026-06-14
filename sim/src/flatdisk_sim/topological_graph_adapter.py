"""Convert saved Rerun topological graphs into NoMaD semantic topomaps."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .semantic_topomap import SemanticTopomap, appearance_descriptor


GRAPH_SCHEMA = "flatdisk_topological_graph.v1"


def convert_topological_graph(
    graph_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    clean: bool = False,
    bidirectional_temporal: bool = False,
) -> SemanticTopomap:
    graph_dir = Path(graph_dir).expanduser().resolve()
    graph_path = graph_dir / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if graph.get("schema") != GRAPH_SCHEMA:
        raise RuntimeError(f"Unsupported topological graph schema in {graph_path}: {graph.get('schema')!r}")

    output_dir = graph_dir / "nomad_semantic_topomap" if output_dir is None else Path(output_dir).expanduser()
    output_dir = output_dir.resolve()
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    nodes: list[dict[str, Any]] = []
    descriptors: list[np.ndarray] = []
    for node in graph.get("nodes", []):
        node_id = str(node["id"])
        image = _node_image(graph_dir, node)
        image_rel = f"images/{node_id}.jpg"
        image.save(output_dir / image_rel, format="JPEG", quality=92)
        descriptors.append(appearance_descriptor(image))
        nodes.append(
            {
                "id": node_id,
                "image": image_rel,
                "pose": _node_pose(node),
                "terms": _node_terms(node),
                "visible_object_count": 0,
                "source_graph_node": {
                    key: node.get(key)
                    for key in (
                        "dataset",
                        "trajectory_id",
                        "trajectory_label",
                        "recording",
                        "recording_name",
                        "frame_index",
                        "trajectory_sample_index",
                        "room_id",
                    )
                    if key in node
                },
            }
        )
    if not nodes:
        raise RuntimeError(f"{graph_path} has no nodes")

    node_ids = {str(node["id"]) for node in nodes}
    edges: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source not in node_ids or target not in node_ids:
            continue
        kind = str(edge.get("kind", "edge"))
        converted: dict[str, Any] = {
            "id": str(edge.get("id") or f"e_{kind}_{source}_{target}"),
            "source": source,
            "target": target,
            "kind": kind,
            "enabled": bool(edge.get("enabled", True)),
            "bidirectional": bool(kind != "temporal" or bidirectional_temporal),
            "weight": float(edge.get("weight", 1.0)),
        }
        for key in ("score", "global_score", "orb_score", "inliers", "good_matches", "trajectory_id", "room_id"):
            if key in edge:
                converted[key] = edge[key]
        edges.append(converted)

    descriptors_matrix = np.vstack(descriptors).astype(np.float32)
    metadata = {
        "builder": "topological_graph_adapter",
        "source_schema": GRAPH_SCHEMA,
        "source_graph_dir": str(graph_dir),
        "source_graph": str(graph_path),
        "converted_unix": time.time(),
        "graph_metadata": graph.get("metadata", {}),
        "runtime_goal_hint": "Use --goal node:<node_id> to drive to a selected graph frame.",
        "bidirectional_temporal": bidirectional_temporal,
    }
    topomap = SemanticTopomap(
        map_dir=output_dir,
        metadata=metadata,
        nodes=nodes,
        edges=edges,
        descriptors=descriptors_matrix,
    )
    topomap.save()
    _write_run_notes(graph_dir, output_dir)
    return topomap


def _node_image(graph_dir: Path, node: dict[str, Any]) -> Image.Image:
    thumb = node.get("thumb")
    if not isinstance(thumb, str) or not thumb:
        raise RuntimeError(f"node {node.get('id')} is missing a thumbnail path")
    path = Path(thumb)
    if not path.is_absolute():
        path = graph_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _node_pose(node: dict[str, Any]) -> dict[str, float]:
    layout = node.get("layout") if isinstance(node.get("layout"), dict) else {}
    x = float(layout.get("x", node.get("trajectory_index", 0.0)))
    z = float(layout.get("y", node.get("trajectory_sample_index", 0.0)))
    return {"x": round(x, 6), "y": 0.0, "z": round(z, 6), "yaw_rad": 0.0, "yaw_deg": 0.0}


def _node_terms(node: dict[str, Any]) -> list[str]:
    values: list[str] = [str(node.get("id", ""))]
    for key in ("dataset", "trajectory_id", "trajectory_label", "recording_name", "room_id"):
        value = node.get(key)
        if isinstance(value, str) and value:
            values.append(value)
    if node.get("frame_index") is not None:
        values.append(f"frame {node['frame_index']}")
    if node.get("trajectory_sample_index") is not None:
        values.append(f"sample {node['trajectory_sample_index']}")
    terms = sorted({value.strip().lower() for value in values if value and value.strip()})
    return terms


def _write_run_notes(graph_dir: Path, map_dir: Path) -> None:
    notes = f"""# NoMaD Topomap

Converted map:

```text
{map_dir}
```

Use graph node ids as goals:

```bash
--goal node:n000123
```

The graph UI's NoMaD panel can generate dry-run and armed commands for the
currently selected goal frame.
"""
    (graph_dir / "nomad_topomap.md").write_text(notes, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path, help="Directory containing graph.json and assets/thumbs.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--bidirectional-temporal",
        action="store_true",
        help="Allow temporal edges to be traversed backward. Default preserves the GUI's forward-temporal route behavior.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topomap = convert_topological_graph(
        args.graph_dir,
        args.output_dir,
        clean=args.clean,
        bidirectional_temporal=args.bidirectional_temporal,
    )
    print(json.dumps(topomap.summary(), indent=2, sort_keys=True))
    print(f"Wrote NoMaD semantic topomap to {topomap.map_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
