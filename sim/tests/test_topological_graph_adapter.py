from __future__ import annotations

import json

from PIL import Image

from flatdisk_sim.semantic_topomap import SemanticTopomap
from flatdisk_sim.topological_graph_adapter import convert_topological_graph


def test_convert_topological_graph_to_semantic_topomap_supports_node_goals(tmp_path) -> None:
    graph_dir = tmp_path / "graph"
    thumb_dir = graph_dir / "assets" / "thumbs"
    thumb_dir.mkdir(parents=True)
    Image.new("RGB", (96, 72), (220, 20, 20)).save(thumb_dir / "a.jpg")
    Image.new("RGB", (96, 72), (20, 220, 20)).save(thumb_dir / "b.jpg")
    graph = {
        "schema": "flatdisk_topological_graph.v1",
        "metadata": {"summary": {"node_count": 2, "edge_count": 1}, "output_dir": str(graph_dir)},
        "nodes": [
            {
                "id": "n000000",
                "thumb": "assets/thumbs/a.jpg",
                "layout": {"x": 0.0, "y": 0.0},
                "dataset": "demo",
                "trajectory_id": "demo/traj",
                "trajectory_label": "traj",
                "recording_name": "traj.rrd",
                "frame_index": 0,
                "trajectory_sample_index": 0,
            },
            {
                "id": "n000001",
                "thumb": "assets/thumbs/b.jpg",
                "layout": {"x": 1.0, "y": 0.0},
                "dataset": "demo",
                "trajectory_id": "demo/traj",
                "trajectory_label": "traj",
                "recording_name": "traj.rrd",
                "frame_index": 1,
                "trajectory_sample_index": 1,
            },
        ],
        "edges": [
            {
                "id": "e_temporal_n000000_n000001",
                "source": "n000000",
                "target": "n000001",
                "kind": "temporal",
                "enabled": True,
                "weight": 1.0,
            }
        ],
    }
    (graph_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")

    converted = convert_topological_graph(graph_dir, clean=True)
    loaded = SemanticTopomap.load(converted.map_dir)
    route = loaded.get_sequence(Image.open(thumb_dir / "a.jpg"), "node:n000001")

    assert (converted.map_dir / "semantic_topomap.json").exists()
    assert (converted.map_dir / "descriptors.npy").exists()
    assert route.start.node_id == "n000000"
    assert route.goal.node_id == "n000001"
    assert route.goal.reason == "node_id"
    assert route.node_ids == ["n000000", "n000001"]
    assert loaded.edges[0]["bidirectional"] is False
