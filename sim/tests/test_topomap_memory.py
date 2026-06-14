from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from flatdisk_sim.semantic_topomap import SemanticTopomap, appearance_descriptor
from flatdisk_sim.topomap_memory import TopomapMemoryConfig, TopomapMemoryTool


def test_topomap_memory_default_returns_image_matches_without_semantic_route(tmp_path) -> None:
    map_dir, current_image = _write_tiny_topomap(tmp_path)
    tool = TopomapMemoryTool(TopomapMemoryConfig(map_dir=map_dir, output_dir=tmp_path / "memory"))

    result = tool.query(image_path=current_image, goal_query="sofa")

    assert result["ok"] is False
    assert result["reason"] == "route_query_requires_clip_embeddings_or_explicit_allow_semantic_terms"
    assert result["current_matches"]
    assert result["routes"] == []
    assert "pose" not in result
    assert "object_metadata" not in result
    assert result["topomap_contact_sheet"]
    assert Path(result["topomap_contact_sheet"]).exists()
    assert Path(result["topomap_memory_manifest_path"]).exists()
    assert Path(result["topomap_query_log_path"]).exists()


def test_topomap_memory_can_route_when_semantic_terms_are_explicitly_allowed(tmp_path) -> None:
    map_dir, current_image = _write_tiny_topomap(tmp_path)
    tool = TopomapMemoryTool(
        TopomapMemoryConfig(
            map_dir=map_dir,
            output_dir=tmp_path / "memory",
            allow_semantic_terms=True,
        )
    )

    result = tool.query(image_path=current_image, goal_query="sofa")

    assert result["ok"] is True
    assert result["matching_mode"] == "semantic_terms_allowed"
    assert result["goal_candidates"][0]["node_id"] == "n000002"
    assert result["routes"][0]["goal_node_id"] == "n000002"
    assert "n000002" in result["routes"][0]["route_node_ids"]
    assert result["topomap_contact_sheet"]
    assert Path(result["topomap_query_log_path"]).read_text(encoding="utf-8").strip()


def _write_tiny_topomap(tmp_path):
    map_dir = tmp_path / "map"
    image_dir = map_dir / "nodes"
    image_dir.mkdir(parents=True)
    img1 = image_dir / "n1.jpg"
    img2 = image_dir / "n2.jpg"
    Image.new("RGB", (64, 48), (30, 60, 110)).save(img1)
    Image.new("RGB", (64, 48), (150, 80, 60)).save(img2)
    descriptors = np.stack(
        [
            appearance_descriptor(Image.open(img1).convert("RGB")),
            appearance_descriptor(Image.open(img2).convert("RGB")),
        ]
    )
    topomap = SemanticTopomap(
        map_dir=map_dir,
        metadata={"source": "unit_test"},
        nodes=[
            {"id": "n000001", "image": "nodes/n1.jpg", "terms": []},
            {"id": "n000002", "image": "nodes/n2.jpg", "terms": ["sofa"]},
        ],
        edges=[{"source": "n000001", "target": "n000002", "weight": 1.0, "bidirectional": True, "enabled": True}],
        descriptors=descriptors,
    )
    topomap.save()
    return map_dir, img1
