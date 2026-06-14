"""Policy-safe semantic topomap query wrapper.

The saved topomap may have been built offline, but this runtime wrapper only
accepts an RGB frame plus a text query and returns image-match/route evidence.
It deliberately omits poses, object metadata, scene metadata, and hidden THOR
fields from the model-facing result.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .semantic_topomap import ClipEmbedder, GoalMatch, ImageMatch, SemanticTopomap


@dataclass(frozen=True)
class TopomapMemoryConfig:
    map_dir: Path
    output_dir: Path
    use_clip: bool = False
    clip_model: str = "ViT-B/32"
    allow_semantic_terms: bool = False
    current_top_k: int = 3
    goal_top_k: int = 3
    max_route_nodes: int = 8


class TopomapMemoryTool:
    def __init__(self, config: TopomapMemoryConfig) -> None:
        self.config = config
        self.topomap = SemanticTopomap.load(config.map_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.query_log_path = self.config.output_dir / "query_log.jsonl"
        self.manifest_path = self.config.output_dir / "topomap_memory_manifest.json"
        self._write_manifest()
        self._clip_embedder: ClipEmbedder | None = None
        self._query_count = 0

    def query(self, *, image_path: Path, goal_query: str, label: str = "query") -> dict[str, Any]:
        self._query_count += 1
        goal_query = goal_query.strip()
        result: dict[str, Any] = {
            "schema": "flatdisk.topomap_memory_result.v1",
            "action": "query_topomap_memory",
            "ok": False,
            "goal_query": goal_query,
            "map_summary": self._safe_map_summary(),
            "inputs_used": ["current_rgb_frame", "prior_topomap_rgb_frames"],
            "matching_mode": "image_current_match",
            "privacy": "returns node ids, image matches, route ids, and contact sheets; omits poses, object metadata, scene metadata, and hidden evaluator state",
            "planner_note": "Use as visual memory only; it is not proof that the final goal is visible or reached.",
            "topomap_memory_manifest_path": str(self.manifest_path),
            "topomap_query_log_path": str(self.query_log_path),
            "current_matches": [],
            "goal_candidates": [],
            "routes": [],
            "topomap_contact_sheet": None,
        }
        if not goal_query:
            result["reason"] = "empty_goal_query"
            self._append_query(result)
            return result
        if not image_path.exists():
            result["reason"] = "image_path_missing"
            self._append_query(result)
            return result

        image = Image.open(image_path).convert("RGB")
        if self.config.use_clip:
            result["inputs_used"].append("text_image_embedding_model")
        current_matches = self.topomap.match_image(image, top_k=self.config.current_top_k)
        result["current_matches"] = [_image_match_json(match) for match in current_matches]

        try:
            route_payload = self._route_payload(image=image, goal_query=goal_query, current_matches=current_matches)
        except Exception as exc:  # noqa: BLE001 - returned as tool evidence for the planner.
            result["reason"] = str(exc)
            if current_matches:
                result["topomap_contact_sheet"] = str(
                    self._contact_sheet(
                        [self.topomap.node_image_path(match.node_id) for match in current_matches],
                        label=label,
                        title="current image matches",
                    )
                )
            self._append_query(result)
            return result

        result.update(route_payload)
        result["ok"] = bool(route_payload.get("routes"))
        if not result["ok"]:
            result["reason"] = route_payload.get("reason", "no_route")
            if current_matches and not result.get("topomap_contact_sheet"):
                result["topomap_contact_sheet"] = str(
                    self._contact_sheet(
                        [self.topomap.node_image_path(match.node_id) for match in current_matches],
                        label=label,
                        title="current image matches",
                    )
                )
        self._append_query(result)
        return result

    def _route_payload(
        self,
        *,
        image: Image.Image,
        goal_query: str,
        current_matches: list[ImageMatch],
    ) -> dict[str, Any]:
        if not current_matches:
            return {"reason": "no_current_image_matches", "matching_mode": "image_current_match"}

        if self.config.use_clip:
            if self.topomap.clip_image_embeddings is None:
                return {"reason": "clip_goal_matching_requested_but_map_has_no_clip_image_embeddings", "matching_mode": "clip_unavailable"}
            goals = self._clip_goal_candidates(goal_query)
            mode = "clip_text_image"
        elif self.config.allow_semantic_terms:
            route = self.topomap.get_sequence(
                image,
                goal_query,
                start_top_k=self.config.current_top_k,
                goal_top_k=self.config.goal_top_k,
                clip_embedder=None,
            )
            goals = [route.goal]
            mode = "semantic_terms_allowed"
        else:
            return {
                "reason": "route_query_requires_clip_embeddings_or_explicit_allow_semantic_terms",
                "matching_mode": "image_current_match_only",
            }

        if not goals:
            return {"reason": "no_goal_candidates", "matching_mode": mode}

        routes: list[dict[str, Any]] = []
        for start in current_matches:
            for goal in goals:
                node_ids, cost = self.topomap.plan_route(start.node_id, goal.node_id)
                if math.isinf(float(cost)):
                    continue
                routes.append(
                    {
                        "start_node_id": start.node_id,
                        "goal_node_id": goal.node_id,
                        "route_length": len(node_ids),
                        "route_node_ids": node_ids[: self.config.max_route_nodes],
                        "route_truncated": len(node_ids) > self.config.max_route_nodes,
                        "cost": round(float(cost), 6),
                        "start_match_score": start.score,
                        "goal_score": goal.score,
                    }
                )
        routes.sort(key=lambda item: (int(item["route_length"]), float(item["cost"]), -float(item["start_match_score"])))
        routes = routes[:3]
        sheet = None
        if routes:
            node_ids = list(routes[0]["route_node_ids"])
            sheet = self._contact_sheet([self.topomap.node_image_path(node_id) for node_id in node_ids], label="route", title="topomap route")
        return {
            "matching_mode": mode,
            "goal_candidates": [_goal_match_json(goal) for goal in goals],
            "routes": routes,
            "topomap_contact_sheet": str(sheet) if sheet else None,
        }

    def _clip_goal_candidates(self, query: str) -> list[GoalMatch]:
        embedder = self._get_clip_embedder()
        text_embedding = embedder.text_embedding(query)
        assert self.topomap.clip_image_embeddings is not None
        scores = self.topomap.clip_image_embeddings @ text_embedding
        count = min(max(1, self.config.goal_top_k), len(scores))
        indices = np.argpartition(-scores, count - 1)[:count]
        ordered = indices[np.argsort(-scores[indices])]
        return [
            GoalMatch(
                node_id=str(self.topomap.nodes[int(index)]["id"]),
                score=round(float(scores[int(index)]), 6),
                rank=rank,
                reason="clip_text_image",
            )
            for rank, index in enumerate(ordered, start=1)
        ]

    def _get_clip_embedder(self) -> ClipEmbedder:
        if self._clip_embedder is None:
            self._clip_embedder = ClipEmbedder(self.config.clip_model)
        return self._clip_embedder

    def _safe_map_summary(self) -> dict[str, Any]:
        summary = self.topomap.summary()
        return {
            "node_count": int(summary.get("node_count", 0)),
            "edge_count": int(summary.get("edge_count", 0)),
            "has_clip_image_embeddings": self.topomap.clip_image_embeddings is not None,
        }

    def _write_manifest(self) -> None:
        manifest = {
            "schema": "flatdisk.topomap_memory_manifest.v1",
            "map_dir": str(self.config.map_dir),
            "output_dir": str(self.config.output_dir),
            "use_clip": self.config.use_clip,
            "clip_model": self.config.clip_model if self.config.use_clip else None,
            "allow_semantic_terms": self.config.allow_semantic_terms,
            "current_top_k": self.config.current_top_k,
            "goal_top_k": self.config.goal_top_k,
            "max_route_nodes": self.config.max_route_nodes,
            "map_summary": self._safe_map_summary(),
            "policy_safety": {
                "returns_pose": False,
                "returns_object_metadata": False,
                "returns_scene_metadata": False,
                "returns_hidden_evaluator_state": False,
            },
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _append_query(self, result: dict[str, Any]) -> None:
        record = {
            "query_index": self._query_count,
            "result": result,
        }
        with self.query_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _contact_sheet(self, paths: list[Path], *, label: str, title: str) -> Path:
        existing = [path for path in paths if path.exists()]
        if not existing:
            raise ValueError("no topomap node images for contact sheet")
        output = self.config.output_dir / f"{self._query_count:04d}_{_safe_id(label)}_topomap.jpg"
        images = [Image.open(path).convert("RGB") for path in existing[: self.config.max_route_nodes]]
        thumb_w = 180
        header_h = 28
        thumbs: list[Image.Image] = []
        for index, image in enumerate(images, start=1):
            scale = thumb_w / max(1, image.width)
            thumb = image.resize((thumb_w, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (thumb.width, thumb.height + header_h), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 7), f"{title} {index}/{len(images)}", fill=(20, 20, 20))
            canvas.paste(thumb, (0, header_h))
            thumbs.append(canvas)
        sheet = Image.new("RGB", (thumb_w * len(thumbs), max(t.height for t in thumbs)), "white")
        x = 0
        for thumb in thumbs:
            sheet.paste(thumb, (x, 0))
            x += thumb.width
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, format="JPEG", quality=90)
        return output


def _image_match_json(match: ImageMatch) -> dict[str, Any]:
    return {"node_id": match.node_id, "score": match.score, "rank": match.rank}


def _goal_match_json(match: GoalMatch) -> dict[str, Any]:
    return {"node_id": match.node_id, "score": match.score, "rank": match.rank, "reason": match.reason}


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return text.strip("-")[:80] or "query"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Query a semantic topomap with policy-safe output.")
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--allow-semantic-terms", action="store_true")
    args = parser.parse_args()

    tool = TopomapMemoryTool(
        TopomapMemoryConfig(
            map_dir=args.map_dir,
            output_dir=args.output_dir,
            use_clip=args.clip,
            allow_semantic_terms=args.allow_semantic_terms,
        )
    )
    print(json.dumps(tool.query(image_path=args.image, goal_query=args.goal), indent=2, sort_keys=True))
    return 0
