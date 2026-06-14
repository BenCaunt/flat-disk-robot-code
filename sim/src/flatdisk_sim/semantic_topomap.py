"""Privileged semantic topological maps for image-goal navigation.

This module is intentionally split between two trust zones:

* map construction may use simulator-only pose, reachable floor positions, and
  object metadata;
* map consumption only needs an RGB frame and an optional text goal.

The saved map is therefore usable by the real robot through the normal Zenoh
camera stream while keeping pose and floor-plan information out of the runtime
policy input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from .thor_backend import (
    DEFAULT_CAMERA_FAR_PLANE_M,
    DEFAULT_CAMERA_FORWARD_OFFSET_M,
    DEFAULT_CAMERA_HEIGHT_M,
    DEFAULT_CAMERA_HORIZONTAL_FOV_DEG,
    DEFAULT_CAMERA_NEAR_PLANE_M,
    DEFAULT_ITHOR_SCENE,
    FlatDiskThorSim,
    ThorSimConfig,
)


SCHEMA = "flatdisk.semantic_topomap.v1"
DESCRIPTOR_NAME = "rgb_hist_gray_edge.v1"
MAP_JSON = "semantic_topomap.json"
DESCRIPTORS_NPY = "descriptors.npy"
CLIP_EMBEDDINGS_NPY = "clip_image_embeddings.npy"


def wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def yaw_to_target(source_x: float, source_z: float, target_x: float, target_z: float) -> float:
    return math.atan2(target_x - source_x, target_z - source_z)


def unit_vector(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        return values
    return values / norm


def tokenize(value: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token]


def semantic_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize(value):
        if token.isdigit():
            continue
        if len(token) >= 6 and re.fullmatch(r"[0-9a-f]+", token):
            continue
        tokens.append(token)
    return tokens


def semantic_terms_for_object(obj: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("objectType", "name"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            words = semantic_tokens(value)
            if words:
                terms.append(" ".join(words))
                terms.extend(words)
    return sorted(set(terms))


def _parse_node_goal_query(query: str) -> str | None:
    cleaned = query.strip()
    lowered = cleaned.lower()
    for prefix in ("node:", "id:"):
        if lowered.startswith(prefix):
            return cleaned[len(prefix) :].strip()
    if re.fullmatch(r"n\d{6,}", cleaned):
        return cleaned
    return None


def appearance_descriptor(image: Image.Image) -> np.ndarray:
    """Compact CPU-only descriptor for start/waypoint image matching.

    This is deliberately dependency-light. Optional CLIP embeddings can be
    stored beside the map for semantic search, but this descriptor keeps map
    matching available in a bare simulator install.
    """

    small = image.convert("RGB").resize((96, 72), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0

    quantized = np.floor(arr * 7.999).astype(np.int32)
    hist_index = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
    hist = np.bincount(hist_index.reshape(-1), minlength=512).astype(np.float32)
    hist /= float(hist.sum()) + 1e-6
    hist = np.sqrt(hist)

    coarse = np.floor(arr * 3.999).astype(np.int32)
    coarse_index = coarse[:, :, 0] * 16 + coarse[:, :, 1] * 4 + coarse[:, :, 2]
    coarse_hist = np.bincount(coarse_index.reshape(-1), minlength=64).astype(np.float32)
    coarse_hist /= float(coarse_hist.sum()) + 1e-6
    coarse_hist = np.sqrt(coarse_hist)

    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    gray_small = np.asarray(
        Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255))).resize((24, 18), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    gray_vec = gray_small.reshape(-1) / 255.0
    gray_vec -= float(gray_vec.mean())

    grad_x = np.zeros_like(gray)
    grad_y = np.zeros_like(gray)
    grad_x[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    grad_y[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edges = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge_small = np.asarray(
        Image.fromarray(np.uint8(np.clip(edges * 255.0 * 3.0, 0, 255))).resize((24, 18), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    edge_vec = edge_small.reshape(-1) / 255.0

    descriptor = np.concatenate([1.05 * hist, 0.8 * coarse_hist, 0.65 * unit_vector(gray_vec), 0.35 * unit_vector(edge_vec)])
    return unit_vector(descriptor.astype(np.float32))


@dataclass(frozen=True)
class Pose2D:
    x: float
    z: float
    yaw_rad: float
    y: float = 0.0

    def as_json(self) -> dict[str, float]:
        return {
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
            "z": round(float(self.z), 6),
            "yaw_rad": round(float(wrap_pi(self.yaw_rad)), 8),
            "yaw_deg": round(math.degrees(wrap_pi(self.yaw_rad)), 4),
        }


@dataclass
class ImageMatch:
    node_id: str
    score: float
    rank: int


@dataclass
class GoalMatch:
    node_id: str
    score: float
    rank: int
    reason: str


@dataclass
class TopomapRoute:
    node_ids: list[str]
    start: ImageMatch
    goal: GoalMatch
    cost: float

    def as_json(self) -> dict[str, Any]:
        return {
            "node_ids": self.node_ids,
            "start": self.start.__dict__,
            "goal": self.goal.__dict__,
            "cost": round(float(self.cost), 6),
        }


@dataclass
class RouteStatus:
    route: TopomapRoute
    cursor: int
    advanced: bool
    reached_goal: bool
    current_match: ImageMatch

    @property
    def current_node_id(self) -> str:
        return self.route.node_ids[min(self.cursor, len(self.route.node_ids) - 1)]

    def as_json(self) -> dict[str, Any]:
        return {
            "route": self.route.as_json(),
            "cursor": self.cursor,
            "current_node_id": self.current_node_id,
            "advanced": self.advanced,
            "reached_goal": self.reached_goal,
            "current_match": self.current_match.__dict__,
        }


@dataclass
class TopomapBuildConfig:
    max_positions: int = 40
    yaw_count: int = 4
    min_position_separation_m: float = 0.35
    object_view_count: int = 2
    object_view_max_distance_m: float = 3.5
    nearby_object_radius_m: float = 2.75
    nearby_object_fov_deg: float = 100.0
    neighbor_count: int = 5
    edge_radius_m: float = 1.25
    edge_max_heading_error_deg: float = 120.0
    image_quality: int = 92
    clean: bool = False


@dataclass
class SemanticTopomap:
    map_dir: Path
    metadata: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    descriptors: np.ndarray
    clip_image_embeddings: np.ndarray | None = None
    _node_by_id: dict[str, dict[str, Any]] = field(init=False, repr=False)
    _edge_by_pair: dict[tuple[str, str], dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.map_dir = self.map_dir.resolve()
        self._node_by_id = {str(node["id"]): node for node in self.nodes}
        self._edge_by_pair = {}
        for edge in self.edges:
            source = str(edge["source"])
            target = str(edge["target"])
            self._edge_by_pair[(source, target)] = edge
            if bool(edge.get("bidirectional", True)):
                self._edge_by_pair[(target, source)] = edge

    @classmethod
    def load(cls, map_dir: Path | str) -> "SemanticTopomap":
        map_dir = Path(map_dir).expanduser().resolve()
        manifest = json.loads((map_dir / MAP_JSON).read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise RuntimeError(f"Unsupported semantic topomap schema in {map_dir / MAP_JSON}")
        descriptors = np.load(map_dir / manifest.get("descriptors", DESCRIPTORS_NPY)).astype(np.float32)
        clip_path = manifest.get("clip_image_embeddings")
        clip_embeddings = None
        if clip_path:
            candidate = map_dir / clip_path
            if candidate.exists():
                clip_embeddings = np.load(candidate).astype(np.float32)
        return cls(
            map_dir=map_dir,
            metadata=dict(manifest.get("metadata", {})),
            nodes=list(manifest["nodes"]),
            edges=list(manifest["edges"]),
            descriptors=descriptors,
            clip_image_embeddings=clip_embeddings,
        )

    def save(self) -> None:
        self.map_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.map_dir / DESCRIPTORS_NPY, self.descriptors.astype(np.float32))
        clip_rel: str | None = None
        if self.clip_image_embeddings is not None:
            np.save(self.map_dir / CLIP_EMBEDDINGS_NPY, self.clip_image_embeddings.astype(np.float32))
            clip_rel = CLIP_EMBEDDINGS_NPY
        manifest = {
            "schema": SCHEMA,
            "created_unix": time.time(),
            "descriptor": DESCRIPTOR_NAME,
            "descriptors": DESCRIPTORS_NPY,
            "clip_image_embeddings": clip_rel,
            "metadata": self.metadata,
            "nodes": self.nodes,
            "edges": self.edges,
            "summary": self.summary(),
        }
        (self.map_dir / MAP_JSON).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def summary(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for edge in self.edges:
            kind = str(edge.get("kind", "edge"))
            by_kind[kind] = by_kind.get(kind, 0) + 1
        terms = sorted({term for node in self.nodes for term in node.get("terms", [])})
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "edges_by_kind": by_kind,
            "semantic_term_count": len(terms),
            "semantic_terms_sample": terms[:60],
        }

    def node(self, node_id: str) -> dict[str, Any]:
        return self._node_by_id[node_id]

    def node_image_path(self, node_id: str) -> Path:
        return self.map_dir / str(self.node(node_id)["image"])

    def node_image(self, node_id: str) -> Image.Image:
        return Image.open(self.node_image_path(node_id)).convert("RGB")

    def match_image(
        self,
        image: Image.Image,
        *,
        top_k: int = 5,
        scope_id: str | None = None,
        allowed_node_ids: Iterable[str] | None = None,
    ) -> list[ImageMatch]:
        if len(self.nodes) == 0:
            return []
        descriptor = appearance_descriptor(image)
        candidate_indices = np.arange(len(self.nodes), dtype=np.int64)
        if scope_id is not None:
            candidate_indices = np.asarray(
                [
                    index
                    for index, node in enumerate(self.nodes)
                    if self.node_scope_id(str(node["id"])) == scope_id
                ],
                dtype=np.int64,
            )
        if allowed_node_ids is not None:
            allowed = {str(node_id) for node_id in allowed_node_ids}
            candidate_indices = np.asarray(
                [
                    int(index)
                    for index in candidate_indices.tolist()
                    if str(self.nodes[int(index)]["id"]) in allowed
                ],
                dtype=np.int64,
            )
        if candidate_indices.size == 0:
            return []

        scores = self.descriptors[candidate_indices] @ descriptor
        count = min(max(1, top_k), len(scores))
        indices = np.argpartition(-scores, count - 1)[:count]
        ordered = indices[np.argsort(-scores[indices])]
        return [
            ImageMatch(
                node_id=str(self.nodes[int(candidate_indices[int(index)])]["id"]),
                score=round(float(scores[int(index)]), 6),
                rank=rank,
            )
            for rank, index in enumerate(ordered, start=1)
        ]

    def node_scope_id(self, node_id: str) -> str | None:
        node = self.node(node_id)
        source = node.get("source_graph_node", {})
        if isinstance(source, dict):
            for key in ("room_id", "space_id", "scene_id"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
        for key in ("room_id", "space_id", "scene_id"):
            value = node.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _scope_compatible(self, start_node_id: str, goal_node_id: str, *, scope_to_goal: bool) -> bool:
        if not scope_to_goal:
            return True
        goal_scope = self.node_scope_id(goal_node_id)
        if goal_scope is None:
            return True
        start_scope = self.node_scope_id(start_node_id)
        if start_scope is None:
            return True
        return start_scope == goal_scope

    def goal_candidates(
        self,
        query: str,
        *,
        top_k: int = 5,
        clip_embedder: "ClipEmbedder | None" = None,
    ) -> list[GoalMatch]:
        query = query.strip()
        if not query:
            raise ValueError("goal query must be non-empty")
        node_query = _parse_node_goal_query(query)
        if node_query is not None:
            if node_query not in self._node_by_id:
                raise KeyError(f"goal node does not exist in topomap: {node_query}")
            return [GoalMatch(node_id=node_query, score=1.0, rank=1, reason="node_id")]

        clip_scores: dict[str, float] = {}
        if clip_embedder is not None and self.clip_image_embeddings is not None:
            text_embedding = clip_embedder.text_embedding(query)
            scores = self.clip_image_embeddings @ text_embedding
            for index, score in enumerate(scores.tolist()):
                clip_scores[str(self.nodes[index]["id"])] = float(score)

        lexical_scores = {str(node["id"]): _lexical_goal_score(query, node) for node in self.nodes}
        matches: list[GoalMatch] = []
        for node in self.nodes:
            node_id = str(node["id"])
            lexical = lexical_scores[node_id]
            clip_score = clip_scores.get(node_id)
            if clip_score is None:
                score = lexical
                reason = "semantic_terms"
            else:
                score = 0.68 * clip_score + 0.32 * lexical
                reason = "clip+semantic_terms"
            matches.append(GoalMatch(node_id=node_id, score=round(float(score), 6), rank=0, reason=reason))

        matches.sort(key=lambda item: (item.score, -_node_semantic_density(self.node(item.node_id))), reverse=True)
        for rank, match in enumerate(matches[:top_k], start=1):
            match.rank = rank
        return matches[:top_k]

    def plan_route(self, start_node_id: str, goal_node_id: str) -> tuple[list[str], float]:
        if start_node_id not in self._node_by_id:
            raise KeyError(start_node_id)
        if goal_node_id not in self._node_by_id:
            raise KeyError(goal_node_id)
        if start_node_id == goal_node_id:
            return [start_node_id], 0.0

        adjacency: dict[str, list[tuple[str, float]]] = {str(node["id"]): [] for node in self.nodes}
        for edge in self.edges:
            if not edge.get("enabled", True):
                continue
            source = str(edge["source"])
            target = str(edge["target"])
            weight = max(1e-6, float(edge.get("weight", 1.0)))
            adjacency.setdefault(source, []).append((target, weight))
            if bool(edge.get("bidirectional", True)):
                adjacency.setdefault(target, []).append((source, weight))

        queue: list[tuple[float, str]] = [(0.0, start_node_id)]
        best = {start_node_id: 0.0}
        previous: dict[str, str] = {}
        while queue:
            cost, node_id = heapq.heappop(queue)
            if cost > best.get(node_id, math.inf):
                continue
            if node_id == goal_node_id:
                break
            for next_node, weight in adjacency.get(node_id, []):
                next_cost = cost + weight
                if next_cost < best.get(next_node, math.inf):
                    best[next_node] = next_cost
                    previous[next_node] = node_id
                    heapq.heappush(queue, (next_cost, next_node))

        if goal_node_id not in best:
            return [start_node_id, goal_node_id], math.inf

        path = [goal_node_id]
        while path[-1] != start_node_id:
            path.append(previous[path[-1]])
        path.reverse()
        return path, best[goal_node_id]

    def get_sequence(
        self,
        current_frame: Image.Image,
        goal_query: str,
        *,
        start_top_k: int = 5,
        goal_top_k: int = 5,
        clip_embedder: "ClipEmbedder | None" = None,
        scope_to_goal: bool = True,
        min_start_match_score: float = 0.88,
    ) -> TopomapRoute:
        goal_matches = self.goal_candidates(goal_query, top_k=goal_top_k, clip_embedder=clip_embedder)
        if not goal_matches:
            raise RuntimeError(f"no goal candidates for {goal_query!r}")
        best_goal_score = float(goal_matches[0].score)
        if best_goal_score > 0.0:
            min_goal_score = max(0.05, best_goal_score * 0.5)
            goal_matches = [match for match in goal_matches if float(match.score) >= min_goal_score]

        best_route: TopomapRoute | None = None
        fallback_start: ImageMatch | None = None
        fallback_goal: GoalMatch | None = goal_matches[0] if goal_matches else None
        scoped_start_seen = False
        best_start_error: str | None = None
        for goal in goal_matches:
            goal_scope = self.node_scope_id(goal.node_id) if scope_to_goal else None
            scoped_starts = self.match_image(
                current_frame,
                top_k=start_top_k,
                scope_id=goal_scope,
            )
            if not scoped_starts:
                scope_note = f" in goal scope {goal_scope}" if goal_scope else ""
                best_start_error = f"no start image matches{scope_note}"
                continue
            if any(self._scope_compatible(start.node_id, goal.node_id, scope_to_goal=scope_to_goal) for start in scoped_starts):
                scoped_start_seen = True
            best_start_score = float(scoped_starts[0].score)
            absolute_min_start_score = float(min_start_match_score) if goal_scope is not None else 0.2
            min_start_score = max(absolute_min_start_score, best_start_score * 0.9)
            start_matches = [start for start in scoped_starts if float(start.score) >= min_start_score]
            if not start_matches:
                best_start_error = (
                    f"best start image match {scoped_starts[0].node_id} score={scoped_starts[0].score} "
                    f"is below min_start_match_score={min_start_match_score}"
                )
                continue
            for start in start_matches:
                fallback_goal = goal
                if fallback_start is None:
                    fallback_start = start
                node_ids, cost = self.plan_route(start.node_id, goal.node_id)
                if math.isinf(cost):
                    continue
                route = TopomapRoute(node_ids=node_ids, start=start, goal=goal, cost=cost)
                if (
                    best_route is None
                    or route.start.rank < best_route.start.rank
                    or (route.start.rank == best_route.start.rank and route.cost < best_route.cost)
                ):
                    best_route = route
        if best_route is not None:
            return best_route

        if scope_to_goal and not scoped_start_seen and fallback_goal is not None:
            goal_scope = self.node_scope_id(fallback_goal.node_id) or "unknown"
            raise RuntimeError(
                best_start_error or f"no start image matches in goal scope {goal_scope}"
            )

        if fallback_start is None:
            raise RuntimeError(best_start_error or "no sufficiently similar start image matches")

        start = fallback_start
        goal = fallback_goal or goal_matches[0]
        node_ids, cost = self.plan_route(start.node_id, goal.node_id)
        return TopomapRoute(node_ids=node_ids, start=start, goal=goal, cost=cost)

    def at_node_image(self, current_frame: Image.Image, node_id: str, *, threshold: float = 0.88) -> ImageMatch:
        descriptor = appearance_descriptor(current_frame)
        node_index = self._node_index(node_id)
        score = float(self.descriptors[node_index] @ descriptor)
        return ImageMatch(node_id=node_id, score=round(score, 6), rank=1 if score >= threshold else 0)

    def _node_index(self, node_id: str) -> int:
        for index, node in enumerate(self.nodes):
            if str(node["id"]) == node_id:
                return index
        raise KeyError(node_id)


class TopomapRouteFollower:
    """Stateful route cursor for ``get_sequence`` plus image-goal advancement."""

    def __init__(
        self,
        topomap: SemanticTopomap,
        *,
        reached_threshold: float = 0.88,
        start_top_k: int = 5,
        goal_top_k: int = 5,
        clip_embedder: "ClipEmbedder | None" = None,
    ) -> None:
        self.topomap = topomap
        self.reached_threshold = reached_threshold
        self.start_top_k = start_top_k
        self.goal_top_k = goal_top_k
        self.clip_embedder = clip_embedder
        self.route: TopomapRoute | None = None
        self.cursor = 0

    def reset(self, current_frame: Image.Image, goal_query: str) -> TopomapRoute:
        self.route = self.topomap.get_sequence(
            current_frame,
            goal_query,
            start_top_k=self.start_top_k,
            goal_top_k=self.goal_top_k,
            clip_embedder=self.clip_embedder,
        )
        self.cursor = 0
        return self.route

    def current_goal_node_id(self) -> str:
        if self.route is None:
            raise RuntimeError("route follower has no route; call reset first")
        return self.route.node_ids[min(self.cursor, len(self.route.node_ids) - 1)]

    def current_goal_image(self) -> Image.Image:
        return self.topomap.node_image(self.current_goal_node_id())

    def update(self, current_frame: Image.Image, *, allow_visual_advance: bool = True) -> RouteStatus:
        if self.route is None:
            raise RuntimeError("route follower has no route; call reset first")
        current_node = self.current_goal_node_id()
        match = self.topomap.at_node_image(current_frame, current_node, threshold=self.reached_threshold)
        advanced = False
        if allow_visual_advance and match.score >= self.reached_threshold and self.cursor < len(self.route.node_ids) - 1:
            self.cursor += 1
            advanced = True
        reached_goal = bool(match.score >= self.reached_threshold and self.cursor >= len(self.route.node_ids) - 1)
        return RouteStatus(
            route=self.route,
            cursor=self.cursor,
            advanced=advanced,
            reached_goal=reached_goal,
            current_match=match,
        )


class ClipEmbedder:
    """Optional OpenAI CLIP wrapper used only when requested."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto") -> None:
        import torch
        import clip

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def image_embeddings(self, images: Sequence[Image.Image], *, batch_size: int = 32) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                tensor = self.torch.stack([self.preprocess(image.convert("RGB")) for image in batch]).to(self.device)
                embedding = self.model.encode_image(tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                outputs.append(embedding.detach().cpu().float().numpy())
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def text_embedding(self, text: str) -> np.ndarray:
        with self.torch.inference_mode():
            tokens = self.clip.tokenize([text]).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return embedding[0].detach().cpu().float().numpy().astype(np.float32)


def _lexical_goal_score(query: str, node: dict[str, Any]) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    terms = [str(term) for term in node.get("terms", [])]
    term_text = " ".join(terms).lower()
    exact_bonus = 0.35 if query.lower() in term_text else 0.0
    term_tokens = set(token for term in terms for token in tokenize(term))
    if not term_tokens:
        return exact_bonus
    overlap = len(query_tokens & term_tokens) / max(1, len(query_tokens))
    density = min(0.15, 0.01 * len(terms))
    visible_bonus = 0.1 if node.get("visible_object_count", 0) else 0.0
    return min(1.0, exact_bonus + 0.7 * overlap + density + visible_bonus)


def _node_semantic_density(node: dict[str, Any]) -> int:
    return int(node.get("visible_object_count", 0)) + len(node.get("terms", []))


def _position_array(reachable_positions: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(position["x"]), float(position.get("y", 0.0)), float(position["z"])] for position in reachable_positions],
        dtype=np.float32,
    )


def select_coverage_positions(
    reachable_positions: Sequence[dict[str, Any]],
    *,
    max_positions: int,
    min_separation_m: float,
) -> list[dict[str, float]]:
    if not reachable_positions:
        return []
    positions = _position_array(reachable_positions)
    max_positions = min(max(1, max_positions), len(positions))
    center = positions[:, [0, 2]].mean(axis=0)
    first = int(np.argmin(np.linalg.norm(positions[:, [0, 2]] - center[None, :], axis=1)))
    selected_indices = [first]
    nearest = np.linalg.norm(positions[:, [0, 2]] - positions[first, [0, 2]][None, :], axis=1)
    while len(selected_indices) < max_positions:
        candidate = int(np.argmax(nearest))
        if nearest[candidate] < min_separation_m:
            break
        selected_indices.append(candidate)
        dist = np.linalg.norm(positions[:, [0, 2]] - positions[candidate, [0, 2]][None, :], axis=1)
        nearest = np.minimum(nearest, dist)
    return [_pose_position_dict(positions[index]) for index in selected_indices]


def object_view_positions(
    reachable_positions: Sequence[dict[str, Any]],
    objects: Sequence[dict[str, Any]],
    *,
    views_per_object: int,
    max_distance_m: float,
) -> list[Pose2D]:
    if not reachable_positions or views_per_object <= 0:
        return []
    positions = _position_array(reachable_positions)
    views: list[Pose2D] = []
    for obj in objects:
        obj_position = obj.get("position")
        if not isinstance(obj_position, dict):
            continue
        try:
            ox = float(obj_position["x"])
            oz = float(obj_position["z"])
        except (KeyError, TypeError, ValueError):
            continue
        dists = np.linalg.norm(positions[:, [0, 2]] - np.asarray([[ox, oz]], dtype=np.float32), axis=1)
        ordered = np.argsort(dists)
        added = 0
        for index in ordered:
            if float(dists[int(index)]) > max_distance_m:
                break
            x, y, z = [float(value) for value in positions[int(index)]]
            views.append(Pose2D(x=x, y=y, z=z, yaw_rad=yaw_to_target(x, z, ox, oz)))
            added += 1
            if added >= views_per_object:
                break
    return views


def build_semantic_topomap_from_sim(
    sim: FlatDiskThorSim,
    output_dir: Path | str,
    *,
    config: TopomapBuildConfig = TopomapBuildConfig(),
    clip_embedder: ClipEmbedder | None = None,
) -> SemanticTopomap:
    output_dir = Path(output_dir).expanduser().resolve()
    if config.clean and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    reachable_positions = sim.privileged_reachable_positions()
    if not reachable_positions:
        raise RuntimeError("simulator returned no reachable positions")
    all_objects = sim.hidden_objects()
    coverage = select_coverage_positions(
        reachable_positions,
        max_positions=config.max_positions,
        min_separation_m=config.min_position_separation_m,
    )

    yaw_values = [2.0 * math.pi * index / max(1, config.yaw_count) for index in range(max(1, config.yaw_count))]
    poses = [
        Pose2D(x=float(position["x"]), y=float(position.get("y", 0.0)), z=float(position["z"]), yaw_rad=yaw)
        for position in coverage
        for yaw in yaw_values
    ]
    poses.extend(
        object_view_positions(
            reachable_positions,
            all_objects,
            views_per_object=config.object_view_count,
            max_distance_m=config.object_view_max_distance_m,
        )
    )
    poses = _dedupe_poses(poses, position_bin_m=max(0.05, config.min_position_separation_m * 0.5), yaw_bin_deg=15.0)

    nodes: list[dict[str, Any]] = []
    descriptors: list[np.ndarray] = []
    rendered_images: list[Image.Image] = []
    for index, pose in enumerate(poses):
        ok = sim.privileged_teleport(x=pose.x, y=pose.y, z=pose.z, yaw_rad=pose.yaw_rad)
        if not ok:
            continue
        image = sim.render_image()
        hidden = sim.hidden_pose()
        visible_objects = [obj for obj in hidden.get("objects", []) if bool(obj.get("visible", False))]
        nearby_objects = _nearby_facing_objects(
            all_objects,
            pose,
            radius_m=config.nearby_object_radius_m,
            fov_deg=config.nearby_object_fov_deg,
        )
        terms = sorted({term for obj in [*visible_objects, *nearby_objects] for term in semantic_terms_for_object(obj)})
        image_rel = f"images/{index:06d}.jpg"
        image.save(output_dir / image_rel, format="JPEG", quality=config.image_quality)
        descriptor = appearance_descriptor(image)
        node = {
            "id": f"n{index:06d}",
            "image": image_rel,
            "pose": hidden.get("x") is not None and Pose2D(
                x=float(hidden["x"]),
                y=float(hidden.get("y", pose.y)),
                z=float(hidden["z"]),
                yaw_rad=math.radians(float(hidden.get("yaw_deg", math.degrees(pose.yaw_rad)))),
            ).as_json() or pose.as_json(),
            "terms": terms,
            "visible_object_count": len(visible_objects),
            "nearby_object_count": len(nearby_objects),
            "visible_objects": _compact_objects(visible_objects),
            "nearby_objects": _compact_objects(nearby_objects),
        }
        nodes.append(node)
        descriptors.append(descriptor)
        rendered_images.append(image.copy())

    if not nodes:
        raise RuntimeError("no topomap nodes were rendered")

    descriptor_matrix = np.vstack(descriptors).astype(np.float32)
    edges = build_pose_edges(
        nodes,
        neighbor_count=config.neighbor_count,
        edge_radius_m=config.edge_radius_m,
        edge_max_heading_error_deg=config.edge_max_heading_error_deg,
    )
    clip_embeddings = clip_embedder.image_embeddings(rendered_images) if clip_embedder is not None else None
    xs = [float(position["x"]) for position in reachable_positions]
    zs = [float(position["z"]) for position in reachable_positions]
    topomap = SemanticTopomap(
        map_dir=output_dir,
        metadata={
            "builder": "privileged_ai2thor",
            "backend": sim.backend_name,
            "world": sim.world_name,
            "scene": sim.scene_name,
            "reachable_position_count": len(reachable_positions),
            "selected_coverage_position_count": len(coverage),
            "object_count": len(all_objects),
            "floor_plan_bounds": {
                "min_x": round(min(xs), 6),
                "max_x": round(max(xs), 6),
                "min_z": round(min(zs), 6),
                "max_z": round(max(zs), 6),
            },
            "camera": sim.hidden_pose().get("camera", {}),
            "build_config": config.__dict__,
        },
        nodes=nodes,
        edges=edges,
        descriptors=descriptor_matrix,
        clip_image_embeddings=clip_embeddings,
    )
    topomap.save()
    return topomap


def build_pose_edges(
    nodes: Sequence[dict[str, Any]],
    *,
    neighbor_count: int,
    edge_radius_m: float,
    edge_max_heading_error_deg: float,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    poses = [_pose_from_node(node) for node in nodes]
    ids = [str(node["id"]) for node in nodes]
    max_heading_error = math.radians(edge_max_heading_error_deg)
    edge_keys: set[tuple[str, str, str]] = set()

    for i, pose_i in enumerate(poses):
        same_position: list[tuple[float, int]] = []
        spatial: list[tuple[float, float, int]] = []
        for j, pose_j in enumerate(poses):
            if i == j:
                continue
            distance = math.hypot(pose_j.x - pose_i.x, pose_j.z - pose_i.z)
            yaw_error = abs(wrap_pi(pose_j.yaw_rad - pose_i.yaw_rad))
            if distance <= 0.08:
                same_position.append((yaw_error, j))
                continue
            if distance > edge_radius_m:
                continue
            heading = yaw_to_target(pose_i.x, pose_i.z, pose_j.x, pose_j.z)
            heading_error = abs(wrap_pi(heading - pose_i.yaw_rad))
            if heading_error > max_heading_error:
                continue
            spatial.append((distance, heading_error, j))

        same_position.sort(key=lambda item: item[0])
        for yaw_error, j in same_position[:2]:
            _append_edge(
                edges,
                edge_keys,
                source=ids[i],
                target=ids[j],
                kind="turn",
                weight=0.05 + yaw_error / math.pi * 0.25,
                metrics={"yaw_error_deg": round(math.degrees(yaw_error), 3)},
            )

        spatial.sort(key=lambda item: (item[0], item[1]))
        for distance, heading_error, j in spatial[: max(1, neighbor_count)]:
            _append_edge(
                edges,
                edge_keys,
                source=ids[i],
                target=ids[j],
                kind="move",
                weight=distance * (1.0 + 0.35 * heading_error / math.pi),
                metrics={
                    "distance_m": round(distance, 4),
                    "heading_error_deg": round(math.degrees(heading_error), 3),
                },
            )
    return edges


def _append_edge(
    edges: list[dict[str, Any]],
    keys: set[tuple[str, str, str]],
    *,
    source: str,
    target: str,
    kind: str,
    weight: float,
    metrics: dict[str, Any],
) -> None:
    key = tuple(sorted((source, target))) + (kind,)
    if key in keys:
        return
    keys.add(key)
    edges.append(
        {
            "id": f"e_{kind}_{source}_{target}",
            "source": source,
            "target": target,
            "kind": kind,
            "enabled": True,
            "bidirectional": True,
            "weight": round(float(weight), 6),
            **metrics,
        }
    )


def _pose_position_dict(values: np.ndarray) -> dict[str, float]:
    return {"x": float(values[0]), "y": float(values[1]), "z": float(values[2])}


def _pose_from_node(node: dict[str, Any]) -> Pose2D:
    pose = node["pose"]
    return Pose2D(
        x=float(pose["x"]),
        y=float(pose.get("y", 0.0)),
        z=float(pose["z"]),
        yaw_rad=float(pose.get("yaw_rad", math.radians(float(pose.get("yaw_deg", 0.0))))),
    )


def _dedupe_poses(poses: Iterable[Pose2D], *, position_bin_m: float, yaw_bin_deg: float) -> list[Pose2D]:
    seen: set[tuple[int, int, int]] = set()
    result: list[Pose2D] = []
    yaw_bin = math.radians(yaw_bin_deg)
    for pose in poses:
        key = (
            int(round(pose.x / position_bin_m)),
            int(round(pose.z / position_bin_m)),
            int(round(wrap_pi(pose.yaw_rad) / yaw_bin)),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(pose)
    return result


def _nearby_facing_objects(
    objects: Sequence[dict[str, Any]],
    pose: Pose2D,
    *,
    radius_m: float,
    fov_deg: float,
) -> list[dict[str, Any]]:
    half_fov = math.radians(fov_deg) * 0.5
    selected: list[tuple[float, dict[str, Any]]] = []
    for obj in objects:
        position = obj.get("position")
        if not isinstance(position, dict):
            continue
        try:
            ox = float(position["x"])
            oz = float(position["z"])
        except (KeyError, TypeError, ValueError):
            continue
        distance = math.hypot(ox - pose.x, oz - pose.z)
        if distance > radius_m:
            continue
        heading_error = abs(wrap_pi(yaw_to_target(pose.x, pose.z, ox, oz) - pose.yaw_rad))
        if heading_error <= half_fov:
            selected.append((distance, obj))
    selected.sort(key=lambda item: item[0])
    return [obj for _distance, obj in selected[:12]]


def _compact_objects(objects: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for obj in objects:
        position = obj.get("position", {})
        result.append(
            {
                "name": obj.get("name"),
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "visible": bool(obj.get("visible", False)),
                "position": {
                    "x": round(float(position.get("x", 0.0)), 4) if isinstance(position, dict) else 0.0,
                    "z": round(float(position.get("z", 0.0)), 4) if isinstance(position, dict) else 0.0,
                },
            }
        )
    return result


def _build_config_from_args(args: argparse.Namespace) -> TopomapBuildConfig:
    return TopomapBuildConfig(
        max_positions=args.max_positions,
        yaw_count=args.yaw_count,
        min_position_separation_m=args.min_position_separation_m,
        object_view_count=args.object_view_count,
        object_view_max_distance_m=args.object_view_max_distance_m,
        nearby_object_radius_m=args.nearby_object_radius_m,
        nearby_object_fov_deg=args.nearby_object_fov_deg,
        neighbor_count=args.neighbor_count,
        edge_radius_m=args.edge_radius_m,
        edge_max_heading_error_deg=args.edge_max_heading_error_deg,
        image_quality=args.image_quality,
        clean=args.clean,
    )


def build_cli_main() -> int:
    parser = argparse.ArgumentParser(description="Build a privileged semantic topomap from AI2-THOR/ProcTHOR.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="ithor", choices=("ithor", "procthor", "house-json"))
    parser.add_argument("--scene", default=DEFAULT_ITHOR_SCENE)
    parser.add_argument("--house-json", type=Path)
    parser.add_argument("--procthor-seed", type=int, default=42)
    parser.add_argument("--procthor-split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--field-of-view", type=float, default=DEFAULT_CAMERA_HORIZONTAL_FOV_DEG)
    parser.add_argument("--field-of-view-axis", default="horizontal", choices=("horizontal", "vertical"))
    parser.add_argument("--camera-height-m", type=float, default=DEFAULT_CAMERA_HEIGHT_M)
    parser.add_argument("--camera-forward-offset-m", type=float, default=DEFAULT_CAMERA_FORWARD_OFFSET_M)
    parser.add_argument("--camera-near-plane-m", type=float, default=DEFAULT_CAMERA_NEAR_PLANE_M)
    parser.add_argument("--camera-far-plane-m", type=float, default=DEFAULT_CAMERA_FAR_PLANE_M)
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--quality", default="Low")
    parser.add_argument("--grid-size", type=float, default=0.05)
    parser.add_argument("--rotate-step-degrees", type=float, default=5.0)
    parser.add_argument("--max-positions", type=int, default=40)
    parser.add_argument("--yaw-count", type=int, default=4)
    parser.add_argument("--min-position-separation-m", type=float, default=0.35)
    parser.add_argument("--object-view-count", type=int, default=2)
    parser.add_argument("--object-view-max-distance-m", type=float, default=3.5)
    parser.add_argument("--nearby-object-radius-m", type=float, default=2.75)
    parser.add_argument("--nearby-object-fov-deg", type=float, default=100.0)
    parser.add_argument("--neighbor-count", type=int, default=5)
    parser.add_argument("--edge-radius-m", type=float, default=1.25)
    parser.add_argument("--edge-max-heading-error-deg", type=float, default=120.0)
    parser.add_argument("--image-quality", type=int, default=92)
    parser.add_argument("--clip", action="store_true", help="Also store CLIP image embeddings for text-image search.")
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    sim = FlatDiskThorSim(
        ThorSimConfig(
            backend=args.backend,
            scene=args.scene,
            house_json=args.house_json,
            procthor_seed=args.procthor_seed,
            procthor_split=args.procthor_split,
            width=args.render_width,
            height=args.render_height,
            field_of_view=args.field_of_view,
            field_of_view_axis=args.field_of_view_axis,
            camera_height_m=args.camera_height_m,
            camera_forward_offset_m=args.camera_forward_offset_m,
            camera_near_plane_m=args.camera_near_plane_m,
            camera_far_plane_m=args.camera_far_plane_m,
            camera_calibration=args.camera_calibration,
            use_third_party_camera=True,
            grid_size=args.grid_size,
            rotate_step_degrees=args.rotate_step_degrees,
            quality=args.quality,
        )
    )
    try:
        clip_embedder = ClipEmbedder(args.clip_model) if args.clip else None
        topomap = build_semantic_topomap_from_sim(
            sim,
            args.output_dir,
            config=_build_config_from_args(args),
            clip_embedder=clip_embedder,
        )
    finally:
        sim.close()
    print(json.dumps(topomap.summary(), indent=2, sort_keys=True))
    print(f"Wrote semantic topomap to {topomap.map_dir}")
    return 0


def query_cli_main() -> int:
    parser = argparse.ArgumentParser(description="Query a saved semantic topomap.")
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--clip-model", default="ViT-B/32")
    args = parser.parse_args()

    topomap = SemanticTopomap.load(args.map_dir)
    image = Image.open(args.image).convert("RGB")
    clip_embedder = ClipEmbedder(args.clip_model) if args.clip else None
    route = topomap.get_sequence(image, args.goal, clip_embedder=clip_embedder)
    print(json.dumps(route.as_json(), indent=2, sort_keys=True))
    return 0
