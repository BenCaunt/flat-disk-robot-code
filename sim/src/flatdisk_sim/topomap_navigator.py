"""Reusable semantic-topomap navigation API.

The API here is the runtime counterpart to privileged map construction:
``SemanticTopomap`` decides where the robot should go, while a command policy
such as NoMaD converts the current camera context and next goal image into
motor commands.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import time
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from .nomad_policy import WaypointCommand
from .semantic_topomap import (
    ClipEmbedder,
    ImageMatch,
    RouteStatus,
    SemanticTopomap,
    TopomapRoute,
    TopomapRouteFollower,
    appearance_descriptor,
)


class ImageGoalCommandPolicy(Protocol):
    context_size: int

    def command_for_goal_image(
        self,
        context_images: Sequence[Image.Image],
        goal_image: Image.Image,
        **kwargs: Any,
    ) -> WaypointCommand:
        ...


@dataclass(frozen=True)
class NavigatorConfig:
    goal: str
    reached_threshold: float = 0.88
    start_top_k: int = 5
    goal_top_k: int = 5
    context_size: int = 4
    waypoint_index: int = 2
    waypoint_dt_s: float = 1.0 / 4.0
    max_v_mps: float = 0.2
    max_w_rad_s: float = 0.4
    wheel_base_m: float = 0.215
    max_wheel_speed_mps: float = 0.78
    max_abs_output: float = 35.0
    nomad_sample_aggregation: str = "medoid"
    invert_angular: bool = False
    use_nomad_distance_for_progress: bool = False
    use_visual_match_for_progress: bool = False
    use_route_window_progress: bool = True
    route_window_lookahead: int = 4
    route_window_advance_threshold: float = 0.55
    route_window_advance_margin: float = 0.015
    route_window_stable_frames: int = 3
    route_window_max_advance: int = 1
    nomad_close_threshold: float = 3.0
    nomad_advance_margin: float = 0.5
    skip_turn_only_goals: bool = False
    turn_goal_position_epsilon_m: float = 0.12
    require_finite_route_cost: bool = True
    max_initial_route_start_rank: int = 3


@dataclass(frozen=True)
class NavigationStep:
    goal: str
    route_status: RouteStatus | None
    command: tuple[int, int]
    armed: bool = False
    command_ready: bool = False
    reached_goal: bool = False
    nomad_inference_ms: float | None = None
    waypoint: list[float] | None = None
    linear_mps: float | None = None
    angular_rad_s: float | None = None
    raw_waypoint: list[float] | None = None
    sampled_waypoints: list[list[list[float]]] | None = None
    nomad_distances: dict[str, float] | None = None
    progress_debug: dict[str, Any] | None = None
    error: str | None = None
    host_time_unix: float = 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": "flatdisk.nomad_topomap.status.v1",
            "goal": self.goal,
            "armed": self.armed,
            "motor1_percent": self.command[0],
            "motor2_percent": self.command[1],
            "command_ready": self.command_ready,
            "reached_goal": self.reached_goal,
            "nomad_inference_ms": self.nomad_inference_ms,
            "waypoint": self.waypoint,
            "linear_mps": self.linear_mps,
            "angular_rad_s": self.angular_rad_s,
            "raw_waypoint": self.raw_waypoint,
            "sampled_waypoints": self.sampled_waypoints,
            "nomad_distances": self.nomad_distances,
            "progress_debug": self.progress_debug,
            "route": None if self.route_status is None else self.route_status.as_json(),
            "error": self.error,
            "host_time_unix": self.host_time_unix,
        }

    def to_status_payload(self) -> bytes:
        return json.dumps(self.as_json(), sort_keys=True).encode("utf-8")


class ZeroCommandPolicy:
    """Command policy for route-debug runs without NoMaD loaded."""

    context_size = 1

    def command_for_goal_image(
        self,
        context_images: Sequence[Image.Image],
        goal_image: Image.Image,
        **_kwargs: Any,
    ) -> WaypointCommand:
        return WaypointCommand(
            waypoint=np.zeros(2, dtype=np.float32),
            motor_percent=(0, 0),
            linear_mps=0.0,
            angular_rad_s=0.0,
            inference_ms=0.0,
        )


class SemanticTopomapNavigator:
    """Simple API for real-robot and simulator semantic topomap navigation.

    Typical usage:

    ``navigator = SemanticTopomapNavigator(topomap, NavigatorConfig(goal="toilet"), nomad_policy)``
    ``sequence = navigator.get_sequence(current_frame).node_ids``
    ``step = navigator.drive_to_goal(current_frame, armed=True)``
    """

    def __init__(
        self,
        topomap: SemanticTopomap,
        config: NavigatorConfig,
        command_policy: ImageGoalCommandPolicy | None = None,
        *,
        clip_embedder: ClipEmbedder | None = None,
    ) -> None:
        self.topomap = topomap
        self.config = config
        self.command_policy = command_policy or ZeroCommandPolicy()
        self.follower = TopomapRouteFollower(
            topomap,
            reached_threshold=config.reached_threshold,
            start_top_k=config.start_top_k,
            goal_top_k=config.goal_top_k,
            clip_embedder=clip_embedder,
        )
        context_size = max(int(config.context_size), int(getattr(self.command_policy, "context_size", 1)) + 1)
        self.context_images: deque[Image.Image] = deque(maxlen=context_size)
        self.last_nomad_distances: dict[str, float] | None = None
        self.last_progress_debug: dict[str, Any] | None = None
        self._route_window_candidate_node_id: str | None = None
        self._route_window_candidate_hits = 0

    @property
    def route(self) -> TopomapRoute | None:
        return self.follower.route

    def reset(self) -> None:
        self.follower.route = None
        self.follower.cursor = 0
        self.context_images.clear()
        self.last_progress_debug = None
        self._route_window_candidate_node_id = None
        self._route_window_candidate_hits = 0

    def get_sequence(self, current_frame: Image.Image, goal: str | None = None) -> TopomapRoute:
        if goal is None:
            goal = self.config.goal
        route = self.follower.reset(current_frame, goal)
        try:
            self._validate_initial_route(route)
        except Exception:
            self.reset()
            raise
        if len(route.node_ids) > 1:
            self.follower.cursor = 1
        return route

    def current_goal_image(self) -> Image.Image:
        return self.follower.current_goal_image()

    def update(self, current_frame: Image.Image) -> RouteStatus:
        if self.follower.route is None:
            self.get_sequence(current_frame)
        return self._update_route_status(current_frame)

    def drive_to_goal(self, current_frame: Image.Image, *, armed: bool = False) -> NavigationStep:
        self.context_images.append(current_frame)
        self.last_nomad_distances = None
        self.last_progress_debug = None
        host_time = time.time()
        try:
            route_status = self.update(current_frame)
            if route_status.reached_goal:
                return NavigationStep(
                    goal=self.config.goal,
                    route_status=route_status,
                    command=(0, 0),
                    armed=armed,
                    command_ready=True,
                    reached_goal=True,
                    nomad_distances=self.last_nomad_distances,
                    progress_debug=self.last_progress_debug,
                    host_time_unix=host_time,
                )

            min_context = int(getattr(self.command_policy, "context_size", 1)) + 1
            if len(self.context_images) < min_context:
                return NavigationStep(
                    goal=self.config.goal,
                    route_status=route_status,
                    command=(0, 0),
                    armed=armed,
                    command_ready=False,
                    reached_goal=False,
                    nomad_distances=self.last_nomad_distances,
                    progress_debug=self.last_progress_debug,
                    host_time_unix=host_time,
                )

            result = self.command_policy.command_for_goal_image(
                list(self.context_images),
                self.current_goal_image(),
                waypoint_index=self.config.waypoint_index,
                waypoint_dt_s=self.config.waypoint_dt_s,
                max_v_mps=self.config.max_v_mps,
                max_w_rad_s=self.config.max_w_rad_s,
                wheel_base_m=self.config.wheel_base_m,
                max_wheel_speed_mps=self.config.max_wheel_speed_mps,
                max_abs_percent=self.config.max_abs_output,
                sample_aggregation=self.config.nomad_sample_aggregation,
                invert_angular=self.config.invert_angular,
            )
            return NavigationStep(
                goal=self.config.goal,
                route_status=route_status,
                command=result.motor_percent,
                armed=armed,
                command_ready=True,
                reached_goal=False,
                nomad_inference_ms=result.inference_ms,
                waypoint=result.waypoint.astype(float).tolist(),
                linear_mps=result.linear_mps,
                angular_rad_s=result.angular_rad_s,
                raw_waypoint=None if result.raw_waypoint is None else result.raw_waypoint.astype(float).tolist(),
                sampled_waypoints=None if result.sampled_waypoints is None else result.sampled_waypoints.astype(float).tolist(),
                nomad_distances=self.last_nomad_distances,
                progress_debug=self.last_progress_debug,
                host_time_unix=host_time,
            )
        except Exception as exc:  # keep callers able to publish safe stop status.
            return NavigationStep(
                goal=self.config.goal,
                route_status=None,
                command=(0, 0),
                armed=armed,
                command_ready=False,
                reached_goal=False,
                error=f"{type(exc).__name__}: {exc}",
                host_time_unix=host_time,
            )

    def _update_route_status(self, current_frame: Image.Image) -> RouteStatus:
        status = self.follower.update(
            current_frame,
            allow_visual_advance=self.config.use_visual_match_for_progress,
        )
        route = status.route
        cursor = min(status.cursor, len(route.node_ids) - 1)
        self.last_progress_debug = {
            "cursor": cursor,
            "route_length": len(route.node_ids),
            "current_node_id": route.node_ids[cursor],
            "previous_node_id": route.node_ids[cursor - 1] if cursor > 0 else None,
            "next_node_id": route.node_ids[cursor + 1] if cursor + 1 < len(route.node_ids) else None,
            "final_node_id": route.node_ids[-1],
            "visual_match_node_id": status.current_match.node_id,
            "visual_match_score": status.current_match.score,
            "visual_reached_threshold": self.config.reached_threshold,
            "visual_progress_enabled": self.config.use_visual_match_for_progress,
            "visual_advanced": status.advanced,
            "advance_reason": "visual_match" if status.advanced else "waiting",
        }
        if (
            not self.config.use_visual_match_for_progress
            and not status.reached_goal
            and status.current_match.score >= self.config.reached_threshold
        ):
            self.last_progress_debug["advance_reason"] = "visual_match_ignored_for_interim_goal"
        status = self._skip_turn_only_goals(status)
        if status.advanced or status.reached_goal:
            if self.last_progress_debug is not None:
                self.last_progress_debug["cursor"] = status.cursor
                self.last_progress_debug["current_node_id"] = status.current_node_id
                self.last_progress_debug["advanced"] = status.advanced
                self.last_progress_debug["reached_goal"] = status.reached_goal
                if status.reached_goal:
                    self.last_progress_debug["advance_reason"] = "reached_goal"
                elif self.last_progress_debug.get("advance_reason") == "waiting":
                    self.last_progress_debug["advance_reason"] = "turn_only_skip"
            return status
        status = self._route_window_progress(status, current_frame)
        if status.advanced or status.reached_goal:
            return status
        if not self.config.use_nomad_distance_for_progress:
            if self.last_progress_debug is not None:
                self.last_progress_debug["nomad_distance_progress_enabled"] = False
                if self.last_progress_debug.get("advance_reason") == "waiting":
                    self.last_progress_debug["advance_reason"] = "nomad_distance_progress_disabled"
            return status
        if not hasattr(self.command_policy, "predict_distances"):
            if self.last_progress_debug is not None:
                self.last_progress_debug["nomad_distance_progress_enabled"] = True
                if self.last_progress_debug.get("advance_reason") == "waiting":
                    self.last_progress_debug["advance_reason"] = "policy_has_no_distance_head"
            return status
        min_context = int(getattr(self.command_policy, "context_size", 1)) + 1
        if len(self.context_images) < min_context:
            if self.last_progress_debug is not None:
                self.last_progress_debug["advance_reason"] = "waiting_for_context"
                self.last_progress_debug["context_images"] = len(self.context_images)
                self.last_progress_debug["required_context_images"] = min_context
            return status
        if self.follower.route is None:
            return status

        route = self.follower.route
        cursor = min(self.follower.cursor, len(route.node_ids) - 1)
        candidate_ids = route.node_ids[cursor : min(len(route.node_ids), cursor + 2)]
        if not candidate_ids:
            if self.last_progress_debug is not None:
                self.last_progress_debug["advance_reason"] = "no_candidate_nodes"
            return status
        if self.last_progress_debug is not None:
            self.last_progress_debug["nomad_candidate_node_ids"] = candidate_ids
        goal_images = [self.topomap.node_image(node_id) for node_id in candidate_ids]
        distances = np.asarray(
            self.command_policy.predict_distances(list(self.context_images), goal_images),
            dtype=np.float32,
        ).reshape(-1)
        if distances.size == 0:
            return status
        self.last_nomad_distances = {
            node_id: round(float(distances[index]), 4)
            for index, node_id in enumerate(candidate_ids[: len(distances)])
        }
        if self.last_progress_debug is not None:
            self.last_progress_debug["nomad_distances"] = dict(self.last_nomad_distances)
        current_distance = float(distances[0])
        should_advance = current_distance <= self.config.nomad_close_threshold
        advance_reason = (
            "nomad_current_distance_close"
            if should_advance
            else "nomad_distance_not_close"
        )
        if len(distances) > 1:
            next_distance = float(distances[1])
            next_is_closer = next_distance + self.config.nomad_advance_margin < current_distance
            should_advance = should_advance or next_is_closer
            if next_is_closer:
                advance_reason = "nomad_next_distance_closer"
        if should_advance and cursor < len(route.node_ids) - 1:
            previous_node = route.node_ids[cursor]
            self.follower.cursor = cursor + 1
            if self.last_progress_debug is not None:
                self.last_progress_debug["cursor"] = self.follower.cursor
                self.last_progress_debug["current_node_id"] = route.node_ids[self.follower.cursor]
                self.last_progress_debug["advanced"] = True
                self.last_progress_debug["advance_reason"] = advance_reason
            return RouteStatus(
                route=route,
                cursor=self.follower.cursor,
                advanced=True,
                reached_goal=False,
                current_match=ImageMatch(
                    node_id=previous_node,
                    score=round(-current_distance, 6),
                    rank=1,
                ),
            )
        if should_advance and cursor >= len(route.node_ids) - 1:
            if self.last_progress_debug is not None:
                self.last_progress_debug["reached_goal"] = True
                self.last_progress_debug["advance_reason"] = "nomad_final_distance_close"
            return RouteStatus(
                route=route,
                cursor=cursor,
                advanced=False,
                reached_goal=True,
                current_match=ImageMatch(
                    node_id=route.node_ids[cursor],
                    score=round(-current_distance, 6),
                    rank=1,
                ),
            )
        if self.last_progress_debug is not None:
            self.last_progress_debug["advance_reason"] = advance_reason
        return status

    def _route_window_progress(self, status: RouteStatus, current_frame: Image.Image) -> RouteStatus:
        if not self.config.use_route_window_progress or self.follower.route is None:
            return status
        route = self.follower.route
        cursor = min(self.follower.cursor, len(route.node_ids) - 1)
        if cursor >= len(route.node_ids) - 1:
            if self.last_progress_debug is not None:
                descriptor = appearance_descriptor(current_frame)
                node_id = route.node_ids[cursor]
                final_score = float(self.topomap.descriptors[self.topomap._node_index(node_id)] @ descriptor)
                self.last_progress_debug["route_window_final_score"] = round(final_score, 6)
                if self.last_progress_debug.get("advance_reason") == "waiting":
                    self.last_progress_debug["advance_reason"] = "route_window_at_final_goal"
            return status

        lookahead = max(1, int(self.config.route_window_lookahead))
        candidate_ids = route.node_ids[cursor : min(len(route.node_ids), cursor + lookahead + 1)]
        if len(candidate_ids) <= 1:
            return status
        descriptor = appearance_descriptor(current_frame)
        scores = [
            float(self.topomap.descriptors[self.topomap._node_index(node_id)] @ descriptor)
            for node_id in candidate_ids
        ]
        best_offset = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        current_score = float(scores[0])
        best_score = float(scores[best_offset])
        max_advance = max(1, int(self.config.route_window_max_advance))
        selected_offset = min(best_offset, max_advance)
        selected_node_id = candidate_ids[selected_offset] if selected_offset > 0 else candidate_ids[0]
        selected_score = float(scores[selected_offset])
        threshold = float(self.config.route_window_advance_threshold)
        margin = float(self.config.route_window_advance_margin)
        scores_payload = [
            {
                "offset": index,
                "node_id": node_id,
                "score": round(float(score), 6),
            }
            for index, (node_id, score) in enumerate(zip(candidate_ids, scores, strict=False))
        ]
        if self.last_progress_debug is not None:
            self.last_progress_debug.update(
                {
                    "route_window_enabled": True,
                    "route_window_candidate_scores": scores_payload,
                    "route_window_best_offset": best_offset,
                    "route_window_best_node_id": candidate_ids[best_offset],
                    "route_window_best_score": round(best_score, 6),
                    "route_window_selected_offset": selected_offset,
                    "route_window_selected_node_id": selected_node_id,
                    "route_window_selected_score": round(selected_score, 6),
                    "route_window_current_score": round(current_score, 6),
                    "route_window_advance_threshold": threshold,
                    "route_window_advance_margin": margin,
                }
            )

        should_consider = (
            best_offset > 0
            and selected_offset > 0
            and selected_score >= threshold
            and selected_score >= current_score + margin
        )
        if not should_consider:
            self._route_window_candidate_node_id = None
            self._route_window_candidate_hits = 0
            if self.last_progress_debug is not None:
                if best_offset <= 0:
                    reason = "route_window_best_is_current"
                elif selected_score < threshold:
                    reason = "route_window_below_threshold"
                else:
                    reason = "route_window_margin_too_small"
                self.last_progress_debug["advance_reason"] = reason
                self.last_progress_debug["route_window_stable_hits"] = 0
            return status

        if self._route_window_candidate_node_id == selected_node_id:
            self._route_window_candidate_hits += 1
        else:
            self._route_window_candidate_node_id = selected_node_id
            self._route_window_candidate_hits = 1
        required_hits = max(1, int(self.config.route_window_stable_frames))
        if self.last_progress_debug is not None:
            self.last_progress_debug["route_window_stable_hits"] = self._route_window_candidate_hits
            self.last_progress_debug["route_window_required_stable_hits"] = required_hits
        if self._route_window_candidate_hits < required_hits:
            if self.last_progress_debug is not None:
                self.last_progress_debug["advance_reason"] = "route_window_waiting_for_stability"
            return status

        self.follower.cursor = cursor + selected_offset
        self._route_window_candidate_node_id = None
        self._route_window_candidate_hits = 0
        if self.last_progress_debug is not None:
            self.last_progress_debug["cursor"] = self.follower.cursor
            self.last_progress_debug["current_node_id"] = route.node_ids[self.follower.cursor]
            self.last_progress_debug["advanced"] = True
            self.last_progress_debug["advance_reason"] = "route_window_match"
        return RouteStatus(
            route=route,
            cursor=self.follower.cursor,
            advanced=True,
            reached_goal=False,
            current_match=ImageMatch(
                node_id=selected_node_id,
                score=round(selected_score, 6),
                rank=selected_offset + 1,
            ),
        )

    def _validate_initial_route(self, route: TopomapRoute) -> None:
        if self.config.require_finite_route_cost and not np.isfinite(route.cost):
            raise ValueError(
                f"unreachable route from image match {route.start.node_id} to goal {route.goal.node_id}; "
                f"cost={route.cost}"
            )
        max_rank = int(self.config.max_initial_route_start_rank)
        if max_rank > 0 and route.start.rank > max_rank:
            raise ValueError(
                f"route initialized from visual match rank {route.start.rank} ({route.start.node_id}), "
                f"above max_initial_route_start_rank={max_rank}; nearest-frame initialization is ambiguous"
            )

    def _skip_turn_only_goals(self, status: RouteStatus) -> RouteStatus:
        if not self.config.skip_turn_only_goals or self.follower.route is None:
            return status
        route = self.follower.route
        cursor = min(self.follower.cursor, len(route.node_ids) - 1)
        if cursor <= 0 or cursor >= len(route.node_ids) - 1:
            return status

        anchor_id = route.node_ids[cursor - 1]
        advanced = False
        while cursor < len(route.node_ids) - 1 and self._same_position(
            anchor_id,
            route.node_ids[cursor],
            epsilon_m=self.config.turn_goal_position_epsilon_m,
        ):
            cursor += 1
            advanced = True
        if not advanced:
            return status
        self.follower.cursor = cursor
        return RouteStatus(
            route=route,
            cursor=cursor,
            advanced=True,
            reached_goal=False,
            current_match=status.current_match,
        )

    def _same_position(self, a_node_id: str, b_node_id: str, *, epsilon_m: float) -> bool:
        a_pose = self.topomap.node(a_node_id)["pose"]
        b_pose = self.topomap.node(b_node_id)["pose"]
        distance = float(
            np.hypot(float(a_pose["x"]) - float(b_pose["x"]), float(a_pose["z"]) - float(b_pose["z"]))
        )
        return distance <= epsilon_m
