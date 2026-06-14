from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image

from flatdisk_sim.evaluate_nomad_topomap import (
    HiddenPoseLogMonitor,
    parse_video_sample,
    pose_distance_to_node,
    update_hidden_metrics,
    EvaluationMetrics,
)
from flatdisk_sim.nomad_policy import WaypointCommand
from flatdisk_sim.protocol import pack_video_jpeg
from flatdisk_sim.semantic_topomap import GoalMatch, ImageMatch, SemanticTopomap, TopomapRoute, appearance_descriptor
from flatdisk_sim.topomap_navigator import NavigatorConfig, SemanticTopomapNavigator


def solid_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (96, 72), color)


def make_topomap(tmp_path) -> SemanticTopomap:
    images = [solid_image((220, 30, 30)), solid_image((30, 220, 30)), solid_image((30, 30, 220))]
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    nodes = []
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": {"x": float(index), "y": 0.0, "z": 0.0, "yaw_rad": math.pi / 2.0, "yaw_deg": 90.0},
                "terms": ["toilet"] if index == 2 else [],
                "visible_object_count": 1 if index == 2 else 0,
            }
        )
    edges = [
        {"id": "e0", "source": "n000000", "target": "n000001", "kind": "move", "enabled": True, "weight": 1.0},
        {"id": "e1", "source": "n000001", "target": "n000002", "kind": "move", "enabled": True, "weight": 1.0},
    ]
    topomap = SemanticTopomap(tmp_path, {}, nodes, edges, np.vstack([appearance_descriptor(image) for image in images]))
    topomap.save()
    return SemanticTopomap.load(tmp_path)


def make_unreachable_topomap(tmp_path) -> SemanticTopomap:
    images = [solid_image((220, 30, 30)), solid_image((30, 30, 220))]
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    nodes = []
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": {"x": float(index), "y": 0.0, "z": 0.0, "yaw_rad": 0.0, "yaw_deg": 0.0},
                "terms": ["toilet"] if index == 1 else [],
                "visible_object_count": 1 if index == 1 else 0,
            }
        )
    topomap = SemanticTopomap(tmp_path, {}, nodes, [], np.vstack([appearance_descriptor(image) for image in images]))
    topomap.save()
    return SemanticTopomap.load(tmp_path)


def make_low_rank_connected_topomap(tmp_path) -> SemanticTopomap:
    images = [
        solid_image((220, 30, 30)),
        solid_image((210, 35, 35)),
        solid_image((200, 40, 40)),
        solid_image((190, 45, 45)),
        solid_image((30, 30, 220)),
    ]
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    nodes = []
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": {"x": float(index), "y": 0.0, "z": 0.0, "yaw_rad": 0.0, "yaw_deg": 0.0},
                "terms": ["toilet"] if index == 4 else [],
                "visible_object_count": 1 if index == 4 else 0,
            }
        )

    current = appearance_descriptor(images[0])
    basis = np.zeros_like(current)
    basis[0] = 1.0
    basis = basis - float(basis @ current) * current
    basis = basis / max(float(np.linalg.norm(basis)), 1e-8)

    def descriptor_with_score(score: float) -> np.ndarray:
        side = math.sqrt(max(0.0, 1.0 - score * score))
        value = score * current + side * basis
        return value / max(float(np.linalg.norm(value)), 1e-8)

    descriptors = np.vstack(
        [
            descriptor_with_score(1.00),
            descriptor_with_score(0.99),
            descriptor_with_score(0.98),
            descriptor_with_score(0.97),
            descriptor_with_score(0.10),
        ]
    )
    edges = [
        {"id": "e3", "source": "n000003", "target": "n000004", "kind": "move", "enabled": True, "weight": 1.0},
    ]
    topomap = SemanticTopomap(tmp_path, {}, nodes, edges, descriptors)
    topomap.save()
    return SemanticTopomap.load(tmp_path)


def make_route_window_topomap(tmp_path) -> SemanticTopomap:
    images = [
        solid_image((220, 30, 30)),
        solid_image((30, 220, 30)),
        solid_image((30, 30, 220)),
        solid_image((230, 230, 30)),
        solid_image((30, 230, 230)),
    ]
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    nodes = []
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": {"x": float(index), "y": 0.0, "z": 0.0, "yaw_rad": 0.0, "yaw_deg": 0.0},
                "terms": ["toilet"] if index == len(images) - 1 else [],
                "visible_object_count": 1 if index == len(images) - 1 else 0,
            }
        )

    current = appearance_descriptor(images[0])
    basis = np.zeros_like(current)
    basis[0] = 1.0
    basis = basis - float(basis @ current) * current
    basis = basis / max(float(np.linalg.norm(basis)), 1e-8)

    def descriptor_with_score(score: float) -> np.ndarray:
        side = math.sqrt(max(0.0, 1.0 - score * score))
        value = score * current + side * basis
        return value / max(float(np.linalg.norm(value)), 1e-8)

    descriptors = np.vstack(
        [
            descriptor_with_score(1.00),
            descriptor_with_score(0.50),
            descriptor_with_score(0.92),
            descriptor_with_score(0.99),
            descriptor_with_score(0.10),
        ]
    )
    edges = [
        {
            "id": f"e{index}",
            "source": f"n{index:06d}",
            "target": f"n{index + 1:06d}",
            "kind": "move",
            "enabled": True,
            "weight": 1.0,
        }
        for index in range(len(images) - 1)
    ]
    topomap = SemanticTopomap(tmp_path, {}, nodes, edges, descriptors)
    topomap.save()
    return SemanticTopomap.load(tmp_path)


def make_turn_topomap(tmp_path) -> SemanticTopomap:
    images = [solid_image((220, 30, 30)), solid_image((200, 40, 40)), solid_image((30, 30, 220))]
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    nodes = []
    poses = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_rad": 0.0, "yaw_deg": 0.0},
        {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_rad": math.pi / 2.0, "yaw_deg": 90.0},
        {"x": 1.0, "y": 0.0, "z": 0.0, "yaw_rad": math.pi / 2.0, "yaw_deg": 90.0},
    ]
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": poses[index],
                "terms": ["toilet"] if index == 2 else [],
                "visible_object_count": 1 if index == 2 else 0,
            }
        )
    edges = [
        {"id": "e0", "source": "n000000", "target": "n000001", "kind": "turn", "enabled": True, "weight": 0.1},
        {"id": "e1", "source": "n000001", "target": "n000002", "kind": "move", "enabled": True, "weight": 1.0},
    ]
    topomap = SemanticTopomap(tmp_path, {}, nodes, edges, np.vstack([appearance_descriptor(image) for image in images]))
    topomap.save()
    return SemanticTopomap.load(tmp_path)


class FakePolicy:
    context_size = 2

    def command_for_goal_image(self, context_images, goal_image, **kwargs):
        assert len(context_images) >= 3
        assert goal_image.size == (96, 72)
        assert kwargs["waypoint_index"] == 2
        return WaypointCommand(
            waypoint=np.asarray([0.1, 0.2], dtype=np.float32),
            motor_percent=(7, 5),
            linear_mps=0.11,
            angular_rad_s=0.22,
            inference_ms=1.5,
        )


class FailingPolicy:
    context_size = 0

    def command_for_goal_image(self, *_args, **_kwargs):
        raise RuntimeError("policy failed")


class DistanceAdvancePolicy(FakePolicy):
    context_size = 0

    def predict_distances(self, context_images, goal_images):
        assert context_images
        assert len(goal_images) == 2
        return np.asarray([2.0, 6.0], dtype=np.float32)

    def command_for_goal_image(self, context_images, goal_image, **kwargs):
        return WaypointCommand(
            waypoint=np.asarray([0.0, 0.0], dtype=np.float32),
            motor_percent=(0, 0),
            linear_mps=0.0,
            angular_rad_s=0.0,
            inference_ms=0.0,
        )


def test_semantic_topomap_navigator_generates_commands_after_context(tmp_path) -> None:
    navigator = SemanticTopomapNavigator(
        make_topomap(tmp_path),
        NavigatorConfig(goal="toilet"),
        command_policy=FakePolicy(),
    )

    first = navigator.drive_to_goal(solid_image((220, 30, 30)), armed=True)
    second = navigator.drive_to_goal(solid_image((220, 30, 30)), armed=True)
    third = navigator.drive_to_goal(solid_image((220, 30, 30)), armed=True)

    assert first.command_ready is False
    assert second.command_ready is False
    assert third.command_ready is True
    assert third.command == (7, 5)
    assert third.route_status is not None
    assert third.route_status.current_node_id == "n000001"
    payload = json.loads(third.to_status_payload())
    assert payload["schema"] == "flatdisk.nomad_topomap.status.v1"
    assert payload["motor1_percent"] == 7


def test_semantic_topomap_navigator_returns_safe_stop_on_policy_error(tmp_path) -> None:
    navigator = SemanticTopomapNavigator(
        make_topomap(tmp_path),
        NavigatorConfig(goal="toilet", context_size=1),
        command_policy=FailingPolicy(),
    )

    step = navigator.drive_to_goal(solid_image((220, 30, 30)), armed=True)

    assert step.command == (0, 0)
    assert step.command_ready is False
    assert "policy failed" in str(step.error)


def test_semantic_topomap_navigator_can_advance_with_nomad_distance_head(tmp_path) -> None:
    navigator = SemanticTopomapNavigator(
        make_topomap(tmp_path),
        NavigatorConfig(
            goal="toilet",
            context_size=1,
            use_route_window_progress=False,
            use_nomad_distance_for_progress=True,
            nomad_close_threshold=3.0,
        ),
        command_policy=DistanceAdvancePolicy(),
    )

    first = navigator.drive_to_goal(solid_image((220, 30, 30)))

    assert first.route_status is not None
    assert first.route_status.advanced is True
    assert first.route_status.current_node_id == "n000002"
    assert first.nomad_distances == {"n000001": 2.0, "n000002": 6.0}
    assert first.progress_debug is not None
    assert first.progress_debug["advance_reason"] == "nomad_current_distance_close"


def test_route_window_progress_advances_at_most_one_node(tmp_path) -> None:
    topomap = make_route_window_topomap(tmp_path)
    navigator = SemanticTopomapNavigator(
        topomap,
        NavigatorConfig(
            goal="node:n000004",
            use_nomad_distance_for_progress=False,
            route_window_lookahead=3,
            route_window_advance_threshold=0.8,
            route_window_advance_margin=0.0,
            route_window_stable_frames=1,
            route_window_max_advance=1,
        ),
    )
    navigator.follower.route = TopomapRoute(
        node_ids=["n000000", "n000001", "n000002", "n000003", "n000004"],
        start=ImageMatch("n000000", 1.0, 1),
        goal=GoalMatch("n000004", 1.0, 1, "node_id"),
        cost=4.0,
    )
    navigator.follower.cursor = 1

    status = navigator.update(solid_image((220, 30, 30)))

    assert status.advanced is True
    assert status.current_node_id == "n000002"
    assert navigator.last_progress_debug is not None
    assert navigator.last_progress_debug["route_window_best_node_id"] == "n000003"
    assert navigator.last_progress_debug["route_window_selected_node_id"] == "n000002"
    assert navigator.last_progress_debug["advance_reason"] == "route_window_match"


def test_route_window_progress_requires_stable_candidate(tmp_path) -> None:
    topomap = make_route_window_topomap(tmp_path)
    navigator = SemanticTopomapNavigator(
        topomap,
        NavigatorConfig(
            goal="node:n000004",
            use_nomad_distance_for_progress=False,
            route_window_lookahead=3,
            route_window_advance_threshold=0.8,
            route_window_advance_margin=0.0,
            route_window_stable_frames=2,
            route_window_max_advance=1,
        ),
    )
    navigator.follower.route = TopomapRoute(
        node_ids=["n000000", "n000001", "n000002", "n000003", "n000004"],
        start=ImageMatch("n000000", 1.0, 1),
        goal=GoalMatch("n000004", 1.0, 1, "node_id"),
        cost=4.0,
    )
    navigator.follower.cursor = 1

    first = navigator.update(solid_image((220, 30, 30)))
    second = navigator.update(solid_image((220, 30, 30)))

    assert first.advanced is False
    assert first.current_node_id == "n000001"
    assert second.advanced is True
    assert second.current_node_id == "n000002"
    assert navigator.last_progress_debug is not None
    assert navigator.last_progress_debug["route_window_stable_hits"] == 2


def test_semantic_topomap_navigator_skips_turn_only_interim_goals(tmp_path) -> None:
    navigator = SemanticTopomapNavigator(
        make_turn_topomap(tmp_path),
        NavigatorConfig(goal="toilet", skip_turn_only_goals=True),
    )

    route = navigator.get_sequence(solid_image((220, 30, 30)))
    status = navigator.update(solid_image((220, 30, 30)))

    assert route.node_ids == ["n000000", "n000001", "n000002"]
    assert status.advanced is True
    assert status.current_node_id == "n000002"


def test_semantic_topomap_navigator_rejects_unreachable_initial_route(tmp_path) -> None:
    navigator = SemanticTopomapNavigator(
        make_unreachable_topomap(tmp_path),
        NavigatorConfig(goal="toilet"),
    )

    step = navigator.drive_to_goal(solid_image((220, 30, 30)), armed=True)

    assert step.command == (0, 0)
    assert step.command_ready is False
    assert "unreachable route" in str(step.error)


def test_semantic_topomap_navigator_rejects_low_rank_initial_route_start(tmp_path) -> None:
    navigator = SemanticTopomapNavigator(
        make_low_rank_connected_topomap(tmp_path),
        NavigatorConfig(goal="toilet", start_top_k=5, max_initial_route_start_rank=3),
    )

    step = navigator.drive_to_goal(solid_image((220, 30, 30)), armed=True)

    assert step.command == (0, 0)
    assert step.command_ready is False
    assert "visual match rank 4" in str(step.error)


def test_evaluator_hidden_pose_monitor_and_goal_distance(tmp_path) -> None:
    topomap = make_topomap(tmp_path / "map")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "bridge_20260610_010101.jsonl"
    log_path.write_text(
        json.dumps({"t": 0.1, "pose": {"x": 1.5, "z": 0.0}, "collided": False}) + "\n"
        + json.dumps({"t": 0.2, "pose": {"x": 1.9, "z": 0.0}, "collided": True}) + "\n",
        encoding="utf-8",
    )
    monitor = HiddenPoseLogMonitor(log_dir)
    samples = monitor.poll()
    metrics = EvaluationMetrics()

    update_hidden_metrics(metrics, topomap, "n000002", samples)

    assert len(samples) == 2
    assert monitor.latest == samples[-1]
    assert pose_distance_to_node(topomap, "n000002", samples[-1].pose) == 0.10000000000000009
    assert metrics.hidden_pose_samples == 2
    assert metrics.collisions == 1
    assert metrics.best_goal_distance_m == 0.1
    assert metrics.final_goal_distance_m == 0.1


def test_evaluator_video_packet_parser() -> None:
    packet = pack_video_jpeg(jpeg=b"jpeg", width=320, height=240, seq=9, esp_us=123)

    parsed = parse_video_sample(packet)

    assert parsed == (9, 123, 320, 240, b"jpeg")
