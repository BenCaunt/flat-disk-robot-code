from __future__ import annotations

import math

import numpy as np
from PIL import Image

from flatdisk_sim.nomad_policy import twist_to_motor_percent, waypoint_to_twist
from flatdisk_sim.semantic_topomap import (
    SemanticTopomap,
    TopomapRouteFollower,
    appearance_descriptor,
    build_pose_edges,
    semantic_terms_for_object,
)


def solid_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (96, 72), color)


def make_topomap(tmp_path) -> SemanticTopomap:
    images = [solid_image((220, 30, 30)), solid_image((30, 220, 30)), solid_image((30, 30, 220))]
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    nodes = []
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        terms = ["toilet", "bathroom"] if index == 2 else []
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": {"x": float(index), "y": 0.0, "z": 0.0, "yaw_rad": math.pi / 2.0, "yaw_deg": 90.0},
                "terms": terms,
                "visible_object_count": 1 if terms else 0,
            }
        )
    edges = [
        {"id": "e0", "source": "n000000", "target": "n000001", "kind": "move", "enabled": True, "weight": 1.0},
        {"id": "e1", "source": "n000001", "target": "n000002", "kind": "move", "enabled": True, "weight": 1.0},
    ]
    descriptors = np.vstack([appearance_descriptor(image) for image in images])
    topomap = SemanticTopomap(tmp_path, {}, nodes, edges, descriptors)
    topomap.save()
    return SemanticTopomap.load(tmp_path)


def make_scoped_topomap(tmp_path, *, scoped_start_score: float = 0.95) -> SemanticTopomap:
    images = [solid_image((220, 30, 30)), solid_image((210, 35, 35)), solid_image((30, 30, 220))]
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rooms = ["space_a", "space_b", "space_b"]
    nodes = []
    for index, image in enumerate(images):
        image.save(image_dir / f"{index:06d}.jpg")
        nodes.append(
            {
                "id": f"n{index:06d}",
                "image": f"images/{index:06d}.jpg",
                "pose": {"x": float(index), "y": 0.0, "z": 0.0, "yaw_rad": 0.0, "yaw_deg": 0.0},
                "source_graph_node": {"room_id": rooms[index]},
                "terms": ["toilet"] if index == 2 else [],
                "visible_object_count": 1 if index == 2 else 0,
            }
        )
    edges = [
        {"id": "e0", "source": "n000001", "target": "n000002", "kind": "move", "enabled": True, "weight": 1.0},
    ]
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
            descriptor_with_score(1.0),
            descriptor_with_score(scoped_start_score),
            descriptor_with_score(0.1),
        ]
    )
    topomap = SemanticTopomap(tmp_path, {}, nodes, edges, descriptors)
    topomap.save()
    return SemanticTopomap.load(tmp_path)


def test_semantic_topomap_matches_start_and_routes_to_text_goal(tmp_path) -> None:
    topomap = make_topomap(tmp_path)

    route = topomap.get_sequence(solid_image((222, 32, 32)), "toilet")

    assert route.start.node_id == "n000000"
    assert route.goal.node_id == "n000002"
    assert route.node_ids == ["n000000", "n000001", "n000002"]
    assert route.cost == 2.0


def test_get_sequence_scopes_start_match_to_goal_room(tmp_path) -> None:
    topomap = make_scoped_topomap(tmp_path)

    route = topomap.get_sequence(solid_image((220, 30, 30)), "node:n000002", start_top_k=3)
    scoped_matches = topomap.match_image(solid_image((220, 30, 30)), top_k=3, scope_id="space_b")

    assert topomap.match_image(solid_image((220, 30, 30)), top_k=1)[0].node_id == "n000000"
    assert [match.node_id for match in scoped_matches] == ["n000001", "n000002"]
    assert route.start.node_id == "n000001"
    assert route.start.rank == 1
    assert route.goal.node_id == "n000002"
    assert route.node_ids == ["n000001", "n000002"]


def test_get_sequence_refuses_when_current_frame_is_outside_goal_room(tmp_path) -> None:
    topomap = make_scoped_topomap(tmp_path, scoped_start_score=0.1)

    try:
        topomap.get_sequence(solid_image((220, 30, 30)), "node:n000002", start_top_k=3)
    except RuntimeError as exc:
        assert "below min_start_match_score" in str(exc)
    else:
        raise AssertionError("expected scoped route initialization to fail")


def test_route_follower_advances_on_goal_image_match(tmp_path) -> None:
    topomap = make_topomap(tmp_path)
    follower = TopomapRouteFollower(topomap, reached_threshold=0.98)
    follower.reset(solid_image((220, 30, 30)), "toilet")

    status0 = follower.update(solid_image((220, 30, 30)))
    status1 = follower.update(solid_image((30, 220, 30)))
    status2 = follower.update(solid_image((30, 30, 220)))

    assert status0.advanced is True
    assert status0.current_node_id == "n000001"
    assert status1.advanced is True
    assert status1.current_node_id == "n000002"
    assert status2.reached_goal is True


def test_build_pose_edges_uses_heading_alignment() -> None:
    nodes = [
        {"id": "a", "pose": {"x": 0.0, "z": 0.0, "yaw_rad": math.pi / 2.0}},
        {"id": "b", "pose": {"x": 1.0, "z": 0.0, "yaw_rad": math.pi / 2.0}},
        {"id": "c", "pose": {"x": 0.0, "z": 1.0, "yaw_rad": math.pi}},
    ]

    edges = build_pose_edges(nodes, neighbor_count=2, edge_radius_m=1.2, edge_max_heading_error_deg=45.0)
    pairs = {(edge["source"], edge["target"]) for edge in edges}

    assert ("a", "b") in pairs
    assert ("a", "c") not in pairs


def test_nomad_waypoint_conversion_to_flatdisk_motors() -> None:
    linear, angular = waypoint_to_twist(
        [0.05, 0.01],
        waypoint_dt_s=0.1,
        max_v_mps=0.5,
        max_w_rad_s=1.0,
    )
    motors = twist_to_motor_percent(
        linear,
        angular,
        wheel_base_m=0.2,
        max_wheel_speed_mps=1.0,
        max_abs_percent=50.0,
    )

    assert linear > 0.0
    assert angular > 0.0
    assert motors[0] > motors[1]
    assert max(abs(motors[0]), abs(motors[1])) <= 50


def test_semantic_terms_ignore_numeric_object_id_noise() -> None:
    terms = semantic_terms_for_object(
        {
            "objectType": "ToiletPaperHanger",
            "name": "ToiletPaperHanger_1686be5e_44",
            "objectId": "ToiletPaperHanger|+01.23|+04.56",
        }
    )

    assert "toilet" in terms
    assert "paper" in terms
    assert "hanger" in terms
    assert "1686be5e" not in terms
    assert "44" not in terms
