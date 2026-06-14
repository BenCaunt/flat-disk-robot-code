from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from flatdisk_sim.thor_backend import (
    DEFAULT_CAMERA_FORWARD_OFFSET_M,
    DEFAULT_CAMERA_HEIGHT_M,
    FlatDiskThorSim,
    ThorSimConfig,
)
from flatdisk_sim.semantic_topomap import TopomapBuildConfig, build_semantic_topomap_from_sim


class FakeThorEvent:
    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        frame: np.ndarray,
        third_party_camera_frames: list[np.ndarray] | None = None,
    ) -> None:
        self.metadata = metadata
        self.frame = frame
        self.third_party_camera_frames = third_party_camera_frames or []

    def __bool__(self) -> bool:
        return bool(self.metadata.get("lastActionSuccess", True))


class FakeThorController:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.actions: list[tuple[str, dict[str, Any]]] = []
        self.position = {"x": 0.0, "y": 0.9, "z": 0.0}
        self.yaw_deg = 0.0
        self.house: dict[str, Any] | None = None
        self.third_party_cameras: list[dict[str, Any]] = []
        self.last_event = self._event()
        self.stopped = False

    def step(self, *, action: str, **kwargs: Any) -> FakeThorEvent:
        self.actions.append((action, kwargs))
        if action == "CreateHouse":
            self.house = kwargs["house"]
        elif action == "GetReachablePositions":
            return self._event(action_return=[{"x": 1.0, "y": 0.9, "z": 2.0}])
        elif action == "TeleportFull":
            self.position = {"x": float(kwargs["x"]), "y": float(kwargs["y"]), "z": float(kwargs["z"])}
            self.yaw_deg = float(kwargs.get("rotation", {}).get("y", self.yaw_deg))
        elif action == "RotateRight":
            self.yaw_deg += float(kwargs["degrees"])
        elif action == "RotateLeft":
            self.yaw_deg -= float(kwargs["degrees"])
        elif action == "MoveAhead":
            self._move(float(kwargs["moveMagnitude"]))
        elif action == "MoveBack":
            self._move(-float(kwargs["moveMagnitude"]))
        elif action == "AddThirdPartyCamera":
            camera = {
                "position": dict(kwargs["position"]),
                "rotation": dict(kwargs["rotation"]),
                "fieldOfView": kwargs["fieldOfView"],
            }
            self.third_party_cameras.append(camera)
        elif action == "UpdateThirdPartyCamera":
            camera_id = int(kwargs["thirdPartyCameraId"])
            self.third_party_cameras[camera_id] = {
                "position": dict(kwargs["position"]),
                "rotation": dict(kwargs["rotation"]),
                "fieldOfView": kwargs["fieldOfView"],
            }
        self.last_event = self._event()
        return self.last_event

    def stop(self) -> None:
        self.stopped = True

    def _move(self, magnitude: float) -> None:
        yaw_rad = math.radians(self.yaw_deg)
        self.position["x"] += math.sin(yaw_rad) * magnitude
        self.position["z"] += math.cos(yaw_rad) * magnitude

    def _event(self, *, success: bool = True, action_return: Any = None) -> FakeThorEvent:
        width = int(self.kwargs.get("width", 64))
        height = int(self.kwargs.get("height", 48))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 64
        frame[:, :, 1] = 96
        frame[:, :, 2] = 128
        metadata = {
            "agent": {
                "position": dict(self.position),
                "rotation": {"x": 0.0, "y": self.yaw_deg, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "objects": [
                {
                    "name": "ArmChair_1",
                    "objectId": "ArmChair|1",
                    "objectType": "ArmChair",
                    "visible": True,
                    "position": {"x": 1.25, "y": 0.0, "z": 2.5},
                    "axisAlignedBoundingBox": {
                        "center": {"x": 1.25, "y": 0.5, "z": 2.5},
                        "size": {"x": 1.0, "y": 1.0, "z": 1.4},
                    },
                }
            ],
            "lastActionSuccess": success,
            "errorMessage": "",
        }
        if action_return is not None:
            metadata["actionReturn"] = action_return
        if self.third_party_cameras:
            metadata["thirdPartyCameras"] = list(self.third_party_cameras)
        third_party_frames = [self._third_party_frame(camera) for camera in self.third_party_cameras]
        return FakeThorEvent(metadata=metadata, frame=frame, third_party_camera_frames=third_party_frames)

    def _third_party_frame(self, camera: dict[str, Any]) -> np.ndarray:
        width = int(self.kwargs.get("width", 64))
        height = int(self.kwargs.get("height", 48))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        y = float(camera["position"]["y"])
        frame[:, :, 0] = min(255, int(round(y * 1000.0)))
        frame[:, :, 1] = 32
        frame[:, :, 2] = 16
        return frame


def make_sim(config: ThorSimConfig) -> tuple[FlatDiskThorSim, FakeThorController]:
    controllers: list[FakeThorController] = []

    def factory(**kwargs: Any) -> FakeThorController:
        controller = FakeThorController(**kwargs)
        controllers.append(controller)
        return controller

    sim = FlatDiskThorSim(config, controller_factory=factory)
    return sim, controllers[0]


def test_ithor_backend_initializes_controller_and_frame_size() -> None:
    sim, controller = make_sim(ThorSimConfig(backend="ithor", scene="FloorPlan301", width=80, height=60))

    assert controller.kwargs["scene"] == "FloorPlan301"
    assert controller.kwargs["snapToGrid"] is False
    assert controller.kwargs["width"] == 80
    assert controller.kwargs["height"] == 60
    assert controller.kwargs["cameraY"] == DEFAULT_CAMERA_HEIGHT_M
    assert controller.kwargs["cameraNearPlane"] == 0.03
    assert controller.kwargs["cameraFarPlane"] == 20.0
    assert math.isclose(controller.kwargs["fieldOfView"], 53.667999, rel_tol=1e-6)
    assert math.isclose(sim.hidden_pose()["camera"]["horizontal_fov_deg"], 68.0)
    assert sim.hidden_pose()["camera"]["source"] == "third-party"
    add_camera = [(name, args) for name, args in controller.actions if name == "AddThirdPartyCamera"]
    assert add_camera
    assert add_camera[-1][1]["position"]["x"] == 0.0
    assert add_camera[-1][1]["position"]["y"] == DEFAULT_CAMERA_HEIGHT_M
    assert math.isclose(add_camera[-1][1]["position"]["z"], DEFAULT_CAMERA_FORWARD_OFFSET_M)
    image = sim.render_image()
    update_camera = [(name, args) for name, args in controller.actions if name == "UpdateThirdPartyCamera"]
    assert update_camera
    assert update_camera[-1][1]["position"]["x"] == 0.0
    assert update_camera[-1][1]["position"]["y"] == DEFAULT_CAMERA_HEIGHT_M
    assert math.isclose(update_camera[-1][1]["position"]["z"], DEFAULT_CAMERA_FORWARD_OFFSET_M)
    assert image.size == (80, 60)
    assert image.getpixel((0, 0)) == (105, 32, 16)
    assert sim.hidden_pose()["scene"] == "FloorPlan301"
    assert sim.hidden_pose()["objects"][0]["objectType"] == "ArmChair"
    assert sim.hidden_pose()["objects"][0]["axisAlignedBoundingBox"]["min"]["x"] == 0.75


def test_vertical_fov_axis_passes_ai2thor_field_of_view_directly() -> None:
    sim, controller = make_sim(
        ThorSimConfig(
            backend="ithor",
            scene="FloorPlan301",
            width=640,
            height=480,
            field_of_view=48.0,
            field_of_view_axis="vertical",
            camera_height_m=0.033125,
        )
    )

    assert controller.kwargs["fieldOfView"] == 48.0
    assert controller.kwargs["cameraY"] == 0.033125
    assert controller.actions[-1][1]["position"]["y"] == 0.033125
    assert sim.camera_settings.source == "vertical-fov"


def test_robot_camera_forward_offset_follows_heading() -> None:
    sim, controller = make_sim(
        ThorSimConfig(
            backend="ithor",
            scene="FloorPlan301",
            camera_forward_offset_m=0.2,
            start_yaw_deg=90.0,
        )
    )

    add_camera = [(name, args) for name, args in controller.actions if name == "AddThirdPartyCamera"]
    assert add_camera
    assert math.isclose(add_camera[-1][1]["position"]["x"], 0.2)
    assert math.isclose(add_camera[-1][1]["position"]["z"], 0.0, abs_tol=1e-9)
    assert sim.hidden_pose()["camera"]["forward_offset_m"] == 0.2


def test_exact_start_pose_overrides_default_agent_pose() -> None:
    sim, controller = make_sim(
        ThorSimConfig(
            backend="ithor",
            scene="FloorPlan301",
            start_x=1.0,
            start_y=0.9,
            start_z=2.0,
            start_yaw_deg=135.0,
        )
    )

    teleports = [(name, args) for name, args in controller.actions if name == "TeleportFull"]
    assert teleports
    assert teleports[-1][1]["x"] == 1.0
    assert teleports[-1][1]["y"] == 0.9
    assert teleports[-1][1]["z"] == 2.0
    assert teleports[-1][1]["rotation"]["y"] == 135.0
    assert teleports[-1][1]["standing"] is True
    assert sim.hidden_pose()["x"] == 1.0
    assert sim.hidden_pose()["z"] == 2.0
    assert math.isclose(sim.hidden_pose()["yaw_deg"], 135.0)


def test_privileged_reachable_positions_and_teleport_for_map_building() -> None:
    sim, controller = make_sim(ThorSimConfig(backend="ithor", scene="FloorPlan301"))

    positions = sim.privileged_reachable_positions()
    ok = sim.privileged_teleport(x=1.0, y=0.9, z=2.0, yaw_rad=math.radians(90.0))

    assert positions == [{"x": 1.0, "y": 0.9, "z": 2.0}]
    assert ok is True
    assert sim.hidden_pose()["x"] == 1.0
    assert sim.hidden_pose()["z"] == 2.0
    assert math.isclose(sim.hidden_pose()["yaw_deg"], 90.0)
    update_camera = [(name, args) for name, args in controller.actions if name == "UpdateThirdPartyCamera"]
    assert update_camera
    assert math.isclose(update_camera[-1][1]["position"]["x"], 1.0 + DEFAULT_CAMERA_FORWARD_OFFSET_M)


def test_privileged_semantic_topomap_builder_writes_images_and_manifest(tmp_path) -> None:
    sim, _controller = make_sim(ThorSimConfig(backend="ithor", scene="FloorPlan301", width=64, height=48))

    topomap = build_semantic_topomap_from_sim(
        sim,
        tmp_path / "map",
        config=TopomapBuildConfig(max_positions=1, yaw_count=2, object_view_count=0, clean=True),
    )

    assert (topomap.map_dir / "semantic_topomap.json").exists()
    assert (topomap.map_dir / "descriptors.npy").exists()
    assert len(topomap.nodes) == 2
    assert len(topomap.edges) >= 1
    assert (topomap.map_dir / topomap.nodes[0]["image"]).exists()
    assert "arm chair" in topomap.nodes[0]["terms"]


def test_camera_calibration_overrides_field_of_view(tmp_path) -> None:
    calibration_path = tmp_path / "camera_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "image_width": 640,
                "image_height": 480,
                "camera_matrix": [[533.5, 0.0, 320.0], [0.0, 533.5, 240.0], [0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )

    sim, controller = make_sim(
        ThorSimConfig(
            backend="ithor",
            scene="FloorPlan301",
            width=320,
            height=240,
            field_of_view=120.0,
            camera_calibration=calibration_path,
        )
    )

    assert math.isclose(controller.kwargs["fieldOfView"], 48.442093, rel_tol=1e-6)
    assert sim.camera_settings.source == str(calibration_path)


def test_motor_percent_translates_to_ai2thor_moveahead() -> None:
    sim, controller = make_sim(ThorSimConfig(backend="ithor", scene="FloorPlan201"))

    sim.set_motor_percent(20.0, 20.0)
    sim.step(0.25)

    move_actions = [(name, args) for name, args in controller.actions if name == "MoveAhead"]
    assert move_actions
    assert move_actions[-1][1]["moveMagnitude"] > 0.0
    assert sim.hidden_pose()["z"] > 0.0


def test_differential_motor_translates_to_ai2thor_rotation() -> None:
    sim, controller = make_sim(ThorSimConfig(backend="ithor", scene="FloorPlan201"))

    sim.set_motor_percent(30.0, -30.0)
    sim.step(0.25)

    rotate_actions = [(name, args) for name, args in controller.actions if name == "RotateRight"]
    assert rotate_actions
    assert rotate_actions[-1][1]["degrees"] > 0.0
    assert sim.hidden_pose()["yaw_deg"] > 0.0


def test_house_json_backend_creates_procedural_scene_and_random_start(tmp_path) -> None:
    house_path = tmp_path / "house.json"
    house_path.write_text(json.dumps({"metadata": {"roomSpecId": "test_spec"}}), encoding="utf-8")

    sim, controller = make_sim(
        ThorSimConfig(backend="house-json", house_json=house_path, random_start=True, procthor_seed=7)
    )

    assert controller.kwargs["scene"] == "Procedural"
    assert controller.kwargs["branch"] == "nanna"
    assert controller.house == {"metadata": {"roomSpecId": "test_spec", "schema": "1.0.0"}, "proceduralParameters": {}}
    assert [name for name, _args in controller.actions] == [
        "CreateHouse",
        "GetReachablePositions",
        "TeleportFull",
        "AddThirdPartyCamera",
    ]
    assert sim.hidden_pose()["world"] == str(house_path)


def test_house_metadata_agent_pose_is_used_without_random_start(tmp_path) -> None:
    house_path = tmp_path / "house.json"
    house_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "agent": {
                        "x": 1.25,
                        "y": 0.9,
                        "z": -0.5,
                        "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                        "horizon": 0.0,
                    }
                },
                "proceduralParameters": {"ceilingMaterial": "PureWhite"},
                "rooms": [{"floorMaterial": "Wood"}],
                "walls": [{"material": "Drywall"}],
            }
        ),
        encoding="utf-8",
    )

    sim, controller = make_sim(ThorSimConfig(backend="house-json", house_json=house_path))

    assert controller.house["proceduralParameters"]["ceilingMaterial"] == {"name": "PureWhite"}
    assert controller.house["rooms"][0]["floorMaterial"] == {"name": "Wood"}
    assert controller.house["walls"][0]["material"] == {"name": "Drywall"}
    assert [name for name, _args in controller.actions] == ["CreateHouse", "TeleportFull", "AddThirdPartyCamera"]
    assert sim.hidden_pose()["x"] == 1.25
    assert sim.hidden_pose()["z"] == -0.5
    assert controller.actions[-1][1]["rotation"]["y"] == 90.0
