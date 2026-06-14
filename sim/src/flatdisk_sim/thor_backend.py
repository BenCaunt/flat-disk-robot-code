"""AI2-THOR and ProcTHOR simulation backend for the flat disk bridge."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Callable
import warnings

import numpy as np
from PIL import Image

from .protocol import clamp, wrap_pi


ControllerFactory = Callable[..., Any]

DEFAULT_ITHOR_SCENE = "FloorPlan201"
DEFAULT_CAMERA_HEIGHT_M = 0.105
DEFAULT_CAMERA_FORWARD_OFFSET_M = 0.22
URDF_CAMERA_FORWARD_OFFSET_M = 0.102
DEFAULT_CAMERA_HORIZONTAL_FOV_DEG = 68.0
DEFAULT_CAMERA_NEAR_PLANE_M = 0.03
DEFAULT_CAMERA_FAR_PLANE_M = 20.0
ITHOR_SCENE_GROUPS: dict[str, tuple[str, ...]] = {
    "kitchens": tuple(f"FloorPlan{i}" for i in range(1, 31)),
    "living_rooms": tuple(f"FloorPlan{i}" for i in range(201, 231)),
    "bedrooms": tuple(f"FloorPlan{i}" for i in range(301, 331)),
    "bathrooms": tuple(f"FloorPlan{i}" for i in range(401, 431)),
}
ITHOR_SCENES: tuple[str, ...] = tuple(scene for scenes in ITHOR_SCENE_GROUPS.values() for scene in scenes)


class ThorBackendUnavailable(RuntimeError):
    """Raised when AI2-THOR or ProcTHOR is not installed."""


@dataclass(frozen=True)
class ThorSimConfig:
    backend: str = "procthor"
    scene: str = DEFAULT_ITHOR_SCENE
    house_json: Path | None = None
    procthor_seed: int = 42
    procthor_split: str = "train"
    random_start: bool = False
    width: int = 640
    height: int = 480
    field_of_view: float = DEFAULT_CAMERA_HORIZONTAL_FOV_DEG
    field_of_view_axis: str = "horizontal"
    camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M
    camera_forward_offset_m: float = DEFAULT_CAMERA_FORWARD_OFFSET_M
    camera_near_plane_m: float = DEFAULT_CAMERA_NEAR_PLANE_M
    camera_far_plane_m: float = DEFAULT_CAMERA_FAR_PLANE_M
    camera_calibration: Path | None = None
    use_third_party_camera: bool = True
    grid_size: float = 0.05
    rotate_step_degrees: float = 5.0
    visibility_distance: float = 1.5
    quality: str = "Low"
    wheel_base_m: float = 0.215
    max_wheel_speed_mps: float = 0.78
    min_move_m: float = 0.01
    min_rotate_deg: float = 1.0
    start_x: float | None = None
    start_y: float | None = None
    start_z: float | None = None
    start_yaw_deg: float | None = None


@dataclass(frozen=True)
class ThorCameraSettings:
    vertical_fov_deg: float
    horizontal_fov_deg: float
    source: str


@dataclass
class ThorRobotState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0
    motor1_percent: float = 0.0
    motor2_percent: float = 0.0
    linear_accel_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    collided: bool = False
    last_action_success: bool = True
    error_message: str = ""


class FlatDiskThorSim:
    """Translate flat disk motor commands into AI2-THOR navigation actions."""

    def __init__(
        self,
        config: ThorSimConfig,
        *,
        controller_factory: ControllerFactory | None = None,
    ) -> None:
        self.config = config
        self.backend_name = _normalize_backend(config.backend, config.house_json)
        self.world_name = ""
        self.scene_name = ""
        self.state = ThorRobotState()
        self._controller_factory = controller_factory
        self._controller: Any | None = None
        self._event: Any | None = None
        self._rng = random.Random(config.procthor_seed)
        self.camera_settings = _resolve_camera_settings(config)
        self._robot_camera_id = 0
        self._robot_camera_added = False
        self._house: dict[str, Any] | None = None
        self._move_remainder_m = 0.0
        self._yaw_remainder_rad = 0.0
        self._last_forward_speed_mps = 0.0
        self._last_lateral_speed_mps = 0.0
        self._start_controller()

    def set_motor_percent(self, motor1: float, motor2: float) -> None:
        self.state.motor1_percent = clamp(motor1, -100.0, 100.0)
        self.state.motor2_percent = clamp(motor2, -100.0, 100.0)

    def set_motor_us(self, motor1_us: float, motor2_us: float) -> None:
        m1 = (motor1_us - 1500.0) / 5.0
        m2 = (motor2_us - 1500.0) / 5.0
        self.set_motor_percent(m1, m2)

    def stop(self) -> None:
        self.set_motor_percent(0.0, 0.0)
        self._move_remainder_m = 0.0
        self._yaw_remainder_rad = 0.0
        self.state.yaw_rate = 0.0
        self.state.linear_accel_body = (0.0, 0.0, 0.0)

    def close(self) -> None:
        if self._controller is None:
            return
        close = getattr(self._controller, "stop", None) or getattr(self._controller, "close", None)
        if close is not None:
            close()
        self._controller = None

    def step(self, dt: float) -> None:
        dt = max(0.0, min(dt, 0.25))
        if dt <= 0.0:
            return

        m1 = self.state.motor1_percent / 100.0 * self.config.max_wheel_speed_mps
        m2 = self.state.motor2_percent / 100.0 * self.config.max_wheel_speed_mps
        forward_speed = (m1 + m2) * 0.5
        yaw_rate = (m1 - m2) / self.config.wheel_base_m

        self._move_remainder_m += forward_speed * dt
        self._yaw_remainder_rad += yaw_rate * dt
        self.state.yaw_rate = yaw_rate

        collided = False
        yaw_delta_deg = math.degrees(self._yaw_remainder_rad)
        if abs(yaw_delta_deg) >= self.config.min_rotate_deg:
            action = "RotateRight" if yaw_delta_deg > 0.0 else "RotateLeft"
            event = self._step_controller(action=action, degrees=abs(yaw_delta_deg))
            collided = collided or not _event_success(event)
            self._yaw_remainder_rad = 0.0

        move_delta_m = self._move_remainder_m
        if abs(move_delta_m) >= self.config.min_move_m:
            action = "MoveAhead" if move_delta_m > 0.0 else "MoveBack"
            event = self._step_controller(action=action, moveMagnitude=abs(move_delta_m))
            collided = collided or not _event_success(event)
            self._move_remainder_m = 0.0

        accel_x = (forward_speed - self._last_forward_speed_mps) / dt if dt > 1e-6 else 0.0
        accel_y = (0.0 - self._last_lateral_speed_mps) / dt if dt > 1e-6 else 0.0
        self.state.linear_accel_body = (accel_x, accel_y, 0.0)
        self.state.collided = collided
        self._last_forward_speed_mps = forward_speed
        self._last_lateral_speed_mps = 0.0
        self._sync_state_from_event(yaw_rate=yaw_rate, collided=collided)

    def render_image(self) -> Image.Image:
        event = self._sync_robot_camera() if self.config.use_third_party_camera else self._event
        frame = None
        if self.config.use_third_party_camera:
            frames = getattr(event, "third_party_camera_frames", []) if event is not None else []
            if len(frames) > self._robot_camera_id:
                frame = frames[self._robot_camera_id]
        if frame is None:
            frame = getattr(event, "frame", None) if event is not None else None
        if frame is None:
            event = self._step_controller(action="Done")
            if self.config.use_third_party_camera:
                frames = getattr(event, "third_party_camera_frames", [])
                if len(frames) > self._robot_camera_id:
                    frame = frames[self._robot_camera_id]
            if frame is None:
                frame = getattr(event, "frame", None)
        if frame is None:
            raise RuntimeError("AI2-THOR did not return an RGB frame")

        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise RuntimeError(f"AI2-THOR returned an unexpected frame shape: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr[:, :, :3])

    def privileged_reachable_positions(self) -> list[dict[str, float]]:
        """Return AI2-THOR reachable floor positions for offline map building."""

        event = self._step_controller(action="GetReachablePositions")
        values = event.metadata.get("actionReturn", []) if event is not None else []
        positions: list[dict[str, float]] = []
        if not isinstance(values, list):
            return positions
        for value in values:
            if not isinstance(value, dict):
                continue
            try:
                positions.append(
                    {
                        "x": float(value["x"]),
                        "y": float(value.get("y", self.state.y)),
                        "z": float(value["z"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return positions

    def privileged_teleport(
        self,
        *,
        x: float,
        z: float,
        yaw_rad: float,
        y: float | None = None,
    ) -> bool:
        """Teleport the simulator agent for privileged offline data collection."""

        event = self._step_controller(
            action="TeleportFull",
            x=float(x),
            y=float(self.state.y if y is None else y),
            z=float(z),
            rotation={"x": 0.0, "y": math.degrees(yaw_rad), "z": 0.0},
            horizon=0.0,
            standing=True,
        )
        ok = _event_success(event)
        self._move_remainder_m = 0.0
        self._yaw_remainder_rad = 0.0
        self._sync_state_from_event(yaw_rate=0.0, collided=not ok)
        if ok and self.config.use_third_party_camera:
            self._sync_robot_camera()
        return ok

    def hidden_pose(self) -> dict[str, Any]:
        return {
            "x": self.state.x,
            "y": self.state.y,
            "z": self.state.z,
            "yaw_deg": math.degrees(self.state.yaw),
            "backend": self.backend_name,
            "world": self.world_name,
            "scene": self.scene_name,
            "objects": self.hidden_objects(),
            "camera": {
                "width": self.config.width,
                "height": self.config.height,
                "height_m": self.config.camera_height_m,
                "forward_offset_m": self.config.camera_forward_offset_m,
                "horizontal_fov_deg": self.camera_settings.horizontal_fov_deg,
                "vertical_fov_deg": self.camera_settings.vertical_fov_deg,
                "near_plane_m": self.config.camera_near_plane_m,
                "far_plane_m": self.config.camera_far_plane_m,
                "calibration": str(self.config.camera_calibration) if self.config.camera_calibration else None,
                "source": "third-party" if self.config.use_third_party_camera else "agent",
                "fov_source": self.camera_settings.source,
            },
        }

    def hidden_objects(self) -> list[dict[str, Any]]:
        metadata = getattr(self._event, "metadata", {}) if self._event is not None else {}
        objects = metadata.get("objects", [])
        result: list[dict[str, Any]] = []
        if not isinstance(objects, list):
            return result
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            position = obj.get("position", {})
            if not isinstance(position, dict):
                continue
            try:
                x = float(position["x"])
                y = float(position.get("y", 0.0))
                z = float(position["z"])
            except (KeyError, TypeError, ValueError):
                continue
            hidden_obj: dict[str, Any] = {
                "name": obj.get("name"),
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "visible": bool(obj.get("visible", False)),
                "position": {"x": x, "y": y, "z": z},
            }
            bounds = _hidden_axis_aligned_bounds(obj.get("axisAlignedBoundingBox"))
            if bounds is not None:
                hidden_obj["axisAlignedBoundingBox"] = bounds
            result.append(hidden_obj)
        return result

    def _start_controller(self) -> None:
        if self.backend_name == "ithor":
            scene = self._choose_ithor_scene()
            self.scene_name = scene
            self.world_name = scene
            self._controller = self._make_controller(scene=scene)
            self._event = getattr(self._controller, "last_event", None)
            if self._event is None:
                self._event = self._step_controller(action="Done")
        elif self.backend_name in {"procthor", "house-json"}:
            house = self._load_house()
            self._house = house
            self.scene_name = "Procedural"
            self.world_name = _house_name(self.config, house)
            self._controller = self._make_controller(scene="Procedural")
            self._event = self._step_controller(action="CreateHouse", house=house)
            if not _event_success(self._event):
                raise RuntimeError(f"AI2-THOR CreateHouse failed: {_event_error(self._event)}")
        else:
            raise ValueError(f"unsupported THOR backend: {self.backend_name}")

        self._place_agent()
        self._sync_state_from_event(yaw_rate=0.0, collided=False)
        if self.config.use_third_party_camera:
            self._sync_robot_camera()

    def _make_controller(self, *, scene: str) -> Any:
        params = {
            "agentMode": "default",
            "visibilityDistance": self.config.visibility_distance,
            "scene": scene,
            "gridSize": self.config.grid_size,
            "snapToGrid": False,
            "rotateStepDegrees": self.config.rotate_step_degrees,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": self.config.width,
            "height": self.config.height,
            "fieldOfView": self.camera_settings.vertical_fov_deg,
            "cameraY": self.config.camera_height_m,
            "cameraNearPlane": self.config.camera_near_plane_m,
            "cameraFarPlane": self.config.camera_far_plane_m,
            "quality": self.config.quality,
        }
        if scene == "Procedural":
            params["branch"] = "nanna"
        if self._controller_factory is not None:
            return self._controller_factory(**params)

        try:
            from ai2thor.controller import Controller
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ThorBackendUnavailable(
                "AI2-THOR is not installed. Run `uv sync --extra thor` from sim/."
            ) from exc
        return Controller(**params)

    def _load_house(self) -> dict[str, Any]:
        if self.config.house_json is not None:
            house = json.loads(self.config.house_json.read_text(encoding="utf-8"))
        else:
            house = _generate_procthor_house(seed=self.config.procthor_seed, split=self.config.procthor_split)
        return _normalize_house_schema_for_ai2thor(house)

    def _choose_ithor_scene(self) -> str:
        scene = self.config.scene
        if scene == "random":
            return self._rng.choice(ITHOR_SCENES)
        if scene in ITHOR_SCENE_GROUPS:
            return self._rng.choice(ITHOR_SCENE_GROUPS[scene])
        return scene

    def _place_agent(self) -> None:
        if self.config.start_x is not None and self.config.start_z is not None:
            agent = _event_agent(self._event)
            position = agent.get("position", {})
            yaw = self.config.start_yaw_deg
            if yaw is None:
                yaw = float(agent.get("rotation", {}).get("y", 0.0))
            self._event = self._step_controller(
                action="TeleportFull",
                x=float(self.config.start_x),
                y=float(position.get("y", 0.0) if self.config.start_y is None else self.config.start_y),
                z=float(self.config.start_z),
                rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
                horizon=0.0,
                standing=True,
            )
            return

        if self.config.random_start:
            event = self._step_controller(action="GetReachablePositions")
            positions = event.metadata.get("actionReturn", []) if event is not None else []
            if positions:
                position = self._rng.choice(positions)
                yaw = self.config.start_yaw_deg
                if yaw is None:
                    yaw = self._rng.randrange(0, 360)
                self._event = self._step_controller(
                    action="TeleportFull",
                    x=position["x"],
                    y=position["y"],
                    z=position["z"],
                    rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
                    horizon=0.0,
                )
            return

        if self.backend_name in {"procthor", "house-json"}:
            agent_pose = _house_agent_pose(self._house or {})
            if agent_pose is not None:
                self._event = self._step_controller(action="TeleportFull", **_teleport_args_from_agent_pose(agent_pose))
                return

        if self.config.start_yaw_deg is not None:
            agent = _event_agent(self._event)
            position = agent.get("position", {})
            self._event = self._step_controller(
                action="TeleportFull",
                x=position.get("x", 0.0),
                y=position.get("y", 0.0),
                z=position.get("z", 0.0),
                rotation={"x": 0.0, "y": float(self.config.start_yaw_deg), "z": 0.0},
                horizon=agent.get("cameraHorizon", 0.0),
            )

    def _step_controller(self, **kwargs: Any) -> Any:
        if self._controller is None:
            raise RuntimeError("AI2-THOR controller is not started")
        event = self._controller.step(**kwargs)
        self._event = event
        return event

    def _sync_robot_camera(self) -> Any:
        forward_x = math.sin(self.state.yaw) * self.config.camera_forward_offset_m
        forward_z = math.cos(self.state.yaw) * self.config.camera_forward_offset_m
        position = {
            "x": self.state.x + forward_x,
            "y": self.config.camera_height_m,
            "z": self.state.z + forward_z,
        }
        rotation = {
            "x": 0.0,
            "y": math.degrees(self.state.yaw),
            "z": 0.0,
        }
        if self._robot_camera_added:
            return self._step_controller(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self._robot_camera_id,
                position=position,
                rotation=rotation,
                fieldOfView=self.camera_settings.vertical_fov_deg,
            )
        event = self._step_controller(
            action="AddThirdPartyCamera",
            position=position,
            rotation=rotation,
            fieldOfView=self.camera_settings.vertical_fov_deg,
        )
        self._robot_camera_added = True
        return event

    def _sync_state_from_event(self, *, yaw_rate: float, collided: bool) -> None:
        event = self._event
        agent = _event_agent(event)
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})
        self.state.x = float(position.get("x", self.state.x))
        self.state.y = float(position.get("y", self.state.y))
        self.state.z = float(position.get("z", self.state.z))
        self.state.yaw = wrap_pi(math.radians(float(rotation.get("y", math.degrees(self.state.yaw)))))
        self.state.yaw_rate = yaw_rate
        self.state.collided = collided
        self.state.last_action_success = _event_success(event)
        self.state.error_message = _event_error(event)


def _hidden_axis_aligned_bounds(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    center = payload.get("center")
    size = payload.get("size")
    if not isinstance(center, dict) or not isinstance(size, dict):
        return None
    try:
        cx = float(center["x"])
        cy = float(center.get("y", 0.0))
        cz = float(center["z"])
        sx = float(size["x"])
        sy = float(size.get("y", 0.0))
        sz = float(size["z"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "center": {"x": cx, "y": cy, "z": cz},
        "size": {"x": sx, "y": sy, "z": sz},
        "min": {"x": cx - sx * 0.5, "y": cy - sy * 0.5, "z": cz - sz * 0.5},
        "max": {"x": cx + sx * 0.5, "y": cy + sy * 0.5, "z": cz + sz * 0.5},
    }


def _resolve_camera_settings(config: ThorSimConfig) -> ThorCameraSettings:
    if config.width <= 0 or config.height <= 0:
        raise ValueError("camera width and height must be positive")
    if config.camera_height_m <= 0.0:
        raise ValueError("camera_height_m must be positive")
    if config.camera_forward_offset_m < 0.0:
        raise ValueError("camera_forward_offset_m must be non-negative")
    if config.camera_near_plane_m <= 0.0:
        raise ValueError("camera_near_plane_m must be positive")
    if config.camera_far_plane_m <= config.camera_near_plane_m:
        raise ValueError("camera_far_plane_m must be greater than camera_near_plane_m")

    if config.camera_calibration is not None:
        horizontal_fov, vertical_fov = _fovs_from_calibration(
            config.camera_calibration,
            width=config.width,
            height=config.height,
        )
        return ThorCameraSettings(
            vertical_fov_deg=vertical_fov,
            horizontal_fov_deg=horizontal_fov,
            source=str(config.camera_calibration),
        )

    axis = config.field_of_view_axis.lower()
    if axis in {"horizontal", "x"}:
        horizontal_fov = _validated_fov(config.field_of_view, label="horizontal field of view")
        vertical_fov = _vertical_fov_from_horizontal(horizontal_fov, width=config.width, height=config.height)
    elif axis in {"vertical", "y"}:
        vertical_fov = _validated_fov(config.field_of_view, label="vertical field of view")
        horizontal_fov = _horizontal_fov_from_vertical(vertical_fov, width=config.width, height=config.height)
    else:
        raise ValueError("field_of_view_axis must be 'horizontal' or 'vertical'")

    return ThorCameraSettings(
        vertical_fov_deg=vertical_fov,
        horizontal_fov_deg=horizontal_fov,
        source=f"{axis}-fov",
    )


def _fovs_from_calibration(calibration_path: Path, *, width: int, height: int) -> tuple[float, float]:
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_width = int(data.get("image_width", width))
    source_height = int(data.get("image_height", height))
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"{calibration_path} has invalid image_width/image_height")

    fx: float | None = None
    fy: float | None = None

    camera_matrix = data.get("camera_matrix")
    if isinstance(camera_matrix, list) and len(camera_matrix) >= 2:
        try:
            fx = float(camera_matrix[0][0])
            fy = float(camera_matrix[1][1])
        except (TypeError, ValueError, IndexError):
            fx = None
            fy = None

    if fx is None or fy is None:
        colmap = data.get("colmap", {})
        model = str(colmap.get("camera_model", "")).upper()
        params = colmap.get("camera_params")
        if not isinstance(params, list):
            params_string = colmap.get("camera_params_string", "")
            if isinstance(params_string, str) and params_string:
                params = [float(part) for part in params_string.replace(",", " ").split()]
        if isinstance(params, list):
            fx, fy = _focal_lengths_from_colmap(model, [float(value) for value in params])

    if fx is None or fy is None or fx <= 0.0 or fy <= 0.0:
        raise RuntimeError(
            f"{calibration_path} must contain camera_matrix or colmap camera parameters with positive focal lengths"
        )

    scaled_fx = fx * width / source_width
    scaled_fy = fy * height / source_height
    horizontal_fov = math.degrees(2.0 * math.atan(width / (2.0 * scaled_fx)))
    vertical_fov = math.degrees(2.0 * math.atan(height / (2.0 * scaled_fy)))
    return (
        _validated_fov(horizontal_fov, label="calibrated horizontal field of view"),
        _validated_fov(vertical_fov, label="calibrated vertical field of view"),
    )


def _focal_lengths_from_colmap(model: str, params: list[float]) -> tuple[float | None, float | None]:
    single_focal_models = {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }
    dual_focal_models = {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    }
    if model in single_focal_models and len(params) >= 1:
        return params[0], params[0]
    if model in dual_focal_models and len(params) >= 2:
        return params[0], params[1]
    return None, None


def _vertical_fov_from_horizontal(horizontal_fov_deg: float, *, width: int, height: int) -> float:
    horizontal_rad = math.radians(horizontal_fov_deg)
    vertical_rad = 2.0 * math.atan((height / width) * math.tan(horizontal_rad * 0.5))
    return _validated_fov(math.degrees(vertical_rad), label="derived vertical field of view")


def _horizontal_fov_from_vertical(vertical_fov_deg: float, *, width: int, height: int) -> float:
    vertical_rad = math.radians(vertical_fov_deg)
    horizontal_rad = 2.0 * math.atan((width / height) * math.tan(vertical_rad * 0.5))
    return _validated_fov(math.degrees(horizontal_rad), label="derived horizontal field of view")


def _validated_fov(value: float, *, label: str) -> float:
    value = float(value)
    if not 1.0 <= value <= 179.0:
        raise ValueError(f"{label} must be between 1 and 179 degrees")
    return value


def _normalize_backend(backend: str, house_json: Path | None) -> str:
    normalized = backend.lower().replace("_", "-")
    if normalized in {"procedural", "proc-thor"}:
        return "procthor"
    if normalized in {"house", "json", "house-json"}:
        return "house-json"
    if normalized in {"ithor", "ai2thor"}:
        return "ithor"
    if house_json is not None:
        return "house-json"
    return normalized


def _event_agent(event: Any | None) -> dict[str, Any]:
    if event is None:
        return {}
    metadata = getattr(event, "metadata", {}) or {}
    agent = metadata.get("agent", {})
    return agent if isinstance(agent, dict) else {}


def _event_success(event: Any | None) -> bool:
    if event is None:
        return False
    metadata = getattr(event, "metadata", {}) or {}
    return bool(metadata.get("lastActionSuccess", bool(event)))


def _event_error(event: Any | None) -> str:
    if event is None:
        return "no event"
    metadata = getattr(event, "metadata", {}) or {}
    return str(metadata.get("errorMessage", ""))


def _house_name(config: ThorSimConfig, house: dict[str, Any]) -> str:
    if config.house_json is not None:
        return str(config.house_json)
    room_spec = house.get("metadata", {}).get("roomSpecId", "generated")
    return f"ProcTHOR {config.procthor_split} seed={config.procthor_seed} room_spec={room_spec}"


def _house_agent_pose(house: dict[str, Any]) -> dict[str, Any] | None:
    pose = house.get("metadata", {}).get("agent")
    return pose if isinstance(pose, dict) else None


def _teleport_args_from_agent_pose(pose: dict[str, Any]) -> dict[str, Any]:
    if "position" not in pose:
        return pose
    position = pose["position"]
    return {
        "x": position.get("x", 0.0),
        "y": position.get("y", 0.0),
        "z": position.get("z", 0.0),
        "rotation": pose.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0}),
        "horizon": 0.0,
        "standing": pose.get("standing", True),
    }


def _normalize_house_schema_for_ai2thor(house: dict[str, Any]) -> dict[str, Any]:
    """Make legacy ProcTHOR house JSON acceptable to current procedural builds."""

    house = _upgrade_legacy_procthor_house(house)

    def normalize(value: Any, *, key: str = "") -> Any:
        if isinstance(value, str) and key in {"material", "floorMaterial", "ceilingMaterial"}:
            return {"name": value}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {child_key: normalize(child_value, key=child_key) for child_key, child_value in value.items()}
        return value

    normalized = normalize(house)
    if not isinstance(normalized, dict):
        raise TypeError("house JSON must be an object")
    return normalized


def _upgrade_legacy_procthor_house(house: dict[str, Any]) -> dict[str, Any]:
    metadata = house.get("metadata")
    schema = metadata.get("schema") if isinstance(metadata, dict) else None
    if schema not in {None, "0.0.1"}:
        return house

    upgraded = copy.deepcopy(house)
    procedural = upgraded.setdefault("proceduralParameters", {})
    _materialize(procedural, "ceilingMaterial")
    _move_to_material(procedural, ["ceilingColor"], ["ceilingMaterial", "color"], delete=False)
    _move_to_material(procedural, ["ceilingMaterialTilingXDivisor"], ["ceilingMaterial", "tilingDivisorX"])
    _move_to_material(procedural, ["ceilingMaterialTilingYDivisor"], ["ceilingMaterial", "tilingDivisorY"])

    for room in upgraded.get("rooms") or []:
        _materialize(room, "floorMaterial")
        _move_to_material(room, ["floorColor"], ["floorMaterial", "color"])
        _move_to_material(room, ["floorMaterialTilingXDivisor"], ["floorMaterial", "tilingDivisorX"])
        _move_to_material(room, ["floorMaterialTilingYDivisor"], ["floorMaterial", "tilingDivisorY"])
        for ceiling in room.get("ceilings") or []:
            if "materialProperties" in ceiling:
                ceiling["material"] = ceiling.pop("materialProperties")
            _materialize(ceiling, "material")
            _move_to_material(ceiling, ["tilingDivisorX"], ["material", "tilingDivisorX"])
            _move_to_material(ceiling, ["tilingDivisorY"], ["material", "tilingDivisorY"])

    for wall in upgraded.get("walls") or []:
        if "materialId" in wall:
            wall.setdefault("material", {})["name"] = wall.pop("materialId")
        if "materialProperties" in wall:
            wall["material"] = wall.pop("materialProperties")
        _materialize(wall, "material")
        _move_to_material(wall, ["color"], ["material", "color"], delete=False)
        wall_id = str(wall.get("id", ""))
        parts = wall_id.split("|")
        if len(parts) > 1 and parts[1] == "exterior":
            wall["roomId"] = "exterior"

    for hole in [*(upgraded.get("windows") or []), *(upgraded.get("doors") or [])]:
        _materialize(hole, "material")
        _move_to_material(hole, ["color"], ["material", "color"], delete=False)
        _upgrade_legacy_hole(hole)

    for obj in upgraded.get("objects") or []:
        if "materialProperties" in obj:
            obj["material"] = obj.pop("materialProperties")
        _materialize(obj, "material")
        _move_to_material(obj, ["color"], ["material", "color"])

    upgraded.setdefault("metadata", {})["schema"] = "1.0.0"
    return upgraded


def _materialize(container: dict[str, Any], key: str) -> None:
    value = container.get(key)
    if isinstance(value, str):
        container[key] = {"name": value}
    elif value is None:
        return
    elif not isinstance(value, dict):
        container[key] = {"name": str(value)}


def _move_to_material(container: dict[str, Any], source: list[str], target: list[str], *, delete: bool = True) -> None:
    current: Any = container
    for key in source:
        if not isinstance(current, dict) or key not in current:
            return
        current = current[key]

    out = container
    for key in target[:-1]:
        value = out.get(key)
        if isinstance(value, str):
            value = {"name": value}
            out[key] = value
        elif value is None:
            value = {}
            out[key] = value
        out = value
    out[target[-1]] = current

    if delete:
        parent = container
        for key in source[:-1]:
            parent = parent[key]
        parent.pop(source[-1], None)


def _upgrade_legacy_hole(hole: dict[str, Any]) -> None:
    if "boundingBox" not in hole or "assetOffset" not in hole or "assetId" not in hole:
        return
    try:
        import procthor.databases as procthor_databases
    except Exception:
        return
    if hasattr(procthor_databases, "DEFAULT_PROCTHOR_DATABASE"):
        asset_database = procthor_databases.DEFAULT_PROCTHOR_DATABASE.ASSET_ID_DATABASE
    else:
        asset_database = procthor_databases.asset_id_database
    asset = asset_database.get(hole["assetId"])
    if asset is None:
        return
    bbox = hole.pop("boundingBox")
    offset = hole.pop("assetOffset")
    hole["holePolygon"] = [bbox["min"], bbox["max"]]
    hole["assetPosition"] = {
        "x": bbox["min"]["x"] + offset["x"] + asset["boundingBox"]["x"] / 2.0,
        "y": bbox["min"]["y"] + offset["y"] + asset["boundingBox"]["y"] / 2.0,
        "z": 0,
    }


def _generate_procthor_house(*, seed: int, split: str) -> dict[str, Any]:
    try:
        import procthor.generation as procthor_generation
        from procthor.generation import PROCTHOR10K_ROOM_SPEC_SAMPLER, HouseGenerator
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ThorBackendUnavailable(
            "ProcTHOR generation is not installed. Run `uv sync --extra procthor` "
            "or pass `--backend house-json --house-json path/to/house.json`."
        ) from exc

    original_add_small_objects = procthor_generation.add_small_objects

    def add_small_objects_with_fallback(*args: Any, **kwargs: Any) -> None:
        try:
            original_add_small_objects(*args, **kwargs)
        except AssertionError as exc:
            if "Unable to CreateHouse" not in str(exc):
                raise
            warnings.warn(
                "ProcTHOR small-object placement failed; continuing with furniture and room geometry only.",
                RuntimeWarning,
                stacklevel=2,
            )

    procthor_generation.add_small_objects = add_small_objects_with_fallback
    generator = HouseGenerator(split=split, seed=seed, room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER)
    try:
        house, _ = generator.sample()
        return json.loads(house.to_json())
    finally:
        procthor_generation.add_small_objects = original_add_small_objects
        controller = getattr(generator, "controller", None)
        stop = getattr(controller, "stop", None)
        if stop is not None:
            stop()
