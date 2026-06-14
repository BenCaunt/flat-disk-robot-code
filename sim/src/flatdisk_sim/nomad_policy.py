"""NoMaD adapter and flat-disk command conversion.

The upstream NoMaD release is ROS-oriented and depends on its training package
plus Stanford's ``diffusion_policy`` package. This module wraps that code behind
an optional adapter so the rest of the topological navigation stack can be
tested without importing heavy model dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image


DEFAULT_NOMAD_CONFIG: dict[str, Any] = {
    "model_type": "nomad",
    "vision_encoder": "nomad_vint",
    "encoding_size": 256,
    "obs_encoder": "efficientnet-b0",
    "cond_predict_scale": False,
    "mha_num_attention_heads": 4,
    "mha_num_attention_layers": 4,
    "mha_ff_dim_factor": 4,
    "down_dims": [64, 128, 256],
    "num_diffusion_iters": 10,
    "normalize": True,
    "context_size": 3,
    "len_traj_pred": 8,
    "learn_angle": False,
    "image_size": [96, 96],
}
ACTION_STATS = {
    "min": np.asarray([-2.5, -4.0], dtype=np.float32),
    "max": np.asarray([5.0, 4.0], dtype=np.float32),
}


class NoMaDUnavailable(RuntimeError):
    """Raised when optional upstream NoMaD dependencies are unavailable."""


@dataclass(frozen=True)
class WaypointCommand:
    waypoint: np.ndarray
    motor_percent: tuple[int, int]
    linear_mps: float
    angular_rad_s: float
    inference_ms: float
    distance: float | None = None
    selected_route_offset: int | None = None
    raw_waypoint: np.ndarray | None = None
    sampled_waypoints: np.ndarray | None = None


def resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def waypoint_to_twist(
    waypoint: Sequence[float],
    *,
    waypoint_dt_s: float,
    max_v_mps: float,
    max_w_rad_s: float,
) -> tuple[float, float]:
    values = np.asarray(waypoint, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError("NoMaD waypoint must contain at least dx, dy")
    dx = float(values[0])
    dy = float(values[1])
    if abs(dx) < 1e-6:
        v = 0.0
        w = math.copysign(max_w_rad_s, dy) if abs(dy) > 1e-6 else 0.0
    else:
        v = max(0.0, dx / max(waypoint_dt_s, 1e-3))
        w = math.atan2(dy, max(dx, 1e-6)) / max(waypoint_dt_s, 1e-3)
    return float(np.clip(v, 0.0, max_v_mps)), float(np.clip(w, -max_w_rad_s, max_w_rad_s))


def twist_to_motor_percent(
    linear_mps: float,
    angular_rad_s: float,
    *,
    wheel_base_m: float,
    max_wheel_speed_mps: float,
    max_abs_percent: float,
) -> tuple[int, int]:
    left_mps = linear_mps + angular_rad_s * wheel_base_m * 0.5
    right_mps = linear_mps - angular_rad_s * wheel_base_m * 0.5
    if max_wheel_speed_mps <= 0:
        raise ValueError("max_wheel_speed_mps must be positive")
    left = 100.0 * left_mps / max_wheel_speed_mps
    right = 100.0 * right_mps / max_wheel_speed_mps
    max_abs_percent = float(max_abs_percent)
    if max_abs_percent <= 0:
        return 0, 0
    max_command = max(abs(left), abs(right))
    if max_command > max_abs_percent:
        scale = max_abs_percent / max_command
        left *= scale
        right *= scale
    return int(round(left)), int(round(right))


class NoMaDPolicy:
    def __init__(
        self,
        *,
        checkpoint: Path | str,
        visualnav_repo: Path | str,
        diffusion_policy_repo: Path | str | None = None,
        config: dict[str, Any] | None = None,
        device: str = "auto",
        num_samples: int = 8,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.visualnav_repo = Path(visualnav_repo).expanduser().resolve()
        self.diffusion_policy_repo = (
            None if diffusion_policy_repo is None else Path(diffusion_policy_repo).expanduser().resolve()
        )
        self.config = dict(DEFAULT_NOMAD_CONFIG if config is None else config)
        self.num_samples = int(num_samples)
        self.device = resolve_device(device)
        self._load_upstream_symbols()
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.noise_scheduler = self.DDPMScheduler(
            num_train_timesteps=int(self.config["num_diffusion_iters"]),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.context_size = int(self.config["context_size"])
        self.image_size = tuple(int(value) for value in self.config["image_size"])
        self.len_traj_pred = int(self.config["len_traj_pred"])

    def _load_upstream_symbols(self) -> None:
        train_dir = self.visualnav_repo / "train"
        if not train_dir.exists():
            raise NoMaDUnavailable(
                f"{self.visualnav_repo} does not look like robodhruv/visualnav-transformer; "
                "expected train/."
            )
        if str(train_dir) not in sys.path:
            sys.path.insert(0, str(train_dir))
        if self.diffusion_policy_repo is not None:
            if not self.diffusion_policy_repo.exists():
                raise NoMaDUnavailable(f"diffusion_policy repo path does not exist: {self.diffusion_policy_repo}")
            path = str(self.diffusion_policy_repo)
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
            from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
            from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
        except Exception as exc:  # pragma: no cover - depends on optional model env
            raise NoMaDUnavailable(
                "NoMaD dependencies are not installed. Install the upstream visualnav-transformer "
                "training package, diffusers, efficientnet_pytorch, einops, and "
                f"real-stanford/diffusion_policy before running live NoMaD inference. Import error: {exc}"
            ) from exc
        self.DDPMScheduler = DDPMScheduler
        self.ConditionalUnet1D = ConditionalUnet1D
        self.DenseNetwork = DenseNetwork
        self.NoMaD = NoMaD
        self.NoMaD_ViNT = NoMaD_ViNT
        self.replace_bn_with_gn = replace_bn_with_gn

    def _load_model(self):
        import torch

        if str(self.config.get("model_type")) != "nomad":
            raise NoMaDUnavailable("NoMaDPolicy only supports model_type='nomad'")
        model = self._build_nomad_model()
        checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device)

    def _build_nomad_model(self):
        if self.config["vision_encoder"] == "nomad_vint":
            vision_encoder = self.NoMaD_ViNT(
                obs_encoding_size=self.config["encoding_size"],
                context_size=self.config["context_size"],
                mha_num_attention_heads=self.config["mha_num_attention_heads"],
                mha_num_attention_layers=self.config["mha_num_attention_layers"],
                mha_ff_dim_factor=self.config["mha_ff_dim_factor"],
            )
            vision_encoder = self.replace_bn_with_gn(vision_encoder)
        elif self.config["vision_encoder"] == "vit":
            try:
                from vint_train.models.vint.vit import ViT
            except Exception as exc:  # pragma: no cover - alternate upstream encoder
                raise NoMaDUnavailable("vision_encoder='vit' requires the upstream ViT dependencies") from exc
            vision_encoder = ViT(
                obs_encoding_size=self.config["encoding_size"],
                context_size=self.config["context_size"],
                image_size=self.config["image_size"],
                patch_size=self.config["patch_size"],
                mha_num_attention_heads=self.config["mha_num_attention_heads"],
                mha_num_attention_layers=self.config["mha_num_attention_layers"],
            )
            vision_encoder = self.replace_bn_with_gn(vision_encoder)
        else:
            raise NoMaDUnavailable(f"unsupported NoMaD vision encoder: {self.config['vision_encoder']}")

        noise_pred_net = self.ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=self.config["encoding_size"],
            down_dims=self.config["down_dims"],
            cond_predict_scale=self.config["cond_predict_scale"],
        )
        dist_pred_network = self.DenseNetwork(embedding_dim=self.config["encoding_size"])
        return self.NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    @staticmethod
    def transform_images(pil_imgs: Image.Image | Sequence[Image.Image], image_size: Sequence[int], center_crop: bool = False):
        import torch

        if not isinstance(pil_imgs, list):
            pil_imgs = [pil_imgs]
        tensors = []
        for image in pil_imgs:
            image = image.convert("RGB")
            if center_crop:
                image = _center_crop_aspect(image, 4.0 / 3.0)
            image = image.resize(tuple(int(value) for value in image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
                [0.229, 0.224, 0.225],
                dtype=np.float32,
            )
            tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            tensors.append(tensor)
        return torch.cat(tensors, dim=1)

    @staticmethod
    def get_action(diffusion_output):
        import torch

        ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2).detach().cpu().numpy()
        ndeltas = (ndeltas + 1.0) / 2.0
        ndeltas = ndeltas * (ACTION_STATS["max"] - ACTION_STATS["min"]) + ACTION_STATS["min"]
        actions = np.cumsum(ndeltas, axis=1).astype(np.float32)
        return torch.from_numpy(actions).to(diffusion_output.device)

    @staticmethod
    def select_sampled_waypoint(sampled: np.ndarray, waypoint_index: int, aggregation: str = "first") -> np.ndarray:
        if sampled.ndim != 3:
            raise ValueError(f"expected sampled waypoints with shape (samples, horizon, 2), got {sampled.shape}")
        waypoint_index = min(max(0, int(waypoint_index)), sampled.shape[1] - 1)
        candidates = sampled[:, waypoint_index, :]
        aggregation = aggregation.strip().lower()
        if aggregation == "first":
            return candidates[0].astype(np.float32)
        if aggregation == "mean":
            return np.mean(candidates, axis=0).astype(np.float32)
        if aggregation == "median":
            return np.median(candidates, axis=0).astype(np.float32)
        if aggregation == "medoid":
            center = np.median(candidates, axis=0)
            index = int(np.argmin(np.linalg.norm(candidates - center[None, :], axis=1)))
            return candidates[index].astype(np.float32)
        raise ValueError(f"unsupported NoMaD sample aggregation: {aggregation}")

    def _context_tensor(self, context_images: Sequence[Image.Image]):
        if len(context_images) < self.context_size + 1:
            raise ValueError(f"need at least {self.context_size + 1} context images")
        images = list(context_images)[-(self.context_size + 1) :]
        obs_images = self.transform_images(images, list(self.image_size), center_crop=False)
        obs_images = self.torch_cat_context(obs_images)
        return obs_images.to(self.device)

    def torch_cat_context(self, obs_images):
        import torch

        chunks = torch.split(obs_images, 3, dim=1)
        return torch.cat(chunks, dim=1)

    def predict_distances(self, context_images: Sequence[Image.Image], goal_images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        if not goal_images:
            return np.empty((0,), dtype=np.float32)
        obs_images = self._context_tensor(context_images)
        goal_tensor = torch.cat(
            [self.transform_images(goal, list(self.image_size), center_crop=False).to(self.device) for goal in goal_images],
            dim=0,
        )
        mask = torch.zeros(len(goal_images), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            obsgoal_cond = self.model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_images), 1, 1, 1),
                goal_img=goal_tensor,
                input_goal_mask=mask,
            )
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        return self.to_numpy(dists.flatten()).astype(np.float32)

    def predict_waypoints(self, context_images: Sequence[Image.Image], goal_image: Image.Image) -> np.ndarray:
        import torch

        obs_images = self._context_tensor(context_images)
        goal_tensor = self.transform_images(goal_image, list(self.image_size), center_crop=False).to(self.device)
        mask = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            obs_cond = self.model(
                "vision_encoder",
                obs_img=obs_images,
                goal_img=goal_tensor,
                input_goal_mask=mask,
            )
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.num_samples, 1, 1)
            naction = torch.randn((self.num_samples, self.len_traj_pred, 2), device=self.device)
            self.noise_scheduler.set_timesteps(int(self.config["num_diffusion_iters"]))
            for timestep in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=timestep,
                    global_cond=obs_cond,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=naction,
                ).prev_sample
        return self.to_numpy(self.get_action(naction)).astype(np.float32)

    def command_for_goal_image(
        self,
        context_images: Sequence[Image.Image],
        goal_image: Image.Image,
        *,
        waypoint_index: int = 2,
        waypoint_dt_s: float = 1.0 / 4.0,
        max_v_mps: float = 0.2,
        max_w_rad_s: float = 0.4,
        wheel_base_m: float = 0.215,
        max_wheel_speed_mps: float = 0.78,
        max_abs_percent: float = 35.0,
        sample_aggregation: str = "first",
        invert_angular: bool = False,
    ) -> WaypointCommand:
        started = time.perf_counter()
        sampled = self.predict_waypoints(context_images, goal_image)
        raw_waypoint = self.select_sampled_waypoint(sampled, waypoint_index, sample_aggregation)
        waypoint = raw_waypoint.copy()
        if bool(self.config.get("normalize", True)):
            waypoint = waypoint.copy()
            waypoint[:2] *= max_v_mps * waypoint_dt_s
        linear, angular = waypoint_to_twist(
            waypoint,
            waypoint_dt_s=waypoint_dt_s,
            max_v_mps=max_v_mps,
            max_w_rad_s=max_w_rad_s,
        )
        if invert_angular:
            angular = -angular
        motors = twist_to_motor_percent(
            linear,
            angular,
            wheel_base_m=wheel_base_m,
            max_wheel_speed_mps=max_wheel_speed_mps,
            max_abs_percent=max_abs_percent,
        )
        return WaypointCommand(
            waypoint=waypoint,
            motor_percent=motors,
            linear_mps=linear,
            angular_rad_s=angular,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            raw_waypoint=raw_waypoint,
            sampled_waypoints=sampled[: min(len(sampled), 4)].astype(np.float32, copy=False),
        )


def _center_crop_aspect(image: Image.Image, aspect_ratio: float) -> Image.Image:
    width, height = image.size
    if width / max(height, 1) > aspect_ratio:
        crop_width = int(round(height * aspect_ratio))
        left = max(0, (width - crop_width) // 2)
        return image.crop((left, 0, left + crop_width, height))
    crop_height = int(round(width / aspect_ratio))
    top = max(0, (height - crop_height) // 2)
    return image.crop((0, top, width, top + crop_height))
