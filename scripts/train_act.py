#!/usr/bin/env python3
"""Train a small ACT-style policy from flat-disk Rerun recordings.

The dataset is expected to contain camera frames at `/camera/image` and motor
percent commands at `/commands/motor1_percent` and `/commands/motor2_percent`.
Each training item uses image history plus past motor commands and predicts a
chunk of future motor commands. Commands can be represented directly as
left/right motor percentages or as mixed forward/steer controls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from rerun.recording import load_archive
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_ENTITY = "/camera/image"
LEFT_COMMAND_ENTITY = "/commands/motor1_percent"
RIGHT_COMMAND_ENTITY = "/commands/motor2_percent"
SCALAR_COLUMN = "Scalars:scalars"
IMAGE_COLUMN = "EncodedImage:blob"
LEFT_RIGHT_ACTIONS = "left_right"
FORWARD_STEER_ACTIONS = "forward_steer"


def duration_to_ns(value: object) -> int | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return int(value.value)
    if isinstance(value, np.timedelta64):
        return int(value / np.timedelta64(1, "ns"))
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[ns]").astype(np.int64))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def first_scalar(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return first_scalar(value[0])
    return float(value)


def first_blob(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, list):
        if not value:
            return None
        inner = value[0]
        if isinstance(inner, bytes):
            return inner
        return bytes(inner)
    return None


def row_time_ns(row: dict[str, list[object]], index: int, columns: tuple[str, ...] | None = None) -> int | None:
    if columns is None:
        columns = ("synced_monotonic", "device_time", "log_time", "host_monotonic")
    for column in columns:
        values = row.get(column)
        if values is None:
            continue
        t_ns = duration_to_ns(values[index])
        if t_ns is not None:
            return t_ns
    return None


def canonical_action_representation(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"left_right", "lr", "motors", "motor"}:
        return LEFT_RIGHT_ACTIONS
    if normalized in {"forward_steer", "forward_turn", "mixed", "mix", "stick"}:
        return FORWARD_STEER_ACTIONS
    raise argparse.ArgumentTypeError(
        f"Unknown action representation {value!r}; use left_right or forward_steer"
    )


def encode_actions(motor_actions: np.ndarray, action_representation: str) -> np.ndarray:
    """Convert left/right motor percentages into the model action space."""
    action_representation = canonical_action_representation(action_representation)
    actions = np.asarray(motor_actions, dtype=np.float32)
    if action_representation == LEFT_RIGHT_ACTIONS:
        return actions.copy()

    forward = (actions[..., 0] + actions[..., 1]) * 0.5
    steer = (actions[..., 0] - actions[..., 1]) * 0.5
    return np.stack([forward, steer], axis=-1).astype(np.float32)


def decode_actions(model_actions: np.ndarray, action_representation: str) -> np.ndarray:
    """Convert model action-space commands back to left/right motor percentages."""
    action_representation = canonical_action_representation(action_representation)
    actions = np.asarray(model_actions, dtype=np.float32)
    if action_representation == LEFT_RIGHT_ACTIONS:
        return actions.copy()

    left = actions[..., 0] + actions[..., 1]
    right = actions[..., 0] - actions[..., 1]
    return np.stack([left, right], axis=-1).astype(np.float32)


def normalize_model_actions(
    model_actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    return ((np.asarray(model_actions, dtype=np.float32) - action_mean) / action_std).astype(np.float32)


def denormalize_model_actions(
    normalized_actions: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> np.ndarray:
    return (np.asarray(normalized_actions, dtype=np.float32) * action_std + action_mean).astype(np.float32)


def action_stats(
    episodes: list["Episode"],
    action_representation: str,
    action_scale: float,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    values = [
        encode_actions(episode.frame_actions, action_representation) / action_scale
        for episode in episodes
        if len(episode.frame_actions) > 0
    ]
    if not values:
        return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    stacked = np.concatenate(values, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.maximum(std, np.asarray([1e-3, 1e-3], dtype=np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class Episode:
    path: str
    frame_times_ns: np.ndarray
    frame_jpegs: list[bytes]
    frame_actions: np.ndarray


@dataclass
class ActSampleRef:
    episode_index: int
    frame_index: int


def extract_entity_rows(rrd_path: Path, entity_path: str) -> Iterable[dict[str, list[object]]]:
    archive = load_archive(rrd_path)
    recordings = archive.all_recordings()
    if len(recordings) != 1:
        raise ValueError(f"{rrd_path} contains {len(recordings)} recordings; expected 1")
    for chunk in recordings[0].chunks():
        if chunk.entity_path == entity_path:
            yield chunk.to_record_batch().to_pydict()


def read_scalar_series(
    rrd_path: Path,
    entity_path: str,
    time_columns: tuple[str, ...] = ("host_monotonic", "synced_monotonic", "device_time", "log_time"),
) -> tuple[np.ndarray, np.ndarray]:
    times: list[int] = []
    values: list[float] = []
    for row in extract_entity_rows(rrd_path, entity_path):
        if SCALAR_COLUMN not in row:
            continue
        for i, raw_value in enumerate(row[SCALAR_COLUMN]):
            t_ns = row_time_ns(row, i, time_columns)
            scalar = first_scalar(raw_value)
            if t_ns is not None and scalar is not None:
                times.append(t_ns)
                values.append(scalar)
    order = np.argsort(times)
    return np.asarray(times, dtype=np.int64)[order], np.asarray(values, dtype=np.float32)[order]


def read_image_series(rrd_path: Path) -> tuple[np.ndarray, list[bytes]]:
    rows: list[tuple[int, bytes]] = []
    for row in extract_entity_rows(rrd_path, IMAGE_ENTITY):
        if IMAGE_COLUMN not in row:
            continue
        for i, raw_blob in enumerate(row[IMAGE_COLUMN]):
            t_ns = row_time_ns(row, i, ("synced_monotonic", "device_time", "log_time", "host_monotonic"))
            blob = first_blob(raw_blob)
            if t_ns is not None and blob is not None:
                rows.append((t_ns, blob))
    rows.sort(key=lambda item: item[0])
    return np.asarray([r[0] for r in rows], dtype=np.int64), [r[1] for r in rows]


def build_episode(
    rrd_path: Path,
    max_command_age_s: float,
    min_sample_interval_s: float = 0.0,
    action_time_offset_s: float = 0.0,
) -> Episode | None:
    frame_times, frame_jpegs = read_image_series(rrd_path)
    left_times, left = read_scalar_series(rrd_path, LEFT_COMMAND_ENTITY)
    right_times, right = read_scalar_series(rrd_path, RIGHT_COMMAND_ENTITY)
    if len(frame_times) == 0 or len(left_times) == 0 or len(right_times) == 0:
        return None

    actions = np.full((len(frame_times), 2), np.nan, dtype=np.float32)
    max_age_ns = int(max_command_age_s * 1_000_000_000)
    action_time_offset_ns = int(action_time_offset_s * 1_000_000_000)
    for i, t_ns in enumerate(frame_times):
        action_t_ns = int(t_ns) + action_time_offset_ns
        li = np.searchsorted(left_times, action_t_ns, side="right") - 1
        ri = np.searchsorted(right_times, action_t_ns, side="right") - 1
        if li < 0 or ri < 0:
            continue
        if action_t_ns - left_times[li] > max_age_ns or action_t_ns - right_times[ri] > max_age_ns:
            continue
        actions[i] = (left[li], right[ri])

    valid = np.isfinite(actions).all(axis=1)
    if not valid.any():
        return None
    frame_times = frame_times[valid]
    actions = actions[valid]
    frame_jpegs = [jpeg for jpeg, keep in zip(frame_jpegs, valid, strict=True) if keep]
    if min_sample_interval_s > 0 and len(frame_times) > 1:
        min_interval_ns = int(min_sample_interval_s * 1_000_000_000)
        keep_indices = [0]
        last_kept_ns = int(frame_times[0])
        for i in range(1, len(frame_times)):
            if int(frame_times[i]) - last_kept_ns >= min_interval_ns:
                keep_indices.append(i)
                last_kept_ns = int(frame_times[i])
        frame_times = frame_times[keep_indices]
        actions = actions[keep_indices]
        frame_jpegs = [frame_jpegs[i] for i in keep_indices]
    return Episode(str(rrd_path), frame_times, frame_jpegs, actions)


class RerunActDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        episodes: list[Episode],
        *,
        image_size: int,
        image_history: int,
        past_action_steps: int,
        chunk_len: int,
        action_scale: float,
        action_representation: str,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        past_action_noise_std_percent: float,
        past_action_dropout_prob: float,
        max_samples_per_episode: int,
    ) -> None:
        self.episodes = episodes
        self.image_size = image_size
        self.image_history = image_history
        self.past_action_steps = past_action_steps
        self.chunk_len = chunk_len
        self.action_scale = action_scale
        self.action_representation = canonical_action_representation(action_representation)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.past_action_noise_std_percent = past_action_noise_std_percent
        self.past_action_dropout_prob = past_action_dropout_prob
        self.samples: list[ActSampleRef] = []

        for episode_index, episode in enumerate(episodes):
            usable = len(episode.frame_jpegs) - chunk_len + 1
            if usable <= 0:
                continue
            indices = list(range(usable))
            if max_samples_per_episode > 0 and len(indices) > max_samples_per_episode:
                stride = len(indices) / max_samples_per_episode
                indices = [int(i * stride) for i in range(max_samples_per_episode)]
            self.samples.extend(ActSampleRef(episode_index, i) for i in indices)

    def __len__(self) -> int:
        return len(self.samples)

    def decode_image(self, jpeg: bytes) -> torch.Tensor:
        import io

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self.samples[index]
        episode = self.episodes[ref.episode_index]
        i = ref.frame_index

        image_indices = [max(0, i - offset) for offset in reversed(range(self.image_history))]
        images = torch.stack([self.decode_image(episode.frame_jpegs[j]) for j in image_indices])

        action_indices = [max(0, i - offset - 1) for offset in reversed(range(self.past_action_steps))]
        past_motor_actions = episode.frame_actions[action_indices].copy()
        if self.past_action_noise_std_percent > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.past_action_noise_std_percent,
                size=past_motor_actions.shape,
            ).astype(np.float32)
            past_motor_actions = past_motor_actions + noise
        past_model_actions = encode_actions(past_motor_actions, self.action_representation)
        past_model_actions = normalize_model_actions(
            past_model_actions / self.action_scale,
            self.action_mean,
            self.action_std,
        )
        if self.past_action_dropout_prob > 0:
            keep = np.random.random((past_model_actions.shape[0], 1)) >= self.past_action_dropout_prob
            past_model_actions = past_model_actions * keep.astype(np.float32)
        past_actions = torch.from_numpy(past_model_actions).float()

        chunk_model_actions = encode_actions(episode.frame_actions[i : i + self.chunk_len], self.action_representation)
        chunk_model_actions = normalize_model_actions(
            chunk_model_actions / self.action_scale,
            self.action_mean,
            self.action_std,
        )
        chunk = torch.from_numpy(chunk_model_actions).float()
        return images, past_actions, chunk


class ActPolicy(nn.Module):
    def __init__(
        self,
        *,
        image_history: int,
        past_action_steps: int,
        chunk_len: int,
        action_dim: int = 2,
        width: int = 192,
        heads: int = 4,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.image_history = image_history
        self.past_action_steps = past_action_steps
        self.chunk_len = chunk_len

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.image_proj = nn.Linear(width, width)
        self.action_proj = nn.Linear(action_dim, width)
        self.pos = nn.Parameter(torch.randn(image_history + past_action_steps, width) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.chunk_queries = nn.Parameter(torch.randn(chunk_len, width) * 0.02)
        self.action_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

    def forward(self, images: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
        batch, history, channels, height, width = images.shape
        flat_images = images.reshape(batch * history, channels, height, width)
        image_features = self.cnn(flat_images).flatten(1).reshape(batch, history, -1)
        image_tokens = self.image_proj(image_features)
        action_tokens = self.action_proj(past_actions)
        tokens = torch.cat([image_tokens, action_tokens], dim=1) + self.pos.unsqueeze(0)
        memory = self.encoder(tokens)
        queries = self.chunk_queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)


class TemporalEnsembler:
    """Inference helper for ACT overlapping action chunks."""

    def __init__(self, chunk_len: int, action_dim: int, decay: float) -> None:
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.decay = decay
        self.pending: list[tuple[int, np.ndarray]] = []
        self.step_index = 0

    def add_prediction(self, chunk: np.ndarray) -> np.ndarray:
        self.pending.append((self.step_index, chunk.copy()))
        weighted = np.zeros(self.action_dim, dtype=np.float32)
        weight_sum = 0.0
        keep: list[tuple[int, np.ndarray]] = []
        for start_step, pred in self.pending:
            offset = self.step_index - start_step
            if 0 <= offset < self.chunk_len:
                weight = math.exp(-self.decay * offset)
                weighted += weight * pred[offset]
                weight_sum += weight
                keep.append((start_step, pred))
        self.pending = keep
        self.step_index += 1
        return weighted / max(weight_sum, 1e-8)


@dataclass
class TrainConfig:
    data: str
    output_dir: str
    image_size: int
    image_history: int
    past_action_steps: int
    chunk_len: int
    action_scale: float
    action_representation: str
    normalize_actions: bool
    action_mean: list[float]
    action_std: list[float]
    temporal_ensembling: bool
    temporal_ensemble_decay: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    val_fraction: float
    max_files: int
    max_samples_per_episode: int
    max_command_age_s: float
    min_sample_interval_s: float
    action_time_offset_s: float
    past_action_noise_std_percent: float
    past_action_dropout_prob: float


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_episodes(args: argparse.Namespace) -> list[Episode]:
    data_dir = Path(args.data).expanduser()
    paths = sorted(data_dir.glob("*.rrd"))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No .rrd files found in {data_dir}")

    episodes: list[Episode] = []
    for i, path in enumerate(paths, start=1):
        episode = build_episode(
            path,
            args.max_command_age_s,
            args.min_sample_interval_s,
            args.action_time_offset_s,
        )
        if episode is not None and len(episode.frame_jpegs) >= args.chunk_len:
            episodes.append(episode)
            print(f"[{i:03d}/{len(paths):03d}] {path.name}: {len(episode.frame_jpegs)} aligned frames", flush=True)
        else:
            print(f"[{i:03d}/{len(paths):03d}] {path.name}: skipped", flush=True)
    if not episodes:
        raise RuntimeError("No usable episodes found after aligning camera frames and motor commands")
    return episodes


def split_episodes(episodes: list[Episode], val_fraction: float, seed: int) -> tuple[list[Episode], list[Episode]]:
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def run_epoch(
    model: ActPolicy,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    action_std: torch.Tensor,
    action_scale: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_items = 0
    total_abs_percent = 0.0
    total_sq_percent = 0.0
    with torch.set_grad_enabled(train):
        for images, past_actions, target in loader:
            images = images.to(device)
            past_actions = past_actions.to(device)
            target = target.to(device)
            pred = model(images, past_actions)
            loss = torch.nn.functional.mse_loss(pred, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            count = target.numel()
            metric_std = action_std.to(device=device, dtype=pred.dtype).view(1, 1, -1)
            diff_percent = (pred.detach() - target.detach()) * metric_std * action_scale
            total_loss += float(loss.detach().cpu()) * count
            total_items += count
            total_abs_percent += float(diff_percent.abs().sum().cpu())
            total_sq_percent += float((diff_percent * diff_percent).sum().cpu())
    return {
        "loss": total_loss / max(total_items, 1),
        "mae_percent": total_abs_percent / max(total_items, 1),
        "rmse_percent": math.sqrt(total_sq_percent / max(total_items, 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/Users/bencaunt/Documents/flat-disk-robot-code/captures/vla-act-ctrlr")
    parser.add_argument("--output-dir", default="runs/act_vla")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--image-history", type=int, default=1)
    parser.add_argument("--past-action-steps", type=int, default=8)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--action-scale", type=float, default=100.0)
    parser.add_argument("--action-representation", type=canonical_action_representation, default=LEFT_RIGHT_ACTIONS,
                        help="Action space to train in: left_right or forward_steer.")
    parser.add_argument("--normalize-actions", action="store_true",
                        help="Center and scale each model action dimension using train-split statistics.")
    parser.add_argument("--temporal-ensembling", action="store_true")
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-samples-per-episode", type=int, default=0)
    parser.add_argument("--max-command-age-s", type=float, default=1.5)
    parser.add_argument("--min-sample-interval-s", type=float, default=0.0,
                        help="Drop aligned frames closer together than this interval.")
    parser.add_argument("--action-time-offset-s", type=float, default=0.0,
                        help="Pair each image at time t with commands at t + offset.")
    parser.add_argument("--past-action-noise-std-percent", type=float, default=0.0,
                        help="Train-only Gaussian noise added to past motor-percent inputs before encoding.")
    parser.add_argument("--past-action-dropout-prob", type=float, default=0.0,
                        help="Train-only probability of replacing a past-action token with zero/mean.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args)
    train_episodes, val_episodes = split_episodes(episodes, args.val_fraction, args.seed)
    action_mean, action_std = action_stats(
        train_episodes,
        args.action_representation,
        args.action_scale,
        args.normalize_actions,
    )
    train_ds = RerunActDataset(
        train_episodes,
        image_size=args.image_size,
        image_history=args.image_history,
        past_action_steps=args.past_action_steps,
        chunk_len=args.chunk_len,
        action_scale=args.action_scale,
        action_representation=args.action_representation,
        action_mean=action_mean,
        action_std=action_std,
        past_action_noise_std_percent=args.past_action_noise_std_percent,
        past_action_dropout_prob=args.past_action_dropout_prob,
        max_samples_per_episode=args.max_samples_per_episode,
    )
    val_ds = RerunActDataset(
        val_episodes,
        image_size=args.image_size,
        image_history=args.image_history,
        past_action_steps=args.past_action_steps,
        chunk_len=args.chunk_len,
        action_scale=args.action_scale,
        action_representation=args.action_representation,
        action_mean=action_mean,
        action_std=action_std,
        past_action_noise_std_percent=0.0,
        past_action_dropout_prob=0.0,
        max_samples_per_episode=args.max_samples_per_episode,
    )
    if len(train_ds) == 0:
        raise RuntimeError("Training split produced no samples")
    if len(val_ds) == 0:
        val_ds = train_ds

    device = choose_device(args.device)
    model = ActPolicy(
        image_history=args.image_history,
        past_action_steps=args.past_action_steps,
        chunk_len=args.chunk_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = TrainConfig(
        data=args.data,
        output_dir=args.output_dir,
        image_size=args.image_size,
        image_history=args.image_history,
        past_action_steps=args.past_action_steps,
        chunk_len=args.chunk_len,
        action_scale=args.action_scale,
        action_representation=args.action_representation,
        normalize_actions=args.normalize_actions,
        action_mean=action_mean.tolist(),
        action_std=action_std.tolist(),
        temporal_ensembling=args.temporal_ensembling,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_files=args.max_files,
        max_samples_per_episode=args.max_samples_per_episode,
        max_command_age_s=args.max_command_age_s,
        min_sample_interval_s=args.min_sample_interval_s,
        action_time_offset_s=args.action_time_offset_s,
        past_action_noise_std_percent=args.past_action_noise_std_percent,
        past_action_dropout_prob=args.past_action_dropout_prob,
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    print(
        f"episodes={len(episodes)} train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"device={device} action_representation={args.action_representation} "
        f"normalize_actions={args.normalize_actions} action_mean={action_mean.tolist()} "
        f"action_std={action_std.tolist()} past_action_noise_std_percent={args.past_action_noise_std_percent} "
        f"past_action_dropout_prob={args.past_action_dropout_prob} "
        f"action_time_offset_s={args.action_time_offset_s} "
        f"temporal_ensembling={args.temporal_ensembling}",
        flush=True,
    )

    best_val = float("inf")
    history: list[dict[str, float]] = []
    action_std_tensor = torch.from_numpy(action_std)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            action_std=action_std_tensor,
            action_scale=args.action_scale,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            action_std=action_std_tensor,
            action_scale=args.action_scale,
        )
        row = {
            "epoch": epoch,
            "seconds": time.time() - start,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.6f} train_mae={row['train_mae_percent']:.2f}% "
            f"val_loss={row['val_loss']:.6f} val_mae={row['val_mae_percent']:.2f}% "
            f"val_rmse={row['val_rmse_percent']:.2f}% seconds={row['seconds']:.1f}",
            flush=True,
        )
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                output_dir / "best.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "epoch": args.epochs,
            "val_loss": history[-1]["val_loss"],
        },
        output_dir / "last.pt",
    )
    print(f"saved best checkpoint: {output_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
