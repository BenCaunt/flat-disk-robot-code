from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from flatdisk_sim.nomad_policy import NoMaDPolicy, twist_to_motor_percent


torch = pytest.importorskip("torch")


def test_nomad_transform_images_matches_expected_context_shape() -> None:
    images = [
        Image.new("RGB", (120, 90), (255, 0, 0)),
        Image.new("RGB", (120, 90), (0, 255, 0)),
        Image.new("RGB", (120, 90), (0, 0, 255)),
    ]

    tensor = NoMaDPolicy.transform_images(images, (96, 96))

    assert tensor.shape == (1, 9, 96, 96)
    expected_red = (1.0 - 0.485) / 0.229
    assert float(tensor[0, 0, 0, 0]) == pytest.approx(expected_red)


def test_nomad_get_action_unnormalizes_and_integrates_deltas() -> None:
    normalized = torch.zeros((1, 2, 2), dtype=torch.float32)

    actions = NoMaDPolicy.get_action(normalized).detach().cpu().numpy()

    np.testing.assert_allclose(
        actions,
        np.asarray([[[1.25, 0.0], [2.5, 0.0]]], dtype=np.float32),
    )


def test_twist_to_motor_percent_preserves_turn_ratio_when_limited() -> None:
    motors = twist_to_motor_percent(
        linear_mps=0.2,
        angular_rad_s=0.4,
        wheel_base_m=0.215,
        max_wheel_speed_mps=0.78,
        max_abs_percent=8.0,
    )

    assert motors[0] == 8
    assert 0 < motors[1] < motors[0]


def test_nomad_select_sampled_waypoint_supports_robust_aggregation() -> None:
    sampled = np.asarray(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[1.5, 11.0], [3.0, 21.0]],
            [[9.0, 99.0], [4.0, 80.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(NoMaDPolicy.select_sampled_waypoint(sampled, 1, "first"), [2.0, 20.0])
    np.testing.assert_allclose(NoMaDPolicy.select_sampled_waypoint(sampled, 1, "median"), [3.0, 21.0])
    np.testing.assert_allclose(NoMaDPolicy.select_sampled_waypoint(sampled, 1, "mean"), [3.0, 40.333332], rtol=1e-6)
    np.testing.assert_allclose(NoMaDPolicy.select_sampled_waypoint(sampled, 1, "medoid"), [3.0, 21.0])
