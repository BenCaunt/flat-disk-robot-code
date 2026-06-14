from __future__ import annotations

from flatdisk_sim.compete_text_goals import _policy_score, build_competition_suite, competition_suites
from flatdisk_sim.policy_registry import policy_registry


def test_competition_suites_have_dev_and_heldout() -> None:
    suites = competition_suites()

    assert set(suites) == {"dev", "heldout"}
    assert len(suites["heldout"].episodes) >= 2
    assert suites["heldout"].episodes[0].scene != suites["dev"].episodes[0].scene


def test_random_competition_suite_is_seeded_and_excludes_dev_heldout() -> None:
    suite = build_competition_suite("random", random_seed=123)
    scenes = [episode.scene for episode in suite.episodes]

    assert suite.name == "random_seed_123"
    assert scenes == [episode.scene for episode in build_competition_suite("random", random_seed=123).episodes]
    assert set(scenes).isdisjoint({"FloorPlan201", "FloorPlan202", "FloorPlan301", "FloorPlan302", "FloorPlan402", "FloorPlan403"})


def test_policy_score_ranks_success_and_wall_clock_inputs() -> None:
    score = _policy_score(
        "example",
        [
            {"success": True, "final_distance_m": 0.4},
            {"success": False, "final_distance_m": 1.2},
        ],
        12.345,
    )

    assert score == {
        "policy": "example",
        "success_count": 1,
        "episode_count": 2,
        "wall_clock_s": 12.345,
        "mean_final_distance_m": 0.8,
    }


def test_control_policy_registered() -> None:
    assert {"control_vlm", "memory_vlm", "sprinter", "hf_scout"} <= set(policy_registry())
