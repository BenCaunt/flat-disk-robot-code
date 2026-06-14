from __future__ import annotations

import argparse

from flatdisk_sim.nomad_topomap_doctor import run_checks


def test_nomad_topomap_doctor_reports_required_sections(tmp_path) -> None:
    repo = tmp_path / "visualnav-transformer"
    (repo / "train" / "vint_train").mkdir(parents=True)

    result = run_checks(
        argparse.Namespace(
            checkpoint=tmp_path / "missing_nomad.pth",
            visualnav_repo=repo,
            diffusion_policy_repo=None,
        )
    )

    assert result["schema"] == "flatdisk.nomad_topomap.doctor.v1"
    assert "ai2thor" in result["modules"]
    assert "diffusion_policy" in result["modules"]
    assert "einops" in result["modules"]
    assert result["checkpoint"]["ok"] is False
    assert result["visualnav_repo"]["ok"] is True
    assert result["nomad_ready"] is False
