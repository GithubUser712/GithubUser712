"""Tests for cross-track policy memory."""

from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

from raceline.rl.policy_memory import (
    find_transfer_policy,
    policy_memory_dir,
    register_policy,
    scan_entries,
    track_label_from_artifacts,
)


def _fake_policy(path: Path) -> None:
    """Minimal zip so copy/register has a real file."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data", "test")


def test_register_and_find_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "raceline.rl.policy_memory.find_project_root", lambda: tmp_path)
    mem = policy_memory_dir(tmp_path)
    policy_a = tmp_path / "a.zip"
    _fake_policy(policy_a)
    register_policy(
        policy_a,
        track_label="monaco",
        artifacts_dir=tmp_path / "artifacts_monaco",
        source_step=2,
        policy_kind="base",
        observation_shape=(31,),
        track_length_m=333.0,
        timesteps=500_000,
        root=tmp_path,
    )
    policy_b = tmp_path / "b.zip"
    _fake_policy(policy_b)
    register_policy(
        policy_b,
        track_label="spa",
        artifacts_dir=tmp_path / "artifacts_spa",
        source_step=3,
        policy_kind="tuned",
        observation_shape=(31,),
        track_length_m=699.0,
        timesteps=600_000,
        root=tmp_path,
    )
    assert len(scan_entries(tmp_path)) == 2
    found = find_transfer_policy(
        (31,), exclude_artifacts=tmp_path / "artifacts_new", root=tmp_path)
    assert found is not None
    path, meta = found
    assert meta["policy_kind"] == "tuned"
    assert meta["track_label"] == "spa"
    assert path.is_file()


def test_track_label_from_meta(tmp_path):
    art = tmp_path / "artifacts_monaco"
    art.mkdir()
    with open(art / "track_meta.yaml", "w") as f:
        yaml.safe_dump({"track_label": "monaco", "source_image": "maps/monaco.png"}, f)
    assert track_label_from_artifacts(art) == "monaco"
