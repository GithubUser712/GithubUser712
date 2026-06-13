"""Tests for Gazebo world export."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from raceline.gazebo import build_world_sdf, export_gazebo_world


@pytest.fixture
def mini_artifacts(tmp_path):
    occ = np.zeros((20, 20), dtype=np.uint8)
    occ[0, :] = 1
    occ[-1, :] = 1
    occ[:, 0] = 1
    occ[:, -1] = 1
    import cv2
    cv2.imwrite(str(tmp_path / "map.pgm"), np.where(occ == 1, 0, 254).astype(np.uint8))
    with open(tmp_path / "map.yaml", "w") as f:
        yaml.dump({"image": "map.pgm", "resolution": 0.1}, f)
    with open(tmp_path / "centerline.csv", "w") as f:
        f.write("x_m,y_m,clearance_m\n0,0,1\n")
    with open(tmp_path / "track_meta.yaml", "w") as f:
        yaml.dump({"track_length_m": 1.0}, f)
    return tmp_path


def test_build_world_sdf_contains_walls(mini_artifacts):
    sdf = build_world_sdf(mini_artifacts, wall_stride=2)
    assert "wall_" in sdf
    assert "ground_plane" in sdf


def test_export_writes_file(mini_artifacts, tmp_path):
    out = export_gazebo_world(mini_artifacts, tmp_path / "world.sdf")
    assert out.is_file()
    assert out.read_text().startswith("<sdf")
