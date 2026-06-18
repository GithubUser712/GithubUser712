"""Export pipeline artifacts to Gazebo Harmonic (gz sim) SDF worlds."""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

from raceline.core.exceptions import ArtifactError


def _read_map(artifacts_dir: Path) -> tuple[np.ndarray, float, tuple[int, int]]:
    with open(artifacts_dir / "map.yaml") as f:
        cfg = yaml.safe_load(f)
    img = cv2.imread(str(artifacts_dir / cfg["image"]), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ArtifactError(f"Cannot read map image in {artifacts_dir}")
    occ = np.where(img < 127, 1, 0).astype(np.uint8)
    res = float(cfg["resolution"])
    return occ, res, occ.shape


def _wall_segments(occ: np.ndarray, res: float, stride: int = 4
                   ) -> list[tuple[float, float, float, float, float]]:
    """Downsampled wall voxels -> (cx, cy, cz, sx, sy) boxes in world frame."""
    rows, cols = occ.shape
    boxes: list[tuple[float, float, float, float, float]] = []
    for r in range(0, rows, stride):
        for c in range(0, cols, stride):
            if occ[r, c] != 1:
                continue
            x = (c + 0.5) * res
            y = (rows - 1 - r + 0.5) * res
            boxes.append((x, y, 0.15, res * stride, res * stride))
    return boxes


def build_world_sdf(
    artifacts_dir: str | Path,
    wall_height_m: float = 0.30,
    wall_stride: int = 4,
    include_ground: bool = True,
) -> str:
    """Generate an SDF world string from step-1 artifacts."""
    d = Path(artifacts_dir)
    occ, res, (rows, cols) = _read_map(d)
    extent_x = cols * res
    extent_y = rows * res

    world = ET.Element("sdf", version="1.9")
    model = ET.SubElement(world, "world", name="raceline_track")
    ET.SubElement(model, "gravity").text = "0 0 -9.81"
    ET.SubElement(model, "magnetic_field").text = "0 0 0"

    if include_ground:
        ground = ET.SubElement(model, "model", name="ground_plane")
        static = ET.SubElement(ground, "static")
        static.text = "true"
        link = ET.SubElement(ground, "link", name="link")
        col = ET.SubElement(link, "collision", name="collision")
        geom = ET.SubElement(col, "geometry")
        plane = ET.SubElement(geom, "plane")
        ET.SubElement(plane, "normal").text = "0 0 1"
        ET.SubElement(plane, "size").text = f"{extent_x + 2} {extent_y + 2}"

    boxes = _wall_segments(occ, res, wall_stride)
    for i, (cx, cy, cz, sx, sy) in enumerate(boxes):
        m = ET.SubElement(model, "model", name=f"wall_{i}")
        st = ET.SubElement(m, "static")
        st.text = "true"
        pose = ET.SubElement(m, "pose")
        pose.text = f"{cx:.3f} {cy:.3f} {wall_height_m / 2:.3f} 0 0 0"
        link = ET.SubElement(m, "link", name="link")
        col = ET.SubElement(link, "collision", name="collision")
        geom = ET.SubElement(col, "geometry")
        box = ET.SubElement(geom, "box")
        ET.SubElement(box, "size").text = (
            f"{sx:.3f} {sy:.3f} {wall_height_m:.3f}")

    return ET.tostring(world, encoding="unicode")


def export_gazebo_world(
    artifacts_dir: str | Path,
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Write an SDF world file ready for ``gz sim``."""
    sdf = build_world_sdf(artifacts_dir, **kwargs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sdf, encoding="utf-8")
    return out


def export_model_sdf(output_dir: str | Path) -> Path:
    """Write a minimal F1TENTH-style ackermann vehicle SDF stub."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "f1tenth_raceline.sdf"
    path.write_text(_VEHICLE_SDF, encoding="utf-8")
    return path


_VEHICLE_SDF = """<?xml version="1.0"?>
<sdf version="1.9">
  <model name="f1tenth_raceline">
    <pose>0 0 0.05 0 0 0</pose>
    <link name="base_link">
      <inertial>
        <mass>3.5</mass>
        <inertia><ixx>0.02</ixx><iyy>0.04</iyy><izz>0.04</izz></inertia>
      </inertial>
      <collision name="chassis_collision">
        <geometry><box><size>0.50 0.30 0.10</size></box></geometry>
      </collision>
      <visual name="chassis_visual">
        <geometry><box><size>0.50 0.30 0.10</size></box></geometry>
        <material><ambient>0.2 0.4 0.8 1</ambient></material>
      </visual>
    </link>
    <!-- Attach ros2_control / gz-sim ackermann plugin in launch file -->
  </model>
</sdf>
"""
