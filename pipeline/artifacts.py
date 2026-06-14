"""Step 1d: write everything later stages (and ROS / f1tenth_gym) will consume.

Outputs, all inside the chosen artifacts directory:

  map.pgm + map.yaml  occupancy grid in ROS map_server / f1tenth_gym convention
  centerline.csv      ordered waypoints: x_m, y_m, clearance_m
  track_meta.yaml     resolution, start pose, track length / width statistics
  debug.png           visual overlay so you can eyeball the result
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe; we only ever save figures to disk
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .centerline import Centerline
from .map_ingestion import WALL, TrackMap


def save_artifacts(track: TrackMap, start_xy, heading_rad: float,
                   centerline: Centerline, out_dir: str | Path,
                   *, track_label: str | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- occupancy grid (map_server convention: 0 = occupied, 254 = free) ---
    pgm = np.where(track.occupancy == WALL, 0, 254).astype(np.uint8)
    pgm_path = out_dir / "map.pgm"
    cv2.imwrite(str(pgm_path), pgm)
    written.append(pgm_path)

    yaml_path = out_dir / "map.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({
            "image": "map.pgm",
            "resolution": float(track.resolution),
            "origin": [0.0, 0.0, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }, f, sort_keys=False)
    written.append(yaml_path)

    # --- centerline waypoints ---
    csv_path = out_dir / "centerline.csv"
    data = np.column_stack([centerline.points, centerline.clearance])
    np.savetxt(csv_path, data, delimiter=",", fmt="%.4f",
               header="x_m,y_m,clearance_m", comments="")
    written.append(csv_path)

    # --- metadata ---
    widths = 2.0 * centerline.clearance
    n = len(centerline.points)
    meta_path = out_dir / "track_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump({
            "track_label": track_label or Path(track.source_path).stem,
            "source_image": str(track.source_path),
            "resolution_m_per_px": float(track.resolution),
            "grid_size_px": {"cols": int(track.shape[1]), "rows": int(track.shape[0])},
            "start_pose": {
                "x_m": round(float(start_xy[0]), 4),
                "y_m": round(float(start_xy[1]), 4),
                "yaw_rad": round(float(heading_rad), 6),
            },
            "track_length_m": round(centerline.length_m, 3),
            "num_waypoints": n,
            "waypoint_spacing_m": round(centerline.length_m / n, 4),
            "track_width_m": {
                "min": round(float(widths.min()), 3),
                "mean": round(float(widths.mean()), 3),
                "max": round(float(widths.max()), 3),
            },
        }, f, sort_keys=False)
    written.append(meta_path)

    written.append(_save_debug_png(track, start_xy, heading_rad, centerline,
                                   out_dir / "debug.png"))
    return written


def _save_debug_png(track: TrackMap, start_xy, heading_rad: float,
                    centerline: Centerline, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 10 * track.shape[0] / track.shape[1]))
    ax.imshow(track.occupancy, cmap="gray_r", interpolation="nearest")

    px = track.pixel_from_world(centerline.points)        # (N, 2) row, col
    sc = ax.scatter(px[:, 1], px[:, 0], c=2.0 * centerline.clearance,
                    cmap="viridis", s=3)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="track width (m)")

    start_px = track.pixel_from_world(start_xy)
    # image rows grow downwards, so a +y (north) heading points to smaller rows
    arrow_len = 0.06 * max(track.shape)
    ax.arrow(start_px[1], start_px[0],
             arrow_len * np.cos(heading_rad), -arrow_len * np.sin(heading_rad),
             color="red", width=arrow_len * 0.06,
             head_width=arrow_len * 0.3, length_includes_head=True, zorder=5)
    ax.plot(px[0, 1], px[0, 0], "r*", markersize=14, zorder=5,
            label="waypoint 0 / start")

    ax.set_title("Step 1 result: occupancy grid + ordered centerline")
    ax.legend(loc="upper right")
    ax.set_xlabel("col (px)")
    ax.set_ylabel("row (px)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
