"""Generate a sample closed-loop track image for testing step 1.

Also writes maps/sample_track.meta.yaml with the exact start pixel and heading
so `python -m pipeline.step1 --preset sample` needs zero manual input.

Higher resolution (finer simulation grid):
  python tools/make_sample_map.py --resolution 0.02
  # doubles pixel count to keep the same physical track size (~50 m x 35 m)

Or pass --resolution directly to step 1 for any map PNG:
  python -m pipeline.step1 --preset sample --resolution 0.02 --spacing 0.02
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE_WIDTH, BASE_HEIGHT = 1000, 700
BASE_CORRIDOR_PX = 70
BASE_WALL_PX = 12
BASE_RESOLUTION = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate maps/sample_track.png")
    ap.add_argument("--resolution", type=float, default=BASE_RESOLUTION,
                    help="metres per pixel for step 1 (default 0.05; use 0.02 "
                         "or 0.01 for finer grid)")
    ap.add_argument("--scale", type=float, default=None,
                    help="multiply image pixel size (default: auto from resolution "
                         "to keep physical track dimensions)")
    args = ap.parse_args()

    if args.resolution <= 0:
        print("ERROR: --resolution must be positive")
        return 2

    scale = args.scale if args.scale is not None else BASE_RESOLUTION / args.resolution
    width = int(round(BASE_WIDTH * scale))
    height = int(round(BASE_HEIGHT * scale))
    corridor_px = max(int(round(BASE_CORRIDOR_PX * scale)), 10)
    wall_px = max(int(round(BASE_WALL_PX * scale)), 2)

    root = Path(__file__).resolve().parent.parent
    t = np.linspace(0.0, 2.0 * np.pi, max(int(1500 * scale), 1500), endpoint=False)
    rx = (330 + 60 * np.sin(2 * t + 0.8)) * scale
    ry = (220 + 40 * np.sin(3 * t)) * scale
    x = width / 2 + rx * np.cos(t)
    y = height / 2 + ry * np.sin(t)
    pts = np.stack([x, y], axis=1).astype(np.int32)

    img = np.full((height, width), 255, dtype=np.uint8)
    cv2.polylines(img, [pts], True, 0, thickness=corridor_px + 2 * wall_px)
    cv2.polylines(img, [pts], True, 255, thickness=corridor_px)

    maps_dir = root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    out = maps_dir / "sample_track.png"
    cv2.imwrite(str(out), img)

    d = pts[10] - pts[0]
    heading_deg = math.degrees(math.atan2(-float(d[1]), float(d[0])))
    start_col, start_row = int(pts[0][0]), int(pts[0][1])

    meta = {
        "resolution": args.resolution,
        "start_col": start_col,
        "start_row": start_row,
        "heading_deg": round(heading_deg, 1),
        "image_size": {"width": width, "height": height},
        "scale": round(scale, 3),
    }
    meta_path = maps_dir / "sample_track.meta.yaml"
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    phys_w = width * args.resolution
    phys_h = height * args.resolution
    print(f"wrote {out} ({width}x{height} px, scale {scale:.2f}x)")
    print(f"  resolution {args.resolution} m/px -> {phys_w:.1f} m x {phys_h:.1f} m")
    print(f"wrote {meta_path}")
    print()
    print("Run step 1:")
    print("  python -m pipeline.step1 --preset sample")
    print("Or override spacing for finer centerline waypoints:")
    print(f"  python -m pipeline.step1 --preset sample --spacing {args.resolution}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
