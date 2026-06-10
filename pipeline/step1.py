"""Step 1 entry point: interactive map ingestion.

Run interactively (it prompts for everything):

    python -m pipeline.step1

or fully scripted:

    python -m pipeline.step1 --map maps/sample_track.png --resolution 0.05 \
        --start 873,350 --heading -69
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .artifacts import save_artifacts
from .centerline import CenterlineError, extract_centerline
from .map_ingestion import FREE, load_track_map
from .validation import validate_track


# ---------------------------------------------------------------- prompting

def _prompt(text: str, parser, error: str):
    """Keep asking until `parser` accepts the input."""
    while True:
        raw = input(text).strip()
        try:
            return parser(raw)
        except (ValueError, OSError):
            print(f"  !! {error}")


def _parse_image_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_file():
        raise ValueError(raw)
    return p


def _parse_positive_float(raw: str) -> float:
    v = float(raw)
    if v <= 0.0:
        raise ValueError(raw)
    return v


def _parse_pixel(raw: str) -> tuple[int, int]:
    """'x,y' in image coordinates (x = column from left, y = row from top)."""
    a, b = raw.replace(" ", "").split(",")
    return int(a), int(b)


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 1: ingest a track map (black = wall, white = free).")
    ap.add_argument("--map", type=str, help="path to the track image")
    ap.add_argument("--resolution", type=float, help="metres per pixel")
    ap.add_argument("--start", type=str,
                    help="start pixel as 'x,y' (x = column, y = row)")
    ap.add_argument("--heading", type=float,
                    help="start heading in degrees (0 = +x/east, CCW positive)")
    ap.add_argument("--out", type=str, default="artifacts",
                    help="output directory (default: artifacts/)")
    ap.add_argument("--spacing", type=float, default=0.05,
                    help="centerline waypoint spacing in metres (default 0.05)")
    ap.add_argument("--wall-threshold", type=int, default=127,
                    help="grey level below which a pixel is a wall (default 127)")
    args = ap.parse_args(argv)

    # 1. map image + resolution -------------------------------------------
    if args.map is not None:
        map_path = Path(args.map).expanduser()
        if not map_path.is_file():
            print(f"ERROR: no such file: {map_path}")
            return 2
    else:
        map_path = _prompt("Path to track image (black = walls, white = free): ",
                           _parse_image_path, "file not found, try again.")

    resolution = args.resolution if args.resolution is not None else _prompt(
        "Map resolution in metres per pixel (e.g. 0.05): ",
        _parse_positive_float, "enter a positive number.")

    track = load_track_map(map_path, resolution, args.wall_threshold)
    rows, cols = track.shape
    free_pct = 100.0 * (track.occupancy == FREE).mean()
    print(f"\nLoaded {map_path}")
    print(f"  grid:       {cols} x {rows} px  "
          f"({cols * resolution:.1f} m x {rows * resolution:.1f} m)")
    print(f"  free space: {free_pct:.1f} % of pixels\n")

    # 2. start pose ---------------------------------------------------------
    if args.start is not None:
        start_col, start_row = _parse_pixel(args.start)
    else:
        start_col, start_row = _prompt(
            f"Start pixel as 'x,y' (x: 0..{cols - 1} from left, "
            f"y: 0..{rows - 1} from top): ",
            _parse_pixel, "enter two integers like 420,310.")

    heading_deg = args.heading if args.heading is not None else _prompt(
        "Start heading in degrees (0 = +x/east, 90 = +y/north, CCW): ",
        float, "enter a number.")
    heading_rad = math.radians(heading_deg)

    # 3. validation ----------------------------------------------------------
    report = validate_track(track, (start_row, start_col))
    for w in report.warnings:
        print(f"WARN:  {w}")
    for e in report.errors:
        print(f"ERROR: {e}")
    if not report.ok:
        print("\nMap rejected -- fix the issues above and rerun step 1.")
        return 1
    print(f"Map OK. Widest point of the corridor: {report.widest_m:.2f} m\n")

    # 4. centerline -----------------------------------------------------------
    start_xy = track.world_from_pixel((start_row, start_col))
    try:
        centerline = extract_centerline(track, report.corridor_mask,
                                        start_xy, heading_rad,
                                        spacing_m=args.spacing)
    except CenterlineError as exc:
        print(f"ERROR: centerline extraction failed: {exc}")
        return 1

    widths = 2.0 * centerline.clearance
    print("Centerline extracted:")
    print(f"  track length: {centerline.length_m:.2f} m")
    print(f"  waypoints:    {len(centerline.points)} "
          f"(every {centerline.length_m / len(centerline.points):.3f} m)")
    print(f"  track width:  min {widths.min():.2f} m / "
          f"mean {widths.mean():.2f} m / max {widths.max():.2f} m\n")

    # 5. artifacts -------------------------------------------------------------
    written = save_artifacts(track, start_xy, heading_rad, centerline, args.out)
    print("Wrote:")
    for p in written:
        print(f"  {p}")
    print("\nStep 1 complete. Open debug.png to verify the centerline, then "
          "step 2 (RL racing line) will consume map.yaml, centerline.csv and "
          "track_meta.yaml from the same directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
