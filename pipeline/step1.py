"""Step 1 entry point: map ingestion.

Foolproof usage (Jetson or any machine):

    python -m pipeline.step1 --preset sample

Presets: sample, monaco, spa, nurburgring  (see config/map_presets.yaml)
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback

from raceline.core.deps import require_step1_deps
from raceline.core.exceptions import ArtifactError
from raceline.core.maps import load_preset
from raceline.core.paths import chdir_to_project, resolve_path


def _prompt(text: str, parser, error: str):
    while True:
        raw = input(text).strip()
        try:
            return parser(raw)
        except (ValueError, OSError):
            print(f"  !! {error}")


def _parse_image_path(raw: str, root):
    p = resolve_path(raw, root)
    if not p.is_file():
        raise ValueError(str(p))
    return p


def _parse_positive_float(raw: str) -> float:
    v = float(raw)
    if v <= 0.0:
        raise ValueError(raw)
    return v


def _parse_pixel(raw: str) -> tuple[int, int]:
    a, b = raw.replace(" ", "").split(",")
    return int(a), int(b)


def _fail(msg: str, code: int = 1) -> int:
    print(f"\nERROR: {msg}")
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 1: ingest a track map (black = wall, white = free).",
        epilog="Easiest:  python -m pipeline.step1 --preset sample")
    ap.add_argument("--preset", type=str,
                    choices=["sample", "monaco", "spa", "nurburgring"],
                    help="built-in map with known-good start pose (recommended)")
    ap.add_argument("--map", type=str, help="path to the track image")
    ap.add_argument("--resolution", type=float, help="metres per pixel")
    ap.add_argument("--start", type=str,
                    help="start pixel as 'x,y' (x = column, y = row)")
    ap.add_argument("--heading", type=float,
                    help="start heading in degrees (0 = +x/east, CCW positive)")
    ap.add_argument("--out", type=str, default="artifacts",
                    help="output directory (default: artifacts/)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="centerline waypoint spacing in metres (default: preset "
                         "spacing or 0.01)")
    ap.add_argument("--wall-threshold", type=int, default=127,
                    help="grey level below which a pixel is a wall (default 127)")
    args = ap.parse_args(argv)

    require_step1_deps()

    from .artifacts import save_artifacts
    from .centerline import CenterlineError, extract_centerline
    from .map_ingestion import FREE, load_track_map
    from .validation import validate_track

    try:
        root = chdir_to_project()
    except FileNotFoundError as exc:
        return _fail(str(exc), 2)

    if not (root / "pipeline" / "step1.py").is_file():
        return _fail(
            "pipeline/step1.py not found — you may be on the empty main branch.\n"
            "  git fetch origin\n"
            "  git checkout cursor/step5-6-vesc-3d48", 2)

    try:
        if args.preset:
            preset = load_preset(args.preset, root)
            map_path = preset.map_path
            resolution = preset.resolution
            start_col, start_row = preset.start_col, preset.start_row
            heading_deg = preset.heading_deg
            if args.spacing is None:
                spacing = preset.spacing
            else:
                spacing = args.spacing
            print(f"Preset '{args.preset}' @ {resolution} m/px:")
            print(f"  map        {map_path}")
            print(f"  resolution {resolution} m/px")
            print(f"  start      {start_col},{start_row}  heading {heading_deg} deg")
            print(f"  spacing    {spacing} m between waypoints")
        else:
            if args.map is not None:
                map_path = resolve_path(args.map, root)
                if not map_path.is_file():
                    return _fail(f"no such file: {map_path}", 2)
            else:
                map_path = _prompt(
                    "Path to track image (black = walls, white = free): ",
                    lambda r: _parse_image_path(r, root),
                    "file not found, try again.")

            resolution = args.resolution if args.resolution is not None else _prompt(
                "Map resolution in metres per pixel (e.g. 0.05): ",
                _parse_positive_float, "enter a positive number.")

            if args.start is not None:
                start_col, start_row = _parse_pixel(args.start)
            else:
                track_tmp = load_track_map(map_path, resolution, args.wall_threshold)
                rows, cols = track_tmp.shape
                start_col, start_row = _prompt(
                    f"Start pixel as 'x,y' (x: 0..{cols - 1}, y: 0..{rows - 1}): ",
                    _parse_pixel, "enter two integers like 420,310.")

            heading_deg = args.heading if args.heading is not None else _prompt(
                "Start heading in degrees (0 = +x/east, 90 = +y/north, CCW): ",
                float, "enter a number.")
            spacing = args.spacing if args.spacing is not None else 0.01
    except ArtifactError as exc:
        return _fail(str(exc), 2)

    heading_rad = math.radians(heading_deg)

    try:
        track = load_track_map(map_path, resolution, args.wall_threshold)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(str(exc), 2)

    rows, cols = track.shape
    free_pct = 100.0 * (track.occupancy == FREE).mean()
    print(f"\nLoaded {map_path.name}")
    print(f"  grid:       {cols} x {rows} px  "
          f"({cols * resolution:.1f} m x {rows * resolution:.1f} m)")
    print(f"  free space: {free_pct:.1f} % of pixels\n")

    if not (0 <= start_row < rows and 0 <= start_col < cols):
        return _fail(
            f"start pixel ({start_col},{start_row}) outside image "
            f"(valid x: 0..{cols - 1}, y: 0..{rows - 1})", 2)

    report = validate_track(track, (start_row, start_col))
    for w in report.warnings:
        print(f"WARN:  {w}")
    for e in report.errors:
        print(f"ERROR: {e}")
    if not report.ok:
        print("\nMap rejected — fix the issues above and rerun step 1.")
        print("Tips:")
        print("  - walls must be black/dark, free space white/light")
        print("  - track must be a closed loop (inner + outer wall)")
        print("  - start pixel must be on the white corridor, not a wall")
        return 1
    print(f"Map OK. Widest corridor: {report.widest_m:.2f} m\n")

    start_xy = track.world_from_pixel((start_row, start_col))
    try:
        centerline = extract_centerline(
            track, report.corridor_mask, start_xy, heading_rad,
            spacing_m=spacing)
    except CenterlineError as exc:
        return _fail(
            f"centerline extraction failed: {exc}\n"
            "  Try a different --start point on the start/finish straight.", 1)

    widths = 2.0 * centerline.clearance
    print("Centerline extracted:")
    print(f"  track length: {centerline.length_m:.2f} m")
    print(f"  waypoints:    {len(centerline.points)} "
          f"(every {centerline.length_m / len(centerline.points):.3f} m)")
    print(f"  track width:  min {widths.min():.2f} m / "
          f"mean {widths.mean():.2f} m / max {widths.max():.2f} m\n")

    out_dir = resolve_path(args.out, root)
    try:
        written = save_artifacts(track, start_xy, heading_rad, centerline, out_dir)
    except Exception as exc:
        print(traceback.format_exc())
        return _fail(f"failed to write artifacts: {exc}", 1)

    print("Wrote:")
    for p in written:
        print(f"  {p}")
    print(f"\nStep 1 complete.  Open {out_dir / 'debug.png'} to verify.")
    print("Next:")
    print(f"  python -m pipeline.step2 --artifacts {out_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
