"""Generate RC-scale (1:10) track maps of famous circuits and road courses.

    python tools/make_f1_tracks.py            # all tracks
    python tools/make_f1_tracks.py monaco     # just one

Centerlines are projected to metres, scaled 1:10 (F1TENTH convention), and
rasterised as white free space between two closed black walls at 0.01 m/px.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import interpolate

SCALE = 0.10              # 1:10 of the real circuit
WALL_M = 0.30             # wall thickness on each side, metres at RC scale
MARGIN_M = 2.0            # white border around the outer wall
RESOLUTION = 0.01
MAX_IMAGE_PX = 65500        # OpenCV practical limit; warn above this

TRACKS = {
    "monaco": dict(source="mc-1929.geojson", width_m=1.00, import_script=None),
    "spa": dict(source="be-1925.geojson", width_m=1.35, import_script=None),
    "nurburgring": dict(
        source="de-gesamtstrecke.geojson",
        width_m=1.30,
        import_script="import_jrt_nring.py",
    ),
    "isle_of_man": dict(
        source="im-tt.geojson",
        width_m=0.65,
        import_script="import_osm_relation.py",
        import_args=["isle_of_man"],
    ),
    "targa_florio": dict(
        source="it-targa-grande.geojson",
        width_m=0.60,
        import_script="import_targa_florio.py",
        import_args=[],
    ),
}

SOURCES_DIR = Path("maps/sources")
OUT_DIR = Path("maps")


def ensure_source(name: str, cfg: dict) -> None:
    src = SOURCES_DIR / cfg["source"]
    if src.is_file():
        return
    script_name = cfg.get("import_script")
    if not script_name:
        raise FileNotFoundError(
            f"Missing source {src} for '{name}' — add the geojson or an import script")
    script = Path("tools") / script_name
    if not script.is_file():
        raise FileNotFoundError(f"Import script missing: {script}")
    import subprocess
    args = [sys.executable, str(script), *cfg.get("import_args", [])]
    print(f"Importing {src.name} (first run) ...")
    subprocess.run(args, cwd=Path("."), check=True)


def load_centerline_m(geojson_path: Path) -> tuple[np.ndarray, dict]:
    """Closed circuit centerline in metres at RC scale, plus properties."""
    gj = json.loads(geojson_path.read_text())
    feat = gj["features"][0]
    coords = feat["geometry"]["coordinates"]
    if feat["geometry"]["type"] == "MultiLineString":
        coords = coords[0]
    pts = np.asarray(coords, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    props = feat["properties"]
    if gj.get("coordinate_system") == "local_meters":
        # Already in full-scale metres (e.g. JRT-Trackmaps import).
        return pts * SCALE, props

    # WGS84 lon/lat -> local equirectangular projection
    lon0, lat0 = pts.mean(axis=0)
    R = 6371000.0
    x = np.radians(pts[:, 0] - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(pts[:, 1] - lat0) * R
    return np.stack([x, y], axis=1) * SCALE, props


def densify(pts: np.ndarray, step_m: float) -> np.ndarray:
    """Periodic spline through the sparse polyline, resampled every step_m."""
    tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0.0, per=True)
    u = np.linspace(0.0, 1.0, 20000, endpoint=False)
    dense = np.stack(interpolate.splev(u, tck), axis=1)
    closed = np.vstack([dense, dense[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(arc[-1] / step_m)
    s_t = np.linspace(0.0, arc[-1], n, endpoint=False)
    return np.stack([np.interp(s_t, arc, closed[:, 0]),
                     np.interp(s_t, arc, closed[:, 1])], axis=1), float(arc[-1])


def rasterize(pts: np.ndarray, width_m: float, res: float
              ) -> tuple[np.ndarray, tuple[int, int], float]:
    """Draw the corridor; returns (image, start_px (x, y), start heading deg)."""
    pad = MARGIN_M + width_m / 2.0 + WALL_M
    pts = pts - pts.min(axis=0) + pad
    extent = pts.max(axis=0) + pad
    w_px, h_px = int(math.ceil(extent[0] / res)), int(math.ceil(extent[1] / res))

    # world -> pixel, consistent with step 1: row 0 at top, y up in world
    cols = np.rint(pts[:, 0] / res).astype(np.int32)
    rows = (h_px - 1) - np.rint(pts[:, 1] / res).astype(np.int32)
    poly = np.stack([cols, rows], axis=1)

    img = np.full((h_px, w_px), 255, np.uint8)
    t_outer = int(round((width_m + 2 * WALL_M) / res))
    t_inner = int(round(width_m / res))
    cv2.polylines(img, [poly], True, 0, thickness=t_outer)
    cv2.polylines(img, [poly], True, 255, thickness=t_inner)

    d = pts[min(10, len(pts) - 1)] - pts[0]
    heading_deg = math.degrees(math.atan2(d[1], d[0]))
    return img, (int(cols[0]), int(rows[0])), heading_deg


def main(names: list[str]) -> int:
    for name in names:
        cfg = TRACKS[name]
        ensure_source(name, cfg)
        src = SOURCES_DIR / cfg["source"]
        centerline, props = load_centerline_m(src)
        res = RESOLUTION
        dense, lap_m = densify(centerline, step_m=res)
        pad = MARGIN_M + cfg["width_m"] / 2.0 + WALL_M
        extent = dense.max(axis=0) - dense.min(axis=0) + 2 * pad
        est_w = int(math.ceil(extent[0] / res))
        est_h = int(math.ceil(extent[1] / res))
        if est_w > MAX_IMAGE_PX or est_h > MAX_IMAGE_PX:
            print(
                f"\nERROR: '{name}' at {res} m/px would be ~{est_w} x {est_h} px "
                f"(>{MAX_IMAGE_PX}). RC lap ~{lap_m:.0f} m — needs a machine with "
                f"very large RAM or a tiled map pipeline.")
            return 1
        img, start, heading = rasterize(dense, cfg["width_m"], res)

        out = OUT_DIR / f"{name}.png"
        cv2.imwrite(str(out), img)

        meta_path = OUT_DIR / f"{name}.meta.yaml"
        with open(meta_path, "w") as f:
            yaml.safe_dump({
                "resolution": res,
                "spacing": res,
                "start_col": int(start[0]),
                "start_row": int(start[1]),
                "heading_deg": round(heading, 1),
                "image_size": {"width": int(img.shape[1]), "height": int(img.shape[0])},
                "lap_length_m": round(lap_m, 1),
                "variant": props.get("Name", name),
            }, f, sort_keys=False)

        print(f"\n{props['Name']}  (real lap {props['length']} m)")
        print(f"  wrote {out}: {img.shape[1]} x {img.shape[0]} px, {res} m/px")
        print(f"  wrote {meta_path}")
        print(f"  RC scale: lap {lap_m:.0f} m, track width {cfg['width_m']} m, "
              f"walls {WALL_M} m")
        print(f"  run step 1 with:\n"
              f"    python -m pipeline.step1 --preset {name} "
              f"--out artifacts_{name}")
    return 0


if __name__ == "__main__":
    chosen = sys.argv[1:] or list(TRACKS)
    bad = [n for n in chosen if n not in TRACKS]
    if bad:
        print(f"unknown track(s): {bad}; choose from {list(TRACKS)}")
        sys.exit(2)
    sys.exit(main(chosen))
