"""Import Targa Florio Grande Circuit (92 mi / 148.823 km).

The Grande Madonie layout is approximated as a smooth closed centerline through
its documented control towns (anticlockwise). Arc length is normalized to the
historic 148.823 km before 1:10 RC scaling.

    python tools/import_targa_florio.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_geojson import write_geojson

OUT = Path("maps/sources/it-targa-grande.geojson")
TARGET_M = 148_823.0

# lon, lat — Grande Circuito control towns (historic route, anticlockwise)
WAYPOINTS_LONLAT: list[tuple[float, float]] = [
    (13.9053, 37.9877),   # Campofelice di Roccella (start/finish)
    (13.9833, 37.9036),   # Cerda / Floriopoli
    (13.8917, 37.8278),   # Caltavuturo
    (13.8689, 37.7894),   # Castellana Sicula
    (14.0928, 37.8072),   # Petralia Sottana
    (14.1069, 37.7947),   # Petralia Soprana
    (14.1536, 37.8667),   # Geraci Siculo
    (14.0881, 37.9317),   # Castelbuono
    (14.0067, 37.9428),   # Isnello
    (13.9389, 37.9767),   # Collesano
]


def _to_local_m(lonlat: list[tuple[float, float]]) -> np.ndarray:
    lons = np.array([p[0] for p in lonlat])
    lats = np.array([p[1] for p in lonlat])
    lon0, lat0 = lons.mean(), lats.mean()
    r = 6371000.0
    x = np.radians(lons - lon0) * r * math.cos(math.radians(lat0))
    y = np.radians(lats - lat0) * r
    return np.stack([x, y], axis=1)


def _smooth_loop(pts_m: np.ndarray, target_m: float, step_m: float = 50.0) -> np.ndarray:
    tck, _ = interpolate.splprep([pts_m[:, 0], pts_m[:, 1]], s=0.0, per=True)
    u = np.linspace(0.0, 1.0, 8000, endpoint=False)
    dense = np.stack(interpolate.splev(u, tck), axis=1)
    raw_len = float(np.sum(np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1)))
    dense *= target_m / raw_len
    closed = np.vstack([dense, dense[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = max(int(target_m / step_m), 200)
    s_t = np.linspace(0.0, target_m, n, endpoint=False)
    return np.stack([np.interp(s_t, arc, closed[:, 0]),
                     np.interp(s_t, arc, closed[:, 1])], axis=1)


def _to_lonlat(xy_m: np.ndarray, lonlat_ref: list[tuple[float, float]]) -> list[tuple[float, float]]:
    lons = np.array([p[0] for p in lonlat_ref])
    lats = np.array([p[1] for p in lonlat_ref])
    lon0, lat0 = lons.mean(), lats.mean()
    r = 6371000.0
    cos0 = math.cos(math.radians(lat0))
    lon = np.degrees(xy_m[:, 0] / (r * cos0)) + lon0
    lat = np.degrees(xy_m[:, 1] / r) + lat0
    return [(float(lo), float(la)) for lo, la in zip(lon, lat)]


def main() -> int:
    print("Building Targa Florio Grande centerline through control towns ...")
    local = _to_local_m(WAYPOINTS_LONLAT)
    loop_arr = _smooth_loop(local, TARGET_M)
    length_m = float(np.sum(np.linalg.norm(np.diff(np.vstack([loop_arr, loop_arr[:1]]), axis=0), axis=1)))
    lonlat = _to_lonlat(loop_arr, WAYPOINTS_LONLAT)

    write_geojson(
        OUT,
        name="Targa Florio Grande Circuit",
        gid="it-targa-grande",
        coords_lonlat=lonlat,
        length_m=TARGET_M,
        location="Madonie, Sicily, Italy",
        source="Smoothed loop through historic Grande control towns",
        note="Grande Circuito delle Madonie; arc length fixed to 148.823 km",
    )
    print(f"wrote {OUT}")
    print(f"  {len(lonlat)} points, full-scale length {TARGET_M / 1000:.3f} km")
    print(f"  resampled polyline {length_m / 1000:.3f} km")
    print(f"  RC lap @ 1:10 scale: {TARGET_M * 0.1:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
