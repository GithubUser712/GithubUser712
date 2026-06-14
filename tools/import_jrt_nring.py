"""Import Nürburgring Gesamtstrecke (Kurzanbindung) from JRT-Trackmaps coords.

Source: drjackyl/JRT-Trackmaps — nurburgring combinedshorta
(~24 km full-scale; Nordschleife + GP link, not the standalone GP loop).

    python tools/import_jrt_nring.py
"""

from __future__ import annotations

import json
import math
import pickle
import ssl
import urllib.request
from pathlib import Path

import yaml

JRT_URL = (
    "https://raw.githubusercontent.com/drjackyl/JRT-Trackmaps/master/"
    "nurburgring%20combinedshorta/coord"
)
OUT_GEOJSON = Path("maps/sources/de-gesamtstrecke.geojson")


def fetch_jrt_coords() -> list[tuple[float, float]]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    raw = urllib.request.urlopen(JRT_URL, context=ctx).read()
    d = pickle.loads(raw)
    n = int(d["k_max"])
    return [(float(d["x"][i]), float(d["y"][i])) for i in range(n)]


def to_geojson(xy: list[tuple[float, float]], length_m: float) -> dict:
    # Store as local metres (x east, y north) — not WGS84.
    coords = [[x, y] for x, y in xy]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return {
        "type": "FeatureCollection",
        "name": "de-gesamtstrecke",
        "coordinate_system": "local_meters",
        "features": [{
            "type": "Feature",
            "properties": {
                "id": "de-gesamtstrecke",
                "Name": "Nürburgring Gesamtstrecke (Kurzanbindung)",
                "Location": "Nürburg",
                "length": int(round(length_m)),
                "source": "JRT-Trackmaps combinedshorta",
                "note": "Nordschleife + GP connector; not the standalone GP loop",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
        }],
    }


def main() -> int:
    xy = fetch_jrt_coords()
    length_m = sum(
        math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
        for i in range(len(xy) - 1)
    )
    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_GEOJSON.write_text(
        json.dumps(to_geojson(xy, length_m), indent=2), encoding="utf-8")
    print(f"wrote {OUT_GEOJSON}")
    print(f"  {len(xy)} points, full-scale length {length_m / 1000:.2f} km")
    print(f"  RC lap @ 1:10 scale: {length_m * 0.1:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
