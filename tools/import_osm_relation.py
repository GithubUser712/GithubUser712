"""Import a road-course centerline from an OpenStreetMap route relation.

    python tools/import_osm_relation.py isle_of_man
    python tools/import_osm_relation.py --relation 188240 --out maps/sources/im-tt.geojson

Uses the OSM API 0.6 /full dump (more reliable than Overpass on Windows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_geojson import (
    fetch_osm_relation_full,
    flatten_osm_route,
    haversine_path_m,
    write_geojson,
)

PRESETS = {
    "isle_of_man": dict(
        relation_id=188240,
        out=Path("maps/sources/im-tt.geojson"),
        gid="im-tt",
        name="Isle of Man TT Mountain Course",
        location="Douglas, Isle of Man",
        source="OpenStreetMap relation 188240",
        note="Snaefell Mountain Course (~37.7 mi / 60.7 km full-scale)",
    ),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    ap.add_argument("--relation", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--name", type=str, default="OSM route")
    args = ap.parse_args(argv)

    if args.preset:
        cfg = PRESETS[args.preset]
    else:
        if args.relation is None or args.out is None:
            ap.error("provide a preset name or both --relation and --out")
        cfg = dict(
            relation_id=args.relation,
            out=args.out,
            gid=args.out.stem,
            name=args.name,
            location="",
            source=f"OpenStreetMap relation {args.relation}",
            note="",
        )

    rid = cfg["relation_id"]
    print(f"Fetching OSM relation {rid} ...")
    xml = fetch_osm_relation_full(rid)
    coords, tags = flatten_osm_route(xml, rid)
    length_m = haversine_path_m(coords)
    if "length" in tags:
        raw = tags["length"].strip().lower()
        try:
            if raw.endswith("km"):
                length_m = float(raw.removesuffix("km").strip()) * 1000.0
            elif raw.endswith("m"):
                length_m = float(raw.removesuffix("m").strip())
        except ValueError:
            pass

    write_geojson(
        cfg["out"],
        name=tags.get("name", cfg["name"]),
        gid=cfg["gid"],
        coords_lonlat=coords,
        length_m=length_m,
        location=cfg.get("location", ""),
        source=cfg.get("source", ""),
        note=cfg.get("note", ""),
    )
    print(f"wrote {cfg['out']}")
    print(f"  {len(coords)} points, full-scale length {length_m / 1000:.2f} km")
    print(f"  RC lap @ 1:10 scale: {length_m * 0.1:.0f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
