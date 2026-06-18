"""Shared helpers for building track GeoJSON sources."""

from __future__ import annotations

import json
import math
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def polyline_length_m(coords: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
        for i in range(len(coords) - 1)
    )


def write_geojson(
    path: Path,
    *,
    name: str,
    gid: str,
    coords_lonlat: list[tuple[float, float]],
    length_m: float,
    location: str = "",
    source: str = "",
    note: str = "",
) -> None:
    """Write WGS84 lon/lat centerline GeoJSON (f1-circuits format)."""
    closed = list(coords_lonlat)
    if closed and closed[0] != closed[-1]:
        closed.append(closed[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "type": "FeatureCollection",
            "name": gid,
            "features": [{
                "type": "Feature",
                "properties": {
                    "id": gid,
                    "Name": name,
                    "Location": location,
                    "length": int(round(length_m)),
                    "source": source,
                    "note": note,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in closed],
                },
            }],
        }, indent=2),
        encoding="utf-8",
    )


def fetch_osm_relation_full(relation_id: int) -> str:
    url = f"https://www.openstreetmap.org/api/0.6/relation/{relation_id}/full"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "raceline-track-import/1.0"})
    with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
        return resp.read().decode("utf-8")


def _node_coord(nodes: dict[int, tuple[float, float]], nid: int) -> tuple[float, float]:
    if nid not in nodes:
        raise KeyError(f"OSM node {nid} missing from relation/full dump")
    return nodes[nid]


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def haversine_path_m(coords: list[tuple[float, float]]) -> float:
    return sum(_haversine_m(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _stitch_ways(
    way_ids: list[int],
    ways: dict[int, list[int]],
    nodes: dict[int, tuple[float, float]],
    *,
    max_gap_m: float = 75.0,
) -> list[tuple[float, float]]:
    """Greedy endpoint matching — typical for OSM route relations."""
    if not way_ids:
        return []
    remaining = list(way_ids)
    cur_id = remaining.pop(0)
    nds = ways[cur_id]
    coords = [_node_coord(nodes, n) for n in nds]

    while remaining:
        end = coords[-1]
        best_i = -1
        best_rev = False
        best_d = max_gap_m
        for i, wid in enumerate(remaining):
            wnds = ways[wid]
            p0 = _node_coord(nodes, wnds[0])
            p1 = _node_coord(nodes, wnds[-1])
            d0 = _haversine_m(end, p0)
            d1 = _haversine_m(end, p1)
            if d0 < best_d:
                best_d, best_i, best_rev = d0, i, False
            if d1 < best_d:
                best_d, best_i, best_rev = d1, i, True
        if best_i < 0:
            break
        wid = remaining.pop(best_i)
        pts = [_node_coord(nodes, n) for n in ways[wid]]
        if best_rev:
            pts = list(reversed(pts))
        if _dist2(coords[-1], pts[0]) < 1e-14:
            coords.extend(pts[1:])
        else:
            coords.extend(pts)
    return coords


def flatten_osm_route(xml_text: str, relation_id: int) -> tuple[list[tuple[float, float]], dict]:
    """Flatten an OSM route relation (nodes + ways) to ordered lon/lat points."""
    root = ET.fromstring(xml_text)
    nodes: dict[int, tuple[float, float]] = {}
    ways: dict[int, list[int]] = {}
    rel_tags: dict[str, str] = {}

    for elem in root:
        tag = elem.tag
        if tag == "node":
            nid = int(elem.attrib["id"])
            nodes[nid] = (float(elem.attrib["lon"]), float(elem.attrib["lat"]))
        elif tag == "way":
            wid = int(elem.attrib["id"])
            ways[wid] = [int(nd.attrib["ref"]) for nd in elem if nd.tag == "nd"]
        elif tag == "relation" and int(elem.attrib["id"]) == relation_id:
            rel_tags = {t.attrib["k"]: t.attrib["v"] for t in elem if t.tag == "tag"}
            members = [(m.attrib["type"], int(m.attrib["ref"])) for m in elem if m.tag == "member"]
            break
    else:
        raise ValueError(f"relation {relation_id} not found in OSM dump")

    coords: list[tuple[float, float]] = []
    way_members = [(t, r) for t, r in members if t == "way"]
    if way_members:
        coords = _stitch_ways([r for _, r in way_members], ways, nodes)
    else:
        for mtype, ref in members:
            if mtype == "node":
                pt = _node_coord(nodes, ref)
                if not coords or _dist2(coords[-1], pt) > 1e-14:
                    coords.append(pt)
            elif mtype == "way":
                nds = ways.get(ref)
                if not nds:
                    continue
                pts = [_node_coord(nodes, n) for n in nds]
                if not coords:
                    coords.extend(pts)
                    continue
                d0 = _dist2(coords[-1], pts[0])
                d1 = _dist2(coords[-1], pts[-1])
                ordered = pts if d0 <= d1 else list(reversed(pts))
                if _dist2(coords[-1], ordered[0]) < 1e-14:
                    coords.extend(ordered[1:])
                else:
                    coords.extend(ordered)

    if len(coords) < 3:
        raise ValueError(f"relation {relation_id}: too few coordinates after flattening")
    return coords, rel_tags
