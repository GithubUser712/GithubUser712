"""Step 1c: extract a smooth, ordered, evenly-spaced centerline from the corridor.

Pipeline:  corridor mask
        -> morphological skeleton           (skimage.skeletonize)
        -> prune spurs / keep largest loop  (a ring has no endpoints, spurs do)
        -> walk the loop pixel by pixel     (gives an ORDERED list of points)
        -> periodic smoothing spline        (removes pixel staircase)
        -> resample at uniform arc length   (constant waypoint spacing in metres)
        -> orient & roll                    (index 0 = nearest the start pose,
                                             direction matches the start heading)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, ndimage
from skimage.morphology import skeletonize

from .map_ingestion import TrackMap


class CenterlineError(RuntimeError):
    """Raised when no clean closed centerline can be traced."""


@dataclass
class Centerline:
    points: np.ndarray     # (N, 2) world coords [x_m, y_m], ordered around the loop
    clearance: np.ndarray  # (N,) metres from each waypoint to the nearest wall
    length_m: float        # total loop length


# 4-connected neighbours first so the walk prefers straight steps over diagonals
_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]
_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def extract_centerline(track: TrackMap, corridor_mask: np.ndarray,
                       start_xy, heading_rad: float,
                       spacing_m: float = 0.05,
                       smoothing_m: float | None = None) -> Centerline:
    if smoothing_m is None:
        # allow the spline to deviate ~2 px from the raw skeleton: enough to
        # kill the pixel staircase without changing the track shape
        smoothing_m = 2.0 * track.resolution

    skeleton = skeletonize(corridor_mask.astype(bool))
    skeleton = _prune_spurs(skeleton)
    skeleton = _largest_component(skeleton)
    loop_rc = _order_loop(skeleton)

    raw_xy = track.world_from_pixel(loop_rc)
    points, length_m = _smooth_resample(raw_xy, spacing_m, smoothing_m)
    points = _orient(points, np.asarray(start_xy, dtype=float), heading_rad)
    clearance = _sample_clearance(track, points)
    return Centerline(points=points, clearance=clearance, length_m=length_m)


def _prune_spurs(skel: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Repeatedly delete endpoints (pixels with <= 1 neighbour).

    Dead-end branches of the skeleton shrink away one pixel per iteration;
    the closed loop itself has no endpoints, so it survives untouched.
    """
    skel = skel.copy()
    kernel = _EIGHT_CONNECTED.copy()
    kernel[1, 1] = 0
    for _ in range(max_iter):
        neighbours = ndimage.convolve(skel.astype(int), kernel, mode="constant")
        endpoints = skel & (neighbours <= 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def _largest_component(skel: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(skel, structure=_EIGHT_CONNECTED)
    if n == 0:
        raise CenterlineError("skeleton vanished after pruning -- the corridor "
                              "is probably not a closed loop")
    if n == 1:
        return skel
    sizes = ndimage.sum(skel, labels, index=range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _order_loop(skel: np.ndarray) -> np.ndarray:
    """Walk the loop pixel-by-pixel to produce an ordered (M, 2) (row, col) array."""
    remaining = {(int(r), int(c)) for r, c in np.argwhere(skel)}
    total = len(remaining)
    if total < 8:
        raise CenterlineError("skeleton too small to be a track loop")

    current = min(remaining)            # deterministic starting pixel
    path = [current]
    remaining.discard(current)
    while True:
        step = None
        for dr, dc in _NEIGHBOURS:
            cand = (current[0] + dr, current[1] + dc)
            if cand in remaining:
                step = cand
                break
        if step is None:                # no unvisited neighbour -> loop finished
            break
        path.append(step)
        remaining.discard(step)
        current = step

    # Diagonal staircases legitimately leave a few redundant pixels behind,
    # but losing a large fraction means the walk got stuck on a branch.
    if len(remaining) > 0.05 * total:
        raise CenterlineError(f"centerline trace abandoned {len(remaining)}/{total} "
                              "pixels -- skeleton is not a single clean loop")
    first, last = path[0], path[-1]
    if max(abs(first[0] - last[0]), abs(first[1] - last[1])) > 2:
        raise CenterlineError("traced centerline does not close back on itself")
    return np.asarray(path, dtype=float)


def _smooth_resample(pts: np.ndarray, spacing_m: float,
                     smoothing_m: float) -> tuple[np.ndarray, float]:
    """Fit a periodic spline through the ordered points and resample it
    at uniform arc-length intervals of `spacing_m` metres."""
    x, y = pts[:, 0], pts[:, 1]
    s = len(x) * smoothing_m**2          # scipy: sum of squared residuals <= s
    tck, _ = interpolate.splprep([x, y], s=s, per=True)

    u = np.linspace(0.0, 1.0, max(4 * len(x), 2000), endpoint=False)
    dense = np.stack(interpolate.splev(u, tck), axis=1)

    closed = np.vstack([dense, dense[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(arc[-1])

    n = max(int(round(length / spacing_m)), 50)
    s_targets = np.linspace(0.0, length, n, endpoint=False)
    rx = np.interp(s_targets, arc, closed[:, 0])
    ry = np.interp(s_targets, arc, closed[:, 1])
    return np.stack([rx, ry], axis=1), length


def _orient(points: np.ndarray, start_xy: np.ndarray, heading_rad: float) -> np.ndarray:
    """Roll the loop so index 0 is the waypoint nearest the start pose and
    flip its direction if needed so the loop runs along the start heading."""
    i0 = int(np.argmin(np.linalg.norm(points - start_xy, axis=1)))
    points = np.roll(points, -i0, axis=0)
    tangent = points[1] - points[0]
    heading = np.array([np.cos(heading_rad), np.sin(heading_rad)])
    if float(tangent @ heading) < 0.0:
        points = np.roll(points[::-1], 1, axis=0)   # reverse, keep index 0 in place
    return points


def _sample_clearance(track: TrackMap, points: np.ndarray) -> np.ndarray:
    rc = np.rint(track.pixel_from_world(points)).astype(int)
    rows, cols = track.shape
    r = np.clip(rc[:, 0], 0, rows - 1)
    c = np.clip(rc[:, 1], 0, cols - 1)
    return track.distance_m[r, c]
