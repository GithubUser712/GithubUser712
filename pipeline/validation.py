"""Step 1b: sanity-check the drivable area before anything expensive runs.

Checks performed (all on the connected free region that contains the start):

1. start point must lie on free space
2. the drivable corridor must NOT touch the image border
   (if it does, the outer wall has a gap -> the "track" leaks off the page)
3. the corridor must enclose an infield, i.e. be a closed loop
   (a topological ring has a hole; filling its holes must add pixels)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .map_ingestion import FREE, WALL, TrackMap

_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


@dataclass
class ValidationReport:
    ok: bool
    corridor_mask: np.ndarray | None      # bool (rows, cols): the drivable region
    widest_m: float                       # widest point of the corridor (track width)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_track(track: TrackMap, start_rc: tuple[int, int]) -> ValidationReport:
    occ = track.occupancy
    rows, cols = occ.shape
    r, c = int(round(start_rc[0])), int(round(start_rc[1]))
    errors: list[str] = []
    warnings: list[str] = []

    if not (0 <= r < rows and 0 <= c < cols):
        return ValidationReport(False, None, 0.0,
                                [f"start pixel (row={r}, col={c}) is outside the image"])
    if occ[r, c] == WALL:
        return ValidationReport(False, None, 0.0,
                                [f"start pixel (row={r}, col={c}) lies on a wall pixel"])

    free = occ == FREE
    labels, n_regions = ndimage.label(free, structure=_EIGHT_CONNECTED)
    corridor = labels == labels[r, c]

    # 1. leak check: a closed outer boundary keeps the corridor off the border
    if (corridor[0, :].any() or corridor[-1, :].any()
            or corridor[:, 0].any() or corridor[:, -1].any()):
        errors.append("drivable area touches the image border -- the outer "
                      "boundary is not a closed wall (find and close the gap)")

    # 2. loop check: a closed circuit must enclose an infield
    filled = ndimage.binary_fill_holes(corridor)
    if filled.sum() == corridor.sum():
        errors.append("drivable area encloses no infield -- the track is not a "
                      "closed loop (inner boundary missing or has a gap)")

    # 3. extra free regions (infield, outside the track) are normal -- just say so
    if n_regions > 1:
        warnings.append(f"image contains {n_regions} disconnected free regions; "
                        "using the one containing the start point (the others are "
                        "treated as unreachable, e.g. the infield)")

    widest_m = float(2.0 * track.distance_m[corridor].max()) if corridor.any() else 0.0
    if widest_m < 0.30:
        warnings.append(f"track is only {widest_m:.2f} m wide at its widest point -- "
                        "check the map resolution, an F1TENTH car is ~0.3 m wide")

    return ValidationReport(ok=not errors, corridor_mask=corridor,
                            widest_m=widest_m, errors=errors, warnings=warnings)
