"""Step 1a: load a track image and turn it into a binary occupancy grid.

Conventions used by the whole project:

* occupancy grid : uint8 array, shape (rows, cols); 1 = wall, 0 = free space
* pixel coords   : (row, col) with row 0 at the TOP of the image
* world coords   : metres; origin at the BOTTOM-LEFT corner of the image,
                   x to the right, y upwards.  This matches the ROS
                   map_server / f1tenth_gym convention with origin [0, 0, 0].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WALL = 1
FREE = 0


@dataclass
class TrackMap:
    """A thresholded track image plus everything derived directly from it."""

    occupancy: np.ndarray        # uint8 (rows, cols), 1 = wall, 0 = free
    resolution: float            # metres per pixel
    source_path: Path
    distance_m: np.ndarray = field(init=False)  # metres from each free pixel to nearest wall

    def __post_init__(self) -> None:
        free = (self.occupancy == FREE).astype(np.uint8)
        # distanceTransform measures, for every non-zero pixel, the distance
        # to the nearest zero pixel -- i.e. free-pixel -> nearest wall.
        self.distance_m = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape  # (rows, cols)

    def world_from_pixel(self, rc) -> np.ndarray:
        """(row, col) -> (x, y) metres.  Accepts a single pair or an (N, 2) array."""
        rows = self.occupancy.shape[0]
        r, c = np.asarray(rc, dtype=float).T
        x = (c + 0.5) * self.resolution
        y = (rows - 1 - r + 0.5) * self.resolution
        return np.stack([x, y], axis=-1)

    def pixel_from_world(self, xy) -> np.ndarray:
        """(x, y) metres -> (row, col).  Accepts a single pair or an (N, 2) array."""
        rows = self.occupancy.shape[0]
        x, y = np.asarray(xy, dtype=float).T
        c = x / self.resolution - 0.5
        r = (rows - 1) - (y / self.resolution - 0.5)
        return np.stack([r, c], axis=-1)


def load_track_map(path: str | Path, resolution: float, wall_threshold: int = 127) -> TrackMap:
    """Read an image and threshold it: pixels darker than `wall_threshold`
    become walls (1), everything else becomes free space (0).

    The threshold (instead of an exact black/white test) makes the loader
    tolerant of anti-aliased drawings, JPEG noise and SLAM-generated maps.
    """
    path = Path(path).expanduser()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if resolution <= 0.0:
        raise ValueError(f"resolution must be > 0 (got {resolution})")
    occupancy = np.where(img < wall_threshold, WALL, FREE).astype(np.uint8)
    return TrackMap(occupancy=occupancy, resolution=float(resolution), source_path=path)
