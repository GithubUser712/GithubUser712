"""Generate a sample closed-loop track image for testing step 1.

Draws a wobbly oval: first a thick black band along a parametric closed
curve, then a slightly thinner white band on top of it.  What survives is a
pair of closed black walls with a white corridor in between -- exactly the
black = 1 / white = 0 format step 1 expects.
"""

import math
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1000, 700
CORRIDOR_PX = 70          # corridor width  (70 px * 0.05 m/px = 3.5 m)
WALL_PX = 12              # wall thickness on each side
RESOLUTION = 0.05         # suggested metres per pixel


def main() -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 1500, endpoint=False)
    rx = 330 + 60 * np.sin(2 * t + 0.8)
    ry = 220 + 40 * np.sin(3 * t)
    x = WIDTH / 2 + rx * np.cos(t)
    y = HEIGHT / 2 + ry * np.sin(t)
    pts = np.stack([x, y], axis=1).astype(np.int32)

    img = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    cv2.polylines(img, [pts], True, 0, thickness=CORRIDOR_PX + 2 * WALL_PX)
    cv2.polylines(img, [pts], True, 255, thickness=CORRIDOR_PX)

    out = Path("maps/sample_track.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)

    # heading of the curve at the start point, in WORLD coords (image y is down);
    # use a ~10-point stride so integer rounding doesn't zero out the tangent
    d = pts[10] - pts[0]
    heading_deg = math.degrees(math.atan2(-float(d[1]), float(d[0])))

    print(f"wrote {out} ({WIDTH}x{HEIGHT} px)")
    print("test step 1 with:\n")
    print(f"  python -m pipeline.step1 --map {out} --resolution {RESOLUTION} "
          f"--start {pts[0][0]},{pts[0][1]} --heading {heading_deg:.0f}")


if __name__ == "__main__":
    main()
