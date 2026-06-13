#!/usr/bin/env python3
"""Export step-1 artifacts to a Gazebo Harmonic world file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raceline.gazebo import export_gazebo_world, export_model_sdf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export artifacts to Gazebo SDF world.")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="ros2/raceline_gazebo/worlds/track.sdf")
    ap.add_argument("--wall-stride", type=int, default=4,
                    help="downsample walls (higher = fewer collision boxes)")
    ap.add_argument("--vehicle-model", action="store_true",
                    help="also write f1tenth_raceline.sdf vehicle stub")
    args = ap.parse_args(argv)

    out = export_gazebo_world(
        args.artifacts, args.out, wall_stride=args.wall_stride)
    print(f"Wrote Gazebo world: {out}")

    if args.vehicle_model:
        model = export_model_sdf(Path(args.out).parent / "models")
        print(f"Wrote vehicle model: {model}")

    print("\nLaunch in ROS 2 (Jazzy + Gazebo Harmonic):")
    print("  ros2 launch raceline_gazebo sim.launch.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
