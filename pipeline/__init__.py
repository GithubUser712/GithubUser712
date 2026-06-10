"""Offline raceline pipeline for the self-driving RC car.

Step 1 (this package, modules below): map ingestion
    map_ingestion.py  -- load image -> binary occupancy grid + distance transform
    validation.py     -- closed-loop / connectivity checks on the drivable area
    centerline.py     -- skeleton -> ordered loop -> smooth resampled centerline
    artifacts.py      -- write map.pgm/map.yaml, centerline.csv, track_meta.yaml, debug.png
    step1.py          -- interactive CLI entry point (python -m pipeline.step1)

Later steps (RL racing line, vehicle parameters, braking/acceleration zones)
consume the artifacts produced here and never re-read the raw drawing.
"""
