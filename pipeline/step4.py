"""Step 4 entry point: dictate braking and acceleration zones.

    python -m pipeline.step4

Takes the recalculated racing line from step 3 plus the saved car parameters
and computes the fastest physically-possible speed at every waypoint using
the standard three-pass method:

  pass 1: cornering cap        v <= sqrt(mu * g / |curvature|)
  pass 2: forward (accelerate) limited by the friction circle -- grip spent
          on cornering is not available for acceleration
  pass 3: backward (brake)     same, so the car always brakes early enough

Wherever the resulting profile rises the car is in an ACCELERATION zone,
wherever it falls it is in a BRAKING zone; the rest is HOLD.  The zones are
printed (dictated), saved to zones.yaml, and drawn over the map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .step3 import G
from .track_env import load_track_data

ZONE_ACCEL = "ACCEL"
ZONE_BRAKE = "BRAKE"
ZONE_HOLD = "HOLD"
_MIN_ZONE_LEN_M = 0.5
_DV_EPS = 0.005           # m/s change per waypoint below which speed is "held"


# ----------------------------------------------------------- speed profile

def velocity_profile(curvature: np.ndarray, ds: float, mu: float,
                     v_max: float, v_min: float, a_accel: float,
                     a_brake: float, mass_kg: float = 3.5,
                     cda_m2: float = 0.04, crr: float = 0.02
                     ) -> tuple[np.ndarray, int]:
    """Fastest speed at each waypoint of the closed loop (forward-backward
    passes with a friction circle, drag and rolling resistance).
    Returns (speeds, n_grip_violations)."""
    a_lat_max = mu * G
    k = np.maximum(np.abs(curvature), 1e-6)
    v = np.minimum(np.sqrt(a_lat_max / k), v_max)
    n = len(v)

    def friction_circle(v_here: float, k_here: float, a_cap: float) -> float:
        ay_frac = min((v_here**2 * k_here) / a_lat_max, 1.0)
        return a_cap * np.sqrt(max(1.0 - ay_frac**2, 0.0))

    def resistive_decel(v_here: float) -> float:
        return (0.5 * 1.2 * cda_m2 * v_here**2 + crr * mass_kg * G) / mass_kg

    # The track is a loop, so each pass runs twice around: the second lap
    # propagates constraints across the seam at index 0.
    for _ in range(2):
        for i in range(2 * n):                       # forward: accel limits
            j, jn = i % n, (i + 1) % n
            ax = friction_circle(v[j], k[j], a_accel) - resistive_decel(v[j])
            v[jn] = min(v[jn], np.sqrt(max(v[j]**2 + 2.0 * ax * ds, 0.0)))
        for i in range(2 * n, 0, -1):                # backward: brake limits
            j, jp = i % n, (i - 1) % n
            ax = friction_circle(v[j], k[j], a_brake) + resistive_decel(v[j])
            v[jp] = min(v[jp], np.sqrt(v[j]**2 + 2.0 * ax * ds))

    violations = int(np.sum(v < v_min))
    return np.maximum(v, v_min), violations


# ------------------------------------------------------------------ zones

def dictate_zones(speeds: np.ndarray, ds: float) -> list[dict]:
    """Split the loop into ACCEL / BRAKE / HOLD segments."""
    n = len(speeds)
    dv = np.roll(speeds, -1) - speeds
    labels = np.where(dv > _DV_EPS, ZONE_ACCEL,
                      np.where(dv < -_DV_EPS, ZONE_BRAKE, ZONE_HOLD))

    # rotate so the loop seam (index 0) falls on a label change, which makes
    # every run of equal labels a contiguous slice of the rolled array
    start = 0
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            start = i
            break
    rolled = np.roll(labels, -start)

    zones = []
    seg_start = 0
    for i in range(1, n + 1):
        if i == n or rolled[i] != rolled[i - 1]:
            zones.append({
                "type": str(rolled[seg_start]),
                "start_idx": int((seg_start + start) % n),
                "length_m": float((i - seg_start) * ds),
            })
            seg_start = i

    # merge blips shorter than _MIN_ZONE_LEN_M into their predecessor
    merged = []
    for z in zones:
        if merged and (z["length_m"] < _MIN_ZONE_LEN_M
                       or z["type"] == merged[-1]["type"]):
            merged[-1]["length_m"] += z["length_m"]
        else:
            merged.append(z)
    # the loop wraps: if first and last ended up the same type, join them
    if len(merged) > 1 and merged[0]["type"] == merged[-1]["type"]:
        last = merged.pop()
        merged[0]["start_idx"] = last["start_idx"]
        merged[0]["length_m"] += last["length_m"]

    # attach start/end positions and speeds
    for z in merged:
        i0 = z["start_idx"]
        i1 = (i0 + int(round(z["length_m"] / ds))) % n
        z["s_start_m"] = round(i0 * ds, 2)
        z["s_end_m"] = round((i0 * ds + z["length_m"]) % (n * ds), 2)
        z["v_start_mps"] = round(float(speeds[i0]), 2)
        z["v_end_mps"] = round(float(speeds[i1]), 2)
        z["length_m"] = round(z["length_m"], 2)
    return merged


# -------------------------------------------------------------- plotting

def save_zone_plots(track, raceline_xy, speeds, zones, ds, out_dir: Path):
    written = []

    fig, ax = plt.subplots(figsize=(10, 10 * track.grid.shape[0] / track.grid.shape[1]))
    ax.imshow(track.grid.occupancy, cmap="gray_r", interpolation="nearest")
    px = track.grid.pixel_from_world(raceline_xy)
    sc = ax.scatter(px[:, 1], px[:, 0], c=speeds, cmap="RdYlGn", s=4)
    fig.colorbar(sc, ax=ax, fraction=0.04, label="target speed (m/s)")
    for z in zones:
        if z["type"] == ZONE_BRAKE:
            i = z["start_idx"]
            ax.annotate("BRAKE", (px[i, 1], px[i, 0]), color="red",
                        fontsize=8, fontweight="bold",
                        xytext=(px[i, 1] + 12, px[i, 0] - 12),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1))
    ax.set_title("Step 4: target speeds and braking points")
    fig.tight_layout()
    p = out_dir / "zones_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    s_axis = np.arange(len(speeds)) * ds
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_axis, speeds, color="black", lw=1.5)
    colors = {ZONE_ACCEL: "#2ca02c", ZONE_BRAKE: "#d62728", ZONE_HOLD: "#cccccc"}
    for z in zones:
        s0 = z["start_idx"] * ds
        ax.axvspan(s0, s0 + z["length_m"], color=colors[z["type"]], alpha=0.3)
    ax.set_xlabel("distance along racing line (m)")
    ax.set_ylabel("target speed (m/s)")
    ax.set_title("Speed profile (green = accelerate, red = brake, grey = hold)")
    ax.set_xlim(0, s_axis[-1] + ds)
    fig.tight_layout()
    p = out_dir / "speed_profile.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 4: braking and acceleration zone dictation.")
    ap.add_argument("--artifacts", type=str, default="artifacts")
    ap.add_argument("--params-file", type=str,
                    help="override car_params.yaml (e.g. to test a different "
                         "speed cap without retraining)")
    args = ap.parse_args(argv)

    out_dir = Path(args.artifacts)
    track = load_track_data(out_dir)

    params_path = Path(args.params_file) if args.params_file \
        else out_dir / "car_params.yaml"
    if not params_path.is_file():
        print("ERROR: car_params.yaml not found -- run step 3 first.")
        return 2
    with open(params_path) as f:
        cp = yaml.safe_load(f)
    mu = cp["measured"]["tire_grip_mu"]
    v_max = cp["measured"]["max_speed_mps"]
    v_min = cp["measured"]["min_speed_mps"]
    a_acc = cp["derived"]["max_accel_mps2"]
    a_brk = cp["derived"]["max_brake_mps2"]
    mass = cp["derived"].get("mass_kg", 3.5)
    cda = cp["derived"].get("cda_m2", 0.04)
    crr = cp["derived"].get("crr", 0.02)

    raceline_path = out_dir / "raceline_tuned.csv"
    if not raceline_path.is_file():
        raceline_path = out_dir / "raceline.csv"
        print("WARN: raceline_tuned.csv not found, falling back to the "
              "step-2 raceline (generic physics).")
    if not raceline_path.is_file():
        print("ERROR: no raceline found -- run steps 2 and 3 first.")
        return 2
    rl = np.loadtxt(raceline_path, delimiter=",", skiprows=1)
    xy, curvature = rl[:, :2], rl[:, 2]
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    length = float(seg.sum())
    ds = length / len(xy)
    print(f"Racing line: {length:.1f} m, {len(xy)} waypoints "
          f"({raceline_path.name})")

    speeds, violations = velocity_profile(curvature, ds, mu, v_max, v_min,
                                          a_acc, a_brk, mass, cda, crr)
    if violations:
        print(f"WARN: min speed ({v_min} m/s) exceeds the grip limit at "
              f"{violations} waypoints -- the car may slide there.")

    lap_estimate = float(np.sum(ds / speeds))
    zones = dictate_zones(speeds, ds)

    # ------------------------------------------------------- dictation
    n_acc = sum(z["type"] == ZONE_ACCEL for z in zones)
    n_brk = sum(z["type"] == ZONE_BRAKE for z in zones)
    print(f"\nSpeed range {speeds.min():.2f}-{speeds.max():.2f} m/s, "
          f"estimated lap time {lap_estimate:.2f} s")
    print(f"\nZONE DICTATION ({n_acc} acceleration, {n_brk} braking):\n")
    for i, z in enumerate(zones, 1):
        arrow = {"ACCEL": "/\\", "BRAKE": "\\/", "HOLD": "--"}[z["type"]]
        print(f"  zone {i:2d} | {z['type']:5s} {arrow} | "
              f"s = {z['s_start_m']:7.2f} -> {z['s_end_m']:7.2f} m "
              f"({z['length_m']:6.2f} m) | "
              f"{z['v_start_mps']:5.2f} -> {z['v_end_mps']:5.2f} m/s")

    # ------------------------------------------------------- outputs
    s_col = (np.arange(len(xy)) * ds).reshape(-1, 1)
    zone_col = np.full(len(xy), ZONE_HOLD, dtype=object)
    for z in zones:
        i0 = z["start_idx"]
        count = int(round(z["length_m"] / ds))
        idx = (i0 + np.arange(count)) % len(xy)
        zone_col[idx] = z["type"]

    final_path = out_dir / "raceline_final.csv"
    with open(final_path, "w") as f:
        f.write("s_m,x_m,y_m,curvature_1pm,speed_mps,zone\n")
        for i in range(len(xy)):
            f.write(f"{s_col[i, 0]:.4f},{xy[i, 0]:.4f},{xy[i, 1]:.4f},"
                    f"{curvature[i]:.4f},{speeds[i]:.4f},{zone_col[i]}\n")

    zones_path = out_dir / "zones.yaml"
    with open(zones_path, "w") as f:
        yaml.safe_dump({"estimated_lap_time_s": round(lap_estimate, 2),
                        "zones": zones}, f, sort_keys=False)

    written = save_zone_plots(track, xy, speeds, zones, ds, out_dir)
    print("\nWrote:")
    for p in [final_path, zones_path, *written]:
        print(f"  {p}")
    print("\nStep 4 complete. raceline_final.csv (position + speed + zone "
          "per waypoint) is the file the f1tenth_gym tracker and the ROS 2 "
          "pure-pursuit node will follow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
