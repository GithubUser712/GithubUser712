"""Map preset loading and auto-generation for foolproof step 1."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .exceptions import ArtifactError
from .paths import find_project_root, resolve_path


@dataclass(frozen=True)
class MapPreset:
    name: str
    map_path: Path
    resolution: float
    start_col: int
    start_row: int
    heading_deg: float


def _load_presets_yaml(root: Path) -> dict:
    path = root / "config" / "map_presets.yaml"
    if not path.is_file():
        raise ArtifactError(f"Missing preset file: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_sample_meta(root: Path) -> dict:
    meta = root / "maps" / "sample_track.meta.yaml"
    if not meta.is_file():
        return {}
    with open(meta) as f:
        return yaml.safe_load(f) or {}


def ensure_sample_map(root: Path) -> Path:
    """Generate maps/sample_track.png if missing."""
    out = root / "maps" / "sample_track.png"
    if out.is_file():
        return out
    script = root / "tools" / "make_sample_map.py"
    if not script.is_file():
        raise ArtifactError(
            f"Sample map not found at {out} and generator missing at {script}.\n"
            "  git checkout cursor/step5-6-vesc-3d48   # main branch has no code yet")
    print(f"Generating sample map (first run) ...")
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)
    if not out.is_file():
        raise ArtifactError(f"Map generation failed — expected {out}")
    return out


def load_preset(name: str, root: Path | None = None) -> MapPreset:
    """Load a named preset; auto-generates sample map when needed."""
    base = root or find_project_root()
    presets = _load_presets_yaml(base)
    if name not in presets:
        known = ", ".join(sorted(presets))
        raise ArtifactError(
            f"Unknown preset '{name}'.  Choose one of: {known}")

    cfg = presets[name]
    map_rel = cfg["map"]

    if name == "sample":
        ensure_sample_map(base)
        meta = _load_sample_meta(base)
        if not meta:
            raise ArtifactError(
                "maps/sample_track.meta.yaml missing — run:\n"
                "  python tools/make_sample_map.py")
        return MapPreset(
            name=name,
            map_path=resolve_path(map_rel, base),
            resolution=float(meta.get("resolution", cfg["resolution"])),
            start_col=int(meta["start_col"]),
            start_row=int(meta["start_row"]),
            heading_deg=float(meta["heading_deg"]),
        )

    map_path = resolve_path(map_rel, base)
    if not map_path.is_file():
        raise ArtifactError(
            f"Map file missing for preset '{name}': {map_path}\n"
            f"  git checkout cursor/step5-6-vesc-3d48   # ensures maps/ exist")

    start_col, start_row = (int(x) for x in cfg["start"].split(","))
    return MapPreset(
        name=name,
        map_path=map_path,
        resolution=float(cfg["resolution"]),
        start_col=start_col,
        start_row=start_row,
        heading_deg=float(cfg["heading"]),
    )
