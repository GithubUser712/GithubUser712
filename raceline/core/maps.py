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
    spacing: float
    start_col: int
    start_row: int
    heading_deg: float


def _load_presets_yaml(root: Path) -> dict:
    path = root / "config" / "map_presets.yaml"
    if not path.is_file():
        raise ArtifactError(f"Missing preset file: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_meta(root: Path, meta_rel: str) -> dict:
    meta_path = resolve_path(meta_rel, root)
    if not meta_path.is_file():
        return {}
    with open(meta_path) as f:
        return yaml.safe_load(f) or {}


def _ensure_map(root: Path, name: str, script: Path, args: list[str]) -> None:
    cfg = _load_presets_yaml(root)[name]
    map_path = resolve_path(cfg["map"], root)
    if map_path.is_file():
        return
    if not script.is_file():
        raise ArtifactError(f"Map generator missing: {script}")
    print(f"Generating {map_path.name} (first run) ...")
    subprocess.run([sys.executable, str(script), *args], cwd=root, check=True)


def ensure_sample_map(root: Path) -> Path:
    out = root / "maps" / "sample_track.png"
    _ensure_map(root, "sample", root / "tools" / "make_sample_map.py",
                ["--resolution", "0.01"])
    return out


def ensure_f1_map(root: Path, name: str) -> Path:
    cfg = _load_presets_yaml(root)[name]
    out = resolve_path(cfg["map"], root)
    _ensure_map(root, name, root / "tools" / "make_f1_tracks.py", [name])
    return out


def load_preset(name: str, root: Path | None = None) -> MapPreset:
    """Load a named preset; auto-generates map + meta when missing."""
    base = root or find_project_root()
    presets = _load_presets_yaml(base)
    if name not in presets:
        known = ", ".join(sorted(presets))
        raise ArtifactError(
            f"Unknown preset '{name}'.  Choose one of: {known}")

    cfg = presets[name]
    map_rel = cfg["map"]
    meta_rel = cfg.get("meta", "")

    if name == "sample":
        ensure_sample_map(base)
    elif name in ("monaco", "spa", "nurburgring"):
        ensure_f1_map(base, name)

    map_path = resolve_path(map_rel, base)
    if not map_path.is_file():
        raise ArtifactError(
            f"Map file missing for preset '{name}': {map_path}\n"
            f"  Run: python tools/make_f1_tracks.py {name}" if name != "sample"
            else "  Run: python tools/make_sample_map.py")

    meta = _load_meta(base, meta_rel) if meta_rel else {}
    if not meta:
        raise ArtifactError(
            f"Missing meta file for preset '{name}': {meta_rel}\n"
            f"  Regenerate maps at 0.01 m/px:\n"
            f"    python tools/make_sample_map.py --resolution 0.01\n"
            f"    python tools/make_f1_tracks.py")

    return MapPreset(
        name=name,
        map_path=map_path,
        resolution=float(meta.get("resolution", cfg["resolution"])),
        spacing=float(cfg.get("spacing", meta.get("resolution", 0.01))),
        start_col=int(meta["start_col"]),
        start_row=int(meta["start_row"]),
        heading_deg=float(meta["heading_deg"]),
    )
