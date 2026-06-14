"""Persistent policy memory — reuse driving skill across tracks.

After each step 2/3 run the trained PPO weights are archived under
``policy_memory/``.  When you upload a new track and run step 2 again,
training warm-starts from the best compatible prior policy instead of
random weights.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from raceline.core.paths import find_project_root

POLICY_FILENAME = "policy.zip"
META_FILENAME = "meta.yaml"


def policy_memory_dir(root: Path | None = None) -> Path:
    base = root or find_project_root()
    d = base / "policy_memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sanitize_label(label: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", label.strip().lower())
    return s[:48] or "track"


def _entry_id(track_label: str, saved_at: datetime) -> str:
    stamp = saved_at.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_sanitize_label(track_label)}"


def scan_entries(root: Path | None = None) -> list[tuple[Path, dict]]:
    """Return (entry_dir, meta) for every saved policy, newest first."""
    mem = policy_memory_dir(root)
    out: list[tuple[Path, dict]] = []
    for child in mem.iterdir():
        if not child.is_dir():
            continue
        meta_path = child / META_FILENAME
        policy_path = child / POLICY_FILENAME
        if not meta_path.is_file() or not policy_path.is_file():
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        meta["_entry_dir"] = str(child)
        meta["_policy_path"] = str(policy_path)
        out.append((child, meta))
    out.sort(key=lambda t: t[1].get("saved_at", ""), reverse=True)
    return out


def register_policy(
    policy_zip: Path,
    *,
    track_label: str,
    artifacts_dir: Path,
    source_step: int,
    policy_kind: str,
    observation_shape: tuple[int, ...],
    track_length_m: float,
    timesteps: int,
    lap_time_s: float | None = None,
    root: Path | None = None,
) -> Path:
    """Archive a trained policy for future cross-track transfer."""
    mem = policy_memory_dir(root)
    saved_at = datetime.now(timezone.utc)
    entry = mem / _entry_id(track_label, saved_at)
    entry.mkdir(parents=True, exist_ok=False)
    shutil.copy2(policy_zip, entry / POLICY_FILENAME)
    meta = {
        "track_label": track_label,
        "artifacts_dir": str(Path(artifacts_dir).resolve()),
        "source_step": int(source_step),
        "policy_kind": policy_kind,
        "observation_shape": list(observation_shape),
        "track_length_m": round(float(track_length_m), 3),
        "timesteps": int(timesteps),
        "lap_time_s": round(lap_time_s, 3) if lap_time_s is not None else None,
        "saved_at": saved_at.isoformat(),
    }
    with open(entry / META_FILENAME, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    return entry


def _shape_match(saved: list, wanted: tuple[int, ...]) -> bool:
    try:
        return tuple(int(x) for x in saved) == tuple(int(x) for x in wanted)
    except (TypeError, ValueError):
        return False


def _kind_rank(kind: str) -> int:
    return 2 if kind == "tuned" else 1 if kind == "base" else 0


def find_transfer_policy(
    observation_shape: tuple[int, ...],
    *,
    exclude_artifacts: Path | str | None = None,
    root: Path | None = None,
) -> tuple[Path, dict] | None:
    """Pick the best prior policy compatible with the current simulator."""
    exclude = str(Path(exclude_artifacts).resolve()) if exclude_artifacts else None
    candidates: list[tuple[int, int, str, Path, dict]] = []
    for entry_dir, meta in scan_entries(root):
        if not _shape_match(meta.get("observation_shape", []), observation_shape):
            continue
        if exclude and meta.get("artifacts_dir") == exclude:
            continue
        policy_path = entry_dir / POLICY_FILENAME
        if not policy_path.is_file():
            continue
        candidates.append((
            _kind_rank(str(meta.get("policy_kind", ""))),
            int(meta.get("timesteps", 0)),
            str(meta.get("saved_at", "")),
            policy_path,
            meta,
        ))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    _, _, _, policy_path, meta = candidates[0]
    return policy_path, meta


def load_warm_start_model(
    model_cls,
    venv,
    device: str,
    observation_shape: tuple[int, ...],
    *,
    local_paths: list[Path],
    artifacts_dir: Path,
    fresh: bool = False,
    resume: bool = False,
    root: Path | None = None,
):
    """Load PPO from local checkpoint, resume, or cross-track memory.

    Returns (model, source_description) or (None, None) to train from scratch.
    """
    if fresh:
        return None, None

    for path in local_paths:
        if not path.is_file():
            continue
        try:
            model = model_cls.load(path, env=venv, device=device)
            label = "resuming" if resume else "local checkpoint"
            return model, f"{label}: {path.name}"
        except Exception:
            continue

    if resume:
        return None, None

    found = find_transfer_policy(
        observation_shape, exclude_artifacts=artifacts_dir, root=root)
    if found is None:
        return None, None
    policy_path, meta = found
    try:
        model = model_cls.load(policy_path, env=venv, device=device)
    except Exception:
        return None, None
    track = meta.get("track_label", "unknown")
    kind = meta.get("policy_kind", "policy")
    steps = meta.get("timesteps", "?")
    return model, (
        f"cross-track transfer from {track} ({kind}, {steps:,} train steps) "
        f"via {policy_path.parent.name}"
    )


def track_label_from_artifacts(artifacts_dir: Path) -> str:
    """Derive a human label from track_meta or the artifacts folder name."""
    meta_path = artifacts_dir / "track_meta.yaml"
    if meta_path.is_file():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}
        if meta.get("track_label"):
            return str(meta["track_label"])
        src = meta.get("source_image", "")
        if src:
            return Path(src).stem
    return Path(artifacts_dir).name
