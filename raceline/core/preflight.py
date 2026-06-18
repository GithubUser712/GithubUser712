"""Preflight checks run before every pipeline step — fail fast with clear fixes."""

from __future__ import annotations

from pathlib import Path

from .exceptions import ArtifactError, PreflightError

# Files each step requires (in addition to prior-step outputs).
_STEP_REQUIRES: dict[int, tuple[str, ...]] = {
    1: (),
    2: ("map.yaml", "map.pgm", "centerline.csv", "track_meta.yaml"),
    3: ("rl_policy.zip",),
    4: ("car_params.yaml",),
    5: ("raceline_final.csv",),
    6: (),
}

# At least one of these must exist for step 4 raceline input.
_RACELINE_CANDIDATES = ("raceline_tuned.csv", "raceline.csv")


def check_artifacts_dir(path: str | Path) -> Path:
    d = Path(path).expanduser().resolve()
    if not d.is_dir():
        raise ArtifactError(
            f"Artifacts directory does not exist: {d}\n"
            "Run step 1 first or pass --artifacts to an existing directory.")
    return d


def _missing(d: Path, names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not (d / n).is_file()]


def check_step_prerequisites(step: int, artifacts_dir: str | Path) -> Path:
    """Validate that `artifacts_dir` contains everything `step` needs."""
    if step not in _STEP_REQUIRES:
        raise ValueError(f"unknown pipeline step: {step}")

    d = check_artifacts_dir(artifacts_dir)
    missing = _missing(d, _STEP_REQUIRES[step])

    if step == 4:
        if not any((d / n).is_file() for n in _RACELINE_CANDIDATES):
            missing.append("raceline_tuned.csv or raceline.csv")

    if missing:
        hint = _step_hint(step)
        raise PreflightError(
            f"Step {step} cannot start — missing in {d}:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + (f"\n\n{hint}" if hint else ""))
    return d


def _step_hint(step: int) -> str:
    hints = {
        2: "Run:  python -m pipeline.step1",
        3: "Run:  python -m pipeline.step2",
        4: "Run:  python -m pipeline.step3",
        5: "Run:  python -m pipeline.step4",
    }
    return hints.get(step, "")
