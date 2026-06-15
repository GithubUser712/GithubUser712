"""Shared training profiles for RL steps (bullet / quick / deep)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from raceline.physics import CarParams

MODE_CHOICES = ("bullet_learn", "quick_train", "deep_learn", "deep_learn_xhigh")

_TRACK_ENV_KEYS = (
    "random_spawn",
    "physics_substeps",
    "dt",
    "max_steps",
    "n_beams",
    "max_range_m",
    "fov_deg",
)

_MODE_SETTINGS: dict[str, dict[str, Any]] = {
    "bullet_learn": {
        "label": "bullet_learn (bare minimum, ~10-20 min on sample)",
        "step2_timesteps": 75_000,
        "step3_timesteps": 50_000,
        "n_envs": 1,
        "autosave_every": 10_000,
        "env": {
            "random_spawn": False,
            "physics_substeps": 1,
            "dt": 0.10,
            "max_steps": 1000,
            "use_motor_dynamics": False,
        },
        "ppo": {
            "learning_rate": 8e-4,
            "n_steps": 256,
            "batch_size": 64,
            "gamma": 0.98,
            "gae_lambda": 0.90,
            "ent_coef": 0.02,
            "net_arch": [64, 64],
        },
    },
    "quick_train": {
        "label": "quick_train (~30-45 min target on sample / small tracks)",
        "step2_timesteps": 250_000,
        "step3_timesteps": 200_000,
        "n_envs": 2,
        "autosave_every": 25_000,
        "env": {
            "random_spawn": False,
            "physics_substeps": 2,
        },
        "ppo": {
            "learning_rate": 5e-4,
            "n_steps": 512,
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "net_arch": [128, 128],
        },
    },
    "deep_learn": {
        "label": "deep_learn (full random-spawn physics + larger PPO budget)",
        "step2_timesteps": 1_000_000,
        "step3_timesteps": 600_000,
        "n_envs": 4,
        "autosave_every": 50_000,
        "env": {
            "random_spawn": True,
            "physics_substeps": 4,
        },
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "ent_coef": 0.005,
            "net_arch": [256, 256],
        },
    },
    "deep_learn_xhigh": {
        "label": "deep_learn xhigh (6 envs, 6 substeps, 5000 max steps/run)",
        "step2_timesteps": 2_000_000,
        "step3_timesteps": 1_000_000,
        "n_envs": 6,
        "autosave_every": 75_000,
        "env": {
            "random_spawn": True,
            "physics_substeps": 6,
            "max_steps": 5000,
        },
        "ppo": {
            "learning_rate": 2.5e-4,
            "n_steps": 1024,
            "batch_size": 384,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "ent_coef": 0.005,
            "net_arch": [256, 256],
        },
    },
}


def mode_settings(mode: str) -> dict[str, Any]:
    if mode not in _MODE_SETTINGS:
        known = ", ".join(MODE_CHOICES)
        raise ValueError(f"Unknown mode '{mode}'. Choose one of: {known}")
    return _MODE_SETTINGS[mode]


def resolve_budget(
    mode: str,
    step: int,
    *,
    timesteps: int | None = None,
    n_envs: int | None = None,
) -> tuple[int, int, dict[str, Any]]:
    """Return (timesteps, n_envs, full mode config) for step 2 or 3."""
    cfg = mode_settings(mode)
    if step == 2:
        default_ts = int(cfg["step2_timesteps"])
    elif step == 3:
        default_ts = int(cfg["step3_timesteps"])
    else:
        raise ValueError(f"resolve_budget only supports step 2 or 3 (got {step})")
    ts = default_ts if timesteps is None else timesteps
    ne = int(cfg["n_envs"]) if n_envs is None else n_envs
    return ts, ne, cfg


def env_build_options(
    cfg: dict[str, Any],
    params: CarParams | None = None,
) -> tuple[CarParams | None, dict[str, Any]]:
    """Split mode env config into CarParams overrides and TrackEnv kwargs."""
    raw = dict(cfg["env"])
    motor = raw.pop("use_motor_dynamics", None)
    if motor is not None:
        base = params or CarParams()
        params = replace(base, use_motor_dynamics=motor)
    opts = {k: raw[k] for k in _TRACK_ENV_KEYS if k in raw}
    return params, opts


def ppo_kwargs(cfg: dict[str, Any], *, seed: int, device: str) -> dict[str, Any]:
    """Keyword arguments for ``PPO('MlpPolicy', env, **kwargs)``."""
    ppo = cfg["ppo"]
    return dict(
        seed=seed,
        verbose=0,
        learning_rate=ppo["learning_rate"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        ent_coef=ppo["ent_coef"],
        policy_kwargs=dict(net_arch=ppo["net_arch"]),
        device=device,
    )


def format_mode_summary(mode: str, cfg: dict[str, Any], *,
                        timesteps: int, n_envs: int) -> str:
    env = cfg["env"]
    extras = []
    if "dt" in env:
        extras.append(f"dt={env['dt']}")
    if "max_steps" in env:
        extras.append(f"max_steps={env['max_steps']}")
    if env.get("use_motor_dynamics") is False:
        extras.append("motor_dynamics=off")
    extra_s = (" | " + ", ".join(extras)) if extras else ""
    return (
        f"Mode: {mode} ({cfg['label']}) | timesteps {timesteps:,} | "
        f"n-envs {n_envs} | random_spawn={env['random_spawn']} | "
        f"physics_substeps={env['physics_substeps']}{extra_s}"
    )
