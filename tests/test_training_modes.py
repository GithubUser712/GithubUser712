from raceline.rl.training_modes import (
    MODE_CHOICES,
    env_build_options,
    format_mode_summary,
    resolve_budget,
)


def test_mode_choices():
    assert "bullet_learn" in MODE_CHOICES
    assert "quick_train" in MODE_CHOICES
    assert "deep_learn_xhigh" in MODE_CHOICES


def test_bullet_learn_budgets():
    ts2, ne2, cfg = resolve_budget("bullet_learn", 2)
    ts3, ne3, _ = resolve_budget("bullet_learn", 3)
    assert ts2 == 75_000
    assert ts3 == 50_000
    assert ne2 == 1
    assert cfg["env"]["physics_substeps"] == 1
    assert cfg["env"]["use_motor_dynamics"] is False
    assert cfg["ppo"]["net_arch"] == [64, 64]


def test_env_build_options_motor_off():
    _, _, cfg = resolve_budget("bullet_learn", 2)
    params, opts = env_build_options(cfg)
    assert params is not None
    assert params.use_motor_dynamics is False
    assert opts["dt"] == 0.10
    assert opts["max_steps"] == 1000



def test_quick_train_budgets():
    ts2, ne2, cfg = resolve_budget("quick_train", 2)
    ts3, ne3, _ = resolve_budget("quick_train", 3)
    assert ts2 == 250_000
    assert ts3 == 200_000
    assert ne2 == 2
    assert ne3 == 2
    assert cfg["env"]["random_spawn"] is False
    assert cfg["env"]["physics_substeps"] == 2


def test_deep_learn_budgets():
    ts2, ne2, cfg = resolve_budget("deep_learn", 2)
    ts3, ne3, _ = resolve_budget("deep_learn", 3)
    assert ts2 == 1_000_000
    assert ts3 == 600_000
    assert ne2 == 4
    assert cfg["env"]["random_spawn"] is True
    assert "Mode: deep_learn" in format_mode_summary(
        "deep_learn", cfg, timesteps=ts2, n_envs=ne2)


def test_deep_learn_xhigh_budgets():
    ts2, ne2, cfg = resolve_budget("deep_learn_xhigh", 2)
    ts3, ne3, _ = resolve_budget("deep_learn_xhigh", 3)
    assert ts2 == 2_000_000
    assert ts3 == 1_000_000
    assert ne2 == 6
    assert ne3 == 6
    assert cfg["env"]["random_spawn"] is True
    assert cfg["env"]["physics_substeps"] == 6
    assert cfg["env"]["max_steps"] == 5000
    assert "max_steps=5000" in format_mode_summary(
        "deep_learn_xhigh", cfg, timesteps=ts2, n_envs=ne2)


def test_overrides():
    ts, ne, _ = resolve_budget("quick_train", 2, timesteps=99, n_envs=1)
    assert ts == 99
    assert ne == 1
