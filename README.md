# MAD RL

Reinforcement learning pipelines for the Unitree G1 in Isaac Lab 3.0 for MAD clug research activities.

## Prerequisites

- Make sure you have the `mad_assets` repository cloned and available at the same
directory level as this repository (`../mad_assets`). Override with 
`MAD_ASSETS_PATH` in a `.env` file if it lives elsewhere.
- [Pixi](https://pixi.sh) (if you already have it, don't forget to keep it up with updates):
  ```bash
  curl -fsSL https://pixi.sh/install.sh | sh
  ```
- An NVIDIA GPU with a driver new enough for CUDA 13 (torch 2.11).

## Quick start

Isaac Lab 3.0 is not published on PyPI, so it is vendored as a git submodule at
`external/IsaacLab` and installed as an editable path dependency (~26GB):

```bash
pixi run setup
```

Then:

```bash
pixi run list_envs                                   # list the Mad-Rl-* tasks
pixi run zero_agent --task Mad-Rl-Rough-v0 --num_envs 16  # smoke-test the env, no policy
pixi run train --task Mad-Rl-Rough-v0                # train (headless by default)
pixi run play  --task Mad-Rl-Rough-Play-v0           # play a checkpoint
pixi run tensorboard                                 # watch the curves
```

Add `--viz kit` to any of those to open an Omniverse viewport; without it the simulator runs
headless by default. Other options are `newton`, `rerun`, `viser`. They can even be 
specified via comma `--viz kit,newton`. 

Every `pixi run` task above is Isaac Lab's own `isaaclab` CLI, so every upstream
flag (`--num_envs`, `--max_iterations`, `physics=` presets, hydra overrides)
works. Two pieces of wiring make that possible without wrapper scripts: the
`isaaclab.tasks` entry point in `source/mad_rl/setup.py` registers the `Mad-Rl-*`
gym ids, and `default_agent` in each `gym.register` picks the RL backend, so
`--rl_library` only has to be passed when overriding it.

### Updating Isaac Lab

The submodule tracks upstream's `develop` branch:

```bash
git submodule update --remote external/IsaacLab
pixi install          # re-solve against the new dependency contract
```

`pixi.toml` restates IsaacLab's `[tool.uv] override-dependencies` and resolver
settings by hand, because pixi does not read a path dependency's `[tool.uv]`
table. Re-check that block against `external/IsaacLab/pyproject.toml` after a
bump.

## Docker

`Dockerfile` / `compose.yaml` predate the submodule and do **not** build: the
cached `pixi install --frozen` layer bind-mounts only `pixi.lock` and
`pixi.toml`, which cannot resolve `path = "external/IsaacLab"`. Local pixi is the
supported path for now.

## Code formatting

Pre-commit formats the code:

```bash
pip install pre-commit
pre-commit install
```

## References
- [IsaacLab 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html)