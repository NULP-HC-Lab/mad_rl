# MAD RL

Reinforcement learning pipelines for the Unitree G1 in Isaac Lab 3.0 for MAD clug research activities.

## Prerequisites

- Make sure you have the `mad_assets` repository cloned and available at the same
directory level as this repository (`../mad_assets`). Override with 
`MAD_ASSETS_PATH` in a `.env` file if it lives elsewhere.
- [Pixi](https://pixi.sh) (if you already have it, don't forget to keep it up updated):
  ```bash
  curl -fsSL https://pixi.sh/install.sh | sh
  ```
- An NVIDIA GPU with a driver new enough for CUDA 13 (torch 2.11).

## Quick start

Isaac Lab is installed from the published `isaaclab==3.0.0` wheel with Isaac Sim 6.2
(~26GB):

```bash
pixi install
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

Every `pixi run` task above is Isaac Lab's own `isaaclab` CLI, so every common
flag (`--num_envs`, `--max_iterations`, `physics=` presets, hydra overrides) works.

### Updating Isaac Lab
Bump the `isaaclab` pin in `pixi.toml` and replace the
`[pypi-options.dependency-overrides]` block with that release's
[`tools/wheel_builder/uv-overrides.txt`](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/tools/wheel_builder/uv-overrides.txt)
(the `--overrides` file upstream's `isaaclab[isaacsim]` install uses; pixi has no
`--overrides` flag), then run `pixi install`.

## Browser demo (mjswan)

`scripts/mjswan/` runs the `Mad-Rl-Rough` policy in MuJoCo on the lab course from
`mad_assets/scenes/default.usd` (12 cm and 25 cm stairs, ramp, brick road) and builds it into a
static browser app with [mjswan](https://mjswan.readthedocs.io). Needs `uv`, nothing from pixi:

```bash
uv run scripts/mjswan/rollout.py            # headless check + videos in scripts/mjswan/videos
uv run scripts/mjswan/build.py              # build scripts/mjswan/dist and serve on localhost:8080
uv run scripts/mjswan/build.py --publish    # upload to mjswan Cloud (GitHub sign-in)
```

`MAD_RL_CHECKPOINT` picks the policy: an rsl-rl `model_*.pt` or an exported `policy.pt`
(default: the blind policy in `2026-03-21_12-27-55__blind/`). `task.py` mirrors the Isaac Lab task cfgs by hand:
change it when observations, actions or gains change there.

## Docker

`Dockerfile` / `compose.yaml` predate Isaac Lab 3.0 and are not updated yet.

## Code formatting

Pre-commit formats the code:

```bash
pip install pre-commit
pre-commit install
```

## References
- [IsaacLab 3.0 migration guide](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/migrating.html)