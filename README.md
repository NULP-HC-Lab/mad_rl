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

`scripts/mjswan/` runs ETH's AME-2 G1 teacher
([leggedrobotics/ame2_minimal](https://github.com/leggedrobotics/ame2_minimal),
`modelzoo/g1_gaze_64000env_10h/model_2400.pt`, GPL-3.0) in MuJoCo on the lab course
(12 cm stairs, ramp, brick road) plus wide 20 cm stairs, and builds it into a static browser
app with [mjswan](https://mjswan.readthedocs.io). Needs `uv`, nothing from pixi:

```bash
uv run scripts/mjswan/rollout.py            # 32 jittered trials per station + videos in scripts/mjswan/videos (GPU)
uv run scripts/mjswan/build.py              # build scripts/mjswan/dist and serve on localhost:8080
uv run scripts/mjswan/check.py              # CPU: the built app's policy and scenes, as the browser runs them
uv run scripts/mjswan/check.py --sweep      # ... and under pushes (1-2 m/s) and slippery feet (μ 0.5, 0.3)
uv run scripts/mjswan/rollout.py --push 1.5 --friction 0.5   # the same perturbations in mjlab, with videos
uv run scripts/mjswan/compare.py            # AME-2 against the previous blind policy, same trials (GPU)
uv run scripts/mjswan/smoke.py              # every scene loads and runs in headless Chromium
uv run scripts/mjswan/build.py --publish    # upload to mjswan Cloud (GitHub sign-in)
uv run scripts/mjswan/bake_course.py        # re-bake course.xml after the course changes in mad_assets
uv run scripts/mjswan/bake_course.py --check   # exit 1 if course.xml no longer matches mad_assets
```

The course geometry is baked into `scripts/mjswan/course.xml` from `mad_assets/scenes/default.usd`,
so nothing here needs mad_assets except `bake_course.py`.

The checkpoint is downloaded once (83 MB, pinned commit) into `scripts/mjswan/.cache/`;
`MAD_RL_CHECKPOINT` points elsewhere. The policy is goal-reaching: each scene has a fixed goal
past its obstacle. `task.py` mirrors the authors' Isaac Lab 2.3 cfgs by hand (23-DoF G1 with
welded wrists, their gains, 26 x 15 xyz height map) and transcribes their `AME2GazeModel`;
its output was checked against the original class.

Not the lab's 25 cm stairs: they are 0.61 m wide and the policy walks around them (on wide
stairs it climbs 25 cm in 14 of 16 trials).

CI (`.github/workflows/mjswan.yml`, CPU only) runs on changes under `scripts/mjswan/`:
`build.py`, then `check.py` (ONNX matches `task.actor()`, shuffling any observation term
changes the actions, every scene reaches its goal in at least 3 of 4 seeded trials), then `smoke.py`; a
`lint` job runs the repo's pre-commit hooks on these files. `smoke.py` catches a broken JS runtime or mjswan
upgrade, not gait. `check.py` reads the rays, rest pose and action scale from the built manifest,
so a build that drops one fails there. Ready PRs also get a non-blocking comment with 32-trial
success rates for the PR and its base, plus the PR's `--sweep` table.

The input check exists because the previous checkpoint was trained with its height scan scaled
by 0: shuffling its 55 scan inputs changes its actions by exactly 0. `compare.py` runs in
MuJoCo-Warp, which keeps about 4 contacts per foot-heightfield pair (C MuJoCo up to 50); AME-2
falls on the brick road (the only heightfield) in a few of 32 trials there and in none in `check.py`.

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