# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mjswan[examples]==0.11.6"]
# ///
"""AME-2 against the previous, local policy (blind, 12 leg joints, walks straight at 0.6 m/s), in mjlab.

    uv run scripts/mjswan/compare.py [--trials 32] [--seconds 20]   # GPU; needs git and git-lfs

The old policy's `task.py` and checkpoint come from git history at `BLIND_REV` (removed in
8140be0). Both policies get the same course, the same jittered resets, pushes and foot
friction (`task.env_cfg`). The blind policy has no goal input, so the common metric is
crossing the goal line (the spawn-to-goal distance minus 0.5 m, along that line) before falling.

Brick road is a heightfield, and MuJoCo-Warp keeps about 4 contacts per foot-heightfield pair where
C MuJoCo keeps up to 50: AME-2 falls there in a few of 32 trials here and in none in `check.py`.
"""

import argparse
import importlib.util
import os
import subprocess
import time

import task
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv

BLIND_REV = "bd1ed17"  # feat: mjswan policy with stairs
CONDITIONS = {
    "nominal": {},
    "push 1.0 m/s": dict(push=1.0),
    "push 1.5 m/s": dict(push=1.5),
    "foot μ 0.5": dict(friction=0.5),
}


def blind_task():
    """The old `task.py`, its checkpoint from git LFS, on the current course."""
    cache = task.CHECKPOINT.parent
    ckpt = cache / f"blind_{BLIND_REV}.pt"
    if not ckpt.exists():
        pointer = subprocess.run(
            ["git", "show", f"{BLIND_REV}:2026-03-21_12-27-55__blind/policy.pt"], capture_output=True, check=True
        ).stdout
        ckpt.write_bytes(
            subprocess.run(["git", "lfs", "smudge"], input=pointer, capture_output=True, check=True).stdout
        )
    src = cache / f"blind_task_{BLIND_REV}.py"
    src.write_bytes(
        subprocess.run(["git", "show", f"{BLIND_REV}:scripts/mjswan/task.py"], capture_output=True, check=True).stdout
    )
    os.environ["MAD_RL_CHECKPOINT"] = str(ckpt)
    spec = importlib.util.spec_from_file_location("blind_task", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blind_cfg(old, station: dict, num_envs: int, episode_s: float, push: float = 0.0, friction: float = 1.0):
    new = task.env_cfg(**station, episode_s=episode_s, num_envs=num_envs, jitter=True, push=push)
    cfg = old.env_cfg(station["spawn"], station["yaw"], episode_s)
    cfg.scene.num_envs, cfg.scene.env_spacing = num_envs, 0.0
    cfg.scene.entities["course"] = EntityCfg(spec_fn=task.course_spec)
    cfg.scene.entities["robot"].collisions[0].friction = {r"^(left|right)_foot[1-7]_collision$": (friction,)}
    cfg.events = new.events  # the same resets and pushes
    return cfg


def run(cfg, policy, station: dict, seconds: float, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per trial: crossed the goal line, fell first, time to cross (s)."""
    env = ManagerBasedRlEnv(cfg, device=device)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    spawn, goal = torch.tensor(station["spawn"], device=device), torch.tensor(station["goal"], device=device)
    axis = (goal - spawn) / torch.linalg.norm(goal - spawn)
    line = float(torch.linalg.norm(goal - spawn)) - 0.5
    n = cfg.scene.num_envs
    crossed, fell = torch.zeros(n, dtype=torch.bool, device=device), torch.zeros(n, dtype=torch.bool, device=device)
    t_cross = torch.full((n,), float("nan"), device=device)
    with torch.no_grad():
        for step in range(int(seconds / env.step_dt)):
            obs, _, terminated, _, _ = env.step(policy(*(obs[k] for k in obs)))
            fell |= terminated & ~crossed
            now = ((robot.data.root_link_pos_w[:, :2] - spawn) @ axis >= line) & ~crossed & ~fell
            t_cross[now] = (step + 1) * env.step_dt
            crossed |= now
    env.close()
    return crossed.cpu(), fell.cpu(), t_cross.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    old = blind_task()
    policies = {"AME-2 (ETH)": task.actor().to(device), f"blind ({BLIND_REV})": old.actor().to(device)}

    rows = []
    for name, station in task.STATIONS.items():
        for cond, kw in CONDITIONS.items():
            cells = []
            for label, policy in policies.items():
                cfg = (
                    task.env_cfg(**station, episode_s=args.seconds + 1.0, num_envs=args.trials, jitter=True, **kw)
                    if label.startswith("AME")
                    else blind_cfg(old, station, args.trials, args.seconds + 1.0, **kw)
                )
                t0 = time.time()
                crossed, fell, t = run(cfg, policy, station, args.seconds, device)
                med = f"{t[crossed].median():.1f} s" if crossed.any() else "-"
                cells.append(f"{int(crossed.sum())}/{args.trials}, {int(fell.sum())} falls, {med}")
                print(f"  {name} / {cond} / {label}: {cells[-1]} ({time.time() - t0:.0f} s)", flush=True)
            rows.append(f"| {name} | {cond} | " + " | ".join(cells) + " |")

    print(
        f"\n{args.trials} jittered trials of {args.seconds:g} s: crossed the goal line / falls / median time to"
        " cross.\n"
    )
    print("| Scene | Condition | " + " | ".join(policies) + " |\n|---|---|" + "---|" * len(policies))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
