# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mjswan[examples]==0.11.6"]
# [tool.uv]
# exclude-newer = "2026-10-27T00:00:00Z"  # reproducible resolution; bump deliberately
# ///
"""Headless check of the policy in mjlab on every lab station: falls, goals reached, one video each.

    uv run scripts/mjswan/rollout.py [--trials 32] [--seconds 20] [--out scripts/mjswan/videos]
    uv run scripts/mjswan/rollout.py --push 1.5 --friction 0.5   # perturbed (task.env_cfg)

Every trial but the filmed one (env 0) starts with a jittered pose, velocity and joints.
Exits 1 if a station reaches its goal in under 90 % of trials.
"""

import argparse
from pathlib import Path

import mediapy
import task
import torch
from mjlab.envs import ManagerBasedRlEnv

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--trials", type=int, default=32)
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "videos")
parser.add_argument("--station", action="append", help="default: all")
parser.add_argument("--push", type=float, default=0.0, help="kick speed (m/s) every 3-5 s")
parser.add_argument("--friction", type=float, default=1.0, help="the feet's friction")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = task.actor().to(device)
failed = False

for name in args.station or task.STATIONS:
    station = task.STATIONS[name]
    cfg = task.env_cfg(
        **station,
        episode_s=args.seconds + 1.0,
        num_envs=args.trials,
        jitter=True,
        push=args.push,
        friction=args.friction,
    )
    env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
    obs, _ = env.reset()
    robot = env.scene["robot"]
    # env 0 starts exactly at the station's spawn, for the video.
    pose = robot.data.default_root_state[:1].clone()
    pose[:, :3] += env.scene.env_origins[:1]
    robot.write_root_state_to_sim(pose, env_ids=torch.tensor([0], device=device))
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[:1], robot.data.default_joint_vel[:1], env_ids=torch.tensor([0], device=device)
    )
    obs = env.observation_manager.compute()
    goal = torch.tensor(station["goal"], device=device)
    fell = torch.zeros(args.trials, dtype=torch.bool, device=device)
    rise = torch.zeros(args.trials, device=device)
    frames = []
    with torch.no_grad():
        for step in range(int(args.seconds / env.step_dt)):
            obs, _, terminated, _, _ = env.step(policy(obs["actor"], obs["map"]))
            if step == 1:  # the actor reads the older of the two stacked actions
                assert torch.equal(obs["actor"][:, 55:78], env.action_manager.prev_action)
            fell |= terminated
            rise = torch.maximum(rise, robot.data.root_link_pos_w[:, 2] - task.SPAWN_Z)
            if step % 2 == 0:  # 25 fps
                frames.append(env.render())
    dist = torch.linalg.norm(robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - goal, dim=-1)
    reached = (dist < 0.5) & ~fell
    print(
        f"{name:18s} falls {int(fell.sum())}/{args.trials}  reached goal {int(reached.sum())}/{args.trials}  "
        f"pelvis rise min/max {float(rise.min()):+.2f}/{float(rise.max()):+.2f} m  "
        f"film run: fell={bool(fell[0])} dist={float(dist[0]):.2f} m rise={float(rise[0]):+.2f} m"
    )
    failed |= int(reached.sum()) < 0.9 * args.trials
    mediapy.write_video(args.out / f"{name.lower().replace(' ', '_')}.mp4", frames, fps=25)
    env.close()

raise SystemExit(1 if failed else 0)
