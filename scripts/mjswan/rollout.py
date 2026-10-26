# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mjswan[examples]==0.11.6", "usd-core==26.8"]
# ///
"""Headless check of the policy in mjlab on every lab station: progress, falls, one video each.

    uv run scripts/mjswan/rollout.py [--seconds 30] [--out scripts/mjswan/videos]
"""

import argparse
from pathlib import Path

import mediapy
import torch
from mjlab.envs import ManagerBasedRlEnv

import task

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "videos")
parser.add_argument("--station", action="append", help="default: all")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = task.actor().to(device)

for name in args.station or task.STATIONS:
    station = task.STATIONS[name]
    env = ManagerBasedRlEnv(task.env_cfg(**station), device=device, render_mode="rgb_array")
    obs, _ = env.reset()
    robot = env.scene["robot"]
    start = robot.data.root_link_pos_w[0].clone()
    frames, falls, top_z = [], 0, 0.0
    with torch.no_grad():
        for step in range(int(args.seconds / env.step_dt)):
            obs, _, terminated, truncated, _ = env.step(policy(obs["actor"]))
            pos = robot.data.root_link_pos_w[0]
            top_z = max(top_z, float(pos[2]) - task.SPAWN_Z)
            falls += int(terminated[0])
            if step % 2 == 0:  # 25 fps
                frames.append(env.render())
    travelled = float(torch.linalg.norm((pos - start)[:2]))
    print(f"{name:18s} travelled {travelled:5.2f} m  max pelvis rise {top_z:+.2f} m  falls {falls}")
    mediapy.write_video(args.out / f"{name.lower().replace(' ', '_')}.mp4", frames, fps=25)
    env.close()
