# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mjswan[examples]==0.11.6", "usd-core==26.8"]
# ///
"""Build the Mad-Rl-Rough policy on the lab course into a browser app with mjswan.

    uv run scripts/mjswan/build.py             # build scripts/mjswan/dist and serve it locally
    uv run scripts/mjswan/build.py --no-launch # build only
    uv run scripts/mjswan/build.py --publish   # build and upload to mjswan Cloud (GitHub login)
"""

import argparse
from pathlib import Path

import mjswan

import task

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "dist")
parser.add_argument("--no-launch", action="store_true")
parser.add_argument("--publish", action="store_true")
args = parser.parse_args()

policy = task.export_onnx()
builder = mjswan.Builder()
project = builder.add_project(name="MAD G1 stairs")
for name, station in task.STATIONS.items():
    scene = project.add_scene_mjlab(name, env_cfg=task.env_cfg(**station))
    # mjlab's G1 is the MuJoCo Menagerie model; detection misses it behind our spec_fn.
    scene.add_attribution("unitree_g1", license="BSD-3-Clause", copyright="Unitree Robotics")
    scene.add_policy(
        name="Rough walker",
        policy=policy,
        policy_joint_names=[f"robot/{j}" for j in task.MODEL_LEG_JOINTS],
        clip_actions=None,
        default=True,
    )

app = builder.build(args.out)
if args.publish:
    print(app.publish(title="MAD G1: stairs, ramp and brick road", tags=["g1", "locomotion", "stairs"]))
elif not args.no_launch:
    app.launch()
