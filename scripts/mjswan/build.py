# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mjswan[examples]==0.11.6"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Build the AME-2 G1 policy on the lab course into a browser app with mjswan.

uv run scripts/mjswan/build.py             # build scripts/mjswan/dist and serve it locally
uv run scripts/mjswan/build.py --no-launch # build only
uv run scripts/mjswan/build.py --publish   # build and upload to mjswan Cloud (GitHub login)
"""

import argparse
import re
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
project = builder.add_project(name="MAD G1 AME-2")
for name, station in task.STATIONS.items():
    scene = project.add_scene_mjlab(name, env_cfg=task.env_cfg(**station))
    # mjlab's G1 is the MuJoCo Menagerie model; detection misses it behind our spec_fn.
    scene.add_attribution("unitree_g1", license="BSD-3-Clause", copyright="Unitree Robotics")
    scene.add_policy(
        name="AME-2 teacher (ETH)",
        policy=policy,
        policy_joint_names=[f"robot/{j}" for j in task.MODEL_JOINTS],
        in_keys=["actor", "map"],
        # mjswan fills the rest pose only from mjlab-exported ONNX metadata; without it the
        # browser offsets actions from zero.
        default_joint_pos=[
            next((v for k, v in task.DEFAULT_POS.items() if re.fullmatch(k, j)), 0.0) for j in task.MODEL_JOINTS
        ],
        clip_actions=None,
        default=True,
    )

app = builder.build(args.out)
if args.publish:
    print(app.publish(title="MAD G1 AME-2: stairs, ramp and brick road", tags=["g1", "locomotion", "stairs", "ame-2"]))
elif not args.no_launch:
    app.launch()
