# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mjswan[examples]==0.11.6"]
# [tool.uv]
# exclude-newer = "2026-10-27T00:00:00Z"  # reproducible resolution; bump deliberately
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""CPU check of the built browser app, run the way the browser runs it (CI uses this).

    uv run scripts/mjswan/build.py --no-launch
    uv run scripts/mjswan/check.py [--trials 4] [--dist scripts/mjswan/dist]

1. The policy ONNX in dist matches `task.actor()`, and reads every observation term: shuffling any
   one of them (each actor term, the map) changes the actions. The old blind checkpoint, trained
   with its height scan scaled by 0, fails this: its scan term changes nothing.
2. Every scene runs in C MuJoCo from its scene.mjz with the per-term observation graphs. Rays,
   rest pose, action scale and the fall angle come from the manifest, as in the browser, so a
   build that drops any of them fails here. Trial 0 starts at the spawn, the rest are jittered
   (seeded, ranges from `task.env_cfg(jitter=True)`).
3. With --sweep, the same trials again under pushes and slippery feet (`task.env_cfg(push=,
   friction=)`). Informational: only the unperturbed trials are gated.

Prints markdown tables; exits 1 if a scene reaches its goal in under 75 % of unperturbed trials.
"""

import argparse
import json
import math
import multiprocessing
import os
import re
import zipfile
from functools import cache
from pathlib import Path

# No rendering here. Else `import task` (mjlab) sets MUJOCO_GL=egl, the spawned workers inherit
# it and die importing mujoco on machines without EGL (CI runners), and the pool respawns forever.
os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np
import onnxruntime as ort
import task
import torch

MIN_RATE = 0.75
SWEEP = {  # 2 m/s already topples a quarter of the trials
    **{f"push {p} m/s": dict(push=p) for p in (1.0, 1.5, 2.0)},
    **{f"foot μ {f}": dict(friction=f) for f in (0.5, 0.3)},
}


@cache
def _session(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1  # deterministic, and faster for a batch of 1
    return ort.InferenceSession(path, opts)


@cache  # one model per friction: a worker's trials share it
def _model(mjz: str, friction: float | None = None) -> mujoco.MjModel:
    with zipfile.ZipFile(mjz) as z:
        files = {n: z.read(n) for n in z.namelist()}
    xml = next(n for n in files if n.endswith(".xml"))
    m = mujoco.MjModel.from_xml_string(files[xml].decode(), files)
    if friction is not None:
        feet = [g for g in range(m.ngeom) if re.fullmatch(r"robot/(left|right)_foot\d_collision", m.geom(g).name)]
        assert len(feet) == 14
        m.geom_friction[feet, 0] = friction
    return m


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def trial(
    scene_dir: str, scene: dict, seconds: float, seed: int, push: float = 0.0, friction: float | None = None
) -> tuple[bool, float]:
    """One episode; returns (fell, distance from the goal at the end)."""
    m = _model(f"{scene_dir}/{scene['scene']}", friction)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    pol = scene["policies"][0]
    mdp = next(x for x in scene["mdps"] if x["id"] == pol["mdp"])
    station = task.STATIONS[scene["name"]]
    root = m.jnt_bodyid[list(m.jnt_type).index(mujoco.mjtJoint.mjJNT_FREE)]
    joints = [
        j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and m.joint(j).name.startswith("robot/")
    ]
    qadr, vadr = m.jnt_qposadr[joints], m.jnt_dofadr[joints]

    events = task.env_cfg(**station, jitter=True, push=push).events
    rng = np.random.default_rng(seed)
    if seed:  # the jittered resets of rollout.py
        pose, vel = events["reset_base"].params["pose_range"], events["reset_base"].params["velocity_range"]
        d.qpos[:2] += rng.uniform(*zip(pose["x"], pose["y"]))
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, _yaw_quat(rng.uniform(*pose["yaw"])), d.qpos[3:7])
        d.qpos[3:7] = quat
        d.qvel[:6] += rng.uniform(*zip(*(vel[k] for k in ("x", "y", "z", "roll", "pitch", "yaw"))))
        d.qpos[qadr] += rng.uniform(*events["reset_robot_joints"].params["position_range"], len(qadr))
    mujoco.mj_forward(m, d)

    actions = mdp["actions"]["joint_pos"]
    act_ids = [m.actuator(n).id for n in actions["actuator_names"]]
    default = np.array(pol["default_joint_pos"])  # KeyError: mjswan dropped the rest pose
    assert len(act_ids) == len(default)
    lanes = mdp["terminations"]["__fused__"]["lanes"]
    limit = next(x["params"]["limit_angle"] for x in lanes if x["name"] == "fell_over")
    last_action = np.zeros(len(act_ids), np.float32)

    def rays(desc: dict) -> dict:
        body = m.body(desc["frames"][0]["name"]).id
        R, p = d.xmat[body].reshape(3, 3), d.xpos[body].copy()
        yaw = math.atan2(R[1, 0], R[0, 0])
        Ry = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        mask = np.isin(np.arange(mujoco.mjNGROUP), desc["include_geom_groups"]).astype(np.uint8)
        starts = p + np.array(desc["local_offsets"]) @ Ry.T
        dirs = np.array(desc["local_directions"]) @ Ry.T
        dist, hit, gid = np.full(len(starts), -1.0), starts.copy(), np.zeros(1, np.int32)
        exclude = body if desc["exclude_parent_body"] else -1
        for i, (s, v) in enumerate(zip(starts, dirs)):
            t = mujoco.mj_ray(m, d, s, v, mask, 1, exclude, gid)
            if 0 <= t <= desc["max_distance"]:
                dist[i], hit[i] = t, s + t * v
        return {"distances": dist[None], "frame_pos_w": p[None, None], "hit_pos_w": hit[None]}

    def slot(sl: dict, group: dict) -> np.ndarray:
        if "action" in sl:
            return last_action[None]
        if "sensor" in sl:  # TypeError/KeyError: mjswan sent no raycast descriptor for this group
            return rays(group["sensors"][sl["sensor"]])[sl["field"]]
        if sl["entity"] != "robot":  # the free-walk flag: its bodies' positions
            assert sl["field"] == "body_link_pos_w", sl
            return d.xpos[[b for b in range(m.nbody) if m.body(b).name.startswith(f"{sl['entity']}/")]]
        R, p, w_b = d.xmat[root].reshape(3, 3), d.xpos[root], d.qvel[3:6]
        v_com_w = d.qvel[:3] + np.cross(R @ w_b, d.xipos[root] - p)
        return {
            "root_com_lin_vel_b": lambda: R.T @ v_com_w,
            "root_link_ang_vel_b": lambda: w_b,
            "projected_gravity_b": lambda: R.T @ np.array([0.0, 0.0, -1.0]),
            "joint_pos": lambda: d.qpos[qadr],
            "joint_vel": lambda: d.qvel[vadr],
            "heading_w": lambda: np.array(math.atan2(R[1, 0], R[0, 0])),
            "root_link_pos_w": lambda: p,
        }[sl["field"]]()

    history: dict[str, list] = {}

    def observe(group) -> np.ndarray:
        terms = (
            [{"name": "fused", "onnx": group["fused"], "input_slots": group["input_slots"]}]
            if isinstance(group, dict)
            else group
        )
        parts = []
        for t in terms:
            feeds = {
                sl["input"]: np.asarray(slot(sl, group), np.float32).reshape(sl["shape"]) for sl in t["input_slots"]
            }
            v = _session(f"{scene_dir}/{t['onnx']}").run(None, feeds)[0].reshape(-1)
            if t.get("history_length", 0) > 1:
                h = history.setdefault(t["name"], [v] * t["history_length"])
                h.pop(0)
                h.append(v)
                v = np.concatenate(h)
            parts.append(v)
        return np.concatenate(parts)[None].astype(np.float32)

    policy = _session(f"{scene_dir}/{pol['onnx']}")
    goal = np.array(station["goal"])
    if kick := events.get("push"):  # mjlab's interval event: after the physics steps, per-env timer
        kick_vel, kick_in = kick.params["velocity_range"], rng.uniform(*kick.interval_range_s)
    for _ in range(int(seconds / scene["control_dt"])):
        obs = {k: observe(mdp["observations"][k]) for k in pol["in_keys"]}
        last_action = policy.run(None, {i.name: obs[k] for i, k in zip(policy.get_inputs(), pol["in_keys"])})[0][0]
        d.ctrl[act_ids] = actions["scale"] * last_action + default
        for _ in range(round(scene["control_dt"] / m.opt.timestep)):
            mujoco.mj_step(m, d)
        if math.acos(np.clip(d.xmat[root][8], -1, 1)) > limit:  # bad_orientation
            return True, float(np.linalg.norm(d.xpos[root][:2] - goal))
        if kick and (kick_in := kick_in - scene["control_dt"]) < 1e-6:
            d.qvel[:2] += rng.uniform(*zip(kick_vel["x"], kick_vel["y"]))  # free joint: world-frame root velocity
            kick_in = rng.uniform(*kick.interval_range_s)
    return False, float(np.linalg.norm(d.xpos[root][:2] - goal))


def check_parity(dist: Path) -> tuple[float, dict[str, float]]:
    """Max |ONNX - task.actor()|, and per observation term the max action change when it alone is shuffled."""
    project = json.loads((dist / "manifest.json").read_text())["projects"][0]
    scene = project["scenes"][0]
    terms = next(x for x in scene["mdps"] if x["id"] == scene["policies"][0]["mdp"])["observations"]["actor"]
    onnx_path = dist / project["id"] / scene["id"] / scene["policies"][0]["onnx"]
    obs, height_map = torch.randn(64, task.OBS_DIM), torch.randn(64, task.MAP_DIM) * 0.5
    with torch.no_grad():
        want = task.actor()(obs, height_map).numpy()
    sess, feeds = _session(str(onnx_path)), {"obs": obs.numpy(), "map": height_map.numpy()}
    got = sess.run(None, feeds)[0]
    rng = np.random.default_rng(0)  # shuffles across the batch: same distribution, other samples

    def change(name: str, cols: slice) -> float:
        x = feeds[name].copy()
        x[:, cols] = rng.permutation(x[:, cols])
        return float(np.abs(sess.run(None, {**feeds, name: x})[0] - got).max())

    used, start = {}, 0
    for t in terms:  # the actor group, concatenated in this order
        n = t["size"] * t.get("history_length", 1)
        used[t["name"]], start = change("obs", slice(start, start + n)), start + n
    assert start == task.OBS_DIM, start
    used["map"] = change("map", slice(None))
    return float(np.abs(got - want).max()), used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path(__file__).parent / "dist")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--sweep", action="store_true", help="also run the perturbed trials")
    parser.add_argument("--sweep-trials", type=int, help="trials per scene and perturbation (default: --trials)")
    args = parser.parse_args()

    diff, used = check_parity(args.dist)
    print(f"policy ONNX vs task.actor(): max abs diff {diff:.1e}")
    print("max action change, one term shuffled: " + ", ".join(f"{k} {v:.2f}" for k, v in used.items()) + "\n")
    failed = diff > 1e-4 or min(used.values()) < 1e-3  # a term the policy ignores
    project = json.loads((args.dist / "manifest.json").read_text())["projects"][0]
    scenes = project["scenes"]
    conditions = [{}, *SWEEP.values()] if args.sweep else [{}]
    n_sweep = args.sweep_trials or args.trials
    keys, jobs = [], []  # keys[k] = (condition, scene) of jobs[k]
    for c, cond in enumerate(conditions):
        for i, s in enumerate(scenes):
            for seed in range(n_sweep if c else args.trials):
                keys.append((c, i))
                path = str(args.dist / project["id"] / s["id"])
                jobs.append((path, s, args.seconds, seed, cond.get("push", 0.0), cond.get("friction")))
    with multiprocessing.get_context("spawn").Pool(len(os.sched_getaffinity(0))) as pool:
        results = pool.starmap(trial, jobs, chunksize=1)

    def score(c: int, i: int) -> tuple[int, int]:  # (reached, falls) of condition c, scene i
        res = [r for k, r in zip(keys, results) if k == (c, i)]
        return sum(not fell and dist < 0.5 for fell, dist in res), sum(fell for fell, _ in res)

    print(f"{args.trials} trials of {args.seconds:g} s per scene, trial 0 at the spawn:\n")
    print("| Scene | Reached goal | Falls |\n|---|---|---|")
    for i, s in enumerate(scenes):
        reached, falls = score(0, i)
        print(f"| {s['name']} | {reached}/{args.trials} | {falls} |")
        failed |= reached < MIN_RATE * args.trials
    if args.sweep:
        print("\nRobustness, not gated: reached goal / falls. Pushes kick the base every 3-5 s by up to the")
        print("given speed in x and y; foot μ replaces the feet's friction (nominal 1.0). Trials per scene:")
        print(f"{args.trials} nominal, {n_sweep} per perturbation.\n")
        print("| Scene | nominal | " + " | ".join(SWEEP) + " |\n|---|---|" + "---|" * len(SWEEP))
        for i, s in enumerate(scenes):
            print(f"| {s['name']} | " + " | ".join("%d / %d" % score(c, i) for c in range(len(conditions))) + " |")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
