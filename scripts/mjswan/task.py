"""Mad-Rl-Rough-v0 rebuilt as an mjlab env, for running the trained policy in MuJoCo / mjswan.

Mirrors `source/mad_rl/.../mad_rl/*_cfg.py`. If an observation, action, gain or the height
scan changes there, change it here too: the policy reads the exact 103-dim vector below.

Terrain is the lab course from `$MAD_ASSETS_PATH/scenes/default.usd` (stairs, ramps, pallets,
brick road), converted from its collision prims.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION, get_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

REPO = Path(__file__).resolve().parents[2]
MAD_ASSETS = Path(os.environ.get("MAD_ASSETS_PATH", REPO.parent / "mad_assets"))
CHECKPOINT = Path(
    os.environ.get(
        "MAD_RL_CHECKPOINT",
        REPO / "2026-03-21_12-27-55__blind/policy.pt",
    )
)

# utils/constants.py: policy joint order (obs and actions).
LEG_JOINTS = [
    f"{side}_{j}_joint"
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    for side in ("left", "right")
]
# MuJoCo's own order (left leg, then right). mjlab ignores `preserve_order` for joint
# actions, so the sim side uses this order everywhere and the exported ONNX permutes.
MODEL_LEG_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
# scene_cfg.py init_state.
DEFAULT_POS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 1.134,
    ".*_shoulder_pitch_joint": 0.261,
    "left_shoulder_roll_joint": 0.261,
    "right_shoulder_roll_joint": -0.261,
}
SPAWN_Z = 0.7945
# G1_upd carries Dex3 hands (palm + 7 finger links, ~0.70 kg) on wrist_yaw (0.085 kg);
# mjlab's G1 has a 0.255 kg rubber hand there. ponytail: lumped onto the wrist link, no hand joints.
HAND_MASS = 0.085 + 0.3728 + 2 * 0.0589 + 2 * 0.0203 + 0.0862 + 0.0589 + 0.0203
# Isaac Lab drives the joints the policy doesn't control (arms, wrists) from the USD at
# kp=40, kd=10 toward the init pose; the waist at kp=57296, i.e. rigid.
ARM_KP, ARM_KD = 40.0, 10.0
ARMATURE = 0.01


def _default(name: str) -> float:
    import re

    return next((v for k, v in DEFAULT_POS.items() if re.fullmatch(k, name)), 0.0)


def robot_spec() -> mujoco.MjSpec:
    spec = get_spec()
    for joint in list(spec.joints):
        # PhysX joint limits are hard; MuJoCo's are soft. The blind policy locks its knees by
        # driving them into the stop at 300 Nm, which hyperextends a default-soft knee ~8 deg
        # and pitches the robot forward. Stiffest limit the 5 ms step allows (timeconst 2*dt).
        joint.solref_limit = [0.01, 1.0]
        joint.solimp_limit = [0.99, 0.999, 0.001, 0.5, 2.0]
        if joint.name.startswith("waist_"):
            spec.delete(joint)
        elif joint.type == mujoco.mjtJoint.mjJNT_HINGE and joint.name not in LEG_JOINTS:
            joint.stiffness = [ARM_KP, 0, 0]
            joint.damping = [ARM_KD, 0, 0]
            joint.springref = _default(joint.name)
            joint.armature = ARMATURE
    for side in ("left", "right"):
        spec.body(f"{side}_wrist_yaw_link").mass = HAND_MASS
    return spec


def _leg_actuator(joints: tuple[str, ...], kp: float, kd: float, effort: float) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=joints, stiffness=kp, damping=kd, effort_limit=effort, armature=ARMATURE
    )


def robot_cfg(pos: tuple[float, float], yaw: float) -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(*pos, SPAWN_Z),
            rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
            joint_pos=dict(DEFAULT_POS),
            joint_vel={".*": 0.0},
        ),
        # Isaac Lab: terrain friction 1.0, multiply-combined.
        collisions=(
            CollisionCfg(
                geom_names_expr=FULL_COLLISION.geom_names_expr,
                contype=1,
                conaffinity=1,
                condim=FULL_COLLISION.condim,
                priority=FULL_COLLISION.priority,
                friction={r"^(left|right)_foot[1-7]_collision$": (1.0,)},
            ),
        ),
        spec_fn=robot_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(
                _leg_actuator((".*_hip_yaw_joint", ".*_hip_roll_joint"), 150.0, 5.0, 300.0),
                _leg_actuator((".*_hip_pitch_joint", ".*_knee_joint"), 200.0, 5.0, 300.0),
                _leg_actuator((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 20.0, 2.0, 20.0),
            ),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


class IsaacGridPatternCfg(GridPatternCfg):
    """Isaac Lab's `GridPatternCfg(ordering="yx")` shifted by the sensor offset.

    mjlab orders its grid y-major; Isaac's "yx" is x-major (x outer, y inner). The
    RayCaster offset (0.3, 0, 0) moves the ray starts only, not `pos_w`, so the scan
    stays `pelvis_z - hit_z - 0.5`.
    """

    def __init__(self) -> None:
        super().__init__(size=(0.4, 1.0), resolution=0.1)

    def generate_rays(self, mj_model, device):
        x = torch.arange(-0.2, 0.2 + 1e-9, 0.1, device=device) + 0.3
        y = torch.arange(-0.5, 0.5 + 1e-9, 0.1, device=device)
        gx, gy = torch.meshgrid(x, y, indexing="ij")
        offsets = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(gx.numel(), device=device)], 1)
        directions = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(offsets).clone()
        return offsets, directions


def course_spec() -> mujoco.MjSpec:
    """Collision geometry of the lab scene as static MuJoCo geoms (also used as visuals)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(MAD_ASSETS / "scenes" / "default.usd"))
    spec = mujoco.MjSpec()
    root = spec.worldbody
    spec.add_material(name="wood", rgba=(0.62, 0.48, 0.33, 1))
    spec.add_material(name="steel", rgba=(0.55, 0.57, 0.6, 1))
    spec.add_material(name="stone", rgba=(0.45, 0.42, 0.4, 1))

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not path.startswith(("/env", "/equipment")):
            continue
        if "/floor/" in path:  # the terrain plane
            continue
        name = path.strip("/").replace("/", "__")
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if prim.IsA(UsdGeom.Cube):
            t = Gf.Transform(xf)
            q = t.GetRotation().GetQuat()
            half = np.array(t.GetScale()) * UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            pos = np.array(t.GetTranslation())
            if "/step_" in path:
                # The USD steps are 2-5 cm slabs floating over open risers; the toe catches
                # under the next slab and the robot trips. The visual stairs are solid to the
                # floor, so fill each step down to z=0.
                top = pos[2] + half[2]
                half[2], pos[2] = top / 2, top / 2
            root.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=half,
                pos=pos,
                quat=[q.GetReal(), *q.GetImaginary()],
                material="wood" if "pallet" in path else "steel",
            )
            continue
        # Meshes: bake the world transform into the points (row-vector convention).
        pts = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        m = np.array(xf)
        pts = pts @ m[:3, :3] + m[3, :3]
        if "brick_road" in path:
            _add_heightfield(spec, name, pts)
        else:  # the ramp: a wedge, its convex hull is exact enough
            spec.add_mesh(name=name, uservert=pts.astype(np.float32).ravel())
            root.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, material="wood")
    return spec


def _add_heightfield(spec: mujoco.MjSpec, name: str, pts: np.ndarray, res: float = 0.02) -> None:
    """Rasterize a dense surface mesh (1.3 M points for the brick road) into an hfield."""
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    ncol, nrow = np.ceil((hi - lo) / res).astype(int) + 1
    ix = np.clip(((pts[:, 0] - lo[0]) / res).astype(int), 0, ncol - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / res).astype(int), 0, nrow - 1)
    z = np.full((nrow, ncol), -np.inf)
    np.maximum.at(z, (iy, ix), pts[:, 2])
    while np.isinf(z).any():  # fill the few empty cells from their neighbours
        pad = np.pad(z, 1, constant_values=-np.inf)
        nb = np.max([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]], axis=0)
        z = np.where(np.isinf(z), nb, z)
    zmax = z.max()
    half = (hi - lo) / 2
    # MuJoCo hfield: row 0 is -y, column 0 is -x; data normalized to [0, 1] * size[2].
    spec.add_hfield(name=name, nrow=nrow, ncol=ncol, size=[half[0], half[1], zmax, 0.05], userdata=(z / zmax).ravel())
    spec.worldbody.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, pos=[*(lo + half), 0], material="stone"
    )


def env_cfg(spawn: tuple[float, float], yaw: float, episode_s: float = 60.0) -> ManagerBasedRlEnvCfg:
    joints = SceneEntityCfg("robot", joint_names=tuple(MODEL_LEG_JOINTS))
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=IsaacGridPatternCfg(),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # terrain + course only; robot collisions are group 3
    )
    # Same terms, same order as observations_cfg.py (noise off: play config).
    actor = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "velocity_commands": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": joints}),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": joints}),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "height_scan": ObservationTermCfg(
            func=mdp.height_scan, params={"sensor_name": "terrain_scan", "offset": 0.5}, clip=(-1.0, 1.0)
        ),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg(spawn, yaw), "course": EntityCfg(spec_fn=course_spec)},
            sensors=(terrain_scan,),
            num_envs=1,
        ),
        observations={"actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=False)},
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot", actuator_names=tuple(MODEL_LEG_JOINTS), scale=0.5, use_default_offset=True,
            )
        },
        # Trained with lin_vel_x in [0, 1] only, y and yaw fixed at 0.
        commands={
            "twist": UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(1e9, 1e9),
                rel_standing_envs=0.0,
                rel_heading_envs=0.0,
                heading_command=False,
                debug_vis=True,
                ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.6, 0.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
            )
        },
        events={
            "reset_base": EventTermCfg(
                func=vel_mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}}
            ),
            "reset_robot_joints": EventTermCfg(
                func=vel_mdp.reset_joints_by_offset,
                mode="reset",
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
        # Isaac Lab ends on torso contact; a fall is the same thing for a viewer.
        terminations={
            "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(func=vel_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.5,
            elevation=-15.0,
            azimuth=math.degrees(yaw) + 135.0,
        ),
        sim=SimulationCfg(
            nconmax=300,
            njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=episode_s,
    )


# One scene per lab station; the policy only walks forward, so each spawn faces its obstacle.
STATIONS = {
    "Lab stairs 12 cm": dict(spawn=(-2.0, 2.3), yaw=math.pi / 2),
    "Lab stairs 25 cm": dict(spawn=(0.0, 2.3), yaw=math.pi / 2),
    "Lab ramp": dict(spawn=(-2.0, -1.8), yaw=-math.pi / 2),
    "Brick road": dict(spawn=(-1.0, 0.0), yaw=0.0),
}


class _Actor(torch.nn.Module):
    """rsl-rl 5.4 MLPModel actor: EmpiricalNormalization (eps=1e-2) + ELU MLP, mean action.

    Takes and returns joint blocks in MODEL_LEG_JOINTS order; the trained net sees LEG_JOINTS.
    Inputs the policy never saw vary (normalizer std 0, e.g. the height scan of a blind policy
    trained with scale 0) are pinned to their training mean.
    """

    def __init__(self, sd: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"])
        self.register_buffer("std", sd["obs_normalizer._std"])
        self.register_buffer("seen", sd["obs_normalizer._std"] > 0)
        n = len([k for k in sd if k.startswith("mlp.") and k.endswith(".weight")])
        layers: list[torch.nn.Module] = []
        for i in range(n):
            w = sd[f"mlp.{2 * i}.weight"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": sd[f"mlp.{2 * i}.bias"]})
            layers += [lin, torch.nn.ELU()] if i < n - 1 else [lin]
        self.mlp = torch.nn.Sequential(*layers)
        to_policy = [MODEL_LEG_JOINTS.index(j) for j in LEG_JOINTS]
        # obs = 12 root/command dims, then joint_pos, joint_vel, last_action (12 each), then the scan.
        obs_idx = list(range(12)) + [12 + 12 * b + i for b in range(3) for i in to_policy] + list(range(48, 103))
        self.register_buffer("obs_idx", torch.tensor(obs_idx))
        self.register_buffer("act_idx", torch.tensor([LEG_JOINTS.index(j) for j in MODEL_LEG_JOINTS]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.where(self.seen, obs[:, self.obs_idx], self.mean)
        return self.mlp((obs - self.mean) / (self.std + 1e-2))[:, self.act_idx]


def actor(checkpoint: Path = CHECKPOINT) -> torch.nn.Module:
    """From an rsl-rl training checkpoint (model_*.pt) or Isaac Lab's exported TorchScript policy.pt."""
    try:
        jit = torch.jit.load(checkpoint, map_location="cpu")
    except RuntimeError:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    else:
        assert float(jit.normalizer.eps) == 1e-2
        sd = {k.replace("actor.", "mlp.").replace("normalizer.", "obs_normalizer."): v for k, v in jit.state_dict().items()}
    return _Actor(sd).eval()


def export_onnx(checkpoint: Path = CHECKPOINT) -> onnx.ModelProto:
    """The actor as one self-contained ONNX graph: obs [B, 103] -> actions [B, 12]."""
    import io

    buf = io.BytesIO()
    torch.onnx.export(
        actor(checkpoint), torch.zeros(1, 103), buf, input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, dynamo=False,
    )
    return onnx.load_from_string(buf.getvalue())
