"""AME-2 G1 teacher (leggedrobotics/ame2_minimal) rebuilt as an mjlab env, for MuJoCo / mjswan.

The policy is the authors' `modelzoo/g1_gaze_64000env_10h/model_2400.pt`: a goal-reaching
23-DoF G1 (wrists welded) that reads proprioception, a goal pose and a 26 x 15 xyz height map.
Everything below mirrors their Isaac Lab 2.3 cfgs at commit `AME2_SHA`
(`ame2/ame2/tasks/ame2_env_cfg.py`, `assets/unitree.py`, `rsl_rl/models/AME2_models.py`).

Terrain is the lab course (stairs, ramps, pallets, brick road) baked into `course.xml`, plus
wide stairs.
"""

from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path

import mujoco
import onnx
import torch
import torch.nn.functional as F
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig
from torch import nn

AME2_SHA = "8beb9a688d67882828019583f7d5a78e3517f233"
AME2_URL = (
    f"https://github.com/leggedrobotics/ame2_minimal/raw/{AME2_SHA}/modelzoo/g1_gaze_64000env_10h/model_2400.pt"
)
CHECKPOINT = Path(
    os.environ.get("MAD_RL_CHECKPOINT", Path(__file__).parent / ".cache" / "ame2_g1_gaze_model_2400.pt")
)

# Isaac Lab's (PhysX breadth-first) joint order: the policy's obs and actions. Taken from
# `_switch_joints_lr_g1` in ame2/tasks/mdp/symmetry.py.
POLICY_JOINTS = [
    f"{p}_joint"
    for p in (
        "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll", "waist_roll",
        "left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee", "right_knee", "left_shoulder_pitch",
        "right_shoulder_pitch", "left_ankle_pitch", "right_ankle_pitch", "left_shoulder_roll",
        "right_shoulder_roll", "left_ankle_roll", "right_ankle_roll", "left_shoulder_yaw",
        "right_shoulder_yaw", "left_elbow", "right_elbow",
    )
]
# MuJoCo's own order. mjlab ignores `preserve_order` for joint actions, so the sim side uses
# this order everywhere and the exported ONNX permutes.
MODEL_JOINTS = [
    *(f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")),
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *(f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")),
]
# assets/unitree.py init_state.
DEFAULT_POS = {
    ".*_hip_pitch_joint": -0.30,
    ".*_knee_joint": 0.62,
    ".*_ankle_pitch_joint": -0.33,
    ".*_elbow_joint": 0.30,
    ".*_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
}
SPAWN_Z = 0.78
# The authors' USD keeps head_link (1.036 kg) as its own body on torso_link (8.562 kg);
# mjlab's torso_link (7.818 kg) has the head mesh but not its mass.
TORSO_MASS = 8.562 + 1.036
# Height map: RayCaster on the pelvis, rays from 20 m up, grid x in [-0.5, 1.5], y in
# [-0.56, 0.56] at 0.08 m, x-major (Isaac "yx" ordering). z = hit - pelvis + 0.74.
MAP_X = torch.arange(-1.0, 1.0 + 1e-5, 0.08) + 0.5
MAP_Y = torch.arange(-0.56, 0.56 + 1e-5, 0.08)
MAP_OFFSET = 0.74
GOAL_CLIP = 2.0  # teacher_prop pose_commands dist_thr
PROP_DIM = 82  # 9 base + 3 x 23 joints/actions + 4 goal


def robot_spec() -> mujoco.MjSpec:
    spec = get_spec()
    for joint in list(spec.joints):
        # PhysX joint limits are hard, MuJoCo's soft; stiffest the 5 ms step allows.
        joint.solref_limit = [0.01, 1.0]
        joint.solimp_limit = [0.99, 0.999, 0.001, 0.5, 2.0]
        if "_wrist_" in joint.name:  # fixed joints in the authors' USD
            spec.delete(joint)
    spec.body("torso_link").mass = TORSO_MASS
    return spec


def _actuator(joints: tuple[str, ...], kp: float, kd: float, effort: float, armature: float) -> BuiltinPositionActuatorCfg:
    return BuiltinPositionActuatorCfg(
        target_names_expr=joints, stiffness=kp, damping=kd, effort_limit=effort, armature=armature
    )


# assets/unitree.py actuators (DCMotorCfg; its speed-torque saturation is not modelled).
ACTUATORS = (
    _actuator((".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"), 88.0, 2.0, 88.0, 0.01017752004),
    _actuator((".*_hip_roll_joint",), 139.0, 3.0, 88.0, 0.025101925),
    _actuator((".*_knee_joint",), 139.0, 3.0, 139.0, 0.025101925),
    _actuator(("waist_roll_joint", "waist_pitch_joint"), 50.0, 1.0, 50.0, 0.00721945),
    _actuator((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 35.0, 0.8, 50.0, 0.00721945),
    _actuator((".*_shoulder_pitch_joint",), 35.0, 0.8, 25.0, 0.003609725),
    _actuator((".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"), 40.0, 1.0, 25.0, 0.003609725),
    _actuator((".*_elbow_joint",), 40.0, 1.0, 25.0, 0.003597),
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
        articulation=EntityArticulationInfoCfg(actuators=ACTUATORS, soft_joint_pos_limit_factor=0.9),
    )


class AmeMapPatternCfg(GridPatternCfg):
    """The authors' scan grid, x-major (mjlab's own grid is y-major), rays 20 m above the pelvis."""

    def __init__(self) -> None:
        super().__init__(size=(2.0, 1.12), resolution=0.08)

    def generate_rays(self, mj_model, device):
        gx, gy = torch.meshgrid(MAP_X.to(device), MAP_Y.to(device), indexing="ij")
        offsets = torch.stack([gx.flatten(), gy.flatten(), torch.full((gx.numel(),), 20.0, device=device)], 1)
        directions = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(offsets).clone()
        return offsets, directions


def height_map(env, sensor_name: str) -> torch.Tensor:
    """`ame2_teacher_mapping`: (B, 26 * 15 * 3), channels (x, y, z) per cell, x-major."""
    z = (MAP_OFFSET - mdp.height_scan(env, sensor_name)).clamp(-1.5, 1.5)  # height_scan = pelvis - hit
    gx, gy = torch.meshgrid(MAP_X, MAP_Y, indexing="ij")
    xy = torch.stack([gx.flatten(), gy.flatten()], -1).to(z.device).expand(z.shape[0], -1, -1)
    return torch.cat([xy, z.unsqueeze(-1)], -1).flatten(1)


def goal_command(env, goal: tuple[float, float, float]) -> torch.Tensor:
    """`ame2_commands_obs` of a fixed world goal (x, y, heading): [x_b, y_b, sin dyaw, cos dyaw].

    The goal is clipped to 2 m. Isaac then randomizes dyaw (the true one is in distribution).
    """
    data = env.scene["robot"].data
    yaw = data.heading_w
    d = torch.tensor(goal[:2], device=yaw.device) - data.root_link_pos_w[:, :2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    p = torch.stack([c * d[:, 0] + s * d[:, 1], c * d[:, 1] - s * d[:, 0]], -1)
    p = p * (GOAL_CLIP / p.norm(dim=-1, keepdim=True).clamp(min=GOAL_CLIP))
    dyaw = goal[2] - yaw
    return torch.cat([p, torch.sin(dyaw).unsqueeze(-1), torch.cos(dyaw).unsqueeze(-1)], -1)


def base_lin_vel(env) -> torch.Tensor:  # Isaac Lab 2.x root_lin_vel_b is the COM's
    return env.scene["robot"].data.root_com_lin_vel_b


def env_cfg(
    spawn: tuple[float, float], yaw: float, goal: tuple[float, float], episode_s: float = 20.0, num_envs: int = 1,
    jitter: bool = False,
) -> ManagerBasedRlEnvCfg:
    joints = SceneEntityCfg("robot", joint_names=tuple(MODEL_JOINTS))
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=AmeMapPatternCfg(),
        max_distance=30.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # terrain + course only; robot collisions are group 3
    )
    # teacher_prop, noise off (play config). The ONNX actor reorders joints.
    actor = {
        "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"asset_cfg": joints}),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, params={"asset_cfg": joints}),
        # `ame2_last_action` is ActionManager.prev_action, one step older than mjlab's
        # last_action. The browser serves only the current action, so keep two and let
        # the actor read the older one.
        "actions": ObservationTermCfg(func=mdp.last_action, history_length=2),
        "pose_commands": ObservationTermCfg(func=goal_command, params={"goal": (*goal, yaw)}),
    }
    reset_pose = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.3, 0.3)} if jitter else {}
    reset_vel = {k: (-0.3, 0.3) for k in ("x", "y", "z", "roll", "pitch", "yaw")} if jitter else {}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg(spawn, yaw), "course": EntityCfg(spec_fn=course_spec)},
            sensors=(terrain_scan,),
            num_envs=num_envs,
            env_spacing=0.0,  # the course is static geometry at world coordinates
        ),
        # The map is its own group (second ONNX input): mjswan 0.11.6 sends the browser a
        # group's raycast descriptor only for a group it fuses, and per-term history (above)
        # stops the fusion. Without it the browser's rays read zeros.
        observations={
            "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=False),
            "map": ObservationGroupCfg(
                terms={"teacher_mapping": ObservationTermCfg(func=height_map, params={"sensor_name": "terrain_scan"})},
                concatenate_terms=True,
                enable_corruption=False,
            ),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot", actuator_names=tuple(MODEL_JOINTS), scale=0.25, use_default_offset=True,
            )
        },
        events={
            "reset_base": EventTermCfg(
                func=vel_mdp.reset_root_state_uniform,
                mode="reset",
                params={"pose_range": reset_pose, "velocity_range": reset_vel},
            ),
            "reset_robot_joints": EventTermCfg(
                func=vel_mdp.reset_joints_by_offset,
                mode="reset",
                params={"position_range": (-0.1, 0.1) if jitter else (0.0, 0.0), "velocity_range": (0.0, 0.0)},
            ),
        },
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
            max_extra_envs=0,
            width=1280,
            height=720,
        ),
        sim=SimulationCfg(
            nconmax=300,
            njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=episode_s,
    )


def course_spec() -> mujoco.MjSpec:
    """The lab course (`course.xml`, baked from mad_assets by `bake_course.py`) plus wide stairs."""
    spec = mujoco.MjSpec.from_file(str(Path(__file__).parent / "course.xml"))
    root = spec.worldbody
    # Not in the lab: wide stairs up and down on the free floor, like the authors' pyramid stairs.
    for i in range(2 * STAIRS_N - 1):
        level = min(i, 2 * STAIRS_N - 2 - i) + 1
        length = STAIRS_PLATFORM if i == STAIRS_N - 1 else STAIRS_TREAD
        x = STAIRS_X0 + i * STAIRS_TREAD + (STAIRS_PLATFORM - STAIRS_TREAD) * (i >= STAIRS_N) + length / 2
        root.add_geom(
            name=f"wide_stairs_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, material="steel",
            size=[length / 2, 1.5, level * STAIRS_STEP / 2], pos=[x, 0.0, level * STAIRS_STEP / 2],
        )
    return spec


STAIRS_X0, STAIRS_STEP, STAIRS_N, STAIRS_TREAD, STAIRS_PLATFORM = -9.0, 0.20, 4, 0.30, 1.2


# One scene per station: spawn facing the obstacle, goal past it with the same heading.
# Not the lab's 25 cm stairs: 0.61 m wide, the policy walks around them to the goal (and
# will not climb them to a goal on top); on wide stairs it climbs 25 cm in 14/16 trials.
STATIONS = {
    "Lab stairs 12 cm": dict(spawn=(-2.0, 2.3), yaw=math.pi / 2, goal=(-2.0, 6.6)),
    "Lab ramp": dict(spawn=(-2.0, -1.8), yaw=-math.pi / 2, goal=(-2.0, -6.2)),
    "Brick road": dict(spawn=(-1.0, 0.0), yaw=0.0, goal=(7.0, 0.0)),
    "Wide stairs 20 cm": dict(spawn=(-10.5, 0.0), yaw=0.0, goal=(-4.0, 0.0)),
}


def _mlp(i: int, o: int) -> nn.Sequential:  # rsl_rl.modules.ame2_modules.MLP_512
    return nn.Sequential(nn.Linear(i, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(), nn.Linear(512, o))


class _Actor(nn.Module):
    """`AME2GazeModel` (rsl_rl/models/AME2_models.py) inference path, mean action.

    Takes the mjlab "actor" and "map" observations (joints in MODEL_JOINTS order) and returns actions in
    MODEL_JOINTS order. The straight-through gaze crop is bitwise its hard crop in the
    forward pass, and the MHA is written out so the legacy ONNX exporter takes it.
    """

    L, W, CROP_X, CROP_Y, HEADS = len(MAP_X), len(MAP_Y), 12, 10, 32

    def __init__(self) -> None:
        super().__init__()
        self.fc_part = nn.Sequential(nn.Linear(3, 16), nn.ELU())
        self.cnn_part = nn.Sequential(nn.Conv2d(1, 8, 5, padding=2), nn.ELU(), nn.Conv2d(8, 48, 5, padding=2), nn.ELU())
        self.local_proj = nn.Sequential(nn.Linear(64, 96), nn.ELU())
        self.prop_encoder = _mlp(PROP_DIM, 128)
        self.global_cnn = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ELU(), nn.Conv2d(8, 32, 3, dilation=2), nn.ELU())
        self.global_proj = nn.Sequential(nn.Linear(32, 64), nn.ELU())
        self.gaze_head = nn.Sequential(nn.Linear(192, 64), nn.ELU(), nn.Linear(64, 2))
        self.query_mlp = _mlp(192, 96)
        self.mha = nn.MultiheadAttention(96, self.HEADS, batch_first=True)
        self.output_mlp = _mlp(288, 23)
        to_policy = torch.tensor([MODEL_JOINTS.index(j) for j in POLICY_JOINTS])
        # obs: 9 base, joint_pos, joint_vel, actions (t-1, t), goal (4), map. Prop wants a_{t-1}.
        self.register_buffer("prop_idx", torch.cat([torch.arange(9), 9 + to_policy, 32 + to_policy, 55 + to_policy, torch.arange(101, 105)]), persistent=False)
        self.register_buffer("act_idx", torch.tensor([POLICY_JOINTS.index(j) for j in MODEL_JOINTS]), persistent=False)

    def forward(self, obs: torch.Tensor, height_map: torch.Tensor) -> torch.Tensor:
        B, L, W, cx, cy = obs.shape[0], self.L, self.W, self.CROP_X, self.CROP_Y
        prop = obs[:, self.prop_idx]
        m = height_map.reshape(B, L, W, 3)
        # Stage 1: global context and gaze.
        g = self.global_cnn(m.permute(0, 3, 1, 2)).flatten(2).max(dim=2).values
        glob = self.global_proj(g)
        pe = self.prop_encoder(prop)
        r = torch.sigmoid(self.gaze_head(torch.cat([glob, pe], -1)))
        ox = torch.clamp(torch.round(r[:, 0] * float(L - cx)).long(), 0, L - cx)
        oy = torch.clamp(torch.round(r[:, 1] * float(W - cy)).long(), 0, W - cy)
        rows = ox.unsqueeze(1) + torch.arange(cx, device=obs.device)
        cols = oy.unsqueeze(1) + torch.arange(cy, device=obs.device)
        idx = (rows.unsqueeze(2) * W + cols.unsqueeze(1)).reshape(B, -1, 1).expand(-1, -1, 3)
        patch = m.reshape(B, L * W, 3).gather(1, idx).view(B, cx, cy, 3)
        # Stage 2: crop encoder, attention with a [prop; context] query.
        fc = self.fc_part(patch)
        cnn = self.cnn_part(patch[..., 2:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        local = self.local_proj(torch.cat([fc, cnn], -1)).view(B, cx * cy, 96)
        query = self.query_mlp(torch.cat([pe, glob], -1)).unsqueeze(1)
        attn = self._mha(query, local).squeeze(1)
        return self.output_mlp(torch.cat([attn, pe, glob], -1))[:, self.act_idx]

    def _mha(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, N, H, d = kv.shape[0], kv.shape[1], self.HEADS, 96 // self.HEADS
        wq, wk, wv = self.mha.in_proj_weight.chunk(3)
        bq, bk, bv = self.mha.in_proj_bias.chunk(3)
        q = F.linear(q, wq, bq).view(B, 1, H, d).transpose(1, 2)
        k = F.linear(kv, wk, bk).view(B, N, H, d).transpose(1, 2)
        v = F.linear(kv, wv, bv).view(B, N, H, d).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), -1)
        return self.mha.out_proj((a @ v).transpose(1, 2).reshape(B, 1, 96))


def actor(checkpoint: Path = CHECKPOINT) -> torch.nn.Module:
    """The authors' rsl-rl checkpoint, downloaded once (83 MB, pinned commit)."""
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(AME2_URL, checkpoint.with_suffix(".part"))
        checkpoint.with_suffix(".part").rename(checkpoint)
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    sd = {k.replace(".net.", "."): v for k, v in sd.items() if k != "log_std"}
    net = _Actor()
    net.load_state_dict(sd, strict=True)
    return net.eval()


OBS_DIM, MAP_DIM = 105, 3 * len(MAP_X) * len(MAP_Y)


def export_onnx(checkpoint: Path = CHECKPOINT) -> onnx.ModelProto:
    """The actor as one self-contained ONNX graph: (obs [B, OBS_DIM], map [B, MAP_DIM]) -> actions [B, 23]."""
    import io

    buf = io.BytesIO()
    torch.onnx.export(
        actor(checkpoint), (torch.zeros(1, OBS_DIM), torch.zeros(1, MAP_DIM)), buf, input_names=["obs", "map"],
        output_names=["actions"], dynamic_axes={k: {0: "batch"} for k in ("obs", "map", "actions")}, dynamo=False,
    )
    return onnx.load_from_string(buf.getvalue())
