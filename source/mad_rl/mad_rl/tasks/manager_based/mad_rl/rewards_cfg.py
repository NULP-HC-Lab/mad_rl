from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .utils.constants import FOOT_BODY_NAMES
from .utils.rewards import (
    feet_air_time_positive_biped,
    feet_slide,
    horizontal_contact_forces,
    jump,
    vertical_contact_forces,
)


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )

    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )

    feet_air_time = RewTerm(
        func=feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "left_foot_sensor_cfg": SceneEntityCfg("left_foot_contact_forces"),
            "right_foot_sensor_cfg": SceneEntityCfg("right_foot_contact_forces"),
            "threshold": 0.15,
        },
    )

    feet_slide = RewTerm(
        func=feet_slide,
        weight=-0.1,
        params={
            "left_foot_sensor_cfg": SceneEntityCfg("left_foot_contact_forces"),
            "right_foot_sensor_cfg": SceneEntityCfg("right_foot_contact_forces"),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )

    vertical_contact_forces = RewTerm(
        func=vertical_contact_forces,
        weight=0.05,
        params={
            "left_foot_contact_sensor_cfg": SceneEntityCfg("left_foot_contact_forces"),
            "right_foot_contact_sensor_cfg": SceneEntityCfg("right_foot_contact_forces"),
        },
    )

    horizontal_contact_forces = RewTerm(
        func=horizontal_contact_forces,
        weight=-1.0,
        params={
            "left_foot_contact_sensor_cfg": SceneEntityCfg("left_foot_contact_forces"),
            "right_foot_contact_sensor_cfg": SceneEntityCfg("right_foot_contact_forces"),
        },
    )

    jump = RewTerm(
        func=jump,
        weight=-10.0,
        params={
            "left_foot_contact_sensor_cfg": SceneEntityCfg("left_foot_contact_forces"),
            "right_foot_contact_sensor_cfg": SceneEntityCfg("right_foot_contact_forces"),
        },
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.075,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*"])},
    )
