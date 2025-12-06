from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "torso_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ],
        scale=0.5,
        use_default_offset=True,
    )
