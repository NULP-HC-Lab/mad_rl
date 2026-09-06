from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.velocity import mdp

from .utils.constants import CONTROLLABLE_JOINT_NAMES


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=CONTROLLABLE_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
