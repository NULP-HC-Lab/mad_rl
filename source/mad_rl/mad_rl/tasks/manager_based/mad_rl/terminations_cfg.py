from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp

from .utils.terminations import jump


@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("torso_contact_forces"),
            "threshold": 1.0,
        },
    )

    jump = DoneTerm(
        func=jump,
        params={
            "left_foot_contact_sensor_cfg": SceneEntityCfg("left_foot_contact_forces"),
            "right_foot_contact_sensor_cfg": SceneEntityCfg("right_foot_contact_forces"),
        },
    )
