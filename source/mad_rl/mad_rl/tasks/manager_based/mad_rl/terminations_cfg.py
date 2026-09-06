from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.velocity import mdp


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
