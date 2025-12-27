from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity import mdp


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )
