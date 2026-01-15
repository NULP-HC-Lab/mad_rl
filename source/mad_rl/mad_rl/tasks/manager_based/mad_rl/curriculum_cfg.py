import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
