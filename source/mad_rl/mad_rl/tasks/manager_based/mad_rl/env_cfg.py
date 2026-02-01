from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .actions_cfg import ActionsCfg
from .commands_cfg import CommandsCfg
from .curriculum_cfg import CurriculumCfg
from .events_cfg import EventsCfg
from .observations_cfg import ObservationsCfg
from .rewards_cfg import RewardsCfg
from .scene_cfg import RoughSceneCfg, SceneCfg
from .terminations_cfg import TerminationsCfg


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.seed = 42

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.torso_contact_forces is not None:
            self.scene.torso_contact_forces.update_period = self.sim.dt
        if self.scene.left_foot_contact_forces is not None:
            self.scene.left_foot_contact_forces.update_period = self.sim.dt
        if self.scene.right_foot_contact_forces is not None:
            self.scene.right_foot_contact_forces.update_period = self.sim.dt
        if self.scene.camera is not None:
            self.scene.camera.update_period = self.decimation * self.sim.dt


@configclass
class EnvCfg_PLAY(EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)

        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.observations.policy.enable_corruption = False


class _RoughEnvCfgMixin:
    scene: RoughSceneCfg = RoughSceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class RoughEnvCfg(_RoughEnvCfgMixin, EnvCfg):
    pass


@configclass
class RoughEnvCfg_PLAY(_RoughEnvCfgMixin, EnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 6
