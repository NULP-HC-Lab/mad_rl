import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .actions_cfg import ActionsCfg
from .commands_cfg import CommandsCfg
from .events_cfg import EventsCfg
from .observations_cfg import ObservationsCfg
from .rewards_cfg import RewardsCfg
from .scene_cfg import SceneCfg
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


@configclass
class EnvCfg_PLAY(EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=os.path.join(
                os.environ["DOCKER_MAD_ASSETS_PATH"],
                "models",
                "floor",
                "floor.usd",
            ),
            collision_group=-1,
            debug_vis=True,
        )
        self.scene.num_envs = 1

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.81)

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
