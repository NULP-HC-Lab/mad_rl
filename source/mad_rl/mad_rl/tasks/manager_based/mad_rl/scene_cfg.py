import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.environ["DOCKER_MAD_ASSETS_PATH"], "models", "G1_upd", "G1.usd"),
            variants={"Configuration": "RL"},
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.7945),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "lower_body": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 300,
                    ".*_hip_roll_joint": 300,
                    ".*_hip_pitch_joint": 300,
                    ".*_knee_joint": 300,
                    ".*_ankle_pitch_joint": 20,
                    ".*_ankle_roll_joint": 20,
                },
                stiffness={
                    ".*_hip_yaw_joint": 150.0,
                    ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0,
                    ".*_knee_joint": 200.0,
                    ".*_ankle_pitch_joint": 20.0,
                    ".*_ankle_roll_joint": 20.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0,
                    ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0,
                    ".*_knee_joint": 5.0,
                    ".*_ankle_pitch_joint": 2.0,
                    ".*_ankle_roll_joint": 2.0,
                },
                armature={
                    ".*_hip_.*": 0.01,
                    ".*_knee_joint": 0.01,
                    ".*_ankle_pitch_joint": 0.01,
                    ".*_ankle_roll_joint": 0.01,
                },
            ),
        },
        actuator_value_resolution_debug_print=True,
    )

    torso_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/G1_base/base/torso",
        history_length=3,
        track_air_time=False,
    )

    left_foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/G1_base/base/left_ankle_roll",
        history_length=3,
        track_air_time=True,
    )

    right_foot_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/G1_base/base/right_ankle_roll",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class RoughSceneCfg(SceneCfg):
    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_type = "generator"
        self.terrain.max_init_terrain_level = 0
        self.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            curriculum=True,
            color_scheme="height",
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=6,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.1,
                    step_height_range=(0.05, 0.3),
                    step_width=0.27,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.1,
                    step_height_range=(0.05, 0.3),
                    step_width=0.27,
                    platform_width=3.0,
                    border_width=1.0,
                    holes=False,
                ),
                "rough_ground": terrain_gen.MeshRepeatedCylindersTerrainCfg(
                    proportion=0.2,
                    platform_width=2.0,
                    object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                        num_objects=1500,
                        height=0.05,
                        radius=0.25,
                        max_yx_angle=0,
                        degrees=True,
                    ),
                    object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                        num_objects=3500,
                        height=0.3,
                        radius=0.15,
                        max_yx_angle=10,
                        degrees=True,
                    ),
                    abs_height_noise=(0.0, 0.35),
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=0.1,
                    slope_range=(0.0, 0.4),
                    platform_width=2.0,
                    border_width=0.25,
                ),
            },
        )
