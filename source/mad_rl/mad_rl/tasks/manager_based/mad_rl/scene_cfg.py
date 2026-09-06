import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from .utils.constants import HEIGHT_SCAN_RESOLUTION, HEIGHT_SCAN_SIZE


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
            usd_path=os.path.join(os.environ["MAD_ASSETS_PATH"], "models", "G1_upd", "G1.usd"),
            variants={"Configuration": "RL"},
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.7945),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_joint": 1.134,
                ".*_shoulder_pitch_joint": 0.261,
                "left_shoulder_roll_joint": 0.261,
                "right_shoulder_roll_joint": -0.261,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
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
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_elbow_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                ],
                effort_limit_sim={
                    ".*_elbow_joint": 300,
                    ".*_shoulder_pitch_joint": 300,
                    ".*_shoulder_roll_joint": 300,
                },
                stiffness={
                    ".*_elbow_joint": 40.0,
                    ".*_shoulder_pitch_joint": 40.0,
                    ".*_shoulder_roll_joint": 40.0,
                },
                damping={
                    ".*_elbow_joint": 10.0,
                    ".*_shoulder_pitch_joint": 10.0,
                    ".*_shoulder_roll_joint": 10.0,
                },
                armature={
                    ".*_elbow_joint": 0.01,
                    ".*_shoulder_pitch_joint": 0.01,
                    ".*_shoulder_roll_joint": 0.01,
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

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/G1_base/base/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=HEIGHT_SCAN_SIZE, ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
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
            num_cols=9,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                    step_height_range=(0.05, 0.17),
                    step_width=0.27,
                    platform_width=1.5,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
                    step_height_range=(0.05, 0.17),
                    step_width=0.24,
                    platform_width=1.5,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_wide": terrain_gen.MeshPyramidStairsTerrainCfg(
                    step_height_range=(0.05, 0.17),
                    step_width=0.3,
                    platform_width=1.5,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    step_height_range=(0.05, 0.17),
                    step_width=0.27,
                    platform_width=1.5,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv_narrow": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    step_height_range=(0.05, 0.17),
                    step_width=0.24,
                    platform_width=1.5,
                    border_width=1.0,
                    holes=False,
                ),
                "pyramid_stairs_inv_wide": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    step_height_range=(0.05, 0.17),
                    step_width=0.3,
                    platform_width=1.5,
                    border_width=1.0,
                    holes=False,
                ),
                "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    slope_range=(0.0, 0.4),
                    platform_width=1.5,
                    border_width=0.25,
                ),
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    slope_range=(0.0, 0.4),
                    platform_width=1.5,
                    border_width=0.25,
                ),
                "plane": terrain_gen.MeshPlaneTerrainCfg(),
            },
        )
