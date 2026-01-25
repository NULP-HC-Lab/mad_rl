from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_default_joint_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]

    default_joint_pos = robot.data.default_joint_pos[env_ids, asset_cfg.joint_ids].clone()

    robot.set_joint_position_target(
        target=default_joint_pos,
        env_ids=env_ids,
    )

    robot.write_joint_position_to_sim(
        position=default_joint_pos,
        env_ids=env_ids,
    )

    robot.write_data_to_sim()
