from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def jump(
    env: ManagerBasedRLEnv,
    left_foot_contact_sensor_cfg: SceneEntityCfg,
    right_foot_contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when both feet are simultaneously off the ground."""
    left_foot_contact_sensor: ContactSensor = env.scene.sensors[left_foot_contact_sensor_cfg.name]
    right_foot_contact_sensor: ContactSensor = env.scene.sensors[right_foot_contact_sensor_cfg.name]
    air_time = torch.cat(
        [
            left_foot_contact_sensor.data.current_air_time[:, left_foot_contact_sensor_cfg.body_ids],
            right_foot_contact_sensor.data.current_air_time[:, right_foot_contact_sensor_cfg.body_ids],
        ],
        dim=1,
    )
    return air_time.min(dim=1)[0] > 0.2
