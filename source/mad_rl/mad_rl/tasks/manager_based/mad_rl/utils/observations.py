from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    # Importing the sensor classes for real pulls in `pxr`, and Isaac Sim segfaults
    # if USD is loaded before SimulationApp starts. Config modules are imported
    # during task resolution, i.e. before launch, so these stay type-only.
    # This `pxr` is pissing me off.
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import RayCaster

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor

from .io_descriptors import record_height_scan_params


@generic_io_descriptor(on_inspect=[record_height_scan_params])
def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
