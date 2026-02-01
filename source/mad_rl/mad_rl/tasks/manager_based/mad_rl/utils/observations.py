from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from .misc import crop_image

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


from isaaclab.envs.utils.io_descriptors import generic_io_descriptor

from .io_descriptors import record_clip_range, record_crop_ratios


@generic_io_descriptor(on_inspect=[record_crop_ratios, record_clip_range])
def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    left_crop_ratio: float = 0.35,
    right_crop_ratio: float = 0.35,
    top_crop_ratio: float = 0.75,
    bottom_crop_ratio: float = 0.0,
    clip_range: tuple[float, float] = (0.0, 2.0),
) -> torch.Tensor:
    """Observation term for cropped and clipped depth image from a camera sensor."""
    camera = env.scene.sensors[sensor_cfg.name]
    depth_image = camera.data.output["distance_to_image_plane"]
    cropped_depth_image = crop_image(
        depth_image,
        left_crop_ratio,
        right_crop_ratio,
        top_crop_ratio,
        bottom_crop_ratio,
    )
    clipped_depth_image = torch.clamp(cropped_depth_image, min=clip_range[0], max=clip_range[1])

    return clipped_depth_image.flatten(start_dim=1)
