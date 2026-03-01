from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


from isaaclab.envs.utils.io_descriptors import GenericObservationIODescriptor


def record_height_scan_params(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    sensor_cfg = kwargs["env"].scene[kwargs["sensor_cfg"].name].cfg
    length, width = sensor_cfg.pattern_cfg.size
    x = sensor_cfg.offset.pos[0]
    y = sensor_cfg.offset.pos[1]
    x_min = x - length / 2
    x_max = x + length / 2
    y_min = y - width / 2
    y_max = y + width / 2
    descriptor.offset = kwargs["offset"]
    descriptor.x_min = x_min
    descriptor.x_max = x_max
    descriptor.y_min = y_min
    descriptor.y_max = y_max
    descriptor.res = sensor_cfg.pattern_cfg.resolution
