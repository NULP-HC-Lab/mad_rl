from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


from isaaclab.envs.utils.io_descriptors import GenericObservationIODescriptor


def record_crop_ratios(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    descriptor.left_crop_ratio = kwargs["left_crop_ratio"]
    descriptor.right_crop_ratio = kwargs["right_crop_ratio"]
    descriptor.top_crop_ratio = kwargs["top_crop_ratio"]
    descriptor.bottom_crop_ratio = kwargs["bottom_crop_ratio"]


def record_clip_range(output: torch.Tensor, descriptor: GenericObservationIODescriptor, **kwargs) -> None:
    descriptor.clip_range = kwargs["clip_range"]
