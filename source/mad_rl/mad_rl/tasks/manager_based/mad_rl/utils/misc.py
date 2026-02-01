import torch


def crop_image(
    image: torch.Tensor,
    left_crop_ratio: float,
    right_crop_ratio: float,
    top_crop_ratio: float,
    bottom_crop_ratio: float,
) -> torch.Tensor:
    h, w = image.shape[1:3]
    return image.clone()[
        :,
        int(h * top_crop_ratio) : h - int(h * bottom_crop_ratio),
        int(w * left_crop_ratio) : w - int(w * right_crop_ratio),
    ]
