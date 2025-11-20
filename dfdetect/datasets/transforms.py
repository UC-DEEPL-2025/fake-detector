from typing import Any, Sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(
    *,
    img_size: int,
    normalize: bool,
    norm_mean: Sequence[float],
    norm_std: Sequence[float],
    is_train: bool,
    hflip_p: float,
    brightness_contrast_p: float,
) -> Any:
    transforms = []
    transforms.append(A.Resize(img_size, img_size, interpolation=1))
    if is_train:
        if hflip_p and hflip_p > 0:
            transforms.append(A.HorizontalFlip(p=hflip_p))
        if brightness_contrast_p and brightness_contrast_p > 0:
            transforms.append(A.RandomBrightnessContrast(p=brightness_contrast_p))
    if normalize:
        transforms.append(A.Normalize(mean=tuple(norm_mean), std=tuple(norm_std)))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)
