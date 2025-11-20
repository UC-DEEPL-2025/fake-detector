import os
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def _list_images(root: str, classes: List[str], extensions: List[str]) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    exts = set(e.lower() for e in extensions)
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in os.listdir(cls_dir):
            full = os.path.join(cls_dir, name)
            if os.path.isfile(full) and name.lower().split('.')[-1] in exts:
                paths.append(full)
                labels.append(idx)
    return paths, labels


class DeepFakeDataset(Dataset):
    def __init__(self, root_dir: str, split: str, classes: List[str], transform, *, extensions: List[str], subset_fraction: float):
        self.split_dir = os.path.join(root_dir, split)
        self.classes = classes
        self.transform = transform
        self.paths, self.labels = _list_images(self.split_dir, classes, extensions)
        if len(self.paths) == 0:
            raise ValueError(f"No images found under {self.split_dir} for classes {classes}")
        # Apply subset fraction for trial runs
        if 0.0 < subset_fraction < 1.0:
            n_subset = max(1, int(len(self.paths) * subset_fraction))
            self.paths = self.paths[:n_subset]
            self.labels = self.labels[:n_subset]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        out = self.transform(image=img_np)
        img_t = out["image"]
        target = torch.tensor(label, dtype=torch.long)
        return {"image": img_t, "label": target, "path": path}


def make_dataloaders(
    *,
    root_dir: str,
    classes: List[str],
    img_size: int,
    normalize: bool,
    norm_mean: List[float],
    norm_std: List[float],
    batch_size: int,
    num_workers: int,
    train_split: str,
    val_split: str,
    test_split: str,
    build_transforms,
    hflip_p: float,
    brightness_contrast_p: float,
    allowed_extensions: List[str],
    subset_fraction: float,
) -> Dict[str, DataLoader]:
    t_train = build_transforms(
        img_size=img_size,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        is_train=True,
        hflip_p=hflip_p,
        brightness_contrast_p=brightness_contrast_p,
    )
    t_eval = build_transforms(
        img_size=img_size,
        normalize=normalize,
        norm_mean=norm_mean,
        norm_std=norm_std,
        is_train=False,
        hflip_p=hflip_p,
        brightness_contrast_p=brightness_contrast_p,
    )

    ds_train = DeepFakeDataset(root_dir, train_split, classes, transform=t_train, extensions=allowed_extensions, subset_fraction=subset_fraction)
    ds_val = DeepFakeDataset(root_dir, val_split, classes, transform=t_eval, extensions=allowed_extensions, subset_fraction=subset_fraction)
    ds_test = DeepFakeDataset(root_dir, test_split, classes, transform=t_eval, extensions=allowed_extensions, subset_fraction=subset_fraction)

    def make(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return {"train": make(ds_train, True), "val": make(ds_val, False), "test": make(ds_test, False)}
