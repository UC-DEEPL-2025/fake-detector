import sys
from omegaconf import DictConfig
import hydra

from dfdetect.utils.seed import set_seed
from dfdetect.datasets.segmentation import make_dataloaders
from dfdetect.datasets.transforms import build_transforms
from dfdetect.training.loop import train_loop


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    verbose = bool(cfg.get("verbose", True))
    if verbose:
        print(f"[INFO] Setting seed: {cfg.train.seed}")
    set_seed(cfg.train.seed)

    if verbose:
        print(f"[INFO] Building dataloaders...")
        print(f"  - root_dir: {cfg.data.root_dir}")
        print(f"  - subset_fraction: {cfg.data.subset_fraction}")
        print(f"  - batch_size: {cfg.data.batch_size}")
    
    loaders = make_dataloaders(
        root_dir=cfg.data.root_dir,
        classes=list(cfg.data.classes),
        img_size=int(cfg.data.img_size),
        normalize=bool(cfg.data.normalize),
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        build_transforms=build_transforms,
        hflip_p=float(cfg.data.aug.hflip_p),
        brightness_contrast_p=float(cfg.data.aug.brightness_contrast_p),
        allowed_extensions=list(cfg.data.allowed_extensions),
        subset_fraction=float(cfg.data.subset_fraction),
    )
    
    if verbose:
        print(f"[INFO] Dataloaders created:")
        print(f"  - train: {len(loaders['train'].dataset)} samples, {len(loaders['train'])} batches")
        print(f"  - val: {len(loaders['val'].dataset)} samples, {len(loaders['val'])} batches")
        print(f"  - test: {len(loaders['test'].dataset)} samples, {len(loaders['test'])} batches")
        print(f"[INFO] Starting training for {cfg.train.epochs} epochs...")
    
    result = train_loop(cfg, loaders)
    print(f"Training complete. Best checkpoint: {result.get('best_checkpoint')}\nRun dir: {result.get('run_dir')}")


if __name__ == "__main__":
    main()
