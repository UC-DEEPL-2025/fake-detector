import os
import csv
import torch
from omegaconf import DictConfig
import hydra

from dfdetect.utils.seed import set_seed
from dfdetect.datasets.segmentation import make_dataloaders
from dfdetect.datasets.transforms import build_transforms
from dfdetect.models.registry import build_model
from dfdetect.utils.logging import ensure_dir, write_json


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    verbose = bool(cfg.get("verbose", True))
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path required (e.g., checkpoint_path=runs/xxx/best_unet.pt)")
    
    if verbose:
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if verbose:
        print(f"[INFO] Setting seed: {cfg.train.seed}")
    set_seed(cfg.train.seed)
    
    if verbose:
        print(f"[INFO] Building dataloaders...")
        print(f"  - subset_fraction: {cfg.data.subset_fraction}")
    
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
    
    test_loader = loaders["test"]
    
    if verbose:
        print(f"[INFO] Test set: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Using device: {device}")
        print(f"[INFO] Building model...")
    
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    if verbose:
        print(f"[INFO] Running inference on test set...")
    
    y_true = []
    y_prob = []
    paths = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist()
            labels = batch["label"].cpu().numpy().tolist()
            batch_paths = batch["path"]
            
            y_true.extend(labels)
            y_prob.extend(probs)
            paths.extend(batch_paths)
            
            if verbose and (i + 1) % max(1, len(test_loader) // 5) == 0:
                print(f"  [Batch {i+1}/{len(test_loader)}]")
    
    # simple accuracy
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    correct = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    preds_path = os.path.join(checkpoint_dir, "test_preds.csv")
    with open(preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "prob", "pred"])
        writer.writeheader()
        for p, yt, pr, yp in zip(paths, y_true, y_prob, y_pred):
            writer.writerow({"path": p, "label": yt, "prob": pr, "pred": yp})
    
    print(f"\nTEST RESULTS\n")
    print(f"Samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"preds saved to: {preds_path}")


if __name__ == "__main__":
    main()
