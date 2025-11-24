from typing import Dict, Any

import os
import math
from time import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dfdetect.models.registry import build_model
from dfdetect.utils.logging import ensure_dir, write_json, timestamp


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _val_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n = 0
    y_true_all = []
    y_prob_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy().tolist()
            y_true_all.extend(batch["label"].cpu().numpy().tolist())
            y_prob_all.extend(probs)
    avg_loss = total_loss / max(1, n)
    return {"val_loss": avg_loss, "y_true": y_true_all, "y_prob": y_prob_all}


def train_loop(cfg: Dict[str, Any], loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
    verbose = bool(cfg.get("verbose", True))
    device = _device()
    if verbose:
        print(f"[INFO] Using device: {device}")
    
    # Check if resuming from checkpoint
    resume_path = cfg.get("checkpoint_path")
    start_epoch = 1
    best_val = math.inf
    
    if resume_path and os.path.exists(resume_path):
        if verbose:
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
        # Extract run_dir from checkpoint path
        run_dir = os.path.dirname(resume_path)
    else:
        out_root = os.path.abspath(cfg.get("output_dir", "runs"))
        exp_name = cfg.get("experiment_name", "exp") + "_" + timestamp()
        run_dir = os.path.join(out_root, exp_name)
        ensure_dir(run_dir)
    
    if verbose:
        print(f"[INFO] Run directory: {run_dir}")

    if verbose:
        print(f"[INFO] Building model: {cfg['model']['type']}")
    model = build_model(cfg["model"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt_name = str(cfg.train["optimizer"]).lower()
    lr = float(cfg.train["lr"])
    wd = float(cfg.train["weight_decay"])
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if verbose:
        print(f"[INFO] Optimizer: {opt_name}, lr={lr}, weight_decay={wd}")

    use_amp = bool(cfg.train["mixed_precision"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Load checkpoint if resuming
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val = checkpoint.get("best_val_loss", checkpoint.get("val_loss", math.inf))
        if verbose:
            print(f"[INFO] Resumed from epoch {start_epoch-1}, best_val_loss={best_val:.4f}")

    epochs = int(cfg.train["epochs"])
    best_path = os.path.join(run_dir, "best_unet.pt")
    last_path = os.path.join(run_dir, "last_unet.pt")

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    for epoch in range(start_epoch, epochs + 1):
        if verbose:
            print(f"\n[EPOCH {epoch}/{epochs}] Starting training...")
        model.train()
        total = 0
        total_loss = 0.0
        t0 = time()
        for i, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total += x.size(0)
            total_loss += loss.item() * x.size(0)
            
            if verbose and (i + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  [Batch {i+1}/{len(train_loader)}] loss={loss.item():.4f}")
        
        tr_loss = total_loss / max(1, total)
        train_time = time() - t0
        if verbose:
            print(f"[EPOCH {epoch}/{epochs}] Train complete: loss={tr_loss:.4f}, time={train_time:.1f}s")
            print(f"[EPOCH {epoch}/{epochs}] Running validation...")

        val_out = _val_epoch(model, val_loader, device, criterion)
        val_loss = val_out["val_loss"]
        
        if verbose:
            print(f"[EPOCH {epoch}/{epochs}] Validation complete: loss={val_loss:.4f}")

        # save the best
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "config": dict(cfg),
            }, best_path)
            if verbose:
                print(f"[EPOCH {epoch}/{epochs}] New best model saved (val_loss={val_loss:.4f})")

        # Always save the latest checkpoint for resuming
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "config": dict(cfg),
        }, last_path)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

    # after training / write best summary and a validation preds.csv for external metrics
    val_out = _val_epoch(model, val_loader, device, criterion)
    import csv
    preds_csv = os.path.join(run_dir, "val_preds.csv")
    with open(preds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "prob"])
        writer.writeheader()
        i = 0
        for batch in val_loader:
            # recompute on CPU side using precomputed probs to avoid double forward
            # but we did not store paths per-sample during _val_epoch; recompute simply
            x = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist()
            labels = batch["label"].cpu().numpy().tolist()
            paths = batch["path"]
            for p, y, pr in zip(paths, labels, probs):
                writer.writerow({"path": p, "label": y, "prob": pr})

    best_summary = {
        "best_val_loss": best_val,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "run_dir": run_dir,
        "model": "unet",
    }
    write_json(best_summary, os.path.join(run_dir, "metrics_unet_best.json"))
    if verbose:
        print(f"\n[INFO] Training complete!")
        print(f"[INFO] Best checkpoint: {best_path}")
        print(f"[INFO] Last checkpoint (for resume): {last_path}")
    return best_summary
