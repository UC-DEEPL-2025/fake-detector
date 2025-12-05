import os
import sys

# Add repo root to sys.path so 'dfdetect' package is visible
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dfdetect.preprocessing import preprocess
from dfdetect.engine import engine  # still imported in case you need it
from dfdetect.model import model_builder

# ----- Config (match train.py) -----
NUM_EPOCHS = 1          # not used here, but kept for consistency
BATCH_SIZE = 64
MODEL_PATH = r"/home/luudh/luudh/MyFile/fake-detector/model/ResNet152.pth"
BASE_DIR = r"/home/luudh/luudh/MyFile/fake-detector/manjilkarki/deepfake-and-real-images/versions/1/Dataset/"


def load_model(gpu, world_size, model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Rebuild the ResNet model and load trained weights."""

    # --- Only initialize distributed if we're actually using >1 GPU ---
    if world_size > 1:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=gpu,
        )

    # Build base model
    model = model_builder.ResNet(
        block=model_builder.Bottlenet_block,
        layers=[3, 8, 36, 3],   # same as in train.py
        num_classes=num_classes
    ).to(device)

    # Wrap in DDP only if multi-GPU
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    if gpu == 0:
        print(f"[INFO] Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if gpu == 0:
        print("[INFO] Model loaded successfully")

    return model


def create_test_dataloader(batch_size: int):
    """Create a test dataloader using the same transforms as training."""
    train_dir = os.path.join(BASE_DIR, "Train")
    test_dir = os.path.join(BASE_DIR, "Test")  # used as test set

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Same transforms as in train.py
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Reuse your helper; we only care about test_dataloader and class_names
    train_dataloader, test_dataloader, class_names = preprocess.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size,
        distributed=False  # no DistributedSampler for testing
    )

    return test_dataloader, class_names


def _get_sample_path(dataset, idx: int):
    """
    Try to recover an image path for dataset[idx].
    Handles common cases: Subset(ImageFolder), ImageFolder, custom datasets with .samples or .imgs.
    """
    # If this is a Subset, map through to underlying dataset
    if isinstance(dataset, Subset):
        base_idx = dataset.indices[idx]
        return _get_sample_path(dataset.dataset, base_idx)

    # Typical torchvision ImageFolder-style datasets
    if hasattr(dataset, "samples") and len(dataset.samples) > 0:
        # samples: List[(path, label)]
        return dataset.samples[idx][0]
    if hasattr(dataset, "imgs") and len(dataset.imgs) > 0:
        return dataset.imgs[idx][0]

    # Some custom datasets might store paths explicitly
    if hasattr(dataset, "image_paths"):
        return dataset.image_paths[idx]
    if hasattr(dataset, "paths"):
        return dataset.paths[idx]

    # Fallback: if __getitem__ returns (image, label, path) or dict with 'path'
    sample = dataset[idx]
    if isinstance(sample, (tuple, list)) and len(sample) >= 3:
        return sample[2]
    if isinstance(sample, dict) and "path" in sample:
        return sample["path"]

    # If all else fails
    return None


def compute_metrics(model, dataloader, device):
    """
    Run a full pass over dataloader and compute:
    - avg loss
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc
    Also returns a dict of up to 2 example paths for each:
    - true positive (tp)
    - true negative (tn)
    - false positive (fp)
    - false negative (fn)
    """
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model.eval()

    all_logits = []
    all_labels = []
    all_paths = []

    total_loss = 0.0
    num_batches = 0

    dataset = dataloader.dataset
    global_idx = 0  # index within the test subset

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            batch_size = labels.size(0)

            # Recover paths for each sample in this batch
            for i in range(batch_size):
                path = _get_sample_path(dataset, global_idx + i)
                all_paths.append(path)

            global_idx += batch_size

            # Move to CPU for metric computation later
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if num_batches == 0:
        raise RuntimeError("No batches in dataloader — check your dataset paths.")

    # Concatenate over all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Probabilities (softmax over classes)
    probs = torch.softmax(all_logits, dim=1)

    # Predictions = argmax over classes
    y_pred = probs.argmax(dim=1).numpy()
    y_true = all_labels.numpy()

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary" if probs.shape[1] == 2 else "macro"
    )

    # ROC–AUC
    if probs.shape[1] == 2:
        # Binary: prob of class-1 as positive
        roc_auc = roc_auc_score(y_true, probs[:, 1].numpy())
        pos_probs = probs[:, 1].numpy()
    else:
        roc_auc = roc_auc_score(y_true, probs.numpy(), multi_class="ovr")
        pos_probs = None  # not used for multi-class confusion examples

    avg_loss = total_loss / num_batches

    # Collect example paths for confusion matrix categories (binary) 
    examples = {"tp": [], "tn": [], "fp": [], "fn": []}
    if probs.shape[1] == 2:
        for yt, yp, path, p_pos in zip(y_true, y_pred, all_paths, pos_probs):
            if path is None:
                continue

            if yt == 1 and yp == 1:
                if len(examples["tp"]) < 2:
                    examples["tp"].append({"path": path, "label": int(yt), "pred": int(yp), "prob": float(p_pos)})
            elif yt == 0 and yp == 0:
                if len(examples["tn"]) < 2:
                    examples["tn"].append({"path": path, "label": int(yt), "pred": int(yp), "prob": float(p_pos)})
            elif yt == 0 and yp == 1:
                if len(examples["fp"]) < 2:
                    examples["fp"].append({"path": path, "label": int(yt), "pred": int(yp), "prob": float(p_pos)})
            elif yt == 1 and yp == 0:
                if len(examples["fn"]) < 2:
                    examples["fn"].append({"path": path, "label": int(yt), "pred": int(yp), "prob": float(p_pos)})

            # Stop if we already have 2 of each
            if all(len(examples[k]) >= 2 for k in examples):
                break

    return avg_loss, acc, precision, recall, f1, roc_auc, examples


def evaluate(gpu, world_size):
    """Evaluate the saved model on the test dataset."""
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if use_cuda else "cpu")
    if gpu == 0:
        print(f"[INFO] Using device: {device}")

    # 1. Build test dataloader
    test_dataloader, class_names = create_test_dataloader(batch_size=BATCH_SIZE)
    if gpu == 0:
        print(f"[INFO] Number of classes: {len(class_names)} -> {class_names}")

    # 2. Build and load model
    model = load_model(gpu, world_size, MODEL_PATH, num_classes=len(class_names), device=device)

    # 3. Compute metrics + confusion examples
    avg_loss, acc, precision, recall, f1, roc_auc, examples = compute_metrics(model, test_dataloader, device)

    if gpu == 0:
        print("\n[RESULTS]")
        print(f"  Test Loss   : {avg_loss:.4f}")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"  F1-score    : {f1:.4f}")
        print(f"  ROC–AUC     : {roc_auc:.4f}")

        # Print example paths for inspection
        print("\n[EXAMPLE IMAGES] (up to 2 each)")
        for k in ["tp", "tn", "fp", "fn"]:
            print(f"\n{k.upper()}:")
            if len(examples[k]) == 0:
                print("  (no examples found)")
            else:
                for ex in examples[k]:
                    print(f"  path={ex['path']}, label={ex['label']}, pred={ex['pred']}, prob={ex['prob']:.4f}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count() or 1
    print(f"Using {world_size} GPUs for DistributedDataParallel testing")

    if world_size > 1:
        mp.spawn(evaluate, args=(world_size,), nprocs=world_size, join=True)
    else:
        # Fall back to single-process single-GPU / CPU eval
        evaluate(0, world_size)

