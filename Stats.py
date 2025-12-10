# stats.py
# -*- coding: utf-8 -*-
"""
Statistics & evaluation script for ViT Deepfake classifier
"""

import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from train import ViTForImageClassification, DeepfakeDataset  # reuse classes


# ----------------------
# Full validation (return predictions + probabilities)
# ----------------------
def validate_full(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits, _ = model(pixel_values)
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # probability of Fake class

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ----------------------
# Main statistics function
# ----------------------
def run_statistics():
    VAL_DIR = r'C:\Users\ayush\Downloads\archive\Dataset\Validation'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    val_dataset = DeepfakeDataset(VAL_DIR, processor, split='val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = ViTForImageClassification().to(DEVICE)
    model.load_state_dict(torch.load('best_vit_deepfake_model.pth', map_location=DEVICE))

    preds, labels, probs = validate_full(model, val_loader, DEVICE)

    # Metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, probs)

    print("\n==== Evaluation Metrics ====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_curve.png")
    print("Saved roc_curve.png")


if __name__ == "__main__":
    run_statistics()
