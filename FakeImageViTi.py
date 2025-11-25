# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:22:17 2025

@author: ayush
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Your ViT Model
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=3):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss
        else:
            return logits, None

# Custom Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, processor, split='train'):
        """
        root_dir should contain subdirectories: 'Real', 'Fake'
        split: 'train' or 'val'
        """
        self.processor = processor
        self.images = []
        self.labels = []

        # Map class names to indices
        self.class_to_idx = {'Real': 0, 'Fake': 1}

        # Load images from each class folder
        for class_name in ['Real', 'Fake']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist!")
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"{split} dataset: {len(self.images)} images loaded")
        print(f"Class distribution: Real={self.labels.count(0)}, Fake={self.labels.count(1)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            # Process image using ViT processor
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)

            return pixel_values, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(images=dummy_image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0), label

# Training Function
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc='Training')
    for pixel_values, labels in pbar:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, loss = model(pixel_values, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1

# Validation Function
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits, loss = model(pixel_values, labels)

            if loss is not None:
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1, all_preds, all_labels

# Main Training Script
def main():
    # Configuration
    TRAIN_DIR = r'C:\Users\ayush\Downloads\archive\Dataset\Train'  # Update this path
    VAL_DIR = r'C:\Users\ayush\Downloads\archive\Dataset\Test'  # Update this path
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    NUM_LABELS = 2  # Real and Fake
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")

    # Initialize processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification(num_labels=NUM_LABELS).to(DEVICE)

    # Create datasets
    train_dataset = DeepfakeDataset(TRAIN_DIR, processor, split='train')
    val_dataset = DeepfakeDataset(VAL_DIR, processor, split='val')

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)

        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(model, val_loader, DEVICE)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:  
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_deepfake_model.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("\nTraining curves saved to 'training_curves.png'")

    # Final confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\nFinal Confusion Matrix:")
    print(cm)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
