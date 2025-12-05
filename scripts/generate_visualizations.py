import os
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from PIL import Image

from dfdetect.models.registry import build_model
from dfdetect.datasets.segmentation import DeepFakeDataset
from dfdetect.datasets.transforms import build_transforms

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

with open("configs/model/unet.yaml", "r") as f:
	model_cfg = yaml.safe_load(f)
with open("configs/data/default.yaml", "r") as f:
	data_cfg = yaml.safe_load(f)

ckpt_path = "runs/baseline_20251123T005114Z/best_unet.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = build_model(model_cfg)
model.load_state_dict(checkpoint["model"])
model.eval()

transforms = build_transforms(
	img_size=data_cfg["img_size"],
	normalize=data_cfg.get("normalize", False),
	norm_mean=data_cfg.get("norm_mean", [0.0, 0.0, 0.0]),
	norm_std=data_cfg.get("norm_std", [1.0, 1.0, 1.0]),
	is_train=False,
	hflip_p=0.0,
	brightness_contrast_p=0.0,
)

os.makedirs("statistics", exist_ok=True)
predictions_file = "statistics/predictions.json"

if os.path.exists(predictions_file):
	print(f"Loading existing predictions from {predictions_file}...")
	with open(predictions_file, 'r') as f:
		results = json.load(f)
	
	y_true = np.array([r['true_label'] for r in results])
	y_pred = np.array([r['predicted_label'] for r in results])
	y_prob = np.array([r['probability'] for r in results])
	
	print(f"Loaded {len(results)} predictions from file.")
else:
	dataset = DeepFakeDataset(
		root_dir=data_cfg["root_dir"],
		split=data_cfg["test_split"],
		classes=data_cfg["classes"],
		transform=transforms,
		extensions=data_cfg["allowed_extensions"],
		subset_fraction=1.0,  # Use full test set
	)

	print(f"Running inference on {len(dataset)} test samples...")

	y_true = []
	y_prob = []
	y_pred = []
	results = []

	for idx, sample in enumerate(dataset):
		img = sample["image"].unsqueeze(0)
		label = sample["label"]
		
		with torch.no_grad():
			logit = model(img)
			prob = torch.sigmoid(logit).squeeze().cpu().item()
			pred = int(prob >= 0.5)
		
		y_true.append(label)
		y_prob.append(prob)
		y_pred.append(pred)
		
		results.append({
			"path": sample["path"],
			"true_label": int(label),
			"predicted_label": pred,
			"probability": float(prob),
		})
		
		if (idx + 1) % 1000 == 0:
			print(f"Processed {idx + 1}/{len(dataset)} samples...")

	y_true = np.array(y_true)
	y_prob = np.array(y_prob)
	y_pred = np.array(y_pred)

	with open(predictions_file, "w") as f:
		json.dump(results, f, indent=2)

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
auc_score = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"accuracy:  {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")
print(f"f1 score:  {f1:.4f}")
print(f"roc-auc:   {auc_score:.4f}")
print(f"\nconfusion matrix:")
print(f"tn: {tn:6d}  fp: {fp:6d}")
print(f"fn: {fn:6d}  tp: {tp:6d}")

os.makedirs("statistics", exist_ok=True)

if not os.path.exists(predictions_file):
	with open(predictions_file, "w") as f:
		json.dump(results, f, indent=2)

summary = {
	"checkpoint": ckpt_path,
	"dataset_split": data_cfg["test_split"],
	"num_samples": len(results),
	"metrics": {
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1_score": float(f1),
		"roc_auc": float(auc_score),
	},
	"confusion_matrix": {
		"true_negative": int(tn),
		"false_positive": int(fp),
		"false_negative": int(fn),
		"true_positive": int(tp),
	}
}

with open("statistics/summary.json", "w") as f:
	json.dump(summary, f, indent=2)

print(f"\nresults saved to:")
print(f"  - statistics/summary.json")
print(f"  - statistics/predictions.json")

# for poster

fig_dir = "statistics/figures"
os.makedirs(fig_dir, exist_ok=True)

print("confusion matrix heatmap...")
fig, ax = plt.subplots(figsize=(8, 6))
cm_display = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '1_confusion_matrix.png'), bbox_inches='tight')
plt.close()

# roc curve
fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
roc_auc_value = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '2_roc_curve.png'), bbox_inches='tight')
plt.close()

# precision-recall curve
precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recalls, precisions)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '3_precision_recall_curve.png'), bbox_inches='tight')
plt.close()

# prediction distribution histogram
fig, ax = plt.subplots(figsize=(10, 6))
real_probs = y_prob[y_true == 0]
fake_probs = y_prob[y_true == 1]
ax.hist(real_probs, bins=50, alpha=0.6, label='Real Images', color='green', edgecolor='black')
ax.hist(fake_probs, bins=50, alpha=0.6, label='Fake Images', color='red', edgecolor='black')
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Classification Threshold')
ax.set_xlabel('Predicted Probability (Fake)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Predicted Probabilities')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '4_prediction_distribution.png'), bbox_inches='tight')
plt.close()

# metrics bar chart
print("Creating metrics bar chart...")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, auc_score]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylim([0, 1])
ax.set_ylabel('Score')
ax.set_title('Model Performance Metrics')
ax.grid(True, alpha=0.3, axis='y')
# value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '5_metrics_bar_chart.png'), bbox_inches='tight')
plt.close()

# threshold analysis
print("Creating threshold analysis...")
thresholds_analysis = np.linspace(0, 1, 100)
accuracies = []
precisions_list = []
recalls_list = []
f1_scores = []

for thresh in thresholds_analysis:
    y_pred_thresh = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_pred_thresh)
    if y_pred_thresh.sum() > 0:  # Avoid division by zero
        prec, rec, f1_t, _ = precision_recall_fscore_support(y_true, y_pred_thresh, average='binary', zero_division=0)
    else:
        prec, rec, f1_t = 0, 0, 0
    accuracies.append(acc)
    precisions_list.append(prec)
    recalls_list.append(rec)
    f1_scores.append(f1_t)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds_analysis, accuracies, label='Accuracy', linewidth=2)
ax.plot(thresholds_analysis, precisions_list, label='Precision', linewidth=2)
ax.plot(thresholds_analysis, recalls_list, label='Recall', linewidth=2)
ax.plot(thresholds_analysis, f1_scores, label='F1 Score', linewidth=2)
ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Current Threshold (0.5)')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('Metrics vs Classification Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '6_threshold_analysis.png'), bbox_inches='tight')
plt.close()

real_mask = y_true == 0
fake_mask = y_true == 1

real_pred = y_pred[real_mask]
real_true = y_true[real_mask]
fake_pred = y_pred[fake_mask]
fake_true = y_true[fake_mask]

real_precision, real_recall, real_f1, _ = precision_recall_fscore_support(
    real_true, real_pred, average='binary', pos_label=0, zero_division=0
)
fake_precision, fake_recall, fake_f1, _ = precision_recall_fscore_support(
    fake_true, fake_pred, average='binary', pos_label=1, zero_division=0
)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35
real_scores = [real_precision, real_recall, real_f1]
fake_scores = [fake_precision, fake_recall, fake_f1]

bars1 = ax.bar(x - width/2, real_scores, width, label='Real Images', color='green', edgecolor='black')
bars2 = ax.bar(x + width/2, fake_scores, width, label='Fake Images', color='red', edgecolor='black')

ax.set_ylabel('Score')
ax.set_title('Per-Class Performance Metrics')
ax.set_xticks(x)
ax.set_xticklabels(['Precision', 'Recall', 'F1 Score'])
ax.legend()
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '7_per_class_metrics.png'), bbox_inches='tight')
plt.close()

tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

# 2 examples from each category
np.random.seed(42)
examples = []
categories = [
    (tp_indices, 'True Positive', 'green'),
    (tn_indices, 'True Negative', 'blue'),
    (fp_indices, 'False Positive', 'orange'),
    (fn_indices, 'False Negative', 'red')
]

for indices, cat_name, color in categories:
    if len(indices) >= 2:
        selected = np.random.choice(indices, 2, replace=False)
        for idx in selected:
            examples.append((idx, cat_name, color))

if len(examples) > 0:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (idx, cat_name, color) in enumerate(examples):
        img_path = results[idx]['path']
        true_label = results[idx]['true_label']
        pred_label = results[idx]['predicted_label']
        prob = results[idx]['probability']
        
        # Load and display image
        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_class = 'Fake' if true_label == 1 else 'Real'
        pred_class = 'Fake' if pred_label == 1 else 'Real'
        
        title = f'{cat_name}\nTrue: {true_class} | Pred: {pred_class}\nProb: {prob:.3f}'
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    plt.suptitle('Sample Predictions from Each Category', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '8_sample_predictions.png'), bbox_inches='tight')
    plt.close()

# error analysis - top confident mistakes
print("Creating error analysis...")
# top 10 most confident false positives and false negatives (i think this is the way to do it)
fp_confidences = [(i, y_prob[i]) for i in fp_indices]
fn_confidences = [(i, 1 - y_prob[i]) for i in fn_indices]  # confidence but wrong class

fp_sorted = sorted(fp_confidences, key=lambda x: x[1], reverse=True)[:10]
fn_sorted = sorted(fn_confidences, key=lambda x: x[1], reverse=True)[:10]

# also get one representative true positive
tp_example = None
if len(tp_indices) > 0:
    # h igh confidence true positive
    tp_confidences = [(i, y_prob[i]) for i in tp_indices]
    tp_sorted = sorted(tp_confidences, key=lambda x: x[1], reverse=True)
    tp_example_idx = tp_sorted[0][0]  # Most confident TP
    tp_example = {
        "path": results[tp_example_idx]['path'],
        "true_label": "Fake",
        "predicted_label": "Fake",
        "predicted_prob_fake": float(y_prob[tp_example_idx]),
        "confidence": float(y_prob[tp_example_idx])
    }
    tp_img = Image.open(results[tp_example_idx]['path']).convert('RGB')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(tp_img)
    ax.axis('off')
    ax.set_title(f"True Positive Example\nTrue: Fake | Predicted: Fake\nConfidence: {y_prob[tp_example_idx]:.3f}", 
                 fontsize=14, color='green', fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '9_true_positive_example.png'), bbox_inches='tight')
    plt.close()

error_analysis = {
    "top_false_positives": [
        {
            "path": results[idx]['path'],
            "true_label": "Real",
            "predicted_prob_fake": float(conf),
            "rank": rank + 1
        }
        for rank, (idx, conf) in enumerate(fp_sorted)
    ],
    "top_false_negatives": [
        {
            "path": results[idx]['path'],
            "true_label": "Fake",
            "predicted_prob_fake": float(y_prob[idx]),
            "predicted_prob_real": float(1 - y_prob[idx]),
            "rank": rank + 1
        }
        for rank, (idx, conf) in enumerate(fn_sorted)
    ],
    "true_positive_example": tp_example
}

with open("statistics/error_analysis.json", "w") as f:
    json.dump(error_analysis, f, indent=2)

print(f"\nfigures saved to: {fig_dir}/")
print("\nGenerated figures:")
print("1. Confusion Matrix Heatmap")
print("2. ROC Curve")
print("3. Precision-Recall Curve")
print("4. Prediction Distribution Histogram")
print("5. Metrics Bar Chart")
print("6. Threshold Analysis")
print("7. Per-Class Metrics")
print("8. Sample Predictions Grid")
print("9. True Positive Example")

print("\n")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc_score:.4f}")
