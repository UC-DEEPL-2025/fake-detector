import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import model and dataset utilities
from dfdetect.models.registry import build_model
from dfdetect.datasets.segmentation import DeepFakeDataset
from dfdetect.datasets.transforms import build_transforms

# Load config
with open("configs/model/unet.yaml", "r") as f:
	model_cfg = yaml.safe_load(f)
with open("configs/data/default.yaml", "r") as f:
	data_cfg = yaml.safe_load(f)


# Build model and load weights (same as test.py)
ckpt_path = "runs/baseline_20251123T005114Z/last_unet.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = build_model(model_cfg)
model.load_state_dict(checkpoint["model"])
model.eval()

# Prepare transforms and dataset
transforms = build_transforms(
	img_size=data_cfg["img_size"],
	normalize=data_cfg.get("normalize", False),
	norm_mean=data_cfg.get("norm_mean", [0.0, 0.0, 0.0]),
	norm_std=data_cfg.get("norm_std", [1.0, 1.0, 1.0]),
	is_train=False,
	hflip_p=data_cfg["aug"]["hflip_p"],
	brightness_contrast_p=data_cfg["aug"]["brightness_contrast_p"],
)
dataset = DeepFakeDataset(
	root_dir=data_cfg["root_dir"],
	split=data_cfg["val_split"],
	classes=data_cfg["classes"],
	transform=transforms,
	extensions=data_cfg["allowed_extensions"],
	subset_fraction=0.05,  # Use a small subset for statistics
)

# Run inference and collect statistics
results = []
for sample in dataset:
	img = sample["image"].unsqueeze(0)
	orig_img = np.array(Image.open(sample["path"]).convert("RGB"))
	with torch.no_grad():
		pred = model(img)
		pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
		pred_bin = (pred_mask > 0.5).astype(np.uint8)
	# Statistics: pixel change summary
	changed_pixels = np.sum(pred_bin)
	total_pixels = pred_bin.size
	percent_changed = 100.0 * changed_pixels / total_pixels
	results.append({
		"path": sample["path"],
		"changed_pixels": int(changed_pixels),
		"total_pixels": int(total_pixels),
		"percent_changed": percent_changed,
	})
	# Visualization: overlay and difference
	overlay = orig_img.copy()
	overlay[pred_bin == 1] = [255, 0, 0]  # Red overlay for changed
	diff_map = np.zeros_like(orig_img)
	diff_map[pred_bin == 1] = [255, 255, 255]
	# Save poster-style image
	fig, axs = plt.subplots(1, 3, figsize=(12, 4))
	axs[0].imshow(orig_img)
	axs[0].set_title("Original")
	axs[1].imshow(overlay)
	axs[1].set_title("Model Overlay")
	axs[2].imshow(diff_map)
	axs[2].set_title("Change Map")
	for ax in axs:
		ax.axis('off')
	plt.suptitle(f"{os.path.basename(sample['path'])} | Changed: {percent_changed:.2f}%")
	out_dir = "statistics/poster"
	os.makedirs(out_dir, exist_ok=True)
	plt.savefig(os.path.join(out_dir, f"{os.path.basename(sample['path'])}_poster.png"), bbox_inches='tight')
	plt.close()

# Save statistics summary
import json
os.makedirs("statistics", exist_ok=True)
with open("statistics/summary.json", "w") as f:
	json.dump(results, f, indent=2)
