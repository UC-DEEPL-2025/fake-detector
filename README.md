# fake-detector

## Model pipelines by branch

This repository uses separate branches for each major model pipeline to keep `main` clean and focused on shared utilities (datasets, transforms, evaluation, configs, etc.).

### U-Net pipeline

- **Branch:** `feat/basic-pipeline-unet-demo`  
- **Description:** Contains the U-Net demo pipeline, including data loading, training, and evaluation code for segmentation experiments.

### ResNet pipeline

- **Branch:** `feat/pipeline-resnet`  
- **Description:** Contains the ResNet-based classification pipeline, with training scripts, configs, and evaluation helpers.

### Vision Transformer (ViT) pipeline

- **Branch:** `pipeline-transformers`  
- **Description:** Contains the Vision Transformer pipeline and related experiments using transformer-based architectures.

---

When working with a specific architecture, switch to its corresponding branch and follow the instructions in that branch’s own README for setup, training, and evaluation details.
