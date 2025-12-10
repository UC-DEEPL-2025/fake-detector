# deepfake detection with pytorch

currently implements vision transformer (ViT) for binary classification on the pipeline-transformers branch.
the project is being restructured to support config-driven workflows with hydra for easier experimentation.

## what it does

trains a binary classifier (real/fake) on images using pytorch and transformers:
- vision transformer (ViT) architecture using pretrained google/vit-base-patch16-224-in21k
- fine-tuned for deepfake detection with custom classification head
- trains on gpu with cuda support
- saves best model automatically based on validation accuracy
- tracks metrics: accuracy, f1-score, confusion matrix

## how to use

### setup
```bash
python -m venv .venv
source .venv/bin/activate  # on Linux/Mac
pip install torch torchvision transformers pillow scikit-learn matplotlib tqdm
```

### train a model
```bash
python FakeImageViTi.py
```

before running, update the dataset paths in the script:
- `TRAIN_DIR`: path to your training data
- `VAL_DIR`: path to your validation/test data

this will:
- load your dataset from the specified folders
- train the ViT model for 10 epochs
- save the best checkpoint as `best_vit_deepfake_model.pth`
- generate training curves as `training_curves.png`
- print train/val loss, accuracy, and f1-score each epoch

### test the model
```bash
python stats.py
```
stats.py will:
- Load the saved model (best_vit_deepfake_model.pth)
- Evaluate it on the validation dataset

Generate:
- evaluation_report.txt

No training occurs in this script — only inference + statistics.

## dataset format

your data should be organized as:
```
Dataset/
├── Train/
│   ├── Real/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── Fake/
│       ├── img001.jpg
│       └── ...
└── Test/
    ├── Real/
    └── Fake/
```

supported image formats: jpg, jpeg, png

classes must be named "Real" and "Fake" (case-sensitive).

## current configuration

hardcoded in `FakeImageViTi.py` (moving to yaml configs soon):
- batch size: 16
- epochs: 10
- learning rate: 2e-5
- image size: 224x224 (ViT standard)
- optimizer: AdamW
- num workers: 4

## outputs

after training:
- `best_vit_deepfake_model.pth` - saved model weights (best validation accuracy)
- `training_curves.png` - loss and accuracy plots over epochs
- console logs showing metrics per epoch
- confusion matrix printed at the end

## tips

**check gpu availability**: the script automatically uses cuda if available, otherwise falls back to cpu
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**adjust batch size**: if you run out of memory, edit `BATCH_SIZE` in the script
```python
BATCH_SIZE = 8  # or lower
```

**change number of epochs**: edit `NUM_EPOCHS` in the script
```python
NUM_EPOCHS = 20
```

**modify learning rate**: edit `LEARNING_RATE` in the script
```python
LEARNING_RATE = 1e-5  # lower for more stable training
```

## project structure

```
fake-detector/
├── FakeImageViTi.py          # main training script with ViT model
├── dfdetect/                 # package being developed
│   ├── datasets/            # dataset classes (in progress)
│   └── models/              # model architectures (in progress)
└── README.md
```

## roadmap

todo:
- migrate to config-driven architecture with hydra
- add configs/ directory for yaml configurations
- proper metrics tracking and logging
- support for additional models: ResNet, EfficientNet
- mixed precision training
- data augmentation pipeline
- model checkpointing with full config
- inference script for single images
- wandb/tensorboard integration

## notes

this branch (pipeline-transformers) is focused on implementing the transformer-based approach.
the codebase is being refactored to support multiple architectures and config-driven experimentation.

currently, all hyperparameters are hardcoded in the script.
future versions will use hydra for configuration management, allowing easy experimentation without code changes.
