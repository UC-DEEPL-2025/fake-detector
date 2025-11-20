# deepfake detection with pytorch

everything's config-driven with hydra so you don't have to mess with the code to try different settings
(after building the actual model)

## what it does

trains a binary classifier (real/fake) on images using pytorch:
- unet architecture with configurable depth and channels
- mixed precision training for faster training on gpu
- saves best model automatically based on validation loss
- all hyperparameters in yaml files (no hardcoded values)

## how to use

### setup
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### train a model
```bash
python scripts/train.py
```

this will:
- load your dataset from the `Dataset/` folder
- train the unet model
- save checkpoints to `runs/<experiment_name>_<timestamp>/`
- print train/val loss each epoch

you can override any config setting from the command line:
```bash
python scripts/train.py train.epochs=20 data.batch_size=32 model.params.depth=4
```

### test the model
```bash
python scripts/test.py checkpoint_path=runs/your_experiment/best_unet.pt
```

saves predictions to a csv file and prints accuracy.

## configs explained

all settings are in yaml files under `configs/`. no magic numbers in the code.

**`configs/data/default.yaml`**
- dataset path, image size
- batch size, number of workers
- augmentation probabilities (horizontal flip, brightness/contrast) NOT SURE HOW CORRECT THIS IS... THIS NEEDS WORK.
- subset_fraction: use a fraction of data for quick testing (1.0 = full dataset)

**`configs/train/default.yaml`**
- epochs, learning rate, optimizer (rn... adam/adamw)
- mixed precision training (true/false)
- random seed for reproducibility

^ I think I will merge this with /model/<model>.yaml

**`configs/model/unet.yaml`**
- depth: how many encoder/decoder blocks
- base_channels: starting number of channels
- kernel sizes, activation functions

you can create new config files and switch between them:

```bash
python scripts/train.py model=unet data=default train=default
```

## how it works

**dataset**: expects images organized as `Dataset/<Split>/<Class>/<image>.<ext>`. (jpg, jpeg, png). 

**augmentations**: Not sure if needed or correct as is... was not from my original model... this was a suggestion by gpt

**training**: well... training

**testing**: loads checkpoint, runs inference on test set, computes simple accuracy (threshold=0.5), saves predictions with probabilities to csv.

todo: testing should accept .keras weights and dataset

## dataset format

your data should look like:
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
├── Validation/
│   ├── Real/
│   └── Fake/
└── Test/
    ├── Real/
    └── Fake/
```

classes (splits) ["Real", "Fake"] above, should match `configs/data/<config>.yaml` classes list

## outputs

after training, you get:
- `runs/<exp_name>_<timestamp>/best_unet.pt` - saved checkpoint (model weights + config)
- `runs/<exp_name>_<timestamp>/val_preds.csv` - validation predictions
- console logs showing train/val loss per epoch

after testing:
- `test_preds.csv` in the checkpoint directory with columns: path, label, prob, pred
- printed accuracy on test set

## tips

**quick test run**: use a small subset to make sure everything works
```bash
python scripts/train.py data.subset_fraction=0.01 train.epochs=2
```

**adjust batch size**: if you run out of memory, lower the batch size
```bash
python scripts/train.py data.batch_size=16
```

**change image size**: smaller images = faster training
```bash
python scripts/train.py data.img_size=128
```

**verbose logging**: set to false to reduce console spam
```bash
python scripts/train.py verbose=false
```

# notes

todo: 
- proper metrics
- models: ResNet, Transformers
