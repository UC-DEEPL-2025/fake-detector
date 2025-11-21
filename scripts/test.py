import os
import sys
import code

# Add repo root to sys.path so 'code' package is visible
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import torch
from torchvision import transforms
import torch.distributed as dist

if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '29500'

from dfdetect.preprocessing import preprocess
from dfdetect.engine import engine
from dfdetect.model import model_builder
import torch.multiprocessing as mp

# ----- Config (match train.py) -----
NUM_EPOCHS = 1          # not used here, but kept for consistency
BATCH_SIZE = 64
MODEL_PATH = r"/home/luudh/luudh/MyFile/fake-detector/model/ResNet152.pth"
BASE_DIR = r"/home/luudh/luudh/MyFile/fake-detector/manjilkarki/deepfake-and-real-images/versions/1/Dataset/"

def load_model(gpu, world_size, model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Rebuild the ResNet model and load trained weights."""
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=gpu)
    model = model_builder.ResNet(
        block=model_builder.Bottlenet_block,
        layers=[3, 8, 36, 3],   # same as in train.py
        num_classes=num_classes
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    print(f"[INFO] Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[INFO Model loaded successfully]")
    return model


def create_test_dataloader(batch_size: int):
    """Create a test dataloader using the same transforms as training."""
    train_dir = os.path.join(BASE_DIR, "Train")
    test_dir = os.path.join(BASE_DIR, "Test")  # used as test set

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test/Test directory not found: {test_dir}")

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
        distributed=False  # no DDP for testing
    )

    return test_dataloader, class_names


def evaluate(gpu, world_size):
    """Evaluate the saved model on the test dataset."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Build test dataloader
    test_dataloader, class_names = create_test_dataloader(batch_size=BATCH_SIZE)
    print(f"[INFO] Number of classes: {len(class_names)} -> {class_names}")

    # 2. Build and load model
    model = load_model(gpu, world_size, MODEL_PATH, num_classes=len(class_names), device=device)

    # 3. Define loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # 4. Run evaluation using your existing engine.test_step
    test_loss, test_acc = engine.test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device
    )

    print(f"\n[RESULTS] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DistributedDataParallel testing")
    mp.spawn(evaluate, args=(world_size,), nprocs=world_size, join=True)