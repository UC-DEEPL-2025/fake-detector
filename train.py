import os

if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '29500'

import torch
import time
from torchvision import transforms
import torch.distributed as dist
import torch.multiprocessing as mp
import preprocess, engine, model_builder, utils

# Global constants
NUM_EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_UNITS = 10
LEARNING_RATE = 0.0001
MODEL_PATH = r"/home/luudh/luudh/MyFile/fake-detector/model/ResNet152.pth"
BASE_DIR = r"/home/luudh/luudh/MyFile/fake-detector/manjilkarki/deepfake-and-real-images/versions/1/Dataset/"

def main_worker(gpu, world_size):
    start_time = time.time()
    # Build the paths for train and test data
    train_dir = os.path.join(BASE_DIR, "Train")
    validation_dir = os.path.join(BASE_DIR, "Validation")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f"Testing directory not found: {validation_dir}")

    # Set the GPU device for this process
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=gpu)

    # Create train and test transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create dataloaders (make sure your data_setup.create_dataloaders uses DistributedSampler if distributed=True)
    train_dataloader, test_dataloader, class_names = preprocess.create_dataloaders(
        train_dir=train_dir,
        test_dir=validation_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=BATCH_SIZE,
        distributed=True
    )

    print("Class Names: ", class_names)
    print("CPU Cores Used:", torch.get_num_threads())
    print("CUDA Available:", torch.cuda.is_available())
    print("Using Device:", device)

    # Create model and wrap it in DistributedDataParallel
    model = model_builder.ResNet(block=model_builder.Bottlenet_block, layers=[3,8,36,3], num_classes=len(class_names)).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Create optimizer and loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load model if it exists
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model from: {MODEL_PATH} ...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("[INFO] Model loaded successfully")
    else:
        print("[INFO] Model not found, starting training from scratch ...")

    # Run the training engine
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )
    end_time = time.time()

    # Save the model from GPU 0 only
    if gpu == 0:
        utils.save_model(model=model, optimizer=optimizer, target_dir="model", model_name="ResNet152.pth")
        print(f"[INFO] Model saved to: {MODEL_PATH}")

    print(f"Executed in: {(end_time-start_time):.3f}")
    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DistributedDataParallel training")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

"""
To run:
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=3,4 python train.py
"""