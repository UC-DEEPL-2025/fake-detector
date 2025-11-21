import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, train_transform: transforms.Compose, test_transform: transforms.Compose, batch_size: int, num_workers: int=NUM_WORKERS, distributed: bool = False):
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    #get class names
    class_names = train_data.classes
    
    #if using DDP, set up distributed samplers
    if distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        test_sampler = DistributedSampler(test_data, shuffle=False)
        shuffle_flag = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle_flag = True
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=test_sampler
    )
    return train_dataloader, test_dataloader, class_names
