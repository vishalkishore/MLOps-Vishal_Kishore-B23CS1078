"""
Custom CIFAR-10 DataLoader with augmentation and train/val split.
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_cifar10_dataloaders(batch_size=128, num_workers=4, val_split=0.2, seed=42):
    """
    Create custom CIFAR-10 dataloaders with train/val/test splits.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    
    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Validation/Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Download and load CIFAR-10
    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=test_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Split training data into train and validation
    num_train = len(full_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Dataset sizes - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    # Test the dataloader
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(batch_size=64)
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {[classes[l] for l in labels[:5]]}")
