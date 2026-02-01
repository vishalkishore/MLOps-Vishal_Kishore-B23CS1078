import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_API_KEY"] = "e6a1f43a2c9b91102b234914523e00a9ca00d8f2"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from thop import profile, clever_format

from dataloader import get_cifar10_dataloaders


def get_resnet18_cifar10():
    model = models.resnet18(weights=None)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    return model


# ============ FLOP Counting ============
def count_flops(model, input_size=(1, 3, 32, 32)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    
    return flops, params, flops_str, params_str


# ============ Gradient Flow Visualization ============
def plot_gradient_flow(named_parameters, epoch):
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name.replace('.weight', '').replace('.bias', ''))
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())
    
    if not layers:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, ave_grads, width, label='Mean Gradient', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, max_grads, width, label='Max Gradient', color='coral', alpha=0.8)
    
    ax.set_xlabel('Layers')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title(f'Gradient Flow - Epoch {epoch}')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============ Weight Stats ============
def get_weight_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[name] = {
                'mean': param.data.mean().cpu().item(),
                'std': param.data.std().cpu().item(),
                'min': param.data.min().cpu().item(),
                'max': param.data.max().cpu().item(),
                'norm': param.data.norm().cpu().item()
            }
    return stats


def plot_weight_update_flow(prev_stats, curr_stats, epoch):
    if prev_stats is None:
        return None
    
    layers = []
    update_magnitudes = []
    
    for name in prev_stats.keys():
        if name in curr_stats:
            prev_norm = prev_stats[name]['norm']
            curr_norm = curr_stats[name]['norm']
            if prev_norm > 0:
                update_mag = abs(curr_norm - prev_norm) / prev_norm
                layers.append(name.replace('.weight', '').replace('.bias', ''))
                update_magnitudes.append(update_mag)
    
    if not layers:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(range(len(layers)), update_magnitudes, color='green', alpha=0.7)
    ax.set_xlabel('Layers')
    ax.set_ylabel('Relative Weight Update Magnitude')
    ax.set_title(f'Weight Update Flow - Epoch {epoch}')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============ Training Functions ============
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total


# ============ Main Training Loop ============
def main():
    # Configuration
    config = {
        'batch_size': 128,
        'epochs': 25,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'model': 'ResNet-18-CIFAR10',
        'optimizer': 'SGD',
        'scheduler': 'CosineAnnealingLR'
    }
    
    # Initialize WandB
    wandb.init(
        project="cifar10-cnn-training",
        config=config,
        name=f"resnet18-{config['epochs']}epochs"
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, test_loader, classes = get_cifar10_dataloaders(
        batch_size=config['batch_size']
    )
    
    # Model
    model = get_resnet18_cifar10().to(device)
    
    # WandB watch for gradient and weight logging
    wandb.watch(model, log="all", log_freq=100)
    
    # Count FLOPs
    flops, params, flops_str, params_str = count_flops(model)
    print(f"Model FLOPs: {flops_str}, Parameters: {params_str}")
    wandb.log({
        "model/flops": flops,
        "model/params": params,
        "model/flops_formatted": flops_str,
        "model/params_formatted": params_str
    })
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_val_acc = 0.0
    prev_weight_stats = None
    
    print(f"\nStarting training for {config['epochs']} epochs...")
    print("=" * 60)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Get weight stats before training (for weight update visualization)
        curr_weight_stats = get_weight_stats(model)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Get weight stats after training
        new_weight_stats = get_weight_stats(model)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Gradient flow visualization
        grad_fig = plot_gradient_flow(model.named_parameters(), epoch)
        
        # Weight update flow visualization
        weight_update_fig = plot_weight_update_flow(prev_weight_stats, new_weight_stats, epoch)
        
        # Log to WandB
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": current_lr,
        }
        
        if grad_fig:
            log_dict["gradient_flow"] = wandb.Image(grad_fig)
            plt.close(grad_fig)
        
        if weight_update_fig:
            log_dict["weight_update_flow"] = wandb.Image(weight_update_fig)
            plt.close(weight_update_fig)
        
        # Log weight statistics per layer
        for name, stats in new_weight_stats.items():
            layer_name = name.replace('.', '_')
            log_dict[f"weights/{layer_name}/mean"] = stats['mean']
            log_dict[f"weights/{layer_name}/std"] = stats['std']
            log_dict[f"weights/{layer_name}/norm"] = stats['norm']
        
        wandb.log(log_dict)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Update previous weight stats for next epoch
        prev_weight_stats = new_weight_stats
    
    # Final test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    wandb.log({
        "test/loss": test_loss,
        "test/accuracy": test_acc,
        "best_val_accuracy": best_val_acc
    })
    
    # Summary
    wandb.run.summary["best_val_accuracy"] = best_val_acc
    wandb.run.summary["final_test_accuracy"] = test_acc
    wandb.run.summary["total_flops"] = flops_str
    wandb.run.summary["total_params"] = params_str
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Model FLOPs: {flops_str}")
    print(f"Model Parameters: {params_str}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
