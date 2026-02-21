import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os

# ==========================
# Configuration
# ==========================
DATA_DIR = "data/train/"
BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(pretrained=False)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ==========================
# Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Training Loop
# ==========================
train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} "
          f"Accuracy: {epoch_acc:.2f}%")

print("Training Complete!")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "trained_model.pth", _use_new_zipfile_serialization=False)
print("Model saved!")

# ==========================
# Evaluate on Training Set
# ==========================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Overall Accuracy & F1
overall_acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")
print(f"F1 Score (macro): {macro_f1:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

# Class-wise Accuracy
cm = confusion_matrix(all_labels, all_preds)
print("Class-wise Accuracy:")
for i, cls in enumerate(dataset.classes):
    cls_total = cm[i].sum()
    cls_correct = cm[i][i]
    acc = 100 * cls_correct / cls_total if cls_total > 0 else 0
    print(f"  Class {cls}: {acc:.2f}% ({cls_correct}/{cls_total})")


# ==========================
# Plot & Save Training Curves
# ==========================
epochs_range = range(1, EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_range, train_losses, 'b-o')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(epochs_range, train_accuracies, 'r-o')
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves_batch64.png', dpi=150)
print("Training curves saved to training_curves.png")
