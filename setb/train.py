import os

os.environ["WANDB_API_KEY"] = "WANDB_APIKEY"
# ==========================
# HuggingFace Login
# ==========================
login(token="HUUGING_FACE")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
from huggingface_hub import login, HfApi
from datasets import load_dataset




# ==========================
# Load Dataset from HuggingFace
# ==========================
ds = load_dataset("Chiranjeev007/STL-10_Subset")
print(ds)
print("Train samples:", len(ds["train"]))
print("Val samples:", len(ds["validation"]))
print("Test samples:", len(ds["test"]))

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

# ==========================
# Configuration
# ==========================
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 5
LR = 1e-3
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_model.pth"
HF_REPO_ID = "Chiranjeev007/STL-10-ResNet18"
MY_REPO_ID = "vishalkishore01/model"

print(f"Using device: {DEVICE}")

# ==========================
# Transforms
# ==========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================
# Custom Dataset Wrapper
# ==========================
class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================
# DataLoaders
# ==========================
train_dataset = HFImageDataset(ds["train"], transform=train_transform)
val_dataset = HFImageDataset(ds["validation"], transform=val_transform)
test_dataset = HFImageDataset(ds["test"], transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================
# Initialize W&B
# ==========================
wandb.init(project="stl10-resnet18", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "patience": PATIENCE,
    "model": "resnet18-pretrained",
})

# ==========================
# Load Pretrained ResNet-18
# ==========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Training & Validation Loop
# ==========================
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct / total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # --- Validation ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    epoch_val_loss = running_loss / len(val_loader)
    epoch_val_acc = 100 * correct / total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    # --- Log to W&B ---
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc,
    })

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

    # --- Early Stopping ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH, _use_new_zipfile_serialization=False)
        print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  ✗ No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("Training Complete!")

# ==========================
# Plot Train/Val Loss & Accuracy
# ==========================
epochs_range = range(1, len(train_losses) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
ax1.plot(epochs_range, val_losses, 'r-o', label='Val Loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, train_accs, 'b-o', label='Train Acc')
ax2.plot(epochs_range, val_accs, 'r-o', label='Val Acc')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("Training curves saved to training_curves.png")

# ==========================
# Confusion Matrix on Test Set → W&B
# ==========================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Log confusion matrix to W&B
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=CLASS_NAMES,
        title="Test Set Confusion Matrix"
    )
})

test_acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\nOverall Test Accuracy: {test_acc:.2f}%")
wandb.log({"test_acc": test_acc})

# ==========================
# Log 10 Correct & 10 Incorrect to W&B
# ==========================
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

correct_images, incorrect_images = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images_gpu = images.to(DEVICE)
        outputs = model(images_gpu)
        _, preds = torch.max(outputs, 1)

        for i in range(len(labels)):
            img = inv_normalize(images[i]).clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pred_label = CLASS_NAMES[preds[i].item()]
            true_label = CLASS_NAMES[labels[i].item()]

            if preds[i].item() == labels[i].item() and len(correct_images) < 10:
                correct_images.append((img_np, pred_label, true_label))
            elif preds[i].item() != labels[i].item() and len(incorrect_images) < 10:
                incorrect_images.append((img_np, pred_label, true_label))

            if len(correct_images) >= 10 and len(incorrect_images) >= 10:
                break
        if len(correct_images) >= 10 and len(incorrect_images) >= 10:
            break

# Log correct predictions
correct_table = wandb.Table(columns=["Image", "Predicted", "Actual"])
for img_np, pred, actual in correct_images:
    correct_table.add_data(wandb.Image(img_np), pred, actual)
wandb.log({"Correct Predictions": correct_table})

# Log incorrect predictions
incorrect_table = wandb.Table(columns=["Image", "Predicted", "Actual"])
for img_np, pred, actual in incorrect_images:
    incorrect_table.add_data(wandb.Image(img_np), pred, actual)
wandb.log({"Incorrect Predictions": incorrect_table})

print(f"Logged {len(correct_images)} correct and {len(incorrect_images)} incorrect predictions to W&B")

# ==========================
# Class-wise Accuracy
# ==========================
cm = confusion_matrix(all_labels, all_preds)
class_accs = []
print("\nClass-wise Accuracy:")
for i, cls in enumerate(CLASS_NAMES):
    cls_total = cm[i].sum()
    cls_correct = cm[i][i]
    acc = 100 * cls_correct / cls_total if cls_total > 0 else 0
    class_accs.append(acc)
    print(f"  {cls:>10s}: {acc:.2f}% ({cls_correct}/{cls_total})")

# ==========================
# Bar Plot: Class-wise Accuracy
# ==========================
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, NUM_CLASSES))
bars = ax.bar(CLASS_NAMES, class_accs, color=colors, edgecolor='black')

for bar, acc in zip(bars, class_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Class')
ax.set_ylabel('Accuracy (%)')
ax.set_title(f'Class-wise Test Accuracy (Overall: {test_acc:.2f}%)')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('classwise_accuracy.png', dpi=150)
print("Class-wise accuracy plot saved to classwise_accuracy.png")

# ==========================
# Push Best Model to HuggingFace
# ==========================
api = HfApi()
try:
    api.create_repo(repo_id=MY_REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=MODEL_SAVE_PATH,
        path_in_repo="best_model.pth",
        repo_id=MY_REPO_ID,
    )
    print(f"Model pushed to https://huggingface.co/{MY_REPO_ID}")
except Exception as e:
    print(f"Failed to push model: {e}")

wandb.finish()
print("Done!")
