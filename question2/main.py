import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter


BASE_DIR = Path(__file__).resolve().parent
RGB_DIR = BASE_DIR / "CameraRGB"
MASK_DIR = BASE_DIR / "CameraMask"
ARTIFACT_DIR = BASE_DIR / "Question2"
MODEL_PATH = ARTIFACT_DIR / "unet_segmentation.pt"
HISTORY_PATH = ARTIFACT_DIR / "history.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SPLIT_PATH = ARTIFACT_DIR / "test_split.json"
LOSS_PLOT_PATH = ARTIFACT_DIR / "train_loss_curve.png"
MIOU_PLOT_PATH = ARTIFACT_DIR / "train_miou_curve.png"
MDICE_PLOT_PATH = ARTIFACT_DIR / "train_mdice_curve.png"
PREVIEW_PATH = ARTIFACT_DIR / "test_predictions_preview.png"

NUM_CLASSES = 23
IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 15
SEED = 42
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_COLORS = np.stack(
    [
        np.array(
            [
                (index * 37) % 255,
                (index * 67) % 255,
                (index * 97) % 255,
            ],
            dtype=np.uint8,
        )
        for index in range(NUM_CLASSES)
    ]
)


@dataclass
class TrainingResult:
    test_miou: float
    test_mdice: float
    train_count: int
    test_count: int


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples():
    samples = []
    for image_path in sorted(RGB_DIR.glob("*.png")):
        mask_path = MASK_DIR / image_path.name
        if mask_path.exists():
            samples.append((image_path, mask_path))
    if not samples:
        raise FileNotFoundError("No paired samples found under CameraRGB/ and CameraMask/.")
    return samples


def split_samples(samples, train_ratio=0.8, seed=SEED):
    rng = random.Random(seed)
    samples = samples[:]
    rng.shuffle(samples)
    split_index = int(len(samples) * train_ratio)
    return samples[:split_index], samples[split_index:]


def mask_to_class_ids(mask):
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_array = mask_array.astype(np.int64)
    if mask_array.max() >= NUM_CLASSES:
        raise ValueError("Mask contains class ids outside the expected 0-22 range.")
    return torch.from_numpy(mask_array)


def image_to_tensor(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array)


class SegmentationDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        self.color_jitter = ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment and random.random() < 0.6:
            image = self.color_jitter(image)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_class_ids(mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor.long(),
            "name": image_path.name,
        }


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, features=(32, 64, 128, 256)):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_in = in_channels
        for feature in features:
            self.down_blocks.append(DoubleConv(current_in, feature))
            current_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=0.1)

        for feature in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.up_blocks.append(DoubleConv(feature * 2, feature))

        self.head = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for up_transpose, up_block, skip in zip(self.up_transpose, self.up_blocks, skip_connections):
            x = up_transpose(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = up_block(x)

        return self.head(x)


def multiclass_dice_loss(logits, targets, num_classes):
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dims)
    denominator = torch.sum(probabilities + one_hot, dims)
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    ce_loss = F.cross_entropy(logits, targets, label_smoothing=0.05)
    dice_loss = multiclass_dice_loss(logits, targets, NUM_CLASSES)
    return ce_loss + dice_loss


def update_confusion_matrix(confusion, logits, targets):
    predictions = torch.argmax(logits, dim=1)
    labels = targets * NUM_CLASSES + predictions
    bincount = torch.bincount(labels.reshape(-1), minlength=NUM_CLASSES * NUM_CLASSES)
    confusion += bincount.reshape(NUM_CLASSES, NUM_CLASSES)
    return confusion


def scores_from_confusion(confusion):
    confusion = confusion.float()
    true_positive = torch.diag(confusion)
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    support = confusion.sum(dim=1)

    present = support > 0
    iou = true_positive / (true_positive + false_positive + false_negative + 1e-6)
    dice = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + 1e-6)

    mean_iou = iou[present].mean().item() if present.any() else 0.0
    mean_dice = dice[present].mean().item() if present.any() else 0.0
    return mean_iou, mean_dice


def create_data_loaders():
    samples = collect_samples()
    train_samples, test_samples = split_samples(samples, train_ratio=0.8, seed=SEED)

    train_dataset = SegmentationDataset(train_samples, augment=True)
    eval_train_dataset = SegmentationDataset(train_samples, augment=False)
    test_dataset = SegmentationDataset(test_samples, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        eval_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, train_eval_loader, test_loader, train_samples, test_samples


def evaluate(model, data_loader):
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=DEVICE)
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)
            logits = model(images)
            total_loss += combined_loss(logits, masks).item()
            update_confusion_matrix(confusion, logits, masks)

    mean_iou, mean_dice = scores_from_confusion(confusion)
    return total_loss / len(data_loader), mean_iou, mean_dice


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_curves(history):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", color="#d35400")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_miou"], marker="o", color="#0b84a5")
    plt.title("Training mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MIOU_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_mdice"], marker="o", color="#2ca02c")
    plt.title("Training mDice")
    plt.xlabel("Epoch")
    plt.ylabel("mDice")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MDICE_PLOT_PATH, dpi=180)
    plt.close()


def colorize_mask(mask_array):
    mask_array = np.asarray(mask_array, dtype=np.int64)
    return CLASS_COLORS[mask_array]


def save_prediction_preview(model, test_samples, count=4):
    figure, axes = plt.subplots(count, 3, figsize=(12, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (image_path, mask_path) in enumerate(test_samples[:count]):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image_tensor = image_to_tensor(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prediction = torch.argmax(model(image_tensor), dim=1).squeeze(0).cpu().numpy()

        ground_truth = mask_to_class_ids(mask).numpy()
        resized_image = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

        axes[row, 0].imshow(resized_image)
        axes[row, 0].set_title(f"Input: {image_path.name}")
        axes[row, 1].imshow(colorize_mask(ground_truth))
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(colorize_mask(prediction))
        axes[row, 2].set_title("Prediction")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(PREVIEW_PATH, dpi=180)
    plt.close()


def train(epochs=NUM_EPOCHS):
    set_seed(SEED)
    train_loader, train_eval_loader, test_loader, train_samples, test_samples = create_data_loaders()

    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history = {
        "epoch": [],
        "train_loss": [],
        "train_miou": [],
        "train_mdice": [],
    }

    print(f"device={DEVICE}")
    print(f"total samples: {len(train_samples) + len(test_samples)}")
    print(f"train samples: {len(train_samples)}")
    print(f"test samples: {len(test_samples)}")

    best_state = None
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = combined_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        train_loss, train_miou, train_mdice = evaluate(model, train_eval_loader)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["train_mdice"].append(train_mdice)

        score = train_miou + train_mdice
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        print(
            f"epoch={epoch}/{epochs} "
            f"batch_loss={running_loss / len(train_loader):.4f} "
            f"train_loss={train_loss:.4f} "
            f"train_miou={train_miou:.4f} "
            f"train_mdice={train_mdice:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_PATH)
    save_curves(history)
    save_prediction_preview(model, test_samples)

    _, test_miou, test_mdice = evaluate(model, test_loader)

    save_json(HISTORY_PATH, history)
    save_json(
        METRICS_PATH,
        {
            "test_miou": test_miou,
            "test_mdice": test_mdice,
            "train_count": len(train_samples),
            "test_count": len(test_samples),
            "epochs": epochs,
            "seed": SEED,
            "image_size": IMAGE_SIZE,
        },
    )
    save_json(SPLIT_PATH, {"test_samples": [image_path.name for image_path, _ in test_samples]})

    print(f"saved model to: {MODEL_PATH}")
    print(f"saved loss plot to: {LOSS_PLOT_PATH}")
    print(f"saved mIoU plot to: {MIOU_PLOT_PATH}")
    print(f"saved mDice plot to: {MDICE_PLOT_PATH}")
    print(f"saved preview plot to: {PREVIEW_PATH}")
    print(f"Test set : mIOU: {test_miou:.4f}")
    print(f"Test set : mDICE: {test_mdice:.4f}")

    return TrainingResult(
        test_miou=test_miou,
        test_mdice=test_mdice,
        train_count=len(train_samples),
        test_count=len(test_samples),
    )


def load_metrics():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Training metrics not found. Run `python3.11 main.py train` first.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_test_split():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError("Test split file not found. Run training first.")
    return set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["test_samples"])


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model checkpoint not found. Run training first.")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def build_metrics_html(metrics):
    return f"""
    <div style="padding:16px;border-radius:12px;background:#f6f7fb;border:1px solid #dde3ec;">
      <h2 style="margin:0 0 8px 0;">Question2 Test Metrics</h2>
      <p style="margin:4px 0;"><strong>mIoU:</strong> {metrics['test_miou']:.4f}</p>
      <p style="margin:4px 0;"><strong>mDice:</strong> {metrics['test_mdice']:.4f}</p>
      <p style="margin:4px 0;"><strong>Train/Test split:</strong> {metrics['train_count']} / {metrics['test_count']}</p>
      <p style="margin:4px 0;"><strong>Epochs:</strong> {metrics['epochs']}</p>
    </div>
    """


def predict_uploaded_images(files):
    if not files:
        return None, None, "Upload exactly 4 images from the saved test split."

    if len(files) != 4:
        return None, None, "Please upload exactly 4 test images."

    allowed_files = load_test_split()
    model = load_trained_model()

    gt_gallery = []
    pred_gallery = []

    for file_info in files:
        image_name = Path(file_info).name
        if image_name not in allowed_files:
            return None, None, f"{image_name} is not part of the saved test split."

        image = Image.open(file_info).convert("RGB")
        mask = Image.open(MASK_DIR / image_name)
        image_tensor = image_to_tensor(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = torch.argmax(model(image_tensor), dim=1).squeeze(0).cpu().numpy()

        gt_gallery.append((colorize_mask(mask_to_class_ids(mask).numpy()), f"{image_name} ground truth"))
        pred_gallery.append((colorize_mask(prediction), f"{image_name} prediction"))

    return gt_gallery, pred_gallery, "Ground-truth masks and predictions generated successfully."


def launch_app():
    metrics = load_metrics()

    with gr.Blocks(title="Question2 Segmentation Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Question2 Segmentation Dashboard")
        gr.Markdown("UNet-based semantic segmentation for 23 classes with an 80/20 split and seed 42.")

        with gr.Tab("Training Summary"):
            gr.HTML(build_metrics_html(metrics))
            with gr.Row():
                gr.Image(value=str(LOSS_PLOT_PATH), label="Training Loss", interactive=False)
                gr.Image(value=str(MIOU_PLOT_PATH), label="Training mIoU", interactive=False)
                gr.Image(value=str(MDICE_PLOT_PATH), label="Training mDice", interactive=False)
            gr.Image(value=str(PREVIEW_PATH), label="Preview on Test Samples", interactive=False)

        with gr.Tab("Predict Test Images"):
            gr.Markdown("Upload exactly 4 images from the saved test split to compare ground-truth and predicted masks.")
            file_input = gr.File(
                label="Upload 4 RGB test images",
                file_count="multiple",
                file_types=["image"],
            )
            run_button = gr.Button("Run Segmentation")
            status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                gt_gallery = gr.Gallery(label="Ground Truth Masks", columns=2, height=480)
                pred_gallery = gr.Gallery(label="Predicted Masks", columns=2, height=480)

            run_button.click(
                fn=predict_uploaded_images,
                inputs=file_input,
                outputs=[gt_gallery, pred_gallery, status],
            )

    demo.launch(server_name="0.0.0.0", server_port=7860)


def parse_args():
    parser = argparse.ArgumentParser(description="Question2 segmentation training and app runner")
    parser.add_argument(
        "command",
        nargs="?",
        default="train",
        choices=["train", "app"],
        help="Use `train` to train and save artifacts, or `app` to launch the Gradio UI.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs to run when using the `train` command.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        result = train(epochs=args.epochs)
        print(json.dumps(asdict(result), indent=2))
    else:
        launch_app()


if __name__ == "__main__":
    main()
