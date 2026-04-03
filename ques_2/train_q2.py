from __future__ import annotations
import os

os.environ.setdefault("WANDB_API_KEY", "YOUR_WANDB_API_KEY")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLE_SYSTEM_STATS"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_GIT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["HOSTNAME"] = "vishal"
os.environ["WANDB_HOST"] = "vishal"

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import socket
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from art.attacks.evasion import BasicIterativeMethod, FastGradientMethod, ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

socket.gethostname = lambda: "vishal"


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Q2Config:
    command: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    output_dir: str
    data_dir: str
    image_size: int
    val_split: float
    device: str
    eps: float
    alpha: float
    attack_steps: int
    wandb_project: str | None = None
    wandb_entity: str | None = None
    use_amp: bool = True
    compile_model: bool = False
    max_visuals: int = 10
    classifier_checkpoint: str | None = None
    detector_attack: str | None = None
    eps_list: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 5 Q2: adversarial attacks and detection on CIFAR-10.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--epochs", type=int, default=20)
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument("--learning-rate", type=float, default=3e-4)
        subparser.add_argument("--weight-decay", type=float, default=1e-4)
        subparser.add_argument("--num-workers", type=int, default=8)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "q2"))
        subparser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
        subparser.add_argument("--image-size", type=int, default=32)
        subparser.add_argument("--val-split", type=float, default=0.1)
        subparser.add_argument("--device", type=str, default="cuda")
        subparser.add_argument("--wandb-project", type=str, default=None)
        subparser.add_argument("--wandb-entity", type=str, default=None)
        subparser.add_argument("--no-amp", action="store_true")
        subparser.add_argument("--compile", action="store_true")

    classifier = subparsers.add_parser("train-classifier", help="Train a clean ResNet18 classifier.")
    add_shared(classifier)

    fgsm = subparsers.add_parser("fgsm-report", help="Compare scratch FGSM and IBM ART FGSM.")
    add_shared(fgsm)
    fgsm.add_argument("--classifier-checkpoint", type=str, default=None)
    fgsm.add_argument("--eps-list", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1])
    fgsm.add_argument("--max-visuals", type=int, default=10)

    detector = subparsers.add_parser("train-detector", help="Train a ResNet34 adversarial detector.")
    add_shared(detector)
    detector.add_argument("--classifier-checkpoint", type=str, default=None)
    detector.add_argument("--detector-attack", choices=["pgd", "bim"], required=True)
    detector.add_argument("--eps", type=float, default=0.03)
    detector.add_argument("--alpha", type=float, default=0.007)
    detector.add_argument("--attack-steps", type=int, default=10)

    compare = subparsers.add_parser("compare-detectors", help="Train PGD and BIM detectors and compare results.")
    add_shared(compare)
    compare.add_argument("--classifier-checkpoint", type=str, default=None)
    compare.add_argument("--eps", type=float, default=0.03)
    compare.add_argument("--alpha", type=float, default=0.007)
    compare.add_argument("--attack-steps", type=int, default=10)
    compare.add_argument("--max-visuals", type=int, default=10)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_enable_speedups(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_tfms, eval_tfms


def inverse_normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def build_cifar10_loaders(config: Q2Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms, eval_tfms = get_transforms(config.image_size)
    train_full = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=train_tfms)
    train_eval_view = datasets.CIFAR10(root=config.data_dir, train=True, download=False, transform=eval_tfms)
    test_ds = datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=eval_tfms)
    val_len = max(1, int(len(train_full) * config.val_split))
    train_len = len(train_full) - val_len
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(train_full, [train_len, val_len], generator=generator)
    val_indices = val_subset.indices

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(torch.utils.data.Subset(train_eval_view, val_indices), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_resnet18(compile_model: bool) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def build_resnet34_binary(compile_model: bool) -> nn.Module:
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model


def get_optimizer_and_scheduler(model: nn.Module, config: Q2Config) -> tuple[AdamW, CosineAnnealingLR]:
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    return optimizer, scheduler


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_acc += compute_accuracy(logits, targets) * images.size(0)
    size = len(loader.dataset)
    return total_loss / size, total_acc / size


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    predictions: list[int] = []
    targets_all: list[int] = []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        total_loss += loss.item() * images.size(0)
        total_acc += compute_accuracy(logits, targets) * images.size(0)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
    size = len(loader.dataset)
    return total_loss / size, total_acc / size, predictions, targets_all


def save_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))


def write_csv(rows: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


def plot_curves(history: list[dict], destination: Path) -> None:
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_eps(rows: list[dict], destination: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["eps"], frame["scratch_fgsm_accuracy"], marker="o", label="FGSM Scratch")
    axes[0].plot(frame["eps"], frame["art_fgsm_accuracy"], marker="o", label="FGSM ART")
    axes[0].plot(frame["eps"], frame["clean_accuracy"], linestyle="--", color="black", label="Clean")
    axes[0].set_title("Accuracy vs Epsilon")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Accuracy")
    axes[1].plot(frame["eps"], frame["scratch_drop"], marker="o", label="FGSM Scratch")
    axes[1].plot(frame["eps"], frame["art_drop"], marker="o", label="FGSM ART")
    axes[1].set_title("Performance Drop vs Epsilon")
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylabel("Accuracy Drop")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pair_grid(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    attack_name: str,
    destination: Path,
) -> None:
    n_samples = min(len(clean), len(adversarial))
    fig, axes = plt.subplots(2, n_samples, figsize=(2.5 * n_samples, 5))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=1)
    for idx in range(n_samples):
        axes[0, idx].imshow(clean[idx].permute(1, 2, 0).cpu().numpy())
        axes[0, idx].set_title("Clean")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(adversarial[idx].permute(1, 2, 0).cpu().numpy())
        axes[1, idx].set_title(attack_name)
        axes[1, idx].axis("off")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def maybe_init_wandb(config: Q2Config, run_name: str) -> object | None:
    if not config.wandb_project:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Add it with `uv add wandb` or omit --wandb-project.")
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        config=asdict(config),
        reinit=True,
    )


def train_clean_classifier(config: Q2Config) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    maybe_enable_speedups(device)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    output_dir = Path(config.output_dir) / "classifier"
    train_loader, val_loader, test_loader = build_cifar10_loaders(config)
    model = build_resnet18(config.compile_model).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    wandb_run = maybe_init_wandb(config, "q2-classifier")

    best_acc = -1.0
    best_state = None
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler, amp_enabled)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, amp_enabled)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_accuracy": round(val_acc, 6),
        }
        history.append(row)
        if wandb_run is not None:
            wandb.log(row)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc, predictions, targets = evaluate(model, test_loader, device, amp_enabled)
    checkpoint_path = output_dir / "best_model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    write_csv(history, output_dir / "epoch_metrics.csv")
    plot_curves(history, output_dir / "training_curves.png")
    summary = {
        "best_val_accuracy": round(best_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 6),
        "target_accuracy_met": bool(test_acc >= 0.72),
        "checkpoint_path": str(checkpoint_path),
        "num_test_samples": len(targets),
        "num_correct": int(accuracy_score(targets, predictions, normalize=False)),
    }
    save_json(summary, output_dir / "summary.json")
    if wandb_run is not None:
        wandb.log(summary)
        wandb.finish()
    return summary


def load_classifier_for_eval(config: Q2Config, device: torch.device) -> tuple[nn.Module, PyTorchClassifier]:
    checkpoint = config.classifier_checkpoint or str(Path(config.output_dir) / "classifier" / "best_model.pt")
    model = build_resnet18(False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=AdamW(model.parameters(), lr=config.learning_rate),
        input_shape=(3, config.image_size, config.image_size),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        preprocessing=(np.array(MEAN, dtype=np.float32), np.array(STD, dtype=np.float32)),
    )
    return model, classifier


def fgsm_scratch_attack(model: nn.Module, images: torch.Tensor, targets: torch.Tensor, eps: float, device: torch.device) -> torch.Tensor:
    images = images.clone().detach().to(device)
    targets = targets.to(device)
    images.requires_grad_(True)
    logits = model(normalize_tensor(images))
    loss = F.cross_entropy(logits, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.clamp(images + eps * images.grad.sign(), 0.0, 1.0).detach()


@torch.no_grad()
def eval_image_tensor(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> float:
    predictions = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        logits = model(normalize_tensor(batch))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return float(accuracy_score(labels.tolist(), predictions))


def save_visual_grid(
    originals: torch.Tensor,
    scratch_adv: torch.Tensor,
    art_adv: torch.Tensor,
    labels: torch.Tensor,
    destination: Path,
) -> None:
    fig, axes = plt.subplots(len(originals), 3, figsize=(8, 2.5 * len(originals)))
    if len(originals) == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = ["Original", "FGSM Scratch", "FGSM ART"]
    for row in range(len(originals)):
        for col, image in enumerate([originals[row], scratch_adv[row], art_adv[row]]):
            axes[row, col].imshow(image.permute(1, 2, 0).cpu().numpy())
            axes[row, col].set_title(f"{titles[col]} | y={labels[row].item()}")
            axes[row, col].axis("off")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_fgsm_report(config: Q2Config, eps_list: list[float]) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    maybe_enable_speedups(device)
    output_dir = Path(config.output_dir) / "fgsm"
    _, _, test_loader = build_cifar10_loaders(config)
    model, art_classifier = load_classifier_for_eval(config, device)
    wandb_run = maybe_init_wandb(config, "q2-fgsm-report")

    clean_images = []
    clean_labels = []
    for images, labels in test_loader:
        clean_images.append(inverse_normalize(images))
        clean_labels.append(labels)
    clean_images_tensor = torch.cat(clean_images, dim=0)
    clean_labels_tensor = torch.cat(clean_labels, dim=0)
    clean_acc = eval_image_tensor(model, clean_images_tensor, clean_labels_tensor, config.batch_size, device)

    rows = []
    for eps in eps_list:
        scratch_adv_batches = []
        for start in range(0, len(clean_images_tensor), config.batch_size):
            images = clean_images_tensor[start : start + config.batch_size]
            labels = clean_labels_tensor[start : start + config.batch_size]
            scratch_adv_batches.append(fgsm_scratch_attack(model, images, labels, eps, device).cpu())
        scratch_adv = torch.cat(scratch_adv_batches, dim=0)

        art_adv = torch.from_numpy(FastGradientMethod(estimator=art_classifier, eps=eps).generate(x=clean_images_tensor.numpy())).float()

        scratch_acc = eval_image_tensor(model, scratch_adv, clean_labels_tensor, config.batch_size, device)
        art_acc = eval_image_tensor(model, art_adv, clean_labels_tensor, config.batch_size, device)
        rows.append(
            {
                "eps": eps,
                "clean_accuracy": round(clean_acc, 6),
                "scratch_fgsm_accuracy": round(scratch_acc, 6),
                "art_fgsm_accuracy": round(art_acc, 6),
                "scratch_drop": round(clean_acc - scratch_acc, 6),
                "art_drop": round(clean_acc - art_acc, 6),
            }
        )

        limit = min(config.max_visuals, len(clean_images_tensor))
        visual_path = output_dir / f"visual_comparison_eps_{eps:.3f}.png"
        save_visual_grid(
            clean_images_tensor[:limit],
            scratch_adv[:limit],
            art_adv[:limit],
            clean_labels_tensor[:limit],
            visual_path,
        )
        scratch_pair_path = output_dir / f"fgsm_scratch_pairs_eps_{eps:.3f}.png"
        art_pair_path = output_dir / f"fgsm_art_pairs_eps_{eps:.3f}.png"
        save_pair_grid(clean_images_tensor[:limit], scratch_adv[:limit], "FGSM Scratch", scratch_pair_path)
        save_pair_grid(clean_images_tensor[:limit], art_adv[:limit], "FGSM ART", art_pair_path)
        if wandb_run is not None:
            wandb.log({
                f"fgsm_visual_comparison_eps_{eps:.3f}": wandb.Image(str(visual_path)),
                f"fgsm_scratch_pairs_eps_{eps:.3f}": wandb.Image(str(scratch_pair_path)),
                f"fgsm_art_pairs_eps_{eps:.3f}": wandb.Image(str(art_pair_path)),
                "eps": eps,
                "clean_accuracy": clean_acc,
                "scratch_fgsm_accuracy": scratch_acc,
                "art_fgsm_accuracy": art_acc,
            })

    write_csv(rows, output_dir / "fgsm_metrics.csv")
    plot_metric_vs_eps(rows, output_dir / "fgsm_eps_vs_accuracy.png")
    summary = {
        "clean_accuracy": round(clean_acc, 6),
        "results": rows,
        "best_scratch_eps": max(rows, key=lambda row: row["scratch_drop"])["eps"],
        "best_art_eps": max(rows, key=lambda row: row["art_drop"])["eps"],
    }
    save_json(summary, output_dir / "summary.json")
    if wandb_run is not None:
        wandb.log({
            "fgsm_eps_vs_accuracy": wandb.Image(str(output_dir / "fgsm_eps_vs_accuracy.png")),
            **summary,
        })
        wandb.finish()
    return summary


class AdversarialTensorDataset(Dataset):
    def __init__(self, clean: torch.Tensor, adversarial: torch.Tensor) -> None:
        self.images = torch.cat([clean, adversarial], dim=0)
        self.labels = torch.cat(
            [torch.zeros(len(clean), dtype=torch.long), torch.ones(len(adversarial), dtype=torch.long)],
            dim=0,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = normalize_tensor(self.images[index].unsqueeze(0)).squeeze(0)
        return image, self.labels[index]


def generate_adversarial_tensor(classifier: PyTorchClassifier, images: torch.Tensor, attack_name: str, config: Q2Config) -> torch.Tensor:
    if attack_name == "pgd":
        attack = ProjectedGradientDescentPyTorch(
            estimator=classifier,
            eps=config.eps,
            eps_step=config.alpha,
            max_iter=config.attack_steps,
        )
    elif attack_name == "bim":
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=config.eps,
            eps_step=config.alpha,
            max_iter=config.attack_steps,
        )
    else:
        raise ValueError(f"Unsupported attack: {attack_name}")
    return torch.from_numpy(attack.generate(x=images.numpy())).float()


def subset_to_clean_tensor(subset: Dataset, config: Q2Config) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(subset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    images_all = []
    labels_all = []
    for images, labels in loader:
        images_all.append(inverse_normalize(images))
        labels_all.append(labels)
    return torch.cat(images_all, dim=0), torch.cat(labels_all, dim=0)


def build_detector_loaders(config: Q2Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    if config.detector_attack is None:
        raise ValueError("detector_attack must be set.")
    device = resolve_device(config.device)
    _, _, test_loader = build_cifar10_loaders(config)
    _, eval_tfms = get_transforms(config.image_size)
    train_ds = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=eval_tfms)
    generator = torch.Generator().manual_seed(config.seed)
    val_len = max(1, int(len(train_ds) * config.val_split))
    train_len = len(train_ds) - val_len
    train_subset, val_subset = random_split(train_ds, [train_len, val_len], generator=generator)

    _, classifier = load_classifier_for_eval(config, device)
    train_clean, _ = subset_to_clean_tensor(train_subset, config)
    val_clean, _ = subset_to_clean_tensor(val_subset, config)

    test_clean_images = []
    for images, _ in test_loader:
        test_clean_images.append(inverse_normalize(images))
    test_clean = torch.cat(test_clean_images, dim=0)

    train_adv = generate_adversarial_tensor(classifier, train_clean, config.detector_attack, config)
    val_adv = generate_adversarial_tensor(classifier, val_clean, config.detector_attack, config)
    test_adv = generate_adversarial_tensor(classifier, test_clean, config.detector_attack, config)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    return (
        DataLoader(AdversarialTensorDataset(train_clean, train_adv), shuffle=True, **loader_kwargs),
        DataLoader(AdversarialTensorDataset(val_clean, val_adv), shuffle=False, **loader_kwargs),
        DataLoader(AdversarialTensorDataset(test_clean, test_adv), shuffle=False, **loader_kwargs),
    )


def save_detector_samples(test_loader: DataLoader, output_dir: Path, attack_name: str, max_visuals: int) -> Path:
    dataset = test_loader.dataset
    if not isinstance(dataset, AdversarialTensorDataset):
        raise TypeError("Expected test_loader.dataset to be an AdversarialTensorDataset.")
    n_samples = min(max_visuals, len(dataset.images) // 2)
    clean_samples = normalize_tensor(dataset.images[:n_samples])
    adv_samples = normalize_tensor(dataset.images[len(dataset.images) // 2 : len(dataset.images) // 2 + n_samples])
    display_clean = inverse_normalize(clean_samples)
    display_adv = inverse_normalize(adv_samples)
    sample_grid_path = output_dir / f"sample_{attack_name}_clean_vs_adv.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pair_grid(display_clean, display_adv, attack_name.upper(), sample_grid_path)
    return sample_grid_path


def train_detector(config: Q2Config) -> dict:
    if config.detector_attack is None:
        raise ValueError("detector_attack must be set.")
    set_seed(config.seed)
    device = resolve_device(config.device)
    maybe_enable_speedups(device)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    output_dir = Path(config.output_dir) / f"detector_{config.detector_attack}"

    train_loader, val_loader, test_loader = build_detector_loaders(config)
    model = build_resnet34_binary(config.compile_model).to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    wandb_run = maybe_init_wandb(config, f"q2-detector-{config.detector_attack}")

    sample_grid_path = save_detector_samples(test_loader, output_dir, config.detector_attack, config.max_visuals)
    if wandb_run is not None:
        wandb.log({f"{config.detector_attack}_sample_clean_vs_adv": wandb.Image(str(sample_grid_path))})

    best_acc = -1.0
    best_state = None
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler, amp_enabled)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, amp_enabled)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "val_accuracy": round(val_acc, 6),
        }
        history.append(row)
        if wandb_run is not None:
            wandb.log(row)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Detector training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, device, amp_enabled)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    write_csv(history, output_dir / "epoch_metrics.csv")
    plot_curves(history, output_dir / "training_curves.png")
    summary = {
        "attack": config.detector_attack,
        "best_val_accuracy": round(best_acc, 6),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 6),
        "target_accuracy_met": bool(test_acc >= 0.70),
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(summary, output_dir / "summary.json")
    if wandb_run is not None:
        wandb.log({
            **summary,
            "training_curves": wandb.Image(str(output_dir / "training_curves.png")),
        })
        wandb.finish()
    return summary


def compare_detectors(config: Q2Config) -> dict:
    results = []
    for attack_name in ("pgd", "bim"):
        attack_config = Q2Config(**{**asdict(config), "command": "train-detector", "detector_attack": attack_name})
        results.append(train_detector(attack_config))

    comparison_dir = Path(config.output_dir) / "detector_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, comparison_dir / "detector_comparison.csv")
    save_json({"results": results}, comparison_dir / "summary.json")

    frame = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(frame["attack"], frame["test_accuracy"], color=["#1f77b4", "#ff7f0e"])
    ax.axhline(0.70, linestyle="--", color="red", label="Target 70%")
    ax.set_ylabel("Detection Accuracy")
    ax.set_title("PGD vs BIM Detector Accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(comparison_dir / "detector_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    wandb_run = maybe_init_wandb(config, "q2-detector-comparison")
    if wandb_run is not None:
        wandb.log({
            "detector_comparison": wandb.Image(str(comparison_dir / "detector_comparison.png")),
            "detector_results": results,
        })
        wandb.finish()
    return {"results": results}


def namespace_to_config(args: argparse.Namespace) -> Q2Config:
    return Q2Config(
        command=args.command,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        image_size=args.image_size,
        val_split=args.val_split,
        device=args.device,
        eps=getattr(args, "eps", 0.03),
        alpha=getattr(args, "alpha", 0.007),
        attack_steps=getattr(args, "attack_steps", 10),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        use_amp=not args.no_amp,
        compile_model=args.compile,
        max_visuals=getattr(args, "max_visuals", 10),
        classifier_checkpoint=getattr(args, "classifier_checkpoint", None),
        detector_attack=getattr(args, "detector_attack", None),
        eps_list=getattr(args, "eps_list", None),
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    config = namespace_to_config(args)
    if args.command == "train-classifier":
        train_clean_classifier(config)
    elif args.command == "fgsm-report":
        run_fgsm_report(config, args.eps_list)
    elif args.command == "train-detector":
        train_detector(config)
    elif args.command == "compare-detectors":
        compare_detectors(config)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
