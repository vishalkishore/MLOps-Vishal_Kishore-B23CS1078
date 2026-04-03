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
import csv
import json
import math
import random
import socket
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import confusion_matrix
from timm import create_model
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.auto import tqdm

socket.gethostname = lambda: "vishal"

try:
    import wandb
except ImportError:
    wandb = None


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    mode: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    output_dir: str
    data_dir: str
    model_name: str
    num_classes: int
    image_size: int
    val_split: float
    rank: int | None = None
    alpha: int | None = None
    dropout: float = 0.1
    lora_targets: tuple[str, ...] = ("qkv",)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    optuna_trials: int = 0
    optuna_metric: str = "best_val_accuracy"
    device: str = "cuda"
    use_amp: bool = True
    compile_model: bool = False
    save_best_only: bool = True

    @property
    def run_name(self) -> str:
        if self.mode == "baseline":
            return "baseline-head-only"
        return f"lora-r{self.rank}-a{self.alpha}-d{self.dropout}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assignment 5 Q1: ViT-S on CIFAR-100 with LoRA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--epochs", type=int, default=10)
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument("--learning-rate", type=float, default=3e-4)
        subparser.add_argument("--weight-decay", type=float, default=1e-4)
        subparser.add_argument("--num-workers", type=int, default=4)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "q1"))
        subparser.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
        subparser.add_argument("--model-name", type=str, default="vit_small_patch16_224.augreg_in21k_ft_in1k")
        subparser.add_argument("--image-size", type=int, default=224)
        subparser.add_argument("--val-split", type=float, default=0.1)
        subparser.add_argument("--dropout", type=float, default=0.1)
        subparser.add_argument("--wandb-project", type=str, default=None)
        subparser.add_argument("--wandb-entity", type=str, default=None)
        subparser.add_argument("--device", type=str, default="cuda")
        subparser.add_argument("--num-classes", type=int, default=100)
        subparser.add_argument("--no-amp", action="store_true", help="Disable mixed precision on CUDA.")
        subparser.add_argument("--compile", action="store_true", help="Use torch.compile for the model.")

    baseline = subparsers.add_parser("baseline", help="Train only the classification head.")
    add_shared_arguments(baseline)

    lora = subparsers.add_parser("lora", help="Train a single LoRA configuration.")
    add_shared_arguments(lora)
    lora.add_argument("--rank", type=int, required=True)
    lora.add_argument("--alpha", type=int, required=True)

    grid = subparsers.add_parser("grid", help="Run all required LoRA combinations.")
    add_shared_arguments(grid)
    grid.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    grid.add_argument("--alphas", type=int, nargs="+", default=[2, 4, 8])

    optuna_parser = subparsers.add_parser("optuna", help="Search LoRA hyperparameters with Optuna.")
    add_shared_arguments(optuna_parser)
    optuna_parser.add_argument("--optuna-trials", type=int, default=10)
    optuna_parser.add_argument("--metric", type=str, default="best_val_accuracy")
    optuna_parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8,16])
    optuna_parser.add_argument("--alphas", type=int, nargs="+", default=[2, 4, 8,16])

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


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms, eval_tfms = build_transforms(config.image_size)
    train_full = datasets.CIFAR100(root=config.data_dir, train=True, download=True, transform=train_tfms)
    train_eval_view = datasets.CIFAR100(root=config.data_dir, train=True, download=False, transform=eval_tfms)
    test_ds = datasets.CIFAR100(root=config.data_dir, train=False, download=True, transform=eval_tfms)

    val_len = max(1, int(len(train_full) * config.val_split))
    train_len = len(train_full) - val_len
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(train_full, [train_len, val_len], generator=generator)

    # Validation should not use augmentation.
    val_indices = val_subset.indices
    val_subset.dataset = train_eval_view
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(train_eval_view, val_indices),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def build_model(config: ExperimentConfig) -> nn.Module:
    model = create_model(config.model_name, pretrained=True, num_classes=config.num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("Expected timm ViT model to expose a trainable `head` module.")

    if config.mode == "lora":
        lora_cfg = LoraConfig(
            task_type=None,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.lora_targets),
            bias="none",
            modules_to_save=["head"],
        )
        model = inject_adapter_in_model(lora_cfg, model)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_optimizer(model: nn.Module, config: ExperimentConfig) -> AdamW:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    gradient_norms = []

    for images, targets in tqdm(loader, leave=False, desc="train"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        lora_grad_sq_sum = 0.0
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and "lora_" in name:
                lora_grad_sq_sum += parameter.grad.norm(2).item() ** 2
        gradient_norms.append(math.sqrt(lora_grad_sq_sum) if lora_grad_sq_sum > 0 else 0.0)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += compute_accuracy(logits, targets) * images.size(0)

    dataset_size = len(loader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size, float(np.mean(gradient_norms))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    for images, targets in tqdm(loader, leave=False, desc="eval"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        running_loss += loss.item() * images.size(0)
        running_acc += compute_accuracy(logits, targets) * images.size(0)
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    dataset_size = len(loader.dataset)
    return running_loss / dataset_size, running_acc / dataset_size, all_predictions, all_targets


def plot_training_curves(history: list[dict], destination: Path) -> None:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["lora_grad_norm"])
    axes[2].set_title("LoRA Gradient Norm")
    axes[2].set_ylim(bottom=0)

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)

    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_classwise_histogram(classwise_accuracy: np.ndarray, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(len(classwise_accuracy)), classwise_accuracy)
    ax.set_title("Class-wise Test Accuracy")
    ax.set_xlabel("Class index")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))


def write_epoch_table(history: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "train_accuracy",
                "val_accuracy",
                "lora_grad_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def to_relative_path_string(path_like: str | Path) -> str:
    path = Path(path_like)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sanitize_wandb_payload(payload: dict) -> dict:
    sanitized = dict(payload)
    for key in ("output_dir", "data_dir", "checkpoint_path"):
        if key in sanitized and sanitized[key] is not None:
            sanitized[key] = to_relative_path_string(sanitized[key])
    return sanitized


def maybe_init_wandb(config: ExperimentConfig, run_name: str, payload: dict) -> object | None:
    if not config.wandb_project:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed. Add it with `uv add wandb` or omit --wandb-project.")
    return wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        config=payload,
        reinit=True,
        settings=wandb.Settings(
            _disable_stats=True,
            x_disable_meta=True
        )
    )


def train_experiment(config: ExperimentConfig) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    optimizer = get_optimizer(model, config)
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    trainable_params, total_params = count_trainable_parameters(model)

    wandb_run = maybe_init_wandb(config, config.run_name, sanitize_wandb_payload(asdict(config)))
    if wandb_run is not None:
        # Full parameter + gradient watching adds substantial per-step overhead for ViT training.
        wandb.watch(model, log="gradients", log_freq=500)
    best_state = None
    best_val_accuracy = -float("inf")
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy, grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            amp_enabled,
        )
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_accuracy": round(val_accuracy, 6),
            "lora_grad_norm": round(grad_norm, 6),
        }
        history.append(row)

        if wandb_run is not None:
            wandb.log(row)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_accuracy, test_predictions, test_targets = evaluate(model, test_loader, device, amp_enabled)

    conf = confusion_matrix(test_targets, test_predictions, labels=list(range(config.num_classes)))
    classwise_accuracy = conf.diagonal() / np.maximum(conf.sum(axis=1), 1)

    checkpoint_path = output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    write_epoch_table(history, output_dir / "epoch_metrics.csv")
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_classwise_histogram(classwise_accuracy, output_dir / "classwise_accuracy.png")

    summary = {
        "run_name": config.run_name,
        "mode": config.mode,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "best_val_accuracy": round(best_val_accuracy, 6),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 6),
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "trainable_percent": round(100 * trainable_params / total_params, 4),
        "checkpoint_path": to_relative_path_string(checkpoint_path),
    }
    save_json(summary, output_dir / "summary.json")

    if wandb_run is not None:
        wandb.log(
            {
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "classwise_test_accuracy_mean": float(classwise_accuracy.mean()),
                "checkpoint_path": to_relative_path_string(checkpoint_path),
                "classwise_accuracy_histogram": wandb.Image(str(output_dir / "classwise_accuracy.png")),
                "training_curves": wandb.Image(str(output_dir / "training_curves.png")),
            }
        )
        wandb.finish()

    return {
        "summary": summary,
        "history": history,
        "classwise_accuracy": classwise_accuracy.tolist(),
    }


def append_grid_summary(rows: Iterable[dict], destination: Path) -> None:
    frame = pd.DataFrame(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def run_grid(args: argparse.Namespace) -> None:
    rows = []
    for rank in args.ranks:
        for alpha in args.alphas:
            config = ExperimentConfig(
                mode="lora",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                seed=args.seed,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                model_name=args.model_name,
                num_classes=args.num_classes,
                image_size=args.image_size,
                val_split=args.val_split,
                rank=rank,
                alpha=alpha,
                dropout=args.dropout,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                device=args.device,
                use_amp=not args.no_amp,
                compile_model=args.compile,
            )
            result = train_experiment(config)
            rows.append(result["summary"])

    append_grid_summary(rows, Path(args.output_dir) / "grid_summary.csv")


def run_optuna(args: argparse.Namespace) -> None:
    def objective(trial: optuna.Trial) -> float:
        config = ExperimentConfig(
            mode="lora",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed + trial.number,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            model_name=args.model_name,
            num_classes=args.num_classes,
            image_size=args.image_size,
            val_split=args.val_split,
            rank=trial.suggest_categorical("rank", args.ranks),
            alpha=trial.suggest_categorical("alpha", args.alphas),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            device=args.device,
            use_amp=not args.no_amp,
            compile_model=args.compile,
        )
        result = train_experiment(config)
        return float(result["summary"][args.metric])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optuna_trials)

    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
    }
    save_json(best_payload, Path(args.output_dir) / "optuna_summary.json")


def namespace_to_config(args: argparse.Namespace, mode: str) -> ExperimentConfig:
    return ExperimentConfig(
        mode=mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        val_split=args.val_split,
        rank=getattr(args, "rank", None),
        alpha=getattr(args, "alpha", None),
        dropout=args.dropout,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        device=args.device,
        use_amp=not args.no_amp,
        compile_model=args.compile,
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.command == "baseline":
        train_experiment(namespace_to_config(args, "baseline"))
    elif args.command == "lora":
        train_experiment(namespace_to_config(args, "lora"))
    elif args.command == "grid":
        run_grid(args)
    elif args.command == "optuna":
        run_optuna(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
