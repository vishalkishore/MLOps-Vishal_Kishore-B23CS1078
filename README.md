## Assignment 5

 "matplotlib>=3.10.8",
    "optuna>=4.8.0",
    "pandas>=3.0.2",
    "peft>=0.18.1",
    "scikit-learn>=1.8.0",
    "timm>=1.0.26",
    "torch>=2.11.0",
    "torchvision>=0.26.0",
    "tqdm>=4.67.3",
    "wandb>=0.25.1",

This repository contains the Question 1 training pipeline for Assignment 5:

- ViT-S baseline finetuning on CIFAR-100 with the classification head trainable
- ViT-S + LoRA experiments on the attention `qkv` projections
- LoRA grid search for ranks `2, 4, 8` and alphas `2, 4, 8`
- Optuna-based LoRA hyperparameter search

## Setup

`uv` is the package manager used in this repo.

```bash
uv add torch torchvision timm peft optuna pandas matplotlib scikit-learn tqdm wandb
uv sync
```

Docker build:

```bash
docker build -t assignment5-q1 .
```

## Run

Baseline:

```bash
uv run python main.py baseline --epochs 10 --batch-size 128
```

Single LoRA experiment:

```bash
uv run python main.py lora --rank 4 --alpha 8 --epochs 10 --batch-size 128
```

Required LoRA grid:

```bash
uv run python main.py grid --ranks 2 4 8 --alphas 2 4 8 --epochs 10 --batch-size 128
```

Optuna search:

```bash
uv run python main.py optuna --optuna-trials 10 --ranks 2 4 8 --alphas 2 4 8 --epochs 10
```

The same commands can be executed inside the container, for example:

```bash
docker run --rm -it assignment5-q1 uv run python main.py baseline --epochs 10 --batch-size 128
```

## Outputs

Each run writes files under `outputs/q1/<run-name>/`:

- `best_model.pt`
- `epoch_metrics.csv`
- `training_curves.png`
- `classwise_accuracy.png`
- `summary.json`

Grid search additionally writes `outputs/q1/grid_summary.csv`, and Optuna writes `outputs/q1/optuna_summary.json`.
