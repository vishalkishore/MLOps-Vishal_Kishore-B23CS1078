# Assignment 5

**Student:** Vishal Kishore  
**Roll Number:** B23CS1078

This repository contains the code, experiment outputs, checkpoints, and report assets for Assignment 5.

## Repository Structure

- `ques_1/` - Q1 training pipeline for ViT-S baseline, LoRA grid search, and Optuna search on CIFAR-100
- `ques_2/` - Q2 training pipeline for CIFAR-10 classification, FGSM attacks, and adversarial detectors
- `outputs/q1/` - Q1 checkpoints, CSV metrics, summaries, and plots
- `outputs/q2/` - Q2 checkpoints, CSV metrics, summaries, and plots
- `report/` - LaTeX report source and report images
- `data/` - CIFAR-10 and CIFAR-100 datasets

## Requirements

The project uses Python with PyTorch and the following core libraries:

- `torch==2.6.0`
- `torchvision==0.21.0`
- `timm>=1.0.26`
- `peft>=0.18.1`
- `optuna>=4.8.0`
- `adversarial-robustness-toolbox>=1.20.1`
- `pandas>=3.0.2`
- `matplotlib>=3.10.8`
- `scikit-learn>=1.8.0`
- `tqdm>=4.67.3`
- `wandb>=0.25.1`

All dependencies are also listed in [requirements.txt](requirements.txt) and [pyproject.toml](pyproject.toml).

## Installation

### Option 1: `uv`

```bash
uv sync
```

### Option 2: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker

The assignment requires experiments to be run using Docker. A Dockerfile is included in the repository.

Build the container:

```bash
docker build -t assignment5 .
```

Open a shell inside the container:

```bash
docker run --rm -it assignment5 bash
```

Run a command directly:

```bash
docker run --rm -it assignment5 uv run python main.py --help
```

## How To Run

The main entrypoint is [main.py](main.py).

### Question 1: ViT-S on CIFAR-100

Baseline head-only fine-tuning:

```bash
uv run python main.py baseline --epochs 10 --batch-size 128
```

Single LoRA experiment:

```bash
uv run python main.py lora --rank 4 --alpha 8 --dropout 0.1 --epochs 10 --batch-size 128
```

Required LoRA grid:

```bash
uv run python main.py grid --ranks 2 4 8 --alphas 2 4 8 --dropout 0.1 --epochs 10 --batch-size 128
```

Optuna search for LoRA hyperparameters:

```bash
uv run python main.py optuna --optuna-trials 20 --ranks 2 4 8 16 --alphas 2 4 8 16 --epochs 10 --batch-size 128
```

Optional WandB logging for Q1:

```bash
uv run python main.py grid --ranks 2 4 8 --alphas 2 4 8 --dropout 0.1 --epochs 10 --wandb-project <project_name> --wandb-entity <entity_name>
```

### Question 2: Adversarial Attacks and Detection on CIFAR-10

Train clean ResNet-18 classifier:

```bash
uv run python main.py q2 train-classifier --epochs 20 --batch-size 128
```

Run FGSM comparison from scratch vs IBM ART:

```bash
uv run python main.py q2 fgsm-report --classifier-checkpoint outputs/q2/classifier/best_model.pt --eps-list 0.01 0.03 0.05 0.1
```

Train PGD detector:

```bash
uv run python main.py q2 train-detector --detector-attack pgd --classifier-checkpoint outputs/q2/classifier/best_model.pt --epochs 15 --eps 0.03 --alpha 0.007 --attack-steps 10
```

Train BIM detector:

```bash
uv run python main.py q2 train-detector --detector-attack bim --classifier-checkpoint outputs/q2/classifier/best_model.pt --epochs 15 --eps 0.03 --alpha 0.007 --attack-steps 10
```

Train and compare both detectors in one command:

```bash
uv run python main.py q2 compare-detectors --classifier-checkpoint outputs/q2/classifier/best_model.pt --epochs 15 --eps 0.03 --alpha 0.007 --attack-steps 10
```

Optional WandB logging for Q2:

```bash
uv run python main.py q2 fgsm-report --classifier-checkpoint outputs/q2/classifier/best_model.pt --wandb-project <project_name> --wandb-entity <entity_name>
```

## Results

### Q1 Baseline: ViT-S Head-Only Fine-Tuning

| Model | Best Val Accuracy | Test Accuracy | Test Loss | Trainable Parameters | Trainable Percent |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-S head-only | 0.8190 | 0.8143 | 0.6215 | 38,500 | 0.1774% |

### Q1 Required LoRA Grid

LoRA is injected into the ViT attention `qkv` projection while keeping the classification head trainable.

| LoRA layers | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| with LoRA on `qkv` | 2 | 2 | 0.1 | 0.8999 | 75,364 |
| with LoRA on `qkv` | 2 | 4 | 0.1 | 0.9017 | 75,364 |
| with LoRA on `qkv` | 2 | 8 | 0.1 | 0.9020 | 75,364 |
| with LoRA on `qkv` | 4 | 2 | 0.1 | 0.8994 | 112,228 |
| with LoRA on `qkv` | 4 | 4 | 0.1 | 0.8997 | 112,228 |
| with LoRA on `qkv` | 4 | 8 | 0.1 | 0.9012 | 112,228 |
| with LoRA on `qkv` | 8 | 2 | 0.1 | 0.9016 | 185,956 |
| with LoRA on `qkv` | 8 | 4 | 0.1 | 0.9017 | 185,956 |
| with LoRA on `qkv` | 8 | 8 | 0.1 | 0.9012 | 185,956 |

### Q1 Best LoRA Configuration from Optuna

| Source | Rank | Alpha | Dropout | Best Val Accuracy | Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optuna best trial | 8 | 4 | 0.2962490251548571 | 0.9070 | 0.9049 |

Observations:

- LoRA substantially improves over the baseline head-only setup on CIFAR-100.
- The required grid over `rank={2,4,8}` and `alpha={2,4,8}` with dropout `0.1` is complete.
- Gradient norm tracking for LoRA weights and class-wise test accuracy histograms are generated for Q1 runs.

### Q2 Clean Classifier

| Model | Best Val Accuracy | Test Accuracy | Test Loss | Meets 72% Target |
| --- | ---: | ---: | ---: | --- |
| ResNet-18 from scratch | 0.7878 | 0.7836 | 0.6305 | Yes |

### Q2 FGSM: Scratch vs IBM ART

| Epsilon | Clean Accuracy | Scratch FGSM Accuracy | ART FGSM Accuracy | Scratch Drop | ART Drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.7838 | 0.2341 | 0.3504 | 0.5497 | 0.4334 |
| 0.03 | 0.7838 | 0.0196 | 0.1312 | 0.7642 | 0.6526 |
| 0.05 | 0.7838 | 0.0065 | 0.1113 | 0.7773 | 0.6725 |
| 0.10 | 0.7838 | 0.0090 | 0.0914 | 0.7748 | 0.6924 |

Observations:

- Both FGSM implementations significantly reduce performance compared with clean evaluation.
- In these saved runs, the scratch FGSM attack produces a stronger drop than IBM ART FGSM.
- Qualitative comparisons for original vs adversarial images are saved for multiple epsilon values.

### Q2 Adversarial Detectors

| Detector Attack | Best Val Accuracy | Test Accuracy | Test Loss | Meets 70% Target |
| --- | ---: | ---: | ---: | --- |
| PGD | 0.9981 | 0.9985 | 0.008833 | Yes |
| BIM | 0.9979 | 0.9983 | 0.008720 | Yes |

Observations:

- Both detectors comfortably exceed the required 70% detection accuracy threshold.
- PGD and BIM adversarial sample pairs are saved as qualitative examples.

## Saved Outputs

### Q1

- Baseline checkpoint: [outputs/q1/baseline-head-only/best_model.pt](outputs/q1/baseline-head-only/best_model.pt)
- Grid summary: [outputs/q1/grid_summary.csv](outputs/q1/grid_summary.csv)
- Optuna summary: [outputs/q1/optuna_summary.json](outputs/q1/optuna_summary.json)
- Best Optuna checkpoint: [outputs/q1/lora-r8-a4-d0.2962490251548571/best_model.pt](outputs/q1/lora-r8-a4-d0.2962490251548571/best_model.pt)

Each Q1 run directory contains:

- `best_model.pt`
- `epoch_metrics.csv`
- `training_curves.png`
- `classwise_accuracy.png`
- `summary.json`

### Q2

- Clean classifier checkpoint: [outputs/q2/classifier/best_model.pt](outputs/q2/classifier/best_model.pt)
- FGSM summary: [outputs/q2/fgsm/summary.json](outputs/q2/fgsm/summary.json)
- PGD detector checkpoint: [outputs/q2/detector_pgd/best_model.pt](outputs/q2/detector_pgd/best_model.pt)
- BIM detector checkpoint: [outputs/q2/detector_bim/best_model.pt](outputs/q2/detector_bim/best_model.pt)

Q2 output folders also include CSV metrics, summaries, training curves, and qualitative visualizations.

