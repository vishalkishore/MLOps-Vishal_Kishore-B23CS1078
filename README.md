# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

**Student:** Vishal Kishore (B23CS1078)  
**Course:** ML/DL Ops  
**HuggingFace Model:** [vishalkishore01/lab3](https://huggingface.co/vishalkishore01/lab3)

---

## Project Overview

Fine-tune a **DistilBERT** model on Goodreads book reviews to classify text by genre (poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult). The full pipeline is containerized with Docker.

## Model Selection

**Model:** `distilbert-base-cased`

- **Why DistilBERT?** It is a distilled version of BERT that retains ~97% of BERT's language understanding while being 60% faster and 40% smaller. This makes it ideal for a classification assignment — fast to train even on CPU, with strong performance.
- **Why cased?** Casing can carry meaningful information in book reviews (e.g., proper nouns, titles).
- **Num labels:** 8 genres


## Setup & Usage

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for containerized runs)

### Local Development

```bash
# Install dependencies
uv sync

# Train the model (downloads data, trains, evaluates, pushes to HF)
uv run python train.py

# Evaluate model from HuggingFace repo
uv run python eval.py
```

### Docker

#### Training Image
```bash
# Build
docker build -t lab3-train .

# Run training
docker run lab3-train
```

#### Production Evaluation Image
```bash
# Build
docker build -f Dockerfile.eval -t lab3-eval .

# Run evaluation (auto-pulls model from HuggingFace)
docker run lab3-eval
```

## Training Summary

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-cased` |
| Epochs | 3 |
| Batch Size (train) | 10 |
| Batch Size (eval) | 16 |
| Learning Rate | 5e-5 |
| Warmup Steps | 100 |
| Weight Decay | 0.01 |

## Evaluation Results

Evaluation is performed twice:
1. **Local model** (directly after training) → `eval_results.json`
2. **HuggingFace model** (loaded from `vishalkishore01/lab3`) → `eval_results_from_hf.json`

Both should produce identical metrics since they are the same model.

## Challenges

1. **Large data downloads**: The UCSD Goodreads dataset is hosted as gzipped JSON. Streaming was used to avoid loading entire files into memory.
2. **GPU vs CPU**: Training on CPU takes significantly longer. The code auto-detects CUDA availability.
3. **Docker image size**: The PyTorch + Transformers stack makes images large. A slim base image and `.dockerignore` help minimize size.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token for model push/pull |

See `.env.sample` for the template.
