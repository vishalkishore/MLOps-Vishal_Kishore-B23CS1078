FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e ".[dev]"

COPY . .


ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/workspace/.cache/huggingface \
    CUDA_VISIBLE_DEVICES=0,1

CMD ["python", "translate.py"]
