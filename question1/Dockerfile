FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1059
ARG GID=1059

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl vim build-essential cmake nlohmann-json3-dev \
    && if ! getent group appuser >/dev/null; then groupadd --gid "${GID}" appuser; fi \
    && if ! id -u appuser >/dev/null 2>&1; then useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash appuser; fi \
    && python -m venv /opt/venv \
    && chown -R appuser:appuser /opt/venv /home/appuser \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/venv/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/workspace/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0,1

WORKDIR /workspace

COPY pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python /opt/venv/bin/python --no-install-project

COPY --chown=appuser:appuser . .

RUN mkdir -p /workspace/.cache/huggingface \
    && chown -R appuser:appuser /workspace

USER appuser

CMD ["bash"]
