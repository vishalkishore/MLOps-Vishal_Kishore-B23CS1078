FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.23 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock .python-version README.md /app/

RUN uv sync --frozen

COPY . /app

CMD ["uv", "run", "python", "main.py", "--help"]
