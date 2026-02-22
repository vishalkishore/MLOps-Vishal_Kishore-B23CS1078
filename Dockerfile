# Build:  docker build -t lab3-train .
# Run:    docker run lab3-train

FROM python:3.12-slim


RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY .env .env
COPY utils.py data.py train.py eval.py ./

CMD ["uv", "run", "python", "train.py"]
