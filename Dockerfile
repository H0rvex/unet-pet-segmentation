# CPU-only image: install the package and run the test suite (no dataset download in default tests).
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pytest -q

# Default: re-run tests; override CMD to run training or inference with mounted volumes.
CMD ["pytest", "-q"]
