# Contributing

## Environment

- Python **3.10** or **3.11** (matches CI).
- Install in editable mode with dev tools:

```bash
pip install -e ".[dev]"
```

Optional ONNX / ONNX Runtime: `pip install -e ".[deploy]"`.

## Local checks (same gates as CI)

```bash
make ci
```

This runs, in order: `ruff check`, `ruff format --check`, `pyright`, and `pytest` with coverage on `unet_pet_seg`.

Individual targets: `make lint`, `make fmt-check`, `make typecheck`, `make test`, `make test-cov`.

## Tests

```bash
make test
```

Training smoke tests use tiny tensors on CPU and a temporary run directory; they do not require downloading the full Oxford-IIIT Pet dataset.

## Style

- **Ruff** for lint (`E`, `F`, `I`) and format at line length 100.
- **Pyright** in basic mode for static checks on `src/` and `scripts/`.
