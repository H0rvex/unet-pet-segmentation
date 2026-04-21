.PHONY: install train train-aug baseline eval viz curves export-onnx bench test lint fmt ci clean

install:
	pip install -e ".[dev]"

train:
	python scripts/train.py --config configs/unet_base.yaml

train-aug:
	python scripts/train.py --config configs/unet_256_aug.yaml

baseline:
	python scripts/train.py --config configs/baseline_fcn.yaml

eval:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make eval CHECKPOINT=runs/<ts>/best.pth"; exit 1; }
	python scripts/evaluate.py --checkpoint $(CHECKPOINT)

viz:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make viz CHECKPOINT=runs/<ts>/best.pth"; exit 1; }
	python scripts/visualize.py --checkpoint $(CHECKPOINT)

curves:
	@[ "$(RUN_DIR)" ] || { echo "Usage: make curves RUN_DIR=runs/<ts>"; exit 1; }
	python scripts/plot_curves.py --run-dir $(RUN_DIR)

export-onnx:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make export-onnx CHECKPOINT=runs/<ts>/best.pth"; exit 1; }
	python scripts/export_onnx.py --checkpoint $(CHECKPOINT)

bench:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make bench CHECKPOINT=runs/<ts>/best.pth"; exit 1; }
	python scripts/benchmark.py --checkpoint $(CHECKPOINT)

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

ci: lint test

clean:
	rm -rf runs/ artifacts/*.pth __pycache__ .pytest_cache .ruff_cache
