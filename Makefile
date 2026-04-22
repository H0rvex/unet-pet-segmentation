.PHONY: install train train-aug baseline eval viz infer curves export-onnx bench test test-cov lint fmt fmt-check typecheck ci clean

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

infer:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make infer CHECKPOINT=runs/<ts>/best.pth INPUT=path/to/img_or_dir"; exit 1; }
	@[ "$(INPUT)" ] || { echo "Usage: make infer CHECKPOINT=... INPUT=path/to/img_or_dir"; exit 1; }
	python scripts/infer.py --checkpoint $(CHECKPOINT) --input $(INPUT)

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

test-cov:
	pytest -q --cov=unet_pet_seg --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff format .

fmt-check:
	ruff format --check .

typecheck:
	pyright

ci: lint fmt-check typecheck test-cov

clean:
	rm -rf runs/ artifacts/*.pth __pycache__ .pytest_cache .ruff_cache
