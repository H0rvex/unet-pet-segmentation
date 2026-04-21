.PHONY: install train eval test lint fmt clean

install:
	pip install -e ".[dev]"

train:
	python scripts/train.py

eval:
	@[ "$(CHECKPOINT)" ] || { echo "Usage: make eval CHECKPOINT=artifacts/unet.pth"; exit 1; }
	python scripts/evaluate.py --checkpoint $(CHECKPOINT)

test:
	pytest -q

lint:
	ruff check .

fmt:
	ruff format .

clean:
	rm -rf runs/ artifacts/*.pth artifacts/*.png artifacts/*.json __pycache__ .pytest_cache .ruff_cache
