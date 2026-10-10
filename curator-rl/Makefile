-include requirements.lock   # optional pin file (E1: the file is not in the repo)
SRC := src/curator_rl

.PHONY: install test lint smoke

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest tests

lint:
	ruff check .

smoke:
	python -m curator_rl.cli init-run --config configs/experiment/smoke.yaml --set experiment.seed=0
