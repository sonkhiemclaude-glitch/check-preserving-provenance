# All targets run on synthetic data; none downloads anything.
PY ?= python3
export PYTHONPATH := src

.PHONY: help install test lint typecheck demo tables plots clean-demo reference-e1 reference-e2 reference-e4

help:
	@echo "install    pip install -e .[dev]"
	@echo "test       pytest -q (synthetic data only)"
	@echo "lint       ruff check ."
	@echo "typecheck  mypy src"
	@echo "demo       synthetic smoke test of all stages -> outputs/synthetic_demo (not reference results)"
	@echo "reference-e1   reference protocol E1/E3 (~3 min on 2 vCPU)"
	@echo "reference-e2   reference protocol E2 + sensitivity (~50 min on 2 vCPU)"
	@echo "reference-e4   reference protocol E4 (several GB of cache)"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .

typecheck:
	mypy src

demo:
	$(PY) scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml

tables:
	$(PY) scripts/make_tables.py --input outputs/synthetic_demo/results --output outputs/synthetic_demo/tables

plots:
	$(PY) scripts/make_plots.py --input outputs/synthetic_demo/results --output outputs/synthetic_demo/figures

reference-e1:
	$(PY) scripts/run_experiment.py --config configs/e1_e3.yaml

reference-e2:
	$(PY) scripts/run_experiment.py --config configs/e2.yaml

reference-e4:
	$(PY) scripts/run_experiment.py --config configs/e4.yaml
