VENV := .venv
PYTHON := $(if $(wildcard $(VENV)/Scripts/python.exe),$(VENV)/Scripts/python.exe,$(VENV)/bin/python)
EXP ?= main

.PHONY: help setup data decode run figures test smoke lint clean

help:
	@echo "setup      create the venv and install the package with dev extras"
	@echo "data       download and preprocess BCI IV-2a"
	@echo "decode     train decoders and cache posteriors"
	@echo "run        execute an experiment matrix (make run EXP=lambda_sweep)"
	@echo "figures    render the paper figures from a run directory"
	@echo "test       pytest"
	@echo "smoke      synthetic posteriors, 1 subject, 2 episodes, seconds"
	@echo "lint       ruff check + mypy"
	@echo "clean      remove caches and build leftovers"

setup:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/00_download_data.py
	$(PYTHON) scripts/01_preprocess.py

decode:
	$(PYTHON) scripts/02_cache_posteriors.py

run:
	$(PYTHON) scripts/03_run_experiment.py experiment=$(EXP)

figures:
	$(PYTHON) scripts/05_make_figures.py

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/03_run_experiment.py experiment=smoke

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
