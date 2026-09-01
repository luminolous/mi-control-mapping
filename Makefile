VENV := .venv
PYTHON := $(if $(wildcard $(VENV)/Scripts/python.exe),$(VENV)/Scripts/python.exe,$(VENV)/bin/python)
EXP ?= main
RUN ?=

.PHONY: help setup data decoders decode run analyze figures test smoke lint clean

help:
	@echo "setup      create the venv and install the package with dev extras"
	@echo "data       download BCI IV-2a"
	@echo "decoders   fit decoders and report kappa against the published range"
	@echo "decode     cache posteriors for the runner"
	@echo "run        execute an experiment matrix (make run EXP=lambda_sweep)"
	@echo "analyze    fit the hypothesis models for a run (make analyze RUN=<dir>)"
	@echo "figures    render the paper figures (make figures RUN="<dir> <dir> ...")"
	@echo "test       pytest"
	@echo "smoke      synthetic posteriors, 1 subject, 4 mappings, 4 episodes, seconds"
	@echo "lint       ruff check + mypy"
	@echo "clean      remove caches and build leftovers"

setup:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/00_download_data.py

# A guard, not a production stage: it reports kappa against the published range
# and is what catches a preprocessing error. Folding it into `decode` would mean
# nobody runs it on its own. See docs/decisions.md D13b.
decoders:
	$(PYTHON) scripts/01_fit_decoders.py

decode:
	$(PYTHON) scripts/02_cache_posteriors.py

run:
	$(PYTHON) scripts/03_run_experiment.py experiment=$(EXP)

analyze:
	$(PYTHON) scripts/04_analyze.py $(RUN)

figures:
	$(PYTHON) scripts/05_make_figures.py $(RUN)

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
