.PHONY: check test smoke clean-notebooks

PYTHON ?= python

check: test smoke

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/smoke_run.py

clean-notebooks:
	$(PYTHON) scripts/clean_notebooks.py notebooks
