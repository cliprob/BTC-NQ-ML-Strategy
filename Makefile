.PHONY: check test smoke full-reproduction clean-notebooks

PYTHON ?= python

check: test smoke

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) scripts/smoke_run.py

full-reproduction:
	$(PYTHON) scripts/full_reproduction.py

clean-notebooks:
	$(PYTHON) scripts/clean_notebooks.py notebooks
