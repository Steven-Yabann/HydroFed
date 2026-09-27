PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup test

setup:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest tests/ -v

model-demo:
	$(PYTHON) scripts/inspect_model.py