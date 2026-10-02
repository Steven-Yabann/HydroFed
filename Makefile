PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup test data partition partition-synthetic baseline federated evaluate monitor-up monitor-down

setup:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest tests/ -v

model-demo:
	$(PYTHON) scripts/inspect_model.py

data:
	$(PYTHON) src/hydrofed/data/pipeline.py
partition:
	$(PYTHON) scripts/prepare_data.py --download --input data/SKAB --output data/partitions
partition-synthetic:
	$(PYTHON) scripts/prepare_data.py --synthetic --output data/partitions
baseline:
	$(PYTHON) scripts/run_baseline.py
federated:
	$(PYTHON) scripts/run_experiment.py
evaluate:
	$(PYTHON) scripts/evaluate_model.py artifacts/baselines/plant_1.pt data/partitions/plant_1.npz
monitor-up:
	docker compose -f monitoring/docker-compose.yml up -d
monitor-down:
	docker compose -f monitoring/docker-compose.yml down
