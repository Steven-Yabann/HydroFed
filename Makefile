PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup test data partition partition-skab partition-synthetic baseline federated evaluate monitor-up monitor-down operator-feed publish admin api dashboard

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
	$(PYTHON) scripts/prepare_data.py --download --input data/SKAB --output data/partitions --partition-strategy temporal --minimum-anomalies 100
partition-skab:
	$(PYTHON) scripts/prepare_data.py --download --input data/SKAB --output data/partitions_skab --window 10 --stride 1 --partition-strategy temporal --minimum-anomalies 100
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
operator-feed:
	$(PYTHON) scripts/build_operator_feed.py
# Copy the results the hosted API serves into deploy/published, which is
# committed so the Render build (which cannot see artifacts/ or data/) has them.
publish:
	rm -rf deploy/published && mkdir -p deploy/published/operator deploy/published/comparison
	cp artifacts/operator/*.json deploy/published/operator/
	cp artifacts/comparison/regimes.json artifacts/comparison/models.json artifacts/comparison/scenarios.json artifacts/comparison/personalised.json deploy/published/comparison/
	cp data/partitions_skab/partition_summary.json deploy/published/
admin:
	$(PYTHON) scripts/create_user.py --role admin --email $(EMAIL) --name "$(NAME)"
api:
	$(PYTHON) -m uvicorn hydrofed.api:create_app --factory --reload --port 8000
dashboard:
	cd frontend && npm run dev
