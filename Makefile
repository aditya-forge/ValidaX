# Common tasks. Run from the repository root (Linux/macOS, or Git Bash on Windows).
PYTHON ?= python
IMAGE  ?= validax-console
PORT   ?= 8501

.PHONY: help install install-dev lint test pipeline app sql docker-build docker-run docker-stop

help:
	@echo "install       install the pinned project environment (requirements.txt)"
	@echo "install-dev   install the environment plus pytest and ruff"
	@echo "lint          run ruff on src/, app/ and tests/"
	@echo "test          run the unit and smoke tests"
	@echo "pipeline      rebuild data/processed, outputs/ and app/model.pkl from data/raw (overwrites results)"
	@echo "app           start the Streamlit console on port $(PORT)"
	@echo "sql           run sql/fraud_analysis.sql in DuckDB"
	@echo "docker-build  build the container image"
	@echo "docker-run    run the container (4 GB, 4 CPUs, read-only, port $(PORT))"
	@echo "docker-stop   stop and remove the container"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install pytest==9.2.0 ruff==0.17.6

lint:
	$(PYTHON) -m ruff check src app tests

test:
	$(PYTHON) -m pytest

pipeline:
	$(PYTHON) src/data_processing.py
	$(PYTHON) src/features.py
	$(PYTHON) src/model.py
	$(PYTHON) src/scoring.py

app:
	PORT=$(PORT) PYTHON=$(PYTHON) ./scripts/run_app.sh

sql:
	duckdb -c ".read sql/fraud_analysis.sql"

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run -d --name $(IMAGE) \
		--memory=4g --memory-swap=4g --cpus=4 --read-only --tmpfs /tmp \
		-p $(PORT):8501 \
		-v "$(CURDIR)/data/processed:/srv/fraud-console/data/processed:ro" \
		$(IMAGE)

docker-stop:
	docker stop $(IMAGE) && docker rm $(IMAGE)
