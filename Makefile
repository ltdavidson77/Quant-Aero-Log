# Makefile - Development & Ops tasks for Quant-Aero-Log

setup:
	@echo "[SETUP] Creating virtual environment and installing packages..."
	python3 -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

run:
	@echo "[RUN] Launching main orchestrator..."
	source venv/bin/activate && python main.py

clean:
	@echo "[CLEAN] Removing Python cache and compiled files..."
	rm -rf __pycache__ */__pycache__ *.pyc *.pyo *.pyd build/ dist/ *.egg-info

reset-db:
	@echo "[DB] Dropping and recreating PostgreSQL schema..."
	psql -U $$DB_USER -d $$DB_NAME -f storage/init_db_schema.sql

lint:
	@echo "[LINT] Checking Python syntax..."
	flake8 . --exclude=venv

freeze:
	@echo "[REQUIREMENTS] Freezing packages..."
	source venv/bin/activate && pip freeze > requirements.txt
