#!/bin/bash
# -----------------------------------------
# Shell script to initialize environment and run main orchestrator

source .env 2>/dev/null || echo "[INFO] .env file not found. Continuing with exported environment vars."

# Activate virtual environment if exists
if [ -d "venv" ]; then
  echo "[START] Activating virtual environment..."
  source venv/bin/activate
fi

# Install required Python packages
echo "[START] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Launch main pipeline
echo "[START] Running Quant-Aero-Log Engine..."
python main.py

# Log end
echo "[COMPLETE] Quant-Aero-Log run finished."
