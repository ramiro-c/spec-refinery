#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PYTHON="${VENV}/bin/python"
UVICORN="${VENV}/bin/uvicorn"
STREAMLIT="${VENV}/bin/streamlit"

if [[ ! -d .chroma ]]; then
  echo "[run] .chroma ausente — ejecutando ingest..."
  "${PYTHON}" ingest.py
fi

echo "[run] API en :8000..."
"${UVICORN}" app:app --port 8000 &
API_PID=$!

cleanup() {
  echo "[run] deteniendo API (pid ${API_PID})..."
  kill "${API_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[run] Streamlit en :8501..."
"${STREAMLIT}" run ui.py --server.port 8501
