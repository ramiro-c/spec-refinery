#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PYTHON="${VENV}/bin/python"
UVICORN="${VENV}/bin/uvicorn"
STREAMLIT="${VENV}/bin/streamlit"

require_free_port() {
  local port="$1" name="$2"
  if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[run] ERROR: port ${port} (${name}) is already in use." >&2
    echo "[run] Free it with: lsof -nP -iTCP:${port} -sTCP:LISTEN" >&2
    exit 1
  fi
}

require_free_port 8000 "API"
require_free_port 8501 "UI"

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
"${STREAMLIT}" run ui.py --server.port 8501 --server.headless true
