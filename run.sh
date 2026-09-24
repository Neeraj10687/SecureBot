#!/usr/bin/env bash
# SecureBot launcher — Arch-friendly, uses project-local venv.
# Usage:  ./run.sh            # default port 8501
#         ./run.sh 8505       # custom port
set -euo pipefail

cd "$(dirname "$0")"
PORT="${1:-8501}"

# On Arch, prefer 'python' (python3 is often just a symlink or missing)
PY="/usr/bin/python3.12"

if [ -z "$PY" ]; then
    echo "[ERROR] No Python found. Install with: sudo pacman -S python python-pip"
    exit 1
fi

# Rebuild venv if streamlit isn't actually installed
if [ ! -x "./venv/bin/streamlit" ]; then
    echo "==> Setting up venv with: $($PY --version)"

    rm -rf venv
    "$PY" -m venv venv --copies

    if [ ! -x "./venv/bin/python" ]; then
        echo "[ERROR] venv creation failed."
        echo "        Make sure python-pip is installed: sudo pacman -S python-pip"
        exit 1
    fi

    echo "==> Upgrading pip..."
    ./venv/bin/python -m pip install --upgrade pip

    echo "==> Installing dependencies (first run takes ~60s)..."
    ./venv/bin/pip install -r requirements.txt

    if [ ! -x "./venv/bin/streamlit" ]; then
        echo "[ERROR] streamlit failed to install."
        echo "        Try manually: ./venv/bin/pip install streamlit"
        exit 1
    fi
    echo "==> Setup complete."
fi

echo "==> Launching Streamlit on port $PORT ..."
exec ./venv/bin/streamlit run app.py --server.port "$PORT" --server.headless true
