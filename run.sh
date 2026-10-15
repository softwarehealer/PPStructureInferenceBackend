#!/usr/bin/env bash
echo
echo "========================================================================"
echo "    PPStructure API - Starting Server"
echo "========================================================================"
echo

# Set the script directory as current directory
cd "$(dirname "$0")" || exit 1

export PYTHONPATH=src
export PYTHONPYCACHEPREFIX=../tmp/pycache

# Python to use: $PYTHON if set, else the project virtualenv, else python on PATH
if [ -z "$PYTHON" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python
    fi
fi

# Port 8000 is what the BBox viewer calls by default
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"

# Run the application
"$PYTHON" -m uvicorn main:app --host "$HOST" --port "$PORT" --reload

# If server stops, pause before closing
echo ""
echo "========================================================================"
echo "Server stopped"
echo "========================================================================"
read -p "Press [Enter] to exit"
