#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev,rl]"
python -m pip install ruff build

ruff check vcsel_lib.py
python -m compileall -q vcsel_lib.py examples rl tests
pytest -q
python -m build
