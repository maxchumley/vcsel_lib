#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install ruff build

ruff check vcsel_lib.py
pytest -q
python -m build
