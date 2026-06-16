#!/usr/bin/env bash
# Automated environment setup for the production extension.
#
# Creates a Python 3.11 virtualenv (.venv) and installs pinned dependencies.
# TensorFlow 2.15 (needed for parity with the published notebooks) supports
# Python 3.9-3.11 only, so we deliberately do NOT use the system Python 3.13/3.14.
#
# Usage:  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN=""
for c in python3.11 python3.10 python3.9; do
  if command -v "$c" >/dev/null 2>&1; then PYBIN="$c"; break; fi
done

if [ -z "$PYBIN" ]; then
  echo "No TensorFlow-compatible Python (3.9-3.11) found."
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "Installing python@3.11 via Homebrew..."
    brew install python@3.11
    PYBIN="$(brew --prefix)/bin/python3.11"
  else
    echo "Please install Python 3.11 (e.g. 'brew install python@3.11' or pyenv) and re-run."
    exit 1
  fi
fi
echo "Using interpreter: $PYBIN ($($PYBIN --version))"

"$PYBIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Apple Silicon detected -> installing requirements-macos.txt (metal GPU)"
  pip install -r requirements-macos.txt
else
  pip install -r requirements.txt
fi

# Make the src/ package importable.
pip install -e .

echo
echo "Done. Activate with:  source .venv/bin/activate"
