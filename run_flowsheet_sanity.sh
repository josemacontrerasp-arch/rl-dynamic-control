#!/usr/bin/env bash
# Run the flowsheet_graph.py sanity check locally.
# Tested on Python 3.10. Install torch CPU first (sandbox proxy blocks
# download.pytorch.org, so we do this locally).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python -m pip install --upgrade pip

# 1) PyTorch CPU (pick the variant you have disk space for)
if ! python -c "import torch" 2>/dev/null; then
  echo "Installing torch (CPU)…"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# 2) PyTorch Geometric (pure-Python, no C++ extensions needed)
python -m pip install torch-geometric

# 3) sklearn + joblib for surrogate loading (if not already installed)
python -m pip install scikit-learn joblib pytest

echo
echo "── Sanity check ──────────────────────────────────────────────"
python flowsheet_graph.py

echo
echo "── Unit tests ────────────────────────────────────────────────"
pytest -q test_flowsheet_graph.py
