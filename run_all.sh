#!/usr/bin/env bash
# Compatibility verification for the bahrain_rent_solution package.
#
#   ./run_all.sh [python-interpreter]
#
# Runs check_setup.py (seconds) then smoke_test.py (~2-4 min) with the given
# interpreter (default: python3). Use the interpreter you plan to run the
# notebook with, e.g.:  ./run_all.sh ../venv/bin/python
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${1:-python3}"

echo "==================================================================="
echo " bahrain_rent_solution — verification"
echo " interpreter: $PY"
echo "==================================================================="
echo

echo "--- 1/2: check_setup.py (fast compatibility check) ----------------"
"$PY" "$HERE/check_setup.py"
echo

echo "--- 2/2: smoke_test.py (reduced-scale end-to-end run) -------------"
"$PY" "$HERE/smoke_test.py"
echo

echo "==================================================================="
echo " ALL CHECKS PASSED"
echo "==================================================================="
echo
echo "To produce the real submission (~1.5-2 h):"
echo
echo "  cd \"$HERE\""
echo "  $PY -m jupyter nbconvert --to notebook --execute final_solution_simplified.ipynb"
echo
echo "or open final_solution_simplified.ipynb in Jupyter and 'Run All'."
echo "Output file: submission_simplified.csv"
echo
echo "Note: training creates a harmless catboost_info/ log folder."
