#!/usr/bin/env bash
# B0 floor zip. Not the default team submission — that is ../submission.zip
# from submission/build.sh (A1 RAG). This packs main.py only, same idea as
# the starter kit's 01_zero_shot/build.sh.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
ZIP="$ROOT/zero_shot_submission.zip"
rm -f "$ZIP"
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
zip -r "$ZIP" main.py -x '__pycache__/*' '*.pyc' '*.pyo'
echo "-> $ZIP"
echo "NOTE: B0 floor only. Upload submission/build.sh --with-models for A1."
"$ROOT/scripts/check_zip_budget.sh" "$ZIP"
