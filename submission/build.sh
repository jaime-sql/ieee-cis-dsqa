#!/usr/bin/env bash
# Build ../submission.zip — the A1 (RAG + prompt polish) CodaBench upload.
#
# Adapted from the starter kit's 02_rag/build.sh (MIT, Copyright (c) 2026 jucls):
# https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit
#
# Docling and sentence-transformers are already installed in the evaluation
# image. Two weight bundles they need are not, and that image has no
# internet: Docling's layout model, and sentence-transformers/all-MiniLM-L6-v2.
# ``--with-models`` downloads both here (where there is internet) and packs
# them into the zip. This script never downloads OLMo 3.1 32B Instruct.
#
# Default (no flags) packs code and prompt templates only, so CI does not
# need those downloads. That zip is NOT eval-ready. Upload only a zip built
# with --with-models.
#
#   ./build.sh                 # code + prompts, warns, still checks 1 GB
#   ./build.sh --with-models   # also bundle embed_model/ and docling_models/
#
# --with-models needs ``docling-tools`` and ``sentence-transformers``
# importable at the versions pinned in ../requirements-participant.txt,
# plus ``zip``. It does not need a GPU.
set -euo pipefail
cd "$(dirname "$0")"

WITH_MODELS=0
if [[ $# -gt 1 ]]; then
  echo "usage: ./build.sh [--with-models]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --with-models) WITH_MODELS=1 ;;
    -h|--help)
      echo "usage: ./build.sh [--with-models]"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: ./build.sh [--with-models]" >&2
      exit 2
      ;;
  esac
fi

ROOT="$(cd .. && pwd)"
ZIP="$ROOT/submission.zip"

if [[ "$WITH_MODELS" -eq 1 ]]; then
  rm -rf docling_models embed_model
  # HF_*_OFFLINE=0 in case the shell inherited an offline flag.
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    docling-tools models download layout -o ./docling_models
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('embed_model')
"
else
  echo "NOTE: code-only zip (no embed_model/ or docling_models/ download)."
  echo "      Not eval-ready. Re-run ./build.sh --with-models before CodaBench upload."
fi

rm -f "$ZIP"
find . -type d -name '__pycache__' -prune -exec rm -rf {} +

paths=(main.py)
if [[ -d prompts ]]; then
  paths+=(prompts)
fi
if [[ -d embed_model ]]; then
  paths+=(embed_model)
fi
if [[ -d docling_models ]]; then
  paths+=(docling_models)
fi
# A2 only: include adapters when ML has actually exported one.
if [[ -f adapters/adapter_config.json ]]; then
  paths+=(adapters)
fi

zip -r "$ZIP" "${paths[@]}" \
  -x '__pycache__/*' '*/__pycache__/*' '*.pyc' '*/*.pyc' '*.pyo'

echo "-> $ZIP"
if [[ ! -d embed_model || ! -d docling_models ]]; then
  echo "WARNING: RAG aux models are not in this zip. Eval cannot download them."
fi

"$ROOT/scripts/check_zip_budget.sh" "$ZIP"
