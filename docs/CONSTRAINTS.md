# Eval constraints and packaging

Sources: the [join page](https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/), the [evaluation page](https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/evaluation), and the [starter kit README](https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit). This file records those rules for the team. It does not add house metrics and it does not claim a CodaBench registration.

## Hard limits

| Rule | Value |
|---|---|
| GPU | One H100. Inference is `transformers` + `accelerate`, attention `sdpa` with `eager` fallback. No vLLM, no flash-attn. |
| Wall clock | **1 hour** for the whole run: model load, index build, any other preprocessing, and inference on **every** question. Overruns are killed and marked failed. |
| Network at eval | **None.** Internet is allowed only while preparing files that go into the zip. |
| Base model | **OLMo 3.1 32B Instruct**, already in the eval environment. The harness passes `model_id`. Load it from the local cache (`local_files_only=True`). Do not download it at runtime and do not put the 32B weights in the zip. The starter does not hard-code the hub id; do not invent one here. |
| Zip size | **≤ 1 GB.** This repo enforces `size ≤ 1_000_000_000` bytes (decimal GB), which is also under 1 GiB. |
| Dependencies | Only the pins in `requirements-participant.txt`. |
| Entry | Zip root contains `main.py` with `answer(question: str, corpus_dir: str, model_id: str) -> str`. Same process for every question, so load the model once. |
| Corpus | Blind PDFs in `corpus_dir`. No `document_id`. Retrieval searches the whole directory. |
| Cadence | One submission per day per team. Teams of at most four. |
| Fair play | Public external data is allowed if cited. Do not manipulate or jailbreak the judge. |

The starter's `02_rag` notes a measured run of about **36 minutes** for the full corpus plus 50 questions, inside the hour, with the ~10k-token context cap. That number is theirs. Do not quote it as a timing we reproduced. Their note is why A1 keeps `MAX_CONTEXT_TOKENS = 10_000` instead of the model's 65,536-token window (they report that a near-full window OOMs an 80 GB GPU and is too slow).

## What the zip may contain

Ship only auxiliary files the offline run needs:

- `main.py` and `prompts/`
- Docling layout weights (`docling_models/`) and MiniLM (`embed_model/`), from `./build.sh --with-models`
- Optional later: a small PEFT adapter under `adapters/` (A2). Not in the A1 default until ML trains one.

Do not ship:

- OLMo 3.1 32B Instruct weights
- a prebuilt index of the official eval corpus (build that index inside the hour, from `corpus_dir`, as `02_rag` does)
- `__pycache__`, `.pyc`, local virtualenvs, or raw crawl dumps

## Size budget

Keep a margin under 10^9 bytes. After `./build.sh --with-models`, read the table printed by `scripts/check_zip_budget.sh` and `du -sh submission/embed_model submission/docling_models`.

| Piece | Guidance |
|---|---|
| `main.py` + prompts | Negligible (kilobytes). |
| `embed_model/` (all-MiniLM-L6-v2) | Small encoder. Confirm with `du` after download; it is the smaller of the two aux bundles. |
| `docling_models/` (layout only) | The other aux bundle. Layout only — the script does not pull OCR or table models, matching the starter (`do_ocr=False`, `do_table_structure=False`). |
| A2 adapter | Only if it still leaves headroom for load time and the 1 GB cap. Rank and target modules are ML's choice later. Full fine-tunes of 32B do not fit and are out of scope. |
| Code-only zip | Used for CI. It passes the size check and is **not** a valid A1 upload, because eval cannot fetch MiniLM or Docling layout. |

If the measured aux models plus an adapter would cross 1 GB, drop the adapter first. A1 does not need it.

## Packaging checklist

- [ ] `answer(question, corpus_dir, model_id)` is at the root of the zip and returns `str`.
- [ ] Model load is cached. Index build runs once per `corpus_dir`, inside the run.
- [ ] `local_files_only=True` on the base tokenizer and model. No hub download at eval.
- [ ] Attention is `sdpa`, then `eager`. No flash-attn import.
- [ ] Zip built with `submission/build.sh --with-models` (not the code-only default).
- [ ] `scripts/check_zip_budget.sh submission.zip` prints `OK` and shows no `__pycache__`.
- [ ] Zip has no 32B weight shards (`*.safetensors` from OLMo, multi-GB files).
- [ ] Imports are limited to `requirements-participant.txt` (see `docs/DEPS.md`).
- [ ] `sample/` is not required inside the zip. It is dev-only.
- [ ] B0 `zero_shot_submission.zip` is not uploaded as the team default.

## Scoring (do not invent numbers)

Phase 1 is a hidden LLM-as-judge with a binary verdict per question, ranked by domain-balanced accuracy on IFMIF-DONES (D) and AI-research papers (P). Tie-break is lower total inference time. During the competition the public leaderboard uses 50% of the benchmark; the full set is scored after the deadline. Phase 2 (top 20) weights accuracy, expert answer quality, and innovation. The judge identity is not public.

This repository does not compute those scores. Do not report sample-set string overlap, or any local run, as a CodaBench or Phase 1 number. The 25 pairs in `sample/sample_qa.json` are illustrative only.
