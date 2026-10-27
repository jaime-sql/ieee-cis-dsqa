# IEEE CIS DSQA — team submission skeleton

Packaging and run skeleton for the [IEEE-CIS Technical Challenge on Domain-Specific Question Answering with LLMs](https://cis.taskforce.ieee.org/gpais/tc-domain-qa-llms/join-the-challenge/) (GPAIS / DSQA). Submissions go to [CodaBench competition 17954](https://www.codabench.org/competitions/17954/). Official starter (MIT, Copyright (c) 2026 jucls): [jucls/IEEE-CIS-DSQA-Starter-Kit](https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit). This repo adapts that starter. It does not vendor a second copy of the base model.

This checkout is the packaging skeleton for the Wed Oct 28, 2026 end-of-day America/El_Salvador review. It does not register the team on CodaBench and it does not report leaderboard scores.

## Tracks

| Track | Where | Role |
|---|---|---|
| **A1 (default zip)** | `submission/` | Starter `02_rag`: Docling layout + MiniLM embeddings + cosine retrieval, context capped at about 10k tokens, plus domain-aware prompts. |
| **B0 (floor, not the default zip)** | `baselines/zero_shot/` | Starter `01_zero_shot`: frozen base model, no retrieval. Pipeline smoke and latency floor. |
| **A2 (not in this zip)** | hook in `submission/main.py` | Optional LoRA later. TODO(ML). Train offline; ship adapter weights only. |

`submission/main.py` exposes `answer(question: str, corpus_dir: str, model_id: str) -> str`. The base model is loaded once per process and reused. `model_id` is the harness argument; weights come from the eval cache (`local_files_only=True`). Do not download OLMo 3.1 32B Instruct into this repo.

## Ownership

- **AI Eng** owns this packaging and the run skeleton (`main.py` contract, `build.sh`, zip budget).
- **Data** owns corpus prep from `sample/sites.txt` for local iteration and future adapter data. See `data/README.md`. At official eval the index is built inside the run from the blind `corpus_dir` PDFs. Do not ship a prebuilt index of the eval corpus.
- **ML** owns B0 vs A1 vs A2. A1 is the default. A2 stays a TODO until A1 has been looked at on the public sample only.

## Build the zip

From a machine that has `zip`:

```bash
cd submission
./build.sh                  # code + prompts only; CI / smoke; NOT eval-ready
./build.sh --with-models    # also download and bundle Docling layout + MiniLM
```

`./build.sh` writes `../submission.zip`. Aux-model download is off unless you pass `--with-models`, so CI does not pull those weights. The eval image cannot download them later (no internet), so the zip you upload must be the `--with-models` build. The script refuses a zip over 1 GB (10^9 bytes) via `scripts/check_zip_budget.sh`. It never downloads the 32B base weights.

B0, separate zip, not the team default:

```bash
cd baselines/zero_shot && ./build.sh    # -> ../../zero_shot_submission.zip
```

## Smoke test locally

```bash
python -m py_compile submission/main.py baselines/zero_shot/main.py
python scripts/smoke_answer.py
```

`scripts/smoke_answer.py` checks syntax, domain-prompt selection, chunking, and that `answer()` returns `str` with the model load cached. The real A1 call on a code-only tree fails with `FileNotFoundError` until aux models exist. The real B0 call fails until the eval cache has OLMo. Neither path downloads the 32B weights. You cannot fully test the OLMo load outside the challenge servers.

## Dependencies

Eval packages are pinned in [`requirements-participant.txt`](requirements-participant.txt), copied from the starter. What this skeleton actually imports is listed in [`docs/DEPS.md`](docs/DEPS.md). Nothing outside that file can be added. No vLLM and no flash-attn.

Rules, the 1 GB budget, and the packaging checklist: [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md).

## Sample files

[`sample/sample_qa.json`](sample/sample_qa.json) and [`sample/sites.txt`](sample/sites.txt) are copied from the starter kit (25 illustrative QA pairs; 32 source sites). They are not the sealed benchmark and not the public leaderboard slice. See [`sample/README.md`](sample/README.md).

## Starter license

Substantial parts of `submission/main.py` and `baselines/zero_shot/main.py` are adapted from the starter kit, which is MIT, Copyright (c) 2026 jucls. The license text is in [`NOTICE`](NOTICE).
