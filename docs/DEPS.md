# Dependencies

The evaluation image installs exactly the pins in [`requirements-participant.txt`](../requirements-participant.txt). That file is a byte copy of the starter kit's `requirements-participant.txt` ([jucls/IEEE-CIS-DSQA-Starter-Kit](https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit), commit `5e596be2f9d9d5ff8d8c89bf8bd8d43b42ec8e8a`). Do not add packages, loosen pins, or generate a separate lockfile for the submission. There is no internet at eval time, so an import that is not already in the image fails the run.

`main.py` cannot assume vLLM or flash-attn. They are not in the list. Attention is `sdpa`, with `eager` if `sdpa` load fails. Generation goes through `transformers` and `accelerate` (`device_map="cuda"`).

## What the skeleton imports

| Package | Pin (from the participant file) | Where |
|---|---|---|
| `torch` | `2.8.0` | A1 and B0 load / generate |
| `transformers` | `5.17.0` | `AutoModelForCausalLM`, `AutoTokenizer` |
| `accelerate` | `1.15.0` | pulled in by `device_map="cuda"` (not imported by name) |
| `sentence-transformers` | `6.0.1` | A1 query and chunk embeddings (`all-MiniLM-L6-v2`, local folder) |
| `docling` | `2.127.0` | A1 PDF parse. Layout artifacts bundled in the zip; OCR and table structure off |
| `numpy` | `2.2.6` | A1 cosine rank, same as starter `02_rag` |
| `peft` | `0.21.0` | **Optional A2 stub.** Imported only if `submission/adapters/adapter_config.json` exists. Not trained in this skeleton. |

`faiss-cpu==1.15.0` is in the image. A1 does not import it. The starter path uses a normalized numpy dot product, which is what they timed. A TODO in `submission/main.py` leaves a later swap to FAISS for ML if profiling says the matmul matters.

Other pins (LangChain, LlamaIndex, Chroma, rank-bm25, eval metrics, PDF extras, and the rest) are available and unused by this skeleton. Do not import them "just in case".

## Not used, and not allowed

- vLLM, flash-attn, or any wheel absent from `requirements-participant.txt`
- A runtime `pip install`
- Downloading OLMo 3.1 32B Instruct, MiniLM, or Docling weights from inside `answer()`

MiniLM and Docling layout are downloaded **only** by `submission/build.sh --with-models`, on a machine with internet, then stored in the zip. The default `./build.sh` skips that download so CI stays small.
