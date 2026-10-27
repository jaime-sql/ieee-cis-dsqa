# Data placeholder

**Owner: Data Engineer.** This directory is the landing zone for local corpus work. The packaging skeleton does not crawl anything.

## What to build later

`sample/sites.txt` (copied from the [starter kit](https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit)) lists the 32 site roots the official corpus was drawn from, including distractors. Use it as the source list for a reproducible local corpus and, if ML asks, for text that could train an A2 adapter. Cite every external source you keep.

Suggested layout once you start (all gitignored except this README):

```
data/raw/         # downloads, unchanged
data/corpus/      # PDFs or text you would point a local corpus_dir at
data/processed/   # notes, manifests, licenses
```

## What not to put in the submission zip

Official eval passes its own blind `corpus_dir` of PDFs. `submission/main.py` parses `corpus_dir/*.pdf` with Docling, chunks, and embeds **during the one-hour run**, matching starter `02_rag`. A prebuilt index of *their* corpus cannot be shipped, because we do not have that corpus ahead of time.

Ship embedding and layout weights only (`submission/build.sh --with-models`). Do not commit raw crawls, PDF dumps, or `*.npy` / FAISS indexes.

## Contract for the index built at eval

Implemented in `submission/main.py` (`_build_index`):

- Input: every `*.pdf` in `corpus_dir` (sorted by name). One failed PDF is skipped.
- Chunks: paragraph packing, about 800 characters, 150-character overlap.
- Vectors: `all-MiniLM-L6-v2` from `submission/embed_model`, L2-normalized, CPU embedder so the H100 stays on the 32B model.
- Rank: cosine via a numpy dot product. Top chunks packed up to ~10_000 tokens.

TODO(Data): a manifest format and a local dry-run corpus. TODO(ML): A2 training data drawn from that corpus, only after A1 is the default path you are iterating on. Do not treat `sample/sample_qa.json` as the test set for leaderboard claims.
