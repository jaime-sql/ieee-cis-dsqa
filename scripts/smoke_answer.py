"""Local contract smoke for the DSQA skeleton.

Runs without a GPU and without OLMo 3.1 32B weights.

1. ``python -m py_compile`` on the A1 and B0 entrypoints.
2. Import ``answer`` and check prompt / chunk helpers (no model).
3. Call A1 ``answer()`` with a temp corpus. Missing Docling/MiniLM assets
   raise ``FileNotFoundError`` before any download or GPU use. That is the
   graceful code-only path.
4. Monkeypatch load/generate and assert ``answer()`` returns ``str`` and
   that the model load is cached across two calls.
5. Call B0 ``answer()``. Without a local OLMo cache this raises before
   generation (import error or ``local_files_only``). Then monkeypatch and
   assert a string plus a single load.

This does not score answers. The 25-pair sample is not the leaderboard.
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A1_PATH = ROOT / "submission" / "main.py"
B0_PATH = ROOT / "baselines" / "zero_shot" / "main.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset(module) -> None:
    module._STATE["llm_id"] = None
    module._STATE["tokenizer"] = None
    module._STATE["model"] = None
    if "index_key" in module._STATE:
        module._STATE["index_key"] = None
        module._STATE["chunks"] = None
        module._STATE["embeddings"] = None
        module._STATE["embedder"] = None


def _check_prompts(a1) -> None:
    dones_q = (
        "At what point in time does Be7 stop being the main source of "
        "activity in the Heat Exchanger of DONES?"
    )
    ai_q = (
        "In the paper, why do transformer models use a learned tokenizer "
        "instead of a fixed vocabulary?"
    )
    neutral_q = "What is the boiling point of argon at standard pressure?"
    both_q = "In the DONES paper, how does the transformer attend?"

    assert a1.detect_domain(dones_q) == "dones", a1.detect_domain(dones_q)
    assert a1.detect_domain(ai_q) == "ai_papers", a1.detect_domain(ai_q)
    assert a1.detect_domain(neutral_q) == "neutral"
    assert a1.detect_domain(both_q) == "neutral"

    dones_prompt = a1.select_system_prompt(dones_q)
    ai_prompt = a1.select_system_prompt(ai_q)
    neutral_prompt = a1.select_system_prompt(neutral_q)
    assert isinstance(dones_prompt, str) and "DONES" in dones_prompt
    assert isinstance(ai_prompt, str) and "machine-learning" in ai_prompt
    assert isinstance(neutral_prompt, str) and neutral_prompt != dones_prompt

    procedural = a1.build_user_content("", "What is the decay constant of Be7?")
    assert "calculation steps" in procedural
    assert procedural.strip().endswith("Be7?")

    chunks = a1._chunk_markdown("Para one.\n\nPara two is a bit longer.\n\nPara three.")
    assert chunks and all(isinstance(chunk, str) for chunk in chunks)


def _expect_graceful(label: str, fn) -> None:
    try:
        value = fn()
    except Exception as exc:
        print(f"graceful {label}: {type(exc).__name__}: {exc}")
        return
    raise SystemExit(
        f"{label} returned {type(value).__name__} without a model cache; "
        "expected an exception before generation"
    )


def _assert_cached_string(module, corpus_dir: str, patch_index: bool) -> None:
    calls = {"llm": 0, "index": 0, "gen": 0}
    _reset(module)

    def _load_llm(model_id: str):
        calls["llm"] += 1
        return (f"tok:{model_id}", f"model:{model_id}")

    def _generate(*args, **kwargs):
        del kwargs
        calls["gen"] += 1
        return "stub-answer"

    module._load_llm = _load_llm
    module._generate = _generate
    if patch_index:
        module._require_rag_assets = lambda: None

        def _build_index(corpus_dir: str):
            del corpus_dir
            calls["index"] += 1
            return (["chunk"], "embeddings", "embedder")

        module._build_index = _build_index
        module._build_context = lambda *args, **kwargs: "Context fragment.\n\n"

    first = module.answer("What is a tokamak?", corpus_dir, "local-model-id")
    second = module.answer("Second question about DONES lithium?", corpus_dir, "local-model-id")
    assert isinstance(first, str) and first == "stub-answer"
    assert isinstance(second, str) and second == "stub-answer"
    assert calls["llm"] == 1, calls
    assert calls["gen"] == 2, calls
    if patch_index:
        assert calls["index"] == 1, calls
    print(
        f"contract {module.__name__}: answer() -> str, "
        f"llm loads={calls['llm']}, index builds={calls['index']}"
    )


def main() -> None:
    # Keep the unmocked calls from hitting the hub. Eval is offline too.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    for path in (A1_PATH, B0_PATH):
        py_compile.compile(str(path), doraise=True)
        print(f"py_compile ok: {path.relative_to(ROOT)}")

    a1 = _load_module("dsqa_a1", A1_PATH)
    b0 = _load_module("dsqa_b0", B0_PATH)
    _check_prompts(a1)
    print("prompt and chunk helpers ok")

    with tempfile.TemporaryDirectory(prefix="dsqa-corpus-") as corpus_dir:
        _expect_graceful(
            "a1-missing-assets",
            lambda: a1.answer(
                "What does a destructive beam diagnostic do?",
                corpus_dir,
                "olmo-local-cache",
            ),
        )
        _assert_cached_string(a1, corpus_dir, patch_index=True)

        _expect_graceful(
            "b0-no-local-olmo",
            lambda: b0.answer(
                "What does a destructive beam diagnostic do?",
                corpus_dir,
                "olmo-local-cache",
            ),
        )
        _assert_cached_string(b0, corpus_dir, patch_index=False)

    print("smoke ok")
    print(
        "NOTE: this does not load OLMo 3.1 32B. Full model load only works "
        "in the eval environment, which has the weights cached and no internet."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
