"""
B0 floor — zero-shot baseline for pipeline smoke and a latency floor.

Adapted from the IEEE-CIS DSQA starter kit (MIT, Copyright (c) 2026 jucls):
https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit
This is the kit's ``01_zero_shot`` submission: the frozen base model, a
short system prompt, greedy decode, ``max_new_tokens=2048``. No retrieval
and no PEFT. ``corpus_dir`` is accepted so the entry signature matches
and is otherwise unused.

Not the default team zip. The default is A1 in ``submission/`` (RAG +
prompt polish). Build this floor with ``baselines/zero_shot/build.sh``.

``model_id`` is resolved from the local cache only
(``local_files_only=True``). Do not download OLMo 3.1 32B Instruct here.

Entry contract::

    answer(question: str, corpus_dir: str, model_id: str) -> str
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a scientific expert. Answer the following question as accurately "
    "and concisely as possible based on your knowledge."
)

MAX_NEW_TOKENS = 2048

# One load for the whole run. ``answer()`` is called once per question.
_STATE: dict = {"llm_id": None, "tokenizer": None, "model": None}


def _load_llm(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    common = dict(
        dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="sdpa", **common
        )
    except Exception as exc:
        if "out of memory" in str(exc).lower():
            raise
        print(f"[b0] sdpa load failed ({exc}); falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", **common
        )
    model.eval()
    return tokenizer, model


def _load(model_id: str):
    if _STATE["llm_id"] != model_id or _STATE["model"] is None:
        tokenizer, model = _load_llm(model_id)
        _STATE["llm_id"] = model_id
        _STATE["tokenizer"] = tokenizer
        _STATE["model"] = model
    return _STATE["tokenizer"], _STATE["model"]


def _generate(tokenizer, model, question: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


def answer(question: str, corpus_dir: str, model_id: str) -> str:
    """Answer from the base model only. ``corpus_dir`` is unused."""
    if not isinstance(question, str):
        raise TypeError("question must be str")
    if not isinstance(corpus_dir, str):
        raise TypeError("corpus_dir must be str")
    if not isinstance(model_id, str) or not model_id:
        raise TypeError("model_id must be a non-empty str")

    del corpus_dir  # signature compatibility with the harness; no retrieval
    tokenizer, model = _load(model_id)
    text = _generate(tokenizer, model, question)
    if not isinstance(text, str):
        raise TypeError("generation did not return str")
    return text
