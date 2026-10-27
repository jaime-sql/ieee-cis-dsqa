"""
A1 default submission — RAG over the eval corpus, plus prompt polish.

Adapted from the IEEE-CIS DSQA starter kit (MIT, Copyright (c) 2026 jucls):
https://github.com/jucls/IEEE-CIS-DSQA-Starter-Kit
The retrieval loop follows ``02_rag``: Docling layout (no OCR, no table
structure), ``all-MiniLM-L6-v2`` embeddings, cosine ranking, and a fixed
context budget of about 10_000 tokens. The starter measured that filling
OLMo 3.1 32B Instruct's full 65_536-token window OOMs an 80 GB H100 and
that this budget stayed inside the one-hour limit on their run (~36 min
for the full corpus plus 50 questions). That timing is the starter's
measurement, not one we re-ran here.

Prompt polish (this repo, not the starter): pick a domain system prompt
when the question text has DONES or AI-paper cues, otherwise stay with a
neutral scientific expert. Ask for units and calculation steps when the
question looks procedural. Ground the answer in retrieved context when
that context is present.

This is the zip produced by ``submission/build.sh``. The zero-shot floor
lives in ``baselines/zero_shot/`` and is not the default submission.

Entry contract (eval calls this once per question, same process)::

    answer(question: str, corpus_dir: str, model_id: str) -> str

``model_id`` is whatever the harness passes. Resolve it from the local
cache only. Do not download OLMo 3.1 32B Instruct at runtime, and do not
put those weights in the zip.

Heavy imports are deferred so a checkout without the eval image can still
import this module. The eval image already has every package listed in
``requirements-participant.txt``.
"""

from __future__ import annotations

import re
from pathlib import Path

# Loaded once per process. ``answer()`` is called once per question.
_STATE: dict = {
    "llm_id": None,
    "tokenizer": None,
    "model": None,
    "index_key": None,
    "chunks": None,
    "embeddings": None,
    "embedder": None,
}

_ROOT = Path(__file__).parent
_PROMPTS = _ROOT / "prompts"
_PROMPT_CACHE: dict[str, str] = {}

# Local copies bundled by ``build.sh --with-models``. The eval environment
# has the libraries, not these weight files, and it has no internet.
EMBED_MODEL_PATH = _ROOT / "embed_model"
DOCLING_MODELS_PATH = _ROOT / "docling_models"
# TODO(ML): A2 stretch. Train a small LoRA offline and drop
# ``adapter_config.json`` plus adapter weights in this directory.
# Ship adapters only — never the 32B base weights. Leave the directory
# absent for the A1 zip.
ADAPTER_DIR = _ROOT / "adapters"

MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Starter's measured budget. Do not raise this toward the full context
# window without a fresh H100 memory check; prefill near 65k tokens OOMs.
MAX_CONTEXT_TOKENS = 10_000

SYSTEM_NEUTRAL = (
    "You are a scientific expert. Answer the question accurately and "
    "concisely. When context extracted from the corpus is provided, ground "
    "your answer in that context. Do not invent citations or numbers that "
    "the context does not support."
)
SYSTEM_DONES = (
    "You are a scientific expert on the IFMIF-DONES facility and the "
    "related fusion, accelerator, and nuclear-engineering literature. "
    "Answer accurately and concisely. When context extracted from the "
    "corpus is provided, ground your answer in that context, and keep "
    "physical units explicit. Do not invent citations or numbers that the "
    "context does not support."
)
SYSTEM_AI = (
    "You are a scientific expert on machine-learning and AI research "
    "papers. Answer accurately and concisely. When context extracted from "
    "the corpus is provided, ground your answer in that context. Do not "
    "invent citations or numbers that the context does not support."
)
PROCEDURAL_HINT = (
    "This question asks for a procedure or a derived quantity. Show the "
    "calculation steps and the units, and end with the explicit result."
)

# Generic domain vocabulary, not titles from the 25-pair sample. If both
# sides match, stay neutral. TODO(ML): revise only against the public
# sample and qualitative review — never against sealed questions.
_DONES_CUES = (
    "dones",
    "ifmif",
    "be7",
    "be-7",
    "tritium",
    "lithium",
    "li loop",
    "neutronic",
    "scintillation",
    "optical transition radiation",
    "beam diagnostic",
    "beam imaging",
    "neutron source",
    "heat exchanger",
    "otr",
)
_AI_CUES = (
    "in the paper",
    "transformer",
    "neural",
    "diffusion",
    "federated",
    "language model",
    "zero-shot",
    "fine-tun",
    "tokenizer",
    "gradient",
    "embedding",
    "pretrain",
    "attention",
    "byzantine",
    "fuzzy cognitive",
    "benchmark",
    "dataset",
    "ablation",
    "theorem",
    "augmentation",
)
_PROCEDURAL_CUES = (
    "calculate",
    "compute",
    "decay constant",
    "flow rate",
    "half life",
    "half-life",
    "ratio",
    "volume",
    "volumen",
    "mathematically",
    "formula",
    "how many",
    "how much",
    "derive",
)


def _prompt(filename: str, fallback: str) -> str:
    cached = _PROMPT_CACHE.get(filename)
    if cached is not None:
        return cached
    path = _PROMPTS / filename
    text = fallback
    if path.is_file():
        loaded = path.read_text(encoding="utf-8").strip()
        if loaded:
            text = loaded
    _PROMPT_CACHE[filename] = text
    return text


def _cue_hit(text: str, cues: tuple[str, ...]) -> bool:
    for cue in cues:
        if " " not in cue and len(cue) <= 4:
            if re.search(rf"(?<![a-z0-9]){re.escape(cue)}(?![a-z0-9])", text):
                return True
        elif cue in text:
            return True
    return False


def detect_domain(question: str) -> str:
    """Return ``dones``, ``ai_papers``, or ``neutral``.

    Cues are a heuristic on the question string alone. The harness does
    not pass ``domain`` or ``document_id``.
    """
    text = question.lower()
    dones = _cue_hit(text, _DONES_CUES)
    ai = _cue_hit(text, _AI_CUES)
    if dones and not ai:
        return "dones"
    if ai and not dones:
        return "ai_papers"
    return "neutral"


def needs_procedural_hint(question: str) -> bool:
    return _cue_hit(question.lower(), _PROCEDURAL_CUES)


def select_system_prompt(question: str) -> str:
    domain = detect_domain(question)
    if domain == "dones":
        return _prompt("system_dones.txt", SYSTEM_DONES)
    if domain == "ai_papers":
        return _prompt("system_ai_papers.txt", SYSTEM_AI)
    return _prompt("system_neutral.txt", SYSTEM_NEUTRAL)


def build_user_content(context: str, question: str) -> str:
    hint = ""
    if needs_procedural_hint(question):
        hint = _prompt("procedural_hint.txt", PROCEDURAL_HINT).strip() + "\n\n"
    return f"{context}{hint}{question.strip()}"


def _require_rag_assets() -> None:
    missing = []
    for path in (EMBED_MODEL_PATH, DOCLING_MODELS_PATH):
        if not path.is_dir() or not any(path.iterdir()):
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "A1 RAG assets missing: "
            + ", ".join(missing)
            + ". On a machine with internet, run submission/build.sh "
            "--with-models to bundle Docling's layout model and "
            "all-MiniLM-L6-v2. Do not download OLMo 32B into the zip. "
            "The code-only zip is for CI smoke, not for CodaBench upload."
        )


def _chunk_markdown(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Pack paragraphs into ~chunk_size-character chunks.

    Cuts land on paragraph boundaries. The tail of each chunk is carried
    into the next one. Same packing rule as the starter's ``02_rag``.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def _docling_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        artifacts_path=str(DOCLING_MODELS_PATH),
        do_ocr=False,  # born-digital text, not scans (starter setting)
        do_table_structure=False,  # avoids cv2/libGL, not in the image
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _build_index(corpus_dir: str):
    """Parse, chunk, and embed every PDF in ``corpus_dir``. Once per run.

    TODO(Data): own local corpus prep from ``sample/sites.txt`` under
    ``data/`` for iteration and, later, A2 training text. Official eval
    still receives a blind ``corpus_dir`` of PDFs. Build this index inside
    the run, as the starter does. Do not ship a prebuilt index of the
    eval corpus, and do not crawl the sites in this skeleton.

    TODO(ML): numpy cosine matches the timed ``02_rag`` path. ``faiss-cpu``
    is available in the image if a later profile shows the matmul is the
    problem; do not swap it until that profile exists.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    converter = _docling_converter()
    embedder = SentenceTransformer(str(EMBED_MODEL_PATH), device="cpu")

    all_chunks: list[str] = []
    pdfs = sorted(Path(corpus_dir).glob("*.pdf"))
    for pdf_path in pdfs:
        try:
            result = converter.convert(str(pdf_path))
            text = result.document.export_to_markdown()
        except Exception as exc:
            # One bad PDF should not take down the run.
            print(f"[rag] skipping {pdf_path.name}: {exc}")
            continue
        all_chunks.extend(_chunk_markdown(text))

    print(f"[rag] parsed {len(pdfs)} PDFs into {len(all_chunks)} chunks")
    if not all_chunks:
        dim = embedder.get_sentence_embedding_dimension()
        empty = np.zeros((0, dim), dtype=np.float32)
        return all_chunks, empty, embedder

    embeddings = embedder.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )
    return all_chunks, np.asarray(embeddings, dtype=np.float32), embedder


def _maybe_attach_adapter(model):
    """Attach a PEFT adapter when A2 weights are present. No-op otherwise.

    TODO(ML): A2 is stretch, not this skeleton. Train the adapter offline
    on Data's self-built corpus. Load it here atop the frozen base. Keep
    the adapter small enough that the zip stays ≤ 1 GB and load time still
    fits the one-hour budget. Do not full-fine-tune the 32B weights.
    """
    config = ADAPTER_DIR / "adapter_config.json"
    if not config.is_file():
        return model
    from peft import PeftModel

    print(f"[a1] loading PEFT adapter from {ADAPTER_DIR}")
    return PeftModel.from_pretrained(model, str(ADAPTER_DIR))


def _load_llm(model_id: str):
    """Load the harness-provided base model from the local cache. Once."""
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
        print(f"[a1] sdpa load failed ({exc}); falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation="eager", **common
        )
    model.eval()
    model = _maybe_attach_adapter(model)
    return tokenizer, model


def _load(model_id: str, corpus_dir: str):
    """Cache the LLM across questions, and the index across one corpus."""
    _require_rag_assets()
    if _STATE["llm_id"] != model_id or _STATE["model"] is None:
        tokenizer, model = _load_llm(model_id)
        _STATE["llm_id"] = model_id
        _STATE["tokenizer"] = tokenizer
        _STATE["model"] = model

    index_key = str(Path(corpus_dir).resolve())
    if _STATE["index_key"] != index_key or _STATE["chunks"] is None:
        chunks, embeddings, embedder = _build_index(corpus_dir)
        _STATE["index_key"] = index_key
        _STATE["chunks"] = chunks
        _STATE["embeddings"] = embeddings
        _STATE["embedder"] = embedder

    return (
        _STATE["tokenizer"],
        _STATE["model"],
        _STATE["chunks"],
        _STATE["embeddings"],
        _STATE["embedder"],
    )


def _build_context(question, chunks, embeddings, embedder, tokenizer, max_tokens=MAX_CONTEXT_TOKENS) -> str:
    """Pack top-ranked chunks into ``max_tokens``.

    Blind over the whole corpus: nothing here knows which PDF a question
    came from. Same ranking and truncation rule as the starter's ``02_rag``.
    """
    import numpy as np

    if embeddings is None or len(chunks) == 0 or getattr(embeddings, "shape", (0,))[0] == 0:
        return ""

    query_emb = embedder.encode(
        [question], normalize_embeddings=True, show_progress_bar=False
    )[0]
    order = np.argsort(-(embeddings @ query_emb))

    selected: list[str] = []
    used = 0
    for i in order:
        chunk = chunks[int(i)]
        n = len(tokenizer.encode(chunk))
        if used + n > max_tokens:
            if not selected and n > 0:
                ratio = max_tokens / n
                selected.append(chunk[: int(len(chunk) * ratio)])
            break
        selected.append(chunk)
        used += n

    if not selected:
        return ""
    body = "\n\n".join(f"[Fragment {i}]\n{c}" for i, c in enumerate(selected, 1))
    return (
        "Context extracted from a blind search over the corpus:\n\n"
        f"{body}\n\n---\n\n"
    )


def _generate(tokenizer, model, system: str, user: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
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
    """Answer one question. Model and index are cached on later calls."""
    if not isinstance(question, str):
        raise TypeError("question must be str")
    if not isinstance(corpus_dir, str):
        raise TypeError("corpus_dir must be str")
    if not isinstance(model_id, str) or not model_id:
        raise TypeError("model_id must be a non-empty str")

    tokenizer, model, chunks, embeddings, embedder = _load(model_id, corpus_dir)
    context = _build_context(question, chunks, embeddings, embedder, tokenizer)
    system = select_system_prompt(question)
    user = build_user_content(context, question)
    text = _generate(tokenizer, model, system, user)
    if not isinstance(text, str):
        raise TypeError("generation did not return str")
    return text
