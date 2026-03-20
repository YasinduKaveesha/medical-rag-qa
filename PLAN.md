# Medical RAG Q&A — Module 0: Project Setup

## What I understood from the project plan

A 12-module Medical RAG Q&A system that:
- Ingests clinical PDF documents (WHO/NIH guidelines) via PyMuPDF
- Embeds chunks with HuggingFace `all-MiniLM-L6-v2` (384-dim, CPU-only)
- Stores vectors + metadata in Qdrant (Docker)
- Retrieves top-20 candidates → cross-encoder reranker → top-5 chunks
- Generates cited answers via Groq (llama-3.3-70b-versatile, free tier); refuses if max similarity < 0.35
- Exposes a FastAPI `/ask` endpoint + `/health` endpoint
- Evaluates with RAGAS (faithfulness + answer relevancy)
- Compares LangChain vs LlamaIndex pipelines
- Deploys via Docker Compose + GitHub Actions CI + Gradio on HF Spaces

Engineering rules: conventional commits, no hardcoded secrets, type hints on all
functions, docstrings on all public functions, `logging` not `print`, tests before
advancing to the next module.

---

## Module 0 — What I will build

Module 0 is pure scaffolding. No ML logic yet — just a stable base that every
subsequent module imports from.

### Files created

| File | Why |
|------|-----|
| `pyproject.toml` | All runtime + dev deps; ruff config; pytest config |
| `.env.example` | 10 env var templates — documents required secrets |
| `src/__init__.py` | Makes `src` a package |
| `src/config.py` | Typed `Settings` dataclass + `get_settings()` singleton + `setup_logging()` |
| `src/ingestion/__init__.py` | Package marker |
| `src/embeddings/__init__.py` | Package marker |
| `src/retrieval/__init__.py` | Package marker |
| `src/generation/__init__.py` | Package marker |
| `src/evaluation/__init__.py` | Package marker |
| `src/frameworks/__init__.py` | Package marker |
| `app/__init__.py` | Package marker |
| `tests/__init__.py` | Package marker |
| `tests/test_config.py` | Smoke-tests Settings loading and singleton behaviour |
| `data/raw/.gitkeep` | Preserves empty folder in git |
| `reports/figures/.gitkeep` | Preserves empty folder in git |

`.gitignore` addition: `data/raw/*.pdf`

---

## Architecture decisions

### `pyproject.toml`
- **Runtime deps:** pymupdf, sentence-transformers, qdrant-client, openai,
  fastapi, uvicorn, ragas, langchain, langchain-openai, langchain-community,
  llama-index, gradio, mlflow, python-dotenv, tqdm, numpy, pandas, matplotlib
- **Dev deps:** ruff, pytest, pytest-asyncio, httpx
- **Ruff:** `line-length = 100`, `target-version = "py311"`

### `src/config.py`

```python
@dataclass
class Settings:
    groq_api_key: str
    llm_base_url: str = "https://api.groq.com/openai/v1"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "medical_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    similarity_threshold: float = 0.35
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

def setup_logging() -> None:
    """Configure root logger at INFO level with standard format."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )

def get_settings() -> Settings:
    """Return the singleton Settings instance, loading .env on first call."""
    ...  # singleton; calls setup_logging() on first load
```

---

## Questions / concerns

None — the project plan is explicit for Module 0.

---

## Verification

1. `pip install -e ".[dev]"` — no errors
2. Copy `.env.example` → `.env`, fill `GROQ_API_KEY`
3. `python -c "from src.config import get_settings; print(get_settings())"` — prints Settings
4. `ruff check src/ app/ tests/` — zero errors
5. `pytest tests/test_config.py -v` — all pass

---

---

## Module 7 — Generation

### Files to create

| File | Why |
|------|-----|
| `src/generation/prompt_builder.py` | Assembles system prompt + context chunks + user query into a single string |
| `src/generation/llm_client.py` | `LLMClient` wrapping the `openai` package pointed at Groq |
| `src/generation/refusal.py` | Decides whether to refuse based on retrieval similarity scores |
| `src/generation/citations.py` | Maps answer sentences to the source chunks that support them |
| `tests/test_generation.py` | Tests for all four modules — no real LLM calls |

---

### Context from previous modules

`RetrievalPipeline.retrieve()` returns:

```python
[
    {
        "chunk_text": str,
        "metadata": {
            "source_document": str,
            "document_type":   str,
            "section_title":   str,
            "page_number":     int,
            "chunk_id":        str,
            "chunk_index":     int,
            "chunking_strategy": str,
        },
        "score":          float,   # cosine similarity
        "reranker_score": float,   # cross-encoder logit
    },
    ...
]
```

`Settings` provides: `groq_api_key`, `llm_base_url="https://api.groq.com/openai/v1"`,
`llm_model="llama-3.3-70b-versatile"`, `similarity_threshold=0.35`.

---

### `prompt_builder.py`

**Signature:**

```python
def build_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble a system prompt, numbered context blocks, and the user query."""
```

**Output format** — a single string:

```
SYSTEM:
You are a medical information assistant. Answer the user's question using
ONLY the information provided in the context below. For each claim in your
answer, cite the source document and page number in the format
[source_document, p.PAGE_NUMBER]. If the answer cannot be determined from
the provided context, respond with exactly:
"I cannot answer from the provided documents."
Do not use any prior knowledge outside the provided context.

CONTEXT:
[1] Source: WHO-MHP-HPS-EML-2023.02-eng.pdf | Page: 3 | Section: 2.3 Medicines for palliative care
Amitriptyline is a tricyclic antidepressant...

[2] Source: IDF_Rec_2025.pdf | Page: 47 | Section: 4.1 Pharmacotherapy
The recommended starting dose...

QUESTION:
What is the recommended dose of amitriptyline?
```

**Design decisions:**
- Chunks are numbered `[1]`, `[2]`, ... so the LLM can reference them by index
  and `citations.py` can map the index back to the source chunk.
- Each context block header shows `source_document`, `page_number`, and
  `section_title` — the three fields the LLM needs to cite correctly.
- Returns `""` immediately when `chunks` is empty (refusal is handled upstream
  by `should_refuse()`; prompt builder does not know about thresholds).

---

### `llm_client.py`

**Class design:**

```python
from openai import OpenAI

class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        _client: OpenAI | None = None,
    ) -> None: ...

    def generate(self, prompt: str) -> str: ...

    @property
    def model(self) -> str: ...
```

**Groq integration** — `openai` is already in `pyproject.toml`. Groq exposes
an OpenAI-compatible REST API, so only `base_url` and the key change:

```python
s = get_settings()
self._client = OpenAI(
    api_key=s.groq_api_key,
    base_url=s.llm_base_url,   # "https://api.groq.com/openai/v1"
)
self._model = model or s.llm_model
```

**`generate(prompt)` implementation:**

```python
def generate(self, prompt: str) -> str:
    response = self._client.chat.completions.create(
        model=self._model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

`temperature=0` gives deterministic outputs — required for reproducible RAGAS
evaluation in Module 9.

**`_client` injection for testing** — when `_client` is not `None`, it is used
directly (tests pass a `MagicMock` with `chat.completions.create` configured
to return a pre-set response). When `None`, the real `OpenAI(...)` is built.

**`get_llm_client()` singleton** — same pattern as all previous singletons.

---

### `refusal.py`

**Signature:**

```python
def should_refuse(chunks: list[dict]) -> bool:
    """Return True if retrieval quality is too low to answer safely."""
```

**Logic:**

```python
def should_refuse(chunks: list[dict]) -> bool:
    if not chunks:
        return True
    max_score = max(c["score"] for c in chunks)
    threshold = get_settings().similarity_threshold   # default 0.35
    return max_score < threshold
```

**Why `score` not `reranker_score`?** Cosine similarity is bounded `[0, 1]`
and calibrated directly against the `similarity_threshold` value in Settings.
Cross-encoder logits are unbounded and not comparable to `0.35`.

**No class needed** — a module-level function is the right abstraction.
There is no state to manage; the threshold is read from Settings each call.

---

### `citations.py`

**Signature:**

```python
def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Map answer sentences to the source chunks that support them."""
```

**Algorithm:**

1. Split *answer* into sentences using the punctuation-boundary regex
   (`re.split(r"(?<=[.!?])\s+", answer)`).
2. For each sentence, scan for `[N]` patterns (e.g. `[1]`, `[2]`).
3. For each `[N]` found, look up `chunks[N-1]` (1-based index).
4. Emit a citation dict per reference found.

**Output format:**

```python
[
    {
        "claim":           str,   # the sentence containing the citation
        "source_chunk":    str,   # chunks[N-1]["chunk_text"]
        "page_number":     int,   # chunks[N-1]["metadata"]["page_number"]
        "source_document": str,   # chunks[N-1]["metadata"]["source_document"]
    },
    ...
]
```

**Edge cases:**
- `[N]` index out of range (LLM hallucinated a chunk number) — skip silently,
  log a warning.
- Empty answer or empty chunks — return `[]`.
- Duplicate `[N]` across multiple sentences — each sentence gets its own
  citation dict (traces provenance per claim, not per chunk).

---

### `tests/test_generation.py` — what will be tested

#### `prompt_builder` tests

| Test | What it checks |
|------|---------------|
| `test_build_prompt_returns_string` | Return type is `str` |
| `test_build_prompt_contains_query` | Query text appears in the output |
| `test_build_prompt_contains_chunk_text` | Each chunk's text appears in the output |
| `test_build_prompt_contains_source_document` | `source_document` metadata appears |
| `test_build_prompt_contains_page_number` | `page_number` metadata appears |
| `test_build_prompt_numbered_chunks` | `[1]`, `[2]`, ... markers are present |
| `test_build_prompt_system_instruction_present` | System instruction text is in the output |
| `test_build_prompt_empty_chunks` | Returns `""` when chunks is `[]` |

#### `llm_client` tests

| Test | What it checks |
|------|---------------|
| `test_generate_returns_string` | Return type is `str` |
| `test_generate_calls_chat_completions` | `client.chat.completions.create` is called |
| `test_generate_passes_prompt_as_user_message` | Prompt appears as `{"role": "user", "content": prompt}` |
| `test_generate_uses_configured_model` | Model name from settings is passed to the API |
| `test_generate_temperature_zero` | `temperature=0` is passed |
| `test_generate_returns_stripped_content` | Leading/trailing whitespace is stripped |
| `test_get_llm_client_singleton` | Two calls return the same instance |
| `test_get_llm_client_reset` | After `_llm_client = None`, next call returns a new instance |

#### `refusal` tests

| Test | What it checks |
|------|---------------|
| `test_should_refuse_empty_chunks` | Returns `True` for `[]` |
| `test_should_refuse_below_threshold` | Returns `True` when max score < 0.35 |
| `test_should_not_refuse_above_threshold` | Returns `False` when max score >= 0.35 |
| `test_should_refuse_exactly_at_threshold` | Returns `False` when max score == 0.35 (boundary) |
| `test_should_refuse_uses_max_not_mean` | One high-score chunk prevents refusal even if others are low |
| `test_should_refuse_custom_threshold` | Threshold is read from `Settings.similarity_threshold` |

#### `citations` tests

| Test | What it checks |
|------|---------------|
| `test_extract_citations_returns_list` | Return type is `list` |
| `test_extract_citations_empty_answer` | Returns `[]` for empty answer |
| `test_extract_citations_no_markers` | Returns `[]` when answer has no `[N]` markers |
| `test_extract_citations_single_citation` | `[1]` in answer -> one citation with correct chunk text and page |
| `test_extract_citations_multiple_markers` | `[1]` and `[2]` -> two citations |
| `test_extract_citations_claim_text` | `"claim"` equals the sentence containing the marker |
| `test_extract_citations_out_of_range_skipped` | `[99]` with 2 chunks -> no citation, no crash |
| `test_extract_citations_source_document_present` | `"source_document"` key is populated from metadata |

---

### Concerns / decisions

1. **`openai` package is already installed** — in `pyproject.toml` from Module 0.
   No new dependency needed.

2. **`temperature=0`** — Groq's `llama-3.3-70b-versatile` supports it.
   Deterministic outputs are required for reproducible RAGAS scores in Module 9.

3. **Citation format `[N]`** — numbered indices are more reliable than asking
   the LLM to embed exact source titles inline. If the LLM hallucinates an
   index, the extractor skips it gracefully rather than crashing.

4. **`should_refuse` is a module-level function, not a class** — no state, no
   model loading, no singleton needed.

5. **No streaming** — `generate()` returns the full response string. Streaming
   can be added in Module 8 (FastAPI) without changing this module.

---

### Verification

```bash
pytest tests/test_generation.py -v
ruff check src/generation/ tests/test_generation.py
```
