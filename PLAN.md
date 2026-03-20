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

---

---

## Module 8 — FastAPI Endpoint

### Files to create

| File | Why |
|------|-----|
| `app/schemas.py` | Pydantic models for all request/response shapes |
| `app/main.py` | FastAPI app with `/ask` and `/health` endpoints |
| `tests/test_api.py` | Full endpoint tests using `TestClient` and dependency overrides — no real services |

---

### Context from previous modules

`RetrievalPipeline.retrieve(query, top_k, filters)` returns:
```python
[{"chunk_text": str, "metadata": dict, "score": float, "reranker_score": float}, ...]
```

`should_refuse(chunks) -> bool` — True when chunks is empty or max `score` < `similarity_threshold`.

`build_prompt(query, chunks) -> str` — returns `""` for empty chunks.

`LLMClient.generate(prompt) -> str` — returns stripped LLM response.

`extract_citations(answer, chunks) -> list[dict]` — returns `[{"claim", "source_chunk", "page_number", "source_document"}, ...]`.

`QdrantStore.get_collection_info()` — returns a Qdrant `CollectionInfo` object (`.vectors_count`, `.points_count`, `.status`).

---

### `app/schemas.py`

Four Pydantic models:

```python
class AskRequest(BaseModel):
    question: str
    filters: dict | None = None

class CitationSource(BaseModel):
    claim: str
    source_chunk: str
    page_number: int | None
    source_document: str

class AskResponse(BaseModel):
    answer: str
    sources: list[CitationSource]
    confidence: float       # max cosine score from retrieval; 0.0 on refusal
    model_version: str      # Settings.llm_model

class HealthResponse(BaseModel):
    status: str             # "ok" or "degraded"
    model_version: str
    collection_info: dict   # vectors_count, points_count, status from Qdrant
```

`CitationSource` mirrors the dict shape returned by `extract_citations`, allowing
`AskResponse.sources` to be typed rather than `list[dict]`.

---

### `app/main.py`

#### Dependency functions (for testability)

FastAPI's `Depends()` system lets tests override dependencies without monkeypatching
module globals. Three thin wrapper functions are defined at module level:

```python
def _get_pipeline() -> RetrievalPipeline:
    from src.retrieval.pipeline import get_pipeline
    return get_pipeline()

def _get_llm_client() -> LLMClient:
    from src.generation.llm_client import get_llm_client
    return get_llm_client()

def _get_store() -> QdrantStore:
    from src.retrieval.vector_store import get_store
    return get_store()
```

Tests swap them with `app.dependency_overrides[_get_pipeline] = lambda: mock`.

#### `POST /ask` — orchestration logic

```
1. Validate AskRequest (Pydantic handles type errors → 422 automatically)
2. Strip and check question — raise HTTPException(400) if empty after strip
3. pipeline.retrieve(question, top_k=settings.top_k_rerank, filters=filters)
4. should_refuse(chunks) → if True:
       return AskResponse(
           answer="I cannot answer from the provided documents.",
           sources=[],
           confidence=max(c["score"] for c in chunks) if chunks else 0.0,
           model_version=settings.llm_model,
       )
5. prompt = build_prompt(question, chunks)
6. answer = llm_client.generate(prompt)
7. citations = extract_citations(answer, chunks)
8. confidence = max(c["score"] for c in chunks)
9. return AskResponse(answer=answer, sources=citations,
                      confidence=confidence, model_version=settings.llm_model)
```

Wrap steps 3–9 in `try/except`:
- `ConnectionError` → `HTTPException(503, detail="Vector store unavailable")`
- Any other `Exception` → `HTTPException(500, detail="Internal server error")`

Logging: INFO at entry (question length, filters), INFO at exit (answer length, num sources, confidence).

#### `GET /health`

```
1. settings = get_settings()
2. try:
       info = store.get_collection_info()
       collection_info = {
           "name": settings.collection_name,
           "vectors_count": info.vectors_count,
           "points_count": info.points_count,
           "status": str(info.status),
       }
       status = "ok"
   except ConnectionError:
       collection_info = {"error": "Qdrant unavailable"}
       status = "degraded"
3. return HealthResponse(status=status, model_version=settings.llm_model,
                         collection_info=collection_info)
```

Always returns 200 — health endpoints must not raise 5xx or load balancers
will drop the instance. Callers inspect the `status` field to detect degradation.

---

### Error handling strategy

| Scenario | HTTP code | How triggered |
|---|---|---|
| Empty/whitespace `question` | 400 | Explicit `HTTPException(400)` before pipeline call |
| Pydantic type mismatch | 422 | FastAPI automatic validation |
| Qdrant unreachable | 503 | `except ConnectionError` wrapping `pipeline.retrieve` |
| Any other unexpected error | 500 | `except Exception` catch-all in `/ask` |
| Refusal (low similarity) | 200 | Valid answer, just the refusal phrase — not an error |

A custom `exception_handler` is **not** needed — `HTTPException` already produces
structured JSON `{"detail": "..."}` responses, which is sufficient for this stage.

---

### Testing strategy — `tests/test_api.py`

**No real services.** `TestClient` (Starlette, included with FastAPI) runs the ASGI
app in-process. Dependencies are replaced with lightweight fakes:

```python
@pytest.fixture
def mock_pipeline():
    m = MagicMock()
    m.retrieve.return_value = _make_chunks(2)   # reuse helper from test_generation.py
    return m

@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.generate.return_value = "The dose is 25 mg [1]."
    m.model = "llama-3.3-70b-versatile"
    return m

@pytest.fixture
def client(mock_pipeline, mock_llm):
    from app.main import app, _get_pipeline, _get_llm_client
    app.dependency_overrides[_get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

The `mock_pipeline` fixture returns pre-built chunks with `score=0.9` (above threshold),
so the default path exercises the full happy path without any model loading.

#### Tests planned

**`/ask` happy path**

| Test | What it checks |
|------|----------------|
| `test_ask_returns_200` | Status code 200 |
| `test_ask_response_has_answer` | `response["answer"]` is a non-empty string |
| `test_ask_response_has_sources` | `response["sources"]` is a list |
| `test_ask_response_has_confidence` | `response["confidence"]` is a float |
| `test_ask_response_has_model_version` | `response["model_version"]` is present |
| `test_ask_passes_question_to_pipeline` | `pipeline.retrieve` called with the question |
| `test_ask_passes_filters_to_pipeline` | `filters` dict forwarded to `pipeline.retrieve` |
| `test_ask_sources_populated_from_citations` | `[1]` in answer → `sources` list has one entry |
| `test_ask_confidence_is_max_score` | `confidence == 0.9` (max of chunk scores) |

**`/ask` refusal path**

| Test | What it checks |
|------|----------------|
| `test_ask_refusal_returns_200` | Refusal is a 200, not an error |
| `test_ask_refusal_answer_text` | `answer` equals the refusal phrase |
| `test_ask_refusal_empty_sources` | `sources == []` |
| `test_ask_refusal_confidence_zero` | `confidence == 0.0` when no chunks |

**`/ask` error paths**

| Test | What it checks |
|------|----------------|
| `test_ask_empty_question_returns_400` | `""` question → 400 |
| `test_ask_whitespace_question_returns_400` | `"   "` question → 400 |
| `test_ask_pipeline_connection_error_returns_503` | `pipeline.retrieve` raises `ConnectionError` → 503 |
| `test_ask_pipeline_unexpected_error_returns_500` | `pipeline.retrieve` raises `RuntimeError` → 500 |

**`/health`**

| Test | What it checks |
|------|----------------|
| `test_health_returns_200` | Status code 200 |
| `test_health_status_ok` | `response["status"] == "ok"` when store responds |
| `test_health_model_version` | `response["model_version"]` matches settings |
| `test_health_collection_info_present` | `response["collection_info"]` is a dict |
| `test_health_degraded_when_store_unavailable` | `status == "degraded"` when store raises `ConnectionError` |

Total: **~22 tests**.

---

### Architecture decisions

1. **`Depends()` over monkeypatching** — FastAPI's dependency injection is the
   standard pattern for overriding services in tests. It's explicit and doesn't
   risk leaking state between tests via module globals.

2. **503 for Qdrant unavailability, not 500** — `ConnectionError` from the store
   is a known failure mode (Qdrant container not running). 503 signals "service
   unavailable" to clients and load balancers more accurately than 500.

3. **Refusal is 200, not 4xx** — the system understood the question and searched
   the corpus; it just lacked sufficient evidence. This is a valid answer, not a
   client or server error.

4. **`CitationSource` Pydantic model** — typing `sources` as `list[CitationSource]`
   rather than `list[dict]` gives automatic validation, OpenAPI schema generation,
   and IDE support for consumers of the API.

5. **Lazy imports inside dependency functions** — `_get_pipeline()` imports inside
   the function body to avoid loading ML models at import time (same pattern as
   `RetrievalPipeline.__init__`). This keeps the test suite fast.

6. **No streaming** — `/ask` returns the full response. Streaming (SSE/WebSocket)
   is a potential future enhancement but out of scope for Module 8.

---

### Verification

```bash
pytest tests/test_api.py -v
ruff check app/ tests/test_api.py
```
