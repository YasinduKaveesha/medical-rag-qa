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

---

---

## Module 9 — RAGAS Evaluation

### Files to create

| File | Why |
|------|-----|
| `src/evaluation/test_queries.json` | 20+ medical Q&A pairs with ground truth answers |
| `src/evaluation/ragas_eval.py` | Run RAGAS (faithfulness + answer_relevancy) across all test queries |
| `src/evaluation/chunking_comparison.py` | Run same queries across all 3 chunking strategies, compare RAGAS scores |
| `tests/test_evaluation.py` | Lightweight tests — mocked pipeline/LLM/RAGAS; no real API calls |

---

### RAGAS 0.4.x API notes

RAGAS 0.4.3 is installed. Key API facts discovered by inspection:

**Metrics require an LLM at construction time:**
```python
from ragas.llms import llm_factory
from openai import OpenAI

ragas_llm = llm_factory(
    "llama-3.3-70b-versatile",
    provider="openai",
    client=OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1"),
)

from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy

metrics = [Faithfulness(llm=ragas_llm), AnswerRelevancy(llm=ragas_llm)]
```

**Dataset schema (`SingleTurnSample` fields used):**
- `user_input` — the question
- `response` — LLM-generated answer
- `retrieved_contexts` — list of chunk texts passed to the LLM
- `reference` — ground truth answer (used by some metrics)

**`evaluate()` call:**
```python
from ragas import evaluate
from ragas.run_config import RunConfig

result = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    run_config=RunConfig(max_retries=10, max_wait=60, timeout=180),
    batch_size=5,          # limits concurrent requests
    raise_exceptions=False,  # returns NaN on failures, never crashes
)
df = result.to_pandas()
```

---

### `src/evaluation/test_queries.json`

A JSON array of 20+ entries. Each entry:

```json
{
    "id": "q001",
    "category": "dosing",
    "question": "What is the recommended starting dose of amitriptyline for pain management?",
    "ground_truth": "The recommended starting dose of amitriptyline for pain management is 10–25 mg at night.",
    "expected_source_keywords": ["amitriptyline", "dose", "palliative"]
}
```

**Fields:**
- `id` — unique identifier for tracking results per query
- `category` — topic tag (`dosing`, `contraindications`, `indications`, `mechanism`, `monitoring`, `interactions`) for per-category breakdowns
- `question` — natural language query sent to the RAG pipeline
- `ground_truth` — reference answer for RAGAS `reference` field
- `expected_source_keywords` — used in lightweight tests to verify the JSON is well-formed (not used by RAGAS)

**20 question categories covered:**
- 5× dosing (amitriptyline, metformin, insulin, morphine, aspirin)
- 4× contraindications (β-blockers, NSAIDs, warfarin, lithium)
- 4× indications (WHO essential medicines categories)
- 3× mechanism of action
- 2× drug monitoring (INR, renal function)
- 2× drug interactions

All questions are grounded in WHO Essential Medicines List and standard clinical
guidelines — the same corpus ingested in Module 1.

---

### `src/evaluation/ragas_eval.py`

**Public API:**

```python
def load_test_queries(path: str | Path) -> list[dict]:
    """Load and return test query entries from a JSON file."""

def run_rag_pipeline(
    queries: list[dict],
    pipeline: RetrievalPipeline,
    llm_client: LLMClient,
) -> list[dict]:
    """Run each query through retrieve → generate; return result records."""

def build_eval_dataset(results: list[dict]) -> EvaluationDataset:
    """Convert pipeline results into a RAGAS EvaluationDataset."""

def run_ragas(
    dataset: EvaluationDataset,
    ragas_llm: InstructorBaseRagasLLM,
) -> pd.DataFrame:
    """Run faithfulness + answer_relevancy; return scores as a DataFrame."""

def save_results(df: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Save scores to CSV and a bar chart PNG; return (csv_path, chart_path)."""

def main() -> None:
    """CLI entry point: load queries, run eval, save outputs."""
```

**`run_rag_pipeline` — result record shape:**
```python
{
    "id":                str,   # query id
    "category":          str,   # query category
    "question":          str,   # user question
    "ground_truth":      str,   # reference answer
    "answer":            str,   # LLM response (or refusal phrase)
    "retrieved_contexts": list[str],  # chunk texts passed to prompt
    "refused":           bool,  # True if should_refuse() fired
    "max_score":         float, # max cosine similarity
}
```

If `should_refuse()` is True the answer is set to
`"I cannot answer from the provided documents."` and the record is still
included (RAGAS will give it low faithfulness/relevancy scores, which is
the correct signal).

**`build_eval_dataset` — mapping to `SingleTurnSample`:**

| Result field | SingleTurnSample field |
|---|---|
| `question` | `user_input` |
| `answer` | `response` |
| `retrieved_contexts` | `retrieved_contexts` |
| `ground_truth` | `reference` |

**`save_results` — output files:**
- `reports/figures/ragas_results_{timestamp}.csv` — one row per query with
  `id`, `category`, `faithfulness`, `answer_relevancy`
- `reports/figures/ragas_scores_{timestamp}.png` — horizontal bar chart
  showing mean faithfulness and answer_relevancy with individual query dots

---

### `src/evaluation/chunking_comparison.py`

**Assumption:** three Qdrant collections must already be ingested before
running this script, one per strategy:

| Strategy | Default collection name |
|---|---|
| `fixed_size` | `medical_docs_fixed` |
| `sentence` | `medical_docs_sentence` |
| `semantic` | `medical_docs_semantic` |

Collection names are passed as a dict argument (or read from env vars
`COLLECTION_FIXED`, `COLLECTION_SENTENCE`, `COLLECTION_SEMANTIC`) so users
can override them without changing code.

**Public API:**

```python
def evaluate_strategy(
    strategy_name: str,
    collection_name: str,
    queries: list[dict],
    llm_client: LLMClient,
    ragas_llm: InstructorBaseRagasLLM,
) -> pd.DataFrame:
    """Run the full eval loop for one chunking strategy; return scores DF."""

def run_comparison(
    collections: dict[str, str],
    queries: list[dict],
    llm_client: LLMClient,
    ragas_llm: InstructorBaseRagasLLM,
) -> pd.DataFrame:
    """Run evaluate_strategy for all strategies; return combined DF."""

def save_comparison(df: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Save comparison CSV + grouped bar chart."""

def main() -> None:
    """CLI entry point."""
```

**`run_comparison` — combined DataFrame shape:**
```
strategy | faithfulness_mean | answer_relevancy_mean | faithfulness_std | answer_relevancy_std
```

**`save_comparison` — output files:**
- `reports/figures/chunking_comparison_{timestamp}.csv`
- `reports/figures/chunking_comparison_{timestamp}.png` — grouped bar chart:
  - X axis: chunking strategy (fixed_size, sentence, semantic)
  - Two bars per group: faithfulness (blue) and answer_relevancy (orange)
  - Error bars: ± 1 std dev

Each strategy is evaluated sequentially (not in parallel) to stay within
Groq's free-tier rate limit.

---

### Groq rate limiting strategy

Groq free tier allows ~30 requests/minute.  Each RAGAS metric call makes
multiple internal LLM requests per sample.

**Mitigations baked into both scripts:**

1. `RunConfig(max_retries=10, max_wait=60, timeout=180)` — RAGAS retries with
   exponential backoff on rate-limit errors (HTTP 429).
2. `evaluate(..., batch_size=5, raise_exceptions=False)` — limits concurrent
   requests; NaN on individual failures rather than crashing the whole run.
3. `sleep_between_queries: float = 1.0` parameter in `run_rag_pipeline` —
   adds a 1-second pause between pipeline calls (before RAGAS).  Configurable
   so users can increase it on the free tier.
4. Sequential strategy evaluation in `chunking_comparison.py` with a
   `sleep_between_strategies: float = 10.0` pause between strategies.

---

### Testing strategy — `tests/test_evaluation.py`

**Key principle:** RAGAS `evaluate()` makes real LLM API calls — it cannot
be fully unit-tested without a running Groq key. All tests mock it.

**Fixtures:**
```python
@pytest.fixture
def sample_queries():
    return [{"id": "q1", "category": "dosing",
              "question": "What is the dose?",
              "ground_truth": "25 mg.", "expected_source_keywords": ["dose"]}]

@pytest.fixture
def sample_results(sample_queries):
    return [{**q, "answer": "The dose is 25 mg.", "refused": False,
             "max_score": 0.85, "retrieved_contexts": ["Context text."]}
            for q in sample_queries]
```

#### `ragas_eval` tests

| Test | What it checks |
|------|----------------|
| `test_load_test_queries_returns_list` | Return type is `list` |
| `test_load_test_queries_has_required_fields` | Each entry has `id`, `question`, `ground_truth`, `category` |
| `test_load_test_queries_count` | At least 20 entries in the shipped JSON |
| `test_run_rag_pipeline_calls_retrieve` | `pipeline.retrieve` called once per query |
| `test_run_rag_pipeline_calls_generate` | `llm_client.generate` called for non-refused queries |
| `test_run_rag_pipeline_result_fields` | Each result has `answer`, `retrieved_contexts`, `refused` |
| `test_run_rag_pipeline_refusal_sets_flag` | When `should_refuse` fires, `refused=True` and no LLM call |
| `test_build_eval_dataset_type` | Returns `EvaluationDataset` instance |
| `test_build_eval_dataset_sample_count` | One sample per result |
| `test_build_eval_dataset_user_input` | `user_input` matches question |
| `test_build_eval_dataset_response` | `response` matches answer |
| `test_build_eval_dataset_contexts` | `retrieved_contexts` is a list of strings |
| `test_save_results_creates_csv` | CSV file exists after call |
| `test_save_results_creates_chart` | PNG file exists after call |
| `test_save_results_csv_columns` | CSV has `faithfulness` and `answer_relevancy` columns |

#### `chunking_comparison` tests

| Test | What it checks |
|------|----------------|
| `test_run_comparison_calls_all_strategies` | `evaluate_strategy` called 3 times |
| `test_run_comparison_result_has_strategy_column` | `strategy` column present in output DF |
| `test_save_comparison_creates_csv` | Comparison CSV file exists |
| `test_save_comparison_creates_chart` | Comparison PNG file exists |
| `test_save_comparison_all_strategies_in_csv` | All 3 strategy names in CSV |

Total: **~20 tests**.

---

### Architecture decisions

1. **`llm_factory` over `LangchainLLMWrapper`** — `LangchainLLMWrapper` is
   deprecated in RAGAS 0.4.x. Use `llm_factory(model, provider, client=OpenAI(...))`
   which is the officially recommended path and returns an `InstructorBaseRagasLLM`.

2. **Same Groq key, different client** — RAGAS uses its own OpenAI client
   instance (separate from the `LLMClient` in Module 7). Both share the same
   `GROQ_API_KEY` from Settings but are independent objects.

3. **Scripts, not importable modules** — `ragas_eval.py` and
   `chunking_comparison.py` have `main()` entry points runnable as
   `python -m src.evaluation.ragas_eval`. All heavy logic is in named
   functions so tests can import and mock them directly.

4. **`raise_exceptions=False`** — ensures a single rate-limit failure on one
   query doesn't abort the entire eval run. NaN scores are preserved in the CSV
   so the analyst can see which queries failed.

5. **Timestamp in output filenames** — prevents accidental overwrite when re-running
   eval after tuning the pipeline.

6. **`test_queries.json` ships in-repo** — ground truth answers are static
   reference data, not generated. Version-controlled alongside the code so
   any change to the question set is visible in git diff.

---

### Verification

```bash
# Lightweight tests (no API calls)
pytest tests/test_evaluation.py -v
ruff check src/evaluation/ tests/test_evaluation.py

# Full eval run (requires Qdrant + GROQ_API_KEY):
# python -m src.evaluation.ragas_eval
# python -m src.evaluation.chunking_comparison
```

---

---

## Module 10 — Framework Comparison (LangChain vs LlamaIndex)

### Files to create

| File | Why |
|------|-----|
| `src/frameworks/langchain_pipeline.py` | Full RAG pipeline using LangChain LCEL |
| `src/frameworks/llamaindex_pipeline.py` | Full RAG pipeline using LlamaIndex with a Qdrant bridge |
| `tests/test_frameworks.py` | Mocked tests for both pipelines — no real Qdrant or Groq calls |

---

### Installed package versions (discovered by inspection)

| Package | Version | Notes |
|---|---|---|
| `langchain` | 1.2.13 | |
| `langchain_openai` | 1.1.11 | `ChatOpenAI` → Groq |
| `langchain_community` | 0.4.1 | `Qdrant`, `HuggingFaceEmbeddings` |
| `llama-index-core` | 0.14.18 | |
| `llama-index-llms-openai` | 0.7.2 | supports `api_base` → Groq |
| `llama-index-embeddings-openai` | 0.6.0 | |
| `llama-index-vector-stores-qdrant` | **NOT INSTALLED** | Bridge needed |

**Critical asymmetry:** LangChain can connect to Qdrant directly via
`langchain_community.vectorstores.Qdrant`.  LlamaIndex has no Qdrant
integration installed.  Both pipelines are made comparable by having
LlamaIndex use a thin `BaseRetriever` subclass that delegates to our
existing `QdrantStore` + `EmbeddingEncoder`.

---

### Shared output format

Both `LangChainPipeline.query()` and `LlamaIndexPipeline.query()` return
the same dict shape so comparison code can treat them identically:

```python
{
    "answer":      str,        # LLM-generated answer
    "sources":     list[dict], # retrieved chunks with metadata
    "latency_ms":  float,      # wall-clock time for full query in milliseconds
}
```

Each source dict:
```python
{
    "chunk_text":       str,
    "source_document":  str,
    "page_number":      int | None,
    "score":            float,
}
```

---

### `src/frameworks/langchain_pipeline.py`

#### Class design

```python
class LangChainPipeline:
    def __init__(
        self,
        _vectorstore: VectorStore | None = None,
        _llm: BaseChatModel | None = None,
    ) -> None: ...

    def query(self, question: str, top_k: int = 5) -> dict: ...
```

#### Construction (when no injected mocks)

```python
s = get_settings()

# 1. Embeddings (same model as our pipeline for fair comparison)
embeddings = HuggingFaceEmbeddings(model_name=s.embedding_model)

# 2. Qdrant vectorstore — connects to existing collection
vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    host=s.qdrant_host,
    port=s.qdrant_port,
    collection_name=s.collection_name,
)

# 3. LLM — ChatOpenAI pointed at Groq
llm = ChatOpenAI(
    base_url=s.llm_base_url,
    api_key=s.groq_api_key,
    model=s.llm_model,
    temperature=0,
)
```

#### LCEL chain (`query()`)

```python
retriever = self._vectorstore.as_retriever(search_kwargs={"k": top_k})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the context below..."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | self._llm
    | StrOutputParser()
)

t0 = time.perf_counter()
answer = chain.invoke(question)
latency_ms = (time.perf_counter() - t0) * 1000
```

`_format_docs` joins `Document.page_content` strings with `"\n\n"`.

Sources are obtained via a separate `retriever.invoke(question)` call (or
the chain is split at the retriever step to capture intermediate results).

---

### `src/frameworks/llamaindex_pipeline.py`

#### The Qdrant bridge — `_QdrantBridgeRetriever`

Since `llama-index-vector-stores-qdrant` is not installed, a thin
`BaseRetriever` subclass wraps our existing `QdrantStore` +
`EmbeddingEncoder`:

```python
class _QdrantBridgeRetriever(BaseRetriever):
    """LlamaIndex-compatible retriever backed by our QdrantStore."""

    def __init__(
        self,
        store: QdrantStore,
        encoder: EmbeddingEncoder,
        top_k: int = 5,
    ) -> None: ...

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_vector = self._encoder.encode(query_bundle.query_str)
        candidates = self._store.search(query_vector, top_k=self._top_k)
        nodes = []
        for c in candidates:
            chunk = c["chunk"]
            node = TextNode(
                text=chunk.get("text", ""),
                metadata={k: v for k, v in chunk.items() if k != "text"},
            )
            nodes.append(NodeWithScore(node=node, score=c["score"]))
        return nodes
```

This retriever produces `NodeWithScore` objects in the standard LlamaIndex
interface — everything downstream (synthesis, citation extraction) uses
pure LlamaIndex APIs.

#### Class design

```python
class LlamaIndexPipeline:
    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        store: QdrantStore | None = None,
        _query_engine: RetrieverQueryEngine | None = None,
    ) -> None: ...

    def query(self, question: str, top_k: int = 5) -> dict: ...
```

#### Construction (when no injected mock)

```python
s = get_settings()

# LLM — LlamaIndex OpenAI pointed at Groq
llm = LlamaOpenAI(
    model=s.llm_model,
    api_key=s.groq_api_key,
    api_base=s.llm_base_url,
    temperature=0,
)

# Bridge retriever (wraps our QdrantStore + EmbeddingEncoder)
retriever = _QdrantBridgeRetriever(
    store=store or get_store(),
    encoder=encoder or get_encoder(),
    top_k=5,
)

# Query engine — uses LlamaIndex's synthesis pipeline
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm,
)
```

#### `query()` implementation

```python
def query(self, question: str, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    response = self._query_engine.query(question)
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = str(response)
    sources = [
        {
            "chunk_text": ns.node.get_content(),
            "source_document": ns.node.metadata.get("source_document", ""),
            "page_number": ns.node.metadata.get("page_number"),
            "score": ns.score or 0.0,
        }
        for ns in (response.source_nodes or [])
    ]
    return {"answer": answer, "sources": sources, "latency_ms": latency_ms}
```

---

### Comparison metrics

Both pipelines expose the same `query() -> dict` interface, so comparison
is straightforward:

| Metric | How measured |
|---|---|
| **Latency** | `latency_ms` in each pipeline's output dict (wall-clock, includes retrieval + generation) |
| **Faithfulness** | RAGAS `Faithfulness` score via `src.evaluation.ragas_eval.run_ragas` |
| **Answer relevancy** | RAGAS `AnswerRelevancy` score via `src.evaluation.ragas_eval.run_ragas` |

A manual comparison run (not part of the test suite) calls both pipelines
on the test queries from `test_queries.json`, feeds results to `run_ragas`,
and saves a side-by-side CSV and chart using `save_comparison` from
`chunking_comparison.py` (reusing the same save logic with
`strategy="langchain"` / `strategy="llamaindex"`).

---

### Testing strategy — `tests/test_frameworks.py`

**No real services.** All Qdrant, LLM, and embedding calls are replaced
with `MagicMock` or lightweight fakes injected via constructor parameters.

#### LangChain pipeline fixtures

```python
@pytest.fixture
def mock_vectorstore():
    doc = Document(
        page_content="Amitriptyline 10–25 mg at night.",
        metadata={"source_document": "WHO.pdf", "page_number": 3},
    )
    m = MagicMock()
    m.as_retriever.return_value.invoke.return_value = [doc]
    return m

@pytest.fixture
def mock_chat_llm():
    m = MagicMock()
    # Make it behave like a Runnable returning an AIMessage-like object
    m.invoke.return_value = AIMessage(content="The dose is 25 mg.")
    return m

@pytest.fixture
def langchain_pipeline(mock_vectorstore, mock_chat_llm, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    import src.config as cfg; cfg._settings = None
    return LangChainPipeline(_vectorstore=mock_vectorstore, _llm=mock_chat_llm)
```

#### LlamaIndex pipeline fixtures

```python
@pytest.fixture
def mock_query_engine():
    from llama_index.core.base.response.schema import Response
    from llama_index.core.schema import NodeWithScore, TextNode
    node = NodeWithScore(
        node=TextNode(text="Chunk text.", metadata={"source_document": "WHO.pdf"}),
        score=0.85,
    )
    mock_response = Response(response="The dose is 25 mg.", source_nodes=[node])
    m = MagicMock()
    m.query.return_value = mock_response
    return m

@pytest.fixture
def llamaindex_pipeline(mock_query_engine, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    import src.config as cfg; cfg._settings = None
    return LlamaIndexPipeline(_query_engine=mock_query_engine)
```

#### Tests planned

**LangChain pipeline**

| Test | What it checks |
|------|----------------|
| `test_langchain_query_returns_dict` | Return value is a `dict` |
| `test_langchain_query_has_answer` | `"answer"` key is a non-empty string |
| `test_langchain_query_has_sources` | `"sources"` key is a list |
| `test_langchain_query_has_latency` | `"latency_ms"` key is a float ≥ 0 |
| `test_langchain_sources_have_required_fields` | Each source has `chunk_text`, `source_document`, `score` |
| `test_langchain_calls_retriever` | `vectorstore.as_retriever()` is called |
| `test_langchain_passes_question` | Question text appears in the invocation path |

**LlamaIndex pipeline**

| Test | What it checks |
|------|----------------|
| `test_llamaindex_query_returns_dict` | Return value is a `dict` |
| `test_llamaindex_query_has_answer` | `"answer"` key is a non-empty string |
| `test_llamaindex_query_has_sources` | `"sources"` key is a list |
| `test_llamaindex_query_has_latency` | `"latency_ms"` key is a float ≥ 0 |
| `test_llamaindex_sources_from_response_nodes` | Sources are extracted from `response.source_nodes` |
| `test_llamaindex_calls_query_engine` | `query_engine.query()` is called with the question |

**Shared / comparison**

| Test | What it checks |
|------|----------------|
| `test_output_formats_match` | Both pipelines return the same top-level keys |
| `test_latency_is_positive` | `latency_ms > 0` for both pipelines |

Total: **~16 tests**.

---

### Architecture decisions

1. **LlamaIndex `_QdrantBridgeRetriever`** — since
   `llama-index-vector-stores-qdrant` is not installed, subclassing
   `BaseRetriever` is the standard extension point.  It keeps LlamaIndex's
   synthesis pipeline (prompt assembly, citation tracking via `source_nodes`)
   intact while sourcing vectors from our existing `QdrantStore`.

2. **`_query_engine` injection for LlamaIndex** — injecting the entire
   `RetrieverQueryEngine` (rather than just the LLM) is the cleanest seam
   for testing: one mock replaces the entire LlamaIndex stack without needing
   to understand `ServiceContext` or `Settings` global state.

3. **`_vectorstore` + `_llm` injection for LangChain** — the LCEL chain
   is built inside `query()` from these two injected objects, so tests control
   both retrieval and generation independently.

4. **Same system prompt text** — both pipelines use the same instruction
   text ("Answer using ONLY the context below…") so prompt wording does not
   confound the comparison.

5. **`temperature=0` for both** — consistent with `LLMClient` in Module 7
   and required for reproducible RAGAS scores.

6. **Latency includes retrieval + generation** — the timer wraps the entire
   `query()` call so both pipelines are measured end-to-end consistently.

---

### Verification

```bash
pytest tests/test_frameworks.py -v
ruff check src/frameworks/ tests/test_frameworks.py
```

---

## Module 11 — Infrastructure: Docker, CI/CD, Gradio

### Files to create / modify

| File | Purpose |
|------|---------|
| `Dockerfile` | Containerise the FastAPI app on Python 3.11-slim |
| `docker-compose.yml` | Orchestrate Qdrant + FastAPI on a shared network |
| `.github/workflows/ci.yml` | GitHub Actions: lint + mocked test suite on every push to `main` |
| `app/gradio_demo.py` | Standalone Gradio UI for HuggingFace Spaces |

No new test file — Docker and CI correctness is verified by running them.
`app/gradio_demo.py` is exercised manually/on HF Spaces; writing pytest tests for a Gradio UI block brings negligible value relative to cost.

---

### Dockerfile design

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by some wheels (e.g. sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Key decisions:
- **`python:3.11-slim`** — matches project requirement; slim avoids unnecessary OS packages.
- **`pip install -e .`** — installs from `pyproject.toml`; editable install makes `src.*` importable without `PYTHONPATH` hacks.
- **Copy `pyproject.toml` first** — Docker layer-caches the dependency install so code-only changes don't trigger a full `pip install`.
- **`--host 0.0.0.0`** — required inside Docker (default `127.0.0.1` is unreachable from the host).
- **No `.env` baked in** — secrets are injected at runtime via `docker run -e` or `docker-compose` `environment:` block.
- Dev extras (`ruff`, `pytest`) are **not** installed — keeps the image lean.

---

### docker-compose.yml design

Two services on a user-defined bridge network `rag-net`:

| Service | Image | Ports | Role |
|---------|-------|-------|------|
| `qdrant` | `qdrant/qdrant:v1.9.2` | `6333:6333` | Vector store |
| `fastapi` | built from `./Dockerfile` | `8000:8000` | API server |

```yaml
version: "3.9"

networks:
  rag-net:

services:
  qdrant:
    image: qdrant/qdrant:v1.9.2
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - rag-net

  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - QDRANT_HOST=qdrant          # service name resolves on rag-net
      - QDRANT_PORT=6333
      - COLLECTION_NAME=${COLLECTION_NAME:-medical_docs}
      - LLM_MODEL=${LLM_MODEL:-llama-3.3-70b-versatile}
    depends_on:
      - qdrant
    networks:
      - rag-net

volumes:
  qdrant_data:
```

Key decisions:
- **Named volume `qdrant_data`** — Qdrant data survives `docker-compose down` (without `--volumes`).
- **`QDRANT_HOST=qdrant`** — Docker DNS resolves service names on a user-defined network; `localhost` would not work here.
- **`depends_on`** — ensures Qdrant starts before FastAPI (start order only; no health-check polling).
- **`${GROQ_API_KEY}` from host env** — keeps secrets out of the compose file; user runs `export GROQ_API_KEY=...` or uses a `.env` file that compose auto-loads.
- **Pinned Qdrant image tag** — reproducible; avoids surprise breaking changes from `:latest`.

---

### .github/workflows/ci.yml design

Trigger: `push` to `main` and `pull_request` targeting `main`.

Steps:
1. `actions/checkout@v4`
2. `actions/setup-python@v5` with Python `3.11`
3. `pip install -e ".[dev]"` — installs runtime + dev extras
4. `ruff check src/ app/ tests/` — lint; fail fast on any error
5. `pytest tests/ -v --tb=short` — full mocked test suite (no Qdrant, no Groq)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check src/ app/ tests/

      - name: Test
        run: pytest tests/ -v --tb=short
        env:
          GROQ_API_KEY: gsk-ci-placeholder   # satisfies Settings validation; no real calls made
```

Key decisions:
- **`GROQ_API_KEY: gsk-ci-placeholder`** — all tests monkeypatch this env var anyway via `reset_settings` fixture; the env var is only needed so `get_settings()` doesn't raise at import time in modules that call it at module scope.
- **No Qdrant service** — all tests use mocks/dependency overrides; no real vector store is required.
- **`--tb=short`** — concise tracebacks in CI output without the full local-debugging verbosity.
- **No caching** — kept simple for now; can add `actions/cache` for pip wheels in Module 12 if build times matter.

---

### app/gradio_demo.py design

A self-contained Gradio app that calls the FastAPI `/ask` endpoint over HTTP, suitable for HuggingFace Spaces deployment (Spaces injects `FASTAPI_URL` or uses localhost if co-located).

Public interface:
```python
def ask_question(question: str) -> tuple[str, str]:
    """Send *question* to the /ask endpoint; return (answer, formatted sources)."""
```

UI layout:
- **Textbox** — question input, placeholder text, submit on Enter
- **Textbox** — answer output (read-only)
- **Textbox** — sources output (read-only, multi-line, formatted as numbered list)
- **Button** — "Ask" to trigger, "Clear" to reset
- **Title + description** — "Medical RAG Q&A" with brief context note

Configuration:
- `FASTAPI_URL` environment variable (default `http://localhost:8000`) — allows Spaces to point at a deployed API without code changes.
- `REQUEST_TIMEOUT` environment variable (default `30` seconds).
- Graceful error display — if the endpoint is unreachable, display the error message in the answer box rather than raising.

Sources formatted as:
```
[1] WHO-EML-2023.pdf  p.3
    Amitriptyline 10–25 mg at night...
```

Key decisions:
- **Calls FastAPI over HTTP** (not importing `src.*` directly) — the Gradio app and FastAPI server can run in separate processes/containers; keeps Spaces deployment simple (just needs the API URL).
- **`gradio.Blocks`** — more layout control than `Interface`; allows side-by-side answer + sources.
- **`share=False`** — HF Spaces handles public exposure; no need for Gradio's tunnel.
- **No authentication** — demo-grade; add if needed.

---

### README additions

Three-command quickstart (under 5 commands):

```bash
# 1. Copy env template and fill in your key
cp .env.example .env && echo "GROQ_API_KEY=gsk-..." >> .env

# 2. Build and start all services
docker compose up --build -d

# 3. Test the endpoint
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the dose of amitriptyline for neuropathic pain?"}' | python -m json.tool
```

Health check (optional 4th command):
```bash
curl http://localhost:8000/health
```

---

### Verification

```bash
# Docker build (no compose)
docker build -t medical-rag-qa .

# Compose up (requires running Docker)
docker compose up --build

# CI — runs automatically on push to main; verify locally with:
ruff check src/ app/ tests/
pytest tests/ -v
```
