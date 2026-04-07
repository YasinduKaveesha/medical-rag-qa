# CLAUDE.md — Multimodal RAG Extension (Phase 2)

> This file is the single source of truth for AI coding assistants working on the multimodal extension.
> Read it fully before writing any code. This extends the existing medical-rag-qa project (Phase 1).

## Project Identity

- **Name**: medical-rag-qa (same repo — Phase 2 extension)
- **Purpose**: Extend the existing text-only RAG system with image understanding — CLIP embeddings, BLIP captioning, multimodal retrieval and generation
- **Author**: Yasindu Kaveesha
- **Repository**: github.com/YasinduKaveesha/medical-rag-qa
- **Phase 1**: Text-only RAG — complete (252 tests, deployed)
- **Phase 2**: Multimodal RAG — this document
- **Domain**: Medical documents — clinical guidelines with embedded images (diagrams, flowcharts, X-rays)
- **Budget**: Zero — all free-tier services only
- **GPU**: RTX 3050 6GB available, prefer CPU where possible

## Critical Rule: Do NOT Break Phase 1

Every change in Phase 2 must preserve backward compatibility:
- `POST /ask` still works exactly as before
- `GET /health` still works exactly as before
- All 252 existing tests still pass
- Existing classes and functions keep their signatures unchanged
- New functionality is ADDED alongside existing code, never replacing it

## Architecture Overview

Phase 1 (existing): Single-collection text RAG
Phase 2 (new): Dual-collection multimodal RAG layered on top

### Existing P1 Data Flow (DO NOT MODIFY)
```
PDF → parse_pdf() → attach_metadata() → Chunker.chunk() → EmbeddingEncoder.encode_batch()
→ QdrantStore.upsert_chunks() → RetrievalPipeline.retrieve() → should_refuse()
→ build_prompt() → LLMClient.generate() → extract_citations() → POST /ask
```

### New P2 Data Flow (ADDED ALONGSIDE)
```
PDF → parse_pdf() [reuse] → attach_metadata() [reuse] → Chunker.chunk() [reuse]
  └→ ImageExtractor.extract_images_from_pdf() → ImageCaptioner.caption_extracted_images()
     └→ CLIPEncoder.encode_image() → MultiModalVectorStore.upsert_images() [CLIP collection]
     └→ EmbeddingEncoder.encode_batch(captions) → MultiModalVectorStore.upsert_image_captions() [text collection]

Query → MultiModalRetrievalPipeline.retrieve()
  ├→ EmbeddingEncoder.encode() → search text collection → CrossEncoderReranker.rerank()
  └→ CLIPEncoder.encode_text() → search CLIP collection
  → reciprocal_rank_fusion() [with image_id deduplication]
  → build_multimodal_prompt() → LLMClient.generate() → extract_multimodal_citations()
  → POST /ask-multimodal
```

### Deduplication Rule

During ingestion, image captions are embedded with MiniLM and stored in the text collection with `type="image_caption"` and an `image_id` field. A query can retrieve the same image twice — once from CLIP (image embedding) and once from text (caption match). The fusion step MUST deduplicate by `image_id`: keep the higher RRF score, merge metadata from both hits.

## Existing P1 Classes and Functions (Reference — DO NOT MODIFY SIGNATURES)

| Module | Class/Function | Key Methods/Signature |
|---|---|---|
| `src/config.py` | `Settings` dataclass | `get_settings() -> Settings` |
| `src/ingestion/pdf_parser.py` | `parse_pdf(path)` | Returns `list[dict]` — pages with text |
| `src/ingestion/metadata.py` | `attach_metadata(pages, source_document, pdf_path=None)` | Returns pages with metadata |
| `src/ingestion/chunkers.py` | `FixedSizeChunker`, `SentenceChunker`, `SemanticChunker` | `.chunk(text, metadata) -> list[dict]` |
| `src/embeddings/encoder.py` | `EmbeddingEncoder` | `.encode(text) -> np.ndarray`, `.encode_batch(texts, batch_size=32) -> list[np.ndarray]`, `get_encoder()` |
| `src/retrieval/vector_store.py` | `QdrantStore` | `.__init__(host, port, collection_name, _client=None)`, `.create_collection(vector_size=384)`, `.upsert_chunks(chunks, embeddings)`, `.search(query_vector, top_k=20, filters=None)`, `.get_collection_info()`, `.delete_collection()`, `get_store()` |
| `src/retrieval/reranker.py` | `CrossEncoderReranker` | `.rerank(query, candidates, top_k=5) -> list[dict]`, `get_reranker()` |
| `src/retrieval/pipeline.py` | `RetrievalPipeline` | `.__init__(encoder, store, reranker)`, `.retrieve(query, top_k=5, filters=None) -> list[dict]`, `get_pipeline()` |
| `src/generation/prompt_builder.py` | `build_prompt(query, chunks)` | Returns prompt string |
| `src/generation/llm_client.py` | `LLMClient` | `.__init__(model, _client=None)`, `.generate(prompt) -> str`, `get_llm_client()` |
| `src/generation/refusal.py` | `should_refuse(chunks)` | Returns bool |
| `src/generation/citations.py` | `extract_citations(answer, chunks)` | Returns `list[dict]` |
| `app/schemas.py` | `AskRequest`, `AskResponse`, `CitationSource`, `HealthResponse` | Pydantic models |
| `app/main.py` | FastAPI app | `POST /ask`, `GET /health` with `Depends()` |

## Test Data — Medical PDFs with Images

Phase 1 already has 3 PDFs in `data/raw/`:
- `WHO-MHP-HPS-EML-2023.02-eng.pdf`
- `IDF_Rec_2025.pdf`
- `9789240081888-eng.pdf`

For Phase 2, add PDFs that contain embedded images:

1. **WHO Clinical Guidelines** (open access): https://www.who.int/publications — search "pneumonia management"
2. **OpenStax Anatomy & Physiology** (CC-BY): https://openstax.org/details/books/anatomy-and-physiology-2e
3. **PubMed Central Open Access**: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/

Place 2-3 image-rich PDFs in `data/raw/` alongside existing ones. Aim for ~10-20 extractable images.

Unit tests use programmatically generated PDFs (reportlab) — never commit real medical images.

## New Files to Create

```
src/ingestion/image_extractor.py         # NEW
src/ingestion/image_captioner.py         # NEW
src/embeddings/clip_encoder.py           # NEW
src/retrieval/fusion.py                  # NEW
src/retrieval/multimodal_pipeline.py     # NEW
src/evaluation/multimodal_eval.py        # NEW
src/evaluation/multimodal_test_queries.json  # NEW

data/extracted_images/                   # NEW directory (gitignored)

scripts/ingest_multimodal.py             # NEW

tests/test_image_extractor.py            # NEW
tests/test_image_captioner.py            # NEW
tests/test_clip_encoder.py               # NEW
tests/test_fusion.py                     # NEW
tests/test_multimodal_pipeline.py        # NEW
tests/test_multimodal_generation.py      # NEW
tests/test_multimodal_api.py             # NEW
tests/test_multimodal_eval.py            # NEW
tests/test_gradio_multimodal.py          # NEW
```

## Files to Extend (Add Code, Keep All Existing Code)

```
pyproject.toml                     # Add transformers, Pillow, reportlab
.env.example                       # Add new variables
.gitignore                         # Add data/extracted_images/
src/config.py                      # Add new fields to Settings
src/retrieval/vector_store.py      # Add MultiModalVectorStore subclass
src/generation/prompt_builder.py   # Add build_multimodal_prompt()
src/generation/llm_client.py       # Add generate_with_vision() method
src/generation/citations.py        # Add extract_multimodal_citations()
app/schemas.py                     # Add MultiModalAskRequest/Response
app/main.py                        # Add POST /ask-multimodal, GET /images/{image_id}
app/gradio_demo.py                 # Add multimodal tab
tests/conftest.py                  # Add test fixtures
```

## Module Build Order

| # | Module | Key files | Depends on |
|---|---|---|---|
| 0 | Setup | pyproject.toml, config.py, .env.example, conftest.py | None |
| 1 | Image extraction | src/ingestion/image_extractor.py | Module 0 |
| 2 | Image captioning | src/ingestion/image_captioner.py | Module 1 |
| 3 | CLIP encoder | src/embeddings/clip_encoder.py | Module 0 |
| 4 | Vector store extension | src/retrieval/vector_store.py (extend) | Module 3 |
| 5a | RRF fusion | src/retrieval/fusion.py | Module 4 |
| 5b | Multimodal pipeline | src/retrieval/multimodal_pipeline.py | Module 5a |
| 6a | Prompt builder extension | src/generation/prompt_builder.py (extend) | Module 5b |
| 6b | LLM client extension | src/generation/llm_client.py (extend) | Module 6a |
| 6c | Citations extension | src/generation/citations.py (extend) | Module 6b |
| 7 | FastAPI endpoints | app/main.py, app/schemas.py (extend) | Module 6c |
| 8 | Evaluation | src/evaluation/multimodal_eval.py | Module 7 |
| 9 | Gradio + infra | app/gradio_demo.py, scripts/ | Module 8 |

---

## Module Specifications

### Module 0: Setup (Extend Existing Config)

**pyproject.toml** — add to existing dependencies:
- `transformers>=4.36.0`
- `Pillow>=10.0.0`
- Dev: `reportlab>=4.0`

**.env.example** — add these lines below existing variables:
```
# Phase 2 — Multimodal
CLIP_COLLECTION_NAME=multimodal_clip
CLIP_MODEL=openai/clip-vit-base-patch32
CAPTION_MODEL=Salesforce/blip-image-captioning-base
VISION_LLM_MODEL=llama-3.2-11b-vision-preview
RRF_K=60
DEVICE=cpu
EXTRACTED_IMAGES_DIR=data/extracted_images
```

**src/config.py** — add fields to existing `Settings` dataclass (after existing fields):
```python
clip_collection_name: str = "multimodal_clip"
clip_model: str = "openai/clip-vit-base-patch32"
caption_model: str = "Salesforce/blip-image-captioning-base"
vision_llm_model: str = "llama-3.2-11b-vision-preview"
rrf_k: int = 60
device: str = "cpu"
extracted_images_dir: str = "data/extracted_images"
```
Update `get_settings()` to load these from env vars. Add DEVICE auto-detection: if `torch.cuda.is_available()` and env DEVICE != "cpu", use "cuda".

**.gitignore** — add: `data/extracted_images/`

**tests/conftest.py** — add these fixtures (keep existing `matplotlib.use("Agg")`):
- `sample_image` → 200x200 RGB PIL Image with shapes (Pillow ImageDraw)
- `sample_pdf_with_images` → reportlab-generated tiny PDF with one embedded image
- `sample_text_chunks` → list of 3 dicts: `{text, source_document, page_number, section_title, chunk_id, score}`
- `temp_image_dir` → `tmp_path` based
- `in_memory_qdrant` → `QdrantClient(":memory:")`

**Tests (~3 added to existing tests/test_config.py)**:
- test_config_has_clip_fields
- test_config_has_caption_fields
- test_config_has_device_field

---

### Module 1: Image Extraction

**File**: `src/ingestion/image_extractor.py` (NEW)

**ExtractedImage** dataclass: `image_path, source_pdf, page_number, xref, width, height, image_id`
- `image_id` format: `f"{pdf_stem}_p{page_num}_x{xref}"`

**ImageExtractor** class:
- `__init__(self, output_dir: str)` — create dir if missing
- `extract_images_from_pdf(self, pdf_path: str) -> list[ExtractedImage]`
- `extract_images_from_page(self, page: fitz.Page, pdf_path: str, page_num: int) -> list[ExtractedImage]`
- `_save_image(self, image_bytes: bytes, xref: int, pdf_path: str, page_num: int) -> str`
- `_is_valid_image(self, image_bytes: bytes, min_size: tuple = (50, 50)) -> bool`

Uses `page.get_images(full=True)`, `doc.extract_image(xref)`. Dedup by xref. Python logging.

**Tests (~12 in tests/test_image_extractor.py)**

---

### Module 2: Image Captioning

**File**: `src/ingestion/image_captioner.py` (NEW)

**CaptionedImage** dataclass: all ExtractedImage fields + `caption: str`, `caption_model: str`

**ImageCaptioner** class:
- `__init__(self, model_name="Salesforce/blip-image-captioning-base", device="cpu")`
- `caption_image(self, image: Image.Image | str) -> str` — clean "arafed" BLIP artifact
- `caption_batch(self, images, batch_size=8) -> list[str]`
- `caption_extracted_images(self, extracted_images) -> list[CaptionedImage]`

Mock model for fast tests. `@pytest.mark.slow` for real model.

**Tests (~10 in tests/test_image_captioner.py)**

---

### Module 3: CLIP Encoder

**File**: `src/embeddings/clip_encoder.py` (NEW)

**CLIPEncoder** class (same singleton pattern as `EmbeddingEncoder`):
- `__init__(self, model_name="openai/clip-vit-base-patch32", device="cpu")`
- `encode_image(self, image) -> np.ndarray` — 512-dim, L2-normalized
- `encode_text(self, text) -> np.ndarray` — 512-dim, L2-normalized
- `encode_images_batch(self, images, batch_size=16) -> list[np.ndarray]`
- `encode_texts_batch(self, texts, batch_size=32) -> list[np.ndarray]`
- `compute_similarity(self, emb_a, emb_b) -> float`

Singleton: `_clip_encoder: CLIPEncoder | None = None` + `get_clip_encoder() -> CLIPEncoder`

Uses `torch.no_grad()`, returns numpy arrays.

**Tests (~12 in tests/test_clip_encoder.py)**

---

### Module 4: Vector Store Extension

**File**: `src/retrieval/vector_store.py` (EXTEND — add new class below existing `QdrantStore`)

**MultiModalVectorStore(QdrantStore)** — inherits from existing QdrantStore:
- `__init__(self, host, port, text_collection, clip_collection, _client=None)`
  - Calls `super().__init__(host, port, text_collection, _client)` for P1 compatibility
  - Stores `clip_collection` name separately
- `create_clip_collection(self, vector_size=512)` — Cosine distance, payload indexes on type/source_pdf/page_number/image_id
- `upsert_images(self, collection, images: list[CaptionedImage], embeddings) -> int`
  - Payload: `{type: "image", image_id, image_path, caption, source_pdf, page_number, width, height}`
- `upsert_image_captions(self, collection, captions: list[dict], embeddings) -> int`
  - Payload: `{type: "image_caption", image_id, text: caption, source_document, page_number, image_path}`
- `search_clip(self, collection, query_vector, top_k=20) -> list[dict]`
- `get_clip_collection_info(self) -> dict`
- `delete_clip_collection(self) -> None`

All existing `QdrantStore` methods remain unchanged. `get_store()` still returns the P1 store.

New singleton: `_mm_store` + `get_multimodal_store() -> MultiModalVectorStore`

**Tests (~10 NEW tests added — keep existing 27 unchanged)**

---

### Module 5a: RRF Fusion

**File**: `src/retrieval/fusion.py` (NEW)

**Function**: `reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_k: int = 10) -> list[dict]`

Algorithm: standard RRF scoring + deduplication by `image_id`. Pure function, no side effects.

**Deduplication**: Group by `image_id`, keep highest rrf_score entry, merge metadata, tag `retrieval_sources`.

**Tests (~9 in tests/test_fusion.py)** — includes 2 dedup-specific tests

---

### Module 5b: Multimodal Retrieval Pipeline

**File**: `src/retrieval/multimodal_pipeline.py` (NEW)

**RetrievalResult** dataclass: `text_chunks, images, fusion_scores, retrieval_time_ms`

**MultiModalRetrievalPipeline**:
- `__init__(self, text_encoder: EmbeddingEncoder, clip_encoder: CLIPEncoder, store: MultiModalVectorStore, reranker: CrossEncoderReranker, config: Settings)`
- `retrieve(self, query, top_k=5) -> RetrievalResult`
- `_retrieve_text(self, query, top_k=20)` — EmbeddingEncoder → QdrantStore.search → reranker
- `_retrieve_images(self, query, top_k=20)` — CLIPEncoder → search_clip (no rerank)
- `_classify_results(self, fused)` — split by type: "text" → text_chunks, "image"/"image_caption" → images

Singleton: `get_multimodal_pipeline()`

**Tests (~8 in tests/test_multimodal_pipeline.py)** — mock all dependencies

---

### Module 6a: Prompt Builder Extension

**File**: `src/generation/prompt_builder.py` (EXTEND)

**Add** `build_multimodal_prompt(query: str, text_chunks: list[dict], images: list[dict]) -> str`

Existing `build_prompt(query, chunks)` stays unchanged.

**Tests (~5 in tests/test_multimodal_generation.py)**

---

### Module 6b: LLM Client Extension

**File**: `src/generation/llm_client.py` (EXTEND — add method to existing `LLMClient` class)

**Add method**: `LLMClient.generate_with_vision(self, prompt: str, image_paths: list[str], max_images: int = 2) -> str`
- Uses `self._client` (existing openai-compatible client) with vision model
- Base64 encodes images, builds content list
- Falls back to `self.generate(prompt)` on ANY exception
- Vision is a bonus feature — the demo must work without it

Existing `LLMClient.generate(prompt)` and `LLMClient.__init__` stay unchanged.

**Tests (~4 added to tests/test_multimodal_generation.py)**

---

### Module 6c: Citations Extension

**File**: `src/generation/citations.py` (EXTEND)

**Add** `extract_multimodal_citations(answer: str, text_chunks: list[dict], images: list[dict]) -> list[dict]`

Existing `extract_citations(answer, chunks)` stays unchanged.

**Tests (~3 added to tests/test_multimodal_generation.py)**

---

### Module 7: FastAPI Endpoints

**app/schemas.py** (EXTEND — add new models):
```python
class MultiModalAskRequest(BaseModel):
    question: str
    use_vision: bool = False
    top_k: int = 5
    include_images: bool = True

class ImageResult(BaseModel):
    image_id: str
    image_path: str
    caption: str
    source_pdf: str
    page_number: int
    relevance_score: float

class MultiModalAskResponse(BaseModel):
    answer: str
    text_sources: list[dict]
    image_sources: list[ImageResult]
    used_vision_model: bool
    retrieval_time_ms: float
    model_version: str
```

**app/main.py** (EXTEND — add endpoints):
- `POST /ask-multimodal` — uses MultiModalRetrievalPipeline
- `GET /images/{image_id}` — serves images from extracted_images dir

Existing `POST /ask` and `GET /health` stay unchanged.

**Tests (~8 in tests/test_multimodal_api.py)**

---

### Module 8: Evaluation (Custom Metrics, No RAGAS)

**Files**: `src/evaluation/multimodal_eval.py` (NEW), `src/evaluation/multimodal_test_queries.json` (NEW)

15 test queries (8 image, 7 text) with `expected_type` and `expected_keywords`.

**MultiModalEvaluator**:
- `evaluate_retrieval(top_k=5) -> dict` — precision, modality accuracy
- `image_retrieval_precision(query, result) -> float`
- `text_retrieval_precision(query, result) -> float`
- `modality_accuracy(query, result) -> float`
- `compare_text_only_vs_multimodal() -> dict`
- `generate_report(results) -> str` — markdown table

Deterministic, free, no API calls needed.

**Tests (~8 in tests/test_multimodal_eval.py)**

---

### Module 9: Gradio + Ingestion Script

**app/gradio_demo.py** (EXTEND — add multimodal tab alongside existing text-only interface)

**scripts/ingest_multimodal.py** (NEW):
```bash
python scripts/ingest_multimodal.py --pdf-dir data/raw/ --output-dir data/extracted_images/
```

Steps:
1. Parse PDFs with `parse_pdf()` (reuse P1)
2. Extract images with `ImageExtractor`
3. Caption with `ImageCaptioner`
4. Chunk text (reuse P1 chunkers)
5. Embed text → upsert to text collection
6. Embed images with CLIP → upsert to CLIP collection
7. Embed captions with MiniLM → upsert to text collection with `type="image_caption"` + `image_id`

**Tests (~5 in tests/test_gradio_multimodal.py)**

---

## Non-Negotiable Rules

### Carried from Phase 1
1. Commit daily. Never bulk-upload.
2. Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
3. Never force-push to main.
4. Tests before next module.
5. No hardcoded API keys.
6. Type hints + docstrings + logging (Python `logging` module, not print).
7. Do not start next module until current one works.

### Phase 2 Specific
8. **Never modify existing P1 function signatures.** Add new functions/methods alongside.
9. **Run `pytest tests/ -v` after every module** — all 252 P1 tests must still pass.
10. **CPU-first.** GPU is optional.
11. **Vision LLM is optional.** Demo works with text-only generation + captions.
12. **Deduplicate by image_id in fusion.**
13. **Image paths are relative** to project root.
14. **Test images generated programmatically** — never commit real medical images.

## README Update Plan

After Phase 2, update README.md:
1. "Phase 2: Multimodal Extension" section
2. Comparison table: text-only vs multimodal metrics
3. Updated architecture diagram with dual-collection flow
4. New API docs for `POST /ask-multimodal`
5. Updated project structure
6. "What's Next: Phase 3 — Agentic RAG with LangGraph"
