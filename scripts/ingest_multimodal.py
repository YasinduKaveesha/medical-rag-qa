"""Multimodal ingestion script — Phase 2.

Processes PDF files in a directory, extracts text chunks and embedded images,
captions the images with BLIP, and ingests everything into Qdrant:

- Text chunks → ``text`` collection (MiniLM embeddings)
- Images → CLIP collection (CLIP embeddings)
- Image captions → ``text`` collection with ``type="image_caption"`` + ``image_id``

Usage
-----
::

    python scripts/ingest_multimodal.py --pdf-dir data/raw/ --output-dir data/extracted_images/
    python scripts/ingest_multimodal.py --pdf-dir data/raw/ --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("ingest_multimodal")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal ingestion: text + image embeddings into Qdrant.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw/",
        help="Directory containing PDF files to ingest (default: data/raw/).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/extracted_images/",
        help="Directory for extracted PNG images (default: data/extracted_images/).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for CLIP and BLIP models (default: cpu).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: C901
    args = parse_args(argv)

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)

    if not pdf_dir.exists():
        logger.error("pdf_dir does not exist: %s", pdf_dir)
        sys.exit(1)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s — nothing to ingest.", pdf_dir)
        return

    logger.info("Found %d PDF(s) in %s", len(pdf_paths), pdf_dir)
    logger.info("Extracted images will be saved to: %s", output_dir)
    logger.info("Device: %s", args.device)

    # ------------------------------------------------------------------
    # Lazy imports — keep startup fast and avoid GPU init until needed
    # ------------------------------------------------------------------
    from src.config import get_settings
    from src.embeddings.encoder import get_encoder
    from src.embeddings.clip_encoder import get_clip_encoder
    from src.ingestion.image_captioner import ImageCaptioner
    from src.ingestion.image_extractor import ImageExtractor
    from src.ingestion.chunkers import SentenceChunker
    from src.ingestion.metadata import attach_metadata
    from src.ingestion.pdf_parser import parse_pdf
    from src.retrieval.vector_store import get_multimodal_store

    s = get_settings()
    store = get_multimodal_store()
    text_encoder = get_encoder()
    clip_encoder = get_clip_encoder()
    chunker = SentenceChunker()
    extractor = ImageExtractor(output_dir=str(output_dir))
    captioner = ImageCaptioner(
        model_name=s.caption_model,
        device=args.device,
    )

    # Ensure collections exist
    store.create_collection(vector_size=384)
    store.create_clip_collection(vector_size=512)

    total_chunks = 0
    total_images = 0
    total_captions = 0

    for pdf_path in pdf_paths:
        logger.info("--- Processing: %s ---", pdf_path.name)

        # ---- Text ingestion (reuse P1) --------------------------------
        pages = parse_pdf(str(pdf_path))
        pages = attach_metadata(pages, source_document=pdf_path.name, pdf_path=str(pdf_path))

        chunks: list[dict] = []
        for page in pages:
            page_chunks = chunker.chunk(page["text"], page.get("metadata", {}))
            chunks.extend(page_chunks)

        if chunks:
            texts = [c.get("chunk_text", c.get("text", "")) for c in chunks]
            embeddings = text_encoder.encode_batch(texts)
            n = store.upsert_chunks(chunks, embeddings)
            total_chunks += n
            logger.info("  Text chunks ingested: %d", n)
        else:
            logger.info("  No text chunks extracted.")

        # ---- Image extraction + captioning ----------------------------
        extracted = extractor.extract_images_from_pdf(str(pdf_path))
        logger.info("  Images extracted: %d", len(extracted))

        if not extracted:
            continue

        captioned = captioner.caption_extracted_images(extracted)
        logger.info("  Images captioned: %d", len(captioned))

        # ---- CLIP embeddings → clip collection ------------------------
        from PIL import Image as PILImage

        pil_images = []
        valid_captioned = []
        for ci in captioned:
            try:
                pil_images.append(PILImage.open(ci.image_path).convert("RGB"))
                valid_captioned.append(ci)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  Skipping %s: %s", ci.image_path, exc)

        if pil_images:
            clip_embeddings = clip_encoder.encode_images_batch(pil_images)
            n_img = store.upsert_images(s.clip_collection_name, valid_captioned, clip_embeddings)
            total_images += n_img
            logger.info("  CLIP image vectors upserted: %d", n_img)

        # ---- Caption embeddings → text collection (type=image_caption) -
        caption_dicts = [
            {
                "image_id": ci.image_id,
                "caption": ci.caption,
                "source_document": ci.source_pdf,
                "page_number": ci.page_number,
                "image_path": ci.image_path,
            }
            for ci in valid_captioned
        ]
        if caption_dicts:
            caption_texts = [d["caption"] for d in caption_dicts]
            caption_embeddings = text_encoder.encode_batch(caption_texts)
            n_cap = store.upsert_image_captions(
                s.collection_name, caption_dicts, caption_embeddings
            )
            total_captions += n_cap
            logger.info("  Caption vectors upserted: %d", n_cap)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("Ingestion complete.")
    logger.info("  PDFs processed  : %d", len(pdf_paths))
    logger.info("  Text chunks     : %d", total_chunks)
    logger.info("  CLIP images     : %d", total_images)
    logger.info("  Caption vectors : %d", total_captions)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
