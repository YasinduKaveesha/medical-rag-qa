"""Gradio demo for the Medical RAG Q&A system.

Provides a simple web UI with two tabs:

- **Tab 1 "Text Q&A"**: forwards questions to ``POST /ask`` and displays
  the answer with formatted source citations.
- **Tab 2 "Multimodal Q&A"**: forwards questions to ``POST /ask-multimodal``
  and displays the answer, retrieved images in a gallery, and JSON citations.

Intended for local use and HuggingFace Spaces deployment.  Configure the
target API via environment variables:

- ``FASTAPI_URL`` — base URL of the FastAPI service
  (default: ``http://localhost:8000``).
- ``REQUEST_TIMEOUT`` — HTTP request timeout in seconds (default: ``30``).

Typical usage
-------------
::

    # Local (FastAPI must be running)
    python app/gradio_demo.py

    # HuggingFace Spaces — set FASTAPI_URL as a Space secret pointing at
    # the deployed API, then push this file as app.py.
"""

from __future__ import annotations

import logging
import os

import gradio as gr
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FASTAPI_URL: str = os.environ.get("FASTAPI_URL", "http://localhost:8000")
_REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------


def ask_question(question: str) -> tuple[str, str]:
    """Send *question* to the FastAPI ``/ask`` endpoint.

    Args:
        question: The user's clinical question.

    Returns:
        ``(answer, sources_text)`` — both are plain strings ready for display.
        On error the answer contains the error message and sources is empty.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    url = f"{_FASTAPI_URL}/ask"
    try:
        response = requests.post(
            url,
            json={"question": question},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        msg = f"Could not connect to the API at {_FASTAPI_URL}. Is the server running?"
        logger.warning("gradio_demo: ConnectionError -> %s", url)
        return msg, ""
    except requests.exceptions.Timeout:
        msg = f"Request timed out after {_REQUEST_TIMEOUT}s."
        logger.warning("gradio_demo: Timeout -> %s", url)
        return msg, ""
    except requests.exceptions.HTTPError as exc:
        msg = f"API error {exc.response.status_code}: {exc.response.text}"
        logger.warning("gradio_demo: HTTPError -> %s  %s", url, exc)
        return msg, ""

    data = response.json()
    answer: str = data.get("answer", "")
    sources: list[dict] = data.get("sources", [])

    sources_text = _format_sources(sources)
    logger.info(
        "gradio_demo: answer_len=%d  sources=%d", len(answer), len(sources)
    )
    return answer, sources_text


def ask_multimodal_question(question: str, use_vision: bool) -> tuple[str, list, dict]:
    """Send *question* to the FastAPI ``/ask-multimodal`` endpoint.

    Args:
        question: The user's clinical question.
        use_vision: When ``True``, request the vision LLM path.

    Returns:
        ``(answer, image_paths, citations)`` where *image_paths* is a list of
        local file paths for the :class:`gradio.Gallery` component, and
        *citations* is a raw dict for the :class:`gradio.JSON` component.
        On error the answer contains the error message; images and citations
        are empty.
    """
    question = question.strip()
    if not question:
        return "Please enter a question.", [], {}

    url = f"{_FASTAPI_URL}/ask-multimodal"
    try:
        response = requests.post(
            url,
            json={"question": question, "use_vision": use_vision, "include_images": True},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        msg = f"Could not connect to the API at {_FASTAPI_URL}. Is the server running?"
        logger.warning("gradio_demo: multimodal ConnectionError -> %s", url)
        return msg, [], {}
    except requests.exceptions.Timeout:
        msg = f"Request timed out after {_REQUEST_TIMEOUT}s."
        logger.warning("gradio_demo: multimodal Timeout -> %s", url)
        return msg, [], {}
    except requests.exceptions.HTTPError as exc:
        msg = f"API error {exc.response.status_code}: {exc.response.text}"
        logger.warning("gradio_demo: multimodal HTTPError -> %s  %s", url, exc)
        return msg, [], {}

    data = response.json()
    answer: str = data.get("answer", "")
    image_sources: list[dict] = data.get("image_sources", [])
    text_sources: list[dict] = data.get("text_sources", [])

    # Build gallery paths — serve via /images/{image_id}
    image_paths: list[str] = []
    for img in image_sources[:4]:
        image_id = img.get("image_id", "")
        if image_id:
            img_url = f"{_FASTAPI_URL}/images/{image_id}"
            image_paths.append(img_url)

    citations = {
        "text_sources": text_sources,
        "image_sources": image_sources,
        "used_vision_model": data.get("used_vision_model", False),
        "retrieval_time_ms": data.get("retrieval_time_ms", 0.0),
    }

    logger.info(
        "gradio_demo: multimodal answer_len=%d  images=%d  text_sources=%d",
        len(answer),
        len(image_paths),
        len(text_sources),
    )
    return answer, image_paths, citations


def _format_sources(sources: list[dict]) -> str:
    """Format *sources* as a numbered plain-text list.

    Args:
        sources: List of source dicts from the ``/ask`` response, each with
            ``source_document``, ``page_number``, and ``source_chunk`` fields.

    Returns:
        Multi-line string, one block per source, or ``""`` when *sources* is
        empty.
    """
    if not sources:
        return ""

    lines: list[str] = []
    for i, src in enumerate(sources, start=1):
        doc = src.get("source_document", "")
        page = src.get("page_number")
        chunk = src.get("source_chunk", "")

        page_label = f"  p.{page}" if page is not None else ""
        lines.append(f"[{i}] {doc}{page_label}")
        if chunk:
            # Indent the excerpt for readability
            lines.append(f"    {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    """Build and return the Gradio ``Blocks`` app with two tabs.

    - **Tab 1 "Text Q&A"**: the original text-only interface (unchanged).
    - **Tab 2 "Multimodal Q&A"**: new interface using ``POST /ask-multimodal``.

    Returns:
        Configured :class:`gradio.Blocks` instance.
    """
    with gr.Blocks(title="Medical RAG Q&A") as demo:
        gr.Markdown(
            """
            # Medical RAG Q&A
            Ask clinical questions about drug dosing, contraindications,
            indications, and mechanisms of action.  Answers are grounded in
            WHO Essential Medicines List guidelines and retrieved via a
            Retrieval-Augmented Generation pipeline.

            > **Note:** Answers are for educational purposes only and do not
            > constitute medical advice.
            """
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1 — Text Q&A (original interface, unchanged)
            # ------------------------------------------------------------------
            with gr.Tab("Text Q&A"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_box = gr.Textbox(
                            label="Question",
                            placeholder=(
                                "e.g. What is the recommended starting dose of amitriptyline"
                                " for neuropathic pain?"
                            ),
                            lines=3,
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Ask", variant="primary")
                            clear_btn = gr.ClearButton(
                                components=[question_box],
                                value="Clear",
                            )

                with gr.Row():
                    answer_box = gr.Textbox(
                        label="Answer",
                        lines=6,
                        interactive=False,
                    )

                with gr.Row():
                    sources_box = gr.Textbox(
                        label="Sources",
                        lines=8,
                        interactive=False,
                    )

                submit_btn.click(
                    fn=ask_question,
                    inputs=question_box,
                    outputs=[answer_box, sources_box],
                )
                question_box.submit(
                    fn=ask_question,
                    inputs=question_box,
                    outputs=[answer_box, sources_box],
                )
                clear_btn.click(
                    fn=lambda: ("", ""),
                    outputs=[answer_box, sources_box],
                )

            # ------------------------------------------------------------------
            # Tab 2 — Multimodal Q&A (new)
            # ------------------------------------------------------------------
            with gr.Tab("Multimodal Q&A"):
                gr.Markdown(
                    "Retrieve text **and** images from clinical PDFs.  "
                    "Enable *Use vision model* to send retrieved images to the LLM."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        mm_question_box = gr.Textbox(
                            label="Question",
                            placeholder=(
                                "e.g. Show me the chest X-ray findings for pneumonia."
                            ),
                            lines=3,
                        )
                        mm_vision_checkbox = gr.Checkbox(
                            label="Use vision model",
                            value=False,
                        )
                        with gr.Row():
                            mm_submit_btn = gr.Button("Ask", variant="primary")
                            mm_clear_btn = gr.ClearButton(
                                components=[mm_question_box],
                                value="Clear",
                            )

                with gr.Row():
                    mm_answer_box = gr.Textbox(
                        label="Answer",
                        lines=6,
                        interactive=False,
                    )

                with gr.Row():
                    mm_gallery = gr.Gallery(
                        label="Retrieved Images",
                        columns=2,
                        rows=2,
                        height="auto",
                    )

                with gr.Row():
                    mm_citations_box = gr.JSON(label="Citations")

                mm_submit_btn.click(
                    fn=ask_multimodal_question,
                    inputs=[mm_question_box, mm_vision_checkbox],
                    outputs=[mm_answer_box, mm_gallery, mm_citations_box],
                )
                mm_question_box.submit(
                    fn=ask_multimodal_question,
                    inputs=[mm_question_box, mm_vision_checkbox],
                    outputs=[mm_answer_box, mm_gallery, mm_citations_box],
                )
                mm_clear_btn.click(
                    fn=lambda: ("", [], {}),
                    outputs=[mm_answer_box, mm_gallery, mm_citations_box],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo = build_demo()
    demo.launch(share=False)
