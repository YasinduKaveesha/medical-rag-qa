"""LLM client wrapping the OpenAI SDK pointed at Groq's API.

Groq exposes an OpenAI-compatible REST endpoint, so the standard ``openai``
package is used with a custom ``base_url``.  All credentials come from
:func:`src.config.get_settings` — no secrets are hardcoded.

Typical usage
-------------
::

    from src.generation.llm_client import get_llm_client

    client = get_llm_client()
    answer = client.generate(prompt)
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMClient:
    """Wrapper around the OpenAI client pointed at Groq.

    The underlying HTTP client is constructed once in ``__init__`` using
    credentials from :func:`src.config.get_settings`.  Use
    :func:`get_llm_client` to obtain the process-level singleton.

    Args:
        model: Model identifier to use for generation.  Defaults to
            ``Settings.llm_model`` (``"llama-3.3-70b-versatile"``).
        _client: Optional pre-constructed OpenAI client.  When provided it
            is used directly, bypassing real client construction.  Intended
            for testing only — pass a :class:`unittest.mock.MagicMock` to
            avoid making real API calls.
    """

    def __init__(
        self,
        model: str | None = None,
        _client: Any | None = None,
    ) -> None:
        s = get_settings()
        self._model = model or s.llm_model

        if _client is not None:
            self._client = _client
            logger.debug("LLMClient using injected client (test mode)")
        else:
            from openai import OpenAI  # noqa: PLC0415

            logger.info(
                "Initialising OpenAI client: base_url=%s  model=%s",
                s.llm_base_url,
                self._model,
            )
            self._client = OpenAI(
                api_key=s.groq_api_key,
                base_url=s.llm_base_url,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier used for generation."""
        return self._model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response text.

        Uses ``temperature=0`` for deterministic, reproducible output —
        required for consistent RAGAS evaluation in Module 9.

        Args:
            prompt: Fully assembled prompt string as returned by
                :func:`src.generation.prompt_builder.build_prompt`.

        Returns:
            Stripped response text from the LLM.

        Raises:
            openai.OpenAIError: On API errors (rate limit, auth failure, etc.).
        """
        logger.info("LLMClient.generate: model=%s  prompt_chars=%d", self._model, len(prompt))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        answer = response.choices[0].message.content.strip()

        logger.info("LLMClient.generate: response_chars=%d", len(answer))
        return answer


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the process-level singleton :class:`LLMClient`.

    On the first call, reads model and credentials from
    :func:`src.config.get_settings` and constructs the client.  Subsequent
    calls return the cached instance.

    Returns:
        The singleton :class:`LLMClient` instance.
    """
    global _llm_client
    if _llm_client is None:
        logger.info("Initialising singleton LLMClient")
        _llm_client = LLMClient()
    return _llm_client
