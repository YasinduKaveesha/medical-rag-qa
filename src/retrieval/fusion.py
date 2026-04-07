"""Reciprocal Rank Fusion (RRF) for multimodal retrieval results.

Merges ranked result lists from heterogeneous retrievers (dense text, CLIP
image) into a single ranked list.  Image results that appear in both the CLIP
collection and the text caption collection are deduplicated by ``image_id``:
only the highest-scoring entry survives, with metadata merged from all hits.

Typical usage
-------------
::

    from src.retrieval.fusion import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        [text_results, clip_results],
        k=60,
        top_k=5,
    )
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result in a list must contain one of: ``chunk_id``, ``image_id``, or
    ``id`` — used as the deduplication key across lists.  Results that share
    an ``image_id`` are further deduplicated: the entry with the highest RRF
    score is kept and metadata from all matching hits is merged.  Text results
    (no ``image_id``) pass through unchanged.

    Args:
        result_lists: List of ranked result lists.  Each inner list contains
            dicts representing retrieved items.  Order within each list
            determines the RRF rank (index 0 = rank 1).
        k: RRF constant controlling the rank penalty.  Defaults to ``60``
            (standard value from Cormack et al., 2009).
        top_k: Maximum number of results to return.  Defaults to ``10``.

    Returns:
        List of merged result dicts, each with an added ``rrf_score`` field,
        sorted by ``rrf_score`` descending.  At most *top_k* items are
        returned.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Accumulate RRF scores and collect ALL metadata variants      #
    # ------------------------------------------------------------------ #
    rrf_scores: dict[str, float] = defaultdict(float)
    all_metadata: dict[str, list[dict]] = defaultdict(list)

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            key = (
                result.get("chunk_id")
                or result.get("image_id")
                or result.get("id", "")
            )
            if not key:
                logger.warning("fusion: result missing id key — skipping: %s", result)
                continue

            rrf_scores[key] += 1.0 / (k + rank)
            all_metadata[key].append(dict(result))

    # ------------------------------------------------------------------ #
    # Step 2: Build result list — merge metadata from all variants         #
    # ------------------------------------------------------------------ #
    results: list[dict] = []
    for key, score in rrf_scores.items():
        variants = all_metadata[key]

        if len(variants) == 1:
            entry = dict(variants[0])
        else:
            # Start from richest variant, fill missing fields from others
            sorted_variants = sorted(variants, key=len, reverse=True)
            entry = dict(sorted_variants[0])
            for variant in sorted_variants[1:]:
                for field, value in variant.items():
                    if field not in entry:
                        entry[field] = value
            # Record all contributing types as retrieval_sources
            types: list[str] = []
            for v in variants:
                t = v.get("type", "")
                if t and t not in types:
                    types.append(t)
            if len(types) > 1:
                entry["retrieval_sources"] = types

        entry["rrf_score"] = score
        results.append(entry)

    # ------------------------------------------------------------------ #
    # Step 3: Deduplicate by image_id                                      #
    # ------------------------------------------------------------------ #
    results = _deduplicate_by_image_id(results)

    # ------------------------------------------------------------------ #
    # Step 4: Sort descending by rrf_score and truncate                    #
    # ------------------------------------------------------------------ #
    results.sort(key=lambda r: r["rrf_score"], reverse=True)

    logger.debug(
        "fusion: %d input keys → %d after dedup → returning top %d",
        len(rrf_scores),
        len(results),
        top_k,
    )
    return results[:top_k]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _deduplicate_by_image_id(results: list[dict]) -> list[dict]:
    """Collapse entries that share the same ``image_id``.

    Text-only results (no ``image_id``) pass through unchanged.  For image
    groups the entry with the highest ``rrf_score`` is kept as the base;
    all other entries' fields are merged into it.  A ``retrieval_sources``
    field lists every ``type`` value that contributed to the merge.

    Args:
        results: Flat list of result dicts, each with an ``rrf_score`` field.

    Returns:
        Deduplicated list — same ordering is not guaranteed; call-site sorts.
    """
    # Separate image results (have image_id) from text results
    image_groups: dict[str, list[dict]] = defaultdict(list)
    text_results: list[dict] = []

    for result in results:
        image_id = result.get("image_id")
        if image_id:
            image_groups[image_id].append(result)
        else:
            text_results.append(result)

    deduplicated: list[dict] = list(text_results)

    for image_id, group in image_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue

        # Sort group by rrf_score descending — winner is base
        group.sort(key=lambda r: r["rrf_score"], reverse=True)
        merged = dict(group[0])  # start from highest-scoring entry

        # Collect retrieval_sources and merge missing fields from all group entries
        sources: list[str] = []
        for entry in group:
            entry_type = entry.get("type", "")
            if entry_type and entry_type not in sources:
                sources.append(entry_type)
            for field, value in entry.items():
                # Always add fields that are absent in the merged base
                if field not in merged:
                    merged[field] = value

        # Use the maximum rrf_score across the group
        merged["rrf_score"] = group[0]["rrf_score"]
        merged["retrieval_sources"] = sources
        deduplicated.append(merged)

    return deduplicated
