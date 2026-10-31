"""Normalization for bounded Web Scraping cluster settings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

CLUSTER_EMBED_DIM = 128
CLUSTER_MAX_EMBED_DIM = 4096
CLUSTER_PREFILTER_THRESHOLD = 0.2
CLUSTER_SIM_THRESHOLD = 0.4
CLUSTER_MIN_BLOCK_CHARS = 40
CLUSTER_MIN_WORDS = 8
CLUSTER_MAX_BLOCKS = 60
CLUSTER_LINKAGE = "average"
CLUSTER_LINKAGES = frozenset({"average", "complete", "single"})
CLUSTER_TAG_TOP_K = 3
CLUSTER_MAX_TAG_TOP_K = 20


def _first_present(settings: Mapping[str, Any], names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in settings:
            return settings[name]
    return default


def _positive_int(value: Any, default: int, *, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return min(parsed, maximum) if maximum is not None else parsed


def _non_negative_int(value: Any, default: int, *, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return default
    if parsed < 0:
        return default
    return min(parsed, maximum)


def _unit_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and 0.0 <= parsed <= 1.0 else default


def normalize_cluster_settings(
    settings: Mapping[str, Any] | None,
    *,
    env_similarity: float | None = None,
    env_min_words: int | None = None,
    env_linkage: str = "",
) -> dict[str, Any]:
    """Return canonical, finite, resource-bounded cluster settings."""

    raw = dict(settings or {})
    normalized = dict(raw)
    normalized["min_block_chars"] = _positive_int(
        raw.get("min_block_chars", CLUSTER_MIN_BLOCK_CHARS),
        CLUSTER_MIN_BLOCK_CHARS,
    )
    normalized["min_word_count"] = _positive_int(
        _first_present(
            raw,
            ("min_word_count", "min_words", "word_count_threshold"),
            env_min_words if env_min_words is not None else CLUSTER_MIN_WORDS,
        ),
        CLUSTER_MIN_WORDS,
    )
    normalized["max_blocks"] = _positive_int(
        raw.get("max_blocks", CLUSTER_MAX_BLOCKS),
        CLUSTER_MAX_BLOCKS,
        maximum=CLUSTER_MAX_BLOCKS,
    )
    normalized["prefilter_threshold"] = _unit_float(
        raw.get("prefilter_threshold", CLUSTER_PREFILTER_THRESHOLD),
        CLUSTER_PREFILTER_THRESHOLD,
    )
    normalized["cluster_threshold"] = _unit_float(
        _first_present(
            raw,
            ("cluster_threshold", "similarity_threshold"),
            env_similarity if env_similarity is not None else CLUSTER_SIM_THRESHOLD,
        ),
        CLUSTER_SIM_THRESHOLD,
    )
    normalized["embed_dims"] = _positive_int(
        raw.get("embed_dims", CLUSTER_EMBED_DIM),
        CLUSTER_EMBED_DIM,
        maximum=CLUSTER_MAX_EMBED_DIM,
    )
    method = str(_first_present(raw, ("method", "cluster_method"), "greedy")).strip().lower()
    normalized["method"] = method if method in {"greedy", "hierarchical"} else "greedy"
    linkage = str(_first_present(raw, ("linkage", "cluster_linkage"), env_linkage or CLUSTER_LINKAGE)).strip().lower()
    normalized["linkage"] = linkage if linkage in CLUSTER_LINKAGES else CLUSTER_LINKAGE
    normalized["tag_top_k"] = _non_negative_int(
        raw.get("tag_top_k", CLUSTER_TAG_TOP_K),
        CLUSTER_TAG_TOP_K,
        maximum=CLUSTER_MAX_TAG_TOP_K,
    )
    return normalized


def has_valid_hierarchical_linkage(
    settings: Mapping[str, Any] | None,
    *,
    env_linkage: str = "",
) -> bool:
    """Return whether a requested hierarchical mode has a supported linkage."""

    raw = dict(settings or {})
    method = str(_first_present(raw, ("method", "cluster_method"), "greedy")).strip().lower()
    if method != "hierarchical":
        return True
    linkage = str(_first_present(raw, ("linkage", "cluster_linkage"), env_linkage or CLUSTER_LINKAGE)).strip().lower()
    return linkage in CLUSTER_LINKAGES


def normalize_cluster_rule(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize supplied router fields without expanding the public rule mapping."""

    normalized = normalize_cluster_settings(settings)
    result = dict(settings)
    canonical_by_key = {
        "min_block_chars": "min_block_chars",
        "min_word_count": "min_word_count",
        "min_words": "min_word_count",
        "word_count_threshold": "min_word_count",
        "max_blocks": "max_blocks",
        "prefilter_threshold": "prefilter_threshold",
        "cluster_threshold": "cluster_threshold",
        "similarity_threshold": "cluster_threshold",
        "embed_dims": "embed_dims",
        "method": "method",
        "cluster_method": "method",
        "linkage": "linkage",
        "cluster_linkage": "linkage",
        "tag_top_k": "tag_top_k",
    }
    for key, canonical_key in canonical_by_key.items():
        if key in result:
            result[key] = normalized[canonical_key]
    return result
