"""Cluster-based article extraction strategy."""

import hashlib
import math
import os
import re
from typing import Any, Callable, Optional

from bs4 import BeautifulSoup

from ...cluster_settings import (
    CLUSTER_LINKAGES,
    CLUSTER_MAX_BLOCKS,
    has_valid_hierarchical_linkage,
    normalize_cluster_settings,
)
from ..caches import _cluster_cache_get, _cluster_cache_put
from ..dependencies import ExtractionDependencies, build_default_dependencies
from ..metrics import emit_counter

_CLUSTER_LINKAGES = CLUSTER_LINKAGES
_CLUSTER_MAX_BLOCKS = CLUSTER_MAX_BLOCKS
_DEFAULT_CLUSTER_TAG_KEYWORDS: dict[str, list[str]] = {
    "marketing": ["subscribe", "newsletter", "promotion", "marketing"],
    "commerce": ["price", "pricing", "cost", "$"],
    "product": ["feature", "release", "roadmap", "product"],
    "research": ["study", "research", "paper", "dataset"],
    "security": ["security", "encrypt", "token", "oauth"],
}
_METRIC_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)


def _env_int(name: str) -> Optional[int]:  # noqa: UP045
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float(name: str) -> Optional[float]:  # noqa: UP045
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _increment_counter(
    dependencies: ExtractionDependencies,
    name: str,
    *,
    labels: dict[str, str],
) -> None:
    try:
        emit_counter(dependencies, name, labels=labels)
    except _METRIC_NONCRITICAL_EXCEPTIONS:
        return


def _tokenize_cluster_text(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def _cluster_word_count(text: str) -> int:
    return len(_tokenize_cluster_text(text))


def _normalize_vector(vec: list[float]) -> list[float]:
    if not vec:
        return vec
    norm = math.sqrt(sum(val * val for val in vec))
    if norm <= 0.0:
        return vec
    return [val / norm for val in vec]


def _hash_embedding(text: str, dims: int) -> list[float]:
    tokens = _tokenize_cluster_text(text)
    if not tokens:
        return [0.0] * dims
    vec = [0.0] * dims
    for token in tokens:
        token_hash = hashlib.md5(
            token.encode("utf-8", errors="ignore"),
            usedforsecurity=False,
        ).hexdigest()
        idx = int(token_hash, 16) % dims
        vec[idx] += 1.0
    return _normalize_vector(vec)


def _cluster_embedding(
    text: str,
    dims: int,
    *,
    increment_counter: Callable[..., None] | None = None,
) -> list[float]:
    text_hash = hashlib.sha1(
        text.encode("utf-8", errors="ignore"),
        usedforsecurity=False,
    ).hexdigest()
    key = f"{dims}:{text_hash}"
    cached = _cluster_cache_get(key, increment_counter=increment_counter)
    if cached is not None:
        return list(cached)
    vec = _hash_embedding(text, dims)
    _cluster_cache_put(key, vec)
    return list(vec)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return float(dot)


def _extract_cluster_blocks(
    html_text: str,
    *,
    min_block_chars: int,
    min_word_count: int,
    max_blocks: int,
) -> list[str]:
    if not html_text:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks = [tag.get_text(" ", strip=True) for tag in soup.find_all(["p", "li"])]
    if not blocks:
        raw_text = soup.get_text("\n", strip=True)
        blocks = [line.strip() for line in raw_text.splitlines() if line.strip()]
    filtered = [
        block for block in blocks if len(block) >= min_block_chars and _cluster_word_count(block) >= min_word_count
    ]
    if not filtered and blocks:
        filtered = [max(blocks, key=len)]
    if len(filtered) > max_blocks:
        indexed = list(enumerate(filtered))
        top = sorted(indexed, key=lambda item: len(item[1]), reverse=True)[:max_blocks]
        keep_indexes = {idx for idx, _value in top}
        filtered = [block for idx, block in indexed if idx in keep_indexes]
    return filtered


def _extract_cluster_title(html_text: str) -> Optional[str]:  # noqa: UP045
    if not html_text:
        return None
    soup = BeautifulSoup(html_text, "html.parser")
    title_tag = soup.find("title")
    if not title_tag:
        return None
    title = title_tag.get_text(strip=True)
    return title or None


def _cluster_assignments_hierarchical(
    vectors: list[list[float]],
    *,
    similarity_threshold: float,
    linkage: str,
) -> Optional[list[int]]:  # noqa: UP045
    if not vectors:
        return None
    if len(vectors) == 1:
        return [0]
    try:
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
    except _METRIC_NONCRITICAL_EXCEPTIONS:
        return None
    distance_threshold = max(0.0, 1.0 - similarity_threshold)
    size = len(vectors)
    distances = [[0.0 for _ in range(size)] for _ in range(size)]
    for index in range(size):
        for other_index in range(index + 1, size):
            similarity = _cosine_similarity(vectors[index], vectors[other_index])
            distance = max(0.0, 1.0 - similarity)
            distances[index][other_index] = distance
            distances[other_index][index] = distance
    if linkage not in _CLUSTER_LINKAGES:
        return None
    try:
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage=linkage,
                distance_threshold=distance_threshold,
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage=linkage,
                distance_threshold=distance_threshold,
            )
        labels = clusterer.fit_predict(distances)
    except _METRIC_NONCRITICAL_EXCEPTIONS:
        return None
    return [int(label) for label in labels]


def _build_clusters_from_assignments(
    assignments: list[int],
    items: list[tuple[int, str, list[float], float]],
) -> list[dict[str, Any]]:
    clusters: dict[int, dict[str, Any]] = {}
    for label, item in zip(assignments, items):
        index, block, vector, similarity_to_document = item
        cluster = clusters.get(label)
        if cluster is None:
            cluster = {
                "members": [],
                "sum_vec": [0.0 for _ in vector],
                "centroid": [0.0 for _ in vector],
                "total_chars": 0,
            }
            clusters[label] = cluster
        cluster["members"].append((index, block, similarity_to_document))
        cluster["sum_vec"] = [a + b for a, b in zip(cluster["sum_vec"], vector)]
        cluster["total_chars"] += len(block)
    for cluster in clusters.values():
        cluster["centroid"] = _normalize_vector(cluster["sum_vec"])
    return list(clusters.values())


def _cluster_blocks_greedy(
    items: list[tuple[int, str, list[float], float]],
    *,
    cluster_threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for index, block, vector, similarity_to_document in items:
        best_index = None
        best_similarity = -1.0
        for cluster_index, cluster in enumerate(clusters):
            similarity = _cosine_similarity(vector, cluster["centroid"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = cluster_index
        if best_index is None or best_similarity < cluster_threshold:
            clusters.append(
                {
                    "members": [(index, block, similarity_to_document)],
                    "sum_vec": list(vector),
                    "centroid": list(vector),
                    "total_chars": len(block),
                }
            )
            continue
        cluster = clusters[best_index]
        cluster["members"].append((index, block, similarity_to_document))
        cluster["sum_vec"] = [a + b for a, b in zip(cluster["sum_vec"], vector)]
        cluster["centroid"] = _normalize_vector(cluster["sum_vec"])
        cluster["total_chars"] += len(block)
    return clusters


def _tag_cluster_text(
    text: str,
    *,
    tag_keywords: dict[str, list[str]],
    top_k: int,
) -> tuple[list[str], dict[str, int]]:
    if top_k <= 0 or not text:
        return [], {}
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for tag, keywords in tag_keywords.items():
        if not keywords:
            continue
        score = sum(text_lower.count(str(keyword).lower()) for keyword in keywords if keyword)
        if score > 0:
            scores[tag] = score
    if not scores:
        return [], {}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [tag for tag, _score in ranked[:top_k]], scores


def _extract_cluster_entities_with_dependencies(
    html_text: str,
    url: str,
    *,
    dependencies: ExtractionDependencies,
    cluster_settings: Optional[dict[str, Any]] = None,  # noqa: UP045
) -> dict[str, Any]:
    """Extract the dominant content cluster from article HTML."""
    result: dict[str, Any] = {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "extraction_successful": False,
        "cluster_blocks": [],
        "cluster_block_count": 0,
    }
    dependencies.cancellation_checkpoint()
    if not html_text:
        result["cluster_error"] = "cluster_empty_html"
        return result

    raw_settings = dict(cluster_settings or {})
    env_similarity = _env_float("SIM_THRESHOLD")
    env_min_words = _env_int("WORD_COUNT_THRESHOLD")
    env_linkage = os.getenv("CLUSTER_LINKAGE", "").strip().lower()
    valid_hierarchical_linkage = has_valid_hierarchical_linkage(
        raw_settings,
        env_linkage=env_linkage,
    )
    settings = normalize_cluster_settings(
        raw_settings,
        env_similarity=env_similarity,
        env_min_words=env_min_words,
        env_linkage=env_linkage,
    )
    min_block_chars = settings["min_block_chars"]
    min_word_count = settings["min_word_count"]
    max_blocks = settings["max_blocks"]
    prefilter_threshold = settings["prefilter_threshold"]
    cluster_threshold = settings["cluster_threshold"]
    embed_dims = settings["embed_dims"]
    method = settings["method"]
    linkage = settings["linkage"]
    tag_top_k = settings["tag_top_k"]
    tag_keywords = settings.get("tag_keywords") or _DEFAULT_CLUSTER_TAG_KEYWORDS
    if not isinstance(tag_keywords, dict):
        tag_keywords = _DEFAULT_CLUSTER_TAG_KEYWORDS

    _increment_counter(dependencies, "extraction_cluster_total", labels={"status": "started"})
    blocks = _extract_cluster_blocks(
        html_text,
        min_block_chars=min_block_chars,
        min_word_count=min_word_count,
        max_blocks=max_blocks,
    )
    dependencies.cancellation_checkpoint()
    if not blocks:
        result["cluster_error"] = "cluster_no_blocks"
        _increment_counter(dependencies, "extraction_cluster_total", labels={"status": "no_blocks"})
        return result

    def cache_counter(name: str, *, labels: dict[str, str]) -> None:
        _increment_counter(dependencies, name, labels=labels)

    document_vector = _cluster_embedding(
        " ".join(blocks),
        embed_dims,
        increment_counter=cache_counter,
    )
    dependencies.cancellation_checkpoint()
    scored_blocks: list[tuple[int, str, list[float], float]] = []
    for index, block in enumerate(blocks):
        dependencies.cancellation_checkpoint()
        vector = _cluster_embedding(block, embed_dims, increment_counter=cache_counter)
        dependencies.cancellation_checkpoint()
        similarity = _cosine_similarity(vector, document_vector)
        scored_blocks.append((index, block, vector, similarity))

    kept = [item for item in scored_blocks if item[3] >= prefilter_threshold]
    if not kept:
        kept = sorted(scored_blocks, key=lambda item: item[3], reverse=True)[: min(2, len(scored_blocks))]

    clusters: list[dict[str, Any]] = []
    cluster_method = method
    if method == "hierarchical":
        assignments = (
            _cluster_assignments_hierarchical(
                [item[2] for item in kept],
                similarity_threshold=cluster_threshold,
                linkage=linkage,
            )
            if valid_hierarchical_linkage
            else None
        )
        dependencies.cancellation_checkpoint()
        if assignments and len(assignments) == len(kept):
            clusters = _build_clusters_from_assignments(assignments, kept)
        else:
            cluster_method = "greedy_fallback"
            clusters = _cluster_blocks_greedy(kept, cluster_threshold=cluster_threshold)
    else:
        cluster_method = "greedy"
        clusters = _cluster_blocks_greedy(kept, cluster_threshold=cluster_threshold)
    dependencies.cancellation_checkpoint()

    if not clusters:
        result["cluster_error"] = "cluster_no_clusters"
        _increment_counter(dependencies, "extraction_cluster_total", labels={"status": "no_clusters"})
        return result

    best_cluster = max(
        clusters,
        key=lambda cluster: (int(cluster.get("total_chars", 0)), len(cluster.get("members", []))),
    )
    ordered_members = sorted(best_cluster["members"], key=lambda item: item[0])
    content_blocks = [block for _index, block, _similarity in ordered_members if block]
    content = "\n\n".join(content_blocks).strip()
    if not content:
        result["cluster_error"] = "cluster_empty_content"
        _increment_counter(dependencies, "extraction_cluster_total", labels={"status": "empty"})
        return result

    title = _extract_cluster_title(html_text)
    if title:
        result["title"] = title
    result["content"] = content
    result["cluster_blocks"] = content_blocks
    result["cluster_block_count"] = len(content_blocks)
    result["cluster_prefiltered_count"] = len(kept)
    result["cluster_total_blocks"] = len(blocks)
    result["cluster_cluster_count"] = len(clusters)
    result["cluster_method"] = cluster_method
    if method == "hierarchical" and valid_hierarchical_linkage:
        result["cluster_linkage"] = linkage
    result["cluster_similarity_threshold"] = cluster_threshold
    result["cluster_word_threshold"] = min_word_count
    tags, tag_scores = _tag_cluster_text(content, tag_keywords=tag_keywords, top_k=tag_top_k)
    if tags:
        result["cluster_tags"] = tags
        result["cluster_tag_scores"] = tag_scores
    dependencies.cancellation_checkpoint()
    result["extraction_successful"] = True
    _increment_counter(dependencies, "extraction_cluster_total", labels={"status": "success"})
    return result


def extract_cluster_entities(
    html_text: str,
    url: str,
    *,
    cluster_settings: Optional[dict[str, Any]] = None,  # noqa: UP045
) -> dict[str, Any]:
    """Extract the dominant content cluster from article HTML."""
    return _extract_cluster_entities_with_dependencies(
        html_text,
        url,
        dependencies=build_default_dependencies(),
        cluster_settings=cluster_settings,
    )
