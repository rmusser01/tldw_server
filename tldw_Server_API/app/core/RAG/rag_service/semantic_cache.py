"""
Semantic caching implementation for RAG service.

This module provides semantic similarity-based caching that can find
cached results for semantically similar queries, not just exact matches.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
from loguru import logger

from .advanced_cache import CacheEntry


@dataclass
class SemanticCacheEntry(CacheEntry):
    """Extended cache entry with semantic information.

    Inherits timing and access fields from CacheEntry (created_at, last_accessed,
    access_count, ttl). Adds query text and optional normalized embedding.
    """
    query: str = ""
    embedding: Optional[np.ndarray] = None

    def access(self) -> None:
        """Compatibility helper used by older code paths."""
        self.update_access()


class SemanticCache:
    """
    Semantic similarity-based cache for RAG queries.

    This cache can:
    - Find cached results for semantically similar queries
    - Use embeddings to measure query similarity
    - Fall back to exact matching when embeddings unavailable
    - Track usage statistics for cache optimization
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        ttl: Optional[int] = 3600,
        persist_path: Optional[str] = None,
        embedding_model: Optional[Any] = None,
        namespace: Optional[str] = None,
    ):
        """
        Initialize semantic cache.

        Args:
            max_size: Maximum number of cached items
            similarity_threshold: Minimum similarity for semantic match (0-1)
            ttl: Default time-to-live in seconds
            persist_path: Optional path for cache persistence
            embedding_model: Optional model for generating embeddings
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = ttl
        # Optional logical namespace for multi-tenant environments.
        # When provided, it is used to prefix persisted state and metrics so
        # cache entries from different tenants/users do not collide.
        self.namespace = (str(namespace).strip() or None) if namespace is not None else None
        self.persist_path = persist_path
        self.embedding_model = embedding_model

        # Main cache storage
        self._cache: dict[str, SemanticCacheEntry] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._exact_hits = 0
        self._evictions = 0
        self._warned_no_embedding = False
        self._metrics_available: Optional[bool] = None

        # Load persisted cache if available
        if persist_path:
            self.load()

    def _generate_key(self, query: str) -> str:
        """Generate a cache key from query."""
        return hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()

    def _emit_counter(self, metric_name: str) -> None:
        """Emit a counter increment to Prometheus/OTEL if available."""
        if self._metrics_available is False:
            return
        try:
            from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
            cache_type = "semantic" if self.embedding_model else "exact"
            increment_counter(metric_name, labels={"cache_type": cache_type})
            self._metrics_available = True
        except (ImportError, AttributeError, TypeError):
            self._metrics_available = False

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if unavailable
        """
        if not self.embedding_model:
            logger.debug("No embedding model configured for semantic cache")
            return None

        try:
            # Check if embedding_model has the expected interface
            if hasattr(self.embedding_model, 'encode'):
                # Sentence transformers style
                embedding = self.embedding_model.encode(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            elif hasattr(self.embedding_model, 'embed'):
                # Generic embedding interface
                embedding = await self.embedding_model.embed(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            elif callable(self.embedding_model):
                # Function-based embedding model
                embedding = await self.embedding_model(text) if asyncio.iscoroutinefunction(self.embedding_model) else self.embedding_model(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            else:
                logger.warning(f"Embedding model does not have expected interface: {type(self.embedding_model)}")
                return None

            embedding_array = np.asarray(embedding)
            # Normalize the embedding
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

            return embedding_array
        except (AttributeError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError):
            logger.error("Failed to generate embedding")
            return None

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def find_similar(self, query: str, embedding: Optional[np.ndarray] = None) -> Optional[tuple[str, str, float]]:
        """
        Find most similar cached query.

        Args:
            query: Query to match
            embedding: Pre-computed embedding (optional)

        Returns:
            Tuple of (cache_key, cached_query, similarity_score) or None
        """
        if embedding is None:
            embedding = await self.get_embedding(query)

        if embedding is None or not self._embeddings:
            return None

        with self._lock:
            best_key = None
            best_similarity = 0.0

            for key, cached_embedding in self._embeddings.items():
                similarity = self._compute_similarity(embedding, cached_embedding)

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    # Check if entry is still valid
                    if key in self._cache and not self._cache[key].is_expired():
                        best_similarity = similarity
                        best_key = key

            if best_key:
                entry = self._cache.get(best_key)
                cached_query = getattr(entry, "query", None) if entry else None
                if cached_query:
                    logger.debug(f"Found similar query with similarity {best_similarity:.3f}")
                    return best_key, cached_query, best_similarity

        return None

    async def get(self, query: str, use_semantic: bool = True) -> Optional[Any]:
        """
        Get cached result for query.

        Args:
            query: Query string
            use_semantic: Whether to use semantic matching

        Returns:
            Cached value or None
        """
        key = self._generate_key(query)

        with self._lock:
            # Try exact match first
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    entry.access()
                    self._hits += 1
                    self._exact_hits += 1
                    self._emit_counter("rag_cache_hits_total")
                    # Avoid logging raw query text; use hash + length for traceback without leakage
                    try:
                        logger.debug(f"Exact cache hit for key={key} (len={len(query)})")
                    except (AttributeError, TypeError, ValueError):
                        logger.debug("Exact cache hit")
                    return entry.value
                else:
                    # Remove expired entry
                    del self._cache[key]
                    if key in self._embeddings:
                        del self._embeddings[key]

        # Try semantic match if enabled
        if use_semantic and not self.embedding_model:
            # Warn once that semantic matching is not active
            if not self._warned_no_embedding:
                logger.info("Semantic cache running in exact-match mode (no embedding model configured)")
                self._warned_no_embedding = True
        if use_semantic and self.embedding_model:
            embedding = await self.get_embedding(query)
            if embedding is not None:
                similar_result = await self.find_similar(query, embedding)

                if similar_result:
                    similar_key, cached_query, similarity = similar_result
                    with self._lock:
                        if similar_key in self._cache:
                            entry = self._cache[similar_key]
                            entry.access()
                            self._hits += 1
                            self._semantic_hits += 1
                            self._emit_counter("rag_cache_hits_total")
                            # Log similar cache key and similarity without raw query
                            logger.info(
                                f"Semantic cache hit (similarity={similarity:.3f}) for key={similar_key}"
                            )
                            return entry.value

        with self._lock:
            self._misses += 1
            self._emit_counter("rag_cache_misses_total")
        return None

    async def set(
        self,
        query: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Cache a query result.

        Args:
            query: Query string
            value: Value to cache
            ttl: Time-to-live in seconds
            metadata: Optional metadata to store
        """
        key = self._generate_key(query)

        # Generate embedding for semantic matching
        embedding = None
        if self.embedding_model:
            embedding = await self.get_embedding(query)

        now = time.time()
        with self._lock:
            # Create cache entry aligned with CacheEntry fields
            entry = SemanticCacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                ttl=ttl or self.default_ttl,
                query=query,
                embedding=embedding,
            )

            # Store entry
            self._cache[key] = entry
            if embedding is not None:
                self._embeddings[key] = embedding

            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                self._evict_lru()

            try:
                logger.debug(f"Cached result for key={key} (len={len(query)})")
            except (AttributeError, TypeError, ValueError):
                logger.debug("Cached result")

    @property
    def hit_rate(self) -> float:
        """Compute cache hit rate as hits / (hits + misses)."""
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

    @property
    def evictions(self) -> int:
        """Total number of evictions that have occurred."""
        return self._evictions

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )

        # Remove from cache
        del self._cache[lru_key]
        if lru_key in self._embeddings:
            del self._embeddings[lru_key]

        self._evictions += 1
        self._emit_counter("rag_cache_evictions_total")
        logger.debug(f"Evicted LRU cache entry: {lru_key}")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0
            self._exact_hits = 0
            self._evictions = 0
            logger.info("Semantic cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns a dict suitable for JSON serialization and Prometheus metrics
        export. Includes hits, misses, evictions, hit_rate, and more.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            semantic_hit_rate = self._semantic_hits / self._hits if self._hits > 0 else 0

            stats: dict[str, Any] = {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_hits": self._hits,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "semantic_hit_rate": semantic_hit_rate,
                "total_requests": total_requests,
                "similarity_threshold": self.similarity_threshold,
                "has_embeddings": len(self._embeddings),
            }
            if self.namespace is not None:
                stats["namespace"] = self.namespace
            return stats

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]
                if key in self._embeddings:
                    del self._embeddings[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def save(self) -> None:
        """Save cache state to disk."""
        if not self.persist_path:
            return

        try:
            with self._lock:
                # Prepare serializable state
                state: dict[str, Any] = {
                    "cache": {},
                    "stats": {
                        "hits": self._hits,
                        "misses": self._misses,
                        "semantic_hits": self._semantic_hits,
                        "exact_hits": self._exact_hits,
                        "evictions": self._evictions
                    },
                    "config": {
                        "similarity_threshold": self.similarity_threshold,
                        "max_size": self.max_size,
                        "default_ttl": self.default_ttl
                    }
                }

                # Convert cache entries to serializable format
                for key, entry in self._cache.items():
                    state["cache"][key] = {
                        "value": entry.value,  # Note: Complex objects may need special handling
                        "query": entry.query,
                        "timestamp": entry.created_at,
                        "ttl": entry.ttl,
                        "access_count": entry.access_count,
                        "last_access": entry.last_accessed,
                    }

                # Save embeddings separately as binary
                embeddings_path = Path(self.persist_path).with_suffix('.embeddings.npz')
                if self._embeddings:
                    savez = cast(Any, np.savez_compressed)
                    savez(embeddings_path, **self._embeddings)

                # Save main state as JSON
                with open(self.persist_path, 'w') as f:
                    json.dump(state, f, indent=2)

                logger.info(f"Saved semantic cache state ({len(self._cache)} entries)")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to save semantic cache: {type(e).__name__}")

    def load(self) -> None:
        """Load cache state from disk."""
        if not self.persist_path:
            return

        try:
            # Load main state
            with open(self.persist_path) as f:
                state = json.load(f)

            with self._lock:
                # Restore configuration
                config = state.get("config", {})
                self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)

                # Restore cache entries
                for key, entry_data in state.get("cache", {}).items():
                    created_at = entry_data.get("created_at", entry_data.get("timestamp", time.time()))
                    last_accessed = entry_data.get("last_accessed", entry_data.get("last_access", created_at))
                    entry = SemanticCacheEntry(
                        value=entry_data["value"],
                        created_at=created_at,
                        last_accessed=last_accessed,
                        ttl=entry_data.get("ttl"),
                        query=entry_data.get("query", ""),
                    )

                    if not entry.is_expired():
                        self._cache[key] = entry

                # Load embeddings
                embeddings_path = Path(self.persist_path).with_suffix('.embeddings.npz')
                if embeddings_path.exists():
                    embeddings_data = np.load(embeddings_path)
                    self._embeddings = {
                        key: embeddings_data[key]
                        for key in embeddings_data.files
                        if key in self._cache  # Only load embeddings for valid cache entries
                    }

                # Restore statistics
                stats = state.get("stats", {})
                self._hits = stats.get("hits", 0)
                self._misses = stats.get("misses", 0)
                self._semantic_hits = stats.get("semantic_hits", 0)
                self._exact_hits = stats.get("exact_hits", 0)
                self._evictions = stats.get("evictions", 0)

                logger.info(f"Loaded semantic cache ({len(self._cache)} entries, {len(self._embeddings)} embeddings)")
        except FileNotFoundError:
            logger.debug(f"No cache file found at {self.persist_path}")
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to load semantic cache: {type(e).__name__}")


class AdaptiveCache(SemanticCache):
    """
    Adaptive cache that adjusts its behavior based on usage patterns.

    Features:
    - Dynamic similarity threshold adjustment
    - Automatic TTL optimization
    - Pattern-based prefetching hints
    """

    def __init__(self, *args, **kwargs):
        """Initialize adaptive cache with pattern tracking."""
        super().__init__(*args, **kwargs)

        # Pattern tracking
        self._query_patterns: dict[str, int] = {}
        self._similarity_scores: list[float] = []
        self._ttl_effectiveness: dict[str, float] = {}

        # Adaptive parameters
        self._adaptive_threshold = self.similarity_threshold
        self._adaptive_ttl = self.default_ttl

    async def get(self, query: str, use_semantic: bool = True) -> Optional[Any]:
        """Get with pattern tracking."""
        result = await super().get(query, use_semantic)

        # Track query patterns
        self._track_pattern(query)

        # Adjust parameters periodically
        if self._hits + self._misses > 0 and (self._hits + self._misses) % 100 == 0:
            self._adjust_parameters()

        return result

    def _track_pattern(self, query: str) -> None:
        """Track query patterns for optimization."""
        # Extract pattern (simplified - in production use NLP)
        words = query.lower().split()[:3]  # First 3 words as pattern
        pattern = " ".join(words)

        with self._lock:
            self._query_patterns[pattern] = self._query_patterns.get(pattern, 0) + 1

    def _adjust_parameters(self) -> None:
        """Adjust cache parameters based on usage patterns."""
        with self._lock:
            # Adjust similarity threshold based on hit rate
            hit_rate = self._hits / (self._hits + self._misses)

            if hit_rate < 0.3 and self._adaptive_threshold > 0.7:
                # Low hit rate - be more lenient with similarity
                self._adaptive_threshold -= 0.05
                self.similarity_threshold = self._adaptive_threshold
                logger.info(f"Lowered similarity threshold to {self._adaptive_threshold:.2f}")
            elif hit_rate > 0.7 and self._adaptive_threshold < 0.95:
                # High hit rate - can be more strict
                self._adaptive_threshold += 0.05
                self.similarity_threshold = self._adaptive_threshold
                logger.info(f"Raised similarity threshold to {self._adaptive_threshold:.2f}")

    def get_patterns(self) -> list[tuple[str, int]]:
        """Get most common query patterns."""
        with self._lock:
            return sorted(
                self._query_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

    def suggest_prefetch(self) -> list[str]:
        """Suggest queries to prefetch based on patterns."""
        patterns = self.get_patterns()
        suggestions = []

        for pattern, count in patterns[:5]:
            if count > 10:  # Only suggest frequently used patterns
                suggestions.append(pattern)

        return suggestions


_SHARED_CACHE_LOCK = threading.Lock()
_SHARED_CACHES: dict[tuple[type[SemanticCache], str, float, Optional[int], Optional[int], Optional[str]], SemanticCache] = {}
_DEFAULT_CACHE_DIR: Optional[Path] = None


def _normalize_namespace(namespace: Optional[str]) -> str:
    raw = str(namespace).strip() if namespace is not None else ""
    if not raw:
        return "default"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def _normalize_namespace_key_for_filename(namespace_key: str, max_length: int = 64) -> str:
    """
    Ensure the namespace key used in filenames is safely normalized and bounded.

    This protects against unexpected characters and excessively long filenames,
    even if callers pass an untrusted or unnormalized value.
    """
    # Reuse the existing namespace normalization to enforce the allowed character set.
    normalized = _normalize_namespace(namespace_key)
    # Truncate to keep filenames reasonably small and avoid filesystem issues.
    if len(normalized) > max_length:
        normalized = normalized[:max_length]
    # As a final safeguard, ensure we never return an empty string.
    return normalized or "default"


def _resolve_default_cache_dir() -> Optional[Path]:
    global _DEFAULT_CACHE_DIR
    if _DEFAULT_CACHE_DIR is not None:
        return _DEFAULT_CACHE_DIR
    base_dir = os.getenv("RAG_SEMANTIC_CACHE_DIR") or os.getenv("RAG_CACHE_DIR")
    if base_dir:
        # Resolve to an absolute path so downstream checks can reliably enforce containment.
        _DEFAULT_CACHE_DIR = Path(base_dir).expanduser().resolve(strict=False)
        return _DEFAULT_CACHE_DIR
    project_root = None
    try:
        from tldw_Server_API.app.core.config import load_and_log_configs
        cfg = load_and_log_configs() or {}
        project_root = cfg.get("PROJECT_ROOT")
    except (ImportError, AttributeError, KeyError, TypeError) as exc:
        logger.warning("Semantic cache: could not load config for PROJECT_ROOT: {}", type(exc).__name__)
        return None
    if project_root:
        try:
            # Use a fixed subdirectory under the project root for cache storage.
            _DEFAULT_CACHE_DIR = (Path(project_root) / "Databases" / "cache").expanduser().resolve(strict=False)
            return _DEFAULT_CACHE_DIR
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error(
                "Semantic cache: failed to resolve cache path from PROJECT_ROOT: {}",
                type(exc).__name__,
            )
            return None
        except Exception as exc:
            logger.error(
                "Semantic cache: unexpected error resolving cache path from PROJECT_ROOT: {}",
                type(exc).__name__,
            )
            raise
    return None


def _default_persist_path(namespace_key: str) -> Optional[str]:
    base_dir = _resolve_default_cache_dir()
    if not base_dir:
        return None
    try:
        base_dir_resolved = base_dir.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error(
            "Semantic cache: failed to resolve base dir for default persist path: {}",
            type(exc).__name__,
        )
        return None
    except Exception as exc:
        logger.error(
            "Semantic cache: unexpected error resolving base dir for default persist path: {}",
            type(exc).__name__,
        )
        raise
    # Normalize and bound the namespace key before embedding it in the filename.
    safe_key = _normalize_namespace_key_for_filename(namespace_key)
    candidate = base_dir_resolved / f"semantic_cache_{safe_key}.json"
    try:
        full_path = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error(
            "Semantic cache: failed to resolve default persist path: {}",
            type(exc).__name__,
        )
        return None
    except Exception as exc:
        logger.error(
            "Semantic cache: unexpected error resolving default persist path: {}",
            type(exc).__name__,
        )
        raise
    # Verify that the final path is contained within the base cache directory.
    try:
        if not (full_path == base_dir_resolved or base_dir_resolved in full_path.parents):
            logger.warning("Refusing to use out-of-root semantic cache path.")
            return None
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error(
            "Semantic cache: failed to validate cache path against base dir: {}",
            type(exc).__name__,
        )
        return None
    except Exception as exc:
        logger.error(
            "Semantic cache: unexpected error validating cache path against base dir: {}",
            type(exc).__name__,
        )
        raise
    return str(full_path)


def _sanitize_persist_path(persist_path: Optional[str], namespace_key: str) -> Optional[str]:
    """Normalize persist_path and ensure it stays under the cache base directory.

    If the base cache directory cannot be determined or resolved, we fall back to
    a default path derived from the namespace, or disable persistence entirely.
    """
    if not persist_path:
        return None
    base_dir = _resolve_default_cache_dir()
    if not base_dir:
        # Without a trusted base directory, do not trust an arbitrary persist_path.
        fallback = _default_persist_path(namespace_key)
        if fallback:
            logger.warning("No base cache dir for semantic cache; using default cache path.")
            return fallback
        logger.warning("No base cache dir for semantic cache; persistence disabled.")
        return None
    try:
        base_dir_resolved = base_dir.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error(
            "Semantic cache: failed to resolve base cache dir in sanitize: {}",
            type(exc).__name__,
        )
        # If we cannot resolve the base directory safely, use the default path or disable.
        fallback = _default_persist_path(namespace_key)
        if fallback:
            logger.warning("Failed to resolve base cache dir; using default cache path.")
            return fallback
        logger.warning("Failed to resolve base cache dir; persistence disabled.")
        return None
    except Exception as exc:
        logger.error(
            "Semantic cache: unexpected error resolving base cache dir in sanitize: {}",
            type(exc).__name__,
        )
        raise
    candidate_path = Path(persist_path).expanduser()
    try:
        if candidate_path.is_absolute():
            resolved_path = candidate_path.resolve(strict=False)
        else:
            resolved_path = (base_dir_resolved / candidate_path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Semantic cache: failed to resolve persist_path: {}", type(exc).__name__)
        logger.warning("Failed to resolve semantic cache persist_path; using default cache path.")
        return _default_persist_path(namespace_key)
    except Exception as exc:
        logger.error(
            "Semantic cache: unexpected error resolving persist_path: {}",
            type(exc).__name__,
        )
        raise
    if not resolved_path.is_relative_to(base_dir_resolved):
        fallback = _default_persist_path(namespace_key)
        if fallback:
            logger.warning("Rejected semantic cache persist_path outside base dir; using default cache path.")
            return fallback
        logger.warning("Rejected semantic cache persist_path outside base dir; persistence disabled.")
        return None
    return str(resolved_path)


def get_shared_cache(
    cache_cls: type[SemanticCache],
    *,
    similarity_threshold: float,
    ttl: Optional[int],
    max_size: Optional[int],
    persist_path: Optional[str] = None,
    embedding_model: Optional[Any] = None,
    namespace: Optional[str] = None,
) -> SemanticCache:
    namespace_key = _normalize_namespace(namespace)
    persist_path = persist_path or _default_persist_path(namespace_key)
    persist_path = _sanitize_persist_path(persist_path, namespace_key)
    if persist_path:
        with contextlib.suppress(OSError):
            Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
    ttl_key = int(ttl) if ttl is not None else None
    max_size_key = int(max_size) if max_size is not None else None
    cache_key = (cache_cls, namespace_key, float(similarity_threshold), ttl_key, max_size_key, persist_path)
    with _SHARED_CACHE_LOCK:
        cache = _SHARED_CACHES.get(cache_key)
        if cache is None:
            try:
                cache = cache_cls(
                    max_size=max_size or 1000,
                    similarity_threshold=similarity_threshold,
                    ttl=ttl,
                    persist_path=persist_path,
                    embedding_model=embedding_model,
                    namespace=namespace,
                )
            except TypeError:
                cache = cache_cls(similarity_threshold=similarity_threshold)
            _SHARED_CACHES[cache_key] = cache
    return cache


def clear_shared_caches(namespace: Optional[str] = None) -> int:
    """Clear shared semantic caches, optionally scoped to a namespace."""
    cleared = 0
    namespace_key = _normalize_namespace(namespace) if namespace is not None else None
    with _SHARED_CACHE_LOCK:
        for cache_key, cache in list(_SHARED_CACHES.items()):
            _, key_namespace, *_ = cache_key
            if namespace_key is None or key_namespace == namespace_key:
                with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                    cache.clear()
                cleared += 1
    return cleared
