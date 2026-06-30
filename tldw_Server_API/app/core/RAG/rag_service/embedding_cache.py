"""
LRU cache for embeddings with persistence support.

This module provides an efficient caching mechanism for embeddings
to avoid recomputing them for frequently accessed content.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger


class EmbeddingCache:
    """
    LRU cache for embeddings with optional persistence.

    Features:
    - Thread-safe operations
    - LRU eviction policy
    - Configurable size limits
    - Optional disk persistence
    - Cache statistics
    - TTL support
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 1024,  # 1GB
        ttl_seconds: Optional[float] = 86400,  # 24 hours
        persist_path: Optional[str] = None,
        enable_stats: bool = True
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
            persist_path: Optional path for persistent cache storage
            enable_stats: Whether to track cache statistics
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path
        self.enable_stats = enable_stats

        # Cache storage (OrderedDict maintains insertion order for LRU)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

        # Memory tracking
        self._current_memory = 0

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "ttl_expirations": 0,
            "memory_evictions": 0,
            "total_compute_time": 0.0,
            "total_cache_time": 0.0
        }

        # Load persistent cache if available
        if persist_path:
            self._load_cache()

    def _generate_key(self, text: str, model_name: Optional[str] = None) -> str:
        """
        Generate a cache key for text.

        Args:
            text: Text to generate key for
            model_name: Optional model name to include in key

        Returns:
            Cache key
        """
        key_parts = [text]
        if model_name:
            key_parts.append(model_name)

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _estimate_memory(self, embedding: np.ndarray) -> int:
        """
        Estimate memory usage of an embedding.

        Args:
            embedding: Numpy array embedding

        Returns:
            Estimated memory in bytes
        """
        if isinstance(embedding, np.ndarray):
            return embedding.nbytes + 200  # Array size + overhead
        elif isinstance(embedding, list):
            return len(embedding) * 8 + 200  # Assume float64 + overhead
        else:
            return 1000  # Default estimate

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """
        Check if a cache entry has expired.

        Args:
            entry: Cache entry

        Returns:
            True if expired
        """
        if self.ttl_seconds is None:
            return False

        age = time.time() - float(entry["timestamp"])
        ttl_seconds = float(self.ttl_seconds)
        return age > ttl_seconds

    def _evict_lru(self) -> Optional[str]:
        """
        Evict least recently used entry.

        Returns:
            Key of evicted entry
        """
        if not self._cache:
            return None

        # Get least recently used (first item in OrderedDict)
        lru_key = next(iter(self._cache))
        entry = self._cache.pop(lru_key)

        # Update memory tracking
        self._current_memory -= entry.get("memory_size", 0)

        if self.enable_stats:
            self._stats["evictions"] += 1

        logger.debug(f"Evicted LRU entry: {lru_key[:8]}...")
        return lru_key

    def _evict_for_memory(self, required_memory: int):
        """
        Evict entries to free up required memory.

        Args:
            required_memory: Memory needed in bytes
        """
        while self._current_memory + required_memory > self.max_memory_bytes:
            if not self._cache:
                break

            self._evict_lru()

            if self.enable_stats:
                self._stats["memory_evictions"] += 1

    def get(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            text: Text to get embedding for
            model_name: Optional model name

        Returns:
            Cached embedding or None if not found
        """
        key = self._generate_key(text, model_name)

        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if self._is_expired(entry):
                    self._cache.pop(key)
                    self._current_memory -= entry.get("memory_size", 0)

                    if self.enable_stats:
                        self._stats["ttl_expirations"] += 1
                        self._stats["misses"] += 1

                    return None

                # Move to end (most recently used)
                self._cache.move_to_end(key)

                # Update access time
                entry["last_access"] = time.time()
                entry["access_count"] += 1

                if self.enable_stats:
                    self._stats["hits"] += 1

                embedding = entry.get("embedding")
                if isinstance(embedding, np.ndarray):
                    return embedding
                return None

            if self.enable_stats:
                self._stats["misses"] += 1

            return None

    def put(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ):
        """
        Store embedding in cache.

        Args:
            text: Text the embedding represents
            embedding: Embedding vector
            model_name: Optional model name
            metadata: Optional metadata to store
        """
        key = self._generate_key(text, model_name)
        memory_size = self._estimate_memory(embedding)

        with self._lock:
            # Check if we need to evict for size
            while len(self._cache) >= self.max_size:
                self._evict_lru()

            # Check if we need to evict for memory
            self._evict_for_memory(memory_size)

            # Store entry
            self._cache[key] = {
                "embedding": embedding,
                "text_hash": hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(),
                "timestamp": time.time(),
                "last_access": time.time(),
                "access_count": 0,
                "memory_size": memory_size,
                "model_name": model_name,
                "metadata": metadata or {}
            }

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Update memory tracking
            self._current_memory += memory_size

    def batch_get(
        self,
        texts: list[str],
        model_name: Optional[str] = None
    ) -> tuple[list[Optional[np.ndarray]], list[int]]:
        """
        Get multiple embeddings from cache.

        Args:
            texts: List of texts to get embeddings for
            model_name: Optional model name

        Returns:
            Tuple of (embeddings, missing_indices)
            where missing_indices are indices of texts not in cache
        """
        embeddings = []
        missing_indices = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model_name)
            embeddings.append(embedding)
            if embedding is None:
                missing_indices.append(i)

        return embeddings, missing_indices

    def batch_put(
        self,
        texts: list[str],
        embeddings: list[np.ndarray],
        model_name: Optional[str] = None
    ):
        """
        Store multiple embeddings in cache.

        Args:
            texts: List of texts
            embeddings: List of embeddings
            model_name: Optional model name
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding, model_name)

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0

            if self.enable_stats:
                logger.info(f"Cache cleared. Stats: {self.get_stats()}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "ttl_expirations": self._stats["ttl_expirations"],
                "memory_evictions": self._stats["memory_evictions"]
            }

    def _save_cache(self):
        """Save cache to disk."""
        if not self.persist_path:
            return

        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                # Convert numpy arrays to lists for JSON serialization
                cache_data = {}
                for key, entry in self._cache.items():
                    cache_entry = entry.copy()
                    if isinstance(cache_entry["embedding"], np.ndarray):
                        cache_entry["embedding"] = cache_entry["embedding"].tolist()
                    cache_data[key] = cache_entry

                # Save as JSON
                with open(path, 'w') as f:
                    json.dump({
                        "cache": cache_data,
                        "stats": self._stats,
                        "memory": self._current_memory
                    }, f)

                logger.info(f"Saved cache to {self.persist_path} ({len(self._cache)} entries)")

        except Exception as e:
            logger.error(f"Failed to save cache: {type(e).__name__}")

    def _load_cache(self):
        """Load cache from disk."""
        if not self.persist_path:
            return

        try:
            path = Path(self.persist_path)
            if not path.exists():
                return

            with open(path) as f:
                data = json.load(f)

            with self._lock:
                # Restore cache entries
                for key, entry in data.get("cache", {}).items():
                    # Convert lists back to numpy arrays
                    if isinstance(entry["embedding"], list):
                        entry["embedding"] = np.array(entry["embedding"])

                    # Skip expired entries
                    if not self._is_expired(entry):
                        self._cache[key] = entry

                # Restore stats
                if "stats" in data:
                    self._stats.update(data["stats"])

                # Restore memory tracking
                self._current_memory = data.get("memory", 0)

                logger.info(f"Loaded cache from {self.persist_path} ({len(self._cache)} entries)")

        except Exception as e:
            logger.error(f"Failed to load cache: {type(e).__name__}")

    def save(self):
        """Manually save cache to disk."""
        self._save_cache()

    def __del__(self):
        """Save cache on deletion."""
        if self.persist_path:
            self._save_cache()


class EmbeddingCacheManager:
    """
    Manages multiple embedding caches for different models/purposes.
    """

    def __init__(self, default_config: Optional[dict[str, Any]] = None):
        """
        Initialize cache manager.

        Args:
            default_config: Default configuration for new caches
        """
        self._caches: dict[str, EmbeddingCache] = {}
        self._lock = threading.RLock()
        self._default_config = default_config or {
            "max_size": 10000,
            "max_memory_mb": 1024,
            "ttl_seconds": 86400,
            "enable_stats": True
        }

    def get_cache(self, cache_name: str = "default", **config) -> EmbeddingCache:
        """
        Get or create a cache.

        Args:
            cache_name: Name of the cache
            **config: Optional configuration overrides

        Returns:
            EmbeddingCache instance
        """
        with self._lock:
            if cache_name not in self._caches:
                # Merge configurations
                cache_config = self._default_config.copy()
                cache_config.update(config)

                # Add persistence path if not specified
                if "persist_path" not in cache_config:
                    cache_config["persist_path"] = f"cache/embeddings_{cache_name}.json"

                self._caches[cache_name] = EmbeddingCache(**cache_config)
                logger.info(f"Created embedding cache: {cache_name}")

            return self._caches[cache_name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary mapping cache names to statistics
        """
        with self._lock:
            return {
                name: cache.get_stats()
                for name, cache in self._caches.items()
            }

    def clear_all(self):
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()

    def save_all(self):
        """Save all caches to disk."""
        with self._lock:
            for cache in self._caches.values():
                cache.save()


# Global cache manager
_global_cache_manager: Optional[EmbeddingCacheManager] = None
_manager_lock = threading.Lock()


def get_global_cache_manager() -> EmbeddingCacheManager:
    """
    Get or create the global embedding cache manager.

    Returns:
        Global EmbeddingCacheManager instance
    """
    global _global_cache_manager

    with _manager_lock:
        if _global_cache_manager is None:
            _global_cache_manager = EmbeddingCacheManager()
        return _global_cache_manager
