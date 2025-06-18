"""
Cache implementations for the RAG service.
"""

import time
import json
from typing import Dict, Optional, Any, Tuple
from collections import OrderedDict
import threading
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    value: Any
    timestamp: float
    ttl: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    
    This cache automatically evicts least recently used items when
    the cache reaches its maximum size.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = no expiration)
        """
        with self._lock:
            # Remove if already exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl
            )
            
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def delete(self, key: str) -> None:
        """Delete an item from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """Check if an object can be safely serialized to JSON."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False


class MultiLevelCache:
    """
    Multi-level cache with different TTLs for different data types.
    
    This allows for more granular caching strategies where embeddings
    might be cached longer than search results.
    """
    
    def __init__(self, config: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-level cache.
        
        Args:
            config: Configuration for each cache level
                   e.g., {"embeddings": {"max_size": 10000, "ttl": 3600},
                          "search": {"max_size": 1000, "ttl": 300}}
        """
        self._caches: Dict[str, LRUCache] = {}
        self._config = config
        
        for level, level_config in config.items():
            self._caches[level] = LRUCache(
                max_size=level_config.get("max_size", 1000)
            )
    
    def get(self, level: str, key: str) -> Optional[Any]:
        """Get from a specific cache level."""
        if level not in self._caches:
            return None
        return self._caches[level].get(key)
    
    def set(self, level: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in a specific cache level."""
        if level not in self._caches:
            logger.warning(f"Unknown cache level: {level}")
            return
        
        # Use level-specific TTL if not provided
        if ttl is None and level in self._config:
            ttl = self._config[level].get("ttl")
        
        self._caches[level].set(key, value, ttl)
    
    def delete(self, level: str, key: str) -> None:
        """Delete from a specific cache level."""
        if level in self._caches:
            self._caches[level].delete(key)
    
    def clear(self, level: Optional[str] = None) -> None:
        """Clear a specific level or all levels."""
        if level:
            if level in self._caches:
                self._caches[level].clear()
        else:
            for cache in self._caches.values():
                cache.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all cache levels."""
        return {
            level: cache.get_stats()
            for level, cache in self._caches.items()
        }
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries in all levels."""
        return {
            level: cache.cleanup_expired()
            for level, cache in self._caches.items()
        }


class PersistentCache(LRUCache):
    """
    LRU cache with optional persistence to disk.
    
    This cache can save its state to disk and restore it on startup,
    useful for caching expensive computations like embeddings.
    """
    
    def __init__(self, max_size: int = 1000, persist_path: Optional[str] = None):
        """
        Initialize persistent cache.
        
        Args:
            max_size: Maximum number of items
            persist_path: Path to save cache state
        """
        super().__init__(max_size)
        self.persist_path = persist_path
        
        # Try to load existing cache
        if persist_path:
            self.load()
    
    def save(self) -> None:
        """Save cache state to disk using JSON."""
        if not self.persist_path:
            return
        
        try:
            with self._lock:
                # Convert cache entries to serializable format
                serializable_cache = {}
                for key, entry in self._cache.items():
                    # Only serialize basic types that are JSON-safe
                    if self._is_json_serializable(entry.value):
                        serializable_cache[key] = {
                            "value": entry.value,
                            "timestamp": entry.timestamp,
                            "ttl": entry.ttl
                        }
                
                state = {
                    "cache": serializable_cache,
                    "stats": {
                        "hits": self._hits,
                        "misses": self._misses
                    }
                }
                
                with open(self.persist_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved cache state to {self.persist_path} ({len(serializable_cache)} entries)")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load(self) -> None:
        """Load cache state from disk using JSON."""
        if not self.persist_path:
            return
        
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            with self._lock:
                # Clear expired entries before loading
                loaded_cache = state.get("cache", {})
                for key, entry_data in loaded_cache.items():
                    # Reconstruct CacheEntry from JSON data
                    entry = CacheEntry(
                        value=entry_data["value"],
                        timestamp=entry_data["timestamp"],
                        ttl=entry_data.get("ttl")
                    )
                    if not entry.is_expired():
                        self._cache[key] = entry
                
                # Restore stats
                stats = state.get("stats", {})
                self._hits = stats.get("hits", 0)
                self._misses = stats.get("misses", 0)
            
            logger.info(f"Loaded cache state from {self.persist_path} ({len(self._cache)} entries)")
        except FileNotFoundError:
            logger.debug(f"No cache file found at {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")