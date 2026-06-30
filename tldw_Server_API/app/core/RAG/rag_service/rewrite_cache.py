"""
Lightweight persistent cache for query → effective rewrite(s).

Stores per-corpus (index_namespace) rewrite candidates keyed by a stable
"intent cluster" to encourage reuse with decay. Designed to have minimal
runtime overhead and zero external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def _exception_type_name(exc: BaseException) -> str:
    return type(exc).__name__


def _safe_path(user_id: str | None) -> Path:
    try:
        base = os.getenv("RAG_REWRITE_CACHE_PATH")
        if base:
            p = Path(base)
            p.parent.mkdir(parents=True, exist_ok=True)
            path = p
        else:
            path = DatabasePaths.get_user_rewrite_cache_path(user_id)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.error("Rewrite cache: failed to resolve cache path: {}", _exception_type_name(exc))
        raise
    else:
        return path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_base = base.resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        return resolved_path.is_relative_to(resolved_base)
    except AttributeError:
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            return False
        else:
            return True


def _normalize_query(q: str) -> str:
    try:
        q = (q or "").strip().lower()
        # Remove punctuation and collapse spaces
        out = []
        prev_space = False
        for ch in q:
            if ch.isalnum():
                out.append(ch)
                prev_space = False
            else:
                if not prev_space:
                    out.append(" ")
                    prev_space = True
        return "".join(out).strip()
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Rewrite cache: failed to normalize query; returning fallback: {}", _exception_type_name(exc))
        return q or ""


def _cluster_id(query: str, intent: str | None = None, corpus: str | None = None) -> str:
    base = _normalize_query(query)
    key = f"{intent or 'unknown'}|{corpus or 'default'}|{base}"
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass
class RewriteEntry:
    cluster_id: str
    corpus: str | None
    intent: str | None
    rewrites: list[str]
    weight: float = 1.0
    last_used: float = 0.0
    created_at: float = 0.0


class RewriteCache:
    """Append-only JSONL cache with in-memory index."""

    def __init__(
        self,
        path: str | None = None,
        half_life_hours: float = 72.0,
        user_id: str | None = None,
        ttl_hours: float | None = None,
    ) -> None:
        # Determine storage path (per-user if provided)
        if path is None:
            self.path = str(_safe_path(user_id))
        else:
            self.path = str(Path(path))
        # Decay settings
        self.half_life = max(1.0, float(half_life_hours))
        # TTL expiry for cache entries (hard expiration)
        try:
            _ttl_env = os.getenv("RAG_REWRITE_CACHE_TTL_HOURS")
            ttl_fallback = float(_ttl_env) if _ttl_env is not None else None
        except (TypeError, ValueError):
            ttl_fallback = None
        self.ttl_hours = float(ttl_hours) if ttl_hours is not None else (ttl_fallback if ttl_fallback is not None else None)
        self._index: dict[str, RewriteEntry] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            p = Path(self.path)
            if not p.exists():
                self._loaded = True
                return
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        entry = RewriteEntry(
                            cluster_id=data.get("cluster_id", ""),
                            corpus=data.get("corpus"),
                            intent=data.get("intent"),
                            rewrites=list(data.get("rewrites") or []),
                            weight=float(data.get("weight", 1.0)),
                            last_used=float(data.get("last_used", 0.0)),
                            created_at=float(data.get("created_at", 0.0)),
                        )
                        if entry.cluster_id:
                            self._index[entry.cluster_id] = entry
                    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                        continue
        finally:
            self._loaded = True

    def _decay(self, entry: RewriteEntry) -> float:
        # Exponential decay based on half-life
        try:
            now = time.time()
            dt = max(0.0, now - (entry.last_used or entry.created_at or now))
            half_life_sec = self.half_life * 3600.0
            if half_life_sec <= 0:
                return entry.weight
            # weight * 0.5^(dt/half_life)
            decay_factor = pow(0.5, dt / half_life_sec)
            return float(entry.weight) * float(decay_factor)
        except (TypeError, ValueError, OverflowError):
            return float(entry.weight)

    def get(self, query: str, *, intent: str | None = None, corpus: str | None = None) -> list[str] | None:
        self._load()
        cid = _cluster_id(query, intent=intent, corpus=corpus)
        entry = self._index.get(cid)
        if not entry:
            # metrics: miss (no entry)
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                increment_counter("rag_rewrite_cache_misses_total", 1, labels={"corpus": str(corpus or ""), "intent": str(intent or ""), "reason": "empty"})
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
            return None
        # Hard TTL expiry
        if self.ttl_hours is not None:
            now = time.time()
            last = float(entry.last_used or entry.created_at or now)
            ttl_sec = float(self.ttl_hours) * 3600.0
            if now - last > ttl_sec:
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                    increment_counter("rag_rewrite_cache_misses_total", 1, labels={"corpus": str(corpus or ""), "intent": str(intent or ""), "reason": "expired"})
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                return None
        # Apply decay to reorder (heavier first)
        try:
            scored = [(self._decay(entry), r) for r in (entry.rewrites or [])]
            scored.sort(key=lambda x: x[0], reverse=True)
            out = [r for _, r in scored if isinstance(r, str) and r.strip()][:5]
            if out:
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                    increment_counter("rag_rewrite_cache_hits_total", 1, labels={"corpus": str(corpus or ""), "intent": str(intent or "")})
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            else:
                try:
                    from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                    increment_counter("rag_rewrite_cache_misses_total", 1, labels={"corpus": str(corpus or ""), "intent": str(intent or ""), "reason": "empty_rewrites"})
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                    pass
        except (TypeError, ValueError, RuntimeError):
            return entry.rewrites[:5]
        else:
            return out

    def put(self, query: str, rewrites: list[str], *, intent: str | None = None, corpus: str | None = None) -> None:
        if not rewrites:
            return
        self._load()
        cid = _cluster_id(query, intent=intent, corpus=corpus)
        now = time.time()
        entry = self._index.get(cid)
        if entry is None:
            entry = RewriteEntry(
                cluster_id=cid,
                corpus=corpus,
                intent=intent,
                rewrites=[],
                weight=1.0,
                last_used=now,
                created_at=now,
            )
            self._index[cid] = entry
        # Merge rewrites (dedupe, keep order)
        existing = {r.strip(): None for r in (entry.rewrites or [])}
        for r in rewrites:
            s = (r or "").strip()
            if s and s not in existing:
                entry.rewrites.append(s)
                existing[s] = None
        entry.last_used = now
        entry.weight = min(10.0, float(entry.weight) + 0.25)
        # Append to JSONL for durability (append-only)
        try:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "cluster_id": entry.cluster_id,
                    "corpus": entry.corpus,
                    "intent": entry.intent,
                    "rewrites": entry.rewrites[-20:],  # bound
                    "weight": entry.weight,
                    "last_used": entry.last_used,
                    "created_at": entry.created_at,
                }) + "\n")
            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
                increment_counter("rag_rewrite_cache_puts_total", 1, labels={"corpus": str(corpus or ""), "intent": str(intent or "")})
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                pass
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Failed to persist rewrite cache: {}", _exception_type_name(exc))
