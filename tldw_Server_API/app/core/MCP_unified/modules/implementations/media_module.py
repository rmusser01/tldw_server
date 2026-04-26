"""
Media Module for Unified MCP

Production-ready media management module with full MCP compliance.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import os
import socket
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from loguru import logger

from ....DB_Management.db_path_utils import DatabasePaths
from ....DB_Management.media_db.errors import DatabaseError
from ....DB_Management.media_db.api import (
    create_media_database,
    get_document_version,
    get_latest_transcription,
    get_media_by_id,
    get_media_transcripts,
    get_unvectorized_anchor_index_for_offset,
    get_unvectorized_chunk_index_by_uuid,
    get_unvectorized_chunks_in_range,
    has_unvectorized_chunks,
    permanently_delete_item,
    search_media,
)
from ...persona_scope import get_explicit_scope_ids, merge_requested_ids_with_scope
from ..base import BaseModule, create_resource_definition, create_tool_definition
from ..disk_space import get_free_disk_space_gb

MediaDbLike = Any

_MEDIA_MODULE_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
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
    json.JSONDecodeError,
)


class UnsupportedMediaQueueBackendError(RuntimeError):
    """Raised when a configured media queue backend is recognized but not implemented."""


class MediaModule(BaseModule):
    """
    Enhanced Media Module with production features.

    Provides tools for:
    - Media search (full-text and semantic)
    - Media ingestion (URLs, files)
    - Transcript retrieval
    - Media metadata management
    - Summary generation
    """

    async def on_initialize(self) -> None:
        """Initialize media module with connection pooling"""
        try:
            # Get database path from config
            default_db_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
            db_path = self.config.settings.get("db_path") or default_db_path

            # Ensure database directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # Initialize database with async support
            self.db = create_media_database(
                client_id=f"mcp_media_{self.config.name}",
                db_path=db_path,
            )
            self._module_db_owner = self.db

            # Initialize connection pool if supported
            if hasattr(self.db, 'initialize_pool'):
                await self.db.initialize_pool(
                    pool_size=self.config.settings.get("pool_size", 10)
                )

            # Cache for frequently accessed data
            self._media_cache = {}
            self._cache_ttl = self.config.settings.get("cache_ttl", 300)  # 5 minutes
            self._semantic_retrievers: dict[tuple[Optional[str], Optional[str]], Any] = {}
            self._ingestion_jobs: dict[str, dict[str, Any]] = {}
            self._ingestion_jobs_lock = asyncio.Lock()
            self._user_db_cache: OrderedDict[str, tuple[MediaDbLike, float]] = OrderedDict()
            self._user_db_cache_lock = threading.Lock()
            # Per-user DB cache bounds (TTL + LRU)
            self._user_db_cache_ttl_seconds = int(self.config.settings.get("user_db_cache_ttl_seconds", 900))
            self._user_db_cache_max_size = int(self.config.settings.get("user_db_cache_max_size", 100))
            if self._user_db_cache_ttl_seconds <= 0:
                self._user_db_cache_ttl_seconds = 900
            if self._user_db_cache_max_size <= 0:
                self._user_db_cache_max_size = 1

            logger.info(f"Media module initialized with database: {db_path}")

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to initialize media module: {e}")
            raise

    async def on_shutdown(self) -> None:
        """Graceful shutdown with connection cleanup"""
        try:
            # Clear cache
            self._media_cache.clear()
            try:
                if hasattr(self, "_semantic_retrievers"):
                    for retriever in self._semantic_retrievers.values():
                        close_fn = getattr(retriever, "close", None)
                        if callable(close_fn):
                            with contextlib.suppress(_MEDIA_MODULE_NONCRITICAL_EXCEPTIONS):
                                close_fn()
                    self._semantic_retrievers.clear()
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                pass

            try:
                if hasattr(self, "_user_db_cache"):
                    for entry in self._user_db_cache.values():
                        db = entry[0] if isinstance(entry, tuple) else entry
                        self._close_media_db_instance(db)
                    self._user_db_cache.clear()
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                pass

            # Close database connections
            if hasattr(self.db, 'close_pool'):
                await self.db.close_pool()

            logger.info("Media module shutdown complete")

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error during media module shutdown: {e}")

    async def check_health(self) -> dict[str, bool]:
        """Comprehensive health checks"""
        checks = {
            "database_connection": False,
            "database_writable": False,
            "disk_space": False,
            "service_available": True
        }

        try:
            # Check database connection (backend-agnostic)
            try:
                cur = self.db.execute_query("SELECT 1")
                _ = cur.fetchone()
                checks["database_connection"] = True
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as _db_e:
                logger.debug(f"Media DB connection check failed: {_db_e}")
                checks["database_connection"] = False

            # Check if database is writable (use a test table or transaction)
            # This is a simplified check - implement proper health check table
            try:
                # Use a short-lived transaction to avoid leaving artifacts
                with self.db.transaction():
                    self.db.execute_query("CREATE TABLE IF NOT EXISTS _mcp_healthcheck (k TEXT PRIMARY KEY, v TEXT)")
                    self.db.execute_query("INSERT OR REPLACE INTO _mcp_healthcheck(k, v) VALUES (?, ?)", ("ping", datetime.utcnow().isoformat()))
                    # Best-effort cleanup to keep DB tidy (ignore errors for non-SQLite backends)
                    with contextlib.suppress(_MEDIA_MODULE_NONCRITICAL_EXCEPTIONS):
                        self.db.execute_query("DELETE FROM _mcp_healthcheck WHERE k = ?", ("ping",))
                checks["database_writable"] = True
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as _w_e:
                logger.debug(f"Media DB writable check failed: {_w_e}")
                checks["database_writable"] = False

            # Check disk space
            default_db_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
            db_path = self.config.settings.get("db_path", default_db_path)
            db_dir = os.path.dirname(db_path) or "."
            free_gb = get_free_disk_space_gb(db_dir)
            checks["disk_space"] = free_gb > 1  # At least 1GB free

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Health check failed: {e}")

        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available media tools"""
        tools = [
            # Context-search tools (FTS-only v1)
            create_tool_definition(
                name="media.search",
                description="Search media by title/content with optional filters (FTS-only).",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                        "media_types": {"type": "array", "items": {"type": "string"}},
                        "date_from": {"type": "string", "description": "ISO 8601 start date"},
                        "date_to": {"type": "string", "description": "ISO 8601 end date"},
                        "order_by": {"type": "string", "enum": ["relevance", "recent"], "default": "relevance"}
                    },
                    "required": ["query"],
                },
                metadata={"category": "search", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="media.get",
                description="Retrieve media content or snippet by id.",
                parameters={
                    "properties": {
                        "media_id": {"type": "integer"},
                        "retrieval": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string", "enum": ["snippet", "full"], "default": "snippet"},
                                "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                            }
                        }
                    },
                    "required": ["media_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="search_media",
                description="Search for media content using keywords or semantic search",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                            "minLength": 1,
                            "maxLength": 1000
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["keyword", "semantic", "hybrid"],
                            "default": "keyword",
                            "description": "Type of search to perform"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10,
                            "description": "Maximum number of results"
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Pagination offset"
                        }
                    },
                    "required": ["query"]
                },
                metadata={"category": "search", "auth_required": True}
            ),

            create_tool_definition(
                name="get_transcript",
                description="Get the transcript for a specific media item",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "include_timestamps": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include timestamps in transcript"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "srt", "vtt", "json"],
                            "default": "text",
                            "description": "Output format"
                        }
                    },
                    "required": ["media_id"]
                },
                metadata={"category": "retrieval", "auth_required": True}
            ),

            create_tool_definition(
                name="get_media_metadata",
                description="Get metadata for a specific media item",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "include_stats": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include usage statistics"
                        }
                    },
                    "required": ["media_id"]
                },
                metadata={"category": "metadata", "auth_required": True}
            ),

            create_tool_definition(
                name="ingest_media",
                description="Ingest a new media item from URL or file",
                parameters={
                    "properties": {
                        "url": {
                            "type": "string",
                            "format": "uri",
                            "description": "URL of the media to ingest"
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title for the media"
                        },
                        "process_type": {
                            "type": "string",
                            "enum": ["transcribe", "summarize", "both", "none"],
                            "default": "transcribe",
                            "description": "Type of processing to perform"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                            "default": "normal",
                            "description": "Processing priority"
                        }
                    },
                    "required": ["url"]
                },
                metadata={"category": "ingestion", "auth_required": True, "admin_only": False}
            ),

            create_tool_definition(
                name="update_media",
                description="Update media metadata or content",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "updates": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["media_id", "updates"]
                },
                metadata={"category": "management", "auth_required": True}
            ),

            create_tool_definition(
                name="delete_media",
                description="Delete a media item (soft delete)",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "permanent": {
                            "type": "boolean",
                            "default": False,
                            "description": "Permanently delete (admin only)"
                        }
                    },
                    "required": ["media_id"]
                },
                metadata={"category": "management", "auth_required": True}
            )
        ]
        return tools

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        """Execute media tool with validation and error handling"""
        # Validate and sanitize inputs
        arguments = self.sanitize_input(arguments)
        # High-risk operations: validate against stricter schema
        try:
            self.validate_tool_arguments(tool_name, arguments)
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve

        # Log tool execution
        logger.info(f"Executing media tool: {tool_name}", extra={"audit": True})

        try:
            if tool_name == "media.search":
                return await self._media_search_normalized(context=context, **arguments)
            elif tool_name == "media.get":
                return await self._media_get_normalized(context=context, **arguments)
            elif tool_name == "search_media":
                return await self._search_media(context=context, **arguments)

            elif tool_name == "get_transcript":
                return await self._get_transcript(context=context, **arguments)

            elif tool_name == "get_media_metadata":
                return await self._get_media_metadata(context=context, **arguments)

            elif tool_name == "ingest_media":
                return await self._ingest_media(**arguments)

            elif tool_name == "update_media":
                return await self._update_media(context=context, **arguments)

            elif tool_name == "delete_media":
                return await self._delete_media(context=context, **arguments)

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise

    def _allow_anonymous_access(self) -> bool:
        try:
            return bool(self.config.settings.get("allow_anonymous_access", False))
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            return False

    def _normalize_media_db_path(self, value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if text == ":memory:" or text.startswith("file:"):
            return text
        return os.path.abspath(text)

    def _is_module_media_db_owner(self, db: Any) -> bool:
        owner = getattr(self, "_module_db_owner", None)
        if owner is not None:
            return db is owner
        if not callable(getattr(db, "close_connection", None)):
            return False
        db_path = self._normalize_media_db_path(getattr(db, "db_path_str", None))
        if db_path is None:
            return False
        configured_path = self.config.settings.get("db_path")
        if configured_path:
            return db_path == self._normalize_media_db_path(configured_path)
        default_path = DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())
        return db_path == self._normalize_media_db_path(default_path)

    def _close_media_db_instance(self, db: MediaDbLike) -> None:
        with contextlib.suppress(_MEDIA_MODULE_NONCRITICAL_EXCEPTIONS):
            db.close_connection()
        try:
            pool = db.backend.get_pool()
            pool.close_all()
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            pass

    def _evict_user_db_cache_locked(self, now: Optional[float] = None) -> None:
        cache = getattr(self, "_user_db_cache", None)
        if not cache:
            return
        now_ts = now if now is not None else time.monotonic()
        ttl = max(1, int(getattr(self, "_user_db_cache_ttl_seconds", 900)))
        max_size = max(1, int(getattr(self, "_user_db_cache_max_size", 100)))

        expired = [k for k, (_, last_used) in cache.items() if now_ts - last_used > ttl]
        for key in expired:
            try:
                db, _ = cache.pop(key)
                self._close_media_db_instance(db)
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                pass

        while len(cache) > max_size:
            try:
                key, (db, _) = cache.popitem(last=False)
                self._close_media_db_instance(db)
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                break

    def _get_or_create_user_db(self, db_path: str) -> MediaDbLike:
        lock = getattr(self, "_user_db_cache_lock", None)
        if lock is None:
            logger.warning("User DB cache lock not initialized; bypassing cache for {}", db_path)
            return create_media_database(
                client_id=f"mcp_media_{self.config.name}",
                db_path=db_path,
            )
        with lock:
            cache = getattr(self, "_user_db_cache", None)
            if cache is None:
                self._user_db_cache = OrderedDict()
                cache = self._user_db_cache
            now_ts = time.monotonic()
            self._evict_user_db_cache_locked(now_ts)
            cached = cache.get(db_path)
            if cached is not None:
                db, last_used = cached
                ttl = max(1, int(getattr(self, "_user_db_cache_ttl_seconds", 900)))
                if now_ts - last_used <= ttl:
                    cache[db_path] = (db, now_ts)
                    cache.move_to_end(db_path)
                    return db
                try:
                    cache.pop(db_path, None)
                    self._close_media_db_instance(db)
                except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                    pass
            db = create_media_database(
                client_id=f"mcp_media_{self.config.name}",
                db_path=db_path,
            )
            cache[db_path] = (db, now_ts)
            cache.move_to_end(db_path)
            self._evict_user_db_cache_locked(now_ts)
            return db

    def _open_media_db(self, context: Any | None) -> MediaDbLike:
        """Open per-user media DB when context provides one; fallback to module DB."""
        db = getattr(self, "db", None)
        if db is None:
            raise ValueError("Media database not initialized")
        if not self._is_module_media_db_owner(db):
            return db
        if context is None or getattr(context, "user_id", None) is None:
            if self._allow_anonymous_access():
                return db
            raise PermissionError("User context required for media access")
        db_paths = getattr(context, "db_paths", None)
        if not isinstance(db_paths, dict):
            raise PermissionError("Media DB path not available in context")
        user_media_path = db_paths.get("media")
        if not user_media_path:
            raise PermissionError("Media DB path not available in context")
        if str(user_media_path) == str(getattr(db, "db_path_str", "")):
            return db
        return self._get_or_create_user_db(str(user_media_path))

    def _normalize_scores(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        # If relevance_score present (bm25-like), lower is better
        if all((r.get("relevance_score") is not None) for r in rows):
            vals = [float(r.get("relevance_score") or 0.0) for r in rows]
            mn, mx = min(vals), max(vals)
            if mx - mn < 1e-9:
                return [1.0 for _ in vals]
            return [(mx - v) / (mx - mn + 1e-9) for v in vals]
        # Fallback positional decay
        n = len(rows)
        if n == 1:
            return [1.0]
        return [1.0 - (i / max(1, n - 1)) for i in range(n)]

    def _make_snippet(self, text: Optional[str], query: Optional[str], length: int = 300) -> str:
        if not text:
            return ""
        length = max(50, min(length, 2000))
        t = " ".join(text.split())
        if not query:
            return t[:length]
        try:
            idx = t.lower().find(query.lower())
            if idx == -1:
                return t[:length]
            half = max(0, length // 2)
            start = max(0, idx - half)
            end = min(len(t), start + length)
            return t[start:end]
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            return t[:length]

    def _get_latest_description(self, dbi: MediaDbLike, media_id: int) -> Optional[str]:
        try:
            latest = get_document_version(
                dbi,
                media_id=media_id,
                version_number=None,
                include_content=False,
            )
            return (latest or {}).get("analysis_content")
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            return None

    def _sanitize_media_metadata(self, row: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(row)
        for key in ("content", "client_id", "vector_embedding"):
            sanitized.pop(key, None)
        return sanitized

    def _media_ids_scope(self, context: Any | None) -> set[str] | None:
        return get_explicit_scope_ids(context, "media_id")

    def _effective_media_ids_filter(self, requested: Any, context: Any | None) -> list[int] | None:
        scoped_ids = self._media_ids_scope(context)
        merged = merge_requested_ids_with_scope(requested, scoped_ids=scoped_ids)
        if merged is None:
            return None
        parsed: list[int] = []
        for value in sorted(merged):
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                continue
        return parsed

    def _assert_media_scope_allowed(self, media_id: int, context: Any | None) -> None:
        scoped_ids = self._media_ids_scope(context)
        if scoped_ids is None:
            return
        if str(media_id) not in scoped_ids:
            raise PermissionError("Access denied for this media item by persona scope")

    async def _media_search_normalized(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        snippet_length: int = 300,
        media_types: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        order_by: str = "relevance",
        media_ids_filter: Optional[list[Any]] = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._media_search_normalized_sync,
            query,
            limit,
            offset,
            snippet_length,
            media_types,
            date_from,
            date_to,
            order_by,
            media_ids_filter,
            context,
        )

    def _media_search_normalized_sync(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        snippet_length: int = 300,
        media_types: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        order_by: str = "relevance",
        media_ids_filter: Optional[list[Any]] = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        # Session defaults
        try:
            if context and getattr(context, "metadata", None):
                sc = context.metadata.get("safe_config")
                if isinstance(sc, dict):
                    snippet_length = int(sc.get("snippet_length", snippet_length))
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            pass

        # Build date_range
        date_range = None
        try:
            if date_from or date_to:
                from datetime import datetime as _dt
                date_range = {}
                if date_from:
                    date_range["start_date"] = _dt.fromisoformat(date_from)
                if date_to:
                    date_range["end_date"] = _dt.fromisoformat(date_to)
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            date_range = None

        # Map order
        sort_by = "relevance" if order_by == "relevance" else "last_modified_desc"
        page_size = max(1, limit)
        page = (offset // page_size) + 1
        local_offset = offset % page_size
        effective_media_ids_filter = self._effective_media_ids_filter(media_ids_filter, context)
        if effective_media_ids_filter is not None and not effective_media_ids_filter:
            return {
                "results": [],
                "has_more": False,
                "next_offset": None,
                "total_estimated": 0,
            }
        dbi = self._open_media_db(context)
        results, total = search_media(
            dbi,
            search_query=query,
            search_fields=["title", "content"],
            media_types=media_types or None,
            date_range=date_range,
            sort_by=sort_by,
            media_ids_filter=effective_media_ids_filter,
            page=page,
            results_per_page=page_size,
            include_trash=False,
            include_deleted=False,
        )
        if local_offset:
            if len(results) == page_size and (offset + limit) < total:
                more_results, _ = search_media(
                    dbi,
                    search_query=query,
                    search_fields=["title", "content"],
                    media_types=media_types or None,
                    date_range=date_range,
                    sort_by=sort_by,
                    media_ids_filter=effective_media_ids_filter,
                    page=page + 1,
                    results_per_page=page_size,
                    include_trash=False,
                    include_deleted=False,
                )
                results = results + more_results
            results = results[local_offset: local_offset + limit]
        else:
            results = results[:limit]
        scores = self._normalize_scores(results)
        out = []
        for i, r in enumerate(results):
            mid = r.get("id")
            content = r.get("content") or ""
            # Compute an approximate offset of query within content as a location hint
            approx_offset = None
            try:
                idx = content.lower().find(query.lower()) if query else -1
                if idx >= 0:
                    approx_offset = idx
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                approx_offset = None
            # Try to map to prechunked chunk_index when available
            loc_hint = None
            try:
                if approx_offset is not None and isinstance(mid, (int, str)):
                    mid_int = int(mid)
                    if has_unvectorized_chunks(dbi, mid_int):
                        cidx = get_unvectorized_anchor_index_for_offset(
                            dbi,
                            mid_int,
                            int(approx_offset),
                        )
                        if cidx is not None:
                            loc_hint = {"chunk_index": cidx}
            except DatabaseError as exc:
                logger.debug("Chunk anchor lookup failed for media {}: {}", mid, exc)
                loc_hint = None
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                loc_hint = None
            out.append({
                "id": mid,
                "source": "media",
                "title": r.get("title"),
                "snippet": self._make_snippet(content, query, snippet_length),
                "uri": f"media://{mid}",
                "score": float(scores[i] if i < len(scores) else 0.0),
                "score_type": "fts",
                "created_at": r.get("ingestion_date"),
                "last_modified": r.get("last_modified"),
                "version": r.get("version"),
                "tags": None,
                "media_type": r.get("type"),
                "url": r.get("url"),
                "loc": (loc_hint if loc_hint is not None else ({"approx_offset": approx_offset} if approx_offset is not None else None)),
            })
        return {
            "results": out,
            "has_more": (offset + len(results)) < total,
            "next_offset": (offset + len(results)) if (offset + len(results)) < total else None,
            "total_estimated": total,
        }

    async def _media_get_normalized(self, media_id: int, retrieval: Optional[dict[str, Any]] = None, context: Any | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._media_get_normalized_sync,
            media_id,
            retrieval,
            context,
        )

    def _media_get_normalized_sync(self, media_id: int, retrieval: Optional[dict[str, Any]] = None, context: Any | None = None) -> dict[str, Any]:
        retrieval = retrieval or {}
        self._assert_media_scope_allowed(media_id, context)
        mode = retrieval.get("mode", "snippet")
        snippet_length = int(retrieval.get("snippet_length", 300))
        max_tokens = retrieval.get("max_tokens")
        cpt = int(retrieval.get("chars_per_token", 4))
        if cpt <= 0:
            raise ValueError("chars_per_token must be a positive integer")
        chunk_size_tokens = int(retrieval.get("chunk_size_tokens", 1000 if max_tokens else 500))
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be a positive integer")
        sibling_window = int(retrieval.get("sibling_window", 1))
        loc = retrieval.get("loc") or {}
        # Session defaults
        try:
            if context and getattr(context, "metadata", None):
                sc = context.metadata.get("safe_config")
                if isinstance(sc, dict):
                    snippet_length = int(sc.get("snippet_length", snippet_length))
                    if isinstance(sc.get("chars_per_token"), int):
                        cpt = int(sc.get("chars_per_token"))
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            pass

        dbi = self._open_media_db(context)
        # Use active media row for metadata + content
        meta = get_media_by_id(dbi, media_id, include_deleted=False, include_trash=False)
        if not meta:
            raise ValueError(f"Media not found: {media_id}")
        # Ownership check
        self._assert_media_access(media_id, context, dbi)
        description = self._get_latest_description(dbi, media_id)
        content = meta.get("content") or ""
        item = {
            "id": meta.get("id"),
            "source": "media",
            "title": meta.get("title"),
            "snippet": self._make_snippet(content, None, snippet_length),
            "uri": f"media://{meta.get('id')}",
            "score": 1.0,
            "score_type": "fts",
            "created_at": meta.get("ingestion_date"),
            "last_modified": meta.get("last_modified"),
            "version": meta.get("version"),
            "tags": None,
            "media_type": meta.get("type"),
            "url": meta.get("url"),
            "loc": None,
        }
        item["description"] = description
        if mode == "full":
            body = content
            return {"meta": item, "content": body, "attachments": None}

        # Helper for on-the-fly chunking
        def _chunkify(text: str, size_chars: int) -> list[str]:
            if size_chars <= 0:
                size_chars = 1000 * cpt
            chunks = []
            i = 0
            n = len(text)
            while i < n:
                chunks.append(text[i:i+size_chars])
                i += size_chars
            return chunks

        def _estimate_tokens(chars: int) -> int:
            return max(1, (chars + cpt - 1) // cpt)

        # If chunking modes requested:
        if mode in {"chunk", "chunk_with_siblings", "auto"}:
            if mode == "auto" and max_tokens:
                mode = "chunk_with_siblings"
            elif mode == "auto":
                mode = "snippet"

            # Prefer prechunked retrieval when available
            prefer_prechunked = bool(retrieval.get("prefer_prechunked", True))
            anchor_index: Optional[int] = None
            anchor_uuid: Optional[str] = None

            # Extract loc hints
            approx_offset = None
            try:
                if isinstance(loc, dict) and isinstance(loc.get("approx_offset"), int):
                    approx_offset = int(loc.get("approx_offset"))
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                approx_offset = None
            if isinstance(loc, dict) and isinstance(loc.get("chunk_index"), int):
                anchor_index = int(loc.get("chunk_index"))
            if isinstance(loc, dict) and isinstance(loc.get("chunk_uuid"), str):
                anchor_uuid = str(loc.get("chunk_uuid"))

            if prefer_prechunked:
                try:
                    if has_unvectorized_chunks(dbi, media_id):
                        # Resolve anchor by uuid or offset
                        if anchor_index is None and anchor_uuid:
                            anchor_index = get_unvectorized_chunk_index_by_uuid(dbi, media_id, anchor_uuid)
                        if anchor_index is None and isinstance(approx_offset, int):
                            anchor_index = get_unvectorized_anchor_index_for_offset(
                                dbi,
                                media_id,
                                approx_offset,
                            )
                        if anchor_index is None:
                            anchor_index = 0

                        anchor_list = get_unvectorized_chunks_in_range(
                            dbi,
                            media_id,
                            anchor_index,
                            anchor_index,
                        )
                        if anchor_list:
                            anchor_chunk = anchor_list[0]
                            anchor_uuid = anchor_chunk.get("uuid") or anchor_uuid

                            if mode == "chunk":
                                body = anchor_chunk.get("chunk_text") or ""
                                item["loc"] = {"chunk_index": anchor_index, "chunk_uuid": anchor_uuid}
                                return {"meta": item, "content": body, "attachments": None}

                            # chunk_with_siblings budgeted expansion
                            budget_tokens = int(max_tokens) if max_tokens else None
                            selected_indexes: list[int] = [anchor_index]
                            selected_texts: dict[int, str] = {anchor_index: anchor_chunk.get("chunk_text") or ""}
                            if budget_tokens is None:
                                for d in range(1, max(1, sibling_window) + 1):
                                    li = anchor_index - d
                                    ri = anchor_index + d
                                    if li >= 0:
                                        left_chunks = get_unvectorized_chunks_in_range(dbi, media_id, li, li)
                                        if left_chunks:
                                            selected_indexes.insert(0, li)
                                            selected_texts[li] = left_chunks[0].get("chunk_text") or ""
                                    right_chunks = get_unvectorized_chunks_in_range(dbi, media_id, ri, ri)
                                    if right_chunks:
                                        selected_indexes.append(ri)
                                        selected_texts[ri] = right_chunks[0].get("chunk_text") or ""
                            else:
                                current_tokens = _estimate_tokens(len(selected_texts[anchor_index]))
                                left = anchor_index - 1
                                right = anchor_index + 1
                                while True:
                                    progressed = False
                                    if left >= 0:
                                        lc = get_unvectorized_chunks_in_range(dbi, media_id, left, left)
                                        if lc:
                                            t_add = _estimate_tokens(len(lc[0].get("chunk_text") or ""))
                                            if current_tokens + t_add <= budget_tokens:
                                                selected_indexes.insert(0, left)
                                                selected_texts[left] = lc[0].get("chunk_text") or ""
                                                current_tokens += t_add
                                                left -= 1
                                                progressed = True
                                    rc = get_unvectorized_chunks_in_range(dbi, media_id, right, right)
                                    if rc:
                                        t_add = _estimate_tokens(len(rc[0].get("chunk_text") or ""))
                                        if current_tokens + t_add <= budget_tokens:
                                            selected_indexes.append(right)
                                            selected_texts[right] = rc[0].get("chunk_text") or ""
                                            current_tokens += t_add
                                            right += 1
                                            progressed = True
                                    if not progressed:
                                        break

                            body = "\n\n".join([selected_texts[i] for i in selected_indexes])
                            item["loc"] = {"chunk_index": anchor_index, "chunk_uuid": anchor_uuid}
                            return {"meta": item, "content": body, "attachments": None}
                except DatabaseError as exc:
                    logger.debug("Prechunked retrieval failed for media {}: {}", media_id, exc)

            # Fallback: on-the-fly chunking from full content.
            # Preserve anchor_index if it was already resolved (e.g. from
            # chunk_index loc hint or uuid/offset lookup) before the DB call
            # that raised DatabaseError.
            size_chars = max(1, chunk_size_tokens * cpt)
            chunks = _chunkify(content, size_chars)
            if not chunks:
                body = self._make_snippet(content, None, snippet_length)
                return {"meta": item, "content": body, "attachments": None}

            if anchor_index is not None:
                # Clamp the previously-resolved index to the on-the-fly chunk list bounds.
                anchor_index = max(0, min(len(chunks) - 1, anchor_index))
            elif approx_offset is not None:
                anchor_index = max(0, min(len(chunks) - 1, approx_offset // size_chars))
            else:
                anchor_index = 0

            if mode == "chunk":
                body = chunks[anchor_index]
                item["loc"] = {"chunk_index": anchor_index, "approx_offset": anchor_index * size_chars}
                return {"meta": item, "content": body, "attachments": None}

            budget_tokens = int(max_tokens) if max_tokens else None
            sel = [anchor_index]
            if budget_tokens is None:
                for d in range(1, max(1, sibling_window) + 1):
                    if anchor_index - d >= 0:
                        sel.insert(0, anchor_index - d)
                    if anchor_index + d < len(chunks):
                        sel.append(anchor_index + d)
            else:
                current_tokens = _estimate_tokens(len(chunks[anchor_index]))
                left = anchor_index - 1
                right = anchor_index + 1
                while True:
                    progressed = False
                    if left >= 0:
                        t_add = _estimate_tokens(len(chunks[left]))
                        if current_tokens + t_add <= budget_tokens:
                            sel.insert(0, left)
                            current_tokens += t_add
                            left -= 1
                            progressed = True
                    if right < len(chunks):
                        t_add = _estimate_tokens(len(chunks[right]))
                        if current_tokens + t_add <= budget_tokens:
                            sel.append(right)
                            current_tokens += t_add
                            right += 1
                            progressed = True
                    if not progressed:
                        break

            selected_chunks = [chunks[i] for i in sel]
            body = "\n\n".join(selected_chunks)
            item["loc"] = {"chunk_index": anchor_index, "approx_offset": anchor_index * size_chars}
            return {"meta": item, "content": body, "attachments": None}

        # Default snippet mode
        body = self._make_snippet(content, None, snippet_length)
        return {"meta": item, "content": body, "attachments": None}

    async def _search_media(
        self,
        query: str,
        search_type: str = "keyword",
        limit: int = 10,
        offset: int = 0,
        context: Any | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Search media with caching and unified keyword/semantic support."""
        if not query or len(query) > 1000:
            raise ValueError("Invalid search query")

        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))

        dbi = self._open_media_db(context)

        search_fields = kwargs.get("search_fields")
        media_types = kwargs.get("media_types")
        date_range = kwargs.get("date_range")
        must_have_keywords = kwargs.get("must_have_keywords")
        must_not_have_keywords = kwargs.get("must_not_have_keywords")
        sort_by_value = kwargs.get("sort_by")
        order_by_param = kwargs.get("order_by")
        if sort_by_value is None and isinstance(order_by_param, str):
            order_key = order_by_param.strip().lower()
            if order_key == "relevance":
                sort_by_value = "relevance"
            elif order_key in {"recent", "last_modified_desc"}:
                sort_by_value = "last_modified_desc"
            elif order_key == "last_modified_asc":
                sort_by_value = "last_modified_asc"
            elif order_key in {"date_desc", "ingestion_desc"}:
                sort_by_value = "date_desc"
            elif order_key in {"date_asc", "ingestion_asc"}:
                sort_by_value = "date_asc"
            elif order_key in {"title_asc", "title_desc"}:
                sort_by_value = order_key

        media_ids_filter = kwargs.get("media_ids_filter")
        effective_media_ids_filter = self._effective_media_ids_filter(media_ids_filter, context)
        if effective_media_ids_filter is not None and not effective_media_ids_filter:
            return {
                "query": query,
                "type": search_type,
                "count": 0,
                "results": [],
                "offset": offset,
                "limit": limit,
                "total": 0,
            }
        include_trash = bool(kwargs.get("include_trash", False))
        include_deleted = bool(kwargs.get("include_deleted", False))
        metadata_filter = kwargs.get("metadata_filter")
        index_namespace = kwargs.get("index_namespace")
        query_vector = kwargs.get("query_vector")

        cache_payload = {
            "db_path": getattr(dbi, "db_path_str", None),
            "query": query,
            "search_type": search_type,
            "limit": limit,
            "offset": offset,
            "search_fields": search_fields,
            "media_types": media_types,
            "date_range": date_range,
            "must_have_keywords": must_have_keywords,
            "must_not_have_keywords": must_not_have_keywords,
            "sort_by": sort_by_value,
            "order_by": order_by_param,
            "media_ids_filter": effective_media_ids_filter,
            "include_trash": include_trash,
            "include_deleted": include_deleted,
            "metadata_filter": metadata_filter,
            "index_namespace": index_namespace,
            "user_id": getattr(context, "user_id", None),
        }
        if query_vector is not None:
            cache_payload["query_vector"] = query_vector

        cache_key = self._make_cache_key("search_media", cache_payload)
        cached = self._media_cache.get(cache_key)
        if cached and (datetime.utcnow() - cached["time"]).total_seconds() < self._cache_ttl:
            logger.debug(f"Cache hit for search: {cache_key}")
            return cached["data"]

        try:
            if search_type == "keyword":
                rows, total = await asyncio.to_thread(
                    self._keyword_search,
                    dbi,
                    query,
                    limit,
                    offset,
                    search_fields=search_fields,
                    media_types=media_types,
                    date_range=date_range,
                    must_have_keywords=must_have_keywords,
                    must_not_have_keywords=must_not_have_keywords,
                    sort_by_value=sort_by_value,
                    media_ids_filter=effective_media_ids_filter,
                    include_trash=include_trash,
                    include_deleted=include_deleted,
                )
            elif search_type == "semantic":
                rows, total = await self._semantic_search(
                    query=query,
                    limit=limit,
                    offset=offset,
                    dbi=dbi,
                    media_types=media_types,
                    metadata_filter=metadata_filter,
                    index_namespace=index_namespace,
                    media_ids_filter=effective_media_ids_filter,
                    context=context,
                    query_vector=query_vector,
                )
            elif search_type == "hybrid":
                rows, total = await self._hybrid_search(
                    query=query,
                    limit=limit,
                    offset=offset,
                    dbi=dbi,
                    media_types=media_types,
                    date_range=date_range,
                    must_have_keywords=must_have_keywords,
                    must_not_have_keywords=must_not_have_keywords,
                    sort_by_value=sort_by_value,
                    media_ids_filter=effective_media_ids_filter,
                    include_trash=include_trash,
                    include_deleted=include_deleted,
                    search_fields=search_fields,
                    metadata_filter=metadata_filter,
                    index_namespace=index_namespace,
                    context=context,
                    query_vector=query_vector,
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            formatted_results = {
                "query": query,
                "type": search_type,
                "count": len(rows),
                "results": rows,
                "offset": offset,
                "limit": limit,
                "total": total,
            }

            self._media_cache[cache_key] = {
                "time": datetime.utcnow(),
                "data": formatted_results,
            }
            await self._clean_cache()
            return formatted_results

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Search failed: {e}")
            raise

    def _serialize_for_cache(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._serialize_for_cache(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_for_cache(v) for v in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (bytes, bytearray)):
            return value.hex()
        return value

    def _make_cache_key(self, namespace: str, payload: dict[str, Any]) -> str:
        try:
            normalised = self._serialize_for_cache(payload)
            raw = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            raw = repr(payload)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def _keyword_search(
        self,
        dbi: MediaDbLike,
        query: str,
        limit: int,
        offset: int,
        *,
        search_fields: Optional[list[str]],
        media_types: Optional[list[str]],
        date_range: Optional[dict[str, Any]],
        must_have_keywords: Optional[list[str]],
        must_not_have_keywords: Optional[list[str]],
        sort_by_value: Optional[str],
        media_ids_filter: Optional[list[Any]],
        include_trash: bool,
        include_deleted: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        page_size = max(1, limit)
        page = (offset // page_size) + 1
        local_offset = offset % page_size
        rows, total = search_media(
            dbi,
            search_query=query,
            search_fields=search_fields,
            media_types=media_types,
            date_range=date_range,
            must_have_keywords=must_have_keywords,
            must_not_have_keywords=must_not_have_keywords,
            sort_by=sort_by_value or "last_modified_desc",
            media_ids_filter=media_ids_filter,
            page=page,
            results_per_page=page_size,
            include_trash=include_trash,
            include_deleted=include_deleted,
        )
        if local_offset:
            if len(rows) == page_size and (offset + limit) < total:
                more_rows, _ = search_media(
                    dbi,
                    search_query=query,
                    search_fields=search_fields,
                    media_types=media_types,
                    date_range=date_range,
                    must_have_keywords=must_have_keywords,
                    must_not_have_keywords=must_not_have_keywords,
                    sort_by=sort_by_value or "last_modified_desc",
                    media_ids_filter=media_ids_filter,
                    page=page + 1,
                    results_per_page=page_size,
                    include_trash=include_trash,
                    include_deleted=include_deleted,
                )
                rows = rows + more_rows
            rows = rows[local_offset: local_offset + limit]
        else:
            rows = rows[:limit]
        return rows, total

    def _keyword_search_head(
        self,
        dbi: MediaDbLike,
        query: str,
        size: int,
        *,
        search_fields: Optional[list[str]],
        media_types: Optional[list[str]],
        date_range: Optional[dict[str, Any]],
        must_have_keywords: Optional[list[str]],
        must_not_have_keywords: Optional[list[str]],
        sort_by_value: Optional[str],
        media_ids_filter: Optional[list[Any]],
        include_trash: bool,
        include_deleted: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        fetch_ceiling = int(self.config.settings.get("hybrid_fetch_ceiling", 200))
        fetch_size = max(1, min(int(size), fetch_ceiling))
        rows, total = search_media(
            dbi,
            search_query=query,
            search_fields=search_fields,
            media_types=media_types,
            date_range=date_range,
            must_have_keywords=must_have_keywords,
            must_not_have_keywords=must_not_have_keywords,
            sort_by=sort_by_value or "last_modified_desc",
            media_ids_filter=media_ids_filter,
            page=1,
            results_per_page=fetch_size,
            include_trash=include_trash,
            include_deleted=include_deleted,
        )
        return rows[:fetch_size], total

    def _get_semantic_retriever(self, dbi: MediaDbLike, context: Any | None) -> Any | None:
        db_path = getattr(dbi, "db_path_str", None)
        user_key = str(getattr(context, "user_id", None) or "0")
        retriever_key = (db_path, user_key)
        retriever = self._semantic_retrievers.get(retriever_key)
        if retriever is False:
            return None
        if retriever is None:
            try:
                from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (  # lazy import
                    MediaDBRetriever,
                    RetrievalConfig,
                )
                config = RetrievalConfig(max_results=20, use_fts=True, use_vector=True)
                retriever = MediaDBRetriever(
                    db_path=db_path,
                    config=config,
                    user_id=user_key,
                    media_db=dbi,
                )
                self._semantic_retrievers[retriever_key] = retriever
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Semantic retriever unavailable: {exc}")
                self._semantic_retrievers[retriever_key] = False  # sentinel to avoid retries
                return None
        return retriever

    async def _semantic_search(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        dbi: MediaDbLike,
        media_types: Optional[list[str]],
        metadata_filter: Optional[dict[str, Any]],
        index_namespace: Any,
        media_ids_filter: Optional[list[Any]],
        context: Any | None,
        query_vector: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        retriever = self._get_semantic_retriever(dbi, context)
        if retriever is None:
            return [], 0
        max_results = max(1, limit + offset)
        with contextlib.suppress(_MEDIA_MODULE_NONCRITICAL_EXCEPTIONS):
            retriever.config.max_results = max_results

        media_type_arg: Optional[str] = None
        if isinstance(media_types, list) and len(media_types) == 1:
            media_type_arg = media_types[0]

        try:
            docs = await retriever._retrieve_vector(  # type: ignore[attr-defined]
                query,
                media_type=media_type_arg,
                metadata_filter=metadata_filter,
                index_namespace=index_namespace,
                allowed_media_ids=media_ids_filter,
                query_vector=query_vector,
            )
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Semantic retrieval failed, falling back to empty set: {exc}")
            docs = []

        allowed_types = {t.lower() for t in media_types} if media_types else None
        rows: list[dict[str, Any]] = []
        for doc in docs or []:
            meta = getattr(doc, "metadata", {}) or {}
            media_type_val = (
                meta.get("media_type")
                or meta.get("type")
                or meta.get("mediaType")
                or meta.get("kind")
            )
            if allowed_types:
                if not media_type_val:
                    continue
                if str(media_type_val).lower() not in allowed_types:
                    continue
            doc_id = getattr(doc, "id", None)
            try:
                media_id = int(doc_id) if doc_id is not None else None
            except (TypeError, ValueError):
                media_id = doc_id
            record = {
                "id": media_id,
                "title": meta.get("title"),
                "content": getattr(doc, "content", None),
                "type": media_type_val or meta.get("type"),
                "media_type": media_type_val or meta.get("type"),
                "url": meta.get("url"),
                "ingestion_date": meta.get("created_at") or meta.get("ingestion_date"),
                "last_modified": meta.get("last_modified"),
                "semantic_score": float(getattr(doc, "score", 0.0) or 0.0),
                "score_type": "semantic",
            }
            rows.append(record)

        total = len(rows)
        return rows[offset: offset + limit], total

    async def _hybrid_search(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        dbi: MediaDbLike,
        media_types: Optional[list[str]],
        date_range: Optional[dict[str, Any]],
        must_have_keywords: Optional[list[str]],
        must_not_have_keywords: Optional[list[str]],
        sort_by_value: Optional[str],
        media_ids_filter: Optional[list[Any]],
        include_trash: bool,
        include_deleted: bool,
        search_fields: Optional[list[str]],
        metadata_filter: Optional[dict[str, Any]],
        index_namespace: Any,
        context: Any | None,
        query_vector: Any,
    ) -> tuple[list[dict[str, Any]], int]:
        fetch_size = limit + offset
        keyword_rows_all, keyword_total = await asyncio.to_thread(
            self._keyword_search_head,
            dbi,
            query,
            fetch_size,
            search_fields=search_fields,
            media_types=media_types,
            date_range=date_range,
            must_have_keywords=must_have_keywords,
            must_not_have_keywords=must_not_have_keywords,
            sort_by_value=sort_by_value,
            media_ids_filter=media_ids_filter,
            include_trash=include_trash,
            include_deleted=include_deleted,
        )

        semantic_rows_all, semantic_total = await self._semantic_search(
            query=query,
            limit=max(1, fetch_size),
            offset=0,
            dbi=dbi,
            media_types=media_types,
            metadata_filter=metadata_filter,
            index_namespace=index_namespace,
            media_ids_filter=media_ids_filter,
            context=context,
            query_vector=query_vector,
        )

        combined: OrderedDict[Any, dict[str, Any]] = OrderedDict()
        for row in keyword_rows_all:
            key = row.get("id")
            if key is None:
                continue
            merged = dict(row)
            merged.setdefault("score_type", "fts")
            combined[key] = merged

        for row in semantic_rows_all:
            key = row.get("id")
            if key is None:
                continue
            merged = dict(row)
            existing = combined.get(key)
            if existing:
                if "semantic_score" in merged:
                    existing["semantic_score"] = merged["semantic_score"]
                existing["score_type"] = "hybrid"
            else:
                combined[key] = merged

        combined_rows = list(combined.values())
        total_estimate = max(keyword_total, semantic_total, len(combined_rows))
        paged = combined_rows[offset: offset + limit]
        for row in paged:
            row.setdefault("score_type", "hybrid")
        return paged, total_estimate

    async def _get_transcript(
        self,
        media_id: int,
        include_timestamps: bool = False,
        format: str = "text",
        context: Any | None = None,
        **kwargs
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._get_transcript_sync,
            media_id,
            include_timestamps,
            format,
            context,
            kwargs,
        )

    def _get_transcript_sync(
        self,
        media_id: int,
        include_timestamps: bool = False,
        format: str = "text",
        context: Any | None = None,
        _kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Get media transcript with formatting options"""
        try:
            self._assert_media_scope_allowed(media_id, context)
            # Ownership check first
            dbi = self._open_media_db(context)
            self._assert_media_access(media_id, context, dbi)
            # Get transcript from database (latest text)
            transcript_text = get_latest_transcription(dbi, media_id)
            if transcript_text is None:
                raise ValueError(f"No transcript found for media ID: {media_id}")

            # Format based on requested type
            if format == "json":
                formatted = get_media_transcripts(dbi, media_id)
            elif format == "text":
                formatted = (
                    self._format_transcript_with_timestamps(transcript_text)
                    if include_timestamps
                    else transcript_text
                )
            elif format == "srt":
                formatted = self._convert_to_srt(transcript_text)
            elif format == "vtt":
                formatted = self._convert_to_vtt(transcript_text)
            else:
                formatted = transcript_text

            return {
                "media_id": media_id,
                "format": format,
                "include_timestamps": include_timestamps,
                "transcript": formatted,
            }

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get transcript: {e}")
            raise

    async def _get_media_metadata(
        self,
        media_id: int,
        include_stats: bool = False,
        context: Any | None = None,
        **kwargs
    ) -> dict[str, Any]:
        metadata = await asyncio.to_thread(
            self._get_media_metadata_sync,
            media_id,
            context,
            kwargs,
        )
        if include_stats:
            stats = await self._get_media_stats(media_id)
            metadata["statistics"] = stats
        return metadata

    def _get_media_metadata_sync(
        self,
        media_id: int,
        context: Any | None = None,
        _kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Get comprehensive media metadata"""
        try:
            self._assert_media_scope_allowed(media_id, context)
            # Ownership check
            dbi = self._open_media_db(context)
            self._assert_media_access(media_id, context, dbi)
            # Get basic metadata
            metadata = get_media_by_id(
                dbi,
                media_id,
                include_deleted=False,
                include_trash=False,
            )

            if not metadata:
                raise ValueError(f"Media not found: {media_id}")

            description = self._get_latest_description(dbi, media_id)
            metadata = self._sanitize_media_metadata(metadata)
            metadata["description"] = description

            return metadata

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get metadata: {e}")
            raise

    async def _ingest_media(
        self,
        url: str,
        title: Optional[str] = None,
        process_type: str = "transcribe",
        tags: Optional[list[str]] = None,
        priority: str = "normal",
        **kwargs
    ) -> dict[str, Any]:
        """Ingest new media with processing options"""
        try:
            # Validate URL
            if not self._validate_url(url):
                raise ValueError("Invalid or unsupported URL")

            # Create ingestion job
            job_id = await self._create_ingestion_job(
                url=url,
                title=title or url,
                process_type=process_type,
                tags=tags or [],
                priority=priority
            )

            # Start processing based on priority
            queued = False
            if priority == "high":
                # Process immediately
                await self._process_media_job(job_id)
            else:
                # Queue for background processing when configured, otherwise fallback to immediate
                queue_backend = (
                    self.config.settings.get("ingestion_queue")
                    or self.config.settings.get("queue_backend")
                    or self.config.settings.get("queue_provider")
                )
                if queue_backend:
                    try:
                        await self._queue_media_job(job_id)
                        queued = True
                    except UnsupportedMediaQueueBackendError as exc:
                        logger.warning("Falling back to inline ingestion for {}: {}", queue_backend, exc)
                        await self._process_media_job(job_id)
                else:
                    await self._process_media_job(job_id)

            return {
                "job_id": job_id,
                "status": "queued" if queued else "processing",
                "url": url,
                "title": title,
                "process_type": process_type
            }

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to ingest media: {e}")
            raise

    async def _update_media(
        self,
        media_id: int,
        updates: dict[str, Any],
        context: Any | None = None,
        **kwargs
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._update_media_sync,
            media_id,
            updates,
            context,
            kwargs,
        )

    def _update_media_sync(
        self,
        media_id: int,
        updates: dict[str, Any],
        context: Any | None = None,
        _kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Update media with validation"""
        try:
            self._assert_media_scope_allowed(media_id, context)
            # Ownership check
            dbi = self._open_media_db(context)
            self._assert_media_access(media_id, context, dbi)
            # Validate media exists
            existing = get_media_by_id(
                dbi,
                media_id,
                include_deleted=False,
                include_trash=False,
            )
            if not existing:
                raise ValueError(f"Media not found: {media_id}")

            updated_fields: list[str] = []
            new_title = updates.get("title")
            new_description = updates.get("description")
            new_tags = updates.get("tags")

            with dbi.transaction() as conn:
                row = dbi._fetchone_with_connection(
                    conn,
                    "SELECT uuid, version, title, content FROM Media WHERE id = ? AND deleted = 0 AND is_trash = 0",
                    (media_id,),
                )
                if not row:
                    raise ValueError(f"Media not found: {media_id}")
                media_uuid = row["uuid"]
                current_version = row["version"]
                current_title = row.get("title")
                current_content = row.get("content")

                new_version = current_version + 1
                now = dbi._get_current_utc_timestamp_str()
                set_parts = ["last_modified = ?", "version = ?", "client_id = ?"]
                params: list[Any] = [now, new_version, dbi.client_id]

                if new_title is not None:
                    set_parts.append("title = ?")
                    params.append(new_title)
                    updated_fields.append("title")

                set_clause = ", ".join(set_parts)
                update_sql_template = "UPDATE Media SET {set_clause} WHERE id = ? AND version = ?"
                update_sql = update_sql_template.format_map(locals())  # nosec B608
                update_params = (*params, media_id, current_version)
                update_cursor = dbi._execute_with_connection(conn, update_sql, update_params)
                if update_cursor.rowcount == 0:
                    raise ValueError(f"Failed to update media {media_id} (conflict)")

                if new_title is not None and new_title != current_title:
                    dbi._update_fts_media(conn, media_id, new_title, current_content or "")

                if new_tags is not None:
                    dbi.update_keywords_for_media(media_id, list(new_tags), conn=conn)
                    updated_fields.append("tags")

                if new_description is not None:
                    if current_content is None:
                        raise ValueError("Cannot update description without media content")
                    dbi.create_document_version(
                        media_id=media_id,
                        content=current_content,
                        prompt=None,
                        analysis_content=new_description,
                    )
                    updated_fields.append("description")

                updated_media_info = dbi._fetchone_with_connection(
                    conn,
                    "SELECT * FROM Media WHERE id = ?",
                    (media_id,),
                ) or {}
                dbi._log_sync_event(conn, "Media", media_uuid, "update", new_version, updated_media_info)

            # Clear cache for this media
            self._clear_media_cache(media_id)

            return {
                "media_id": media_id,
                "updated_fields": updated_fields,
                "success": True,
            }

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to update media: {e}")
            raise

    async def _delete_media(
        self,
        media_id: int,
        permanent: bool = False,
        context: Any | None = None,
        **kwargs
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._delete_media_sync,
            media_id,
            permanent,
            context,
            kwargs,
        )

    def _delete_media_sync(
        self,
        media_id: int,
        permanent: bool = False,
        context: Any | None = None,
        _kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Delete media with soft/hard delete options"""
        try:
            self._assert_media_scope_allowed(media_id, context)
            # Ownership check
            dbi = self._open_media_db(context)
            self._assert_media_access(media_id, context, dbi)
            # Validate media exists
            existing = get_media_by_id(
                dbi,
                media_id,
                include_deleted=False,
                include_trash=False,
            )
            if not existing:
                raise ValueError(f"Media not found: {media_id}")

            if permanent:
                # Hard delete (requires admin)
                if not self._is_admin(context):
                    raise PermissionError("Admin role required for permanent delete")
                deleted = permanently_delete_item(dbi, media_id)
                if not deleted:
                    raise ValueError(f"Media not found: {media_id}")
                action = "permanently_deleted"
            else:
                # Soft delete
                deleted = dbi.soft_delete_media(media_id, cascade=True)
                if not deleted:
                    raise ValueError(f"Media not found: {media_id}")
                action = "soft_deleted"

            # Clear cache
            self._clear_media_cache(media_id)

            return {
                "media_id": media_id,
                "action": action,
                "success": True,
            }

        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to delete media: {e}")
            raise

    async def get_resources(self) -> list[dict[str, Any]]:
        """Get available media resources"""
        return [
            create_resource_definition(
                uri="media://recent",
                name="Recent Media",
                description="Recently added media items",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="media://popular",
                name="Popular Media",
                description="Most accessed media items",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="media://types",
                name="Media Types",
                description="Distinct media types present in the database",
                mime_type="application/json"
            ),
        ]

    async def read_resource(self, uri: str, context: Any | None = None) -> dict[str, Any]:
        """Read media resource"""
        return await asyncio.to_thread(self._read_resource_sync, uri, context)

    def _read_resource_sync(self, uri: str, context: Any | None = None) -> dict[str, Any]:
        dbi = self._open_media_db(context)

        def _rows_to_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            items = []
            for row in rows:
                items.append({
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "type": row.get("type"),
                    "media_type": row.get("type"),
                    "ingestion_date": row.get("ingestion_date"),
                    "last_modified": row.get("last_modified"),
                    "url": row.get("url"),
                })
            return items

        if uri == "media://recent":
            rows, _ = search_media(
                dbi,
                search_query=None,
                search_fields=None,
                media_types=None,
                date_range=None,
                must_have_keywords=None,
                must_not_have_keywords=None,
                sort_by="last_modified_desc",
                media_ids_filter=None,
                page=1,
                results_per_page=20,
                include_trash=False,
                include_deleted=False,
            )
            return {
                "uri": uri,
                "type": "media_list",
                "items": _rows_to_items(rows),
            }

        if uri == "media://popular":
            rows, _ = search_media(
                dbi,
                search_query=None,
                search_fields=None,
                media_types=None,
                date_range=None,
                must_have_keywords=None,
                must_not_have_keywords=None,
                sort_by="date_desc",
                media_ids_filter=None,
                page=1,
                results_per_page=20,
                include_trash=False,
                include_deleted=False,
            )
            return {
                "uri": uri,
                "type": "media_list",
                "items": _rows_to_items(rows),
            }
        if uri == "media://types":
            types = dbi.get_distinct_media_types()
            return {"uri": uri, "type": "media_types", "items": types}

        raise ValueError(f"Unknown resource URI: {uri}")

    # Helper methods

    async def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = []

        for key, value in self._media_cache.items():
            if (current_time - value["time"]).total_seconds() > self._cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._media_cache[key]

    def _clear_media_cache(self, media_id: int):
        """Clear cached search results (all entries)."""
        self._media_cache.clear()

    def _validate_url(self, url: str) -> bool:
        """Validate URL for ingestion with SSRF safeguards.

        Rules:
        - Only http/https schemes
        - Optional allowed_domains allowlist
        - Enforce port allowlist (default 80/443)
        - Reject hosts resolving to private/loopback/link-local/reserved/multicast
        - Reject .local TLDs and empty hosts
        """
        try:
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"}:
                return False
            host = parts.hostname or ""
            if not host:
                return False
            # Disallow .local
            if host.endswith(".local"):
                return False

            # Allowlist check if configured
            allowed_domains = self.config.settings.get("allowed_domains", []) or []
            if allowed_domains:
                host_l = host.lower()
                ok = False
                for d in allowed_domains:
                    d = str(d).lower().lstrip(".")
                    if host_l == d or host_l.endswith("." + d):
                        ok = True
                        break
                if not ok:
                    return False

            # Blocklist check (substring match)
            blocked_domains = self.config.settings.get("blocked_domains", []) or []
            for d in blocked_domains:
                try:
                    if d and d.lower() in host.lower():
                        return False
                except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                    pass

            # Port allowlist (default http/https)
            port = parts.port
            allowed_ports = set(self.config.settings.get("allowed_ports", [80, 443]))
            if port is not None and port not in allowed_ports:
                return False

            # Resolve and reject private/bad ranges
            try:
                addrinfos = socket.getaddrinfo(host, port or (80 if parts.scheme == "http" else 443))
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                return False
            if not addrinfos:
                return False
            for ai in addrinfos:
                try:
                    ip_str = ai[4][0]
                    ip = ipaddress.ip_address(ip_str)
                    if (
                        ip.is_private
                        or ip.is_loopback
                        or ip.is_link_local
                        or ip.is_reserved
                        or ip.is_multicast
                    ):
                        return False
                except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                    return False
            return True
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            return False

    def _is_admin(self, context: Any | None) -> bool:
        try:
            roles = (getattr(context, "metadata", {}) or {}).get("roles")
            return isinstance(roles, list) and any(str(r).lower() == "admin" for r in roles)
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            return False

    def _assert_media_access(self, media_id: int, context: Any | None, dbi: Optional[MediaDbLike] = None) -> None:
        """Enforce that non-admin users can only access their own media when ownership is present."""
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import is_multi_user_mode
            strict_ownership = is_multi_user_mode()
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            strict_ownership = False

        if context is None or getattr(context, "user_id", None) is None:
            if self._allow_anonymous_access() and not strict_ownership:
                return
            raise PermissionError("User context required for media access")
        if self._is_admin(context):
            return
        dbi = dbi or self._open_media_db(context)
        try:
            row = get_media_by_id(
                dbi,
                media_id,
                include_deleted=False,
                include_trash=False,
            )
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS as exc:
            if strict_ownership:
                raise PermissionError("Access denied: ownership lookup failed") from exc
            raise
        if not row:
            return
        owner = row.get("owner_user_id")
        if owner is None:
            if strict_ownership:
                raise PermissionError("Access denied: ownership metadata missing")
            return
        if str(owner) != str(context.user_id):
            raise PermissionError("Access denied for this media item")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        """Stricter validation for high-risk tools."""
        if tool_name == "media.search":
            q = arguments.get("query")
            if not isinstance(q, str) or not (1 <= len(q) <= 1000):
                raise ValueError("query must be 1..1000 chars")
            limit = int(arguments.get("limit", 10))
            offset = int(arguments.get("offset", 0))
            snip = int(arguments.get("snippet_length", 300))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
            if offset < 0:
                raise ValueError("offset must be >= 0")
            if snip < 50 or snip > 2000:
                raise ValueError("snippet_length must be 50..2000")
            order_by = arguments.get("order_by")
            if order_by is not None and order_by not in {"relevance", "recent"}:
                raise ValueError("order_by must be relevance|recent when provided")
            media_types = arguments.get("media_types")
            if media_types is not None and (not isinstance(media_types, list) or any(not isinstance(m, str) for m in media_types)):
                raise ValueError("media_types must be list[str] when provided")
        elif tool_name == "media.get":
            mid = arguments.get("media_id")
            if not isinstance(mid, int) or mid <= 0:
                raise ValueError("media_id must be a positive integer")
            retrieval = arguments.get("retrieval") or {}
            if not isinstance(retrieval, dict):
                raise ValueError("retrieval must be an object")
            mode = retrieval.get("mode", "snippet")
            if mode not in {"snippet", "full", "chunk", "chunk_with_siblings", "auto"}:
                raise ValueError("retrieval.mode invalid")
            snip = int(retrieval.get("snippet_length", 300))
            if snip < 50 or snip > 2000:
                raise ValueError("retrieval.snippet_length must be 50..2000")
            if retrieval.get("chars_per_token") is not None:
                cpt = int(retrieval.get("chars_per_token"))
                if cpt <= 0:
                    raise ValueError("chars_per_token must be a positive integer")
            if retrieval.get("max_tokens") is not None:
                mt = int(retrieval.get("max_tokens"))
                if mt <= 0:
                    raise ValueError("max_tokens must be a positive integer")
            if retrieval.get("chunk_size_tokens") is not None:
                cst = int(retrieval.get("chunk_size_tokens"))
                if cst <= 0:
                    raise ValueError("chunk_size_tokens must be a positive integer")
            if retrieval.get("sibling_window") is not None:
                sw = int(retrieval.get("sibling_window"))
                if sw < 0:
                    raise ValueError("sibling_window must be >= 0")
        elif tool_name == "ingest_media":
            url = arguments.get("url")
            if not isinstance(url, str) or not self._validate_url(url) or len(url) > 2048:
                raise ValueError("url must be a valid http(s) URL <= 2048 chars")
            title = arguments.get("title")
            if title is not None and (not isinstance(title, str) or len(title) > 512):
                raise ValueError("title must be a string <= 512 chars")
            tags = arguments.get("tags")
            if tags is not None:
                if not isinstance(tags, list) or any(not isinstance(t, str) or len(t) > 64 for t in tags):
                    raise ValueError("tags must be list[str] with each tag <= 64 chars")
            process_type = arguments.get("process_type", "transcribe")
            if process_type not in {"transcribe", "summarize", "both", "none"}:
                raise ValueError("process_type invalid")
            priority = arguments.get("priority", "normal")
            if priority not in {"low", "normal", "high"}:
                raise ValueError("priority invalid")

        elif tool_name == "update_media":
            mid = arguments.get("media_id")
            if not isinstance(mid, int) or mid <= 0:
                raise ValueError("media_id must be a positive integer")
            updates = arguments.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("updates must be a non-empty object")
            for k, v in updates.items():
                if k == "title":
                    if not isinstance(v, str) or len(v) > 512:
                        raise ValueError("title must be a string <= 512 chars")
                elif k == "description":
                    if not isinstance(v, str) or len(v) > 4096:
                        raise ValueError("description must be a string <= 4096 chars")
                elif k == "tags":
                    if not isinstance(v, list) or any(not isinstance(t, str) or len(t) > 64 for t in v):
                        raise ValueError("tags must be list[str] with each tag <= 64 chars")
                else:
                    raise ValueError(f"unsupported update field: {k}")

        elif tool_name == "delete_media":
            mid = arguments.get("media_id")
            if not isinstance(mid, int) or mid <= 0:
                raise ValueError("media_id must be a positive integer")
            if "permanent" in arguments and not isinstance(arguments.get("permanent"), bool):
                raise ValueError("permanent must be a boolean")

    async def _create_ingestion_job(self, **kwargs) -> str:
        """Create media ingestion job"""
        # Generate job ID
        job_id = str(uuid.uuid4())
        try:
            job_payload = dict(kwargs)
            job_payload["created_at"] = datetime.utcnow()
            lock = getattr(self, "_ingestion_jobs_lock", None)
            if lock:
                async with lock:
                    self._ingestion_jobs[job_id] = job_payload
            else:
                self._ingestion_jobs[job_id] = job_payload
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            pass
        # Store job details (in production, use job queue)
        # For now, return job ID
        return job_id

    async def _process_media_job(self, job_id: str):
        """Process media ingestion job"""
        job = None
        try:
            lock = getattr(self, "_ingestion_jobs_lock", None)
            if lock:
                async with lock:
                    job = self._ingestion_jobs.get(job_id)
            else:
                job = self._ingestion_jobs.get(job_id)
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            job = None

        if not job:
            raise ValueError(f"Ingestion job not found: {job_id}")

        url = job.get("url")
        if not isinstance(url, str) or not self._validate_url(url):
            raise ValueError("URL failed validation before processing")

        try:
            # Placeholder for actual processing
            await asyncio.sleep(0.1)
        finally:
            try:
                lock = getattr(self, "_ingestion_jobs_lock", None)
                if lock:
                    async with lock:
                        self._ingestion_jobs.pop(job_id, None)
                else:
                    self._ingestion_jobs.pop(job_id, None)
            except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                pass

    async def _queue_media_job(self, job_id: str):
        """Queue media job for background processing"""
        queue_backend = (
            self.config.settings.get("ingestion_queue")
            or self.config.settings.get("queue_backend")
            or self.config.settings.get("queue_provider")
        )
        if not queue_backend:
            raise RuntimeError("Ingestion queue not configured; set ingestion_queue to enable background jobs")
        raise UnsupportedMediaQueueBackendError(f"Ingestion queue backend '{queue_backend}' not implemented")

    async def _get_media_stats(self, media_id: int) -> dict[str, Any]:
        """Get media statistics"""
        return {
            "views": 0,
            "likes": 0,
            "transcriptions": 0,
            "last_accessed": None
        }

    def _format_timestamp(self, total_seconds: Any, ms_separator: str) -> str:
        try:
            total_ms = int(round(max(float(total_seconds or 0.0), 0.0) * 1000))
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            total_ms = 0
        seconds, ms = divmod(total_ms, 1000)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{ms_separator}{ms:03d}"

    def _coerce_time_value(self, value: Any, *, is_ms: bool = False) -> float:
        try:
            val = float(value)
        except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
            return 0.0
        if is_ms:
            val = val / 1000.0
        return max(val, 0.0)

    def _normalize_segments(self, segments: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if not isinstance(segments, list):
            return normalized
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = seg.get("text")
            if text is None:
                text = seg.get("Text")
            if text is None:
                text = seg.get("transcription")
            if text is None:
                text = seg.get("transcript")
            text = str(text or "").strip()

            start = seg.get("start")
            end = seg.get("end")
            if start is None:
                start = seg.get("start_seconds")
            if end is None:
                end = seg.get("end_seconds")
            if start is None:
                start = seg.get("start_time")
            if end is None:
                end = seg.get("end_time")
            if start is None:
                start = seg.get("start_ms")
            if end is None:
                end = seg.get("end_ms")
            is_ms = "start_ms" in seg or "end_ms" in seg
            start_val = self._coerce_time_value(start, is_ms=is_ms)
            end_val = self._coerce_time_value(end, is_ms=is_ms)
            if end_val < start_val:
                end_val = start_val
            normalized.append({"text": text, "start": start_val, "end": end_val})
        return normalized

    def _coerce_transcript_payload(self, transcript_data: Any) -> tuple[str, list[dict[str, Any]]]:
        if transcript_data is None:
            return "", []
        if isinstance(transcript_data, str):
            stripped = transcript_data.strip()
            if stripped and stripped[0] in "[{":
                try:
                    parsed = json.loads(stripped)
                    return self._coerce_transcript_payload(parsed)
                except _MEDIA_MODULE_NONCRITICAL_EXCEPTIONS:
                    pass
            return transcript_data, []
        if isinstance(transcript_data, dict):
            segments = self._normalize_segments(transcript_data.get("segments") or transcript_data.get("Segments") or [])
            text = transcript_data.get("text")
            if text is None:
                text = transcript_data.get("transcription")
            if text is None:
                text = ""
            if not text and segments:
                text = " ".join(seg["text"] for seg in segments if seg.get("text"))
            return str(text), segments
        if isinstance(transcript_data, list):
            segments = self._normalize_segments(transcript_data)
            text = " ".join(seg["text"] for seg in segments if seg.get("text"))
            return text, segments
        return str(transcript_data), []

    def _format_transcript_with_timestamps(self, transcript_data: Any) -> str:
        """Format transcript with timestamps when segment metadata is available."""
        text, segments = self._coerce_transcript_payload(transcript_data)
        if not segments:
            return text
        lines: list[str] = []
        for seg in segments:
            seg_text = seg.get("text") or ""
            if not seg_text:
                continue
            start_ts = self._format_timestamp(seg.get("start", 0.0), ".")
            lines.append(f"{start_ts} {seg_text}")
        return "\n".join(lines).strip()

    def _convert_to_srt(self, transcript_data: Any) -> str:
        """Convert transcript to SRT format."""
        text, segments = self._coerce_transcript_payload(transcript_data)
        if segments:
            lines: list[str] = []
            idx = 1
            for seg in segments:
                seg_text = seg.get("text") or ""
                if not seg_text:
                    continue
                start_ts = self._format_timestamp(seg.get("start", 0.0), ",")
                end_ts = self._format_timestamp(seg.get("end", seg.get("start", 0.0)), ",")
                lines.append(str(idx))
                lines.append(f"{start_ts} --> {end_ts}")
                lines.append(seg_text)
                lines.append("")
                idx += 1
            if lines:
                return "\n".join(lines).rstrip() + "\n"
        if not text:
            return ""
        return f"1\n00:00:00,000 --> 00:00:10,000\n{text}\n"

    def _convert_to_vtt(self, transcript_data: Any) -> str:
        """Convert transcript to WebVTT format."""
        text, segments = self._coerce_transcript_payload(transcript_data)
        if segments:
            lines: list[str] = ["WEBVTT", ""]
            for seg in segments:
                seg_text = seg.get("text") or ""
                if not seg_text:
                    continue
                start_ts = self._format_timestamp(seg.get("start", 0.0), ".")
                end_ts = self._format_timestamp(seg.get("end", seg.get("start", 0.0)), ".")
                lines.append(f"{start_ts} --> {end_ts}")
                lines.append(seg_text)
                lines.append("")
            return "\n".join(lines).rstrip() + "\n"
        if not text:
            return "WEBVTT\n\n"
        return f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{text}\n"
