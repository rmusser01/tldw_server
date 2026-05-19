"""
Notes Module for Unified MCP

FTS-only search and retrieval for user Notes stored in ChaChaNotes DB.
Returns normalized result schema with 0-1 scores and 300-char snippets by default.
"""

import asyncio
from collections.abc import Iterable
from typing import Any, Optional

from loguru import logger

from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB
from ...persona_scope import assert_identifier_in_scope, get_explicit_scope_ids, merge_requested_ids_with_scope
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb

_NOTES_MODULE_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
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
    UnicodeDecodeError,
    ValueError,
)


def _normalize_scores(results: list[dict[str, Any]], score_key: Optional[str] = None) -> list[float]:
    if not results:
        return []
    # Prefer a numeric score if present (e.g., bm25 or ts_rank), otherwise use position-based decay
    if score_key and all(isinstance(r.get(score_key), (int, float)) for r in results):
        vals = [float(r.get(score_key)) for r in results]
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return [1.0 for _ in vals]
        # If this is bm25 (lower is better), invert scale
        if score_key and "bm25" in score_key.lower():
            return [(mx - v) / (mx - mn + 1e-9) for v in vals]
        # Otherwise assume higher is better
        return [(v - mn) / (mx - mn + 1e-9) for v in vals]
    # Positional fallback
    n = len(results)
    if n == 1:
        return [1.0]
    # simple linear decay from 1.0 → ~0.0
    return [1.0 - (i / max(1, n - 1)) for i in range(n)]


def _make_snippet(text: Optional[str], query: Optional[str], length: int = 300) -> str:
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
    except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
        return t[:length]


class NotesModule(BaseModule):
    """FTS search/get over user notes"""

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Notes module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Notes module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        try:
            # Verify DB driver class is importable
            _ = CharactersRAGDB  # noqa: F401
            checks["driver_available"] = True
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            checks["driver_available"] = False
        # Check Databases directory has free space
        try:
            from pathlib import Path
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                base = Path(get_project_root())
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
                # Anchor to package root if project root resolution fails
                base = Path(__file__).resolve().parents[5]
            free_gb = get_free_disk_space_gb(base)
            checks["disk_space"] = free_gb > 1
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            checks["disk_space"] = False
        # Optional ephemeral DB write test (heavy) for deeper validation
        try:
            import os
            if str(os.getenv("MCP_HEALTHCHECK_DB_WRITE_TEST", "")).lower() in {"1", "true", "yes"}:
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(prefix="mcp_notes_health_", suffix=".db", delete=True) as tf:
                    db = CharactersRAGDB(db_path=tf.name, client_id=f"mcp_notes_{self.config.name}")
                    # A trivial read to confirm
                    _ = db.get_note_by_id("nonexistent")
                checks["ephemeral_db_ok"] = True
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            checks["ephemeral_db_ok"] = False

        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="notes.search",
                description="Search notes by title/content (FTS-only).",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                    },
                    "required": ["query"],
                },
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="notes.get",
                description="Retrieve a note by id (snippet or full).",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "retrieval": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string", "enum": ["snippet", "full"], "default": "snippet"},
                                "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                            }
                        }
                    },
                    "required": ["note_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="notes.create",
                description="Create a new note.",
                parameters={
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 512},
                        "content": {"type": "string", "minLength": 1, "maxLength": 500000},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "content"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="notes.update",
                description="Update note title/content.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "updates": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "maxLength": 512},
                                "content": {"type": "string", "maxLength": 500000},
                            },
                        },
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["note_id", "updates"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="notes.delete",
                description="Delete a note (soft delete by default; permanent delete requires admin).",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "permanent": {"type": "boolean", "default": False},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["note_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="notes.tags.add",
                description="Add tags to a note.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["note_id", "tags"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="notes.tags.remove",
                description="Remove tags from a note.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["note_id", "tags"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="notes.tags.set",
                description="Replace tags on a note with the provided list.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["note_id", "tags"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="notes.tags.list",
                description="List tags for a note, or list all tags when note_id is omitted.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve
        if tool_name == "notes.search":
            return await self._search_notes(args, context)
        if tool_name == "notes.get":
            return await self._get_note(args, context)
        if tool_name == "notes.create":
            return await self._create_note(args, context)
        if tool_name == "notes.update":
            return await self._update_note(args, context)
        if tool_name == "notes.delete":
            return await self._delete_note(args, context)
        if tool_name == "notes.tags.add":
            return await self._tags_add(args, context)
        if tool_name == "notes.tags.remove":
            return await self._tags_remove(args, context)
        if tool_name == "notes.tags.set":
            return await self._tags_set(args, context)
        if tool_name == "notes.tags.list":
            return await self._tags_list(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _open_db(self, context: Any) -> CharactersRAGDB:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Notes access")
        chacha_path = context.db_paths.get("chacha")
        if not chacha_path:
            raise ValueError("ChaChaNotes DB path not available in context")
        return CharactersRAGDB(db_path=chacha_path, client_id=f"mcp_notes_{self.config.name}")

    async def _search_notes(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        query: str = args.get("query")
        limit: int = int(args.get("limit", 10))
        offset: int = int(args.get("offset", 0))
        snippet_len: int = int(args.get("snippet_length", 300))
        note_ids_filter = args.get("note_ids_filter")
        # Apply session defaults if present
        try:
            if context and isinstance(getattr(context, "metadata", {}), dict):
                sc = context.metadata.get("safe_config") or {}
                if isinstance(sc, dict):
                    snippet_len = int(sc.get("snippet_length", snippet_len))
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            pass
        snippet_len = max(50, min(2000, snippet_len))

        return await asyncio.to_thread(
            self._search_notes_sync,
            context,
            query,
            limit,
            offset,
            snippet_len,
            note_ids_filter,
        )

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "notes.search":
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
        elif tool_name == "notes.get":
            note_id = arguments.get("note_id")
            if not isinstance(note_id, str) or not note_id:
                raise ValueError("note_id must be a non-empty string")
            retrieval = arguments.get("retrieval") or {}
            if not isinstance(retrieval, dict):
                raise ValueError("retrieval must be an object")
            mode = retrieval.get("mode", "snippet")
            if mode not in {"snippet", "full"}:
                raise ValueError("retrieval.mode must be 'snippet' or 'full'")
            snip = int(retrieval.get("snippet_length", 300))
            if snip < 50 or snip > 2000:
                raise ValueError("retrieval.snippet_length must be 50..2000")
        elif tool_name == "notes.create":
            title = arguments.get("title")
            content = arguments.get("content")
            if not isinstance(title, str) or not (1 <= len(title.strip()) <= 512):
                raise ValueError("title must be 1..512 chars")
            if not isinstance(content, str) or not (1 <= len(content) <= 500000):
                raise ValueError("content must be 1..500000 chars")
            tags = arguments.get("tags")
            if tags is not None:
                self._validate_tags(tags, allow_empty=False)
        elif tool_name == "notes.update":
            note_id = arguments.get("note_id")
            if not isinstance(note_id, str) or not note_id:
                raise ValueError("note_id must be a non-empty string")
            updates = arguments.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("updates must be a non-empty object")
            for k, v in updates.items():
                if k == "title":
                    if not isinstance(v, str) or len(v) > 512:
                        raise ValueError("title must be a string <= 512 chars")
                elif k == "content":
                    if not isinstance(v, str) or len(v) > 500000:
                        raise ValueError("content must be a string <= 500000 chars")
                else:
                    raise ValueError(f"unsupported update field: {k}")
            if arguments.get("expected_version") is not None:
                ev = int(arguments.get("expected_version"))
                if ev <= 0:
                    raise ValueError("expected_version must be a positive integer")
        elif tool_name == "notes.delete":
            note_id = arguments.get("note_id")
            if not isinstance(note_id, str) or not note_id:
                raise ValueError("note_id must be a non-empty string")
            if "permanent" in arguments and not isinstance(arguments.get("permanent"), bool):
                raise ValueError("permanent must be a boolean")
            if arguments.get("expected_version") is not None:
                ev = int(arguments.get("expected_version"))
                if ev <= 0:
                    raise ValueError("expected_version must be a positive integer")
        elif tool_name in {"notes.tags.add", "notes.tags.remove"}:
            note_id = arguments.get("note_id")
            if not isinstance(note_id, str) or not note_id:
                raise ValueError("note_id must be a non-empty string")
            tags = arguments.get("tags")
            self._validate_tags(tags, allow_empty=False)
        elif tool_name == "notes.tags.set":
            note_id = arguments.get("note_id")
            if not isinstance(note_id, str) or not note_id:
                raise ValueError("note_id must be a non-empty string")
            tags = arguments.get("tags")
            self._validate_tags(tags, allow_empty=True)
        elif tool_name == "notes.tags.list":
            note_id = arguments.get("note_id")
            if note_id is not None and (not isinstance(note_id, str) or not note_id):
                raise ValueError("note_id must be a non-empty string when provided")
            limit = int(arguments.get("limit", 50)) if arguments.get("limit") is not None else 50
            offset = int(arguments.get("offset", 0)) if arguments.get("offset") is not None else 0
            if limit < 1 or limit > 200:
                raise ValueError("limit must be 1..200")
            if offset < 0:
                raise ValueError("offset must be >= 0")

    async def _get_note(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id: str = args.get("note_id")
        retrieval = args.get("retrieval") or {}
        note_ids_filter = args.get("note_ids_filter")
        mode = (retrieval or {}).get("mode", "snippet")
        snippet_len = int((retrieval or {}).get("snippet_length", 300))
        # Apply session defaults if present
        try:
            if context and isinstance(getattr(context, "metadata", {}), dict):
                sc = context.metadata.get("safe_config") or {}
                if isinstance(sc, dict):
                    snippet_len = int(sc.get("snippet_length", snippet_len))
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            pass
        snippet_len = max(50, min(2000, snippet_len))

        return await asyncio.to_thread(
            self._get_note_sync,
            context,
            note_id,
            mode,
            snippet_len,
            note_ids_filter,
        )

    async def _create_note(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        title = args.get("title")
        content = args.get("content")
        tags = args.get("tags") or []
        return await asyncio.to_thread(self._create_note_sync, context, title, content, tags)

    async def _update_note(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id = args.get("note_id")
        updates = args.get("updates") or {}
        expected_version = args.get("expected_version")
        return await asyncio.to_thread(self._update_note_sync, context, note_id, updates, expected_version)

    async def _delete_note(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id = args.get("note_id")
        permanent = bool(args.get("permanent", False))
        expected_version = args.get("expected_version")
        return await asyncio.to_thread(self._delete_note_sync, context, note_id, permanent, expected_version)

    async def _tags_add(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id = args.get("note_id")
        tags = args.get("tags") or []
        return await asyncio.to_thread(self._tags_add_sync, context, note_id, tags)

    async def _tags_remove(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id = args.get("note_id")
        tags = args.get("tags") or []
        return await asyncio.to_thread(self._tags_remove_sync, context, note_id, tags)

    async def _tags_set(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id = args.get("note_id")
        tags = args.get("tags") or []
        return await asyncio.to_thread(self._tags_set_sync, context, note_id, tags)

    async def _tags_list(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        note_id = args.get("note_id")
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        return await asyncio.to_thread(self._tags_list_sync, context, note_id, limit, offset)

    def _search_notes_sync(
        self,
        context: Any | None,
        query: str,
        limit: int,
        offset: int,
        snippet_len: int,
        note_ids_filter: Any,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            scoped_note_ids = get_explicit_scope_ids(context, "note_id")
            effective_note_ids = merge_requested_ids_with_scope(
                note_ids_filter,
                scoped_ids=scoped_note_ids,
            )
            if effective_note_ids is not None and not effective_note_ids:
                return {
                    "results": [],
                    "has_more": False,
                    "next_offset": None,
                    "total_estimated": 0,
                }

            fetch_limit = limit + 1  # fetch one extra row to detect additional pages
            if effective_note_ids is not None:
                fetch_limit = min(max(limit + offset + 1, (limit + offset + 1) * 5), 1000)
            raw = db.search_notes(query, limit=fetch_limit, offset=0 if effective_note_ids is not None else offset)
            filtered_rows = raw
            if effective_note_ids is not None:
                filtered_rows = [row for row in raw if str(row.get("id") or "") in effective_note_ids]
                rows = filtered_rows[offset: offset + limit]
                has_more = len(filtered_rows) > (offset + limit)
            else:
                rows = filtered_rows[:limit]
                has_more = len(raw) > limit
            # Detect score key if backend provided it
            score_key = None
            if rows:
                first = rows[0]
                if isinstance(first.get("rank"), (int, float)):
                    score_key = "rank"
                elif isinstance(first.get("bm25_score"), (int, float)):
                    score_key = "bm25_score"
            scores = _normalize_scores(rows, score_key=score_key)

            next_offset = (offset + len(rows)) if has_more else None

            results = []
            for i, r in enumerate(rows):
                note_id = r.get("id")
                title = r.get("title")
                content = r.get("content") or ""
                created_at = r.get("created_at")
                last_modified = r.get("last_modified") or r.get("updated_at")
                # Approximate offset of query within content
                approx_offset = None
                try:
                    idx = content.lower().find(query.lower()) if query else -1
                    if idx >= 0:
                        approx_offset = idx
                except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
                    approx_offset = None
                results.append({
                    "id": note_id,
                    "source": "notes",
                    "title": title,
                    "snippet": _make_snippet(content, query, snippet_len),
                    "uri": f"notes://{note_id}",
                    "score": float(scores[i] if i < len(scores) else 0.0),
                    "score_type": "fts",
                    "created_at": created_at,
                    "last_modified": last_modified,
                    "version": r.get("version"),
                    "tags": None,
                    "loc": ({"approx_offset": approx_offset} if approx_offset is not None else None),
                })

            try:
                total_estimated = db.count_notes_matching(query)
                if effective_note_ids is not None:
                    total_estimated = min(total_estimated, len(filtered_rows))
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
                total_estimated = offset + len(rows) + (1 if has_more else 0)

            return {
                "results": results,
                "has_more": has_more,
                "next_offset": next_offset,
                "total_estimated": total_estimated,
            }
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after search: {}", exc)

    def _get_note_sync(
        self,
        context: Any | None,
        note_id: str,
        mode: str,
        snippet_len: int,
        note_ids_filter: Any,
    ) -> dict[str, Any]:
        scoped_note_ids = get_explicit_scope_ids(context, "note_id")
        effective_note_ids = merge_requested_ids_with_scope(
            note_ids_filter,
            scoped_ids=scoped_note_ids,
        )
        if effective_note_ids is not None and str(note_id or "") not in effective_note_ids:
            raise PermissionError("Note access denied by persona scope")

        db = self._open_db(context)
        try:
            row = db.get_note_by_id(note_id)
            if not row:
                raise ValueError(f"Note not found: {note_id}")
            content = row.get("content") or ""
            meta = {
                "id": row.get("id"),
                "source": "notes",
                "title": row.get("title"),
                "snippet": _make_snippet(content, None, snippet_len),
                "uri": f"notes://{row.get('id')}",
                "score": 1.0,
                "score_type": "fts",
                "created_at": row.get("created_at"),
                "last_modified": row.get("last_modified") or row.get("updated_at"),
                "version": row.get("version"),
                "tags": None,
                "loc": None,
            }

            body = content if mode == "full" else _make_snippet(content, None, snippet_len)

            return {
                "meta": meta,
                "content": body,
                "attachments": None,
            }
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after note fetch: {}", exc)

    def _create_note_sync(
        self,
        context: Any | None,
        title: str,
        content: str,
        tags: Iterable[str],
    ) -> dict[str, Any]:
        scoped_note_ids = get_explicit_scope_ids(context, "note_id")
        if scoped_note_ids is not None:
            raise PermissionError("Cannot create a note outside explicit persona scope")
        db = self._open_db(context)
        try:
            note_id = db.add_note(title=title, content=content)
            if not note_id:
                raise ValueError("Failed to create note")
            norm_tags = self._normalize_tags(tags)
            if norm_tags:
                self._apply_tags(db, note_id, norm_tags)
            row = db.get_note_by_id(note_id)
            if not row:
                raise ValueError("Created note not found")
            meta = self._build_note_meta(row, snippet_len=300)
            return {"note_id": note_id, "success": True, "meta": meta}
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after create: {}", exc)

    def _update_note_sync(
        self,
        context: Any | None,
        note_id: str,
        updates: dict[str, Any],
        expected_version: Any,
    ) -> dict[str, Any]:
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        db = self._open_db(context)
        try:
            row = db.get_note_by_id(note_id)
            if not row:
                raise ValueError(f"Note not found: {note_id}")
            current_version = int(row.get("version") or 1)
            ev = int(expected_version) if expected_version is not None else current_version
            updated_fields = list(updates.keys())
            db.update_note(note_id, updates, ev)
            return {"note_id": note_id, "updated_fields": updated_fields, "success": True}
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after update: {}", exc)

    def _delete_note_sync(
        self,
        context: Any | None,
        note_id: str,
        permanent: bool,
        expected_version: Any,
    ) -> dict[str, Any]:
        if permanent and not self._is_admin(context):
            raise PermissionError("Admin role required for permanent delete")
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        db = self._open_db(context)
        try:
            row = db.get_note_by_id(note_id)
            if not row and not permanent:
                raise ValueError(f"Note not found: {note_id}")
            current_version = int(row.get("version") or 1) if row else None
            ev = int(expected_version) if expected_version is not None else current_version
            deleted = db.delete_note(note_id, expected_version=ev, hard_delete=permanent)
            if not deleted:
                raise ValueError(f"Note not found: {note_id}")
            return {
                "note_id": note_id,
                "action": "permanently_deleted" if permanent else "soft_deleted",
                "success": True,
            }
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after delete: {}", exc)

    def _tags_add_sync(self, context: Any | None, note_id: str, tags: Iterable[str]) -> dict[str, Any]:
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        db = self._open_db(context)
        try:
            if not db.get_note_by_id(note_id):
                raise ValueError(f"Note not found: {note_id}")
            norm_tags = self._normalize_tags(tags)
            if norm_tags:
                existing = {t.lower() for t in self._tags_for_note(db, note_id)}
                for tag in norm_tags:
                    if tag in existing:
                        continue
                    kid = self._ensure_keyword(db, tag)
                    if kid is not None:
                        db.link_note_to_keyword(note_id, int(kid))
            return {"note_id": note_id, "tags": self._tags_for_note(db, note_id), "success": True}
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after tags add: {}", exc)

    def _tags_remove_sync(self, context: Any | None, note_id: str, tags: Iterable[str]) -> dict[str, Any]:
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        db = self._open_db(context)
        try:
            if not db.get_note_by_id(note_id):
                raise ValueError(f"Note not found: {note_id}")
            norm_tags = self._normalize_tags(tags)
            for tag in norm_tags:
                kw = db.get_keyword_by_text(tag)
                if kw and kw.get("id") is not None:
                    db.unlink_note_from_keyword(note_id, int(kw["id"]))
            return {"note_id": note_id, "tags": self._tags_for_note(db, note_id), "success": True}
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after tags remove: {}", exc)

    def _tags_set_sync(self, context: Any | None, note_id: str, tags: Iterable[str]) -> dict[str, Any]:
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        db = self._open_db(context)
        try:
            if not db.get_note_by_id(note_id):
                raise ValueError(f"Note not found: {note_id}")
            desired = set(self._normalize_tags(tags))
            existing_rows = db.get_keywords_for_note(note_id)
            existing = {str(r.get("keyword")).lower(): int(r.get("id")) for r in existing_rows if r.get("keyword") is not None}

            # Remove tags not desired
            for tag, kid in existing.items():
                if tag not in desired:
                    db.unlink_note_from_keyword(note_id, int(kid))

            # Add missing tags
            for tag in desired:
                if tag in existing:
                    continue
                kid = self._ensure_keyword(db, tag)
                if kid is not None:
                    db.link_note_to_keyword(note_id, int(kid))

            return {"note_id": note_id, "tags": self._tags_for_note(db, note_id), "success": True}
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after tags set: {}", exc)

    def _tags_list_sync(self, context: Any | None, note_id: Optional[str], limit: int, offset: int) -> dict[str, Any]:
        scoped_note_ids = get_explicit_scope_ids(context, "note_id")
        if note_id is None and scoped_note_ids is not None:
            raise PermissionError("Cannot list all tags outside explicit persona scope")
        if note_id is not None:
            assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        db = self._open_db(context)
        try:
            if note_id:
                if not db.get_note_by_id(note_id):
                    raise ValueError(f"Note not found: {note_id}")
                tags = self._tags_for_note(db, note_id)
                sliced = tags[offset: offset + limit]
                has_more = (offset + len(sliced)) < len(tags)
                return {
                    "note_id": note_id,
                    "tags": sliced,
                    "has_more": has_more,
                    "next_offset": (offset + len(sliced)) if has_more else None,
                }
            rows = db.list_keywords(limit=limit, offset=offset)
            tags = [str(r.get("keyword")) for r in rows if r.get("keyword") is not None]
            total = db.count_keywords()
            has_more = (offset + len(tags)) < total
            return {
                "note_id": None,
                "tags": tags,
                "has_more": has_more,
                "next_offset": (offset + len(tags)) if has_more else None,
            }
        finally:
            try:
                db.close_all_connections()
            except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after tags list: {}", exc)

    def _build_note_meta(self, row: dict[str, Any], snippet_len: int = 300) -> dict[str, Any]:
        content = row.get("content") or ""
        return {
            "id": row.get("id"),
            "source": "notes",
            "title": row.get("title"),
            "snippet": _make_snippet(content, None, snippet_len),
            "uri": f"notes://{row.get('id')}",
            "score": 1.0,
            "score_type": "fts",
            "created_at": row.get("created_at"),
            "last_modified": row.get("last_modified") or row.get("updated_at"),
            "version": row.get("version"),
            "tags": None,
            "loc": None,
        }

    def _is_admin(self, context: Any | None) -> bool:
        try:
            roles = (getattr(context, "metadata", {}) or {}).get("roles")
            return isinstance(roles, list) and any(str(r).lower() == "admin" for r in roles)
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            return False

    def _validate_tags(self, tags: Any, *, allow_empty: bool) -> None:
        if not isinstance(tags, list):
            raise ValueError("tags must be a list of strings")
        if not tags and not allow_empty:
            raise ValueError("tags cannot be empty")
        for t in tags:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("tags must be non-empty strings")
            if len(t.strip()) > 64:
                raise ValueError("each tag must be <= 64 chars")
        if len(tags) > 50:
            raise ValueError("tags must contain <= 50 items")

    def _normalize_tags(self, tags: Iterable[str]) -> list[str]:
        out: list[str] = []
        seen = set()
        for t in tags or []:
            if not isinstance(t, str):
                continue
            norm = t.strip().lower()
            if not norm:
                continue
            if len(norm) > 64:
                raise ValueError("each tag must be <= 64 chars")
            if norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
            if len(out) > 50:
                raise ValueError("tags must contain <= 50 items")
        return out

    def _ensure_keyword(self, db: CharactersRAGDB, tag: str) -> Optional[int]:
        try:
            existing = db.get_keyword_by_text(tag)
            if existing and existing.get("id") is not None:
                return int(existing["id"])
            kid = db.add_keyword(tag)
            return int(kid) if kid is not None else None
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            return None

    def _apply_tags(self, db: CharactersRAGDB, note_id: str, tags: list[str]) -> None:
        for tag in tags:
            kid = self._ensure_keyword(db, tag)
            if kid is None:
                continue
            db.link_note_to_keyword(note_id, int(kid))

    def _tags_for_note(self, db: CharactersRAGDB, note_id: str) -> list[str]:
        rows = db.get_keywords_for_note(note_id)
        return [str(r.get("keyword")).lower() for r in rows if r.get("keyword") is not None]
