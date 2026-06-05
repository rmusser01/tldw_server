"""
Notes Module for Unified MCP

FTS-only search and retrieval for user Notes stored in ChaChaNotes DB.
Returns normalized result schema with 0-1 scores and 300-char snippets by default.
"""

import asyncio
import copy
import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from datetime import date
from typing import Any, Optional

from loguru import logger

from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from ....Notes_Tasks import NotesTaskService, TaskActor
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

_TASK_WRITE_TOOLS = {
    "notes.tasks.create",
    "notes.tasks.update",
    "notes.tasks.set_status",
    "notes.tasks.delete",
    "notes.tasks.reconcile_note",
}
_TASK_IDEMPOTENT_WRITE_TOOLS = {
    "notes.tasks.create",
    "notes.tasks.update",
    "notes.tasks.set_status",
    "notes.tasks.delete",
}
_TASK_STATUSES = {"open", "done"}
_TASK_PROJECTION_STATUSES = {"live", "unlinked", "ambiguous", "deleted"}
_TASK_METADATA_KEYS = {"due_date", "priority", "estimate"}
_TASK_PRIORITY_VALUES = {"high", "medium", "low"}
_TASK_MAX_TEXT_CHARS = 2000
_TASK_MAX_BATCH_UPDATES = 50
_TASK_IDEMPOTENCY_CACHE_SIZE = 128
_TASK_LIST_MAX_OFFSET = 500
_TASK_LIST_MAX_QUERY_CHARS = 1000


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_service = NotesTaskService()
        self._task_idempotency_cache: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self._task_idempotency_lock = asyncio.Lock()

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
        tools = [
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
        tools.extend(self._task_tool_definitions())
        return tools

    def is_write_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        if tool_name == "notes.tasks.list":
            return self._is_task_list_write_call(arguments)
        return super().is_write_tool_call(tool_name, arguments, tool_def=tool_def)

    def _task_tool_definitions(self) -> list[dict[str, Any]]:
        write_metadata = {
            "category": "management",
            "auth_required": True,
            "requires_confirmation": True,
            "agent_write_policy": "approval_required",
            "autonomous_writes": "denied",
            "governance_preflight_required": True,
            "sensitive": True,
        }

        def strict_tool(
            *,
            name: str,
            description: str,
            parameters: dict[str, Any],
            metadata: dict[str, Any],
        ) -> dict[str, Any]:
            tool = create_tool_definition(
                name=name,
                description=description,
                parameters=parameters,
                metadata=metadata,
            )
            tool["inputSchema"]["additionalProperties"] = False
            return tool

        metadata_schema = {
            "type": "object",
            "properties": {
                "due_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
                "priority": {"type": "string", "enum": sorted(_TASK_PRIORITY_VALUES)},
                "estimate": {"type": "string", "pattern": r"^\d+[mhd]$"},
            },
            "additionalProperties": False,
        }
        status_update_schema = {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
                "status": {"type": "string", "enum": sorted(_TASK_STATUSES)},
                "expected_task_version": {"type": "integer", "minimum": 1},
                "expected_note_version": {"type": "integer", "minimum": 1},
                "record_only": {"type": "boolean", "default": False},
            },
            "required": ["task_id", "status", "expected_task_version", "expected_note_version"],
            "additionalProperties": False,
        }
        insertion_schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["append"], "default": "append"},
            },
            "additionalProperties": False,
        }
        idempotency_schema = {"type": "string", "minLength": 1, "maxLength": 256}

        return [
            strict_tool(
                name="notes.tasks.list",
                description=(
                    "List note-backed tasks with reconciliation-aware discovery. "
                    "Set reconcile_limit=0 for an explicitly read-only list."
                ),
                parameters={
                    "properties": {
                        "note_id": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": sorted(_TASK_STATUSES)},
                        "projection_status": {"type": "string", "enum": sorted(_TASK_PROJECTION_STATUSES)},
                        "query": {"type": "string", "minLength": 1, "maxLength": _TASK_LIST_MAX_QUERY_CHARS},
                        "metadata_filters": metadata_schema,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                        "offset": {"type": "integer", "minimum": 0, "maximum": _TASK_LIST_MAX_OFFSET, "default": 0},
                        "include_unlinked": {"type": "boolean", "default": False},
                        "reconcile_limit": {"type": "integer", "minimum": 0, "maximum": 100, "default": 25},
                    },
                    "required": [],
                },
                metadata={**write_metadata, "readOnlyHint": False},
            ),
            strict_tool(
                name="notes.tasks.get",
                description="Retrieve one note-backed task by id.",
                parameters={
                    "properties": {
                        "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    },
                    "required": ["task_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            strict_tool(
                name="notes.tasks.create",
                description="Create a task by appending a checklist line to a note.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "minLength": 1, "maxLength": _TASK_MAX_TEXT_CHARS},
                        "status": {"type": "string", "enum": sorted(_TASK_STATUSES), "default": "open"},
                        "metadata": metadata_schema,
                        "expected_note_version": {"type": "integer", "minimum": 1},
                        "insertion": insertion_schema,
                        "idempotencyKey": idempotency_schema,
                        "idempotency_key": idempotency_schema,
                    },
                    "required": ["note_id", "text", "expected_note_version"],
                },
                metadata=write_metadata,
            ),
            strict_tool(
                name="notes.tasks.update",
                description="Update projected task text or metadata.",
                parameters={
                    "properties": {
                        "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "text": {"type": "string", "minLength": 1, "maxLength": _TASK_MAX_TEXT_CHARS},
                        "metadata": metadata_schema,
                        "expected_task_version": {"type": "integer", "minimum": 1},
                        "expected_note_version": {"type": "integer", "minimum": 1},
                        "record_only": {"type": "boolean", "default": False},
                        "idempotencyKey": idempotency_schema,
                        "idempotency_key": idempotency_schema,
                    },
                    "required": ["task_id", "expected_task_version", "expected_note_version"],
                },
                metadata=write_metadata,
            ),
            strict_tool(
                name="notes.tasks.set_status",
                description="Set one or more task statuses and return succeeded, failed, and skipped items.",
                parameters={
                    "properties": {
                        "updates": {
                            "type": "array",
                            "items": status_update_schema,
                            "minItems": 1,
                            "maxItems": _TASK_MAX_BATCH_UPDATES,
                        },
                        "items": {
                            "type": "array",
                            "items": status_update_schema,
                            "minItems": 1,
                            "maxItems": _TASK_MAX_BATCH_UPDATES,
                        },
                        "idempotencyKey": idempotency_schema,
                        "idempotency_key": idempotency_schema,
                    },
                    "required": [],
                },
                metadata=write_metadata,
            ),
            strict_tool(
                name="notes.tasks.delete",
                description="Delete a note-backed task, updating the projected note when required.",
                parameters={
                    "properties": {
                        "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
                        "expected_task_version": {"type": "integer", "minimum": 1},
                        "expected_note_version": {"type": "integer", "minimum": 1},
                        "record_only": {"type": "boolean", "default": False},
                        "record_only_if_unlinked": {"type": "boolean", "default": False},
                        "idempotencyKey": idempotency_schema,
                        "idempotency_key": idempotency_schema,
                    },
                    "required": ["task_id", "expected_task_version", "expected_note_version"],
                },
                metadata=write_metadata,
            ),
            strict_tool(
                name="notes.tasks.reconcile_note",
                description="Reconcile the current checklist projections for one note.",
                parameters={
                    "properties": {
                        "note_id": {"type": "string", "minLength": 1},
                        "expected_note_version": {"type": "integer", "minimum": 1},
                        "work_limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["note_id"],
                },
                metadata=write_metadata,
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
        if tool_name == "notes.tasks.list":
            if self._is_task_list_write_call(args):
                policy_decision = self._preflight_task_agent_write_policy(tool_name, args, context)
                if policy_decision is not None:
                    return policy_decision
            return await self._tasks_list(args, context)
        if tool_name == "notes.tasks.get":
            return await self._tasks_get(args, context)
        if tool_name in _TASK_WRITE_TOOLS:
            policy_decision = self._preflight_task_agent_write_policy(tool_name, args, context)
            if policy_decision is not None:
                return policy_decision
            return await self._execute_task_write(tool_name, args, context)
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
        elif tool_name == "notes.tasks.list":
            self._validate_task_list_arguments(arguments)
        elif tool_name == "notes.tasks.get":
            self._validate_task_allowed_fields(arguments, {"task_id"})
            self._validate_required_text(arguments, "task_id", max_length=128)
        elif tool_name == "notes.tasks.create":
            self._validate_task_allowed_fields(
                arguments,
                {
                    "note_id",
                    "text",
                    "status",
                    "metadata",
                    "expected_note_version",
                    "insertion",
                    "idempotencyKey",
                    "idempotency_key",
                },
            )
            self._validate_required_text(arguments, "note_id")
            self._validate_task_text(arguments.get("text"))
            self._validate_task_status(arguments.get("status", "open"))
            self._validate_task_metadata(arguments.get("metadata") or {})
            self._validate_expected_version(arguments.get("expected_note_version"), "expected_note_version")
            self._validate_task_insertion(arguments.get("insertion"))
            self._validate_optional_idempotency_key(arguments)
        elif tool_name == "notes.tasks.update":
            self._validate_task_allowed_fields(
                arguments,
                {
                    "task_id",
                    "text",
                    "metadata",
                    "expected_task_version",
                    "expected_note_version",
                    "record_only",
                    "idempotencyKey",
                    "idempotency_key",
                },
            )
            self._validate_required_text(arguments, "task_id", max_length=128)
            has_text = "text" in arguments and arguments.get("text") is not None
            has_metadata = "metadata" in arguments and arguments.get("metadata") is not None
            if not has_text and not has_metadata:
                raise ValueError("At least one task field must be provided")
            if has_text:
                self._validate_task_text(arguments.get("text"))
            if has_metadata:
                self._validate_task_metadata(arguments.get("metadata"))
            self._validate_expected_version(arguments.get("expected_task_version"), "expected_task_version")
            self._validate_expected_version(arguments.get("expected_note_version"), "expected_note_version")
            self._validate_optional_bool(arguments, "record_only")
            self._validate_optional_idempotency_key(arguments)
        elif tool_name == "notes.tasks.set_status":
            self._validate_task_allowed_fields(arguments, {"updates", "items", "idempotencyKey", "idempotency_key"})
            self._validate_task_status_batch_arguments(arguments)
            self._validate_optional_idempotency_key(arguments)
        elif tool_name == "notes.tasks.delete":
            self._validate_task_allowed_fields(
                arguments,
                {
                    "task_id",
                    "expected_task_version",
                    "expected_note_version",
                    "record_only",
                    "record_only_if_unlinked",
                    "idempotencyKey",
                    "idempotency_key",
                },
            )
            self._validate_required_text(arguments, "task_id", max_length=128)
            self._validate_expected_version(arguments.get("expected_task_version"), "expected_task_version")
            self._validate_expected_version(arguments.get("expected_note_version"), "expected_note_version")
            self._validate_optional_bool(arguments, "record_only")
            self._validate_optional_bool(arguments, "record_only_if_unlinked")
            if "record_only" in arguments and "record_only_if_unlinked" in arguments:
                raise ValueError("Use only one of record_only_if_unlinked or record_only")
            self._validate_optional_idempotency_key(arguments)
        elif tool_name == "notes.tasks.reconcile_note":
            self._validate_task_allowed_fields(arguments, {"note_id", "expected_note_version", "work_limit"})
            self._validate_required_text(arguments, "note_id")
            if arguments.get("expected_note_version") is not None:
                self._validate_expected_version(arguments.get("expected_note_version"), "expected_note_version")
            if arguments.get("work_limit") is not None:
                self._validate_work_limit(arguments.get("work_limit"))

    @staticmethod
    def _validate_required_text(arguments: dict[str, Any], field: str, *, max_length: int | None = None) -> None:
        value = arguments.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        if max_length is not None and len(value) > max_length:
            raise ValueError(f"{field} must be <= {max_length} chars")

    @staticmethod
    def _validate_expected_version(value: Any, field: str) -> None:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a positive integer")
        try:
            version = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive integer") from exc
        if version <= 0:
            raise ValueError(f"{field} must be a positive integer")

    @staticmethod
    def _validate_optional_bool(arguments: dict[str, Any], field: str) -> None:
        if field in arguments and not isinstance(arguments.get(field), bool):
            raise ValueError(f"{field} must be a boolean")

    @staticmethod
    def _validate_task_allowed_fields(arguments: dict[str, Any], allowed: set[str]) -> None:
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"unsupported task argument: {', '.join(unknown)}")

    @staticmethod
    def _validate_task_status(status: Any) -> None:
        if status not in _TASK_STATUSES:
            raise ValueError("status must be 'open' or 'done'")

    @staticmethod
    def _validate_task_text(text: Any) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > _TASK_MAX_TEXT_CHARS:
            raise ValueError(f"text must be <= {_TASK_MAX_TEXT_CHARS} chars")
        if "\n" in text or "\r" in text:
            raise ValueError("text cannot contain newline characters")

    @staticmethod
    def _validate_task_metadata(metadata: Any) -> None:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        unknown = sorted(set(metadata) - _TASK_METADATA_KEYS)
        if unknown:
            raise ValueError(f"metadata contains unsupported keys: {', '.join(unknown)}")
        due_date = metadata.get("due_date")
        if due_date is not None:
            if not isinstance(due_date, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", due_date) is None:
                raise ValueError("metadata.due_date must use YYYY-MM-DD format")
            try:
                date.fromisoformat(due_date)
            except ValueError as exc:
                raise ValueError("metadata.due_date must be a real ISO date") from exc
        priority = metadata.get("priority")
        if priority is not None and priority not in _TASK_PRIORITY_VALUES:
            raise ValueError("metadata.priority must be high, medium, or low")
        estimate = metadata.get("estimate")
        if estimate is not None and (not isinstance(estimate, str) or re.fullmatch(r"\d+[mhd]", estimate) is None):
            raise ValueError("metadata.estimate must match '<number><m|h|d>'")

    @staticmethod
    def _validate_task_insertion(insertion: Any) -> None:
        if insertion is None:
            return
        if insertion == "append":
            return
        if not isinstance(insertion, dict):
            raise ValueError("insertion currently supports append mode only")
        unknown = sorted(set(insertion) - {"mode"})
        if unknown:
            raise ValueError(f"insertion contains unsupported keys: {', '.join(unknown)}")
        mode = insertion.get("mode", "append")
        if mode != "append":
            raise ValueError("insertion currently supports append mode only")

    @staticmethod
    def _validate_work_limit(value: Any) -> None:
        if isinstance(value, bool):
            raise ValueError("work_limit must be an integer 1..100")
        try:
            work_limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("work_limit must be an integer 1..100") from exc
        if work_limit < 1 or work_limit > 100:
            raise ValueError("work_limit must be an integer 1..100")

    def _validate_task_list_arguments(self, arguments: dict[str, Any]) -> None:
        self._validate_task_allowed_fields(
            arguments,
            {
                "note_id",
                "status",
                "projection_status",
                "query",
                "metadata_filters",
                "limit",
                "offset",
                "include_unlinked",
                "reconcile_limit",
            },
        )
        note_id = arguments.get("note_id")
        if note_id is not None and (not isinstance(note_id, str) or not note_id.strip()):
            raise ValueError("note_id must be a non-empty string when provided")
        status = arguments.get("status")
        if status is not None:
            self._validate_task_status(status)
        projection_status = arguments.get("projection_status")
        if projection_status is not None and projection_status not in _TASK_PROJECTION_STATUSES:
            raise ValueError("projection_status must be live, unlinked, ambiguous, or deleted")
        query = arguments.get("query")
        if query is not None:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string when provided")
            if len(query) > _TASK_LIST_MAX_QUERY_CHARS:
                raise ValueError(f"query must be <= {_TASK_LIST_MAX_QUERY_CHARS} chars")
        if arguments.get("metadata_filters") is not None:
            self._validate_task_metadata(arguments.get("metadata_filters"))
        self._validate_optional_bool(arguments, "include_unlinked")
        try:
            limit = int(arguments.get("limit", 100))
            offset = int(arguments.get("offset", 0))
            reconcile_limit = int(arguments.get("reconcile_limit", 25))
        except (TypeError, ValueError) as exc:
            raise ValueError("limit, offset, and reconcile_limit must be integers") from exc
        if limit < 1 or limit > 500:
            raise ValueError("limit must be 1..500")
        if offset < 0 or offset > _TASK_LIST_MAX_OFFSET:
            raise ValueError(f"offset must be 0..{_TASK_LIST_MAX_OFFSET}")
        if reconcile_limit < 0 or reconcile_limit > 100:
            raise ValueError("reconcile_limit must be 0..100")

    def _validate_task_status_batch_arguments(self, arguments: dict[str, Any]) -> None:
        updates = self._task_status_updates(arguments)
        if not isinstance(updates, list) or not updates:
            raise ValueError("updates must be a non-empty list")
        if len(updates) > _TASK_MAX_BATCH_UPDATES:
            raise ValueError(f"updates must contain <= {_TASK_MAX_BATCH_UPDATES} items")
        for index, item in enumerate(updates):
            if not isinstance(item, dict):
                raise ValueError(f"updates[{index}] must be an object")
            self._validate_task_allowed_fields(
                item,
                {"task_id", "status", "expected_task_version", "expected_note_version", "record_only"},
            )
            self._validate_required_text(item, "task_id", max_length=128)
            self._validate_task_status(item.get("status"))
            self._validate_expected_version(item.get("expected_task_version"), "expected_task_version")
            self._validate_expected_version(item.get("expected_note_version"), "expected_note_version")
            self._validate_optional_bool(item, "record_only")

    @staticmethod
    def _is_task_list_write_call(arguments: dict[str, Any] | None) -> bool:
        args = arguments if isinstance(arguments, dict) else {}
        raw_limit = args.get("reconcile_limit", 25)
        if isinstance(raw_limit, bool):
            return True
        try:
            return int(raw_limit) > 0
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _task_status_updates(arguments: dict[str, Any]) -> Any:
        has_updates = "updates" in arguments and arguments.get("updates") is not None
        has_items = "items" in arguments and arguments.get("items") is not None
        if has_updates and has_items:
            raise ValueError("Use only one of updates or items")
        return arguments.get("items") if has_items else arguments.get("updates")

    def _validate_optional_idempotency_key(self, arguments: dict[str, Any]) -> None:
        key = self._get_idempotency_key(arguments)
        if key is None:
            return
        if not isinstance(key, str) or not key.strip() or len(key) > 256:
            raise ValueError("idempotencyKey must be a non-empty string <= 256 chars")

    async def _tasks_list(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._tasks_list_sync, context, args)

    async def _tasks_get(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._tasks_get_sync, context, args)

    async def _execute_task_write(self, tool_name: str, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        idempotency_key = self._get_idempotency_key(args)
        if tool_name in _TASK_IDEMPOTENT_WRITE_TOOLS and isinstance(idempotency_key, str) and idempotency_key:
            async with self._task_idempotency_lock:
                cache_key = self._task_idempotency_cache_key(context, tool_name, idempotency_key)
                fingerprint = self._task_arguments_fingerprint(args)
                cached = self._task_idempotency_cache.get(cache_key)
                if cached is not None:
                    cached_fingerprint, cached_result = cached
                    if cached_fingerprint != fingerprint:
                        raise ValueError("idempotencyKey was reused with different arguments")
                    self._task_idempotency_cache.move_to_end(cache_key)
                    return copy.deepcopy(cached_result)
                result = await self._execute_task_write_uncached(tool_name, args, context)
                self._task_idempotency_cache[cache_key] = (fingerprint, copy.deepcopy(result))
                self._task_idempotency_cache.move_to_end(cache_key)
                while len(self._task_idempotency_cache) > _TASK_IDEMPOTENCY_CACHE_SIZE:
                    self._task_idempotency_cache.popitem(last=False)
                return result
        return await self._execute_task_write_uncached(tool_name, args, context)

    async def _execute_task_write_uncached(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: Any | None,
    ) -> dict[str, Any]:
        if tool_name == "notes.tasks.create":
            return await asyncio.to_thread(self._tasks_create_sync, context, args, tool_name)
        if tool_name == "notes.tasks.update":
            return await asyncio.to_thread(self._tasks_update_sync, context, args, tool_name)
        if tool_name == "notes.tasks.set_status":
            return await asyncio.to_thread(self._tasks_set_status_sync, context, args, tool_name)
        if tool_name == "notes.tasks.delete":
            return await asyncio.to_thread(self._tasks_delete_sync, context, args, tool_name)
        if tool_name == "notes.tasks.reconcile_note":
            return await asyncio.to_thread(self._tasks_reconcile_note_sync, context, args, tool_name)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _tasks_list_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        note_id = args.get("note_id")
        status = args.get("status")
        projection_status = args.get("projection_status")
        limit = int(args.get("limit", 100))
        offset = int(args.get("offset", 0))
        reconcile_limit = int(args.get("reconcile_limit", 25))
        query = args.get("query")
        metadata_filters = dict(args.get("metadata_filters") or {})
        include_unlinked = bool(args.get("include_unlinked", False))
        actor = self._task_actor(context)

        db = self._open_db(context)
        try:
            if note_id is not None:
                assert_identifier_in_scope(context, "note_id", note_id, label="Note")
                if reconcile_limit > 0:
                    reconciliation = self._task_service.ensure_note_reconciled(
                        db=db,
                        note_id=str(note_id),
                        actor=actor,
                    )
                else:
                    reconciliation = {"status": "clean", "processed_notes": 0, "remaining_stale_notes": 0}
                tasks = db.list_tasks(
                    note_id=str(note_id),
                    status=status,
                    projection_status=projection_status,
                    query=query,
                    metadata_filters=metadata_filters,
                    offset=offset,
                    include_unlinked=include_unlinked,
                    limit=limit,
                )
            else:
                scoped_note_ids = get_explicit_scope_ids(context, "note_id")
                if scoped_note_ids is None:
                    if reconcile_limit > 0:
                        reconciliation = self._task_service.reconcile_stale_notes(
                            db=db,
                            limit=reconcile_limit,
                            actor=actor,
                        )
                    else:
                        reconciliation = {"status": "clean", "processed_notes": 0, "remaining_stale_notes": 0}
                    tasks = db.list_tasks(
                        status=status,
                        projection_status=projection_status,
                        query=query,
                        metadata_filters=metadata_filters,
                        offset=offset,
                        include_unlinked=include_unlinked,
                        limit=limit,
                    )
                else:
                    fetched_tasks = []
                    processed = 0
                    skipped_reconciliation = 0
                    target_fetch = min(limit + offset, 500)
                    for scoped_note_id in sorted(scoped_note_ids):
                        if len(fetched_tasks) >= target_fetch:
                            break
                        if reconcile_limit > 0:
                            if processed < reconcile_limit:
                                self._task_service.ensure_note_reconciled(
                                    db=db,
                                    note_id=str(scoped_note_id),
                                    actor=actor,
                                )
                                processed += 1
                            else:
                                skipped_reconciliation += 1
                        fetched_tasks.extend(
                            db.list_tasks(
                                note_id=str(scoped_note_id),
                                status=status,
                                projection_status=projection_status,
                                query=query,
                                metadata_filters=metadata_filters,
                                offset=0,
                                include_unlinked=include_unlinked,
                                limit=target_fetch - len(fetched_tasks),
                            )
                        )
                    tasks = fetched_tasks[offset:offset + limit]
                    reconciliation = {
                        "status": "incomplete" if skipped_reconciliation else "clean",
                        "processed_notes": processed,
                        "remaining_stale_notes": skipped_reconciliation,
                    }

            return {
                "tasks": [self._task_response(db, task) for task in tasks],
                "reconciliation": self._task_reconciliation_response(reconciliation),
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "returned": len(tasks),
                },
            }
        finally:
            self._close_task_db(db, "task list")

    @staticmethod
    def _filter_task_list(tasks: list[dict[str, Any]], args: dict[str, Any]) -> list[dict[str, Any]]:
        include_unlinked = bool(args.get("include_unlinked", False))
        projection_status = args.get("projection_status")
        query = str(args.get("query") or "").strip().lower()
        metadata_filters = dict(args.get("metadata_filters") or {})
        filtered: list[dict[str, Any]] = []
        for task in tasks:
            if not include_unlinked and projection_status is None and task.get("projection_status") == "unlinked":
                continue
            if query and query not in str(task.get("text") or "").lower():
                continue
            metadata = dict(task.get("metadata_json") or {})
            if any(metadata.get(key) != value for key, value in metadata_filters.items()):
                continue
            filtered.append(task)
        return filtered

    def _tasks_get_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        task_id = str(args.get("task_id"))
        db = self._open_db(context)
        try:
            task = self._require_scoped_task(db, context, task_id)
            return self._task_response(db, task)
        finally:
            self._close_task_db(db, "task fetch")

    def _tasks_create_sync(self, context: Any | None, args: dict[str, Any], tool_name: str) -> dict[str, Any]:
        self._require_task_user_context(context)
        note_id = str(args.get("note_id"))
        text = str(args.get("text"))
        status = str(args.get("status") or "open")
        metadata = dict(args.get("metadata") or {})
        expected_note_version = int(args.get("expected_note_version"))
        db = self._open_db(context)
        try:
            self._require_scoped_note(db, context, note_id)
            task = self._task_service.create_task_for_note(
                db=db,
                note_id=note_id,
                text=text,
                status=status,
                metadata=metadata,
                expected_note_version=expected_note_version,
                actor=self._task_actor(
                    context,
                    tool_name=tool_name,
                    idempotency_key=self._get_idempotency_key(args),
                ),
            )
            response = self._task_response(db, task)
            response["insertion"] = {"mode": "append"}
            return response
        finally:
            self._close_task_db(db, "task create")

    def _tasks_update_sync(self, context: Any | None, args: dict[str, Any], tool_name: str) -> dict[str, Any]:
        self._require_task_user_context(context)
        task_id = str(args.get("task_id"))
        db = self._open_db(context)
        try:
            self._require_scoped_task(db, context, task_id)
            task = self._task_service.update_task(
                db=db,
                task_id=task_id,
                expected_task_version=int(args.get("expected_task_version")),
                expected_note_version=int(args["expected_note_version"]),
                text=args.get("text") if args.get("text") is not None else None,
                metadata=dict(args["metadata"]) if args.get("metadata") is not None else None,
                actor=self._task_actor(
                    context,
                    tool_name=tool_name,
                    idempotency_key=self._get_idempotency_key(args),
                ),
                record_only=bool(args.get("record_only", False)),
            )
            return self._task_response(db, task)
        finally:
            self._close_task_db(db, "task update")

    def _tasks_set_status_sync(self, context: Any | None, args: dict[str, Any], tool_name: str) -> dict[str, Any]:
        self._require_task_user_context(context)
        db = self._open_db(context)
        succeeded: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        seen_task_ids: set[str] = set()
        actor = self._task_actor(
            context,
            tool_name=tool_name,
            idempotency_key=self._get_idempotency_key(args),
        )
        try:
            for item in self._task_status_updates(args) or []:
                task_id = str(item.get("task_id") or "")
                if task_id in seen_task_ids:
                    skipped.append({"task_id": task_id, "reason": "duplicate_in_batch"})
                    continue
                seen_task_ids.add(task_id)
                try:
                    current = self._require_scoped_task(db, context, task_id)
                    requested_status = str(item.get("status"))
                    self._require_task_expected_versions(
                        db,
                        current,
                        expected_task_version=int(item.get("expected_task_version")),
                        expected_note_version=int(item["expected_note_version"]),
                    )
                    if current.get("status") == requested_status and current.get("projection_status") == "live":
                        skipped.append({
                            "task_id": task_id,
                            "reason": f"already_{requested_status}",
                            "task": self._task_response(db, current),
                        })
                        continue
                    task = self._task_service.update_task(
                        db=db,
                        task_id=task_id,
                        expected_task_version=int(item.get("expected_task_version")),
                        expected_note_version=int(item["expected_note_version"]),
                        status=requested_status,
                        actor=actor,
                        record_only=bool(item.get("record_only", False)),
                    )
                    succeeded.append({"task_id": task_id, "task": self._task_response(db, task)})
                except Exception as exc:
                    failed.append({
                        "task_id": task_id,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    })
            return {
                "succeeded": succeeded,
                "failed": failed,
                "skipped": skipped,
            }
        finally:
            self._close_task_db(db, "task status update")

    def _require_task_expected_versions(
        self,
        db: CharactersRAGDB,
        task: dict[str, Any],
        *,
        expected_task_version: int,
        expected_note_version: int,
    ) -> None:
        task_id = str(task.get("id") or "")
        actual_task_version = int(task.get("version") or 0)
        if actual_task_version != int(expected_task_version):
            raise ConflictError(
                f"Task version mismatch for ID '{task_id}'. "
                f"Expected {expected_task_version}, found {actual_task_version}.",
                entity="tasks",
                entity_id=task_id,
            )
        if task.get("projection_status") != "live":
            return
        task_store = getattr(db, "task_store", None)
        projection = task_store._fetch_projection(task_id) if task_store is not None else None
        if projection is None:
            raise ConflictError(
                f"Task projection is missing for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )
        actual_note_version = int(projection.get("note_version") or 0)
        if actual_note_version != int(expected_note_version):
            raise ConflictError(
                f"Task projection is stale for task '{task_id}'.",
                entity="tasks",
                entity_id=task_id,
            )

    def _tasks_delete_sync(self, context: Any | None, args: dict[str, Any], tool_name: str) -> dict[str, Any]:
        self._require_task_user_context(context)
        task_id = str(args.get("task_id"))
        db = self._open_db(context)
        try:
            self._require_scoped_task(db, context, task_id)
            task = self._task_service.delete_task(
                db=db,
                task_id=task_id,
                expected_task_version=int(args.get("expected_task_version")),
                expected_note_version=int(args["expected_note_version"]),
                record_only=bool(args.get("record_only_if_unlinked", args.get("record_only", False))),
                actor=self._task_actor(
                    context,
                    tool_name=tool_name,
                    idempotency_key=self._get_idempotency_key(args),
                ),
            )
            return self._task_response(db, task)
        finally:
            self._close_task_db(db, "task delete")

    def _tasks_reconcile_note_sync(self, context: Any | None, args: dict[str, Any], tool_name: str) -> dict[str, Any]:
        self._require_task_user_context(context)
        note_id = str(args.get("note_id"))
        db = self._open_db(context)
        try:
            note = self._require_scoped_note(db, context, note_id)
            if args.get("expected_note_version") is not None:
                expected_note_version = int(args["expected_note_version"])
                if int(note.get("version") or 0) != expected_note_version:
                    raise ValueError(
                        f"Note version mismatch for ID '{note_id}'. "
                        f"Expected {expected_note_version}, found {note.get('version')}."
                    )
            result = self._task_service.reconcile_note_current(
                db=db,
                note_id=note_id,
                actor=self._task_actor(context, tool_name=tool_name),
            )
            response = self._task_reconciliation_response(result)
            if args.get("work_limit") is not None:
                response["work_limit"] = int(args["work_limit"])
            return response
        finally:
            self._close_task_db(db, "task reconcile")

    def _require_task_user_context(self, context: Any | None) -> None:
        if context is None or not str(getattr(context, "user_id", "") or "").strip():
            raise ValueError("Missing user context for Notes task write")

    def _require_scoped_note(self, db: CharactersRAGDB, context: Any | None, note_id: str) -> dict[str, Any]:
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        note = db.get_note_by_id(note_id)
        if not note:
            raise ValueError(f"Note not found: {note_id}")
        return dict(note)

    def _require_scoped_task(self, db: CharactersRAGDB, context: Any | None, task_id: str) -> dict[str, Any]:
        task = db.get_task(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")
        note_id = str(task.get("note_id") or "")
        assert_identifier_in_scope(context, "note_id", note_id, label="Note")
        if not db.get_note_by_id(note_id):
            raise ValueError(f"Task note not found: {note_id}")
        return dict(task)

    def _task_response(self, db: CharactersRAGDB, task: dict[str, Any]) -> dict[str, Any]:
        task_id = str(task.get("id"))
        note_id = str(task.get("note_id"))
        note = db.get_note_by_id(note_id)
        projection = None
        try:
            task_store = getattr(db, "task_store", None)
            if task_store is not None:
                projection = task_store._fetch_projection(task_id)
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS:
            projection = None

        return {
            "id": task_id,
            "note_id": note_id,
            "text": str(task.get("text") or ""),
            "status": task.get("status"),
            "metadata": dict(task.get("metadata_json") or {}),
            "projection_status": task.get("projection_status"),
            "version": int(task.get("version") or 0),
            "created_at": task.get("created_at"),
            "updated_at": task.get("updated_at"),
            "completed_at": task.get("completed_at"),
            "note": (
                {
                    "id": str(note.get("id")),
                    "title": str(note.get("title") or ""),
                    "version": int(note.get("version") or 0),
                }
                if note
                else None
            ),
            "projection": self._task_projection_response(projection),
        }

    @staticmethod
    def _task_projection_response(projection: Any) -> dict[str, Any] | None:
        if not isinstance(projection, dict):
            return None
        return {
            "note_id": str(projection.get("note_id")),
            "note_version": int(projection.get("note_version") or 0),
            "line_number": int(projection.get("line_number") or 0),
            "start_offset": int(projection.get("start_offset") or 0),
            "end_offset": int(projection.get("end_offset") or 0),
            "raw_line": str(projection.get("raw_line") or ""),
            "has_child_content": bool(projection.get("has_child_content")),
            "projection_status": projection.get("projection_status"),
        }

    @staticmethod
    def _task_reconciliation_response(result: Any) -> dict[str, Any]:
        if result is None:
            return {"status": "clean", "processed_notes": 0, "remaining_stale_notes": 0}
        if isinstance(result, dict):
            return {
                "status": result.get("status", "clean"),
                "note_id": result.get("note_id"),
                "note_version": result.get("note_version"),
                "parsed_count": result.get("parsed_count"),
                "created_count": int(result.get("created_count", 0) or 0),
                "updated_count": int(result.get("updated_count", 0) or 0),
                "unlinked_count": int(result.get("unlinked_count", 0) or 0),
                "ambiguous_count": int(result.get("ambiguous_count", 0) or 0),
                "warning_count": int(result.get("warning_count", 0) or 0),
                "processed_notes": int(result.get("processed_notes", 0) or 0),
                "remaining_stale_notes": int(result.get("remaining_stale_notes", 0) or 0),
            }
        if hasattr(result, "processed_notes"):
            return {
                "status": getattr(result, "status", "clean"),
                "processed_notes": int(getattr(result, "processed_notes", 0) or 0),
                "remaining_stale_notes": int(getattr(result, "remaining_stale_notes", 0) or 0),
            }
        warning_count = int(getattr(result, "warning_count", 0) or 0)
        return {
            "status": "clean" if warning_count == 0 else "warnings",
            "note_id": getattr(result, "note_id", None),
            "note_version": getattr(result, "note_version", None),
            "parsed_count": getattr(result, "parsed_count", None),
            "created_count": int(getattr(result, "created_count", 0) or 0),
            "updated_count": int(getattr(result, "updated_count", 0) or 0),
            "unlinked_count": int(getattr(result, "unlinked_count", 0) or 0),
            "ambiguous_count": int(getattr(result, "ambiguous_count", 0) or 0),
            "warning_count": warning_count,
            "processed_notes": 0,
            "remaining_stale_notes": 0,
        }

    def _task_actor(
        self,
        context: Any | None,
        *,
        tool_name: str | None = None,
        idempotency_key: str | None = None,
    ) -> TaskActor:
        metadata = self._context_metadata(context)
        agent_id = self._agent_id_from_metadata(metadata)
        policy_mode = self._policy_mode_from_metadata(metadata)
        approval_id = self._approval_id_from_metadata(metadata)
        if agent_id:
            return TaskActor(
                actor_type="agent",
                actor_id=agent_id,
                tool_name=tool_name,
                policy_mode=policy_mode,
                approval_id=approval_id,
                idempotency_key=idempotency_key,
            )
        actor_id = str(getattr(context, "user_id", "") or getattr(context, "client_id", "") or "") or None
        return TaskActor(
            actor_type="user",
            actor_id=actor_id,
            tool_name=tool_name,
            policy_mode=policy_mode,
            approval_id=approval_id,
            idempotency_key=idempotency_key,
        )

    def _preflight_task_agent_write_policy(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: Any | None,
    ) -> dict[str, Any] | None:
        metadata = self._context_metadata(context)
        if not self._is_agent_context(metadata):
            return None

        if self._is_autonomous_agent_context(metadata):
            return self._task_policy_decision(
                tool_name,
                context,
                status="denied",
                action="deny",
                reason_code="autonomous_notes_task_write_activity_notice_required",
                message="Autonomous Notes task writes are disabled until persistent activity notices are enabled.",
            )

        if self._has_write_confirmation(metadata):
            return None

        return self._task_policy_decision(
            tool_name,
            context,
            status="approval_required",
            action="require_approval",
            reason_code="agent_write_confirmation_required",
            message="Agent Notes task writes require user confirmation or MCP Hub approval.",
        )

    @staticmethod
    def _context_metadata(context: Any | None) -> dict[str, Any]:
        metadata = getattr(context, "metadata", None) if context is not None else None
        return metadata if isinstance(metadata, dict) else {}

    def _is_agent_context(self, metadata: dict[str, Any]) -> bool:
        agent_context = metadata.get("agent_context")
        if isinstance(agent_context, dict):
            return True
        actor_type = str(metadata.get("actor_type") or metadata.get("client_type") or "").strip().lower()
        if actor_type in {"agent", "assistant", "autonomous_agent"}:
            return True
        return any(bool(metadata.get(key)) for key in ("agent_id", "is_agent", "agent"))

    def _is_autonomous_agent_context(self, metadata: dict[str, Any]) -> bool:
        agent_context = metadata.get("agent_context")
        if isinstance(agent_context, dict) and bool(agent_context.get("autonomous")):
            return True
        mode = str(metadata.get("execution_mode") or metadata.get("agent_mode") or "").strip().lower()
        return bool(metadata.get("autonomous") or metadata.get("autonomous_agent")) or mode == "autonomous"

    def _agent_id_from_metadata(self, metadata: dict[str, Any]) -> str | None:
        agent_context = metadata.get("agent_context")
        if isinstance(agent_context, dict):
            raw_agent_id = agent_context.get("agent_id") or agent_context.get("id")
            if raw_agent_id is not None and str(raw_agent_id).strip():
                return str(raw_agent_id)
        raw_agent_id = metadata.get("agent_id")
        return str(raw_agent_id).strip() if raw_agent_id is not None and str(raw_agent_id).strip() else None

    @staticmethod
    def _policy_mode_from_metadata(metadata: dict[str, Any]) -> str | None:
        approval = metadata.get("approval")
        if isinstance(approval, dict):
            for key in ("mode", "policy_mode", "approval_mode"):
                value = approval.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        effective_policy = metadata.get("_mcp_effective_tool_policy")
        if isinstance(effective_policy, dict):
            value = effective_policy.get("approval_mode") or effective_policy.get("policy_mode")
            if value is not None and str(value).strip():
                return str(value).strip()
        for key in ("policy_mode", "approval_mode"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    @staticmethod
    def _approval_id_from_metadata(metadata: dict[str, Any]) -> str | None:
        approval = metadata.get("approval")
        if isinstance(approval, dict):
            for key in ("id", "approval_id"):
                value = approval.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        value = metadata.get("approval_id")
        return str(value).strip() if value is not None and str(value).strip() else None

    @staticmethod
    def _has_write_confirmation(metadata: dict[str, Any]) -> bool:
        for key in ("user_confirmed_write", "write_confirmed", "mcp_hub_approval_granted", "approval_granted"):
            if bool(metadata.get(key)):
                return True
        approval = metadata.get("approval")
        return isinstance(approval, dict) and str(approval.get("status") or "").lower() == "approved"

    def _task_policy_decision(
        self,
        tool_name: str,
        context: Any | None,
        *,
        status: str,
        action: str,
        reason_code: str,
        message: str,
    ) -> dict[str, Any]:
        policy_decision = {
            "surface": "mcp_unified",
            "tool_name": tool_name,
            "action": action,
            "status": status,
            "reason_code": reason_code,
            "message": message,
            "mutation_allowed": False,
            "request_id": getattr(context, "request_id", None),
            "user_id": getattr(context, "user_id", None),
            "client_id": getattr(context, "client_id", None),
            "session_id": getattr(context, "session_id", None),
        }
        return {
            "status": status,
            "tool_name": tool_name,
            "mutated": False,
            "policy_decision": policy_decision,
        }

    @staticmethod
    def _get_idempotency_key(args: dict[str, Any]) -> Any | None:
        raw = args.get("idempotencyKey")
        if raw is None:
            raw = args.get("idempotency_key")
        if raw is None:
            return None
        if not isinstance(raw, str):
            return raw
        value = raw.strip()
        return value or None

    @staticmethod
    def _task_arguments_fingerprint(args: dict[str, Any]) -> str:
        filtered = {
            key: value
            for key, value in args.items()
            if key not in {"idempotencyKey", "idempotency_key"}
        }
        return json.dumps(filtered, sort_keys=True, default=str, separators=(",", ":"))

    @staticmethod
    def _task_idempotency_cache_key(context: Any | None, tool_name: str, idempotency_key: str) -> str:
        owner = (
            f"user:{getattr(context, 'user_id', None)}"
            if getattr(context, "user_id", None) is not None
            else f"client:{getattr(context, 'client_id', None)}"
            if getattr(context, "client_id", None) is not None
            else "anon"
        )
        session_id = str(getattr(context, "session_id", "") or "")
        return f"{owner}|session:{session_id}|tool:{tool_name}|key:{idempotency_key}"

    def _close_task_db(self, db: CharactersRAGDB, operation: str) -> None:
        try:
            db.close_all_connections()
        except _NOTES_MODULE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to close ChaChaNotes DB connections after {}: {}", operation, exc)

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
