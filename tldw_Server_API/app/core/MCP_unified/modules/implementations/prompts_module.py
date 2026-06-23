"""
Prompts Module for Unified MCP

Search/get prompts via PromptsDatabase (per-user prompts DB path).
"""

import asyncio
from sqlite3 import Error as SQLiteError
from typing import Any

from loguru import logger

from ....DB_Management.Prompts_DB import PromptsDatabase
from ....exceptions import PromptCatalogError
from ...persona_scope import assert_identifier_in_scope
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb
from .prompts_catalog import (
    CONFIG_PROMPT_PREFIX,
    LIBRARY_PROMPT_PREFIX,
    ConfigPromptCatalogSource,
    MCPPromptFormatter,
    PromptCatalogCursor,
    UserPromptCatalogSource,
    clamp_prompt_page_size,
    decode_prompt_cursor,
    encode_prompt_cursor,
)

_PROMPTS_HEALTHCHECK_EXCEPTIONS = (
    OSError,
    RuntimeError,
    SQLiteError,
    TypeError,
    ValueError,
)
_PROMPTS_CLOSE_EXCEPTIONS = (OSError, RuntimeError, SQLiteError, TypeError, ValueError)


class PromptsModule(BaseModule):
    async def on_initialize(self) -> None:
        """Initialize prompt catalog sources from module settings."""
        logger.info(f"Initializing Prompts module: {self.name}")
        settings = self.config.settings or {}
        self._prompt_list_page_size = clamp_prompt_page_size(settings.get("prompt_list_page_size", 50))
        try:
            max_rendered_prompt_chars = int(settings.get("max_rendered_prompt_chars", 100000) or 100000)
        except (TypeError, ValueError):
            max_rendered_prompt_chars = 100000
        self._prompt_formatter = MCPPromptFormatter(max_rendered_chars=max_rendered_prompt_chars)
        self._user_prompt_source = UserPromptCatalogSource(self._prompt_formatter)
        config_prompts = settings.get("config_prompts")
        self._config_prompt_source = ConfigPromptCatalogSource(
            self._prompt_formatter,
            config_prompts if isinstance(config_prompts, dict) else {},
        )

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Prompts module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        try:
            _ = PromptsDatabase  # noqa: F401
            checks["driver_available"] = True
        except NameError:
            checks["driver_available"] = False
        try:
            import os
            base = os.path.dirname("./Databases/test.db") or "."
            free_gb = get_free_disk_space_gb(base)
            checks["disk_space"] = free_gb > 1
        except (AttributeError, OSError, TypeError, ValueError):
            checks["disk_space"] = False
        # Optional ephemeral DB write test (heavy) for deeper validation
        try:
            import os
            if str(os.getenv("MCP_HEALTHCHECK_DB_WRITE_TEST", "")).lower() in {"1", "true", "yes"}:
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(prefix="mcp_prompts_health_", suffix=".db", delete=True) as tf:
                    db = PromptsDatabase(db_path=tf.name, client_id=f"mcp_prompts_{self.config.name}")
                    # Trivial read to confirm
                    _ = db.get_prompt_by_name("nonexistent")
                checks["ephemeral_db_ok"] = True
        except _PROMPTS_HEALTHCHECK_EXCEPTIONS:
            checks["ephemeral_db_ok"] = False

        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="prompts.search",
                description="Search prompts by name/details/system_prompt/user_prompt/author/keywords.",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "fields": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                    },
                    "required": ["query"],
                },
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="prompts.get",
                description="Get a prompt by id or name.",
                parameters={
                    "properties": {
                        "prompt_id_or_name": {"type": "string"}
                    },
                    "required": ["prompt_id_or_name"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except (TypeError, ValueError) as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve
        if tool_name == "prompts.search":
            return await self._search(args, context)
        if tool_name == "prompts.get":
            return await self._get(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def get_prompts_for_context(self, context: Any, params: dict[str, Any]) -> dict[str, Any]:
        """List MCP prompts visible to the request context."""

        cursor = decode_prompt_cursor((params or {}).get("cursor"))
        page_size = getattr(self, "_prompt_list_page_size", 50)
        library_result = await asyncio.to_thread(
            self._user_prompt_source.list_prompts,
            context=context,
            cursor=cursor,
            limit=page_size,
        )

        prompts = list(library_result.prompts)
        warnings = list(library_result.warnings)
        next_cursor = library_result.next_cursor
        remaining = page_size - len(prompts)

        if remaining > 0 and library_result.next_cursor is None:
            config_result = await asyncio.to_thread(
                self._config_prompt_source.list_prompts,
                cursor=cursor,
                limit=remaining,
            )
            prompts.extend(config_result.prompts)
            warnings.extend(config_result.warnings)
            next_cursor = config_result.next_cursor
        elif (
            remaining == 0
            and library_result.next_cursor is None
            and self._config_prompt_source.has_entries_after(cursor.config_index)
        ):
            next_cursor = PromptCatalogCursor(
                library_done=True,
                config_index=cursor.config_index,
            )

        result: dict[str, Any] = {"prompts": prompts}
        encoded_cursor = encode_prompt_cursor(next_cursor)
        if encoded_cursor:
            result["nextCursor"] = encoded_cursor
        if warnings:
            result["_meta"] = {"tldw": {"warnings": warnings}}
        return result

    async def get_prompt_for_context(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        """Get one MCP prompt by catalog namespace for the request context."""

        if isinstance(name, str) and name.startswith(LIBRARY_PROMPT_PREFIX):
            return await asyncio.to_thread(
                self._user_prompt_source.get_prompt,
                context=context,
                name=name,
                arguments=arguments,
            )
        if isinstance(name, str) and name.startswith(CONFIG_PROMPT_PREFIX):
            return await asyncio.to_thread(
                self._config_prompt_source.get_prompt,
                name=name,
                arguments=arguments,
            )
        raise PromptCatalogError("prompt_not_found", "Prompt not found")

    def _open_db(self, context: Any) -> PromptsDatabase:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Prompts access")
        ppath = context.db_paths.get("prompts")
        if not ppath:
            raise ValueError("Prompts DB path not available in context")
        return PromptsDatabase(db_path=ppath, client_id=f"mcp_prompts_{self.config.name}")

    async def _search(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        query: str = args.get("query")
        fields: list[str] = args.get("fields") or []
        limit: int = int(args.get("limit", 10))
        offset: int = int(args.get("offset", 0))
        snippet_len: int = int(args.get("snippet_length", 300))
        return await asyncio.to_thread(
            self._search_sync,
            context,
            query,
            fields,
            limit,
            offset,
            snippet_len,
        )

    async def _get(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        ident: str = args.get("prompt_id_or_name")
        return await asyncio.to_thread(self._get_sync, context, ident)

    def _search_sync(
        self,
        context: Any | None,
        query: str,
        fields: list[str],
        limit: int,
        offset: int,
        snippet_len: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            page_size = max(1, limit)
            page = (offset // page_size) + 1
            local_offset = offset % page_size
            rows, total = db.search_prompts(
                search_query=query,
                search_fields=fields or None,
                page=page,
                results_per_page=page_size,
                include_deleted=False,
            )
            if local_offset:
                if len(rows) == page_size and (offset + limit) < total:
                    next_rows, _ = db.search_prompts(
                        search_query=query,
                        search_fields=fields or None,
                        page=page + 1,
                        results_per_page=page_size,
                        include_deleted=False,
                    )
                    rows = rows + next_rows
                rows = rows[local_offset: local_offset + limit]
            else:
                rows = rows[:limit]
            out = []
            scope_filtered = False
            for r in rows:
                try:
                    assert_identifier_in_scope(context, "prompt_id", r.get("id"), label="Prompt")
                except PermissionError:
                    scope_filtered = True
                    continue
                desc = r.get("details") or r.get("system_prompt") or ""
                out.append({
                    "id": r.get("id"),
                    "source": "prompts",
                    "title": r.get("name"),
                    "snippet": " ".join(desc.split())[:snippet_len],
                    "uri": f"prompts://{r.get('id')}",
                    "score": 1.0,
                    "score_type": "fts",
                    "created_at": r.get("created_at"),
                    "last_modified": r.get("last_modified"),
                    "version": r.get("version"),
                    "tags": r.get("keywords") or None,
                    "loc": None,
                })
            visible_total = len(out) if scope_filtered else total
            has_more = False if scope_filtered else (offset + len(out)) < visible_total
            next_offset = (offset + len(out)) if has_more else None
            return {
                "results": out,
                "has_more": has_more,
                "next_offset": next_offset,
                "total_estimated": visible_total,
            }
        finally:
            try:
                db.close_connection()
            except _PROMPTS_CLOSE_EXCEPTIONS as exc:
                logger.debug("Failed to close Prompts DB connections after prompts search: {}", exc)

    def _get_sync(self, context: Any | None, ident: str) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            row = None
            try:
                pid = int(ident)
                row = db.get_prompt_by_id(pid)
            except (TypeError, ValueError):
                row = db.get_prompt_by_name(ident)
            if not row:
                raise ValueError(f"Prompt not found: {ident}")
            assert_identifier_in_scope(context, "prompt_id", row.get("id"), label="Prompt")
            desc = row.get("details") or ""
            meta = {
                "id": row.get("id"),
                "source": "prompts",
                "title": row.get("name"),
                "snippet": " ".join(desc.split())[:300],
                "uri": f"prompts://{row.get('id')}",
                "score": 1.0,
                "score_type": "fts",
                "created_at": row.get("created_at"),
                "last_modified": row.get("last_modified"),
                "version": row.get("version"),
                "tags": None,
                "loc": None,
            }
            content = {
                k: row.get(k)
                for k in ("name", "author", "details", "system_prompt", "user_prompt")
            }
            return {"meta": meta, "content": content, "attachments": None}
        finally:
            try:
                db.close_connection()
            except _PROMPTS_CLOSE_EXCEPTIONS as exc:
                logger.debug("Failed to close Prompts DB connections after prompts get: {}", exc)

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "prompts.search":
            q = arguments.get("query")
            if not isinstance(q, str) or not (1 <= len(q) <= 1000):
                raise ValueError("query must be 1..1000 chars")
            fields = arguments.get("fields")
            if fields is not None and (not isinstance(fields, list) or any(not isinstance(f, str) for f in fields)):
                raise ValueError("fields must be list[str] if provided")
            limit = int(arguments.get("limit", 10))
            offset = int(arguments.get("offset", 0))
            snip = int(arguments.get("snippet_length", 300))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
            if offset < 0:
                raise ValueError("offset must be >= 0")
            if snip < 50 or snip > 2000:
                raise ValueError("snippet_length must be 50..2000")
        elif tool_name == "prompts.get":
            pid = arguments.get("prompt_id_or_name")
            if not isinstance(pid, str) or not pid:
                raise ValueError("prompt_id_or_name must be a non-empty string")
