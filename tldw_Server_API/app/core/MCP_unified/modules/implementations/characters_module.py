"""
Characters Module for Unified MCP

Search/get character cards via ChaChaNotes DB FTS.
"""

import asyncio
from sqlite3 import Error as SQLiteError
from typing import Any

from loguru import logger

from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB
from ...persona_scope import assert_identifier_in_scope
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb

_CHARACTERS_HEALTHCHECK_EXCEPTIONS = (
    OSError,
    RuntimeError,
    SQLiteError,
    TypeError,
    ValueError,
)
_CHARACTERS_CLOSE_EXCEPTIONS = (
    OSError,
    RuntimeError,
    SQLiteError,
    TypeError,
    ValueError,
)


class CharactersModule(BaseModule):
    async def on_initialize(self) -> None:
        logger.info(f"Initializing Characters module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Characters module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        try:
            _ = CharactersRAGDB  # noqa: F401
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
                with NamedTemporaryFile(prefix="mcp_characters_health_", suffix=".db", delete=True) as tf:
                    db = CharactersRAGDB(db_path=tf.name, client_id=f"mcp_characters_{self.config.name}")
                    # Trivial read
                    _ = db.get_character_card_by_id(-1)
                checks["ephemeral_db_ok"] = True
        except _CHARACTERS_HEALTHCHECK_EXCEPTIONS:
            checks["ephemeral_db_ok"] = False

        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="characters.search",
                description="Search character cards (name, description, personality, scenario, system_prompt).",
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
                name="characters.get",
                description="Get character card by id.",
                parameters={
                    "properties": {
                        "character_id": {"type": "integer"}
                    },
                    "required": ["character_id"],
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
        if tool_name == "characters.search":
            return await self._search(args, context)
        if tool_name == "characters.get":
            return await self._get(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _open_db(self, context: Any) -> CharactersRAGDB:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Characters access")
        chacha_path = context.db_paths.get("chacha")
        if not chacha_path:
            raise ValueError("ChaChaNotes DB path not available in context")
        return CharactersRAGDB(db_path=chacha_path, client_id=f"mcp_characters_{self.config.name}")

    async def _search(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        query: str = args.get("query")
        limit: int = int(args.get("limit", 10))
        offset: int = int(args.get("offset", 0))
        snippet_len: int = int(args.get("snippet_length", 300))
        return await asyncio.to_thread(
            self._search_sync,
            context,
            query,
            limit,
            offset,
            snippet_len,
        )

    async def _get(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        character_id: int = int(args.get("character_id"))
        return await asyncio.to_thread(self._get_sync, context, character_id)

    def _search_sync(
        self,
        context: Any | None,
        query: str,
        limit: int,
        offset: int,
        snippet_len: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            rows = db.search_character_cards(query, limit=limit + offset)
            rows = rows[offset: offset + limit]
            out = []
            for r in rows:
                desc = r.get("description") or r.get("system_prompt") or ""
                out.append({
                    "id": r.get("id"),
                    "source": "characters",
                    "title": r.get("name"),
                    "snippet": " ".join(desc.split())[:snippet_len],
                    "uri": f"characters://{r.get('id')}",
                    "score": 1.0,
                    "score_type": "fts",
                    "created_at": r.get("created_at"),
                    "last_modified": r.get("last_modified"),
                    "version": r.get("version"),
                    "tags": None,
                    "loc": None,
                })
            return {"results": out, "has_more": False, "next_offset": None, "total_estimated": len(out) + offset}
        finally:
            try:
                db.close_all_connections()
            except _CHARACTERS_CLOSE_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after characters search: {}", exc)

    def _get_sync(self, context: Any | None, character_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            r = db.get_character_card_by_id(character_id)
            if not r:
                raise ValueError(f"Character not found: {character_id}")
            assert_identifier_in_scope(context, "character_id", r.get("id"), label="Character")
            desc = r.get("description") or ""
            meta = {
                "id": r.get("id"),
                "source": "characters",
                "title": r.get("name"),
                "snippet": " ".join(desc.split())[:300],
                "uri": f"characters://{r.get('id')}",
                "score": 1.0,
                "score_type": "fts",
                "created_at": r.get("created_at"),
                "last_modified": r.get("last_modified"),
                "version": r.get("version"),
                "tags": None,
                "loc": None,
            }
            # content as a dict with important fields
            content = {
                k: r.get(k)
                for k in ("name", "description", "personality", "scenario", "system_prompt")
            }
            return {"meta": meta, "content": content, "attachments": None}
        finally:
            try:
                db.close_all_connections()
            except _CHARACTERS_CLOSE_EXCEPTIONS as exc:
                logger.debug("Failed to close ChaChaNotes DB connections after characters get: {}", exc)

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "characters.search":
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
        elif tool_name == "characters.get":
            cid = arguments.get("character_id")
            if not isinstance(cid, int) or cid <= 0:
                raise ValueError("character_id must be a positive integer")
