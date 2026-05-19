"""
Knowledge Aggregator Module for Unified MCP

Provides knowledge.search and knowledge.get by fanning out to source modules
(notes, media, chats, characters, prompts). Stage 4 initial implementation:
- FTS-only sources
- Normalized inputs/outputs using common schema
- Per-request dedupe by URI; persists across WS sessions via context.metadata
- Retrieval modes supported minimally (snippet|full). chunk_with_siblings deferred.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from ...persona_scope import get_explicit_scope_ids
from ..base import BaseModule, create_tool_definition
from ..registry import get_module_registry


def _iso_to_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class KnowledgeModule(BaseModule):
    """Aggregator for knowledge search/get"""

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Knowledge module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Knowledge module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="knowledge.search",
                description="Unified search across notes, media, chats, characters, prompts (FTS-only).",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                        "order_by": {"type": "string", "enum": ["relevance", "recent"], "default": "relevance"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["notes", "media", "chats", "characters", "prompts"]},
                        },
                        "filters": {"type": "object"}
                    },
                    "required": ["query"],
                },
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="knowledge.get",
                description="Retrieve an item by source + id with basic retrieval modes (snippet/full).",
                parameters={
                    "properties": {
                        "source": {"type": "string", "enum": ["notes", "media", "chats", "characters", "prompts"]},
                        "id": {"type": ["string", "number"]},
                        "retrieval": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string", "enum": ["snippet", "full", "chunk", "chunk_with_siblings", "auto"], "default": "snippet"},
                                "snippet_length": {"type": "integer", "minimum": 50, "maximum": 2000, "default": 300},
                                "max_tokens": {"type": "integer"},
                                "chars_per_token": {"type": "integer"},
                            }
                        }
                    },
                    "required": ["source", "id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except (OverflowError, TypeError, ValueError) as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve
        if tool_name == "knowledge.search":
            return await self._search(args, context)
        if tool_name == "knowledge.get":
            return await self._get(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    async def _call_tool(self, tool: str, arguments: dict[str, Any], context: Any | None) -> Any:
        registry = get_module_registry()
        module = await registry.find_module_for_tool(tool)
        if not module:
            return None
        return await module.execute_with_circuit_breaker(module.execute_tool, tool, arguments, context)

    async def _resolve_tool_write_flag(self, tool: str, module: Any) -> Optional[bool]:
        """Best-effort determination of whether a tool is write-capable."""
        try:
            if module is None:
                return None
            tool_def = None
            try:
                get_def = getattr(module, "get_tool_def", None)
                if callable(get_def):
                    tool_def = await get_def(tool)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                tool_def = None
            if tool_def is None:
                try:
                    tool_defs = await module.get_tools()
                    for _t in tool_defs:
                        if isinstance(_t, dict) and _t.get("name") == tool:
                            tool_def = _t
                            break
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    tool_def = None
            if tool_def is not None:
                return module.is_write_tool_def(tool_def)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        return None

    async def _tool_allowed(self, tool: str, context: Any | None) -> bool:
        """Enforce per-source RBAC for underlying tools."""
        if context is None:
            return False
        try:
            # Use MCPProtocol's permission logic for consistency.
            from tldw_Server_API.app.core.MCP_unified.server import get_mcp_server
            server = get_mcp_server()
            protocol = getattr(server, "protocol", None)
            if protocol is None:
                from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
                protocol = MCPProtocol()
            registry = get_module_registry()
            module = await registry.find_module_for_tool(tool)
            is_write = await self._resolve_tool_write_flag(tool, module)
            return await protocol._has_tool_permission(context, tool, is_write=is_write)  # type: ignore[attr-defined]
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return False

    def _tool_for_source(self, source: str, action: str) -> Optional[str]:
        """Map knowledge sources to underlying tool names."""
        mapping = {
            "notes": f"notes.{action}",
            "media": f"media.{action}",
            "chats": f"chats.{action}",
            "characters": f"characters.{action}",
            "prompts": f"prompts.{action}",
        }
        return mapping.get(source)

    def _collect_seen(self, context: Any | None) -> set[str]:
        seen: set[str] = set()
        try:
            if context and getattr(context, "metadata", None):
                if isinstance(context.metadata.get("seen_uris"), list):
                    seen.update([str(u) for u in context.metadata.get("seen_uris")])
        except (AttributeError, TypeError, ValueError):
            pass
        return seen

    def _update_seen(self, context: Any | None, new_uris: list[str]):
        try:
            if context and getattr(context, "metadata", None):
                arr = list(set([str(u) for u in (context.metadata.get("seen_uris") or [])] + new_uris))
                context.metadata["seen_uris"] = arr
        except (AttributeError, TypeError, ValueError):
            pass

    def _combine_and_sort(self, buckets: list[list[dict[str, Any]]], order_by: str) -> list[dict[str, Any]]:
        items = [it for b in buckets for it in (b or [])]
        if order_by == "recent":
            items.sort(key=lambda x: (_iso_to_dt(x.get("last_modified")) or datetime.min), reverse=True)
        else:
            items.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return items

    @staticmethod
    def _id_matches_scope(candidate: Any, scoped_ids: set[str] | None) -> bool:
        if scoped_ids is None:
            return True
        return str(candidate or "").strip() in scoped_ids

    def _item_allowed_by_scope(self, item: dict[str, Any], context: Any | None) -> bool:
        source = str(item.get("source") or "").strip().lower()
        if source == "media":
            scoped_media_ids = get_explicit_scope_ids(context, "media_id")
            candidate = item.get("id")
            if candidate is None:
                uri = str(item.get("uri") or "")
                if uri.startswith("media://"):
                    candidate = uri.split("media://", 1)[1].split("#", 1)[0]
            return self._id_matches_scope(candidate, scoped_media_ids)

        if source == "notes":
            scoped_note_ids = get_explicit_scope_ids(context, "note_id")
            candidate = item.get("id")
            if candidate is None:
                uri = str(item.get("uri") or "")
                if uri.startswith("notes://"):
                    candidate = uri.split("notes://", 1)[1].split("#", 1)[0]
            return self._id_matches_scope(candidate, scoped_note_ids)

        if source == "chats":
            scoped_conv_ids = get_explicit_scope_ids(context, "conversation_id")
            candidate = item.get("conversation_id") or item.get("id")
            if candidate is None:
                uri = str(item.get("uri") or "")
                if uri.startswith("chats://"):
                    candidate = uri.split("chats://", 1)[1].split("#", 1)[0]
            return self._id_matches_scope(candidate, scoped_conv_ids)

        if source == "characters":
            scoped_char_ids = get_explicit_scope_ids(context, "character_id")
            candidate = item.get("character_id") or item.get("id")
            if candidate is None:
                uri = str(item.get("uri") or "")
                if uri.startswith("characters://"):
                    candidate = uri.split("characters://", 1)[1].split("#", 1)[0]
            return self._id_matches_scope(candidate, scoped_char_ids)

        if source == "prompts":
            scoped_prompt_ids = get_explicit_scope_ids(context, "prompt_id")
            candidate = item.get("prompt_id") or item.get("id")
            if candidate is None:
                uri = str(item.get("uri") or "")
                if uri.startswith("prompts://"):
                    candidate = uri.split("prompts://", 1)[1].split("#", 1)[0]
            return self._id_matches_scope(candidate, scoped_prompt_ids)

        return True

    async def _search(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        query: str = args.get("query")
        limit: int = int(args.get("limit", 20))
        offset: int = int(args.get("offset", 0))
        snippet_len: int = int(args.get("snippet_length", 300))
        order_by: str = args.get("order_by", "relevance")
        default_sources = ["notes", "media", "chats", "characters", "prompts"]
        sources: list[str] = args.get("sources") or default_sources
        filters: dict[str, Any] = args.get("filters") or {}

        # Apply session defaults
        try:
            if context and getattr(context, "metadata", None):
                sc = context.metadata.get("safe_config") or {}
                if isinstance(sc, dict):
                    snippet_len = int(sc.get("snippet_length", snippet_len))
        except (AttributeError, OverflowError, TypeError, ValueError):
            pass

        # Seed dedupe with session-persisted URIs if any
        seen = self._collect_seen(context)

        scoped_note_ids = get_explicit_scope_ids(context, "note_id")
        scoped_media_ids = get_explicit_scope_ids(context, "media_id")
        scoped_conv_ids = get_explicit_scope_ids(context, "conversation_id")
        scoped_char_ids = get_explicit_scope_ids(context, "character_id")
        scoped_prompt_ids = get_explicit_scope_ids(context, "prompt_id")

        # Enforce per-source RBAC before fan-out calls
        allowed_sources: list[str] = []
        for src in sources:
            tool = self._tool_for_source(src, "search")
            if tool is None:
                continue
            allowed = await self._tool_allowed(tool, context)
            if allowed:
                allowed_sources.append(src)
            else:
                try:
                    if context and getattr(context, "logger", None):
                        context.logger.debug("Knowledge source filtered by RBAC", source=src, tool=tool)
                except (AttributeError, TypeError, ValueError):
                    pass
        sources = allowed_sources

        if not sources:
            return {"results": [], "has_more": False, "next_offset": None, "total_estimated": 0}

        # Fan-out calls
        tasks = []
        if "notes" in sources:
            tasks.append(
                self._call_tool(
                    "notes.search",
                    {
                        "query": query,
                        "limit": limit + offset,
                        "offset": 0,
                        "snippet_length": snippet_len,
                        "note_ids_filter": sorted(scoped_note_ids) if scoped_note_ids is not None else None,
                    },
                    context,
                )
            )
        if "media" in sources:
            f = (filters or {}).get("media") or {}
            tasks.append(
                self._call_tool(
                    "media.search",
                    {
                        "query": query,
                        "limit": limit + offset,
                        "offset": 0,
                        "snippet_length": snippet_len,
                        "media_types": f.get("media_types"),
                        "date_from": f.get("date_from"),
                        "date_to": f.get("date_to"),
                        "order_by": f.get("order_by", order_by),
                        "media_ids_filter": sorted(scoped_media_ids) if scoped_media_ids is not None else None,
                    },
                    context,
                )
            )
        if "chats" in sources:
            f = (filters or {}).get("chats") or {}
            character_filter = f.get("character_id")
            if character_filter is None and scoped_char_ids and len(scoped_char_ids) == 1:
                try:
                    character_filter = int(next(iter(scoped_char_ids)))
                except (TypeError, ValueError):
                    character_filter = None
            tasks.append(self._call_tool("chats.search", {
                "query": query,
                "by": f.get("by", "both"),
                "limit": limit + offset,
                "offset": 0,
                "snippet_length": snippet_len,
                "character_id": character_filter,
                "sender": f.get("sender"),
                "conversation_ids_filter": sorted(scoped_conv_ids) if scoped_conv_ids is not None else None,
            }, context))
        if "characters" in sources:
            tasks.append(
                self._call_tool(
                    "characters.search",
                    {
                        "query": query,
                        "limit": limit + offset,
                        "offset": 0,
                        "snippet_length": snippet_len,
                        "character_ids_filter": sorted(scoped_char_ids) if scoped_char_ids is not None else None,
                    },
                    context,
                )
            )
        if "prompts" in sources:
            f = (filters or {}).get("prompts") or {}
            tasks.append(
                self._call_tool(
                    "prompts.search",
                    {
                        "query": query,
                        "fields": f.get("fields"),
                        "limit": limit + offset,
                        "offset": 0,
                        "snippet_length": snippet_len,
                        "prompt_ids_filter": sorted(scoped_prompt_ids) if scoped_prompt_ids is not None else None,
                    },
                    context,
                )
            )

        results_raw = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []

        buckets: list[list[dict[str, Any]]] = []
        for r in results_raw:
            try:
                if isinstance(r, dict) and isinstance(r.get("results"), list):
                    buckets.append(r.get("results"))
            except (AttributeError, TypeError, ValueError):
                continue

        combined = self._combine_and_sort(buckets, order_by=order_by)

        # Dedupe by URI (skip items already seen in session/request)
        out: list[dict[str, Any]] = []
        new_uris: list[str] = []
        for item in combined:
            if not self._item_allowed_by_scope(item, context):
                continue
            uri = str(item.get("uri") or "")
            if not uri or uri in seen:
                continue
            seen.add(uri)
            new_uris.append(uri)
            out.append(item)

        # Apply offset/limit across combined results
        sliced = out[offset: offset + limit]
        has_more = (len(out) > (offset + len(sliced)))
        next_offset = (offset + len(sliced)) if has_more else None

        # Persist dedupe across WS sessions via context.metadata
        self._update_seen(context, new_uris)

        return {
            "results": sliced,
            "has_more": has_more,
            "next_offset": next_offset,
            "total_estimated": len(combined),
        }

    async def _get(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        source: str = args.get("source")
        idv = args.get("id")
        retrieval: dict[str, Any] = args.get("retrieval") or {}
        mode = retrieval.get("mode", "snippet")
        int(retrieval.get("snippet_length", 300))
        # If auto and a token budget is provided, prefer chunk_with_siblings
        if mode == "auto" and isinstance(retrieval.get("max_tokens"), (int, float)):
            retrieval = dict(retrieval)
            retrieval["mode"] = "chunk_with_siblings"

        tool = self._tool_for_source(source, "get")
        if not tool:
            raise ValueError(f"Unsupported source for get: {source}")

        if source == "notes":
            if not self._id_matches_scope(idv, get_explicit_scope_ids(context, "note_id")):
                raise PermissionError("Source/id not allowed by persona scope")
        if source == "media":
            if not self._id_matches_scope(idv, get_explicit_scope_ids(context, "media_id")):
                raise PermissionError("Source/id not allowed by persona scope")
        if source == "chats":
            if not self._id_matches_scope(idv, get_explicit_scope_ids(context, "conversation_id")):
                raise PermissionError("Source/id not allowed by persona scope")
        if source == "characters":
            if not self._id_matches_scope(idv, get_explicit_scope_ids(context, "character_id")):
                raise PermissionError("Source/id not allowed by persona scope")

        # Enforce per-source RBAC for direct fetch
        allowed = await self._tool_allowed(tool, context)
        if not allowed:
            raise PermissionError(f"Permission denied for source: {source}")

        # Map to source tools
        if source == "notes":
            note_args: dict[str, Any] = {
                "note_id": str(idv),
                "retrieval": retrieval,
            }
            scoped_note_ids = get_explicit_scope_ids(context, "note_id")
            if scoped_note_ids is not None:
                note_args["note_ids_filter"] = sorted(scoped_note_ids)
            return await self._call_tool("notes.get", note_args, context)
        if source == "media":
            return await self._call_tool(
                "media.get",
                {"media_id": int(idv), "retrieval": retrieval},
                context,
            )
        if source == "chats":
            chats_args: dict[str, Any] = {
                "conversation_id": str(idv),
                "retrieval": retrieval,
            }
            scoped_conv_ids = get_explicit_scope_ids(context, "conversation_id")
            if scoped_conv_ids is not None:
                chats_args["conversation_ids_filter"] = sorted(scoped_conv_ids)
            return await self._call_tool("chats.get", chats_args, context)
        if source == "characters":
            return await self._call_tool("characters.get", {"character_id": int(idv)}, context)
        if source == "prompts":
            out = await self._call_tool("prompts.get", {"prompt_id_or_name": str(idv)}, context)
            meta = out.get("meta") if isinstance(out, dict) else None
            if isinstance(meta, dict) and not self._item_allowed_by_scope(meta, context):
                raise PermissionError("Source/id not allowed by persona scope")
            return out
        # Unsupported source
        raise ValueError(f"Unsupported source for get: {source}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "knowledge.search":
            q = arguments.get("query")
            if not isinstance(q, str) or not (1 <= len(q) <= 1000):
                raise ValueError("query must be 1..1000 chars")
            limit = int(arguments.get("limit", 20))
            offset = int(arguments.get("offset", 0))
            snip = int(arguments.get("snippet_length", 300))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
            if offset < 0:
                raise ValueError("offset must be >= 0")
            if snip < 50 or snip > 2000:
                raise ValueError("snippet_length must be 50..2000")
            sources = arguments.get("sources")
            if sources is not None and (not isinstance(sources, list) or any(s not in {"notes","media","chats","characters","prompts"} for s in sources)):
                raise ValueError("sources must be subset of notes|media|chats|characters|prompts")
        elif tool_name == "knowledge.get":
            source = arguments.get("source")
            if source not in {"notes","media","chats","characters","prompts"}:
                raise ValueError("source invalid")
            if arguments.get("id") is None:
                raise ValueError("id is required")
            retrieval = arguments.get("retrieval") or {}
            if not isinstance(retrieval, dict):
                raise ValueError("retrieval must be an object")
            mode = retrieval.get("mode", "snippet")
            if mode not in {"snippet","full","chunk","chunk_with_siblings","auto"}:
                raise ValueError("retrieval.mode invalid")
