"""
Slides Module for Unified MCP

CRUD operations for presentations stored in SlidesDatabase.
Supports manual creation, AI generation from various sources,
version history, and export to multiple formats.
"""

import asyncio
import json
from collections.abc import Iterable
from typing import Any, Optional

from loguru import logger

from ....DB_Management.media_db.api import managed_media_database
from ....Slides.slides_db import ConflictError
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb

# Available Reveal.js themes
REVEAL_THEMES = [
    "black", "white", "league", "beige", "sky", "night",
    "serif", "simple", "solarized", "blood", "moon", "dracula"
]


class SlidesModule(BaseModule):
    """Presentation/slides management module for MCP"""

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Slides module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Slides module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        try:
            from ....Slides.slides_db import SlidesDatabase
            _ = SlidesDatabase
            checks["driver_available"] = True
        except (ImportError, AttributeError):
            checks["driver_available"] = False
        try:
            from pathlib import Path
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                base = Path(get_project_root())
            except (ImportError, AttributeError, RuntimeError):
                base = Path(__file__).resolve().parents[5]
            free_gb = get_free_disk_space_gb(base)
            checks["disk_space"] = free_gb > 1
        except (AttributeError, OSError, TypeError, ValueError, RuntimeError):
            checks["disk_space"] = False
        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            # Presentations CRUD
            create_tool_definition(
                name="slides.presentations.list",
                description="List presentations with sorting.",
                parameters={
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "sort_column": {
                            "type": "string",
                            "enum": ["created_at", "last_modified", "title"],
                            "default": "last_modified",
                        },
                        "sort_direction": {"type": "string", "enum": ["ASC", "DESC"], "default": "DESC"},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="slides.presentations.search",
                description="Full-text search presentations.",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="slides.presentations.get",
                description="Get a presentation by ID.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                    "required": ["presentation_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="slides.presentations.create",
                description="Create a new presentation manually.",
                parameters={
                    "properties": {
                        "title": {"type": "string", "minLength": 1, "maxLength": 256},
                        "description": {"type": "string", "maxLength": 2000},
                        "theme": {
                            "type": "string",
                            "enum": REVEAL_THEMES,
                            "default": "black",
                        },
                        "marp_theme": {"type": "string", "maxLength": 64},
                        "template_id": {"type": "string"},
                        "slides": {"type": "string", "description": "Markdown or JSON slides content"},
                        "custom_css": {"type": "string", "maxLength": 50000},
                        "settings": {"type": "object", "description": "Presentation settings JSON"},
                    },
                    "required": ["title", "slides"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.presentations.update",
                description="Full update of a presentation.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "updates": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "maxLength": 256},
                                "description": {"type": "string", "maxLength": 2000},
                                "theme": {"type": "string", "enum": REVEAL_THEMES},
                                "marp_theme": {"type": "string", "maxLength": 64},
                                "template_id": {"type": "string"},
                                "slides": {"type": "string"},
                                "custom_css": {"type": "string", "maxLength": 50000},
                                "settings": {"type": "object"},
                            },
                        },
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["presentation_id", "updates", "expected_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.presentations.patch",
                description="Partial update of a presentation.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "patch": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "maxLength": 256},
                                "description": {"type": "string", "maxLength": 2000},
                                "theme": {"type": "string", "enum": REVEAL_THEMES},
                            },
                        },
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["presentation_id", "patch", "expected_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.presentations.reorder",
                description="Reorder slides in a presentation.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "slide_order": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "New order of slide indices",
                        },
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["presentation_id", "slide_order", "expected_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.presentations.delete",
                description="Soft-delete a presentation.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["presentation_id", "expected_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.presentations.restore",
                description="Restore a soft-deleted presentation.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["presentation_id", "expected_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Templates
            create_tool_definition(
                name="slides.templates.list",
                description="List available presentation templates.",
                parameters={
                    "properties": {
                        "category": {"type": "string", "description": "Filter by category"},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="slides.templates.get",
                description="Get template details.",
                parameters={
                    "properties": {
                        "template_id": {"type": "string"},
                    },
                    "required": ["template_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            # Versions
            create_tool_definition(
                name="slides.versions.list",
                description="List presentation versions.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "required": ["presentation_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="slides.versions.get",
                description="Get a specific version of a presentation.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "version": {"type": "integer", "minimum": 1},
                    },
                    "required": ["presentation_id", "version"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="slides.versions.restore",
                description="Restore presentation to a previous version.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "version": {"type": "integer", "minimum": 1},
                        "expected_current_version": {"type": "integer"},
                    },
                    "required": ["presentation_id", "version", "expected_current_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # AI Generation
            create_tool_definition(
                name="slides.generate.from_prompt",
                description="Generate slides from a text prompt.",
                parameters={
                    "properties": {
                        "prompt": {"type": "string", "minLength": 10, "maxLength": 50000},
                        "title_hint": {"type": "string", "maxLength": 256},
                        "num_slides": {"type": "integer", "minimum": 3, "maximum": 30, "default": 8},
                        "theme": {"type": "string", "enum": REVEAL_THEMES, "default": "black"},
                        "provider": {"type": "string", "description": "LLM provider"},
                        "model": {"type": "string", "description": "LLM model"},
                    },
                    "required": ["prompt"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.generate.from_media",
                description="Generate slides from media transcript/content.",
                parameters={
                    "properties": {
                        "media_id": {"type": "integer"},
                        "title_hint": {"type": "string", "maxLength": 256},
                        "num_slides": {"type": "integer", "minimum": 3, "maximum": 30, "default": 8},
                        "theme": {"type": "string", "enum": REVEAL_THEMES, "default": "black"},
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                    "required": ["media_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.generate.from_notes",
                description="Generate slides from note IDs.",
                parameters={
                    "properties": {
                        "note_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20},
                        "title_hint": {"type": "string", "maxLength": 256},
                        "num_slides": {"type": "integer", "minimum": 3, "maximum": 30, "default": 8},
                        "theme": {"type": "string", "enum": REVEAL_THEMES, "default": "black"},
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                    "required": ["note_ids"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.generate.from_chat",
                description="Generate slides from a conversation.",
                parameters={
                    "properties": {
                        "conversation_id": {"type": "string"},
                        "title_hint": {"type": "string", "maxLength": 256},
                        "num_slides": {"type": "integer", "minimum": 3, "maximum": 30, "default": 8},
                        "theme": {"type": "string", "enum": REVEAL_THEMES, "default": "black"},
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                    "required": ["conversation_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="slides.generate.from_rag",
                description="Generate slides from RAG query results.",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 3, "maxLength": 1000},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                        "title_hint": {"type": "string", "maxLength": 256},
                        "num_slides": {"type": "integer", "minimum": 3, "maximum": 30, "default": 8},
                        "theme": {"type": "string", "enum": REVEAL_THEMES, "default": "black"},
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                    "required": ["query"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Export
            create_tool_definition(
                name="slides.export",
                description="Export presentation to various formats.",
                parameters={
                    "properties": {
                        "presentation_id": {"type": "string"},
                        "format": {
                            "type": "string",
                            "enum": ["reveal", "json", "markdown", "pdf"],
                            "default": "reveal",
                        },
                    },
                    "required": ["presentation_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "slides.presentations.list":
            limit = int(arguments.get("limit", 50))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
        elif tool_name == "slides.presentations.search":
            query = arguments.get("query")
            if not isinstance(query, str) or not (1 <= len(query) <= 500):
                raise ValueError("query must be 1..500 chars")
        elif tool_name == "slides.presentations.get":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
        elif tool_name == "slides.presentations.create":
            title = arguments.get("title")
            if not isinstance(title, str) or not (1 <= len(title.strip()) <= 256):
                raise ValueError("title must be 1..256 chars")
            slides = arguments.get("slides")
            if not isinstance(slides, str) or not slides.strip():
                raise ValueError("slides must be a non-empty string")
            theme = arguments.get("theme", "black")
            if theme not in REVEAL_THEMES:
                raise ValueError(f"theme must be one of: {', '.join(REVEAL_THEMES)}")
        elif tool_name == "slides.presentations.update":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            updates = arguments.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("updates must be a non-empty object")
            ev = arguments.get("expected_version")
            if not isinstance(ev, int) or ev < 1:
                raise ValueError("expected_version must be a positive integer")
        elif tool_name == "slides.presentations.patch":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            patch = arguments.get("patch")
            if not isinstance(patch, dict) or not patch:
                raise ValueError("patch must be a non-empty object")
            ev = arguments.get("expected_version")
            if not isinstance(ev, int) or ev < 1:
                raise ValueError("expected_version must be a positive integer")
        elif tool_name == "slides.presentations.reorder":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            order = arguments.get("slide_order")
            if not isinstance(order, list) or not order:
                raise ValueError("slide_order must be a non-empty list")
        elif tool_name in {"slides.presentations.delete", "slides.presentations.restore"}:
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            ev = arguments.get("expected_version")
            if not isinstance(ev, int) or ev < 1:
                raise ValueError("expected_version must be a positive integer")
        elif tool_name == "slides.templates.get":
            tid = arguments.get("template_id")
            if not isinstance(tid, str) or not tid.strip():
                raise ValueError("template_id must be a non-empty string")
        elif tool_name == "slides.versions.list":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
        elif tool_name == "slides.versions.get":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            version = arguments.get("version")
            if not isinstance(version, int) or version < 1:
                raise ValueError("version must be a positive integer")
        elif tool_name == "slides.versions.restore":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            version = arguments.get("version")
            if not isinstance(version, int) or version < 1:
                raise ValueError("version must be a positive integer")
            ev = arguments.get("expected_current_version")
            if not isinstance(ev, int) or ev < 1:
                raise ValueError("expected_current_version must be a positive integer")
        elif tool_name == "slides.generate.from_prompt":
            prompt = arguments.get("prompt")
            if not isinstance(prompt, str) or not (10 <= len(prompt) <= 50000):
                raise ValueError("prompt must be 10..50000 chars")
        elif tool_name == "slides.generate.from_media":
            media_id = arguments.get("media_id")
            if not isinstance(media_id, int) or media_id < 1:
                raise ValueError("media_id must be a positive integer")
        elif tool_name == "slides.generate.from_notes":
            note_ids = arguments.get("note_ids")
            if not isinstance(note_ids, list) or not (1 <= len(note_ids) <= 20):
                raise ValueError("note_ids must be a list of 1..20 items")
        elif tool_name == "slides.generate.from_chat":
            cid = arguments.get("conversation_id")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("conversation_id must be a non-empty string")
        elif tool_name == "slides.generate.from_rag":
            query = arguments.get("query")
            if not isinstance(query, str) or not (3 <= len(query) <= 1000):
                raise ValueError("query must be 3..1000 chars")
        elif tool_name == "slides.export":
            pid = arguments.get("presentation_id")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("presentation_id must be a non-empty string")
            fmt = arguments.get("format", "reveal")
            if fmt not in {"reveal", "json", "markdown", "pdf"}:
                raise ValueError("format must be reveal, json, markdown, or pdf")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except (ValueError, TypeError, KeyError) as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve

        # Presentations CRUD
        if tool_name == "slides.presentations.list":
            return await self._list_presentations(args, context)
        if tool_name == "slides.presentations.search":
            return await self._search_presentations(args, context)
        if tool_name == "slides.presentations.get":
            return await self._get_presentation(args, context)
        if tool_name == "slides.presentations.create":
            return await self._create_presentation(args, context)
        if tool_name == "slides.presentations.update":
            return await self._update_presentation(args, context)
        if tool_name == "slides.presentations.patch":
            return await self._patch_presentation(args, context)
        if tool_name == "slides.presentations.reorder":
            return await self._reorder_slides(args, context)
        if tool_name == "slides.presentations.delete":
            return await self._delete_presentation(args, context)
        if tool_name == "slides.presentations.restore":
            return await self._restore_presentation(args, context)
        # Templates
        if tool_name == "slides.templates.list":
            return await self._list_templates(args, context)
        if tool_name == "slides.templates.get":
            return await self._get_template(args, context)
        # Versions
        if tool_name == "slides.versions.list":
            return await self._list_versions(args, context)
        if tool_name == "slides.versions.get":
            return await self._get_version(args, context)
        if tool_name == "slides.versions.restore":
            return await self._restore_version(args, context)
        # Generation
        if tool_name == "slides.generate.from_prompt":
            return await self._generate_from_prompt(args, context)
        if tool_name == "slides.generate.from_media":
            return await self._generate_from_media(args, context)
        if tool_name == "slides.generate.from_notes":
            return await self._generate_from_notes(args, context)
        if tool_name == "slides.generate.from_chat":
            return await self._generate_from_chat(args, context)
        if tool_name == "slides.generate.from_rag":
            return await self._generate_from_rag(args, context)
        # Export
        if tool_name == "slides.export":
            return await self._export_presentation(args, context)

        raise ValueError(f"Unknown tool: {tool_name}")

    def _open_db(self, context: Any):
        from ....Slides.slides_db import SlidesDatabase
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Slides access")
        slides_path = context.db_paths.get("slides")
        if not slides_path:
            raise ValueError("Slides DB path not available in context")
        return SlidesDatabase(db_path=slides_path, client_id=f"mcp_slides_{self.config.name}")

    def _presentation_to_dict(self, row) -> dict[str, Any]:
        """Convert PresentationRow dataclass to dict."""
        return {
            "id": row.id,
            "title": row.title,
            "description": row.description,
            "theme": row.theme,
            "marp_theme": row.marp_theme,
            "template_id": row.template_id,
            "settings": row.settings,
            "studio_data": getattr(row, "studio_data", None),
            "slides": row.slides,
            "slides_text": row.slides_text,
            "source_type": row.source_type,
            "source_ref": row.source_ref,
            "source_query": row.source_query,
            "custom_css": row.custom_css,
            "created_at": row.created_at,
            "last_modified": row.last_modified,
            "deleted": bool(row.deleted),
            "client_id": row.client_id,
            "version": row.version,
        }

    def _version_to_dict(self, row) -> dict[str, Any]:
        """Convert PresentationVersionRow dataclass to dict."""
        return {
            "presentation_id": row.presentation_id,
            "version": row.version,
            "payload": json.loads(row.payload_json) if row.payload_json else None,
            "created_at": row.created_at,
            "client_id": row.client_id,
        }

    # Presentations CRUD

    async def _list_presentations(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_presentations_sync, context, args)

    def _list_presentations_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            rows, total = db.list_presentations(
                limit=int(args.get("limit", 50)),
                offset=int(args.get("offset", 0)),
                include_deleted=bool(args.get("include_deleted", False)),
                sort_column=args.get("sort_column", "last_modified"),
                sort_direction=args.get("sort_direction", "DESC"),
            )
            presentations = [self._presentation_to_dict(r) for r in rows]
            int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            has_more = offset + len(presentations) < total
            return {
                "presentations": presentations,
                "total": total,
                "has_more": has_more,
                "next_offset": offset + len(presentations) if has_more else None,
            }
        finally:
            db.close_connection()

    async def _search_presentations(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._search_presentations_sync, context, args)

    def _search_presentations_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            rows, total = db.search_presentations(
                query=args.get("query"),
                limit=int(args.get("limit", 50)),
                offset=int(args.get("offset", 0)),
                include_deleted=bool(args.get("include_deleted", False)),
            )
            presentations = [self._presentation_to_dict(r) for r in rows]
            int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            has_more = offset + len(presentations) < total
            return {
                "presentations": presentations,
                "total": total,
                "has_more": has_more,
                "next_offset": offset + len(presentations) if has_more else None,
            }
        finally:
            db.close_connection()

    async def _get_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_presentation_sync, context, args)

    def _get_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            row = db.get_presentation_by_id(
                args.get("presentation_id"),
                include_deleted=bool(args.get("include_deleted", False)),
            )
            return {"presentation": self._presentation_to_dict(row)}
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        finally:
            db.close_connection()

    async def _create_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_presentation_sync, context, args)

    def _create_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            slides = args.get("slides", "")
            # Extract plain text for FTS
            slides_text = self._extract_slides_text(slides)
            settings = args.get("settings")
            settings_json = json.dumps(settings) if settings else None
            studio_data = args.get("studio_data")
            studio_data_json = json.dumps(studio_data) if studio_data else None

            row = db.create_presentation(
                presentation_id=None,
                title=args.get("title"),
                description=args.get("description"),
                theme=args.get("theme", "black"),
                marp_theme=args.get("marp_theme"),
                template_id=args.get("template_id"),
                settings=settings_json,
                studio_data=studio_data_json,
                slides=slides,
                slides_text=slides_text,
                source_type="manual",
                source_ref=None,
                source_query=None,
                custom_css=args.get("custom_css"),
            )
            return {
                "presentation_id": row.id,
                "success": True,
                "presentation": self._presentation_to_dict(row),
            }
        finally:
            db.close_connection()

    def _extract_slides_text(self, slides: str) -> str:
        """Extract plain text from slides content for FTS."""
        if not slides:
            return ""
        # If JSON, extract text fields
        try:
            data = json.loads(slides)
            if isinstance(data, dict) and "slides" in data:
                texts = []
                for slide in data.get("slides", []):
                    if isinstance(slide, dict):
                        texts.append(slide.get("title", ""))
                        texts.append(slide.get("content", ""))
                        texts.append(slide.get("speaker_notes", ""))
                return " ".join(t for t in texts if t)
        except json.JSONDecodeError:
            pass
        # Otherwise return as-is (markdown)
        return slides

    def _normalize_presentation_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(updates)
        normalized.pop("slides_text", None)
        if "settings" in normalized and isinstance(normalized["settings"], dict):
            normalized["settings"] = json.dumps(normalized["settings"])
        if "studio_data" in normalized and isinstance(normalized["studio_data"], dict):
            normalized["studio_data"] = json.dumps(normalized["studio_data"])
        if "slides" in normalized:
            normalized["slides_text"] = self._extract_slides_text(normalized["slides"])
        return normalized

    @staticmethod
    def _validate_slide_order(slide_order: list[int], slide_count: int) -> None:
        if any(isinstance(index, bool) or not isinstance(index, int) for index in slide_order) or sorted(slide_order) != list(range(slide_count)):
            raise ValueError(f"slide_order must be a permutation of 0..{slide_count - 1}")

    async def _update_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_presentation_sync, context, args)

    def _update_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            updates = self._normalize_presentation_updates(args.get("updates", {}))
            expected_version = args.get("expected_version")

            row = db.update_presentation(
                presentation_id=pid,
                update_fields=updates,
                expected_version=expected_version,
            )
            return {
                "presentation_id": pid,
                "success": True,
                "presentation": self._presentation_to_dict(row),
            }
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            db.close_connection()

    async def _patch_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._patch_presentation_sync, context, args)

    def _patch_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            patch = self._normalize_presentation_updates(args.get("patch", {}))
            expected_version = args.get("expected_version")

            db.update_presentation(
                presentation_id=pid,
                update_fields=patch,
                expected_version=expected_version,
            )
            return {
                "presentation_id": pid,
                "success": True,
                "updated_fields": list(patch.keys()),
            }
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            db.close_connection()

    async def _reorder_slides(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._reorder_slides_sync, context, args)

    def _reorder_slides_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            slide_order = list(args.get("slide_order", []))
            expected_version = args.get("expected_version")

            # Get current presentation
            pres = db.get_presentation_by_id(pid)
            slides_content = pres.slides

            # Parse and reorder slides
            try:
                data = json.loads(slides_content)
                if isinstance(data, dict) and "slides" in data:
                    old_slides = data["slides"]
                    self._validate_slide_order(slide_order, len(old_slides))
                    new_slides = [old_slides[i] for i in slide_order]
                    data["slides"] = new_slides
                    new_content = json.dumps(data)
                else:
                    raise ValueError("Cannot reorder non-JSON slides content")
            except json.JSONDecodeError:
                raise ValueError("Cannot reorder markdown slides; use update instead") from None

            db.update_presentation(
                presentation_id=pid,
                update_fields={
                    "slides": new_content,
                    "slides_text": self._extract_slides_text(new_content),
                },
                expected_version=expected_version,
            )
            return {
                "presentation_id": pid,
                "success": True,
                "new_order": slide_order,
            }
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            db.close_connection()

    async def _delete_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._delete_presentation_sync, context, args)

    def _delete_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            expected_version = args.get("expected_version")
            db.soft_delete_presentation(pid, expected_version)
            return {
                "presentation_id": pid,
                "action": "soft_deleted",
                "success": True,
            }
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            db.close_connection()

    async def _restore_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._restore_presentation_sync, context, args)

    def _restore_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            expected_version = args.get("expected_version")
            row = db.restore_presentation(pid, expected_version)
            return {
                "presentation_id": pid,
                "action": "restored",
                "success": True,
                "presentation": self._presentation_to_dict(row),
            }
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            db.close_connection()

    # Templates

    async def _list_templates(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """List available presentation templates."""
        # Built-in templates
        templates = [
            {
                "id": "default",
                "name": "Default",
                "description": "Simple presentation with title and content slides",
                "category": "general",
                "theme": "black",
            },
            {
                "id": "business",
                "name": "Business",
                "description": "Professional business presentation",
                "category": "business",
                "theme": "white",
            },
            {
                "id": "academic",
                "name": "Academic",
                "description": "Academic/research presentation",
                "category": "education",
                "theme": "serif",
            },
            {
                "id": "tech",
                "name": "Tech/Developer",
                "description": "Technical presentation with code highlighting",
                "category": "technology",
                "theme": "dracula",
            },
            {
                "id": "minimal",
                "name": "Minimal",
                "description": "Clean, minimal design",
                "category": "general",
                "theme": "simple",
            },
        ]
        category = args.get("category")
        if category:
            templates = [t for t in templates if t["category"] == category]
        return {"templates": templates}

    async def _get_template(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """Get template details with sample structure."""
        template_id = args.get("template_id")
        templates = {
            "default": {
                "id": "default",
                "name": "Default",
                "theme": "black",
                "structure": {
                    "slides": [
                        {"order": 0, "layout": "title", "title": "Presentation Title", "content": "Subtitle"},
                        {"order": 1, "layout": "content", "title": "Overview", "content": "- Point 1\n- Point 2\n- Point 3"},
                        {"order": 2, "layout": "content", "title": "Details", "content": "Main content here"},
                        {"order": 3, "layout": "content", "title": "Conclusion", "content": "Summary and next steps"},
                    ]
                },
            },
            "business": {
                "id": "business",
                "name": "Business",
                "theme": "white",
                "structure": {
                    "slides": [
                        {"order": 0, "layout": "title", "title": "Business Proposal", "content": "Company Name"},
                        {"order": 1, "layout": "content", "title": "Executive Summary", "content": "Key points"},
                        {"order": 2, "layout": "content", "title": "Problem Statement", "content": "Challenge we address"},
                        {"order": 3, "layout": "content", "title": "Solution", "content": "Our approach"},
                        {"order": 4, "layout": "content", "title": "Next Steps", "content": "Action items"},
                    ]
                },
            },
            "academic": {
                "id": "academic",
                "name": "Academic",
                "theme": "serif",
                "structure": {
                    "slides": [
                        {"order": 0, "layout": "title", "title": "Research Topic", "content": "Author / Institution"},
                        {"order": 1, "layout": "content", "title": "Background", "content": "Motivation and context"},
                        {"order": 2, "layout": "content", "title": "Methodology", "content": "Approach and methods"},
                        {"order": 3, "layout": "content", "title": "Findings", "content": "Key results"},
                        {"order": 4, "layout": "content", "title": "Conclusion", "content": "Summary and future work"},
                    ]
                },
            },
            "tech": {
                "id": "tech",
                "name": "Tech/Developer",
                "theme": "dracula",
                "structure": {
                    "slides": [
                        {"order": 0, "layout": "title", "title": "Tech Talk", "content": "Project or Feature"},
                        {"order": 1, "layout": "content", "title": "Problem", "content": "What we are solving"},
                        {"order": 2, "layout": "content", "title": "Architecture", "content": "Key components"},
                        {"order": 3, "layout": "content", "title": "Demo", "content": "Code or live example"},
                        {"order": 4, "layout": "content", "title": "Next Steps", "content": "Roadmap"},
                    ]
                },
            },
            "minimal": {
                "id": "minimal",
                "name": "Minimal",
                "theme": "simple",
                "structure": {
                    "slides": [
                        {"order": 0, "layout": "title", "title": "Minimal Deck", "content": "Subtitle"},
                        {"order": 1, "layout": "content", "title": "Point One", "content": "Short and clear"},
                        {"order": 2, "layout": "content", "title": "Point Two", "content": "Keep it concise"},
                    ]
                },
            },
        }
        template = templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        return {"template": template}

    # Versions

    async def _list_versions(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_versions_sync, context, args)

    def _list_versions_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            rows, total = db.list_presentation_versions(
                presentation_id=args.get("presentation_id"),
                limit=int(args.get("limit", 20)),
                offset=int(args.get("offset", 0)),
            )
            versions = [self._version_to_dict(r) for r in rows]
            int(args.get("limit", 20))
            offset = int(args.get("offset", 0))
            has_more = offset + len(versions) < total
            return {
                "versions": versions,
                "total": total,
                "has_more": has_more,
                "next_offset": offset + len(versions) if has_more else None,
            }
        finally:
            db.close_connection()

    async def _get_version(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_version_sync, context, args)

    def _get_version_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            row = db.get_presentation_version(
                presentation_id=args.get("presentation_id"),
                version=args.get("version"),
            )
            return {"version": self._version_to_dict(row)}
        except KeyError:
            raise ValueError("Version not found") from None
        finally:
            db.close_connection()

    async def _restore_version(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._restore_version_sync, context, args)

    def _restore_version_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            version = args.get("version")
            expected_current_version = args.get("expected_current_version")

            # Get the version to restore
            version_row = db.get_presentation_version(presentation_id=pid, version=version)
            payload = json.loads(version_row.payload_json) if version_row.payload_json else {}

            # Extract fields to restore
            restore_fields = {}
            for key in ["title", "description", "theme", "marp_theme", "settings", "studio_data", "slides", "custom_css"]:
                if key in payload:
                    restore_fields[key] = payload[key]

            restore_fields = self._normalize_presentation_updates(restore_fields)

            row = db.update_presentation(
                presentation_id=pid,
                update_fields=restore_fields,
                expected_version=expected_current_version,
            )
            return {
                "presentation_id": pid,
                "restored_from_version": version,
                "new_version": row.version,
                "success": True,
            }
        except KeyError:
            raise ValueError("Presentation or version not found") from None
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            db.close_connection()

    # Generation helpers

    def _get_generator(self):
        from ....Slides.slides_generator import SlidesGenerator
        return SlidesGenerator()

    async def _generate_from_prompt(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """Generate slides from text prompt."""
        prompt = args.get("prompt")
        title_hint = args.get("title_hint")
        theme = args.get("theme", "black")
        provider = args.get("provider") or "openai"
        model = args.get("model")

        generator = self._get_generator()
        result = await asyncio.to_thread(
            generator.generate_from_text,
            source_text=prompt,
            title_hint=title_hint,
            provider=provider,
            model=model,
            api_key=None,
            temperature=0.7,
            max_tokens=4000,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=1000,
            summary_tokens=200,
        )

        # Save as presentation
        return await self._save_generated_presentation(
            context=context,
            result=result,
            theme=theme,
            source_type="generated",
            source_query=prompt[:500],
        )

    async def _generate_from_media(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """Generate slides from media content."""
        media_id = args.get("media_id")
        title_hint = args.get("title_hint")
        theme = args.get("theme", "black")
        provider = args.get("provider") or "openai"
        model = args.get("model")

        # Get media content
        media_content = await asyncio.to_thread(
            self._get_media_content, context, media_id
        )
        if not media_content:
            raise ValueError(f"Media not found or no content: {media_id}")

        generator = self._get_generator()
        result = await asyncio.to_thread(
            generator.generate_from_text,
            source_text=media_content,
            title_hint=title_hint,
            provider=provider,
            model=model,
            api_key=None,
            temperature=0.7,
            max_tokens=4000,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=1000,
            summary_tokens=200,
        )

        return await self._save_generated_presentation(
            context=context,
            result=result,
            theme=theme,
            source_type="media",
            source_ref=str(media_id),
        )

    async def _generate_from_notes(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """Generate slides from notes."""
        note_ids = args.get("note_ids", [])
        title_hint = args.get("title_hint")
        theme = args.get("theme", "black")
        provider = args.get("provider") or "openai"
        model = args.get("model")

        # Get notes content
        notes_content = await asyncio.to_thread(
            self._get_notes_content, context, note_ids
        )
        if not notes_content:
            raise ValueError("No notes content found")

        generator = self._get_generator()
        result = await asyncio.to_thread(
            generator.generate_from_text,
            source_text=notes_content,
            title_hint=title_hint,
            provider=provider,
            model=model,
            api_key=None,
            temperature=0.7,
            max_tokens=4000,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=1000,
            summary_tokens=200,
        )

        return await self._save_generated_presentation(
            context=context,
            result=result,
            theme=theme,
            source_type="notes",
            source_ref=",".join(note_ids),
        )

    async def _generate_from_chat(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """Generate slides from conversation."""
        conversation_id = args.get("conversation_id")
        title_hint = args.get("title_hint")
        theme = args.get("theme", "black")
        provider = args.get("provider") or "openai"
        model = args.get("model")

        # Get conversation content
        chat_content = await asyncio.to_thread(
            self._get_chat_content, context, conversation_id
        )
        if not chat_content:
            raise ValueError(f"Conversation not found: {conversation_id}")

        generator = self._get_generator()
        result = await asyncio.to_thread(
            generator.generate_from_text,
            source_text=chat_content,
            title_hint=title_hint,
            provider=provider,
            model=model,
            api_key=None,
            temperature=0.7,
            max_tokens=4000,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=1000,
            summary_tokens=200,
        )

        return await self._save_generated_presentation(
            context=context,
            result=result,
            theme=theme,
            source_type="chat",
            source_ref=conversation_id,
        )

    async def _generate_from_rag(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        """Generate slides from RAG query results."""
        query = args.get("query")
        top_k = int(args.get("top_k", 5))
        title_hint = args.get("title_hint")
        theme = args.get("theme", "black")
        provider = args.get("provider") or "openai"
        model = args.get("model")

        # Perform RAG search
        rag_content = await self._get_rag_content(context, query, top_k)
        if not rag_content:
            raise ValueError("No RAG results found for query")

        generator = self._get_generator()
        result = await asyncio.to_thread(
            generator.generate_from_text,
            source_text=rag_content,
            title_hint=title_hint or f"Presentation: {query}",
            provider=provider,
            model=model,
            api_key=None,
            temperature=0.7,
            max_tokens=4000,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=1000,
            summary_tokens=200,
        )

        return await self._save_generated_presentation(
            context=context,
            result=result,
            theme=theme,
            source_type="rag",
            source_query=query,
        )

    async def _save_generated_presentation(
        self,
        context: Any,
        result: dict[str, Any],
        theme: str,
        source_type: str,
        source_ref: Optional[str] = None,
        source_query: Optional[str] = None,
    ) -> dict[str, Any]:
        """Save generated slides as a presentation."""
        return await asyncio.to_thread(
            self._save_generated_presentation_sync,
            context, result, theme, source_type, source_ref, source_query
        )

    def _save_generated_presentation_sync(
        self,
        context: Any,
        result: dict[str, Any],
        theme: str,
        source_type: str,
        source_ref: Optional[str],
        source_query: Optional[str],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            title = result.get("title", "Generated Presentation")
            slides = result.get("slides", [])
            slides_json = json.dumps({"title": title, "slides": slides})
            slides_text = self._extract_slides_text(slides_json)

            row = db.create_presentation(
                presentation_id=None,
                title=title,
                description=f"Generated from {source_type}",
                theme=theme,
                marp_theme=None,
                template_id=None,
                settings=None,
                studio_data=None,
                slides=slides_json,
                slides_text=slides_text,
                source_type=source_type,
                source_ref=source_ref,
                source_query=source_query,
                custom_css=None,
            )
            return {
                "presentation_id": row.id,
                "success": True,
                "presentation": self._presentation_to_dict(row),
                "slides_count": len(slides),
            }
        finally:
            db.close_connection()

    def _get_media_content(self, context: Any, media_id: int) -> Optional[str]:
        """Get media content for slide generation."""
        try:
            media_path = context.db_paths.get("media")
            if not media_path:
                return None
            with managed_media_database(
                "mcp_slides_gen",
                db_path=media_path,
                initialize=False,
            ) as db:
                media = db.get_media_by_id(media_id)
                if not media:
                    return None
                return media.get("content") or media.get("transcript") or media.get("summary")
        except (ImportError, AttributeError, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to get media content: {e}")
            return None

    def _get_notes_content(self, context: Any, note_ids: list[str]) -> Optional[str]:
        """Get notes content for slide generation."""
        try:
            chacha_path = context.db_paths.get("chacha")
            if not chacha_path:
                return None
            from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB
            db = CharactersRAGDB(db_path=chacha_path, client_id="mcp_slides_gen")
            try:
                contents = []
                for note_id in note_ids:
                    note = db.get_note_by_id(note_id)
                    if note:
                        contents.append(f"# {note.get('title', 'Untitled')}\n{note.get('content', '')}")
                return "\n\n".join(contents) if contents else None
            finally:
                db.close_all_connections()
        except (ImportError, AttributeError, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to get notes content: {e}")
            return None

    def _get_chat_content(self, context: Any, conversation_id: str) -> Optional[str]:
        """Get chat content for slide generation."""
        try:
            chacha_path = context.db_paths.get("chacha")
            if not chacha_path:
                return None
            from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB
            db = CharactersRAGDB(db_path=chacha_path, client_id="mcp_slides_gen")
            try:
                messages = db.get_messages_for_conversation(conversation_id)
                if not messages:
                    return None
                texts = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    texts.append(f"{role.upper()}: {content}")
                return "\n\n".join(texts)
            finally:
                db.close_all_connections()
        except (ImportError, AttributeError, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to get chat content: {e}")
            return None

    async def _get_rag_content(self, context: Any, query: str, top_k: int) -> Optional[str]:
        """Get RAG search results for slide generation via unified pipeline."""
        try:
            media_path = context.db_paths.get("media")
            chacha_path = context.db_paths.get("chacha")
            if not media_path and not chacha_path:
                return None
            from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
            sources = []
            if media_path:
                sources.append("media_db")
            if chacha_path:
                sources.append("notes")
            result = await unified_rag_pipeline(
                query=query,
                sources=sources or ["media_db"],
                media_db_path=media_path,
                notes_db_path=chacha_path,
                search_mode="hybrid",
                enable_reranking=True,
                reranking_strategy="hybrid",
                top_k=top_k,
                enable_generation=False,
                user_id=str(getattr(context, "user_id", "") or ""),
            )
            documents = getattr(result, "documents", None) or []
            content = self._format_rag_documents(documents)
            if not content and getattr(result, "generated_answer", None):
                content = str(result.generated_answer)
            return content or None
        except (ImportError, AttributeError, OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
            logger.error(f"Failed to get RAG content: {e}")
            return None

    def _format_rag_documents(self, documents: Iterable[Any]) -> str:
        parts: list[str] = []
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            title = metadata.get("title") or metadata.get("source_title") or getattr(doc, "id", "source")
            content = getattr(doc, "content", "")
            if title:
                parts.append(f"# {title}")
            if content:
                parts.append(str(content))
        return "\n\n".join(parts).strip()

    # Export

    async def _export_presentation(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._export_presentation_sync, context, args)

    def _export_presentation_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            pid = args.get("presentation_id")
            fmt = args.get("format", "reveal")

            from tldw_Server_API.app.core.Slides.slides_export import (
                SlidesAssetsMissingError,
                SlidesExportError,
                SlidesExportInputError,
                export_presentation_bundle,
                export_presentation_json,
                export_presentation_markdown,
                export_presentation_pdf,
            )

            pres = db.get_presentation_by_id(pid)
            pres_dict = self._presentation_to_dict(pres)
            slides = self._parse_slides_for_export(pres)
            settings = self._parse_settings(pres.settings)
            visual_style_snapshot = self._parse_visual_style_snapshot(
                getattr(pres, "visual_style_snapshot", None)
            )

            if fmt == "json":
                content = export_presentation_json(pres_dict)
                mime_type = "application/json"
                payload = content.encode("utf-8")
            elif fmt == "markdown":
                try:
                    content = export_presentation_markdown(
                        title=pres.title,
                        slides=slides,
                        theme=pres.theme,
                        marp_theme=getattr(pres, "marp_theme", None),
                    )
                except (SlidesExportInputError, SlidesExportError) as exc:
                    raise ValueError(str(exc)) from exc
                mime_type = "text/markdown"
                payload = content.encode("utf-8")
            elif fmt == "reveal":
                try:
                    payload = export_presentation_bundle(
                        title=pres.title,
                        slides=slides,
                        theme=pres.theme,
                        settings=settings,
                        custom_css=pres.custom_css,
                        visual_style_snapshot=visual_style_snapshot,
                    )
                except SlidesAssetsMissingError as exc:
                    raise ValueError("slides_assets_missing") from exc
                except (SlidesExportInputError, SlidesExportError) as exc:
                    raise ValueError(str(exc)) from exc
                mime_type = "application/zip"
            elif fmt == "pdf":
                try:
                    payload = export_presentation_pdf(
                        title=pres.title,
                        slides=slides,
                        theme=pres.theme,
                        settings=settings,
                        custom_css=pres.custom_css,
                        visual_style_snapshot=visual_style_snapshot,
                    )
                except (SlidesExportInputError, SlidesExportError) as exc:
                    raise ValueError(str(exc)) from exc
                mime_type = "application/pdf"
            else:
                raise ValueError(f"Unknown format: {fmt}")

            import base64
            content_b64 = base64.b64encode(payload).decode("utf-8")

            return {
                "presentation_id": pid,
                "format": fmt,
                "mime_type": mime_type,
                "content_base64": content_b64,
                "size_bytes": len(payload),
                "success": True,
            }
        except KeyError:
            raise ValueError(f"Presentation not found: {args.get('presentation_id')}") from None
        finally:
            db.close_connection()

    def _parse_slides_for_export(self, pres: Any) -> list[dict[str, Any]]:
        raw = pres.slides
        slides: list[dict[str, Any]] = []
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "slides" in data:
                items = data.get("slides") or []
            elif isinstance(data, list):
                items = data
            else:
                items = [{"content": raw}]
        except json.JSONDecodeError:
            items = [{"content": raw}]
        for idx, item in enumerate(items):
            slide = dict(item) if isinstance(item, dict) else {"content": str(item)}
            if "order" not in slide or not isinstance(slide.get("order"), int):
                slide["order"] = idx
            slides.append(slide)
        return slides

    def _parse_settings(self, settings: Any) -> Optional[dict[str, Any]]:
        if settings is None:
            return None
        if isinstance(settings, dict):
            return settings
        if isinstance(settings, str):
            try:
                parsed = json.loads(settings)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _parse_visual_style_snapshot(self, snapshot: Any) -> Optional[dict[str, Any]]:
        if snapshot is None:
            return None
        if isinstance(snapshot, dict):
            return snapshot
        if isinstance(snapshot, str):
            try:
                parsed = json.loads(snapshot)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _to_markdown(self, pres) -> str:
        """Convert presentation to markdown."""
        lines = [f"# {pres.title}", ""]
        if pres.description:
            lines.append(f"_{pres.description}_")
            lines.append("")

        try:
            data = json.loads(pres.slides)
            if isinstance(data, dict) and "slides" in data:
                for slide in data["slides"]:
                    title = slide.get("title", "")
                    content = slide.get("content", "")
                    lines.append(f"## {title}")
                    lines.append("")
                    lines.append(content)
                    lines.append("")
                    lines.append("---")
                    lines.append("")
            else:
                lines.append(pres.slides)
        except json.JSONDecodeError:
            lines.append(pres.slides)

        return "\n".join(lines)

    def _to_reveal_html(self, pres) -> str:
        """Convert presentation to Reveal.js HTML."""
        theme = pres.theme or "black"
        title = pres.title

        slides_html = []
        try:
            data = json.loads(pres.slides)
            if isinstance(data, dict) and "slides" in data:
                for slide in data["slides"]:
                    slide_title = slide.get("title", "")
                    content = slide.get("content", "").replace("\n", "<br>")
                    slides_html.append(f"""
                    <section>
                        <h2>{slide_title}</h2>
                        <p>{content}</p>
                    </section>
                    """)
            else:
                slides_html.append(f"<section><pre>{pres.slides}</pre></section>")
        except json.JSONDecodeError:
            slides_html.append(f"<section><pre>{pres.slides}</pre></section>")

        custom_css = pres.custom_css or ""

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/theme/{theme}.css">
    <style>{custom_css}</style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
            {''.join(slides_html)}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@4/dist/reveal.js"></script>
    <script>
        Reveal.initialize({{
            hash: true,
            slideNumber: true
        }});
    </script>
</body>
</html>"""
