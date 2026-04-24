"""
Flashcards Module for Unified MCP

CRUD operations for flashcard decks and cards with SM-2 spaced repetition,
stored in ChaChaNotes DB. Supports deck management, card creation, review
tracking, tags, and export.
"""

import asyncio
import re
from typing import Any

from loguru import logger

from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb

_FLASHCARDS_NONCRITICAL_EXCEPTIONS = (
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

_FLASHCARDS_VALIDATION_EXCEPTIONS = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
)


class FlashcardsModule(BaseModule):
    """Flashcard management module with spaced repetition for MCP"""

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Flashcards module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Flashcards module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        checks["driver_available"] = CharactersRAGDB is not None
        try:
            from pathlib import Path
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                base = Path(get_project_root())
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS:
                base = Path(__file__).resolve().parents[5]
            free_gb = get_free_disk_space_gb(base)
            checks["disk_space"] = free_gb > 1
        except _FLASHCARDS_NONCRITICAL_EXCEPTIONS:
            checks["disk_space"] = False
        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            # Decks
            create_tool_definition(
                name="flashcards.decks.list",
                description="List flashcard decks.",
                parameters={
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 100},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="flashcards.decks.get",
                description="Get a deck by ID.",
                parameters={
                    "properties": {
                        "deck_id": {"type": "integer"},
                    },
                    "required": ["deck_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="flashcards.decks.create",
                description="Create a new flashcard deck.",
                parameters={
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 256},
                        "description": {"type": "string", "maxLength": 2000},
                    },
                    "required": ["name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Cards
            create_tool_definition(
                name="flashcards.list",
                description="List flashcards with filters.",
                parameters={
                    "properties": {
                        "deck_id": {"type": "integer", "description": "Filter by deck ID"},
                        "tag": {"type": "string", "maxLength": 64, "description": "Filter by tag"},
                        "due_status": {
                            "type": "string",
                            "enum": ["new", "learning", "due", "all"],
                            "default": "all",
                            "description": "Filter by due status",
                        },
                        "q": {"type": "string", "maxLength": 500, "description": "Search query"},
                        "order_by": {
                            "type": "string",
                            "enum": ["due_at", "created_at", "last_modified"],
                            "default": "due_at",
                        },
                        "include_deleted": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="flashcards.get",
                description="Get a flashcard by UUID.",
                parameters={
                    "properties": {
                        "card_uuid": {"type": "string", "description": "Card UUID"},
                    },
                    "required": ["card_uuid"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="flashcards.create",
                description="Create a new flashcard.",
                parameters={
                    "properties": {
                        "deck_id": {"type": "integer", "description": "Target deck ID"},
                        "front": {"type": "string", "minLength": 1, "maxLength": 10000},
                        "back": {"type": "string", "minLength": 1, "maxLength": 10000},
                        "notes": {"type": "string", "maxLength": 5000},
                        "extra": {"type": "string", "maxLength": 5000},
                        "model_type": {
                            "type": "string",
                            "enum": ["basic", "basic_reverse", "cloze"],
                            "default": "basic",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["front", "back"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="flashcards.create_bulk",
                description="Bulk create flashcards.",
                parameters={
                    "properties": {
                        "cards": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "deck_id": {"type": "integer"},
                                    "front": {"type": "string", "minLength": 1, "maxLength": 10000},
                                    "back": {"type": "string", "minLength": 1, "maxLength": 10000},
                                    "notes": {"type": "string", "maxLength": 5000},
                                    "extra": {"type": "string", "maxLength": 5000},
                                    "model_type": {"type": "string", "enum": ["basic", "basic_reverse", "cloze"]},
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["front", "back"],
                            },
                            "minItems": 1,
                            "maxItems": 100,
                        },
                    },
                    "required": ["cards"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="flashcards.update",
                description="Update a flashcard.",
                parameters={
                    "properties": {
                        "card_uuid": {"type": "string"},
                        "updates": {
                            "type": "object",
                            "properties": {
                                "deck_id": {"type": "integer"},
                                "front": {"type": "string", "maxLength": 10000},
                                "back": {"type": "string", "maxLength": 10000},
                                "notes": {"type": "string", "maxLength": 5000},
                                "extra": {"type": "string", "maxLength": 5000},
                                "model_type": {"type": "string", "enum": ["basic", "basic_reverse", "cloze"]},
                            },
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["card_uuid", "updates"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="flashcards.delete",
                description="Soft-delete a flashcard.",
                parameters={
                    "properties": {
                        "card_uuid": {"type": "string"},
                        "expected_version": {"type": "integer"},
                    },
                    "required": ["card_uuid", "expected_version"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Spaced Repetition
            create_tool_definition(
                name="flashcards.review",
                description="Record a flashcard review with SM-2 scheduling.",
                parameters={
                    "properties": {
                        "card_uuid": {"type": "string"},
                        "rating": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 5,
                            "description": "Anki-style rating: 0=again, 1=hard, 2=okay, 3=good, 4=easy, 5=perfect",
                        },
                        "answer_time_ms": {"type": "integer", "minimum": 0, "description": "Time to answer in ms"},
                    },
                    "required": ["card_uuid", "rating"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            # Tags
            create_tool_definition(
                name="flashcards.tags.set",
                description="Set tags on a flashcard (replaces existing).",
                parameters={
                    "properties": {
                        "card_uuid": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["card_uuid", "tags"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="flashcards.tags.get",
                description="Get tags for a flashcard.",
                parameters={
                    "properties": {
                        "card_uuid": {"type": "string"},
                    },
                    "required": ["card_uuid"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            # Export
            create_tool_definition(
                name="flashcards.export",
                description="Export flashcards to CSV/TSV or APKG.",
                parameters={
                    "properties": {
                        "deck_id": {"type": "integer", "description": "Filter by deck"},
                        "tag": {"type": "string", "maxLength": 64, "description": "Filter by tag"},
                        "q": {"type": "string", "maxLength": 500, "description": "Search query"},
                        "format": {
                            "type": "string",
                            "enum": ["csv", "tsv", "apkg"],
                            "default": "csv",
                            "description": "Export format",
                        },
                        "include_reverse": {"type": "boolean", "default": False, "description": "Include reverse cards in APKG export"},
                        "include_header": {"type": "boolean", "default": True},
                        "extended_header": {"type": "boolean", "default": False},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "flashcards.decks.list":
            limit = int(arguments.get("limit", 100))
            if limit < 1 or limit > 100:
                raise ValueError("limit must be 1..100")
        elif tool_name == "flashcards.decks.get":
            deck_id = arguments.get("deck_id")
            if not isinstance(deck_id, int) or deck_id < 1:
                raise ValueError("deck_id must be a positive integer")
        elif tool_name == "flashcards.decks.create":
            name = arguments.get("name")
            if not isinstance(name, str) or not (1 <= len(name.strip()) <= 256):
                raise ValueError("name must be 1..256 chars")
        elif tool_name == "flashcards.list":
            limit = int(arguments.get("limit", 100))
            if limit < 1 or limit > 200:
                raise ValueError("limit must be 1..200")
            due_status = arguments.get("due_status", "all")
            if due_status not in {"new", "learning", "due", "all"}:
                raise ValueError("due_status must be new, learning, due, or all")
        elif tool_name == "flashcards.get":
            card_uuid = arguments.get("card_uuid")
            if not isinstance(card_uuid, str) or not card_uuid.strip():
                raise ValueError("card_uuid must be a non-empty string")
        elif tool_name == "flashcards.create":
            front = arguments.get("front")
            back = arguments.get("back")
            if not isinstance(front, str) or not (1 <= len(front) <= 10000):
                raise ValueError("front must be 1..10000 chars")
            if not isinstance(back, str) or not (1 <= len(back) <= 10000):
                raise ValueError("back must be 1..10000 chars")
            model_type = arguments.get("model_type", "basic")
            if model_type == "cloze":
                self._validate_cloze_front(front)
        elif tool_name == "flashcards.create_bulk":
            cards = arguments.get("cards")
            if not isinstance(cards, list) or not (1 <= len(cards) <= 100):
                raise ValueError("cards must be a list of 1..100 items")
            for i, card in enumerate(cards):
                if not isinstance(card, dict):
                    raise ValueError(f"card[{i}] must be an object")
                front = card.get("front")
                back = card.get("back")
                if not isinstance(front, str) or not (1 <= len(front) <= 10000):
                    raise ValueError(f"card[{i}].front must be 1..10000 chars")
                if not isinstance(back, str) or not (1 <= len(back) <= 10000):
                    raise ValueError(f"card[{i}].back must be 1..10000 chars")
                if card.get("model_type") == "cloze":
                    self._validate_cloze_front(front)
        elif tool_name == "flashcards.update":
            card_uuid = arguments.get("card_uuid")
            if not isinstance(card_uuid, str) or not card_uuid.strip():
                raise ValueError("card_uuid must be a non-empty string")
            updates = arguments.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ValueError("updates must be a non-empty object")
            if "front" in updates:
                front = updates["front"]
                if not isinstance(front, str) or len(front) > 10000:
                    raise ValueError("front must be <= 10000 chars")
            if "model_type" in updates and updates["model_type"] == "cloze":
                front = updates.get("front")
                if front:
                    self._validate_cloze_front(front)
        elif tool_name == "flashcards.delete":
            card_uuid = arguments.get("card_uuid")
            if not isinstance(card_uuid, str) or not card_uuid.strip():
                raise ValueError("card_uuid must be a non-empty string")
            ev = arguments.get("expected_version")
            if not isinstance(ev, int) or ev < 1:
                raise ValueError("expected_version must be a positive integer")
        elif tool_name == "flashcards.review":
            card_uuid = arguments.get("card_uuid")
            if not isinstance(card_uuid, str) or not card_uuid.strip():
                raise ValueError("card_uuid must be a non-empty string")
            rating = arguments.get("rating")
            if not isinstance(rating, int) or rating < 0 or rating > 5:
                raise ValueError("rating must be 0..5")
        elif tool_name == "flashcards.tags.set":
            card_uuid = arguments.get("card_uuid")
            if not isinstance(card_uuid, str) or not card_uuid.strip():
                raise ValueError("card_uuid must be a non-empty string")
            tags = arguments.get("tags")
            self._validate_tags(tags)
        elif tool_name == "flashcards.tags.get":
            card_uuid = arguments.get("card_uuid")
            if not isinstance(card_uuid, str) or not card_uuid.strip():
                raise ValueError("card_uuid must be a non-empty string")
        elif tool_name == "flashcards.export":
            fmt = arguments.get("format", "csv")
            if fmt not in {"csv", "tsv", "apkg"}:
                raise ValueError("format must be csv, tsv, or apkg")

    def _validate_cloze_front(self, front: str) -> None:
        """Validate cloze deletion format: must contain {{cN::...}} pattern."""
        if not re.search(r'\{\{c\d+::', front):
            raise ValueError("Cloze card front must contain {{cN::...}} pattern (e.g., {{c1::answer}})")

    def _validate_tags(self, tags: Any) -> None:
        if not isinstance(tags, list):
            raise ValueError("tags must be a list of strings")
        for t in tags:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("tags must be non-empty strings")
            if len(t.strip()) > 64:
                raise ValueError("each tag must be <= 64 chars")
        if len(tags) > 50:
            raise ValueError("tags must contain <= 50 items")

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except _FLASHCARDS_VALIDATION_EXCEPTIONS as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve

        if tool_name == "flashcards.decks.list":
            return await self._list_decks(args, context)
        if tool_name == "flashcards.decks.get":
            return await self._get_deck(args, context)
        if tool_name == "flashcards.decks.create":
            return await self._create_deck(args, context)
        if tool_name == "flashcards.list":
            return await self._list_cards(args, context)
        if tool_name == "flashcards.get":
            return await self._get_card(args, context)
        if tool_name == "flashcards.create":
            return await self._create_card(args, context)
        if tool_name == "flashcards.create_bulk":
            return await self._create_cards_bulk(args, context)
        if tool_name == "flashcards.update":
            return await self._update_card(args, context)
        if tool_name == "flashcards.delete":
            return await self._delete_card(args, context)
        if tool_name == "flashcards.review":
            return await self._review_card(args, context)
        if tool_name == "flashcards.tags.set":
            return await self._set_tags(args, context)
        if tool_name == "flashcards.tags.get":
            return await self._get_tags(args, context)
        if tool_name == "flashcards.export":
            return await self._export_cards(args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    def _open_db(self, context: Any) -> CharactersRAGDB:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Flashcards access")
        chacha_path = context.db_paths.get("chacha")
        if not chacha_path:
            raise ValueError("ChaChaNotes DB path not available in context")
        return CharactersRAGDB(db_path=chacha_path, client_id=f"mcp_flashcards_{self.config.name}")

    def _workspace_id_from_context(self, context: Any | None) -> str | None:
        metadata = getattr(context, "metadata", None) or {}
        workspace_id = metadata.get("workspace_id")
        if workspace_id is None:
            return None
        text = str(workspace_id).strip()
        return text or None

    def _assert_card_workspace(self, card: dict[str, Any] | None, workspace_id: str | None) -> None:
        if workspace_id is None or card is None:
            return
        if str(card.get("workspace_id") or "").strip() != workspace_id:
            raise PermissionError("Flashcard access denied for workspace")

    def _assert_deck_workspace(self, deck: dict[str, Any] | None, workspace_id: str | None) -> None:
        if workspace_id is None or deck is None:
            return
        if str(deck.get("workspace_id") or "").strip() != workspace_id:
            raise PermissionError("Flashcard access denied for workspace")

    # Decks

    async def _list_decks(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        limit = int(args.get("limit", 100))
        offset = int(args.get("offset", 0))
        include_deleted = bool(args.get("include_deleted", False))
        return await asyncio.to_thread(self._list_decks_sync, context, limit, offset, include_deleted)

    def _list_decks_sync(
        self, context: Any, limit: int, offset: int, include_deleted: bool
    ) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            decks = db.list_decks(
                limit=limit,
                offset=offset,
                include_deleted=include_deleted,
                workspace_id=workspace_id,
                include_workspace_items=True,
            )
            return {
                "decks": decks,
                "total": len(decks),
                "has_more": len(decks) >= limit,
                "next_offset": offset + len(decks) if len(decks) >= limit else None,
            }
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _get_deck(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        deck_id = args.get("deck_id")
        return await asyncio.to_thread(self._get_deck_sync, context, deck_id)

    def _get_deck_sync(self, context: Any, deck_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            deck = db.get_deck(deck_id)
            self._assert_deck_workspace(deck, workspace_id)
            if not deck:
                raise ValueError(f"Deck not found: {deck_id}")
            return {"deck": deck}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _create_deck(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_deck_sync, context, args)

    def _create_deck_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            deck_id = db.add_deck(
                name=args.get("name"),
                description=args.get("description"),
                workspace_id=workspace_id,
            )
            deck = db.get_deck(deck_id)
            return {"deck_id": deck_id, "success": True, "deck": deck}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Cards

    async def _list_cards(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_cards_sync, context, args)

    def _list_cards_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            cards = db.list_flashcards(
                deck_id=args.get("deck_id"),
                workspace_id=workspace_id,
                include_workspace_items=True,
                tag=args.get("tag"),
                due_status=args.get("due_status", "all"),
                q=args.get("q"),
                include_deleted=bool(args.get("include_deleted", False)),
                limit=int(args.get("limit", 100)),
                offset=int(args.get("offset", 0)),
                order_by=args.get("order_by", "due_at"),
            )
            count = db.count_flashcards(
                deck_id=args.get("deck_id"),
                workspace_id=workspace_id,
                include_workspace_items=True,
                tag=args.get("tag"),
                due_status=args.get("due_status", "all"),
                q=args.get("q"),
                include_deleted=bool(args.get("include_deleted", False)),
            )
            int(args.get("limit", 100))
            offset = int(args.get("offset", 0))
            has_more = offset + len(cards) < count
            return {
                "cards": cards,
                "total": count,
                "has_more": has_more,
                "next_offset": offset + len(cards) if has_more else None,
            }
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _get_card(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        card_uuid = args.get("card_uuid")
        return await asyncio.to_thread(self._get_card_sync, context, card_uuid)

    def _get_card_sync(self, context: Any, card_uuid: str) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            card = db.get_flashcard(card_uuid)
            self._assert_card_workspace(card, workspace_id)
            if not card:
                raise ValueError(f"Card not found: {card_uuid}")
            return {"card": card}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _create_card(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_card_sync, context, args)

    def _create_card_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            model_type = args.get("model_type", "basic")
            is_cloze = model_type == "cloze"
            deck_id = args.get("deck_id")
            deck = db.get_deck(deck_id)
            if deck is None:
                raise ValueError(f"Deck not found: {deck_id}")
            self._assert_deck_workspace(deck, workspace_id)
            card_data = {
                "front": args.get("front"),
                "back": args.get("back"),
                "deck_id": deck_id,
                "notes": args.get("notes"),
                "extra": args.get("extra"),
                "is_cloze": is_cloze,
                "model_type": model_type,
                "reverse": model_type == "basic_reverse",
            }
            tags = args.get("tags")
            if tags:
                import json
                card_data["tags_json"] = json.dumps(tags)
            card_uuid = db.add_flashcard(card_data)
            card = db.get_flashcard(card_uuid)
            return {"card_uuid": card_uuid, "success": True, "card": card}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _create_cards_bulk(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._create_cards_bulk_sync, context, args)

    def _create_cards_bulk_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            cards_input = args.get("cards", [])
            cards_data = []
            for card_args in cards_input:
                model_type = card_args.get("model_type", "basic")
                is_cloze = model_type == "cloze"
                deck_id = card_args.get("deck_id")
                deck = db.get_deck(deck_id)
                if deck is None:
                    raise ValueError(f"Deck not found: {deck_id}")
                self._assert_deck_workspace(deck, workspace_id)
                card_data = {
                    "front": card_args.get("front"),
                    "back": card_args.get("back"),
                    "deck_id": deck_id,
                    "notes": card_args.get("notes"),
                    "extra": card_args.get("extra"),
                    "is_cloze": is_cloze,
                    "model_type": model_type,
                    "reverse": model_type == "basic_reverse",
                }
                tags = card_args.get("tags")
                if tags:
                    import json
                    card_data["tags_json"] = json.dumps(tags)
                cards_data.append(card_data)

            card_uuids = db.add_flashcards_bulk(cards_data)
            return {
                "card_uuids": card_uuids,
                "count": len(card_uuids),
                "success": True,
            }
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _update_card(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._update_card_sync, context, args)

    def _update_card_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            card_uuid = args.get("card_uuid")
            updates = args.get("updates", {})
            expected_version = args.get("expected_version")
            tags = args.get("tags")
            workspace_id = self._workspace_id_from_context(context)

            card = db.get_flashcard(card_uuid)
            self._assert_card_workspace(card, workspace_id)

            if "deck_id" in updates and workspace_id is not None:
                destination_deck = db.get_deck(updates["deck_id"])
                if destination_deck is None:
                    raise ValueError(f"Deck not found: {updates['deck_id']}")
                self._assert_deck_workspace(destination_deck, workspace_id)

            # Handle model_type changes
            if "model_type" in updates:
                model_type = updates["model_type"]
                updates["is_cloze"] = model_type == "cloze"
                updates["reverse"] = model_type == "basic_reverse"

            success = db.update_flashcard(
                card_uuid=card_uuid,
                updates=updates,
                expected_version=expected_version,
                tags=tags,
            )
            if not success:
                raise ValueError(f"Card not found or version conflict: {card_uuid}")
            return {"card_uuid": card_uuid, "success": True, "updated_fields": list(updates.keys())}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _delete_card(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._delete_card_sync, context, args)

    def _delete_card_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            card_uuid = args.get("card_uuid")
            expected_version = args.get("expected_version")
            workspace_id = self._workspace_id_from_context(context)

            card = db.get_flashcard(card_uuid)
            self._assert_card_workspace(card, workspace_id)
            success = db.soft_delete_flashcard(card_uuid, expected_version)
            if not success:
                raise ValueError(f"Card not found or version conflict: {card_uuid}")
            return {"card_uuid": card_uuid, "action": "soft_deleted", "success": True}
        except ConflictError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Spaced Repetition

    async def _review_card(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._review_card_sync, context, args)

    def _review_card_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            card_uuid = args.get("card_uuid")
            rating = args.get("rating")
            answer_time_ms = args.get("answer_time_ms")
            workspace_id = self._workspace_id_from_context(context)

            card = db.get_flashcard(card_uuid)
            self._assert_card_workspace(card, workspace_id)
            result = db.review_flashcard(
                card_uuid=card_uuid,
                rating=rating,
                answer_time_ms=answer_time_ms,
            )
            return {"card_uuid": card_uuid, "review_result": result, "success": True}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Tags

    async def _set_tags(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._set_tags_sync, context, args)

    def _set_tags_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            card_uuid = args.get("card_uuid")
            tags = args.get("tags", [])
            workspace_id = self._workspace_id_from_context(context)

            card = db.get_flashcard(card_uuid)
            self._assert_card_workspace(card, workspace_id)
            # Normalize tags
            norm_tags = [t.strip().lower() for t in tags if t.strip()][:50]
            success = db.set_flashcard_tags(card_uuid, norm_tags)
            if not success:
                raise ValueError(f"Card not found: {card_uuid}")
            return {"card_uuid": card_uuid, "tags": norm_tags, "success": True}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    async def _get_tags(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_tags_sync, context, args)

    def _get_tags_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            card_uuid = args.get("card_uuid")
            workspace_id = self._workspace_id_from_context(context)
            card = db.get_flashcard(card_uuid)
            self._assert_card_workspace(card, workspace_id)
            keywords = db.get_keywords_for_flashcard(card_uuid)
            tags = [str(kw.get("keyword", "")).lower() for kw in keywords if kw.get("keyword")]
            return {"card_uuid": card_uuid, "tags": tags}
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")

    # Export

    async def _export_cards(self, args: dict[str, Any], context: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._export_cards_sync, context, args)

    def _export_cards_sync(self, context: Any, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        try:
            workspace_id = self._workspace_id_from_context(context)
            deck_id = args.get("deck_id")
            tag = args.get("tag")
            q = args.get("q")
            fmt = args.get("format", "csv")
            include_reverse = bool(args.get("include_reverse", False))
            include_header = bool(args.get("include_header", True))
            extended_header = bool(args.get("extended_header", False))

            if fmt == "apkg":
                from ....Flashcards.apkg_exporter import export_apkg_from_rows
                items = db.list_flashcards(
                    deck_id=deck_id,
                    workspace_id=workspace_id,
                    include_workspace_items=True,
                    tag=tag,
                    q=q,
                    due_status="all",
                    include_deleted=False,
                    limit=100000,
                    offset=0,
                )
                apkg_bytes = export_apkg_from_rows(items, include_reverse=include_reverse)
                import base64
                content_b64 = base64.b64encode(apkg_bytes).decode("utf-8")
                return {
                    "format": fmt,
                    "mime_type": "application/apkg",
                    "content_base64": content_b64,
                    "size_bytes": len(apkg_bytes),
                    "success": True,
                }

            delimiter = "\t" if fmt == "tsv" else ","
            csv_bytes = db.export_flashcards_csv(
                deck_id=deck_id,
                workspace_id=workspace_id,
                include_workspace_items=True,
                tag=tag,
                q=q,
                delimiter=delimiter,
                include_header=include_header,
                extended_header=extended_header,
            )

            # Return as base64 for safe transport
            import base64
            content_b64 = base64.b64encode(csv_bytes).decode("utf-8")

            return {
                "format": fmt,
                "mime_type": "text/tab-separated-values" if fmt == "tsv" else "text/csv",
                "content_base64": content_b64,
                "size_bytes": len(csv_bytes),
                "success": True,
            }
        finally:
            try:
                db.close_all_connections()
            except _FLASHCARDS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to close DB: {exc}")
