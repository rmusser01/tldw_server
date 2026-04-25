"""
Kanban Module for Unified MCP

Lightweight MCP wrapper around the Kanban DB for boards, lists, and cards.
"""

import asyncio
import uuid
from typing import Any, Optional

from loguru import logger

from ....DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)
from ..base import BaseModule, create_tool_definition
from ..disk_space import get_free_disk_space_gb


def _ensure_positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive integer") from None
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _parse_bool_like(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{field} must be a boolean-like value")


def _bool_like_case_variants(token: str) -> set[str]:
    variants = {""}
    for char in token:
        if char.isalpha():
            variants = {prefix + variant for prefix in variants for variant in (char.lower(), char.upper())}
        else:
            variants = {prefix + char for prefix in variants}
    return variants


_BOOL_LIKE_STRING_TOKENS = tuple(
    sorted(
        {
            variant
            for token in ("1", "true", "yes", "y", "on", "0", "false", "no", "n", "off")
            for variant in _bool_like_case_variants(token)
        }
    )
)


class KanbanModule(BaseModule):
    """Kanban MCP module exposing boards, lists, and cards."""

    @staticmethod
    def _bool_like_schema(description: str | None = None) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "oneOf": [
                {"type": "boolean"},
                {"type": "integer", "enum": [0, 1]},
                {"type": "string", "enum": list(_BOOL_LIKE_STRING_TOKENS)},
            ],
        }
        if description:
            schema["description"] = description
        return schema

    async def on_initialize(self) -> None:
        logger.info(f"Initializing Kanban module: {self.name}")

    async def on_shutdown(self) -> None:
        logger.info(f"Shutting down Kanban module: {self.name}")

    async def check_health(self) -> dict[str, bool]:
        checks = {"initialized": True, "driver_available": False, "disk_space": False}
        try:
            _ = KanbanDB  # noqa: F401
            checks["driver_available"] = True
        except NameError:
            checks["driver_available"] = False
        try:
            from pathlib import Path
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                base = Path(get_project_root())
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                base = Path(__file__).resolve().parents[5]
            free_gb = get_free_disk_space_gb(base)
            checks["disk_space"] = free_gb > 1
        except (AttributeError, OSError, TypeError, ValueError):
            checks["disk_space"] = False
        return checks

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="kanban.boards.list",
                description="List Kanban boards for the current user.",
                parameters={
                    "properties": {
                        "include_archived": {"type": "boolean", "default": False},
                        "include_deleted": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    }
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.boards.get",
                description="Get a Kanban board by id.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.boards.create",
                description="Create a new Kanban board.",
                parameters={
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "description": {"type": "string"},
                        "client_id": {"type": "string"},
                        "activity_retention_days": {"type": "integer", "minimum": 1},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.list",
                description="List labels for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.create",
                description="Create a new label for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "color": {"type": "string"},
                    },
                    "required": ["board_id", "name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.update",
                description="Update a label name or color.",
                parameters={
                    "properties": {
                        "label_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "color": {"type": "string"},
                    },
                    "required": ["label_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.delete",
                description="Delete a label.",
                parameters={
                    "properties": {
                        "label_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["label_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.assign",
                description="Assign a label to a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "label_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["card_id", "label_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.remove",
                description="Remove a label from a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "label_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["card_id", "label_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.labels.list_for_card",
                description="List labels assigned to a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["card_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.lists.list",
                description="List lists for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                        "include_archived": {"type": "boolean", "default": False},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.lists.create",
                description="Create a new list in a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "client_id": {"type": "string"},
                        "position": {"type": "integer", "minimum": 0},
                    },
                    "required": ["board_id", "name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.cards.list",
                description="List cards for a list.",
                parameters={
                    "properties": {
                        "list_id": {"type": "integer", "minimum": 1},
                        "include_archived": {"type": "boolean", "default": False},
                        "include_deleted": {"type": "boolean", "default": False},
                    },
                    "required": ["list_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.cards.create",
                description="Create a new card in a list.",
                parameters={
                    "properties": {
                        "list_id": {"type": "integer", "minimum": 1},
                        "title": {"type": "string", "minLength": 1, "maxLength": 500},
                        "client_id": {"type": "string"},
                        "description": {"type": "string"},
                        "position": {"type": "integer", "minimum": 0},
                        "due_date": {"type": "string"},
                        "start_date": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                        "metadata": {"type": "object"},
                    },
                    "required": ["list_id", "title"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.cards.move",
                description="Move a card to a different list (lane).",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "target_list_id": {"type": "integer", "minimum": 1},
                        "position": {"type": "integer", "minimum": 0},
                    },
                    "required": ["card_id", "target_list_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.cards.search",
                description="Search cards with FTS.",
                parameters={
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 1000},
                        "board_id": {"type": "integer", "minimum": 1},
                        "label_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                        "include_archived": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "required": ["query"],
                },
                metadata={"category": "search", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.comments.list",
                description="List comments for a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "include_deleted": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "required": ["card_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.comments.create",
                description="Create a comment on a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "content": {"type": "string", "minLength": 1, "maxLength": 10000},
                    },
                    "required": ["card_id", "content"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.comments.update",
                description="Update a comment.",
                parameters={
                    "properties": {
                        "comment_id": {"type": "integer", "minimum": 1},
                        "content": {"type": "string", "minLength": 1, "maxLength": 10000},
                    },
                    "required": ["comment_id", "content"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.comments.delete",
                description="Delete a comment.",
                parameters={
                    "properties": {
                        "comment_id": {"type": "integer", "minimum": 1},
                        "hard_delete": {"type": "boolean", "default": False},
                    },
                    "required": ["comment_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.list",
                description="List checklists for a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["card_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.create",
                description="Create a checklist on a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "position": {"type": "integer", "minimum": 0},
                    },
                    "required": ["card_id", "name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.update",
                description="Update a checklist name.",
                parameters={
                    "properties": {
                        "checklist_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                    },
                    "required": ["checklist_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.delete",
                description="Delete a checklist.",
                parameters={
                    "properties": {
                        "checklist_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["checklist_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.items.list",
                description="List checklist items.",
                parameters={
                    "properties": {
                        "checklist_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["checklist_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.items.create",
                description="Create a checklist item.",
                parameters={
                    "properties": {
                        "checklist_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 500},
                        "position": {"type": "integer", "minimum": 0},
                        "checked": {"type": "boolean", "default": False},
                    },
                    "required": ["checklist_id", "name"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.items.update",
                description="Update a checklist item (name or checked).",
                parameters={
                    "properties": {
                        "item_id": {"type": "integer", "minimum": 1},
                        "name": {"type": "string", "minLength": 1, "maxLength": 500},
                        "checked": {"type": "boolean"},
                    },
                    "required": ["item_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.checklists.items.delete",
                description="Delete a checklist item.",
                parameters={
                    "properties": {
                        "item_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["item_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.policy.get",
                description="Get workflow policy for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.policy.upsert",
                description="Upsert workflow policy, statuses, and transitions for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                        "statuses": {"type": "array", "items": {"type": "object"}},
                        "transitions": {"type": "array", "items": {"type": "object"}},
                        "is_paused": self._bool_like_schema("Boolean-like value: true/false, yes/no, on/off, or 1/0."),
                        "is_draining": self._bool_like_schema("Boolean-like value: true/false, yes/no, on/off, or 1/0."),
                        "default_lease_ttl_sec": {"type": "integer", "minimum": 1},
                        "strict_projection": self._bool_like_schema("Boolean-like value: true/false, yes/no, on/off, or 1/0."),
                        "metadata": {"type": "object"},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.statuses.list",
                description="List workflow statuses for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.transitions.list",
                description="List workflow transitions for a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.state.get",
                description="Get workflow runtime state for a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["card_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.state.patch",
                description="Patch workflow runtime state for a card using optimistic concurrency.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "workflow_status_key": {"type": "string", "minLength": 1},
                        "expected_version": {"type": "integer", "minimum": 1},
                        "lease_owner": {"type": "string"},
                        "idempotency_key": {"type": "string", "minLength": 1},
                        "correlation_id": {"type": "string", "minLength": 1},
                        "last_actor": {"type": "string", "minLength": 1},
                    },
                    "required": ["card_id", "expected_version", "idempotency_key"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.claim",
                description="Claim a workflow lease for a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "owner": {"type": "string", "minLength": 1},
                        "lease_ttl_sec": {"type": "integer", "minimum": 1},
                        "idempotency_key": {"type": "string", "minLength": 1},
                        "correlation_id": {"type": "string", "minLength": 1},
                    },
                    "required": ["card_id", "owner", "idempotency_key"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.release",
                description="Release a workflow lease for a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "owner": {"type": "string", "minLength": 1},
                        "idempotency_key": {"type": "string", "minLength": 1},
                        "correlation_id": {"type": "string", "minLength": 1},
                    },
                    "required": ["card_id", "owner", "idempotency_key"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.transition",
                description="Transition a card workflow status with policy/CAS enforcement.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "to_status_key": {"type": "string", "minLength": 1},
                        "actor": {"type": "string", "minLength": 1},
                        "expected_version": {"type": "integer", "minimum": 1},
                        "idempotency_key": {"type": "string", "minLength": 1},
                        "correlation_id": {"type": "string", "minLength": 1},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "card_id",
                        "to_status_key",
                        "actor",
                        "expected_version",
                        "idempotency_key",
                        "correlation_id",
                    ],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.approval.decide",
                description="Record approval decision for pending workflow transition.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "reviewer": {"type": "string", "minLength": 1},
                        "decision": {"type": "string", "enum": ["approved", "rejected"]},
                        "expected_version": {"type": "integer", "minimum": 1},
                        "idempotency_key": {"type": "string", "minLength": 1},
                        "correlation_id": {"type": "string", "minLength": 1},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "card_id",
                        "reviewer",
                        "decision",
                        "expected_version",
                        "idempotency_key",
                        "correlation_id",
                    ],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.task.events.list",
                description="List workflow events for a card.",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "required": ["card_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.control.pause",
                description="Pause workflow transitions for a board (admin only).",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "management", "auth_required": True, "admin_only": True},
            ),
            create_tool_definition(
                name="kanban.workflow.control.resume",
                description="Resume workflow transitions for a board (admin only).",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "management", "auth_required": True, "admin_only": True},
            ),
            create_tool_definition(
                name="kanban.workflow.control.drain",
                description="Enable workflow drain mode for a board (admin only).",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["board_id"],
                },
                metadata={"category": "management", "auth_required": True, "admin_only": True},
            ),
            create_tool_definition(
                name="kanban.workflow.recovery.list_stale_claims",
                description="List stale workflow claims optionally scoped to a board.",
                parameters={
                    "properties": {
                        "board_id": {"type": "integer", "minimum": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="kanban.workflow.recovery.force_reassign",
                description="Force-reassign workflow claim to another owner (admin only).",
                parameters={
                    "properties": {
                        "card_id": {"type": "integer", "minimum": 1},
                        "new_owner": {"type": "string", "minLength": 1},
                        "idempotency_key": {"type": "string", "minLength": 1},
                        "correlation_id": {"type": "string", "minLength": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["card_id", "new_owner", "idempotency_key", "correlation_id"],
                },
                metadata={"category": "management", "auth_required": True, "admin_only": True},
            ),
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments)
        try:
            self.validate_tool_arguments(tool_name, args)
        except (TypeError, ValueError) as ve:
            raise ValueError(f"Invalid arguments for {tool_name}: {ve}") from ve
        try:
            if tool_name == "kanban.boards.list":
                return await self._list_boards(args, context)
            if tool_name == "kanban.boards.get":
                return await self._get_board(args, context)
            if tool_name == "kanban.boards.create":
                return await self._create_board(args, context)
            if tool_name == "kanban.labels.list":
                return await self._list_labels(args, context)
            if tool_name == "kanban.labels.create":
                return await self._create_label(args, context)
            if tool_name == "kanban.labels.update":
                return await self._update_label(args, context)
            if tool_name == "kanban.labels.delete":
                return await self._delete_label(args, context)
            if tool_name == "kanban.labels.assign":
                return await self._assign_label(args, context)
            if tool_name == "kanban.labels.remove":
                return await self._remove_label(args, context)
            if tool_name == "kanban.labels.list_for_card":
                return await self._list_card_labels(args, context)
            if tool_name == "kanban.lists.list":
                return await self._list_lists(args, context)
            if tool_name == "kanban.lists.create":
                return await self._create_list(args, context)
            if tool_name == "kanban.cards.list":
                return await self._list_cards(args, context)
            if tool_name == "kanban.cards.create":
                return await self._create_card(args, context)
            if tool_name == "kanban.cards.move":
                return await self._move_card(args, context)
            if tool_name == "kanban.cards.search":
                return await self._search_cards(args, context)
            if tool_name == "kanban.comments.list":
                return await self._list_comments(args, context)
            if tool_name == "kanban.comments.create":
                return await self._create_comment(args, context)
            if tool_name == "kanban.comments.update":
                return await self._update_comment(args, context)
            if tool_name == "kanban.comments.delete":
                return await self._delete_comment(args, context)
            if tool_name == "kanban.checklists.list":
                return await self._list_checklists(args, context)
            if tool_name == "kanban.checklists.create":
                return await self._create_checklist(args, context)
            if tool_name == "kanban.checklists.update":
                return await self._update_checklist(args, context)
            if tool_name == "kanban.checklists.delete":
                return await self._delete_checklist(args, context)
            if tool_name == "kanban.checklists.items.list":
                return await self._list_checklist_items(args, context)
            if tool_name == "kanban.checklists.items.create":
                return await self._create_checklist_item(args, context)
            if tool_name == "kanban.checklists.items.update":
                return await self._update_checklist_item(args, context)
            if tool_name == "kanban.checklists.items.delete":
                return await self._delete_checklist_item(args, context)
            if tool_name == "kanban.workflow.policy.get":
                return await self._workflow_policy_get(args, context)
            if tool_name == "kanban.workflow.policy.upsert":
                return await self._workflow_policy_upsert(args, context)
            if tool_name == "kanban.workflow.statuses.list":
                return await self._workflow_statuses_list(args, context)
            if tool_name == "kanban.workflow.transitions.list":
                return await self._workflow_transitions_list(args, context)
            if tool_name == "kanban.workflow.task.state.get":
                return await self._workflow_task_state_get(args, context)
            if tool_name == "kanban.workflow.task.state.patch":
                return await self._workflow_task_state_patch(args, context)
            if tool_name == "kanban.workflow.task.claim":
                return await self._workflow_task_claim(args, context)
            if tool_name == "kanban.workflow.task.release":
                return await self._workflow_task_release(args, context)
            if tool_name == "kanban.workflow.task.transition":
                return await self._workflow_task_transition(args, context)
            if tool_name == "kanban.workflow.task.approval.decide":
                return await self._workflow_task_approval_decide(args, context)
            if tool_name == "kanban.workflow.task.events.list":
                return await self._workflow_task_events_list(args, context)
            if tool_name == "kanban.workflow.control.pause":
                return await self._workflow_control_pause(args, context)
            if tool_name == "kanban.workflow.control.resume":
                return await self._workflow_control_resume(args, context)
            if tool_name == "kanban.workflow.control.drain":
                return await self._workflow_control_drain(args, context)
            if tool_name == "kanban.workflow.recovery.list_stale_claims":
                return await self._workflow_recovery_list_stale_claims(args, context)
            if tool_name == "kanban.workflow.recovery.force_reassign":
                return await self._workflow_recovery_force_reassign(args, context)
        except (InputError, ConflictError, NotFoundError, KanbanDBError) as exc:
            raise ValueError(str(exc)) from exc
        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]):
        if tool_name == "kanban.boards.list":
            limit = int(arguments.get("limit", 50))
            offset = int(arguments.get("offset", 0))
            if limit < 1 or limit > 200:
                raise ValueError("limit must be 1..200")
            if offset < 0:
                raise ValueError("offset must be >= 0")
        elif tool_name == "kanban.boards.get":
            _ensure_positive_int(arguments.get("board_id"), "board_id")
        elif tool_name == "kanban.boards.create":
            name = arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("name must be a non-empty string")
        elif tool_name == "kanban.labels.list":
            _ensure_positive_int(arguments.get("board_id"), "board_id")
        elif tool_name == "kanban.labels.create":
            _ensure_positive_int(arguments.get("board_id"), "board_id")
            name = arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("name must be a non-empty string")
        elif tool_name == "kanban.labels.update":
            _ensure_positive_int(arguments.get("label_id"), "label_id")
            name = arguments.get("name")
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise ValueError("name must be a non-empty string")
            if name is None and arguments.get("color") is None:
                raise ValueError("must provide name or color")
        elif tool_name == "kanban.labels.delete":
            _ensure_positive_int(arguments.get("label_id"), "label_id")
        elif tool_name == "kanban.labels.assign" or tool_name == "kanban.labels.remove":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            _ensure_positive_int(arguments.get("label_id"), "label_id")
        elif tool_name == "kanban.labels.list_for_card":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
        elif tool_name == "kanban.lists.list":
            _ensure_positive_int(arguments.get("board_id"), "board_id")
        elif tool_name == "kanban.lists.create":
            _ensure_positive_int(arguments.get("board_id"), "board_id")
            name = arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("name must be a non-empty string")
        elif tool_name == "kanban.cards.list":
            _ensure_positive_int(arguments.get("list_id"), "list_id")
        elif tool_name == "kanban.cards.create":
            _ensure_positive_int(arguments.get("list_id"), "list_id")
            title = arguments.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string")
        elif tool_name == "kanban.cards.move":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            _ensure_positive_int(arguments.get("target_list_id"), "target_list_id")
        elif tool_name == "kanban.cards.search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            limit = int(arguments.get("limit", 50))
            offset = int(arguments.get("offset", 0))
            if limit < 1 or limit > 200:
                raise ValueError("limit must be 1..200")
            if offset < 0:
                raise ValueError("offset must be >= 0")
        elif tool_name == "kanban.comments.list":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            limit = int(arguments.get("limit", 50))
            offset = int(arguments.get("offset", 0))
            if limit < 1 or limit > 200:
                raise ValueError("limit must be 1..200")
            if offset < 0:
                raise ValueError("offset must be >= 0")
        elif tool_name == "kanban.comments.create":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be a non-empty string")
        elif tool_name == "kanban.comments.update":
            _ensure_positive_int(arguments.get("comment_id"), "comment_id")
            content = arguments.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be a non-empty string")
        elif tool_name == "kanban.comments.delete":
            _ensure_positive_int(arguments.get("comment_id"), "comment_id")
        elif tool_name == "kanban.checklists.list":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
        elif tool_name == "kanban.checklists.create":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            name = arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("name must be a non-empty string")
        elif tool_name == "kanban.checklists.update":
            _ensure_positive_int(arguments.get("checklist_id"), "checklist_id")
            name = arguments.get("name")
            if name is None or not isinstance(name, str) or not name.strip():
                raise ValueError("name must be a non-empty string")
        elif tool_name == "kanban.checklists.delete" or tool_name == "kanban.checklists.items.list":
            _ensure_positive_int(arguments.get("checklist_id"), "checklist_id")
        elif tool_name == "kanban.checklists.items.create":
            _ensure_positive_int(arguments.get("checklist_id"), "checklist_id")
            name = arguments.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("name must be a non-empty string")
        elif tool_name == "kanban.checklists.items.update":
            _ensure_positive_int(arguments.get("item_id"), "item_id")
            name = arguments.get("name")
            if name is not None and (not isinstance(name, str) or not name.strip()):
                raise ValueError("name must be a non-empty string")
            if name is None and arguments.get("checked") is None:
                raise ValueError("must provide name or checked")
        elif tool_name == "kanban.checklists.items.delete":
            _ensure_positive_int(arguments.get("item_id"), "item_id")
        elif tool_name in {"kanban.workflow.policy.get", "kanban.workflow.statuses.list", "kanban.workflow.transitions.list"}:
            _ensure_positive_int(arguments.get("board_id"), "board_id")
        elif tool_name == "kanban.workflow.policy.upsert":
            _ensure_positive_int(arguments.get("board_id"), "board_id")
            lease_ttl = arguments.get("default_lease_ttl_sec")
            if lease_ttl is not None:
                _ensure_positive_int(lease_ttl, "default_lease_ttl_sec")
        elif tool_name == "kanban.workflow.task.state.get":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
        elif tool_name == "kanban.workflow.task.state.patch":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            _ensure_positive_int(arguments.get("expected_version"), "expected_version")
            idem = arguments.get("idempotency_key")
            if not isinstance(idem, str) or not idem.strip():
                raise ValueError("idempotency_key must be a non-empty string")
        elif tool_name == "kanban.workflow.task.claim":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            owner = arguments.get("owner")
            if not isinstance(owner, str) or not owner.strip():
                raise ValueError("owner must be a non-empty string")
            idem = arguments.get("idempotency_key")
            if not isinstance(idem, str) or not idem.strip():
                raise ValueError("idempotency_key must be a non-empty string")
            ttl = arguments.get("lease_ttl_sec")
            if ttl is not None:
                _ensure_positive_int(ttl, "lease_ttl_sec")
        elif tool_name == "kanban.workflow.task.release":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            owner = arguments.get("owner")
            if not isinstance(owner, str) or not owner.strip():
                raise ValueError("owner must be a non-empty string")
            idem = arguments.get("idempotency_key")
            if not isinstance(idem, str) or not idem.strip():
                raise ValueError("idempotency_key must be a non-empty string")
        elif tool_name == "kanban.workflow.task.transition":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            _ensure_positive_int(arguments.get("expected_version"), "expected_version")
            to_status_key = arguments.get("to_status_key")
            actor = arguments.get("actor")
            idem = arguments.get("idempotency_key")
            corr = arguments.get("correlation_id")
            if not isinstance(to_status_key, str) or not to_status_key.strip():
                raise ValueError("to_status_key must be a non-empty string")
            if not isinstance(actor, str) or not actor.strip():
                raise ValueError("actor must be a non-empty string")
            if not isinstance(idem, str) or not idem.strip():
                raise ValueError("idempotency_key must be a non-empty string")
            if not isinstance(corr, str) or not corr.strip():
                raise ValueError("correlation_id must be a non-empty string")
        elif tool_name == "kanban.workflow.task.approval.decide":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            _ensure_positive_int(arguments.get("expected_version"), "expected_version")
            reviewer = arguments.get("reviewer")
            decision = arguments.get("decision")
            idem = arguments.get("idempotency_key")
            corr = arguments.get("correlation_id")
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError("reviewer must be a non-empty string")
            if decision not in {"approved", "rejected"}:
                raise ValueError("decision must be 'approved' or 'rejected'")
            if not isinstance(idem, str) or not idem.strip():
                raise ValueError("idempotency_key must be a non-empty string")
            if not isinstance(corr, str) or not corr.strip():
                raise ValueError("correlation_id must be a non-empty string")
        elif tool_name == "kanban.workflow.task.events.list":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            limit = int(arguments.get("limit", 100))
            offset = int(arguments.get("offset", 0))
            if limit < 1 or limit > 500:
                raise ValueError("limit must be 1..500")
            if offset < 0:
                raise ValueError("offset must be >= 0")
        elif tool_name in {"kanban.workflow.control.pause", "kanban.workflow.control.resume", "kanban.workflow.control.drain"}:
            _ensure_positive_int(arguments.get("board_id"), "board_id")
        elif tool_name == "kanban.workflow.recovery.list_stale_claims":
            board_id = arguments.get("board_id")
            if board_id is not None:
                _ensure_positive_int(board_id, "board_id")
            limit = int(arguments.get("limit", 100))
            if limit < 1 or limit > 500:
                raise ValueError("limit must be 1..500")
        elif tool_name == "kanban.workflow.recovery.force_reassign":
            _ensure_positive_int(arguments.get("card_id"), "card_id")
            new_owner = arguments.get("new_owner")
            idem = arguments.get("idempotency_key")
            corr = arguments.get("correlation_id")
            if not isinstance(new_owner, str) or not new_owner.strip():
                raise ValueError("new_owner must be a non-empty string")
            if not isinstance(idem, str) or not idem.strip():
                raise ValueError("idempotency_key must be a non-empty string")
            if not isinstance(corr, str) or not corr.strip():
                raise ValueError("correlation_id must be a non-empty string")

    def _open_db(self, context: Any) -> KanbanDB:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for Kanban access")
        kanban_path = context.db_paths.get("kanban")
        if not kanban_path:
            raise ValueError("Kanban DB path not available in context")
        user_id = getattr(context, "user_id", None)
        if user_id is None:
            raise ValueError("Missing user_id for Kanban access")
        return KanbanDB(db_path=kanban_path, user_id=str(user_id))

    async def _list_boards(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        include_archived = bool(args.get("include_archived", False))
        include_deleted = bool(args.get("include_deleted", False))
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        return await asyncio.to_thread(
            self._list_boards_sync,
            context,
            include_archived,
            include_deleted,
            limit,
            offset,
        )

    def _list_boards_sync(
        self,
        context: Any | None,
        include_archived: bool,
        include_deleted: bool,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        boards, total = db.list_boards(
            include_archived=include_archived,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return {
            "boards": boards,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _get_board(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        include_deleted = bool(args.get("include_deleted", False))
        return await asyncio.to_thread(self._get_board_sync, context, board_id, include_deleted)

    def _get_board_sync(
        self,
        context: Any | None,
        board_id: int,
        include_deleted: bool,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        board = db.get_board(board_id, include_deleted=include_deleted)
        if not board:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)
        return {"board": board}

    async def _create_board(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        name = str(args.get("name")).strip()
        client_id = str(args.get("client_id") or uuid.uuid4())
        description = args.get("description")
        activity_retention_days = args.get("activity_retention_days")
        metadata = args.get("metadata")
        return await asyncio.to_thread(
            self._create_board_sync,
            context,
            name,
            client_id,
            description,
            activity_retention_days,
            metadata,
        )

    def _create_board_sync(
        self,
        context: Any | None,
        name: str,
        client_id: str,
        description: Optional[str],
        activity_retention_days: Optional[int],
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        board = db.create_board(
            name=name,
            client_id=client_id,
            description=description,
            activity_retention_days=activity_retention_days,
            metadata=metadata,
        )
        return {"board": board}

    async def _list_lists(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        include_archived = bool(args.get("include_archived", False))
        include_deleted = bool(args.get("include_deleted", False))
        return await asyncio.to_thread(
            self._list_lists_sync,
            context,
            board_id,
            include_archived,
            include_deleted,
        )

    def _list_lists_sync(
        self,
        context: Any | None,
        board_id: int,
        include_archived: bool,
        include_deleted: bool,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        lists = db.list_lists(
            board_id,
            include_archived=include_archived,
            include_deleted=include_deleted,
        )
        return {"board_id": board_id, "lists": lists}

    async def _create_list(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        name = str(args.get("name")).strip()
        client_id = str(args.get("client_id") or uuid.uuid4())
        position = args.get("position")
        return await asyncio.to_thread(
            self._create_list_sync,
            context,
            board_id,
            name,
            client_id,
            position,
        )

    def _create_list_sync(
        self,
        context: Any | None,
        board_id: int,
        name: str,
        client_id: str,
        position: Optional[int],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        lst = db.create_list(
            board_id=board_id,
            name=name,
            client_id=client_id,
            position=position,
        )
        return {"list": lst}

    async def _list_cards(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        list_id = _ensure_positive_int(args.get("list_id"), "list_id")
        include_archived = bool(args.get("include_archived", False))
        include_deleted = bool(args.get("include_deleted", False))
        return await asyncio.to_thread(
            self._list_cards_sync,
            context,
            list_id,
            include_archived,
            include_deleted,
        )

    def _list_cards_sync(
        self,
        context: Any | None,
        list_id: int,
        include_archived: bool,
        include_deleted: bool,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        cards = db.list_cards(
            list_id,
            include_archived=include_archived,
            include_deleted=include_deleted,
        )
        return {"list_id": list_id, "cards": cards}

    async def _create_card(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        list_id = _ensure_positive_int(args.get("list_id"), "list_id")
        title = str(args.get("title")).strip()
        client_id = str(args.get("client_id") or uuid.uuid4())
        description = args.get("description")
        position = args.get("position")
        due_date = args.get("due_date")
        start_date = args.get("start_date")
        priority = args.get("priority")
        metadata = args.get("metadata")
        return await asyncio.to_thread(
            self._create_card_sync,
            context,
            list_id,
            title,
            client_id,
            description,
            position,
            due_date,
            start_date,
            priority,
            metadata,
        )

    def _create_card_sync(
        self,
        context: Any | None,
        list_id: int,
        title: str,
        client_id: str,
        description: Optional[str],
        position: Optional[int],
        due_date: Optional[str],
        start_date: Optional[str],
        priority: Optional[str],
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        card = db.create_card(
            list_id=list_id,
            title=title,
            client_id=client_id,
            description=description,
            position=position,
            due_date=due_date,
            start_date=start_date,
            priority=priority,
            metadata=metadata,
        )
        return {"card": card}

    async def _search_cards(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        query = str(args.get("query")).strip()
        board_id = args.get("board_id")
        label_ids = args.get("label_ids")
        priority = args.get("priority")
        include_archived = bool(args.get("include_archived", False))
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        return await asyncio.to_thread(
            self._search_cards_sync,
            context,
            query,
            board_id,
            label_ids,
            priority,
            include_archived,
            limit,
            offset,
        )

    def _search_cards_sync(
        self,
        context: Any | None,
        query: str,
        board_id: Optional[int],
        label_ids: Optional[list[int]],
        priority: Optional[str],
        include_archived: bool,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        board_id_val = _ensure_positive_int(board_id, "board_id") if board_id is not None else None
        results, total = db.search_cards(
            query=query,
            board_id=board_id_val,
            label_ids=label_ids,
            priority=priority,
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        )
        return {
            "cards": results,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _list_labels(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._list_labels_sync, context, board_id)

    def _list_labels_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        labels = db.list_labels(board_id)
        return {"board_id": board_id, "labels": labels}

    async def _create_label(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        name = str(args.get("name")).strip()
        color = args.get("color")
        return await asyncio.to_thread(self._create_label_sync, context, board_id, name, color)

    def _create_label_sync(
        self,
        context: Any | None,
        board_id: int,
        name: str,
        color: Optional[str],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        label = db.create_label(board_id=board_id, name=name, color=color)
        return {"label": label}

    async def _update_label(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        label_id = _ensure_positive_int(args.get("label_id"), "label_id")
        name = args.get("name")
        color = args.get("color")
        return await asyncio.to_thread(self._update_label_sync, context, label_id, name, color)

    def _update_label_sync(
        self,
        context: Any | None,
        label_id: int,
        name: Optional[str],
        color: Optional[str],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        label = db.update_label(label_id, name=name, color=color)
        return {"label": label}

    async def _delete_label(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        label_id = _ensure_positive_int(args.get("label_id"), "label_id")
        return await asyncio.to_thread(self._delete_label_sync, context, label_id)

    def _delete_label_sync(self, context: Any | None, label_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        deleted = db.delete_label(label_id)
        return {"deleted": bool(deleted), "label_id": label_id}

    async def _assign_label(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        label_id = _ensure_positive_int(args.get("label_id"), "label_id")
        return await asyncio.to_thread(self._assign_label_sync, context, card_id, label_id)

    def _assign_label_sync(
        self,
        context: Any | None,
        card_id: int,
        label_id: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        assigned = db.assign_label_to_card(card_id, label_id)
        labels = db.get_card_labels(card_id)
        return {"assigned": bool(assigned), "card_id": card_id, "labels": labels}

    async def _remove_label(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        label_id = _ensure_positive_int(args.get("label_id"), "label_id")
        return await asyncio.to_thread(self._remove_label_sync, context, card_id, label_id)

    def _remove_label_sync(
        self,
        context: Any | None,
        card_id: int,
        label_id: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        removed = db.remove_label_from_card(card_id, label_id)
        labels = db.get_card_labels(card_id)
        return {"removed": bool(removed), "card_id": card_id, "labels": labels}

    async def _list_card_labels(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        return await asyncio.to_thread(self._list_card_labels_sync, context, card_id)

    def _list_card_labels_sync(self, context: Any | None, card_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        labels = db.get_card_labels(card_id)
        return {"card_id": card_id, "labels": labels}

    async def _move_card(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        target_list_id = _ensure_positive_int(args.get("target_list_id"), "target_list_id")
        position = args.get("position")
        return await asyncio.to_thread(self._move_card_sync, context, card_id, target_list_id, position)

    def _move_card_sync(
        self,
        context: Any | None,
        card_id: int,
        target_list_id: int,
        position: Optional[int],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        card = db.move_card(card_id=card_id, target_list_id=target_list_id, position=position)
        return {"card": card}

    async def _list_comments(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        include_deleted = bool(args.get("include_deleted", False))
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        return await asyncio.to_thread(
            self._list_comments_sync,
            context,
            card_id,
            include_deleted,
            limit,
            offset,
        )

    def _list_comments_sync(
        self,
        context: Any | None,
        card_id: int,
        include_deleted: bool,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        comments, total = db.list_comments(
            card_id,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
        return {
            "card_id": card_id,
            "comments": comments,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def _create_comment(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        content = str(args.get("content")).strip()
        return await asyncio.to_thread(self._create_comment_sync, context, card_id, content)

    def _create_comment_sync(self, context: Any | None, card_id: int, content: str) -> dict[str, Any]:
        db = self._open_db(context)
        comment = db.create_comment(card_id=card_id, content=content)
        return {"comment": comment}

    async def _update_comment(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        comment_id = _ensure_positive_int(args.get("comment_id"), "comment_id")
        content = str(args.get("content")).strip()
        return await asyncio.to_thread(self._update_comment_sync, context, comment_id, content)

    def _update_comment_sync(self, context: Any | None, comment_id: int, content: str) -> dict[str, Any]:
        db = self._open_db(context)
        comment = db.update_comment(comment_id=comment_id, content=content)
        return {"comment": comment}

    async def _delete_comment(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        comment_id = _ensure_positive_int(args.get("comment_id"), "comment_id")
        hard_delete = bool(args.get("hard_delete", False))
        return await asyncio.to_thread(self._delete_comment_sync, context, comment_id, hard_delete)

    def _delete_comment_sync(
        self,
        context: Any | None,
        comment_id: int,
        hard_delete: bool,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        deleted = db.delete_comment(comment_id=comment_id, hard_delete=hard_delete)
        return {"deleted": bool(deleted), "comment_id": comment_id}

    async def _list_checklists(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        return await asyncio.to_thread(self._list_checklists_sync, context, card_id)

    def _list_checklists_sync(self, context: Any | None, card_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        checklists = db.list_checklists(card_id)
        return {"card_id": card_id, "checklists": checklists}

    async def _create_checklist(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        name = str(args.get("name")).strip()
        position = args.get("position")
        return await asyncio.to_thread(self._create_checklist_sync, context, card_id, name, position)

    def _create_checklist_sync(
        self,
        context: Any | None,
        card_id: int,
        name: str,
        position: Optional[int],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        checklist = db.create_checklist(card_id=card_id, name=name, position=position)
        return {"checklist": checklist}

    async def _update_checklist(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        checklist_id = _ensure_positive_int(args.get("checklist_id"), "checklist_id")
        name = str(args.get("name")).strip()
        return await asyncio.to_thread(self._update_checklist_sync, context, checklist_id, name)

    def _update_checklist_sync(
        self,
        context: Any | None,
        checklist_id: int,
        name: str,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        checklist = db.update_checklist(checklist_id=checklist_id, name=name)
        return {"checklist": checklist}

    async def _delete_checklist(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        checklist_id = _ensure_positive_int(args.get("checklist_id"), "checklist_id")
        return await asyncio.to_thread(self._delete_checklist_sync, context, checklist_id)

    def _delete_checklist_sync(self, context: Any | None, checklist_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        deleted = db.delete_checklist(checklist_id)
        return {"deleted": bool(deleted), "checklist_id": checklist_id}

    async def _list_checklist_items(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        checklist_id = _ensure_positive_int(args.get("checklist_id"), "checklist_id")
        return await asyncio.to_thread(self._list_checklist_items_sync, context, checklist_id)

    def _list_checklist_items_sync(self, context: Any | None, checklist_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        items = db.list_checklist_items(checklist_id)
        return {"checklist_id": checklist_id, "items": items}

    async def _create_checklist_item(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        checklist_id = _ensure_positive_int(args.get("checklist_id"), "checklist_id")
        name = str(args.get("name")).strip()
        position = args.get("position")
        checked = args.get("checked")
        return await asyncio.to_thread(
            self._create_checklist_item_sync,
            context,
            checklist_id,
            name,
            position,
            checked,
        )

    def _create_checklist_item_sync(
        self,
        context: Any | None,
        checklist_id: int,
        name: str,
        position: Optional[int],
        checked: Optional[bool],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        item = db.create_checklist_item(
            checklist_id=checklist_id,
            name=name,
            position=position,
            checked=bool(checked) if checked is not None else False,
        )
        return {"item": item}

    async def _update_checklist_item(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        item_id = _ensure_positive_int(args.get("item_id"), "item_id")
        name = args.get("name")
        checked = args.get("checked")
        return await asyncio.to_thread(self._update_checklist_item_sync, context, item_id, name, checked)

    def _update_checklist_item_sync(
        self,
        context: Any | None,
        item_id: int,
        name: Optional[str],
        checked: Optional[bool],
    ) -> dict[str, Any]:
        db = self._open_db(context)
        item = db.update_checklist_item(item_id=item_id, name=name, checked=checked)
        return {"item": item}

    async def _delete_checklist_item(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        item_id = _ensure_positive_int(args.get("item_id"), "item_id")
        return await asyncio.to_thread(self._delete_checklist_item_sync, context, item_id)

    def _delete_checklist_item_sync(self, context: Any | None, item_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        deleted = db.delete_checklist_item(item_id)
        return {"deleted": bool(deleted), "item_id": item_id}

    def _is_admin(self, context: Any | None) -> bool:
        try:
            if bool(getattr(context, "is_admin", False)):
                return True
            roles = (getattr(context, "metadata", {}) or {}).get("roles")
            return isinstance(roles, list) and any(str(r).lower() == "admin" for r in roles)
        except (AttributeError, TypeError, ValueError):
            return False

    def _require_admin(self, context: Any | None) -> None:
        if not self._is_admin(context):
            raise ValueError("forbidden: admin privileges required")

    async def _workflow_policy_get(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_policy_get_sync, context, board_id)

    def _workflow_policy_get_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        policy = db.get_workflow_policy(board_id)
        if not policy:
            raise NotFoundError("Workflow policy not found", entity="workflow_policy", entity_id=board_id)
        return {"policy": policy}

    async def _workflow_policy_upsert(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_policy_upsert_sync, context, board_id, args)

    def _workflow_policy_upsert_sync(self, context: Any | None, board_id: int, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        upsert_kwargs: dict[str, Any] = {
            "board_id": board_id,
            "statuses": args.get("statuses"),
            "transitions": args.get("transitions"),
            "is_paused": _parse_bool_like(args.get("is_paused", False), "is_paused"),
            "is_draining": _parse_bool_like(args.get("is_draining", False), "is_draining"),
            "default_lease_ttl_sec": int(args.get("default_lease_ttl_sec", 900)),
            "strict_projection": _parse_bool_like(args.get("strict_projection", True), "strict_projection"),
        }
        if "metadata" in args:
            upsert_kwargs["metadata"] = args.get("metadata")

        policy = db.upsert_workflow_policy(
            **upsert_kwargs,
        )
        return {"policy": policy}

    async def _workflow_statuses_list(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_statuses_list_sync, context, board_id)

    def _workflow_statuses_list_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        return {"board_id": board_id, "statuses": db.list_workflow_statuses(board_id)}

    async def _workflow_transitions_list(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_transitions_list_sync, context, board_id)

    def _workflow_transitions_list_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        return {"board_id": board_id, "transitions": db.list_workflow_transitions(board_id)}

    async def _workflow_task_state_get(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        return await asyncio.to_thread(self._workflow_task_state_get_sync, context, card_id)

    def _workflow_task_state_get_sync(self, context: Any | None, card_id: int) -> dict[str, Any]:
        db = self._open_db(context)
        return {"state": db.get_card_workflow_state(card_id)}

    async def _workflow_task_state_patch(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._workflow_task_state_patch_sync, context, args)

    def _workflow_task_state_patch_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        state = db.patch_card_workflow_state(
            card_id=_ensure_positive_int(args.get("card_id"), "card_id"),
            workflow_status_key=args.get("workflow_status_key"),
            expected_version=_ensure_positive_int(args.get("expected_version"), "expected_version"),
            lease_owner=args.get("lease_owner"),
            idempotency_key=str(args.get("idempotency_key")).strip(),
            correlation_id=args.get("correlation_id"),
            last_actor=args.get("last_actor"),
        )
        return {"state": state}

    async def _workflow_task_claim(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._workflow_task_claim_sync, context, args)

    def _workflow_task_claim_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        state = db.claim_card_workflow(
            card_id=_ensure_positive_int(args.get("card_id"), "card_id"),
            owner=str(args.get("owner")).strip(),
            lease_ttl_sec=args.get("lease_ttl_sec"),
            idempotency_key=str(args.get("idempotency_key")).strip(),
            correlation_id=args.get("correlation_id"),
        )
        return {"state": state}

    async def _workflow_task_release(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._workflow_task_release_sync, context, args)

    def _workflow_task_release_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        state = db.release_card_workflow(
            card_id=_ensure_positive_int(args.get("card_id"), "card_id"),
            owner=str(args.get("owner")).strip(),
            idempotency_key=str(args.get("idempotency_key")).strip(),
            correlation_id=args.get("correlation_id"),
        )
        return {"state": state}

    async def _workflow_task_transition(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._workflow_task_transition_sync, context, args)

    def _workflow_task_transition_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        state = db.transition_card_workflow(
            card_id=_ensure_positive_int(args.get("card_id"), "card_id"),
            to_status_key=str(args.get("to_status_key")).strip(),
            actor=str(args.get("actor")).strip(),
            expected_version=_ensure_positive_int(args.get("expected_version"), "expected_version"),
            idempotency_key=str(args.get("idempotency_key")).strip(),
            correlation_id=str(args.get("correlation_id")).strip(),
            reason=args.get("reason"),
        )
        return {"state": state}

    async def _workflow_task_approval_decide(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._workflow_task_approval_decide_sync, context, args)

    def _workflow_task_approval_decide_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        db = self._open_db(context)
        state = db.decide_card_workflow_approval(
            card_id=_ensure_positive_int(args.get("card_id"), "card_id"),
            reviewer=str(args.get("reviewer")).strip(),
            decision=str(args.get("decision")).strip(),
            expected_version=_ensure_positive_int(args.get("expected_version"), "expected_version"),
            idempotency_key=str(args.get("idempotency_key")).strip(),
            correlation_id=str(args.get("correlation_id")).strip(),
            reason=args.get("reason"),
        )
        return {"state": state}

    async def _workflow_task_events_list(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        card_id = _ensure_positive_int(args.get("card_id"), "card_id")
        limit = int(args.get("limit", 100))
        offset = int(args.get("offset", 0))
        return await asyncio.to_thread(self._workflow_task_events_list_sync, context, card_id, limit, offset)

    def _workflow_task_events_list_sync(
        self,
        context: Any | None,
        card_id: int,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        events = db.list_card_workflow_events(card_id=card_id, limit=limit, offset=offset)
        return {"card_id": card_id, "events": events, "limit": limit, "offset": offset}

    async def _workflow_control_pause(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_control_pause_sync, context, board_id)

    def _workflow_control_pause_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        self._require_admin(context)
        db = self._open_db(context)
        updated = db.update_workflow_policy_flags(board_id=board_id, is_paused=True)
        return {"success": True, "policy": updated}

    async def _workflow_control_resume(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_control_resume_sync, context, board_id)

    def _workflow_control_resume_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        self._require_admin(context)
        db = self._open_db(context)
        updated = db.update_workflow_policy_flags(board_id=board_id, is_paused=False)
        return {"success": True, "policy": updated}

    async def _workflow_control_drain(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = _ensure_positive_int(args.get("board_id"), "board_id")
        return await asyncio.to_thread(self._workflow_control_drain_sync, context, board_id)

    def _workflow_control_drain_sync(self, context: Any | None, board_id: int) -> dict[str, Any]:
        self._require_admin(context)
        db = self._open_db(context)
        updated = db.update_workflow_policy_flags(board_id=board_id, is_draining=True)
        return {"success": True, "policy": updated}

    async def _workflow_recovery_list_stale_claims(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        board_id = args.get("board_id")
        board_id_val = _ensure_positive_int(board_id, "board_id") if board_id is not None else None
        limit = int(args.get("limit", 100))
        return await asyncio.to_thread(
            self._workflow_recovery_list_stale_claims_sync,
            context,
            board_id_val,
            limit,
        )

    def _workflow_recovery_list_stale_claims_sync(
        self,
        context: Any | None,
        board_id: int | None,
        limit: int,
    ) -> dict[str, Any]:
        db = self._open_db(context)
        return {"stale_claims": db.list_stale_workflow_claims(board_id=board_id, limit=limit)}

    async def _workflow_recovery_force_reassign(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        return await asyncio.to_thread(self._workflow_recovery_force_reassign_sync, context, args)

    def _workflow_recovery_force_reassign_sync(self, context: Any | None, args: dict[str, Any]) -> dict[str, Any]:
        self._require_admin(context)
        db = self._open_db(context)
        state = db.force_reassign_workflow_claim(
            card_id=_ensure_positive_int(args.get("card_id"), "card_id"),
            new_owner=str(args.get("new_owner")).strip(),
            idempotency_key=str(args.get("idempotency_key")).strip(),
            correlation_id=str(args.get("correlation_id")).strip(),
            reason=args.get("reason"),
        )
        return {"state": state}
