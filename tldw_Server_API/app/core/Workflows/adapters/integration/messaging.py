"""Messaging and app integration adapters.

This module includes adapters for application-specific integrations:
- kanban: Manage Kanban boards
- chatbooks: Manage chatbooks (export/import)
- character_chat: Chat with AI characters
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)
from tldw_Server_API.app.core.exceptions import AdapterError
from tldw_Server_API.app.core.testing import is_test_mode, is_truthy
from tldw_Server_API.app.core.Workflows.adapters._common import resolve_context_user_id
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import (
    CharacterChatConfig,
    ChatbooksConfig,
    KanbanConfig,
)


_SAFE_ADAPTER_ERROR_PREFIXES = ("invalid_", "missing_")


def _safe_adapter_error_message(exc: AdapterError) -> str:
    message = str(exc).strip()
    if message.startswith(_SAFE_ADAPTER_ERROR_PREFIXES) and all(ch.isalnum() or ch == "_" for ch in message):
        return message
    return "adapter_error"


def _safe_kanban_error_detail(exc: Exception) -> str:
    if isinstance(exc, InputError):
        return "Invalid kanban request"
    if isinstance(exc, NotFoundError):
        return "Kanban resource not found"
    if isinstance(exc, ConflictError):
        return "Kanban conflict"
    return "Kanban operation failed"


@registry.register(
    "kanban",
    category="integration",
    description="Manage Kanban boards",
    parallelizable=True,
    tags=["integration", "kanban"],
    config_model=KanbanConfig,
)
async def run_kanban_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Read/write Kanban boards, lists, and cards for workflow steps.

    Config:
      - action: str (required) e.g. board.list, board.create, list.create, card.update
      - entity ids: board_id, list_id, card_id
      - include flags: include_archived, include_deleted, include_details
      - create/update fields: name, description, title, metadata, activity_retention_days
      - card fields: due_date, start_date, due_complete, priority, position
      - move/copy: target_list_id, new_client_id, new_title, copy_checklists, copy_labels
      - search/filter: query, label_ids, priority, limit, offset
    Output: action-specific payload (board/list/card/etc.)
    """
    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    user_id = resolve_context_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return is_truthy(value)
        return bool(value)

    def _coerce_optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            raw = value.strip().lower()
            if is_truthy(raw):
                return True
            if raw in {"0", "false", "no", "off", "n"}:
                return False
        return bool(value)

    def _coerce_int(value: Any, field: str, allow_none: bool = False) -> int | None:
        if value is None or value == "":
            if allow_none:
                return None
            raise AdapterError(f"missing_{field}")
        if isinstance(value, bool):
            raise AdapterError(f"invalid_{field}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"invalid_{field}") from exc

    def _coerce_int_list(value: Any) -> list[int]:
        if value is None:
            return []
        raw_value = _render(value) if isinstance(value, str) else value
        items: list[Any] = []
        if isinstance(raw_value, list):
            items = raw_value
        elif isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError:
                parsed = None
            items = parsed if isinstance(parsed, list) else [s.strip() for s in raw_value.split(",") if s.strip()]
        else:
            items = [raw_value]
        out: list[int] = []
        for item in items:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out

    def _coerce_limit(value: Any, default: int) -> int:
        try:
            parsed = int(value) if value is not None else default
        except (TypeError, ValueError):
            parsed = default
        return max(1, parsed)

    def _coerce_date_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        try:
            rendered = str(value)
        except (TypeError, ValueError):
            return None
        return rendered.strip() or None

    try:
        user_id_int: Any = int(user_id) if str(user_id).isdigit() else user_id
    except (OverflowError, TypeError, ValueError):
        user_id_int = user_id

    db_path = DatabasePaths.get_kanban_db_path(user_id_int)
    db = KanbanDB(db_path=str(db_path), user_id=user_id)

    try:
        if action == "board.list":
            include_archived = _coerce_bool(config.get("include_archived"), False)
            include_deleted = _coerce_bool(config.get("include_deleted"), False)
            limit = _coerce_limit(_render(config.get("limit")), 50)
            offset = max(0, _coerce_int(_render(config.get("offset", 0)), "offset", allow_none=True) or 0)
            boards, total = db.list_boards(
                include_archived=include_archived,
                include_deleted=include_deleted,
                limit=limit,
                offset=offset,
            )
            return {"boards": boards, "total": total, "limit": limit, "offset": offset}

        if action == "board.get":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            include_children = _coerce_bool(config.get("include_children"), True)
            include_archived = _coerce_bool(config.get("include_archived"), False)
            include_deleted = _coerce_bool(config.get("include_deleted"), False)
            if include_children:
                board = db.get_board_with_lists_and_cards(board_id, include_archived=include_archived)
            else:
                board = db.get_board(board_id, include_deleted=include_deleted)
            if not board:
                return {"error": "not_found", "entity": "board", "entity_id": board_id}
            return {"board": board}

        if action == "board.create":
            name = str(_render(config.get("name") or "")).strip()
            if not name:
                return {"error": "missing_name"}
            client_id = str(_render(config.get("client_id") or "")).strip()
            if not client_id:
                import uuid as _uuid
                client_id = f"wf_{_uuid.uuid4().hex}"
            description = _render(config.get("description"))
            activity_retention_days = _coerce_int(
                _render(config.get("activity_retention_days")),
                "activity_retention_days",
                allow_none=True,
            )
            metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else None
            board = db.create_board(
                name=name,
                client_id=client_id,
                description=str(description) if isinstance(description, str) else description,
                activity_retention_days=activity_retention_days,
                metadata=metadata,
            )
            return {"board": board}

        if action == "board.update":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            name = _render(config.get("name"))
            description = _render(config.get("description"))
            activity_retention_days = _coerce_int(
                _render(config.get("activity_retention_days")),
                "activity_retention_days",
                allow_none=True,
            )
            metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else None
            expected_version = _coerce_int(_render(config.get("expected_version")), "expected_version", allow_none=True)
            board = db.update_board(
                board_id=board_id,
                name=str(name) if isinstance(name, str) else name,
                description=str(description) if isinstance(description, str) else description,
                activity_retention_days=activity_retention_days,
                metadata=metadata,
                expected_version=expected_version,
            )
            return {"board": board}

        if action in {"board.archive", "board.unarchive"}:
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            archive_flag = action != "board.unarchive"
            if "archive" in config:
                archive_flag = _coerce_bool(config.get("archive"), archive_flag)
            board = db.archive_board(board_id, archive=archive_flag)
            return {"board": board}

        if action == "board.delete":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            hard_delete = _coerce_bool(config.get("hard_delete"), False)
            success = db.delete_board(board_id, hard_delete=hard_delete)
            return {"success": success}

        if action == "board.restore":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            board = db.restore_board(board_id)
            return {"board": board}

        if action == "list.list":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            include_archived = _coerce_bool(config.get("include_archived"), False)
            include_deleted = _coerce_bool(config.get("include_deleted"), False)
            lists = db.list_lists(
                board_id=board_id,
                include_archived=include_archived,
                include_deleted=include_deleted,
            )
            return {"lists": lists}

        if action == "list.get":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            include_deleted = _coerce_bool(config.get("include_deleted"), False)
            lst = db.get_list(list_id, include_deleted=include_deleted)
            if not lst:
                return {"error": "not_found", "entity": "list", "entity_id": list_id}
            return {"list": lst}

        if action == "list.create":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            name = str(_render(config.get("name") or "")).strip()
            if not name:
                return {"error": "missing_name"}
            client_id = str(_render(config.get("client_id") or "")).strip()
            if not client_id:
                import uuid as _uuid
                client_id = f"wf_{_uuid.uuid4().hex}"
            position = _coerce_int(_render(config.get("position")), "position", allow_none=True)
            lst = db.create_list(
                board_id=board_id,
                name=name,
                client_id=client_id,
                position=position,
            )
            return {"list": lst}

        if action == "list.update":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            name = _render(config.get("name"))
            expected_version = _coerce_int(_render(config.get("expected_version")), "expected_version", allow_none=True)
            lst = db.update_list(
                list_id=list_id,
                name=str(name) if isinstance(name, str) else name,
                expected_version=expected_version,
            )
            return {"list": lst}

        if action == "list.reorder":
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            list_ids = _coerce_int_list(config.get("list_ids") or config.get("ids"))
            if not list_ids:
                return {"error": "missing_list_ids"}
            db.reorder_lists(board_id=board_id, list_ids=list_ids)
            return {"success": True, "count": len(list_ids)}

        if action in {"list.archive", "list.unarchive"}:
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            archive_flag = action != "list.unarchive"
            if "archive" in config:
                archive_flag = _coerce_bool(config.get("archive"), archive_flag)
            lst = db.archive_list(list_id, archive=archive_flag)
            return {"list": lst}

        if action == "list.delete":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            hard_delete = _coerce_bool(config.get("hard_delete"), False)
            success = db.delete_list(list_id, hard_delete=hard_delete)
            return {"success": success}

        if action == "list.restore":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            lst = db.restore_list(list_id)
            return {"list": lst}

        if action == "card.list":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            include_archived = _coerce_bool(config.get("include_archived"), False)
            include_deleted = _coerce_bool(config.get("include_deleted"), False)
            cards = db.list_cards(
                list_id=list_id,
                include_archived=include_archived,
                include_deleted=include_deleted,
            )
            return {"cards": cards}

        if action == "card.get":
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            include_details = _coerce_bool(config.get("include_details"), True)
            include_deleted = _coerce_bool(config.get("include_deleted"), False)
            if include_details:
                card = db.get_card_with_details(card_id, include_deleted=include_deleted)
            else:
                card = db.get_card(card_id, include_deleted=include_deleted)
            if not card:
                return {"error": "not_found", "entity": "card", "entity_id": card_id}
            return {"card": card}

        if action == "card.create":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            title = str(_render(config.get("title") or "")).strip()
            if not title:
                return {"error": "missing_title"}
            client_id = str(_render(config.get("client_id") or "")).strip()
            if not client_id:
                import uuid as _uuid
                client_id = f"wf_{_uuid.uuid4().hex}"
            description = _render(config.get("description"))
            position = _coerce_int(_render(config.get("position")), "position", allow_none=True)
            due_date = _coerce_date_str(_render(config.get("due_date")))
            start_date = _coerce_date_str(_render(config.get("start_date")))
            priority = _render(config.get("priority"))
            metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else None
            card = db.create_card(
                list_id=list_id,
                title=title,
                client_id=client_id,
                description=str(description) if isinstance(description, str) else description,
                position=position,
                due_date=due_date,
                start_date=start_date,
                priority=str(priority) if isinstance(priority, str) and priority.strip() else None,
                metadata=metadata,
            )
            return {"card": card}

        if action == "card.update":
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            title = _render(config.get("title"))
            description = _render(config.get("description"))
            due_date = _coerce_date_str(_render(config.get("due_date")))
            due_complete = _coerce_optional_bool(config.get("due_complete"))
            start_date = _coerce_date_str(_render(config.get("start_date")))
            priority = _render(config.get("priority"))
            metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else None
            expected_version = _coerce_int(_render(config.get("expected_version")), "expected_version", allow_none=True)
            card = db.update_card(
                card_id=card_id,
                title=str(title) if isinstance(title, str) else title,
                description=str(description) if isinstance(description, str) else description,
                due_date=due_date,
                due_complete=due_complete,
                start_date=start_date,
                priority=str(priority) if isinstance(priority, str) and priority.strip() else None,
                metadata=metadata,
                expected_version=expected_version,
            )
            return {"card": card}

        if action == "card.reorder":
            list_id = _coerce_int(_render(config.get("list_id")), "list_id")
            card_ids = _coerce_int_list(config.get("card_ids") or config.get("ids"))
            if not card_ids:
                return {"error": "missing_card_ids"}
            db.reorder_cards(list_id=list_id, card_ids=card_ids)
            return {"success": True, "count": len(card_ids)}

        if action == "card.move":
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            target_list_id = _coerce_int(_render(config.get("target_list_id")), "target_list_id")
            position = _coerce_int(_render(config.get("position")), "position", allow_none=True)
            card = db.move_card(card_id=card_id, target_list_id=target_list_id, position=position)
            return {"card": card}

        if action == "card.copy":
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            target_list_id = _coerce_int(_render(config.get("target_list_id")), "target_list_id")
            new_client_id = str(_render(config.get("new_client_id") or "")).strip()
            if not new_client_id:
                import uuid as _uuid
                new_client_id = f"wf_{_uuid.uuid4().hex}"
            new_title = _render(config.get("new_title"))
            position = _coerce_int(_render(config.get("position")), "position", allow_none=True)
            copy_checklists = _coerce_bool(config.get("copy_checklists"), True)
            copy_labels = _coerce_bool(config.get("copy_labels"), True)
            card = db.copy_card_with_checklists(
                card_id=card_id,
                target_list_id=target_list_id,
                new_client_id=new_client_id,
                position=position,
                new_title=str(new_title) if isinstance(new_title, str) and new_title.strip() else None,
                copy_checklists=copy_checklists,
                copy_labels=copy_labels,
            )
            return {"card": card}

        if action in {"card.archive", "card.unarchive"}:
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            archive_flag = action != "card.unarchive"
            if "archive" in config:
                archive_flag = _coerce_bool(config.get("archive"), archive_flag)
            card = db.archive_card(card_id, archive=archive_flag)
            return {"card": card}

        if action == "card.delete":
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            hard_delete = _coerce_bool(config.get("hard_delete"), False)
            success = db.delete_card(card_id, hard_delete=hard_delete)
            return {"success": success}

        if action == "card.restore":
            card_id = _coerce_int(_render(config.get("card_id")), "card_id")
            card = db.restore_card(card_id)
            return {"card": card}

        if action == "card.search":
            query = str(_render(config.get("query") or "")).strip()
            if not query:
                return {"error": "missing_query"}
            board_id = _coerce_int(_render(config.get("board_id")), "board_id", allow_none=True)
            label_ids = _coerce_int_list(config.get("label_ids"))
            priority = _render(config.get("priority"))
            include_archived = _coerce_bool(config.get("include_archived"), False)
            limit = _coerce_limit(_render(config.get("limit")), 50)
            offset = max(0, _coerce_int(_render(config.get("offset", 0)), "offset", allow_none=True) or 0)
            cards, total = db.search_cards(
                query=query,
                board_id=board_id,
                label_ids=label_ids or None,
                priority=str(priority) if isinstance(priority, str) and priority.strip() else None,
                include_archived=include_archived,
                limit=limit,
                offset=offset,
            )
            return {"cards": cards, "total": total, "limit": limit, "offset": offset}

        if action in {"board.cards.filter", "cards.filter"}:
            board_id = _coerce_int(_render(config.get("board_id")), "board_id")
            filters = config.get("filters") if isinstance(config.get("filters"), dict) else {}

            def _f(key: str) -> Any:
                return config.get(key) if key in config else filters.get(key)

            label_ids = _coerce_int_list(_f("label_ids"))
            priority = _render(_f("priority"))
            due_before = _render(_f("due_before"))
            due_after = _render(_f("due_after"))
            overdue = _coerce_optional_bool(_f("overdue"))
            has_due_date = _coerce_optional_bool(_f("has_due_date"))
            has_checklist = _coerce_optional_bool(_f("has_checklist"))
            is_complete = _coerce_optional_bool(_f("is_complete"))
            include_archived = _coerce_bool(_f("include_archived"), False)
            include_deleted = _coerce_bool(_f("include_deleted"), False)
            limit = _coerce_limit(_render(_f("limit") or _f("per_page")), 50)
            offset = _coerce_int(_render(_f("offset")), "offset", allow_none=True)
            if offset is None:
                page = _coerce_int(_render(_f("page")), "page", allow_none=True)
                offset = (page - 1) * limit if page is not None and page > 0 else 0
            cards, total = db.get_board_cards_filtered(
                board_id=board_id,
                label_ids=label_ids or None,
                priority=str(priority) if isinstance(priority, str) and priority.strip() else None,
                due_before=str(due_before) if isinstance(due_before, str) and due_before.strip() else None,
                due_after=str(due_after) if isinstance(due_after, str) and due_after.strip() else None,
                overdue=overdue,
                has_due_date=has_due_date,
                has_checklist=has_checklist,
                is_complete=is_complete,
                include_archived=include_archived,
                include_deleted=include_deleted,
                limit=limit,
                offset=offset,
            )
            return {"cards": cards, "total": total, "limit": limit, "offset": offset}

        return {"error": f"unsupported_action:{action}"}

    except AdapterError as exc:
        return {"error": _safe_adapter_error_message(exc)}
    except (InputError, ConflictError, NotFoundError, KanbanDBError) as exc:
        return {
            "error": "kanban_error",
            "error_type": exc.__class__.__name__,
            "detail": _safe_kanban_error_detail(exc),
        }
    finally:
        with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
            db.close()


@registry.register(
    "chatbooks",
    category="integration",
    description="Manage chatbooks",
    parallelizable=False,
    tags=["integration", "chatbooks"],
    config_model=ChatbooksConfig,
)
async def run_chatbooks_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Export and import chatbooks within a workflow step.

    Config:
      - action: Literal["export", "import", "list_jobs", "get_job", "preview"]
      - content_types: Optional[List[str]] (for export: "conversations", "notes", "prompts", "media")
      - name: Optional[str] (templated, for export)
      - description: Optional[str] (templated, for export)
      - file_path: Optional[str] (for import)
      - job_id: Optional[str] (for get_job)
    Output:
      - {"job_id": str, "status": str, "artifact_uri": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = resolve_context_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Test mode simulation
    if is_test_mode():
        if action == "export":
            return {
                "job_id": "test-job-123",
                "status": "completed",
                "name": _render(config.get("name") or "Test Export"),
                "simulated": True,
            }
        if action == "import":
            return {"job_id": "test-import-123", "status": "completed", "imported": 0, "simulated": True}
        if action == "list_jobs":
            return {"jobs": [], "count": 0, "simulated": True}
        if action == "get_job":
            return {"job": {"id": config.get("job_id"), "status": "completed"}, "simulated": True}
        if action == "preview":
            return {"preview": {"conversations": 0, "notes": 0, "prompts": 0}, "simulated": True}
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

        # Initialize user's notes database for chatbook service
        try:
            user_id_int = int(user_id)
            notes_db_path = DatabasePaths.get_chacha_db_path(user_id_int)
        except (OverflowError, TypeError, ValueError):
            user_id_int = None
            notes_db_path = Path("Databases") / "user_databases" / user_id / "ChaChaNotes.db"

        db = CharactersRAGDB(db_path=notes_db_path, client_id="workflow_engine")
        service = ChatbookService(user_id=user_id, db=db, user_id_int=user_id_int)

        if action == "export":
            content_types = config.get("content_types") or ["conversations", "notes"]
            if isinstance(content_types, str):
                content_types = [c.strip() for c in content_types.split(",") if c.strip()]
            name = _render(config.get("name") or "Workflow Export")
            description = _render(config.get("description") or "")

            job_info = service.create_export_job(
                name=name,
                description=description,
                content_types=content_types,
            )
            return {
                "job_id": job_info.get("job_id"),
                "status": job_info.get("status", "pending"),
                "name": name,
                "content_types": content_types,
            }

        if action == "import":
            file_path = _render(config.get("file_path") or "")
            if not file_path:
                return {"error": "missing_file_path"}
            # Note: Import is async and job-based; we just start the job here
            return {"error": "import_not_yet_supported_in_workflows", "file_path": file_path}

        if action == "list_jobs":
            status_filter = config.get("status")
            limit = int(config.get("limit") or 100)
            jobs = service.list_export_jobs(status=status_filter, limit=limit)
            return {
                "jobs": [
                    {"id": j.job_id, "name": j.name, "status": j.status.value if hasattr(j.status, 'value') else str(j.status)}
                    for j in jobs
                ],
                "count": len(jobs),
            }

        if action == "get_job":
            job_id = config.get("job_id")
            if not job_id:
                return {"error": "missing_job_id"}
            job = service.get_export_job(str(job_id))
            if job is None:
                return {"error": "job_not_found", "job_id": job_id}
            return {
                "job": {
                    "id": job.job_id,
                    "name": job.name,
                    "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                }
            }

        if action == "preview":
            content_types = config.get("content_types") or ["conversations", "notes", "prompts"]
            if isinstance(content_types, str):
                content_types = [c.strip() for c in content_types.split(",") if c.strip()]
            preview = service.preview_export(content_types=content_types)
            return {"preview": preview}

        return {"error": f"unknown_action:{action}"}

    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        logger.exception("Chatbooks adapter error")
        return {"error": "chatbooks_error"}


@registry.register(
    "character_chat",
    category="integration",
    description="Character chat",
    parallelizable=True,
    tags=["integration", "chat"],
    config_model=CharacterChatConfig,
)
async def run_character_chat_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Chat with AI characters using character cards within a workflow step.

    Config:
      - action: Literal["start", "message", "load"] (default: "message")
      - character_id: Optional[int] - for start
      - conversation_id: Optional[str] - for message/load
      - message: Optional[str] (templated) - for message action
      - api_name: Optional[str] - LLM provider
      - temperature: float = 0.8
      - user_name: Optional[str] - user display name for placeholders
    Output:
      - {"response": str, "conversation_id": str, "character_name": str, "turn_count": int}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = resolve_context_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    action = str(config.get("action") or "message").strip().lower()

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    user_name = _render(config.get("user_name") or "User")

    # Test mode simulation
    if is_test_mode():
        if action == "start":
            return {
                "conversation_id": "test-conv-123",
                "character_name": "Test Character",
                "character_id": config.get("character_id"),
                "greeting": "Hello! How can I help you today?",
                "simulated": True,
            }
        if action == "message":
            return {
                "response": f"This is a simulated response to: {_render(config.get('message') or '')}",
                "conversation_id": config.get("conversation_id") or "test-conv-123",
                "character_name": "Test Character",
                "turn_count": 2,
                "simulated": True,
            }
        if action == "load":
            return {
                "conversation_id": config.get("conversation_id"),
                "character_name": "Test Character",
                "history": [],
                "turn_count": 0,
                "simulated": True,
            }
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Character_Chat.modules.character_chat import (
            load_chat_and_character,
            post_message_to_conversation,
            start_new_chat_session,
        )
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

        # Initialize user's character DB
        try:
            user_id_int = int(user_id)
            db_path = DatabasePaths.get_chacha_db_path(user_id_int)
        except (OverflowError, TypeError, ValueError):
            db_path = Path("Databases") / "user_databases" / user_id / "ChaChaNotes.db"

        db = CharactersRAGDB(db_path=db_path, client_id="workflow_engine")

        if action == "start":
            character_id = config.get("character_id")
            if character_id is None:
                return {"error": "missing_character_id"}

            custom_title = _render(config.get("title")) if config.get("title") else None

            conversation_id, char_data, initial_history, _ = start_new_chat_session(
                db=db,
                character_id=int(character_id),
                user_name=user_name,
                custom_title=custom_title,
            )

            if not conversation_id:
                return {"error": "failed_to_start_chat_session"}

            char_name = char_data.get("name", "Character") if char_data else "Character"
            greeting = ""
            if initial_history and initial_history[0]:
                greeting = initial_history[0][1] or ""

            return {
                "conversation_id": conversation_id,
                "character_name": char_name,
                "character_id": character_id,
                "greeting": greeting,
                "turn_count": 1 if greeting else 0,
            }

        if action == "load":
            conversation_id = config.get("conversation_id")
            if not conversation_id:
                return {"error": "missing_conversation_id"}

            char_data, history, _ = load_chat_and_character(
                db=db,
                conversation_id_str=str(conversation_id),
                user_name=user_name,
            )

            if char_data is None:
                return {"error": "conversation_not_found", "conversation_id": conversation_id}

            char_name = char_data.get("name", "Character") if char_data else "Unknown"

            # Format history for output
            formatted_history = []
            for user_msg, char_msg in history:
                if user_msg:
                    formatted_history.append({"role": "user", "content": user_msg})
                if char_msg:
                    formatted_history.append({"role": "character", "content": char_msg})

            return {
                "conversation_id": conversation_id,
                "character_name": char_name,
                "character_id": char_data.get("id") if char_data else None,
                "history": formatted_history,
                "turn_count": len(history),
            }

        if action == "message":
            conversation_id = config.get("conversation_id")
            if not conversation_id:
                return {"error": "missing_conversation_id"}

            message = _render(config.get("message") or "")
            if not message:
                return {"error": "missing_message"}

            api_name = _render(config.get("api_name") or config.get("api_provider") or "openai")
            temperature = float(config.get("temperature") or 0.8)

            # Post message and get response
            result = await post_message_to_conversation(
                db=db,
                conversation_id=str(conversation_id),
                user_message=message,
                user_name=user_name,
                api_name=api_name,
                temperature=temperature,
            )

            if result is None:
                return {"error": "failed_to_post_message"}

            response_text = result.get("response") or result.get("content") or ""
            char_name = result.get("character_name") or "Character"

            return {
                "response": response_text,
                "text": response_text,  # Alias for downstream steps
                "conversation_id": conversation_id,
                "character_name": char_name,
                "turn_count": result.get("turn_count", 0),
            }

        return {"error": f"unknown_action:{action}"}

    except (AttributeError, ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        logger.exception("Character chat adapter error")
        return {"error": "character_chat_error"}
