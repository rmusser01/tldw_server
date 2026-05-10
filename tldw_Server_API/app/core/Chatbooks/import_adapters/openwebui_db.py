"""OpenWebUI SQLite database import adapter."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .openwebui import (
    OpenWebUIConversationPlan,
    OpenWebUIMessagePlan,
    _build_title,
    _is_branched,
    _parse_message,
)


SQLITE_MAGIC = b"SQLite format 3\x00"
UNFILED_FOLDER_PATH = ["Unfiled"]

REQUIRED_SCHEMA: dict[str, set[str]] = {
    "user": {"id", "name", "email", "created_at", "updated_at"},
    "folder": {
        "id",
        "parent_id",
        "user_id",
        "name",
        "items",
        "meta",
        "is_expanded",
        "created_at",
        "updated_at",
    },
    "chat": {
        "id",
        "user_id",
        "title",
        "chat",
        "created_at",
        "updated_at",
        "share_id",
        "archived",
        "pinned",
        "meta",
        "folder_id",
    },
}
TABLE_INFO_QUERIES = {
    "user": "PRAGMA table_info(user)",
    "folder": "PRAGMA table_info(folder)",
    "chat": "PRAGMA table_info(chat)",
}


@dataclass(frozen=True)
class OpenWebUIDatabaseUserPreview:
    """Aggregate preview for one OpenWebUI source user."""

    source_user_id: str
    display_label: str
    email: str | None
    chat_count: int
    folder_count: int
    message_count: int
    branched_chat_count: int
    duplicate_chat_count: int
    archived_chat_count: int
    pinned_chat_count: int
    attachment_reference_count: int
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable user preview without chat/message content."""
        return {
            "source_user_id": self.source_user_id,
            "display_label": self.display_label,
            "email": self.email,
            "chat_count": self.chat_count,
            "folder_count": self.folder_count,
            "message_count": self.message_count,
            "branched_chat_count": self.branched_chat_count,
            "duplicate_chat_count": self.duplicate_chat_count,
            "archived_chat_count": self.archived_chat_count,
            "pinned_chat_count": self.pinned_chat_count,
            "attachment_reference_count": self.attachment_reference_count,
            "warning_count": self.warning_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OpenWebUIDatabasePreview:
    """OpenWebUI database preview grouped by source user."""

    users: list[OpenWebUIDatabaseUserPreview]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable database preview."""
        return {
            "user_count": len(self.users),
            "users": [user.to_dict() for user in self.users],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OpenWebUIDatabaseFolderPlan:
    """OpenWebUI source folder placement for a normalized conversation."""

    source_folder_id: str | None
    source_parent_id: str | None
    source_path: list[str]
    source_meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpenWebUIDatabaseExtractionResult:
    """Selected-user OpenWebUI database extraction result."""

    selected_user_id: str
    selected_user_label: str
    chats: list[OpenWebUIConversationPlan]
    folder_plans_by_external_ref: dict[str, OpenWebUIDatabaseFolderPlan]
    warnings: list[str] = field(default_factory=list)


def preview_openwebui_db(
    file_path: str | Path,
    duplicate_lookup: Callable[[str], bool] | None = None,
) -> OpenWebUIDatabasePreview:
    """Build a per-user preview for an uploaded OpenWebUI SQLite database."""
    duplicate_lookup = duplicate_lookup or (lambda _external_ref: False)
    with _open_validated_db(file_path) as conn:
        users = _load_users(conn)
        user_previews: list[OpenWebUIDatabaseUserPreview] = []
        warnings: list[str] = []
        for user in users:
            user_id = str(user["id"])
            chats = _load_chats_for_user(conn, user_id)
            folders = _load_folders_for_user(conn, user_id)
            parsed_chats: list[OpenWebUIConversationPlan] = []
            user_warnings: list[str] = []

            for chat_row in chats:
                chat_plan, chat_warnings = _conversation_plan_from_chat_row(chat_row)
                if chat_plan is None:
                    user_warnings.extend(chat_warnings)
                    continue
                parsed_chats.append(chat_plan)
                user_warnings.extend(chat_warnings)
                folder_plan = _folder_plan_for_chat(chat_row, folders)
                user_warnings.extend(folder_plan.warnings)

            duplicate_chat_count = sum(1 for chat in parsed_chats if duplicate_lookup(chat.external_ref))
            user_previews.append(
                OpenWebUIDatabaseUserPreview(
                    source_user_id=user_id,
                    display_label=_display_label_for_user(user),
                    email=_optional_str(user["email"]),
                    chat_count=len(parsed_chats),
                    folder_count=len(folders),
                    message_count=sum(len(chat.messages) for chat in parsed_chats),
                    branched_chat_count=sum(1 for chat in parsed_chats if chat.is_branched),
                    duplicate_chat_count=duplicate_chat_count,
                    archived_chat_count=sum(1 for row in chats if _sqlite_truthy(row["archived"])),
                    pinned_chat_count=sum(1 for row in chats if _sqlite_truthy(row["pinned"])),
                    attachment_reference_count=sum(
                        chat.attachment_reference_count for chat in parsed_chats
                    ),
                    warning_count=len(user_warnings),
                    warnings=user_warnings,
                )
            )
            warnings.extend(user_warnings)

        return OpenWebUIDatabasePreview(users=user_previews, warnings=warnings)


def extract_openwebui_db_user(
    file_path: str | Path,
    *,
    selected_user_id: str,
) -> OpenWebUIDatabaseExtractionResult:
    """Extract normalized conversations for one selected OpenWebUI source user."""
    if not selected_user_id or not str(selected_user_id).strip():
        raise ValueError("selected_user_id is required for OpenWebUI database imports")

    with _open_validated_db(file_path) as conn:
        user = _load_user(conn, selected_user_id)
        if user is None:
            raise ValueError("Selected OpenWebUI user was not found in the database")

        folders = _load_folders_for_user(conn, selected_user_id)
        chats: list[OpenWebUIConversationPlan] = []
        folder_plans_by_external_ref: dict[str, OpenWebUIDatabaseFolderPlan] = {}
        warnings: list[str] = []

        for chat_row in _load_chats_for_user(conn, selected_user_id):
            chat_plan, chat_warnings = _conversation_plan_from_chat_row(
                chat_row,
                source_user_id=selected_user_id,
            )
            warnings.extend(chat_warnings)
            if chat_plan is None:
                continue

            folder_plan = _folder_plan_for_chat(chat_row, folders)
            warnings.extend(folder_plan.warnings)
            folder_plans_by_external_ref[chat_plan.external_ref] = folder_plan
            chats.append(chat_plan)

        return OpenWebUIDatabaseExtractionResult(
            selected_user_id=selected_user_id,
            selected_user_label=_display_label_for_user(user),
            chats=chats,
            folder_plans_by_external_ref=folder_plans_by_external_ref,
            warnings=warnings,
        )


@contextmanager
def _open_validated_db(file_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(file_path)
    try:
        with path.open("rb") as handle:
            if handle.read(len(SQLITE_MAGIC)) != SQLITE_MAGIC:
                raise ValueError("Invalid OpenWebUI SQLite database")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("Unable to read OpenWebUI SQLite database") from exc

    resolved = path.resolve()
    uri = f"file:{quote(str(resolved), safe='/:')}?mode=ro"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.OperationalError):
            pass
        _validate_schema(conn)
        yield conn
    except ValueError:
        raise
    except sqlite3.Error as exc:
        raise ValueError("Invalid OpenWebUI SQLite database") from exc
    finally:
        if conn is not None:
            conn.close()


def _validate_schema(conn: sqlite3.Connection) -> None:
    table_rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
        tuple(REQUIRED_SCHEMA.keys()),
    ).fetchall()
    tables = {str(row["name"]) for row in table_rows}
    missing_tables = sorted(set(REQUIRED_SCHEMA) - tables)
    if missing_tables:
        missing = ", ".join(missing_tables)
        raise ValueError(f"missing required OpenWebUI table: {missing}")

    for table, required_columns in REQUIRED_SCHEMA.items():
        column_rows = conn.execute(TABLE_INFO_QUERIES[table]).fetchall()
        columns = {str(row["name"]) for row in column_rows}
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"missing required OpenWebUI column in {table}: {missing}")


def _load_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, name, email, created_at, updated_at
            FROM user
            ORDER BY COALESCE(name, email, id), id
            """
        ).fetchall()
    )


def _load_user(conn: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT id, name, email, created_at, updated_at
        FROM user
        WHERE id = ?
        """,
        (user_id,),
    ).fetchone()


def _load_chats_for_user(conn: sqlite3.Connection, user_id: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, user_id, title, chat, created_at, updated_at, share_id, archived, pinned, meta, folder_id
            FROM chat
            WHERE user_id = ?
            ORDER BY COALESCE(updated_at, created_at, 0), id
            """,
            (user_id,),
        ).fetchall()
    )


def _load_folders_for_user(conn: sqlite3.Connection, user_id: str) -> dict[str, sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT id, parent_id, user_id, name, items, meta, is_expanded, created_at, updated_at
        FROM folder
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()
    return {str(row["id"]): row for row in rows}


def _conversation_plan_from_chat_row(
    row: sqlite3.Row,
    *,
    source_user_id: str | None = None,
) -> tuple[OpenWebUIConversationPlan | None, list[str]]:
    external_ref = str(row["id"])
    warnings: list[str] = []
    try:
        chat_payload = json.loads(row["chat"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return None, [f"OpenWebUI chat {external_ref} has malformed chat JSON and was skipped."]

    if not isinstance(chat_payload, dict):
        return None, [f"OpenWebUI chat {external_ref} chat payload is not an object and was skipped."]

    history = chat_payload.get("history")
    messages_map = history.get("messages") if isinstance(history, dict) else None
    if not isinstance(messages_map, dict):
        return None, [f"OpenWebUI chat {external_ref} does not contain history.messages and was skipped."]

    messages: list[OpenWebUIMessagePlan] = []
    for source_key, value in messages_map.items():
        message = _parse_message(str(source_key), value)
        if message is None:
            warnings.append(f"OpenWebUI chat {external_ref} message {source_key} is malformed and was skipped.")
            continue
        messages.append(message)

    meta = _loads_json_object(row["meta"])
    folder_id = _optional_str(row["folder_id"])
    chat_payload["title"] = row["title"] or chat_payload.get("title")
    attachment_count = sum(len(message.attachment_refs) for message in messages)
    return (
        OpenWebUIConversationPlan(
            external_ref=external_ref,
            title=_build_title(chat_payload, messages),
            messages=messages,
            history_current_id=_optional_str(history.get("currentId") or history.get("current_id")),
            is_branched=_is_branched(messages),
            attachment_reference_count=attachment_count,
            source_metadata={
                "source_kind": "openwebui_db",
                "source_user_id": source_user_id or _optional_str(row["user_id"]),
                "row_id": external_ref,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "share_id": _optional_str(row["share_id"]),
                "archived": _sqlite_truthy(row["archived"]),
                "pinned": _sqlite_truthy(row["pinned"]),
                "folder_id": folder_id,
                "meta": meta,
                "models": chat_payload.get("models"),
                "options": chat_payload.get("options"),
            },
            warnings=warnings,
        ),
        warnings,
    )


def _folder_plan_for_chat(
    chat_row: sqlite3.Row,
    folders: dict[str, sqlite3.Row],
) -> OpenWebUIDatabaseFolderPlan:
    chat_id = str(chat_row["id"])
    folder_id = _optional_str(chat_row["folder_id"])
    warnings: list[str] = []
    if not folder_id:
        return OpenWebUIDatabaseFolderPlan(
            source_folder_id=None,
            source_parent_id=None,
            source_path=list(UNFILED_FOLDER_PATH),
        )

    folder = folders.get(folder_id)
    if folder is None:
        return OpenWebUIDatabaseFolderPlan(
            source_folder_id=folder_id,
            source_parent_id=None,
            source_path=list(UNFILED_FOLDER_PATH),
            warnings=[f"OpenWebUI chat {chat_id} references missing folder {folder_id}; routed to Unfiled."],
        )

    path: list[str] = []
    current = folder
    seen: set[str] = set()
    while current is not None:
        current_id = str(current["id"])
        if current_id in seen:
            return OpenWebUIDatabaseFolderPlan(
                source_folder_id=folder_id,
                source_parent_id=_optional_str(folder["parent_id"]),
                source_path=list(UNFILED_FOLDER_PATH),
                source_meta=_loads_json_object(folder["meta"]),
                warnings=[f"OpenWebUI folder cycle detected at folder {current_id}; routed chat {chat_id} to Unfiled."],
            )
        seen.add(current_id)
        path.append(str(current["name"] or current_id))
        parent_id = _optional_str(current["parent_id"])
        if not parent_id:
            break
        current = folders.get(parent_id)
        if current is None:
            return OpenWebUIDatabaseFolderPlan(
                source_folder_id=folder_id,
                source_parent_id=_optional_str(folder["parent_id"]),
                source_path=list(UNFILED_FOLDER_PATH),
                source_meta=_loads_json_object(folder["meta"]),
                warnings=[
                    f"OpenWebUI folder {current_id} references missing parent {parent_id}; routed chat {chat_id} to Unfiled."
                ],
            )

    if not _folder_items_reference_chat(folder, chat_id):
        warnings.append(
            f"OpenWebUI folder.items drift for chat {chat_id}: chat.folder_id {folder_id} is authoritative."
        )

    return OpenWebUIDatabaseFolderPlan(
        source_folder_id=folder_id,
        source_parent_id=_optional_str(folder["parent_id"]),
        source_path=list(reversed(path)) if path else list(UNFILED_FOLDER_PATH),
        source_meta=_loads_json_object(folder["meta"]),
        warnings=warnings,
    )


def _folder_items_reference_chat(folder: sqlite3.Row, chat_id: str) -> bool:
    items = _loads_json_array(folder["items"])
    return chat_id in {str(item) for item in items}


def _loads_json_object(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _loads_json_array(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _display_label_for_user(user: sqlite3.Row) -> str:
    for key in ("name", "email", "id"):
        value = _optional_str(user[key])
        if not value:
            continue
        if key == "email":
            return value.split("@", 1)[0] or value
        return value
    return "OpenWebUI user"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sqlite_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)
