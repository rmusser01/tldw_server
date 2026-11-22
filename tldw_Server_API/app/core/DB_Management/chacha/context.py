"""The stores' view of the CharactersRAGDB they sit on (TASK-13950.22).

``CharactersRAGDB.ctx`` builds one ``ChaChaContext`` per DB. A store that takes it reads the DB's public
state and its few shared helpers here, and reaches no private DB member.

Rules:
  * every member reads the DB at call time and never captures a bound method, so a test that patches
    ``db.execute_query`` or ``CharactersRAGDB._CURRENT_SCHEMA_VERSION`` after construction still lands;
  * its only private DB reads are ``PINNED_DB_PRIVATES``;
  * its only private imports are ``PINNED_MODULE_PRIVATES``.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.query_utils import (
    prepare_backend_statement,
    transform_sqlite_query_for_postgres,
)

# R27: ChaChaNotes_DB keeps using these two. Moving them here would make ChaChaNotes_DB import this module at
# load, which cycles. The import is safe because ChaChaNotes_DB imports this module only inside ``ctx``.
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _SAFE_IDENTIFIER_RE,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    OwnerScope,
    UniqueConstraintError,
    _owner_is_scoped,
)
from tldw_Server_API.app.core.Sync_Contracts.notes_moodboard_studio_contract import (
    canonical_json_bytes as moodboard_studio_canonical_json_bytes,
)
from tldw_Server_API.app.core.Sync_Contracts.notes_task_contract import canonical_json_bytes

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# Class attributes tests patch or init mutates. ``_runtime_schema_version`` is written by the schema bootstrap, so a
# context property reads it (Task 9 makes it public, and the pin goes with it: R29).
PINNED_DB_PRIVATES: frozenset[str] = frozenset({"_CURRENT_SCHEMA_VERSION", "_SCHEMA_NAME", "_runtime_schema_version"})
PINNED_MODULE_PRIVATES: frozenset[str] = frozenset({"_owner_is_scoped", "_SAFE_IDENTIFIER_RE"})

_ORDER_BY_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


class ChaChaContext:
    """Public DB state, call-throughs and pure helpers shared by the ChaCha stores."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    # --- State -------------------------------------------------------------------------------------

    @property
    def db(self) -> CharactersRAGDB:
        """The DB itself, for public calls into another domain."""
        return self._db

    @property
    def backend(self) -> Any:
        """The DB's active backend."""
        return self._db.backend

    @property
    def backend_type(self) -> BackendType:
        """The DB's backend type."""
        return self._db.backend_type

    @property
    def client_id(self) -> str:
        """The client id the DB stamps on its writes."""
        return self._db.client_id

    @property
    def owner_user_id(self) -> str:
        """The DB's construction-time owner."""
        return self._db.owner_user_id

    @property
    def db_path_str(self) -> str:
        """The DB path as a string, or ``:memory:``."""
        return self._db.db_path_str

    @property
    def is_memory_db(self) -> bool:
        """Whether the DB is an in-memory SQLite database."""
        return self._db.is_memory_db

    @property
    def code_schema_version(self) -> int:
        """The schema version this code builds."""
        return self._db._CURRENT_SCHEMA_VERSION

    @property
    def schema_name(self) -> str:
        """The schema's name in the version table."""
        return self._db._SCHEMA_NAME

    @property
    def runtime_schema_version(self) -> int | None:
        """The schema version the bootstrap recorded for this DB, or None before it ran."""
        return getattr(self._db, "_runtime_schema_version", None)

    # --- Call-throughs ------------------------------------------------------------------------------

    def transaction(self, *, preserve_existing: bool = False) -> Any:
        """Open the DB's transaction context manager."""
        if preserve_existing:  # the default keeps the call shape of a fake that takes no keyword
            return self._db.transaction(preserve_existing=True)
        return self._db.transaction()

    def get_connection(self) -> Any:
        """Return the DB's connection for this thread."""
        return self._db.get_connection()

    def execute_query(self, query: str, params: Any = None, **kwargs: Any) -> Any:
        """Run one statement through the DB."""
        if params is None:
            return self._db.execute_query(query, **kwargs)
        return self._db.execute_query(query, params, **kwargs)

    def execute_many(self, query: str, params_list: Any, *, commit: bool = False) -> Any:
        """Run one statement over many parameter sets through the DB."""
        return self._db.execute_many(query, params_list, commit=commit)

    # --- Helpers ------------------------------------------------------------------------------------

    def utc_now_iso(self) -> str:
        """
        Generates the current UTC timestamp in ISO 8601 format with 'Z' for UTC.

        Example: "2023-10-27T10:30:00.123Z"
        """
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def new_uuid(self) -> str:
        """Generate a new UUID version 4 as a string."""
        return str(uuid.uuid4())

    @staticmethod
    def normalize_nullable_text(value: Any) -> str | None:
        """Normalize optional text fields; returns None for empty/whitespace."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return str(value)

    def map_table(self, table_name: str) -> str:
        """Map logical table names to backend-specific physical names.

        This avoids name collisions with other modules when sharing a Postgres database.
        """
        if self.backend_type == BackendType.POSTGRESQL:
            mapping = {
                "keywords": "chacha_keywords",
            }
            return mapping.get(table_name, table_name)
        return table_name

    def owner_filter(
        self, owner_client_id: str | OwnerScope | None, alias: str = "", *, table: str = "?",
    ) -> tuple[str, tuple[str, ...]]:
        """Apply the PostgreSQL owner; SQLite IDs label devices. None raises on PostgreSQL."""
        if not _owner_is_scoped(self.backend_type, owner_client_id, "_selected_owner_filter", table):
            return "", ()
        if alias and not _SAFE_IDENTIFIER_RE.fullmatch(alias):
            raise InputError("Invalid owner filter alias")  # noqa: TRY003
        column = f"{alias}.client_id" if alias else "client_id"
        return f" AND {column} = ?", (owner_client_id,)

    def current_db_version(
        self, conn: Any, table_name: str, pk_col_name: str, pk_value: Any,
        *, owner_client_id: str | OwnerScope | None = None,
    ) -> int:
        """
        Fetches the current version of an active (not soft-deleted) record.

        Used for optimistic locking checks before an update or soft delete.

        Raises:
            ConflictError: If the record is not found (with `entity` and `entity_id` attributes
                           set in the exception) or if the record is found but is soft-deleted.
        """
        if not (_SAFE_IDENTIFIER_RE.fullmatch(table_name or "") and _SAFE_IDENTIFIER_RE.fullmatch(pk_col_name or "")):
            raise CharactersRAGDBError(  # noqa: TRY003
                f"Unsafe identifier in version lookup: table={table_name!r}, column={pk_col_name!r}"
            )
        owner_clause, owner_params = self.owner_filter(owner_client_id, table=table_name)
        cursor = conn.execute(f"SELECT version, deleted FROM {table_name} WHERE {pk_col_name} = ?{owner_clause}", (pk_value,) + owner_params)  # nosec B608
        row = cursor.fetchone()

        if not row:
            logger.warning(f"Record not found in {table_name} with {pk_col_name} = {pk_value} for version check.")
            raise ConflictError(f"Record not found in {table_name}.", entity=table_name, entity_id=pk_value)  # noqa: TRY003

        if row['deleted']:
            logger.warning(f"Record in {table_name} with {pk_col_name} = {pk_value} is soft-deleted.")
            raise ConflictError(f"Record is soft-deleted in {table_name}.", entity=table_name, entity_id=pk_value)  # noqa: TRY003

        version: int = row["version"]
        return version

    def prepare_statement(
        self, query: str, params: tuple | list | dict | None = None,
    ) -> tuple[str, tuple | list | dict | None]:
        """Convert a SQLite-style statement for the active backend (R27: no DB private is read)."""
        return prepare_backend_statement(
            self.backend_type,
            query,
            params,
            transformer=transform_sqlite_query_for_postgres if self.backend_type == BackendType.POSTGRESQL else None,
            ensure_returning=False,
        )

    def keyword_link_filter(
        self, link_table: str, alias: str, *, owner_client_id: str | OwnerScope | None,
    ) -> tuple[str, tuple[str, ...]]:
        """Scope both ends of the keyword relationship. None raises on PostgreSQL."""
        if not _owner_is_scoped(self.backend_type, owner_client_id, "_selected_keyword_link_filter", link_table):
            return "", ()
        parents = {
            "note_keywords": ("notes", "note_id"),
            "conversation_keywords": ("conversations", "conversation_id"),
            "collection_keywords": ("keyword_collections", "collection_id"),
            "flashcard_keywords": ("flashcards", "card_id"),
        }
        parent_table, parent_column = parents[link_table]
        if not _SAFE_IDENTIFIER_RE.fullmatch(alias):
            raise InputError("Invalid keyword link alias")  # noqa: TRY003
        keyword_table = self.map_table("keywords")
        return (
            f" AND EXISTS (SELECT 1 FROM {parent_table} owner_parent "  # nosec B608
            f"WHERE owner_parent.id = {alias}.{parent_column} AND owner_parent.client_id = ?)"
            f" AND EXISTS (SELECT 1 FROM {keyword_table} owner_keyword "
            f"WHERE owner_keyword.id = {alias}.keyword_id AND owner_keyword.client_id = ?)",
            (owner_client_id, owner_client_id),
        )

    def require_owner_row(
        self, conn: Any, table_name: str, item_id: Any, owner_client_id: str | OwnerScope | None,
        *, include_deleted: bool = False,
    ) -> None:
        """Lock a scoped parent until its existing transaction completes. None raises on PostgreSQL."""
        if item_id is None:
            return
        if not _owner_is_scoped(self.backend_type, owner_client_id, "_require_selected_owner_row", table_name):
            return
        if table_name == "messages":
            # Message client IDs may name sync devices. Its conversation owns it.
            # Match the existing message-edit lock order: message, then conversation.
            row = conn.execute(
                "SELECT conversation_id FROM messages WHERE id = ? AND deleted = FALSE FOR UPDATE",
                (item_id,),
            ).fetchone()
            if row is not None:
                try:
                    self.require_owner_row(conn, "conversations", row["conversation_id"], owner_client_id)
                except ConflictError:
                    row = None
            if row is None:
                raise ConflictError("Referenced message not found", entity="messages", entity_id=item_id)  # noqa: TRY003
            return
        if table_name not in {"notes", "conversations", "keywords", "keyword_collections", "note_folders", "study_packs", "flashcards"}:
            raise InputError("Unsupported Notes parent table")  # noqa: TRY003
        lookup_column = "uuid" if table_name == "flashcards" else "id"
        table_name = self.map_table(table_name)
        deleted_clause = "" if include_deleted else " AND deleted = FALSE"
        row = conn.execute(
            f"SELECT id FROM {table_name} WHERE {lookup_column} = ? AND client_id = ?{deleted_clause} FOR UPDATE",  # nosec B608
            (item_id, owner_client_id),
        ).fetchone()
        if row is None:
            raise ConflictError("Referenced record not found", entity=table_name, entity_id=item_id)  # noqa: TRY003

    def unique_text_match_sql(self, table_name: str, column: str) -> str:
        """Return the equality predicate for a unique text column.

        Keywords and keyword collections are case-insensitive (TASK-13950.5). SQLite
        gets that from the columns' ``COLLATE NOCASE``; PostgreSQL strips NOCASE and
        indexes ``LOWER(keyword)``/``LOWER(name)`` instead, so it compares ``LOWER`` too.
        """
        if table_name in {"keywords", "keyword_collections"} and self.backend_type == BackendType.POSTGRESQL:
            return f"LOWER({column}) = LOWER(?)"
        return f"{column} = ?"

    def case_insensitive_order_expression(self, column: str, direction: str | None = None) -> str:
        """Return a case-insensitive ORDER BY expression for ``column``, validated against injection."""
        column_clean = (column or '').strip()
        if not _ORDER_BY_COLUMN_RE.match(column_clean):
            raise InputError(f"Invalid order-by column: {column!r}")  # noqa: TRY003
        direction_clean = (direction or '').strip()
        direction_clause = f" {direction_clean}" if direction_clean else ""
        if self.backend_type == BackendType.POSTGRESQL:
            return f"LOWER({column_clean}){direction_clause}"
        return f"{column_clean} COLLATE NOCASE{direction_clause}"

    def case_insensitive_order_clause(self, column: str, direction: str | None = None) -> str:
        """Return ``ORDER BY`` followed by the case-insensitive expression for ``column``."""
        return f"ORDER BY {self.case_insensitive_order_expression(column, direction)}"

    def is_unique_violation(self, error: Exception) -> bool:
        """Return True if the provided backend error represents a unique constraint violation."""
        if isinstance(error, UniqueConstraintError):
            return True
        message = str(error).lower()
        return "unique constraint" in message or "duplicate key" in message

    @staticmethod
    def chunk_list(lst: list, size: int) -> list[list]:
        """Split *lst* into sub-lists of at most *size* elements."""
        return [lst[i:i + size] for i in range(0, len(lst), size)]

    @staticmethod
    def canonical_json_text_v61(value: object) -> str:
        """Return the canonical moodboard/Studio JSON text of ``value``."""
        return moodboard_studio_canonical_json_bytes(value).decode("utf-8")

    @staticmethod
    def note_task_v60_hash(value: object) -> str:
        """Return the ``sha256:`` hash of the canonical JSON bytes of ``value``."""
        return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()

    def deserialize_row_fields(self, row: Any, json_fields: Sequence[str]) -> dict[str, Any] | None:
        """
        Converts a sqlite3.Row object to a dictionary, deserializing specified JSON fields.

        If a field listed in `json_fields` contains a string, it attempts to
        parse it as JSON. If parsing fails, the field value is set to None
        and a warning is logged.

        Args:
            row: The sqlite3.Row object to convert. If None, returns None.
            json_fields: A sequence of field names that should be treated as JSON strings
                         and deserialized.

        Returns:
            A dictionary representing the row with specified fields deserialized,
            or None if the input `row` is None.
        """
        if not row:
            return None
        item = dict(row)
        for field in json_fields:
            if field in item and isinstance(item[field], str):
                try:
                    item[field] = json.loads(item[field])
                except json.JSONDecodeError:
                    pk_val = item.get('id') or item.get('uuid', 'N/A')  # Try to get an identifier
                    logger.warning(
                        f"Failed to decode JSON for field '{field}' in row (ID: {pk_val}). Value: '{item[field][:100]}...'")
                    item[field] = None  # Or sensible default
        return item
