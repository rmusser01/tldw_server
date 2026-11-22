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

import uuid
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
    _owner_is_scoped,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

PINNED_DB_PRIVATES: frozenset[str] = frozenset({"_CURRENT_SCHEMA_VERSION", "_SCHEMA_NAME"})
PINNED_MODULE_PRIVATES: frozenset[str] = frozenset({"_owner_is_scoped", "_SAFE_IDENTIFIER_RE"})


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
