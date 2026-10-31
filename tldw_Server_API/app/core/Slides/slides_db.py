"""SlidesDatabase: per-user SQLite storage for presentations."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.Slides.slides_migrations import (
    SLIDES_SCHEMA_VERSION,
    migrate_slides_schema,
    slides_schema_v2_is_complete,
)
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationResult,
)


class SlidesDatabaseError(Exception):
    """Base exception for SlidesDatabase."""


class SchemaError(SlidesDatabaseError):
    """Raised when schema migrations fail."""


class ConflictError(SlidesDatabaseError):
    """Raised when optimistic locking fails or duplicates exist."""

    def __init__(self, message: str, *, entity: str | None = None, identifier: str | None = None):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier


class InputError(ValueError):
    """Raised for invalid inputs."""


_STANDALONE_HTML_MAX_DOCUMENT_BYTES = 1_048_576
_STANDALONE_HTML_SNAPSHOT_MAX_BYTES = 2 * _STANDALONE_HTML_MAX_DOCUMENT_BYTES + 65_536
_STANDALONE_HTML_MAX_VERSION_RETENTION = 25
_STANDALONE_HTML_MAX_EMPTY_SLIDES_JSON_CHARS = 64
_GENERATION_RECONCILIATION_BATCH_MAX = 500
_GENERATION_RECEIPT_MISSING_CODE = "generation_receipt_unresolved_pending"
_SNAPSHOT_SCHEMA_VERSION = 1


def decode_presentation_version_payload(payload_json: str) -> dict[str, Any]:
    """Decode a snapshot without retaining source-bearing JSON exceptions."""
    payload: Any = None
    try:
        payload = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError, RecursionError):
        pass
    if not isinstance(payload, dict):
        raise InputError("version_payload_invalid")
    return payload


def bind_validated_standalone_source(
    html_document: str | bytes,
    validation_result: StandaloneHtmlValidationResult,
) -> str:
    """Bind one immutable validator result to the exact source bytes."""
    if not isinstance(validation_result, StandaloneHtmlValidationResult):
        raise InputError("standalone_html_validation_result_mismatch")
    source: str | None = None
    if isinstance(html_document, str):
        source = html_document
    elif isinstance(html_document, bytes):
        try:
            source = html_document.decode("utf-8", "strict")
        except UnicodeDecodeError:
            pass
    if source is None:
        raise InputError("standalone_html_unsupported_encoding")
    encoded: bytes | None = None
    try:
        encoded = source.encode("utf-8", "strict")
    except UnicodeEncodeError:
        pass
    if encoded is None:
        raise InputError("standalone_html_unsupported_encoding")
    if (
        validation_result.html_bytes != len(encoded)
        or validation_result.html_sha256 != hashlib.sha256(encoded).hexdigest()
    ):
        raise InputError("standalone_html_validation_result_mismatch")
    return source


@dataclass
class PresentationRow:
    id: str
    title: str
    description: str | None
    theme: str
    marp_theme: str | None
    template_id: str | None
    visual_style_id: str | None
    visual_style_scope: str | None
    visual_style_name: str | None
    visual_style_version: int | None
    visual_style_snapshot: str | None
    settings: str | None
    studio_data: str | None
    slides: str
    slides_text: str
    source_type: str | None
    source_ref: str | None
    source_query: str | None
    custom_css: str | None
    created_at: str
    last_modified: str
    deleted: int
    client_id: str
    version: int
    content_kind: str
    html_document: str | None
    html_sha256: str | None
    html_bytes: int | None
    html_slide_count: int | None
    generation_job_uuid: str | None
    generation_provenance_json: str | None


PresentationDetailRow = PresentationRow


@dataclass
class PresentationSummaryRow:
    id: str
    title: str
    description: str | None
    theme: str
    content_kind: str
    source_kind: str | None
    provider: str | None
    model: str | None
    slide_count: int | None
    html_slide_count: int | None
    html_bytes: int | None
    created_at: str
    last_modified: str
    deleted: int
    version: int


@dataclass
class PresentationKindRow:
    id: str
    content_kind: str
    version: int
    deleted: int
    last_modified: str


@dataclass
class PresentationSourceIdentityRow:
    id: str
    title: str
    content_kind: str
    version: int
    deleted: int
    last_modified: str
    html_sha256: str | None
    html_bytes: int | None


@dataclass
class PresentationVersionRow:
    presentation_id: str
    version: int
    payload_json: str
    created_at: str
    client_id: str


@dataclass
class PresentationVersionMetadataRow:
    presentation_id: str
    version: int
    created_at: str
    client_id: str
    title: str | None
    deleted: int | None


@dataclass
class SlidesGenerationReceiptRow:
    id: str
    owner_user_id: str
    digest_key_id: str
    idempotency_key_hmac_sha256: str
    jobs_idempotency_key: str
    client_request_hmac_sha256: str
    execution_hmac_sha256: str
    job_id: int | None
    job_uuid: str | None
    presentation_id: str | None
    receipt_status: str
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    expires_at: str | None


@dataclass
class SlidesGenerationInputRow:
    receipt_id: str
    source_kind: str
    source_text: str
    source_hmac_sha256: str
    source_bytes: int
    provenance_json: str
    html_options_json: str
    provider: str
    model: str
    adapter_id: str
    endpoint_identity: str
    system_prompt: str
    prompt_sha256: str
    prompt_contract_version: str
    input_expires_at: str
    created_at: str


@dataclass(frozen=True, slots=True)
class SlidesGenerationReconciliationRow:
    """Source-free receipt metadata used by the dormant-database reconciler."""

    id: str
    owner_user_id: str
    digest_key_id: str
    jobs_idempotency_key: str
    job_id: int | None
    job_uuid: str | None
    presentation_id: str | None
    receipt_status: str
    error_code: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    expires_at: str | None
    input_expires_at: str | None
    input_exists: bool
    presentation_exists: bool
    presentation_content_kind: str | None
    presentation_generation_job_uuid: str | None


@dataclass(frozen=True, slots=True)
class SlidesGenerationClaimResult:
    """Atomic receipt/input claim result."""

    receipt: SlidesGenerationReceiptRow
    generation_input: SlidesGenerationInputRow | None
    created: bool


@dataclass(frozen=True, slots=True)
class SlidesGenerationCommitResult:
    """Atomic standalone commit result, including idempotent recovery."""

    presentation: PresentationRow
    created: bool


@dataclass
class VisualStyleRow:
    id: str
    name: str
    scope: str
    style_payload: str
    created_at: str
    updated_at: str


_PRESENTATION_DETAIL_COLUMNS = (
    "id",
    "title",
    "description",
    "theme",
    "marp_theme",
    "template_id",
    "visual_style_id",
    "visual_style_scope",
    "visual_style_name",
    "visual_style_version",
    "visual_style_snapshot",
    "settings",
    "studio_data",
    "slides",
    "slides_text",
    "source_type",
    "source_ref",
    "source_query",
    "custom_css",
    "created_at",
    "last_modified",
    "deleted",
    "client_id",
    "version",
    "content_kind",
    "html_document",
    "html_sha256",
    "html_bytes",
    "html_slide_count",
    "generation_job_uuid",
    "generation_provenance_json",
)
_PRESENTATION_DETAIL_PROJECTION = ", ".join(_PRESENTATION_DETAIL_COLUMNS)
_PRESENTATION_DETAIL_PROJECTION_QUALIFIED = ", ".join(f"p.{column}" for column in _PRESENTATION_DETAIL_COLUMNS)


def _build_presentation_summary_projection(table_alias: str | None = None) -> str:
    """Build the source-free summary projection with an optional table alias."""

    prefix = f"{table_alias}." if table_alias else ""
    expressions = (
        *(f"{prefix}{column}" for column in ("id", "title", "description", "theme", "content_kind")),
        f"json_extract({prefix}generation_provenance_json, '$.source_kind') AS source_kind",
        f"json_extract({prefix}generation_provenance_json, '$.provider') AS provider",
        f"json_extract({prefix}generation_provenance_json, '$.model') AS model",
        (
            f"CASE WHEN {prefix}content_kind = 'standalone_html' "
            f"THEN {prefix}html_slide_count ELSE json_array_length({prefix}slides) "
            "END AS slide_count"
        ),
        *(
            f"{prefix}{column}"
            for column in ("html_slide_count", "html_bytes", "created_at", "last_modified", "deleted", "version")
        ),
    )
    return ",\n    ".join(expressions)


_PRESENTATION_SUMMARY_PROJECTION = _build_presentation_summary_projection()
_PRESENTATION_SUMMARY_PROJECTION_QUALIFIED = _build_presentation_summary_projection("p")

_RECEIPT_PROJECTION = """
    id, owner_user_id, digest_key_id, idempotency_key_hmac_sha256,
    jobs_idempotency_key, client_request_hmac_sha256,
    execution_hmac_sha256, job_id, job_uuid, presentation_id,
    receipt_status, error_code, error_message, created_at, updated_at,
    expires_at
"""
_INPUT_PROJECTION = """
    receipt_id, source_kind, source_text, source_hmac_sha256, source_bytes,
    provenance_json, html_options_json, provider, model, adapter_id,
    endpoint_identity, system_prompt, prompt_sha256, prompt_contract_version,
    input_expires_at, created_at
"""
_GENERATION_RECONCILIATION_PROJECTION = """
    r.id, r.owner_user_id, r.digest_key_id, r.jobs_idempotency_key,
    r.job_id, r.job_uuid, r.presentation_id, r.receipt_status,
    r.error_code, r.error_message, r.created_at, r.updated_at, r.expires_at,
    i.input_expires_at,
    CASE WHEN i.receipt_id IS NULL THEN 0 ELSE 1 END AS input_exists,
    CASE WHEN p.id IS NULL THEN 0 ELSE 1 END AS presentation_exists,
    p.content_kind AS presentation_content_kind,
    p.generation_job_uuid AS presentation_generation_job_uuid
"""

_REQUIRED_BASE_SCHEMA_OBJECTS = {
    ("table", "schema_version"),
    ("table", "presentations"),
    ("table", "presentations_versions"),
    ("table", "presentations_fts"),
    ("table", "sync_log"),
    ("table", "visual_styles"),
    ("index", "idx_presentations_deleted"),
    ("index", "idx_presentations_created"),
    ("index", "idx_presentations_versions_unique"),
    ("index", "idx_presentations_versions_pid"),
    ("index", "idx_presentations_versions_created"),
    ("index", "idx_sync_log_ts"),
    ("index", "idx_sync_log_entity_uuid"),
    ("index", "idx_sync_log_client_id"),
    ("index", "idx_visual_styles_scope"),
    ("index", "idx_visual_styles_name"),
    ("trigger", "presentations_ai"),
    ("trigger", "presentations_ad"),
    ("trigger", "presentations_au"),
}


class SlidesDatabase:
    _SCHEMA_VERSION = SLIDES_SCHEMA_VERSION

    def __init__(
        self,
        db_path: str | Path,
        client_id: str,
        *,
        standalone_html_version_retention: int = _STANDALONE_HTML_MAX_VERSION_RETENTION,
    ) -> None:
        if not client_id:
            raise ValueError("client_id is required")
        if not 1 <= standalone_html_version_retention <= _STANDALONE_HTML_MAX_VERSION_RETENTION:
            raise ValueError("standalone_html_version_retention must be between 1 and 25")
        self.client_id = str(client_id)
        self.standalone_html_version_retention = standalone_html_version_retention
        if isinstance(db_path, Path):
            self.db_path = db_path.resolve()
            self._db_path_str = str(self.db_path)
        else:
            self._db_path_str = str(db_path)
            self.db_path = Path(self._db_path_str).resolve() if self._db_path_str != ":memory:" else Path(":memory:")
        self._sqlite_uri = False
        self._expected_file_identity: tuple[int, int, int] | None = None
        self._expected_directory_identities: tuple[tuple[Path, tuple[int, int, int]], ...] = ()
        self._local = threading.local()
        self._ensure_schema()

    @classmethod
    def open_existing_complete(
        cls,
        db_path: str | Path,
        client_id: str,
        *,
        expected_file_identity: tuple[int, int, int],
        expected_directory_identities: tuple[
            tuple[str | Path, tuple[int, int, int]],
            ...,
        ] = (),
        standalone_html_version_retention: int = _STANDALONE_HTML_MAX_VERSION_RETENTION,
    ) -> SlidesDatabase:
        """Open one existing complete database without creating or migrating it."""
        if not client_id:
            raise ValueError("client_id is required")
        if not 1 <= standalone_html_version_retention <= _STANDALONE_HTML_MAX_VERSION_RETENTION:
            raise ValueError("standalone_html_version_retention must be between 1 and 25")
        if (
            not isinstance(expected_file_identity, tuple)
            or len(expected_file_identity) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in expected_file_identity)
        ):
            raise ValueError("expected_file_identity is invalid")
        if expected_file_identity[2] != stat.S_IFREG or not isinstance(expected_directory_identities, tuple):
            raise ValueError("expected_file_identity is invalid")
        normalized_directories: list[tuple[Path, tuple[int, int, int]]] = []
        for entry in expected_directory_identities:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[1], tuple)
                or len(entry[1]) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in entry[1])
                or entry[1][2] != stat.S_IFDIR
            ):
                raise ValueError("expected_directory_identities is invalid")
            normalized_directories.append(
                (
                    Path(os.path.abspath(os.fspath(Path(entry[0])))),
                    entry[1],
                )
            )

        lexical_path = Path(os.path.abspath(os.fspath(Path(db_path))))
        instance = cls.__new__(cls)
        instance.client_id = str(client_id)
        instance.standalone_html_version_retention = standalone_html_version_retention
        instance.db_path = lexical_path
        instance._db_path_str = lexical_path.as_uri() + "?mode=rw"
        instance._sqlite_uri = True
        instance._expected_file_identity = expected_file_identity
        instance._expected_directory_identities = tuple(normalized_directories)
        instance._local = threading.local()
        try:
            instance.get_connection()
        except Exception as exc:
            instance.close_connection()
            if isinstance(exc, SchemaError) and str(exc) == "Slides database is unavailable or incomplete":
                raise
            raise SchemaError("Slides database is unavailable or incomplete") from exc
        return instance

    @staticmethod
    def _schema_is_complete(conn: sqlite3.Connection) -> bool:
        """Return whether every base and v2 schema object is present."""
        if not slides_schema_v2_is_complete(conn):
            return False
        objects = {
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master " "WHERE type IN ('table', 'index', 'trigger')"
            ).fetchall()
        }
        if not _REQUIRED_BASE_SCHEMA_OBJECTS.issubset(objects):
            return False
        columns = {
            str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
            for row in conn.execute("PRAGMA table_info(presentations)").fetchall()
        }
        return set(_PRESENTATION_DETAIL_COLUMNS).issubset(columns)

    @staticmethod
    def _execute_schema_statements(conn: sqlite3.Connection, script: str) -> None:
        """Execute a DDL script one complete statement at a time."""
        pending: list[str] = []
        for line in script.splitlines():
            pending.append(line)
            statement = "\n".join(pending).strip()
            if statement and sqlite3.complete_statement(statement):
                conn.execute(statement)
                pending.clear()
        if any(line.strip() for line in pending):
            raise SchemaError("Slides base schema contains an incomplete statement")

    def _ensure_schema(self) -> None:
        conn = self.get_connection()
        try:
            if self._schema_is_complete(conn):
                return
            conn.execute("BEGIN IMMEDIATE")
            if self._schema_is_complete(conn):
                conn.rollback()
                return
            self._execute_schema_statements(
                conn,
                """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS presentations (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        description TEXT,
                        theme TEXT DEFAULT 'black',
                        marp_theme TEXT,
                        template_id TEXT,
                        visual_style_id TEXT,
                        visual_style_scope TEXT,
                        visual_style_name TEXT,
                        visual_style_version INTEGER,
                        visual_style_snapshot TEXT,
                        settings TEXT,
                        studio_data TEXT,
                        slides TEXT NOT NULL,
                        slides_text TEXT NOT NULL,
                        source_type TEXT,
                        source_ref TEXT,
                        source_query TEXT,
                        custom_css TEXT,
                        created_at DATETIME NOT NULL,
                        last_modified DATETIME NOT NULL,
                        deleted INTEGER DEFAULT 0,
                        client_id TEXT NOT NULL,
                        version INTEGER DEFAULT 1
                    );

                    CREATE INDEX IF NOT EXISTS idx_presentations_deleted ON presentations(deleted);
                    CREATE INDEX IF NOT EXISTS idx_presentations_created ON presentations(created_at);

                    CREATE TABLE IF NOT EXISTS presentations_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        presentation_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        client_id TEXT NOT NULL,
                        title TEXT NULL,
                        deleted INTEGER NULL
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_presentations_versions_unique
                        ON presentations_versions(presentation_id, version);
                    CREATE INDEX IF NOT EXISTS idx_presentations_versions_pid
                        ON presentations_versions(presentation_id);
                    CREATE INDEX IF NOT EXISTS idx_presentations_versions_created
                        ON presentations_versions(created_at);

                    CREATE VIRTUAL TABLE IF NOT EXISTS presentations_fts USING fts5(
                        title,
                        slides_text,
                        content=presentations,
                        content_rowid=rowid
                    );

                    CREATE TRIGGER IF NOT EXISTS presentations_ai AFTER INSERT ON presentations BEGIN
                      INSERT INTO presentations_fts(rowid, title, slides_text)
                      VALUES (new.rowid, new.title, new.slides_text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS presentations_ad AFTER DELETE ON presentations BEGIN
                      INSERT INTO presentations_fts(presentations_fts, rowid, title, slides_text)
                      VALUES ('delete', old.rowid, old.title, old.slides_text);
                    END;

                    CREATE TRIGGER IF NOT EXISTS presentations_au AFTER UPDATE ON presentations BEGIN
                      INSERT INTO presentations_fts(presentations_fts, rowid, title, slides_text)
                      VALUES ('delete', old.rowid, old.title, old.slides_text);
                      INSERT INTO presentations_fts(rowid, title, slides_text)
                      VALUES (new.rowid, new.title, new.slides_text);
                    END;

                    CREATE TABLE IF NOT EXISTS sync_log (
                        change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity TEXT NOT NULL,
                        entity_uuid TEXT NOT NULL,
                        operation TEXT NOT NULL CHECK(operation IN ('create','update','delete','restore')),
                        timestamp DATETIME NOT NULL,
                        client_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        payload TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
                    CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id);

                    CREATE TABLE IF NOT EXISTS visual_styles (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        style_payload TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_visual_styles_scope ON visual_styles(scope);
                    CREATE INDEX IF NOT EXISTS idx_visual_styles_name ON visual_styles(name);
                """,
            )
            self._ensure_marp_theme_column(conn)
            self._ensure_template_id_column(conn)
            self._ensure_presentation_visual_style_columns(conn)
            self._ensure_studio_data_column(conn)
            self._ensure_visual_styles_table(conn)
            conn.commit()
            migrate_slides_schema(conn)
            if not self._schema_is_complete(conn):
                raise SchemaError("Slides schema initialization did not reach version 2")
        except Exception as exc:
            conn.rollback()
            if isinstance(exc, SchemaError):
                raise
            raise SchemaError(f"Failed to initialize Slides DB schema: {exc}") from exc

    def get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            try:
                conn = sqlite3.connect(
                    self._db_path_str,
                    check_same_thread=False,
                    uri=self._sqlite_uri,
                )
                conn.row_factory = sqlite3.Row
                if self._expected_file_identity is not None:
                    self._validate_opened_existing_connection(conn)
                configure_sqlite_connection(conn)
            except Exception:
                if conn is not None:
                    conn.close()
                raise
            self._local.connection = conn
        return conn

    def _validate_opened_existing_connection(self, conn: sqlite3.Connection) -> None:
        """Bind a strict-open connection to the validated regular-file inode."""
        expected = self._expected_file_identity
        if expected is None:
            return

        def require_identity(
            path: Path,
            expected_identity: tuple[int, int, int],
            *,
            directory: bool,
        ) -> None:
            metadata = os.stat(path, follow_symlinks=False)
            identity = (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))
            expected_type = stat.S_ISDIR if directory else stat.S_ISREG
            if stat.S_ISLNK(metadata.st_mode) or not expected_type(metadata.st_mode) or identity != expected_identity:
                raise SchemaError("Slides database is unavailable or incomplete")

        try:
            for directory_path, directory_identity in self._expected_directory_identities:
                require_identity(directory_path, directory_identity, directory=True)
            require_identity(self.db_path, expected, directory=False)
            if not self._schema_is_complete(conn):
                raise SchemaError("Slides database is unavailable or incomplete")
            for directory_path, directory_identity in self._expected_directory_identities:
                require_identity(directory_path, directory_identity, directory=True)
            require_identity(self.db_path, expected, directory=False)
        except (OSError, sqlite3.Error, SchemaError, ValueError) as exc:
            if isinstance(exc, SchemaError):
                raise
            raise SchemaError("Slides database is unavailable or incomplete") from exc

    def close_connection(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterable[sqlite3.Connection]:
        conn = self.get_connection()
        if conn.in_transaction:
            raise SlidesDatabaseError("nested Slides transactions are not supported")
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _canonical_utc_timestamp(value: object) -> datetime:
        """Parse one second-precision canonical UTC timestamp."""
        if not isinstance(value, str):
            raise InputError("timestamp must be canonical UTC")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise InputError("timestamp must be canonical UTC") from None
        if (
            parsed.tzinfo is None
            or parsed.utcoffset() != timedelta(0)
            or parsed.replace(microsecond=0).isoformat() != value
        ):
            raise InputError("timestamp must be canonical UTC")
        return parsed

    @staticmethod
    def _generation_job_uuid_cas(expected_job_uuid: str | None, *, alias: str = "") -> tuple[str, tuple[str, ...]]:
        """Build an exact nullable Jobs UUID predicate for internal static SQL."""
        column = f"{alias}.job_uuid" if alias else "job_uuid"
        if expected_job_uuid is None:
            return f"{column} IS NULL", ()
        return f"{column} = ?", (expected_job_uuid,)

    def _insert_sync_log(
        self,
        conn: sqlite3.Connection,
        /,
        *,
        entity_uuid: str,
        operation: str,
        version: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        payload_json = json.dumps(payload) if payload is not None else None
        conn.execute(
            """
            INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "presentations",
                entity_uuid,
                operation,
                self._utcnow_iso(),
                self.client_id,
                version,
                payload_json,
            ),
        )

    @staticmethod
    def _ensure_marp_theme_column(conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "marp_theme" for col in columns):
            return
        conn.execute("ALTER TABLE presentations ADD COLUMN marp_theme TEXT")

    @staticmethod
    def _ensure_template_id_column(conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "template_id" for col in columns):
            return
        conn.execute("ALTER TABLE presentations ADD COLUMN template_id TEXT")

    @staticmethod
    def _ensure_studio_data_column(conn: sqlite3.Connection) -> None:
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "studio_data" for col in columns):
            return
        conn.execute("ALTER TABLE presentations ADD COLUMN studio_data TEXT")

    @staticmethod
    def _ensure_presentation_visual_style_columns(conn: sqlite3.Connection) -> None:
        columns = {col["name"] for col in conn.execute("PRAGMA table_info(presentations)").fetchall()}
        if "visual_style_id" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_id TEXT")
        if "visual_style_scope" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_scope TEXT")
        if "visual_style_name" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_name TEXT")
        if "visual_style_version" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_version INTEGER")
        if "visual_style_snapshot" not in columns:
            conn.execute("ALTER TABLE presentations ADD COLUMN visual_style_snapshot TEXT")

    @staticmethod
    def _ensure_visual_styles_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visual_styles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                scope TEXT NOT NULL,
                style_payload TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_styles_scope ON visual_styles(scope)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_styles_name ON visual_styles(name)")

    @staticmethod
    def _validate_presentation_candidate(candidate: Mapping[str, Any]) -> None:
        """Validate the complete discriminated row before it is persisted."""
        slides_json = candidate.get("slides")
        content_kind = candidate.get("content_kind")
        if content_kind == "standalone_html" and (
            not isinstance(slides_json, str) or len(slides_json) > _STANDALONE_HTML_MAX_EMPTY_SLIDES_JSON_CHARS
        ):
            raise InputError("standalone_html slides must be an empty JSON list")
        try:
            parsed_slides = json.loads(slides_json)
        except (TypeError, json.JSONDecodeError, RecursionError) as exc:
            raise InputError("slides must be a valid JSON list") from exc
        if not isinstance(parsed_slides, list):
            raise InputError("slides must be a valid JSON list")

        standalone_fields = (
            "html_document",
            "html_sha256",
            "html_bytes",
            "html_slide_count",
            "generation_job_uuid",
            "generation_provenance_json",
        )
        if content_kind == "structured_slides":
            if any(candidate.get(field) is not None for field in standalone_fields):
                raise InputError("structured_slides cannot contain standalone presentation fields")
            return
        if content_kind != "standalone_html":
            raise InputError("content_kind must be one of: structured_slides, standalone_html")
        if parsed_slides != []:
            raise InputError("standalone_html slides must be an empty JSON list")

        html_document = candidate.get("html_document")
        if not isinstance(html_document, str) or not html_document.strip():
            raise InputError("standalone_html html_document must be nonblank")
        try:
            html_bytes = html_document.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InputError("standalone_html html_document must be valid UTF-8") from exc

        expected_digest = hashlib.sha256(html_bytes).hexdigest()
        if candidate.get("html_sha256") != expected_digest:
            raise InputError("standalone_html html_sha256 does not match html_document")
        stored_bytes = candidate.get("html_bytes")
        if isinstance(stored_bytes, bool) or stored_bytes != len(html_bytes):
            raise InputError("standalone_html html_bytes does not match html_document")
        slide_count = candidate.get("html_slide_count")
        if isinstance(slide_count, bool) or not isinstance(slide_count, int) or not 1 <= slide_count <= 30:
            raise InputError("standalone_html html_slide_count must be between 1 and 30")

        generation_job_uuid = candidate.get("generation_job_uuid")
        if not isinstance(generation_job_uuid, str) or not generation_job_uuid.strip():
            raise InputError("standalone_html generation_job_uuid is required")
        provenance_json = candidate.get("generation_provenance_json")
        if not isinstance(provenance_json, str):
            raise InputError("standalone_html generation_provenance_json is required")
        try:
            encoded_provenance = provenance_json.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InputError("standalone_html generation_provenance_json must be valid JSON") from exc
        if len(encoded_provenance) > 4096:
            raise InputError("standalone_html generation_provenance_json exceeds 4096 bytes")
        try:
            provenance = json.loads(provenance_json)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise InputError("standalone_html generation_provenance_json must be valid JSON") from exc
        if not isinstance(provenance, dict) or not provenance:
            raise InputError("standalone_html generation_provenance_json must be a nonempty object")

    @classmethod
    def presentation_row_invariant_holds(cls, row: PresentationRow) -> bool:
        """Check the persisted-row invariant without exposing invalid row data."""
        try:
            cls._validate_presentation_candidate(vars(row))
        except InputError:
            return False
        return True

    @staticmethod
    def _fetch_presentation_by_id(
        conn: sqlite3.Connection, presentation_id: str, include_deleted: bool
    ) -> PresentationRow:
        query = (
            f"SELECT {_PRESENTATION_DETAIL_PROJECTION} "  # nosec B608
            "FROM presentations WHERE id = ?"
        )
        params: list[Any] = [presentation_id]
        if not include_deleted:
            query += " AND deleted = 0"
        row = conn.execute(query, tuple(params)).fetchone()
        if not row:
            raise KeyError("presentation_not_found")
        return PresentationRow(**dict(row))

    @staticmethod
    def _is_fts_query_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "fts5: syntax error",
                "fts5 syntax error",
                "malformed match",
                "no such column:",
                "unterminated string",
                "syntax error near",
            )
        )

    @staticmethod
    def _normalize_content_kinds(
        accepted_content_kinds: Iterable[str] | None,
    ) -> tuple[str, ...]:
        if accepted_content_kinds is None:
            return ("structured_slides", "standalone_html")
        requested = set(accepted_content_kinds)
        kinds = tuple(kind for kind in ("structured_slides", "standalone_html") if kind in requested)
        if not kinds:
            raise InputError("accepted_content_kinds must include a supported kind")
        return kinds

    @staticmethod
    def _build_version_payload(row: PresentationRow) -> dict[str, Any]:
        common = {
            "snapshot_schema_version": _SNAPSHOT_SCHEMA_VERSION,
            "content_kind": row.content_kind,
            "id": row.id,
            "title": row.title,
            "description": row.description,
            "source_type": row.source_type,
            "source_ref": row.source_ref,
            "source_query": row.source_query,
            "created_at": row.created_at,
            "last_modified": row.last_modified,
            "deleted": int(row.deleted or 0),
            "client_id": row.client_id,
            "version": int(row.version),
        }

        if row.content_kind == "standalone_html":
            return {
                **common,
                "html_document": row.html_document,
                "html_sha256": row.html_sha256,
                "html_bytes": row.html_bytes,
                "html_slide_count": row.html_slide_count,
                "generation_job_uuid": row.generation_job_uuid,
                "generation_provenance_json": row.generation_provenance_json,
            }
        return {
            **common,
            "theme": row.theme,
            "marp_theme": row.marp_theme,
            "template_id": row.template_id,
            "visual_style_id": row.visual_style_id,
            "visual_style_scope": row.visual_style_scope,
            "visual_style_name": row.visual_style_name,
            "visual_style_version": row.visual_style_version,
            "visual_style_snapshot": row.visual_style_snapshot,
            "settings": row.settings,
            "studio_data": row.studio_data,
            "slides": row.slides,
            "slides_text": row.slides_text,
            "custom_css": row.custom_css,
        }

    def _insert_version_snapshot(self, conn: sqlite3.Connection, row: PresentationRow) -> None:
        payload_json = json.dumps(
            self._build_version_payload(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if (
            row.content_kind == "standalone_html"
            and len(payload_json.encode("utf-8")) > _STANDALONE_HTML_SNAPSHOT_MAX_BYTES
        ):
            raise InputError("standalone_html_storage_limit")
        conn.execute(
            """
            INSERT INTO presentations_versions (
                presentation_id, version, payload_json, created_at, client_id,
                title, deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.id,
                int(row.version),
                payload_json,
                row.last_modified,
                row.client_id,
                row.title,
                int(row.deleted or 0),
            ),
        )
        if row.content_kind == "standalone_html":
            conn.execute(
                """
                DELETE FROM presentations_versions
                WHERE presentation_id = ?
                  AND id NOT IN (
                    SELECT id FROM presentations_versions
                    WHERE presentation_id = ?
                    ORDER BY version DESC
                    LIMIT ?
                  )
                """,
                (
                    row.id,
                    row.id,
                    self.standalone_html_version_retention,
                ),
            )

    def _insert_presentation_in_connection(
        self,
        conn: sqlite3.Connection,
        candidate: Mapping[str, Any],
    ) -> PresentationRow:
        """Insert one validated presentation and its transactional side effects."""
        self._validate_presentation_candidate(candidate)
        conn.execute(
            """
            INSERT INTO presentations (
                id, title, description, theme, marp_theme, template_id,
                visual_style_id, visual_style_scope, visual_style_name,
                visual_style_version, visual_style_snapshot, settings,
                studio_data, slides, slides_text, source_type, source_ref,
                source_query, custom_css, created_at, last_modified, deleted,
                client_id, version, content_kind, html_document, html_sha256,
                html_bytes, html_slide_count, generation_job_uuid,
                generation_provenance_json
            ) VALUES (
                :id, :title, :description, :theme, :marp_theme, :template_id,
                :visual_style_id, :visual_style_scope, :visual_style_name,
                :visual_style_version, :visual_style_snapshot, :settings,
                :studio_data, :slides, :slides_text, :source_type, :source_ref,
                :source_query, :custom_css, :created_at, :last_modified, 0,
                :client_id, 1, :content_kind, :html_document, :html_sha256,
                :html_bytes, :html_slide_count, :generation_job_uuid,
                :generation_provenance_json
            )
            """,
            candidate,
        )
        presentation_id = str(candidate["id"])
        row = self._fetch_presentation_by_id(
            conn,
            presentation_id,
            include_deleted=True,
        )
        self._insert_version_snapshot(conn, row)
        self._insert_sync_log(
            conn,
            entity_uuid=presentation_id,
            operation="create",
            version=1,
            payload={"title": row.title, "theme": row.theme},
        )
        return row

    @staticmethod
    def _normalize_visual_style_payload(style_payload: str) -> str:
        if not style_payload:
            raise InputError("style_payload is required")
        try:
            parsed = json.loads(style_payload)
        except json.JSONDecodeError as exc:
            raise InputError("style_payload must be valid JSON") from exc
        return json.dumps(parsed, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def _validate_visual_style_scope(scope: str) -> str:
        if scope not in {"builtin", "user"}:
            raise InputError("scope must be one of: builtin, user")
        return scope

    @staticmethod
    def _fetch_visual_style_by_id(conn: sqlite3.Connection, style_id: str) -> VisualStyleRow:
        row = conn.execute("SELECT * FROM visual_styles WHERE id = ?", (style_id,)).fetchone()
        if not row:
            raise KeyError("visual_style_not_found")
        return VisualStyleRow(**dict(row))

    def create_visual_style(
        self,
        *,
        name: str,
        scope: str,
        style_payload: str,
        style_id: str | None = None,
    ) -> VisualStyleRow:
        """Create and persist a visual style for the current user."""
        if not name:
            raise InputError("name is required")
        resolved_scope = self._validate_visual_style_scope(scope)
        normalized_payload = self._normalize_visual_style_payload(style_payload)
        resolved_style_id = style_id or str(uuid.uuid4())
        now = self._utcnow_iso()
        try:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO visual_styles (id, name, scope, style_payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_style_id,
                        name,
                        resolved_scope,
                        normalized_payload,
                        now,
                        now,
                    ),
                )
                return self._fetch_visual_style_by_id(conn, resolved_style_id)
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper() or "PRIMARY" in str(exc).upper():
                raise ConflictError(
                    "visual style already exists",
                    entity="visual_styles",
                    identifier=resolved_style_id,
                ) from exc
            raise SlidesDatabaseError(f"Failed to create visual style: {exc}") from exc

    def get_visual_style_by_id(self, style_id: str) -> VisualStyleRow:
        """Fetch a single visual style by identifier."""
        conn = self.get_connection()
        return self._fetch_visual_style_by_id(conn, style_id)

    def count_visual_styles(self) -> int:
        """Return the number of persisted user visual styles."""

        conn = self.get_connection()
        count_row = conn.execute("SELECT COUNT(*) AS cnt FROM visual_styles").fetchone()
        return int(count_row["cnt"]) if count_row else 0

    def list_visual_styles(self, *, limit: int, offset: int) -> tuple[list[VisualStyleRow], int]:
        """List persisted user visual styles with pagination metadata."""
        if limit < 1:
            raise InputError("limit must be >= 1")
        if offset < 0:
            raise InputError("offset must be >= 0")
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT * FROM visual_styles
            ORDER BY updated_at DESC, created_at DESC, name ASC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        count_row = conn.execute("SELECT COUNT(*) AS cnt FROM visual_styles").fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [VisualStyleRow(**dict(row)) for row in rows], total

    def update_visual_style(
        self,
        *,
        style_id: str,
        name: str,
        style_payload: str,
        expected_updated_at: str,
    ) -> VisualStyleRow:
        """Update a stored visual style and return the refreshed row."""
        if not name:
            raise InputError("name is required")
        normalized_payload = self._normalize_visual_style_payload(style_payload)
        if not expected_updated_at:
            raise InputError("expected_updated_at is required")
        with self.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE visual_styles
                SET name = ?, style_payload = ?, updated_at = ?
                WHERE id = ? AND updated_at = ?
                """,
                (name, normalized_payload, self._utcnow_iso(), style_id, expected_updated_at),
            )
            if cur.rowcount == 0:
                current_row = conn.execute(
                    "SELECT updated_at FROM visual_styles WHERE id = ?",
                    (style_id,),
                ).fetchone()
                if not current_row:
                    raise KeyError("visual_style_not_found")
                raise ConflictError(
                    "visual style update conflicted with a newer revision",
                    entity="visual_styles",
                    identifier=style_id,
                )
            return self._fetch_visual_style_by_id(conn, style_id)

    def delete_visual_style(self, style_id: str) -> bool:
        """Delete a stored visual style by identifier."""
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT 1 FROM visual_styles WHERE id = ?",
                (style_id,),
            ).fetchone()
            if not existing:
                return False
            in_use = conn.execute(
                """
                SELECT 1
                FROM presentations
                WHERE visual_style_id = ?
                  AND deleted = 0
                LIMIT 1
                """,
                (style_id,),
            ).fetchone()
            if in_use:
                raise ConflictError(
                    "visual style is still referenced by presentations",
                    entity="visual_styles",
                    identifier=style_id,
                )
            cur = conn.execute("DELETE FROM visual_styles WHERE id = ?", (style_id,))
            return cur.rowcount > 0

    def create_presentation(
        self,
        *,
        presentation_id: str | None,
        title: str,
        description: str | None,
        theme: str,
        marp_theme: str | None,
        settings: str | None,
        studio_data: str | None,
        template_id: str | None = None,
        visual_style_id: str | None = None,
        visual_style_scope: str | None = None,
        visual_style_name: str | None = None,
        visual_style_version: int | None = None,
        visual_style_snapshot: str | None = None,
        slides: str,
        slides_text: str,
        source_type: str | None,
        source_ref: str | None,
        source_query: str | None,
        custom_css: str | None,
        content_kind: str = "structured_slides",
        html_document: str | None = None,
        html_sha256: str | None = None,
        html_bytes: int | None = None,
        html_slide_count: int | None = None,
        generation_job_uuid: str | None = None,
        generation_provenance_json: str | None = None,
    ) -> PresentationRow:
        if not title:
            raise InputError("title is required")
        pres_id = presentation_id or str(uuid.uuid4())
        now = self._utcnow_iso()
        candidate = {
            "id": pres_id,
            "title": title,
            "description": description,
            "theme": theme,
            "marp_theme": marp_theme,
            "template_id": template_id,
            "visual_style_id": visual_style_id,
            "visual_style_scope": visual_style_scope,
            "visual_style_name": visual_style_name,
            "visual_style_version": visual_style_version,
            "visual_style_snapshot": visual_style_snapshot,
            "settings": settings,
            "studio_data": studio_data,
            "slides": slides,
            "slides_text": slides_text,
            "source_type": source_type,
            "source_ref": source_ref,
            "source_query": source_query,
            "custom_css": custom_css,
            "created_at": now,
            "last_modified": now,
            "client_id": self.client_id,
            "content_kind": content_kind,
            "html_document": html_document,
            "html_sha256": html_sha256,
            "html_bytes": html_bytes,
            "html_slide_count": html_slide_count,
            "generation_job_uuid": generation_job_uuid,
            "generation_provenance_json": generation_provenance_json,
        }
        try:
            with self.transaction(immediate=True) as conn:
                row = self._insert_presentation_in_connection(conn, candidate)
            return row
        except sqlite3.IntegrityError as exc:
            message = str(exc).lower()
            if "presentations.generation_job_uuid" in message:
                raise ConflictError(
                    "generation_job_uuid_conflict",
                    entity="presentations",
                    identifier=generation_job_uuid,
                ) from exc
            if "UNIQUE" in str(exc).upper() or "PRIMARY" in str(exc).upper():
                raise ConflictError("presentation already exists", entity="presentations", identifier=pres_id) from exc
            raise SlidesDatabaseError(f"Failed to create presentation: {exc}") from exc

    def get_presentation_by_id(self, presentation_id: str, *, include_deleted: bool = False) -> PresentationRow:
        conn = self.get_connection()
        return self._fetch_presentation_by_id(conn, presentation_id, include_deleted)

    def probe_health(self) -> None:
        """Verify that the presentation table is readable without loading content."""
        failed = False
        try:
            self.get_connection().execute("SELECT 1 FROM presentations LIMIT 1").fetchone()
        except sqlite3.Error:
            failed = True
        if failed:
            raise SlidesDatabaseError("slides_health_probe_failed")

    def get_presentation_kind(
        self,
        presentation_id: str,
        *,
        include_deleted: bool = False,
    ) -> PresentationKindRow:
        """Fetch only fields needed to guard an operation by content kind."""
        query = """
            SELECT id, content_kind, version, deleted, last_modified
            FROM presentations
            WHERE id = ?
        """
        if not include_deleted:
            query += " AND deleted = 0"
        row = self.get_connection().execute(query, (presentation_id,)).fetchone()
        if not row:
            raise KeyError("presentation_not_found")
        return PresentationKindRow(**dict(row))

    def get_presentation_source_identity(
        self,
        presentation_id: str,
        *,
        include_deleted: bool = False,
    ) -> PresentationSourceIdentityRow:
        """Fetch source-free identity metadata for save reconciliation."""
        query = """
            SELECT id, title, content_kind, version, deleted, last_modified,
                   html_sha256, html_bytes
            FROM presentations
            WHERE id = ?
        """
        if not include_deleted:
            query += " AND deleted = 0"
        row = self.get_connection().execute(query, (presentation_id,)).fetchone()
        if not row:
            raise KeyError("presentation_not_found")
        return PresentationSourceIdentityRow(**dict(row))

    def get_presentation_summary(
        self,
        presentation_id: str,
        *,
        include_deleted: bool = False,
    ) -> PresentationSummaryRow:
        """Fetch one source-free presentation metadata projection by owner-local ID."""
        query = f"""
            SELECT {_PRESENTATION_SUMMARY_PROJECTION}
            FROM presentations
            WHERE id = ?
        """  # nosec B608 - projection is a module constant
        if not include_deleted:
            query += " AND deleted = 0"
        row = self.get_connection().execute(query, (presentation_id,)).fetchone()
        if not row:
            raise KeyError("presentation_not_found")
        return PresentationSummaryRow(**dict(row))

    def list_presentations(
        self,
        *,
        limit: int,
        offset: int,
        include_deleted: bool,
        sort_column: str,
        sort_direction: str,
    ) -> tuple[list[PresentationRow], int]:
        if limit < 1:
            raise InputError("limit must be >= 1")
        allowed_columns = {
            "created_at": "created_at",
            "last_modified": "last_modified",
            "title": "title",
        }
        safe_column = allowed_columns.get(sort_column, "created_at")
        safe_direction = "DESC" if sort_direction.upper() == "DESC" else "ASC"
        where = "" if include_deleted else "WHERE deleted = 0"
        query_template = (
            f"SELECT {_PRESENTATION_DETAIL_PROJECTION} FROM presentations "  # nosec B608
            "{where} ORDER BY {safe_column} {safe_direction} LIMIT ? OFFSET ?"
        )
        query = query_template.format_map(locals())  # nosec B608
        count_query_template = "SELECT COUNT(*) AS cnt FROM presentations {where}"
        count_query = count_query_template.format_map(locals())  # nosec B608
        conn = self.get_connection()
        rows = conn.execute(query, (limit, offset)).fetchall()
        count_row = conn.execute(count_query).fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationRow(**dict(row)) for row in rows], total

    def list_presentation_summaries(
        self,
        *,
        limit: int,
        offset: int,
        include_deleted: bool,
        sort_column: str,
        sort_direction: str,
        accepted_content_kinds: Iterable[str] | None = None,
    ) -> tuple[list[PresentationSummaryRow], int]:
        """List source-free presentation summaries."""
        if limit < 1:
            raise InputError("limit must be >= 1")
        allowed_columns = {
            "created_at": "created_at",
            "last_modified": "last_modified",
            "title": "title",
        }
        safe_column = allowed_columns.get(sort_column, "created_at")
        safe_direction = "DESC" if sort_direction.upper() == "DESC" else "ASC"
        kinds = self._normalize_content_kinds(accepted_content_kinds)
        placeholders = ", ".join("?" for _ in kinds)
        clauses = [f"content_kind IN ({placeholders})"]
        if not include_deleted:
            clauses.append("deleted = 0")
        where = "WHERE " + " AND ".join(clauses)
        query_template = (
            f"SELECT {_PRESENTATION_SUMMARY_PROJECTION} FROM presentations "  # nosec B608
            "{where} ORDER BY {safe_column} {safe_direction} LIMIT ? OFFSET ?"
        )
        query = query_template.format_map(locals())  # nosec B608
        count_query_template = "SELECT COUNT(*) AS cnt FROM presentations {where}"
        count_query = count_query_template.format_map(locals())  # nosec B608
        conn = self.get_connection()
        rows = conn.execute(query, (*kinds, limit, offset)).fetchall()
        count_row = conn.execute(count_query, kinds).fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationSummaryRow(**dict(row)) for row in rows], total

    def search_presentations(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        include_deleted: bool,
    ) -> tuple[list[PresentationRow], int]:
        if not query:
            raise InputError("query is required")
        if limit < 1:
            raise InputError("limit must be >= 1")
        where = "" if include_deleted else "AND p.deleted = 0"
        search_sql_template = (
            f"SELECT {_PRESENTATION_DETAIL_PROJECTION_QUALIFIED} FROM presentations p "  # nosec B608
            "JOIN presentations_fts fts ON p.rowid = fts.rowid "
            "WHERE presentations_fts MATCH ? "
            "{where} "
            "ORDER BY p.last_modified DESC LIMIT ? OFFSET ?"
        )
        sql = search_sql_template.format_map(locals())  # nosec B608
        count_sql_template = (
            "SELECT COUNT(*) AS cnt FROM presentations p "
            "JOIN presentations_fts fts ON p.rowid = fts.rowid "
            "WHERE presentations_fts MATCH ? "
            "{where}"
        )
        count_sql = count_sql_template.format_map(locals())  # nosec B608
        conn = self.get_connection()
        try:
            rows = conn.execute(sql, (query, limit, offset)).fetchall()
            count_row = conn.execute(count_sql, (query,)).fetchone()
        except sqlite3.OperationalError as exc:
            if not self._is_fts_query_error(exc):
                raise
            raise InputError("search query is invalid") from exc
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationRow(**dict(row)) for row in rows], total

    def search_presentation_summaries(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        include_deleted: bool,
        accepted_content_kinds: Iterable[str] | None = None,
    ) -> tuple[list[PresentationSummaryRow], int]:
        """Search presentations without selecting standalone source."""
        if not query:
            raise InputError("query is required")
        if limit < 1:
            raise InputError("limit must be >= 1")
        kinds = self._normalize_content_kinds(accepted_content_kinds)
        placeholders = ", ".join("?" for _ in kinds)
        clauses = [f"p.content_kind IN ({placeholders})"]
        if not include_deleted:
            clauses.append("p.deleted = 0")
        where = "AND " + " AND ".join(clauses)
        search_sql_template = (
            f"SELECT {_PRESENTATION_SUMMARY_PROJECTION_QUALIFIED} "  # nosec B608
            "FROM presentations p "
            "JOIN presentations_fts fts ON p.rowid = fts.rowid "
            "WHERE presentations_fts MATCH ? "
            "{where} "
            "ORDER BY p.last_modified DESC LIMIT ? OFFSET ?"
        )
        sql = search_sql_template.format_map(locals())  # nosec B608
        count_sql_template = (
            "SELECT COUNT(*) AS cnt FROM presentations p "
            "JOIN presentations_fts fts ON p.rowid = fts.rowid "
            "WHERE presentations_fts MATCH ? "
            "{where}"
        )
        count_sql = count_sql_template.format_map(locals())  # nosec B608
        conn = self.get_connection()
        try:
            rows = conn.execute(sql, (query, *kinds, limit, offset)).fetchall()
            count_row = conn.execute(count_sql, (query, *kinds)).fetchone()
        except sqlite3.OperationalError as exc:
            if not self._is_fts_query_error(exc):
                raise
            raise InputError("search query is invalid") from exc
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationSummaryRow(**dict(row)) for row in rows], total

    def update_presentation(
        self,
        *,
        presentation_id: str,
        update_fields: dict[str, Any],
        expected_version: int,
        operation: str = "update",
    ) -> PresentationRow:
        if not update_fields:
            raise InputError("update_fields is required")
        allowed = {
            "title",
            "description",
            "theme",
            "marp_theme",
            "template_id",
            "visual_style_id",
            "visual_style_scope",
            "visual_style_name",
            "visual_style_version",
            "visual_style_snapshot",
            "settings",
            "studio_data",
            "slides",
            "slides_text",
            "source_type",
            "source_ref",
            "source_query",
            "custom_css",
            "deleted",
            "content_kind",
            "html_document",
            "html_sha256",
            "html_bytes",
            "html_slide_count",
            "generation_job_uuid",
            "generation_provenance_json",
        }
        valid_updates = {key: value for key, value in update_fields.items() if key in allowed}
        if not valid_updates:
            raise InputError("no valid fields to update")
        sets: list[str] = []
        params: list[Any] = []
        for key, value in valid_updates.items():
            sets.append(f"{key} = ?")
            params.append(value)
        next_version = expected_version + 1
        sets.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([self._utcnow_iso(), next_version, self.client_id, presentation_id, expected_version])
        # Column names come from the allowlist above; values are always bound params.
        set_clause_sql = ", ".join(sets)
        update_sql_template = "UPDATE presentations SET {set_clause_sql} WHERE id = ? AND version = ?"
        sql = update_sql_template.format_map(locals())  # nosec B608
        with self.transaction(immediate=True) as conn:
            current = self._fetch_presentation_by_id(
                conn,
                presentation_id,
                include_deleted=True,
            )
            if current.version != expected_version:
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            immutable_errors = {
                "content_kind": "content_kind_immutable",
                "generation_job_uuid": "generation_job_uuid_immutable",
                "generation_provenance_json": "generation_provenance_immutable",
            }
            for field, error_code in immutable_errors.items():
                if field in valid_updates and valid_updates[field] != getattr(current, field):
                    raise InputError(error_code)
            if current.content_kind == "standalone_html":
                delete_transition = operation == "delete" and current.deleted == 0 and valid_updates == {"deleted": 1}
                restore_transition = operation == "restore" and current.deleted == 1 and valid_updates == {"deleted": 0}
                if not delete_transition and not restore_transition:
                    raise InputError("operation_not_supported_for_content_kind")
            candidate = vars(current).copy()
            candidate.update(valid_updates)
            self._validate_presentation_candidate(candidate)
            cur = conn.execute(sql, tuple(params))
            if cur.rowcount == 0:
                existing = conn.execute("SELECT version FROM presentations WHERE id = ?", (presentation_id,)).fetchone()
                if not existing:
                    raise KeyError("presentation_not_found")
                raise ConflictError("version_conflict", entity="presentations", identifier=presentation_id)
            row = self._fetch_presentation_by_id(conn, presentation_id, include_deleted=True)
            self._insert_version_snapshot(conn, row)
            self._insert_sync_log(
                conn,
                entity_uuid=presentation_id,
                operation=operation,
                version=next_version,
                payload={"fields": list(valid_updates.keys())},
            )
        return row

    def save_standalone_html_source(
        self,
        *,
        presentation_id: str,
        html_document: str | bytes,
        validation_result: StandaloneHtmlValidationResult,
        expected_version: int,
    ) -> PresentationRow:
        """Atomically store one source already accepted by the shared pool."""
        source = bind_validated_standalone_source(html_document, validation_result)
        with self.transaction(immediate=True) as conn:
            current = self._fetch_presentation_by_id(conn, presentation_id, include_deleted=False)
            if current.content_kind != "standalone_html":
                raise InputError("operation_not_supported_for_content_kind")
            if current.version != expected_version:
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            derived = validation_result
            if (
                current.html_document,
                current.title,
                current.html_sha256,
                current.html_bytes,
                current.html_slide_count,
                current.slides_text,
            ) == (
                source,
                derived.title,
                derived.html_sha256,
                derived.html_bytes,
                derived.slide_count,
                derived.indexable_text,
            ):
                return current

            next_version = expected_version + 1
            modified = self._utcnow_iso()
            candidate = vars(current).copy()
            candidate.update(
                {
                    "title": derived.title,
                    "html_document": source,
                    "html_sha256": derived.html_sha256,
                    "html_bytes": derived.html_bytes,
                    "html_slide_count": derived.slide_count,
                    "slides_text": derived.indexable_text,
                    "last_modified": modified,
                    "version": next_version,
                }
            )
            self._validate_presentation_candidate(candidate)
            cur = conn.execute(
                """
                UPDATE presentations
                SET title = ?, html_document = ?, html_sha256 = ?,
                    html_bytes = ?, html_slide_count = ?, slides_text = ?,
                    last_modified = ?, version = ?
                WHERE id = ? AND version = ?
                """,
                (
                    derived.title,
                    source,
                    derived.html_sha256,
                    derived.html_bytes,
                    derived.slide_count,
                    derived.indexable_text,
                    modified,
                    next_version,
                    presentation_id,
                    expected_version,
                ),
            )
            if cur.rowcount != 1:  # pragma: no cover - held write transaction
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            row = self._fetch_presentation_by_id(conn, presentation_id, include_deleted=True)
            self._insert_version_snapshot(conn, row)
            self._insert_sync_log(
                conn,
                entity_uuid=presentation_id,
                operation="update",
                version=next_version,
                payload={"fields": ["html_document"]},
            )
            return row

    def restore_standalone_html_version(
        self,
        *,
        presentation_id: str,
        version: int,
        expected_version: int,
        html_document: str,
        validation_result: StandaloneHtmlValidationResult,
        expected_payload_json: str,
    ) -> PresentationRow:
        """Restore an exact snapshot already accepted by the shared pool."""
        source = bind_validated_standalone_source(html_document, validation_result)
        with self.transaction(immediate=True) as conn:
            current = self._fetch_presentation_by_id(conn, presentation_id, include_deleted=True)
            if current.content_kind != "standalone_html":
                raise InputError("operation_not_supported_for_content_kind")
            if current.version != expected_version:
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            version_row = conn.execute(
                """
                SELECT payload_json
                FROM presentations_versions
                WHERE presentation_id = ? AND version = ?
                """,
                (presentation_id, version),
            ).fetchone()
            if not version_row:
                raise KeyError("presentation_version_not_found")
            if version_row["payload_json"] != expected_payload_json:
                raise InputError("version_payload_invalid")
            payload = decode_presentation_version_payload(version_row["payload_json"])
            if payload.get("content_kind") != "standalone_html":
                raise InputError("version_content_kind_mismatch")
            if payload.get("generation_job_uuid", current.generation_job_uuid) != current.generation_job_uuid:
                raise InputError("generation_job_uuid_immutable")
            if payload.get("generation_provenance_json") != current.generation_provenance_json:
                raise InputError("generation_provenance_immutable")
            if payload.get("html_document") != source:
                raise InputError("version_payload_invalid")
            derived = validation_result
            expected_metadata = (
                payload.get("html_sha256"),
                payload.get("html_bytes"),
                payload.get("html_slide_count"),
                payload.get("title"),
            )
            actual_metadata = (
                derived.html_sha256,
                derived.html_bytes,
                derived.slide_count,
                derived.title,
            )
            if expected_metadata != actual_metadata:
                raise InputError("version_payload_invalid")
            updates = {
                "title": derived.title,
                "description": payload.get("description"),
                "html_document": source,
                "html_sha256": derived.html_sha256,
                "html_bytes": derived.html_bytes,
                "html_slide_count": derived.slide_count,
                "slides_text": derived.indexable_text,
                "source_type": payload.get("source_type"),
                "source_ref": payload.get("source_ref"),
                "source_query": payload.get("source_query"),
                "deleted": 0,
            }

            next_version = expected_version + 1
            modified = self._utcnow_iso()
            candidate = vars(current).copy()
            candidate.update(updates)
            candidate.update(
                {
                    "last_modified": modified,
                    "version": next_version,
                }
            )
            self._validate_presentation_candidate(candidate)
            set_clause = ", ".join(f"{field} = ?" for field in updates)
            sql = (
                f"UPDATE presentations SET {set_clause}, last_modified = ?, "  # nosec B608
                "version = ? WHERE id = ? AND version = ?"
            )
            cur = conn.execute(
                sql,
                (
                    *updates.values(),
                    modified,
                    next_version,
                    presentation_id,
                    expected_version,
                ),
            )
            if cur.rowcount != 1:  # pragma: no cover - held write transaction
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            row = self._fetch_presentation_by_id(conn, presentation_id, include_deleted=True)
            self._insert_version_snapshot(conn, row)
            self._insert_sync_log(
                conn,
                entity_uuid=presentation_id,
                operation="restore",
                version=next_version,
                payload={"version": version},
            )
            return row

    def restore_validated_standalone_presentation(
        self,
        *,
        presentation_id: str,
        html_document: str,
        validation_result: StandaloneHtmlValidationResult,
        expected_version: int,
    ) -> PresentationRow:
        """Atomically undelete one exact standalone source accepted by the pool."""
        source = bind_validated_standalone_source(html_document, validation_result)
        with self.transaction(immediate=True) as conn:
            current = self._fetch_presentation_by_id(
                conn,
                presentation_id,
                include_deleted=True,
            )
            if current.content_kind != "standalone_html":
                raise InputError("operation_not_supported_for_content_kind")
            if current.version != expected_version:
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            if current.deleted != 1:
                raise InputError("operation_not_supported_for_content_kind")
            derived = validation_result
            if (
                current.html_document,
                current.title,
                current.html_sha256,
                current.html_bytes,
                current.html_slide_count,
                current.slides_text,
            ) != (
                source,
                derived.title,
                derived.html_sha256,
                derived.html_bytes,
                derived.slide_count,
                derived.indexable_text,
            ):
                raise InputError("standalone_html_response_invalid")

            next_version = expected_version + 1
            modified = self._utcnow_iso()
            candidate = vars(current).copy()
            candidate.update(
                {
                    "deleted": 0,
                    "last_modified": modified,
                    "version": next_version,
                    "client_id": self.client_id,
                }
            )
            self._validate_presentation_candidate(candidate)
            cur = conn.execute(
                """
                UPDATE presentations
                SET deleted = 0, last_modified = ?, version = ?, client_id = ?
                WHERE id = ? AND version = ? AND deleted = 1
                """,
                (
                    modified,
                    next_version,
                    self.client_id,
                    presentation_id,
                    expected_version,
                ),
            )
            if cur.rowcount != 1:  # pragma: no cover - held write transaction
                raise ConflictError(
                    "version_conflict",
                    entity="presentations",
                    identifier=presentation_id,
                )
            row = self._fetch_presentation_by_id(
                conn,
                presentation_id,
                include_deleted=True,
            )
            self._insert_version_snapshot(conn, row)
            self._insert_sync_log(
                conn,
                entity_uuid=presentation_id,
                operation="restore",
                version=next_version,
                payload={"fields": ["deleted"]},
            )
            return row

    def list_presentation_versions(
        self,
        *,
        presentation_id: str,
        limit: int,
        offset: int,
    ) -> tuple[list[PresentationVersionRow], int]:
        if limit < 1:
            raise InputError("limit must be >= 1")
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT presentation_id, version, payload_json, created_at, client_id
            FROM presentations_versions
            WHERE presentation_id = ?
            ORDER BY version DESC
            LIMIT ? OFFSET ?
            """,
            (presentation_id, limit, offset),
        ).fetchall()
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM presentations_versions WHERE presentation_id = ?",
            (presentation_id,),
        ).fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationVersionRow(**dict(row)) for row in rows], total

    def list_presentation_version_metadata(
        self,
        *,
        presentation_id: str,
        limit: int,
        offset: int,
    ) -> tuple[list[PresentationVersionMetadataRow], int]:
        """List version metadata without selecting snapshot payloads."""
        if limit < 1:
            raise InputError("limit must be >= 1")
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT presentation_id, version, created_at, client_id, title, deleted
            FROM presentations_versions
            WHERE presentation_id = ?
            ORDER BY version DESC
            LIMIT ? OFFSET ?
            """,
            (presentation_id, limit, offset),
        ).fetchall()
        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM presentations_versions WHERE presentation_id = ?",
            (presentation_id,),
        ).fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        return [PresentationVersionMetadataRow(**dict(row)) for row in rows], total

    def get_presentation_version(
        self,
        *,
        presentation_id: str,
        version: int,
    ) -> PresentationVersionRow:
        conn = self.get_connection()
        row = conn.execute(
            """
            SELECT presentation_id, version, payload_json, created_at, client_id
            FROM presentations_versions
            WHERE presentation_id = ? AND version = ?
            """,
            (presentation_id, version),
        ).fetchone()
        if not row:
            raise KeyError("presentation_version_not_found")
        return PresentationVersionRow(**dict(row))

    @staticmethod
    def _generation_receipt_from_connection(
        conn: sqlite3.Connection,
        receipt_id: str,
        owner_user_id: str,
    ) -> SlidesGenerationReceiptRow:
        row = conn.execute(
            f"SELECT {_RECEIPT_PROJECTION} FROM slides_generation_receipts "  # nosec B608
            "WHERE id = ? AND owner_user_id = ?",
            (receipt_id, owner_user_id),
        ).fetchone()
        if not row:
            raise KeyError("slides_generation_receipt_not_found")
        return SlidesGenerationReceiptRow(**dict(row))

    @staticmethod
    def _generation_input_from_connection(
        conn: sqlite3.Connection,
        receipt_id: str,
        owner_user_id: str,
    ) -> SlidesGenerationInputRow:
        row = conn.execute(
            f"SELECT {_INPUT_PROJECTION} FROM slides_generation_inputs i "  # nosec B608
            "WHERE i.receipt_id = ? AND EXISTS ("
            "SELECT 1 FROM slides_generation_receipts r "
            "WHERE r.id = i.receipt_id AND r.owner_user_id = ?)",
            (receipt_id, owner_user_id),
        ).fetchone()
        if not row:
            raise KeyError("slides_generation_input_not_found")
        return SlidesGenerationInputRow(**dict(row))

    def list_generation_receipts_for_reconciliation(
        self,
        *,
        owner_user_id: str,
        after_receipt_id: str | None,
        limit: int,
    ) -> list[SlidesGenerationReconciliationRow]:
        """List one deterministic, source-free physical-database receipt page."""
        if not isinstance(owner_user_id, str) or not owner_user_id:
            raise InputError("owner_user_id is required")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _GENERATION_RECONCILIATION_BATCH_MAX
        ):
            raise InputError("reconciliation limit must be between 1 and 500")
        after_clause = "" if after_receipt_id is None else "AND r.id > ?"
        parameters: tuple[Any, ...] = (limit,) if after_receipt_id is None else (after_receipt_id, limit)
        rows = (
            self.get_connection()
            .execute(
                f"SELECT {_GENERATION_RECONCILIATION_PROJECTION} "  # nosec B608
                "FROM slides_generation_receipts r "
                "LEFT JOIN slides_generation_inputs i ON i.receipt_id = r.id "
                "LEFT JOIN presentations p ON p.id = r.presentation_id "
                f"WHERE 1 = 1 {after_clause} "  # nosec B608
                "ORDER BY r.id ASC LIMIT ?",
                parameters,
            )
            .fetchall()
        )
        result: list[SlidesGenerationReconciliationRow] = []
        for row in rows:
            values = dict(row)
            values["input_exists"] = bool(values["input_exists"])
            values["presentation_exists"] = bool(values["presentation_exists"])
            result.append(SlidesGenerationReconciliationRow(**values))
        return result

    def repair_generation_receipt_job(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        expected_job_uuid: str | None,
        job_id: int | None,
        job_uuid: str,
        receipt_status: str,
        updated_at: str,
    ) -> bool:
        """CAS one nonterminal receipt to authoritative Jobs binding/state."""
        if receipt_status not in {"queued", "running"}:
            raise InputError("generation receipt status must be queued or running")
        if not isinstance(job_uuid, str) or not job_uuid:
            raise InputError("job_uuid is required")
        if expected_job_uuid is not None and expected_job_uuid != job_uuid:
            raise InputError("generation repair cannot replace a Jobs UUID")
        if job_id is not None and (isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0):
            raise InputError("job_id must be a positive integer or null")
        self._canonical_utc_timestamp(updated_at)
        job_uuid_clause, job_uuid_parameters = self._generation_job_uuid_cas(expected_job_uuid)
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET "
                "job_id = COALESCE(?, job_id), job_uuid = ?, receipt_status = ?, "
                "error_message = CASE WHEN error_code = ? THEN NULL ELSE error_message END, "
                "error_code = CASE WHEN error_code = ? THEN NULL ELSE error_code END, "
                "updated_at = ? WHERE id = ? AND owner_user_id = ? "
                "AND receipt_status IN ('claimed', 'queued', 'running') "
                f"AND {job_uuid_clause}",  # nosec B608
                (
                    job_id,
                    job_uuid,
                    receipt_status,
                    _GENERATION_RECEIPT_MISSING_CODE,
                    _GENERATION_RECEIPT_MISSING_CODE,
                    updated_at,
                    receipt_id,
                    owner_user_id,
                    *job_uuid_parameters,
                ),
            )
            return cursor.rowcount == 1

    def mark_generation_receipt_job_missing(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        expected_job_uuid: str | None,
        observed_at: str,
    ) -> str | None:
        """Persist and return the first authoritative Jobs-miss timestamp."""
        self._canonical_utc_timestamp(observed_at)
        job_uuid_clause, job_uuid_parameters = self._generation_job_uuid_cas(expected_job_uuid)
        with self.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT error_code, updated_at FROM slides_generation_receipts "
                "WHERE id = ? AND owner_user_id = ? "
                "AND receipt_status IN ('claimed', 'queued', 'running') "
                f"AND {job_uuid_clause}",  # nosec B608
                (receipt_id, owner_user_id, *job_uuid_parameters),
            ).fetchone()
            if row is None:
                return None
            if row["error_code"] == _GENERATION_RECEIPT_MISSING_CODE:
                return str(row["updated_at"])
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET error_code = ?, "
                "error_message = NULL, updated_at = ? "
                "WHERE id = ? AND owner_user_id = ? "
                "AND receipt_status IN ('claimed', 'queued', 'running') "
                f"AND {job_uuid_clause}",  # nosec B608
                (
                    _GENERATION_RECEIPT_MISSING_CODE,
                    observed_at,
                    receipt_id,
                    owner_user_id,
                    *job_uuid_parameters,
                ),
            )
            return observed_at if cursor.rowcount == 1 else None

    def terminalize_expired_generation_receipt(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        expected_job_uuid: str | None,
        as_of: str,
    ) -> bool:
        """Fail an overdue receipt at its immutable logical input deadline."""
        as_of_timestamp = self._canonical_utc_timestamp(as_of)
        job_uuid_clause, job_uuid_parameters = self._generation_job_uuid_cas(
            expected_job_uuid,
            alias="r",
        )
        with self.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT r.created_at AS receipt_created_at, "
                "i.created_at AS input_created_at, i.input_expires_at "
                "FROM slides_generation_receipts r "
                "LEFT JOIN slides_generation_inputs i ON i.receipt_id = r.id "
                "WHERE r.id = ? AND r.owner_user_id = ? "
                "AND r.receipt_status IN ('claimed', 'queued', 'running') "
                f"AND {job_uuid_clause}",  # nosec B608
                (receipt_id, owner_user_id, *job_uuid_parameters),
            ).fetchone()
            if row is None:
                return False
            try:
                receipt_created_at = self._canonical_utc_timestamp(row["receipt_created_at"])
            except InputError:
                raise SlidesDatabaseError("generation_timestamp_invalid") from None
            input_deadline = receipt_created_at + timedelta(days=1)
            try:
                input_created_at = self._canonical_utc_timestamp(row["input_created_at"])
                stored_input_deadline = self._canonical_utc_timestamp(row["input_expires_at"])
            except InputError:
                # Corrupt input metadata cannot extend the receipt-derived deadline.
                pass
            else:
                if input_created_at == receipt_created_at and stored_input_deadline == input_deadline:
                    input_deadline = stored_input_deadline
            if as_of_timestamp < input_deadline:
                return False
            terminal_at = input_deadline.isoformat()
            expires_at = (input_deadline + timedelta(days=30)).isoformat()
            unaliased_job_clause, unaliased_job_parameters = self._generation_job_uuid_cas(expected_job_uuid)
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET receipt_status = 'failed', "
                "error_code = 'generation_expired', "
                "error_message = 'Generation input expired.', "
                "updated_at = ?, expires_at = ? "
                "WHERE id = ? AND owner_user_id = ? "
                "AND receipt_status IN ('claimed', 'queued', 'running') "
                f"AND {unaliased_job_clause}",  # nosec B608
                (
                    terminal_at,
                    expires_at,
                    receipt_id,
                    owner_user_id,
                    *unaliased_job_parameters,
                ),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                "DELETE FROM slides_generation_inputs WHERE receipt_id = ?",
                (receipt_id,),
            )
            return True

    def delete_terminal_generation_input(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
    ) -> bool:
        """Delete an orphaned input only for an owner-scoped terminal receipt."""
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                "DELETE FROM slides_generation_inputs WHERE receipt_id = ? "
                "AND EXISTS (SELECT 1 FROM slides_generation_receipts r "
                "WHERE r.id = slides_generation_inputs.receipt_id "
                "AND r.owner_user_id = ? "
                "AND r.receipt_status IN ('completed', 'failed', 'cancelled'))",
                (receipt_id, owner_user_id),
            )
            return cursor.rowcount == 1

    def delete_expired_generation_receipts(
        self,
        *,
        owner_user_id: str,
        expires_before: str,
        limit: int,
    ) -> int:
        """Delete bounded expired terminal receipt metadata, never presentations."""
        self._canonical_utc_timestamp(expires_before)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _GENERATION_RECONCILIATION_BATCH_MAX
        ):
            raise InputError("reconciliation limit must be between 1 and 500")
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                "DELETE FROM slides_generation_receipts WHERE id IN ("
                "SELECT id FROM slides_generation_receipts "
                "WHERE owner_user_id = ? "
                "AND receipt_status IN ('completed', 'failed', 'cancelled') "
                "AND expires_at IS NOT NULL AND expires_at <= ? "
                "ORDER BY id ASC LIMIT ?)",
                (owner_user_id, expires_before, limit),
            )
            return max(cursor.rowcount, 0)

    def count_unexpired_generation_receipts_for_digest_key(
        self,
        *,
        owner_user_id: str,
        digest_key_id: str,
        as_of: str,
    ) -> int:
        """Count physical-database live receipt references to one digest key."""
        if not isinstance(owner_user_id, str) or not owner_user_id:
            raise InputError("owner_user_id is required")
        self._canonical_utc_timestamp(as_of)
        row = (
            self.get_connection()
            .execute(
                "SELECT COUNT(*) AS count FROM slides_generation_receipts "
                "WHERE digest_key_id = ? "
                "AND (receipt_status IN ('claimed', 'queued', 'running') "
                "OR (receipt_status IN ('completed', 'failed', 'cancelled') "
                "AND (expires_at IS NULL OR expires_at > ?)))",
                (digest_key_id, as_of),
            )
            .fetchone()
        )
        return int(row["count"]) if row is not None else 0

    def find_generation_receipt_by_idempotency_digests(
        self,
        *,
        owner_user_id: str,
        digest_candidates: Iterable[str],
    ) -> SlidesGenerationReceiptRow | None:
        """Find one owner-scoped receipt using a bounded HMAC candidate set."""
        candidates = tuple(digest_candidates)
        if not candidates:
            return None
        placeholders = ",".join("?" for _ in candidates)
        rows = (
            self.get_connection()
            .execute(
                f"SELECT {_RECEIPT_PROJECTION} FROM slides_generation_receipts "  # nosec B608
                f"WHERE owner_user_id = ? AND idempotency_key_hmac_sha256 IN ({placeholders}) "
                "LIMIT 2",
                (owner_user_id, *candidates),
            )
            .fetchall()
        )
        if len(rows) > 1:
            raise SlidesDatabaseError("generation_receipt_correlation_ambiguous")
        return SlidesGenerationReceiptRow(**dict(rows[0])) if rows else None

    def claim_generation_receipt_input(
        self,
        *,
        receipt: Mapping[str, Any],
        generation_input: Mapping[str, Any],
        replay_digest_candidates: Iterable[str],
    ) -> SlidesGenerationClaimResult:
        """Atomically claim one durable receipt plus its immutable input."""
        candidates = tuple(replay_digest_candidates)
        owner_user_id = str(receipt["owner_user_id"])
        with self.transaction(immediate=True) as conn:
            if candidates:
                placeholders = ",".join("?" for _ in candidates)
                rows = conn.execute(
                    f"SELECT {_RECEIPT_PROJECTION} FROM slides_generation_receipts "  # nosec B608
                    f"WHERE owner_user_id = ? AND idempotency_key_hmac_sha256 IN ({placeholders}) "
                    "LIMIT 2",
                    (owner_user_id, *candidates),
                ).fetchall()
                if len(rows) > 1:
                    raise SlidesDatabaseError("generation_receipt_correlation_ambiguous")
                if rows:
                    existing = SlidesGenerationReceiptRow(**dict(rows[0]))
                    try:
                        stored_input = self._generation_input_from_connection(
                            conn,
                            existing.id,
                            owner_user_id,
                        )
                    except KeyError:
                        stored_input = None
                    return SlidesGenerationClaimResult(
                        receipt=existing,
                        generation_input=stored_input,
                        created=False,
                    )
            conn.execute(
                """
                INSERT INTO slides_generation_receipts (
                    id, owner_user_id, digest_key_id,
                    idempotency_key_hmac_sha256, jobs_idempotency_key,
                    client_request_hmac_sha256, execution_hmac_sha256,
                    job_id, job_uuid, presentation_id, receipt_status,
                    error_code, error_message, created_at, updated_at, expires_at
                ) VALUES (
                    :id, :owner_user_id, :digest_key_id,
                    :idempotency_key_hmac_sha256, :jobs_idempotency_key,
                    :client_request_hmac_sha256, :execution_hmac_sha256,
                    NULL, NULL, NULL, 'claimed', NULL, NULL,
                    :created_at, :updated_at, NULL
                )
                """,
                receipt,
            )
            conn.execute(
                """
                INSERT INTO slides_generation_inputs (
                    receipt_id, source_kind, source_text, source_hmac_sha256,
                    source_bytes, provenance_json, html_options_json, provider,
                    model, adapter_id, endpoint_identity, system_prompt,
                    prompt_sha256, prompt_contract_version, input_expires_at,
                    created_at
                ) VALUES (
                    :receipt_id, :source_kind, :source_text,
                    :source_hmac_sha256, :source_bytes, :provenance_json,
                    :html_options_json, :provider, :model, :adapter_id,
                    :endpoint_identity, :system_prompt, :prompt_sha256,
                    :prompt_contract_version, :input_expires_at, :created_at
                )
                """,
                generation_input,
            )
            return SlidesGenerationClaimResult(
                receipt=self._generation_receipt_from_connection(
                    conn,
                    str(receipt["id"]),
                    owner_user_id,
                ),
                generation_input=self._generation_input_from_connection(
                    conn,
                    str(receipt["id"]),
                    owner_user_id,
                ),
                created=True,
            )

    def delete_unbound_generation_claim(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
    ) -> bool:
        """Remove a deterministically rejected, never-bound claim and input."""
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                "DELETE FROM slides_generation_receipts "
                "WHERE id = ? AND owner_user_id = ? "
                "AND receipt_status = 'claimed' AND job_uuid IS NULL",
                (receipt_id, owner_user_id),
            )
            return cursor.rowcount == 1

    def bind_generation_job(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        job_id: int | None,
        job_uuid: str,
        updated_at: str,
    ) -> SlidesGenerationReceiptRow:
        """Bind an immutable Jobs UUID, storing its numeric ID only alongside it."""
        with self.transaction(immediate=True) as conn:
            current = self._generation_receipt_from_connection(
                conn,
                receipt_id,
                owner_user_id,
            )
            if current.job_uuid is not None and current.job_uuid != job_uuid:
                raise ConflictError("generation_correlation_mismatch")
            if current.job_id is not None and job_id is not None and current.job_id != int(job_id):
                raise ConflictError("generation_correlation_mismatch")
            stored_job_id = (
                current.job_id if current.job_id is not None else (int(job_id) if job_id is not None else None)
            )
            next_status = "queued" if current.receipt_status == "claimed" else current.receipt_status
            conn.execute(
                "UPDATE slides_generation_receipts SET job_id = ?, job_uuid = ?, "
                "receipt_status = ?, updated_at = ? WHERE id = ? AND owner_user_id = ?",
                (
                    stored_job_id,
                    job_uuid,
                    next_status,
                    updated_at,
                    receipt_id,
                    owner_user_id,
                ),
            )
            return self._generation_receipt_from_connection(
                conn,
                receipt_id,
                owner_user_id,
            )

    def set_generation_receipt_running(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        job_uuid: str,
        updated_at: str,
    ) -> SlidesGenerationReceiptRow:
        """Move a bound nonterminal receipt to running."""
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET receipt_status = 'running', "
                "error_code = NULL, error_message = NULL, updated_at = ? "
                "WHERE id = ? AND owner_user_id = ? AND job_uuid = ? "
                "AND receipt_status IN ('queued', 'running')",
                (updated_at, receipt_id, owner_user_id, job_uuid),
            )
            if cursor.rowcount != 1:
                raise ConflictError("generation_correlation_mismatch")
            return self._generation_receipt_from_connection(
                conn,
                receipt_id,
                owner_user_id,
            )

    def reset_generation_receipt_queued(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        job_uuid: str,
        error_code: str,
        error_message: str,
        updated_at: str,
    ) -> bool:
        """Return retryable precommit work to queued without deleting input."""
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET receipt_status = 'queued', "
                "error_code = ?, error_message = ?, updated_at = ? "
                "WHERE id = ? AND owner_user_id = ? AND job_uuid = ? "
                "AND receipt_status IN ('queued', 'running')",
                (
                    error_code,
                    error_message,
                    updated_at,
                    receipt_id,
                    owner_user_id,
                    job_uuid,
                ),
            )
            return cursor.rowcount == 1

    def terminalize_generation_receipt(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        job_uuid: str | None,
        status: str,
        error_code: str,
        error_message: str,
        terminal_at: str,
        expires_at: str,
    ) -> bool:
        """CAS a nonterminal receipt and delete input only when the CAS wins."""
        if status not in {"failed", "cancelled"}:
            raise InputError("generation terminal status is invalid")
        with self.transaction(immediate=True) as conn:
            conditions = [
                "id = ?",
                "owner_user_id = ?",
                "receipt_status IN ('claimed', 'queued', 'running')",
            ]
            parameters: list[Any] = [receipt_id, owner_user_id]
            if job_uuid is not None:
                conditions.append("job_uuid = ?")
                parameters.append(job_uuid)
            else:
                conditions.append("job_uuid IS NULL")
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET receipt_status = ?, "
                "error_code = ?, error_message = ?, updated_at = ?, expires_at = ? "
                f"WHERE {' AND '.join(conditions)}",  # nosec B608
                (status, error_code, error_message, terminal_at, expires_at, *parameters),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                "DELETE FROM slides_generation_inputs WHERE receipt_id = ?",
                (receipt_id,),
            )
            return True

    def commit_generation_presentation(
        self,
        *,
        receipt_id: str,
        owner_user_id: str,
        job_uuid: str,
        html_document: str | bytes,
        validation_result: StandaloneHtmlValidationResult,
        generation_provenance_json: str,
        committed_at: str,
        expires_at: str,
        now: Callable[[], datetime] | None = None,
    ) -> SlidesGenerationCommitResult:
        """Atomically insert one presentation, complete its receipt, and delete input."""
        source = bind_validated_standalone_source(html_document, validation_result)
        with self.transaction(immediate=True) as conn:
            receipt = self._generation_receipt_from_connection(
                conn,
                receipt_id,
                owner_user_id,
            )
            if receipt.receipt_status == "completed":
                if not receipt.presentation_id:
                    raise ConflictError("generation_correlation_mismatch")
                presentation = self._fetch_presentation_by_id(
                    conn,
                    receipt.presentation_id,
                    include_deleted=True,
                )
                if presentation.generation_job_uuid != job_uuid:
                    raise ConflictError("generation_correlation_mismatch")
                return SlidesGenerationCommitResult(presentation, created=False)
            if receipt.receipt_status not in {"queued", "running"} or receipt.job_uuid != job_uuid:
                raise ConflictError("generation_correlation_mismatch")
            generation_input = self._generation_input_from_connection(
                conn,
                receipt_id,
                owner_user_id,
            )

            def canonical_utc(value: object) -> datetime:
                if not isinstance(value, str):
                    raise ConflictError("generation_correlation_mismatch")
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    raise ConflictError("generation_correlation_mismatch") from None
                if (
                    parsed.tzinfo is None
                    or parsed.utcoffset() != timedelta(0)
                    or parsed.replace(microsecond=0).isoformat() != value
                ):
                    raise ConflictError("generation_correlation_mismatch")
                return parsed

            receipt_created = canonical_utc(receipt.created_at)
            input_created = canonical_utc(generation_input.created_at)
            input_deadline = canonical_utc(generation_input.input_expires_at)
            if (
                input_created != receipt_created
                or input_deadline != receipt_created + timedelta(hours=24)
                or generation_provenance_json != generation_input.provenance_json
            ):
                raise ConflictError("generation_correlation_mismatch")
            try:
                commit_time = datetime.fromisoformat(committed_at)
                transaction_time = (now or (lambda: datetime.now(timezone.utc)))()
            except Exception as exc:
                raise SlidesDatabaseError("generation_timestamp_invalid") from exc
            if (
                input_deadline.tzinfo is None
                or commit_time.tzinfo is None
                or transaction_time.tzinfo is None
                or transaction_time.utcoffset() != timedelta(0)
            ):
                raise SlidesDatabaseError("generation_timestamp_invalid")
            if transaction_time >= input_deadline:
                raise ConflictError("generation_expired")

            candidate = {
                "id": receipt_id,
                "title": validation_result.title,
                "description": None,
                "theme": "black",
                "marp_theme": None,
                "template_id": None,
                "visual_style_id": None,
                "visual_style_scope": None,
                "visual_style_name": None,
                "visual_style_version": None,
                "visual_style_snapshot": None,
                "settings": None,
                "studio_data": None,
                "slides": "[]",
                "slides_text": validation_result.indexable_text,
                "source_type": generation_input.source_kind,
                "source_ref": None,
                "source_query": None,
                "custom_css": None,
                "created_at": committed_at,
                "last_modified": committed_at,
                "client_id": self.client_id,
                "content_kind": "standalone_html",
                "html_document": source,
                "html_sha256": validation_result.html_sha256,
                "html_bytes": validation_result.html_bytes,
                "html_slide_count": validation_result.slide_count,
                "generation_job_uuid": job_uuid,
                "generation_provenance_json": generation_provenance_json,
            }
            presentation = self._insert_presentation_in_connection(conn, candidate)
            cursor = conn.execute(
                "UPDATE slides_generation_receipts SET presentation_id = ?, "
                "receipt_status = 'completed', error_code = NULL, "
                "error_message = NULL, updated_at = ?, expires_at = ? "
                "WHERE id = ? AND owner_user_id = ? AND job_uuid = ? "
                "AND receipt_status IN ('queued', 'running')",
                (
                    presentation.id,
                    committed_at,
                    expires_at,
                    receipt_id,
                    owner_user_id,
                    job_uuid,
                ),
            )
            if cursor.rowcount != 1:
                raise ConflictError("generation_correlation_mismatch")
            conn.execute(
                "DELETE FROM slides_generation_inputs WHERE receipt_id = ?",
                (receipt_id,),
            )
            return SlidesGenerationCommitResult(presentation, created=True)

    def get_generation_receipt(
        self,
        receipt_id: str,
        *,
        owner_user_id: str,
    ) -> SlidesGenerationReceiptRow:
        """Fetch one generation receipt scoped to its canonical owner."""
        query = (
            f"SELECT {_RECEIPT_PROJECTION} "  # nosec B608
            "FROM slides_generation_receipts "
            "WHERE id = ? AND owner_user_id = ?"
        )
        row = (
            self.get_connection()
            .execute(
                query,
                (receipt_id, owner_user_id),
            )
            .fetchone()
        )
        if not row:
            raise KeyError("slides_generation_receipt_not_found")
        return SlidesGenerationReceiptRow(**dict(row))

    def get_generation_input(
        self,
        receipt_id: str,
        *,
        owner_user_id: str,
    ) -> SlidesGenerationInputRow:
        """Fetch immutable execution input scoped through its receipt owner."""
        query = (
            f"SELECT {_INPUT_PROJECTION} "  # nosec B608
            "FROM slides_generation_inputs WHERE receipt_id = ? "
            "AND EXISTS ("
            "SELECT 1 FROM slides_generation_receipts r "
            "WHERE r.id = slides_generation_inputs.receipt_id "
            "AND r.owner_user_id = ?"
            ")"
        )
        row = (
            self.get_connection()
            .execute(
                query,
                (receipt_id, owner_user_id),
            )
            .fetchone()
        )
        if not row:
            raise KeyError("slides_generation_input_not_found")
        return SlidesGenerationInputRow(**dict(row))

    def soft_delete_presentation(self, presentation_id: str, expected_version: int) -> PresentationRow:
        return self.update_presentation(
            presentation_id=presentation_id,
            update_fields={"deleted": 1},
            expected_version=expected_version,
            operation="delete",
        )

    def restore_presentation(self, presentation_id: str, expected_version: int) -> PresentationRow:
        return self.update_presentation(
            presentation_id=presentation_id,
            update_fields={"deleted": 0},
            expected_version=expected_version,
            operation="restore",
        )
