import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


def _column_exists(backend, conn, table: str, column: str) -> bool:
    """Return True if the supplied column is present on the table."""

    result = backend.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table, column),
        connection=conn,
    )
    return bool(result.rows)


def _serial_sequence_name(backend, conn, table: str, column: str) -> str:
    """Fetch the fully-qualified sequence backing a serial column."""

    result = backend.execute(
        "SELECT pg_get_serial_sequence(%s, %s) AS seq",
        (table, column),
        connection=conn,
    )
    sequence = result.scalar
    if not sequence:
        raise AssertionError(f"Sequence not found for {table}.{column}")
    return str(sequence)


def _index_exists(backend, conn, index_name: str) -> bool:
    """Return True if the supplied PostgreSQL index exists in public schema."""

    result = backend.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = %s
        LIMIT 1
        """,
        (index_name,),
        connection=conn,
    )
    return bool(result.rows)


def _constraint_exists(backend, conn, constraint_name: str) -> bool:
    """Return True if the supplied PostgreSQL constraint exists on public tables."""

    result = backend.execute(
        """
        SELECT 1
        FROM pg_constraint
        WHERE conname = %s
        LIMIT 1
        """,
        (constraint_name,),
        connection=conn,
    )
    return bool(result.rows)


@pytest.mark.integration
def test_media_postgres_migration_adds_safe_metadata(pg_database_config: DatabaseConfig) -> None:
    """Downgrade schema to v4 and ensure migration restores the v5 metadata column."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration", backend=backend)

    try:
        with backend.transaction() as conn:
            backend.execute(
                "ALTER TABLE DocumentVersions DROP COLUMN IF EXISTS safe_metadata",
                connection=conn,
            )
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (4,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert _column_exists(backend, conn, "documentversions", "safe_metadata")
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar
            assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_migration_reaches_v11_and_restores_mediafiles_table(
    pg_database_config: DatabaseConfig,
) -> None:
    """Downgrade schema to v10 and ensure v11 restores the MediaFiles table."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration-v11", backend=backend)

    try:
        with backend.transaction() as conn:
            backend.execute("DROP TABLE IF EXISTS MediaFiles", connection=conn)
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (10,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert backend.table_exists("MediaFiles", connection=conn)
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar
            assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_migration_reaches_v20_and_restores_workspace_tag(
    pg_database_config: DatabaseConfig,
) -> None:
    """Downgrade schema to v9 and ensure v10–v20 migrations repair late additions."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration-v19", backend=backend)

    try:
        with backend.transaction() as conn:
            backend.execute(
                "DROP INDEX IF EXISTS idx_data_tables_workspace_tag",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE data_tables DROP COLUMN IF EXISTS workspace_tag",
                connection=conn,
            )
            backend.execute(
                "DROP TABLE IF EXISTS claims_monitoring_events",
                connection=conn,
            )
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (9,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert _column_exists(backend, conn, "data_tables", "workspace_tag")
            assert backend.table_exists("claims_monitoring_events", connection=conn)
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar
            assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_migration_reaches_v9_and_restores_visibility_owner_columns(
    pg_database_config: DatabaseConfig,
) -> None:
    """Downgrade schema to v8 and ensure v9 restores visibility-owner artifacts."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration-v9", backend=backend)
    media_uuid = str(uuid.uuid4())

    try:
        with backend.transaction() as conn:
            now = db._get_current_utc_timestamp_str()
            backend.execute(
                """
                INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_uuid,
                    "Visibility migration",
                    "text",
                    f"hash-{media_uuid}",
                    now,
                    1,
                    "321",
                ),
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE media DROP CONSTRAINT IF EXISTS chk_media_visibility",
                connection=conn,
            )
            backend.execute("DROP INDEX IF EXISTS idx_media_visibility", connection=conn)
            backend.execute(
                "DROP INDEX IF EXISTS idx_media_owner_user_id",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE media DROP COLUMN IF EXISTS owner_user_id CASCADE",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE media DROP COLUMN IF EXISTS visibility CASCADE",
                connection=conn,
            )
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (8,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert _column_exists(backend, conn, "media", "visibility")
            assert _column_exists(backend, conn, "media", "owner_user_id")
            assert _index_exists(backend, conn, "idx_media_visibility")
            assert _index_exists(backend, conn, "idx_media_owner_user_id")
            assert _constraint_exists(backend, conn, "chk_media_visibility")
            owner_user_id = backend.execute(
                "SELECT owner_user_id FROM media WHERE uuid = %s",
                (media_uuid,),
                connection=conn,
            ).scalar
            assert int(owner_user_id) == 321
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar
            assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_sequence_sync(pg_database_config: DatabaseConfig) -> None:
    """Sequences are advanced to match table maxima after initialization."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-seq", backend=backend)

    try:
        with backend.transaction() as conn:
            now = db._get_current_utc_timestamp_str()
            backend.execute(
                """
                INSERT INTO Media (title, type, content_hash, uuid, last_modified, version, client_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    "Sequence sync",
                    "text",
                    "hash-123",
                    str(uuid.uuid4()),
                    now,
                    1,
                    db.client_id,
                ),
                connection=conn,
            )

        with backend.transaction() as conn:
            sequence = _serial_sequence_name(backend, conn, "media", "id")
            backend.execute(
                "SELECT setval(%s, %s, false)",
                (sequence, 1),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            sequence = _serial_sequence_name(backend, conn, "media", "id")
            next_value = backend.execute(
                "SELECT nextval(%s)",
                (sequence,),
                connection=conn,
            ).scalar
        assert int(next_value) > 1
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_migration_reaches_v21_and_restores_structure_visual_indexes(
    pg_database_config: DatabaseConfig,
) -> None:
    """Downgrade schema to v20 and ensure v21 restores structure and visual indexes."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration-v21", backend=backend)

    try:
        with backend.transaction() as conn:
            backend.execute("DROP INDEX IF EXISTS idx_dsi_media_path", connection=conn)
            backend.execute("DROP INDEX IF EXISTS idx_visualdocs_caption", connection=conn)
            backend.execute("DROP INDEX IF EXISTS idx_visualdocs_tags", connection=conn)
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (20,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert _index_exists(backend, conn, "idx_dsi_media_path")
            assert _index_exists(backend, conn, "idx_visualdocs_caption")
            assert _index_exists(backend, conn, "idx_visualdocs_tags")
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar
            assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_migration_reaches_v22_and_restores_email_schema(
    pg_database_config: DatabaseConfig,
) -> None:
    """Downgrade schema to v21 and ensure v22 restores stable email-native artifacts."""

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration-v22", backend=backend)

    try:
        with backend.transaction() as conn:
            backend.execute("DROP TABLE IF EXISTS email_sources CASCADE", connection=conn)
            backend.execute(
                "DROP INDEX IF EXISTS idx_email_messages_tenant_date_id",
                connection=conn,
            )
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (21,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert backend.table_exists("email_sources", connection=conn)
            assert _index_exists(backend, conn, "idx_email_messages_tenant_date_id")
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar
            assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_media_postgres_migration_reaches_v23_and_backfills_transcript_run_history(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="pg-migration-v23", backend=backend)

    try:
        media_one_uuid = str(uuid.uuid4())
        media_two_uuid = str(uuid.uuid4())
        transcript_one_uuid = str(uuid.uuid4())
        transcript_two_uuid = str(uuid.uuid4())
        transcript_three_uuid = str(uuid.uuid4())
        transcript_four_uuid = str(uuid.uuid4())

        with backend.transaction() as conn:
            now = db._get_current_utc_timestamp_str()
            backend.execute(
                """
                INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s),
                       (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_one_uuid,
                    "Transcript history media one",
                    "audio",
                    f"hash-{media_one_uuid}",
                    now,
                    1,
                    db.client_id,
                    media_two_uuid,
                    "Transcript history media two",
                    "audio",
                    f"hash-{media_two_uuid}",
                    now,
                    1,
                    db.client_id,
                ),
                connection=conn,
            )
            media_lookup = backend.execute(
                "SELECT id, uuid FROM media WHERE uuid IN (%s, %s) ORDER BY id",
                (media_one_uuid, media_two_uuid),
                connection=conn,
            ).rows
            media_one_id = int(media_lookup[0]["id"])
            media_two_id = int(media_lookup[1]["id"])
            backend.execute(
                """
                INSERT INTO Transcripts (
                    media_id, whisper_model, transcription, created_at, uuid, last_modified, version, client_id, deleted
                )
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s),
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s),
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s),
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_one_id,
                    "small",
                    "newer undeleted transcript",
                    "2024-01-03T00:00:00Z",
                    transcript_one_uuid,
                    "2024-01-03T00:00:00Z",
                    1,
                    db.client_id,
                    False,
                    media_one_id,
                    "large",
                    "older undeleted transcript",
                    "2024-01-01T00:00:00Z",
                    transcript_two_uuid,
                    "2024-01-01T00:00:00Z",
                    1,
                    db.client_id,
                    False,
                    media_one_id,
                    "xlarge",
                    "deleted newest transcript",
                    "2024-01-04T00:00:00Z",
                    transcript_three_uuid,
                    "2024-01-04T00:00:00Z",
                    1,
                    db.client_id,
                    True,
                    media_two_id,
                    "small",
                    "third transcript",
                    "2024-01-02T00:00:00Z",
                    transcript_four_uuid,
                    "2024-01-02T00:00:00Z",
                    1,
                    db.client_id,
                    False,
                ),
                connection=conn,
            )
            backend.execute(
                "DROP INDEX IF EXISTS idx_transcripts_media_run_id",
                connection=conn,
            )
            backend.execute(
                "DROP INDEX IF EXISTS idx_transcripts_supersedes_run_id",
                connection=conn,
            )
            backend.execute(
                "DROP INDEX IF EXISTS idx_transcripts_media_idempotency_key",
                connection=conn,
            )
            backend.execute(
                "DROP INDEX IF EXISTS idx_media_latest_transcription_run_id",
                connection=conn,
            )
            backend.execute(
                "DROP INDEX IF EXISTS idx_media_next_transcription_run_id",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE media DROP COLUMN IF EXISTS latest_transcription_run_id CASCADE",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE media DROP COLUMN IF EXISTS next_transcription_run_id CASCADE",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE transcripts DROP COLUMN IF EXISTS transcription_run_id CASCADE",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE transcripts DROP COLUMN IF EXISTS supersedes_run_id CASCADE",
                connection=conn,
            )
            backend.execute(
                "ALTER TABLE transcripts DROP COLUMN IF EXISTS idempotency_key CASCADE",
                connection=conn,
            )
            backend.execute(
                """
                ALTER TABLE transcripts
                ADD CONSTRAINT transcripts_media_id_whisper_model_key
                UNIQUE (media_id, whisper_model)
                """,
                connection=conn,
            )
            backend.execute(
                "UPDATE schema_version SET version = %s",
                (22,),
                connection=conn,
            )

        db._initialize_schema()

        with backend.transaction() as conn:
            assert _column_exists(backend, conn, "media", "latest_transcription_run_id")
            assert _column_exists(backend, conn, "media", "next_transcription_run_id")
            assert _column_exists(backend, conn, "transcripts", "transcription_run_id")
            assert _column_exists(backend, conn, "transcripts", "supersedes_run_id")
            assert _column_exists(backend, conn, "transcripts", "idempotency_key")
            assert _index_exists(backend, conn, "idx_transcripts_media_run_id")
            assert _index_exists(backend, conn, "idx_transcripts_supersedes_run_id")
            assert _index_exists(backend, conn, "idx_transcripts_media_idempotency_key")
            assert _index_exists(backend, conn, "idx_media_latest_transcription_run_id")
            assert _index_exists(backend, conn, "idx_media_next_transcription_run_id")
            assert not _constraint_exists(backend, conn, "transcripts_media_id_whisper_model_key")
            transcript_rows = backend.execute(
                """
                SELECT id, media_id, whisper_model, created_at, deleted, transcription_run_id, supersedes_run_id, idempotency_key
                FROM transcripts
                ORDER BY media_id, id
                """,
                connection=conn,
            ).rows
            media_rows = backend.execute(
                """
                SELECT id, latest_transcription_run_id, next_transcription_run_id
                FROM media
                ORDER BY id
                """,
                connection=conn,
            ).rows
            version = backend.execute(
                "SELECT version FROM schema_version LIMIT 1",
                connection=conn,
            ).scalar

            assert [
                {
                    "id": int(row["id"]),
                    "media_id": int(row["media_id"]),
                    "whisper_model": row["whisper_model"],
                    "created_at": str(row["created_at"]),
                    "deleted": bool(row["deleted"]),
                    "transcription_run_id": int(row["transcription_run_id"]),
                    "supersedes_run_id": row["supersedes_run_id"],
                    "idempotency_key": row["idempotency_key"],
                }
                for row in transcript_rows
            ] == [
                {
                    "id": 1,
                    "media_id": media_one_id,
                    "whisper_model": "small",
                    "created_at": "2024-01-03 00:00:00+00:00",
                    "deleted": False,
                    "transcription_run_id": 2,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
                {
                    "id": 2,
                    "media_id": media_one_id,
                    "whisper_model": "large",
                    "created_at": "2024-01-01 00:00:00+00:00",
                    "deleted": False,
                    "transcription_run_id": 1,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
                {
                    "id": 3,
                    "media_id": media_one_id,
                    "whisper_model": "xlarge",
                    "created_at": "2024-01-04 00:00:00+00:00",
                    "deleted": True,
                    "transcription_run_id": 3,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
                {
                    "id": 4,
                    "media_id": media_two_id,
                    "whisper_model": "small",
                    "created_at": "2024-01-02 00:00:00+00:00",
                    "deleted": False,
                    "transcription_run_id": 1,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
            ]
            assert [
                {
                    "id": int(row["id"]),
                    "latest_transcription_run_id": (
                        int(row["latest_transcription_run_id"])
                        if row["latest_transcription_run_id"] is not None
                        else None
                    ),
                    "next_transcription_run_id": int(row["next_transcription_run_id"]),
                }
                for row in media_rows
            ] == [
                {
                    "id": media_one_id,
                    "latest_transcription_run_id": 2,
                    "next_transcription_run_id": 4,
                },
                {
                    "id": media_two_id,
                    "latest_transcription_run_id": 1,
                    "next_transcription_run_id": 2,
                },
            ]

            backend.execute(
                """
                INSERT INTO Transcripts (
                    media_id, whisper_model, transcription, created_at, transcription_run_id,
                    idempotency_key, uuid, last_modified, version, client_id, deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_one_id,
                    "nullable-key-one",
                    "nullable key transcript",
                    "2024-01-05T00:00:00Z",
                    4,
                    None,
                    str(uuid.uuid4()),
                    "2024-01-05T00:00:00Z",
                    1,
                    db.client_id,
                    False,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO Transcripts (
                    media_id, whisper_model, transcription, created_at, transcription_run_id,
                    idempotency_key, uuid, last_modified, version, client_id, deleted
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    media_one_id,
                    "idempotency-anchor",
                    "idempotency key anchor",
                    "2024-01-06T00:00:00Z",
                    5,
                    "job-unique",
                    str(uuid.uuid4()),
                    "2024-01-06T00:00:00Z",
                    1,
                    db.client_id,
                    False,
                ),
                connection=conn,
            )

        with pytest.raises(BackendDatabaseError, match="unique|duplicate key"):
            with backend.transaction() as conn:
                backend.execute(
                    """
                    INSERT INTO Transcripts (
                        media_id, whisper_model, transcription, created_at, transcription_run_id,
                        idempotency_key, uuid, last_modified, version, client_id, deleted
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        media_one_id,
                        "duplicate-run",
                        "duplicate run transcript",
                        "2024-01-07T00:00:00Z",
                        2,
                        "job-dup-run",
                        str(uuid.uuid4()),
                        "2024-01-07T00:00:00Z",
                        1,
                        db.client_id,
                        False,
                    ),
                    connection=conn,
                )

        with pytest.raises(BackendDatabaseError, match="unique|duplicate key"):
            with backend.transaction() as conn:
                backend.execute(
                    """
                    INSERT INTO Transcripts (
                        media_id, whisper_model, transcription, created_at, transcription_run_id,
                        idempotency_key, uuid, last_modified, version, client_id, deleted
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        media_one_id,
                        "duplicate-key",
                        "duplicate key transcript",
                        "2024-01-08T00:00:00Z",
                        6,
                        "job-unique",
                        str(uuid.uuid4()),
                        "2024-01-08T00:00:00Z",
                        1,
                        db.client_id,
                        False,
                    ),
                    connection=conn,
                )

        assert int(version) == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()
