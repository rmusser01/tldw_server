"""Routine request handles must not deadlock a real Claims/Media read."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime.session import MediaDbFactory
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.mark.timeout(60)
def test_claims_read_can_finish_while_request_handle_reopens(pg_restricted_backend, monkeypatch):
    backend = pg_restricted_backend
    with scoped_context(user_id=341, is_admin=False):
        db = MediaDatabase(":memory:", client_id="341", backend=backend)
        media_id, _, _ = db.add_media_with_keywords(
            title="S05 concurrency source",
            media_type="text",
            content="The telescope is named Selene.",
            keywords=None,
            owner_user_id=341,
            visibility="personal",
        )
        db.upsert_claims(
            [
                {
                    "media_id": media_id,
                    "claim_text": "The telescope is named Selene.",
                    "extractor": "uat-manual",
                    "extractor_version": "s05-lock-order",
                    "chunk_hash": "controlled-concurrency-source",
                    "reviewer_id": 341,
                }
            ]
        )
        claim_id = db.get_claims_by_media(media_id)[0]["id"]
        db.release_context_connection()
    factory = MediaDbFactory(":memory:", client_id="341", backend=backend)
    claims_held = threading.Event()
    reopen_boundary = threading.Event()
    create_fts = backend.create_fts_table

    def observe_fts_boundary(*args, **kwargs):
        if threading.current_thread().name.endswith("_1") and kwargs.get("source_table") == "claims":
            reopen_boundary.set()
        return create_fts(*args, **kwargs)

    monkeypatch.setattr(backend, "create_fts_table", observe_fts_boundary)

    def reader():
        with scoped_context(user_id=341, is_admin=False):
            try:
                with db.transaction():
                    assert [row["id"] for row in db.get_claims_by_media(media_id)] == [claim_id]  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
                    claims_held.set()
                    assert reopen_boundary.wait(10), "Reopen did not reach its declared boundary"  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
                    assert db.get_media_by_id(media_id)["content"] == "The telescope is named Selene."  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
            finally:
                claims_held.set()
                db.release_context_connection()

    def reopen():
        assert claims_held.wait(10), "Reader did not acquire its real Claims read"  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
        session = None
        with scoped_context(user_id=341, is_admin=False):
            try:
                session = factory.for_request()
                reopen_boundary.set()
                assert [row["id"] for row in session.list_review_queue(status="pending", owner_user_id=341)] == [  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
                    claim_id
                ]
            finally:
                reopen_boundary.set()
                if session is not None:
                    session.release_context_connection()

    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="s05_claims") as workers:
            read = workers.submit(reader)
            reopened = workers.submit(reopen)
            read.result(timeout=40)
            reopened.result(timeout=40)
    finally:
        reopen_boundary.set()
        db.close_connection()


def test_schema_owner_still_repairs_missing_claims_fts_without_changing_source(pg_restricted_backend):
    backend = pg_restricted_backend
    with scoped_context(user_id=341, is_admin=False):
        initial = MediaDatabase(":memory:", client_id="341", backend=backend)
        media_id, _, _ = initial.add_media_with_keywords(
            title="Retained S05 source",
            media_type="text",
            content="The telescope is named Selene.",
            owner_user_id=341,
            visibility="personal",
        )
        initial.close_connection()
        with backend.transaction() as conn:
            backend.execute("DROP TRIGGER update_claims_fts_tsv_trigger ON claims", connection=conn)
        reopened = MediaDbFactory(":memory:", client_id="341", backend=backend).for_request()
        try:
            reopened.upsert_claims(
                [
                    {
                        "media_id": media_id,
                        "claim_text": "The telescope is named Selene.",
                        "extractor": "uat-manual",
                        "extractor_version": "s05-repair",
                        "chunk_hash": "controlled-owner-repair",
                        "reviewer_id": 341,
                    }
                ]
            )
            row = reopened.execute_query(
                "SELECT claims_fts_tsv IS NOT NULL AS populated FROM Claims WHERE media_id = ?",
                (media_id,),
            ).fetchone()
            assert row["populated"] is True  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
            assert reopened.get_media_by_id(media_id)["content"] == "The telescope is named Selene."  # nosec B101 -- Pytest verifies real lock completion, repair and retained source data.
        finally:
            reopened.release_context_connection()


def test_owner_catalog_connection_failure_is_not_repaired(pg_restricted_backend, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.DB_Management.media_db.schema import postgres_runtime_validation

    backend = pg_restricted_backend
    with scoped_context(user_id=341, is_admin=False):
        initial = MediaDatabase(":memory:", client_id="341", backend=backend)
        media_id, _, _ = initial.add_media_with_keywords(
            title="S05 inspection refusal",
            media_type="text",
            content="Retain this source on failed inspection.",
            owner_user_id=341,
            visibility="personal",
        )
        initial.release_context_connection()

        def inspection_unavailable(_db, _conn):
            raise BackendDatabaseError("Catalog connection unavailable")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(postgres_runtime_validation, "validate_postgres_schema_catalog", inspection_unavailable)
                with pytest.raises(DatabaseError, match="Catalog connection unavailable"):
                    MediaDbFactory(":memory:", client_id="341", backend=backend).for_request()
            assert initial.get_media_by_id(media_id)["content"] == "Retain this source on failed inspection."  # nosec B101 -- Pytest verifies no source mutation on a failed catalog connection.
        finally:
            initial.release_context_connection()


@pytest.fixture
def owner_source(pg_restricted_backend):
    with scoped_context(user_id=341, is_admin=False):
        db = MediaDatabase(":memory:", client_id="341", backend=pg_restricted_backend)
        try:
            media_id, _, _ = db.add_media_with_keywords(
                title="S05 owner reconciliation",
                media_type="text",
                content="Keep the Rowan source.",
                owner_user_id=341,
                visibility="personal",
            )
            db.upsert_claims(
                [
                    {
                        "media_id": media_id,
                        "claim_text": "Keep the Rowan source.",
                        "extractor": "uat-manual",
                        "extractor_version": "s05-owner-repair",
                        "chunk_hash": "controlled-owner-source",
                        "reviewer_id": 341,
                    }
                ]
            )
            claim_id = db.get_claims_by_media(media_id)[0]["id"]
            db.release_context_connection()
            yield db, media_id, claim_id
        finally:
            db.release_context_connection()


@pytest.mark.parametrize(
    "index",
    [
        "ux_output_templates_user_name",
        "ux_content_items_user_canonical",
        "ux_content_items_user_hash",
        "idx_tts_history_user_created",
    ],
)
def test_current_owner_reopen_restores_missing_required_index(owner_source, index):
    db, media_id, _ = owner_source
    backend = db.backend
    with backend.transaction() as conn:
        backend.execute(f"DROP INDEX {backend.escape_identifier(index)}", connection=conn)  # nosec B608 -- Closed test identifiers, escaped by the real backend.
    reopened = MediaDbFactory(":memory:", client_id="341", backend=backend).for_request()
    try:
        present = reopened.execute_query(
            "SELECT EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relkind='i' AND c.relname=?) AS present",
            (index,),
        ).fetchone()
        assert present["present"] is True  # nosec B101 -- Pytest verifies real owner index repair after reopen.
        assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies retained content.
    finally:
        reopened.release_context_connection()


def test_current_owner_reopen_backfills_scoped_claim_review_fields(owner_source):
    db, media_id, claim_id = owner_source
    with db.backend.transaction() as conn:
        db.backend.execute("ALTER TABLE claims ALTER COLUMN review_status DROP NOT NULL", connection=conn)
        db.backend.execute("ALTER TABLE claims ALTER COLUMN review_version DROP NOT NULL", connection=conn)
    db.execute_query("UPDATE Claims SET review_status=NULL, review_version=NULL WHERE id=?", (claim_id,), commit=True)
    reopened = MediaDbFactory(":memory:", client_id="341", backend=db.backend).for_request()
    try:
        claim = reopened.get_claims_by_media(media_id)[0]
        assert (claim["review_status"], claim["review_version"]) == ("pending", 1)  # nosec B101 -- Pytest verifies existing scoped backfill behavior.
        assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies retained content.
    finally:
        reopened.release_context_connection()


def test_current_owner_reopen_backfills_legacy_data_table_fields(owner_source):
    db, media_id, _ = owner_source
    with scoped_context(user_id=341, is_admin=True):
        with db.backend.transaction() as conn:
            db.backend.execute("ALTER TABLE data_tables ALTER COLUMN last_modified DROP NOT NULL", connection=conn)
            db.backend.execute(
                "INSERT INTO data_tables(uuid,name,prompt,created_at,updated_at,last_modified,client_id) "
                "VALUES('s05-legacy-row','Legacy S05','Controlled input',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,NULL,'')",
                connection=conn,
            )
        reopened = MediaDbFactory(":memory:", client_id="341", backend=db.backend).for_request()
        try:
            row = reopened.execute_query(
                "SELECT client_id,last_modified FROM data_tables WHERE uuid='s05-legacy-row'"
            ).fetchone()
            assert row["client_id"] == "341" and row["last_modified"] is not None  # nosec B101 -- Pytest verifies both existing legacy backfills.
            assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies retained content.
        finally:
            reopened.release_context_connection()


@pytest.mark.parametrize(
    ("index", "table", "position"),
    [
        ("ux_data_table_columns_table_position_active", "data_table_columns", "position"),
        ("ux_data_table_rows_table_index_active", "data_table_rows", "row_index"),
    ],
)
def test_current_owner_reopen_retires_obsolete_guard_index(owner_source, index, table, position):
    db, media_id, _ = owner_source
    ident = db.backend.escape_identifier
    with db.backend.transaction() as conn:
        db.backend.execute(
            f"CREATE UNIQUE INDEX {ident(index)} ON {ident(table)} (table_id,{ident(position)})", connection=conn
        )  # nosec B608 -- Closed test identifiers, escaped by the real backend.
    reopened = MediaDbFactory(":memory:", client_id="341", backend=db.backend).for_request()
    try:
        row = reopened.execute_query("SELECT to_regclass(?) IS NULL AS retired", (index,)).fetchone()
        assert row["retired"] is True  # nosec B101 -- Pytest verifies actual removal of the obsolete owner guard.
        assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies retained content.
    finally:
        reopened.release_context_connection()


def test_current_owner_reopen_retries_optional_email_acceleration(owner_source):
    db, media_id, _ = owner_source
    with db.backend.transaction() as conn:
        db.backend.execute("DROP INDEX idx_email_text_trigram", connection=conn)
        db.backend.execute("DROP STATISTICS email_media_visibility_stats", connection=conn)
        db.backend.execute("DROP STATISTICS email_text_condition_stats", connection=conn)
    reopened = MediaDbFactory(":memory:", client_id="341", backend=db.backend).for_request()
    try:
        row = reopened.execute_query(
            "SELECT to_regclass('idx_email_text_trigram') IS NOT NULL AS indexed, "
            "(SELECT count(*) FROM pg_statistic_ext s JOIN pg_namespace n ON n.oid=s.stxnamespace "
            "WHERE n.nspname=current_schema() AND s.stxname IN "
            "('email_media_visibility_stats','email_text_condition_stats')) AS statistics"
        ).fetchone()
        assert (row["indexed"], row["statistics"]) == (True, 2)  # nosec B101 -- Pytest verifies the existing optional acceleration attempt after catalog loss.
        assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies retained content.
    finally:
        reopened.release_context_connection()


def test_ready_owner_reopen_keeps_schema_read_only(owner_source, monkeypatch):
    db, media_id, _ = owner_source
    execute = db.backend.execute
    schema_commands = []

    def observe(query, *args, **kwargs):
        command = query.strip().split(maxsplit=1)[0].upper()
        if command in {"ALTER", "CREATE", "DROP"}:
            schema_commands.append(command)
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(db.backend, "execute", observe)
    reopened = MediaDbFactory(":memory:", client_id="341", backend=db.backend).for_request()
    try:
        assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies the actual request-handle result.
        assert schema_commands == []  # nosec B101 -- Pytest verifies that an already-ready owner handle performs no schema writes.
    finally:
        reopened.release_context_connection()


@pytest.mark.parametrize(("table", "alias"), [("media", "media_fts"), ("claims", "claims_fts")])
@pytest.mark.parametrize("state", ["null", "stale"])
def test_owner_reopen_backfills_preexisting_fts_rows(owner_source, table, alias, state):
    from tldw_Server_API.app.core.DB_Management.backends.base import FTSQuery

    db, media_id, claim_id = owner_source
    backend = db.backend
    ident = backend.escape_identifier
    trigger = "update_" + alias + "_tsv_trigger"
    field = alias + "_tsv"
    row_id = media_id if table == "media" else claim_id
    value = "NULL" if state == "null" else "to_tsvector('english','obsolete')"
    with backend.transaction() as conn:
        backend.execute(f"ALTER TABLE {ident(table)} DISABLE TRIGGER {ident(trigger)}", connection=conn)  # nosec B608 -- Closed fixture identifiers, escaped by the real backend.
        backend.execute(f"UPDATE {ident(table)} SET {ident(field)}={value} WHERE id=%s", (row_id,), connection=conn)  # nosec B608 -- Only fixed NULL/obsolete-vector expressions and escaped fixture identifiers.
        backend.execute(f"ALTER TABLE {ident(table)} ENABLE TRIGGER {ident(trigger)}", connection=conn)  # nosec B608 -- Restore the actual trigger before the reopen proof.
    query = FTSQuery(query="Rowan", table=alias, limit=10)
    assert backend.fts_search(query).rows == []  # nosec B101 -- Pytest proves that the existing-row fixture is unsearchable without fallback.
    reopened = MediaDbFactory(":memory:", client_id="341", backend=backend).for_request()
    try:
        assert [row["id"] for row in backend.fts_search(query).rows] == [row_id]  # nosec B101 -- Pytest verifies user-visible FTS recovery for existing content.
        assert reopened.get_media_by_id(media_id)["content"] == "Keep the Rowan source."  # nosec B101 -- Pytest verifies that index repair leaves the source content intact.
    finally:
        reopened.release_context_connection()
