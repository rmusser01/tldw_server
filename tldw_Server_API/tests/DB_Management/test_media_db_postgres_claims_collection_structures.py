import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

pytestmark = pytest.mark.unit


def _load_postgres_claims_collection_structures_module():
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_claims_collection_structures"
    )
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Expected schema helper module spec for {module_name}"
    return importlib.import_module(module_name)


def test_postgres_claims_collection_helper_rebinds_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase

    helper_module = _load_postgres_claims_collection_structures_module()

    assert MediaDatabase.__dict__["_ensure_postgres_claims_tables"].__globals__[
        "__name__"
    ] == helper_module.__name__
    assert MediaDatabase.__dict__["_ensure_postgres_collections_tables"].__globals__[
        "__name__"
    ] == helper_module.__name__
    assert MediaDatabase.__dict__["_ensure_postgres_claims_extensions"].__globals__[
        "__name__"
    ] == helper_module.__name__


def test_ensure_postgres_claims_tables_runs_create_tables_then_extensions_then_other_statements() -> None:
    helper_module = _load_postgres_claims_collection_structures_module()

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def execute(self, query: str, *, connection) -> None:
            calls.append(("execute", query))

    def _convert_sqlite_sql_to_postgres_statements(sql: str) -> list[str]:
        assert sql == "claims sql"
        return [
            "CREATE TABLE claims (id BIGINT)",
            "CREATE TABLE claim_notes (id BIGINT)",
            "CREATE INDEX idx_claims_text ON claims(id)",
            "ALTER TABLE claims ADD COLUMN foo TEXT",
        ]

    db = SimpleNamespace(
        _CLAIMS_TABLE_SQL="claims sql",
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=_convert_sqlite_sql_to_postgres_statements,
        _ensure_postgres_claims_extensions=lambda value: calls.append(("extensions", value)),
    )

    helper_module.ensure_postgres_claims_tables(db, conn)

    assert calls == [
        ("execute", "CREATE TABLE claims (id BIGINT)"),
        ("execute", "CREATE TABLE claim_notes (id BIGINT)"),
        ("extensions", conn),
        ("execute", "CREATE INDEX idx_claims_text ON claims(id)"),
        ("execute", "ALTER TABLE claims ADD COLUMN foo TEXT"),
    ]


def test_ensure_postgres_collections_tables_executes_representative_ddls() -> None:
    helper_module = _load_postgres_claims_collection_structures_module()

    conn = object()
    queries: list[str] = []

    class FakeBackend:
        def execute(self, query: str, *, connection) -> None:
            queries.append(" ".join(query.split()))

    db = SimpleNamespace(backend=FakeBackend())

    helper_module.ensure_postgres_collections_tables(db, conn)

    assert any("CREATE TABLE IF NOT EXISTS output_templates" in query for query in queries)
    assert any("CREATE TABLE IF NOT EXISTS content_items" in query for query in queries)
    assert any(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_canonical" in query
        for query in queries
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS idx_content_items_user_updated" in query
        for query in queries
    )
    assert any("CREATE TABLE IF NOT EXISTS content_item_tags" in query for query in queries)


def test_ensure_postgres_claims_extensions_executes_representative_claims_ddls_and_backfills() -> None:
    helper_module = _load_postgres_claims_collection_structures_module()

    conn = object()
    queries: list[str] = []

    class FakeBackend:
        @staticmethod
        def escape_identifier(name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection) -> None:
            queries.append(" ".join(query.split()))

    db = SimpleNamespace(backend=FakeBackend())

    helper_module.ensure_postgres_claims_extensions(db, conn)

    assert any(
        'ALTER TABLE "claims" ADD COLUMN IF NOT EXISTS "review_status" TEXT DEFAULT \'pending\''
        in query
        for query in queries
    )
    assert any(
        'UPDATE "claims" SET "review_status" = \'pending\'' in query
        for query in queries
    )
    assert any('CREATE INDEX IF NOT EXISTS "idx_claims_cluster_id"' in query for query in queries)
    assert any('CREATE TABLE IF NOT EXISTS "claims_review_log"' in query for query in queries)
    assert any(
        'CREATE TABLE IF NOT EXISTS "claims_review_extractor_metrics_daily"' in query
        for query in queries
    )
    assert any(
        'CREATE TABLE IF NOT EXISTS "claims_monitoring_events"' in query
        for query in queries
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_claims_monitoring_events_delivered"' in query
        for query in queries
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_claims_monitoring_events_user_created_id" '
        'ON "claims_monitoring_events" ("user_id", "created_at", "id")' in query
        for query in queries
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_claims_monitoring_events_user_id" '
        'ON "claims_monitoring_events" ("user_id", "id")' in query
        for query in queries
    )
    assert any("CREATE OR REPLACE FUNCTION tldw_claims_safe_json" in query for query in queries)
    assert any("CREATE OR REPLACE FUNCTION tldw_claims_compact_json" in query for query in queries)
    assert any('CREATE TABLE IF NOT EXISTS "claim_clusters"' in query for query in queries)
    assert any(
        'CREATE TABLE IF NOT EXISTS "claim_cluster_membership"' in query
        for query in queries
    )

    analytics_export_table = next(
        query
        for query in queries
        if 'CREATE TABLE IF NOT EXISTS "claims_analytics_exports"' in query
    )
    assert "job_id BIGINT" in analytics_export_table
    assert "error_code TEXT" in analytics_export_table
    assert "snapshot_at TIMESTAMPTZ" in analytics_export_table
    assert "snapshot_event_id BIGINT" in analytics_export_table
    assert "job_status" not in analytics_export_table
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_claims_analytics_exports_job_id" '
        'ON "claims_analytics_exports" ("job_id")' in query
        for query in queries
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_claims_analytics_exports_user_status_export_id" '
        'ON "claims_analytics_exports" ("user_id", "status", "export_id")' in query
        for query in queries
    )
    assert any(
        'CREATE INDEX IF NOT EXISTS "idx_claims_analytics_exports_user_status_updated_export_id" '
        'ON "claims_analytics_exports" '
        '("user_id", "status", "updated_at", "export_id")' in query
        for query in queries
    )
    assert not any(
        'ALTER COLUMN "job_id" TYPE' in query
        or 'ALTER COLUMN "snapshot_at" TYPE' in query
        for query in queries
    )


def test_ensure_postgres_collections_tables_swallows_backend_errors() -> None:
    helper_module = _load_postgres_claims_collection_structures_module()

    conn = object()

    class FakeBackend:
        def execute(self, query: str, *, connection) -> None:
            raise BackendDatabaseError("boom")

    db = SimpleNamespace(backend=FakeBackend())

    helper_module.ensure_postgres_collections_tables(db, conn)
