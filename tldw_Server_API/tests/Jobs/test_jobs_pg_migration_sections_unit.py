"""SQL and failure contracts for focused PostgreSQL migration sections."""

import pytest

from tldw_Server_API.app.core.Jobs import pg_migrations

pytestmark = pytest.mark.unit


class _RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str) -> None:
        self.statements.append(" ".join(sql.split()))


def test_attachment_schema_creates_table_before_index() -> None:
    cursor = _RecordingCursor()
    pg_migrations._ensure_pg_job_attachments_schema(cursor)
    assert cursor.statements == [
        "CREATE TABLE IF NOT EXISTS job_attachments ( id SERIAL PRIMARY KEY, "
        "job_id INTEGER NOT NULL, kind TEXT NOT NULL, content_text TEXT, url TEXT, "
        "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW() );",
        "CREATE INDEX IF NOT EXISTS idx_job_attachments_job ON job_attachments(job_id)",
    ]


def test_attachment_schema_failure_propagates_to_the_callers_transaction() -> None:
    class _FailingCursor:
        def execute(self, _sql: str) -> None:
            raise RuntimeError("attachment DDL failed")

    with pytest.raises(RuntimeError, match="attachment DDL failed"):
        pg_migrations._ensure_pg_job_attachments_schema(_FailingCursor())


def test_receipt_policies_preserve_read_write_scope_and_statement_order() -> None:
    cursor = _RecordingCursor()
    pg_migrations._ensure_pg_idempotency_receipt_rls(
        cursor, admin_expr="admin_claim", domain_filter="allowed_domain", owner_filter="owned_row"
    )
    assert cursor.statements == [
        "DROP POLICY IF EXISTS job_idempotency_receipts_select ON job_idempotency_receipts",
        "CREATE POLICY job_idempotency_receipts_select ON job_idempotency_receipts FOR SELECT "
        "USING ( admin_claim OR ( allowed_domain AND owned_row ) )",
        "DROP POLICY IF EXISTS job_idempotency_receipts_modify ON job_idempotency_receipts",
        "CREATE POLICY job_idempotency_receipts_modify ON job_idempotency_receipts FOR ALL "
        "USING ( admin_claim OR ( allowed_domain AND owned_row ) ) "
        "WITH CHECK ( admin_claim OR ( allowed_domain AND owned_row ) )",
    ]


def test_receipt_policy_failure_propagates_to_installer_error_boundary() -> None:
    class _FailingCursor:
        def execute(self, _sql: str) -> None:
            raise RuntimeError("receipt policy failed")

    with pytest.raises(RuntimeError, match="receipt policy failed"):
        pg_migrations._ensure_pg_idempotency_receipt_rls(
            _FailingCursor(), admin_expr="admin", domain_filter="domain", owner_filter="owner"
        )
