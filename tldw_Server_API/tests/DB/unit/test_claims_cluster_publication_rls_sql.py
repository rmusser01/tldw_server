"""Publication policy contracts without a PostgreSQL connection."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_chacha_rls_sql

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("table", ("claims_cluster_publication_heads", "claims_cluster_publication_receipts"))
def test_publication_policy_is_guarded_forced_and_owner_only(table: str) -> None:
    statements = [sql for sql in build_chacha_rls_sql() if f"ON {table} FOR ALL" in sql]
    expected = f"""
        DO $claims_cluster_publication_rls$
        BEGIN
          IF to_regclass('{table}') IS NULL THEN RETURN; END IF;
          ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
          ALTER TABLE {table} FORCE ROW LEVEL SECURITY;
          DROP POLICY IF EXISTS {table}_owner ON {table};
          CREATE POLICY {table}_owner ON {table} FOR ALL
            USING (owner_user_id = NULLIF(current_setting('app.current_user_id', true), ''))
            WITH CHECK (owner_user_id = NULLIF(current_setting('app.current_user_id', true), ''));
        END
        $claims_cluster_publication_rls$;
    """
    assert [" ".join(sql.split()) for sql in statements] == [" ".join(expected.split())]


def test_publication_policies_remain_between_study_and_notes_policies() -> None:
    statements = build_chacha_rls_sql()
    publication_indexes = [
        index for index, sql in enumerate(statements) if "DO $claims_cluster_publication_rls$" in sql
    ]
    first, last = publication_indexes
    assert "ON flashcard_citations" in statements[first - 1]
    assert last == first + 1
    assert statements[last + 1] == "ALTER TABLE IF EXISTS notes ENABLE ROW LEVEL SECURITY;"
