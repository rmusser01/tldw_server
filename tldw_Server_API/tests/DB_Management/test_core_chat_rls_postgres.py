"""The core chat policies must isolate accounts on a real PostgreSQL server.

Contract tests elsewhere assert the generated SQL contains the right text.
This runs it: a live server, RLS installed, a non-bypassing role connected, and
two accounts' conversations and messages in the same tables.

Needs CHAT_RLS_DB_URL pointing at a database the test may reshape, e.g.

    docker run -d --rm -e POSTGRES_PASSWORD=pw -p 55436:5432 postgres:18-bookworm
    CHAT_RLS_DB_URL=postgresql://postgres:pw@127.0.0.1:55436/postgres
"""

import os

import pytest

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    build_core_chat_rls_sql,
)

psycopg = pytest.importorskip("psycopg")

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

ROLE = "chat_rls_probe"
ROLE_PASSWORD = "probe"  # noqa: S105 - throwaway role in a disposable database


def _dsn_or_skip() -> str:
    dsn = os.getenv("CHAT_RLS_DB_URL")
    if not dsn:
        pytest.skip("CHAT_RLS_DB_URL not configured for core chat RLS tests")
    return dsn


@pytest.fixture()
def seeded(request) -> str:
    """Build the two tables, seed two accounts, install RLS, return an RLS DSN."""
    admin_dsn = _dsn_or_skip()
    with psycopg.connect(admin_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS messages, conversations CASCADE")
        cur.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, client_id TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL "
            "REFERENCES conversations(id), client_id TEXT NOT NULL)"
        )
        cur.execute("INSERT INTO conversations VALUES ('c-alice','alice'),('c-bob','bob')")
        cur.execute(
            "INSERT INTO messages VALUES ('m-alice','c-alice','alice'),"
            "('m-bob','c-bob','bob')"
        )
        for statement in build_core_chat_rls_sql():
            if "conversations" in statement or "messages" in statement:
                try:
                    cur.execute(statement)
                except psycopg.Error as exc:  # tables outside this fixture
                    if "does not exist" not in str(exc):
                        raise
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,))
        if not cur.fetchone():
            cur.execute(
                f"CREATE ROLE {ROLE} LOGIN PASSWORD '{ROLE_PASSWORD}' "
                "NOSUPERUSER NOBYPASSRLS"
            )
        cur.execute(f"GRANT SELECT, INSERT ON conversations, messages TO {ROLE}")

    scheme, _, rest = admin_dsn.partition("://")
    _, _, hostpart = rest.partition("@")
    return f"{scheme}://{ROLE}:{ROLE_PASSWORD}@{hostpart}"


def _as(dsn: str, tenant: str):
    conn = psycopg.connect(dsn, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('app.current_user_id', %s, false)", (tenant,))
    return conn


def test_an_account_sees_only_its_own_conversations_and_messages(seeded):
    """The regression: both accounts' chat history shared one unpoliced table."""
    with _as(seeded, "alice") as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM conversations ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["c-alice"]
        cur.execute("SELECT id FROM messages ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["m-alice"]


def test_no_tenant_context_sees_nothing(seeded):
    """Missing context must mean no rows, matching the Jobs predicates."""
    with _as(seeded, "") as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM conversations")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM messages")
        assert cur.fetchone()[0] == 0


def test_a_message_cannot_be_written_into_another_accounts_conversation(seeded):
    """What the parent check in the policy buys: USING alone would allow this."""
    with _as(seeded, "bob") as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("INSERT INTO messages VALUES ('m-x','c-alice','bob')")


def test_sync_log_rows_written_by_triggers_still_pass_the_check(seeded):
    """The risk this policy introduces, checked rather than assumed.

    ChaCha writes sync_log from triggers that copy NEW.client_id off the row
    being changed. That row has already satisfied its own table's WITH CHECK,
    so its client_id equals the session tenant and the sync_log insert passes
    too. A row carrying a different client_id is refused, which is the point.
    """
    admin_dsn = _dsn_or_skip()
    with psycopg.connect(admin_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS sync_log CASCADE")
        cur.execute(
            "CREATE TABLE sync_log (change_id SERIAL PRIMARY KEY, "
            "entity TEXT NOT NULL, client_id TEXT NOT NULL)"
        )
        for statement in build_core_chat_rls_sql():
            if "sync_log" in statement:
                cur.execute(statement)
        cur.execute(f"GRANT SELECT, INSERT ON sync_log TO {ROLE}")
        cur.execute(f"GRANT USAGE, SELECT ON SEQUENCE sync_log_change_id_seq TO {ROLE}")

    with _as(seeded, "alice") as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO sync_log (entity, client_id) VALUES ('messages','alice')")
        cur.execute("SELECT count(*) FROM sync_log")
        assert cur.fetchone()[0] == 1

        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("INSERT INTO sync_log (entity, client_id) VALUES ('messages','bob')")


def test_an_account_can_still_write_its_own_message(seeded):
    """Isolation must not cost the account its own writes."""
    with _as(seeded, "bob") as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO messages VALUES ('m-bob2','c-bob','bob')")
        cur.execute("SELECT count(*) FROM messages")
        assert cur.fetchone()[0] == 2
