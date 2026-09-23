"""The core chat policies must isolate accounts on a real PostgreSQL server.

Contract tests elsewhere assert the generated SQL contains the right text. This
runs it: a live server, the policies installed, a non-bypassing role connected,
and two accounts' conversations and messages sharing the same tables.

Everything is created inside a uniquely named schema that is dropped afterwards,
and the unprivileged role is per-run and dropped too, so nothing outside this
test is touched even when CHAT_RLS_DB_URL points at a reused database.

    docker run -d --rm -e POSTGRES_PASSWORD=pw -p 55436:5432 postgres:18-bookworm
    CHAT_RLS_DB_URL=postgresql://postgres:pw@127.0.0.1:55436/postgres
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    build_core_chat_rls_sql,
)

psycopg = pytest.importorskip("psycopg")

pytestmark = pytest.mark.integration

ROLE_PASSWORD = "probe"  # noqa: S105 - throwaway role in a throwaway schema


def _dsn_or_skip() -> str:
    """Return the DSN for a PostgreSQL server, or skip when none is configured."""
    dsn = os.getenv("CHAT_RLS_DB_URL")
    if not dsn:
        pytest.skip("CHAT_RLS_DB_URL not configured for core chat RLS tests")
    return dsn


def _with_role(dsn: str, role: str) -> str:
    """Rewrite a DSN's credentials to connect as the unprivileged test role."""
    scheme, _, rest = dsn.partition("://")
    _, _, hostpart = rest.partition("@")
    return f"{scheme}://{role}:{ROLE_PASSWORD}@{hostpart}"


class _Fixture:
    """Connection details for one isolated run: its schema, role and DSN."""

    def __init__(self, dsn: str, schema: str, role: str) -> None:
        self.dsn = dsn
        self.schema = schema
        self.role = role

    def connect_as(self, tenant: str) -> "psycopg.Connection":
        """Open a connection as the unprivileged role with a tenant bound."""
        conn = psycopg.connect(self.dsn, autocommit=True)
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema}")
            cur.execute("SELECT set_config('app.current_user_id', %s, false)", (tenant,))
        return conn


@pytest.fixture()
def seeded() -> Iterator[_Fixture]:
    """Build the chat tables in a private schema, seed two accounts, install RLS."""
    admin_dsn = _dsn_or_skip()
    suffix = uuid.uuid4().hex[:10]
    schema = f"chat_rls_{suffix}"
    role = f"chat_rls_role_{suffix}"

    with psycopg.connect(admin_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        cur.execute(
            "CREATE TABLE conversations (id TEXT PRIMARY KEY, client_id TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL "
            "REFERENCES conversations(id), client_id TEXT NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE sync_log (change_id SERIAL PRIMARY KEY, "
            "entity TEXT NOT NULL, client_id TEXT NOT NULL)"
        )
        cur.execute("INSERT INTO conversations VALUES ('c-alice','alice'),('c-bob','bob')")
        cur.execute(
            "INSERT INTO messages VALUES ('m-alice','c-alice','alice'),"
            "('m-bob','c-bob','bob')"
        )
        for statement in build_core_chat_rls_sql():
            if any(t in statement for t in ("conversations", "messages", "sync_log")):
                cur.execute(statement)
        cur.execute(
            f"CREATE ROLE {role} LOGIN PASSWORD '{ROLE_PASSWORD}' "
            "NOSUPERUSER NOBYPASSRLS"
        )
        cur.execute(f"GRANT USAGE ON SCHEMA {schema} TO {role}")
        cur.execute(
            f"GRANT SELECT, INSERT ON conversations, messages, sync_log TO {role}"
        )
        cur.execute(f"GRANT USAGE, SELECT ON SEQUENCE sync_log_change_id_seq TO {role}")

    try:
        yield _Fixture(_with_role(admin_dsn, role), schema, role)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            cur.execute(f"DROP OWNED BY {role}")
            cur.execute(f"DROP ROLE IF EXISTS {role}")


def test_an_account_sees_only_its_own_conversations_and_messages(
    seeded: _Fixture,
) -> None:
    """The regression: both accounts' chat history shared one unpoliced table."""
    with seeded.connect_as("alice") as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM conversations ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["c-alice"]
        cur.execute("SELECT id FROM messages ORDER BY id")
        assert [r[0] for r in cur.fetchall()] == ["m-alice"]


def test_no_tenant_context_sees_nothing(seeded: _Fixture) -> None:
    """Missing context must mean no rows, matching the Jobs predicates."""
    with seeded.connect_as("") as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM conversations")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM messages")
        assert cur.fetchone()[0] == 0


def test_a_message_cannot_be_written_into_another_accounts_conversation(
    seeded: _Fixture,
) -> None:
    """What the parent check buys: USING alone would allow this write."""
    with seeded.connect_as("bob") as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("INSERT INTO messages VALUES ('m-x','c-alice','bob')")


def test_an_account_can_still_write_its_own_message(seeded: _Fixture) -> None:
    """Isolation must not cost the account its own writes."""
    with seeded.connect_as("bob") as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO messages VALUES ('m-bob2','c-bob','bob')")
        cur.execute("SELECT count(*) FROM messages")
        assert cur.fetchone()[0] == 2


def test_sync_log_rows_written_by_triggers_still_pass_the_check(
    seeded: _Fixture,
) -> None:
    """The risk this policy introduces, checked rather than assumed.

    ChaCha writes sync_log from triggers that copy NEW.client_id off the row
    being changed. That row has already satisfied its own table's WITH CHECK, so
    its client_id equals the session tenant and the sync_log insert passes too.
    A row carrying a different client_id is refused, which is the point.
    """
    with seeded.connect_as("alice") as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO sync_log (entity, client_id) VALUES ('messages','alice')")
        cur.execute("SELECT count(*) FROM sync_log")
        assert cur.fetchone()[0] == 1

        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute("INSERT INTO sync_log (entity, client_id) VALUES ('messages','bob')")
