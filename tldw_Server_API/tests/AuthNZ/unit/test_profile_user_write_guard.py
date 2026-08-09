from __future__ import annotations

import copy
import tracemalloc
from dataclasses import FrozenInstanceError
from statistics import mean, quantiles
from time import perf_counter

import pytest
from sqlglot import logger as sqlglot_logger

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    ProfileUserWriteRejected,
    _active_capability_count,
    _classification_cache_clear,
    _classification_cache_info,
    _classify_sql,
    _guard_sql,
    _is_canonical_users_bootstrap_sql,
    _mint_profile_user_sql,
    _ProfileUserSql,
    _revoke_profile_user_sql,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("backend", "statement"),
    [
        (
            "sqlite",
            "INSERT INTO users (username, email) VALUES (?, ?)",
        ),
        (
            "sqlite",
            "INSERT OR REPLACE INTO users (id, email) VALUES (?, ?)",
        ),
        ("sqlite", "UPDATE users SET email = ? WHERE id = ?"),
        ("sqlite", 'UPDATE "users" SET "is_active" = 0 WHERE id = 1'),
        ("postgres", "UPDATE auth.users AS u SET role = $1 WHERE u.id = $2"),
        (
            "postgres",
            "WITH changed AS (UPDATE users SET email = $1 RETURNING id) "
            "SELECT id FROM changed",
        ),
        ("postgres", "COPY users (email) FROM STDIN"),
        ("postgres", "PREPARE profile_write AS UPDATE users SET email = $1"),
        ("postgres", "DO $$ BEGIN UPDATE users SET email = 'x'; END $$"),
        ("sqlite", "PRAGMA writable_schema=1"),
        ("sqlite", "PRAGMA writable_schema(1)"),
        ("sqlite", "UPDATE sqlite_master SET sql = ? WHERE name = 'users'"),
        ("sqlite", "INSERT INTO sqlite_schema VALUES (?, ?, ?, ?, ?)"),
        ("sqlite", "DELETE FROM sqlite_temp_master WHERE name = 'users'"),
        ("sqlite", "DELETE FROM users WHERE id = 1"),
        ("sqlite", "UPDATE users SET profile_version = ? WHERE id = ?"),
        ("postgres", "TRUNCATE TABLE users"),
        ("postgres", "DROP TABLE users"),
        ("postgres", "ALTER TABLE users DROP COLUMN email"),
        ("postgres", "ALTER TABLE users RENAME COLUMN email TO contact_email"),
        ("postgres", "ALTER TABLE users ALTER COLUMN email TYPE TEXT"),
        (
            "postgres",
            "ALTER TABLE users ALTER COLUMN profile_version DROP NOT NULL",
        ),
        (
            "postgres",
            "ALTER TABLE public.users ADD COLUMN profile_version TIMESTAMPTZ",
        ),
        (
            "postgres",
            "ALTER TABLE public.users ALTER COLUMN profile_version "
            "SET DEFAULT CURRENT_TIMESTAMP",
        ),
        ("postgres", "DROP SCHEMA public CASCADE"),
        ("postgres", "DROP DATABASE authnz"),
        ("postgres", "TRUNCATE TABLE api_keys CASCADE"),
        ("postgres", "CREATE TEMP TABLE users (id BIGINT PRIMARY KEY)"),
        ("postgres", "CREATE TEMP VIEW users AS SELECT 1 AS id"),
        (
            "postgres",
            "CREATE VIEW public.user_alias AS SELECT id, email FROM public.users",
        ),
        (
            "postgres",
            "CREATE MATERIALIZED VIEW public.user_snapshot AS SELECT id FROM public.users",
        ),
        ("postgres", "SET search_path TO pg_temp, public"),
        ("sqlite", "CREATE TEMP TABLE users (id INTEGER PRIMARY KEY)"),
        ("sqlite", "CREATE TEMP VIEW users AS SELECT 1 AS id"),
    ],
)
def test_raw_protected_users_writes_fail_closed(
    backend: str,
    statement: str,
) -> None:
    with pytest.raises(
        ProfileUserWriteRejected,
        match="Profile-visible AuthNZ users write rejected",
    ):
        _guard_sql(
            statement,
            backend=backend,
            connection_identity=object(),
            operation="execute",
        )


@pytest.mark.parametrize(
    ("backend", "statement"),
    [
        ("sqlite", "DELETE FROM org_members WHERE org_id = ?"),
        ("sqlite", "UPDATE team_members SET role = ? WHERE team_id = ?"),
        (
            "postgres",
            "INSERT INTO public.org_members (org_id, user_id, role) "
            "VALUES ($1, $2, $3)",
        ),
        ("postgres", "DELETE FROM public.team_members WHERE team_id = $1"),
        ("sqlite", "DELETE FROM organizations WHERE id = ?"),
        ("postgres", "DELETE FROM public.teams WHERE id = $1"),
        (
            "postgres",
            "COPY public.org_members (org_id, user_id, role) FROM STDIN",
        ),
        (
            "postgres",
            "COPY public.team_members (team_id, user_id, role) FROM STDIN",
        ),
    ],
)
def test_raw_membership_and_scope_deletion_writes_fail_closed(
    backend: str,
    statement: str,
) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend=backend,
            connection_identity=object(),
            operation="execute",
        )


@pytest.mark.parametrize(
    ("backend", "statement"),
    [
        ("sqlite", "DELETE FROM main.org_members WHERE org_id = ?"),
        ("postgres", "DELETE FROM public.team_members WHERE team_id = $1"),
        ("sqlite", "DELETE FROM main.organizations WHERE id = ?"),
        ("postgres", "DELETE FROM public.teams WHERE id = $1"),
    ],
)
def test_membership_scope_capability_is_connection_bound_and_consumed_once(
    backend: str,
    statement: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
        _mint_membership_scope_sql,
    )

    connection = object()
    capability = _mint_membership_scope_sql(
        statement,
        backend=backend,
        connection_identity=connection,
        execution_mode="execute",
    )

    assert (
        _guard_sql(
            capability,
            backend=backend,
            connection_identity=connection,
            operation="execute",
        )
        == statement
    )
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            capability,
            backend=backend,
            connection_identity=connection,
            operation="execute",
        )


@pytest.mark.parametrize(
    ("backend", "statement"),
    [
        ("sqlite", "SELECT id, email FROM users WHERE id = ?"),
        ("sqlite", "UPDATE users SET password_hash = ? WHERE id = ?"),
        ("sqlite", "UPDATE users SET totp_secret = ? WHERE id = ?"),
        ("sqlite", "UPDATE users SET backup_codes = ? WHERE id = ?"),
        (
            "postgres",
            "UPDATE users SET failed_login_attempts = $1, locked_until = $2 "
            "WHERE id = $3",
        ),
        ("postgres", "ALTER TABLE users ADD COLUMN nickname TEXT"),
        ("postgres", "COPY users TO STDOUT"),
        ("postgres", "COPY public.org_members TO STDOUT"),
        ("postgres", "COPY team_members TO STDOUT"),
        ("postgres", "UPDATE api_keys SET status = $1 WHERE id = $2"),
    ],
)
def test_unprotected_concrete_sql_is_returned_unchanged(
    backend: str,
    statement: str,
) -> None:
    assert (
        _guard_sql(
            statement,
            backend=backend,
            connection_identity=object(),
            operation="execute",
        )
        == statement
    )


def test_unknown_users_update_column_fails_closed() -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            "UPDATE users SET future_profile_field = ? WHERE id = ?",
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )


@pytest.mark.parametrize(
    "statement",
    [
        """
        CREATE FUNCTION rewrite_user() RETURNS void LANGUAGE SQL AS $$
            UPDATE users SET email = 'sql-function@example.com' WHERE id = 1
        $$
        """,
        """
        CREATE OR REPLACE FUNCTION rewrite_user() RETURNS void
        LANGUAGE plpgsql AS $$
        BEGIN
            UPDATE users SET email = 'plpgsql-function@example.com' WHERE id = 1;
        END
        $$
        """,
        """
        CREATE PROCEDURE rewrite_user() LANGUAGE SQL AS $$
            UPDATE users SET email = 'sql-procedure@example.com' WHERE id = 1
        $$
        """,
        """
        CREATE OR REPLACE PROCEDURE rewrite_user() LANGUAGE plpgsql AS $$
        BEGIN
            UPDATE users SET email = 'plpgsql-procedure@example.com' WHERE id = 1;
        END
        $$
        """,
        """
        CREATE TRIGGER rewrite_user
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION rewrite_user()
        """,
        """
        CREATE RULE rewrite_user AS ON UPDATE TO api_keys
        DO ALSO UPDATE users SET email = 'rule@example.com' WHERE id = NEW.user_id
        """,
    ],
)
def test_postgres_routine_creation_fails_closed(statement: str) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )


def test_sqlite_users_writing_trigger_creation_fails_closed() -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            """
            CREATE TRIGGER rewrite_user AFTER UPDATE ON api_keys
            BEGIN
                UPDATE users SET email = 'trigger@example.com' WHERE id = NEW.user_id;
            END
            """,
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE api_keys_new (id BIGINT PRIMARY KEY)",
        "CREATE INDEX api_keys_new_id_idx ON api_keys_new (id)",
    ],
)
def test_postgres_ordinary_table_and_index_ddl_remains_allowed(
    statement: str,
) -> None:
    assert _guard_sql(
        statement,
        backend="postgres",
        connection_identity=object(),
        operation="execute",
    ) == statement


def test_existing_non_profile_email_verified_update_remains_raw_safe() -> None:
    statement = "UPDATE users SET email_verified = ?, updated_at = ? WHERE id = ?"

    assert _guard_sql(
        statement,
        backend="sqlite",
        connection_identity=object(),
        operation="execute",
    ) == statement


def test_safe_users_update_preserves_exact_classification_metadata() -> None:
    classification = _classify_sql(
        "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
        "sqlite",
    )

    assert classification.operation == "update"
    assert classification.columns == ("password_hash", "updated_at")
    assert classification.protected is False


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT 1; SELECT 2",
        "SELECT 1; UPDATE users SET email = 'x' WHERE id = 1",
        "UPDATE users SET email = 'x' WHERE id = 1; SELECT 1",
        "THIS IS NOT SQL",
    ],
)
def test_ambiguous_or_multi_statement_sql_fails_closed(statement: str) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )


def test_tokenizer_failure_is_sanitized_without_literal_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "AUTHNZ_TOKEN_SECRET_SENTINEL_81f6"
    messages: list[str] = []

    def _capture(message: object, *args: object, **_kwargs: object) -> None:
        messages.append(" ".join(str(item) for item in (message, *args)))

    monkeypatch.setattr(sqlglot_logger, "warning", _capture)
    monkeypatch.setattr(sqlglot_logger, "error", _capture)
    with pytest.raises(ProfileUserWriteRejected) as raised:
        _guard_sql(
            f"SELECT '{sentinel}",
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )

    assert str(raised.value) == "Profile-visible AuthNZ users write rejected"
    assert raised.value.__cause__ is None
    assert sentinel not in "\n".join(messages)


def test_rejected_unsupported_sql_does_not_log_literal_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "AUTHNZ_SQL_SECRET_SENTINEL_6e41"
    messages: list[str] = []

    def _capture(message: object, *args: object, **_kwargs: object) -> None:
        messages.append(" ".join(str(item) for item in (message, *args)))

    monkeypatch.setattr(sqlglot_logger, "warning", _capture)
    monkeypatch.setattr(sqlglot_logger, "error", _capture)
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            f"PREPARE profile_write AS UPDATE users SET email = '{sentinel}'",
            backend="postgres",
            connection_identity=object(),
            operation="prepare",
        )

    assert messages
    assert sentinel not in "\n".join(messages)


def test_oversized_sql_fails_closed() -> None:
    statement = "SELECT '" + ("x" * (16 * 1024)) + "'"

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend="sqlite",
            connection_identity=object(),
            operation="execute",
        )


def test_large_sql_is_rejected_before_allocating_a_utf8_copy() -> None:
    statement = "SELECT '" + ("x" * (2 * 1024 * 1024)) + "'"

    tracemalloc.start()
    try:
        with pytest.raises(ProfileUserWriteRejected):
            _guard_sql(
                statement,
                backend="sqlite",
                connection_identity=object(),
                operation="execute",
            )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak_bytes < 256 * 1024


def test_classifier_cache_is_bounded_and_deterministic() -> None:
    _classification_cache_clear()
    connection = object()

    for value in range(4097):
        statement = f"SELECT {value} FROM api_keys"
        assert (
            _guard_sql(
                statement,
                backend="sqlite",
                connection_identity=connection,
                operation="execute",
            )
            == statement
        )

    cache_info = _classification_cache_info()
    assert cache_info.maxsize == 4096
    assert cache_info.currsize == 4096
    assert cache_info.misses == 4097


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE EXTENSION IF NOT EXISTS pgcrypto",
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
    ],
)
def test_uuid_extension_bootstrap_is_narrowly_allowed(statement: str) -> None:
    assert (
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )
        == statement
    )


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE EXTENSION IF NOT EXISTS hstore",
        "CREATE EXTENSION pgcrypto",
        "CREATE EXTENSION IF NOT EXISTS pgcrypto CASCADE",
        "CREATE EXTENSION IF NOT EXISTS pgcrypto; SELECT 1",
        'CREATE EXTENSION IF NOT EXISTS "uuid-oſſp"',
    ],
)
def test_other_extension_commands_remain_rejected(statement: str) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )


def test_sharing_constraint_validation_is_narrowly_allowed() -> None:
    statement = (
        "ALTER TABLE share_tokens "
        "VALIDATE CONSTRAINT ck_share_tokens_resource_type"
    )

    assert (
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )
        == statement
    )


@pytest.mark.parametrize(
    ("backend", "statement"),
    [
        (
            "postgres",
            "ALTER TABLE users "
            "VALIDATE CONSTRAINT ck_share_tokens_resource_type",
        ),
        (
            "postgres",
            "ALTER TABLE share_tokens VALIDATE CONSTRAINT other_constraint",
        ),
        (
            "postgres",
            "ALTER TABLE share_tokens "
            "VALIDATE CONSTRAINT ck_share_tokens_resource_type; SELECT 1",
        ),
        (
            "sqlite",
            "ALTER TABLE share_tokens "
            "VALIDATE CONSTRAINT ck_share_tokens_resource_type",
        ),
        (
            "postgres",
            "ALTER TABLE ſhare_tokens "
            "VALIDATE CONSTRAINT ck_share_tokens_resource_type",
        ),
        (
            "postgres",
            "ALTER TABLE share_tokens "
            "VALIDATE CONSTRAINT ck_ſhare_tokens_resource_type",
        ),
    ],
)
def test_other_constraint_validation_commands_remain_rejected(
    backend: str,
    statement: str,
) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend=backend,
            connection_identity=object(),
            operation="execute",
        )


def test_classifier_cache_meets_miss_and_hit_latency_budgets() -> None:
    _classification_cache_clear()
    connection = object()
    statements = tuple(
        f"SELECT id FROM api_keys WHERE id = {value}"
        for value in range(128)
    )

    miss_samples: list[float] = []
    for statement in statements:
        started = perf_counter()
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=connection,
            operation="execute",
        )
        miss_samples.append(perf_counter() - started)

    hit_samples: list[float] = []
    for _ in range(16):
        for statement in statements:
            started = perf_counter()
            _guard_sql(
                statement,
                backend="postgres",
                connection_identity=connection,
                operation="execute",
            )
            hit_samples.append(perf_counter() - started)

    miss_p95 = quantiles(miss_samples, n=100, method="inclusive")[94]
    hit_p95 = quantiles(hit_samples, n=100, method="inclusive")[94]
    assert miss_p95 < 0.002
    assert mean(hit_samples) < 0.00005
    assert hit_p95 < 0.00005


def test_profile_user_capability_is_frozen_and_consumed_once() -> None:
    connection = object()
    statement = "UPDATE users SET email = ? WHERE id = ?"
    capability = _mint_profile_user_sql(
        statement,
        backend="sqlite",
        connection_identity=connection,
        operation="update",
        columns=("email",),
    )

    with pytest.raises(FrozenInstanceError):
        capability.operation = "insert"
    assert (
        _guard_sql(
            capability,
            backend="sqlite",
            connection_identity=connection,
            operation="execute",
        )
        == statement
    )
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            capability,
            backend="sqlite",
            connection_identity=connection,
            operation="execute",
        )


@pytest.mark.parametrize(
    "statement",
    [
        "ALTER TABLE public.users ADD COLUMN profile_version TIMESTAMPTZ",
        (
            "ALTER TABLE public.users ALTER COLUMN profile_version "
            "SET DEFAULT CURRENT_TIMESTAMP"
        ),
        "ALTER TABLE public.users ALTER COLUMN profile_version SET NOT NULL",
    ],
)
def test_profile_anchor_ddl_requires_one_shot_capability(statement: str) -> None:
    connection = object()
    capability = _mint_profile_user_sql(
        statement,
        backend="postgres",
        connection_identity=connection,
        operation="alter",
        columns=(),
    )

    assert _guard_sql(
        capability,
        backend="postgres",
        connection_identity=connection,
        operation="execute",
    ) == statement
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            capability,
            backend="postgres",
            connection_identity=connection,
            operation="execute",
        )


def test_users_bootstrap_requires_exact_one_shot_capability() -> None:
    connection = object()
    statement = """
        CREATE TABLE IF NOT EXISTS public.users (
            id SERIAL PRIMARY KEY,
            uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            role VARCHAR(50) NOT NULL DEFAULT 'user',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            is_verified BOOLEAN NOT NULL DEFAULT FALSE,
            is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
            email_verified BOOLEAN NOT NULL DEFAULT FALSE,
            two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            failed_login_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TIMESTAMPTZ,
            storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
            storage_used_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMPTZ,
            email_verified_at TIMESTAMPTZ,
            two_factor_secret TEXT,
            totp_secret TEXT,
            backup_codes TEXT,
            created_by INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
            password_changed_at TIMESTAMPTZ
        )
    """

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=connection,
            operation="execute",
        )

    capability = _mint_profile_user_sql(
        statement,
        backend="postgres",
        connection_identity=connection,
        operation="create",
        columns=(),
    )
    assert _guard_sql(
        capability,
        backend="postgres",
        connection_identity=connection,
        operation="execute",
    ) == statement
    assert not _is_canonical_users_bootstrap_sql(
        statement.replace("public.users", "public.user_archive", 1),
        backend="postgres",
    )
    assert not _is_canonical_users_bootstrap_sql(
        statement.replace("public.users", "tenant.public.users", 1),
        backend="postgres",
    )

    invalid_statements = (
        statement.replace("id SERIAL PRIMARY KEY", "id BIGINT PRIMARY KEY", 1),
        statement.replace(
            "metadata JSONB DEFAULT '{}'::jsonb",
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
            1,
        ),
        statement.replace(
            "password_changed_at TIMESTAMPTZ",
            "password_changed_at TIMESTAMPTZ, tenant_id BIGINT NOT NULL",
            1,
        ),
    )
    for invalid_statement in invalid_statements:
        invalid_capability = None
        try:
            invalid_capability = _mint_profile_user_sql(
                invalid_statement,
                backend="postgres",
                connection_identity=object(),
                operation="create",
                columns=(),
            )
        except ProfileUserWriteRejected:
            pass
        finally:
            if invalid_capability is not None:
                _revoke_profile_user_sql(invalid_capability)
        assert invalid_capability is None


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE IF NOT EXISTS public.users AS SELECT 1 AS id",
        "CREATE TABLE IF NOT EXISTS public.users (id BIGINT PRIMARY KEY)",
        """
        CREATE TABLE IF NOT EXISTS public.users (
            id SERIAL PRIMARY KEY,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ],
)
def test_users_bootstrap_capability_rejects_incomplete_shapes(statement: str) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _mint_profile_user_sql(
            statement,
            backend="postgres",
            connection_identity=object(),
            operation="create",
            columns=(),
        )


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE public.user_alias (extra INTEGER) INHERITS (public.users)",
        "CREATE TABLE public.user_partition PARTITION OF public.users "
        "FOR VALUES FROM (1) TO (10)",
    ],
)
def test_users_inheritance_and_partition_ddl_is_rejected(statement: str) -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            statement,
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )


@pytest.mark.parametrize(
    "definition",
    [
        "role VARCHAR(50) NOT NULL DEFAULT 'admin'",
        "is_active BOOLEAN NOT NULL DEFAULT FALSE",
        "is_verified BOOLEAN NOT NULL DEFAULT TRUE",
        "profile_version TEXT NOT NULL DEFAULT 'invalid'",
        "profile_version TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP",
        "last_login TEXT",
    ],
)
def test_users_bootstrap_capability_rejects_noncanonical_column_contracts(
    definition: str,
) -> None:
    definitions = {
        "role": "role VARCHAR(50) NOT NULL DEFAULT 'user'",
        "is_active": "is_active BOOLEAN NOT NULL DEFAULT TRUE",
        "is_verified": "is_verified BOOLEAN NOT NULL DEFAULT FALSE",
        "profile_version": (
            "profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ),
        "last_login": "last_login TIMESTAMPTZ",
    }
    definitions[definition.split(maxsplit=1)[0]] = definition
    statement = f"""
        CREATE TABLE IF NOT EXISTS public.users (
            id SERIAL PRIMARY KEY,
            uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}'::jsonb,
            {definitions['role']},
            {definitions['is_active']},
            {definitions['is_verified']},
            is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
            email_verified BOOLEAN NOT NULL DEFAULT FALSE,
            two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            failed_login_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TIMESTAMPTZ,
            storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
            storage_used_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            {definitions['profile_version']},
            {definitions['last_login']},
            email_verified_at TIMESTAMPTZ,
            two_factor_secret TEXT,
            totp_secret TEXT,
            backup_codes TEXT,
            created_by INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
            password_changed_at TIMESTAMPTZ
        )
    """

    capability = None
    try:
        with pytest.raises(ProfileUserWriteRejected):
            capability = _mint_profile_user_sql(
                statement,
                backend="postgres",
                connection_identity=object(),
                operation="create",
                columns=(),
            )
    finally:
        if capability is not None:
            _revoke_profile_user_sql(capability)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda sql: sql.replace(
            "is_superuser BOOLEAN NOT NULL DEFAULT FALSE,", "", 1
        ),
        lambda sql: sql.replace(" DEFAULT gen_random_uuid()", "", 1),
        lambda sql: sql.replace("locked_until TIMESTAMPTZ", "locked_until TIMESTAMP", 1),
        lambda sql: sql.replace("last_login TIMESTAMPTZ", "last_login TIMESTAMP", 1),
        lambda sql: sql.replace(
            "created_by INTEGER REFERENCES public.users(id) ON DELETE SET NULL",
            "created_by INTEGER",
            1,
        ),
        lambda sql: sql.replace("ON DELETE SET NULL", "ON DELETE CASCADE", 1),
        lambda sql: sql.replace(
            "password_hash TEXT NOT NULL",
            "password_hash TEXT NOT NULL CHECK (length(password_hash) > 0)",
            1,
        ),
        lambda sql: sql.replace(
            "metadata JSONB DEFAULT '{}'::jsonb",
            "metadata JSONB DEFAULT '{}'::jsonb "
            "GENERATED ALWAYS AS ('{}'::jsonb) STORED",
            1,
        ),
        lambda sql: sql.replace("username VARCHAR(255)", "username VARCHAR(1)", 1),
        lambda sql: sql.replace("role VARCHAR(50)", "role VARCHAR(1)", 1),
        lambda sql: sql.replace("password_hash TEXT", "password_hash VARCHAR(1)", 1),
        lambda sql: sql.replace(
            "CREATE TABLE IF NOT EXISTS",
            "CREATE UNLOGGED TABLE IF NOT EXISTS",
            1,
        ),
        lambda sql: sql.replace("public.users", "public.user_archive", 1),
        lambda sql: sql.replace("public.users", "tenant.public.users", 1),
        lambda sql: f"{sql}\nPARTITION BY RANGE (id)",
        lambda sql: f"{sql}\nWITH (fillfactor=70)",
    ],
)
def test_users_bootstrap_capability_rejects_incomplete_auth_schema(mutation) -> None:
    canonical = """
        CREATE TABLE IF NOT EXISTS public.users (
            id SERIAL PRIMARY KEY,
            uuid UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            role VARCHAR(50) NOT NULL DEFAULT 'user',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            is_verified BOOLEAN NOT NULL DEFAULT FALSE,
            is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
            email_verified BOOLEAN NOT NULL DEFAULT FALSE,
            two_factor_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            failed_login_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TIMESTAMPTZ,
            storage_quota_mb INTEGER NOT NULL DEFAULT 5120,
            storage_used_mb DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMPTZ,
            email_verified_at TIMESTAMPTZ,
            two_factor_secret TEXT,
            totp_secret TEXT,
            backup_codes TEXT,
            created_by INTEGER REFERENCES public.users(id) ON DELETE SET NULL,
            password_changed_at TIMESTAMPTZ
        )
    """

    with pytest.raises(ProfileUserWriteRejected):
        _mint_profile_user_sql(
            mutation(canonical),
            backend="postgres",
            connection_identity=object(),
            operation="create",
            columns=(),
        )


def test_alter_table_rename_destination_users_is_protected() -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            "ALTER TABLE public.user_staging RENAME TO users",
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )


def test_unrelated_users_constraint_strengthening_remains_allowed() -> None:
    statement = (
        "ALTER TABLE public.users ADD CONSTRAINT users_email_unique UNIQUE (email)"
    )

    assert _guard_sql(
        statement,
        backend="postgres",
        connection_identity=object(),
        operation="execute",
    ) == statement


def test_profile_anchor_constraint_requires_capability() -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            "ALTER TABLE public.users ADD CONSTRAINT users_profile_version_present "
            "CHECK (profile_version IS NOT NULL)",
            backend="postgres",
            connection_identity=object(),
            operation="execute",
        )


@pytest.mark.parametrize(
    ("changed_backend", "changed_connection", "changed_operation"),
    [
        ("postgres", False, "execute"),
        ("sqlite", True, "execute"),
        ("sqlite", False, "executemany"),
    ],
)
def test_profile_user_capability_rejects_boundary_mismatch(
    changed_backend: str,
    changed_connection: bool,
    changed_operation: str,
) -> None:
    connection = object()
    capability = _mint_profile_user_sql(
        "UPDATE users SET email = ? WHERE id = ?",
        backend="sqlite",
        connection_identity=connection,
        operation="update",
        columns=("email",),
        execution_mode="execute",
    )

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            capability,
            backend=changed_backend,
            connection_identity=object() if changed_connection else connection,
            operation=changed_operation,
        )
    assert _active_capability_count() == 0


def test_profile_user_capability_rejects_copy_and_lookalike() -> None:
    connection = object()
    capability = _mint_profile_user_sql(
        "UPDATE users SET email = ? WHERE id = ?",
        backend="sqlite",
        connection_identity=connection,
        operation="update",
        columns=("email",),
    )
    copied = copy.copy(capability)

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            copied,
            backend="sqlite",
            connection_identity=connection,
            operation="execute",
        )

    class Lookalike:
        text = capability.text
        backend = capability.backend
        operation = capability.operation
        columns = capability.columns

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            Lookalike(),
            backend="sqlite",
            connection_identity=connection,
            operation="execute",
        )
    _revoke_profile_user_sql(capability)
    assert _active_capability_count() == 0


def test_profile_user_capability_rejects_altered_metadata() -> None:
    connection = object()
    capability = _mint_profile_user_sql(
        "UPDATE users SET email = ? WHERE id = ?",
        backend="sqlite",
        connection_identity=connection,
        operation="update",
        columns=("email",),
    )
    object.__setattr__(capability, "columns", ("username",))

    with pytest.raises(ProfileUserWriteRejected):
        _guard_sql(
            capability,
            backend="sqlite",
            connection_identity=connection,
            operation="execute",
        )
    assert _active_capability_count() == 0


def test_profile_user_capability_mint_rejects_inexact_metadata() -> None:
    with pytest.raises(ProfileUserWriteRejected):
        _mint_profile_user_sql(
            "UPDATE users SET email = ? WHERE id = ?",
            backend="sqlite",
            connection_identity=object(),
            operation="update",
            columns=("username",),
        )


def test_profile_user_insert_capability_retains_exact_columns_and_consumes() -> None:
    connection = object()
    statement = (
        "INSERT INTO users (username, email, profile_version) VALUES (?, ?, ?)"
    )
    capability = _mint_profile_user_sql(
        statement,
        backend="sqlite",
        connection_identity=connection,
        operation="insert",
        columns=("username", "email", "profile_version"),
    )

    assert capability.columns == ("username", "email", "profile_version")
    assert (
        _guard_sql(
            capability,
            backend="sqlite",
            connection_identity=connection,
            operation="execute",
        )
        == statement
    )


def test_profile_user_capability_cannot_be_constructed_directly() -> None:
    with pytest.raises(TypeError):
        _ProfileUserSql(
            text="UPDATE users SET email = ? WHERE id = ?",
            backend="sqlite",
            operation="update",
            columns=("email",),
            execution_mode="execute",
            _nonce=object(),
        )
    with pytest.raises(TypeError):
        _ProfileUserSql()
