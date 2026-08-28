from __future__ import annotations

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.tests.Admin_Webhooks.test_migration_sqlite import (
    EXPECTED_COLUMNS,
)

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)
pytestmark = pytest.mark.postgres

CANONICAL_TABLES = {
    "admin_webhook_sequences",
    "admin_webhook_registrations",
    "admin_webhook_events",
    "admin_webhook_deliveries",
    "admin_webhook_delivery_attempts",
    "admin_webhook_idempotency",
    "admin_webhook_migration_state",
    "admin_webhook_runtime_heartbeats",
}


async def _insert_registration(pool, webhook_id: int = 1) -> None:
    await pool.execute(
        """
        INSERT INTO admin_webhook_registrations (
            id, target_ciphertext_json, target_key_id, target_hostname,
            target_display, event_types_json, secret_ciphertext_json,
            secret_key_id, created_by_user_id, updated_by_user_id
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        webhook_id,
        '{"ciphertext":"target"}',
        "key-1",
        "example.com",
        "https://example.com",
        '["user.created"]',
        '{"ciphertext":"secret"}',
        "key-1",
        7,
        7,
    )


async def _insert_command_event(pool, event_id: str = "event-1") -> None:
    await pool.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, source_command_id,
            source_component, body_ciphertext_json, body_key_id, body_size_bytes
        ) VALUES ($1, $2, $3, 'command', $4, $5, $6, $7, $8)
        """,
        event_id,
        "user.created",
        "2026-07-01",
        "command-1",
        "authnz",
        '{"ciphertext":"body"}',
        "key-1",
        42,
    )


@pytest.mark.integration
async def test_postgres_schema_is_additive_idempotent_and_preserves_legacy_rows(
    test_db_pool,
) -> None:
    await test_db_pool.execute(
        """
        CREATE TABLE admin_webhooks (
            id BIGINT PRIMARY KEY,
            url TEXT NOT NULL,
            secret_encrypted TEXT NOT NULL,
            secret_key_id TEXT,
            event_types TEXT NOT NULL,
            description TEXT NOT NULL,
            active BOOLEAN NOT NULL,
            retry_count INTEGER NOT NULL,
            timeout_seconds INTEGER NOT NULL
        )
        """
    )
    await test_db_pool.execute(
        """
        CREATE TABLE admin_webhooks_delivery_log (
            id BIGINT PRIMARY KEY,
            webhook_id BIGINT NOT NULL REFERENCES admin_webhooks(id),
            payload_json TEXT NOT NULL
        )
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhooks (
            id, url, secret_encrypted, secret_key_id, event_types,
            description, active, retry_count, timeout_seconds
        ) VALUES (44, 'https://legacy.example/private?token=unchanged',
                  'legacy-envelope', 'legacy-key', '["*"]', 'legacy row',
                  TRUE, 3, 10)
        """
    )

    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    assert await AdminWebhookRepository(test_db_pool).delivery_schema_ready() is True

    rows = await test_db_pool.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name LIKE 'admin_webhook%'
        """
    )
    names = {str(row["table_name"]) for row in rows}
    assert names >= CANONICAL_TABLES
    assert names >= {"admin_webhooks", "admin_webhooks_delivery_log"}

    column_rows = await test_db_pool.fetch(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = ANY($1::text[])
        """,
        [sorted(CANONICAL_TABLES)],
    )
    actual_columns: dict[str, set[str]] = {
        table: set() for table in CANONICAL_TABLES
    }
    for row in column_rows:
        actual_columns[str(row["table_name"])].add(str(row["column_name"]))
    assert actual_columns == EXPECTED_COLUMNS

    legacy = await test_db_pool.fetchone(
        """
        SELECT id, url, secret_encrypted, secret_key_id, event_types,
               description, active, retry_count, timeout_seconds
        FROM admin_webhooks WHERE id = 44
        """
    )
    assert legacy == {
        "id": 44,
        "url": "https://legacy.example/private?token=unchanged",
        "secret_encrypted": "legacy-envelope",
        "secret_key_id": "legacy-key",
        "event_types": '["*"]',
        "description": "legacy row",
        "active": True,
        "retry_count": 3,
        "timeout_seconds": 10,
    }
    assert await test_db_pool.fetchone(
        "SELECT name, next_value FROM admin_webhook_sequences"
    ) == {"name": "registration", "next_value": 1}
    assert await test_db_pool.fetchone(
        """
        SELECT singleton_id, schema_version, state_revision, phase,
               source_mapping_json, source_rejections_json,
               rollback_retirement_phase
        FROM admin_webhook_migration_state
        """
    ) == {
        "singleton_id": 1,
        "schema_version": 1,
        "state_revision": 1,
        "phase": "migration_pending",
        "source_mapping_json": "{}",
        "source_rejections_json": "[]",
        "rollback_retirement_phase": "not_applicable",
    }


@pytest.mark.integration
async def test_postgres_delivery_schema_ready_requires_recovery_indexes(test_db_pool) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    await test_db_pool.execute("DROP INDEX idx_admin_webhook_runtime_heartbeats_freshness")

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
async def test_postgres_schema_has_equivalent_partial_uniqueness_and_checks(
    test_db_pool,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _insert_registration(test_db_pool)
    await _insert_command_event(test_db_pool)

    with pytest.raises(asyncpg.UniqueViolationError):
        await _insert_command_event(test_db_pool, "event-command-duplicate")

    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES ('delivery-a', 'event-1', 1, 'automatic', 1, 1,
                  'pending', '2026-07-04T00:00:00Z')
        """
    )
    with pytest.raises(asyncpg.UniqueViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_deliveries (
                id, event_id, webhook_id, kind, delivery_config_version,
                secret_version, state, expires_at
            ) VALUES ('delivery-b', 'event-1', 1, 'automatic', 1, 1,
                      'pending', '2026-07-04T00:00:00Z')
            """
        )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES ('delivery-manual', 'event-1', 1, 'manual', 1, 1,
                  'pending', '2026-07-04T00:00:00Z')
        """
    )

    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_events (
                id, event_type, api_version, source_kind, aggregate_type,
                source_command_id, source_component, body_ciphertext_json,
                body_key_id, body_size_bytes
            ) VALUES ('bad-source', 'user.created', '2026-07-01',
                      'command', 'user', 'command-2', 'authnz', '{}', 'key-1', 2)
            """
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            UPDATE admin_webhook_migration_state
            SET rotation_operation_id = 'rotate-1',
                rotation_source_key_id = 'key-1',
                rotation_target_key_id = 'key-2',
                rotation_phase = 'rewriting',
                rotation_started_at = '2026-07-01T00:00:00Z'
            WHERE singleton_id = 1
            """
        )


@pytest.mark.parametrize(
    ("field", "exact_json", "too_large_json"),
    [
        (
            "source_mapping_json",
            '{"x":"' + "a" * (1_048_576 - len('{"x":""}')) + '"}',
            '{"x":"' + "a" * (1_048_577 - len('{"x":""}')) + '"}',
        ),
        (
            "source_rejections_json",
            '["' + "a" * (1_048_576 - len('[""]')) + '"]',
            '["' + "a" * (1_048_577 - len('[""]')) + '"]',
        ),
    ],
)
@pytest.mark.integration
async def test_postgres_schema_enforces_json_utf8_byte_bounds(
    test_db_pool,
    field: str,
    exact_json: str,
    too_large_json: str,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)

    await test_db_pool.execute(
        f"UPDATE admin_webhook_migration_state SET {field} = $1 WHERE singleton_id = 1",
        exact_json,
    )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            f"UPDATE admin_webhook_migration_state SET {field} = $1 WHERE singleton_id = 1",
            too_large_json,
        )
    with pytest.raises(asyncpg.InvalidTextRepresentationError):
        await test_db_pool.execute(
            f"UPDATE admin_webhook_migration_state SET {field} = $1 WHERE singleton_id = 1",
            "not-json",
        )

    multibyte = '{"x":"' + "\u00e9" * 524_285 + '"}'
    assert len(multibyte) < 1_048_576
    assert len(multibyte.encode("utf-8")) > 1_048_576
    if field == "source_mapping_json":
        with pytest.raises(asyncpg.CheckViolationError):
            await test_db_pool.execute(
                """
                UPDATE admin_webhook_migration_state
                SET source_mapping_json = $1 WHERE singleton_id = 1
                """,
                multibyte,
            )
