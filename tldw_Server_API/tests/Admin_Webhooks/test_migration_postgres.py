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
async def test_postgres_attempt_retry_delay_constraint_upgrades_existing_table(
    test_db_pool,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        constraints = await connection.fetch(
            """
            SELECT conname
            FROM pg_constraint
            WHERE conrelid = 'admin_webhook_delivery_attempts'::regclass
              AND contype = 'c'
              AND pg_get_constraintdef(oid) LIKE '%state%'
              AND pg_get_constraintdef(oid)
                  LIKE '%requested_retry_delay_seconds%'
            """
        )
        for row in constraints:
            name = str(row["conname"]).replace('"', '""')
            await connection.execute(
                'ALTER TABLE admin_webhook_delivery_attempts '
                f'DROP CONSTRAINT "{name}"'
            )
        await connection.execute(
            """
            ALTER TABLE admin_webhook_delivery_attempts
            ADD CONSTRAINT legacy_admin_webhook_attempt_retry_delay_state
            CHECK (
                (state = 'retryable' AND requested_retry_delay_seconds IS NOT NULL)
                OR (
                    state != 'retryable'
                    AND requested_retry_delay_seconds IS NULL
                )
            )
            """
        )
    finally:
        await connection.close()

    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _insert_registration(test_db_pool)
    await _insert_command_event(test_db_pool)
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES ('delivery-retry-evidence', 'event-1', 1, 'automatic', 1, 1,
                  'retry_wait', '2026-07-04T00:00:00Z')
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_delivery_attempts (
            id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
            request_timeout_seconds, started_at, finished_at, state,
            reason_code, requested_retry_delay_seconds
        ) VALUES ('attempt-outcome-unknown-retry', 'delivery-retry-evidence', 1,
                  'jobs-1', 'lease-1', 10, '2026-07-01T00:00:00Z',
                  '2026-07-01T00:01:00Z', 'outcome_unknown',
                  'outcome_unknown', 60)
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_delivery_attempts (
            id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
            request_timeout_seconds, started_at, finished_at, state,
            reason_code, requested_retry_delay_seconds
        ) VALUES ('attempt-outcome-unknown-terminal', 'delivery-retry-evidence', 2,
                  'jobs-1', 'lease-2', 10, '2026-07-01T00:02:00Z',
                  '2026-07-01T00:03:00Z', 'outcome_unknown',
                  'outcome_unknown', NULL)
        """
    )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
                request_timeout_seconds, started_at, finished_at, state,
                requested_retry_delay_seconds
            ) VALUES ('attempt-retryable-without-delay',
                      'delivery-retry-evidence', 3, 'jobs-1', 'lease-3', 10,
                      '2026-07-01T00:04:00Z', '2026-07-01T00:05:00Z',
                      'retryable', NULL)
            """
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
                request_timeout_seconds, started_at, finished_at, state,
                requested_retry_delay_seconds
            ) VALUES ('attempt-failed-with-delay', 'delivery-retry-evidence', 4,
                      'jobs-1', 'lease-4', 10, '2026-07-01T00:06:00Z',
                      '2026-07-01T00:07:00Z', 'failed', 60)
            """
        )


@pytest.mark.integration
async def test_postgres_delivery_schema_ready_requires_recovery_indexes(test_db_pool) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute("DROP INDEX idx_admin_webhook_runtime_heartbeats_freshness")
        await connection.execute(
            """
            CREATE INDEX idx_admin_webhook_runtime_heartbeats_freshness
            ON admin_webhook_runtime_heartbeats(component, ready, heartbeat_at)
            """
        )
    finally:
        await connection.close()

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
async def test_postgres_delivery_schema_ready_rejects_wrong_recovery_index_predicate(
    test_db_pool,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute("DROP INDEX idx_admin_webhook_deliveries_recovery")
        await connection.execute(
            """
            CREATE INDEX idx_admin_webhook_deliveries_recovery
            ON admin_webhook_deliveries(
                state, enqueue_claim_expires_at, expires_at, created_at
            )
            WHERE state = 'pending'
            """
        )
    finally:
        await connection.close()

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
@pytest.mark.parametrize(
    ("index_name", "replacement_sql"),
    (
        (
            "idx_admin_webhook_deliveries_recovery",
            """
            CREATE INDEX idx_admin_webhook_deliveries_recovery
            ON admin_webhook_deliveries(
                state DESC, enqueue_claim_expires_at, expires_at, created_at
            )
            WHERE state IN ('pending', 'enqueue_claimed')
            """,
        ),
        (
            "idx_admin_webhook_deliveries_disposition_recovery",
            """
            CREATE INDEX idx_admin_webhook_deliveries_disposition_recovery
            ON admin_webhook_deliveries(
                jobs_disposition_applied DESC,
                pending_jobs_disposition_not_before_at,
                updated_at
            )
            WHERE pending_jobs_disposition IS NOT NULL
            """,
        ),
    ),
)
async def test_postgres_delivery_schema_ready_rejects_wrong_recovery_index_order(
    test_db_pool,
    index_name: str,
    replacement_sql: str,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute(f"DROP INDEX {index_name}")
        await connection.execute(replacement_sql)
    finally:
        await connection.close()

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
async def test_postgres_delivery_schema_ready_rejects_decoy_named_index_on_wrong_table(
    test_db_pool,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute(
            """
            CREATE TABLE admin_webhook_delivery_recovery_decoy (
                state TEXT NOT NULL,
                enqueue_claim_expires_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        await connection.execute("DROP INDEX idx_admin_webhook_deliveries_recovery")
        await connection.execute(
            """
            CREATE INDEX idx_admin_webhook_deliveries_recovery
            ON admin_webhook_delivery_recovery_decoy(
                state, enqueue_claim_expires_at, expires_at, created_at
            ) WHERE state IN ('pending', 'enqueue_claimed')
            """
        )
    finally:
        await connection.close()

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
async def test_postgres_delivery_schema_ready_binds_checks_to_their_owning_table(
    test_db_pool,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        constraint_name = await connection.fetchval(
            """
            SELECT constraint_name
            FROM information_schema.check_constraints
            WHERE check_clause LIKE '%pending_jobs_disposition_token%'
            """
        )
        assert constraint_name is not None
        await connection.execute(
            f'ALTER TABLE admin_webhook_deliveries DROP CONSTRAINT "{constraint_name}"'
        )
        await connection.execute(
            """
            ALTER TABLE admin_webhook_runtime_heartbeats
            ADD COLUMN pending_jobs_disposition_token TEXT CHECK (
                pending_jobs_disposition_token IS NULL
                OR pending_jobs_disposition_token ~ '^[0-9a-f]{64}$'
            )
            """
        )
    finally:
        await connection.close()

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
async def test_postgres_delivery_schema_ready_rejects_incompatible_column_contract(
    test_db_pool,
) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    repository = AdminWebhookRepository(test_db_pool)
    assert await repository.delivery_schema_ready() is True

    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute(
            """
            ALTER TABLE admin_webhook_delivery_attempts
            ALTER COLUMN request_timeout_seconds SET NOT NULL
            """
        )
    finally:
        await connection.close()

    assert await repository.delivery_schema_ready() is False


@pytest.mark.integration
async def test_postgres_095_enforces_delivery_and_heartbeat_boundaries(test_db_pool) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _insert_registration(test_db_pool)
    await _insert_command_event(test_db_pool)
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at, pending_jobs_disposition_token
        ) VALUES ('delivery-boundary', 'event-1', 1, 'test', 1, 1, 'pending',
                  '2026-07-04T00:00:00Z', $1)
        """,
        "a" * 64,
    )
    for invalid_token in ("a" * 63, "A" * 64, "https://receiver.example/secret"):
        with pytest.raises(asyncpg.CheckViolationError):
            await test_db_pool.execute(
                """
                UPDATE admin_webhook_deliveries
                SET pending_jobs_disposition_token = $1
                WHERE id = 'delivery-boundary'
                """,
                invalid_token,
            )
    for attempt_number, timeout in enumerate((1, 30), start=1):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, test_attempt_token, started_at,
                state, request_timeout_seconds
            ) VALUES ($1, 'delivery-boundary', $2, $3, '2026-07-01T00:00:00Z',
                      'processing', $4)
            """,
            f"attempt-{timeout}",
            attempt_number,
            f"test-token-{timeout}",
            timeout,
        )
    for attempt_number, timeout in enumerate((0, 31), start=3):
        with pytest.raises(asyncpg.CheckViolationError):
            await test_db_pool.execute(
                """
                INSERT INTO admin_webhook_delivery_attempts (
                    id, delivery_id, attempt_number, test_attempt_token, started_at,
                    state, request_timeout_seconds
                ) VALUES ($1, 'delivery-boundary', $2, $3, '2026-07-01T00:00:00Z',
                          'processing', $4)
                """,
                f"attempt-invalid-{timeout}",
                attempt_number,
                f"test-token-invalid-{timeout}",
                timeout,
            )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_runtime_heartbeats (
            component, instance_id, ready, reason_code, heartbeat_at
        ) VALUES ('worker', 'runtime-unready', FALSE, 'database_unavailable',
                  CURRENT_TIMESTAMP)
        """
    )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, reason_code, heartbeat_at
            ) VALUES ('worker', 'unready-with-null-reason', FALSE, NULL,
                      CURRENT_TIMESTAMP)
            """
        )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_runtime_heartbeats (
            component, instance_id, ready, heartbeat_at, last_success_at
        ) VALUES ('retention', $1, TRUE, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        "x" * 128,
    )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, heartbeat_at
            ) VALUES ('retention', $1, TRUE, CURRENT_TIMESTAMP)
            """,
            "x" * 129,
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, reason_code, heartbeat_at
            ) VALUES ('worker', 'arbitrary-reason', FALSE,
                      'https://receiver.example/secret', CURRENT_TIMESTAMP)
            """
        )
    with pytest.raises(asyncpg.NotNullViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, heartbeat_at
            ) VALUES ('worker', 'invalid-ready', NULL, CURRENT_TIMESTAMP)
            """
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, reason_code, heartbeat_at
            ) VALUES ('worker', 'ready-with-reason', TRUE, 'database_unavailable',
                      CURRENT_TIMESTAMP)
            """
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, heartbeat_at
            ) VALUES ('unknown', 'invalid-component', TRUE, CURRENT_TIMESTAMP)
            """
        )
    with pytest.raises(asyncpg.CheckViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, heartbeat_at
            ) VALUES ('worker', '', TRUE, CURRENT_TIMESTAMP)
            """
        )
    with pytest.raises(asyncpg.NotNullViolationError):
        await test_db_pool.execute(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, reason_code
            ) VALUES ('worker', 'missing-heartbeat', FALSE, 'database_unavailable')
            """
        )


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
