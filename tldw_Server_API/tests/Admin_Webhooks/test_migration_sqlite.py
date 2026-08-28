from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.migrations import (
    CANONICAL_ADMIN_WEBHOOK_SQLITE_DDL,
    apply_authnz_migrations,
    migration_094_create_canonical_admin_webhook_tables,
)

CANONICAL_TABLES = {
    "admin_webhook_sequences",
    "admin_webhook_registrations",
    "admin_webhook_events",
    "admin_webhook_deliveries",
    "admin_webhook_delivery_attempts",
    "admin_webhook_idempotency",
    "admin_webhook_migration_state",
}

EXPECTED_COLUMNS = {
    "admin_webhook_sequences": {"name", "next_value"},
    "admin_webhook_registrations": {
        "id",
        "description",
        "target_ciphertext_json",
        "target_key_id",
        "target_hostname",
        "target_display",
        "event_types_json",
        "active",
        "timeout_seconds",
        "delivery_config_version",
        "target_version",
        "secret_ciphertext_json",
        "secret_key_id",
        "secret_version",
        "secret_rotation_required",
        "revision",
        "created_by_user_id",
        "updated_by_user_id",
        "created_at",
        "updated_at",
        "deleted_at",
        "deleted_by_user_id",
    },
    "admin_webhook_events": {
        "id",
        "event_type",
        "api_version",
        "source_kind",
        "aggregate_type",
        "aggregate_id",
        "aggregate_version",
        "source_command_id",
        "source_component",
        "source_request_id",
        "body_ciphertext_json",
        "body_key_id",
        "body_size_bytes",
        "created_at",
    },
    "admin_webhook_deliveries": {
        "id",
        "event_id",
        "webhook_id",
        "kind",
        "delivery_config_version",
        "secret_version",
        "jobs_job_id",
        "enqueue_claim_token",
        "enqueue_claim_expires_at",
        "state",
        "attempt_count",
        "current_attempt_id",
        "status_code",
        "latency_ms",
        "reason_code",
        "pending_jobs_disposition",
        "pending_jobs_disposition_delay_seconds",
        "jobs_disposition_applied",
        "completed_after_config_change",
        "terminal_at",
        "expires_at",
        "redelivery_of_id",
        "created_at",
        "updated_at",
    },
    "admin_webhook_delivery_attempts": {
        "id",
        "delivery_id",
        "attempt_number",
        "jobs_job_id",
        "jobs_lease_id",
        "test_attempt_token",
        "started_at",
        "finished_at",
        "state",
        "status_code",
        "latency_ms",
        "reason_code",
        "requested_retry_delay_seconds",
        "jobs_disposition_applied",
        "created_at",
    },
    "admin_webhook_idempotency": {
        "id",
        "lookup_digest",
        "actor_id",
        "operation",
        "route",
        "webhook_id",
        "delivery_id",
        "request_fingerprint",
        "state",
        "resource_id",
        "resource_version",
        "secret_version",
        "replay_secret_ciphertext_json",
        "replay_secret_key_id",
        "test_delivery_id",
        "test_attempt_id",
        "response_status",
        "response_metadata_json",
        "created_at",
        "updated_at",
        "expires_at",
    },
    "admin_webhook_migration_state": {
        "singleton_id",
        "schema_version",
        "state_revision",
        "phase",
        "import_operation_id",
        "import_operator_id",
        "import_started_at",
        "import_approved_at",
        "artifacts_ready_at",
        "database_committed_at",
        "fingerprint_key_id",
        "active_primary_key_id",
        "system_ops_webhook_fingerprint",
        "legacy_table_fingerprint",
        "source_mapping_json",
        "redacted_report_digest",
        "protected_backup_ciphertext_digest",
        "source_rejections_json",
        "completed_at",
        "active_report_path",
        "active_backup_path",
        "active_key_path",
        "staging_report_path",
        "staging_backup_path",
        "staging_key_path",
        "report_owner_id",
        "report_group_id",
        "report_mode",
        "report_file_identity",
        "backup_owner_id",
        "backup_group_id",
        "backup_mode",
        "backup_file_identity",
        "rollback_key_owner_id",
        "rollback_key_group_id",
        "rollback_key_mode",
        "rollback_key_file_identity",
        "rollback_expires_at",
        "rollback_retirement_phase",
        "rollback_retirement_operator_id",
        "rollback_retirement_started_at",
        "rollback_retirement_completed_at",
        "expected_ciphertext_digest",
        "first_canonical_activity_at",
        "first_canonical_activity_kind",
        "rotation_operation_id",
        "rotation_source_key_id",
        "rotation_target_key_id",
        "rotation_phase",
        "rotation_table_cursor",
        "rotation_key_cursor",
        "rotation_processed_count",
        "rotation_verified_count",
        "rotation_started_at",
        "rotation_completed_at",
        "updated_at",
    },
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _current_schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0] or 0)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _insert_registration(conn: sqlite3.Connection, webhook_id: int = 1) -> None:
    conn.execute(
        """
        INSERT INTO admin_webhook_registrations (
            id, target_ciphertext_json, target_key_id, target_hostname,
            target_display, event_types_json, secret_ciphertext_json,
            secret_key_id, created_by_user_id, updated_by_user_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
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
        ),
    )


def _insert_command_event(conn: sqlite3.Connection, event_id: str = "event-1") -> None:
    conn.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, source_command_id,
            source_component, body_ciphertext_json, body_key_id, body_size_bytes
        ) VALUES (?, ?, ?, 'command', ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            "user.created",
            "2026-07-01",
            "command-1",
            "authnz",
            '{"ciphertext":"body"}',
            "key-1",
            42,
        ),
    )


@pytest.mark.parametrize("starting_version", [0, 79, 80, 82, 93])
@pytest.mark.unit
def test_sqlite_094_is_additive_across_supported_upgrade_points(
    tmp_path: Path,
    starting_version: int,
) -> None:
    db_path = tmp_path / f"auth-{starting_version}.db"
    if starting_version:
        apply_authnz_migrations(db_path, target_version=starting_version)

    legacy_row: tuple[object, ...] | None = None
    if starting_version >= 80:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO admin_webhooks (
                    id, url, secret_encrypted, secret_key_id, event_types,
                    description, active, retry_count, timeout_seconds, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    44,
                    "https://legacy.example/private?token=unchanged",
                    "legacy-envelope",
                    "legacy-key",
                    '["*"]',
                    "legacy row",
                    1,
                    3,
                    10,
                    7,
                ),
            )
            conn.commit()
            legacy_row = conn.execute(
                "SELECT * FROM admin_webhooks WHERE id = 44"
            ).fetchone()

    apply_authnz_migrations(db_path)

    with sqlite3.connect(db_path) as conn:
        names = _table_names(conn)
        assert names >= CANONICAL_TABLES
        assert names >= {"admin_webhooks", "admin_webhooks_delivery_log"}
        assert _current_schema_version(conn) == 94
        if legacy_row is not None:
            assert conn.execute(
                "SELECT * FROM admin_webhooks WHERE id = 44"
            ).fetchone() == legacy_row


@pytest.mark.unit
def test_sqlite_094_has_exact_columns_foreign_keys_and_seed_state() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    migration_094_create_canonical_admin_webhook_tables(conn)

    assert _table_names(conn) >= CANONICAL_TABLES
    for table, expected in EXPECTED_COLUMNS.items():
        assert _columns(conn, table) == expected

    assert conn.execute(
        "SELECT name, next_value FROM admin_webhook_sequences"
    ).fetchall() == [("registration", 1)]
    assert conn.execute(
        """
        SELECT singleton_id, schema_version, state_revision, phase,
               source_mapping_json, source_rejections_json,
               rollback_retirement_phase, rotation_processed_count,
               rotation_verified_count
        FROM admin_webhook_migration_state
        """
    ).fetchone() == (
        1,
        1,
        1,
        "migration_pending",
        "{}",
        "[]",
        "not_applicable",
        0,
        0,
    )

    delivery_fks = {
        (str(row[2]), str(row[3]), str(row[4]))
        for row in conn.execute(
            "PRAGMA foreign_key_list(admin_webhook_deliveries)"
        ).fetchall()
    }
    assert (
        "admin_webhook_registrations",
        "webhook_id",
        "id",
    ) in delivery_fks
    assert ("admin_webhook_events", "event_id", "id") in delivery_fks
    assert (
        "admin_webhook_deliveries",
        "redelivery_of_id",
        "id",
    ) in delivery_fks


@pytest.mark.unit
def test_sqlite_094_enforces_source_and_delivery_uniqueness() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    migration_094_create_canonical_admin_webhook_tables(conn)
    _insert_registration(conn)

    aggregate = (
        "incident.updated",
        "2026-07-01",
        "aggregate",
        "incident",
        "inc-1",
        "4",
        "system_ops",
        '{"ciphertext":"body"}',
        "key-1",
        42,
    )
    conn.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, aggregate_type,
            aggregate_id, aggregate_version, source_component,
            body_ciphertext_json, body_key_id, body_size_bytes
        ) VALUES ('event-a', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        aggregate,
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_events (
                id, event_type, api_version, source_kind, aggregate_type,
                aggregate_id, aggregate_version, source_component,
                body_ciphertext_json, body_key_id, body_size_bytes
            ) VALUES ('event-b', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            aggregate,
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_events (
                id, event_type, api_version, source_kind, aggregate_type,
                source_command_id, source_component, body_ciphertext_json,
                body_key_id, body_size_bytes
            ) VALUES ('bad-source', 'user.created', '2026-07-01',
                      'command', 'user', 'command-2', 'authnz', '{}', 'key-1', 2)
            """
        )

    _insert_command_event(conn, "event-command")
    delivery = (
        "event-command",
        1,
        "automatic",
        1,
        1,
        "pending",
        "2026-07-04T00:00:00Z",
    )
    conn.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES ('delivery-a', ?, ?, ?, ?, ?, ?, ?)
        """,
        delivery,
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_deliveries (
                id, event_id, webhook_id, kind, delivery_config_version,
                secret_version, state, expires_at
            ) VALUES ('delivery-b', ?, ?, ?, ?, ?, ?, ?)
            """,
            delivery,
        )
    conn.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES ('delivery-manual', ?, ?, 'manual', ?, ?, ?, ?)
        """,
        (delivery[0], delivery[1], delivery[3], delivery[4], delivery[5], delivery[6]),
    )


@pytest.mark.unit
def test_sqlite_094_enforces_attempt_and_idempotency_invariants() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    migration_094_create_canonical_admin_webhook_tables(conn)
    _insert_registration(conn)
    _insert_command_event(conn)
    conn.execute(
        """
        INSERT INTO admin_webhook_deliveries (
            id, event_id, webhook_id, kind, delivery_config_version,
            secret_version, state, expires_at
        ) VALUES ('delivery-1', 'event-1', 1, 'test', 1, 1,
                  'processing', '2026-07-04T00:00:00Z')
        """
    )

    conn.execute(
        """
        INSERT INTO admin_webhook_delivery_attempts (
            id, delivery_id, attempt_number, test_attempt_token,
            started_at, state
        ) VALUES ('attempt-1', 'delivery-1', 1, 'test-token',
                  '2026-07-01T00:00:00Z', 'processing')
        """
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, test_attempt_token,
                started_at, state
            ) VALUES ('attempt-2', 'delivery-1', 1, 'other-token',
                      '2026-07-01T00:00:00Z', 'processing')
            """
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, jobs_job_id,
                test_attempt_token, started_at, state
            ) VALUES ('attempt-3', 'delivery-1', 2, 'job-1',
                      'test-token', '2026-07-01T00:00:00Z', 'processing')
            """
        )

    idempotency = (
        "sha256:" + "a" * 64,
        "7",
        "create",
        "/api/v1/admin/webhooks",
        "hmac-sha256:" + "b" * 64,
        "in_progress",
        "2026-07-02T00:00:00Z",
    )
    conn.execute(
        """
        INSERT INTO admin_webhook_idempotency (
            lookup_digest, actor_id, operation, route,
            request_fingerprint, state, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        idempotency,
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_idempotency (
                lookup_digest, actor_id, operation, route,
                request_fingerprint, state, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            idempotency,
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO admin_webhook_idempotency (
                lookup_digest, actor_id, operation, route,
                request_fingerprint, state, replay_secret_ciphertext_json,
                expires_at
            ) VALUES (?, '7', 'rotate', '/api/v1/admin/webhooks/1/rotate-secret',
                      ?, 'completed', '{}', '2026-07-02T00:00:00Z')
            """,
            ("sha256:" + "c" * 64, "hmac-sha256:" + "d" * 64),
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
@pytest.mark.unit
def test_sqlite_094_enforces_json_utf8_byte_bounds(
    field: str,
    exact_json: str,
    too_large_json: str,
) -> None:
    conn = sqlite3.connect(":memory:")
    migration_094_create_canonical_admin_webhook_tables(conn)

    conn.execute(
        f"UPDATE admin_webhook_migration_state SET {field} = ? WHERE singleton_id = 1",
        (exact_json,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"UPDATE admin_webhook_migration_state SET {field} = ? WHERE singleton_id = 1",
            (too_large_json,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"UPDATE admin_webhook_migration_state SET {field} = ? WHERE singleton_id = 1",
            ("not-json",),
        )

    multibyte = '{"x":"' + "\u00e9" * 524_285 + '"}'
    assert len(multibyte) < 1_048_576
    assert len(multibyte.encode("utf-8")) > 1_048_576
    if field == "source_mapping_json":
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                UPDATE admin_webhook_migration_state
                SET source_mapping_json = ? WHERE singleton_id = 1
                """,
                (multibyte,),
            )


@pytest.mark.unit
def test_sqlite_094_enforces_migration_and_rotation_state_machine() -> None:
    conn = sqlite3.connect(":memory:")
    migration_094_create_canonical_admin_webhook_tables(conn)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            UPDATE admin_webhook_migration_state
            SET first_canonical_activity_at = '2026-07-01T00:00:00Z'
            WHERE singleton_id = 1
            """
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            UPDATE admin_webhook_migration_state
            SET first_canonical_activity_at = '2026-07-01T00:00:00Z',
                first_canonical_activity_kind = 'unknown'
            WHERE singleton_id = 1
            """
        )
    conn.execute(
        """
        UPDATE admin_webhook_migration_state
        SET first_canonical_activity_at = '2026-07-01T00:00:00Z',
            first_canonical_activity_kind = 'registration_mutation'
        WHERE singleton_id = 1
        """
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            UPDATE admin_webhook_migration_state
            SET phase = 'artifacts_pending'
            WHERE singleton_id = 1
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
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

    conn.execute(
        """
        UPDATE admin_webhook_migration_state
        SET phase = 'complete',
            completed_at = '2026-07-01T00:00:00Z',
            import_operation_id = 'whmig_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            import_operator_id = 7,
            import_started_at = '2026-07-01T00:00:00Z',
            import_approved_at = '2026-07-01T00:00:00Z',
            database_committed_at = '2026-07-01T00:00:00Z',
            fingerprint_key_id = 'key-1',
            active_primary_key_id = 'key-1',
            system_ops_webhook_fingerprint =
                'hmac-sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            legacy_table_fingerprint =
                'hmac-sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            redacted_report_digest =
                'sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
            active_report_path = '/tmp/webhook-report.json',
            staging_report_path = '/tmp/.webhook-report.stage',
            report_owner_id = 501,
            report_group_id = 20,
            report_mode = 384,
            report_file_identity = '1:2',
            rotation_operation_id = 'rotate-1',
            rotation_source_key_id = 'key-1',
            rotation_target_key_id = 'key-2',
            rotation_phase = 'rewriting',
            rotation_started_at = '2026-07-01T00:00:00Z'
        WHERE singleton_id = 1
        """
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            UPDATE admin_webhook_migration_state
            SET rotation_processed_count = -1 WHERE singleton_id = 1
            """
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            UPDATE admin_webhook_migration_state
            SET system_ops_webhook_fingerprint = ? WHERE singleton_id = 1
            """,
            ("hmac-sha256:" + "g" * 64,),
        )
    conn.execute(
        """
        UPDATE admin_webhook_migration_state
        SET system_ops_webhook_fingerprint = ?,
            redacted_report_digest = ?
        WHERE singleton_id = 1
        """,
        ("hmac-sha256:" + "a" * 64, "sha256:" + "b" * 64),
    )


@pytest.mark.unit
def test_sqlite_094_rerun_is_idempotent() -> None:
    conn = sqlite3.connect(":memory:")
    migration_094_create_canonical_admin_webhook_tables(conn)
    conn.execute(
        "UPDATE admin_webhook_sequences SET next_value = 9 WHERE name = 'registration'"
    )
    conn.execute(
        "UPDATE admin_webhook_migration_state SET state_revision = 4 WHERE singleton_id = 1"
    )

    migration_094_create_canonical_admin_webhook_tables(conn)

    assert conn.execute(
        "SELECT next_value FROM admin_webhook_sequences WHERE name = 'registration'"
    ).fetchone() == (9,)
    assert conn.execute(
        "SELECT state_revision FROM admin_webhook_migration_state WHERE singleton_id = 1"
    ).fetchone() == (4,)


@pytest.mark.unit
def test_sqlite_094_rolls_back_all_ddl_when_one_statement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "rollback.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO schema_migrations VALUES (93, 'existing', CURRENT_TIMESTAMP)"
        )
        conn.commit()

    monkeypatch.setattr(
        migrations,
        "CANONICAL_ADMIN_WEBHOOK_SQLITE_DDL",
        (CANONICAL_ADMIN_WEBHOOK_SQLITE_DDL[0], "CREATE TABLE invalid ("),
    )

    with pytest.raises(sqlite3.OperationalError):
        apply_authnz_migrations(db_path)

    with sqlite3.connect(db_path) as conn:
        assert "admin_webhook_sequences" not in _table_names(conn)
        assert _current_schema_version(conn) == 93
