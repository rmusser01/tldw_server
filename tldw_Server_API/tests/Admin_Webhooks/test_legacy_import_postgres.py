from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.audit import OperationalAudit
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import WebhookKeyRing
from tldw_Server_API.app.core.Admin_Webhooks.legacy_import import (
    LegacyImportRequest,
    LegacyImportService,
    LegacySecretDecryptor,
)
from tldw_Server_API.app.core.Admin_Webhooks.repository import AdminWebhookRepository
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)
pytestmark = [pytest.mark.postgres, pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 8, 22, 21, 0, tzinfo=timezone.utc)


class _Decryptor(LegacySecretDecryptor):
    def decrypt(self, encrypted_blob: str) -> str:
        assert encrypted_blob == '{"_enc":"legacy"}'
        return "postgres-secret-canary"


def _ring() -> WebhookKeyRing:
    return WebhookKeyRing(
        {"migration-key": base64.b64encode(b"p" * 32).decode("ascii")},
        primary_id="migration-key",
    )


def _settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.MIGRATE,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


async def _reset_canonical(test_db_pool) -> None:
    await test_db_pool.execute(
        """
        TRUNCATE TABLE
            admin_webhook_delivery_attempts,
            admin_webhook_deliveries,
            admin_webhook_events,
            admin_webhook_idempotency,
            admin_webhook_registrations,
            admin_webhook_sequences,
            admin_webhook_migration_state
        RESTART IDENTITY CASCADE
        """
    )
    await test_db_pool.execute("INSERT INTO admin_webhook_sequences (name, next_value) VALUES ('registration', 1)")
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_migration_state (
            singleton_id, schema_version, phase
        ) VALUES (1, 1, 'migration_pending')
        """
    )


async def test_postgres_legacy_snapshot_uses_real_legacy_table(test_db_pool) -> None:
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _reset_canonical(test_db_pool)
    await test_db_pool.execute("DROP TABLE IF EXISTS admin_webhooks CASCADE")
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
            timeout_seconds INTEGER NOT NULL,
            created_by BIGINT,
            created_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ
        )
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhooks (
            id, url, secret_encrypted, secret_key_id, event_types,
            description, active, retry_count, timeout_seconds, created_by
        ) VALUES (17, 'https://hooks.example.com/path', '{"_enc":"legacy"}',
                  'jwt_secret', '["user.created"]', 'Legacy', TRUE, 3, 10, 9)
        """
    )

    snapshot = await AdminWebhookRepository(test_db_pool).get_legacy_import_snapshot()

    assert snapshot.table_present is True
    assert tuple(row.source_identity for row in snapshot.rows) == ("17",)
    assert snapshot.rows[0].values["description"] == "Legacy"


async def test_postgres_database_only_apply_commits_mapping_sequence_and_readback(
    test_db_pool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: type("PolicyResult", (), {"allowed": True})(),
    )
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _reset_canonical(test_db_pool)
    await test_db_pool.execute("DROP TABLE IF EXISTS admin_webhooks CASCADE")
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
            timeout_seconds INTEGER NOT NULL,
            created_by BIGINT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhooks (
            id, url, secret_encrypted, secret_key_id, event_types,
            description, active, retry_count, timeout_seconds, created_by
        ) VALUES (17, 'https://hooks.example.com/private?token=db-canary',
                  '{"_enc":"legacy"}', 'jwt_secret', '["user.created"]',
                  'Legacy', TRUE, 3, 10, 9)
        """
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    store_path = private / "system_ops.json"
    store_path.write_text("{}", encoding="utf-8")
    os.chmod(store_path, 0o600)
    audits: list[OperationalAudit] = []

    async def audit_sink(record: OperationalAudit) -> None:
        audits.append(record)

    repository = AdminWebhookRepository(test_db_pool)
    ring = _ring()
    service = LegacyImportService(
        repository=repository,
        key_ring=ring,
        settings=_settings(),
        system_ops_path=store_path,
        application_data_paths=(private,),
        legacy_secret_decryptor=_Decryptor(),
        audit_sink=audit_sink,
    )
    request = LegacyImportRequest(
        report_path=tmp_path / "postgres-report.json",
        backup_path=None,
        rollback_key_path=None,
        operator_id=9,
        now=NOW,
        allow_legacy_credential_decryption=True,
    )

    plan = await service.build_plan(request)
    completed = await service.apply_plan(
        request,
        approved_report_digest=plan.report_digest,
        request_id="postgres-import-request",
    )

    assert plan.source_mapping == {"database:17": 17}
    assert completed.phase == "complete"
    row = await test_db_pool.fetchone(
        """
        SELECT id, active, secret_rotation_required
        FROM admin_webhook_registrations WHERE id = 17
        """
    )
    assert row == {"id": 17, "active": False, "secret_rotation_required": True}
    stored = await repository.get_protected_registration(17)
    assert stored is not None
    assert (
        ring.decrypt_bytes(
            purpose="registration.secret",
            identity={"registration_id": 17, "secret_version": 1},
            protected=stored.secret,
        )
        == b"postgres-secret-canary"
    )
    async with repository.transaction() as tx:
        assert await tx.allocate_registration_id() == 18
    assert [audit.outcome for audit in audits] == ["accepted", "completed"]
