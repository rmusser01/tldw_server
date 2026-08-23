from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    LegacyImportError,
    LegacyImportRequest,
    LegacyImportService,
    LegacySecretDecryptor,
)
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)
pytestmark = [pytest.mark.postgres, pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 8, 22, 21, 0, tzinfo=timezone.utc)


@dataclass
class _PostgresRollbackFixture:
    repository: AdminWebhookRepository
    service: LegacyImportService
    request: LegacyImportRequest
    store_path: Path
    audits: list[OperationalAudit]


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


async def _applied_file_import(
    test_db_pool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _PostgresRollbackFixture:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: type("PolicyResult", (), {"allowed": True})(),
    )
    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    await _reset_canonical(test_db_pool)
    await test_db_pool.execute("DROP TABLE IF EXISTS admin_webhooks CASCADE")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    store_path = private / "system_ops.json"
    store_path.write_text(
        json.dumps(
            {
                "incidents": [],
                "webhooks": [
                    {
                        "id": "legacy-alpha",
                        "url": "https://hooks.example.com/private",
                        "secret": "postgres-file-secret",
                        "events": ["user.created"],
                        "enabled": True,
                    }
                ],
                "webhook_deliveries": [],
            }
        ),
        encoding="utf-8",
    )
    os.chmod(store_path, 0o600)
    audits: list[OperationalAudit] = []

    async def audit_sink(record: OperationalAudit) -> None:
        audits.append(record)

    repository = AdminWebhookRepository(test_db_pool)
    service = LegacyImportService(
        repository=repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=store_path,
        application_data_paths=(private,),
        audit_sink=audit_sink,
    )
    request = LegacyImportRequest(
        report_path=tmp_path / "postgres-file-report.json",
        backup_path=tmp_path / "postgres-file-backup.enc",
        rollback_key_path=tmp_path / "postgres-file-rollback.key",
        operator_id=9,
        now=NOW,
    )
    plan = await service.build_plan(request)
    await service.apply_plan(
        request,
        approved_report_digest=plan.report_digest,
        request_id="postgres-file-import",
    )
    return _PostgresRollbackFixture(
        repository=repository,
        service=service,
        request=request,
        store_path=store_path,
        audits=audits,
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
    assert snapshot.canonical_non_deleted_count == 0


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


async def test_postgres_extract_rechecks_canonical_activity_before_publication(
    test_db_pool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = await _applied_file_import(test_db_pool, tmp_path, monkeypatch)

    async def audit_sink(record: OperationalAudit) -> None:
        fixture.audits.append(record)
        if record.action == "admin_webhook.rollback.extract" and record.outcome == "accepted":
            async with fixture.repository.transaction() as tx:
                await tx.mark_first_canonical_activity(
                    "registration_mutation",
                    NOW + timedelta(hours=1),
                )

    racing_service = LegacyImportService(
        repository=fixture.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=fixture.store_path,
        application_data_paths=(fixture.store_path.parent,),
        audit_sink=audit_sink,
    )
    output = tmp_path / "postgres-must-not-extract-after-activity.json"

    with pytest.raises(LegacyImportError) as caught:
        await racing_service.extract_rollback_backup(
            backup_path=fixture.request.backup_path,
            rollback_key_path=fixture.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="postgres-extract-racing-activity",
        )

    assert caught.value.code.value == "admin_webhook_rollback_window_closed"
    assert not output.exists()


async def test_postgres_extract_rechecks_retirement_before_publication(
    test_db_pool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = await _applied_file_import(test_db_pool, tmp_path, monkeypatch)

    async def audit_sink(record: OperationalAudit) -> None:
        fixture.audits.append(record)
        if record.action == "admin_webhook.rollback.extract" and record.outcome == "accepted":
            await fixture.service.destroy_rollback_key(
                backup_path=fixture.request.backup_path,
                rollback_key_path=fixture.request.rollback_key_path,
                operator_id=10,
                now=NOW + timedelta(days=8),
                confirmed=True,
                request_id="postgres-retirement-races-extraction",
            )

    racing_service = LegacyImportService(
        repository=fixture.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=fixture.store_path,
        application_data_paths=(fixture.store_path.parent,),
        audit_sink=audit_sink,
    )
    output = tmp_path / "postgres-must-not-extract-after-retirement.json"

    with pytest.raises(LegacyImportError) as caught:
        await racing_service.extract_rollback_backup(
            backup_path=fixture.request.backup_path,
            rollback_key_path=fixture.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="postgres-extract-racing-retirement",
        )

    assert caught.value.code.value == "admin_webhook_rollback_window_closed"
    assert not output.exists()


@pytest.mark.parametrize("closing_action", ["activity", "retirement"])
async def test_postgres_extract_holds_migration_lock_through_plaintext_publication(
    test_db_pool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closing_action: str,
) -> None:
    fixture = await _applied_file_import(test_db_pool, tmp_path, monkeypatch)
    output = tmp_path / f"postgres-locked-extraction-{closing_action}.json"
    published_while_locked = asyncio.Event()
    release_extraction = asyncio.Event()
    original_transaction = fixture.repository.transaction

    @asynccontextmanager
    async def pause_before_transaction_exit():
        async with original_transaction() as tx:
            yield tx
            if output.exists():
                published_while_locked.set()
                await release_extraction.wait()

    monkeypatch.setattr(
        fixture.repository,
        "transaction",
        pause_before_transaction_exit,
    )
    writer_repository = AdminWebhookRepository(test_db_pool)

    async def audit_sink(record: OperationalAudit) -> None:
        fixture.audits.append(record)

    writer_service = LegacyImportService(
        repository=writer_repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=fixture.store_path,
        application_data_paths=(fixture.store_path.parent,),
        audit_sink=audit_sink,
    )

    async def close_rollback_window() -> None:
        if closing_action == "activity":
            async with writer_repository.transaction() as tx:
                await tx.mark_first_canonical_activity(
                    "registration_mutation",
                    NOW + timedelta(hours=1),
                )
            return
        await writer_service.destroy_rollback_key(
            backup_path=fixture.request.backup_path,
            rollback_key_path=fixture.request.rollback_key_path,
            operator_id=10,
            now=NOW + timedelta(days=8),
            confirmed=True,
            request_id="postgres-retirement-waits-for-extraction",
        )

    extraction_task = asyncio.create_task(
        fixture.service.extract_rollback_backup(
            backup_path=fixture.request.backup_path,
            rollback_key_path=fixture.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id=f"postgres-extract-holds-lock-{closing_action}",
        )
    )
    closing_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(published_while_locked.wait(), timeout=3)
        assert output.is_file()
        closing_task = asyncio.create_task(close_rollback_window())
        await asyncio.sleep(0.1)
        assert not closing_task.done()
    finally:
        release_extraction.set()
        pending = [extraction_task]
        if closing_task is not None:
            pending.append(closing_task)
        await asyncio.gather(*pending, return_exceptions=True)

    assert extraction_task.result() == "admin_webhook_rollback_backup_extracted"
    assert closing_task is not None
    closing_task.result()
