from __future__ import annotations

import asyncio
import base64
import json
import os
import sqlite3
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks.audit import OperationalAudit
from tldw_Server_API.app.core.Admin_Webhooks.catalog import EVENT_CATALOG
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import WebhookKeyRing
from tldw_Server_API.app.core.Admin_Webhooks.legacy_import import (
    LegacyImportError,
    LegacyImportRequest,
    LegacyImportService,
    LegacyRejectionReason,
    LegacySecretDecryptor,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    RegistrationInsert,
    RegistrationTarget,
)

NOW = datetime(2026, 8, 22, 20, 0, tzinfo=timezone.utc)
CATALOG_EVENTS = tuple(item.event_type for item in EVENT_CATALOG)


@dataclass
class LegacyImportFixture:
    pool: DatabasePool
    repository: AdminWebhookRepository
    service: LegacyImportService
    request: LegacyImportRequest
    store_path: Path
    original_store_bytes: bytes
    audits: list[OperationalAudit]


def _ring() -> WebhookKeyRing:
    return WebhookKeyRing(
        {"migration-key": base64.b64encode(b"m" * 32).decode("ascii")},
        primary_id="migration-key",
    )


def _settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.MIGRATE,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


@pytest_asyncio.fixture
async def legacy_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> LegacyImportFixture:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: type("PolicyResult", (), {"allowed": True})(),
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    store_path = private / "system_ops.json"
    store = {
        "incidents": [{"id": "incident-1", "status": "open"}],
        "webhooks": [
            {
                "id": "legacy-alpha",
                "url": "https://hooks.example.com/private?token=url-canary",
                "secret": "secret-canary",
                "events": ["*"],
                "enabled": True,
            }
        ],
        "webhook_deliveries": [{"id": "delivery-1"}],
    }
    store_path.write_text(json.dumps(store), encoding="utf-8")
    os.chmod(store_path, 0o600)
    original_store_bytes = store_path.read_bytes()

    database_path = private / "authnz.db"
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{database_path}",
        )
    )
    await pool.initialize()
    repository = AdminWebhookRepository(pool)
    audits: list[OperationalAudit] = []

    async def audit_sink(record: OperationalAudit) -> None:
        audits.append(record)

    service = LegacyImportService(
        repository=repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=store_path,
        application_data_paths=(private,),
        audit_sink=audit_sink,
    )
    request = LegacyImportRequest(
        report_path=tmp_path / "report.json",
        backup_path=tmp_path / "system-ops.backup.enc",
        rollback_key_path=tmp_path / "rollback.key",
        operator_id=9,
        now=NOW,
    )
    fixture = LegacyImportFixture(
        pool=pool,
        repository=repository,
        service=service,
        request=request,
        store_path=store_path,
        original_store_bytes=original_store_bytes,
        audits=audits,
    )
    try:
        yield fixture
    finally:
        await pool.close()


@pytest.mark.unit
async def test_dry_run_is_deterministic_redacted_and_mutates_no_source_or_database(
    legacy_import: LegacyImportFixture,
) -> None:
    first = await legacy_import.service.build_plan(legacy_import.request)
    second = await legacy_import.service.build_plan(legacy_import.request)

    assert first.report_digest == second.report_digest
    assert first.operation_id == second.operation_id
    assert first.source_mapping == second.source_mapping
    assert first.requires_system_ops_backup is True
    assert len(first.accepted) == 1
    assert first.accepted[0].event_types == CATALOG_EVENTS
    assert first.accepted[0].target_display == "https://hooks.example.com"
    assert first.accepted[0].secret_rotation_required is True
    assert first.unresolved == ()

    report_bytes = legacy_import.request.report_path.read_bytes()
    assert b"secret-canary" not in report_bytes
    assert b"url-canary" not in report_bytes
    assert stat.S_IMODE(legacy_import.request.report_path.stat().st_mode) == 0o600
    assert legacy_import.store_path.read_bytes() == legacy_import.original_store_bytes
    assert await legacy_import.repository.count_registrations() == 0
    assert not legacy_import.request.backup_path.exists()
    assert not legacy_import.request.rollback_key_path.exists()


@pytest.mark.unit
async def test_source_snapshot_does_not_reopen_store_after_strict_read(
    legacy_import: LegacyImportFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == legacy_import.store_path:
            raise AssertionError("migration snapshot reopened the source path")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    plan = await legacy_import.service.build_plan(legacy_import.request)

    assert len(plan.accepted) == 1


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"   \n", None),
        (b"[]", "admin_webhook_legacy_source_invalid"),
        (b"{not-json", "admin_webhook_legacy_source_invalid"),
        (
            b'{"webhooks":[],"webhooks":[]}',
            "admin_webhook_legacy_source_invalid",
        ),
    ],
)
@pytest.mark.unit
async def test_strict_system_ops_source_handling(
    legacy_import: LegacyImportFixture,
    payload: bytes,
    expected_code: str | None,
) -> None:
    legacy_import.store_path.write_bytes(payload)
    if expected_code is None:
        request = LegacyImportRequest(
            report_path=legacy_import.request.report_path,
            backup_path=None,
            rollback_key_path=None,
            operator_id=9,
            now=NOW,
        )
        plan = await legacy_import.service.build_plan(request)
        assert plan.accepted == ()
        assert plan.requires_system_ops_backup is False
        return

    with pytest.raises(LegacyImportError) as caught:
        await legacy_import.service.build_plan(legacy_import.request)
    assert caught.value.code.value == expected_code


@pytest.mark.unit
async def test_dry_run_rejects_id_that_leaves_no_next_sequence_value(
    legacy_import: LegacyImportFixture,
) -> None:
    store = json.loads(legacy_import.store_path.read_text(encoding="utf-8"))
    store["webhooks"][0]["id"] = 2**63 - 1
    legacy_import.store_path.write_text(json.dumps(store), encoding="utf-8")

    with pytest.raises(LegacyImportError) as caught:
        await legacy_import.service.build_plan(legacy_import.request)

    assert caught.value.code.value == "admin_webhook_sequence_exhausted"


@pytest.mark.unit
async def test_repository_snapshot_reads_legacy_rows_and_canonical_allocator_state(
    legacy_import: LegacyImportFixture,
) -> None:
    with sqlite3.connect(legacy_import.repository.database_path) as connection:
        connection.execute(
            """
            INSERT INTO admin_webhooks (
                id, url, secret_encrypted, secret_key_id, event_types,
                description, active, retry_count, timeout_seconds, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                17,
                "https://db-hooks.example.com/path",
                '{"_enc":"legacy"}',
                "jwt_secret",
                '["user.created"]',
                "Database legacy row",
                1,
                3,
                10,
                9,
            ),
        )
        connection.commit()

    snapshot = await legacy_import.repository.get_legacy_import_snapshot()

    assert snapshot.table_present is True
    assert len(snapshot.rows) == 1
    assert snapshot.rows[0].source_identity == "17"
    assert snapshot.rows[0].values["secret_encrypted"] == '{"_enc":"legacy"}'
    assert snapshot.canonical_registration_ids == ()
    assert snapshot.canonical_non_deleted_count == 0
    assert snapshot.next_registration_id == 1


@pytest.mark.unit
async def test_apply_imports_inactive_preserves_secret_and_sanitizes_exact_fields(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)

    completed = await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-request-1",
    )

    assert completed.phase == "complete"
    assert completed.import_operation_id == plan.operation_id
    assert completed.rollback_retirement_phase == "retained"
    assert legacy_import.request.backup_path.is_file()
    assert legacy_import.request.rollback_key_path.is_file()
    assert stat.S_IMODE(legacy_import.request.backup_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(legacy_import.request.rollback_key_path.stat().st_mode) == 0o600

    canonical_id = next(iter(plan.source_mapping.values()))
    stored = await legacy_import.repository.get_protected_registration(canonical_id)
    assert stored is not None
    assert stored.registration.active is False
    assert stored.registration.secret_rotation_required is True
    assert (
        _ring().decrypt_text(
            purpose="registration.target",
            identity={"registration_id": canonical_id, "target_version": 1},
            protected=stored.target,
        )
        == "https://hooks.example.com/private?token=url-canary"
    )
    assert (
        _ring().decrypt_bytes(
            purpose="registration.secret",
            identity={"registration_id": canonical_id, "secret_version": 1},
            protected=stored.secret,
        )
        == b"secret-canary"
    )

    sanitized = json.loads(legacy_import.store_path.read_text(encoding="utf-8"))
    assert sanitized == {"incidents": [{"id": "incident-1", "status": "open"}]}
    assert [audit.outcome for audit in legacy_import.audits] == [
        "accepted",
        "completed",
    ]


@pytest.mark.unit
async def test_apply_audit_failure_has_zero_database_source_or_artifact_side_effects(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    state_before = await legacy_import.repository.get_migration_state()
    store_before = legacy_import.store_path.read_bytes()

    async def unavailable(_record: OperationalAudit) -> None:
        raise RuntimeError("unavailable")

    service = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=unavailable,
    )
    with pytest.raises(LegacyImportError) as caught:
        await service.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="legacy-import-request-2",
        )

    assert caught.value.code.value == "admin_webhook_audit_unavailable"
    assert await legacy_import.repository.get_migration_state() == state_before
    assert await legacy_import.repository.count_registrations() == 0
    assert legacy_import.store_path.read_bytes() == store_before
    assert not legacy_import.request.backup_path.exists()
    assert not legacy_import.request.rollback_key_path.exists()


@pytest.mark.unit
async def test_literal_report_approval_detects_payload_tampering_before_audit(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    envelope = json.loads(legacy_import.request.report_path.read_text(encoding="utf-8"))
    envelope["projected_non_deleted_count"] = 99
    legacy_import.request.report_path.write_text(json.dumps(envelope), encoding="utf-8")
    os.chmod(legacy_import.request.report_path, 0o600)

    with pytest.raises(LegacyImportError) as caught:
        await legacy_import.service.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="legacy-import-request-3",
        )

    assert caught.value.code.value == "admin_webhook_legacy_report_invalid"
    assert legacy_import.audits == []
    assert await legacy_import.repository.count_registrations() == 0


@pytest.mark.unit
async def test_fresh_install_apply_completes_without_rollback_artifacts(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    legacy_import.store_path.write_text(
        json.dumps({"incidents": [{"id": "incident-2"}]}),
        encoding="utf-8",
    )
    request = LegacyImportRequest(
        report_path=tmp_path / "fresh-report.json",
        backup_path=None,
        rollback_key_path=None,
        operator_id=9,
        now=NOW,
    )
    plan = await legacy_import.service.build_plan(request)

    completed = await legacy_import.service.apply_plan(
        request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-request-4",
    )

    assert plan.accepted == ()
    assert plan.requires_system_ops_backup is False
    assert completed.phase == "complete"
    assert completed.rollback_retirement_phase == "not_applicable"
    assert await legacy_import.repository.count_registrations() == 0


class _StaticLegacyDecryptor(LegacySecretDecryptor):
    def decrypt(self, encrypted_blob: str) -> str:
        assert encrypted_blob == '{"_enc":"legacy"}'
        return "database-secret-canary"


@pytest.mark.unit
async def test_database_only_import_preserves_collision_mapping_and_advances_sequence(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    legacy_import.store_path.write_text('{"incidents":[]}', encoding="utf-8")
    ring = _ring()
    async with legacy_import.repository.transaction() as tx:
        existing = await tx.insert_registration(
            RegistrationInsert(
                id=1,
                description="Existing canonical",
                target=RegistrationTarget(
                    protected=ring.encrypt_text(
                        purpose="registration.target",
                        identity={"registration_id": 1, "target_version": 1},
                        plaintext="https://existing.example.com/hook",
                    ),
                    hostname="existing.example.com",
                    display="https://existing.example.com",
                ),
                event_types=("user.created",),
                active=False,
                timeout_seconds=10,
                secret=ring.encrypt_bytes(
                    purpose="registration.secret",
                    identity={"registration_id": 1, "secret_version": 1},
                    plaintext=b"existing-secret",
                ),
                secret_rotation_required=False,
                actor_user_id=9,
                now=NOW,
            )
        )
        await tx.soft_delete_registration(
            existing.id,
            expected_revision=existing.revision,
            actor_user_id=9,
            at=NOW + timedelta(minutes=1),
        )
        await tx.ensure_registration_sequence_above(1)
    with sqlite3.connect(legacy_import.repository.database_path) as connection:
        connection.execute(
            """
            INSERT INTO admin_webhooks (
                id, url, secret_encrypted, secret_key_id, event_types,
                description, active, retry_count, timeout_seconds, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "https://database.example.com/private?token=db-url-canary",
                '{"_enc":"legacy"}',
                "jwt_secret",
                '["incident.created"]',
                "Database import",
                1,
                3,
                10,
                9,
            ),
        )
        connection.commit()

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    service = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=ring,
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        legacy_secret_decryptor=_StaticLegacyDecryptor(),
        audit_sink=audit_sink,
    )
    request = LegacyImportRequest(
        report_path=tmp_path / "database-report.json",
        backup_path=None,
        rollback_key_path=None,
        operator_id=9,
        now=NOW,
        allow_legacy_credential_decryption=True,
    )
    plan = await service.build_plan(request)

    assert plan.source_mapping == {"database:1": 2}
    assert plan.projected_non_deleted_count == 1
    completed = await service.apply_plan(
        request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-request-db",
    )
    imported = await legacy_import.repository.get_protected_registration(2)
    assert imported is not None
    assert (
        ring.decrypt_bytes(
            purpose="registration.secret",
            identity={"registration_id": 2, "secret_version": 1},
            protected=imported.secret,
        )
        == b"database-secret-canary"
    )
    async with legacy_import.repository.transaction() as tx:
        assert await tx.allocate_registration_id() == 3
    assert completed.rollback_retirement_phase == "not_applicable"
    assert not (tmp_path / "database.backup").exists()


@pytest.mark.unit
async def test_encrypted_database_source_requires_explicit_decryption_flag(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    legacy_import.store_path.write_text("{}", encoding="utf-8")
    with sqlite3.connect(legacy_import.repository.database_path) as connection:
        connection.execute(
            """
            INSERT INTO admin_webhooks (
                id, url, secret_encrypted, event_types, description,
                active, retry_count, timeout_seconds
            ) VALUES (1, 'https://database.example.com/hook',
                      '{"_enc":"legacy"}', '["user.created"]', '', 1, 3, 10)
            """
        )
        connection.commit()
    request = LegacyImportRequest(
        report_path=tmp_path / "unresolved-report.json",
        backup_path=None,
        rollback_key_path=None,
        operator_id=9,
        now=NOW,
    )

    plan = await legacy_import.service.build_plan(request)

    assert plan.accepted == ()
    assert len(plan.unresolved) == 1
    assert plan.unresolved[0].reason_code == "legacy_credential_decryption_required"
    with pytest.raises(LegacyImportError) as caught:
        await legacy_import.service.apply_plan(
            request,
            approved_report_digest=plan.report_digest,
            request_id="legacy-import-request-unresolved",
        )
    assert caught.value.code.value == "admin_webhook_legacy_unresolved"
    assert legacy_import.audits == []


@pytest.mark.unit
async def test_reject_source_is_audited_and_bound_to_exact_record_fingerprint(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    record = plan.accepted[0]

    await legacy_import.service.reject_source(
        source_kind=record.source_kind,
        source_identity=record.source_identity,
        source_record_fingerprint=record.source_record_fingerprint,
        reason_code=LegacyRejectionReason.OPERATOR_EXCLUDED,
        operator_id=9,
        now=NOW,
        request_id="legacy-reject-request-1",
    )
    rejected_plan = await legacy_import.service.build_plan(legacy_import.request)

    assert rejected_plan.accepted == ()
    assert len(rejected_plan.explicitly_rejected) == 1
    assert rejected_plan.explicitly_rejected[0].reason_code == "operator_excluded"
    assert [audit.action for audit in legacy_import.audits] == [
        "admin_webhook.import.reject_source",
        "admin_webhook.import.reject_source",
    ]
    store = json.loads(legacy_import.store_path.read_text(encoding="utf-8"))
    store["webhooks"][0]["events"] = ["user.created"]
    legacy_import.store_path.write_text(json.dumps(store), encoding="utf-8")
    drifted = await legacy_import.service.build_plan(legacy_import.request)
    assert drifted.explicitly_rejected == ()
    assert len(drifted.accepted) == 1


@pytest.mark.unit
async def test_extract_rollback_backup_writes_distinct_private_plaintext_file(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-extract",
    )
    output = tmp_path / "extracted-system-ops.json"

    result = await legacy_import.service.extract_rollback_backup(
        backup_path=legacy_import.request.backup_path,
        rollback_key_path=legacy_import.request.rollback_key_path,
        output_path=output,
        operator_id=9,
        now=NOW + timedelta(days=1),
        confirmed=True,
        request_id="legacy-extract-request-1",
    )

    assert result == "admin_webhook_rollback_backup_extracted"
    assert output.read_bytes() == legacy_import.original_store_bytes
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [audit.outcome for audit in legacy_import.audits[-2:]] == [
        "accepted",
        "completed",
    ]
    assert [audit.action for audit in legacy_import.audits[-2:]] == [
        "admin_webhook.rollback.extract",
        "admin_webhook.rollback.extract",
    ]


@pytest.mark.unit
async def test_extract_cleans_created_plaintext_when_transaction_exit_is_cancelled(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-cancelled-extract",
    )
    output = tmp_path / "cancelled-extraction.json"
    original_transaction = legacy_import.repository.transaction

    @asynccontextmanager
    async def cancel_after_transaction_exit():
        async with original_transaction() as tx:
            yield tx
        if output.exists():
            raise asyncio.CancelledError

    monkeypatch.setattr(
        legacy_import.repository,
        "transaction",
        cancel_after_transaction_exit,
    )

    with pytest.raises(asyncio.CancelledError):
        await legacy_import.service.extract_rollback_backup(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="legacy-extract-cancelled-on-exit",
        )

    assert not output.exists()


@pytest.mark.unit
async def test_extract_failure_cleanup_preserves_replaced_output_inode(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-replaced-output",
    )
    output = tmp_path / "replaced-extraction.json"
    replacement = b"replacement-owned-by-another-process"
    original_transaction = legacy_import.repository.transaction

    @asynccontextmanager
    async def replace_output_then_fail():
        async with original_transaction() as tx:
            yield tx
        if output.exists():
            output.unlink()
            output.write_bytes(replacement)
            os.chmod(output, 0o600)
            raise RuntimeError("injected transaction-exit failure")

    monkeypatch.setattr(
        legacy_import.repository,
        "transaction",
        replace_output_then_fail,
    )

    with pytest.raises(LegacyImportError):
        await legacy_import.service.extract_rollback_backup(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="legacy-extract-replaced-on-exit",
        )

    assert output.read_bytes() == replacement


@pytest.mark.parametrize("closing_action", ["activity", "retirement"])
@pytest.mark.unit
async def test_extract_holds_migration_lock_through_plaintext_publication(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    closing_action: str,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id=f"legacy-import-before-locked-{closing_action}",
    )
    output = tmp_path / f"locked-extraction-{closing_action}.json"
    published_while_locked = asyncio.Event()
    release_extraction = asyncio.Event()
    original_transaction = legacy_import.repository.transaction

    @asynccontextmanager
    async def pause_before_transaction_exit():
        async with original_transaction() as tx:
            yield tx
            if output.exists():
                published_while_locked.set()
                await release_extraction.wait()

    monkeypatch.setattr(
        legacy_import.repository,
        "transaction",
        pause_before_transaction_exit,
    )
    writer_repository = AdminWebhookRepository(legacy_import.pool)

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    writer_service = LegacyImportService(
        repository=writer_repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
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
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            operator_id=10,
            now=NOW + timedelta(days=8),
            confirmed=True,
            request_id="legacy-retirement-waits-for-extraction",
        )

    extraction_task = asyncio.create_task(
        legacy_import.service.extract_rollback_backup(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id=f"legacy-extract-holds-lock-{closing_action}",
        )
    )
    closing_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(published_while_locked.wait(), timeout=3)
        assert output.read_bytes() == legacy_import.original_store_bytes
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


@pytest.mark.unit
async def test_extract_checks_closed_window_before_artifact_access_or_audit(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-closed-extract",
    )
    async with legacy_import.repository.transaction() as tx:
        await tx.mark_first_canonical_activity(
            "registration_mutation",
            NOW + timedelta(hours=1),
        )
    legacy_import.request.backup_path.unlink()
    legacy_import.request.rollback_key_path.unlink()
    audit_count = len(legacy_import.audits)

    with pytest.raises(LegacyImportError) as caught:
        await legacy_import.service.extract_rollback_backup(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            output_path=tmp_path / "must-not-exist.json",
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="legacy-extract-request-closed",
        )

    assert caught.value.code.value == "admin_webhook_rollback_window_closed"
    assert len(legacy_import.audits) == audit_count
    assert not (tmp_path / "must-not-exist.json").exists()


@pytest.mark.unit
async def test_extract_rechecks_activity_after_accepted_audit_before_artifact_access(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-racing-activity",
    )

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)
        if record.action == "admin_webhook.rollback.extract" and record.outcome == "accepted":
            async with legacy_import.repository.transaction() as tx:
                await tx.mark_first_canonical_activity(
                    "registration_mutation",
                    NOW + timedelta(hours=1),
                )

    racing_service = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
    )
    output = tmp_path / "must-not-extract-after-activity.json"

    with pytest.raises(LegacyImportError) as caught:
        await racing_service.extract_rollback_backup(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="legacy-extract-racing-activity",
        )

    assert caught.value.code.value == "admin_webhook_rollback_window_closed"
    assert not output.exists()


@pytest.mark.unit
async def test_extract_rechecks_retirement_after_accepted_audit_before_artifact_access(
    legacy_import: LegacyImportFixture,
    tmp_path: Path,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-racing-retirement",
    )

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)
        if record.action == "admin_webhook.rollback.extract" and record.outcome == "accepted":
            await legacy_import.service.destroy_rollback_key(
                backup_path=legacy_import.request.backup_path,
                rollback_key_path=legacy_import.request.rollback_key_path,
                operator_id=10,
                now=NOW + timedelta(days=8),
                confirmed=True,
                request_id="legacy-retirement-races-extraction",
            )

    racing_service = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
    )
    output = tmp_path / "must-not-extract-after-retirement.json"

    with pytest.raises(LegacyImportError) as caught:
        await racing_service.extract_rollback_backup(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            output_path=output,
            operator_id=9,
            now=NOW + timedelta(days=1),
            confirmed=True,
            request_id="legacy-extract-racing-retirement",
        )

    assert caught.value.code.value == "admin_webhook_rollback_window_closed"
    assert not output.exists()


@pytest.mark.unit
async def test_destroy_rollback_key_requires_expiry_then_retires_idempotently(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)
    await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="legacy-import-before-destroy",
    )
    audit_count = len(legacy_import.audits)
    with pytest.raises(LegacyImportError) as early:
        await legacy_import.service.destroy_rollback_key(
            backup_path=legacy_import.request.backup_path,
            rollback_key_path=legacy_import.request.rollback_key_path,
            operator_id=9,
            now=NOW + timedelta(days=6),
            confirmed=True,
            request_id="legacy-destroy-too-early",
        )
    assert early.value.code.value == "admin_webhook_rollback_window_closed"
    assert len(legacy_import.audits) == audit_count

    retired = await legacy_import.service.destroy_rollback_key(
        backup_path=legacy_import.request.backup_path,
        rollback_key_path=legacy_import.request.rollback_key_path,
        operator_id=9,
        now=NOW + timedelta(days=8),
        confirmed=True,
        request_id="legacy-destroy-request-1",
    )
    assert retired == "admin_webhook_rollback_artifacts_retired"
    assert not legacy_import.request.backup_path.exists()
    assert not legacy_import.request.rollback_key_path.exists()
    state = await legacy_import.repository.get_migration_state()
    assert state.rollback_retirement_phase == "retired"
    audit_count = len(legacy_import.audits)

    repeated = await legacy_import.service.destroy_rollback_key(
        backup_path=legacy_import.request.backup_path,
        rollback_key_path=legacy_import.request.rollback_key_path,
        operator_id=9,
        now=NOW + timedelta(days=9),
        confirmed=True,
        request_id="legacy-destroy-request-2",
    )
    assert repeated == "admin_webhook_rollback_artifacts_already_retired"
    assert len(legacy_import.audits) == audit_count


@pytest.mark.parametrize(
    "crash_stage",
    [
        "after_artifacts_pending",
        "after_key_publish",
        "after_backup_publish",
        "after_artifacts_ready",
        "after_database_commit",
        "after_canonical_readback",
        "after_source_replace",
        "after_complete",
    ],
)
@pytest.mark.unit
async def test_apply_resumes_from_every_durable_stage_without_duplicate_import(
    legacy_import: LegacyImportFixture,
    crash_stage: str,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)

    def inject(stage: str) -> None:
        if stage == crash_stage:
            raise RuntimeError("injected crash")

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    crashing = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
        failure_injector=inject,
    )
    with pytest.raises(LegacyImportError):
        await crashing.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id=f"crash-{crash_stage}",
        )

    completed = await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id=f"resume-{crash_stage}",
    )

    assert completed.phase == "complete"
    assert await legacy_import.repository.count_registrations() == 1
    sanitized = json.loads(legacy_import.store_path.read_text(encoding="utf-8"))
    assert "webhooks" not in sanitized
    assert "webhook_deliveries" not in sanitized


@pytest.mark.unit
async def test_resume_after_backup_publish_preserves_unrelated_store_changes(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)

    def inject(stage: str) -> None:
        if stage == "after_backup_publish":
            raise RuntimeError("injected crash")

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    crashing = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
        failure_injector=inject,
    )
    with pytest.raises(LegacyImportError):
        await crashing.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="crash-after-backup-before-unrelated-change",
        )

    changed = json.loads(legacy_import.store_path.read_text(encoding="utf-8"))
    changed["incidents"][0]["status"] = "resolved"
    changed["incident_notes"] = ["preserve-on-resume"]
    legacy_import.store_path.write_text(json.dumps(changed), encoding="utf-8")

    completed = await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="resume-after-unrelated-change",
    )

    assert completed.phase == "complete"
    assert json.loads(legacy_import.store_path.read_text(encoding="utf-8")) == {
        "incidents": [{"id": "incident-1", "status": "resolved"}],
        "incident_notes": ["preserve-on-resume"],
    }


@pytest.mark.unit
async def test_resume_rejects_authenticated_backup_with_wrong_webhook_subtree(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)

    def inject(stage: str) -> None:
        if stage == "after_backup_publish":
            raise RuntimeError("injected crash")

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    crashing = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
        failure_injector=inject,
    )
    with pytest.raises(LegacyImportError):
        await crashing.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="crash-before-authenticated-backup-replacement",
        )

    key_payload = json.loads(
        legacy_import.request.rollback_key_path.read_text(encoding="utf-8")
    )
    rollback_ring = WebhookKeyRing(
        {"rollback": key_payload["key_b64"]},
        primary_id="rollback",
    )
    protected = rollback_ring.encrypt_bytes(
        purpose="legacy.system_ops.backup",
        identity={
            "operation_id": plan.operation_id,
            "source_fingerprint": plan.source_fingerprints["system_ops"],
        },
        plaintext=json.dumps(
            {"incidents": [], "webhooks": [], "webhook_deliveries": []},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    replacement = json.dumps(
        {
            "schema_version": 1,
            "key_id": protected.key_id,
            "ciphertext_json": protected.ciphertext_json,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_import.request.backup_path.write_bytes(replacement)
    os.chmod(legacy_import.request.backup_path, 0o600)

    with pytest.raises(LegacyImportError) as caught:
        await legacy_import.service.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="resume-with-wrong-authenticated-backup",
        )

    assert caught.value.code.value == "admin_webhook_operation_failed"
    assert legacy_import.request.backup_path.read_bytes() == replacement
    assert await legacy_import.repository.count_registrations() == 0


@pytest.mark.unit
async def test_reserved_operation_resumes_when_reviewed_report_is_missing(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)

    def inject(stage: str) -> None:
        if stage == "after_artifacts_pending":
            raise RuntimeError("injected crash")

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    crashing = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
        failure_injector=inject,
    )
    with pytest.raises(LegacyImportError):
        await crashing.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="crash-before-missing-report",
        )
    legacy_import.request.report_path.unlink()

    completed = await legacy_import.service.apply_plan(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="resume-without-report",
    )

    assert completed.phase == "complete"
    assert await legacy_import.repository.count_registrations() == 1


@pytest.mark.unit
async def test_public_verify_and_sanitize_resumes_database_committed_import(
    legacy_import: LegacyImportFixture,
) -> None:
    plan = await legacy_import.service.build_plan(legacy_import.request)

    def inject(stage: str) -> None:
        if stage == "after_database_commit":
            raise RuntimeError("injected crash")

    async def audit_sink(record: OperationalAudit) -> None:
        legacy_import.audits.append(record)

    crashing = LegacyImportService(
        repository=legacy_import.repository,
        key_ring=_ring(),
        settings=_settings(),
        system_ops_path=legacy_import.store_path,
        application_data_paths=(legacy_import.store_path.parent,),
        audit_sink=audit_sink,
        failure_injector=inject,
    )
    with pytest.raises(LegacyImportError):
        await crashing.apply_plan(
            legacy_import.request,
            approved_report_digest=plan.report_digest,
            request_id="crash-before-public-verification",
        )

    completed = await legacy_import.service.verify_and_sanitize(
        legacy_import.request,
        approved_report_digest=plan.report_digest,
        request_id="public-verification-resume",
    )

    assert completed.phase == "complete"
    assert await legacy_import.repository.count_registrations() == 1
