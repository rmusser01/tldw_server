from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks.audit import (
    OperationalAudit,
    WebhookOperationalReasonCode,
)
from tldw_Server_API.app.core.Admin_Webhooks.catalog import EVENT_API_VERSION
from tldw_Server_API.app.core.Admin_Webhooks.crypto import WebhookKeyRing
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    PendingIncidentWebhookMarker,
    WebhookError,
    WebhookErrorCode,
    build_idempotency_scope,
    canonical_request_hash,
    idempotency_lookup_digest,
)
from tldw_Server_API.app.core.Admin_Webhooks.key_rotation import (
    PROTECTED_TABLE_ORDER,
    WebhookKeyRotationService,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    ProtectedRow,
    RegistrationInsert,
    RegistrationTarget,
)
from tldw_Server_API.app.services import admin_system_ops_service as system_ops

NOW = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
SOURCE_KEY_ID = "key-2026-01"
TARGET_KEY_ID = "key-2026-08"
OTHER_KEY_ID = "key-2025-06"
OPERATION_ID = "rotation-op-0123456789"
REQUEST_ID = "rotation-request-0123456789"


@dataclass
class RotationFixture:
    pool: DatabasePool
    repository: AdminWebhookRepository
    source_ring: WebhookKeyRing
    target_ring: WebhookKeyRing
    service: WebhookKeyRotationService
    store_path: Path
    audits: list[OperationalAudit]
    registration_ids: tuple[int, ...]

    async def audit_sink(self, record: OperationalAudit) -> None:
        self.audits.append(record)

    async def start(self) -> object:
        return await self.service.start(
            OPERATION_ID,
            SOURCE_KEY_ID,
            TARGET_KEY_ID,
            operator_id=9,
            request_id=REQUEST_ID,
            audit_sink=self.audit_sink,
        )

    def service_with_ring(self, ring: WebhookKeyRing) -> WebhookKeyRotationService:
        return WebhookKeyRotationService(
            repository=self.repository,
            key_ring=ring,
            system_ops_path=self.store_path,
            batch_size=1,
            clock=lambda: NOW,
        )

    async def all_database_rows(self) -> list[ProtectedRow]:
        rows: list[ProtectedRow] = []
        for table in PROTECTED_TABLE_ORDER:
            if table == "pending_incident_markers":
                continue
            after: str | None = None
            while True:
                page = await self.repository.page_protected_rows(
                    table=table,
                    after=after,
                    limit=500,
                )
                if not page:
                    break
                rows.extend(page)
                after = page[-1].row_identity
        return rows


def _ring(*, primary_id: str, include_source: bool = True) -> WebhookKeyRing:
    keys = {
        TARGET_KEY_ID: base64.b64encode(b"t" * 32).decode("ascii"),
        OTHER_KEY_ID: base64.b64encode(b"o" * 32).decode("ascii"),
    }
    if include_source:
        keys[SOURCE_KEY_ID] = base64.b64encode(b"s" * 32).decode("ascii")
    return WebhookKeyRing(keys, primary_id=primary_id)


async def _complete_migration(repository: AdminWebhookRepository) -> None:
    current = await repository.get_migration_state()
    digest = "sha256:" + ("a" * 64)
    fingerprint = "hmac-sha256:" + ("b" * 64)
    async with repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={
                "phase": "complete",
                "import_operation_id": "whmig_" + ("c" * 32),
                "import_operator_id": 9,
                "import_started_at": NOW,
                "import_approved_at": NOW,
                "database_committed_at": NOW,
                "fingerprint_key_id": SOURCE_KEY_ID,
                "active_primary_key_id": SOURCE_KEY_ID,
                "system_ops_webhook_fingerprint": fingerprint,
                "legacy_table_fingerprint": fingerprint,
                "redacted_report_digest": digest,
                "completed_at": NOW,
                "active_report_path": "/srv/tldw/webhook-report.json",
                "staging_report_path": "/srv/tldw/webhook-report.json.staging",
                "report_owner_id": 1000,
                "report_group_id": 1000,
                "report_mode": 384,
                "report_file_identity": "1048576:41",
            },
            at=NOW,
        )


async def _seed_registration(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
    *,
    suffix: str,
) -> int:
    async with repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        target = ring.encrypt_text(
            purpose="registration.target",
            identity={"registration_id": webhook_id, "target_version": 1},
            plaintext=f"https://hooks.example.com/{suffix}?token=private-{suffix}",
        )
        secret = ring.encrypt_text(
            purpose="registration.secret",
            identity={"registration_id": webhook_id, "secret_version": 1},
            plaintext="whsec_" + (suffix[0] * 64),
        )
        await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description=f"Receiver {suffix}",
                target=RegistrationTarget(
                    protected=target,
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=("incident.created",),
                active=False,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=9,
                now=NOW,
            )
        )
    return webhook_id


async def _seed_event(
    pool: DatabasePool,
    ring: WebhookKeyRing,
) -> None:
    protected = ring.encrypt_event_body(
        event_id="event-rotation-1",
        api_version=EVENT_API_VERSION,
        body=b'{"incident_id":"incident-1"}',
    )
    await pool.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, source_command_id,
            source_component, body_ciphertext_json, body_key_id, body_size_bytes,
            created_at
        ) VALUES (?, ?, ?, 'command', ?, ?, ?, ?, ?, ?)
        """,
        "event-rotation-1",
        "incident.created",
        EVENT_API_VERSION,
        "incident-command-1",
        "admin",
        protected.ciphertext_json,
        protected.key_id,
        len(b'{"incident_id":"incident-1"}'),
        NOW,
    )


async def _seed_replay(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
    *,
    registration_id: int,
) -> None:
    key = "0123456789abcdef0123456789abcdef"
    scope = build_idempotency_scope(
        actor_id=9,
        operation="create",
        route="/admin/webhooks",
    )
    digest = idempotency_lookup_digest(key, scope)
    fingerprint = canonical_request_hash(
        key,
        scope=scope,
        body={"description": "rotation fixture"},
        conditional_version=None,
    )
    replay = ring.encrypt_text(
        purpose="idempotency.secret_replay",
        identity={
            "lookup_digest": digest,
            "registration_id": registration_id,
            "secret_version": 1,
        },
        plaintext="whsec_" + ("r" * 64),
    )
    async with repository.transaction() as tx:
        await tx.claim_idempotency(
            lookup_digest=digest,
            scope=scope,
            request_fingerprint=fingerprint,
            now=NOW,
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        await tx.complete_idempotency(
            lookup_digest=digest,
            request_fingerprint=fingerprint,
            resource_id=registration_id,
            resource_version=1,
            secret_version=1,
            replay_secret=replay,
            response_status=201,
            response_metadata={"result_kind": "created"},
            at=NOW,
        )


def _seed_pending_marker(path: Path, ring: WebhookKeyRing) -> None:
    marker = PendingIncidentWebhookMarker(
        event_id="pending-event-1",
        event_type="incident.updated",
        api_version=EVENT_API_VERSION,
        source_kind="command",
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id="incident-command-pending-1",
        source_component="admin_system_ops",
        source_request_id="request-pending-1",
        body=ring.encrypt_bytes(
            purpose="pending_incident.body",
            identity={
                "event_id": "pending-event-1",
                "api_version": EVENT_API_VERSION,
                "source_command_id": "incident-command-pending-1",
            },
            plaintext=b'{"incident_id":"incident-pending"}',
        ),
        body_size_bytes=len(b'{"incident_id":"incident-pending"}'),
        created_at=NOW,
    )
    store = system_ops._default_store()
    store["webhook_pending_events"] = [marker.to_store_record()]
    system_ops._atomic_write_store(path, store)


@pytest_asyncio.fixture
async def rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[RotationFixture]:
    database_path = tmp_path / "rotation.db"
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{database_path}",
        )
    )
    await pool.initialize()
    repository = AdminWebhookRepository(pool)
    source_ring = _ring(primary_id=SOURCE_KEY_ID)
    target_ring = _ring(primary_id=TARGET_KEY_ID)
    await _complete_migration(repository)
    registration_ids = (
        await _seed_registration(repository, source_ring, suffix="alpha"),
        await _seed_registration(repository, source_ring, suffix="beta"),
    )
    await _seed_event(pool, source_ring)
    await _seed_replay(repository, source_ring, registration_id=registration_ids[0])
    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(system_ops, "_STORE_PATH", store_path)
    _seed_pending_marker(store_path, source_ring)
    audits: list[OperationalAudit] = []
    fixture = RotationFixture(
        pool=pool,
        repository=repository,
        source_ring=source_ring,
        target_ring=target_ring,
        service=WebhookKeyRotationService(
            repository=repository,
            key_ring=source_ring,
            system_ops_path=store_path,
            batch_size=1,
            clock=lambda: NOW,
        ),
        store_path=store_path,
        audits=audits,
        registration_ids=registration_ids,
    )
    try:
        yield fixture
    finally:
        await pool.close()


@pytest.mark.unit
async def test_start_audits_before_state_change_and_persists_operation(
    rotation: RotationFixture,
) -> None:
    phases_seen_by_sink: list[str | None] = []

    async def ordered_sink(record: OperationalAudit) -> None:
        if record.outcome == "accepted":
            phases_seen_by_sink.append(
                (await rotation.repository.get_migration_state()).rotation_phase
            )
        rotation.audits.append(record)

    progress = await rotation.service.start(
        OPERATION_ID,
        SOURCE_KEY_ID,
        TARGET_KEY_ID,
        operator_id=9,
        request_id=REQUEST_ID,
        audit_sink=ordered_sink,
    )

    assert phases_seen_by_sink == [None]
    assert progress.operation_id == OPERATION_ID
    assert progress.phase == "rewriting"
    assert progress.table_cursor == PROTECTED_TABLE_ORDER[0]
    assert progress.key_cursor is None
    assert progress.processed_count == 0
    assert progress.verified_count == 0
    assert progress.started_at == NOW
    assert progress.completed_at is None
    assert [record.outcome for record in rotation.audits] == ["accepted", "completed"]
    serialized_audits = json.dumps([asdict(record) for record in rotation.audits], default=str)
    for forbidden in (
        SOURCE_KEY_ID,
        TARGET_KEY_ID,
        str(rotation.store_path),
        "hmac-sha256:",
        "sha256:",
    ):
        assert forbidden not in serialized_audits

    attempted_outcomes: list[str] = []

    async def terminal_unavailable(record: OperationalAudit) -> None:
        attempted_outcomes.append(record.outcome)
        if record.outcome != "accepted":
            raise RuntimeError("terminal audit unavailable")

    replayed = await rotation.service.start(
        OPERATION_ID,
        SOURCE_KEY_ID,
        TARGET_KEY_ID,
        operator_id=9,
        request_id="rotation-start-terminal-audit-unavailable",
        audit_sink=terminal_unavailable,
    )
    assert replayed == progress
    assert attempted_outcomes == ["accepted", "completed"]


@pytest.mark.unit
async def test_preoperation_audit_failure_leaves_rotation_state_untouched(
    rotation: RotationFixture,
) -> None:
    before = await rotation.repository.get_migration_state()

    async def unavailable(_record: OperationalAudit) -> None:
        raise RuntimeError("audit unavailable")

    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.start(
            OPERATION_ID,
            SOURCE_KEY_ID,
            TARGET_KEY_ID,
            operator_id=9,
            request_id=REQUEST_ID,
            audit_sink=unavailable,
        )

    assert exc_info.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert await rotation.repository.get_migration_state() == before


@pytest.mark.parametrize(
    ("source_key_id", "target_key_id", "expected_code"),
    [
        (SOURCE_KEY_ID, SOURCE_KEY_ID, WebhookErrorCode.VALIDATION_FAILED),
        ("missing-source", TARGET_KEY_ID, WebhookErrorCode.KEY_UNAVAILABLE),
        (SOURCE_KEY_ID, "missing-target", WebhookErrorCode.KEY_UNAVAILABLE),
    ],
)
@pytest.mark.unit
async def test_start_rejects_invalid_or_missing_keys_with_correlated_audit(
    rotation: RotationFixture,
    source_key_id: str,
    target_key_id: str,
    expected_code: WebhookErrorCode,
) -> None:
    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.start(
            OPERATION_ID,
            source_key_id,
            target_key_id,
            operator_id=9,
            request_id=REQUEST_ID,
            audit_sink=rotation.audit_sink,
        )

    assert exc_info.value.code is expected_code
    assert [record.outcome for record in rotation.audits] == ["accepted", "failed"]
    assert (await rotation.repository.get_migration_state()).rotation_phase is None


@pytest.mark.unit
async def test_start_rejects_primary_mismatch_incomplete_import_and_active_operation(
    rotation: RotationFixture,
) -> None:
    target_primary_service = rotation.service_with_ring(rotation.target_ring)
    with pytest.raises(WebhookError) as local_mismatch:
        await target_primary_service.start(
            OPERATION_ID,
            SOURCE_KEY_ID,
            TARGET_KEY_ID,
            operator_id=9,
            request_id=REQUEST_ID,
            audit_sink=rotation.audit_sink,
        )
    assert local_mismatch.value.code is WebhookErrorCode.KEY_CONFIGURATION_MISMATCH

    current = await rotation.repository.get_migration_state()
    async with rotation.repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={"phase": "database_committed", "completed_at": None},
            at=NOW,
        )
    with pytest.raises(WebhookError) as incomplete:
        await rotation.service.start(
            OPERATION_ID,
            SOURCE_KEY_ID,
            TARGET_KEY_ID,
            operator_id=9,
            request_id=REQUEST_ID,
            audit_sink=rotation.audit_sink,
        )
    assert incomplete.value.code is WebhookErrorCode.MIGRATION_PENDING


@pytest.mark.unit
async def test_start_rejects_durable_primary_mismatch_without_state_change(
    rotation: RotationFixture,
) -> None:
    current = await rotation.repository.get_migration_state()
    async with rotation.repository.transaction() as tx:
        drifted = await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={"active_primary_key_id": OTHER_KEY_ID},
            at=NOW,
        )

    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.start(
            OPERATION_ID,
            SOURCE_KEY_ID,
            TARGET_KEY_ID,
            operator_id=9,
            request_id="rotation-start-durable-primary-drift",
            audit_sink=rotation.audit_sink,
        )

    assert exc_info.value.code is WebhookErrorCode.KEY_CONFIGURATION_MISMATCH
    assert await rotation.repository.get_migration_state() == drifted


@pytest.mark.unit
async def test_only_one_rotation_operation_can_be_active(
    rotation: RotationFixture,
) -> None:
    await rotation.start()

    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.start(
            "rotation-op-other-012345",
            SOURCE_KEY_ID,
            TARGET_KEY_ID,
            operator_id=9,
            request_id="rotation-request-other-012345",
            audit_sink=rotation.audit_sink,
        )

    assert exc_info.value.code is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS


@pytest.mark.unit
async def test_resume_rewrites_every_inventory_then_verify_and_finalize_cutover(
    rotation: RotationFixture,
) -> None:
    await rotation.start()
    rewritten = await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-0123456789",
        audit_sink=rotation.audit_sink,
    )

    assert rewritten.phase == "verifying"
    assert rewritten.processed_count == 7
    database_rows = await rotation.all_database_rows()
    assert len(database_rows) == 6
    assert {row.protected.key_id for row in database_rows} == {TARGET_KEY_ID}
    for row in database_rows:
        rotation.target_ring.decrypt_bytes(
            purpose=row.purpose,
            identity=row.envelope_identity,
            protected=row.protected,
        )
    store = system_ops._load_store_strict(rotation.store_path)
    marker = PendingIncidentWebhookMarker.from_store_record(
        store["webhook_pending_events"][0]
    )
    assert marker.body.key_id == TARGET_KEY_ID
    rotation.target_ring.decrypt_bytes(
        purpose=marker.envelope_purpose,
        identity=marker.envelope_identity,
        protected=marker.body,
    )

    verified = await rotation.service.verify(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-verify-0123456789",
        audit_sink=rotation.audit_sink,
    )
    assert verified.phase == "awaiting_primary_cutover"
    assert verified.verified_count == verified.processed_count == 7

    with pytest.raises(WebhookError) as before_cutover:
        await rotation.service.finalize(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-finalize-before-cutover",
            audit_sink=rotation.audit_sink,
        )
    assert before_cutover.value.code is WebhookErrorCode.KEY_CONFIGURATION_MISMATCH

    completed = await rotation.service_with_ring(rotation.target_ring).finalize(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-finalize-0123456789",
        audit_sink=rotation.audit_sink,
    )
    assert completed.phase == "complete"
    assert completed.completed_at == NOW
    durable = await rotation.repository.get_migration_state()
    assert durable.active_primary_key_id == TARGET_KEY_ID
    assert durable.rotation_phase == "complete"
    assert durable.rotation_verified_count == durable.rotation_processed_count == 7
    assert [
        (record.action, record.outcome, record.reason_code)
        for record in rotation.audits
    ] == [
        ("admin_webhook.key_rotation.start", "accepted", None),
        ("admin_webhook.key_rotation.start", "completed", None),
        ("admin_webhook.key_rotation.resume", "accepted", None),
        ("admin_webhook.key_rotation.resume", "completed", None),
        ("admin_webhook.key_rotation.verify", "accepted", None),
        ("admin_webhook.key_rotation.verify", "completed", None),
        ("admin_webhook.key_rotation.finalize", "accepted", None),
        (
            "admin_webhook.key_rotation.finalize",
            "failed",
            WebhookOperationalReasonCode.KEY_CONFIGURATION_MISMATCH,
        ),
        ("admin_webhook.key_rotation.finalize", "accepted", None),
        ("admin_webhook.key_rotation.finalize", "completed", None),
    ]
    assert [record.request_id for record in rotation.audits] == [
        REQUEST_ID,
        REQUEST_ID,
        "rotation-resume-0123456789",
        "rotation-resume-0123456789",
        "rotation-verify-0123456789",
        "rotation-verify-0123456789",
        "rotation-finalize-before-cutover",
        "rotation-finalize-before-cutover",
        "rotation-finalize-0123456789",
        "rotation-finalize-0123456789",
    ]


@pytest.mark.unit
async def test_resume_recovers_after_a_committed_database_batch(
    rotation: RotationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await rotation.start()
    original = rotation.service._rewrite_database_batch
    committed_batches = 0

    async def crash_after_commit(*args: object, **kwargs: object) -> object:
        nonlocal committed_batches
        result = await original(*args, **kwargs)
        committed_batches += 1
        if committed_batches == 1:
            raise RuntimeError("injected crash after database batch commit")
        return result

    monkeypatch.setattr(rotation.service, "_rewrite_database_batch", crash_after_commit)
    with pytest.raises(WebhookError) as crashed:
        await rotation.service.resume(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-resume-crash-1",
            audit_sink=rotation.audit_sink,
        )
    assert crashed.value.code is WebhookErrorCode.OPERATION_FAILED

    durable = await rotation.repository.get_migration_state()
    assert durable.rotation_processed_count == 1
    assert durable.rotation_key_cursor is not None
    monkeypatch.setattr(rotation.service, "_rewrite_database_batch", original)
    resumed = await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-crash-2",
        audit_sink=rotation.audit_sink,
    )
    assert resumed.phase == "verifying"
    assert resumed.processed_count == 7


@pytest.mark.unit
async def test_resume_recovers_file_publication_before_cursor_commit_once(
    rotation: RotationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await rotation.start()
    original = rotation.service._persist_file_batch_progress
    failed = False

    async def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected crash before file cursor commit")
        return await original(*args, **kwargs)

    monkeypatch.setattr(rotation.service, "_persist_file_batch_progress", fail_once)
    with pytest.raises(WebhookError) as crashed:
        await rotation.service.resume(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-file-crash-1",
            audit_sink=rotation.audit_sink,
        )
    assert crashed.value.code is WebhookErrorCode.OPERATION_FAILED

    store = system_ops._load_store_strict(rotation.store_path)
    marker = PendingIncidentWebhookMarker.from_store_record(
        store["webhook_pending_events"][0]
    )
    assert marker.body.key_id == TARGET_KEY_ID
    durable = await rotation.repository.get_migration_state()
    assert durable.rotation_processed_count == 6

    monkeypatch.setattr(rotation.service, "_persist_file_batch_progress", original)
    resumed = await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-file-crash-2",
        audit_sink=rotation.audit_sink,
    )
    assert resumed.phase == "verifying"
    assert resumed.processed_count == 7


@pytest.mark.unit
async def test_repository_protected_value_cas_rejects_concurrent_change(
    rotation: RotationFixture,
) -> None:
    rows = await rotation.repository.page_protected_rows(
        table="registration_targets",
        after=None,
        limit=1,
    )
    row = rows[0]
    concurrent = rotation.source_ring.encrypt_text(
        purpose=row.purpose,
        identity=row.envelope_identity,
        plaintext="https://hooks.example.com/concurrent",
    )
    await rotation.pool.execute(
        """
        UPDATE admin_webhook_registrations
        SET target_ciphertext_json = ?, target_key_id = ?
        WHERE id = ?
        """,
        concurrent.ciphertext_json,
        concurrent.key_id,
        int(row.row_identity),
    )
    replacement = rotation.source_ring.reencrypt_to_key(
        row.protected,
        purpose=row.purpose,
        identity=row.envelope_identity,
        target_key_id=TARGET_KEY_ID,
    )

    async with rotation.repository.transaction() as tx:
        replaced = await tx.replace_protected_value(
            row,
            expected_ciphertext=row.protected.ciphertext_json,
            replacement=replacement,
        )

    assert replaced is False


@pytest.mark.unit
async def test_database_rewrite_and_cursor_rollback_together(
    rotation: RotationFixture,
) -> None:
    await rotation.start()
    before = await rotation.repository.get_migration_state()
    row = (
        await rotation.repository.page_protected_rows(
            table="registration_targets",
            after=None,
            limit=1,
        )
    )[0]
    replacement = rotation.source_ring.reencrypt_to_key(
        row.protected,
        purpose=row.purpose,
        identity=row.envelope_identity,
        target_key_id=TARGET_KEY_ID,
    )

    with pytest.raises(TransactionError, match="SQLite transaction"):
        async with rotation.repository.transaction() as tx:
            assert await tx.replace_protected_value(
                row,
                expected_ciphertext=row.protected.ciphertext_json,
                replacement=replacement,
            )
            await tx.compare_and_set_migration_state(
                expected_revision=before.state_revision,
                updates={
                    "rotation_key_cursor": row.row_identity,
                    "rotation_processed_count": 1,
                },
                at=NOW,
            )
            raise RuntimeError("rollback batch")

    durable_row = (
        await rotation.repository.page_protected_rows(
            table="registration_targets",
            after=None,
            limit=1,
        )
    )[0]
    durable_state = await rotation.repository.get_migration_state()
    assert durable_row.protected == row.protected
    assert durable_state.rotation_key_cursor is None
    assert durable_state.rotation_processed_count == 0


@pytest.mark.unit
async def test_already_target_envelope_is_accounted_once(
    rotation: RotationFixture,
) -> None:
    row = (
        await rotation.repository.page_protected_rows(
            table="registration_targets",
            after=None,
            limit=1,
        )
    )[0]
    replacement = rotation.source_ring.reencrypt_to_key(
        row.protected,
        purpose=row.purpose,
        identity=row.envelope_identity,
        target_key_id=TARGET_KEY_ID,
    )
    async with rotation.repository.transaction() as tx:
        assert await tx.replace_protected_value(
            row,
            expected_ciphertext=row.protected.ciphertext_json,
            replacement=replacement,
        )

    await rotation.start()
    progress = await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-already-target",
        audit_sink=rotation.audit_sink,
    )
    assert progress.processed_count == 7


@pytest.mark.unit
async def test_resume_refuses_removed_source_key_and_wrong_operation(
    rotation: RotationFixture,
) -> None:
    await rotation.start()
    target_only = _ring(primary_id=TARGET_KEY_ID, include_source=False)

    with pytest.raises(WebhookError) as missing_source:
        await rotation.service_with_ring(target_only).resume(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-resume-source-removed",
            audit_sink=rotation.audit_sink,
        )
    assert missing_source.value.code is WebhookErrorCode.KEY_UNAVAILABLE

    with pytest.raises(WebhookError) as wrong_operation:
        await rotation.service.resume(
            "rotation-op-wrong-012345",
            operator_id=9,
            request_id="rotation-resume-wrong-op",
            audit_sink=rotation.audit_sink,
        )
    assert wrong_operation.value.code is WebhookErrorCode.PRECONDITION_FAILED


@pytest.mark.unit
async def test_resume_rejects_durable_primary_drift_without_mutation(
    rotation: RotationFixture,
) -> None:
    await rotation.start()
    current = await rotation.repository.get_migration_state()
    async with rotation.repository.transaction() as tx:
        drifted = await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={"active_primary_key_id": OTHER_KEY_ID},
            at=NOW,
        )
    rows_before = await rotation.all_database_rows()
    file_before = rotation.store_path.read_bytes()

    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.resume(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-resume-durable-primary-drift",
            audit_sink=rotation.audit_sink,
        )

    assert exc_info.value.code is WebhookErrorCode.KEY_CONFIGURATION_MISMATCH
    assert await rotation.repository.get_migration_state() == drifted
    assert await rotation.all_database_rows() == rows_before
    assert rotation.store_path.read_bytes() == file_before


@pytest.mark.unit
async def test_followup_commands_require_accepted_audit_before_mutation(
    rotation: RotationFixture,
) -> None:
    async def unavailable(_record: OperationalAudit) -> None:
        raise RuntimeError("audit unavailable")

    await rotation.start()
    rewriting_state = await rotation.repository.get_migration_state()
    rewriting_rows = await rotation.all_database_rows()
    rewriting_file = rotation.store_path.read_bytes()
    with pytest.raises(WebhookError) as resume_error:
        await rotation.service.resume(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-resume-audit-unavailable",
            audit_sink=unavailable,
        )
    assert resume_error.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert await rotation.repository.get_migration_state() == rewriting_state
    assert await rotation.all_database_rows() == rewriting_rows
    assert rotation.store_path.read_bytes() == rewriting_file

    await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-after-audit-restored",
        audit_sink=rotation.audit_sink,
    )
    verifying_state = await rotation.repository.get_migration_state()
    with pytest.raises(WebhookError) as verify_error:
        await rotation.service.verify(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-verify-audit-unavailable",
            audit_sink=unavailable,
        )
    assert verify_error.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert await rotation.repository.get_migration_state() == verifying_state

    await rotation.service.verify(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-verify-after-audit-restored",
        audit_sink=rotation.audit_sink,
    )
    awaiting_state = await rotation.repository.get_migration_state()
    with pytest.raises(WebhookError) as finalize_error:
        await rotation.service_with_ring(rotation.target_ring).finalize(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-finalize-audit-unavailable",
            audit_sink=unavailable,
        )
    assert finalize_error.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert await rotation.repository.get_migration_state() == awaiting_state


@pytest.mark.unit
async def test_expired_replay_secret_is_outside_rotation_inventory(
    rotation: RotationFixture,
) -> None:
    await rotation.pool.execute(
        "UPDATE admin_webhook_idempotency SET expires_at = ?",
        NOW - timedelta(days=1),
    )

    await rotation.start()
    rewritten = await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-expired-replay",
        audit_sink=rotation.audit_sink,
    )
    assert rewritten.phase == "verifying"
    assert rewritten.processed_count == 6
    assert (
        await rotation.repository.page_protected_rows(
            table="idempotency_replay_secrets",
            after=None,
            limit=500,
        )
        == []
    )

    verified = await rotation.service.verify(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-verify-expired-replay",
        audit_sink=rotation.audit_sink,
    )
    assert verified.phase == "awaiting_primary_cutover"
    assert verified.verified_count == verified.processed_count == 6


@pytest.mark.unit
async def test_replay_expiring_after_rotation_start_remains_in_stable_inventory(
    rotation: RotationFixture,
) -> None:
    await rotation.start()
    await rotation.pool.execute(
        "UPDATE admin_webhook_idempotency SET expires_at = ?",
        NOW + timedelta(minutes=1),
    )
    rewritten = await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-before-replay-expiry",
        audit_sink=rotation.audit_sink,
    )
    assert rewritten.processed_count == 7

    verified = await rotation.service.verify(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-verify-after-replay-expiry",
        audit_sink=rotation.audit_sink,
    )

    assert verified.phase == "awaiting_primary_cutover"
    assert verified.verified_count == verified.processed_count == 7


@pytest.mark.unit
async def test_malformed_pending_marker_fails_without_file_or_cursor_publication(
    rotation: RotationFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = system_ops._load_store_strict(rotation.store_path)
    store["webhook_pending_events"] = [{"event_id": "incomplete"}]
    system_ops._atomic_write_store(rotation.store_path, store)
    malformed_file = rotation.store_path.read_bytes()
    publications: list[object] = []
    monkeypatch.setattr(
        system_ops,
        "_atomic_write_store",
        lambda *args, **kwargs: publications.append((args, kwargs)),
    )

    await rotation.start()
    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.resume(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-resume-malformed-marker",
            audit_sink=rotation.audit_sink,
        )

    assert exc_info.value.code is WebhookErrorCode.PRECONDITION_FAILED
    durable = await rotation.repository.get_migration_state()
    assert durable.rotation_phase == "rewriting"
    assert durable.rotation_table_cursor == "pending_incident_markers"
    assert durable.rotation_key_cursor is None
    assert durable.rotation_processed_count == 6
    assert rotation.store_path.read_bytes() == malformed_file
    assert publications == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    ("duplicate_id", "identity_substitution", "key_loss"),
)
async def test_pending_marker_integrity_failure_never_drops_or_rewrites_records(
    rotation: RotationFixture,
    failure: str,
) -> None:
    store = system_ops._load_store_strict(rotation.store_path)
    marker_record = dict(store["webhook_pending_events"][0])
    if failure == "duplicate_id":
        store["webhook_pending_events"] = [marker_record, dict(marker_record)]
    elif failure == "identity_substitution":
        marker_record["source_command_id"] = "incident-command-substituted-1"
        store["webhook_pending_events"] = [marker_record]
    else:
        marker_record["body_key_id"] = "key-not-configured"
        store["webhook_pending_events"] = [marker_record]
    system_ops._atomic_write_store(rotation.store_path, store)
    before = rotation.store_path.read_bytes()

    await rotation.start()
    with pytest.raises(WebhookError) as exc_info:
        await rotation.service.resume(
            OPERATION_ID,
            operator_id=9,
            request_id=f"rotation-resume-marker-{failure}",
            audit_sink=rotation.audit_sink,
        )

    expected_code = (
        WebhookErrorCode.PRECONDITION_FAILED
        if failure == "duplicate_id"
        else WebhookErrorCode.KEY_UNAVAILABLE
    )
    assert exc_info.value.code is expected_code
    durable = await rotation.repository.get_migration_state()
    assert durable.rotation_table_cursor == "pending_incident_markers"
    assert durable.rotation_key_cursor is None
    assert rotation.store_path.read_bytes() == before


@pytest.mark.unit
async def test_finalize_requires_source_key_and_repeats_full_readback(
    rotation: RotationFixture,
) -> None:
    await rotation.start()
    await rotation.service.resume(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-resume-before-rescan",
        audit_sink=rotation.audit_sink,
    )
    awaiting = await rotation.service.verify(
        OPERATION_ID,
        operator_id=9,
        request_id="rotation-verify-before-rescan",
        audit_sink=rotation.audit_sink,
    )
    assert awaiting.phase == "awaiting_primary_cutover"
    awaiting_state = await rotation.repository.get_migration_state()

    target_only = _ring(primary_id=TARGET_KEY_ID, include_source=False)
    with pytest.raises(WebhookError) as missing_source:
        await rotation.service_with_ring(target_only).finalize(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-finalize-source-removed",
            audit_sink=rotation.audit_sink,
        )
    assert missing_source.value.code is WebhookErrorCode.KEY_UNAVAILABLE
    assert await rotation.repository.get_migration_state() == awaiting_state

    row = (
        await rotation.repository.page_protected_rows(
            table="registration_targets",
            after=None,
            limit=1,
        )
    )[0]
    source_again = rotation.target_ring.reencrypt_to_key(
        row.protected,
        purpose=row.purpose,
        identity=row.envelope_identity,
        target_key_id=SOURCE_KEY_ID,
    )
    async with rotation.repository.transaction() as tx:
        assert await tx.replace_protected_value(
            row,
            expected_ciphertext=row.protected.ciphertext_json,
            replacement=source_again,
        )

    with pytest.raises(WebhookError) as rescan_error:
        await rotation.service_with_ring(rotation.target_ring).finalize(
            OPERATION_ID,
            operator_id=9,
            request_id="rotation-finalize-rescan-source-found",
            audit_sink=rotation.audit_sink,
        )
    assert rescan_error.value.code is WebhookErrorCode.PRECONDITION_FAILED
    durable = await rotation.repository.get_migration_state()
    assert durable == awaiting_state
    assert durable.active_primary_key_id == SOURCE_KEY_ID
    assert durable.rotation_phase == "awaiting_primary_cutover"


@pytest.mark.unit
def test_pending_aggregate_marker_round_trips_exact_closed_shape() -> None:
    ring = _ring(primary_id=SOURCE_KEY_ID)
    identity = {
        "event_id": "pending-aggregate-1",
        "api_version": EVENT_API_VERSION,
        "aggregate_type": "incident",
        "aggregate_id": "incident-42",
        "aggregate_version": "7",
    }
    marker = PendingIncidentWebhookMarker(
        event_id="pending-aggregate-1",
        event_type="incident.resolved",
        api_version=EVENT_API_VERSION,
        source_kind="aggregate",
        aggregate_type="incident",
        aggregate_id="incident-42",
        aggregate_version="7",
        source_command_id=None,
        source_component="admin_system_ops",
        source_request_id="request-pending-aggregate-1",
        body=ring.encrypt_bytes(
            purpose="pending_incident.body",
            identity=identity,
            plaintext=b'{"incident_id":"incident-42"}',
        ),
        body_size_bytes=len(b'{"incident_id":"incident-42"}'),
        created_at=NOW,
    )

    record = marker.to_store_record()
    assert PendingIncidentWebhookMarker.from_store_record(record) == marker
    assert marker.envelope_identity == identity
    with pytest.raises(ValueError, match="record is invalid"):
        PendingIncidentWebhookMarker.from_store_record({**record, "plaintext": "no"})


@pytest.mark.unit
def test_rotation_batch_size_is_bounded(rotation: RotationFixture) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        WebhookKeyRotationService(
            repository=rotation.repository,
            key_ring=rotation.source_ring,
            system_ops_path=rotation.store_path,
            batch_size=501,
            clock=lambda: NOW,
        )
