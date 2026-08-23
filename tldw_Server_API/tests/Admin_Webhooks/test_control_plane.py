from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks import control_plane
from tldw_Server_API.app.core.Admin_Webhooks.audit import MutationAudit
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.control_plane import (
    AdminWebhookControlPlane,
    CreateRegistrationCommand,
    DeleteRegistrationCommand,
    PatchRegistrationCommand,
    RegistrationChanges,
    RotateSecretCommand,
    UnavailableDeliveryCapability,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    ValidatedWebhookTarget,
    WebhookError,
    WebhookErrorCode,
    WebhookRegistration,
    build_idempotency_scope,
    build_registration_etag,
    canonical_request_hash,
    idempotency_lookup_digest,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseLockError, TransactionError
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    AdminWebhookUnitOfWork,
    RegistrationInsert,
    RegistrationTarget,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
RAW_URL = "https://hooks.example.com/private/receive?token=url-query-canary"
RAW_SECRET_CANARY = "whsec_" + ("a" * 64)
IDEMPOTENCY_KEY = "0123456789abcdef0123456789abcdef"
ROTATE_KEY = "fedcba9876543210fedcba9876543210"


@dataclass
class ControlPlaneFixture:
    pool: DatabasePool
    repository: AdminWebhookRepository
    ring: WebhookKeyRing
    service: AdminWebhookControlPlane
    database_path: Path


class ReadyDeliveryCapability:
    def is_ready(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def allow_test_webhook_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(url: str, *, allow_http_dev: bool) -> ValidatedWebhookTarget:
        del allow_http_dev
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        return ValidatedWebhookTarget(
            url=url,
            hostname=hostname,
            target_display=f"{parsed.scheme}://{hostname}",
        )

    monkeypatch.setattr(
        control_plane,
        "validate_webhook_target",
        validate,
    )


def _settings(
    *,
    mode: AdminWebhookMode = AdminWebhookMode.ON,
    registration_limit: int = 100,
    active_limit: int = 25,
) -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=mode,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=registration_limit,
        active_limit=active_limit,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


def _ring() -> WebhookKeyRing:
    return WebhookKeyRing(
        {"primary": base64.b64encode(b"p" * 32).decode("ascii")},
        primary_id="primary",
    )


def _available_keys(ring: WebhookKeyRing) -> WebhookKeyRingLoadResult:
    return WebhookKeyRingLoadResult(ring=ring, code=WebhookKeyLoadCode.AVAILABLE)


def _missing_keys() -> WebhookKeyRingLoadResult:
    return WebhookKeyRingLoadResult(
        ring=None,
        code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
    )


async def _complete_migration(
    repository: AdminWebhookRepository,
    *,
    primary_key_id: str = "primary",
    rollback_expires_at: datetime | None = NOW + timedelta(days=7),
    retirement_phase: str = "retained",
) -> None:
    current = await repository.get_migration_state()
    digest = "sha256:" + ("a" * 64)
    fingerprint = "hmac-sha256:" + ("b" * 64)
    async with repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={
                "phase": "complete",
                "import_operation_id": "whmig_" + ("c" * 32),
                "import_operator_id": 7,
                "import_started_at": NOW,
                "import_approved_at": NOW,
                "artifacts_ready_at": NOW,
                "database_committed_at": NOW,
                "fingerprint_key_id": primary_key_id,
                "completed_at": NOW,
                "active_primary_key_id": primary_key_id,
                "system_ops_webhook_fingerprint": fingerprint,
                "legacy_table_fingerprint": fingerprint,
                "redacted_report_digest": digest,
                "protected_backup_ciphertext_digest": digest,
                "active_report_path": "/srv/tldw/webhook-report.json",
                "active_backup_path": "/srv/tldw/webhook-backup.enc",
                "active_key_path": "/srv/tldw/webhook-backup.key",
                "staging_report_path": "/srv/tldw/webhook-report.json.staging",
                "staging_backup_path": "/srv/tldw/webhook-backup.enc.staging",
                "staging_key_path": "/srv/tldw/webhook-backup.key.staging",
                "report_owner_id": 1000,
                "report_group_id": 1000,
                "report_mode": 384,
                "report_file_identity": "1048576:41",
                "backup_owner_id": 1000,
                "backup_group_id": 1000,
                "backup_mode": 384,
                "backup_file_identity": "1048576:42",
                "rollback_key_owner_id": 1000,
                "rollback_key_group_id": 1000,
                "rollback_key_mode": 384,
                "rollback_key_file_identity": "1048576:43",
                "rollback_expires_at": rollback_expires_at,
                "rollback_retirement_phase": retirement_phase,
                "expected_ciphertext_digest": digest,
            },
            at=NOW,
        )


@pytest_asyncio.fixture
async def plane(tmp_path: Path) -> AsyncIterator[ControlPlaneFixture]:
    database_path = tmp_path / "control-plane.db"
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{database_path}",
        )
    )
    await pool.initialize()
    repository = AdminWebhookRepository(pool)
    ring = _ring()
    await _complete_migration(repository)
    fixture = ControlPlaneFixture(
        pool=pool,
        repository=repository,
        ring=ring,
        service=AdminWebhookControlPlane(
            repository=repository,
            settings=_settings(),
            key_ring_result=_available_keys(ring),
            delivery_capability=UnavailableDeliveryCapability(),
        ),
        database_path=database_path,
    )
    try:
        yield fixture
    finally:
        await pool.close()


def _create_command(
    *,
    key: str = IDEMPOTENCY_KEY,
    description: str = "Primary receiver",
    event_types: tuple[str, ...] = ("incident.created", "user.created"),
    url: str = RAW_URL,
    timeout_seconds: int = 10,
    now: datetime = NOW,
) -> CreateRegistrationCommand:
    return CreateRegistrationCommand(
        actor_id=7,
        idempotency_key=key,
        url=url,
        event_types=event_types,
        description=description,
        timeout_seconds=timeout_seconds,
        request_id="request-create-0001",
        now=now,
    )


def _patch_command(
    registration_id: int,
    revision: int,
    changes: RegistrationChanges,
    *,
    now: datetime = NOW + timedelta(minutes=1),
) -> PatchRegistrationCommand:
    return PatchRegistrationCommand(
        actor_id=7,
        webhook_id=registration_id,
        if_match=build_registration_etag(
            webhook_id=registration_id,
            revision=revision,
        ),
        changes=changes,
        request_id="request-patch-0001",
        now=now,
    )


def _delete_command(registration_id: int, revision: int) -> DeleteRegistrationCommand:
    return DeleteRegistrationCommand(
        actor_id=7,
        webhook_id=registration_id,
        if_match=build_registration_etag(
            webhook_id=registration_id,
            revision=revision,
        ),
        request_id="request-delete-0001",
        now=NOW + timedelta(minutes=2),
    )


def _rotate_command(
    registration_id: int,
    revision: int,
    *,
    key: str = ROTATE_KEY,
) -> RotateSecretCommand:
    return RotateSecretCommand(
        actor_id=7,
        webhook_id=registration_id,
        if_match=build_registration_etag(
            webhook_id=registration_id,
            revision=revision,
        ),
        idempotency_key=key,
        request_id="request-rotate-0001",
        now=NOW + timedelta(minutes=3),
    )


def _recording_sink(records: list[MutationAudit]):
    async def sink(record: MutationAudit) -> None:
        records.append(record)

    return sink


async def _seed_registration(
    fixture: ControlPlaneFixture,
    *,
    active: bool = False,
    secret_rotation_required: bool = False,
    event_types: tuple[str, ...] = ("user.created",),
) -> WebhookRegistration:
    async with fixture.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        target = fixture.ring.encrypt_text(
            purpose="registration.target",
            identity={"registration_id": webhook_id, "target_version": 1},
            plaintext=RAW_URL,
        )
        secret = fixture.ring.encrypt_text(
            purpose="registration.secret",
            identity={"registration_id": webhook_id, "secret_version": 1},
            plaintext=RAW_SECRET_CANARY,
        )
        return await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description="Seeded receiver",
                target=RegistrationTarget(
                    protected=target,
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=event_types,
                active=active,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=secret_rotation_required,
                actor_user_id=7,
                now=NOW,
            )
        )


async def test_create_is_inactive_encrypted_and_exact_replay_returns_same_secret(
    plane: ControlPlaneFixture,
) -> None:
    records: list[MutationAudit] = []
    command = _create_command()

    first = await plane.service.create(command, audit_sink=_recording_sink(records))
    replay = await plane.service.create(command, audit_sink=_recording_sink(records))

    assert first.registration.active is False
    assert first.secret.startswith("whsec_")
    assert len(first.secret) == 70
    assert set(first.secret.removeprefix("whsec_")) <= set("0123456789abcdef")
    assert replay.secret == first.secret
    assert replay.replayed is True
    assert replay.registration.id == first.registration.id
    assert first.registration.target_display == "https://hooks.example.com"
    assert first.registration.event_types == ("user.created", "incident.created")
    assert [record.outcome for record in records] == ["accepted", "no_op"]
    assert (
        await plane.service.status(now=NOW + timedelta(minutes=1))
    ).migration.legacy_file_restore_permitted is False

    database_bytes = b"".join(
        path.read_bytes()
        for path in plane.database_path.parent.glob(f"{plane.database_path.name}*")
        if path.is_file()
    )
    assert RAW_URL.encode() not in database_bytes
    assert first.secret.encode() not in database_bytes
    assert IDEMPOTENCY_KEY.encode() not in database_bytes
    assert b"url-query-canary" not in database_bytes


async def test_create_audit_is_awaited_before_commit(plane: ControlPlaneFixture) -> None:
    observed_counts: list[int] = []

    async def sink(_record: MutationAudit) -> None:
        observed_counts.append(await plane.repository.count_registrations())

    await plane.service.create(_create_command(), audit_sink=sink)

    assert observed_counts == [0]
    assert await plane.repository.count_registrations() == 1


async def test_audit_failure_rolls_back_create_claim_and_activity(
    plane: ControlPlaneFixture,
) -> None:
    async def unavailable(_record: MutationAudit) -> None:
        raise MandatoryAuditWriteError("audit unavailable")

    with pytest.raises(WebhookError) as exc_info:
        await plane.service.create(_create_command(), audit_sink=unavailable)

    assert exc_info.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert await plane.repository.count_registrations() == 0
    assert (await plane.repository.get_migration_state()).first_canonical_activity_at is None

    records: list[MutationAudit] = []
    retried = await plane.service.create(
        _create_command(),
        audit_sink=_recording_sink(records),
    )
    assert retried.replayed is False


async def test_fail_once_audit_error_is_preserved_across_sqlite_transaction(
    plane: ControlPlaneFixture,
) -> None:
    calls = 0

    async def fail_once(_record: MutationAudit) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MandatoryAuditWriteError("audit unavailable") from DatabaseLockError()

    with pytest.raises(WebhookError) as exc_info:
        await plane.service.create(_create_command(), audit_sink=fail_once)

    assert exc_info.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert calls == 1
    assert await plane.repository.count_registrations() == 0
    assert (await plane.repository.get_migration_state()).first_canonical_activity_at is None


async def test_audit_failure_rolls_back_patch_delete_and_rotation(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane)

    async def unavailable(_record: MutationAudit) -> None:
        raise MandatoryAuditWriteError("audit unavailable")

    operations = (
        lambda: plane.service.patch(
            _patch_command(
                registration.id,
                registration.revision,
                RegistrationChanges(description="must roll back"),
            ),
            audit_sink=unavailable,
        ),
        lambda: plane.service.rotate_secret(
            _rotate_command(registration.id, registration.revision),
            audit_sink=unavailable,
        ),
        lambda: plane.service.delete(
            _delete_command(registration.id, registration.revision),
            audit_sink=unavailable,
        ),
    )
    for operation in operations:
        with pytest.raises(WebhookError) as exc_info:
            await operation()
        assert exc_info.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
        assert await plane.repository.get_registration(registration.id) == registration
        assert (await plane.repository.get_migration_state()).first_canonical_activity_at is None

    retried_rotation = await plane.service.rotate_secret(
        _rotate_command(registration.id, registration.revision),
        audit_sink=_recording_sink([]),
    )
    assert retried_rotation.replayed is False


async def test_create_commit_failure_attempts_correlated_failed_audit_and_rolls_back(
    plane: ControlPlaneFixture,
) -> None:
    class CommitFailRepository:
        def __init__(self, wrapped: AdminWebhookRepository) -> None:
            self.wrapped = wrapped

        def __getattr__(self, name: str) -> object:
            return getattr(self.wrapped, name)

        @asynccontextmanager
        async def transaction(self) -> AsyncIterator[AdminWebhookUnitOfWork]:
            async with self.wrapped.transaction() as tx:
                yield tx
                raise TransactionError("simulated commit failure")

    service = AdminWebhookControlPlane(
        repository=CommitFailRepository(plane.repository),  # type: ignore[arg-type]
        settings=_settings(),
        key_ring_result=_available_keys(plane.ring),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    records: list[MutationAudit] = []

    with pytest.raises(TransactionError):
        await service.create(_create_command(), audit_sink=_recording_sink(records))

    assert [record.outcome for record in records] == ["accepted", "failed"]
    assert records[0].request_id == records[1].request_id
    assert await plane.repository.count_registrations() == 0


@pytest.mark.parametrize(
    ("description", "event_types", "timeout_seconds"),
    [
        ("x" * 501, ("user.created",), 10),
        ("ok", (), 10),
        ("ok", ("user.created", "user.created"), 10),
        ("ok", ("*",), 10),
        ("ok", ("user.created",), 0),
        ("ok", ("user.created",), 31),
    ],
)
async def test_create_bounds_are_denied_without_unvalidated_audit_metadata(
    plane: ControlPlaneFixture,
    description: str,
    event_types: tuple[str, ...],
    timeout_seconds: int,
) -> None:
    records: list[MutationAudit] = []

    with pytest.raises(WebhookError):
        await plane.service.create(
            _create_command(
                description=description,
                event_types=event_types,
                timeout_seconds=timeout_seconds,
            ),
            audit_sink=_recording_sink(records),
        )

    assert len(records) == 1
    assert records[0].outcome == "denied"
    assert records[0].target_hostname is None
    assert records[0].event_types == ()
    assert await plane.repository.count_registrations() == 0


async def test_target_policy_denial_does_not_put_target_canary_in_audit(
    plane: ControlPlaneFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny(_url: str, *, allow_http_dev: bool) -> object:
        del allow_http_dev
        raise WebhookError(WebhookErrorCode.TARGET_REJECTED)

    monkeypatch.setattr(control_plane, "validate_webhook_target", deny)
    records: list[MutationAudit] = []

    with pytest.raises(WebhookError):
        await plane.service.create(
            _create_command(url="https://canary.example/private?secret=unique"),
            audit_sink=_recording_sink(records),
        )

    assert records[0].target_hostname is None
    assert "canary" not in repr(records)
    assert "private" not in repr(records)


async def test_create_replay_normalizes_event_order_and_conflict_precedes_key_state(
    plane: ControlPlaneFixture,
) -> None:
    records: list[MutationAudit] = []
    first = await plane.service.create(
        _create_command(event_types=("incident.created", "user.created")),
        audit_sink=_recording_sink(records),
    )
    replay = await plane.service.create(
        _create_command(event_types=("user.created", "incident.created")),
        audit_sink=_recording_sink(records),
    )
    no_key_service = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(),
        key_ring_result=_missing_keys(),
        delivery_capability=UnavailableDeliveryCapability(),
    )

    with pytest.raises(WebhookError) as conflict:
        await no_key_service.create(
            _create_command(description="Different request"),
            audit_sink=_recording_sink(records),
        )
    with pytest.raises(WebhookError) as unavailable:
        await no_key_service.create(
            _create_command(event_types=("user.created", "incident.created")),
            audit_sink=_recording_sink(records),
        )

    assert replay.secret == first.secret
    assert conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    assert unavailable.value.code is WebhookErrorCode.KEY_UNAVAILABLE


async def test_concurrent_identical_create_has_one_new_result_and_one_replay(
    plane: ControlPlaneFixture,
) -> None:
    records: list[MutationAudit] = []

    results = await asyncio.gather(
        plane.service.create(_create_command(), audit_sink=_recording_sink(records)),
        plane.service.create(_create_command(), audit_sink=_recording_sink(records)),
    )

    assert sorted(result.replayed for result in results) == [False, True]
    assert results[0].secret == results[1].secret
    assert await plane.repository.count_registrations() == 1


async def test_durable_in_progress_claim_returns_stable_conflict(
    plane: ControlPlaneFixture,
) -> None:
    command = _create_command()
    scope = build_idempotency_scope(
        actor_id=command.actor_id,
        operation="create",
        route="/admin/webhooks",
    )
    body = {
        "description": command.description,
        "event_types": ["user.created", "incident.created"],
        "timeout_seconds": command.timeout_seconds,
        "url": command.url,
    }
    async with plane.repository.transaction() as tx:
        await tx.claim_idempotency(
            lookup_digest=idempotency_lookup_digest(command.idempotency_key, scope),
            scope=scope,
            request_fingerprint=canonical_request_hash(
                command.idempotency_key,
                scope=scope,
                body=body,
                conditional_version=None,
            ),
            now=command.now,
            expires_at=command.now + timedelta(days=1),
        )
    records: list[MutationAudit] = []

    with pytest.raises(WebhookError) as exc_info:
        await plane.service.create(command, audit_sink=_recording_sink(records))

    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS
    assert records[0].outcome == "denied"


async def test_patch_no_op_does_not_change_versions_or_activity(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(
        plane,
        event_types=("user.created", "incident.created"),
    )
    records: list[MutationAudit] = []

    result = await plane.service.patch(
        _patch_command(
            registration.id,
            registration.revision,
            RegistrationChanges(
                description=registration.description,
                url=RAW_URL,
                event_types=("incident.created", "user.created"),
                timeout_seconds=10,
                active=False,
            ),
        ),
        audit_sink=_recording_sink(records),
    )

    assert result.changed is False
    assert result.registration == registration
    assert records[0].outcome == "no_op"
    assert (await plane.repository.get_migration_state()).first_canonical_activity_at is None


@pytest.mark.parametrize(
    "changes",
    [
        RegistrationChanges(description="x" * 501),
        RegistrationChanges(event_types=()),
        RegistrationChanges(event_types=("user.created", "user.created")),
        RegistrationChanges(timeout_seconds=0),
        RegistrationChanges(timeout_seconds=31),
    ],
)
async def test_patch_bounds_are_denied_without_mutation(
    plane: ControlPlaneFixture,
    changes: RegistrationChanges,
) -> None:
    registration = await _seed_registration(plane)
    records: list[MutationAudit] = []

    with pytest.raises(WebhookError):
        await plane.service.patch(
            _patch_command(registration.id, registration.revision, changes),
            audit_sink=_recording_sink(records),
        )

    assert await plane.repository.get_registration(registration.id) == registration
    assert records[0].outcome == "denied"
    assert (await plane.repository.get_migration_state()).first_canonical_activity_at is None


async def test_patch_version_rules_and_catalog_event_order(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane)
    records: list[MutationAudit] = []

    described = await plane.service.patch(
        _patch_command(
            registration.id,
            registration.revision,
            RegistrationChanges(description="Renamed"),
        ),
        audit_sink=_recording_sink(records),
    )
    configured = await plane.service.patch(
        _patch_command(
            registration.id,
            described.registration.revision,
            RegistrationChanges(
                url="https://hooks.example.com/replaced?credential=second-canary",
                event_types=("incident.updated", "user.deleted"),
                timeout_seconds=20,
            ),
            now=NOW + timedelta(minutes=2),
        ),
        audit_sink=_recording_sink(records),
    )

    assert described.registration.revision == 2
    assert described.registration.delivery_config_version == 1
    assert described.registration.target_version == 1
    assert configured.registration.revision == 3
    assert configured.registration.delivery_config_version == 2
    assert configured.registration.target_version == 2
    assert configured.registration.secret_version == 1
    assert configured.registration.event_types == ("user.deleted", "incident.updated")
    assert [record.outcome for record in records] == ["accepted", "accepted"]
    activity = await plane.repository.get_migration_state()
    assert activity.first_canonical_activity_kind == "registration_mutation"


async def test_patch_requires_current_strong_etag_and_existing_registration(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane)
    records: list[MutationAudit] = []

    for if_match, expected in (
        (None, WebhookErrorCode.PRECONDITION_REQUIRED),
        ("bad-etag-canary", WebhookErrorCode.PRECONDITION_FAILED),
        ('"admin-webhook-999-r1"', WebhookErrorCode.PRECONDITION_FAILED),
        (build_registration_etag(webhook_id=registration.id, revision=9), WebhookErrorCode.PRECONDITION_FAILED),
    ):
        with pytest.raises(WebhookError) as exc_info:
            await plane.service.patch(
                PatchRegistrationCommand(
                    actor_id=7,
                    webhook_id=registration.id,
                    if_match=if_match,
                    changes=RegistrationChanges(description="new"),
                    request_id="request-patch-etag",
                    now=NOW,
                ),
                audit_sink=_recording_sink(records),
            )
        assert exc_info.value.code is expected

    with pytest.raises(WebhookError) as missing:
        await plane.service.patch(
            _patch_command(999, 1, RegistrationChanges(description="new")),
            audit_sink=_recording_sink(records),
        )
    assert missing.value.code is WebhookErrorCode.NOT_FOUND
    assert all("bad-etag-canary" not in repr(record) for record in records)


async def test_metadata_patch_disable_and_delete_work_without_keys(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane, active=True)
    service = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(),
        key_ring_result=_missing_keys(),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    records: list[MutationAudit] = []

    metadata = await service.patch(
        _patch_command(
            registration.id,
            registration.revision,
            RegistrationChanges(description="No-key rename", event_types=("user.deleted",), timeout_seconds=11),
        ),
        audit_sink=_recording_sink(records),
    )
    disabled = await service.patch(
        _patch_command(
            registration.id,
            metadata.registration.revision,
            RegistrationChanges(active=False),
            now=NOW + timedelta(minutes=2),
        ),
        audit_sink=_recording_sink(records),
    )
    deleted = await service.delete(
        _delete_command(registration.id, disabled.registration.revision),
        audit_sink=_recording_sink(records),
    )

    assert metadata.changed is True
    assert disabled.registration.active is False
    assert deleted.registration.deleted_at is not None
    assert await plane.repository.get_registration(registration.id) is None


async def test_effective_delete_marks_first_canonical_activity(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane)

    deleted = await plane.service.delete(
        _delete_command(registration.id, registration.revision),
        audit_sink=_recording_sink([]),
    )

    assert deleted.changed is True
    activity = await plane.repository.get_migration_state()
    assert activity.first_canonical_activity_kind == "registration_mutation"
    assert activity.first_canonical_activity_at == NOW + timedelta(minutes=2)


@pytest.mark.parametrize(
    "changes",
    [
        RegistrationChanges(url="https://hooks.example.com/new"),
        RegistrationChanges(active=True),
    ],
)
async def test_protected_or_activation_patch_requires_key(
    plane: ControlPlaneFixture,
    changes: RegistrationChanges,
) -> None:
    registration = await _seed_registration(plane)
    service = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(),
        key_ring_result=_missing_keys(),
        delivery_capability=ReadyDeliveryCapability(),
    )

    with pytest.raises(WebhookError) as exc_info:
        await service.patch(
            _patch_command(registration.id, registration.revision, changes),
            audit_sink=_recording_sink([]),
        )
    assert exc_info.value.code is WebhookErrorCode.KEY_UNAVAILABLE


async def test_activation_requires_delivery_capacity_and_rotated_imported_secret(
    plane: ControlPlaneFixture,
) -> None:
    ordinary = await _seed_registration(plane)
    imported = await _seed_registration(plane, secret_rotation_required=True)
    unavailable_records: list[MutationAudit] = []

    with pytest.raises(WebhookError) as unavailable:
        await plane.service.patch(
            _patch_command(ordinary.id, ordinary.revision, RegistrationChanges(active=True)),
            audit_sink=_recording_sink(unavailable_records),
        )
    assert unavailable.value.code is WebhookErrorCode.DELIVERY_UNAVAILABLE

    ready = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(active_limit=1),
        key_ring_result=_available_keys(plane.ring),
        delivery_capability=ReadyDeliveryCapability(),
    )
    activated = await ready.patch(
        _patch_command(ordinary.id, ordinary.revision, RegistrationChanges(active=True)),
        audit_sink=_recording_sink([]),
    )
    assert activated.registration.active is True

    with pytest.raises(WebhookError) as rotation_required:
        await ready.patch(
            _patch_command(imported.id, imported.revision, RegistrationChanges(active=True)),
            audit_sink=_recording_sink([]),
        )
    assert rotation_required.value.code is WebhookErrorCode.SECRET_ROTATION_REQUIRED

    rotated = await ready.rotate_secret(
        _rotate_command(imported.id, imported.revision),
        audit_sink=_recording_sink([]),
    )
    with pytest.raises(WebhookError) as at_limit:
        await ready.patch(
            _patch_command(
                imported.id,
                rotated.registration.revision,
                RegistrationChanges(active=True),
            ),
            audit_sink=_recording_sink([]),
        )
    assert at_limit.value.code is WebhookErrorCode.ACTIVE_LIMIT


async def test_rotate_clears_marker_and_replay_precedes_changed_revision(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane, secret_rotation_required=True)
    records: list[MutationAudit] = []
    command = _rotate_command(registration.id, registration.revision)

    first = await plane.service.rotate_secret(command, audit_sink=_recording_sink(records))
    replay = await plane.service.rotate_secret(command, audit_sink=_recording_sink(records))

    assert first.secret != RAW_SECRET_CANARY
    assert first.registration.secret_rotation_required is False
    assert first.registration.revision == registration.revision + 1
    assert first.registration.delivery_config_version == registration.delivery_config_version + 1
    assert first.registration.secret_version == registration.secret_version + 1
    assert replay.secret == first.secret
    assert replay.registration.revision == first.registration.revision
    assert replay.replayed is True
    assert [record.outcome for record in records] == ["accepted", "no_op"]
    activity = await plane.repository.get_migration_state()
    assert activity.first_canonical_activity_kind == "registration_mutation"
    assert activity.first_canonical_activity_at == NOW + timedelta(minutes=3)


async def test_rotate_requires_inactive_and_current_etag(plane: ControlPlaneFixture) -> None:
    registration = await _seed_registration(plane, active=True)

    with pytest.raises(WebhookError) as active:
        await plane.service.rotate_secret(
            _rotate_command(registration.id, registration.revision),
            audit_sink=_recording_sink([]),
        )
    assert active.value.code is WebhookErrorCode.REGISTRATION_ACTIVE

    inactive = await _seed_registration(plane)
    with pytest.raises(WebhookError) as stale:
        await plane.service.rotate_secret(
            _rotate_command(inactive.id, inactive.revision + 1),
            audit_sink=_recording_sink([]),
        )
    assert stale.value.code is WebhookErrorCode.PRECONDITION_FAILED


async def test_secret_replays_are_superseded_only_by_rotation_or_delete(
    plane: ControlPlaneFixture,
) -> None:
    records: list[MutationAudit] = []
    create_command = _create_command()
    created = await plane.service.create(create_command, audit_sink=_recording_sink(records))
    described = await plane.service.patch(
        _patch_command(
            created.registration.id,
            created.registration.revision,
            RegistrationChanges(description="Description changed"),
        ),
        audit_sink=_recording_sink(records),
    )

    replay_after_description = await plane.service.create(
        create_command,
        audit_sink=_recording_sink(records),
    )
    assert replay_after_description.secret == created.secret
    assert replay_after_description.registration.revision == described.registration.revision

    rotated = await plane.service.rotate_secret(
        _rotate_command(described.registration.id, described.registration.revision),
        audit_sink=_recording_sink(records),
    )
    with pytest.raises(WebhookError) as rotated_replay:
        await plane.service.create(create_command, audit_sink=_recording_sink(records))
    assert rotated_replay.value.code is WebhookErrorCode.IDEMPOTENCY_RESULT_SUPERSEDED

    await plane.service.delete(
        _delete_command(rotated.registration.id, rotated.registration.revision),
        audit_sink=_recording_sink(records),
    )
    with pytest.raises(WebhookError) as deleted_replay:
        await plane.service.rotate_secret(
            _rotate_command(described.registration.id, described.registration.revision),
            audit_sink=_recording_sink(records),
        )
    assert deleted_replay.value.code is WebhookErrorCode.IDEMPOTENCY_RESULT_SUPERSEDED


async def test_list_get_and_status_do_not_require_decryption(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane, secret_rotation_required=True)
    service = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(),
        key_ring_result=_missing_keys(),
        delivery_capability=UnavailableDeliveryCapability(),
    )

    assert await service.get(registration.id) == registration
    assert await service.list(limit=50) == [registration]
    page = await service.list_page(limit=25, offset=0)
    assert page.items == (registration,)
    assert page.total == 1
    assert page.limit == 25
    assert page.offset == 0
    assert (await service.catalog()).api_version == "2026-07-01"
    status = await service.status(now=NOW + timedelta(days=1))
    assert status.key_state == "admin_webhook_key_unavailable"
    assert status.delivery_capability_ready is False
    assert status.limits.current_registrations == 1
    assert status.limits.current_active_registrations == 0
    assert status.migration.secret_rotation_required_count == 1
    assert status.migration.legacy_file_restore_permitted is True
    assert status.migration.rollback_expires_at == NOW + timedelta(days=7)


async def test_catalog_and_status_expose_only_effective_pr1_contract(
    plane: ControlPlaneFixture,
) -> None:
    catalog = await plane.service.catalog()
    status = await plane.service.status(now=NOW)

    assert catalog.api_version == "2026-07-01"
    assert [event.event_type for event in catalog.events] == [
        "user.created",
        "user.deleted",
        "incident.created",
        "incident.updated",
        "incident.resolved",
        "incident.notify",
    ]
    assert catalog.registration_limit == 100
    assert catalog.active_limit == 25
    assert status.mode == "on"
    assert status.route_selection == "canonical"
    assert status.schema_ready is True
    assert status.key_state == "available"
    assert not hasattr(status, "worker_heartbeat")
    assert "/srv/" not in repr(status)


async def test_mode_and_migration_gates_but_status_remains_available(
    plane: ControlPlaneFixture,
) -> None:
    off = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(mode=AdminWebhookMode.OFF),
        key_ring_result=_available_keys(plane.ring),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    assert (await off.status(now=NOW)).mode == "off"
    with pytest.raises(WebhookError) as disabled:
        await off.catalog()
    assert disabled.value.code is WebhookErrorCode.DISABLED
    assert disabled.value.code.value == "admin_webhooks_disabled"

    current = await plane.repository.get_migration_state()
    async with plane.repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={"phase": "database_committed", "completed_at": None},
            at=NOW + timedelta(minutes=1),
        )
    incomplete_on = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(mode=AdminWebhookMode.ON),
        key_ring_result=_available_keys(plane.ring),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    with pytest.raises(WebhookError) as incomplete:
        await incomplete_on.list(limit=50)
    assert incomplete.value.code is WebhookErrorCode.MIGRATION_PENDING

    migrate = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(mode=AdminWebhookMode.MIGRATE),
        key_ring_result=_available_keys(plane.ring),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    assert (await migrate.status(now=NOW)).migration.phase == "database_committed"
    with pytest.raises(WebhookError) as pending:
        await migrate.list(limit=50)
    assert pending.value.code is WebhookErrorCode.MIGRATION_PENDING


async def test_key_rotation_and_primary_mismatch_block_only_protected_writes(
    plane: ControlPlaneFixture,
) -> None:
    registration = await _seed_registration(plane)
    replay_command = _create_command(key="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    replayable = await plane.service.create(
        replay_command,
        audit_sink=_recording_sink([]),
    )
    current = await plane.repository.get_migration_state()
    async with plane.repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={
                "rotation_operation_id": "rotation-1",
                "rotation_source_key_id": "primary",
                "rotation_target_key_id": "next",
                "rotation_phase": "rewriting",
                "rotation_started_at": NOW,
            },
            at=NOW,
        )

    metadata = await plane.service.patch(
        _patch_command(
            registration.id,
            registration.revision,
            RegistrationChanges(description="Allowed during rotation"),
        ),
        audit_sink=_recording_sink([]),
    )
    with pytest.raises(WebhookError) as rotating:
        await plane.service.patch(
            _patch_command(
                registration.id,
                metadata.registration.revision,
                RegistrationChanges(url="https://hooks.example.com/new"),
            ),
            audit_sink=_recording_sink([]),
        )
    assert rotating.value.code is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS

    with pytest.raises(WebhookError) as create_blocked:
        await plane.service.create(
            _create_command(key="cccccccccccccccccccccccccccccccc"),
            audit_sink=_recording_sink([]),
        )
    assert create_blocked.value.code is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS

    with pytest.raises(WebhookError) as replay_blocked:
        await plane.service.create(
            replay_command,
            audit_sink=_recording_sink([]),
        )
    assert replay_blocked.value.code is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS
    assert replayable.registration.id != registration.id

    with pytest.raises(WebhookError) as secret_rotation_blocked:
        await plane.service.rotate_secret(
            _rotate_command(
                registration.id,
                metadata.registration.revision,
            ),
            audit_sink=_recording_sink([]),
        )
    assert (
        secret_rotation_blocked.value.code
        is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS
    )

    deleted = await plane.service.delete(
        _delete_command(registration.id, metadata.registration.revision),
        audit_sink=_recording_sink([]),
    )
    assert deleted.registration.deleted_at is not None

    state = await plane.repository.get_migration_state()
    async with plane.repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=state.state_revision,
            updates={
                "rotation_phase": "complete",
                "rotation_completed_at": NOW + timedelta(minutes=2),
                "active_primary_key_id": "different",
            },
            at=NOW + timedelta(minutes=2),
        )
    with pytest.raises(WebhookError) as mismatch:
        await plane.service.create(
            _create_command(key="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            audit_sink=_recording_sink([]),
        )
    assert mismatch.value.code is WebhookErrorCode.KEY_CONFIGURATION_MISMATCH


async def test_registration_limit_is_atomic_under_concurrent_create(
    plane: ControlPlaneFixture,
) -> None:
    service = AdminWebhookControlPlane(
        repository=plane.repository,
        settings=_settings(registration_limit=1, active_limit=1),
        key_ring_result=_available_keys(plane.ring),
        delivery_capability=UnavailableDeliveryCapability(),
    )
    records: list[MutationAudit] = []

    outcomes = await asyncio.gather(
        service.create(
            _create_command(key="11111111111111111111111111111111"),
            audit_sink=_recording_sink(records),
        ),
        service.create(
            _create_command(key="22222222222222222222222222222222"),
            audit_sink=_recording_sink(records),
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    errors = [outcome for outcome in outcomes if isinstance(outcome, WebhookError)]
    assert len(errors) == 1
    assert errors[0].code is WebhookErrorCode.REGISTRATION_LIMIT
    assert await plane.repository.count_registrations() == 1


async def test_audit_unavailability_replaces_original_denial(
    plane: ControlPlaneFixture,
) -> None:
    async def unavailable(_record: MutationAudit) -> None:
        raise MandatoryAuditWriteError("audit unavailable")

    with pytest.raises(WebhookError) as exc_info:
        await plane.service.create(
            _create_command(event_types=("*",)),
            audit_sink=unavailable,
        )

    assert exc_info.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE


async def test_errors_and_audits_are_secret_free(plane: ControlPlaneFixture) -> None:
    records: list[MutationAudit] = []
    first = await plane.service.create(
        _create_command(),
        audit_sink=_recording_sink(records),
    )

    with pytest.raises(WebhookError) as conflict:
        await plane.service.create(
            _create_command(description="different"),
            audit_sink=_recording_sink(records),
        )

    serialized = repr((conflict.value, records))
    assert first.secret not in serialized
    assert "url-query-canary" not in serialized
    assert "/private/" not in serialized
    assert IDEMPOTENCY_KEY not in serialized


def test_unavailable_delivery_capability_is_fail_closed() -> None:
    assert UnavailableDeliveryCapability().is_ready() is False
