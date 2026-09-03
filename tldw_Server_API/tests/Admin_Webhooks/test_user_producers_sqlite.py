# The imported fixture must retain its name for pytest discovery.
# ruff: noqa: F401, F811

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from tldw_Server_API.app.api.v1.endpoints.admin import admin_user as admin_user_endpoints
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminPrivilegedActionRequest,
    AdminUserCreateRequest,
)
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    ProtectedValue,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import WebhookError, WebhookErrorCode
from tldw_Server_API.app.core.Admin_Webhooks.producer import AdminWebhookEventProducer
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    RegistrationInsert,
    RegistrationTarget,
    WebhookRepositoryError,
)
from tldw_Server_API.app.services import admin_users_service
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.tests.Admin_Webhooks.test_repository_sqlite import (
    SQLiteRepositoryFixture,
    sqlite_repo,
)

NOW = datetime(2026, 8, 31, 18, 0, tzinfo=timezone.utc)
KEY_ID = "key-2026-08"
EVENT_ID = "86f2e40f-3fe7-4b12-9383-766405be78eb"
DELIVERY_ID = "445389eb-d393-48b6-8cd2-b62bf3ab0288"
COMMAND_ID = "registration-command-1"
DELETION_EVENT_ID = "7f610e4a-c28a-46b4-9c32-c3a124789103"
DELETION_DELIVERY_ID = "9177dbd4-f0bd-41ef-9d30-6f85b5d71438"
DELETION_COMMAND_ID = "deletion-command-1"


class _PasswordServiceStub:
    def validate_password_strength(
        self,
        password: str,
        username: str | None = None,
    ) -> None:
        return None

    def hash_password(self, password: str) -> str:
        return f"hash-{password}"


def _ring() -> WebhookKeyRing:
    return WebhookKeyRing(
        {KEY_ID: base64.b64encode(b"k" * 32).decode("ascii")},
        primary_id=KEY_ID,
    )


def _webhook_settings(
    mode: AdminWebhookMode = AdminWebhookMode.ON,
) -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=mode,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


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
                "import_operator_id": 7,
                "import_started_at": NOW,
                "import_approved_at": NOW,
                "artifacts_ready_at": NOW,
                "database_committed_at": NOW,
                "fingerprint_key_id": KEY_ID,
                "completed_at": NOW,
                "active_primary_key_id": KEY_ID,
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
                "rollback_expires_at": NOW + timedelta(days=7),
                "rollback_retirement_phase": "retained",
                "expected_ciphertext_digest": digest,
            },
            at=NOW,
        )


async def _start_key_rotation(repository: AdminWebhookRepository) -> None:
    current = await repository.get_migration_state()
    async with repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={
                "rotation_operation_id": "rotation-op-0123456789",
                "rotation_source_key_id": KEY_ID,
                "rotation_target_key_id": "key-2026-09",
                "rotation_phase": "rewriting",
                "rotation_table_cursor": "registration_targets",
                "rotation_key_cursor": None,
                "rotation_processed_count": 0,
                "rotation_verified_count": 0,
                "rotation_started_at": NOW,
                "rotation_completed_at": None,
            },
            at=NOW,
        )


async def _seed_matching_registration(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
    *,
    event_types: tuple[str, ...] = ("user.created",),
) -> int:
    async with repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        target = ring.encrypt_text(
            purpose="registration.target",
            identity={"registration_id": webhook_id, "target_version": 1},
            plaintext="https://hooks.example.com/capture",
        )
        secret = ring.encrypt_text(
            purpose="registration.secret",
            identity={"registration_id": webhook_id, "secret_version": 1},
            plaintext="whsec_" + ("1" * 64),
        )
        await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description="User lifecycle receiver",
                target=RegistrationTarget(
                    protected=target,
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=event_types,
                active=True,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=7,
                now=NOW - timedelta(minutes=1),
            )
        )
    return webhook_id


def _registration_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=1024,
        USER_DATA_BASE_PATH=str(tmp_path / "user_data"),
        CHROMADB_BASE_PATH=None,
    )


async def _seed_user(
    sqlite_repo: SQLiteRepositoryFixture,
    *,
    uuid: str,
    username: str,
    email: str,
    role: str,
) -> int:
    gateway = VersionedUserWriteGateway("sqlite")
    async with sqlite_repo.pool.transaction() as connection:
        result = await gateway.insert_user(
            connection,
            values={
                "uuid": uuid,
                "username": username,
                "email": email,
                "password_hash": "hash-admin",
                "role": role,
                "is_active": 1,
                "is_verified": 1,
                "created_by": None,
                "storage_quota_mb": 1024,
            },
        )
    return result.affected_user_ids[0]


async def _seed_admin_actor(sqlite_repo: SQLiteRepositoryFixture) -> int:
    return await _seed_user(
        sqlite_repo,
        uuid="00000000-0000-4000-8000-000000000007",
        username="admin-actor",
        email="admin@example.com",
        role="admin",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("is_active_override", "expected_status"),
    ((None, "active"), (False, "inactive")),
)
async def test_registration_commits_user_audit_event_activity_and_fanout_atomically(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
    is_active_override: bool | None,
    expected_status: str,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    webhook_id = await _seed_matching_registration(sqlite_repo.repository, ring)
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: DELIVERY_ID,
        clock=lambda: NOW,
    )
    actor_user_id = (
        await _seed_admin_actor(sqlite_repo)
        if is_active_override is not None
        else None
    )
    service = RegistrationService(
        db_pool=sqlite_repo.pool,
        password_service=_PasswordServiceStub(),
        settings=_registration_settings(tmp_path),
        webhook_event_producer=producer,
        command_id_factory=lambda: COMMAND_ID,
    )

    result = await service.register_user(
        username="first-user",
        email="private@example.com",
        password="Strong!Pass9",
        is_active_override=is_active_override,
        created_by=actor_user_id,
        source_request_id="request-1",
    )

    with sqlite3.connect(sqlite_repo.path) as connection:
        connection.row_factory = sqlite3.Row
        user = connection.execute(
            "SELECT id, is_active FROM users WHERE username = ?",
            ("first-user",),
        ).fetchone()
        role_count = connection.execute(
            "SELECT COUNT(*) FROM user_roles WHERE user_id = ?",
            (result["user_id"],),
        ).fetchone()[0]
        audit_count = connection.execute(
            "SELECT COUNT(*) FROM audit_logs WHERE action = 'user_registered' AND user_id = ?",
            (result["user_id"],),
        ).fetchone()[0]
        event = connection.execute(
            "SELECT * FROM admin_webhook_events WHERE event_type = 'user.created'",
        ).fetchone()
        delivery = connection.execute(
            "SELECT * FROM admin_webhook_deliveries WHERE event_id = ?",
            (EVENT_ID,),
        ).fetchone()
        activity = connection.execute(
            "SELECT first_canonical_activity_kind FROM admin_webhook_migration_state",
        ).fetchone()[0]

    assert user is not None
    assert bool(user["is_active"]) is (expected_status == "active")
    assert role_count == 1
    assert audit_count == 1
    assert event is not None
    assert event["source_command_id"] == COMMAND_ID
    assert event["source_request_id"] == "request-1"
    assert delivery is not None
    assert delivery["webhook_id"] == webhook_id
    assert activity == "event_capture"
    plaintext = ring.decrypt_event_body(
        event_id=event["id"],
        api_version=event["api_version"],
        protected=ProtectedValue(
            ciphertext_json=event["body_ciphertext_json"],
            key_id=event["body_key_id"],
        ),
    )
    body = json.loads(plaintext)
    assert body["data"] == {
        "user_id": result["user_id"],
        "status": expected_status,
        "resource_version": body["data"]["resource_version"],
        "created_at": body["data"]["created_at"],
        "updated_at": body["data"]["updated_at"],
    }
    assert "email" not in body["data"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scenario", "expected_code"),
    (
        ("migration-pending", WebhookErrorCode.MIGRATION_PENDING),
        ("key-unavailable", WebhookErrorCode.KEY_UNAVAILABLE),
        ("key-mismatch", WebhookErrorCode.KEY_CONFIGURATION_MISMATCH),
        ("key-rotation", WebhookErrorCode.KEY_ROTATION_IN_PROGRESS),
    ),
)
async def test_registration_preflight_failure_writes_nothing(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
    scenario: str,
    expected_code: WebhookErrorCode,
) -> None:
    if scenario != "migration-pending":
        await _complete_migration(sqlite_repo.repository)
    if scenario == "key-rotation":
        await _start_key_rotation(sqlite_repo.repository)
    if scenario == "key-unavailable":
        key_result = WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        )
    elif scenario == "key-mismatch":
        mismatched_ring = WebhookKeyRing(
            {"key-2026-09": base64.b64encode(b"m" * 32).decode("ascii")},
            primary_id="key-2026-09",
        )
        key_result = WebhookKeyRingLoadResult(
            ring=mismatched_ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        )
    else:
        key_result = WebhookKeyRingLoadResult(
            ring=_ring(),
            code=WebhookKeyLoadCode.AVAILABLE,
        )
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(),
        key_ring_result=key_result,
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: DELIVERY_ID,
        clock=lambda: NOW,
    )
    service = RegistrationService(
        db_pool=sqlite_repo.pool,
        password_service=_PasswordServiceStub(),
        settings=_registration_settings(tmp_path),
        webhook_event_producer=producer,
        command_id_factory=lambda: COMMAND_ID,
    )

    with pytest.raises(WebhookError) as exc_info:
        await service.register_user(
            username="blocked-user",
            email="blocked@example.com",
            password="Strong!Pass9",
        )

    assert exc_info.value.code is expected_code
    with sqlite3.connect(sqlite_repo.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM users WHERE username = 'blocked-user'",
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM admin_webhook_events",
        ).fetchone()[0] == 0
    assert not Path(service.settings.USER_DATA_BASE_PATH).exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode",
    (AdminWebhookMode.OFF, AdminWebhookMode.MIGRATE),
)
async def test_registration_preserves_non_on_behavior_without_event(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
    mode: AdminWebhookMode,
) -> None:
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(mode),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: DELIVERY_ID,
        clock=lambda: NOW,
    )
    service = RegistrationService(
        db_pool=sqlite_repo.pool,
        password_service=_PasswordServiceStub(),
        settings=_registration_settings(tmp_path),
        webhook_event_producer=producer,
        command_id_factory=lambda: COMMAND_ID,
    )

    result = await service.register_user(
        username=f"{mode.value}-user",
        email=f"{mode.value}-user@example.com",
        password="Strong!Pass9",
    )

    with sqlite3.connect(sqlite_repo.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM users WHERE id = ?",
            (result["user_id"],),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM admin_webhook_events",
        ).fetchone()[0] == 0


@pytest.mark.unit
async def test_registration_fanout_failure_rolls_back_source_and_directories(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    await _seed_matching_registration(sqlite_repo.repository, ring)
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: "invalid-delivery-id",
        clock=lambda: NOW,
    )
    service = RegistrationService(
        db_pool=sqlite_repo.pool,
        password_service=_PasswordServiceStub(),
        settings=_registration_settings(tmp_path),
        webhook_event_producer=producer,
        command_id_factory=lambda: COMMAND_ID,
    )

    with pytest.raises(WebhookRepositoryError):
        await service.register_user(
            username="rolled-back-user",
            email="rolled-back@example.com",
            password="Strong!Pass9",
        )

    with sqlite3.connect(sqlite_repo.path) as connection:
        for table in (
            "users",
            "audit_logs",
            "admin_webhook_events",
            "admin_webhook_deliveries",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    user_data_root = Path(service.settings.USER_DATA_BASE_PATH)
    assert not user_data_root.exists() or not any(user_data_root.iterdir())


@pytest.mark.unit
async def test_deactivation_commits_user_event_fanout_once_and_audits_afterward(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    webhook_id = await _seed_matching_registration(
        sqlite_repo.repository,
        ring,
        event_types=("user.deleted",),
    )
    actor_user_id = await _seed_admin_actor(sqlite_repo)
    target_user_id = await _seed_user(
        sqlite_repo,
        uuid="00000000-0000-4000-8000-000000000042",
        username="deactivation-target",
        email="private-target@example.com",
        role="user",
    )
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: DELETION_EVENT_ID,
        delivery_id_factory=lambda: DELETION_DELIVERY_ID,
        clock=lambda: NOW,
    )
    emitted_audits: list[dict[str, object]] = []

    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _allow_reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _capture_audit(**kwargs) -> None:
        emitted_audits.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(
        admin_users_service,
        "verify_privileged_action",
        _allow_reauth,
    )
    monkeypatch.setattr(
        admin_users_service,
        "_emit_admin_account_audit_event",
        _capture_audit,
    )
    principal = AuthPrincipal(
        kind="user",
        user_id=actor_user_id,
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )
    request = SimpleNamespace(
        reason="Support case 123",
        admin_password="AdminPass123!",
        admin_reauth_token=None,
    )

    result = await admin_users_service.delete_user(
        principal,
        target_user_id,
        request,
        password_service=object(),
        db_pool=sqlite_repo.pool,
        webhook_event_producer=producer,
        command_id_factory=lambda: DELETION_COMMAND_ID,
        source_request_id="request-delete-1",
    )
    repeated = await admin_users_service.delete_user(
        principal,
        target_user_id,
        request,
        password_service=object(),
        db_pool=sqlite_repo.pool,
        webhook_event_producer=producer,
        command_id_factory=lambda: "deletion-command-2",
        source_request_id="request-delete-2",
    )

    with sqlite3.connect(sqlite_repo.path) as connection:
        connection.row_factory = sqlite3.Row
        user = connection.execute(
            "SELECT is_active FROM users WHERE id = ?",
            (target_user_id,),
        ).fetchone()
        event = connection.execute(
            "SELECT * FROM admin_webhook_events WHERE event_type = 'user.deleted'",
        ).fetchone()
        deliveries = connection.execute(
            "SELECT * FROM admin_webhook_deliveries WHERE event_id = ?",
            (DELETION_EVENT_ID,),
        ).fetchall()
        durable_audits = connection.execute(
            "SELECT * FROM audit_logs WHERE action = 'admin_user_deactivated'",
        ).fetchall()

    assert result == {"message": f"User {target_user_id} has been deactivated"}
    assert repeated == result
    assert user is not None and not bool(user["is_active"])
    assert event is not None
    assert event["source_command_id"] == DELETION_COMMAND_ID
    assert event["source_request_id"] == "request-delete-1"
    assert len(deliveries) == 1
    assert deliveries[0]["webhook_id"] == webhook_id
    assert len(durable_audits) == 1
    assert durable_audits[0]["user_id"] == actor_user_id
    assert durable_audits[0]["resource_type"] == "user_account"
    assert int(durable_audits[0]["resource_id"]) == target_user_id
    assert durable_audits[0]["status"] == "success"
    assert json.loads(durable_audits[0]["details"]) == {
        "actor_id": actor_user_id,
        "target_user_id": target_user_id,
        "reason": "Support case 123",
    }
    plaintext = ring.decrypt_event_body(
        event_id=event["id"],
        api_version=event["api_version"],
        protected=ProtectedValue(
            ciphertext_json=event["body_ciphertext_json"],
            key_id=event["body_key_id"],
        ),
    )
    body = json.loads(plaintext)
    assert body["data"] == {
        "user_id": target_user_id,
        "status": "inactive",
        "resource_version": body["data"]["resource_version"],
        "created_at": body["data"]["created_at"],
        "updated_at": body["data"]["updated_at"],
    }
    assert "email" not in body["data"]
    assert len(emitted_audits) == 1
    assert emitted_audits[0]["action"] == "admin.user.deactivate"


@pytest.mark.unit
async def test_deactivation_fanout_failure_rolls_back_user_and_audit(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    await _seed_matching_registration(
        sqlite_repo.repository,
        ring,
        event_types=("user.deleted",),
    )
    actor_user_id = await _seed_admin_actor(sqlite_repo)
    target_user_id = await _seed_user(
        sqlite_repo,
        uuid="00000000-0000-4000-8000-000000000043",
        username="rollback-target",
        email="rollback-target@example.com",
        role="user",
    )
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: DELETION_EVENT_ID,
        delivery_id_factory=lambda: "invalid-delivery-id",
        clock=lambda: NOW,
    )
    emitted_audits: list[dict[str, object]] = []

    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _allow_reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _capture_audit(**kwargs) -> None:
        emitted_audits.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(
        admin_users_service,
        "verify_privileged_action",
        _allow_reauth,
    )
    monkeypatch.setattr(
        admin_users_service,
        "_emit_admin_account_audit_event",
        _capture_audit,
    )
    principal = AuthPrincipal(
        kind="user",
        user_id=actor_user_id,
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_users_service.delete_user(
            principal,
            target_user_id,
            SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
            password_service=object(),
            db_pool=sqlite_repo.pool,
            webhook_event_producer=producer,
            command_id_factory=lambda: DELETION_COMMAND_ID,
        )

    assert exc_info.value.status_code == 500
    with sqlite3.connect(sqlite_repo.path) as connection:
        assert connection.execute(
            "SELECT is_active FROM users WHERE id = ?",
            (target_user_id,),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM admin_webhook_events",
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM admin_webhook_deliveries",
        ).fetchone()[0] == 0
    assert emitted_audits == []


@pytest.mark.unit
async def test_deactivation_audit_failure_rolls_back_user_and_webhook_event(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    await _seed_matching_registration(
        sqlite_repo.repository,
        ring,
        event_types=("user.deleted",),
    )
    actor_user_id = await _seed_admin_actor(sqlite_repo)
    target_user_id = await _seed_user(
        sqlite_repo,
        uuid="00000000-0000-4000-8000-000000000044",
        username="audit-rollback-target",
        email="audit-rollback-target@example.com",
        role="user",
    )
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_webhook_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: DELETION_EVENT_ID,
        delivery_id_factory=lambda: DELETION_DELIVERY_ID,
        clock=lambda: NOW,
    )
    emitted_audits: list[dict[str, object]] = []

    async def _allow_scope(*_args, **_kwargs) -> None:
        return None

    async def _allow_reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _fail_audit(*_args, **_kwargs) -> None:
        raise RuntimeError("forced transactional audit failure")

    async def _capture_audit(**kwargs) -> None:
        emitted_audits.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow_scope,
    )
    monkeypatch.setattr(
        admin_users_service,
        "verify_privileged_action",
        _allow_reauth,
    )
    monkeypatch.setattr(
        admin_users_service,
        "_insert_user_deactivation_audit",
        _fail_audit,
    )
    monkeypatch.setattr(
        admin_users_service,
        "_emit_admin_account_audit_event",
        _capture_audit,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_users_service.delete_user(
            AuthPrincipal(
                kind="user",
                user_id=actor_user_id,
                roles=["admin"],
                permissions=["*"],
                is_admin=True,
            ),
            target_user_id,
            SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
            password_service=object(),
            db_pool=sqlite_repo.pool,
            webhook_event_producer=producer,
            command_id_factory=lambda: DELETION_COMMAND_ID,
        )

    assert exc_info.value.status_code == 500
    with sqlite3.connect(sqlite_repo.path) as connection:
        assert connection.execute(
            "SELECT is_active FROM users WHERE id = ?",
            (target_user_id,),
        ).fetchone()[0] == 1
        for table in (
            "admin_webhook_events",
            "admin_webhook_deliveries",
            "audit_logs",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert emitted_audits == []


@pytest.mark.unit
async def test_admin_user_endpoints_forward_normalized_request_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, str | None] = {}

    async def _create_user(**kwargs):
        received["create"] = kwargs["source_request_id"]
        return {"id": 42}

    async def _delete_user(*_args, **kwargs):
        received["delete"] = kwargs["source_request_id"]
        return {"message": "deleted"}

    monkeypatch.setattr(admin_users_service, "create_user", _create_user)
    monkeypatch.setattr(admin_users_service, "delete_user", _delete_user)
    principal = AuthPrincipal(
        kind="user",
        user_id=7,
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )
    create_request = Request({"type": "http", "headers": []})
    create_request.state.request_id = "request-create-1"
    delete_request = Request({"type": "http", "headers": []})
    delete_request.state.request_id = "request-delete-1"

    await admin_user_endpoints.admin_create_user(
        AdminUserCreateRequest(
            username="new-user",
            email="new-user@example.com",
            password="Strong!Pass9",
            role="user",
        ),
        create_request,
        principal,
        object(),
    )
    await admin_user_endpoints.delete_user(
        42,
        AdminPrivilegedActionRequest(
            reason="Support case 123",
            admin_password="AdminPass123!",
        ),
        delete_request,
        principal,
        object(),
    )

    assert received == {
        "create": "request-create-1",
        "delete": "request-delete-1",
    }
