# The imported fixture must retain its name for pytest discovery.
# ruff: noqa: F401, F811

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from tldw_Server_API.app.core.Admin_Webhooks.producer import AdminWebhookEventProducer
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    RegistrationInsert,
    RegistrationTarget,
)
from tldw_Server_API.app.services import admin_users_service
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.tests.Admin_Webhooks.test_repository_postgres import (
    PostgreSQLRepositoryFixture,
    _complete_migration,
    pg_repo,
)

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)
pytestmark = pytest.mark.postgres

NOW = datetime(2026, 8, 31, 18, 0, tzinfo=timezone.utc)
KEY_ID = "key-2026-08"
EVENT_ID = "86f2e40f-3fe7-4b12-9383-766405be78eb"
DELIVERY_ID = "445389eb-d393-48b6-8cd2-b62bf3ab0288"
COMMAND_ID = "postgres-user-command-1"


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


def _settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.ON,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


def _producer(
    pg_repo: PostgreSQLRepositoryFixture,
    ring: WebhookKeyRing,
) -> AdminWebhookEventProducer:
    return AdminWebhookEventProducer(
        repository=pg_repo.repository,
        settings=_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: DELIVERY_ID,
        clock=lambda: NOW,
    )


async def _seed_registration(
    pg_repo: PostgreSQLRepositoryFixture,
    ring: WebhookKeyRing,
    *,
    event_type: str,
) -> int:
    async with pg_repo.repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description="User lifecycle receiver",
                target=RegistrationTarget(
                    protected=ring.encrypt_text(
                        purpose="registration.target",
                        identity={
                            "registration_id": webhook_id,
                            "target_version": 1,
                        },
                        plaintext="https://hooks.example.com/capture",
                    ),
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=(event_type,),
                active=True,
                timeout_seconds=10,
                secret=ring.encrypt_text(
                    purpose="registration.secret",
                    identity={
                        "registration_id": webhook_id,
                        "secret_version": 1,
                    },
                    plaintext="whsec_" + ("1" * 64),
                ),
                secret_rotation_required=False,
                actor_user_id=7,
                now=NOW,
            )
        )
    return webhook_id


async def _seed_user(
    pg_repo: PostgreSQLRepositoryFixture,
    *,
    uuid: str,
    username: str,
    email: str,
    role: str,
    created_by: int | None = None,
) -> int:
    gateway = VersionedUserWriteGateway("postgres")
    async with pg_repo.pool.transaction() as connection:
        result = await gateway.insert_user(
            connection,
            values={
                "uuid": uuid,
                "username": username,
                "email": email,
                "password_hash": "hash-user",
                "role": role,
                "is_active": True,
                "is_verified": True,
                "created_by": created_by,
                "storage_quota_mb": 1024,
            },
        )
    return result.affected_user_ids[0]


def _registration_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=1024,
        USER_DATA_BASE_PATH=str(tmp_path / "user_data"),
        CHROMADB_BASE_PATH=None,
    )


@pytest.mark.integration
async def test_postgres_registration_captures_user_and_delivery_atomically(
    pg_repo: PostgreSQLRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    await _complete_migration(pg_repo.repository)
    webhook_id = await _seed_registration(pg_repo, ring, event_type="user.created")
    service = RegistrationService(
        db_pool=pg_repo.pool,
        password_service=_PasswordServiceStub(),
        settings=_registration_settings(tmp_path),
        webhook_event_producer=_producer(pg_repo, ring),
        command_id_factory=lambda: COMMAND_ID,
    )

    result = await service.register_user(
        username="postgres-created-user",
        email="postgres-created@example.com",
        password="Strong!Pass9",
        source_request_id="postgres-create-request",
    )

    event = await pg_repo.pool.fetchrow(
        "SELECT * FROM admin_webhook_events WHERE id = ?",
        EVENT_ID,
    )
    delivery = await pg_repo.pool.fetchrow(
        "SELECT * FROM admin_webhook_deliveries WHERE event_id = ?",
        EVENT_ID,
    )
    assert event is not None and event["source_command_id"] == COMMAND_ID
    assert delivery is not None and int(delivery["webhook_id"]) == webhook_id
    body = json.loads(
        ring.decrypt_event_body(
            event_id=event["id"],
            api_version=event["api_version"],
            protected=ProtectedValue(
                ciphertext_json=event["body_ciphertext_json"],
                key_id=event["body_key_id"],
            ),
        )
    )
    assert body["data"]["user_id"] == result["user_id"]
    assert body["data"]["status"] == "active"
    assert set(body["data"]) == {
        "user_id",
        "status",
        "resource_version",
        "created_at",
        "updated_at",
    }


@pytest.mark.integration
async def test_postgres_deactivation_captures_user_and_delivery_atomically(
    pg_repo: PostgreSQLRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = _ring()
    await _complete_migration(pg_repo.repository)
    webhook_id = await _seed_registration(pg_repo, ring, event_type="user.deleted")
    actor_id = await _seed_user(
        pg_repo,
        uuid="00000000-0000-4000-8000-000000000007",
        username="postgres-admin-actor",
        email="postgres-admin@example.com",
        role="admin",
    )
    target_id = await _seed_user(
        pg_repo,
        uuid="00000000-0000-4000-8000-000000000042",
        username="postgres-delete-target",
        email="postgres-delete-target@example.com",
        role="user",
        created_by=actor_id,
    )
    emitted: list[dict[str, object]] = []

    async def _allow(*_args, **_kwargs) -> None:
        return None

    async def _reauth(*_args, **_kwargs) -> str:
        return "Support case 123"

    async def _audit(**kwargs) -> None:
        emitted.append(kwargs)

    monkeypatch.setattr(
        admin_users_service.admin_scope_service,
        "enforce_admin_user_scope",
        _allow,
    )
    monkeypatch.setattr(admin_users_service, "verify_privileged_action", _reauth)
    monkeypatch.setattr(admin_users_service, "_emit_admin_account_audit_event", _audit)

    await admin_users_service.delete_user(
        AuthPrincipal(
            kind="user",
            user_id=actor_id,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
        ),
        target_id,
        SimpleNamespace(reason="Support case 123", admin_password="AdminPass123!"),
        password_service=object(),
        db_pool=pg_repo.pool,
        webhook_event_producer=_producer(pg_repo, ring),
        command_id_factory=lambda: COMMAND_ID,
        source_request_id="postgres-delete-request",
    )

    event = await pg_repo.pool.fetchrow(
        "SELECT * FROM admin_webhook_events WHERE id = ?",
        EVENT_ID,
    )
    delivery = await pg_repo.pool.fetchrow(
        "SELECT * FROM admin_webhook_deliveries WHERE event_id = ?",
        EVENT_ID,
    )
    assert not bool(
        await pg_repo.pool.fetchval(
            "SELECT is_active FROM users WHERE id = ?",
            target_id,
        )
    )
    assert event is not None and event["source_command_id"] == COMMAND_ID
    assert delivery is not None and int(delivery["webhook_id"]) == webhook_id
    assert len(emitted) == 1
