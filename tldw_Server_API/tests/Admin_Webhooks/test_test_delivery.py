from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Protocol

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AttemptState,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryState,
    WebhookError,
    WebhookErrorCode,
    build_registration_etag,
)
from tldw_Server_API.app.core.Admin_Webhooks.executor import (
    DeliveryAttemptExecutor,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    RegistrationInsert,
    RegistrationTarget,
)
from tldw_Server_API.app.core.exceptions import HTTPHopError
from tldw_Server_API.app.core.Security.egress import URLPolicyResult
from tldw_Server_API.app.core.Security.http_hop import StatusOnlyHTTPHopResponse

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)
KEY_ID = "task9-key"
IDEMPOTENCY_KEY = "0123456789abcdef0123456789abcdef"
TARGET_URL = "https://receiver.example.test/hooks?credential=not-for-metadata"
SIGNING_SECRET = "whsec_" + "1" * 64


def canonical_uuid4(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (
        f"{digest[:8]}-{digest[8:12]}-4{digest[13:16]}-"
        f"8{digest[17:20]}-{digest[20:32]}"
    )


def opaque_token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def key_ring() -> WebhookKeyRing:
    encoded = base64.b64encode(b"t" * 32).decode("ascii")
    return WebhookKeyRing({KEY_ID: encoded}, primary_id=KEY_ID)


def settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.ON,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


class TestRepositoryFixture(Protocol):
    repository: AdminWebhookRepository

    async def execute(self, query: str, *params: object) -> None: ...

    async def fetchval(self, query: str, *params: object) -> object: ...


class EgressRecorder:
    def __init__(self, status_code: int = 204) -> None:
        self.status_code = status_code
        self.requests: list[object] = []

    async def __call__(self, request: object) -> StatusOnlyHTTPHopResponse:
        self.requests.append(request)
        return StatusOnlyHTTPHopResponse(
            status_code=self.status_code,
            latency_ms=1,
            retry_after_seconds=None,
        )


async def seed_ready_registration(
    fixture: TestRepositoryFixture,
    *,
    active: bool = False,
) -> tuple[object, WebhookKeyRing]:
    ring = key_ring()
    migration_at = NOW - timedelta(hours=2)
    digest = "sha256:" + ("a" * 64)
    fingerprint = "hmac-sha256:" + ("b" * 64)
    async with fixture.repository.transaction() as tx:
        migration = await tx.lock_migration_state()
        await tx.compare_and_set_migration_state(
            expected_revision=migration.state_revision,
            updates={
                "phase": "complete",
                "import_operation_id": "whmig_" + ("c" * 32),
                "import_operator_id": 7,
                "import_started_at": migration_at,
                "import_approved_at": migration_at,
                "artifacts_ready_at": migration_at,
                "database_committed_at": migration_at,
                "fingerprint_key_id": KEY_ID,
                "completed_at": migration_at,
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
                "rollback_expires_at": migration_at + timedelta(days=7),
                "rollback_retirement_phase": "retained",
                "expected_ciphertext_digest": digest,
            },
            at=migration_at,
        )
        webhook_id = await tx.allocate_registration_id()
        target = ring.encrypt_text(
            purpose="registration.target",
            identity={"registration_id": webhook_id, "target_version": 1},
            plaintext=TARGET_URL,
        )
        secret = ring.encrypt_text(
            purpose="registration.secret",
            identity={"registration_id": webhook_id, "secret_version": 1},
            plaintext=SIGNING_SECRET,
        )
        registration = await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description="Task 9 test receiver",
                target=RegistrationTarget(
                    protected=target,
                    hostname="receiver.example.test",
                    display="https://receiver.example.test",
                ),
                event_types=("user.created",),
                active=active,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=7,
                now=NOW - timedelta(hours=1),
            )
        )
    return registration, ring


def test_internal_test_contracts_are_frozen_bounded_and_repr_safe() -> None:
    module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    package = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks")
    command = module.TestWebhookCommand(
        actor_id=7,
        webhook_id=3,
        if_match='"admin-webhook-3-r4"',
        delivery_config_version=5,
        idempotency_key=IDEMPOTENCY_KEY,
        request_id="task9-request",
    )

    for record in (
        module.TestWebhookCommand,
        module.TestWebhookResult,
        module.TestWebhookAudit,
    ):
        assert is_dataclass(record)
        assert record.__dataclass_params__.frozen
        assert record.__name__ not in package.__all__
        assert not hasattr(package, record.__name__)
    assert {field.name for field in fields(command)} == {
        "actor_id",
        "webhook_id",
        "if_match",
        "delivery_config_version",
        "idempotency_key",
        "request_id",
    }
    rendered = repr(command)
    assert IDEMPOTENCY_KEY not in rendered
    assert command.if_match not in rendered
    assert "target" not in {field.name for field in fields(command)}
    assert "secret" not in {field.name for field in fields(command)}


async def exercise_test_service_success_and_terminal_replay(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    registration, ring = await seed_ready_registration(fixture, active=False)
    policy_calls: list[str] = []

    def allow_target(url: str) -> URLPolicyResult:
        policy_calls.append(url)
        return URLPolicyResult(True, None, ("203.0.113.10",))

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        allow_target,
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        allow_target,
    )
    egress = EgressRecorder()
    executor = DeliveryAttemptExecutor(egress=egress)
    metric_observations: list[dict[str, object]] = []

    class Metrics:
        def attempt_committed(self, **values: object) -> None:
            metric_observations.append(values)

    ids = iter(
        (
            canonical_uuid4("task9-event"),
            canonical_uuid4("task9-delivery"),
            canonical_uuid4("task9-attempt"),
        )
    )
    service = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: next(ids),
        delivery_id_factory=lambda: next(ids),
        clock=lambda: NOW,
        settings=settings(),
        executor=executor,
        test_attempt_id_factory=lambda: next(ids),
        test_token_factory=lambda: opaque_token("task9-token"),
        metrics=Metrics(),
    )
    command = delivery_module.TestWebhookCommand(
        actor_id=7,
        webhook_id=registration.id,
        if_match=build_registration_etag(
            webhook_id=registration.id,
            revision=registration.revision,
        ),
        delivery_config_version=registration.delivery_config_version,
        idempotency_key=IDEMPOTENCY_KEY,
        request_id="task9-success-request",
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)

    result = await service.test_webhook(command, audit_sink=audit_sink)

    assert result.delivery.kind is DeliveryKind.TEST
    assert result.delivery.state is DeliveryState.SUCCEEDED
    assert result.delivery.attempt_count == 1
    assert result.attempt.state is AttemptState.SUCCEEDED
    assert result.attempt.attempt_number == 1
    assert result.idempotent_replay is False
    assert result.in_progress is False
    assert result.retry_after_seconds is None
    assert len(egress.requests) == 1
    headers = dict(egress.requests[0].headers)  # type: ignore[attr-defined]
    assert headers["x-tldw-webhook-test"] == "true"
    assert headers["x-tldw-webhook-delivery-id"] == result.delivery.id
    assert [audit.outcome for audit in audits] == ["accepted", "succeeded"]
    assert len(policy_calls) == 2

    event_body = await fixture.fetchval(
        "SELECT body_ciphertext_json FROM admin_webhook_events WHERE id = ?",
        result.delivery.event_id,
    )
    assert event_body is not None
    bundle = await fixture.repository.get_delivery_bundle(result.delivery.id)
    assert bundle is not None
    plaintext = ring.decrypt_event_body(
        event_id=bundle.event.id,
        api_version=bundle.event.event.api_version,
        protected=bundle.event.body,
    )
    decoded = json.loads(plaintext)
    assert decoded["type"] == "webhook.test"
    assert decoded["data"] == {"test": True, "webhook_id": registration.id}
    assert bundle.event.source_component == "admin_webhooks.test"
    assert bundle.event.source_request_id == command.request_id
    assert bundle.delivery.jobs_job_id is None
    assert bundle.delivery.pending_jobs_disposition is None

    unavailable = WebhookKeyRingLoadResult(
        ring=None,
        code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
    )
    replay_service = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=unavailable,
        event_id_factory=lambda: canonical_uuid4("unused-replay-event"),
        delivery_id_factory=lambda: canonical_uuid4("unused-replay-delivery"),
        clock=lambda: NOW + timedelta(seconds=1),
        settings=settings(),
        executor=DeliveryAttemptExecutor(egress=EgressRecorder(500)),
        test_attempt_id_factory=lambda: canonical_uuid4("unused-replay-attempt"),
        test_token_factory=lambda: opaque_token("unused-replay-token"),
        metrics=Metrics(),
    )
    replay = await replay_service.test_webhook(command, audit_sink=audit_sink)
    assert replay.delivery == result.delivery
    assert replay.attempt == result.attempt
    assert replay.idempotent_replay is True
    assert len(egress.requests) == 1
    assert len(audits) == 2
    assert len(policy_calls) == 2
    assert metric_observations == [
        {
            "state": DeliveryState.SUCCEEDED,
            "kind": DeliveryKind.TEST,
            "reason_code": None,
            "delivery_reason_code": None,
            "status_code": 204,
            "latency_ms": result.attempt.latency_ms,
        }
    ]


async def exercise_processing_replay_and_conflict_precede_current_state(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    registration, ring = await seed_ready_registration(fixture)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )

    class BlockingEgress(EgressRecorder):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def __call__(self, request: object) -> StatusOnlyHTTPHopResponse:
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            return StatusOnlyHTTPHopResponse(204, 1, None)

    egress = BlockingEgress()
    ids = iter(
        (
            canonical_uuid4("processing-event"),
            canonical_uuid4("processing-delivery"),
            canonical_uuid4("processing-attempt"),
        )
    )
    service = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: next(ids),
        delivery_id_factory=lambda: next(ids),
        clock=lambda: NOW,
        settings=settings(),
        executor=DeliveryAttemptExecutor(egress=egress),
        test_attempt_id_factory=lambda: next(ids),
        test_token_factory=lambda: opaque_token("processing-token"),
    )
    command = delivery_module.TestWebhookCommand(
        actor_id=7,
        webhook_id=registration.id,
        if_match=build_registration_etag(
            webhook_id=registration.id,
            revision=registration.revision,
        ),
        delivery_config_version=registration.delivery_config_version,
        idempotency_key=IDEMPOTENCY_KEY,
        request_id="processing-request",
    )

    async def audit_sink(_record: object) -> None:
        return None

    owner = asyncio.create_task(service.test_webhook(command, audit_sink=audit_sink))
    await egress.started.wait()
    replay = await service.test_webhook(command, audit_sink=audit_sink)
    assert replay.in_progress is True
    assert replay.idempotent_replay is True
    assert replay.retry_after_seconds == 5
    assert replay.delivery.state is DeliveryState.PROCESSING
    assert len(egress.requests) == 1

    unavailable = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        event_id_factory=lambda: canonical_uuid4("unused-conflict-event"),
        delivery_id_factory=lambda: canonical_uuid4("unused-conflict-delivery"),
        clock=lambda: NOW,
        settings=settings(),
        executor=DeliveryAttemptExecutor(egress=EgressRecorder()),
        test_attempt_id_factory=lambda: canonical_uuid4("unused-conflict-attempt"),
        test_token_factory=lambda: opaque_token("unused-conflict-token"),
    )
    conflicting = delivery_module.TestWebhookCommand(
        actor_id=command.actor_id,
        webhook_id=command.webhook_id,
        if_match=command.if_match,
        delivery_config_version=command.delivery_config_version + 1,
        idempotency_key=command.idempotency_key,
        request_id=command.request_id,
    )
    with pytest.raises(WebhookError) as conflict:
        await unavailable.test_webhook(conflicting, audit_sink=audit_sink)
    assert conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT

    egress.release.set()
    completed = await owner
    assert completed.delivery.state is DeliveryState.SUCCEEDED
    assert len(egress.requests) == 1


async def exercise_retry_class_terminalization_and_completion_audit_failure(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    registration, ring = await seed_ready_registration(fixture)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )

    class MutatingEgress(EgressRecorder):
        async def __call__(self, request: object) -> StatusOnlyHTTPHopResponse:
            self.requests.append(request)
            await fixture.execute(
                """
                UPDATE admin_webhook_registrations
                SET revision = revision + 1,
                    delivery_config_version = delivery_config_version + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                NOW + timedelta(seconds=1),
                registration.id,
            )
            return StatusOnlyHTTPHopResponse(
                status_code=503,
                latency_ms=1,
                retry_after_seconds=1_800,
            )

    egress = MutatingEgress()
    ids = iter(
        (
            canonical_uuid4("retry-event"),
            canonical_uuid4("retry-delivery"),
            canonical_uuid4("retry-attempt"),
        )
    )
    service = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: next(ids),
        delivery_id_factory=lambda: next(ids),
        clock=iter((NOW, NOW + timedelta(seconds=2))).__next__,
        settings=settings(),
        executor=DeliveryAttemptExecutor(egress=egress),
        test_attempt_id_factory=lambda: next(ids),
        test_token_factory=lambda: opaque_token("retry-token"),
    )
    command = delivery_module.TestWebhookCommand(
        actor_id=7,
        webhook_id=registration.id,
        if_match=build_registration_etag(
            webhook_id=registration.id,
            revision=registration.revision,
        ),
        delivery_config_version=registration.delivery_config_version,
        idempotency_key=IDEMPOTENCY_KEY,
        request_id="retry-request",
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)
        if len(audits) == 2:
            durable = await fixture.repository.get_delivery_bundle(
                record.delivery_id  # type: ignore[attr-defined]
            )
            assert durable is not None
            assert durable.delivery.delivery.state is DeliveryState.DEAD
            raise RuntimeError("completion audit canary must not escape")

    result = await service.test_webhook(command, audit_sink=audit_sink)
    assert result.delivery.state is DeliveryState.DEAD
    assert result.delivery.reason_code.value == "http_server_error"
    assert result.attempt.state is AttemptState.FAILED
    assert result.attempt.reason_code.value == "http_server_error"
    assert result.attempt.requested_retry_delay_seconds is None
    assert result.retry_after_seconds is None
    assert len(egress.requests) == 1
    bundle = await fixture.repository.get_delivery_bundle(result.delivery.id)
    assert bundle is not None
    assert bundle.delivery.completed_after_config_change is True
    assert bundle.delivery.pending_jobs_disposition is None
    assert bundle.delivery.jobs_job_id is None
    assert [record.outcome for record in audits] == ["accepted", "failed"]
    assert TARGET_URL not in repr(audits)
    assert SIGNING_SECRET not in repr(audits)
    assert IDEMPOTENCY_KEY not in repr(audits)


async def exercise_post_start_semantic_and_rekey_races(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )

    async def mutate_config(registration, _ring: WebhookKeyRing) -> None:
        await fixture.execute(
            """
            UPDATE admin_webhook_registrations
            SET revision = revision + 1,
                delivery_config_version = delivery_config_version + 1,
                updated_at = ?
            WHERE id = ?
            """,
            NOW + timedelta(seconds=1),
            registration.id,
        )

    async def rotate_secret(registration, ring: WebhookKeyRing) -> None:
        next_version = registration.secret_version + 1
        secret = ring.encrypt_text(
            purpose="registration.secret",
            identity={
                "registration_id": registration.id,
                "secret_version": next_version,
            },
            plaintext="whsec_" + "2" * 64,
        )
        await fixture.execute(
            """
            UPDATE admin_webhook_registrations
            SET revision = revision + 1, secret_version = ?,
                secret_ciphertext_json = ?, secret_key_id = ?, updated_at = ?
            WHERE id = ?
            """,
            next_version,
            secret.ciphertext_json,
            secret.key_id,
            NOW + timedelta(seconds=1),
            registration.id,
        )

    async def delete_registration(registration, _ring: WebhookKeyRing) -> None:
        async with fixture.repository.transaction() as tx:
            await tx.soft_delete_registration(
                registration.id,
                expected_revision=registration.revision,
                actor_user_id=7,
                at=NOW + timedelta(seconds=1),
            )

    mutations = (
        ("config", mutate_config),
        ("secret", rotate_secret),
        ("delete", delete_registration),
    )
    outcomes = (
        ("success", 204, None, DeliveryState.SUCCEEDED, None),
        (
            "http-retry",
            503,
            None,
            DeliveryState.DEAD,
            DeliveryReasonCode.HTTP_SERVER_ERROR,
        ),
        (
            "network-retry",
            None,
            HTTPHopError("read_timeout", retryable=True),
            DeliveryState.DEAD,
            DeliveryReasonCode.HTTP_HOP_READ_TIMEOUT,
        ),
    )

    class MutatingEgress(EgressRecorder):
        def __init__(
            self,
            *,
            mutation: Callable[[object, WebhookKeyRing], Awaitable[None]],
            registration: object,
            ring: WebhookKeyRing,
            status_code: int | None,
            hop_error: HTTPHopError | None,
        ) -> None:
            super().__init__()
            self._mutation = mutation
            self._registration = registration
            self._ring = ring
            self._status_code = status_code
            self._hop_error = hop_error

        async def __call__(self, request: object) -> StatusOnlyHTTPHopResponse:
            self.requests.append(request)
            await self._mutation(self._registration, self._ring)
            if self._hop_error is not None:
                raise self._hop_error
            if self._status_code is None:
                raise AssertionError("status is required without a hop error")
            return StatusOnlyHTTPHopResponse(
                status_code=self._status_code,
                latency_ms=1,
                retry_after_seconds=(
                    1_800 if self._status_code == 503 else None
                ),
            )

    for mutation_label, mutation in mutations:
        for outcome_label, status_code, hop_error, state, reason in outcomes:
            label = f"post-start-{mutation_label}-{outcome_label}"
            registration, ring = await seed_ready_registration(fixture)

            ids = iter(
                (
                    canonical_uuid4(f"{label}-event"),
                    canonical_uuid4(f"{label}-delivery"),
                    canonical_uuid4(f"{label}-attempt"),
                )
            )
            egress = MutatingEgress(
                mutation=mutation,
                registration=registration,
                ring=ring,
                status_code=status_code,
                hop_error=hop_error,
            )
            service = delivery_module.AdminWebhookDeliveryService(
                repository=fixture.repository,
                key_ring_result=WebhookKeyRingLoadResult(
                    ring=ring,
                    code=WebhookKeyLoadCode.AVAILABLE,
                ),
                event_id_factory=ids.__next__,
                delivery_id_factory=ids.__next__,
                clock=iter((NOW, NOW + timedelta(seconds=2))).__next__,
                settings=settings(),
                executor=DeliveryAttemptExecutor(egress=egress),
                test_attempt_id_factory=ids.__next__,
                test_token_factory=iter((opaque_token(f"{label}-token"),)).__next__,
            )
            command = delivery_module.TestWebhookCommand(
                actor_id=7,
                webhook_id=registration.id,
                if_match=build_registration_etag(
                    webhook_id=registration.id,
                    revision=registration.revision,
                ),
                delivery_config_version=registration.delivery_config_version,
                idempotency_key=hashlib.sha256(label.encode()).hexdigest()[:32],
                request_id=f"{label}-request",
            )

            result = await service.test_webhook(
                command,
                audit_sink=lambda _record: asyncio.sleep(0),
            )

            assert result.delivery.state is state, label
            assert result.delivery.reason_code is reason, label
            assert result.attempt.state is (
                AttemptState.SUCCEEDED
                if state is DeliveryState.SUCCEEDED
                else AttemptState.FAILED
            ), label
            assert result.attempt.reason_code is reason, label
            assert result.delivery.attempt_count == 1, label
            assert result.attempt.requested_retry_delay_seconds is None, label
            assert len(egress.requests) == 1, label
            bundle = await fixture.repository.get_delivery_bundle(result.delivery.id)
            assert bundle is not None
            assert bundle.delivery.completed_after_config_change is True, label
            assert bundle.delivery.jobs_job_id is None, label
            assert bundle.delivery.pending_jobs_disposition is None, label
            assert (
                await fixture.fetchval(
                    "SELECT COUNT(*) FROM admin_webhook_delivery_attempts WHERE delivery_id = ?",
                    result.delivery.id,
                )
                == 1
            ), label

    registration, ring = await seed_ready_registration(fixture)
    encoded_old = base64.b64encode(b"t" * 32).decode("ascii")
    encoded_new = base64.b64encode(b"r" * 32).decode("ascii")
    rekey_id = "task9-rekey"
    rekey_ring = WebhookKeyRing(
        {KEY_ID: encoded_old, rekey_id: encoded_new},
        primary_id=rekey_id,
    )
    protected_before = await fixture.repository.get_protected_registration(
        registration.id,
        include_deleted=False,
    )
    assert protected_before is not None

    class RekeyingEgress(EgressRecorder):
        async def __call__(self, request: object) -> StatusOnlyHTTPHopResponse:
            self.requests.append(request)
            target = rekey_ring.encrypt_text(
                purpose="registration.target",
                identity={
                    "registration_id": registration.id,
                    "target_version": registration.target_version,
                },
                plaintext=TARGET_URL,
            )
            secret = rekey_ring.encrypt_text(
                purpose="registration.secret",
                identity={
                    "registration_id": registration.id,
                    "secret_version": registration.secret_version,
                },
                plaintext=SIGNING_SECRET,
            )
            await fixture.execute(
                """
                UPDATE admin_webhook_registrations
                SET target_ciphertext_json = ?, target_key_id = ?,
                    secret_ciphertext_json = ?, secret_key_id = ?
                WHERE id = ?
                """,
                target.ciphertext_json,
                target.key_id,
                secret.ciphertext_json,
                secret.key_id,
                registration.id,
            )
            return StatusOnlyHTTPHopResponse(204, 1, None)

    ids = iter(
        (
            canonical_uuid4("post-start-rekey-event"),
            canonical_uuid4("post-start-rekey-delivery"),
            canonical_uuid4("post-start-rekey-attempt"),
        )
    )
    rekey_egress = RekeyingEgress()
    rekey_service = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: next(ids),
        delivery_id_factory=lambda: next(ids),
        clock=iter((NOW, NOW + timedelta(seconds=2))).__next__,
        settings=settings(),
        executor=DeliveryAttemptExecutor(egress=rekey_egress),
        test_attempt_id_factory=lambda: next(ids),
        test_token_factory=lambda: opaque_token("post-start-rekey-token"),
    )
    rekey_result = await rekey_service.test_webhook(
        delivery_module.TestWebhookCommand(
            actor_id=7,
            webhook_id=registration.id,
            if_match=build_registration_etag(
                webhook_id=registration.id,
                revision=registration.revision,
            ),
            delivery_config_version=registration.delivery_config_version,
            idempotency_key="8899aabbccddeeff0011223344556677",
            request_id="post-start-rekey-request",
        ),
        audit_sink=lambda _record: asyncio.sleep(0),
    )
    protected_after = await fixture.repository.get_protected_registration(
        registration.id,
        include_deleted=False,
    )
    assert protected_after is not None
    assert protected_after.registration.revision == registration.revision
    assert (
        protected_after.registration.delivery_config_version
        == registration.delivery_config_version
    )
    assert protected_after.registration.target_version == registration.target_version
    assert protected_after.registration.secret_version == registration.secret_version
    assert protected_after.target != protected_before.target
    assert protected_after.secret != protected_before.secret
    assert (
        rekey_ring.decrypt_text(
            purpose="registration.target",
            identity={
                "registration_id": registration.id,
                "target_version": registration.target_version,
            },
            protected=protected_after.target,
        )
        == TARGET_URL
    )
    assert (
        rekey_ring.decrypt_text(
            purpose="registration.secret",
            identity={
                "registration_id": registration.id,
                "secret_version": registration.secret_version,
            },
            protected=protected_after.secret,
        )
        == SIGNING_SECRET
    )
    rekey_bundle = await fixture.repository.get_delivery_bundle(
        rekey_result.delivery.id
    )
    assert rekey_bundle is not None
    assert rekey_result.delivery.state is DeliveryState.SUCCEEDED
    assert rekey_bundle.delivery.completed_after_config_change is False
    assert len(rekey_egress.requests) == 1


async def exercise_start_races_and_accepted_audit_rollback(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    registration, ring = await seed_ready_registration(fixture)
    protected = await fixture.repository.get_protected_registration(
        registration.id,
        include_deleted=False,
    )
    assert protected is not None
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )

    class RaceRepository:
        def __init__(self, hook) -> None:
            self._hook = hook
            self._used = False

        def __getattr__(self, name: str):
            return getattr(fixture.repository, name)

        @asynccontextmanager
        async def transaction(self):
            if not self._used:
                self._used = True
                await self._hook()
            async with fixture.repository.transaction() as tx:
                yield tx

    def service(repository: object, label: str, egress: EgressRecorder):
        ids = iter(
            (
                canonical_uuid4(f"{label}-event"),
                canonical_uuid4(f"{label}-delivery"),
                canonical_uuid4(f"{label}-attempt"),
            )
        )
        return delivery_module.AdminWebhookDeliveryService(
            repository=repository,
            key_ring_result=WebhookKeyRingLoadResult(
                ring=ring,
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            event_id_factory=lambda: next(ids),
            delivery_id_factory=lambda: next(ids),
            clock=lambda: NOW,
            settings=settings(),
            executor=DeliveryAttemptExecutor(egress=egress),
            test_attempt_id_factory=lambda: next(ids),
            test_token_factory=lambda: opaque_token(f"{label}-token"),
        )

    async def audit_sink(_record: object) -> None:
        return None

    async def run_race(label: str, hook) -> WebhookErrorCode:
        current = await fixture.repository.get_registration(registration.id)
        assert current is not None
        egress = EgressRecorder()
        command = delivery_module.TestWebhookCommand(
            actor_id=7,
            webhook_id=current.id,
            if_match=build_registration_etag(
                webhook_id=current.id,
                revision=current.revision,
            ),
            delivery_config_version=current.delivery_config_version,
            idempotency_key=(
                hashlib.sha256(label.encode()).hexdigest()[:32]
            ),
            request_id=f"{label}-request",
        )
        with pytest.raises(WebhookError) as failure:
            await service(RaceRepository(hook), label, egress).test_webhook(
                command,
                audit_sink=audit_sink,
            )
        assert egress.requests == []
        assert (
            await fixture.fetchval(
                "SELECT COUNT(*) FROM admin_webhook_events WHERE source_request_id = ?",
                command.request_id,
            )
            == 0
        )
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'test'"
        ) == 0
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'test'"
        ) == 0
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_delivery_attempts"
        ) == 0
        return failure.value.code

    async def restore_protected_snapshot() -> None:
        await fixture.execute(
            """
            UPDATE admin_webhook_registrations
            SET target_version = ?, target_ciphertext_json = ?, target_key_id = ?,
                secret_version = ?, secret_ciphertext_json = ?, secret_key_id = ?
            WHERE id = ?
            """,
            protected.registration.target_version,
            protected.target.ciphertext_json,
            protected.target.key_id,
            protected.registration.secret_version,
            protected.secret.ciphertext_json,
            protected.secret.key_id,
            registration.id,
        )

    async def mutate_config() -> None:
        await fixture.execute(
            """
            UPDATE admin_webhook_registrations
            SET revision = revision + 1,
                delivery_config_version = delivery_config_version + 1,
                updated_at = ?
            WHERE id = ?
            """,
            NOW,
            registration.id,
        )

    assert await run_race("config-race", mutate_config) is WebhookErrorCode.PRECONDITION_FAILED
    await fixture.execute(
        """
        UPDATE admin_webhook_registrations
        SET revision = ?, delivery_config_version = ?, updated_at = ?
        WHERE id = ?
        """,
        registration.revision,
        registration.delivery_config_version,
        registration.updated_at,
        registration.id,
    )

    async def require_rotation() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET secret_rotation_required = ? WHERE id = ?",
            True,
            registration.id,
        )

    assert await run_race("rotation-race", require_rotation) is WebhookErrorCode.PRECONDITION_FAILED
    await fixture.execute(
        "UPDATE admin_webhook_registrations SET secret_rotation_required = ? WHERE id = ?",
        False,
        registration.id,
    )

    async def delete_registration() -> None:
        async with fixture.repository.transaction() as tx:
            await tx.soft_delete_registration(
                registration.id,
                expected_revision=registration.revision,
                actor_user_id=7,
                at=NOW,
            )

    assert await run_race("delete-race", delete_registration) is WebhookErrorCode.PRECONDITION_FAILED
    await fixture.execute(
        """
        UPDATE admin_webhook_registrations
        SET active = ?, deleted_at = NULL, deleted_by_user_id = NULL,
            revision = ?, delivery_config_version = ?,
            updated_by_user_id = ?, updated_at = ?
        WHERE id = ?
        """,
        registration.active,
        registration.revision,
        registration.delivery_config_version,
        registration.updated_by_user_id,
        registration.updated_at,
        registration.id,
    )

    async def begin_key_rotation() -> None:
        await fixture.execute(
            """
            UPDATE admin_webhook_migration_state
            SET rotation_phase = ?, rotation_operation_id = ?,
                rotation_source_key_id = ?, rotation_target_key_id = ?,
                rotation_started_at = ?
            WHERE singleton_id = 1
            """,
            "rewriting",
            "rotation-task9",
            KEY_ID,
            "next-key",
            NOW,
        )

    assert (
        await run_race("key-race", begin_key_rotation)
        is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS
    )
    await fixture.execute(
        """
        UPDATE admin_webhook_migration_state
        SET rotation_phase = NULL, rotation_operation_id = NULL,
            rotation_source_key_id = NULL, rotation_target_key_id = NULL,
            rotation_started_at = NULL
        WHERE singleton_id = 1
        """
    )

    async def change_target_version() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET target_version = target_version + 1 WHERE id = ?",
            registration.id,
        )

    assert (
        await run_race("target-version-race", change_target_version)
        is WebhookErrorCode.PRECONDITION_FAILED
    )
    await restore_protected_snapshot()

    async def change_secret_version() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET secret_version = secret_version + 1 WHERE id = ?",
            registration.id,
        )

    assert (
        await run_race("secret-version-race", change_secret_version)
        is WebhookErrorCode.PRECONDITION_FAILED
    )
    await restore_protected_snapshot()

    reencrypted_target = ring.encrypt_text(
        purpose="registration.target",
        identity={
            "registration_id": registration.id,
            "target_version": registration.target_version,
        },
        plaintext=TARGET_URL,
    )

    async def change_target_ciphertext() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET target_ciphertext_json = ? WHERE id = ?",
            reencrypted_target.ciphertext_json,
            registration.id,
        )

    assert (
        await run_race("target-ciphertext-race", change_target_ciphertext)
        is WebhookErrorCode.PRECONDITION_FAILED
    )
    await restore_protected_snapshot()

    async def change_target_key_id() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET target_key_id = ? WHERE id = ?",
            "task9-other-key",
            registration.id,
        )

    assert (
        await run_race("target-key-race", change_target_key_id)
        is WebhookErrorCode.PRECONDITION_FAILED
    )
    await restore_protected_snapshot()

    reencrypted_secret = ring.encrypt_text(
        purpose="registration.secret",
        identity={
            "registration_id": registration.id,
            "secret_version": registration.secret_version,
        },
        plaintext=SIGNING_SECRET,
    )

    async def change_secret_ciphertext() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET secret_ciphertext_json = ? WHERE id = ?",
            reencrypted_secret.ciphertext_json,
            registration.id,
        )

    assert (
        await run_race("secret-ciphertext-race", change_secret_ciphertext)
        is WebhookErrorCode.PRECONDITION_FAILED
    )
    await restore_protected_snapshot()

    async def change_secret_key_id() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_registrations SET secret_key_id = ? WHERE id = ?",
            "task9-other-key",
            registration.id,
        )

    assert (
        await run_race("secret-key-race", change_secret_key_id)
        is WebhookErrorCode.PRECONDITION_FAILED
    )
    await restore_protected_snapshot()

    async def change_active_primary() -> None:
        await fixture.execute(
            "UPDATE admin_webhook_migration_state SET active_primary_key_id = ? WHERE singleton_id = 1",
            "task9-other-key",
        )

    assert (
        await run_race("active-primary-race", change_active_primary)
        is WebhookErrorCode.KEY_CONFIGURATION_MISMATCH
    )
    await fixture.execute(
        "UPDATE admin_webhook_migration_state SET active_primary_key_id = ? WHERE singleton_id = 1",
        KEY_ID,
    )

    current = await fixture.repository.get_registration(registration.id)
    assert current is not None
    rollback_command = delivery_module.TestWebhookCommand(
        actor_id=7,
        webhook_id=current.id,
        if_match=build_registration_etag(
            webhook_id=current.id,
            revision=current.revision,
        ),
        delivery_config_version=current.delivery_config_version,
        idempotency_key="abcdef0123456789abcdef0123456789",
        request_id="accepted-audit-rollback",
    )
    egress = EgressRecorder()

    async def unavailable_audit(_record: object) -> None:
        raise RuntimeError("mandatory audit unavailable")

    with pytest.raises(WebhookError) as unavailable:
        await service(fixture.repository, "audit-rollback", egress).test_webhook(
            rollback_command,
            audit_sink=unavailable_audit,
        )
    assert unavailable.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert egress.requests == []
    assert (
        await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_events WHERE source_request_id = ?",
            rollback_command.request_id,
        )
        == 0
    )


async def exercise_commit_failure_correlated_audit(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    registration, ring = await seed_ready_registration(fixture)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )

    class CommitFailRepository:
        def __getattr__(self, name: str):
            return getattr(fixture.repository, name)

        @asynccontextmanager
        async def transaction(self):
            async with fixture.repository.transaction() as tx:
                yield tx
                raise TransactionError("simulated test-start commit failure")

    async def run_case(label: str, *, failed_audit_fails: bool) -> None:
        ids = iter(
            (
                canonical_uuid4(f"{label}-event"),
                canonical_uuid4(f"{label}-delivery"),
                canonical_uuid4(f"{label}-attempt"),
            )
        )
        egress = EgressRecorder()
        service = delivery_module.AdminWebhookDeliveryService(
            repository=CommitFailRepository(),
            key_ring_result=WebhookKeyRingLoadResult(
                ring=ring,
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            event_id_factory=lambda: next(ids),
            delivery_id_factory=lambda: next(ids),
            clock=lambda: NOW,
            settings=settings(),
            executor=DeliveryAttemptExecutor(egress=egress),
            test_attempt_id_factory=lambda: next(ids),
            test_token_factory=lambda: opaque_token(f"{label}-token"),
        )
        command = delivery_module.TestWebhookCommand(
            actor_id=7,
            webhook_id=registration.id,
            if_match=build_registration_etag(
                webhook_id=registration.id,
                revision=registration.revision,
            ),
            delivery_config_version=registration.delivery_config_version,
            idempotency_key=hashlib.sha256(label.encode()).hexdigest()[:32],
            request_id=f"{label}-request",
        )
        audits: list[object] = []

        async def audit_sink(record: object) -> None:
            audits.append(record)
            if failed_audit_fails and record.outcome == "failed":  # type: ignore[attr-defined]
                raise RuntimeError("follow-up audit unavailable")

        with pytest.raises(WebhookError) as failure:
            await service.test_webhook(command, audit_sink=audit_sink)
        assert failure.value.code is WebhookErrorCode.OPERATION_FAILED
        assert [record.outcome for record in audits] == [  # type: ignore[attr-defined]
            "accepted",
            "failed",
        ]
        accepted, failed = audits
        for field_name in (
            "actor_id",
            "webhook_id",
            "delivery_id",
            "attempt_id",
            "request_id",
        ):
            assert getattr(accepted, field_name) == getattr(failed, field_name)
        assert failed.reason_code is WebhookErrorCode.OPERATION_FAILED  # type: ignore[attr-defined]
        assert egress.requests == []
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'test'"
        ) == 0
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_events WHERE source_request_id = ?",
            command.request_id,
        ) == 0
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'test'"
        ) == 0
        assert await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_delivery_attempts"
        ) == 0

    await run_case("commit-failure-audit-succeeds", failed_audit_fails=False)
    await run_case("commit-failure-audit-fails", failed_audit_fails=True)


async def exercise_concurrent_exact_test_start(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    executor_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )
    registration, ring = await seed_ready_registration(fixture)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, None, ("203.0.113.10",)),
    )

    class SynchronizedLookupRepository:
        def __init__(self) -> None:
            self.arrivals = 0
            self.ready = asyncio.Event()
            self.release = asyncio.Event()

        def __getattr__(self, name: str):
            return getattr(fixture.repository, name)

        async def lookup_idempotency(self, **kwargs):
            lookup = await fixture.repository.lookup_idempotency(**kwargs)
            self.arrivals += 1
            if self.arrivals == 2:
                self.ready.set()
            await self.release.wait()
            return lookup

    synchronized = SynchronizedLookupRepository()
    egress = EgressRecorder()

    def service(label: str):
        ids = iter(
            (
                canonical_uuid4(f"concurrent-{label}-event"),
                canonical_uuid4(f"concurrent-{label}-delivery"),
                canonical_uuid4(f"concurrent-{label}-attempt"),
            )
        )
        return delivery_module.AdminWebhookDeliveryService(
            repository=synchronized,
            key_ring_result=WebhookKeyRingLoadResult(
                ring=ring,
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            event_id_factory=lambda: next(ids),
            delivery_id_factory=lambda: next(ids),
            clock=lambda: NOW,
            settings=settings(),
            executor=DeliveryAttemptExecutor(egress=egress),
            test_attempt_id_factory=lambda: next(ids),
            test_token_factory=lambda: opaque_token(f"concurrent-{label}-token"),
        )

    command = delivery_module.TestWebhookCommand(
        actor_id=7,
        webhook_id=registration.id,
        if_match=build_registration_etag(
            webhook_id=registration.id,
            revision=registration.revision,
        ),
        delivery_config_version=registration.delivery_config_version,
        idempotency_key="fedcba9876543210fedcba9876543210",
        request_id="concurrent-exact-request",
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)

    first = asyncio.create_task(service("first").test_webhook(command, audit_sink=audit_sink))
    second = asyncio.create_task(service("second").test_webhook(command, audit_sink=audit_sink))
    await synchronized.ready.wait()
    synchronized.release.set()
    results = await asyncio.gather(first, second)

    assert len(egress.requests) == 1
    assert sorted(result.idempotent_replay for result in results) == [False, True]
    assert results[0].delivery.id == results[1].delivery.id
    assert results[0].attempt.id == results[1].attempt.id
    assert sum(audit.outcome == "accepted" for audit in audits) == 1
    assert (
        await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_events WHERE source_request_id = ?",
            command.request_id,
        )
        == 1
    )
    assert (
        await fixture.fetchval(
            """
            SELECT COUNT(*)
            FROM admin_webhook_delivery_attempts AS attempt
            JOIN admin_webhook_deliveries AS delivery
              ON delivery.id = attempt.delivery_id
            WHERE delivery.webhook_id = ? AND delivery.kind = 'test'
            """,
            registration.id,
        )
        == 1
    )


async def exercise_prestart_rejections_are_no_io(
    fixture: TestRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    repository_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
    registration, ring = await seed_ready_registration(fixture)
    policy_calls: list[str] = []

    def allow_target(url: str) -> URLPolicyResult:
        policy_calls.append(url)
        return URLPolicyResult(True, None, ("203.0.113.10",))

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        allow_target,
    )
    egress = EgressRecorder()

    def service(
        *,
        repository=fixture.repository,
        key_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
    ):
        return delivery_module.AdminWebhookDeliveryService(
            repository=repository,
            key_ring_result=key_result,
            event_id_factory=lambda: canonical_uuid4("unused-rejection-event"),
            delivery_id_factory=lambda: canonical_uuid4("unused-rejection-delivery"),
            clock=lambda: NOW,
            settings=settings(),
            executor=DeliveryAttemptExecutor(egress=egress),
            test_attempt_id_factory=lambda: canonical_uuid4("unused-rejection-attempt"),
            test_token_factory=lambda: opaque_token("unused-rejection-token"),
        )

    def command(**changes):
        values = {
            "actor_id": 7,
            "webhook_id": registration.id,
            "if_match": build_registration_etag(
                webhook_id=registration.id,
                revision=registration.revision,
            ),
            "delivery_config_version": registration.delivery_config_version,
            "idempotency_key": "00112233445566778899aabbccddeeff",
            "request_id": "prestart-rejection-request",
        }
        values.update(changes)
        return delivery_module.TestWebhookCommand(**values)

    async def audit_sink(_record: object) -> None:
        raise AssertionError("rejected tests must not audit an accepted start")

    missing_id = registration.id + 100_000
    cases = (
        (
            service(),
            command(idempotency_key="too-short"),
            WebhookErrorCode.IDEMPOTENCY_KEY_INVALID,
        ),
        (
            service(),
            command(
                if_match=build_registration_etag(
                    webhook_id=registration.id,
                    revision=registration.revision + 1,
                )
            ),
            WebhookErrorCode.PRECONDITION_FAILED,
        ),
        (
            service(),
            command(
                delivery_config_version=registration.delivery_config_version + 1
            ),
            WebhookErrorCode.PRECONDITION_FAILED,
        ),
        (
            service(),
            command(
                webhook_id=missing_id,
                if_match=build_registration_etag(
                    webhook_id=missing_id,
                    revision=1,
                ),
            ),
            WebhookErrorCode.NOT_FOUND,
        ),
        (
            service(
                key_result=WebhookKeyRingLoadResult(
                    ring=None,
                    code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
                )
            ),
            command(),
            WebhookErrorCode.KEY_UNAVAILABLE,
        ),
    )
    for rejected_service, rejected_command, expected_code in cases:
        with pytest.raises(WebhookError) as failure:
            await rejected_service.test_webhook(
                rejected_command,
                audit_sink=audit_sink,
            )
        assert failure.value.code is expected_code

    class BusyRepository:
        def __getattr__(self, name: str):
            return getattr(fixture.repository, name)

        async def lookup_idempotency(self, **_kwargs):
            raise repository_module.WebhookRepositoryError(
                repository_module.WebhookRepositoryErrorCode.DATABASE_BUSY
            )

    with pytest.raises(WebhookError) as busy:
        await service(repository=BusyRepository()).test_webhook(
            command(),
            audit_sink=audit_sink,
        )
    assert busy.value.code is WebhookErrorCode.DATABASE_BUSY
    assert policy_calls == []
    assert egress.requests == []


@pytest.mark.unit
async def test_stale_test_recovery_pass_is_separate_and_no_jobs() -> None:
    reconciler_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.reconciler"
    )
    repository_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
    candidate = repository_module.StaleTestAttemptCandidate(
        delivery_id=canonical_uuid4("reconciler-test-delivery"),
        attempt_id=canonical_uuid4("reconciler-test-attempt"),
        test_attempt_token=opaque_token("reconciler-test-token"),
        stale_at=NOW,
    )
    metric_observations: list[dict[str, object]] = []
    recovered_snapshot = SimpleNamespace(
        delivery=SimpleNamespace(
            delivery=SimpleNamespace(
                state=DeliveryState.DEAD,
                kind=DeliveryKind.TEST,
                reason_code=DeliveryReasonCode.TEST_ATTEMPT_INTERRUPTED,
                status_code=None,
                latency_ms=None,
            )
        ),
        attempt=SimpleNamespace(
            reason_code=DeliveryReasonCode.OUTCOME_UNKNOWN,
            status_code=None,
            latency_ms=None,
        ),
    )

    class Repository:
        def __init__(self) -> None:
            self.list_call = None
            self.recovery_call = None
            self.recovery_attempts = 0

        async def list_stale_test_attempts(self, *, now, limit):
            self.list_call = (now, limit)
            return (candidate,)

        @asynccontextmanager
        async def transaction(self):
            yield self

        async def recover_stale_test_attempt(
            self,
            delivery_id,
            attempt_id,
            test_attempt_token,
            *,
            now,
        ):
            self.recovery_attempts += 1
            self.recovery_call = (
                delivery_id,
                attempt_id,
                test_attempt_token,
                now,
            )
            return recovered_snapshot if self.recovery_attempts == 1 else None

    repository = Repository()
    reconciler = reconciler_module.AdminWebhookReconciler(
        repository=repository,
        queue=object(),
        token_factory=lambda: opaque_token("unused-jobs-token"),
        clock=lambda: NOW,
        claim_ttl_seconds=30,
        failure_observer=lambda _failure: None,
        metrics=SimpleNamespace(
            attempt_committed=lambda **values: metric_observations.append(values)
        ),
    )

    assert await reconciler.recover_stale_test_attempts_once(limit=7) == 1
    assert await reconciler.recover_stale_test_attempts_once(limit=7) == 0
    assert repository.list_call == (NOW, 7)
    assert repository.recovery_call == (
        candidate.delivery_id,
        candidate.attempt_id,
        candidate.test_attempt_token,
        NOW,
    )
    assert metric_observations == [
        {
            "state": DeliveryState.DEAD,
            "kind": DeliveryKind.TEST,
            "reason_code": DeliveryReasonCode.OUTCOME_UNKNOWN,
            "delivery_reason_code": DeliveryReasonCode.TEST_ATTEMPT_INTERRUPTED,
            "status_code": None,
            "latency_ms": None,
        }
    ]
