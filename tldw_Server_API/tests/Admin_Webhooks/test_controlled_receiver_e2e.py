from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AttemptState,
    DeliveryKind,
    DeliveryState,
    EventSourceKind,
    ValidatedWebhookTarget,
)
from tldw_Server_API.app.core.Admin_Webhooks.executor import (
    AttemptExecutionRequest,
    AttemptOutcome,
    DeliveryAttemptExecutor,
)
from tldw_Server_API.app.core.Admin_Webhooks.producer import (
    AdminWebhookEventProducer,
    build_incident_created_data,
    build_incident_notify_data,
    build_incident_resolved_data,
    build_incident_updated_data,
    build_user_created_data,
    build_user_deleted_data,
)
from tldw_Server_API.app.core.Admin_Webhooks.reconciler import JobsDeliveryQueue
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    RegistrationInsert,
    RegistrationTarget,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    PreparedDispositionKind,
)
from tldw_Server_API.app.core.Security.egress import URLPolicyResult
from tldw_Server_API.app.core.Security.http_hop import (
    NormalizedHTTPHopRequest,
    StatusOnlyHTTPHopResponse,
)
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    NOW,
    canonical_uuid4,
    key_ring,
    mark_migration_ready,
)
from tldw_Server_API.tests.Admin_Webhooks.test_recovery_backend_matrix import (
    BACKEND_PAIRS,
    MatrixWorkerContext,
    MutableClock,
    TokenSource,
    _apply_worker_disposition,
    _auth_repository,
    _jobs_manager,
    _reconciler,
    _seed_worker_delivery,
    _worker_handler,
    matrix_jobs_pg_dsn,  # noqa: F401 - imported fixture is registered by pytest
)

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)

SIGNING_SECRET = "whsec_" + "a" * 64
TARGET_URL = "https://receiver.example.test/admin-webhooks"
PUBLIC_EVENT_TYPES = (
    "user.created",
    "user.deleted",
    "incident.created",
    "incident.updated",
    "incident.resolved",
    "incident.notify",
)
BASE_HEADERS = {
    "content-type",
    "x-tldw-webhook-event",
    "x-tldw-webhook-event-id",
    "x-tldw-webhook-delivery-id",
    "x-tldw-webhook-timestamp",
    "x-tldw-webhook-secret-version",
    "x-tldw-webhook-signature",
}


@dataclass(frozen=True, slots=True)
class CapturedWebhook:
    body: bytes
    headers: dict[str, str]
    duplicate: bool


class ControlledHTTPSReceiver:
    """In-memory HTTPS receiver behind the reviewed status-only transport seam."""

    def __init__(self, secret: str, *statuses: int) -> None:
        self._secret = secret
        self._statuses = list(statuses)
        self._seen: set[tuple[str, str]] = set()
        self.captures: list[CapturedWebhook] = []

    async def __call__(
        self,
        request: NormalizedHTTPHopRequest,
    ) -> StatusOnlyHTTPHopResponse:
        assert request.scheme == "https"
        assert request.method == "POST"
        headers = dict(request.headers)
        expected_headers = set(BASE_HEADERS)
        if headers.get("x-tldw-webhook-test") == "true":
            expected_headers.add("x-tldw-webhook-test")
        assert set(headers) == expected_headers

        timestamp = headers["x-tldw-webhook-timestamp"]
        expected_signature = hmac.new(
            self._secret.encode("ascii"),
            timestamp.encode("ascii") + b"." + request.body,
            hashlib.sha256,
        ).hexdigest()
        assert hmac.compare_digest(
            headers["x-tldw-webhook-signature"],
            f"v1={expected_signature}",
        )

        identity = (
            headers["x-tldw-webhook-event-id"],
            headers["x-tldw-webhook-delivery-id"],
        )
        duplicate = identity in self._seen
        self._seen.add(identity)
        self.captures.append(
            CapturedWebhook(
                body=request.body,
                headers=headers,
                duplicate=duplicate,
            )
        )
        status = self._statuses.pop(0) if self._statuses else 204
        return StatusOnlyHTTPHopResponse(
            status_code=status,
            latency_ms=1,
            retry_after_seconds=None,
        )


class ReceiverClock:
    def __init__(self, clock: MutableClock) -> None:
        self._clock = clock
        self._monotonic = 100.0

    def utc_now(self) -> datetime:
        return self._clock()

    def monotonic(self) -> float:
        self._monotonic += 0.001
        return self._monotonic


def _settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.ON,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


async def _seed_receiver_registration(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
) -> int:
    await mark_migration_ready(repository)
    async with repository.transaction() as tx:
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
        await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description="Controlled HTTPS receiver",
                target=RegistrationTarget(
                    protected=target,
                    hostname="receiver.example.test",
                    display="https://receiver.example.test",
                ),
                event_types=PUBLIC_EVENT_TYPES,
                active=True,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=7,
                now=NOW - timedelta(hours=1),
            )
        )
    return webhook_id


def _event_data() -> dict[str, dict[str, object]]:
    created_at = NOW
    updated_at = NOW + timedelta(minutes=5)
    resolved_at = NOW + timedelta(minutes=10)
    return {
        "user.created": build_user_created_data(
            user_id=71,
            is_active=True,
            resource_version=updated_at,
            created_at=created_at,
            updated_at=updated_at,
        ),
        "user.deleted": build_user_deleted_data(
            user_id=72,
            resource_version=updated_at,
            created_at=created_at,
            updated_at=updated_at,
        ),
        "incident.created": build_incident_created_data(
            incident_id="inc-controlled",
            state="open",
            severity="high",
            resource_version=1,
            created_at=created_at,
            updated_at=created_at,
            resolved_at=None,
        ),
        "incident.updated": build_incident_updated_data(
            incident_id="inc-controlled",
            state="investigating",
            severity="critical",
            resource_version=2,
            created_at=created_at,
            updated_at=updated_at,
            resolved_at=None,
        ),
        "incident.resolved": build_incident_resolved_data(
            incident_id="inc-controlled",
            state="resolved",
            severity="critical",
            resource_version=3,
            created_at=created_at,
            updated_at=resolved_at,
            resolved_at=resolved_at,
        ),
        "incident.notify": build_incident_notify_data(
            incident_id="inc-controlled",
            state="investigating",
            severity="high",
            resource_version=2,
            created_at=created_at,
            updated_at=updated_at,
            resolved_at=None,
            narrative="Mitigation is in progress.",
        ),
    }


def _source_coordinates(event_type: str) -> dict[str, object]:
    if event_type in {"incident.created", "incident.updated", "incident.resolved"}:
        version = {
            "incident.created": "1",
            "incident.updated": "2",
            "incident.resolved": "3",
        }[event_type]
        return {
            "source_kind": EventSourceKind.AGGREGATE,
            "aggregate_type": "incident",
            "aggregate_id": "inc-controlled",
            "aggregate_version": version,
            "source_command_id": None,
        }
    return {
        "source_kind": EventSourceKind.COMMAND,
        "aggregate_type": None,
        "aggregate_id": None,
        "aggregate_version": None,
        "source_command_id": f"controlled-{event_type}-command",
    }


def _request_from_bundle(
    bundle: Any,
    ring: WebhookKeyRing,
    *,
    kind: DeliveryKind | None = None,
    delivery_id: str | None = None,
) -> AttemptExecutionRequest:
    registration = bundle.registration.registration
    target_url = ring.decrypt_text(
        purpose="registration.target",
        identity={
            "registration_id": registration.id,
            "target_version": registration.target_version,
        },
        protected=bundle.registration.target,
    )
    secret = ring.decrypt_text(
        purpose="registration.secret",
        identity={
            "registration_id": registration.id,
            "secret_version": registration.secret_version,
        },
        protected=bundle.registration.secret,
    )
    body = ring.decrypt_event_body(
        event_id=bundle.event.event.id,
        api_version=bundle.event.event.api_version,
        protected=bundle.event.body,
    )
    return AttemptExecutionRequest(
        target=ValidatedWebhookTarget(
            url=target_url,
            hostname=registration.target_hostname,
            target_display=registration.target_display,
        ),
        body=body,
        signing_secret=secret,
        timeout_seconds=registration.timeout_seconds,
        event_type=bundle.event.event.event_type,
        event_id=bundle.event.event.id,
        delivery_id=delivery_id or bundle.delivery.delivery.id,
        attempt_number=1,
        secret_version=registration.secret_version,
        kind=kind or bundle.delivery.delivery.kind,
    )


@pytest.mark.parametrize("auth_backend", ("sqlite", "postgres"))
@pytest.mark.integration
async def test_all_six_production_events_test_and_redelivery_reach_controlled_receiver(
    auth_backend: str,
    tmp_path,
    test_db_pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.executor.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, resolved_ips=("93.184.216.34",)),
    )
    ring = key_ring()
    event_ids = iter(canonical_uuid4(f"controlled-{event_type}") for event_type in PUBLIC_EVENT_TYPES)
    delivery_ids = iter(
        canonical_uuid4(f"controlled-{event_type}-delivery")
        for event_type in PUBLIC_EVENT_TYPES
    )
    clock = MutableClock(NOW + timedelta(hours=1))
    receiver = ControlledHTTPSReceiver(SIGNING_SECRET)
    executor = DeliveryAttemptExecutor(
        egress=receiver,
        clock=ReceiverClock(clock),
    )

    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id = await _seed_receiver_registration(repository, ring)
        producer = AdminWebhookEventProducer(
            repository=repository,
            settings=_settings(),
            key_ring_result=WebhookKeyRingLoadResult(
                ring=ring,
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            event_id_factory=lambda: next(event_ids),
            delivery_id_factory=lambda: next(delivery_ids),
            clock=clock,
        )
        captured_results = []
        first_preparation = None
        for event_type, data in _event_data().items():
            preparation = await producer.begin_capture(
                source_component="controlled-receiver-e2e",
                source_request_id=f"request-{event_type}",
            )
            assert preparation is not None
            async with repository.transaction() as tx:
                captured = await producer.capture_in_transaction(
                    preparation,
                    tx=tx,
                    event_type=event_type,
                    data=data,
                    **_source_coordinates(event_type),
                )
            assert captured.inserted is True
            assert len(captured.deliveries) == 1
            captured_results.append(captured)
            if first_preparation is None:
                first_preparation = preparation

        assert first_preparation is not None
        async with repository.transaction() as tx:
            duplicate = await producer.capture_in_transaction(
                first_preparation,
                tx=tx,
                event_type="user.created",
                data=_event_data()["user.created"],
                **_source_coordinates("user.created"),
            )
        assert duplicate.inserted is False
        assert duplicate.event.event.id == captured_results[0].event.event.id
        assert duplicate.deliveries == captured_results[0].deliveries

        automatic_requests: list[AttemptExecutionRequest] = []
        for captured in captured_results:
            delivery = captured.deliveries[0]
            bundle = await repository.get_delivery_bundle(delivery.delivery.id)
            assert bundle is not None
            request = _request_from_bundle(bundle, ring)
            automatic_requests.append(request)
            result = await executor.execute(request)
            assert result.outcome is AttemptOutcome.SUCCESS

        first_bundle = await repository.get_delivery_bundle(
            captured_results[0].deliveries[0].delivery.id
        )
        assert first_bundle is not None
        test_request = _request_from_bundle(
            first_bundle,
            ring,
            kind=DeliveryKind.TEST,
            delivery_id=canonical_uuid4("controlled-test-delivery"),
        )
        test_result = await executor.execute(test_request)
        assert test_result.outcome is AttemptOutcome.SUCCESS

        manual_request = _request_from_bundle(
            first_bundle,
            ring,
            kind=DeliveryKind.MANUAL,
            delivery_id=canonical_uuid4("controlled-manual-redelivery"),
        )
        manual_result = await executor.execute(manual_request)
        assert manual_result.outcome is AttemptOutcome.SUCCESS
        assert manual_request.event_id == automatic_requests[0].event_id
        assert manual_request.delivery_id != automatic_requests[0].delivery_id

        duplicate_result = await executor.execute(automatic_requests[0])
        assert duplicate_result.outcome is AttemptOutcome.SUCCESS

        decoded = [json.loads(item.body) for item in receiver.captures[:6]]
        assert [item["type"] for item in decoded] == list(PUBLIC_EVENT_TYPES)
        assert [item["id"] for item in decoded] == [
            request.event_id for request in automatic_requests
        ]
        assert all(item["created_at"].endswith("Z") for item in decoded)
        assert receiver.captures[6].headers["x-tldw-webhook-test"] == "true"
        assert "x-tldw-webhook-test" not in receiver.captures[7].headers
        assert receiver.captures[-1].duplicate is True
        assert len(
            {
                (
                    item.headers["x-tldw-webhook-event-id"],
                    item.headers["x-tldw-webhook-delivery-id"],
                )
                for item in receiver.captures
            }
        ) == 8
        history = await repository.list_delivery_history(webhook_id, limit=20)
        assert history.total == 6
        assert len(history.items) == 6


async def _process_worker_attempt(
    *,
    manager,
    handler,
    acquired: dict[str, Any],
) -> Any:
    disposition = await handler(acquired, MatrixWorkerContext(acquired))
    applied = _apply_worker_disposition(manager, acquired, disposition)
    await handler.on_disposition_applied(acquired, disposition, applied)
    return disposition


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_retry_and_terminal_classification_crosses_all_backend_pairs(
    auth_backend: str,
    jobs_backend: str,
    tmp_path,
    test_db_pool,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.domain.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, resolved_ips=("93.184.216.34",)),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Admin_Webhooks.executor.evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, resolved_ips=("93.184.216.34",)),
    )
    jobs_pg_dsn = request.getfixturevalue("matrix_jobs_pg_dsn")
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=jobs_pg_dsn,
    )
    clock = MutableClock(datetime.now(timezone.utc))
    receiver = ControlledHTTPSReceiver(SIGNING_SECRET, 503, 204, 400)
    executor = DeliveryAttemptExecutor(
        egress=receiver,
        clock=ReceiverClock(clock),
    )
    ring = key_ring()
    label = f"controlled-{auth_backend}-{jobs_backend}"

    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, retry_delivery_id = await _seed_worker_delivery(
            repository,
            ring,
            f"{label}-retry",
            now=clock(),
        )
        delivery_queue = JobsDeliveryQueue(manager)
        assert await _reconciler(
            repository,
            delivery_queue,
            clock,
            TokenSource(f"{label}-enqueue"),
        ).reconcile_enqueue_once() == 1
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker-1",
        )
        assert acquired is not None
        handler = _worker_handler(
            repository,
            ring,
            clock,
            executor,
            label,
        )
        retry = await _process_worker_attempt(
            manager=manager,
            handler=handler,
            acquired=acquired,
        )
        assert retry.kind is PreparedDispositionKind.RETRY
        assert retry.delay_seconds == 60
        clock.advance(60)
        assert manager.reschedule_jobs(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            status="queued",
            set_now=True,
        ) == 1
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker-2",
        )
        assert acquired is not None
        completed = await _process_worker_attempt(
            manager=manager,
            handler=handler,
            acquired=acquired,
        )
        assert completed.kind is PreparedDispositionKind.COMPLETE
        retry_attempts = await repository.list_delivery_attempts(
            webhook_id,
            retry_delivery_id,
        )
        assert [attempt.state for attempt in retry_attempts] == [
            AttemptState.RETRYABLE,
            AttemptState.SUCCEEDED,
        ]
        assert retry_attempts[0].requested_retry_delay_seconds == 60
        retry_bundle = await repository.get_delivery_bundle(retry_delivery_id)
        assert retry_bundle is not None
        assert retry_bundle.delivery.delivery.state is DeliveryState.SUCCEEDED

        terminal_webhook_id, terminal_delivery_id = await _seed_worker_delivery(
            repository,
            ring,
            f"{label}-terminal",
            now=clock(),
        )
        assert await _reconciler(
            repository,
            delivery_queue,
            clock,
            TokenSource(f"{label}-terminal-enqueue"),
        ).reconcile_enqueue_once() == 1
        terminal_job = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-terminal-worker",
        )
        assert terminal_job is not None
        terminal = await _process_worker_attempt(
            manager=manager,
            handler=handler,
            acquired=terminal_job,
        )
        assert terminal.kind is PreparedDispositionKind.FAIL
        terminal_attempts = await repository.list_delivery_attempts(
            terminal_webhook_id,
            terminal_delivery_id,
        )
        assert [attempt.state for attempt in terminal_attempts] == [AttemptState.FAILED]
        terminal_bundle = await repository.get_delivery_bundle(terminal_delivery_id)
        assert terminal_bundle is not None
        assert terminal_bundle.delivery.delivery.state is DeliveryState.DEAD

        assert len(receiver.captures) == 3
        assert [
            item.headers["x-tldw-webhook-delivery-id"]
            for item in receiver.captures[:2]
        ] == [retry_delivery_id, retry_delivery_id]
        assert (
            receiver.captures[0].headers["x-tldw-webhook-event-id"]
            == receiver.captures[1].headers["x-tldw-webhook-event-id"]
        )
        assert receiver.captures[1].duplicate is True
        assert receiver.captures[2].headers["x-tldw-webhook-delivery-id"] == terminal_delivery_id
