from __future__ import annotations

import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass
from datetime import timedelta
from typing import Protocol

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryKind,
    DeliveryState,
    EventSourceKind,
    WebhookError,
    WebhookErrorCode,
    build_idempotency_scope,
    build_registration_etag,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    EventInsert,
    WebhookRepositoryError,
    WebhookRepositoryErrorCode,
)
from tldw_Server_API.tests.Admin_Webhooks.test_test_delivery import (
    IDEMPOTENCY_KEY,
    NOW,
    canonical_uuid4,
    seed_ready_registration,
    settings,
)


class DeliveryRepositoryFixture(Protocol):
    repository: AdminWebhookRepository

    async def execute(self, query: str, *params: object) -> None: ...

    async def fetchval(self, query: str, *params: object) -> object: ...

    async def fetchrow(self, query: str, *params: object) -> object: ...


def _schema(name: str) -> type:
    module = importlib.import_module(
        "tldw_Server_API.app.api.v1.schemas.admin_webhooks"
    )
    assert hasattr(module, name), f"Task 10 schema {name} is missing"
    return getattr(module, name)


@pytest.mark.unit
def test_task10_request_schemas_are_strict_closed_and_non_nullable() -> None:
    test_request = _schema("WebhookTestRequest")
    redelivery_request = _schema("WebhookRedeliveryRequest")

    assert test_request(delivery_config_version=3).model_dump() == {
        "delivery_config_version": 3
    }
    assert redelivery_request(
        delivery_config_version=3,
        confirm_changed_configuration=False,
    ).model_dump() == {
        "delivery_config_version": 3,
        "confirm_changed_configuration": False,
    }
    for model, payload in (
        (test_request, {"delivery_config_version": "3"}),
        (test_request, {"delivery_config_version": None}),
        (
            redelivery_request,
            {
                "delivery_config_version": 3,
                "confirm_changed_configuration": 1,
            },
        ),
        (
            redelivery_request,
            {
                "delivery_config_version": 3,
                "confirm_changed_configuration": None,
            },
        ),
        (
            redelivery_request,
            {
                "delivery_config_version": 3,
                "confirm_changed_configuration": False,
                "payload": "secret-canary",
            },
        ),
    ):
        with pytest.raises(ValidationError):
            model(**payload)


@pytest.mark.unit
def test_task10_response_schemas_expose_only_bounded_delivery_metadata() -> None:
    delivery_model = _schema("WebhookDeliveryResponse")
    attempt_model = _schema("WebhookDeliveryAttemptResponse")
    history_model = _schema("WebhookDeliveryHistoryItemResponse")
    list_model = _schema("WebhookDeliveryListResponse")
    test_model = _schema("WebhookTestResponse")
    redelivery_model = _schema("WebhookRedeliveryResponse")

    delivery_fields = set(delivery_model.model_fields)
    assert delivery_fields == {
        "id",
        "event_id",
        "event_type",
        "webhook_id",
        "kind",
        "state",
        "delivery_config_version",
        "secret_version",
        "attempt_count",
        "status_code",
        "latency_ms",
        "reason_code",
        "expires_at",
        "created_at",
        "updated_at",
        "terminal_at",
        "redelivery_of_id",
        "completed_after_config_change",
    }
    assert set(attempt_model.model_fields) == {
        "id",
        "sequence",
        "state",
        "request_timeout_seconds",
        "status_code",
        "latency_ms",
        "reason_code",
        "requested_retry_delay_seconds",
        "started_at",
        "finished_at",
    }
    assert set(history_model.model_fields) == {"delivery", "attempts"}
    assert set(list_model.model_fields) == {"items", "total", "limit", "offset"}
    assert set(test_model.model_fields) == {
        "delivery",
        "attempt",
        "idempotent_replay",
        "in_progress",
    }
    assert set(redelivery_model.model_fields) == {"delivery", "idempotent_replay"}

    forbidden = {
        "body",
        "data",
        "ciphertext",
        "key_id",
        "target",
        "url",
        "path",
        "query",
        "secret",
        "signature",
        "request_headers",
        "response_body",
        "response_headers",
        "jobs_job_id",
        "lease_id",
        "token",
        "idempotency_key",
    }
    assert not delivery_fields & forbidden
    assert not set(attempt_model.model_fields) & forbidden


async def _seed_source_delivery(
    fixture: DeliveryRepositoryFixture,
    *,
    label: str,
) -> tuple[object, WebhookKeyRing, object]:
    registration, ring = await seed_ready_registration(fixture, active=True)
    event_id = canonical_uuid4(f"{label}-event")
    created_at = NOW - timedelta(minutes=5)
    body = json.dumps(
        {
            "api_version": "2026-07-01",
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
            "data": {"user_id": 11},
            "id": event_id,
            "type": "user.created",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    event = EventInsert(
        id=event_id,
        event_type="user.created",
        api_version="2026-07-01",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id=f"{label}-command",
        source_component="task10.tests",
        source_request_id=f"{label}-request",
        body=ring.encrypt_event_body(
            event_id=event_id,
            api_version="2026-07-01",
            body=body,
        ),
        body_size_bytes=len(body),
        created_at=created_at,
    )
    delivery_ids: list[str] = []

    def next_delivery_id() -> str:
        delivery_id = canonical_uuid4(f"{label}-source-{len(delivery_ids) + 1}")
        delivery_ids.append(delivery_id)
        return delivery_id

    async with fixture.repository.transaction() as tx:
        captured = await tx.capture_event_and_expand(
            event,
            next_delivery_id,
            created_at + timedelta(hours=72),
        )
    source = next(
        delivery
        for delivery in captured.deliveries
        if delivery.delivery.webhook_id == registration.id
    )
    return registration, ring, source


async def exercise_history_projection_is_set_based_and_key_independent(
    fixture: DeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    registration, _ring, source = await _seed_source_delivery(
        fixture,
        label="history",
    )
    first_attempt = canonical_uuid4("history-attempt-1")
    second_attempt = canonical_uuid4("history-attempt-2")
    await fixture.execute(
        """
        INSERT INTO admin_webhook_delivery_attempts (
            id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
            request_timeout_seconds, started_at, finished_at, state,
            status_code, latency_ms, reason_code,
            requested_retry_delay_seconds, jobs_disposition_applied, created_at
        ) VALUES (?, ?, 2, ?, ?, 10, ?, NULL, 'processing',
                  NULL, NULL, NULL, NULL, ?, ?)
        """,
        second_attempt,
        source.delivery.id,
        "history-job",
        "history-lease-2",
        NOW - timedelta(minutes=1),
        False,
        NOW - timedelta(minutes=1),
    )
    await fixture.execute(
        """
        INSERT INTO admin_webhook_delivery_attempts (
            id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
            request_timeout_seconds, started_at, finished_at, state,
            status_code, latency_ms, reason_code,
            requested_retry_delay_seconds, jobs_disposition_applied, created_at
        ) VALUES (?, ?, 1, ?, ?, 10, ?, ?, 'retryable',
                  503, 12, 'http_server_error', 60, ?, ?)
        """,
        first_attempt,
        source.delivery.id,
        "history-job",
        "history-lease-1",
        NOW - timedelta(minutes=3),
        NOW - timedelta(minutes=2),
        True,
        NOW - timedelta(minutes=3),
    )
    await fixture.execute(
        """
        UPDATE admin_webhook_deliveries
        SET jobs_job_id = ?, state = 'processing', attempt_count = 2,
            current_attempt_id = ?, completed_after_config_change = ?,
            updated_at = ?
        WHERE id = ?
        """,
        "history-job",
        second_attempt,
        True,
        NOW - timedelta(minutes=1),
        source.delivery.id,
    )
    async with fixture.repository.transaction() as tx:
        newer = await tx.insert_delivery(
            canonical_uuid4("history-newer-manual"),
            event_id=source.delivery.event_id,
            webhook_id=registration.id,
            kind=DeliveryKind.MANUAL,
            expires_at=NOW + timedelta(hours=72),
            now=NOW,
            redelivery_of_id=source.delivery.id,
        )
    await fixture.execute(
        """
        UPDATE admin_webhook_registrations
        SET deleted_at = ?, deleted_by_user_id = ?, active = ?, updated_at = ?
        WHERE id = ?
        """,
        NOW,
        7,
        False,
        NOW,
        registration.id,
    )

    attempt_queries: list[tuple[object, ...]] = []
    original_fetch = repository_module.AdminWebhookUnitOfWork._fetch

    async def traced_fetch(self, query, params=()):
        if "FROM admin_webhook_delivery_attempts AS attempt" in query:
            attempt_queries.append(tuple(params))
        return await original_fetch(self, query, params)

    monkeypatch.setattr(
        repository_module.AdminWebhookUnitOfWork,
        "_fetch",
        traced_fetch,
    )
    assert hasattr(
        delivery_module.AdminWebhookDeliveryService,
        "list_delivery_history",
    ), "Task 10 history service is missing"
    service = delivery_module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        event_id_factory=lambda: canonical_uuid4("unused-history-event"),
        delivery_id_factory=lambda: canonical_uuid4("unused-history-delivery"),
        clock=lambda: NOW,
    )
    page = await service.list_delivery_history(
        registration.id,
        limit=50,
        offset=0,
    )

    assert page.total == 2
    assert [item.delivery.id for item in page.items] == [
        newer.delivery.id,
        source.delivery.id,
    ]
    assert [item.event_type for item in page.items] == [
        "user.created",
        "user.created",
    ]
    assert page.items[0].attempts == ()
    assert [attempt.id for attempt in page.items[1].attempts] == [
        first_attempt,
        second_attempt,
    ]
    assert page.items[1].completed_after_config_change is True
    assert len(attempt_queries) == 1
    assert set(attempt_queries[0]) == {newer.delivery.id, source.delivery.id}
    rendered = repr(page)
    for canary in (
        "credential=not-for-metadata",
        "whsec_",
        "ciphertext",
        "history-job",
        "history-lease",
        "idempotency",
    ):
        assert canary not in rendered

    with pytest.raises(WebhookRepositoryError) as missing:
        await fixture.repository.list_delivery_history(999_999, limit=50, offset=0)
    assert missing.value.code is WebhookRepositoryErrorCode.NOT_FOUND
    with pytest.raises(ValueError):
        await fixture.repository.list_delivery_history(
            registration.id,
            limit=101,
            offset=0,
        )
    with pytest.raises(ValueError):
        await fixture.repository.list_delivery_history(
            registration.id,
            limit=50,
            offset=1_001,
        )


async def exercise_history_loads_only_public_columns(
    fixture: DeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
    registration, _ring, source = await _seed_source_delivery(
        fixture,
        label="history-public-columns",
    )
    test_delivery_id = canonical_uuid4("history-public-test-delivery")
    test_attempt_id = canonical_uuid4("history-public-test-attempt")
    async with fixture.repository.transaction() as tx:
        await tx.insert_delivery(
            test_delivery_id,
            event_id=source.delivery.event_id,
            webhook_id=registration.id,
            kind=DeliveryKind.TEST,
            expires_at=NOW + timedelta(hours=72),
            now=NOW,
        )
    await fixture.execute(
        """
        UPDATE admin_webhook_deliveries
        SET enqueue_claim_token = ?, enqueue_claim_expires_at = ?
        WHERE id = ?
        """,
        "not-a-recovery-token",
        NOW + timedelta(minutes=1),
        source.delivery.id,
    )
    await fixture.execute(
        """
        INSERT INTO admin_webhook_delivery_attempts (
            id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
            test_attempt_token, request_timeout_seconds, started_at,
            finished_at, state, status_code, latency_ms, reason_code,
            requested_retry_delay_seconds, jobs_disposition_applied, created_at
        ) VALUES (?, ?, 1, NULL, NULL, ?, 10, ?, NULL, 'processing',
                  NULL, NULL, NULL, NULL, ?, ?)
        """,
        test_attempt_id,
        test_delivery_id,
        "not-a-test-token",
        NOW,
        False,
        NOW,
    )

    selected_queries: list[str] = []
    original_fetch = repository_module.AdminWebhookUnitOfWork._fetch

    async def traced_fetch(self, query, params=()):
        if "admin_webhook_deliver" in query:
            selected_queries.append(str(query))
        return await original_fetch(self, query, params)

    def forbidden_internal_mapper(_row: object) -> object:
        raise AssertionError("public history used an execution-grade row mapper")

    monkeypatch.setattr(
        repository_module.AdminWebhookUnitOfWork,
        "_fetch",
        traced_fetch,
    )
    monkeypatch.setattr(
        repository_module,
        "_stored_delivery_from_row",
        forbidden_internal_mapper,
    )
    monkeypatch.setattr(
        repository_module,
        "_attempt_from_row",
        forbidden_internal_mapper,
    )

    page = await fixture.repository.list_delivery_history(
        registration.id,
        limit=50,
        offset=0,
    )

    assert page.total == 2
    assert {item.delivery.id for item in page.items} == {
        source.delivery.id,
        test_delivery_id,
    }
    test_item = next(
        item for item in page.items if item.delivery.id == test_delivery_id
    )
    assert [attempt.id for attempt in test_item.attempts] == [test_attempt_id]
    assert len(selected_queries) == 2
    selected_sql = "\n".join(selected_queries).lower()
    for forbidden in (
        "delivery.*",
        "jobs_job_id",
        "jobs_lease_id",
        "enqueue_claim_token",
        "enqueue_claim_expires_at",
        "current_attempt_id",
        "pending_jobs_disposition",
        "pending_jobs_disposition_token",
        "jobs_disposition_applied",
        "test_attempt_token",
        "idempotency",
        "ciphertext",
    ):
        assert forbidden not in selected_sql
    rendered = repr(page)
    assert "not-a-recovery-token" not in rendered
    assert "not-a-test-token" not in rendered


async def exercise_history_reads_one_consistent_snapshot(
    fixture: DeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
    registration, _ring, source = await _seed_source_delivery(
        fixture,
        label="history-snapshot",
    )
    concurrent_delivery_id = canonical_uuid4("history-snapshot-concurrent-delivery")
    concurrent_attempt_id = canonical_uuid4("history-snapshot-concurrent-attempt")
    page_commit_done = False
    attempt_commit_done = False
    original_fetch = repository_module.AdminWebhookUnitOfWork._fetch

    async def commit_between_history_statements(self, query, params=()):
        nonlocal page_commit_done, attempt_commit_done
        if (
            not page_commit_done
            and "FROM admin_webhook_deliveries AS delivery" in query
        ):
            page_commit_done = True
            async with fixture.repository.transaction() as tx:
                await tx.insert_delivery(
                    concurrent_delivery_id,
                    event_id=source.delivery.event_id,
                    webhook_id=registration.id,
                    kind=DeliveryKind.MANUAL,
                    expires_at=NOW + timedelta(hours=72),
                    now=NOW + timedelta(minutes=1),
                    redelivery_of_id=source.delivery.id,
                )
        if (
            not attempt_commit_done
            and "FROM admin_webhook_delivery_attempts AS attempt" in query
        ):
            attempt_commit_done = True
            await fixture.execute(
                """
                INSERT INTO admin_webhook_delivery_attempts (
                    id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
                    request_timeout_seconds, started_at, finished_at, state,
                    status_code, latency_ms, reason_code,
                    requested_retry_delay_seconds, jobs_disposition_applied,
                    created_at
                ) VALUES (?, ?, 1, ?, ?, 10, ?, NULL, 'processing',
                          NULL, NULL, NULL, NULL, ?, ?)
                """,
                concurrent_attempt_id,
                source.delivery.id,
                "history-snapshot-job",
                "history-snapshot-lease",
                NOW + timedelta(minutes=2),
                False,
                NOW + timedelta(minutes=2),
            )
        return await original_fetch(self, query, params)

    monkeypatch.setattr(
        repository_module.AdminWebhookUnitOfWork,
        "_fetch",
        commit_between_history_statements,
    )

    page = await asyncio.wait_for(
        fixture.repository.list_delivery_history(
            registration.id,
            limit=50,
            offset=0,
        ),
        timeout=10,
    )

    assert page_commit_done is True
    assert attempt_commit_done is True
    assert page.total == 1
    assert [item.delivery.id for item in page.items] == [source.delivery.id]
    assert page.items[0].attempts == ()
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE webhook_id = ?",
        registration.id,
    ) == 2
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_delivery_attempts WHERE delivery_id = ?",
        source.delivery.id,
    ) == 1


def _redelivery_command(
    delivery_module,
    registration,
    source_delivery_id: str,
    *,
    key: str = IDEMPOTENCY_KEY,
    confirm: bool = False,
):
    assert hasattr(delivery_module, "RedeliverWebhookCommand"), (
        "Task 10 redelivery command is missing"
    )
    return delivery_module.RedeliverWebhookCommand(
        actor_id=7,
        webhook_id=registration.id,
        source_delivery_id=source_delivery_id,
        if_match=build_registration_etag(
            webhook_id=registration.id,
            revision=registration.revision,
        ),
        delivery_config_version=registration.delivery_config_version,
        confirm_changed_configuration=confirm,
        idempotency_key=key,
        request_id="task10-redelivery-request",
    )


def _redelivery_service(
    delivery_module,
    fixture: DeliveryRepositoryFixture,
    ring_result: WebhookKeyRingLoadResult,
    *,
    label: str,
    repository: object | None = None,
    delivery_id_factory=None,
):
    return delivery_module.AdminWebhookDeliveryService(
        repository=repository or fixture.repository,
        key_ring_result=ring_result,
        event_id_factory=lambda: canonical_uuid4(f"unused-{label}-event"),
        delivery_id_factory=(
            delivery_id_factory
            or (lambda: canonical_uuid4(f"{label}-manual"))
        ),
        clock=lambda: NOW,
        settings=settings(),
    )


async def exercise_redelivery_success_exact_replay_and_malformed_coordinate(
    fixture: DeliveryRepositoryFixture,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    registration, ring, source = await _seed_source_delivery(
        fixture,
        label="redelivery-success",
    )
    await fixture.execute(
        """
        UPDATE admin_webhook_migration_state
        SET first_canonical_activity_at = NULL,
            first_canonical_activity_kind = NULL
        WHERE singleton_id = 1
        """
    )
    command = _redelivery_command(
        delivery_module,
        registration,
        source.delivery.id,
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)

    assert hasattr(
        delivery_module.AdminWebhookDeliveryService,
        "redeliver_webhook",
    ), "Task 10 redelivery service is missing"
    result = await _redelivery_service(
        delivery_module,
        fixture,
        WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        label="redelivery-success",
    ).redeliver_webhook(command, audit_sink=audit_sink)

    assert result.idempotent_replay is False
    assert result.delivery.kind is DeliveryKind.MANUAL
    assert result.delivery.state is DeliveryState.PENDING
    assert result.delivery.event_id == source.delivery.event_id
    assert result.delivery.redelivery_of_id == source.delivery.id
    assert result.delivery.delivery_config_version == registration.delivery_config_version
    assert result.delivery.secret_version == registration.secret_version
    assert result.delivery.expires_at == NOW + timedelta(hours=72)
    assert [record.outcome for record in audits] == ["accepted"]
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_delivery_attempts WHERE delivery_id = ?",
        result.delivery.id,
    ) == 0
    assert await fixture.fetchval(
        "SELECT jobs_job_id FROM admin_webhook_deliveries WHERE id = ?",
        result.delivery.id,
    ) is None
    assert await fixture.fetchval(
        "SELECT first_canonical_activity_kind FROM admin_webhook_migration_state WHERE singleton_id = 1"
    ) == "delivery_attempt"
    assert await fixture.fetchval(
        "SELECT test_delivery_id FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    ) is None
    metadata = await fixture.fetchval(
        "SELECT response_metadata_json FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    )
    assert json.loads(str(metadata)) == {
        "redelivery_delivery_id": result.delivery.id
    }

    replay_service = _redelivery_service(
        delivery_module,
        fixture,
        WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        label="must-not-allocate",
    )
    replay = await replay_service.redeliver_webhook(command, audit_sink=audit_sink)
    assert replay.idempotent_replay is True
    assert replay.delivery == result.delivery
    assert [record.outcome for record in audits] == ["accepted", "no_op"]
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'manual'"
    ) == 1

    conflicting = _redelivery_command(
        delivery_module,
        registration,
        source.delivery.id,
        confirm=True,
    )
    with pytest.raises(WebhookError) as conflict:
        await replay_service.redeliver_webhook(conflicting, audit_sink=audit_sink)
    assert conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT

    await fixture.execute(
        """
        UPDATE admin_webhook_idempotency
        SET response_metadata_json = ?
        WHERE operation = 'redeliver'
        """,
        '{"redelivery_delivery_id":"malformed-coordinate"}',
    )
    with pytest.raises(WebhookError) as malformed:
        await replay_service.redeliver_webhook(command, audit_sink=audit_sink)
    assert malformed.value.code is WebhookErrorCode.DELIVERY_UNAVAILABLE
    assert audits[-1].outcome == "failed"

    await fixture.execute(
        "DELETE FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    )
    malformed_factory = _redelivery_service(
        delivery_module,
        fixture,
        WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        label="malformed-factory",
        delivery_id_factory=lambda: "not-a-canonical-uuid",
    )
    with pytest.raises(WebhookError) as unavailable:
        await malformed_factory.redeliver_webhook(
            command,
            audit_sink=audit_sink,
        )
    assert unavailable.value.code is WebhookErrorCode.DELIVERY_UNAVAILABLE
    assert audits[-1].outcome == "failed"
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    ) == 0


async def exercise_redelivery_key_family_conflicts_across_sources(
    fixture: DeliveryRepositoryFixture,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    registration, ring, source = await _seed_source_delivery(
        fixture,
        label="redelivery-key-family",
    )
    second_source_id = canonical_uuid4("redelivery-key-family-second-source")
    async with fixture.repository.transaction() as tx:
        second_source = await tx.insert_delivery(
            second_source_id,
            event_id=source.delivery.event_id,
            webhook_id=registration.id,
            kind=DeliveryKind.MANUAL,
            expires_at=NOW + timedelta(hours=72),
            now=NOW - timedelta(minutes=1),
            redelivery_of_id=source.delivery.id,
        )
    available = WebhookKeyRingLoadResult(
        ring=ring,
        code=WebhookKeyLoadCode.AVAILABLE,
    )
    service = _redelivery_service(
        delivery_module,
        fixture,
        available,
        label="redelivery-key-family-first",
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)

    first_command = _redelivery_command(
        delivery_module,
        registration,
        source.delivery.id,
        key="77778888999900001111222233334444",
    )
    first = await service.redeliver_webhook(first_command, audit_sink=audit_sink)
    before_deliveries = await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE webhook_id = ?",
        registration.id,
    )
    before_idempotency = await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    )
    before_activity = await fixture.fetchrow(
        """
        SELECT first_canonical_activity_at, first_canonical_activity_kind
        FROM admin_webhook_migration_state WHERE singleton_id = 1
        """
    )

    conflicting_command = _redelivery_command(
        delivery_module,
        registration,
        second_source.delivery.id,
        key=first_command.idempotency_key,
    )
    with pytest.raises(WebhookError) as conflict:
        await _redelivery_service(
            delivery_module,
            fixture,
            available,
            label="redelivery-key-family-second",
        ).redeliver_webhook(
            conflicting_command,
            audit_sink=audit_sink,
        )

    assert conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE webhook_id = ?",
        registration.id,
    ) == before_deliveries
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    ) == before_idempotency
    after_activity = await fixture.fetchrow(
        """
        SELECT first_canonical_activity_at, first_canonical_activity_kind
        FROM admin_webhook_migration_state WHERE singleton_id = 1
        """
    )
    assert dict(after_activity) == dict(before_activity)  # type: ignore[arg-type]
    assert [record.outcome for record in audits] == ["accepted", "denied"]
    assert first.delivery.redelivery_of_id == source.delivery.id

    stored = await fixture.fetchrow(
        """
        SELECT route, webhook_id, delivery_id
        FROM admin_webhook_idempotency WHERE operation = 'redeliver'
        """
    )
    assert stored["route"] == (
        f"/admin/webhooks/{registration.id}/deliveries/"
        f"{source.delivery.id}/redeliver"
    )
    assert stored["webhook_id"] == registration.id
    assert stored["delivery_id"] == source.delivery.id

    foreign_registration, foreign_ring, foreign_source = await _seed_source_delivery(
        fixture,
        label="redelivery-key-family-foreign-webhook",
    )
    foreign_result = await _redelivery_service(
        delivery_module,
        fixture,
        WebhookKeyRingLoadResult(
            ring=foreign_ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        label="redelivery-key-family-foreign",
    ).redeliver_webhook(
        _redelivery_command(
            delivery_module,
            foreign_registration,
            foreign_source.delivery.id,
            key=first_command.idempotency_key,
        ),
        audit_sink=audit_sink,
    )
    assert foreign_result.idempotent_replay is False
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    ) == 2


async def exercise_redelivery_replay_rows_have_exact_action_shape(
    fixture: DeliveryRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    repository_module = importlib.import_module(
        "tldw_Server_API.app.core.DB_Management.admin_webhooks_repository"
    )
    registration, ring, source = await _seed_source_delivery(
        fixture,
        label="redelivery-exact-row",
    )
    command = _redelivery_command(
        delivery_module,
        registration,
        source.delivery.id,
        key="88889999000011112222333344445555",
    )

    async def audit_sink(_record: object) -> None:
        return None

    created = await _redelivery_service(
        delivery_module,
        fixture,
        WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        label="redelivery-exact-row",
    ).redeliver_webhook(command, audit_sink=audit_sink)
    persisted = await fixture.fetchrow(
        """
        SELECT lookup_digest, request_fingerprint
        FROM admin_webhook_idempotency WHERE operation = 'redeliver'
        """
    )
    scope = build_idempotency_scope(
        actor_id=command.actor_id,
        operation="redeliver",
        route=(
            f"/admin/webhooks/{command.webhook_id}/deliveries/"
            f"{command.source_delivery_id}/redeliver"
        ),
        webhook_id=command.webhook_id,
        delivery_id=command.source_delivery_id,
    )
    lookup_args = {
        "lookup_digest": str(persisted["lookup_digest"]),
        "scope": scope,
        "request_fingerprint": str(persisted["request_fingerprint"]),
        "now": NOW,
    }
    valid_metadata = json.dumps(
        {"redelivery_delivery_id": created.delivery.id},
        separators=(",", ":"),
    )

    await fixture.execute(
        """
        UPDATE admin_webhook_registrations
        SET deleted_at = ?, deleted_by_user_id = ?, active = ?, updated_at = ?
        WHERE id = ?
        """,
        NOW,
        7,
        False,
        NOW,
        registration.id,
    )
    registration_reads = 0
    original_registration_row = repository_module.AdminWebhookUnitOfWork._registration_row

    async def tracked_registration_row(self, *args, **kwargs):
        nonlocal registration_reads
        registration_reads += 1
        return await original_registration_row(self, *args, **kwargs)

    monkeypatch.setattr(
        repository_module.AdminWebhookUnitOfWork,
        "_registration_row",
        tracked_registration_row,
    )

    async def reset_row(*, state: str, status: int | None, metadata: str | None) -> None:
        await fixture.execute(
            """
            UPDATE admin_webhook_idempotency
            SET state = ?, resource_id = NULL, resource_version = NULL,
                secret_version = NULL, replay_secret_ciphertext_json = NULL,
                replay_secret_key_id = NULL, test_delivery_id = NULL,
                test_attempt_id = NULL, response_status = ?,
                response_metadata_json = ?
            WHERE operation = 'redeliver'
            """,
            state,
            status,
            metadata,
        )

    await reset_row(state="in_progress", status=None, metadata=None)
    in_progress = await fixture.repository.lookup_idempotency(**lookup_args)
    assert in_progress.kind is repository_module.IdempotencyLookupKind.IN_PROGRESS
    assert in_progress.response_status is None
    assert in_progress.response_metadata is None
    assert in_progress.redelivery_delivery_id is None

    await reset_row(state="completed", status=202, metadata=valid_metadata)
    replay = await fixture.repository.lookup_idempotency(**lookup_args)
    assert replay.kind is repository_module.IdempotencyLookupKind.REPLAY
    assert replay.redelivery_delivery_id == created.delivery.id
    assert replay.response_status == 202
    assert dict(replay.response_metadata or {}) == {
        "redelivery_delivery_id": created.delivery.id
    }
    assert registration_reads == 0

    malformed_updates: tuple[tuple[str, str, tuple[object, ...]], ...] = (
        (
            "wrong_status",
            "UPDATE admin_webhook_idempotency SET response_status = ? WHERE operation = 'redeliver'",
            (200,),
        ),
        (
            "extra_metadata",
            "UPDATE admin_webhook_idempotency SET response_metadata_json = ? WHERE operation = 'redeliver'",
            (json.dumps({"redelivery_delivery_id": created.delivery.id, "status_code": 204}),),
        ),
        (
            "missing_coordinate",
            "UPDATE admin_webhook_idempotency SET response_metadata_json = NULL WHERE operation = 'redeliver'",
            (),
        ),
        (
            "generic_resource",
            "UPDATE admin_webhook_idempotency SET resource_id = ?, resource_version = ? WHERE operation = 'redeliver'",
            (registration.id, 1),
        ),
        (
            "generic_replay_secret",
            """
            UPDATE admin_webhook_idempotency
            SET resource_id = ?, resource_version = ?, secret_version = ?,
                replay_secret_ciphertext_json = ?, replay_secret_key_id = ?
            WHERE operation = 'redeliver'
            """,
            (registration.id, 1, 1, '{"ciphertext":"canary"}', "canary-key"),
        ),
        (
            "test_coordinates",
            """
            UPDATE admin_webhook_idempotency
            SET test_delivery_id = ?, test_attempt_id = ?
            WHERE operation = 'redeliver'
            """,
            (
                canonical_uuid4("redelivery-row-test-delivery"),
                canonical_uuid4("redelivery-row-test-attempt"),
            ),
        ),
        (
            "in_progress_result",
            "UPDATE admin_webhook_idempotency SET state = 'in_progress' WHERE operation = 'redeliver'",
            (),
        ),
    )
    failures: list[str] = []
    for name, query, params in malformed_updates:
        await reset_row(state="completed", status=202, metadata=valid_metadata)
        await fixture.execute(query, *params)
        reads_before = registration_reads
        try:
            await fixture.repository.lookup_idempotency(**lookup_args)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001 - aggregate exact RED evidence
            failures.append(f"{name}:{type(exc).__name__}")
        else:
            failures.append(f"{name}:accepted")
        if registration_reads != reads_before:
            failures.append(f"{name}:registration-read")

    assert failures == []


async def exercise_redelivery_preconditions_and_audit_rollback(
    fixture: DeliveryRepositoryFixture,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    registration, ring, source = await _seed_source_delivery(
        fixture,
        label="redelivery-preconditions",
    )
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
    current = await fixture.repository.get_registration(registration.id)
    assert current is not None
    available = WebhookKeyRingLoadResult(
        ring=ring,
        code=WebhookKeyLoadCode.AVAILABLE,
    )
    service = _redelivery_service(
        delivery_module,
        fixture,
        available,
        label="confirmed-redelivery",
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)

    unconfirmed = _redelivery_command(
        delivery_module,
        current,
        source.delivery.id,
        key="11112222333344445555666677778888",
        confirm=False,
    )
    with pytest.raises(WebhookError) as confirmation:
        await service.redeliver_webhook(unconfirmed, audit_sink=audit_sink)
    assert confirmation.value.code is WebhookErrorCode.REDELIVERY_CONFIRMATION_REQUIRED
    assert audits[-1].outcome == "denied"
    assert audits[-1].redelivery_to_changed_config is True
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    ) == 0

    confirmed = _redelivery_command(
        delivery_module,
        current,
        source.delivery.id,
        key="22223333444455556666777788889999",
        confirm=True,
    )
    accepted = await service.redeliver_webhook(confirmed, audit_sink=audit_sink)
    assert accepted.delivery.delivery_config_version == current.delivery_config_version
    assert audits[-1].outcome == "accepted"
    assert audits[-1].source_config_version == source.delivery.delivery_config_version
    assert audits[-1].current_config_version == current.delivery_config_version

    foreign_registration, _foreign_ring, foreign_source = await _seed_source_delivery(
        fixture,
        label="redelivery-foreign",
    )
    assert foreign_registration.id != current.id
    foreign = _redelivery_command(
        delivery_module,
        current,
        foreign_source.delivery.id,
        key="33334444555566667777888899990000",
        confirm=True,
    )
    with pytest.raises(WebhookError) as not_found:
        await service.redeliver_webhook(foreign, audit_sink=audit_sink)
    assert not_found.value.code is WebhookErrorCode.NOT_FOUND

    before_deliveries = await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'manual'"
    )
    rollback_command = _redelivery_command(
        delivery_module,
        current,
        source.delivery.id,
        key="44445555666677778888999900001111",
        confirm=True,
    )

    async def unavailable_audit(_record: object) -> None:
        raise RuntimeError("mandatory audit unavailable")

    with pytest.raises(WebhookError) as unavailable:
        await _redelivery_service(
            delivery_module,
            fixture,
            available,
            label="audit-rollback",
        ).redeliver_webhook(rollback_command, audit_sink=unavailable_audit)
    assert unavailable.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'manual'"
    ) == before_deliveries
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE lookup_digest IS NOT NULL"
    ) == 1


async def exercise_redelivery_concurrency_and_commit_failure(
    fixture: DeliveryRepositoryFixture,
) -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    registration, ring, source = await _seed_source_delivery(
        fixture,
        label="redelivery-concurrency",
    )
    available = WebhookKeyRingLoadResult(
        ring=ring,
        code=WebhookKeyLoadCode.AVAILABLE,
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
    command = _redelivery_command(
        delivery_module,
        registration,
        source.delivery.id,
        key="55556666777788889999000011112222",
    )
    audits: list[object] = []

    async def audit_sink(record: object) -> None:
        audits.append(record)

    first = asyncio.create_task(
        _redelivery_service(
            delivery_module,
            fixture,
            available,
            label="concurrent-first",
            repository=synchronized,
        ).redeliver_webhook(command, audit_sink=audit_sink)
    )
    second = asyncio.create_task(
        _redelivery_service(
            delivery_module,
            fixture,
            available,
            label="concurrent-second",
            repository=synchronized,
        ).redeliver_webhook(command, audit_sink=audit_sink)
    )
    await synchronized.ready.wait()
    synchronized.release.set()
    results = await asyncio.gather(first, second)
    assert sorted(result.idempotent_replay for result in results) == [False, True]
    assert results[0].delivery.id == results[1].delivery.id
    assert [record.outcome for record in audits].count("accepted") == 1
    assert [record.outcome for record in audits].count("no_op") == 1
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'manual'"
    ) == 1

    class CommitFailRepository:
        def __getattr__(self, name: str):
            return getattr(fixture.repository, name)

        @asynccontextmanager
        async def transaction(self):
            async with fixture.repository.transaction() as tx:
                yield tx
                raise TransactionError("simulated redelivery commit failure")

    commit_command = _redelivery_command(
        delivery_module,
        registration,
        source.delivery.id,
        key="66667777888899990000111122223333",
    )
    commit_audits: list[object] = []

    async def commit_audit_sink(record: object) -> None:
        commit_audits.append(record)
        if record.outcome == "failed":
            raise RuntimeError("follow-up audit unavailable")

    with pytest.raises(WebhookError) as commit_failure:
        await _redelivery_service(
            delivery_module,
            fixture,
            available,
            label="commit-failure",
            repository=CommitFailRepository(),
        ).redeliver_webhook(commit_command, audit_sink=commit_audit_sink)
    assert commit_failure.value.code is WebhookErrorCode.OPERATION_FAILED
    assert [record.outcome for record in commit_audits] == ["accepted", "failed"]
    for name in (
        "actor_id",
        "webhook_id",
        "source_delivery_id",
        "delivery_id",
        "request_id",
    ):
        assert getattr(commit_audits[0], name) == getattr(commit_audits[1], name)
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE kind = 'manual'"
    ) == 1
    assert await fixture.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_idempotency WHERE operation = 'redeliver'"
    ) == 1


@pytest.mark.unit
def test_redelivery_contract_is_frozen_and_hides_conditional_and_idempotency_inputs() -> None:
    module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )
    assert hasattr(module, "RedeliverWebhookCommand"), (
        "Task 10 redelivery command is missing"
    )
    assert hasattr(module, "RedeliverWebhookResult"), (
        "Task 10 redelivery result is missing"
    )
    command = module.RedeliverWebhookCommand(
        actor_id=7,
        webhook_id=41,
        source_delivery_id=canonical_uuid4("contract-source"),
        if_match='"admin-webhook-41-r3"',
        delivery_config_version=4,
        confirm_changed_configuration=False,
        idempotency_key=IDEMPOTENCY_KEY,
        request_id="task10-contract",
    )
    for record in (module.RedeliverWebhookCommand, module.RedeliverWebhookResult):
        assert is_dataclass(record)
        assert record.__dataclass_params__.frozen
    assert {item.name for item in fields(command)} == {
        "actor_id",
        "webhook_id",
        "source_delivery_id",
        "if_match",
        "delivery_config_version",
        "confirm_changed_configuration",
        "idempotency_key",
        "request_id",
    }
    assert IDEMPOTENCY_KEY not in repr(command)
    assert command.if_match not in repr(command)
    assert "target" not in repr(command)


@pytest.mark.unit
async def test_history_service_localizes_repository_not_found_mapping() -> None:
    delivery_module = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )

    class MissingHistoryRepository:
        async def list_delivery_history(self, webhook_id: int, *, limit: int, offset: int):
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)

    service = delivery_module.AdminWebhookDeliveryService(
        repository=MissingHistoryRepository(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        event_id_factory=lambda: canonical_uuid4("unused-missing-history-event"),
        delivery_id_factory=lambda: canonical_uuid4(
            "unused-missing-history-delivery"
        ),
        clock=lambda: NOW,
    )

    with pytest.raises(WebhookError) as missing:
        await service.list_delivery_history(999_999, limit=50, offset=0)
    assert missing.value.code is WebhookErrorCode.NOT_FOUND
    shared_mapping = delivery_module._map_capture_error(
        WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)
    )
    assert shared_mapping.code is WebhookErrorCode.OPERATION_FAILED
