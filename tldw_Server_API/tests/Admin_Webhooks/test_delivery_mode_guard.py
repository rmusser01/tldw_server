from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.Admin_Webhooks import delivery
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    EventSourceKind,
    WebhookError,
    WebhookErrorCode,
)

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
SOURCE_DELIVERY_ID = "12345678-1234-4123-8123-123456789abc"


class _RepositorySpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        async def _unexpected(*_args: object, **_kwargs: object) -> None:
            self.calls.append(name)
            raise AssertionError(f"repository call escaped mode guard: {name}")

        return _unexpected


class _ExecutorSpy:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, _request: object) -> None:
        self.calls += 1
        raise AssertionError("executor call escaped mode guard")


def _settings(mode: AdminWebhookMode) -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=mode,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


def _service(
    repository: _RepositorySpy,
    executor: _ExecutorSpy,
    *,
    settings: AdminWebhookSettings | None,
    factory_calls: dict[str, int],
) -> delivery.AdminWebhookDeliveryService:
    def _value(name: str, value: object):
        def _factory() -> object:
            factory_calls[name] += 1
            return value

        return _factory

    return delivery.AdminWebhookDeliveryService(
        repository=repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        event_id_factory=_value("event", SOURCE_DELIVERY_ID),
        delivery_id_factory=_value("delivery", SOURCE_DELIVERY_ID),
        clock=_value("clock", NOW),
        settings=settings,
        executor=executor,
        test_attempt_id_factory=_value("attempt", SOURCE_DELIVERY_ID),
        test_token_factory=_value("token", "a" * 64),
    )


def _test_command() -> delivery.TestWebhookCommand:
    return delivery.TestWebhookCommand(
        actor_id=7,
        webhook_id=11,
        if_match=None,
        delivery_config_version=1,
        idempotency_key="0123456789abcdef0123456789abcdef",
        request_id="mode-guard-test",
    )


def _redelivery_command() -> delivery.RedeliverWebhookCommand:
    return delivery.RedeliverWebhookCommand(
        actor_id=7,
        webhook_id=11,
        source_delivery_id=SOURCE_DELIVERY_ID,
        if_match=None,
        delivery_config_version=1,
        confirm_changed_configuration=False,
        idempotency_key="abcdef0123456789abcdef0123456789",
        request_id="mode-guard-redelivery",
    )


def _capture_command() -> delivery.CaptureSyntheticEventCommand:
    return delivery.CaptureSyntheticEventCommand(
        actor_id=7,
        request_id="mode-guard-capture",
        event_type="user.created",
        source_kind=EventSourceKind.AGGREGATE,
        aggregate_type="user",
        aggregate_id="7",
        aggregate_version="1",
        source_command_id=None,
        source_component="mode-guard-test",
        source_request_id="request-7",
        data={"private": "event-body-canary"},
    )


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        (AdminWebhookMode.OFF, WebhookErrorCode.DISABLED),
        (AdminWebhookMode.MIGRATE, WebhookErrorCode.MIGRATION_PENDING),
    ],
)
@pytest.mark.asyncio
async def test_delivery_surfaces_fail_closed_before_side_effects(
    mode: AdminWebhookMode,
    expected_code: WebhookErrorCode,
) -> None:
    repository = _RepositorySpy()
    executor = _ExecutorSpy()
    factory_calls = dict.fromkeys(
        ("event", "delivery", "clock", "attempt", "token"), 0
    )
    service = _service(
        repository,
        executor,
        settings=_settings(mode),
        factory_calls=factory_calls,
    )
    test_audits: list[object] = []
    mutation_audits: list[object] = []
    capture_audits: list[object] = []

    async def _record_test(record: object) -> None:
        test_audits.append(record)

    async def _record_mutation(record: object) -> None:
        mutation_audits.append(record)

    async def _record_capture(record: object) -> None:
        capture_audits.append(record)

    for operation in (
        service.test_webhook(_test_command(), audit_sink=_record_test),
        service.list_delivery_history(11, limit=50, offset=0),
        service.redeliver_webhook(_redelivery_command(), audit_sink=_record_mutation),
        service.capture_synthetic_event(_capture_command(), audit_sink=_record_capture),
    ):
        with pytest.raises(WebhookError) as exc_info:
            await operation
        assert exc_info.value.code is expected_code

    assert repository.calls == []
    assert executor.calls == 0
    assert factory_calls == dict.fromkeys(factory_calls, 0)
    assert test_audits == []
    assert len(mutation_audits) == 1
    assert mutation_audits[0].outcome == "denied"
    assert mutation_audits[0].reason_code is expected_code
    assert len(capture_audits) == 1
    assert capture_audits[0].outcome == "failed"
    assert capture_audits[0].reason_code is expected_code
    assert capture_audits[0].event_id is None


@pytest.mark.asyncio
async def test_direct_delivery_service_construction_fails_closed() -> None:
    repository = _RepositorySpy()
    service = _service(
        repository,
        _ExecutorSpy(),
        settings=None,
        factory_calls=dict.fromkeys(
            ("event", "delivery", "clock", "attempt", "token"), 0
        ),
    )

    with pytest.raises(WebhookError) as exc_info:
        await service.list_delivery_history(11, limit=50, offset=0)

    assert exc_info.value.code is WebhookErrorCode.DISABLED
    assert repository.calls == []


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("off", WebhookErrorCode.DISABLED),
        ("migrate", WebhookErrorCode.MIGRATION_PENDING),
    ],
)
@pytest.mark.asyncio
async def test_production_composition_denies_without_resource_initialization(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_code: WebhookErrorCode,
) -> None:
    from tldw_Server_API.app.core.Admin_Webhooks import crypto, observability
    from tldw_Server_API.app.core.AuthNZ import database
    from tldw_Server_API.app.core.DB_Management import (
        admin_webhooks_repository,
    )

    resource_calls = dict.fromkeys(
        ("pool", "repository", "key_ring", "executor", "metrics"), 0
    )

    def _unexpected_resource(name: str):
        def _raise(*_args: object, **_kwargs: object) -> None:
            resource_calls[name] += 1
            raise AssertionError(f"{name} initialized while delivery mode is {mode}")

        return _raise

    async def _unexpected_pool() -> None:
        resource_calls["pool"] += 1
        raise AssertionError(f"pool initialized while delivery mode is {mode}")

    monkeypatch.setenv("TLDW_ADMIN_WEBHOOKS_MODE", mode)
    monkeypatch.setenv("TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT", "false")
    monkeypatch.setattr(database, "get_db_pool", _unexpected_pool)
    monkeypatch.setattr(
        admin_webhooks_repository,
        "AdminWebhookRepository",
        _unexpected_resource("repository"),
    )
    monkeypatch.setattr(
        crypto,
        "load_webhook_key_ring",
        _unexpected_resource("key_ring"),
    )
    monkeypatch.setattr(
        delivery,
        "DeliveryAttemptExecutor",
        _unexpected_resource("executor"),
    )
    monkeypatch.setattr(
        observability,
        "AdminWebhookMetrics",
        _unexpected_resource("metrics"),
    )

    service = await delivery.get_admin_webhook_delivery_service()
    audits: list[object] = []

    async def _record(record: object) -> None:
        audits.append(record)

    with pytest.raises(WebhookError) as exc_info:
        await service.redeliver_webhook(
            _redelivery_command(),
            audit_sink=_record,
        )

    assert exc_info.value.code is expected_code
    assert resource_calls == dict.fromkeys(resource_calls, 0)
    assert len(audits) == 1
    assert audits[0].outcome == "denied"
    assert audits[0].reason_code is expected_code


@pytest.mark.asyncio
async def test_production_composition_constructs_all_resources_in_on_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Admin_Webhooks import crypto, observability
    from tldw_Server_API.app.core.AuthNZ import database
    from tldw_Server_API.app.core.DB_Management import (
        admin_webhooks_repository,
    )

    pool = object()
    repository = object()
    executor = object()
    metrics = object()
    key_result = WebhookKeyRingLoadResult(
        ring=None,
        code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
    )
    calls: list[tuple[str, object]] = []

    async def _get_pool() -> object:
        calls.append(("pool", pool))
        return pool

    def _repository(value: object) -> object:
        calls.append(("repository", value))
        return repository

    def _key_ring() -> WebhookKeyRingLoadResult:
        calls.append(("key_ring", key_result))
        return key_result

    def _executor(*, allow_http_dev: bool) -> object:
        calls.append(("executor", allow_http_dev))
        return executor

    def _metrics() -> object:
        calls.append(("metrics", metrics))
        return metrics

    monkeypatch.setenv("TLDW_ADMIN_WEBHOOKS_MODE", "on")
    monkeypatch.setenv("TLDW_ADMIN_WEBHOOKS_LEGACY_COMPAT", "false")
    monkeypatch.setattr(database, "get_db_pool", _get_pool)
    monkeypatch.setattr(
        admin_webhooks_repository,
        "AdminWebhookRepository",
        _repository,
    )
    monkeypatch.setattr(crypto, "load_webhook_key_ring", _key_ring)
    monkeypatch.setattr(delivery, "DeliveryAttemptExecutor", _executor)
    monkeypatch.setattr(observability, "AdminWebhookMetrics", _metrics)

    service = await delivery.get_admin_webhook_delivery_service()

    assert calls == [
        ("pool", pool),
        ("repository", pool),
        ("key_ring", key_result),
        ("executor", False),
        ("metrics", metrics),
    ]
    assert service._repository is repository
    assert service._key_ring_result is key_result
    assert service._executor is executor
    assert service._metrics is metrics
