import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)


class _BillingRepoWithSubscription:
    def __init__(self, db_pool) -> None:  # noqa: ARG002
        pass

    async def list_all_subscriptions(self):
        return [
            {
                "id": 1,
                "org_id": 123,
                "plan_id": 7,
                "plan_name": "pro",
                "plan_display_name": "Pro",
                "price_usd_monthly": 10,
                "effective_limits": {"llm_tokens_month": 1000},
                "status": "active",
                "created_at": "2026-04-01T00:00:00+00:00",
                "current_period_start": "2026-04-01T00:00:00+00:00",
                "current_period_end": "2026-05-01T00:00:00+00:00",
                "billing_cycle": "monthly",
                "cancel_at_period_end": False,
            }
        ]


class _FailingOrgsRepo:
    def __init__(self, db_pool) -> None:  # noqa: ARG002
        pass

    async def list_organizations(self, limit: int):  # noqa: ARG002
        raise RuntimeError("org lookup exploded at /private/orgs.db")


@pytest.mark.asyncio
async def test_list_subscriptions_org_name_warning_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import billing

    async def _fake_get_db_pool():
        return object()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(billing, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(billing, "AuthnzBillingRepo", _BillingRepoWithSubscription)
    monkeypatch.setattr(billing, "AuthnzOrgsTeamsRepo", _FailingOrgsRepo)
    monkeypatch.setattr(billing, "logger", logger_stub)

    subscriptions = await billing.list_subscriptions(status=None)

    assert len(subscriptions) == 1
    assert subscriptions[0]["org_name"] is None
    assert logger_stub.warnings == ["Failed to resolve org names for subscriptions"]
    assert "org lookup exploded" not in str(logger_stub.warnings)
    assert "/private/orgs.db" not in str(logger_stub.warnings)


@pytest.mark.asyncio
async def test_list_subscriptions_outer_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import billing

    async def _failing_get_db_pool():
        raise RuntimeError("billing backend exploded at /private/billing.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(billing, "get_db_pool", _failing_get_db_pool)
    monkeypatch.setattr(billing, "logger", logger_stub)

    with pytest.raises(RuntimeError):
        await billing.list_subscriptions(status=None)

    assert logger_stub.errors == ["list_subscriptions failed"]
    assert "billing backend exploded" not in str(logger_stub.errors)
    assert "/private/billing.db" not in str(logger_stub.errors)
