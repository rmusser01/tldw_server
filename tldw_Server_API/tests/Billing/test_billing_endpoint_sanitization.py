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


class _FailingBillingRepo:
    def __init__(self, db_pool) -> None:  # noqa: ARG002
        pass

    async def list_subscriptions(self, *, status=None, limit=100, offset=0):  # noqa: ARG002
        raise RuntimeError("subscription lookup exploded at /private/subs.db")


@pytest.mark.asyncio
async def test_list_subscriptions_repo_failure_log_is_sanitized(monkeypatch):
    """Repo failures inside list_subscriptions log a sanitized message.

    The org-name warning path is gone: org names are resolved by the repo's
    SQL JOIN (admin-webui perf plan A stage 1), so the remaining inner
    failure point is the repo call itself.
    """
    from tldw_Server_API.app.api.v1.endpoints import billing

    async def _fake_get_db_pool():
        return object()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(billing, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(billing, "AuthnzBillingRepo", _FailingBillingRepo)
    monkeypatch.setattr(billing, "logger", logger_stub)

    with pytest.raises(RuntimeError):
        await billing.list_subscriptions(status=None)

    assert logger_stub.errors == ["list_subscriptions failed"]
    assert "subscription lookup exploded" not in str(logger_stub.errors)
    assert "/private/subs.db" not in str(logger_stub.errors)


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
