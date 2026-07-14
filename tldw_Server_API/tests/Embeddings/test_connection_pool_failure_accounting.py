import pytest

from tldw_Server_API.app.core.Embeddings import connection_pool as pool_mod
from tldw_Server_API.app.core.Embeddings.connection_pool import ConnectionPool
from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError


class _DummyResponse:
    def __init__(self, status_code: int, text: str = "", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return {"ok": True}

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_connection_pool_propagates_sensitive_observability(monkeypatch):
    pool = ConnectionPool(provider="test-provider", retry_attempts=1)
    captured = []

    async def _fake_afetch(**kwargs):
        captured.append(kwargs)
        return _DummyResponse(status_code=200)

    monkeypatch.setattr(pool_mod, "afetch", _fake_afetch)

    assert await pool.request(
        "POST",
        "https://runtime.private/secret/embeddings",
        sensitive_observability=True,
    ) == {"ok": True}
    assert captured[0]["sensitive_observability"] is True


@pytest.mark.asyncio
async def test_connection_pool_http_error_counts_single_failure(monkeypatch):
    pool = ConnectionPool(provider="test-provider", retry_attempts=1)

    async def _fake_afetch(**_kwargs):
        return _DummyResponse(status_code=503, text="service unavailable")

    monkeypatch.setattr(pool_mod, "afetch", _fake_afetch)

    with pytest.raises(NetworkError):
        await pool.request("POST", "https://example.invalid/embeddings")

    stats = pool.get_stats()
    assert stats["total_requests"] == 1
    assert stats["failed_requests"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        NetworkError("connection reset"),
        RetryExhaustedError("retries exhausted"),
    ],
)
async def test_connection_pool_transport_errors_count_single_failure(monkeypatch, exc):
    pool = ConnectionPool(provider="test-provider", retry_attempts=1)

    async def _fake_afetch(**_kwargs):
        raise exc

    monkeypatch.setattr(pool_mod, "afetch", _fake_afetch)

    with pytest.raises(type(exc)):
        await pool.request("POST", "https://example.invalid/embeddings")

    stats = pool.get_stats()
    assert stats["failed_requests"] == 1


@pytest.mark.asyncio
async def test_credential_scoped_failures_do_not_open_shared_breaker(monkeypatch):
    pool = ConnectionPool(provider="openai", retry_attempts=1)
    calls = []

    async def _fake_afetch(**kwargs):
        authorization = (kwargs.get("headers") or {}).get("Authorization")
        calls.append(authorization)
        if authorization == "Bearer invalid-key":
            return _DummyResponse(status_code=401, text="invalid credential")
        return _DummyResponse(status_code=200)

    monkeypatch.setattr(pool_mod, "afetch", _fake_afetch)

    for _ in range(5):
        with pytest.raises(NetworkError):
            await pool.request(
                "POST",
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": "Bearer invalid-key"},
                bypass_circuit_breaker=True,
            )

    result = await pool.request(
        "POST",
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": "Bearer valid-key"},
        bypass_circuit_breaker=True,
    )

    assert result == {"ok": True}
    assert calls == ["Bearer invalid-key"] * 5 + ["Bearer valid-key"]
