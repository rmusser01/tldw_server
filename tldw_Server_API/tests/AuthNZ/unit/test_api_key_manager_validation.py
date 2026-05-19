import pytest


pytestmark = pytest.mark.unit


class _StubRepo:
    def __init__(self, key_info: dict):
        self._key_info = dict(key_info)

    async def fetch_active_by_hash_candidates(self, hash_candidates):  # noqa: ARG002
        return dict(self._key_info)


class _StubKeyIdRepo:
    def __init__(self, key_info: dict):
        self._key_info = dict(key_info)
        self.calls: list[str] = []

    async def fetch_active_by_key_id(self, key_id: str):
        self.calls.append(key_id)
        return dict(self._key_info)


async def _noop_async(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    return None


@pytest.mark.asyncio
async def test_validate_api_key_parses_zulu_expires_at(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = "tldw_test_zulu_expiry"

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-for-zulu-expiry-1234567890")
    manager._initialized = True

    primary_hash = manager.hash_candidates(api_key)[0]
    repo = _StubRepo(
        {
            "id": 1,
            "user_id": 1,
            "key_hash": primary_hash,
            "expires_at": "2099-01-01T00:00:00Z",
            "allowed_ips": None,
            "scope": "read",
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    result = await manager.validate_api_key(api_key)

    assert result is not None
    assert result["id"] == 1
    assert "key_hash" not in result


@pytest.mark.asyncio
async def test_validate_api_key_kdf_path(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
        format_api_key,
        kdf_hash_api_key,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = format_api_key("deadbeefcafe", "secret-part")

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-kdf-1234567890-abcdef")
    manager._initialized = True

    encoded = kdf_hash_api_key(api_key, salt=b"fixed-salt-123456")
    repo = _StubKeyIdRepo(
        {
            "id": 10,
            "user_id": 7,
            "key_hash": encoded,
            "expires_at": None,
            "allowed_ips": None,
            "scope": "read",
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    result = await manager.validate_api_key(api_key)

    assert result is not None
    assert result["id"] == 10
    assert repo.calls == ["deadbeefcafe"]


@pytest.mark.asyncio
async def test_validate_api_key_denies_missing_scope_for_read(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = "tldw_test_missing_scope"

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-for-missing-scope-1234567890")
    manager._initialized = True

    primary_hash = manager.hash_candidates(api_key)[0]
    repo = _StubRepo(
        {
            "id": 2,
            "user_id": 1,
            "key_hash": primary_hash,
            "expires_at": None,
            "allowed_ips": None,
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    result = await manager.validate_api_key(api_key, required_scope="read")

    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_denies_on_malformed_expires_at(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = "tldw_test_malformed_expiry"

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-for-malformed-expiry-1234567890")
    manager._initialized = True

    primary_hash = manager.hash_candidates(api_key)[0]
    repo = _StubRepo(
        {
            "id": 3,
            "user_id": 1,
            "key_hash": primary_hash,
            "expires_at": "not-a-timestamp",
            "allowed_ips": None,
            "scope": "read",
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    result = await manager.validate_api_key(api_key)

    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_denies_on_invalid_allowed_ips_json(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = "tldw_test_bad_allowed_ips_json"

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-for-bad-allowed-ips-json-1234567890")
    manager._initialized = True

    primary_hash = manager.hash_candidates(api_key)[0]
    repo = _StubRepo(
        {
            "id": 4,
            "user_id": 1,
            "key_hash": primary_hash,
            "expires_at": None,
            "allowed_ips": "not-json",
            "scope": "read",
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    result = await manager.validate_api_key(api_key, ip_address="1.2.3.4")

    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_denies_on_invalid_allowed_ips_type(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = "tldw_test_bad_allowed_ips_type"

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-for-bad-allowed-ips-type-1234567890")
    manager._initialized = True

    primary_hash = manager.hash_candidates(api_key)[0]
    repo = _StubRepo(
        {
            "id": 5,
            "user_id": 1,
            "key_hash": primary_hash,
            "expires_at": None,
            "allowed_ips": 123,
            "scope": "read",
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    result = await manager.validate_api_key(api_key, ip_address="1.2.3.4")

    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_enforces_allowed_ips(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import Settings

    api_key = "tldw_test_allowed_ips_enforced"

    manager = APIKeyManager()
    manager.settings = Settings(AUTH_MODE="multi_user", JWT_SECRET_KEY="test-secret-for-allowed-ips-enforced-1234567890")
    manager._initialized = True

    primary_hash = manager.hash_candidates(api_key)[0]
    repo = _StubRepo(
        {
            "id": 6,
            "user_id": 1,
            "key_hash": primary_hash,
            "expires_at": None,
            "allowed_ips": '["1.2.3.4", " 5.6.7.8 "]',
            "scope": "read",
        }
    )

    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)
    manager._update_usage = _noop_async  # type: ignore[method-assign]

    denied = await manager.validate_api_key(api_key, ip_address="9.9.9.9")

    assert denied is None

    allowed = await manager.validate_api_key(api_key, ip_address="5.6.7.8")

    assert allowed is not None
    assert allowed["id"] == 6
    assert "key_hash" not in allowed


class _UsageRepo:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str | None]] = []

    async def increment_usage(self, key_id: int, ip_address: str | None = None) -> None:
        self.calls.append((key_id, ip_address))


@pytest.mark.asyncio
async def test_update_usage_throttles_hot_key_writes(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    monkeypatch.setenv("API_KEY_USAGE_TOUCH_INTERVAL_SECONDS", "30")
    manager = APIKeyManager()
    repo = _UsageRepo()
    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)

    await manager._update_usage(99, "203.0.113.10")
    await manager._update_usage(99, "203.0.113.10")

    assert repo.calls == [(99, "203.0.113.10")]


@pytest.mark.asyncio
async def test_update_usage_no_throttle_when_interval_disabled(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    monkeypatch.setenv("API_KEY_USAGE_TOUCH_INTERVAL_SECONDS", "0")
    manager = APIKeyManager()
    repo = _UsageRepo()
    monkeypatch.setattr(manager, "_get_repo", lambda: repo, raising=True)

    await manager._update_usage(99, "203.0.113.10")
    await manager._update_usage(99, "203.0.113.10")

    assert repo.calls == [
        (99, "203.0.113.10"),
        (99, "203.0.113.10"),
    ]
