"""Scoped BYOK boundaries preserve authority and retryable storage outcomes."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import shared_keys_scoped as routes
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    ProviderKeyTestRequest,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseConcurrencyConflict,
    DatabaseLockError,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit
_BUSY_ERRORS = (
    DatabaseLockError,
    DatabaseConcurrencyConflict,
    ConnectionPoolExhaustedError,
    TimeoutError,
)
_BUSY_DETAIL = "Authentication database is busy. Please retry shortly."


def _busy_failure(failure_type):
    failure = failure_type()
    failure.args = ("secret at /private/shared-byok.db",)
    return failure


async def _invoke(scope, operation, principal):
    endpoint = getattr(routes, f"{operation}_{scope}_shared_key" + ("s" if operation == "list" else ""))
    kwargs = {f"{scope}_id": 42, "principal": principal}
    if operation == "upsert":
        kwargs.update(
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
        )
    elif operation == "test":
        kwargs.update(payload=ProviderKeyTestRequest(provider="openai"), request=SimpleNamespace())
    elif operation == "delete":
        kwargs["provider"] = "oai"
    return await endpoint(**kwargs)


def _allow_validation(monkeypatch):
    monkeypatch.setattr(routes, "_require_byok_enabled", lambda: None)
    monkeypatch.setattr(routes, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(routes, "is_trusted_base_url_request", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "encrypt_byok_payload", lambda _payload: {})
    monkeypatch.setattr(routes, "dumps_envelope", lambda _envelope: "sealed")
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", lambda _envelope: {"api_key": "sk-test"})


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("org", "team"))
@pytest.mark.parametrize("operation", ("upsert", "list", "test", "delete"))
@pytest.mark.parametrize("failure_type", _BUSY_ERRORS)
async def test_scoped_manager_storage_contention_is_retryable(
    monkeypatch, scope, operation, failure_type,
):
    _allow_validation(monkeypatch)
    provider_calls = []

    async def provider_ok(**kwargs):
        provider_calls.append(kwargs["provider"])
        return "model-test"

    class BusyRepo:
        async def upsert_secret(self, **_kwargs):
            raise _busy_failure(failure_type)

        async def list_secrets_for_manager(self, **_kwargs):
            raise _busy_failure(failure_type)

        async def fetch_secret_for_manager(self, **_kwargs):
            raise _busy_failure(failure_type)

        async def delete_secret(self, *_args, **_kwargs):
            raise _busy_failure(failure_type)

    async def get_repo():
        return BusyRepo()

    monkeypatch.setattr(routes, "test_provider_credentials", provider_ok)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", get_repo)
    principal = AuthPrincipal(kind="user", user_id=7, roles=["admin"])
    for retry_after in (7, 19):
        monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", str(retry_after))
        with pytest.raises(HTTPException) as exc:
            await _invoke(scope, operation, principal)
        assert exc.value.status_code == 503
        assert exc.value.detail == _BUSY_DETAIL
        assert exc.value.headers == {"Retry-After": str(retry_after)}
    assert provider_calls == (["openai", "openai"] if operation == "upsert" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("pool", "schema"))
@pytest.mark.parametrize("failure_type", _BUSY_ERRORS)
async def test_shared_repository_acquisition_contention_is_retryable(
    monkeypatch, stage, failure_type,
):
    class Repo:
        async def ensure_tables(self):
            raise _busy_failure(failure_type)

    async def get_pool():
        if stage == "pool":
            raise _busy_failure(failure_type)
        return object()

    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "17")
    monkeypatch.setattr(routes, "get_db_pool", get_pool)
    monkeypatch.setattr(routes, "AuthnzOrgProviderSecretsRepo", lambda _pool: Repo())
    with pytest.raises(HTTPException) as exc:
        await routes._get_shared_byok_repo()
    assert (exc.value.status_code, exc.value.detail, exc.value.headers) == (
        503, _BUSY_DETAIL, {"Retry-After": "17"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("org", "team"))
@pytest.mark.parametrize("failure_type", _BUSY_ERRORS)
async def test_manager_membership_precheck_preserves_busy_outcome(monkeypatch, scope, failure_type):
    async def busy_members(*_args, **_kwargs):
        raise _busy_failure(failure_type)

    monkeypatch.setattr(routes, f"list_{scope}_members", busy_members)
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "11")
    principal = AuthPrincipal(kind="user", user_id=7, roles=["user"])
    with pytest.raises(HTTPException) as exc:
        await getattr(routes, f"_require_{scope}_manager")(principal, 42)
    assert (exc.value.status_code, exc.value.detail, exc.value.headers) == (
        503, _BUSY_DETAIL, {"Retry-After": "11"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("org", "team"))
@pytest.mark.parametrize("operation", ("upsert", "list", "test", "delete"))
@pytest.mark.parametrize("user_id", (None, 0, -1))
async def test_unbacked_claims_are_rejected_before_provider_or_storage(
    monkeypatch, scope, operation, user_id,
):
    _allow_validation(monkeypatch)

    async def forbidden_side_effect(*_args, **_kwargs):
        pytest.fail("unbacked principal reached provider or storage")

    monkeypatch.setattr(routes, "test_provider_credentials", forbidden_side_effect)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", forbidden_side_effect)
    monkeypatch.setattr(routes, f"list_{scope}_members", forbidden_side_effect)
    principal = AuthPrincipal(kind="service", user_id=user_id, roles=["admin"], permissions=["*"])
    with pytest.raises(HTTPException) as exc:
        await _invoke(scope, operation, principal)
    assert exc.value.status_code == 403
    assert exc.value.detail == f"{scope.title()} manager role required"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("org", "team"))
@pytest.mark.parametrize("failure_type", _BUSY_ERRORS)
async def test_manager_test_last_used_contention_is_retryable(monkeypatch, scope, failure_type):
    _allow_validation(monkeypatch)

    class Repo:
        async def fetch_secret_for_manager(self, **_kwargs):
            return {"encrypted_blob": "sealed"}

        async def fetch_secret(self, *_args):
            return {"encrypted_blob": "sealed"}

        async def touch_last_used(self, *_args):
            raise _busy_failure(failure_type)

    async def get_repo():
        return Repo()

    async def provider_ok(**_kwargs):
        return "model-test"

    monkeypatch.setattr(routes, "_get_shared_byok_repo", get_repo)
    monkeypatch.setattr(routes, "test_provider_credentials", provider_ok)
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "23")
    with pytest.raises(HTTPException) as exc:
        await _invoke(scope, "test", AuthPrincipal(kind="user", user_id=7, roles=["admin"]))
    assert (exc.value.status_code, exc.value.detail, exc.value.headers) == (
        503, _BUSY_DETAIL, {"Retry-After": "23"},
    )
