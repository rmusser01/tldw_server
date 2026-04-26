from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import shared_keys_scoped as routes
from tldw_Server_API.app.api.v1.schemas.user_keys import (
    ProviderKeyTestRequest,
    UserProviderKeyUpsertRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


pytestmark = pytest.mark.unit


class _SharedRepo:
    async def fetch_secret(self, *_args):
        return {"encrypted_blob": "encrypted-shared-provider-secret"}

    async def touch_last_used(self, *_args):
        return None


def _principal() -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=7, roles=["admin"], permissions=["*"], is_admin=True)


async def _allow_scope(*_args, **_kwargs) -> None:
    return None


def _stored_payload(*_args, **_kwargs):
    return {"api_key": "sk-test", "credential_fields": {"base_url": "https://api.example.test"}}


def _install_common_patches(monkeypatch) -> None:
    monkeypatch.setattr(routes, "_require_byok_enabled", lambda: None)
    monkeypatch.setattr(routes, "_require_org_manager", _allow_scope)
    monkeypatch.setattr(routes, "_require_team_manager", _allow_scope)
    monkeypatch.setattr(routes, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(routes, "is_trusted_base_url_request", lambda *_args, **_kwargs: True)


async def _repo() -> _SharedRepo:
    return _SharedRepo()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-shared-credentials.json"),
        ("team", "/private/team-shared-credentials.json"),
    ],
)
async def test_scoped_shared_key_upsert_sanitizes_credential_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError(f"shared credential token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(
                provider="openai",
                api_key="sk-test",
                credential_fields={"base_url": "https://api.example.test"},
            ),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert raw_token not in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-provider-test.json"),
        ("team", "/private/team-provider-test.json"),
    ],
)
async def test_scoped_shared_key_upsert_sanitizes_provider_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError(f"shared provider token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    endpoint = routes.upsert_org_shared_key if scope == "org" else routes.upsert_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=UserProviderKeyUpsertRequest(provider="openai", api_key="sk-test"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert raw_token not in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-stored-credentials.json"),
        ("team", "/private/team-stored-credentials.json"),
    ],
)
async def test_scoped_shared_key_test_sanitizes_stored_credential_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    def fail_validate(*_args, **_kwargs):
        raise ValueError(f"stored credential token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(routes, "validate_credential_fields", fail_validate)

    endpoint = routes.test_org_shared_key if scope == "org" else routes.test_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=ProviderKeyTestRequest(provider="openai"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid provider credential fields"
    assert raw_token not in exc_info.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "raw_token"),
    [
        ("org", "/private/org-stored-provider.json"),
        ("team", "/private/team-stored-provider.json"),
    ],
)
async def test_scoped_shared_key_test_sanitizes_provider_validation(
    monkeypatch,
    scope: str,
    raw_token: str,
) -> None:
    async def fail_provider_test(**_kwargs):
        raise ValueError(f"stored provider token at {raw_token}")

    _install_common_patches(monkeypatch)
    monkeypatch.setattr(routes, "_get_shared_byok_repo", _repo)
    monkeypatch.setattr(routes, "loads_envelope", lambda _blob: {})
    monkeypatch.setattr(routes, "decrypt_byok_payload", _stored_payload)
    monkeypatch.setattr(routes, "validate_credential_fields", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(routes, "test_provider_credentials", fail_provider_test)

    endpoint = routes.test_org_shared_key if scope == "org" else routes.test_team_shared_key
    kwargs = {"org_id": 42} if scope == "org" else {"team_id": 42}

    with pytest.raises(HTTPException) as exc_info:
        await endpoint(
            **kwargs,
            payload=ProviderKeyTestRequest(provider="openai"),
            request=SimpleNamespace(),
            principal=_principal(),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Provider credential validation failed"
    assert raw_token not in exc_info.value.detail
