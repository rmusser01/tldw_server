from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import jwt
from jwt.utils import base64url_encode

from tldw_Server_API.app.core.AuthNZ.federation import oidc_service as oidc_module
from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService


pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _install_fetch_json_stub(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[tuple[str, str], dict[str, Any]],
) -> None:
    async def _fake_afetch_json(*, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        return dict(responses[(method.upper(), url)])

    monkeypatch.setattr(oidc_module, "afetch_json", _fake_afetch_json, raising=False)


async def _async_value(value):  # noqa: ANN001, ANN201
    return value


class _FakeOrgSecretRepo:
    def __init__(self, _pool: object, row: dict | None, touched: list[tuple[str, int, str]] | None = None) -> None:
        self._row = row
        self._touched = touched if touched is not None else []

    async def ensure_tables(self) -> None:
        return None

    async def fetch_secret(self, scope_type: str, scope_id: int, provider: str, *, include_revoked: bool = False):  # noqa: ANN001
        assert include_revoked is False
        if self._row is None:
            return None
        assert scope_type == "org"
        assert scope_id == 123
        assert provider == "idp:corp-oidc"
        return dict(self._row)

    async def touch_last_used(self, scope_type: str, scope_id: int, provider: str, used_at) -> None:  # noqa: ANN001
        assert used_at is not None
        self._touched.append((scope_type, scope_id, provider))


class _MissingOrgSecretRepo(_FakeOrgSecretRepo):
    def __init__(self, _pool: object) -> None:
        super().__init__(_pool, None, [])


def _rsa_signing_material() -> tuple[str, dict[str, str]]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_numbers = private_key.public_key().public_numbers()
    return private_pem, {
        "kty": "RSA",
        "kid": "oidc-key-1",
        "alg": "RS256",
        "use": "sig",
        "n": base64url_encode(
            public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, "big")
        ).decode("utf-8"),
        "e": base64url_encode(
            public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, "big")
        ).decode("utf-8"),
    }


@pytest.mark.asyncio
async def test_build_authorization_request_uses_discovery_authorization_endpoint_when_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_url = "https://issuer.example.com/.well-known/openid-configuration"
    _install_fetch_json_stub(
        monkeypatch,
        {
            ("GET", discovery_url): {
                "issuer": "https://issuer.example.com",
                "authorization_endpoint": "https://issuer.example.com/oauth2/authorize",
                "token_endpoint": "https://issuer.example.com/oauth2/token",
                "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
            }
        },
    )

    auth_request = await OIDCFederationService().build_authorization_request(
        provider={
            "issuer": "https://issuer.example.com",
            "discovery_url": discovery_url,
            "client_id": "client-123",
        },
        redirect_uri="http://testserver/callback",
    )

    parsed = urlparse(auth_request["auth_url"])
    query = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://issuer.example.com/oauth2/authorize"
    assert query["client_id"] == ["client-123"]
    assert query["redirect_uri"] == ["http://testserver/callback"]
    assert query["response_type"] == ["code"]


@pytest.mark.parametrize(
    "authorization_url",
    [
        "http://issuer.example.com/oauth2/authorize",
        "javascript:alert(1)",
        "https://",
        "https://issuer.example.com/oauth2/authorize#fragment",
    ],
)
@pytest.mark.asyncio
async def test_build_authorization_request_rejects_unsafe_authorization_url(
    authorization_url: str,
) -> None:
    with pytest.raises(ValueError, match="authorization_url"):
        await OIDCFederationService().build_authorization_request(
            provider={
                "issuer": "https://issuer.example.com",
                "authorization_url": authorization_url,
                "token_url": "https://issuer.example.com/oauth2/token",
                "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
                "client_id": "client-123",
            },
            redirect_uri="http://testserver/callback",
        )


@pytest.mark.asyncio
async def test_build_authorization_request_rejects_unsafe_discovered_authorization_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_url = "https://issuer.example.com/.well-known/openid-configuration"
    _install_fetch_json_stub(
        monkeypatch,
        {
            ("GET", discovery_url): {
                "issuer": "https://issuer.example.com",
                "authorization_endpoint": "http://issuer.example.com/oauth2/authorize",
                "token_endpoint": "https://issuer.example.com/oauth2/token",
                "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
            }
        },
    )

    with pytest.raises(ValueError, match="authorization_url"):
        await OIDCFederationService().build_authorization_request(
            provider={
                "issuer": "https://issuer.example.com",
                "discovery_url": discovery_url,
                "client_id": "client-123",
            },
            redirect_uri="http://testserver/callback",
        )


@pytest.mark.asyncio
async def test_build_authorization_request_replaces_conflicting_authorization_query_params() -> None:
    auth_request = await OIDCFederationService().build_authorization_request(
        provider={
            "issuer": "https://issuer.example.com",
            "authorization_url": (
                "https://issuer.example.com/oauth2/authorize?"
                "prompt=login&client_id=stale-client&redirect_uri=https%3A%2F%2Fstale.example%2Fcallback&state=stale"
            ),
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
        },
        redirect_uri="http://testserver/callback",
    )

    query = parse_qs(urlparse(auth_request["auth_url"]).query)

    assert query["prompt"] == ["login"]
    assert query["client_id"] == ["client-123"]
    assert query["redirect_uri"] == ["http://testserver/callback"]
    assert query["state"] == [auth_request["state"]]
    assert all(len(query[key]) == 1 for key in ("client_id", "redirect_uri", "state"))


@pytest.mark.asyncio
async def test_oidc_json_fetches_use_response_size_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, jwk = _rsa_signing_material()
    discovery_url = "https://issuer.example.com/.well-known/openid-configuration"
    token_url = "https://issuer.example.com/oauth2/token"
    jwks_url = "https://issuer.example.com/.well-known/jwks.json"
    id_token = jwt.encode(
        {
            "sub": "external-user-capped",
            "email": "capped@example.com",
            "iss": "https://issuer.example.com",
            "aud": "client-123",
            "nonce": "nonce-capped",
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "oidc-key-1"},
    )
    seen_max_bytes: dict[tuple[str, str], int | None] = {}

    async def _fake_afetch_json(*, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        key = (method.upper(), url)
        seen_max_bytes[key] = kwargs.get("max_bytes")
        if key == ("GET", discovery_url):
            return {
                "issuer": "https://issuer.example.com",
                "authorization_endpoint": "https://issuer.example.com/oauth2/authorize",
                "token_endpoint": token_url,
                "jwks_uri": jwks_url,
            }
        if key == ("POST", token_url):
            return {"id_token": id_token}
        if key == ("GET", jwks_url):
            return {"keys": [jwk]}
        raise AssertionError(f"Unexpected OIDC fetch: {method} {url}")

    monkeypatch.setattr(oidc_module, "afetch_json", _fake_afetch_json, raising=False)

    await OIDCFederationService().build_authorization_request(
        provider={
            "issuer": "https://issuer.example.com",
            "discovery_url": discovery_url,
            "client_id": "client-123",
        },
        redirect_uri="http://testserver/callback",
    )
    await OIDCFederationService().exchange_authorization_code(
        provider={
            "issuer": "https://issuer.example.com",
            "token_url": token_url,
            "jwks_url": jwks_url,
            "client_id": "client-123",
        },
        code="code-capped",
        redirect_uri="http://testserver/callback",
        code_verifier="verifier-capped",
        nonce="nonce-capped",
    )

    assert seen_max_bytes[("GET", discovery_url)] == oidc_module.OIDC_DISCOVERY_MAX_BYTES
    assert seen_max_bytes[("POST", token_url)] == oidc_module.OIDC_TOKEN_RESPONSE_MAX_BYTES
    assert seen_max_bytes[("GET", jwks_url)] == oidc_module.OIDC_JWKS_MAX_BYTES


@pytest.mark.asyncio
async def test_exchange_authorization_code_uses_discovery_and_jwks_to_verify_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, jwk = _rsa_signing_material()
    discovery_url = "https://issuer.example.com/.well-known/openid-configuration"
    token_url = "https://issuer.example.com/oauth2/token"
    jwks_url = "https://issuer.example.com/.well-known/jwks.json"
    id_token = jwt.encode(
        {
            "sub": "external-user-123",
            "email": "alice@example.com",
            "iss": "https://issuer.example.com",
            "aud": "client-123",
            "nonce": "nonce-123",
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "oidc-key-1"},
    )

    _install_fetch_json_stub(
        monkeypatch,
        {
            ("GET", discovery_url): {
                "issuer": "https://issuer.example.com",
                "authorization_endpoint": "https://issuer.example.com/oauth2/authorize",
                "token_endpoint": token_url,
                "jwks_uri": jwks_url,
            },
            ("POST", token_url): {"id_token": id_token},
            ("GET", jwks_url): {"keys": [jwk]},
        },
    )

    claims = await OIDCFederationService().exchange_authorization_code(
        provider={
            "issuer": "https://issuer.example.com",
            "discovery_url": discovery_url,
            "client_id": "client-123",
        },
        code="code-123",
        redirect_uri="http://testserver/callback",
        code_verifier="verifier-123",
        nonce="nonce-123",
    )

    assert claims["sub"] == "external-user-123"
    assert claims["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_inspect_provider_configuration_rejects_missing_env_client_secret_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CORP_OIDC_CLIENT_SECRET", raising=False)

    with pytest.raises(
        ValueError,
        match="OIDC client_secret_ref environment variable is not set: CORP_OIDC_CLIENT_SECRET",
    ):
        await OIDCFederationService().inspect_provider_configuration(
            provider={
                "issuer": "https://issuer.example.com",
                "authorization_url": "https://issuer.example.com/oauth2/authorize",
                "token_url": "https://issuer.example.com/oauth2/token",
                "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
                "client_id": "client-123",
                "client_secret_ref": "env:CORP_OIDC_CLIENT_SECRET",
            }
        )


@pytest.mark.asyncio
async def test_build_authorization_request_ignores_discovery_when_runtime_fields_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_fetch(*, method: str, url: str, **kwargs) -> dict:  # noqa: ANN003
        raise AssertionError(f"Discovery should not be fetched for pinned provider config: {method} {url}")

    monkeypatch.setattr(oidc_module, "afetch_json", _unexpected_fetch, raising=False)

    auth_request = await OIDCFederationService().build_authorization_request(
        provider={
            "issuer": "https://issuer.example.com",
            "discovery_url": "https://issuer.example.com/.well-known/openid-configuration",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
        },
        redirect_uri="http://testserver/callback",
    )

    parsed = urlparse(auth_request["auth_url"])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://issuer.example.com/oauth2/authorize"


@pytest.mark.asyncio
async def test_exchange_authorization_code_resolves_env_client_secret_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, jwk = _rsa_signing_material()
    token_url = "https://issuer.example.com/oauth2/token"
    jwks_url = "https://issuer.example.com/.well-known/jwks.json"
    id_token = jwt.encode(
        {
            "sub": "external-user-789",
            "email": "bob@example.com",
            "iss": "https://issuer.example.com",
            "aud": "client-123",
            "nonce": "nonce-789",
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "oidc-key-1"},
    )
    seen_form: dict[str, str] = {}

    async def _fake_afetch_json(*, method: str, url: str, **kwargs) -> dict:  # noqa: ANN003
        if method.upper() == "POST" and url == token_url:
            seen_form.update(kwargs["data"])
            return {"id_token": id_token}
        if method.upper() == "GET" and url == jwks_url:
            return {"keys": [jwk]}
        raise AssertionError(f"Unexpected OIDC fetch: {method} {url}")

    monkeypatch.setenv("CORP_OIDC_CLIENT_SECRET", "super-secret")
    monkeypatch.setattr(oidc_module, "afetch_json", _fake_afetch_json, raising=False)

    claims = await OIDCFederationService().exchange_authorization_code(
        provider={
            "issuer": "https://issuer.example.com",
            "token_url": token_url,
            "jwks_url": jwks_url,
            "client_id": "client-123",
            "client_secret_ref": "env:CORP_OIDC_CLIENT_SECRET",
        },
        code="code-789",
        redirect_uri="http://testserver/callback",
        code_verifier="verifier-789",
        nonce="nonce-789",
    )

    assert seen_form["client_secret"] == "super-secret"
    assert claims["sub"] == "external-user-789"
    assert claims["email"] == "bob@example.com"


@pytest.mark.asyncio
async def test_exchange_authorization_code_resolves_byok_org_client_secret_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_pem, jwk = _rsa_signing_material()
    token_url = "https://issuer.example.com/oauth2/token"
    jwks_url = "https://issuer.example.com/.well-known/jwks.json"
    id_token = jwt.encode(
        {
            "sub": "external-user-999",
            "email": "shared@example.com",
            "iss": "https://issuer.example.com",
            "aud": "client-123",
            "nonce": "nonce-999",
        },
        private_pem,
        algorithm="RS256",
        headers={"kid": "oidc-key-1"},
    )
    seen_form: dict[str, str] = {}
    touched: list[tuple[str, int, str]] = []

    async def _fake_afetch_json(*, method: str, url: str, **kwargs) -> dict:  # noqa: ANN003
        if method.upper() == "POST" and url == token_url:
            seen_form.update(kwargs["data"])
            return {"id_token": id_token}
        if method.upper() == "GET" and url == jwks_url:
            return {"keys": [jwk]}
        raise AssertionError(f"Unexpected OIDC fetch: {method} {url}")

    monkeypatch.setattr(oidc_module, "afetch_json", _fake_afetch_json, raising=False)
    monkeypatch.setattr(oidc_module, "get_db_pool", lambda: _async_value(object()))
    monkeypatch.setattr(
        oidc_module,
        "AuthnzOrgProviderSecretsRepo",
        lambda pool: _FakeOrgSecretRepo(pool, {"encrypted_blob": "blob-123"}, touched),
    )
    monkeypatch.setattr(oidc_module, "loads_envelope", lambda value: {"blob": value}, raising=False)
    monkeypatch.setattr(
        oidc_module,
        "decrypt_byok_payload",
        lambda envelope: {"api_key": f"resolved::{envelope['blob']}"},
        raising=False,
    )

    claims = await OIDCFederationService().exchange_authorization_code(
        provider={
            "issuer": "https://issuer.example.com",
            "token_url": token_url,
            "jwks_url": jwks_url,
            "client_id": "client-123",
            "client_secret_ref": "byok-org:123:idp:corp-oidc",
        },
        code="code-999",
        redirect_uri="http://testserver/callback",
        code_verifier="verifier-999",
        nonce="nonce-999",
    )

    assert seen_form["client_secret"] == "resolved::blob-123"
    assert touched == [("org", 123, "idp:corp-oidc")]
    assert claims["sub"] == "external-user-999"


@pytest.mark.asyncio
async def test_inspect_provider_configuration_does_not_touch_byok_secret_last_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[tuple[str, int, str]] = []

    monkeypatch.setattr(oidc_module, "get_db_pool", lambda: _async_value(object()))
    monkeypatch.setattr(
        oidc_module,
        "AuthnzOrgProviderSecretsRepo",
        lambda pool: _FakeOrgSecretRepo(pool, {"encrypted_blob": "blob-123"}, touched),
    )
    monkeypatch.setattr(oidc_module, "loads_envelope", lambda value: {"blob": value}, raising=False)
    monkeypatch.setattr(
        oidc_module,
        "decrypt_byok_payload",
        lambda envelope: {"api_key": f"resolved::{envelope['blob']}"},
        raising=False,
    )

    result = await OIDCFederationService().inspect_provider_configuration(
        provider={
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "client_secret_ref": "byok-org:123:idp:corp-oidc",
        }
    )

    assert result["ok"] is True
    assert touched == []


@pytest.mark.asyncio
async def test_exchange_authorization_code_rejects_untrusted_token_header_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _private_pem, jwk = _rsa_signing_material()
    token_url = "https://issuer.example.com/oauth2/token"
    jwks_url = "https://issuer.example.com/.well-known/jwks.json"
    id_token = jwt.encode(
        {
            "sub": "external-user-unsafe",
            "email": "mallory@example.com",
            "iss": "https://issuer.example.com",
            "aud": "client-123",
            "nonce": "nonce-unsafe",
        },
        "shared-secret",
        algorithm="HS256",
        headers={"kid": "oidc-key-1"},
    )

    async def _fake_afetch_json(*, method: str, url: str, **kwargs) -> dict:  # noqa: ANN003
        if method.upper() == "POST" and url == token_url:
            return {"id_token": id_token}
        if method.upper() == "GET" and url == jwks_url:
            return {"keys": [jwk]}
        raise AssertionError(f"Unexpected OIDC fetch: {method} {url}")

    monkeypatch.setattr(oidc_module, "afetch_json", _fake_afetch_json, raising=False)

    with pytest.raises(ValueError, match="unsupported signing algorithm"):
        await OIDCFederationService().exchange_authorization_code(
            provider={
                "issuer": "https://issuer.example.com",
                "token_url": token_url,
                "jwks_url": jwks_url,
                "client_id": "client-123",
            },
            code="code-unsafe",
            redirect_uri="http://testserver/callback",
            code_verifier="verifier-unsafe",
            nonce="nonce-unsafe",
        )


@pytest.mark.asyncio
async def test_inspect_provider_configuration_rejects_missing_byok_org_client_secret_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oidc_module, "get_db_pool", lambda: _async_value(object()))
    monkeypatch.setattr(oidc_module, "AuthnzOrgProviderSecretsRepo", _MissingOrgSecretRepo)

    with pytest.raises(
        ValueError,
        match="OIDC client_secret_ref BYOK secret is not available",
    ):
        await OIDCFederationService().inspect_provider_configuration(
            provider={
                "issuer": "https://issuer.example.com",
                "authorization_url": "https://issuer.example.com/oauth2/authorize",
                "token_url": "https://issuer.example.com/oauth2/token",
                "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
                "client_id": "client-123",
                "client_secret_ref": "byok-org:123:idp:corp-oidc",
            }
        )
