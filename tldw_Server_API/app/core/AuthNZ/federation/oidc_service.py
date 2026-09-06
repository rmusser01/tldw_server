from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tldw_Server_API.app.core.Utils import jwt_compat as jwt
from tldw_Server_API.app.core.Utils.jwt_compat import JWTError
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    loads_envelope,
)
from tldw_Server_API.app.core.http_client import afetch_json


def _coerce_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _oauth_code_challenge_s256(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _coerce_string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return {stripped} if stripped else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


_ASYMMETRIC_JWT_ALGS_BY_KTY: dict[str, set[str]] = {
    "EC": {"ES256", "ES384", "ES512"},
    "OKP": {"EdDSA"},
    "RSA": {"PS256", "PS384", "PS512", "RS256", "RS384", "RS512"},
}
OIDC_DISCOVERY_MAX_BYTES = 128 * 1024
OIDC_TOKEN_RESPONSE_MAX_BYTES = 64 * 1024
OIDC_JWKS_MAX_BYTES = 256 * 1024


def _validate_oidc_provider_url(field_name: str, value: Any) -> str | None:
    """Validate and normalize trusted OIDC provider endpoint URLs."""
    text = _coerce_nonempty_string(value)
    if not text:
        return None
    try:
        parsed = urlsplit(text)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"OIDC provider {field_name} is invalid") from exc

    if parsed.scheme.lower() != "https":
        raise ValueError(f"OIDC provider {field_name} must use https")
    if not parsed.hostname:
        raise ValueError(f"OIDC provider {field_name} must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError(f"OIDC provider {field_name} must not include credentials")
    if parsed.fragment:
        raise ValueError(f"OIDC provider {field_name} must not include a URL fragment")

    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _append_query_params(url: str, params: dict[str, str]) -> str:
    """Append query parameters while replacing any existing keys being set."""
    parsed = urlsplit(url)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    query_items = [item for item in query_items if item[0] not in params]
    query_items.extend(params.items())
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            "",
        )
    )


def _parse_byok_secret_ref(secret_ref: str) -> tuple[str, int, str] | None:
    lowered = secret_ref.lower()
    scope_type: str | None = None
    if lowered.startswith("byok-user:"):
        scope_type = "user"
    elif lowered.startswith("byok-org:"):
        scope_type = "org"
    elif lowered.startswith("byok-team:"):
        scope_type = "team"
    if scope_type is None:
        return None

    parts = secret_ref.split(":", 2)
    if len(parts) != 3:
        raise ValueError("OIDC client_secret_ref BYOK reference is invalid")

    scope_id_text = parts[1].strip()
    provider = parts[2].strip()
    if not scope_id_text or not scope_id_text.isdigit() or int(scope_id_text) <= 0:
        raise ValueError("OIDC client_secret_ref BYOK scope id is invalid")
    if not provider:
        raise ValueError("OIDC client_secret_ref BYOK provider key is invalid")
    return scope_type, int(scope_id_text), provider


async def _resolve_byok_secret_ref(secret_ref: str) -> str:
    return await _resolve_byok_secret_ref_with_options(
        secret_ref,
        touch_last_used=True,
    )


async def _resolve_byok_secret_ref_with_options(
    secret_ref: str,
    *,
    touch_last_used: bool,
) -> str:
    parsed = _parse_byok_secret_ref(secret_ref)
    if parsed is None:
        raise ValueError("OIDC client_secret_ref BYOK reference is invalid")

    scope_type, scope_id, provider = parsed
    pool = await get_db_pool()

    if scope_type == "user":
        repo = AuthnzUserProviderSecretsRepo(pool)
        await repo.ensure_tables()
        secret_row = await repo.fetch_secret_for_user(scope_id, provider)
    else:
        repo = AuthnzOrgProviderSecretsRepo(pool)
        await repo.ensure_tables()
        secret_row = await repo.fetch_secret(scope_type, scope_id, provider)

    encrypted_blob = _coerce_nonempty_string(secret_row.get("encrypted_blob")) if isinstance(secret_row, dict) else None
    if not encrypted_blob:
        raise ValueError("OIDC client_secret_ref BYOK secret is not available")

    try:
        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
    except Exception as exc:
        raise ValueError("OIDC client_secret_ref BYOK secret could not be decrypted") from exc

    client_secret = _coerce_nonempty_string(payload.get("api_key")) if isinstance(payload, dict) else None
    if not client_secret:
        raise ValueError("OIDC client_secret_ref BYOK secret is missing api_key")

    if touch_last_used:
        used_at = datetime.now(timezone.utc)
        try:
            if scope_type == "user":
                await repo.touch_last_used(scope_id, provider, used_at)
            else:
                await repo.touch_last_used(scope_type, scope_id, provider, used_at)
        except Exception as exc:
            logger.debug("OIDC BYOK secret touch_last_used failed for {}:{}:{}: {}", scope_type, scope_id, provider, exc)
    return client_secret


async def _resolve_client_secret_ref(
    value: Any,
    *,
    touch_last_used: bool = True,
) -> str | None:
    secret_ref = _coerce_nonempty_string(value)
    if not secret_ref:
        return None
    if _parse_byok_secret_ref(secret_ref) is not None:
        return await _resolve_byok_secret_ref_with_options(
            secret_ref,
            touch_last_used=touch_last_used,
        )
    if not secret_ref.lower().startswith("env:"):
        return secret_ref

    env_name = secret_ref[4:].strip()
    if not env_name:
        raise ValueError("OIDC client_secret_ref env reference is invalid")
    env_value = os.getenv(env_name, "").strip()
    if not env_value:
        raise ValueError(
            f"OIDC client_secret_ref environment variable is not set: {env_name}"
        )
    return env_value


@dataclass
class OIDCFederationService:
    """Construct OIDC authorization requests for trusted identity providers."""

    default_scopes: tuple[str, ...] = ("openid", "email", "profile")
    state_ttl_seconds: int = 600

    async def _fetch_discovery_document(self, provider: dict[str, Any]) -> dict[str, Any]:
        discovery_url = _validate_oidc_provider_url("discovery_url", provider.get("discovery_url"))
        if not discovery_url:
            return {}
        discovery = await afetch_json(
            method="GET",
            url=discovery_url,
            max_bytes=OIDC_DISCOVERY_MAX_BYTES,
        )
        if not isinstance(discovery, dict):
            raise ValueError("OIDC discovery response is invalid")
        explicit_issuer = _coerce_nonempty_string(provider.get("issuer"))
        discovered_issuer = _coerce_nonempty_string(discovery.get("issuer"))
        if explicit_issuer and discovered_issuer and explicit_issuer != discovered_issuer:
            raise ValueError("OIDC discovery issuer mismatch")
        return discovery

    @staticmethod
    def _provider_requires_discovery(provider: dict[str, Any]) -> bool:
        if not _coerce_nonempty_string(provider.get("discovery_url")):
            return False
        required_fields = ("issuer", "authorization_url", "token_url", "jwks_url")
        return any(not _coerce_nonempty_string(provider.get(field_name)) for field_name in required_fields)

    async def _resolve_provider_runtime_config(
        self,
        provider: dict[str, Any],
        *,
        touch_secret_refs: bool = True,
    ) -> dict[str, Any]:
        if _coerce_nonempty_string(provider.get("discovery_url")):
            _validate_oidc_provider_url("discovery_url", provider.get("discovery_url"))
        discovery = (
            await self._fetch_discovery_document(provider)
            if self._provider_requires_discovery(provider)
            else {}
        )
        resolved = {
            "issuer": _coerce_nonempty_string(provider.get("issuer")) or _coerce_nonempty_string(discovery.get("issuer")),
            "authorization_url": _coerce_nonempty_string(provider.get("authorization_url"))
            or _coerce_nonempty_string(discovery.get("authorization_endpoint")),
            "token_url": _coerce_nonempty_string(provider.get("token_url"))
            or _coerce_nonempty_string(discovery.get("token_endpoint")),
            "jwks_url": _coerce_nonempty_string(provider.get("jwks_url"))
            or _coerce_nonempty_string(discovery.get("jwks_uri")),
            "client_id": _coerce_nonempty_string(provider.get("client_id")),
            "client_secret": await _resolve_client_secret_ref(
                provider.get("client_secret_ref"),
                touch_last_used=touch_secret_refs,
            ),
            "signing_algorithms": sorted(
                _coerce_string_set(provider.get("allowed_signing_algs"))
                or _coerce_string_set(provider.get("signing_algorithms"))
                or _coerce_string_set(discovery.get("id_token_signing_alg_values_supported"))
            ),
        }
        for field_name in ("authorization_url", "token_url", "jwks_url"):
            resolved[field_name] = _validate_oidc_provider_url(field_name, resolved.get(field_name))
        return resolved

    async def inspect_provider_configuration(
        self,
        *,
        provider: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_config = await self._resolve_provider_runtime_config(
            provider,
            touch_secret_refs=False,
        )
        missing_fields = [
            field_name
            for field_name in ("issuer", "authorization_url", "token_url", "jwks_url", "client_id")
            if not runtime_config.get(field_name)
        ]
        if missing_fields:
            raise ValueError(
                "OIDC provider is missing "
                + ", ".join(missing_fields)
            )

        warnings: list[str] = []
        discovery_url = _coerce_nonempty_string(provider.get("discovery_url"))
        client_secret_ref = _coerce_nonempty_string(provider.get("client_secret_ref"))
        if discovery_url:
            resolved_via_discovery = [
                field_name
                for field_name, provider_key in (
                    ("authorization_url", "authorization_url"),
                    ("token_url", "token_url"),
                    ("jwks_url", "jwks_url"),
                )
                if not _coerce_nonempty_string(provider.get(provider_key))
            ]
            if resolved_via_discovery:
                warnings.append(
                    "Resolved {} from OIDC discovery".format(", ".join(resolved_via_discovery))
                )
        if (
            client_secret_ref
            and not client_secret_ref.lower().startswith("env:")
            and _parse_byok_secret_ref(client_secret_ref) is None
        ):
            warnings.append("OIDC client_secret_ref is stored inline; prefer env:VAR_NAME")

        return {
            "ok": True,
            "issuer": str(runtime_config["issuer"]),
            "authorization_url": str(runtime_config["authorization_url"]),
            "token_url": str(runtime_config["token_url"]),
            "jwks_url": str(runtime_config["jwks_url"]),
            "client_id": str(runtime_config["client_id"]),
            "warnings": warnings,
        }

    @staticmethod
    def _resolve_jwk_for_token(id_token: str, jwks_payload: dict[str, Any]) -> dict[str, Any]:
        keys = jwks_payload.get("keys") if isinstance(jwks_payload, dict) else None
        if not isinstance(keys, list) or not keys:
            raise ValueError("OIDC JWKS response is invalid")

        header = jwt.get_unverified_header(id_token)
        kid = _coerce_nonempty_string(header.get("kid")) if isinstance(header, dict) else None
        if kid:
            for key in keys:
                if isinstance(key, dict) and _coerce_nonempty_string(key.get("kid")) == kid:
                    return key
            raise ValueError("OIDC JWKS does not contain the signing key")
        if len(keys) == 1 and isinstance(keys[0], dict):
            return keys[0]
        raise ValueError("OIDC id_token header is missing kid")

    @staticmethod
    def _resolve_allowed_signing_algorithms(
        *,
        provider_runtime_config: dict[str, Any],
        jwk: dict[str, Any],
    ) -> set[str]:
        configured = _coerce_string_set(provider_runtime_config.get("signing_algorithms"))
        if configured:
            return configured

        jwk_alg = _coerce_nonempty_string(jwk.get("alg")) if isinstance(jwk, dict) else None
        if jwk_alg:
            return {jwk_alg}

        key_type = _coerce_nonempty_string(jwk.get("kty")) if isinstance(jwk, dict) else None
        allowed = _ASYMMETRIC_JWT_ALGS_BY_KTY.get(str(key_type or "").upper())
        if allowed:
            return set(allowed)
        raise ValueError("OIDC JWKS key type does not define a supported signing algorithm policy")

    async def build_authorization_request(
        self,
        *,
        provider: dict[str, Any],
        redirect_uri: str,
    ) -> dict[str, Any]:
        runtime_config = await self._resolve_provider_runtime_config(provider)
        authorization_url = runtime_config["authorization_url"]
        client_id = runtime_config["client_id"]
        if not authorization_url or not client_id:
            raise ValueError("OIDC provider is missing authorization_url or client_id")

        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(24)
        code_verifier = secrets.token_urlsafe(64)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.state_ttl_seconds)

        query = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self.default_scopes),
            "state": state,
            "nonce": nonce,
            "code_challenge": _oauth_code_challenge_s256(code_verifier),
            "code_challenge_method": "S256",
        }

        return {
            "auth_url": _append_query_params(str(authorization_url), query),
            "state": state,
            "nonce": nonce,
            "code_verifier": code_verifier,
            "expires_at": expires_at,
            "ttl_seconds": self.state_ttl_seconds,
        }

    async def exchange_authorization_code(
        self,
        *,
        provider: dict[str, Any],
        code: str,
        redirect_uri: str,
        code_verifier: str,
        nonce: str | None = None,
    ) -> dict[str, Any]:
        runtime_config = await self._resolve_provider_runtime_config(provider)
        token_url = runtime_config["token_url"]
        client_id = runtime_config["client_id"]
        issuer = runtime_config["issuer"]
        jwks_url = runtime_config["jwks_url"]
        client_secret = runtime_config["client_secret"]
        if not token_url or not client_id or not issuer:
            raise ValueError("OIDC provider is missing token_url, client_id, or issuer")
        if not jwks_url:
            raise ValueError("OIDC provider is missing jwks_url or discovery_url")

        form_data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        if client_secret:
            form_data["client_secret"] = client_secret

        token_payload = await afetch_json(
            method="POST",
            url=token_url,
            data=form_data,
            max_bytes=OIDC_TOKEN_RESPONSE_MAX_BYTES,
        )

        id_token = _coerce_nonempty_string(token_payload.get("id_token"))
        if not id_token:
            raise ValueError("OIDC token response is missing id_token")

        jwks_payload = await afetch_json(
            method="GET",
            url=jwks_url,
            max_bytes=OIDC_JWKS_MAX_BYTES,
        )
        jwk = self._resolve_jwk_for_token(id_token, jwks_payload)
        allowed_algorithms = self._resolve_allowed_signing_algorithms(
            provider_runtime_config=runtime_config,
            jwk=jwk,
        )
        header = jwt.get_unverified_header(id_token)
        algorithm = _coerce_nonempty_string(header.get("alg")) if isinstance(header, dict) else None
        if not algorithm:
            raise ValueError("OIDC id_token header is missing alg")
        if algorithm not in allowed_algorithms:
            raise ValueError(f"OIDC id_token uses unsupported signing algorithm: {algorithm}")

        try:
            claims = jwt.decode(
                id_token,
                jwk,
                algorithms=sorted(allowed_algorithms),
                audience=client_id,
                issuer=issuer,
            )
        except JWTError as exc:
            raise ValueError(f"OIDC id_token verification failed: {exc}") from exc
        if not isinstance(claims, dict):
            raise ValueError("OIDC id_token payload is invalid")

        if nonce:
            token_nonce = _coerce_nonempty_string(claims.get("nonce"))
            if token_nonce != nonce:
                raise ValueError("OIDC nonce mismatch")
        if not _coerce_nonempty_string(claims.get("sub")):
            raise ValueError("OIDC subject is missing")
        if client_id not in _coerce_string_set(claims.get("aud")):
            raise ValueError("OIDC audience mismatch")
        return claims
