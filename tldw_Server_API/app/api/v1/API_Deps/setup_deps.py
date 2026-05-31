"""Dependencies for protecting setup endpoints.

By default we restrict mutating setup actions to local requests only (loopback).
Set the environment variable ``TLDW_SETUP_ALLOW_REMOTE=1`` (or config
``allow_remote_setup_access=true``) to permit remote access. Remote callers must
authenticate as admin; loopback requests remain allowed without admin checks.
In tests, Starlette's TestClient reports host as
``testclient``; this value is treated as local unless an ``X-Forwarded-For``
header is provided.
"""

from __future__ import annotations

import ipaddress
import os
import time
from configparser import ConfigParser

from fastapi import HTTPException, Request, status
from loguru import logger

from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.testing import env_flag_enabled

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
# Treat these as local hostnames for the Host header check.
# Include IPv6 localhost as well.
LOCAL_HOST_HEADERS = {"localhost", "127.0.0.1", "::1", "testserver"}
_FALSEY_ENV_VALUES = {"0", "false", "no", "off", "n"}

_CONFIG_REMOTE_CACHE_TTL = 30.0  # seconds
_config_remote_cached: bool | None = None
_config_remote_cached_at = 0.0
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})
_SETUP_PROXY_HEADERS = (
    "x-forwarded-for",
    "forwarded",
    "x-real-ip",
    "x-forwarded-host",
    "x-forwarded-proto",
)
_REMOTE_SETUP_WRITE_DENIED_DETAIL = (
    "Setup writes are only available from localhost unless remote setup access is explicitly enabled."
)


def _principal_has_admin_claims(principal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (getattr(principal, "roles", None) or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (getattr(principal, "permissions", None) or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


async def _require_admin_for_remote(request: Request) -> None:
    """Enforce admin-level authorization for remote setup access.

    Principals satisfying any of these conditions are permitted:
    - explicit admin claims (role ``admin`` or permissions ``*`` / ``system.configure``)
    - Single-user mode principal (bootstrapped local user)

    Raises:
        HTTPException: 403 Forbidden if the principal lacks required privileges.
    """
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import is_single_user_principal

    principal = await get_auth_principal(request)
    if _principal_has_admin_claims(principal) or is_single_user_principal(principal):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required for remote setup.",
    )


async def require_shared_audio_installer_access(request: Request) -> None:
    """Require authenticated admin-style access for shared audio installer routes.

    Shared installer actions are intentionally decoupled from loopback/local-only
    checks so they can be used from the shared UI after setup completes. This
    accepts principals with admin-equivalent claims, including the bootstrapped
    single-user principal.
    """
    await _require_admin_for_remote(request)


def reset_remote_access_cache(value: bool | None = None) -> None:
    """Reset the cached remote access flag (test helper/administrative hook)."""
    _set_remote_access_cache(value)


def _set_remote_access_cache(value: bool | None) -> None:
    global _config_remote_cached, _config_remote_cached_at
    if value is None:
        _config_remote_cached = None
        _config_remote_cached_at = 0.0
    else:
        _config_remote_cached = value
        _config_remote_cached_at = time.monotonic()


def _should_trust_proxy() -> bool:
    """Decide whether to honor proxy headers for locality checks.

    We default to trusting local reverse proxies unless ops explicitly
    disable it by setting ``TLDW_SETUP_TRUST_PROXY`` to a false-like value
    (e.g. ``0`` or ``false``). This preserves the historical behaviour where
    ``X-Forwarded-For`` was always honored, preventing proxy bypasses by
    remote clients.
    """

    raw = os.getenv("TLDW_SETUP_TRUST_PROXY")
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY_ENV_VALUES


def _first_forwarded_ip(request: Request) -> str | None:
    """Return a loopback forwarded IP only when the complete XFF chain is local."""

    if not _should_trust_proxy():
        return None
    raw = request.headers.get("x-forwarded-for")
    if not raw:
        return None
    try:
        candidates = [
            candidate.strip().lower().strip("[]").split("%", 1)[0]
            for candidate in raw.split(",")
            if candidate.strip()
        ]
    except Exception:  # noqa: BLE001
        return None
    if not candidates:
        return None

    parsed = []
    for candidate in candidates:
        try:
            parsed.append(ipaddress.ip_address(candidate))
        except ValueError:
            return None
    if not all(candidate.is_loopback for candidate in parsed):
        return None
    return str(parsed[0])


def should_trust_setup_proxy_headers(request: Request) -> bool:
    """Return True when setup code may honor proxy-supplied client headers."""

    client_host = request.client.host if request.client else None
    return _should_trust_proxy() and _is_loopback_host(client_host)


def has_setup_proxy_headers(request: Request) -> bool:
    """Return True when a request includes setup-relevant proxy headers."""

    headers = request.headers
    return any(key in headers for key in _SETUP_PROXY_HEADERS)


def _is_loopback_host(host: str | None) -> bool:
    """Return True if the provided hostname/IP represents a local loopback address."""
    if not host:
        return False

    value = host.strip().lower()
    if not value:
        return False

    if value in LOCAL_HOSTS:
        return True

    # Strip IPv6 scope identifiers before parsing
    candidate = value.split("%", 1)[0]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _has_proxy_headers(request: Request) -> bool:
    # Treat any of these headers as evidence of a proxy hop; block by default.
    return has_setup_proxy_headers(request)


def _host_header_is_local(request: Request) -> bool:
    """Return True if the Host header targets a local hostname.

    Handles IPv6 bracketed hosts (e.g., "[::1]:8000").
    """
    raw = request.headers.get("host") or request.headers.get("Host")
    if not raw:
        return False

    host = raw.strip()

    # IPv6 addresses in Host header are bracketed: [::1]:8000
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return False
        hostname = host[1:end].strip().lower()
    else:
        # Strip port if present (hostname:port)
        hostname = host.split(":", 1)[0].strip().lower()

    return hostname in LOCAL_HOST_HEADERS


def _is_local_setup_request(
    *,
    method: str,
    path: str,
    has_proxy_headers: bool,
    client_is_local: bool,
    forwarded_is_local: bool,
    host_header_is_local: bool,
) -> bool:
    """Return True when a setup request should be treated as local-only."""
    if method == "GET" and path.endswith("/api/v1/setup/config"):
        if not host_header_is_local:
            return False
        if not has_proxy_headers and client_is_local:
            return True
        return bool(has_proxy_headers and client_is_local and forwarded_is_local)

    if has_proxy_headers:
        return client_is_local and forwarded_is_local and host_header_is_local

    return client_is_local and host_header_is_local


async def require_local_setup_access(request: Request) -> None:
    """Guard mutating setup operations so they can only be called locally.

    When remote access is enabled, local requests are still allowed while
    remote requests require admin authorization.

    Special-case: Allow GET /api/v1/setup/config when the request targets
    localhost (by Host header or client IP), even if proxy headers are present.
    This keeps the Setup UI functional behind common local reverse proxies,
    while keeping POST/PUT restricted.
    """
    allow_remote_env = env_flag_enabled("TLDW_SETUP_ALLOW_REMOTE")
    allow_remote_config = _config_allows_remote()

    path = (request.url.path or "").lower()
    method = (request.method or "").upper()

    has_proxy_headers = _has_proxy_headers(request)
    client_host = request.client.host if request.client else None
    forwarded_ip = _first_forwarded_ip(request)
    client_is_local = _is_loopback_host(client_host)
    forwarded_is_local = _is_loopback_host(forwarded_ip)
    host_header_is_local = _host_header_is_local(request)
    is_local_request = _is_local_setup_request(
        method=method,
        path=path,
        has_proxy_headers=has_proxy_headers,
        client_is_local=client_is_local,
        forwarded_is_local=forwarded_is_local,
        host_header_is_local=host_header_is_local,
    )

    if allow_remote_env or allow_remote_config:
        if allow_remote_config and not allow_remote_env:
            logger.info("Remote setup access permitted via config.txt (allow_remote_setup_access=true)")
        if is_local_request:
            return
        await _require_admin_for_remote(request)
        return

    # Permit GET to setup config snapshot only if effectively local.
    # If proxy headers are present, require both the proxy connection and forwarded IP to be loopback.
    if method == "GET" and path.endswith("/api/v1/setup/config"):
        if not host_header_is_local:
            logger.warning(
                "Blocked GET to setup config due to non-local Host header: {}",
                request.headers.get("host"),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Setup access is restricted to local requests. Host header must target localhost.",
            )
        if not has_proxy_headers and client_is_local:
            return
        if has_proxy_headers and client_is_local and forwarded_is_local:
            logger.debug("Allowing proxied localhost GET to /api/v1/setup/config with loopback client and forwarded IP")
            return
        logger.warning(
            "Blocked proxied GET to setup config from client={} forwarded={} (local={}/{})",
            client_host,
            forwarded_ip,
            client_is_local,
            forwarded_is_local,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Setup access is restricted to local requests. Forwarded client is not local. "
                "Set TLDW_SETUP_ALLOW_REMOTE=1 or enable allow_remote_setup_access in config.txt to bypass."
            ),
        )

    if has_proxy_headers:
        if not (client_is_local and forwarded_is_local and host_header_is_local):
            logger.warning(
                "Blocked setup access via proxy (client={}, forwarded={}, host_local={})",
                client_host,
                forwarded_ip,
                host_header_is_local,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=_REMOTE_SETUP_WRITE_DENIED_DETAIL,
            )
        return

    if client_is_local and host_header_is_local:
        return

    logger.warning("Blocked remote setup access from host={}", client_host)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=_REMOTE_SETUP_WRITE_DENIED_DETAIL,
    )


def _config_allows_remote() -> bool:
    """Return True when config.txt enables remote setup access."""
    global _config_remote_cached, _config_remote_cached_at

    now = time.monotonic()
    if (
        _config_remote_cached is not None
        and (now - _config_remote_cached_at) < _CONFIG_REMOTE_CACHE_TTL
    ):
        return _config_remote_cached

    allow_remote = False
    try:
        config_path = setup_manager.get_config_file_path()
        parser = ConfigParser()
        parser.read(config_path, encoding="utf-8")
        allow_remote = parser.getboolean("Setup", "allow_remote_setup_access", fallback=False)
    except Exception as exc:  # noqa: BLE001 - best-effort config read should not block setup access checks
        logger.bind(error_type=type(exc).__name__).debug(
            "Unable to read setup remote access configuration; using localhost-only default"
        )

    _set_remote_access_cache(allow_remote)
    return allow_remote


def _on_remote_access_updated(enabled: bool) -> None:
    """Callback fired when config.txt flips remote access."""
    _set_remote_access_cache(enabled)
    if enabled:
        logger.warning(
            "Setup remote access enabled via config.txt; remote clients are now permitted to call setup APIs."
        )
    else:
        logger.info("Setup remote access disabled via config.txt; setup APIs restricted to localhost.")


setup_manager.register_remote_access_hook(_on_remote_access_updated)
