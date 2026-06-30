from __future__ import annotations

import ipaddress
import os
from typing import Callable

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.testing import is_truthy

_SETUP_ACCESS_GUARD_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _env_true(name: str) -> bool:
    raw = os.getenv(name)
    if not raw:
        return False
    return is_truthy(raw)


def _get_peer_ip(request: Request) -> str | None:
    try:
        return request.client.host if request.client else None
    except _SETUP_ACCESS_GUARD_NONCRITICAL_EXCEPTIONS:
        return None


def _is_loopback(ip_str: str | None) -> bool:
    if not ip_str:
        return False
    if ip_str in {"testclient", "localhost"}:
        return True
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return ip_obj.is_loopback
    except ValueError:
        # Non-IP or parse failure; treat as remote
        return False


def setup_remote_access_enabled() -> bool:
    if _env_true("TLDW_SETUP_ALLOW_REMOTE"):
        return True
    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section("Setup"):
            return cp.getboolean("Setup", "allow_remote_setup_access", fallback=False)
    except _SETUP_ACCESS_GUARD_NONCRITICAL_EXCEPTIONS:
        pass
    return False


class SetupAccessGuardMiddleware(BaseHTTPMiddleware):
    """Blocks remote access to /setup unless explicitly allowed by config/env.

    Default: local-only. Enable remote access by setting:
      - TLDW_SETUP_ALLOW_REMOTE=1, or
      - [Setup] allow_remote_setup_access=true in Config_Files/config.txt
    """

    def _parse_allowlist(self, raw: str | None) -> list[ipaddress._BaseNetwork]:
        nets: list[ipaddress._BaseNetwork] = []
        if not raw:
            return nets
        for token in [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]:
            try:
                # Accept CIDR or single IP; for single IP, convert to /32 or /128
                if "/" in token:
                    nets.append(ipaddress.ip_network(token, strict=False))
                else:
                    ip = ipaddress.ip_address(token)
                    nets.append(ipaddress.ip_network(ip.exploded + ("/32" if ip.version == 4 else "/128"), strict=False))
            except ValueError:
                logger.warning(f"Invalid allowlist entry ignored: {token}")
        return nets

    def _load_allowlist(self, section: str, field: str, env_name: str) -> list[ipaddress._BaseNetwork]:
        # ENV takes precedence
        raw_env = os.getenv(env_name)
        if raw_env:
            return self._parse_allowlist(raw_env)
        try:
            cp = load_comprehensive_config()
            if cp and cp.has_section(section):
                raw_cfg = cp.get(section, field, fallback="").strip()
                if raw_cfg:
                    return self._parse_allowlist(raw_cfg)
        except _SETUP_ACCESS_GUARD_NONCRITICAL_EXCEPTIONS:
            pass
        return []

    def _resolve_client_ip(self, request: Request, trusted_proxies: list[ipaddress._BaseNetwork]) -> str | None:
        """Resolve client IP using X-Forwarded-For only when the peer is a trusted proxy.

        - If remote peer is in trusted_proxies and XFF present, walk the chain
          from right to left and return the first untrusted IP.
        - Else, use the socket peer.
        - X-Real-IP is considered only when peer is trusted, XFF is absent, and
          the header is valid.
        """
        peer = _get_peer_ip(request)
        try:
            peer_ip_obj = ipaddress.ip_address(peer) if peer else None
        except ValueError:
            peer_ip_obj = None

        def _is_trusted(ip: ipaddress._BaseAddress | None) -> bool:
            return bool(ip and any(ip in net for net in trusted_proxies))

        if _is_trusted(peer_ip_obj):
            fwd = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
            if fwd:
                parsed_chain: list[ipaddress._BaseAddress] = []
                malformed_entries = False
                for raw_part in fwd.split(","):
                    part = raw_part.strip()
                    if not part:
                        continue
                    try:
                        parsed_chain.append(ipaddress.ip_address(part))
                    except ValueError:
                        malformed_entries = True
                        continue
                if malformed_entries:
                    logger.warning("Malformed X-Forwarded-For entry ignored for trusted Setup proxy")
                for ip_obj in reversed(parsed_chain):
                    if not _is_trusted(ip_obj):
                        return str(ip_obj)
                if parsed_chain:
                    return str(parsed_chain[0])
            xr = request.headers.get("x-real-ip") or request.headers.get("X-Real-IP")
            if xr:
                try:
                    ipaddress.ip_address(xr.strip())
                    return xr.strip()
                except ValueError:
                    pass
        return peer

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path or ""
        if not path.startswith("/setup"):
            return await call_next(request)

        # Build allow/deny lists and trusted proxies (environment or config)
        setup_allowlist = self._load_allowlist("Setup", "setup_ip_allowlist", "TLDW_SETUP_ALLOWLIST")
        setup_denylist = self._load_allowlist("Setup", "setup_ip_denylist", "TLDW_SETUP_DENYLIST")
        trusted_proxies = self._load_allowlist("Server", "trusted_proxies", "TLDW_TRUSTED_PROXIES")

        client_ip = self._resolve_client_ip(request, trusted_proxies)
        if _is_loopback(client_ip):
            return await call_next(request)

        # Convert client_ip to ip_address object for membership checks
        client_ip_obj: ipaddress._BaseAddress | None = None
        try:
            client_ip_obj = ipaddress.ip_address(client_ip) if client_ip else None
        except ValueError:
            client_ip_obj = None

        # Denylist takes precedence (except loopback handled above)
        if setup_denylist and client_ip_obj and any(client_ip_obj in net for net in setup_denylist):
            logger.warning(f"Blocked remote Setup (denylist) from {client_ip}")
            return PlainTextResponse("Access denied by IP denylist.", status_code=403)
        # If an allowlist is provided, only allow listed IPs regardless of remote-access flag.
        if setup_allowlist:
            if client_ip_obj and any(client_ip_obj in net for net in setup_allowlist):
                return await call_next(request)
            logger.warning(f"Blocked remote Setup (allowlist) from {client_ip}")
            return PlainTextResponse("Access denied by IP allowlist.", status_code=403)
        if setup_remote_access_enabled():
            return await call_next(request)

        logger.warning(
            f"Blocked remote Setup access from {client_ip} (enable via config/env toggle)"
        )
        return PlainTextResponse(
            (
                "Remote Setup access is disabled.\n\n"
                "To allow remote Setup access, set TLDW_SETUP_ALLOW_REMOTE=1\n"
                "and run the server with a public host (e.g., uvicorn --host 0.0.0.0),\n"
                "or set [Setup] allow_remote_setup_access=true in Config_Files/config.txt.\n\n"
                "Optionally, restrict by IP/CIDR allowlist:\n"
                "- Env:  TLDW_SETUP_ALLOWLIST=192.168.1.0/24,10.0.0.5\n"
                "- Config: [Setup] setup_ip_allowlist=192.168.1.0/24, 10.0.0.5\n\n"
                "Optionally, [Setup] setup_ip_denylist or TLDW_SETUP_DENYLIST can block IPs."
            ),
            status_code=403,
        )


__all__ = ["SetupAccessGuardMiddleware", "setup_remote_access_enabled"]
