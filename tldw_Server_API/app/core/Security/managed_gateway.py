"""Shared authenticated private-hop validation for bounded managed ingress."""

from __future__ import annotations

import hmac
import ipaddress
import os
import re
from urllib.parse import urlsplit

from starlette.requests import HTTPConnection

_MANAGED_HOP_TOKEN = re.compile(r"[!-~]{32,512}")
_MANAGED_FORWARD_HEADERS = frozenset(
    {
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
    }
)
_MANAGED_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "127.0.0.0/8",
        "fc00::/7",
        "::1/128",
    )
)


def _is_managed_private_address(value: str | None) -> bool:
    """Accept only an unadorned RFC1918/ULA/loopback socket address."""
    if not value or value != value.strip() or "%" in value:
        return False
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return any(address in network for network in _MANAGED_PRIVATE_NETWORKS)


def is_managed_gateway_request(request: HTTPConnection) -> bool:
    """Authenticate the private managed gateway envelope without granting user access.

    Managed Compose supplies the secret and exact public loopback origin. The
    gateway replaces incoming forwarding/control headers with one envelope.
    Callers must bound the eligible routes before using this predicate. It does
    not enable general proxy trust or replace authentication or IP policy.
    """
    if os.getenv("TLDW_MANAGED_GATEWAY") != "1" or os.getenv("AUTH_MODE") != "single_user":
        return False
    secret = os.getenv("TLDW_GATEWAY_HOP_SECRET", "")
    if not _MANAGED_HOP_TOKEN.fullmatch(secret):
        return False
    origin = os.getenv("TLDW_MANAGED_PUBLIC_ORIGIN", "")
    try:
        public = urlsplit(origin)
        public_port = public.port
    except ValueError:
        return False
    if (
        public.scheme != "http"
        or public.hostname not in {"127.0.0.1", "localhost", "::1"}
        or public_port is None
        or not 1 <= public_port <= 65535
        or public.username is not None
        or public.password is not None
        or origin != f"http://{public.netloc}"
        or public.path
        or public.query
        or public.fragment
    ):
        return False
    # Require canonical authority, including the exact persisted port.
    host = f"[{public.hostname}]" if public.hostname == "::1" else public.hostname
    if public.netloc != f"{host}:{public_port}":
        return False
    if not _is_managed_private_address(request.client.host if request.client else None):
        return False
    headers = request.headers
    required = {"host", "x-tldw-gateway-hop"} | _MANAGED_FORWARD_HEADERS
    if any(len(headers.getlist(name)) != 1 for name in required) or len(headers.getlist("origin")) > 1:
        return False
    if any(
        name in {"forwarded", "x-real-ip"}
        or (name.startswith("x-forwarded-") and name not in _MANAGED_FORWARD_HEADERS)
        or (name.startswith("x-tldw-gateway-") and name != "x-tldw-gateway-hop")
        for name in headers
    ):
        return False
    supplied_secret = headers["x-tldw-gateway-hop"]
    if not _MANAGED_HOP_TOKEN.fullmatch(supplied_secret) or not hmac.compare_digest(secret, supplied_secret):
        return False
    return (
        headers["host"] == public.netloc
        and ("origin" not in headers or headers["origin"] == origin)
        and headers["x-forwarded-host"] == public.netloc
        and headers["x-forwarded-proto"] == public.scheme
        and headers["x-forwarded-port"] == str(public_port)
        and _is_managed_private_address(headers["x-forwarded-for"])
    )
