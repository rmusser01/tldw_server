"""Canonical, secret-free endpoint origins for Notes semantic indexing."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit


def canonical_semantic_endpoint_origin(value: object) -> str | None:
    """Return one HTTP origin representation, or ``None`` when invalid."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    try:
        address = ip_address(hostname)
    except ValueError:
        try:
            canonical_host = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError:
            return None
        if not canonical_host:
            return None
        display_host = canonical_host
    else:
        canonical_host = str(address).lower()
        display_host = f"[{canonical_host}]" if address.version == 6 else canonical_host
    suffix = f":{port}" if port is not None else ""
    return f"{scheme}://{display_host}{suffix}"


__all__ = ["canonical_semantic_endpoint_origin"]
