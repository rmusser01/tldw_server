"""Canonical, secret-free endpoint origins for Notes semantic indexing."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit

from httpx import URL, InvalidURL


def canonical_semantic_endpoint_origin(value: object) -> str | None:
    """Return one HTTP origin representation, or ``None`` when invalid."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(value)
        runtime_url = URL(value)
        scheme = parsed.scheme.lower()
        parsed_hostname = parsed.hostname
        raw_host = runtime_url.raw_host
        hostname = raw_host.decode("ascii")
        port = parsed.port
    except (InvalidURL, UnicodeError, ValueError):
        return None
    if b"%" in raw_host:
        return None
    if scheme not in {"http", "https"} or runtime_url.scheme.lower() != scheme or not parsed_hostname or not hostname:
        return None
    try:
        address = ip_address(hostname)
    except ValueError:
        canonical_host = hostname.lower()
        if not canonical_host:
            return None
        display_host = canonical_host
    else:
        canonical_host = str(address).lower()
        display_host = f"[{canonical_host}]" if address.version == 6 else canonical_host
    suffix = f":{port}" if port is not None else ""
    return f"{scheme}://{display_host}{suffix}"


__all__ = ["canonical_semantic_endpoint_origin"]
