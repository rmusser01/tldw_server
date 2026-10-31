"""Neutral, bounded observability helpers for Web Scraping."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_STAGES = frozenset({"llm_extraction", "schema_generation", "regex_generation"})
_CODES = frozenset(
    {
        "provider_error",
        "strict_json_failed",
        "selector_invalid",
        "regex_invalid",
        "regex_too_large",
        "regex_timeout",
    }
)


def _is_canonical_idna_label(label: str) -> bool:
    if not label.startswith("xn--"):
        return True
    try:
        decoded = label.encode("ascii").decode("idna")
        canonical = decoded.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    return canonical == label


def _is_valid_dns_name(host: str) -> bool:
    if len(host) > 253 or host.replace(".", "").isdigit():
        return False
    labels = host.split(".")
    return all(
        1 <= len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isascii() and (character.isalnum() or character == "-") for character in label)
        and _is_canonical_idna_label(label)
        for label in labels
    )


def sanitized_host(url: str) -> str:
    """Return a canonical host label without credentials or URL components."""

    if type(url) is not str or "\\" in url or any(ord(character) < 32 or ord(character) == 127 for character in url):
        return "unknown"
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
            return "unknown"
        if any(ord(character) <= 32 or ord(character) == 127 for character in parsed.netloc):
            return "unknown"
        _port = parsed.port
        host = parsed.hostname
    except (AttributeError, TypeError, UnicodeError, ValueError):
        return "unknown"
    if not host or "%" in host:
        return "unknown"
    if host.endswith("."):
        host = host[:-1]
    if not host or host.endswith("."):
        return "unknown"
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if ":" in host:
            return "unknown"
    try:
        canonical_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return "unknown"
    return canonical_host if _is_valid_dns_name(canonical_host) else "unknown"


def bounded_stage(value: object) -> str:
    """Return an approved extraction stage label."""

    return value if type(value) is str and value in _STAGES else "runtime"


def bounded_code(value: object) -> str:
    """Return an approved extraction failure code."""

    return value if type(value) is str and value in _CODES else "other"
