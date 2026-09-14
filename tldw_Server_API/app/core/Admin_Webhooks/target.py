"""Shared structural parsing for canonical admin webhook targets."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import SplitResult, unquote, urlsplit

_MAX_TARGET_BYTES = 2_048
_DNS_LABEL_PATTERN = re.compile(r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)\Z")
_LEGACY_NUMERIC_HOST_PATTERN = re.compile(
    r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}\Z"
)
_PERCENT_ESCAPE_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def normalize_webhook_hostname(hostname: str) -> str:
    """Return one strict IP or IDNA hostname representation."""

    if not isinstance(hostname, str) or not hostname or "%" in hostname:
        raise ValueError("target host is invalid")
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        try:
            normalized = hostname.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError("target host is invalid") from exc
        labels = normalized.split(".")
        if (
            not normalized
            or len(normalized) > 253
            or _LEGACY_NUMERIC_HOST_PATTERN.fullmatch(normalized) is not None
            or any(_DNS_LABEL_PATTERN.fullmatch(label) is None for label in labels)
        ):
            raise ValueError("target host is invalid") from None
        try:
            canonical_idna = (
                normalized.encode("ascii")
                .decode("idna")
                .encode("idna")
                .decode("ascii")
            )
        except UnicodeError as exc:
            raise ValueError("target host is invalid") from exc
        if canonical_idna != normalized:
            raise ValueError("target host is invalid") from None
        return normalized


def _validate_percent_escapes(value: str) -> None:
    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        match = _PERCENT_ESCAPE_PATTERN.match(value, index)
        if match is None:
            raise ValueError("target contains a malformed percent escape")
        decoded = int(value[index + 1 : index + 3], 16)
        if decoded < 32 or decoded == 127 or decoded == ord("\\"):
            raise ValueError("target contains an ambiguous percent escape")
        index += 3
    try:
        decoded_value = unquote(value, encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("target contains an ambiguous percent escape") from exc
    if "\\" in decoded_value or _contains_control(decoded_value):
        raise ValueError("target contains an ambiguous percent escape")


def parse_webhook_target_url(url: str) -> tuple[SplitResult, str]:
    """Parse one bounded target URL using registration/executor-safe syntax."""

    if not isinstance(url, str) or not url or len(url) > _MAX_TARGET_BYTES:
        raise ValueError("target URL is invalid")
    try:
        encoded = url.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("target URL is invalid") from exc
    if (
        len(encoded) > _MAX_TARGET_BYTES
        or "#" in url
        or "\\" in url
        or _contains_control(url)
    ):
        raise ValueError("target URL is invalid")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("target URL is invalid") from exc
    if (
        not parsed.scheme
        or not parsed.netloc
        or not hostname
        or parsed.netloc.endswith(":")
    ):
        raise ValueError("target URL is invalid")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise ValueError("target URL is invalid")
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("target port is invalid")

    _validate_percent_escapes(parsed.path)
    _validate_percent_escapes(parsed.query)
    if (parsed.path or "/").startswith("//"):
        raise ValueError("target request path is invalid")
    return parsed, normalize_webhook_hostname(hostname)
