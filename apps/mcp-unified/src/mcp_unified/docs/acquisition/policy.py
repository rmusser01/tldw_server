from __future__ import annotations

import hashlib
import ipaddress
from dataclasses import dataclass
from typing import Literal
from urllib.parse import unquote, urlsplit, urlunsplit

from .models import NormalizedURL, SourceDecision

SourceProfile = Literal["locked_down", "local_first", "online_capable"]

_SUPPORTED_SCHEMES = {"http", "https"}
_DEFAULT_PORTS = {"http": 80, "https": 443}
_SOURCE_PROFILES = {"locked_down", "local_first", "online_capable"}
_LOCAL_HOSTNAMES = {
    "broadcasthost",
    "ip6-localhost",
    "ip6-loopback",
    "localdomain",
    "localhost",
    "localhost.localdomain",
}
_LEGACY_IPV4_CHARS = frozenset("0123456789abcdefABCDEFxX.")


class URLPolicyError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class DomainRule:
    raw_pattern: str
    host: str
    wildcard: bool = False

    @classmethod
    def from_pattern(cls, pattern: str) -> "DomainRule | None":
        try:
            return _parse_domain_rule(pattern, "domain rule")
        except ValueError:
            return None

    def matches(self, host: str) -> bool:
        if self.wildcard:
            return host != self.host and host.endswith(f".{self.host}")
        return host == self.host


@dataclass(frozen=True)
class URLPrefixRule:
    matched_rule: str
    scheme: str
    host: str
    port: int | None
    decoded_path: str

    @classmethod
    def from_prefix(cls, prefix: str) -> "URLPrefixRule | None":
        try:
            return _parse_prefix_rule(prefix, "URL prefix")
        except ValueError:
            return None

    def matches(self, url: NormalizedURL) -> bool:
        if (self.scheme, self.host, self.port) != (url.scheme, url.host, url.port):
            return False
        if _has_dot_segment(url.decoded_path):
            return False
        return _path_has_prefix(url.decoded_path, self.decoded_path)


def safe_argument_hash(raw_url: str) -> str:
    return hashlib.sha256(raw_url.encode("utf-8", errors="surrogatepass")).hexdigest()


def has_url_credentials(raw_url: str) -> bool:
    try:
        parts = urlsplit(raw_url)
    except ValueError:
        return False
    return parts.username is not None or parts.password is not None


def normalize_url(raw_url: str) -> NormalizedURL:
    if _has_raw_whitespace_or_control(raw_url):
        raise URLPolicyError("malformed_url")
    try:
        parts = urlsplit(raw_url)
    except ValueError as exc:
        raise URLPolicyError("malformed_url") from exc

    scheme = parts.scheme.lower()
    if scheme not in _SUPPORTED_SCHEMES:
        raise URLPolicyError("unsupported_url_scheme")
    if not parts.netloc or parts.hostname is None:
        raise URLPolicyError("url_host_required")
    if _netloc_has_empty_port(parts.netloc):
        raise URLPolicyError("malformed_url")

    host = _normalize_hostname(parts.hostname)
    try:
        parsed_port = parts.port
    except ValueError as exc:
        raise URLPolicyError("malformed_url") from exc
    port = None if parsed_port == _DEFAULT_PORTS[scheme] else parsed_port

    path = parts.path or "/"
    decoded_path = unquote(path)
    canonical_url = _build_canonical_url(scheme=scheme, host=host, port=port, path=path, query=parts.query)
    redacted_url = _build_redacted_url(scheme=scheme, host=host, port=port, path=path)
    return NormalizedURL(
        scheme=scheme,
        host=host,
        port=port,
        path=path,
        decoded_path=decoded_path,
        canonical_url=canonical_url,
        redacted_url=redacted_url,
    )


class SourcePolicy:
    def __init__(
        self,
        *,
        web_source_profile: SourceProfile = "locked_down",
        preapproved_domains: tuple[str, ...] = (),
        allowed_url_prefixes: tuple[str, ...] = (),
        denied_domains: tuple[str, ...] = (),
        allow_arbitrary_public_domains: bool = False,
    ) -> None:
        if web_source_profile not in _SOURCE_PROFILES:
            raise ValueError("web_source_profile must be one of: locked_down, local_first, online_capable")
        if web_source_profile == "locked_down":
            if preapproved_domains:
                raise ValueError("locked_down source policy cannot configure preapproved_domains")
            if allow_arbitrary_public_domains:
                raise ValueError("locked_down source policy cannot allow arbitrary public domains")
        self.web_source_profile = web_source_profile
        self.preapproved_domains = _parse_domain_rules(preapproved_domains, "preapproved_domains")
        self.allowed_url_prefixes = _parse_prefix_rules(allowed_url_prefixes, "allowed_url_prefixes")
        self.denied_domains = _parse_domain_rules(denied_domains, "denied_domains")
        self.allow_arbitrary_public_domains = allow_arbitrary_public_domains

    def evaluate(self, raw_url: str) -> SourceDecision:
        argument_hash = safe_argument_hash(raw_url)
        if has_url_credentials(raw_url):
            return SourceDecision(
                status="denied",
                reason="url_credentials_denied",
                safe_argument_hash=argument_hash,
            )

        try:
            normalized = normalize_url(raw_url)
        except URLPolicyError as exc:
            return SourceDecision(
                status="denied",
                reason=exc.reason,
                safe_argument_hash=argument_hash,
            )

        if _is_source_host_denied(normalized.host):
            return SourceDecision(
                status="denied",
                reason="source_host_denied",
                safe_argument_hash=argument_hash,
                redacted_url=normalized.redacted_url,
                normalized_url=normalized,
            )

        for rule in self.denied_domains:
            if rule.matches(normalized.host):
                return SourceDecision(
                    status="denied",
                    reason="domain_denied",
                    safe_argument_hash=argument_hash,
                    redacted_url=normalized.redacted_url,
                    normalized_url=normalized,
                    matched_rule=rule.raw_pattern,
                )

        for rule in self.allowed_url_prefixes:
            if rule.matches(normalized):
                return SourceDecision(
                    status="allowed",
                    reason="url_prefix_allowed",
                    safe_argument_hash=argument_hash,
                    redacted_url=normalized.redacted_url,
                    normalized_url=normalized,
                    matched_rule=rule.matched_rule,
                )

        if self.web_source_profile == "locked_down":
            return _approval_required(argument_hash, normalized)

        for rule in self.preapproved_domains:
            if rule.matches(normalized.host):
                return SourceDecision(
                    status="allowed",
                    reason="domain_allowed",
                    safe_argument_hash=argument_hash,
                    redacted_url=normalized.redacted_url,
                    normalized_url=normalized,
                    matched_rule=rule.raw_pattern,
                )

        if self.web_source_profile == "online_capable" and self.allow_arbitrary_public_domains:
            return SourceDecision(
                status="allowed",
                reason="arbitrary_public_domain_allowed",
                safe_argument_hash=argument_hash,
                redacted_url=normalized.redacted_url,
                normalized_url=normalized,
            )

        return _approval_required(argument_hash, normalized)


def _approval_required(argument_hash: str, normalized: NormalizedURL) -> SourceDecision:
    return SourceDecision(
        status="approval_required",
        reason="source_approval_required",
        safe_argument_hash=argument_hash,
        redacted_url=normalized.redacted_url,
        normalized_url=normalized,
    )


def _normalize_hostname(hostname: str) -> str:
    if hostname != hostname.strip() or "%" in hostname:
        raise URLPolicyError("malformed_url")
    host = hostname.rstrip(".")
    if not host:
        raise URLPolicyError("url_host_required")
    if any(character.isspace() for character in host):
        raise URLPolicyError("malformed_url")
    try:
        return host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise URLPolicyError("malformed_url") from exc


def _build_canonical_url(*, scheme: str, host: str, port: int | None, path: str, query: str) -> str:
    display_host = _format_url_host(host)
    netloc = f"{display_host}:{port}" if port is not None else display_host
    return urlunsplit((scheme, netloc, path or "/", query, ""))


def _build_redacted_url(*, scheme: str, host: str, port: int | None, path: str) -> str:
    display_host = _format_url_host(host)
    netloc = f"{display_host}:{port}" if port is not None else display_host
    return urlunsplit((scheme, netloc, path or "/", "", ""))


def _format_url_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.version == 6:
        return f"[{address.compressed}]"
    return address.compressed


def _netloc_has_empty_port(netloc: str) -> bool:
    authority = netloc.rsplit("@", 1)[-1]
    if authority.startswith("["):
        closing_bracket = authority.find("]")
        return closing_bracket >= 0 and authority[closing_bracket + 1 :] == ":"
    return authority.endswith(":")


def _parse_domain_rules(patterns: tuple[str, ...], field_name: str) -> tuple[DomainRule, ...]:
    return tuple(_parse_domain_rule(pattern, field_name) for pattern in patterns)


def _parse_domain_rule(pattern: str, field_name: str) -> DomainRule:
    text = pattern.strip()
    if not text:
        raise ValueError(f"{field_name} contains an empty domain rule")
    if any(separator in text for separator in ("://", "/", "\\", "?", "#", ":")):
        raise ValueError(f"{field_name} must contain host patterns, not URLs")

    wildcard = text.startswith("*.")
    host_text = text[2:] if wildcard else text
    if "*" in host_text:
        raise ValueError(f"{field_name} wildcard rules must start with '*.'")

    try:
        host = _normalize_hostname(host_text)
    except URLPolicyError as exc:
        raise ValueError(f"{field_name} contains invalid host pattern") from exc
    return DomainRule(raw_pattern=text, host=host, wildcard=wildcard)


def _parse_prefix_rules(prefixes: tuple[str, ...], field_name: str) -> tuple[URLPrefixRule, ...]:
    return tuple(_parse_prefix_rule(prefix, field_name) for prefix in prefixes)


def _parse_prefix_rule(prefix: str, field_name: str) -> URLPrefixRule:
    try:
        normalized = normalize_url(prefix)
    except URLPolicyError as exc:
        raise ValueError(f"{field_name} contains invalid URL prefix: {exc.reason}") from exc
    if _has_dot_segment(normalized.decoded_path):
        raise ValueError(f"{field_name} cannot contain dot path segments")
    try:
        ipaddress.ip_address(normalized.host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{field_name} cannot contain IP literal hosts")
    return URLPrefixRule(
        matched_rule=normalized.redacted_url,
        scheme=normalized.scheme,
        host=normalized.host,
        port=normalized.port,
        decoded_path=normalized.decoded_path,
    )


def _is_source_host_denied(host: str) -> bool:
    if (
        host in _LOCAL_HOSTNAMES
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".localdomain")
    ):
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return _is_legacy_ipv4_candidate(host)
    return True


def _is_legacy_ipv4_candidate(host: str) -> bool:
    if not host or any(character not in _LEGACY_IPV4_CHARS for character in host):
        return False
    parts = host.split(".")
    if len(parts) > 4 or any(not part for part in parts):
        return False
    if len(parts) == 1:
        return parts[0].isdigit()
    return all(_is_legacy_ipv4_part_candidate(part) for part in parts)


def _is_legacy_ipv4_part_candidate(part: str) -> bool:
    lower_part = part.lower()
    if lower_part.startswith("0x"):
        return len(part) > 2 and all(character in "0123456789abcdefABCDEF" for character in part[2:])
    return part.isdigit()


def _has_raw_whitespace_or_control(raw_url: str) -> bool:
    return any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in raw_url)


def _has_dot_segment(decoded_path: str) -> bool:
    return any(segment in {".", ".."} for segment in decoded_path.split("/"))


def _path_has_prefix(path: str, prefix: str) -> bool:
    normalized_path = path or "/"
    normalized_prefix = prefix or "/"
    if normalized_prefix == "/":
        return normalized_path.startswith("/")
    if normalized_prefix.endswith("/"):
        return normalized_path.startswith(normalized_prefix)
    return normalized_path == normalized_prefix or normalized_path.startswith(f"{normalized_prefix}/")


__all__ = [
    "DomainRule",
    "SourcePolicy",
    "URLPolicyError",
    "URLPrefixRule",
    "has_url_credentials",
    "normalize_url",
    "safe_argument_hash",
]
