"""Credential-free, one-hop transport facade for research discovery."""

from __future__ import annotations

import ipaddress
import json
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal, TypeAlias
from urllib.parse import parse_qsl, quote_from_bytes, unquote_to_bytes, urljoin, urlsplit

from tldw_Server_API.app.core.Research.discovery.contracts import (
    MAX_PAGINATION_CURSOR,
    AccessRoute,
    BoundedDecimalQueryValuePolicy,
    BoundedTextQueryValuePolicy,
    CredentialRequirement,
    DispatchIntent,
    ExactOrigin,
    ExactQueryValuePolicy,
    JSONBodyPair,
    LiteralTermsQueryValuePolicy,
    OperationKind,
    PathSlot,
    PathSlotKind,
    PathTemplate,
    QueryPair,
    RouteLimits,
    RoutePolicy,
    canonical_policy_digest,
)
from tldw_Server_API.app.core.Security.http_hop import (
    HTTPHopError,
    HTTPHopLimits,
    HTTPHopResponse,
    NormalizedHTTPHopRequest,
    request_http_hop,
)

GatewayErrorCode = Literal[
    "request_rejected",
    "policy_inactive",
    "hop_failed",
    "invalid_hop_response",
]
OneHop: TypeAlias = Callable[[NormalizedHTTPHopRequest], Awaitable[HTTPHopResponse]]
PolicyActivityCheck: TypeAlias = Callable[[str, str], bool]

_ERROR_MESSAGES: dict[GatewayErrorCode, str] = {
    "request_rejected": "Discovery gateway request rejected",
    "policy_inactive": "Discovery gateway policy inactive",
    "hop_failed": "Discovery gateway hop failed",
    "invalid_hop_response": "Discovery gateway hop response rejected",
}
_TIMEOUT_HOP_ERROR_CODES = frozenset(
    {"dns_timeout", "connect_timeout", "read_timeout", "write_timeout", "total_timeout"}
)
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
_BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_CANONICAL_UNSIGNED_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_DATE_SEGMENT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_DOI_REGISTRANT_SEGMENT = re.compile(r"10\.[0-9]{4,9}\Z")
_MAX_CONTENT_TYPE_BYTES = 256
_MAX_RETRY_AFTER_BYTES = 128
_MIN_FINAL_HEADER_BLOCK_BYTES = len(b"HTTP/1.0 200\n\n")
_DENIED_IPV6_TRANSITION_NETWORKS = (
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/32"),
    ipaddress.ip_network("2002::/16"),
)


class DiscoveryGatewayError(Exception):
    """Stable failure without request, response, or provider detail."""

    __slots__ = ("code", "retryable", "timed_out")

    def __init__(
        self,
        code: GatewayErrorCode,
        *,
        retryable: bool = False,
        timed_out: bool = False,
    ) -> None:
        if code not in _ERROR_MESSAGES:
            raise ValueError("Unsupported discovery gateway error code")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean")
        if type(timed_out) is not bool:
            raise TypeError("timed_out must be a boolean")
        self.code = code
        self.retryable = retryable
        self.timed_out = timed_out
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class DiscoveryGatewayTrace:
    """Public and derived evidence for one physical hop."""

    route_id: str
    policy_digest: str
    scheme: str
    requested_host: str
    tls_server_name: str | None
    port: int
    method: str
    path: str
    query_keys: tuple[str, ...]
    timeout_ms: int
    max_response_bytes: int
    http_limits: HTTPHopLimits
    status_code: int
    resolved_ips: tuple[str, ...]
    connected_ip: str
    response_header_bytes: int
    wire_bytes: int
    decoded_bytes: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class DiscoveryGatewayResponse:
    """Bounded body, safe metadata, and evidence from exactly one hop."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    trace: DiscoveryGatewayTrace
    redirect_location: str | None = field(repr=False)
    retry_after: str | None = field(repr=False)


@dataclass(frozen=True, slots=True)
class _BindingSnapshot:
    route_id: str
    policy_digest: str
    scheme: str
    host: str
    port: int
    method: str
    operation_kind: OperationKind
    path: str
    allowed_query_keys: tuple[str, ...]
    query_pairs: tuple[tuple[str, str], ...]
    query_keys: tuple[str, ...]
    json_body_pairs: tuple[tuple[str, str | int], ...]
    timeout_ms: int
    max_response_bytes: int
    route_limits: RouteLimits
    http_limits: HTTPHopLimits


@dataclass(frozen=True, slots=True)
class _ResponseSnapshot:
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    resolved_ips: tuple[str, ...]
    connected_ip: str
    response_header_bytes: int
    wire_bytes: int
    redirect_location: str | None
    retry_after: str | None


def _reject(code: GatewayErrorCode) -> DiscoveryGatewayError:
    return DiscoveryGatewayError(code)


def _hop_limits(limits: RouteLimits) -> HTTPHopLimits:
    timeout_seconds = limits.timeout_ms / 1000
    response_bytes = limits.max_response_bytes
    return HTTPHopLimits(
        dns_timeout_seconds=timeout_seconds,
        connect_timeout_seconds=timeout_seconds,
        read_timeout_seconds=timeout_seconds,
        write_timeout_seconds=timeout_seconds,
        total_timeout_seconds=timeout_seconds,
        max_wire_bytes=response_bytes,
        max_decompressed_bytes=response_bytes,
        max_parser_input_bytes=response_bytes,
        max_request_body_bytes=limits.max_request_body_bytes,
    )


def _snapshot_path_template(template: object) -> PathTemplate | None:
    if type(template) is not PathTemplate or type(template.segments) is not tuple:
        return None
    segments: list[str | PathSlot] = []
    try:
        for segment in template.segments:
            if type(segment) is str:
                segments.append(segment)
            elif type(segment) is PathSlot:
                segments.append(PathSlot(segment.kind, segment.max_chars))
            else:
                return None
        return PathTemplate(tuple(segments), template.pagination_segment_index)
    except (TypeError, ValueError):
        return None


def _decode_canonical_path_slot(raw_segment: str, slot: PathSlot) -> tuple[str, str] | None:
    if not raw_segment or _BAD_PERCENT_ESCAPE.search(raw_segment):
        return None
    try:
        decoded = unquote_to_bytes(raw_segment).decode("utf-8", errors="strict")
    except (UnicodeError, ValueError):
        return None
    if (
        not decoded
        or not decoded.isascii()
        or len(decoded) > slot.max_chars
        or decoded in {".", ".."}
        or any(character in "/\\%" or ord(character) < 32 or ord(character) == 127 for character in decoded)
    ):
        return None

    canonical = quote_from_bytes(decoded.encode("ascii"), safe="")
    if raw_segment != canonical:
        return None
    return decoded, canonical


def _valid_template_path(template: PathTemplate, raw_path: str) -> bool:
    if (
        not raw_path.startswith("/")
        or not raw_path.isascii()
        or "?" in raw_path
        or "#" in raw_path
        or "\\" in raw_path
        or _BAD_PERCENT_ESCAPE.search(raw_path)
    ):
        return False
    raw_segments = raw_path[1:].split("/")
    if len(raw_segments) != len(template.segments):
        return False

    rendered_segments: list[str] = []
    interval_dates: list[date] = []
    for raw_segment, expected in zip(raw_segments, template.segments, strict=True):
        if type(expected) is str:
            if raw_segment != expected:
                return False
            rendered_segments.append(expected)
            continue

        decoded_slot = _decode_canonical_path_slot(raw_segment, expected)
        if decoded_slot is None:
            return False
        decoded, canonical = decoded_slot
        if expected.kind is PathSlotKind.DATE:
            if _DATE_SEGMENT.fullmatch(decoded) is None:
                return False
            try:
                parsed_date = date.fromisoformat(decoded)
            except ValueError:
                return False
            if parsed_date.isoformat() != decoded:
                return False
            interval_dates.append(parsed_date)
        elif expected.kind is PathSlotKind.UINT:
            if _CANONICAL_UNSIGNED_DECIMAL.fullmatch(decoded) is None or int(decoded) > MAX_PAGINATION_CURSOR:
                return False
        elif expected.kind is PathSlotKind.DOI_REGISTRANT:
            if _DOI_REGISTRANT_SEGMENT.fullmatch(decoded) is None:
                return False
        elif expected.kind is PathSlotKind.DOI_SUFFIX:
            if (
                not decoded[0].isalnum()
                or any(character in " /\\%?#" for character in decoded)
                or any(not "!" <= character <= "~" for character in decoded)
            ):
                return False
        else:
            return False
        rendered_segments.append(canonical)

    if len(interval_dates) not in {0, 2}:
        return False
    if interval_dates:
        start, end = interval_dates
        if start > end or (end - start).days + 1 > 366:
            return False
    return raw_path == f"/{'/'.join(rendered_segments)}"


def _snapshot_query_value_policies(
    policies: object,
) -> (
    tuple[
        ExactQueryValuePolicy
        | BoundedDecimalQueryValuePolicy
        | LiteralTermsQueryValuePolicy
        | BoundedTextQueryValuePolicy,
        ...,
    ]
    | None
):
    if type(policies) is not tuple:
        return None
    snapshots: list[
        ExactQueryValuePolicy
        | BoundedDecimalQueryValuePolicy
        | LiteralTermsQueryValuePolicy
        | BoundedTextQueryValuePolicy
    ] = []
    try:
        for policy in policies:
            if type(policy) is ExactQueryValuePolicy:
                snapshot = ExactQueryValuePolicy(policy.name, policy.value, policy.required)
            elif type(policy) is BoundedDecimalQueryValuePolicy:
                snapshot = BoundedDecimalQueryValuePolicy(policy.name, policy.maximum, policy.required)
            elif type(policy) is LiteralTermsQueryValuePolicy:
                snapshot = LiteralTermsQueryValuePolicy(
                    policy.name,
                    policy.fixed_suffix,
                    policy.max_terms,
                    policy.max_term_chars,
                    policy.required,
                )
            elif type(policy) is BoundedTextQueryValuePolicy:
                snapshot = BoundedTextQueryValuePolicy(policy.name, policy.max_chars, policy.required)
            else:
                return None
            snapshots.append(snapshot)
    except (TypeError, ValueError):
        return None
    return tuple(snapshots)


def _valid_query_policy_value(
    policy: (
        ExactQueryValuePolicy
        | BoundedDecimalQueryValuePolicy
        | LiteralTermsQueryValuePolicy
        | BoundedTextQueryValuePolicy
    ),
    value: str,
) -> bool:
    if type(policy) is ExactQueryValuePolicy:
        return value == policy.value
    if type(policy) is BoundedDecimalQueryValuePolicy:
        return (
            _CANONICAL_UNSIGNED_DECIMAL.fullmatch(value) is not None
            and len(value) <= len(str(policy.maximum))
            and int(value) <= policy.maximum
        )
    if type(policy) is LiteralTermsQueryValuePolicy:
        if not value.endswith(policy.fixed_suffix):
            return False
        literal_expression = value[: -len(policy.fixed_suffix)]
        terms = literal_expression.split(" AND ")
        if not 1 <= len(terms) <= policy.max_terms:
            return False
        for quoted_term in terms:
            if len(quoted_term) < 3 or quoted_term[0] != '"' or quoted_term[-1] != '"':
                return False
            term = quoted_term[1:-1]
            if (
                len(term) > policy.max_term_chars
                or unicodedata.normalize("NFKC", term) != term
                or not all(character.isalnum() for character in term)
            ):
                return False
        return True
    return (
        1 <= len(value) <= policy.max_chars
        and unicodedata.normalize("NFKC", value) == value
        and value == value.strip()
        and "  " not in value
        and any(character.isalnum() for character in value)
        and all(character.isalnum() or character in " -&/" for character in value)
    )


def _snapshot_binding(route: object, intent: object) -> _BindingSnapshot | None:
    """Copy and validate every field used after the policy check."""
    try:
        if type(route) is not AccessRoute or type(intent) is not DispatchIntent:
            return None
        policy = route.policy
        origin = policy.origin
        if (
            type(policy) is not RoutePolicy
            or type(origin) is not ExactOrigin
            or type(policy.limits) is not RouteLimits
            or type(intent.limits) is not RouteLimits
            or type(route.route_id) is not str
            or type(intent.route_id) is not str
            or type(policy.policy_version) is not str
            or type(policy.policy_digest) is not str
            or type(intent.policy_digest) is not str
            or type(origin.scheme) is not str
            or type(origin.host) is not str
            or type(origin.port) is not int
            or type(intent.method) is not str
            or type(intent.operation_kind) is not OperationKind
            or type(intent.path) is not str
            or type(policy.methods) is not tuple
            or type(policy.paths) is not tuple
            or (policy.path_template is not None and type(policy.path_template) is not PathTemplate)
            or type(intent.query_pairs) is not tuple
            or type(intent.json_body_pairs) is not tuple
            or type(intent.query_bindings) is not tuple
            or type(policy.allowed_query_keys) is not tuple
            or type(policy.query_value_policies) is not tuple
            or type(policy.allowed_json_body_keys) is not tuple
            or type(policy.integer_json_body_keys) is not tuple
            or (policy.pagination_query_key is not None and type(policy.pagination_query_key) is not str)
            or (policy.pagination_json_body_key is not None and type(policy.pagination_json_body_key) is not str)
            or any(type(value) is not str for value in policy.methods)
            or any(type(value) is not str for value in policy.paths)
            or any(type(value) is not str for value in policy.allowed_query_keys)
            or any(type(value) is not str for value in policy.allowed_json_body_keys)
            or any(type(value) is not str for value in policy.integer_json_body_keys)
            or any(
                type(getattr(limits, name)) is not int
                for limits in (policy.limits, intent.limits)
                for name in (
                    "max_pages",
                    "max_redirects",
                    "max_retries",
                    "timeout_ms",
                    "max_response_bytes",
                    "max_results",
                    "max_request_body_bytes",
                )
            )
        ):
            return None
        query_value_policies = _snapshot_query_value_policies(policy.query_value_policies)
        if query_value_policies is None:
            return None
        if policy.path_template is None:
            path_template = None
        else:
            path_template = _snapshot_path_template(policy.path_template)
            if path_template is None:
                return None
        query_policy_names = tuple(rule.name for rule in query_value_policies)
        if len(set(query_policy_names)) != len(query_policy_names) or (
            query_value_policies and set(query_policy_names) != set(policy.allowed_query_keys)
        ):
            return None
        digest = canonical_policy_digest(policy)
        if (
            route.route_id != intent.route_id
            or policy.policy_digest != digest
            or intent.policy_digest != digest
            or route.credential_requirement is not CredentialRequirement.NONE
            or intent.method not in policy.methods
            or bool(policy.paths) == (path_template is not None)
            or (
                (path_template is None and intent.path not in policy.paths)
                or (path_template is not None and not _valid_template_path(path_template, intent.path))
            )
            or intent.limits != policy.limits
            or len(intent.query_pairs) > len(policy.allowed_query_keys)
            or len(intent.json_body_pairs) > len(policy.allowed_json_body_keys)
            or bool(intent.query_bindings)
            or (bool(intent.json_body_pairs) and intent.method != "POST")
            or (bool(policy.allowed_json_body_keys) and "POST" not in policy.methods)
            or bool(set(policy.allowed_query_keys).intersection(policy.allowed_json_body_keys))
            or len(set(policy.integer_json_body_keys)) != len(policy.integer_json_body_keys)
            or not set(policy.integer_json_body_keys).issubset(policy.allowed_json_body_keys)
            or (
                policy.pagination_query_key is not None and policy.pagination_query_key not in policy.allowed_query_keys
            )
            or (
                policy.pagination_json_body_key is not None
                and policy.pagination_json_body_key not in policy.allowed_json_body_keys
            )
            or sum(
                (
                    policy.pagination_query_key is not None,
                    policy.pagination_json_body_key is not None,
                    path_template is not None and path_template.pagination_segment_index is not None,
                )
            )
            > 1
            or (
                policy.pagination_json_body_key is not None
                and policy.pagination_json_body_key not in policy.integer_json_body_keys
            )
        ):
            return None

        allowed_query_keys = set(policy.allowed_query_keys)
        query_policies_by_name = {rule.name: rule for rule in query_value_policies}
        query_pairs: list[tuple[str, str]] = []
        query_keys: list[str] = []
        seen: set[str] = set()
        for pair in intent.query_pairs:
            if type(pair) is not QueryPair or type(pair.name) is not str or type(pair.value) is not str:
                return None
            validated_pair = QueryPair(pair.name, pair.value)
            if validated_pair.name not in allowed_query_keys or validated_pair.name in seen:
                return None
            value_policy = query_policies_by_name.get(validated_pair.name)
            if value_policy is not None and not _valid_query_policy_value(
                value_policy,
                validated_pair.value,
            ):
                return None
            seen.add(validated_pair.name)
            query_pairs.append((validated_pair.name, validated_pair.value))
            query_keys.append(validated_pair.name)
        if any(rule.required and rule.name not in seen for rule in query_value_policies):
            return None

        allowed_json_body_keys = set(policy.allowed_json_body_keys)
        integer_json_body_keys = set(policy.integer_json_body_keys)
        json_body_pairs: list[tuple[str, str | int]] = []
        seen_body_keys: set[str] = set()
        for pair in intent.json_body_pairs:
            if type(pair) is not JSONBodyPair or type(pair.name) is not str or type(pair.value) not in {str, int}:
                return None
            validated_pair = JSONBodyPair(pair.name, pair.value)
            if validated_pair.name not in allowed_json_body_keys or validated_pair.name in seen_body_keys:
                return None
            if validated_pair.name in integer_json_body_keys:
                if type(validated_pair.value) is not int:
                    return None
            elif type(validated_pair.value) is not str:
                return None
            seen_body_keys.add(validated_pair.name)
            json_body_pairs.append((validated_pair.name, validated_pair.value))
        if policy.pagination_json_body_key is not None:
            pagination_values = tuple(
                value for name, value in json_body_pairs if name == policy.pagination_json_body_key
            )
            if len(pagination_values) != 1 or type(pagination_values[0]) is not int:
                return None

        http_limits = _hop_limits(intent.limits)
        return _BindingSnapshot(
            route_id=route.route_id,
            policy_digest=digest,
            scheme=origin.scheme,
            host=origin.host,
            port=origin.port,
            method=intent.method,
            operation_kind=intent.operation_kind,
            path=intent.path,
            allowed_query_keys=policy.allowed_query_keys,
            query_pairs=tuple(query_pairs),
            query_keys=tuple(query_keys),
            json_body_pairs=tuple(json_body_pairs),
            timeout_ms=intent.limits.timeout_ms,
            max_response_bytes=intent.limits.max_response_bytes,
            route_limits=replace(intent.limits),
            http_limits=http_limits,
        )
    except Exception:  # noqa: BLE001 - hostile mutated contracts fail closed.
        return None


def _build_target(binding: _BindingSnapshot) -> str | None:
    """Percent-encode a bounded target without constructing an oversized aggregate."""
    limit = binding.http_limits.max_request_target_bytes
    if len(binding.path) > limit:
        return None
    try:
        path_bytes = binding.path.encode("ascii")
    except UnicodeError:
        return None
    if len(path_bytes) > limit:
        return None

    parts: list[str] = []
    total_bytes = len(path_bytes)
    for name, value in binding.query_pairs:
        if len(name) > limit or len(value) > limit:
            return None
        try:
            name_bytes = name.encode("ascii")
            value_bytes = value.encode("utf-8", errors="strict")
        except UnicodeError:
            return None
        if len(name_bytes) > limit or len(value_bytes) > limit:
            return None
        encoded_name = quote_from_bytes(name_bytes, safe="")
        encoded_value = quote_from_bytes(value_bytes, safe="")
        separator_bytes = 1
        pair_bytes = len(encoded_name) + 1 + len(encoded_value)
        if pair_bytes > limit - total_bytes - separator_bytes:
            return None
        total_bytes += separator_bytes + pair_bytes
        parts.append(f"{encoded_name}={encoded_value}")

    return binding.path if not parts else f"{binding.path}?{'&'.join(parts)}"


def _build_request(binding: _BindingSnapshot) -> NormalizedHTTPHopRequest | None:
    target = _build_target(binding)
    if target is None:
        return None
    headers: tuple[tuple[str, str], ...] = ()
    body = b""
    if binding.json_body_pairs:
        if binding.method != "POST":
            return None
        if (
            sum(
                len(name) + (len(value) if type(value) is str else len(str(value)))
                for name, value in binding.json_body_pairs
            )
            > binding.http_limits.max_request_body_bytes
        ):
            return None
        try:
            body = json.dumps(
                dict(binding.json_body_pairs),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8", errors="strict")
        except (TypeError, ValueError, UnicodeError):
            return None
        if len(body) > binding.http_limits.max_request_body_bytes:
            return None
        headers = (("content-type", "application/json"),)
    try:
        return NormalizedHTTPHopRequest(
            scheme=binding.scheme,
            host=binding.host,
            port=binding.port,
            method=binding.method,
            target=target,
            headers=headers,
            body=body,
            limits=replace(binding.http_limits),
        )
    except Exception:  # noqa: BLE001 - mutated scalar subclasses fail closed.
        return None


def reconstruct_redirect_intent(
    route: AccessRoute,
    prior_intent: DispatchIntent,
    location: str,
) -> DispatchIntent | None:
    """Rebuild one same-origin redirect as policy-bound declarative work."""
    binding = _snapshot_binding(route, prior_intent)
    if (
        binding is None
        or type(location) is not str
        or binding.method not in {"GET", "HEAD"}
        or binding.json_body_pairs
        or not location
        or "#" in location
        or len(location) > binding.http_limits.max_request_target_bytes
        or location != location.strip(" \t")
        or not location.isascii()
        or any(not " " <= character <= "~" for character in location)
    ):
        return None
    try:
        parsed = urlsplit(location)
        if parsed.fragment or parsed.username is not None or parsed.password is not None:
            return None
        if parsed.scheme or parsed.netloc:
            if not parsed.scheme or not parsed.netloc:
                return None
            default_port = 443 if parsed.scheme == "https" else 80
            if parsed.netloc.endswith(":"):
                return None
            parsed_port = parsed.port
            effective_port = default_port if parsed_port is None else parsed_port
            if parsed.scheme != binding.scheme or parsed.hostname != binding.host or effective_port != binding.port:
                return None
            resolved = parsed
        else:
            resolved = urlsplit(urljoin(binding.path, location))
            if resolved.scheme or resolved.netloc or resolved.fragment:
                return None
        if resolved.path != binding.path or _BAD_PERCENT_ESCAPE.search(resolved.query):
            return None
        raw_pairs = parse_qsl(
            resolved.query,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=len(binding.allowed_query_keys),
            separator="&",
        )
        names = tuple(name for name, _value in raw_pairs)
        if len(set(names)) != len(names) or any(name not in binding.allowed_query_keys for name in names):
            return None
        if frozenset(raw_pairs) != frozenset(binding.query_pairs):
            return None
        pairs = tuple(QueryPair(name, value) for name, value in raw_pairs)
        redirected = DispatchIntent(
            route_id=binding.route_id,
            policy_digest=binding.policy_digest,
            operation_kind=binding.operation_kind,
            method=binding.method,
            path=resolved.path,
            query_pairs=pairs,
            limits=replace(binding.route_limits),
        )
    except (TypeError, ValueError, UnicodeError):
        return None
    return redirected


def _active(
    check: PolicyActivityCheck,
    route_id: str,
    policy_digest: str,
) -> bool:
    try:
        return check(route_id, policy_digest) is True
    except Exception:  # noqa: BLE001 - revocation checks must fail closed.
        return False


def _is_allowed_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        not address.is_global
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_private
    ):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None or address.is_site_local:
            return False
        if any(address in network for network in _DENIED_IPV6_TRANSITION_NETWORKS):
            return False
    return True


def _valid_ip_evidence(resolved_ips: object, connected_ip: object) -> bool:
    if (
        type(resolved_ips) is not tuple
        or not resolved_ips
        or type(connected_ip) is not str
        or connected_ip not in resolved_ips
    ):
        return False
    seen: set[str] = set()
    for raw_address in resolved_ips:
        if type(raw_address) is not str or not raw_address or len(raw_address) > 45 or "%" in raw_address:
            return False
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError:
            return False
        canonical = address.compressed
        if raw_address != canonical or canonical in seen or not _is_allowed_public_address(address):
            return False
        seen.add(canonical)
    return True


def _visible_ascii(value: str, max_bytes: int) -> str | None:
    if len(value) > max_bytes:
        return None
    normalized = value.strip(" \t")
    if not normalized or len(normalized) > max_bytes:
        return None
    if not normalized.isascii() or any(not " " <= character <= "~" for character in normalized):
        return None
    return normalized


def _snapshot_headers(
    headers: object,
    *,
    limits: HTTPHopLimits,
    response_header_bytes: int,
) -> tuple[tuple[tuple[str, str], ...], str | None, str | None] | None:
    if type(headers) is not tuple or len(headers) > limits.max_response_headers:
        return None

    minimum_wire_bytes = _MIN_FINAL_HEADER_BLOCK_BYTES
    content_types: list[str] = []
    locations: list[str] = []
    retry_afters: list[str] = []
    for pair in headers:
        if type(pair) is not tuple or len(pair) != 2:
            return None
        name, value = pair
        if type(name) is not str or type(value) is not str:
            return None
        if len(name) > limits.max_response_header_bytes or len(value) > limits.max_response_header_bytes:
            return None
        try:
            name_bytes = name.encode("ascii")
            value_bytes = value.encode("latin-1")
        except UnicodeError:
            return None
        if not name_bytes or _HEADER_NAME_PATTERN.fullmatch(name) is None:
            return None
        if any(ord(character) < 32 and character != "\t" or ord(character) == 127 for character in value):
            return None
        minimum_wire_bytes += len(name_bytes) + 1 + len(value_bytes) + 1
        if minimum_wire_bytes > response_header_bytes or minimum_wire_bytes > limits.max_response_header_bytes:
            return None

        normalized_name = name.lower()
        if normalized_name == "content-type":
            content_types.append(value)
        elif normalized_name == "location":
            locations.append(value)
        elif normalized_name == "retry-after":
            retry_afters.append(value)

    if len(content_types) > 1 or len(locations) > 1 or len(retry_afters) > 1:
        return None

    safe_headers: tuple[tuple[str, str], ...] = ()
    if content_types:
        content_type = _visible_ascii(content_types[0], _MAX_CONTENT_TYPE_BYTES)
        if content_type is not None:
            safe_headers = (("content-type", content_type),)

    redirect_location = None
    if locations:
        redirect_location = _visible_ascii(locations[0], limits.max_request_target_bytes)
        if redirect_location is None:
            return None

    retry_after = None
    if retry_afters:
        retry_after = _visible_ascii(retry_afters[0], _MAX_RETRY_AFTER_BYTES)
        if retry_after is None:
            return None

    return safe_headers, redirect_location, retry_after


def _valid_body_wire_evidence(
    *,
    method: str,
    status_code: int,
    body: bytes,
    wire_bytes: int,
    headers: tuple[tuple[str, str], ...],
) -> bool:
    if method == "HEAD" or status_code in {204, 304}:
        return not body and wire_bytes == 0
    if status_code == 205:
        if body:
            return False
        if wire_bytes == 0:
            return True
        transfer_encodings = tuple(
            value.strip().lower() for name, value in headers if name.lower() == "transfer-encoding"
        )
        return wire_bytes == 5 and transfer_encodings == ("chunked",)
    return not body or wire_bytes > 0


def _snapshot_response(response: object, limits: HTTPHopLimits, method: str) -> _ResponseSnapshot | None:
    try:
        if type(response) is not HTTPHopResponse:
            return None
        if (
            type(response.status_code) is not int
            or not 200 <= response.status_code <= 599
            or type(response.body) is not bytes
            or len(response.body) > limits.max_parser_input_bytes
            or type(response.response_header_bytes) is not int
            or not _MIN_FINAL_HEADER_BLOCK_BYTES <= response.response_header_bytes <= limits.max_response_header_bytes
            or type(response.wire_bytes) is not int
            or not 0 <= response.wire_bytes <= limits.max_wire_bytes
            or not _valid_ip_evidence(response.resolved_ips, response.connected_ip)
        ):
            return None
        metadata = _snapshot_headers(
            response.headers,
            limits=limits,
            response_header_bytes=response.response_header_bytes,
        )
        if metadata is None:
            return None
        if not _valid_body_wire_evidence(
            method=method,
            status_code=response.status_code,
            body=response.body,
            wire_bytes=response.wire_bytes,
            headers=response.headers,
        ):
            return None
        safe_headers, redirect_location, retry_after = metadata
        return _ResponseSnapshot(
            status_code=response.status_code,
            headers=safe_headers,
            body=response.body,
            resolved_ips=response.resolved_ips,
            connected_ip=response.connected_ip,
            response_header_bytes=response.response_header_bytes,
            wire_bytes=response.wire_bytes,
            redirect_location=redirect_location,
            retry_after=retry_after,
        )
    except Exception:  # noqa: BLE001 - hostile injected responses fail closed.
        return None


def _requested_host(binding: _BindingSnapshot) -> str:
    default_port = 443 if binding.scheme == "https" else 80
    return binding.host if binding.port == default_port else f"{binding.host}:{binding.port}"


def _safe_retryable(error: HTTPHopError) -> bool:
    try:
        value = error.retryable
    except Exception:  # noqa: BLE001 - never retain hostile provider state.
        return False
    return type(value) is bool and value is True


def _safe_timed_out(error: HTTPHopError) -> bool:
    try:
        value = error.code
    except Exception:  # noqa: BLE001 - never retain hostile provider state.
        return False
    return type(value) is str and value in _TIMEOUT_HOP_ERROR_CODES


async def dispatch_once(
    route: AccessRoute,
    intent: DispatchIntent,
    *,
    is_policy_active: PolicyActivityCheck,
    one_hop: OneHop = request_http_hop,
) -> DiscoveryGatewayResponse:
    """Validate and perform exactly one credential-free physical HTTP hop."""
    binding = _snapshot_binding(route, intent)
    if binding is None:
        raise _reject("request_rejected")
    request = _build_request(binding)
    if request is None:
        raise _reject("request_rejected")
    if not _active(is_policy_active, binding.route_id, binding.policy_digest):
        raise _reject("policy_inactive")
    if _snapshot_binding(route, intent) != binding:
        raise _reject("request_rejected")

    started_at = time.monotonic()
    mapped_error: DiscoveryGatewayError | None = None
    raw_response: HTTPHopResponse | None = None
    try:
        raw_response = await one_hop(request)
    except HTTPHopError as error:
        mapped_error = DiscoveryGatewayError(
            "hop_failed",
            retryable=_safe_retryable(error),
            timed_out=_safe_timed_out(error),
        )
    except Exception:  # noqa: BLE001 - never expose unexpected provider detail.
        mapped_error = DiscoveryGatewayError("hop_failed")
    elapsed_ms = max(0, int((time.monotonic() - started_at) * 1000))
    if mapped_error is not None:
        raise mapped_error

    response = _snapshot_response(raw_response, binding.http_limits, binding.method)
    if response is None:
        raise _reject("invalid_hop_response")

    trace = DiscoveryGatewayTrace(
        route_id=binding.route_id,
        policy_digest=binding.policy_digest,
        scheme=binding.scheme,
        requested_host=_requested_host(binding),
        tls_server_name=(binding.host if binding.scheme == "https" else None),
        port=binding.port,
        method=binding.method,
        path=binding.path,
        query_keys=binding.query_keys,
        timeout_ms=binding.timeout_ms,
        max_response_bytes=binding.max_response_bytes,
        http_limits=binding.http_limits,
        status_code=response.status_code,
        resolved_ips=response.resolved_ips,
        connected_ip=response.connected_ip,
        response_header_bytes=response.response_header_bytes,
        wire_bytes=response.wire_bytes,
        decoded_bytes=len(response.body),
        elapsed_ms=elapsed_ms,
    )
    return DiscoveryGatewayResponse(
        status_code=response.status_code,
        headers=response.headers,
        body=response.body,
        trace=trace,
        redirect_location=response.redirect_location,
        retry_after=response.retry_after,
    )


__all__ = [
    "DiscoveryGatewayError",
    "DiscoveryGatewayResponse",
    "DiscoveryGatewayTrace",
    "dispatch_once",
    "reconstruct_redirect_intent",
]
