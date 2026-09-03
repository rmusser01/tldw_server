from __future__ import annotations

import ipaddress
import math
import os
import socket
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from loguru import logger

from tldw_Server_API.app.core.stt_observability_context import (
    get_opaque_stt_endpoint_id,
)
from tldw_Server_API.app.core.testing import is_truthy

DEFAULT_ALLOWED_SCHEMES = {"http", "https"}
DEFAULT_ALLOWED_PORTS = (80, 443, 8080)
DEFAULT_ALLOWED_PORTS_ENV_VALUE = ",".join(str(port) for port in DEFAULT_ALLOWED_PORTS)
ALLOWLIST_ENV = "WORKFLOWS_EGRESS_ALLOWLIST"
DENYLIST_ENV = "WORKFLOWS_EGRESS_DENYLIST"
# Global variants (applied across all usages)
GLOBAL_ALLOWLIST_ENV = "EGRESS_ALLOWLIST"
GLOBAL_DENYLIST_ENV = "EGRESS_DENYLIST"
BLOCK_PRIVATE_ENV = "WORKFLOWS_EGRESS_BLOCK_PRIVATE"
ALLOWED_PORTS_ENV = "WORKFLOWS_EGRESS_ALLOWED_PORTS"
PROFILENAME = "WORKFLOWS_EGRESS_PROFILE"  # strict | permissive | custom

# Webhook-specific per-tenant allow/deny controls
WEBHOOK_ALLOWLIST_ENV = "WORKFLOWS_WEBHOOK_ALLOWLIST"
WEBHOOK_DENYLIST_ENV = "WORKFLOWS_WEBHOOK_DENYLIST"
DNS_RESOLVER_MAX_OUTSTANDING_ENV = "WORKFLOWS_EGRESS_DNS_MAX_OUTSTANDING"
DNS_RESOLVER_SLOT_WAIT_SECONDS_ENV = "WORKFLOWS_EGRESS_DNS_SLOT_WAIT_SECONDS"

_DNS_RESOLVER_MAX_OUTSTANDING_DEFAULT = 64
_DNS_RESOLVER_SLOT_WAIT_SECONDS_DEFAULT = 0.05
_SENSITIVE_LOG_HOST = "sensitive_endpoint"


def _log_invalid_dns_config(name: str, raw: object, default: int | float, reason: str) -> None:
    """Log invalid DNS resolver environment configuration with queryable fields."""
    logger.bind(
        env_var=name,
        raw_value=str(raw),
        default_value=default,
        reason=reason,
        event="invalid_egress_dns_config",
    ).warning("Invalid egress DNS configuration; using default")


def _positive_int_env(name: str, default: int) -> int:
    """Return a positive integer env override or the supplied default."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        _log_invalid_dns_config(name, raw, default, "invalid_integer")
        return default
    if value < 1:
        _log_invalid_dns_config(name, raw, default, "not_positive")
        return default
    return value


def _nonnegative_float_env(name: str, default: float) -> float:
    """Return a finite non-negative float env override or the supplied default."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        _log_invalid_dns_config(name, raw, default, "invalid_float")
        return default
    if not math.isfinite(value) or value < 0:
        _log_invalid_dns_config(name, raw, default, "not_finite_or_negative")
        return default
    return value


_DNS_RESOLVER_MAX_OUTSTANDING = _positive_int_env(
    DNS_RESOLVER_MAX_OUTSTANDING_ENV,
    _DNS_RESOLVER_MAX_OUTSTANDING_DEFAULT,
)
_DNS_RESOLVER_SLOTS = threading.BoundedSemaphore(_DNS_RESOLVER_MAX_OUTSTANDING)


PRIVATE_RANGES = [
    ipaddress.ip_network("0.0.0.0/8"),  # "this" network
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),  # carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/29"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),  # multicast
    ipaddress.ip_network("240.0.0.0/4"),  # reserved
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),  # unspecified
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::ffff:0:0/96"),  # IPv4-mapped IPv6
    ipaddress.ip_network("64:ff9b::/96"),  # IPv4/IPv6 translation
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
]

_METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(raw)
    for raw in (
        "169.254.169.254",
        "169.254.170.2",
        "169.254.170.23",
        "100.100.100.200",
        "168.63.129.16",
        "fd00:ec2::254",
    )
)
_SCOPED_LOCAL_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
)
_SCOPED_FORBIDDEN_SPECIAL_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.31.196.0/24"),
    ipaddress.ip_network("192.52.193.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("192.175.48.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("100::/64"),
    ipaddress.ip_network("2001::/23"),
    ipaddress.ip_network("2001:db8::/32"),
    ipaddress.ip_network("2002::/16"),
    ipaddress.ip_network("2620:4f:8000::/48"),
    ipaddress.ip_network("3fff::/20"),
    ipaddress.ip_network("5f00::/16"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
)


@dataclass(frozen=True)
class URLPolicyResult:
    allowed: bool
    reason: str | None = None
    resolved_ips: tuple[str, ...] = ()
    reason_code: str | None = None


def _normalize_hostname(host: str) -> str:
    if not host:
        return ""
    host = host.strip().rstrip(".")
    # Drop zone identifiers for IPv6 (e.g., fe80::1%eth0)
    if "%" in host:
        host = host.split("%", 1)[0]
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        host = host.lower()
    return host.lower()


def _canonical_ip_literal(host: str) -> str | None:
    """Return a canonical IP literal, leaving DNS hostname handling separate."""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def _canonical_origin(url: str) -> tuple[str, str, int]:
    """Return the canonical HTTP(S) origin for a URL.

    Host normalization deliberately matches the existing egress allow/deny-list
    behavior: IDNA is converted to ASCII, case and a terminal DNS dot are
    ignored, and bracketed IPv6 literals are represented without brackets.
    """
    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in DEFAULT_ALLOWED_SCHEMES:
            raise ValueError("Unsupported URL scheme")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URL userinfo is not allowed")
        host = _normalize_hostname(parsed.hostname or "")
        if not host:
            raise ValueError("URL must include a hostname")
        try:
            host = str(ipaddress.ip_address(host))
        except ValueError:
            pass
        port = parsed.port
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError("Invalid URL origin") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, port


@dataclass(frozen=True)
class ConfiguredEndpointScope:
    """Exact server-configured origin authorized for one local-provider call."""

    scheme: str
    host: str
    port: int

    @classmethod
    def from_url(cls, url: str) -> ConfiguredEndpointScope:
        """Create a scope from a trusted configured endpoint URL."""
        scheme, host, port = _canonical_origin(url)
        return cls(scheme=scheme, host=host, port=port)

    def matches(self, url: str) -> bool:
        """Return whether ``url`` has this scope's canonical exact origin."""
        try:
            return _canonical_origin(url) == (self.scheme, self.host, self.port)
        except ValueError:
            return False


def _get_allowlist(env_value: str | None) -> list[str]:
    if not env_value:
        return []
    entries = []
    for raw in env_value.split(","):
        val = raw.strip().lower()
        if not val:
            continue
        if val.startswith("."):
            val = val[1:]
        entries.append(_normalize_hostname(val))
    return entries


def _host_matches_allowlist(host: str, allowlist: Sequence[str]) -> bool:
    if not allowlist:
        return True
    for allowed in allowlist:
        if not allowed:
            continue
        if host == allowed:
            return True
        if host.endswith(f".{allowed}"):
            return True
    return False


def _dns_slot_wait_seconds(timeout_s: float) -> float:
    """Bound DNS slot wait time by the caller's resolver timeout budget."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return 0.0
    configured = _nonnegative_float_env(
        DNS_RESOLVER_SLOT_WAIT_SECONDS_ENV,
        _DNS_RESOLVER_SLOT_WAIT_SECONDS_DEFAULT,
    )
    return min(configured, timeout_s)


def _dns_log_fields(
    host: str,
    *,
    sensitive_observability: bool = False,
    endpoint_id: str | None = None,
    **fields: object,
) -> dict[str, object]:
    """Return a redacted, opaque, or legacy host identity for DNS logs."""
    opaque_id = endpoint_id or get_opaque_stt_endpoint_id()
    if opaque_id is not None:
        identity = {"endpoint_id": opaque_id}
    elif sensitive_observability:
        identity = {"host": _SENSITIVE_LOG_HOST}
    else:
        identity = {"host": host}
    identity.update(fields)
    return identity


def _release_dns_resolver_slot(
    host: str,
    reason: str,
    *,
    sensitive_observability: bool = False,
    endpoint_id: str | None = None,
) -> None:
    """Release one DNS resolver slot and log impossible double-release cases."""
    try:
        _DNS_RESOLVER_SLOTS.release()
    except ValueError as exc:
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            reason=reason,
            exception_type=type(exc).__name__,
            event="dns_resolver_slot_release_failed",
        )).debug("DNS resolver slot release failed")


def _remaining_dns_budget(start_time: float, timeout_s: float) -> float:
    """Return the remaining DNS timeout budget after elapsed wall-clock time."""
    return max(0.0, timeout_s - (time.monotonic() - start_time))


def _getaddrinfo_with_timeout(
    host: str,
    timeout_s: float = 2.0,
    *,
    sensitive_observability: bool = False,
) -> list[tuple]:
    """Resolve a host with fail-closed timeout and DNS worker saturation guards."""
    endpoint_id = get_opaque_stt_endpoint_id()
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            timeout_s=timeout_s,
            event="dns_resolver_invalid_timeout",
        )).warning("Invalid DNS resolver timeout; failing closed")
        return []

    start_time = time.monotonic()
    slot_wait_s = _dns_slot_wait_seconds(timeout_s)
    acquired = (
        _DNS_RESOLVER_SLOTS.acquire(blocking=False)
        if slot_wait_s <= 0
        else _DNS_RESOLVER_SLOTS.acquire(timeout=slot_wait_s)
    )
    if not acquired:
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            slot_wait_s=slot_wait_s,
            elapsed_s=time.monotonic() - start_time,
            timeout_s=timeout_s,
            event="dns_resolver_slots_exhausted",
        )).warning("DNS resolver slots exhausted; failing closed")
        return []

    remaining_s = _remaining_dns_budget(start_time, timeout_s)
    if remaining_s <= 0:
        _release_dns_resolver_slot(
            host,
            "timeout budget exhausted before worker start",
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
        )
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            elapsed_s=time.monotonic() - start_time,
            timeout_s=timeout_s,
            event="dns_resolver_timeout_budget_exhausted",
        )).warning("DNS resolver timeout budget exhausted before worker start; failing closed")
        return []

    result: list[list[tuple]] = []
    error_types: list[str] = []

    def _worker() -> None:
        """Run the blocking OS resolver and always release the resolver slot."""
        try:
            result.append(
                socket.getaddrinfo(
                    host,
                    None,
                    family=socket.AF_UNSPEC,  # both IPv4 and IPv6
                    type=socket.SOCK_STREAM,
                )
            )
        except Exception as exc:  # noqa: BLE001 - contain resolver worker failures
            error_types.append(type(exc).__name__)
        finally:
            _release_dns_resolver_slot(
                host,
                "worker completion",
                sensitive_observability=sensitive_observability,
                endpoint_id=endpoint_id,
            )

    thread = threading.Thread(target=_worker, daemon=True)
    try:
        thread.start()
    except RuntimeError as exc:
        _release_dns_resolver_slot(
            host,
            "thread start failure",
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
        )
        bound_logger = logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            exception_type=type(exc).__name__,
            event="dns_resolver_worker_start_failed",
        ))
        if sensitive_observability or endpoint_id is not None:
            bound_logger.warning("DNS resolver worker could not start; failing closed")
        else:
            bound_logger.opt(exception=exc).warning("DNS resolver worker could not start; failing closed")
        return []

    remaining_s = _remaining_dns_budget(start_time, timeout_s)
    if remaining_s <= 0:
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            elapsed_s=time.monotonic() - start_time,
            timeout_s=timeout_s,
            event="dns_resolver_timeout",
        )).warning("DNS resolver timed out before waiting for worker; failing closed")
        return []
    thread.join(remaining_s)
    if thread.is_alive():
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            elapsed_s=time.monotonic() - start_time,
            timeout_s=timeout_s,
            event="dns_resolver_timeout",
        )).warning("DNS resolver timed out; failing closed")
        return []
    if error_types:
        logger.bind(**_dns_log_fields(
            host,
            sensitive_observability=sensitive_observability,
            endpoint_id=endpoint_id,
            exception_type=error_types[0],
            event="dns_resolver_error",
        )).debug("DNS resolver failed; failing closed")
        return []
    return result[0] if result else []


def resolve_host_ips(
    host: str,
    timeout_s: float = 2.0,
    *,
    sensitive_observability: bool = False,
) -> tuple[str, ...]:
    """Resolve a host to every A/AAAA address within a bounded timeout.

    The wrapper deliberately does not read egress profiles or allowlists. It
    returns addresses in resolver order with duplicates removed. Any resolver
    or result-shape error fails closed as an empty tuple.
    """
    try:
        if sensitive_observability:
            infos = _getaddrinfo_with_timeout(
                host,
                timeout_s=timeout_s,
                sensitive_observability=True,
            )
        else:
            infos = _getaddrinfo_with_timeout(host, timeout_s=timeout_s)
        if not infos:
            return ()

        addrs: list[str] = []
        for info in infos:
            try:
                _family, _stype, _proto, _canon, sockaddr = info
                if not isinstance(sockaddr, tuple) or not sockaddr:
                    return ()
                # sockaddr[0] is the IP for both AF_INET and AF_INET6
                ip = sockaddr[0]
            except (IndexError, KeyError, TypeError, ValueError):
                return ()
            if not isinstance(ip, str):
                return ()
            addrs.append(ip)
        # Preserve order but deduplicate
        return tuple(dict.fromkeys(addrs))
    except (OSError, TypeError, ValueError) as exc:
        endpoint_id = get_opaque_stt_endpoint_id()
        if sensitive_observability or endpoint_id is not None:
            logger.bind(**_dns_log_fields(
                host,
                sensitive_observability=sensitive_observability,
                endpoint_id=endpoint_id,
                exception_type=type(exc).__name__,
                event="dns_resolver_failed_closed",
            )).debug("Host resolution failed; treating as unsafe")
        else:
            logger.debug(
                "Host resolution failed for {} with {}; treating as unsafe",
                host,
                type(exc).__name__,
            )
        return ()


def _resolve_host_ips(
    host: str,
    *,
    sensitive_observability: bool = False,
) -> list[str]:
    """Compatibility wrapper for callers expecting a mutable address list."""
    return list(
        resolve_host_ips(
            host,
            sensitive_observability=sensitive_observability,
        )
    )


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in PRIVATE_RANGES)
    except ValueError:
        # Treat parsing failures as private for safety
        return True


def _normalize_resolved_ips(ips: Sequence[str] | None) -> tuple[str, ...]:
    if not ips:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in ips:
        ip = str(raw).strip()
        if not ip or ip in seen:
            continue
        seen.add(ip)
        out.append(ip)
    return tuple(out)


def _same_resolved_ip_set(left: Sequence[str], right: Sequence[str]) -> bool:
    return {str(ip).strip() for ip in left if str(ip).strip()} == {str(ip).strip() for ip in right if str(ip).strip()}


def _resolve_and_check_private(
    host: str,
    *,
    sensitive_observability: bool = False,
) -> tuple[bool, list[str]]:
    ips: list[str] = []
    # If the host is already an IP address, check directly
    try:
        ipaddress.ip_address(host)
        ips = [host]
    except ValueError:
        if sensitive_observability:
            ips = _resolve_host_ips(host, sensitive_observability=True)
        else:
            ips = _resolve_host_ips(host)

    if not ips:
        return False, []

    for ip in ips:
        if _is_private_ip(ip):
            return False, ips
    return True, ips


def _resolve_host_or_literal(
    host: str,
    *,
    sensitive_observability: bool = False,
) -> list[str]:
    """Return a literal address or every DNS answer for a scoped hostname."""
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        if sensitive_observability:
            return _resolve_host_ips(host, sensitive_observability=True)
        return _resolve_host_ips(host)


def _is_approved_scoped_address(raw_ip: str) -> bool:
    """Allow local-provider destinations only from explicit safe classes."""
    try:
        address = ipaddress.ip_address(raw_ip)
    except ValueError:
        return False

    if address in _METADATA_ADDRESSES:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return False
    if any(address in network for network in _SCOPED_LOCAL_NETWORKS):
        return True
    if any(address in network for network in _SCOPED_FORBIDDEN_SPECIAL_NETWORKS):
        return False
    if (
        address.is_multicast
        or address.is_link_local
        or (isinstance(address, ipaddress.IPv6Address) and address.is_site_local)
        or address.is_unspecified
        or address.is_reserved
        or address.is_private
    ):
        return False
    return address.is_global


def _normalize_scoped_resolved_ips(ips: Sequence[str] | None) -> tuple[str, ...]:
    """Canonicalize and deduplicate scoped DNS answers without dropping errors."""
    if not ips:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in ips:
        text = str(raw).strip()
        try:
            value = str(ipaddress.ip_address(text))
        except ValueError:
            value = text
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def _should_block_private_env(block_private_override: bool | None = None) -> bool:
    if block_private_override is not None:
        return block_private_override
    env_value = os.getenv(BLOCK_PRIVATE_ENV, "true").lower()
    return is_truthy(env_value)


def evaluate_url_policy(
    url: str,
    *,
    allowlist: Sequence[str] | None = None,
    denylist: Sequence[str] | None = None,
    block_private_override: bool | None = None,
    resolved_ips_override: Sequence[str] | None = None,
    pinned_resolved_ips: Sequence[str] | None = None,
    configured_endpoint: ConfiguredEndpointScope | None = None,
    sensitive_observability: bool = False,
) -> URLPolicyResult:
    """Evaluate whether a URL passes the egress policy."""
    try:
        parsed = urlparse(url)
    except (TypeError, AttributeError, ValueError):
        return URLPolicyResult(False, "Invalid URL", reason_code="invalid_url")

    scheme = (parsed.scheme or "").lower()
    if scheme not in DEFAULT_ALLOWED_SCHEMES:
        return URLPolicyResult(False, "Unsupported URL scheme", reason_code="unsupported_scheme")

    if configured_endpoint is not None and (parsed.username is not None or parsed.password is not None):
        return URLPolicyResult(False, "URL userinfo is not allowed", reason_code="userinfo_not_allowed")

    try:
        host = _normalize_hostname(parsed.hostname or "")
    except ValueError:
        return URLPolicyResult(False, "Invalid URL", reason_code="invalid_url")
    if not host:
        return URLPolicyResult(False, "URL must include a hostname", reason_code="invalid_url")

    # Ports policy (defaults 80/443/8080; override via env)
    def _default_ports() -> list[int]:
        raw = os.getenv(ALLOWED_PORTS_ENV, DEFAULT_ALLOWED_PORTS_ENV_VALUE)
        tokens = [part.strip().lower() for part in (raw or "").split(",") if part.strip()]
        if any(token in {"*", "any", "all"} for token in tokens):
            return []
        out = []
        for p in tokens:
            try:
                out.append(int(p))
            except ValueError:
                continue
        return out or list(DEFAULT_ALLOWED_PORTS)

    allowed_ports = _default_ports()
    if (os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING")) and host in {"localhost", "127.0.0.1", "::1"}:
        allowed_ports = []
    try:
        port = parsed.port
    except ValueError:
        return URLPolicyResult(False, "Invalid URL port", reason_code="invalid_url")
    if port is None:
        port = 443 if scheme == "https" else 80
    if configured_endpoint is None and allowed_ports and port not in allowed_ports:
        return URLPolicyResult(False, f"Port not allowed: {port}", reason_code="port_not_allowed")

    allowlist = list(allowlist) if allowlist is not None else None
    if allowlist is None:
        # Merge global and workflows lists
        gl = _get_allowlist(os.getenv(GLOBAL_ALLOWLIST_ENV, ""))
        wl = _get_allowlist(os.getenv(ALLOWLIST_ENV, ""))
        allowlist = list(dict.fromkeys(gl + wl))
    denylist = list(denylist) if denylist is not None else None
    if denylist is None:
        gd = _get_allowlist(os.getenv(GLOBAL_DENYLIST_ENV, ""))
        wd = _get_allowlist(os.getenv(DENYLIST_ENV, ""))
        denylist = list(dict.fromkeys(gd + wd))

    # Profile handling: strict requires explicit allowlist match
    profile = (os.getenv(PROFILENAME, "") or "").strip().lower()
    if not profile:
        # Per-environment sensible defaults
        env = (os.getenv("ENVIRONMENT") or os.getenv("APP_ENV") or os.getenv("ENV") or "dev").lower()
        profile = "strict" if env in {"prod", "production"} else "permissive"

    # Denylist wins if provided
    if denylist:
        host_ip = _canonical_ip_literal(host)
        for denied in denylist:
            if not denied:
                continue
            if denied.startswith("."):
                denied = denied[1:]
            d = _normalize_hostname(denied)
            same_ip = host_ip is not None and host_ip == _canonical_ip_literal(d)
            if same_ip or host == d or host.endswith(f".{d}"):
                return URLPolicyResult(False, "Host in denylist", reason_code="host_denied")

    if configured_endpoint is not None:
        if not configured_endpoint.matches(url):
            return URLPolicyResult(False, "URL origin does not match configured endpoint", reason_code="origin_mismatch")

    if configured_endpoint is None:
        if profile == "strict":
            if not allowlist:
                return URLPolicyResult(False, "No allowlist configured (strict)", reason_code="host_denied")
            if not _host_matches_allowlist(host, allowlist):
                return URLPolicyResult(False, "Host not in allowlist", reason_code="host_denied")
        else:
            # permissive/custom: if allowlist provided, enforce; else accept any public host
            if allowlist and not _host_matches_allowlist(host, allowlist):
                return URLPolicyResult(False, "Host not in allowlist", reason_code="host_denied")

    resolved_ips: tuple[str, ...] = ()
    if configured_endpoint is not None:
        raw_ips = (
            list(resolved_ips_override)
            if resolved_ips_override is not None
            else _resolve_host_or_literal(
                host,
                sensitive_observability=sensitive_observability,
            )
        )
        resolved_ips = _normalize_scoped_resolved_ips(raw_ips)
        if not resolved_ips:
            return URLPolicyResult(False, "Host could not be resolved", reason_code="dns_unresolved")
        if any(not _is_approved_scoped_address(ip) for ip in resolved_ips):
            return URLPolicyResult(
                False,
                "URL resolves to a forbidden address",
                resolved_ips,
                "address_forbidden",
            )
        pinned_ips = _normalize_scoped_resolved_ips(pinned_resolved_ips)
        if pinned_ips and not _same_resolved_ip_set(resolved_ips, pinned_ips):
            return URLPolicyResult(
                False,
                "DNS resolution changed since policy check",
                resolved_ips,
                "dns_changed",
            )
    elif _should_block_private_env(block_private_override):
        if resolved_ips_override is not None:
            resolved_ips = _normalize_resolved_ips(resolved_ips_override)
            if not resolved_ips:
                return URLPolicyResult(False, "Host could not be resolved", reason_code="dns_unresolved")
            if any(_is_private_ip(ip) for ip in resolved_ips):
                return URLPolicyResult(
                    False,
                    "URL resolves to a private or reserved address",
                    resolved_ips,
                    "address_forbidden",
                )
        else:
            if sensitive_observability:
                ok, ips = _resolve_and_check_private(
                    host,
                    sensitive_observability=True,
                )
            else:
                ok, ips = _resolve_and_check_private(host)
            resolved_ips = _normalize_resolved_ips(ips)
            if not ok:
                if not resolved_ips:
                    return URLPolicyResult(False, "Host could not be resolved", reason_code="dns_unresolved")
                return URLPolicyResult(
                    False,
                    "URL resolves to a private or reserved address",
                    resolved_ips,
                    "address_forbidden",
                )
        pinned_ips = _normalize_resolved_ips(pinned_resolved_ips)
        if pinned_ips and not _same_resolved_ip_set(resolved_ips, pinned_ips):
            return URLPolicyResult(
                False,
                "DNS resolution changed since policy check",
                resolved_ips,
                "dns_changed",
            )
    else:
        resolved_ips = _normalize_resolved_ips(resolved_ips_override)
        pinned_ips = _normalize_resolved_ips(pinned_resolved_ips)
        if pinned_ips and resolved_ips and not _same_resolved_ip_set(resolved_ips, pinned_ips):
            return URLPolicyResult(
                False,
                "DNS resolution changed since policy check",
                resolved_ips,
                "dns_changed",
            )

    return URLPolicyResult(True, None, resolved_ips)


def is_private_ip(ip: str) -> bool:
    """Public helper retained for compatibility."""
    return _is_private_ip(ip)


def is_url_allowed(url: str) -> bool:
    """Check egress policy for a URL using env allowlist and private IP blocks."""
    result = evaluate_url_policy(url)
    return result.allowed


def _parse_list_env(value: str | None) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for raw in value.split(","):
        v = raw.strip()
        if not v:
            continue
        if v.startswith("*."):
            v = v[2:]
        out.append(_normalize_hostname(v))
    return out


def evaluate_platform_webhook_url_policy(url: str) -> URLPolicyResult:
    """Evaluate a platform webhook target with all global policy families."""
    allowlist = list(
        dict.fromkeys(
            [
                *_get_allowlist(os.getenv(GLOBAL_ALLOWLIST_ENV, "")),
                *_get_allowlist(os.getenv(ALLOWLIST_ENV, "")),
                *_parse_list_env(os.getenv(WEBHOOK_ALLOWLIST_ENV)),
            ]
        )
    )
    denylist = list(
        dict.fromkeys(
            [
                *_get_allowlist(os.getenv(GLOBAL_DENYLIST_ENV, "")),
                *_get_allowlist(os.getenv(DENYLIST_ENV, "")),
                *_parse_list_env(os.getenv(WEBHOOK_DENYLIST_ENV)),
            ]
        )
    )
    return evaluate_url_policy(
        url,
        allowlist=allowlist,
        denylist=denylist,
        block_private_override=True,
        sensitive_observability=True,
    )


def evaluate_admin_webhook_e2e_loopback_policy(url: str) -> URLPolicyResult:
    """Allow only the exact IPv4 loopback target used by isolated admin E2E."""
    try:
        parsed = urlparse(url)
        host = _normalize_hostname(parsed.hostname or "")
    except (TypeError, ValueError):
        parsed = None
        host = ""
    if (
        parsed is None
        or parsed.scheme.lower() != "http"
        or host != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        return URLPolicyResult(
            False,
            "Admin webhook E2E loopback target denied",
            reason_code="address_forbidden",
        )
    return evaluate_url_policy(
        url,
        allowlist=[],
        denylist=[],
        block_private_override=False,
        resolved_ips_override=("127.0.0.1",),
        sensitive_observability=True,
    )


def is_webhook_url_allowed_for_tenant(url: str, tenant_id: str) -> bool:
    """Webhook egress evaluation with per-tenant allow/deny lists.

    Env:
      - WORKFLOWS_WEBHOOK_ALLOWLIST, WORKFLOWS_WEBHOOK_DENYLIST (global)
      - WORKFLOWS_WEBHOOK_ALLOWLIST_<TENANT>, WORKFLOWS_WEBHOOK_DENYLIST_<TENANT>
      - WORKFLOWS_EGRESS_BLOCK_PRIVATE (applies to webhooks too)
    """
    import os

    t_key = (tenant_id or "default").upper().replace("-", "_")
    allow = _parse_list_env(os.getenv(f"{WEBHOOK_ALLOWLIST_ENV}_{t_key}") or os.getenv(WEBHOOK_ALLOWLIST_ENV))
    deny = _parse_list_env(os.getenv(f"{WEBHOOK_DENYLIST_ENV}_{t_key}") or os.getenv(WEBHOOK_DENYLIST_ENV))
    result = evaluate_url_policy(url, allowlist=allow if allow else None, denylist=deny if deny else None)
    return result.allowed


def is_url_allowed_for_tenant(url: str, tenant_id: str) -> bool:
    """General egress evaluation with per-tenant overrides.

    Env:
      - WORKFLOWS_EGRESS_ALLOWLIST, WORKFLOWS_EGRESS_DENYLIST (global)
      - WORKFLOWS_EGRESS_ALLOWLIST_<TENANT>, WORKFLOWS_EGRESS_DENYLIST_<TENANT>
      - WORKFLOWS_EGRESS_BLOCK_PRIVATE, WORKFLOWS_EGRESS_PROFILE

    Precedence:
      - Deny at any level wins (global or tenant)
      - Allow lists are unioned (host allowed if present in either global or tenant allow)
      - If no allowlists provided, permissive profile allows public hosts; strict requires allow match
    """
    t_key = (tenant_id or "default").upper().replace("-", "_")
    # Tenant overrides fall back to global lists
    global_allow = _get_allowlist(os.getenv(ALLOWLIST_ENV, ""))
    global_deny = _get_allowlist(os.getenv(DENYLIST_ENV, ""))
    tenant_allow = _get_allowlist(os.getenv(f"{ALLOWLIST_ENV}_{t_key}", ""))
    tenant_deny = _get_allowlist(os.getenv(f"{DENYLIST_ENV}_{t_key}", ""))
    # Deny is union
    deny = list(dict.fromkeys([*global_deny, *tenant_deny]))
    # Allow is union; empty means no constraint for permissive profile
    allow = list(dict.fromkeys([*global_allow, *tenant_allow]))
    return evaluate_url_policy(url, allowlist=(allow or None), denylist=(deny or None)).allowed
