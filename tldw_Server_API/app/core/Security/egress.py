from __future__ import annotations

import ipaddress
import os
import socket
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from tldw_Server_API.app.core.testing import is_truthy

DEFAULT_ALLOWED_SCHEMES = {"http", "https"}
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

_DNS_RESOLVER_MAX_OUTSTANDING = 8
_DNS_RESOLVER_SLOTS = threading.BoundedSemaphore(_DNS_RESOLVER_MAX_OUTSTANDING)


PRIVATE_RANGES = [
    ipaddress.ip_network("0.0.0.0/8"),       # "this" network
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),   # carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/29"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),     # multicast
    ipaddress.ip_network("240.0.0.0/4"),     # reserved
    ipaddress.ip_network("255.255.255.255/32"),
    ipaddress.ip_network("::/128"),          # unspecified
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::ffff:0:0/96"),   # IPv4-mapped IPv6
    ipaddress.ip_network("64:ff9b::/96"),    # IPv4/IPv6 translation
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
]


@dataclass(frozen=True)
class URLPolicyResult:
    allowed: bool
    reason: str | None = None
    resolved_ips: tuple[str, ...] = ()


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


def _getaddrinfo_with_timeout(host: str, timeout_s: float = 2.0) -> list[tuple]:
    if not _DNS_RESOLVER_SLOTS.acquire(blocking=False):
        return []

    result: list[list[tuple]] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(
                socket.getaddrinfo(
                    host,
                    None,
                    family=socket.AF_UNSPEC,  # both IPv4 and IPv6
                    type=socket.SOCK_STREAM,
                )
            )
        except (OSError, ValueError) as exc:
            error.append(exc)
        finally:
            try:
                _DNS_RESOLVER_SLOTS.release()
            except ValueError:
                pass

    thread = threading.Thread(target=_worker, daemon=True)
    try:
        thread.start()
    except RuntimeError:
        try:
            _DNS_RESOLVER_SLOTS.release()
        except ValueError:
            pass
        return []
    thread.join(timeout_s)
    if thread.is_alive() or error:
        return []
    return result[0] if result else []


def _resolve_host_ips(host: str) -> list[str]:
    """Resolve the host to all A/AAAA addresses with a short timeout.

    Returns a de-duplicated list of IP strings. Any resolution error results
    in an empty list which callers must treat as unsafe.
    """
    try:
        infos = _getaddrinfo_with_timeout(host)
        if not infos:
            return []

        addrs: list[str] = []
        for _family, _stype, _proto, _canon, sockaddr in infos:
            try:
                # sockaddr[0] is the IP for both AF_INET and AF_INET6
                ip = sockaddr[0]
                addrs.append(ip)
            except (IndexError, TypeError):
                continue
        # Preserve order but deduplicate
        return list(dict.fromkeys(addrs))
    except (OSError, TypeError, ValueError):
        return []


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
    return {str(ip).strip() for ip in left if str(ip).strip()} == {
        str(ip).strip() for ip in right if str(ip).strip()
    }


def _resolve_and_check_private(host: str) -> tuple[bool, list[str]]:
    ips: list[str] = []
    # If the host is already an IP address, check directly
    try:
        ipaddress.ip_address(host)
        ips = [host]
    except ValueError:
        ips = _resolve_host_ips(host)

    if not ips:
        return False, []

    for ip in ips:
        if _is_private_ip(ip):
            return False, ips
    return True, ips


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
) -> URLPolicyResult:
    """Evaluate whether a URL passes the egress policy."""
    try:
        parsed = urlparse(url)
    except (TypeError, AttributeError, ValueError):
        return URLPolicyResult(False, "Invalid URL")

    scheme = (parsed.scheme or "").lower()
    if scheme not in DEFAULT_ALLOWED_SCHEMES:
        return URLPolicyResult(False, "Unsupported URL scheme")

    host = _normalize_hostname(parsed.hostname or "")
    if not host:
        return URLPolicyResult(False, "URL must include a hostname")

    # Ports policy (defaults 80/443; override via env)
    def _default_ports() -> list[int]:
        raw = os.getenv(ALLOWED_PORTS_ENV, "80,443")
        tokens = [part.strip().lower() for part in (raw or "").split(",") if part.strip()]
        if any(token in {"*", "any", "all"} for token in tokens):
            return []
        out = []
        for p in tokens:
            try:
                out.append(int(p))
            except ValueError:
                continue
        return out or [80, 443]

    allowed_ports = _default_ports()
    if (os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING")) and host in {"localhost", "127.0.0.1", "::1"}:
        allowed_ports = []
    try:
        port = parsed.port
    except ValueError:
        return URLPolicyResult(False, "Invalid URL port")
    if port is None:
        port = 443 if scheme == "https" else 80
    if allowed_ports and port not in allowed_ports:
        return URLPolicyResult(False, f"Port not allowed: {port}")

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
        for denied in denylist:
            if not denied:
                continue
            if denied.startswith("."):
                denied = denied[1:]
            d = _normalize_hostname(denied)
            if host == d or host.endswith(f".{d}"):
                return URLPolicyResult(False, "Host in denylist")

    if profile == "strict":
        if not allowlist:
            return URLPolicyResult(False, "No allowlist configured (strict)")
        if not _host_matches_allowlist(host, allowlist):
            return URLPolicyResult(False, "Host not in allowlist")
    else:
        # permissive/custom: if allowlist provided, enforce; else accept any public host
        if allowlist and not _host_matches_allowlist(host, allowlist):
            return URLPolicyResult(False, "Host not in allowlist")

    resolved_ips: tuple[str, ...] = ()
    if _should_block_private_env(block_private_override):
        if resolved_ips_override is not None:
            resolved_ips = _normalize_resolved_ips(resolved_ips_override)
            if not resolved_ips:
                return URLPolicyResult(False, "Host could not be resolved")
            if any(_is_private_ip(ip) for ip in resolved_ips):
                return URLPolicyResult(False, "URL resolves to a private or reserved address", resolved_ips)
        else:
            ok, ips = _resolve_and_check_private(host)
            resolved_ips = _normalize_resolved_ips(ips)
            if not ok:
                if not resolved_ips:
                    return URLPolicyResult(False, "Host could not be resolved")
                return URLPolicyResult(False, "URL resolves to a private or reserved address", resolved_ips)
        pinned_ips = _normalize_resolved_ips(pinned_resolved_ips)
        if pinned_ips and not _same_resolved_ip_set(resolved_ips, pinned_ips):
            return URLPolicyResult(False, "DNS resolution changed since policy check", resolved_ips)
    else:
        resolved_ips = _normalize_resolved_ips(resolved_ips_override)
        pinned_ips = _normalize_resolved_ips(pinned_resolved_ips)
        if pinned_ips and resolved_ips and not _same_resolved_ip_set(resolved_ips, pinned_ips):
            return URLPolicyResult(False, "DNS resolution changed since policy check", resolved_ips)

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
            v = v[1:]
        out.append(_normalize_hostname(v))
    return out


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
