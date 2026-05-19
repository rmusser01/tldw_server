from __future__ import annotations

import contextlib
import ipaddress
import socket
import subprocess
from collections.abc import Iterable, Sequence
from typing import Callable

from loguru import logger
from tldw_Server_API.app.core.testing import is_truthy

_SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    IndexError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    subprocess.SubprocessError,
)
_IPTABLES_CLEANUP_TIMEOUT_SEC = 5


def _truthy(v: str | None) -> bool:
    return is_truthy(v)


def default_resolver(host: str) -> list[str]:
    """Resolve hostname to IPv4 addresses using getaddrinfo; return unique list.

    Avoids raising on DNS errors, returns empty list if resolution fails.
    """
    out: list[str] = []
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
        for _fam, _sock, _proto, _canon, sa in infos:
            ip = sa[0]
            if ip not in out:
                out.append(ip)
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS:
        return []
    return out


def _normalize_host_token(token: str) -> tuple[str, bool, bool]:
    """Normalize a hostname-like token and detect wildcard/suffix semantics.

    Returns (host, is_wildcard, is_suffix).
    - Accepts tokens like '*.example.com', '.example.com', 'https://example.com'.
    - Lowercases host, strips trailing dot, removes URL schemes.
    """
    tok = str(token).strip()
    # Drop URL scheme if present
    for scheme in ("http://", "https://"):
        if tok.lower().startswith(scheme):
            tok = tok[len(scheme):]
            break
    # Strip path/port if accidentally included
    # e.g., example.com:80/foo -> example.com
    for sep in ("/", ":"):
        if sep in tok:
            tok = tok.split(sep, 1)[0]
    is_wild = False
    is_suffix = False
    if tok.startswith("*."):
        is_wild = True
        tok = tok[2:]
    elif tok.startswith('.'):
        # Suffix-style token: treat like wildcard for a domain suffix
        is_wild = True
        is_suffix = True
        tok = tok[1:]
    host = tok.rstrip('.').lower()
    return host, is_wild, is_suffix


def expand_allowlist_to_targets(
    raw_allowlist: Sequence[str] | str | None,
    *,
    resolver: Callable[[str], list[str]] = default_resolver,
    wildcard_subdomains: Sequence[str] | None = ("", "www", "api"),
) -> list[str]:
    """Expand allowlist inputs (CIDR, IP, hostname, wildcard *.domain) into CIDR/IP targets.

    - CIDR tokens: validated and returned as-is
    - IP tokens: promoted to /32
    - Hostname tokens: resolved to A records and promoted to /32
    - Wildcard tokens (e.g., "*.example.com"): resolve a small set of commonly used subdomains
      ("", "www", "api") plus the apex. This is a pragmatic compromise until a dedicated
      DNS pinning/proxy mechanism is available.
    Returns a de-duplicated list of strings like "1.2.3.4/32" or "1.2.3.0/24" sorted for stability.
    """
    if raw_allowlist is None:
        return []
    if isinstance(raw_allowlist, str):
        tokens = [t.strip() for t in raw_allowlist.split(',') if t.strip()]
    else:
        tokens = [str(t).strip() for t in raw_allowlist if str(t).strip()]
    results: set[str] = set()
    for tok in tokens:
        # CIDR
        try:
            if "/" in tok:
                _ = ipaddress.ip_network(tok, strict=False)
                results.add(str(tok))
                continue
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("network policy: CIDR parse failed for {}: {}", tok, e)
        # Literal IP
        try:
            ipaddress.ip_address(tok)
            results.add(f"{tok}/32")
            continue
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("network policy: IP parse failed for {}: {}", tok, e)
        # Hostname (supports wildcard prefix "*." and suffix ".domain")
        host, is_wild, _is_suffix = _normalize_host_token(tok)
        if not host:
            continue
        # Resolve apex and a small set of common subdomains for wildcard tokens
        to_resolve: list[str] = []
        if is_wild:
            subs = list(wildcard_subdomains or ("",))
            for sub in subs:
                fqdn = f"{sub}.{host}" if sub else host
                to_resolve.append(fqdn)
        else:
            to_resolve.append(host)
        for h in to_resolve:
            for ip in resolver(h):
                try:
                    ipaddress.ip_address(ip)
                    results.add(f"{ip}/32")
                except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS:
                    continue
    return sorted(results)


def pin_dns_map(
    raw_allowlist: Sequence[str] | str | None,
    *,
    resolver: Callable[[str], list[str]] = default_resolver,
    wildcard_subdomains: Sequence[str] | None = ("", "www", "api"),
) -> dict[str, list[str]]:
    """Return a mapping of normalized host tokens to resolved IPv4 addresses.

    CIDR and literal IP inputs are returned as themselves (keyed by the token),
    hostnames and wildcards are expanded and grouped by the base host.
    """
    if raw_allowlist is None:
        return {}
    if isinstance(raw_allowlist, str):
        tokens = [t.strip() for t in raw_allowlist.split(',') if t.strip()]
    else:
        tokens = [str(t).strip() for t in raw_allowlist if str(t).strip()]
    out: dict[str, list[str]] = {}
    for tok in tokens:
        # CIDR or IP
        try:
            if "/" in tok:
                ipaddress.ip_network(tok, strict=False)
                out.setdefault(tok, [])
                continue
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("network policy: CIDR parse failed for {}: {}", tok, e)
        try:
            ipaddress.ip_address(tok)
            out.setdefault(tok, [tok])
            continue
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("network policy: IP parse failed for {}: {}", tok, e)
        # Host tokens
        host, is_wild, _is_suffix = _normalize_host_token(tok)
        if not host:
            continue
        hosts: list[str] = []
        if is_wild:
            for sub in list(wildcard_subdomains or ("",)):
                fqdn = f"{sub}.{host}" if sub else host
                hosts.append(fqdn)
        else:
            hosts.append(host)
        ips: list[str] = []
        for h in hosts:
            for ip in resolver(h):
                try:
                    ipaddress.ip_address(ip)
                    if ip not in ips:
                        ips.append(ip)
                except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS:
                    continue
        out[host] = ips
    return out


def refresh_egress_rules(
    container_ip: str,
    raw_allowlist: Sequence[str] | str | None,
    label: str,
    *,
    resolver: Callable[[str], list[str]] = default_resolver,
    wildcard_subdomains: Sequence[str] | None = ("", "www", "api"),
) -> list[str]:
    """Revoke existing rules by label and apply pinned rules for the current allowlist.

    Performs a best-effort deletion via delete_rules_by_label(), then applies
    new rules computed from the current DNS resolution of hostnames.
    """
    with contextlib.suppress(_SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS):
        delete_rules_by_label(label)
    targets = expand_allowlist_to_targets(raw_allowlist, resolver=resolver, wildcard_subdomains=wildcard_subdomains)
    return apply_egress_rules_atomic(container_ip, targets, label)


def _build_restore_blob(container_ip: str, targets: Iterable[str], label: str) -> str:
    """Build an iptables-restore filter table blob that appends ACCEPT rules for targets
    and a final DROP for the container IP, labeled for later cleanup.
    """
    lines: list[str] = ["*filter"]
    for tgt in targets:
        dspec = tgt if "/" in tgt else f"{tgt}/32"
        lines.append(
            f"-A DOCKER-USER -s {container_ip} -d {dspec} -j ACCEPT -m comment --comment {label}"
        )
    lines.append(
        f"-A DOCKER-USER -s {container_ip} -j DROP -m comment --comment {label}"
    )
    lines.append("COMMIT\n")
    return "\n".join(lines)


def apply_egress_rules_atomic(container_ip: str, targets: Sequence[str], label: str) -> list[str]:
    """Apply iptables rules via iptables-restore --noflush for atomicity.

    Returns a list of rule specs (as in `iptables -S` without the initial action) for deletion fallback.
    On failure, attempts iterative application with `iptables` commands.
    """
    rule_specs: list[str] = []
    try:
        blob = _build_restore_blob(container_ip, targets, label)
        proc = subprocess.run(["iptables-restore", "--noflush"], input=blob.encode("utf-8"), check=False)
        if proc.returncode == 0:
            for tgt in targets:
                dspec = tgt if "/" in tgt else f"{tgt}/32"
                rule_specs.append(f"DOCKER-USER -s {container_ip} -d {dspec} -j ACCEPT -m comment --comment {label}")
            rule_specs.append(f"DOCKER-USER -s {container_ip} -j DROP -m comment --comment {label}")
            return rule_specs
        else:
            logger.debug("iptables-restore failed; falling back to iterative iptables invocations")
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"iptables-restore invocation failed: {e}")
    # Fallback path: iterative `iptables` calls
    for tgt in targets:
        dspec = tgt if "/" in tgt else f"{tgt}/32"
        try:
            subprocess.run([
                "iptables", "-I", "DOCKER-USER", "1",
                "-s", container_ip, "-d", dspec, "-j", "ACCEPT",
                "-m", "comment", "--comment", label,
            ], check=False)
            rule_specs.append(f"DOCKER-USER -s {container_ip} -d {dspec} -j ACCEPT -m comment --comment {label}")
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("network policy: iptables ACCEPT rule failed for {} -> {}: {}", container_ip, dspec, e)
    try:
        subprocess.run([
            "iptables", "-A", "DOCKER-USER",
            "-s", container_ip, "-j", "DROP",
            "-m", "comment", "--comment", label,
        ], check=False)
        rule_specs.append(f"DOCKER-USER -s {container_ip} -j DROP -m comment --comment {label}")
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.debug("network policy: iptables DROP rule failed for {}: {}", container_ip, e)
    return rule_specs


def delete_rules_by_label(label: str) -> None:
    """Delete all rules in DOCKER-USER containing the label comment.

    Attempts precise deletion using rule numbers; falls back to translating `iptables -S` specs.
    """
    # Try deletion by line numbers (descending)
    try:
        out = subprocess.check_output(
            ["iptables", "-L", "DOCKER-USER", "--line-numbers", "-n", "-v"],
            text=True,
            timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
        )
        lines = out.splitlines()
        # Skip header lines, find those with the comment
        numbered: list[int] = []
        for ln in lines:
            if label in ln:
                try:
                    num = int(ln.split()[0])
                    numbered.append(num)
                except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS:
                    continue
        for num in sorted(numbered, reverse=True):
            with contextlib.suppress(_SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS):
                subprocess.run(
                    ["iptables", "-D", "DOCKER-USER", str(num)],
                    check=False,
                    timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
                )
        return
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.debug("network policy: delete_rules_by_label line-number path failed for {}: {}", label, e)
    # Fallback: translate `iptables -S` specs into deletions
    try:
        out2 = subprocess.check_output(
            ["iptables", "-S", "DOCKER-USER"],
            text=True,
            timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
        )
        for line in out2.splitlines():
            if label in line:
                parts = line.strip().split()
                if parts and parts[0] in {"-A", "-I"}:
                    parts[0] = "-D"
                    with contextlib.suppress(_SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS):
                        subprocess.run(
                            ["iptables"] + parts,
                            check=False,
                            timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
                        )
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.debug("network policy: delete_rules_by_label -S fallback failed for {}: {}", label, e)
