from __future__ import annotations

import hashlib
import ipaddress
import re
import shlex
import socket
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Callable

from loguru import logger

from tldw_Server_API.app.core.Infrastructure.distributed_lock import FileLock
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
_EGRESS_LOCK_PATH = Path(tempfile.gettempdir()) / "tldw-sandbox-netfilter.lock"


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
    """Atomically replace pinned destinations without first removing enforcement.

    If atomic replacement is unavailable, an existing policy remains installed
    and the iterative installer fails rather than opening an enforcement gap.
    """
    targets = expand_allowlist_to_targets(raw_allowlist, resolver=resolver, wildcard_subdomains=wildcard_subdomains)
    return apply_egress_rules_atomic(container_ip, targets, label)


def _policy_chain(label: str) -> str:
    """Derive a bounded chain name from a safe ownership label."""
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", label):
        raise ValueError("invalid iptables policy label")
    return "TLDW_SBX_" + hashlib.sha256(label.encode()).hexdigest()[:16]


def _policy_rule_specs(container_ip: str, targets: Iterable[str], label: str) -> list[list[str]]:
    """Validate IPv4 inputs and build canonical attachment and policy specs."""
    chain = _policy_chain(label)
    source = str(ipaddress.IPv4Address(container_ip)) + "/32"
    comment = ["-m", "comment", "--comment", label]
    rules = [
        [entry, "-s", source, *comment, "-j", chain]
        for entry in ("FORWARD", "INPUT")
    ]
    for target in targets:
        network = ipaddress.IPv4Network(target, strict=False)
        destination = ["-d", str(network)] if network.prefixlen else []
        rules.append([chain, *destination, *comment, "-j", "ACCEPT"])
    rules.append([chain, *comment, "-j", "DROP"])
    return rules


def _build_restore_blob(container_ip: str, targets: Iterable[str], label: str) -> str:
    """Build an atomic private policy and jumps ahead of existing filter rules."""
    rules = _policy_rule_specs(container_ip, targets, label)
    lines = ["*filter", f":{_policy_chain(label)} - [0:0]"]
    lines.extend(shlex.join(["-A", *rule]) for rule in rules[2:])
    lines.extend(shlex.join(["-I", rule[0], "1", *rule[1:]]) for rule in rules[:2])
    return "\n".join([*lines, "COMMIT", ""])


def _read_rules() -> list[list[str]]:
    """Read filter rules with bounded execution and shell-compatible quoting."""
    output = subprocess.check_output(  # nosec B603, B607
        ["iptables", "-S"], text=True, timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
    )
    return [shlex.split(line) for line in output.splitlines() if line.strip()]


def _verify_policy(rule_specs: list[list[str]], label: str) -> None:
    """Require head jumps and the complete ordered policy, including final DROP."""
    installed = [parts[1:] for parts in _read_rules() if parts[0] == "-A"]
    for rule in rule_specs[:2]:
        first = next((item for item in installed if item[0] == rule[0]), None)
        if first != rule:
            raise RuntimeError(f"iptables egress verification failed: {rule[0]} attachment is not first")
    chain = _policy_chain(label)
    if [rule for rule in installed if rule[0] == chain] != rule_specs[2:]:
        raise RuntimeError("iptables egress verification failed: private policy differs")


def apply_egress_rules_atomic(container_ip: str, targets: Sequence[str], label: str) -> list[str]:
    """Install and verify a source-scoped IPv4 destination allowlist.

    Permitted IPs/CIDRs allow every port and protocol, including other domains
    sharing an address; this is not HTTP host, TLS SNI, or DNS authorization.
    Hostnames must already be resolved/pinned by expand_allowlist_to_targets.
    The caller must use an IPv4-only isolated bridge and keep execution gated
    until this returns. Both forwarded and host-bound packets enter a private
    chain before existing rules. No unrelated chains are flushed.

    Returns installed rule specs. Iterative fallback constructs the complete
    policy before attaching it; any error raises without releasing execution.
    Writers and cleanup share a cross-process host lock. An uncertain refresh
    retains its deny chain rather than deleting enforcement for a running job.
    """
    _policy_rule_specs(container_ip, targets, label)
    chain = _policy_chain(label)
    with FileLock(_EGRESS_LOCK_PATH, timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC):
        try:
            existing_policy = any(parts[:2] in (["-N", chain], ["-A", chain]) for parts in _read_rules())
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            raise RuntimeError(f"iptables egress verification failed before installation: {e}") from e
        return _apply_egress_rules_locked(container_ip, targets, label, existing_policy=existing_policy)


def _apply_egress_rules_locked(
    container_ip: str, targets: Sequence[str], label: str, *, existing_policy: bool,
) -> list[str]:
    """Install under the host lock, preserving pre-existing enforcement on error."""
    rule_specs = _policy_rule_specs(container_ip, targets, label)
    chain = _policy_chain(label)
    installed = False
    try:
        blob = _build_restore_blob(container_ip, targets, label)
        proc = subprocess.run(  # nosec B603, B607
            ["iptables-restore", "--noflush"], input=blob.encode(), check=False,
            timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
        )
        installed = proc.returncode == 0
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.debug("iptables-restore invocation failed: {}", e)
    created = False
    try:
        if not installed:
            commands = [["iptables", "-N", chain]]
            commands.extend(["iptables", "-A", *rule] for rule in rule_specs[2:])
            commands.extend(["iptables", "-I", rule[0], "1", *rule[1:]] for rule in rule_specs[:2])
            for command in commands:
                proc = subprocess.run(  # nosec B603
                    command, check=False, timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
                )
                if proc.returncode:
                    raise RuntimeError(f"iptables egress rule application failed: {command[1]} returned {proc.returncode}")
                created = True
        try:
            _verify_policy(rule_specs, label)
        except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
            raise RuntimeError(f"iptables egress verification failed: {e}") from e
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        if (installed or created) and not existing_policy:
            _delete_rules_by_label_locked(label)
        raise RuntimeError(f"iptables egress rule application failed: {e}") from e
    return [shlex.join(rule) for rule in rule_specs]


def delete_rules_by_label(label: str) -> None:
    """Best-effort removal of exact owned jumps and private chain.

    Delete by rule spec so concurrent insertions cannot shift a rule number.
    Keep the deny chain intact if any reference cannot safely be removed.
    """
    _policy_chain(label)
    try:
        with FileLock(_EGRESS_LOCK_PATH, timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC):
            _delete_rules_by_label_locked(label)
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.warning("network policy: cleanup lock unavailable for {}: {}", label, e)


def _delete_rules_by_label_locked(label: str) -> None:
    """Remove owned resources while the caller holds the shared host lock."""
    chain = _policy_chain(label)
    try:
        rules = _read_rules()
        foreign_reference = False
        for parts in rules:
            if not parts or parts[0] != "-A" or parts[1] == chain:
                continue
            target = parts[parts.index("-j") + 1] if "-j" in parts else None
            comment = parts[parts.index("--comment") + 1] if "--comment" in parts else None
            if target == chain and comment != label:
                foreign_reference = True
                continue
            legacy = parts[1] == "DOCKER-USER" and target in {"ACCEPT", "DROP"}
            if comment == label and (target == chain or legacy):
                subprocess.run(  # nosec B603, B607
                    ["iptables", "-D", *parts[1:]], check=True,
                    timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
                )
        if foreign_reference:
            logger.warning("network policy: retaining chain {} with an unowned reference", chain)
            return
        if any(parts[:2] in (["-N", chain], ["-A", chain]) for parts in rules):
            for action in ("-F", "-X"):
                subprocess.run(  # nosec B603, B607
                    ["iptables", action, chain], check=True,
                    timeout=_IPTABLES_CLEANUP_TIMEOUT_SEC,
                )
    except _SANDBOX_NET_POLICY_NONCRITICAL_EXCEPTIONS as e:
        logger.warning("network policy: cleanup incomplete for {}: {}", label, e)
