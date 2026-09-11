# TASK-13013.5 Trusted Proxy Identity and Login Lockout Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve client identity safely across trusted proxy chains and isolate password-login lockouts by client/login pair while retaining account-wide and Resource Governor protections.

**Architecture:** Add one pure standard-library trusted-hop resolver under `app/core/Security`, then keep AuthNZ and Resource Governor as thin compatibility wrappers around their existing environment-variable families. Build one versioned deterministic login/client key in the auth endpoint, use it for password-failure checks and records, and carry that opaque key through the existing server-side MFA payload so success resets the original attempt's buckets.

**Tech Stack:** Python 3.11, `ipaddress`, `hashlib`, JSON, FastAPI/Starlette request headers, pytest, Ruff, Bandit, Backlog.md, MkDocs published-doc refresh, Git.

**Spec:** `Docs/superpowers/specs/2026-08-29-task-13013-5-trusted-proxy-login-lockout-design.md`

## Global Constraints

- Base all implementation work on `codex/task-13013-5-trusted-proxy-lockout` at design commit `0291c94412` or a direct descendant whose merge base with `origin/dev` remains `f676e23549ea8ed82ef53493260621a05b281863`.
- Keep `AUTH_TRUST_X_FORWARDED_FOR`, `AUTH_TRUSTED_PROXY_IPS`, `RG_TRUSTED_PROXIES`, and `RG_CLIENT_IP_HEADER` unchanged.
- Use only Python's standard library for the shared resolver and lockout-key format; add no dependency or dependency-manifest change.
- Return only canonical compressed IPv4/IPv6 literals or `None` from the shared resolver; never log raw forwarding values or parser exception text.
- Never rewrite `request.client`; do not change audit, setup, authorization, WebSocket, or unrelated middleware identity behavior.
- Add no database model, migration, retention sweep, or cleanup heuristic for legacy raw-IP rows.
- Encode new client/login lockout identifiers exactly as `login-client-v1:<64 lowercase SHA-256 hex>` over compact JSON `[canonical_client_or_unknown, stripped_lower_login_identifier]`.
- Preserve the existing account-wide bucket keyed by the stored canonical username and the existing Resource Governor controls.
- Treat `Docs/Published` as generated output: edit canonical source docs, run `Helper_Scripts/refresh_docs_published.sh`, and never hand-edit generated mirrors.
- Use the existing `../../.venv` tools; install nothing and modify no system file.
- Keep TASK-13013.5 current through Backlog.md MCP/CLI and keep deferred global middleware work under TASK-13144.

## File Structure

- Create `tldw_Server_API/app/core/Security/trusted_proxy.py`: pure IP/network parsing and trusted-hop selection only.
- Create `tldw_Server_API/tests/Security/test_trusted_proxy.py`: exhaustive resolver contract with no FastAPI dependency.
- Modify `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py`: AuthNZ settings/request adapter and existing allowlist helpers.
- Modify `tldw_Server_API/app/core/Resource_Governance/deps.py`: Resource Governor environment/request adapter; remove the invalid-peer-to-loopback shortcut.
- Modify `tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py`: AuthNZ precedence, repeated-header, sentinel, and compatibility coverage.
- Modify `tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py`: Resource Governor XFF/custom-header and invalid-peer coverage.
- Modify `tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py`: middleware-level identity regression coverage.
- Modify `tldw_Server_API/app/api/v1/endpoints/auth.py`: composite key helpers and password/MFA lockout flow only.
- Modify `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`: deterministic key and exact password/MFA call-sequence coverage.
- Modify `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py`: HTTP-surface isolation and threshold coverage.
- Modify `tldw_Server_API/app/core/Resource_Governance/README.md`, `Docs/Operations/Env_Vars.md`, and `Docs/Deployment/horizontal-scaling.md`: retained variables, chain semantics, and equivalent-config warning.
- Regenerate `Docs/Published/Env_Vars.md` and `Docs/Published/Deployment/horizontal-scaling.md` with the existing refresh script.
- Modify `backlog/tasks/task-13013.5 - Harden-trusted-proxy-client-identity-and-login-lockout-isolation.md` only through Backlog.md MCP/CLI.

---

### Task 1: Implement the pure trusted-hop resolver

**Files:**
- Create: `tldw_Server_API/app/core/Security/trusted_proxy.py`
- Create: `tldw_Server_API/tests/Security/test_trusted_proxy.py`

**Interfaces:**
- Consumes: physical peer `str | None`, trusted proxy host/CIDR strings, ordered XFF field values, and an optional strict single-address field.
- Produces: `resolve_trusted_client_ip(physical_peer, trusted_proxy_entries, *, forwarded_for_values=(), single_forwarded_value=None) -> str | None` and `is_trusted_proxy_peer(physical_peer, trusted_proxy_entries) -> bool`.

- [ ] **Step 1: Write the direct-peer and trust-boundary tests**

Create the test module with literal expectations for canonicalization, disabled forwarding, untrusted spoofing, invalid trusted entries, and invalid peers:

```python
import pytest

from tldw_Server_API.app.core.Security.trusted_proxy import (
    is_trusted_proxy_peer,
    resolve_trusted_client_ip,
)


@pytest.mark.parametrize(
    ("peer", "expected"),
    [
        ("203.0.113.7", "203.0.113.7"),
        ("2001:0db8:0:0::7", "2001:db8::7"),
        (None, None),
        ("testclient", None),
        ("127.0.0.1:8000", None),
    ],
)
def test_direct_peer_is_canonical_or_absent(peer, expected):
    assert resolve_trusted_client_ip(peer, ()) == expected


def test_untrusted_peer_cannot_select_forwarded_identity():
    assert resolve_trusted_client_ip(
        "198.51.100.8",
        ("10.0.0.0/8", "not-a-network"),
        forwarded_for_values=("203.0.113.9",),
        single_forwarded_value="192.0.2.4",
    ) == "198.51.100.8"


def test_trusted_peer_predicate_accepts_hosts_and_cidrs():
    entries = ("192.0.2.10", "2001:db8:abcd::/48", "invalid")
    assert is_trusted_proxy_peer("192.0.2.10", entries) is True
    assert is_trusted_proxy_peer("2001:db8:abcd::4", entries) is True
    assert is_trusted_proxy_peer("203.0.113.4", entries) is False
```

- [ ] **Step 2: Write the XFF and strict-single-field matrix**

Add explicit tests for overwritten/appended XFF, repeated fields in wire order, attacker-prepended values, multi-proxy chains, all-trusted fallback, malformed/empty/decorated tokens, IPv6, and strict single values:

```python
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("203.0.113.9",), "203.0.113.9"),
        (("198.51.100.99, 203.0.113.9, 10.0.0.2",), "203.0.113.9"),
        (("198.51.100.99", "203.0.113.9, 10.0.0.2"), "203.0.113.9"),
        (("2001:db8:ffff::9, 2001:db8:abcd::2",), "2001:db8:ffff::9"),
        (("10.0.0.2, 10.0.0.3",), "10.0.0.1"),
        (("203.0.113.9,,10.0.0.2",), "10.0.0.1"),
        (("203.0.113.9:443,10.0.0.2",), "10.0.0.1"),
        (("[2001:db8::9],10.0.0.2",), "10.0.0.1"),
        (("for=203.0.113.9,10.0.0.2",), "10.0.0.1"),
        (("fe80::1%eth0,10.0.0.2",), "10.0.0.1"),
    ],
)
def test_xff_is_scanned_from_trusted_edge(values, expected):
    assert resolve_trusted_client_ip(
        "10.0.0.1",
        ("10.0.0.0/8", "2001:db8:abcd::/48"),
        forwarded_for_values=values,
    ) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("203.0.113.9", "203.0.113.9"),
        ("2001:0db8::9", "2001:db8::9"),
        ("203.0.113.9, 10.0.0.2", "10.0.0.1"),
        ("", "10.0.0.1"),
        ("[2001:db8::9]", "10.0.0.1"),
    ],
)
def test_single_forwarded_field_accepts_exactly_one_plain_ip(value, expected):
    assert resolve_trusted_client_ip(
        "10.0.0.1",
        ("10.0.0.0/8",),
        single_forwarded_value=value,
    ) == expected
```

- [ ] **Step 3: Run the resolver tests and preserve the RED result**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/Security/test_trusted_proxy.py -q
```

Expected: collection fails because `Security.trusted_proxy` does not exist.

- [ ] **Step 4: Implement the minimal pure resolver**

Create the module with no logging, environment, FastAPI, or request imports:

```python
from __future__ import annotations

import ipaddress
from collections.abc import Iterable

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def _parse_ip(value: str | None) -> IPAddress | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    try:
        return ipaddress.ip_address(token)
    except ValueError:
        return None


def _parse_networks(entries: Iterable[str]) -> tuple[IPNetwork, ...]:
    networks: list[IPNetwork] = []
    for entry in entries:
        token = str(entry).strip()
        if not token:
            continue
        try:
            networks.append(ipaddress.ip_network(token, strict=False))
        except ValueError:
            continue
    return tuple(networks)


def _address_is_trusted(address: IPAddress, networks: tuple[IPNetwork, ...]) -> bool:
    return any(address.version == network.version and address in network for network in networks)


def is_trusted_proxy_peer(
    physical_peer: str | None,
    trusted_proxy_entries: Iterable[str],
) -> bool:
    peer = _parse_ip(physical_peer)
    return peer is not None and _address_is_trusted(peer, _parse_networks(trusted_proxy_entries))


def resolve_trusted_client_ip(
    physical_peer: str | None,
    trusted_proxy_entries: Iterable[str],
    *,
    forwarded_for_values: Iterable[str] = (),
    single_forwarded_value: str | None = None,
) -> str | None:
    peer = _parse_ip(physical_peer)
    if peer is None:
        return None
    networks = _parse_networks(trusted_proxy_entries)
    if not _address_is_trusted(peer, networks):
        return peer.compressed

    xff_values = tuple(str(value) for value in forwarded_for_values)
    if xff_values:
        parsed_chain: list[IPAddress] = []
        for token in ",".join(xff_values).split(","):
            parsed = _parse_ip(token)
            if parsed is None:
                return peer.compressed
            parsed_chain.append(parsed)
        for address in reversed(parsed_chain):
            if not _address_is_trusted(address, networks):
                return address.compressed
        return peer.compressed

    if single_forwarded_value is not None:
        forwarded = _parse_ip(single_forwarded_value)
        return forwarded.compressed if forwarded is not None else peer.compressed
    return peer.compressed
```

- [ ] **Step 5: Run the focused resolver suite and static checks**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/Security/test_trusted_proxy.py -q
../../.venv/bin/ruff check tldw_Server_API/app/core/Security/trusted_proxy.py tldw_Server_API/tests/Security/test_trusted_proxy.py
../../.venv/bin/python -m py_compile tldw_Server_API/app/core/Security/trusted_proxy.py tldw_Server_API/tests/Security/test_trusted_proxy.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 6: Commit the pure resolver**

```bash
git add tldw_Server_API/app/core/Security/trusted_proxy.py tldw_Server_API/tests/Security/test_trusted_proxy.py
git commit -m "fix: resolve trusted proxy chains safely"
```

---

### Task 2: Delegate AuthNZ and Resource Governor wrappers to the resolver

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py:1-102`
- Modify: `tldw_Server_API/app/core/Resource_Governance/deps.py:15-119`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py`
- Modify: `tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py`
- Modify: `tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py`

**Interfaces:**
- Consumes: Task 1 `resolve_trusted_client_ip()` and `is_trusted_proxy_peer()`.
- Produces: unchanged public wrappers `resolve_client_ip(request, settings=None) -> str | None` and `derive_client_ip(request) -> str`, with safe chain semantics and existing configuration names.

- [ ] **Step 1: Add AuthNZ wrapper RED tests**

Use a Starlette `Headers` object constructed with `raw=` so repeated XFF field occurrences are real and ordered. Assert XFF precedence, malformed-primary fallback, canonical output, invalid-peer `None`, disabled-header behavior, and strict X-Real-IP behavior:

```python
from types import SimpleNamespace

from starlette.datastructures import Headers


def _request(peer, raw_headers):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=Headers(raw=raw_headers))


def _proxy_settings(enabled=True):
    return SimpleNamespace(
        AUTH_TRUST_X_FORWARDED_FOR=enabled,
        AUTH_TRUSTED_PROXY_IPS=["10.0.0.0/8"],
    )


def test_resolve_client_ip_combines_repeated_xff_and_ignores_attacker_prefix():
    request = _request(
        "10.0.0.1",
        [(b"x-forwarded-for", b"198.51.100.99"),
         (b"x-forwarded-for", b"203.0.113.9, 10.0.0.2")],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "203.0.113.9"


def test_resolve_client_ip_ignores_forwarding_when_disabled():
    request = _request("10.0.0.1", [(b"x-forwarded-for", b"203.0.113.9")])
    assert resolve_client_ip(request, _proxy_settings(enabled=False)) == "10.0.0.1"


def test_xff_takes_precedence_over_x_real_ip():
    request = _request(
        "10.0.0.1",
        [(b"x-forwarded-for", b"203.0.113.9"), (b"x-real-ip", b"198.51.100.4")],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "203.0.113.9"


def test_malformed_xff_does_not_fall_through_to_x_real_ip():
    request = _request(
        "10.0.0.1",
        [(b"x-forwarded-for", b"bad, 10.0.0.2"), (b"x-real-ip", b"203.0.113.4")],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "10.0.0.1"


def test_repeated_x_real_ip_is_not_treated_as_a_single_address():
    request = _request(
        "10.0.0.1",
        [(b"x-real-ip", b"203.0.113.4"), (b"x-real-ip", b"198.51.100.4")],
    )
    assert resolve_client_ip(request, _proxy_settings()) == "10.0.0.1"


def test_invalid_physical_peer_never_authorizes_forwarded_headers():
    request = _request("testclient", [(b"x-forwarded-for", b"203.0.113.4")])
    assert resolve_client_ip(request, _proxy_settings()) is None
```

- [ ] **Step 2: Add Resource Governor wrapper and middleware RED tests**

Expand the existing request helper to support raw repeated headers, then assert right-to-left XFF, case-insensitive configured XFF, strict custom headers, invalid peer `unknown`, and middleware state propagation:

```python
@pytest.mark.asyncio
async def test_xff_uses_first_untrusted_hop_from_the_right(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "x-FoRwArDeD-fOr")
    request = _make_request(
        peer="10.0.0.1",
        raw_headers=[(b"x-forwarded-for", b"198.51.100.99, 203.0.113.9, 10.0.0.2")],
    )
    assert derive_client_ip(request) == "203.0.113.9"


@pytest.mark.asyncio
async def test_custom_header_is_single_ip_only(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "CF-Connecting-IP")
    request = _make_request(
        peer="10.0.0.1",
        raw_headers=[(b"cf-connecting-ip", b"203.0.113.9, 10.0.0.2")],
    )
    assert derive_client_ip(request) == "10.0.0.1"


@pytest.mark.asyncio
async def test_repeated_custom_header_is_not_treated_as_a_single_address(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "CF-Connecting-IP")
    request = _make_request(
        peer="10.0.0.1",
        raw_headers=[(b"cf-connecting-ip", b"203.0.113.9"),
                     (b"cf-connecting-ip", b"198.51.100.9")],
    )
    assert derive_client_ip(request) == "10.0.0.1"


@pytest.mark.asyncio
async def test_invalid_peer_is_unknown_not_loopback(monkeypatch):
    monkeypatch.setenv("RG_TRUSTED_PROXIES", "127.0.0.1")
    monkeypatch.setenv("RG_CLIENT_IP_HEADER", "X-Forwarded-For")
    request = _make_request(peer="testclient", raw_headers=[(b"x-forwarded-for", b"203.0.113.9")])
    assert derive_client_ip(request) == "unknown"
```

- [ ] **Step 3: Run the wrapper matrix and preserve the RED result**

Run:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py -q
```

Expected: new tests fail because the wrappers still select the leftmost value and Resource Governor still converts invalid peers to loopback.

- [ ] **Step 4: Replace AuthNZ's forwarding parser with a thin adapter**

Import the Task 1 functions. Keep `_ip_in_allowlist()` for non-proxy allowlist behavior, but make `is_trusted_proxy_ip()` call `is_trusted_proxy_peer()`. Add a private header extractor that preserves repeated occurrences and use XFF before X-Real-IP:

```python
def _header_values(request: Any, name: str) -> tuple[str, ...]:
    try:
        headers = request.headers
        getlist = getattr(headers, "getlist", None)
        if callable(getlist):
            return tuple(str(value) for value in getlist(name))
        value = headers.get(name)
        return (str(value),) if value is not None else ()
    except _IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS:
        return ()


def is_trusted_proxy_ip(ip: str | None, settings: Settings | None = None) -> bool:
    resolved_settings = settings or get_settings()
    entries = _normalize_entries(
        getattr(resolved_settings, "AUTH_TRUSTED_PROXY_IPS", None) or []
    )
    return is_trusted_proxy_peer(ip, entries)


def resolve_client_ip(request: Any, settings: Settings | None = None) -> str | None:
    if request is None:
        return None
    try:
        resolved_settings = settings or get_settings()
    except _IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS:
        resolved_settings = settings
    try:
        peer = getattr(getattr(request, "client", None), "host", None)
    except _IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS:
        peer = None
    trusted = _normalize_entries(
        getattr(resolved_settings, "AUTH_TRUSTED_PROXY_IPS", None) or []
    ) if resolved_settings is not None else []
    if not bool(getattr(resolved_settings, "AUTH_TRUST_X_FORWARDED_FOR", False)):
        trusted = []
    xff = _header_values(request, "x-forwarded-for")
    real_ip_values = () if xff else _header_values(request, "x-real-ip")
    real_ip = real_ip_values[0] if len(real_ip_values) == 1 else None
    return resolve_trusted_client_ip(
        peer,
        trusted,
        forwarded_for_values=xff,
        single_forwarded_value=real_ip,
    )
```

- [ ] **Step 5: Replace Resource Governor's duplicated parser with a thin adapter**

Delete `_parse_trusted_proxies()`, `_is_trusted_proxy()`, and the `testclient`-to-loopback shortcut. Split `RG_TRUSTED_PROXIES` into raw entries, distinguish XFF case-insensitively, and map the shared resolver's `None` to `unknown`:

```python
def derive_client_ip(request: Request) -> str:
    try:
        peer = request.client.host if request.client and request.client.host else None
    except _RG_DEPS_NONCRITICAL_EXCEPTIONS:
        peer = None
    trusted = tuple(
        part.strip()
        for part in (os.getenv("RG_TRUSTED_PROXIES") or "").split(",")
        if part.strip()
    )
    header_name = (os.getenv("RG_CLIENT_IP_HEADER") or "").strip()
    xff_values: tuple[str, ...] = ()
    single_value = None
    if header_name:
        if header_name.lower() == "x-forwarded-for":
            xff_values = tuple(request.headers.getlist(header_name))
        else:
            header_values = tuple(request.headers.getlist(header_name))
            single_value = header_values[0] if len(header_values) == 1 else None
    resolved = resolve_trusted_client_ip(
        peer,
        trusted,
        forwarded_for_values=xff_values,
        single_forwarded_value=single_value,
    )
    return resolved or "unknown"
```

- [ ] **Step 6: Run focused and neighboring regression suites**

Run:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Security/test_trusted_proxy.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py -q
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Security/trusted_proxy.py \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/Resource_Governance/deps.py \
  tldw_Server_API/tests/Security/test_trusted_proxy.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py
git diff --check
```

Expected: every command exits 0; the pre-existing baseline of 15 wrapper/middleware tests remains green alongside the new matrix.

- [ ] **Step 7: Commit the compatibility wrappers**

```bash
git add \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/Resource_Governance/deps.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py
git commit -m "fix: share trusted proxy identity resolution"
```

---

### Task 3: Introduce the stable client/login lockout key

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py:1-140, 966-990`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`

**Interfaces:**
- Consumes: Task 2 canonical `_auth_request_client_ip()` output or `unknown`, plus the attempted identifier.
- Produces: `_login_client_lockout_key(client_ip, login_identifier) -> str` and `_validated_login_client_lockout_key(value) -> str | None`.

- [ ] **Step 1: Add exact key-format and validation RED tests**

Add a known vector, normalization equivalence, alias distinction, unknown-client behavior, and malformed-key rejection:

```python
def test_login_client_lockout_key_has_stable_known_vector():
    assert auth._login_client_lockout_key("203.0.113.9", " Alice@Example.COM ") == (
        "login-client-v1:5573603ba3e0013f1788b2e3e4b7d67553500401252965ad3f2189a1a352b014"
    )


def test_login_client_lockout_key_normalizes_identifier_but_preserves_aliases():
    by_email = auth._login_client_lockout_key("2001:db8::9", " USER@example.com ")
    assert by_email == auth._login_client_lockout_key("2001:db8::9", "user@example.com")
    assert by_email != auth._login_client_lockout_key("2001:db8::9", "user")
    assert auth._login_client_lockout_key(None, "user").startswith("login-client-v1:")


@pytest.mark.parametrize(
    "value",
    [None, "", "login-client-v1:", "login-client-v1:" + "A" * 64,
     "login-client-v2:" + "a" * 64, "login-client-v1:" + "a" * 63,
     "login-client-v1:" + "g" * 64],
)
def test_validated_login_client_lockout_key_rejects_noncanonical_values(value):
    assert auth._validated_login_client_lockout_key(value) is None
```

- [ ] **Step 2: Add the invalid-physical-peer mapping RED test**

Construct a request whose physical peer is `testclient` and whose XFF is spoofed. Assert `_auth_request_client_ip(request) == "unknown"`; this freezes removal of the current raw-peer fallback.

- [ ] **Step 3: Run the helper selection and preserve the RED result**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py -q \
  -k "login_client_lockout_key or invalid_physical_peer"
```

Expected: failures because the helpers do not exist and invalid peers still leak the raw placeholder.

- [ ] **Step 4: Implement the versioned key and safe client sentinel**

Import `hashlib`, add a compiled exact-key expression, and implement the helpers near `_auth_request_client_ip()`:

```python
_LOGIN_CLIENT_LOCKOUT_KEY_RE = re.compile(r"login-client-v1:[0-9a-f]{64}\Z")


def _auth_request_client_ip(request: Request) -> str:
    try:
        settings = get_settings()
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        settings = None
    try:
        return resolve_client_ip(request, settings) or "unknown"
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return "unknown"


def _login_client_lockout_key(client_ip: str | None, login_identifier: str) -> str:
    payload = json.dumps(
        [client_ip or "unknown", login_identifier.strip().lower()],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"login-client-v1:{hashlib.sha256(payload).hexdigest()}"


def _validated_login_client_lockout_key(value: object) -> str | None:
    return value if isinstance(value, str) and _LOGIN_CLIENT_LOCKOUT_KEY_RE.fullmatch(value) else None
```

- [ ] **Step 5: Run the helper tests and static checks**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py -q \
  -k "login_client_lockout_key or invalid_physical_peer"
../../.venv/bin/ruff check tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py
../../.venv/bin/python -m py_compile tldw_Server_API/app/api/v1/endpoints/auth.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 6: Commit the stable-key contract**

```bash
git add tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py
git commit -m "fix: add stable login client lockout keys"
```

---

### Task 4: Isolate password-login failures without weakening account lockout

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py:1520-1930`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py`

**Interfaces:**
- Consumes: Task 3 `_login_client_lockout_key()`, existing `AuthGovernor.check_lockout()` / `record_auth_failure()`, and `RateLimiter.reset_failed_attempts()`.
- Produces: pre-fetch composite checking; composite-only unknown-user recording; composite-plus-account known-user recording; exact composite/account success reset.

- [ ] **Step 1: Write unit tests for the exact password-login call order**

Use recording stubs for `check_lockout`, `record_auth_failure`, and `reset_failed_attempts`. Cover these literal expectations:

```python
assert checked == [(composite_key, "login")]
assert recorded_unknown == [(composite_key, "login")]
assert recorded_bad_password == [
    (composite_key, "login"),
    ("stored_username", "login"),
]
assert reset_on_success == [
    (composite_key, "login"),
    ("stored_username", "login"),
]
```

Add separate cases proving:

```python
assert auth._login_client_lockout_key("203.0.113.9", "alice") != \
       auth._login_client_lockout_key("203.0.113.9", "bob")
assert checked_for_email[1] == ("stored_username", "login")
assert checked_for_username[1] == ("stored_username", "login")
```

The first assertion freezes shared-IP isolation; the latter two freeze username/email alias convergence on the account bucket.

- [ ] **Step 2: Add HTTP-surface threshold and isolation tests**

In the existing Postgres-backed integration module, teach `_StubLimiter` to record checked/failed/reset identifiers. Keep the existing same-client/same-login threshold test and add a case that submits three failures for `lockout_user` (the third reaches its 429 threshold), then a failure for `other_identifier`; assert the latter remains 401 rather than inheriting the first composite bucket's lockout. Assert every client-side recorded identifier matches `^login-client-v1:[0-9a-f]{64}$` and no raw physical IP is stored.

- [ ] **Step 3: Run the password-login tests and preserve the RED result**

Run:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py -q \
  -k "lockout or login_client"
```

Expected: new call-sequence and isolation assertions fail because the endpoint still checks, records, and resets raw client IP.

- [ ] **Step 4: Compute the composite before the pre-fetch check**

At login entry, normalize once with the same operation as `fetch_user_by_login_identifier()` and use the composite for the first check:

```python
client_ip = _auth_request_client_ip(request)
login_identifier = form_data.username.strip().lower()
client_login_lockout_key = _login_client_lockout_key(client_ip, login_identifier)

if getattr(rate_limiter, "enabled", False):
    is_locked, lockout_expires = await auth_gov.check_lockout(
        client_login_lockout_key,
        attempt_type="login",
        rate_limiter=rate_limiter,
    )
```

Keep the same sanitized 429 response and expiry calculation, but change internal counter names from IP-specific to client/login-specific language; do not include the opaque key in log messages.

- [ ] **Step 5: Route failure and ordinary-success mutations to exact buckets**

Replace the three raw-IP mutations:

```python
# Unknown user
await auth_gov.record_auth_failure(
    identifier=client_login_lockout_key,
    attempt_type="login",
    rate_limiter=rate_limiter,
)

# Invalid password
client_result = await auth_gov.record_auth_failure(
    identifier=client_login_lockout_key,
    attempt_type="login",
    rate_limiter=rate_limiter,
)
account_result = await auth_gov.record_auth_failure(
    identifier=user["username"],
    attempt_type="login",
    rate_limiter=rate_limiter,
)

# Successful non-MFA login
await rate_limiter.reset_failed_attempts(client_login_lockout_key, "login")
await rate_limiter.reset_failed_attempts(user["username"], "login")
```

Keep the stored-username account lockout check before password verification. Determine the 429 response from either failure result exactly as the current endpoint does; merely rename `ip_result` to `client_result`.

- [ ] **Step 6: Run unit and integration password-login gates**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py -q \
  -k "lockout or login_client"
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py -q
git diff --check
```

Expected: all selected unit tests and the complete HTTP integration module pass.

- [ ] **Step 7: Commit password-login isolation**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/auth.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py
git commit -m "fix: isolate password login lockouts"
```

---

### Task 5: Carry the original composite key through MFA success

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py:1838-1883, 3442-3602`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py:650-880, 1075-1135`

**Interfaces:**
- Consumes: Task 4 `client_login_lockout_key` and Task 3 `_validated_login_client_lockout_key()`.
- Produces: MFA ephemeral JSON `{user_id, session_id, login_lockout_key}` and exact original-composite/account reset after successful MFA.

- [ ] **Step 1: Add MFA payload and different-network RED tests**

Extend the existing MFA challenge test to decode the stored ephemeral JSON and assert exact fields:

```python
cached = json.loads(session_manager.stored_value)
assert cached == {
    "user_id": 7,
    "session_id": 41,
    "login_lockout_key": auth._login_client_lockout_key("203.0.113.9", "mfa_user"),
}
assert "203.0.113.9" not in session_manager.stored_value
assert "mfa_user" not in session_manager.stored_value
```

Extend the MFA completion test so the current request uses `198.51.100.44` while the cached payload contains a valid original key. Assert reset calls are exactly:

```python
assert reset_calls == [
    (original_login_lockout_key, "login"),
    ("mfa_user", "login"),
]
assert all(identifier != "198.51.100.44" for identifier, _ in reset_calls)
```

- [ ] **Step 2: Add malformed/missing cached-key RED tests**

Parameterize payloads with a missing key, raw IP, wrong version, uppercase digest, short digest, and non-string value. Invoke `mfa_login()` and assert HTTP 400 with existing detail `MFA session expired or invalid`; assert no limiter reset, no token update, and no user-selected identifier reaches a limiter call.

- [ ] **Step 3: Run the MFA selection and preserve the RED result**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py -q -k "mfa and (login or lockout)"
```

Expected: challenge payload lacks `login_lockout_key`, completion resets the current request IP, and malformed-key cases do not reject early.

- [ ] **Step 4: Store the opaque key in the challenge payload**

Change only the existing server-side ephemeral value:

```python
payload = {
    "user_id": int(user["id"]),
    "session_id": int(session_id),
    "login_lockout_key": client_login_lockout_key,
}
```

Do not expose the key in the response, token claims, logs, cookies, or audit payload.

- [ ] **Step 5: Validate and use the original key on MFA completion**

Validate the cached key alongside `user_id` and `session_id` before Resource Governor reservation or user lookup:

```python
session_id = payload.get("session_id")
user_id = payload.get("user_id")
login_lockout_key = _validated_login_client_lockout_key(payload.get("login_lockout_key"))
if not session_id or not user_id or login_lockout_key is None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="MFA session expired or invalid",
    )
```

After successful MFA, keep the current request IP for audit/session observability but reset only:

```python
await rate_limiter.reset_failed_attempts(login_lockout_key, "login")
await rate_limiter.reset_failed_attempts(user.get("username", ""), "login")
```

- [ ] **Step 6: Run the complete AuthNZ endpoint unit module**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py -q
../../.venv/bin/ruff check tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py
../../.venv/bin/python -m py_compile tldw_Server_API/app/api/v1/endpoints/auth.py
git diff --check
```

Expected: the complete module and every static command pass.

- [ ] **Step 7: Commit MFA reset propagation**

```bash
git add tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py
git commit -m "fix: reset original login lockout after mfa"
```

---

### Task 6: Document the retained proxy configuration and rollout boundary

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/README.md:115-130`
- Modify: `Docs/Operations/Env_Vars.md:320-340`
- Modify: `Docs/Deployment/horizontal-scaling.md:225-295`
- Regenerate: `Docs/Published/Env_Vars.md`
- Regenerate: `Docs/Published/Deployment/horizontal-scaling.md`

**Interfaces:**
- Consumes: Tasks 1-5 final behavior and existing publishing script.
- Produces: operator-facing configuration, equivalence warning, rollout behavior, and deterministic published mirrors.

- [ ] **Step 1: Update canonical operator docs with exact semantics**

Add the four retained variables and these concrete rules:

```markdown
- Forwarding headers are trusted only when the physical peer is a valid IP in the subsystem's configured trusted-proxy host/CIDR list.
- `X-Forwarded-For` is parsed as a complete chain from the trusted edge inward; malformed chains fall back to the physical peer.
- Other `RG_CLIENT_IP_HEADER` values must contain one plain IP literal.
- If both AuthNZ and Resource Governor forwarding are enabled, configure equivalent trusted-proxy sets and compatible headers so login lockouts and request governance derive the same client identity.
- AuthNZ: `AUTH_TRUST_X_FORWARDED_FOR=true` plus `AUTH_TRUSTED_PROXY_IPS=<proxy IP/CIDR list>`.
- Resource Governor: `RG_CLIENT_IP_HEADER=X-Forwarded-For` plus `RG_TRUSTED_PROXIES=<equivalent proxy IP/CIDR list>`.
```

Document that invalid physical peers resolve to the subsystem's safe unknown sentinel, the feature remains opt-in, and rollout stops checking legacy raw-IP password-login buckets while account-wide lockout and Resource Governor protection remain active.

- [ ] **Step 2: Refresh generated documentation once**

Run:

```bash
bash Helper_Scripts/refresh_docs_published.sh
git diff -- Docs/Published/Env_Vars.md Docs/Published/Deployment/horizontal-scaling.md
git status --short Docs/Published
```

Expected: the two named generated mirrors contain the canonical edits. If the refresh exposes unrelated generated drift, do not edit generated files by hand; preserve only output attributable to the three canonical source-doc changes and record the unrelated drift in TASK-13013.5.

- [ ] **Step 3: Prove the generated mirrors match their canonical sources**

Run:

```bash
cmp Docs/Operations/Env_Vars.md Docs/Published/Env_Vars.md
cmp Docs/Deployment/horizontal-scaling.md Docs/Published/Deployment/horizontal-scaling.md
python3 Helper_Scripts/docs/check_public_private_boundary.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Commit canonical and generated docs**

```bash
git add \
  tldw_Server_API/app/core/Resource_Governance/README.md \
  Docs/Operations/Env_Vars.md \
  Docs/Deployment/horizontal-scaling.md \
  Docs/Published/Env_Vars.md \
  Docs/Published/Deployment/horizontal-scaling.md
git commit -m "docs: explain trusted proxy chain configuration"
```

---

### Task 7: Run completion gates and close the implementation record

**Files:**
- Test: all production, test, and documentation files changed in Tasks 1-6.
- Modify: `backlog/tasks/task-13013.5 - Harden-trusted-proxy-client-identity-and-login-lockout-isolation.md` through Backlog.md MCP/CLI only.

**Interfaces:**
- Consumes: committed Tasks 1-6 and the approved design.
- Produces: fresh verification evidence, updated Backlog acceptance/DoD state, and one review-ready exact branch head.

- [ ] **Step 1: Re-run the complete focused security and wrapper matrix**

Run:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Security/test_trusted_proxy.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py -q
```

Expected: every selected test passes.

- [ ] **Step 2: Run the complete login lockout integration module**

Run:

```bash
../../.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py -q
```

Expected: every test passes against the repository's configured isolated Postgres fixture. A missing configured Postgres service is an environment blocker to record, not permission to skip or weaken the gate.

- [ ] **Step 3: Run Python compilation and Ruff on every touched Python path**

Run:

```bash
../../.venv/bin/python -m py_compile \
  tldw_Server_API/app/core/Security/trusted_proxy.py \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/Resource_Governance/deps.py \
  tldw_Server_API/app/api/v1/endpoints/auth.py \
  tldw_Server_API/tests/Security/test_trusted_proxy.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py
../../.venv/bin/ruff check \
  tldw_Server_API/app/core/Security/trusted_proxy.py \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/Resource_Governance/deps.py \
  tldw_Server_API/app/api/v1/endpoints/auth.py \
  tldw_Server_API/tests/Security/test_trusted_proxy.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/Resource_Governance/test_deps_trusted_proxy.py \
  tldw_Server_API/tests/Resource_Governance/test_middleware_trusted_proxy_ip.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py
```

Expected: both commands exit 0.

- [ ] **Step 4: Run Bandit on touched production Python**

Run:

```bash
../../.venv/bin/bandit -q -ll \
  tldw_Server_API/app/core/Security/trusted_proxy.py \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/Resource_Governance/deps.py \
  tldw_Server_API/app/api/v1/endpoints/auth.py
```

Expected: exit 0 with no Medium/High-severity finding introduced by this change.

- [ ] **Step 5: Verify docs, scope, and absence of forbidden changes**

Run:

```bash
cmp Docs/Operations/Env_Vars.md Docs/Published/Env_Vars.md
cmp Docs/Deployment/horizontal-scaling.md Docs/Published/Deployment/horizontal-scaling.md
python3 Helper_Scripts/docs/check_public_private_boundary.py
git diff --check origin/dev...HEAD
git diff --name-only origin/dev...HEAD
git diff --name-only origin/dev...HEAD | rg '(^|/)(alembic|migrations)(/|$)|(^|/)(requirements[^/]*|pyproject\.toml|package-lock\.json|bun\.lockb?)$|(^|/)tldw_Server_API/app/core/Security/middleware\.py$|websocket' && exit 1 || true
```

Expected: source/generated docs compare equal; the boundary and whitespace checks pass; the exact changed-path review contains only planned code, tests, docs, plan/spec, and Backlog records; the forbidden-path scan prints nothing.

- [ ] **Step 6: Update TASK-13013.5 through Backlog.md**

Append the exact commit SHAs and command results, check each acceptance criterion/Definition of Done item supported by evidence, link this plan and the approved spec, keep TASK-13144 as the deferred global rewrite, and move TASK-13013.5 to `Done` only after all local gates and independent review pass.

- [ ] **Step 7: Commit the final task record and verify the branch**

```bash
git add 'backlog/tasks/task-13013.5 - Harden-trusted-proxy-client-identity-and-login-lockout-isolation.md'
git commit -m "chore: record trusted proxy lockout verification"
git status --short --branch
git log --oneline --decorate origin/dev..HEAD
git diff --check origin/dev...HEAD
```

Expected: clean worktree, reviewable small commits, and no untracked/generated residue.

- [ ] **Step 8: Require independent review and hosted gates before merge**

Request correctness/security review against the exact branch head. Resolve every Critical or Important finding with a test-first follow-up and rerun affected gates. Push only after local review is green, then require the repository's `backend-required`, `security-required`, and `coverage-required` checks to succeed on the unchanged PR head; do not bypass protection or merge without a separate user decision.
