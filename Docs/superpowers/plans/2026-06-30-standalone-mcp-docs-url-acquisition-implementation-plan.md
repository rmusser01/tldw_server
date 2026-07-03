# Standalone MCP Docs URL Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional, standalone-safe single-page URL ingestion to the MCP docs corpus through `docs.ingest_url`, with SQLite + FTS5 retrieval and no mandatory web-scraping dependencies.

**Architecture:** Extend the Stage 1 docs package with a new `mcp_unified.docs.acquisition` package that separates source policy, DNS/IP validation, transport, extraction, and store ingestion. The MCP provider advertises URL ingestion only when enabled, while `docs.status` reports disabled, stdlib-only, and rich-extractor availability without importing optional packages at module import time. The host `tldw_server` shim remains a thin adapter and the standalone package continues to import without `tldw_Server_API`, `requests`, `httpx`, `aiohttp`, `playwright`, `trafilatura`, or `bs4` top-level imports.

**Tech Stack:** Python 3.10+, stdlib URL/DNS/IP/HTTP primitives (`urllib.parse`, `socket`, `ipaddress`, `ssl`, `html.parser`), SQLite + FTS5 through the existing `DocsCatalogStore`, optional lazy `trafilatura` and BeautifulSoup extraction via `importlib.import_module`, pytest with fake resolver/transport seams, Bandit for touched Python paths.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md`
- Stage 1 plan: `Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md`
- Backlog planning task: `TASK-12077`
- Implementation work must have its own Backlog.md task before code edits begin, per repo policy.

## Non-Negotiable Constraints

- `enable_web_acquisition` defaults to false and `docs.ingest_url` is hidden when false.
- Unknown sources under approval-requiring profiles return `approval_required` before resolver or transport calls.
- Tests use fake resolver/transport objects and never require live internet.
- The fetch path must deny private, loopback, link-local, multicast, unspecified, and reserved IP addresses.
- Redirects are manual, capped, and re-run source policy, DNS, and IP validation on every hop.
- Transport must dial the resolver-validated address or be rejected as a DNS-rebinding risk.
- `respect_robots: true` fails closed with `robots_unavailable` before page fetch unless a standalone robots checker is implemented in the same slice with tests. This plan chooses fail-closed and does not implement robots fetching.
- Optional extractors are detected lazily; no optional web-scraping package may be imported at `mcp_unified.docs` package import time.
- Do not add `requests`, `httpx`, `aiohttp`, or `playwright` to this standalone path.
- Do not couple `mcp_unified.docs` to Media DB, ChromaDB, Jobs, Scheduler, AuthNZ, browser profiles, cookies, or `tldw_Server_API`.

## File Structure

Create:

- `mcp_unified/docs/acquisition/__init__.py` - exports acquisition service and safe model types that do not import optional extractors.
- `mcp_unified/docs/acquisition/models.py` - dataclasses and protocols shared by policy, fetcher, extractor, and service.
- `mcp_unified/docs/acquisition/policy.py` - pure URL normalization, domain/prefix matching, and source decisions with no network I/O.
- `mcp_unified/docs/acquisition/resolver.py` - stdlib resolver plus IP classification helpers.
- `mcp_unified/docs/acquisition/fetcher.py` - redirect-aware fetch orchestration and baseline validated-address transport.
- `mcp_unified/docs/acquisition/extract.py` - lazy rich extractors and stdlib HTML/text fallback.
- `mcp_unified/docs/acquisition/service.py` - coordinates policy, robots fail-closed behavior, fetch, extraction, chunking, and store writes.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_policy.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py`

Modify:

- `mcp_unified/docs/settings.py` - add URL acquisition config parsing and safe defaults.
- `mcp_unified/docs/importers/base.py` - allow parsed documents to represent URL sources without fake filesystem paths.
- `mcp_unified/docs/importers/html.py` - expose a string-based static HTML parser used by local files and fetched pages.
- `mcp_unified/docs/importers/local.py` - pass optional `source_path` from generalized parsed documents.
- `mcp_unified/docs/mcp_module.py` - construct acquisition service when enabled, advertise and execute `docs.ingest_url`, and report status.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py` - validate `url` for `docs.ingest_url` in the host shim only.
- `tldw_Server_API/Config_Files/mcp_modules.yaml` - keep `enable_web_acquisition: false`; add safe commented-free defaults if config discoverability is desired.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`

Do not modify:

- `pyproject.toml` for new required dependencies. The monorepo already has web/document packages for the host app, but this standalone feature must remain runtime-optional and import-boundary protected.
- Existing `tldw_Server_API` scraping services. Use them only as reference material for extraction behavior, not as imports.

## Test Command Conventions

Use the project virtual environment explicitly:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q
```

Focused examples below use the same Python path. If a command fails because the virtual environment is absent, stop and report the missing environment rather than switching to global Python.

---

### Task 1: URL Acquisition Settings And Disabled Status

**Files:**

- Modify: `mcp_unified/docs/settings.py`
- Modify: `mcp_unified/docs/mcp_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [ ] **Step 1: Add failing settings tests**

Add these tests to `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py`:

```python
def test_from_mapping_uses_safe_url_acquisition_defaults() -> None:
    settings = DocsSettings.from_mapping({})

    assert settings.enable_web_acquisition is False  # nosec B101
    assert settings.web_source_profile == "locked_down"  # nosec B101
    assert settings.preapproved_domains == ()  # nosec B101
    assert settings.allowed_url_prefixes == ()  # nosec B101
    assert settings.denied_domains == ()  # nosec B101
    assert settings.max_url_redirects == 3  # nosec B101
    assert settings.max_url_body_bytes == 2_000_000  # nosec B101
    assert settings.url_request_timeout_seconds == 10.0  # nosec B101
    assert "text/html" in settings.allowed_content_types  # nosec B101
    assert settings.respect_robots is False  # nosec B101
    assert settings.allow_arbitrary_public_domains is False  # nosec B101


def test_from_mapping_parses_url_acquisition_values() -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": "true",
            "web_source_profile": "local_first",
            "preapproved_domains": "docs.python.org",
            "allowed_url_prefixes": ["https://docs.python.org/3/"],
            "denied_domains": ["blocked.example"],
            "max_url_redirects": "5",
            "max_url_body_bytes": "4096",
            "url_request_timeout_seconds": "2.5",
            "allowed_content_types": "text/plain",
            "url_user_agent": "tldw-docs-test/1",
            "respect_robots": "true",
            "allow_arbitrary_public_domains": "false",
        }
    )

    assert settings.web_source_profile == "local_first"  # nosec B101
    assert settings.preapproved_domains == ("docs.python.org",)  # nosec B101
    assert settings.allowed_url_prefixes == ("https://docs.python.org/3/",)  # nosec B101
    assert settings.denied_domains == ("blocked.example",)  # nosec B101
    assert settings.max_url_redirects == 5  # nosec B101
    assert settings.max_url_body_bytes == 4096  # nosec B101
    assert settings.url_request_timeout_seconds == 2.5  # nosec B101
    assert settings.allowed_content_types == ("text/plain",)  # nosec B101
    assert settings.url_user_agent == "tldw-docs-test/1"  # nosec B101
    assert settings.respect_robots is True  # nosec B101


@pytest.mark.parametrize("profile", ["", "open", "offline", "LOCAL_FIRST"])
def test_from_mapping_rejects_unknown_web_source_profile(profile: str) -> None:
    with pytest.raises(ValueError, match="web_source_profile"):
        DocsSettings.from_mapping({"web_source_profile": profile})


@pytest.mark.parametrize("field", ["max_url_redirects", "max_url_body_bytes"])
def test_from_mapping_rejects_non_positive_url_limits(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        DocsSettings.from_mapping({field: 0})


def test_from_mapping_rejects_non_positive_url_timeout() -> None:
    with pytest.raises(ValueError, match="url_request_timeout_seconds"):
        DocsSettings.from_mapping({"url_request_timeout_seconds": 0})
```

- [ ] **Step 2: Add failing disabled status test**

Extend `test_provider_status_reports_web_acquisition_disabled` in `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`:

```python
assert status["web_source_profile"] == "locked_down"  # nosec B101
assert status["web_acquisition_unavailable_reason"] == "web_acquisition_disabled"  # nosec B101
assert status["web_policy"]["allow_arbitrary_public_domains"] is False  # nosec B101
assert status["web_policy"]["preapproved_domains"] == []  # nosec B101
assert status["web_policy"]["allowed_url_prefixes"] == []  # nosec B101
```

- [ ] **Step 3: Run tests to verify the red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_status_reports_web_acquisition_disabled \
  -q
```

Expected: FAIL with missing attributes such as `web_source_profile`.

- [ ] **Step 4: Implement settings parsing**

In `mcp_unified/docs/settings.py`, add these definitions and fields:

```python
from typing import Literal

SourceProfile = Literal["locked_down", "local_first", "online_capable"]
_SOURCE_PROFILES = {"locked_down", "local_first", "online_capable"}
_DEFAULT_ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")
_DEFAULT_URL_USER_AGENT = "tldw-mcp-docs/0.1"


def _coerce_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        items = (value,)
    else:
        items = tuple(value) if isinstance(value, Iterable) else (value,)
    result = tuple(str(item).strip() for item in items if str(item).strip())
    return result


def _coerce_source_profile(value: object) -> SourceProfile:
    profile = str(value or "locked_down").strip().lower()
    if profile not in _SOURCE_PROFILES:
        raise ValueError("web_source_profile must be locked_down, local_first, or online_capable")
    return profile  # type: ignore[return-value]


def _coerce_positive_float(value: object, field_name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result
```

Extend `DocsSettings`:

```python
@dataclass(frozen=True)
class DocsSettings:
    db_path: Path
    trusted_roots: tuple[Path, ...] = ()
    max_import_file_bytes: int = 2_000_000
    default_scope: AccessScope = AccessScope()
    enable_web_acquisition: bool = False
    web_source_profile: SourceProfile = "locked_down"
    preapproved_domains: tuple[str, ...] = ()
    allowed_url_prefixes: tuple[str, ...] = ()
    denied_domains: tuple[str, ...] = ()
    max_url_redirects: int = 3
    max_url_body_bytes: int = 2_000_000
    url_request_timeout_seconds: float = 10.0
    allowed_content_types: tuple[str, ...] = _DEFAULT_ALLOWED_CONTENT_TYPES
    url_user_agent: str = _DEFAULT_URL_USER_AGENT
    respect_robots: bool = False
    allow_arbitrary_public_domains: bool = False
```

Update `from_mapping()` to pass every new setting through the coercers:

```python
web_source_profile=_coerce_source_profile(values.get("web_source_profile", "locked_down")),
preapproved_domains=_coerce_str_tuple(values.get("preapproved_domains"), "preapproved_domains"),
allowed_url_prefixes=_coerce_str_tuple(values.get("allowed_url_prefixes"), "allowed_url_prefixes"),
denied_domains=_coerce_str_tuple(values.get("denied_domains"), "denied_domains"),
max_url_redirects=_coerce_positive_int(values.get("max_url_redirects", 3), "max_url_redirects"),
max_url_body_bytes=_coerce_positive_int(values.get("max_url_body_bytes", 2_000_000), "max_url_body_bytes"),
url_request_timeout_seconds=_coerce_positive_float(
    values.get("url_request_timeout_seconds", 10.0),
    "url_request_timeout_seconds",
),
allowed_content_types=_coerce_str_tuple(
    values.get("allowed_content_types", _DEFAULT_ALLOWED_CONTENT_TYPES),
    "allowed_content_types",
),
url_user_agent=str(values.get("url_user_agent") or _DEFAULT_URL_USER_AGENT).strip() or _DEFAULT_URL_USER_AGENT,
respect_robots=_coerce_bool(values.get("respect_robots", False), "respect_robots"),
allow_arbitrary_public_domains=_coerce_bool(
    values.get("allow_arbitrary_public_domains", False),
    "allow_arbitrary_public_domains",
),
```

- [ ] **Step 5: Expand disabled status without constructing acquisition**

In `mcp_unified/docs/mcp_module.py`, add a helper:

```python
def _web_policy_status(settings: DocsSettings) -> dict[str, Any]:
    return {
        "profile": settings.web_source_profile,
        "allow_arbitrary_public_domains": settings.allow_arbitrary_public_domains,
        "preapproved_domains": list(settings.preapproved_domains),
        "allowed_url_prefixes": list(settings.allowed_url_prefixes),
        "denied_domains": list(settings.denied_domains),
        "max_url_redirects": settings.max_url_redirects,
        "max_url_body_bytes": settings.max_url_body_bytes,
        "allowed_content_types": list(settings.allowed_content_types),
        "respect_robots": settings.respect_robots,
    }
```

Update the `docs.status` branch:

```python
status["web_acquisition_enabled"] = self.settings.enable_web_acquisition
status["web_acquisition_available"] = False
status["web_source_profile"] = self.settings.web_source_profile
status["web_policy"] = _web_policy_status(self.settings)
status["web_acquisition_unavailable_reason"] = (
    "web_acquisition_disabled" if not self.settings.enable_web_acquisition else "web_acquisition_not_constructed"
)
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_status_reports_web_acquisition_disabled \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add mcp_unified/docs/settings.py mcp_unified/docs/mcp_module.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
git commit -m "feat: add docs url acquisition settings"
```

---

### Task 2: Pure Source Policy And Structured URL Matching

**Files:**

- Create: `mcp_unified/docs/acquisition/__init__.py`
- Create: `mcp_unified/docs/acquisition/models.py`
- Create: `mcp_unified/docs/acquisition/policy.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_policy.py`

- [ ] **Step 1: Add failing source-policy tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_policy.py`:

```python
from __future__ import annotations

import pytest

from mcp_unified.docs.acquisition.policy import SourcePolicy
from mcp_unified.docs.settings import DocsSettings


def _settings(**overrides: object) -> DocsSettings:
    values = {"enable_web_acquisition": True, **overrides}
    return DocsSettings.from_mapping(values)


def test_locked_down_ignores_domain_only_allow_rules() -> None:
    policy = SourcePolicy(_settings(web_source_profile="locked_down", preapproved_domains=["docs.python.org"]))

    decision = policy.evaluate("https://docs.python.org/3/library/sqlite3.html")

    assert decision.status == "approval_required"  # nosec B101
    assert decision.reason_code == "source_approval_required"  # nosec B101


def test_locked_down_allows_explicit_prefix() -> None:
    policy = SourcePolicy(
        _settings(web_source_profile="locked_down", allowed_url_prefixes=["https://docs.python.org/3/"])
    )

    decision = policy.evaluate("https://docs.python.org/3/library/sqlite3.html?token=secret")

    assert decision.status == "allowed"  # nosec B101
    assert decision.normalized_url.redacted_url == "https://docs.python.org/3/library/sqlite3.html"  # nosec B101


def test_prefix_matching_normalizes_host_case_and_default_port() -> None:
    policy = SourcePolicy(
        _settings(web_source_profile="locked_down", allowed_url_prefixes=["https://docs.python.org/3/"])
    )

    decision = policy.evaluate("https://DOCS.PYTHON.ORG:443/3/library/sqlite3.html")

    assert decision.status == "allowed"  # nosec B101
    assert decision.normalized_url.canonical_url == "https://docs.python.org/3/library/sqlite3.html"  # nosec B101


def test_local_first_unknown_public_domain_requires_approval() -> None:
    policy = SourcePolicy(_settings(web_source_profile="local_first"))

    decision = policy.evaluate("https://example.com/docs")

    assert decision.status == "approval_required"  # nosec B101
    assert decision.safe_argument_hash  # nosec B101
    assert "docs" in decision.normalized_url.redacted_url  # nosec B101


def test_online_capable_unknown_public_domain_requires_flag() -> None:
    policy = SourcePolicy(_settings(web_source_profile="online_capable", allow_arbitrary_public_domains=False))

    decision = policy.evaluate("https://example.com/docs")

    assert decision.status == "approval_required"  # nosec B101


def test_online_capable_allows_unknown_public_domain_when_flagged() -> None:
    policy = SourcePolicy(_settings(web_source_profile="online_capable", allow_arbitrary_public_domains=True))

    decision = policy.evaluate("https://example.com/docs")

    assert decision.status == "allowed"  # nosec B101


@pytest.mark.parametrize("url", ["file:///tmp/x.html", "ftp://example.com/x", "https:///missing-host"])
def test_policy_denies_unsupported_or_malformed_urls(url: str) -> None:
    policy = SourcePolicy(_settings(web_source_profile="online_capable", allow_arbitrary_public_domains=True))

    decision = policy.evaluate(url)

    assert decision.status == "denied"  # nosec B101


def test_policy_denies_url_credentials() -> None:
    policy = SourcePolicy(_settings(web_source_profile="online_capable", allow_arbitrary_public_domains=True))

    decision = policy.evaluate("https://user:password@example.com/docs")

    assert decision.status == "denied"  # nosec B101
    assert decision.reason_code == "url_credentials_denied"  # nosec B101


def test_denied_domain_wins_over_allowed_domain() -> None:
    policy = SourcePolicy(
        _settings(
            web_source_profile="local_first",
            preapproved_domains=["docs.example.com"],
            denied_domains=["docs.example.com"],
        )
    )

    decision = policy.evaluate("https://docs.example.com/page")

    assert decision.status == "denied"  # nosec B101
    assert decision.reason_code == "source_domain_denied"  # nosec B101


def test_domain_rules_are_exact_by_default() -> None:
    policy = SourcePolicy(_settings(web_source_profile="local_first", preapproved_domains=["example.com"]))

    assert policy.evaluate("https://example.com/docs").status == "allowed"  # nosec B101
    assert policy.evaluate("https://badexample.com/docs").status == "approval_required"  # nosec B101
    assert policy.evaluate("https://sub.example.com/docs").status == "approval_required"  # nosec B101


def test_wildcard_domain_rule_matches_subdomain_not_apex() -> None:
    policy = SourcePolicy(_settings(web_source_profile="local_first", preapproved_domains=["*.example.com"]))

    assert policy.evaluate("https://docs.example.com/page").status == "allowed"  # nosec B101
    assert policy.evaluate("https://example.com/page").status == "approval_required"  # nosec B101


def test_url_prefix_path_matching_uses_segment_boundaries() -> None:
    policy = SourcePolicy(
        _settings(web_source_profile="locked_down", allowed_url_prefixes=["https://example.com/docs/"])
    )

    assert policy.evaluate("https://example.com/docs/page").status == "allowed"  # nosec B101
    assert policy.evaluate("https://example.com/docs.evil/page").status == "approval_required"  # nosec B101
    assert policy.evaluate("https://example.com/docs%2Eevil/page").status == "approval_required"  # nosec B101


def test_safe_argument_hash_changes_with_query_without_leaking_query() -> None:
    policy = SourcePolicy(_settings(web_source_profile="local_first"))

    first = policy.evaluate("https://example.com/docs?token=one")
    second = policy.evaluate("https://example.com/docs?token=two")

    assert first.safe_argument_hash != second.safe_argument_hash  # nosec B101
    assert "token" not in first.normalized_url.redacted_url  # nosec B101
    assert "one" not in first.normalized_url.redacted_url  # nosec B101
```

- [ ] **Step 2: Run tests to verify the red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_policy.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_unified.docs.acquisition'`.

- [ ] **Step 3: Implement acquisition models**

Create `mcp_unified/docs/acquisition/models.py`:

```python
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

PolicyStatus = Literal["allowed", "approval_required", "denied"]
FetchResultStatus = Literal["fetched", "approval_required", "denied", "failed"]
IngestStatus = Literal["created", "updated", "unchanged", "approval_required", "denied", "failed", "capability_disabled"]


@dataclass(frozen=True)
class NormalizedURL:
    original_url: str
    canonical_url: str
    redacted_url: str
    scheme: str
    host: str
    port: int
    path: str
    query: str


@dataclass(frozen=True)
class SourceDecision:
    status: PolicyStatus
    reason_code: str
    normalized_url: NormalizedURL | None = None
    domain: str | None = None
    safe_argument_hash: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class URLRequest:
    normalized_url: NormalizedURL
    headers: Mapping[str, str]


@dataclass(frozen=True)
class ResolvedAddress:
    host: str
    ip: str
    port: int


@dataclass(frozen=True)
class FetchResponse:
    status_code: int
    headers: Mapping[str, str]
    body_chunks: Sequence[bytes]


class Resolver(Protocol):
    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        ...


class Transport(Protocol):
    dials_validated_address: bool

    def request(self, *, address: ResolvedAddress, request: URLRequest, timeout_seconds: float) -> FetchResponse:
        ...
```

Create `mcp_unified/docs/acquisition/__init__.py`:

```python
from __future__ import annotations

from .models import NormalizedURL, SourceDecision
from .policy import SourcePolicy

__all__ = ["NormalizedURL", "SourceDecision", "SourcePolicy"]
```

- [ ] **Step 4: Implement pure policy**

Create `mcp_unified/docs/acquisition/policy.py` with these public functions and classes:

```python
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from ..settings import DocsSettings
from .models import NormalizedURL, SourceDecision

_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class _DomainRule:
    host: str
    wildcard: bool

    def matches(self, host: str) -> bool:
        if self.wildcard:
            return host.endswith(f".{self.host}") and host != self.host
        return host == self.host


@dataclass(frozen=True)
class _URLPrefixRule:
    scheme: str
    host: str
    port: int
    path: str

    def matches(self, url: NormalizedURL) -> bool:
        if (url.scheme, url.host, url.port) != (self.scheme, self.host, self.port):
            return False
        return _path_has_prefix(url.path, self.path)


class SourcePolicy:
    def __init__(self, settings: DocsSettings) -> None:
        self.settings = settings
        self._allowed_domains = tuple(_parse_domain_rule(item) for item in settings.preapproved_domains)
        self._denied_domains = tuple(_parse_domain_rule(item) for item in settings.denied_domains)
        self._allowed_prefixes = tuple(
            rule for rule in (_parse_prefix_rule(item) for item in settings.allowed_url_prefixes) if rule is not None
        )

    def evaluate(self, raw_url: str) -> SourceDecision:
        if has_url_credentials(raw_url):
            return SourceDecision(status="denied", reason_code="url_credentials_denied")
        normalized = normalize_url(raw_url)
        if normalized is None:
            return SourceDecision(status="denied", reason_code="unsupported_url_scheme")

        if any(rule.matches(normalized.host) for rule in self._denied_domains):
            return SourceDecision(
                status="denied",
                reason_code="source_domain_denied",
                normalized_url=normalized,
                domain=normalized.host,
            )
        if any(rule.matches(normalized) for rule in self._allowed_prefixes):
            return SourceDecision(status="allowed", reason_code="source_allowed_by_prefix", normalized_url=normalized)
        if self.settings.web_source_profile != "locked_down" and any(
            rule.matches(normalized.host) for rule in self._allowed_domains
        ):
            return SourceDecision(status="allowed", reason_code="source_allowed_by_domain", normalized_url=normalized)
        if (
            self.settings.web_source_profile == "online_capable"
            and self.settings.allow_arbitrary_public_domains
        ):
            return SourceDecision(status="allowed", reason_code="source_allowed_by_profile", normalized_url=normalized)
        return SourceDecision(
            status="approval_required",
            reason_code="source_approval_required",
            normalized_url=normalized,
            domain=normalized.host,
            safe_argument_hash=_safe_argument_hash(raw_url),
        )
```

Add helper implementations in the same file:

```python
def normalize_url(raw_url: str) -> NormalizedURL | None:
    parsed = urlsplit(str(raw_url).strip())
    scheme = parsed.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parsed.hostname:
        return None
    host = _normalize_host(parsed.hostname)
    port = parsed.port or _DEFAULT_PORTS[scheme]
    path = _normalize_path(parsed.path or "/")
    canonical_url = urlunsplit((scheme, _netloc(host, port, scheme), path, parsed.query, ""))
    redacted_url = urlunsplit((scheme, _netloc(host, port, scheme), path, "", ""))
    return NormalizedURL(raw_url, canonical_url, redacted_url, scheme, host, port, path, parsed.query)


def has_url_credentials(raw_url: str) -> bool:
    parsed = urlsplit(str(raw_url).strip())
    return parsed.username is not None or parsed.password is not None


def _normalize_host(host: str) -> str:
    return host.encode("idna").decode("ascii").lower().rstrip(".")


def _netloc(host: str, port: int, scheme: str) -> str:
    return host if port == _DEFAULT_PORTS[scheme] else f"{host}:{port}"


def _normalize_path(path: str) -> str:
    decoded = unquote(path)
    if not decoded.startswith("/"):
        decoded = f"/{decoded}"
    safe = quote(decoded, safe="/:@")
    return safe or "/"


def _path_has_prefix(path: str, prefix: str) -> bool:
    decoded_path = unquote(path)
    decoded_prefix = unquote(prefix)
    if decoded_path == decoded_prefix.rstrip("/"):
        return True
    if not decoded_prefix.endswith("/"):
        decoded_prefix = f"{decoded_prefix}/"
    return decoded_path.startswith(decoded_prefix)


def _parse_domain_rule(raw: str) -> _DomainRule:
    text = _normalize_host(str(raw).strip())
    if text.startswith("*."):
        return _DomainRule(host=text[2:], wildcard=True)
    return _DomainRule(host=text, wildcard=False)


def _parse_prefix_rule(raw: str) -> _URLPrefixRule | None:
    normalized = normalize_url(raw)
    if normalized is None:
        return None
    return _URLPrefixRule(
        scheme=normalized.scheme,
        host=normalized.host,
        port=normalized.port,
        path=normalized.path,
    )


def _safe_argument_hash(raw_url: str) -> str:
    return sha256(str(raw_url).encode("utf-8")).hexdigest()
```

Keep the normalization helper pure and do not log raw URL query values.

- [ ] **Step 5: Run policy tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_policy.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add mcp_unified/docs/acquisition \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_policy.py
git commit -m "feat: add docs url source policy"
```

---

### Task 3: Resolver, IP Guard, Redirect-Aware Fetcher

**Files:**

- Create: `mcp_unified/docs/acquisition/resolver.py`
- Create: `mcp_unified/docs/acquisition/fetcher.py`
- Modify: `mcp_unified/docs/acquisition/models.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py`

- [ ] **Step 1: Add failing fetcher tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py`:

```python
from __future__ import annotations

from collections.abc import Iterable

from mcp_unified.docs.acquisition.fetcher import URLFetcher
from mcp_unified.docs.acquisition.models import FetchResponse, ResolvedAddress, URLRequest
from mcp_unified.docs.acquisition.policy import SourcePolicy
from mcp_unified.docs.settings import DocsSettings


class FakeResolver:
    def __init__(self, addresses: dict[str, list[str]]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        return [ResolvedAddress(host=host, ip=ip, port=port) for ip in self.addresses[host]]


class FakeTransport:
    dials_validated_address = True

    def __init__(self, responses: list[FetchResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[ResolvedAddress, URLRequest]] = []

    def request(self, *, address: ResolvedAddress, request: URLRequest, timeout_seconds: float) -> FetchResponse:
        self.calls.append((address, request))
        return self.responses.pop(0)


class ReResolvingTransport(FakeTransport):
    dials_validated_address = False


def _fetcher(
    *,
    resolver: FakeResolver,
    transport: FakeTransport,
    settings: DocsSettings | None = None,
) -> URLFetcher:
    actual_settings = settings or DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
            "max_url_body_bytes": 32,
        }
    )
    return URLFetcher(
        settings=actual_settings,
        policy=SourcePolicy(actual_settings),
        resolver=resolver,
        transport=transport,
    )


def test_fetcher_uses_validated_address_and_identity_encoding() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/html"}, [b"<h1>Ok</h1>"])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "fetched"  # nosec B101
    assert transport.calls[0][0].ip == "93.184.216.34"  # nosec B101
    assert transport.calls[0][1].headers["accept-encoding"] == "identity"  # nosec B101
    assert result.body == b"<h1>Ok</h1>"  # nosec B101


def test_fetcher_denies_private_ip_before_transport() -> None:
    resolver = FakeResolver({"internal.example": ["127.0.0.1"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/html"}, [b"never"])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://internal.example/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "egress_private_address_denied"  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_denies_transport_that_cannot_prove_validated_address_binding() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = ReResolvingTransport([FetchResponse(200, {"content-type": "text/html"}, [b"never"])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "dns_rebinding_risk"  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_returns_approval_required_without_resolver_or_transport() -> None:
    settings = DocsSettings.from_mapping({"enable_web_acquisition": True, "web_source_profile": "local_first"})
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/html"}, [b"never"])])

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/docs")

    assert result.status == "approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_fetcher_revalidates_redirect_target_and_denies_private_redirect() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"], "internal.example": ["10.0.0.5"]})
    transport = FakeTransport([FetchResponse(302, {"location": "https://internal.example/secret"}, [])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/docs")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "egress_private_address_denied"  # nosec B101
    assert len(transport.calls) == 1  # nosec B101


def test_fetcher_denies_redirect_target_that_requires_new_approval() -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "local_first",
            "preapproved_domains": ["example.com"],
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"], "other.example": ["93.184.216.35"]})
    transport = FakeTransport([FetchResponse(302, {"location": "https://other.example/docs"}, [])])

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/start")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "redirect_policy_denied"  # nosec B101
    assert len(transport.calls) == 1  # nosec B101


def test_fetcher_enforces_redirect_limit() -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
            "max_url_redirects": 1,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(302, {"location": "https://example.com/one"}, []),
            FetchResponse(302, {"location": "https://example.com/two"}, []),
        ]
    )

    result = _fetcher(resolver=resolver, transport=transport, settings=settings).fetch("https://example.com/start")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "redirect_limit_exceeded"  # nosec B101


def test_fetcher_denies_content_type_before_body_is_returned() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "application/octet-stream"}, [b"not read"])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/archive.bin")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "content_type_denied"  # nosec B101
    assert result.body == b""  # nosec B101


def test_fetcher_enforces_body_size_limit() -> None:
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/plain"}, [b"a" * 33])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/large.txt")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "content_too_large"  # nosec B101


def test_fetcher_enforces_decoded_body_size_limit_for_gzip() -> None:
    import gzip

    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    compressed = gzip.compress(b"a" * 64)
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/plain", "content-encoding": "gzip"}, [compressed])])

    result = _fetcher(resolver=resolver, transport=transport).fetch("https://example.com/compressed.txt")

    assert result.status == "denied"  # nosec B101
    assert result.reason_code == "content_too_large"  # nosec B101
```

- [ ] **Step 2: Run tests to verify the red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py -q
```

Expected: FAIL with missing `mcp_unified.docs.acquisition.fetcher`.

- [ ] **Step 3: Add fetch result models**

Extend `mcp_unified/docs/acquisition/models.py`:

```python
@dataclass(frozen=True)
class RedirectHop:
    from_url: str
    to_url: str
    status_code: int


@dataclass(frozen=True)
class FetchResult:
    status: FetchResultStatus
    reason_code: str
    final_url: str | None = None
    status_code: int | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    redirects: Sequence[RedirectHop] = ()
    warnings: Sequence[str] = ()
    safe_argument_hash: str | None = None
```

- [ ] **Step 4: Implement IP classification and resolver**

Create `mcp_unified/docs/acquisition/resolver.py`:

```python
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable

from .models import ResolvedAddress


def is_unsafe_egress_ip(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return any(
        (
            ip.is_loopback,
            ip.is_private,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_unspecified,
            ip.is_reserved,
        )
    )


class StdlibResolver:
    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        results = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        seen: set[str] = set()
        addresses: list[ResolvedAddress] = []
        for family, _socktype, _proto, _canonname, sockaddr in results:
            if family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            ip_text = str(sockaddr[0])
            if ip_text in seen:
                continue
            seen.add(ip_text)
            addresses.append(ResolvedAddress(host=host, ip=ip_text, port=port))
        return addresses
```

- [ ] **Step 5: Implement fetcher orchestration**

Create `mcp_unified/docs/acquisition/fetcher.py` with `URLFetcher.fetch()`:

```python
from __future__ import annotations

import socket
import ssl
import gzip
import zlib
from collections.abc import Mapping
from urllib.parse import urljoin

from ..settings import DocsSettings
from .models import FetchResponse, FetchResult, RedirectHop, ResolvedAddress, Transport, URLRequest
from .policy import SourcePolicy
from .resolver import StdlibResolver, is_unsafe_egress_ip

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class URLFetcher:
    def __init__(
        self,
        *,
        settings: DocsSettings,
        policy: SourcePolicy,
        resolver: object | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.resolver = resolver or StdlibResolver()
        self.transport = transport or ValidatedAddressHTTPTransport()

    def fetch(self, raw_url: str) -> FetchResult:
        current_url = raw_url
        redirects: list[RedirectHop] = []
        for _hop in range(self.settings.max_url_redirects + 1):
            decision = self.policy.evaluate(current_url)
            if decision.status != "allowed":
                return FetchResult(
                    status=decision.status,
                    reason_code=decision.reason_code,
                    final_url=decision.normalized_url.redacted_url if decision.normalized_url else None,
                    redirects=tuple(redirects),
                    safe_argument_hash=decision.safe_argument_hash,
                )
            normalized = decision.normalized_url
            if normalized is None:
                return FetchResult(status="denied", reason_code="unsupported_url_scheme", redirects=tuple(redirects))
            if not getattr(self.transport, "dials_validated_address", False):
                return FetchResult(status="denied", reason_code="dns_rebinding_risk", redirects=tuple(redirects))
            addresses = list(self.resolver.resolve(normalized.host, normalized.port))
            if not addresses:
                return FetchResult(status="failed", reason_code="fetch_error", redirects=tuple(redirects))
            for address in addresses:
                if is_unsafe_egress_ip(address.ip):
                    return FetchResult(
                        status="denied",
                        reason_code="egress_private_address_denied",
                        final_url=normalized.redacted_url,
                        redirects=tuple(redirects),
                    )
            request = URLRequest(
                normalized_url=normalized,
                headers={
                    "host": normalized.host,
                    "user-agent": self.settings.url_user_agent,
                    "accept": ", ".join(self.settings.allowed_content_types),
                    "accept-encoding": "identity",
                },
            )
            response = self.transport.request(
                address=addresses[0],
                request=request,
                timeout_seconds=self.settings.url_request_timeout_seconds,
            )
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            if response.status_code in _REDIRECT_STATUSES and "location" in headers:
                if len(redirects) >= self.settings.max_url_redirects:
                    return FetchResult(status="denied", reason_code="redirect_limit_exceeded", redirects=tuple(redirects))
                next_url = urljoin(normalized.canonical_url, headers["location"])
                redirect_decision = self.policy.evaluate(next_url)
                if redirect_decision.status != "allowed" or redirect_decision.normalized_url is None:
                    return FetchResult(
                        status="denied",
                        reason_code="redirect_policy_denied",
                        final_url=normalized.redacted_url,
                        redirects=tuple(redirects),
                    )
                redirects.append(RedirectHop(normalized.redacted_url, redirect_decision.normalized_url.redacted_url, response.status_code))
                current_url = next_url
                continue
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type and content_type not in self.settings.allowed_content_types:
                return FetchResult(
                    status="denied",
                    reason_code="content_type_denied",
                    final_url=normalized.redacted_url,
                    status_code=response.status_code,
                    headers=headers,
                    redirects=tuple(redirects),
                )
            transferred_body = _bounded_body(response.body_chunks, self.settings.max_url_body_bytes)
            if transferred_body is None:
                return FetchResult(status="denied", reason_code="content_too_large", redirects=tuple(redirects))
            body = _decoded_body(transferred_body, headers.get("content-encoding", "identity"), self.settings.max_url_body_bytes)
            if body is None:
                return FetchResult(status="denied", reason_code="content_too_large", redirects=tuple(redirects))
            return FetchResult(
                status="fetched",
                reason_code="ok",
                final_url=normalized.canonical_url,
                status_code=response.status_code,
                headers=headers,
                body=body,
                redirects=tuple(redirects),
            )
        return FetchResult(status="denied", reason_code="redirect_limit_exceeded", redirects=tuple(redirects))
```

Add `_bounded_body()` in the same file:

```python
def _bounded_body(chunks: object, max_bytes: int) -> bytes | None:
    total = 0
    parts: list[bytes] = []
    for chunk in chunks:
        total += len(chunk)
        if total > max_bytes:
            return None
        parts.append(chunk)
    return b"".join(parts)


def _decoded_body(body: bytes, content_encoding: str, max_bytes: int) -> bytes | None:
    encoding = content_encoding.split(",", 1)[0].strip().lower() or "identity"
    if encoding == "identity":
        return body
    if encoding == "gzip":
        decoded = gzip.decompress(body)
    elif encoding == "deflate":
        decoded = zlib.decompress(body)
    else:
        return None
    return decoded if len(decoded) <= max_bytes else None
```

- [ ] **Step 6: Implement baseline validated-address transport**

Add `ValidatedAddressHTTPTransport` in `mcp_unified/docs/acquisition/fetcher.py`. The important property is that it dials `address.ip`, sets `Host`, and uses SNI for HTTPS, so it cannot silently re-resolve `request.normalized_url.host`.

```python
class ValidatedAddressHTTPTransport:
    dials_validated_address = True

    def request(self, *, address: ResolvedAddress, request: URLRequest, timeout_seconds: float) -> FetchResponse:
        host = request.normalized_url.host
        port = request.normalized_url.port
        path = request.normalized_url.path
        if request.normalized_url.query:
            path = f"{path}?{request.normalized_url.query}"
        with socket.create_connection((address.ip, port), timeout=timeout_seconds) as raw_socket:
            if request.normalized_url.scheme == "https":
                context = ssl.create_default_context()
                with context.wrap_socket(raw_socket, server_hostname=host) as wrapped:
                    return _send_http_request(wrapped, request, path)
            return _send_http_request(raw_socket, request, path)
```

Implement `_send_http_request()` as a small HTTP/1.1 GET writer/reader that:

- sends `GET {path} HTTP/1.1`;
- sends all configured headers plus `Connection: close`;
- parses the status line;
- parses response headers until the empty line;
- returns body as one bytes chunk read from the socket after headers.

The tests in this task use `FakeTransport`; add a separate smoke unit for `_bounded_body()` if the socket reader needs focused coverage.

- [ ] **Step 7: Run fetcher tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add mcp_unified/docs/acquisition/models.py \
  mcp_unified/docs/acquisition/resolver.py \
  mcp_unified/docs/acquisition/fetcher.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py
git commit -m "feat: add safe docs url fetcher"
```

---

### Task 4: Lazy Extraction Pipeline And Parsed Document Generalization

**Files:**

- Create: `mcp_unified/docs/acquisition/extract.py`
- Modify: `mcp_unified/docs/importers/base.py`
- Modify: `mcp_unified/docs/importers/html.py`
- Modify: `mcp_unified/docs/importers/local.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py`

- [ ] **Step 1: Add failing extraction tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py`:

```python
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from mcp_unified.docs.acquisition.extract import available_extractors, extract_fetched_document


def test_static_html_fallback_extracts_title_sections_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    parsed = extract_fetched_document(
        url="https://example.com/docs",
        content_type="text/html",
        body=b"<html><body><h1>Guide</h1><p>SQLite FTS5 content.</p><script>skip()</script></body></html>",
    )

    assert parsed.title == "Guide"  # nosec B101
    assert parsed.document_type == "html"  # nosec B101
    assert parsed.source_path is None  # nosec B101
    assert parsed.source_url == "https://example.com/docs"  # nosec B101
    assert parsed.canonical_uri == "https://example.com/docs"  # nosec B101
    assert parsed.extraction_method == "static_html"  # nosec B101
    assert "SQLite FTS5 content." in parsed.text  # nosec B101


def test_plain_text_extraction() -> None:
    parsed = extract_fetched_document(
        url="https://example.com/readme.txt",
        content_type="text/plain",
        body=b"Line one\\nLine two",
    )

    assert parsed.title == "readme.txt"  # nosec B101
    assert parsed.document_type == "text"  # nosec B101
    assert parsed.extraction_method == "text"  # nosec B101
    assert parsed.text == "Line one\\nLine two"  # nosec B101


def test_trafilatura_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    module = SimpleNamespace(extract=lambda text, include_comments=False, include_tables=True: "Rich body")

    def fake_import(name: str) -> object:
        if name == "trafilatura":
            return module
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    parsed = extract_fetched_document(
        url="https://example.com/docs",
        content_type="text/html",
        body=b"<h1>Guide</h1><p>Rich body</p>",
    )

    assert parsed.extraction_method == "trafilatura"  # nosec B101
    assert parsed.text == "Rich body"  # nosec B101


def test_available_extractors_uses_lazy_import_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        if name == "bs4":
            return SimpleNamespace(BeautifulSoup=object)
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert available_extractors() == ["beautifulsoup", "static_html", "text"]  # nosec B101
```

- [ ] **Step 2: Run tests to verify the red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py -q
```

Expected: FAIL with missing `extract_fetched_document` or missing generalized parsed document fields.

- [ ] **Step 3: Generalize parsed documents**

Modify `mcp_unified/docs/importers/base.py`:

```python
@dataclass(frozen=True)
class ParsedDocument:
    title: str
    document_type: str
    text: str
    sections: list[ParsedSection]
    canonical_uri: str
    source_path: str | None = None
    source_url: str | None = None
    extraction_method: str | None = None
    warnings: tuple[str, ...] = ()
```

Modify `mcp_unified/docs/importers/local.py` so the store call uses `source_path=parsed.source_path` and keeps `source_url=parsed.source_url`. Existing local parsers will still set `source_path`.

- [ ] **Step 4: Split static HTML parsing into path and string entry points**

In `mcp_unified/docs/importers/html.py`, add:

```python
def parse_html_document(
    *,
    text: str,
    title_hint: str,
    canonical_uri: str,
    source_path: str | None = None,
    source_url: str | None = None,
    extraction_method: str = "static_html",
    warnings: tuple[str, ...] = (),
) -> ParsedDocument:
    parser = StaticHTMLTextParser()
    parser.feed(text)
    body = "\n".join(part.strip() for part in parser.parts if part.strip())
    title = parser.sections[0].heading if parser.sections else title_hint
    return ParsedDocument(
        title=title,
        document_type="html",
        text=body,
        sections=parser.sections,
        canonical_uri=canonical_uri,
        source_path=source_path,
        source_url=source_url,
        extraction_method=extraction_method,
        warnings=warnings,
    )
```

Update `parse_html(path, text)` to delegate:

```python
def parse_html(path: Path, text: str) -> ParsedDocument:
    return parse_html_document(
        text=text,
        title_hint=path.stem,
        canonical_uri=file_uri(path),
        source_path=str(path.resolve()),
    )
```

- [ ] **Step 5: Implement extraction order with lazy imports**

Create `mcp_unified/docs/acquisition/extract.py`:

```python
from __future__ import annotations

import importlib
from urllib.parse import urlsplit

from ..importers.base import ParsedDocument
from ..importers.html import parse_html_document


def available_extractors() -> list[str]:
    names: list[str] = []
    if _can_import("trafilatura"):
        names.append("trafilatura")
    if _can_import("bs4"):
        names.append("beautifulsoup")
    names.extend(["static_html", "text"])
    return names


def extract_fetched_document(*, url: str, content_type: str, body: bytes) -> ParsedDocument:
    text = _decode_body(body, content_type)
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type in {"text/plain", "text/markdown"}:
        return ParsedDocument(
            title=_title_from_url(url),
            document_type="text",
            text=text.strip(),
            sections=[],
            canonical_uri=url,
            source_url=url,
            extraction_method="text",
        )
    trafilatura_text = _extract_with_trafilatura(text)
    if trafilatura_text:
        return ParsedDocument(
            title=_title_from_url(url),
            document_type="html",
            text=trafilatura_text,
            sections=[],
            canonical_uri=url,
            source_url=url,
            extraction_method="trafilatura",
        )
    soup_text = _extract_with_beautifulsoup(text)
    if soup_text:
        return ParsedDocument(
            title=_title_from_url(url),
            document_type="html",
            text=soup_text,
            sections=[],
            canonical_uri=url,
            source_url=url,
            extraction_method="beautifulsoup",
        )
    return parse_html_document(
        text=text,
        title_hint=_title_from_url(url),
        canonical_uri=url,
        source_url=url,
        extraction_method="static_html",
    )
```

Add helpers in the same file:

```python
def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _decode_body(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    for part in content_type.split(";")[1:]:
        key, _sep, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            charset = value.strip()
    return body.decode(charset, errors="replace")


def _title_from_url(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    return path.rsplit("/", 1)[-1] or urlsplit(url).hostname or "Untitled"


def _extract_with_trafilatura(text: str) -> str | None:
    try:
        trafilatura = importlib.import_module("trafilatura")
    except ImportError:
        return None
    extracted = trafilatura.extract(text, include_comments=False, include_tables=True)
    return extracted.strip() if isinstance(extracted, str) and extracted.strip() else None


def _extract_with_beautifulsoup(text: str) -> str | None:
    try:
        bs4 = importlib.import_module("bs4")
    except ImportError:
        return None
    soup = bs4.BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    extracted = soup.get_text("\n", strip=True)
    return extracted.strip() if extracted.strip() else None
```

- [ ] **Step 6: Run extraction and existing importer tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add mcp_unified/docs/acquisition/extract.py \
  mcp_unified/docs/importers/base.py \
  mcp_unified/docs/importers/html.py \
  mcp_unified/docs/importers/local.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_extract.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py
git commit -m "feat: add lazy docs url extraction"
```

---

### Task 5: Acquisition Service And Store Integration

**Files:**

- Create: `mcp_unified/docs/acquisition/service.py`
- Modify: `mcp_unified/docs/acquisition/__init__.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py`

- [ ] **Step 1: Add failing service tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from mcp_unified.docs.acquisition.models import FetchResponse, ResolvedAddress, URLRequest
from mcp_unified.docs.acquisition.service import DocsAcquisitionService
from mcp_unified.docs.models import AccessScope, ContextRequest, SearchFilters, SearchRequest
from mcp_unified.docs.retrieval.context import DocsContextBuilder
from mcp_unified.docs.retrieval.search import DocsRetrievalService
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore


class FakeResolver:
    def __init__(self, addresses: dict[str, list[str]]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    def resolve(self, host: str, port: int) -> Iterable[ResolvedAddress]:
        self.calls.append((host, port))
        return [ResolvedAddress(host=host, ip=ip, port=port) for ip in self.addresses[host]]


class FakeTransport:
    dials_validated_address = True

    def __init__(self, responses: list[FetchResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[ResolvedAddress, URLRequest]] = []

    def request(self, *, address: ResolvedAddress, request: URLRequest, timeout_seconds: float) -> FetchResponse:
        self.calls.append((address, request))
        return self.responses.pop(0)


def _store(tmp_path: Path) -> DocsCatalogStore:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    return store


def test_service_returns_approval_required_without_fetch(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping({"enable_web_acquisition": True, "web_source_profile": "local_first"})
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/html"}, [b"never"])])
    service = DocsAcquisitionService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/docs")

    assert result["status"] == "approval_required"  # nosec B101
    assert result["reason_code"] == "source_approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_service_fails_closed_when_robots_is_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
            "respect_robots": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(200, {"content-type": "text/html"}, [b"never"])])
    service = DocsAcquisitionService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/docs")

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "robots_unavailable"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_service_ingests_approved_page_into_search_and_context(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ["https://example.com/docs/"],
        }
    )
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [FetchResponse(200, {"content-type": "text/html"}, [b"<h1>SQLite Guide</h1><p>FTS5 indexing details.</p>"])]
    )
    service = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = service.ingest_url(
        scope=scope,
        url="https://example.com/docs/sqlite.html",
        keywords=("sqlite", "fts5"),
        collection_names=("Reference",),
    )
    search = DocsRetrievalService(store).search(
        scope=scope,
        request=SearchRequest(query="FTS5", filters=SearchFilters(collection="Reference", keywords=("fts5",))),
    )
    context = DocsContextBuilder(DocsRetrievalService(store)).build(scope=scope, request=ContextRequest(query="FTS5"))

    assert result["status"] == "created"  # nosec B101
    assert result["document"]["source_url"] == "https://example.com/docs/sqlite.html"  # nosec B101
    assert search["results"][0]["title"] == "SQLite Guide"  # nosec B101
    assert context["chunks"]  # nosec B101


def test_service_reports_unchanged_for_same_content(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(200, {"content-type": "text/plain"}, [b"same body"]),
            FetchResponse(200, {"content-type": "text/plain"}, [b"same body"]),
        ]
    )
    service = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)

    first = service.ingest_url(scope=AccessScope(), url="https://example.com/readme.txt")
    second = service.ingest_url(scope=AccessScope(), url="https://example.com/readme.txt")

    assert first["status"] == "created"  # nosec B101
    assert second["status"] == "unchanged"  # nosec B101
```

- [ ] **Step 2: Run tests to verify the red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py -q
```

Expected: FAIL with missing `DocsAcquisitionService`.

- [ ] **Step 3: Implement service coordination**

Create `mcp_unified/docs/acquisition/service.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any

from ..errors import DocsError
from ..importers.base import chunks_from_text
from ..models import AccessScope
from ..settings import DocsSettings
from ..store.sqlite import DocsCatalogStore
from .extract import extract_fetched_document
from .fetcher import URLFetcher
from .policy import SourcePolicy


class DocsAcquisitionService:
    def __init__(
        self,
        *,
        settings: DocsSettings,
        store: DocsCatalogStore,
        resolver: object | None = None,
        transport: object | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.policy = SourcePolicy(settings)
        self.fetcher = URLFetcher(settings=settings, policy=self.policy, resolver=resolver, transport=transport)

    def ingest_url(
        self,
        *,
        scope: AccessScope,
        url: str,
        keywords: Iterable[str] = (),
        collection_names: Iterable[str] = (),
        title_override: str | None = None,
    ) -> dict[str, Any]:
        if self.settings.respect_robots:
            return {"status": "denied", "reason_code": "robots_unavailable"}
        fetched = self.fetcher.fetch(url)
        if fetched.status != "fetched":
            return {
                "status": fetched.status,
                "reason_code": fetched.reason_code,
                "final_url": fetched.final_url,
                "redirects": [asdict(item) for item in fetched.redirects],
                "safe_argument_hash": fetched.safe_argument_hash,
            }
        content_type = fetched.headers.get("content-type", "text/html")
        parsed = extract_fetched_document(url=fetched.final_url or url, content_type=content_type, body=fetched.body)
        if title_override:
            parsed = replace(parsed, title=title_override)
        if not parsed.text.strip():
            return {"status": "failed", "reason_code": "extract_empty", "final_url": fetched.final_url}

        previous_hash = _existing_content_hash(self.store, scope, parsed.canonical_uri)
        new_hash = sha256(parsed.text.encode("utf-8")).hexdigest()
        chunks = [
            {"text": chunk, "citation": f"{parsed.source_url or parsed.canonical_uri}#{idx + 1}"}
            for idx, chunk in enumerate(chunks_from_text(parsed.text))
        ]
        document_id = self.store.upsert_document(
            scope=scope,
            title=parsed.title,
            document_type=parsed.document_type,
            canonical_uri=parsed.canonical_uri,
            source_path=parsed.source_path,
            source_url=parsed.source_url,
            text=parsed.text,
            sections=[asdict(section) for section in parsed.sections],
            chunks=chunks,
            keywords=keywords,
            collection_names=collection_names,
            metadata={
                "importer": "url",
                "extraction_method": parsed.extraction_method,
                "fetch_status_code": fetched.status_code,
                "redirect_count": len(fetched.redirects),
            },
        )
        status = "created" if previous_hash is None else "unchanged" if previous_hash == new_hash else "updated"
        return {
            "status": status,
            "reason_code": "ok",
            "document": {
                "id": document_id,
                "title": parsed.title,
                "canonical_uri": parsed.canonical_uri,
                "source_url": parsed.source_url,
                "chunks": len(chunks),
                "extraction_method": parsed.extraction_method,
            },
            "fetch": {
                "final_url": fetched.final_url,
                "status_code": fetched.status_code,
                "redirects": [asdict(item) for item in fetched.redirects],
                "content_type": content_type,
                "bytes": len(fetched.body),
            },
        }
```

Add helper:

```python
def _existing_content_hash(store: DocsCatalogStore, scope: AccessScope, canonical_uri: str) -> str | None:
    try:
        document = store.get_document(scope, canonical_uri, mode="snippet")
    except DocsError as exc:
        if exc.code == "document_not_found":
            return None
        raise
    value = document.get("content_hash")
    return str(value) if value else None
```

Update `mcp_unified/docs/acquisition/__init__.py`:

```python
from .service import DocsAcquisitionService

__all__ = ["DocsAcquisitionService", "NormalizedURL", "SourceDecision", "SourcePolicy"]
```

- [ ] **Step 4: Run service tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add mcp_unified/docs/acquisition/service.py \
  mcp_unified/docs/acquisition/__init__.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py
git commit -m "feat: add docs url acquisition service"
```

---

### Task 6: MCP Provider Tool Exposure And Host Shim Validation

**Files:**

- Modify: `mcp_unified/docs/mcp_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`

- [ ] **Step 1: Add failing MCP provider tests**

Add to `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`:

```python
def test_provider_stale_ingest_url_call_is_disabled_when_not_advertised(tmp_path: Path) -> None:
    provider, scope, _document_id = _provider(tmp_path)

    result = provider.execute("docs.ingest_url", {"url": "https://example.com/docs"}, scope=scope)

    assert result["status"] == "capability_disabled"  # nosec B101
    assert result["reason_code"] == "web_acquisition_disabled"  # nosec B101


def test_provider_advertises_ingest_url_when_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ["https://example.com/docs/"],
        }
    )
    provider = DocsMCPToolProvider(settings=settings)

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    assert "docs.ingest_url" in tools  # nosec B101
    assert tools["docs.ingest_url"]["metadata"]["category"] == "ingestion"  # nosec B101
    assert tools["docs.ingest_url"]["metadata"]["readOnlyHint"] is False  # nosec B101


def test_provider_status_reports_enabled_static_extractors(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ["https://example.com/docs/"],
        }
    )
    provider = DocsMCPToolProvider(settings=settings)

    status = provider.execute("docs.status", {}, scope=AccessScope())

    assert status["web_acquisition_enabled"] is True  # nosec B101
    assert status["web_acquisition_available"] is True  # nosec B101
    assert "static_html" in status["web_extractors"]  # nosec B101
```

Add to `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`:

```python
@pytest.mark.asyncio
async def test_docs_module_rejects_empty_ingest_url(tmp_path: Path) -> None:
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={
                "db_path": str(tmp_path / "docs.db"),
                "enable_web_acquisition": True,
                "web_source_profile": "locked_down",
                "allowed_url_prefixes": ["https://example.com/docs/"],
            },
        )
    )
    await module.on_initialize()

    with pytest.raises(ValueError, match="url is required"):
        await module.execute_tool("docs.ingest_url", {"url": "   "}, context=None)
```

- [ ] **Step 2: Run tests to verify the red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  -q
```

Expected: FAIL because `docs.ingest_url` is not handled.

- [ ] **Step 3: Construct acquisition service only when enabled**

In `mcp_unified/docs/mcp_module.py`, import the service and extractor status:

```python
from .acquisition.extract import available_extractors
from .acquisition.service import DocsAcquisitionService
```

Update `DocsMCPToolProvider.__init__`:

```python
self.acquisition = (
    DocsAcquisitionService(settings=settings, store=self.store)
    if settings.enable_web_acquisition
    else None
)
```

- [ ] **Step 4: Advertise `docs.ingest_url` only when enabled**

Change `tool_definitions()` to build a list and append the URL tool conditionally:

```python
tools = [
    _tool("docs.search", "Search the local docs corpus.", {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"], "search"),
    ...
]
if self.acquisition is not None:
    tools.append(
        _tool(
            "docs.ingest_url",
            "Fetch and ingest one approved HTTP or HTTPS page into the local docs corpus.",
            {
                "url": {"type": "string"},
                "keywords": {"type": "array"},
                "collections": {"type": "array"},
                "title": {"type": "string"},
            },
            ["url"],
            "ingestion",
        )
    )
return tools
```

Keep `docs.import_path`, collection management, and Context7 compatibility tools unchanged.

- [ ] **Step 5: Execute `docs.ingest_url` and stale disabled calls**

In `execute()` before `_execute_management_or_list()`:

```python
if tool_name == "docs.ingest_url":
    if self.acquisition is None:
        return {"status": "capability_disabled", "reason_code": "web_acquisition_disabled"}
    return self.acquisition.ingest_url(
        scope=scope,
        url=str(args["url"]),
        keywords=tuple(str(item) for item in args.get("keywords") or ()),
        collection_names=tuple(str(item) for item in args.get("collections") or ()),
        title_override=_optional_str(args.get("title")),
    )
```

Update `docs.status`:

```python
status["web_acquisition_available"] = self.acquisition is not None
status["web_extractors"] = available_extractors() if self.acquisition is not None else []
status["web_acquisition_unavailable_reason"] = (
    None if self.acquisition is not None else "web_acquisition_disabled"
)
```

- [ ] **Step 6: Add host shim validation**

In `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`, extend `validate_tool_arguments()`:

```python
if tool_name == "docs.ingest_url" and not str(arguments.get("url") or "").strip():
    raise ValueError("url is required")
```

- [ ] **Step 7: Run provider and shim tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 6**

Run:

```bash
git add mcp_unified/docs/mcp_module.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py
git commit -m "feat: expose docs url ingestion tool"
```

---

### Task 7: Import Boundary, Config Defaults, And No-Live-Internet Guards

**Files:**

- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`

- [ ] **Step 1: Add stricter import-boundary expectations**

Modify `test_docs_package_does_not_import_optional_web_acquisition_dependencies()`:

```python
def test_docs_package_does_not_import_optional_web_acquisition_dependencies() -> None:
    forbidden = {"playwright", "trafilatura", "requests", "aiohttp", "httpx", "bs4"}
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            root = name.split(".", 1)[0]
            if root in forbidden:
                violations.append((str(path), name))

    assert violations == []  # nosec B101
```

Add a test that package import does not load optional modules into `sys.modules`:

```python
def test_docs_package_import_does_not_load_rich_extractors() -> None:
    import sys

    for name in ["trafilatura", "bs4"]:
        sys.modules.pop(name, None)

    importlib.import_module("mcp_unified.docs")

    assert "trafilatura" not in sys.modules  # nosec B101
    assert "bs4" not in sys.modules  # nosec B101
```

- [ ] **Step 2: Add config default test in the host shim test file**

Add to `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`:

```python
def test_repo_docs_mcp_config_keeps_web_acquisition_disabled() -> None:
    config_path = Path("tldw_Server_API/Config_Files/mcp_modules.yaml")
    text = config_path.read_text(encoding="utf-8")

    assert "id: docs" in text  # nosec B101
    assert "enable_web_acquisition: false" in text  # nosec B101
    assert "allow_arbitrary_public_domains: true" not in text  # nosec B101
```

- [ ] **Step 3: Run tests to verify the current state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py::test_repo_docs_mcp_config_keeps_web_acquisition_disabled \
  -q
```

Expected: PASS if previous tasks kept lazy imports and config disabled. If this fails, fix only the import boundary or config default that caused the failure.

- [ ] **Step 4: Keep config disabled while exposing safe defaults**

In `tldw_Server_API/Config_Files/mcp_modules.yaml`, keep:

```yaml
      enable_web_acquisition: false
```

Add the locked-down defaults explicitly under the docs module settings:

```yaml
      web_source_profile: locked_down
      allow_arbitrary_public_domains: false
      preapproved_domains: []
      allowed_url_prefixes: []
      denied_domains: []
      max_url_redirects: 3
      max_url_body_bytes: 2000000
      url_request_timeout_seconds: 10.0
      allowed_content_types:
        - text/html
        - application/xhtml+xml
        - text/plain
        - text/markdown
      respect_robots: false
```

Do not add any online-capable sample to the enabled repo config.

- [ ] **Step 5: Run import-boundary and config tests again**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py::test_repo_docs_mcp_config_keeps_web_acquisition_disabled \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  tldw_Server_API/Config_Files/mcp_modules.yaml
git commit -m "test: harden docs url acquisition boundaries"
```

---

### Task 8: Full Verification, Security Scan, And Final Review

**Files:**

- Modify: Backlog.md task for the Stage 2 implementation work
- Inspect: every file changed by Tasks 1-7

- [ ] **Step 1: Run all docs MCP tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent write-tool validator tests if present**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified -k "docs or write_tools or validator" -q
```

Expected: PASS or a collected subset that passes. If pytest reports no tests selected for part of the expression, record that in the Backlog task notes.

- [ ] **Step 3: Run import smoke checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python - <<'PY'
import importlib
import sys

for name in ["trafilatura", "bs4", "requests", "httpx", "aiohttp", "playwright"]:
    sys.modules.pop(name, None)

module = importlib.import_module("mcp_unified.docs")
print(module.DocsSettings.from_mapping({}).enable_web_acquisition)
loaded = [name for name in ["trafilatura", "bs4", "requests", "httpx", "aiohttp", "playwright"] if name in sys.modules]
print("loaded_optional=", loaded)
PY
```

Expected output includes:

```text
False
loaded_optional= []
```

- [ ] **Step 4: Run Bandit on touched Python paths**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r mcp_unified/docs/acquisition \
     mcp_unified/docs/settings.py \
     mcp_unified/docs/importers/base.py \
     mcp_unified/docs/importers/html.py \
     mcp_unified/docs/importers/local.py \
     mcp_unified/docs/mcp_module.py \
     tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  -f json -o /tmp/bandit_mcp_docs_url_acquisition.json
```

Expected: PASS with no new high or medium findings in touched code. If Bandit reports findings caused by the socket/SSL baseline transport, fix those findings or document why they are false positives with the exact Bandit IDs and line numbers.

- [ ] **Step 5: Review the security invariants from the spec**

Use this checklist and verify each item has a passing test:

```text
[ ] docs.ingest_url hidden when disabled
[ ] stale direct call returns capability_disabled before policy/fetch
[ ] no-fetch-before-approval with resolver/transport counters
[ ] locked_down ignores domain-only allows and accepts explicit prefixes
[ ] online_capable needs allow_arbitrary_public_domains=true for unknown public domains
[ ] structured exact domain and explicit wildcard matching
[ ] decoded URL prefix path-segment boundary matching
[ ] credentials and unsupported schemes denied
[ ] denied domain wins over allowed domain
[ ] private/loopback/link-local/multicast/unspecified/reserved IPs denied
[ ] transport must dial resolver-validated address
[ ] redirect target re-runs source policy and DNS/IP checks
[ ] redirect limit enforced
[ ] content type denied before body is returned
[ ] transferred body size limit enforced
[ ] respect_robots=true returns robots_unavailable before resolver/transport
[ ] trafilatura and bs4 are lazy and absent from AST import-boundary scan
[ ] approved fake URL ingests into docs store, search, and context
[ ] keywords and collections are applied during URL ingestion
[ ] repo config keeps web acquisition disabled
```

- [ ] **Step 6: Update Backlog task**

Use the official Backlog.md workflow available in the environment. If MCP tools are available, update the implementation task with:

```text
Status: Done
Implementation notes: Added optional standalone docs URL acquisition with source policy, safe fetcher, lazy extraction, MCP provider exposure, host shim validation, and import-boundary tests.
Verification: Include the exact pytest and Bandit commands run and their results.
Touched files: List acquisition package files, settings/importer/provider/shim files, config, and tests.
```

If MCP tools are unavailable, use the CLI fallback:

```bash
backlog task edit <IMPLEMENTATION_TASK_ID> \
  --status Done \
  --notes "Implemented optional standalone MCP docs URL acquisition; verification commands recorded in final summary."
```

Inspect the task file afterward to ensure acceptance criteria and definition-of-done checkboxes remain correctly formatted.

- [ ] **Step 7: Commit final task update if needed**

Run:

```bash
git status --short
git add backlog/tasks/<IMPLEMENTATION_TASK_FILE>.md
git commit -m "chore: close docs url acquisition task"
```

Only run this commit when the Backlog task update is not already included in the previous implementation commit.

- [ ] **Step 8: Final response**

Report:

```text
Implemented optional standalone MCP docs URL acquisition.

Verification:
- <pytest docs command>: PASS
- <adjacent tests command>: PASS
- <Bandit command>: PASS

Notes:
- Web acquisition remains disabled in the repo config by default.
- Rich extractors remain lazy; the standalone docs package import does not load optional web dependencies.
```

---

## Self-Review Checklist For Plan Readers

- Spec coverage: Tasks 1-8 cover settings, source policy, structured domain/prefix matching, DNS/IP denial, DNS rebinding protection through validated-address transport, redirects, content type, body limits, robots fail-closed behavior, lazy extraction, service ingestion, MCP provider/shim exposure, status reporting, import-boundary tests, no live-internet tests, config defaults, and Bandit.
- Type consistency: `DocsSettings.web_source_profile`, `SourcePolicy.evaluate()`, `URLFetcher.fetch()`, `DocsAcquisitionService.ingest_url()`, `ParsedDocument.source_url`, and MCP `docs.ingest_url` argument names are consistent across tasks.
- Dependency boundary: The plan adds no required web dependency and forbids top-level imports of `requests`, `httpx`, `aiohttp`, `playwright`, `trafilatura`, and `bs4`.
- Testing boundary: Every network-sensitive path uses fake resolver and fake transport objects; no test calls live internet.
