# Web Scraping Phase 2 Runtime Policy Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Web_Scraping runtime and policy boundaries, then wire the article lightweight HTTP fetch path through them without changing public behavior.

**Architecture:** Introduce a small `runtime/` package for low-level request, response, fetch, policy protocol, browser, session, timeout, and cancellation contracts. Keep concrete outbound policy adapters outside `runtime/` so runtime stays policy-neutral. Wire only `Article_Extractor_Lib.scrape_article` pre-fetch policy and lightweight `httpx`/`curl` fetch through the new adapters; leave preflight analyzers, Playwright lifecycle, enhanced scraping, WebSearch, crawl, and extraction movement untouched.

**Tech Stack:** Python dataclasses, `typing.Protocol`, existing `tldw_Server_API.app.core.http_client.fetch`, existing `Web_Scraping.outbound_policy`, pytest, monkeypatch, Bandit.

**Backlog:** `TASK-12160`

**Design:** `Docs/superpowers/specs/2026-07-04-web-scraping-phase-2-runtime-policy-boundary-design.md`

---

## Scope

Allowed:

- Create `tldw_Server_API/app/core/Web_Scraping/runtime/`.
- Create `tldw_Server_API/app/core/Web_Scraping/policy/` for concrete scrape-policy adapters.
- Add Phase 2 tests under `tldw_Server_API/tests/Web_Scraping/`.
- Modify only the pre-fetch policy call and lightweight HTTP fetch block inside `Article_Extractor_Lib.scrape_article`.
- Keep public `scrape_article(url, custom_cookies=None)` unchanged.

Not allowed:

- Move `scraper_analyzers/`.
- Change preflight analyzer scoring, recommendations, config keys, or result payload shape.
- Move Playwright browser lifecycle.
- Change enhanced scraper, WebSearch, recursive crawl, sitemap, cookie cloning, or job queue behavior.
- Promote `Article_Extractor_Lib._fetch_with_curl` into a new runtime API.

## File Map

- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`
  - Re-export runtime contracts and default fetch client.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/requests.py`
  - Frozen request/context dataclasses and immutable mapping normalization.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/responses.py`
  - Frozen fetch and policy decision dataclasses, including response normalization from dict-like and object-like values.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/policy.py`
  - Protocol-only outbound policy checker boundary. No concrete policy imports.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py`
  - Fetch client protocol and default fetch adapter over central `http_client.fetch`.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`
  - Contract-only browser launch/context/page protocols and launch options.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/sessions.py`
  - Contract-only runtime cookie/session state dataclasses.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/timeouts.py`
  - Contract-only timeout/budget dataclass.
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/cancellation.py`
  - Cancellation helpers that preserve `asyncio.CancelledError`.
- Create: `tldw_Server_API/app/core/Web_Scraping/policy/__init__.py`
  - Re-export concrete policy adapters.
- Create: `tldw_Server_API/app/core/Web_Scraping/policy/adapters.py`
  - Default adapter delegating to `Web_Scraping.outbound_policy.decide_web_outbound_policy`.
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
  - Use runtime policy/fetch adapters only in `scrape_article`.
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py`
  - Runtime dataclass, contract-only module, and import boundary tests.
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py`
  - Fetch and policy adapter tests.
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`
  - `scrape_article` production seam tests.

## Task 0: Rebase And Recheck Current Touchpoints

**Files:**
- No file edits.

- [ ] **Step 1: Fetch and rebase before code edits**

Run from the worktree root:

```bash
git fetch origin
git rebase origin/dev
```

Expected: branch is based on latest `origin/dev`. If conflicts occur in the Phase 2 design or Backlog task, keep the latest approved Phase 2 design content and rerun the rebase.

- [ ] **Step 2: Confirm the current branch state**

Run:

```bash
git status --short --branch
```

Expected: current branch is clean except known unrelated untracked local files. Do not delete or stage unrelated files under `tldw_Server_API/Config_Files/`.

- [ ] **Step 3: Recheck the current Web_Scraping touchpoints**

Run:

```bash
rg -n "decide_web_outbound_policy|_fetch_with_curl|http_fetch|async def scrape_article|def _resp_get" tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py
rg -n "def fetch\\(|backend=|impersonate|follow_redirects|method" tldw_Server_API/app/core/http_client.py
```

Expected: `scrape_article` still contains the pre-fetch policy call before preflight and the lightweight `curl`/`httpx` block before Playwright fallback. `http_client.fetch` still has separate response-object and simplified Web_Scraping call modes.

## Task 1: Runtime Request And Response Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/requests.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/responses.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py`

- [ ] **Step 1: Write failing tests for runtime request and response contracts**

Create `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py` with this initial content:

```python
from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)


@pytest.mark.unit
def test_runtime_request_context_freezes_metadata() -> None:
    metadata = {"trace": {"id": "abc"}}
    context = RuntimeRequestContext(
        source="article_extract",
        stage="pre_fetch",
        user_id=123,
        request_id="req-1",
        metadata=metadata,
    )

    metadata["trace"]["id"] = "mutated"

    assert isinstance(context.metadata, MappingProxyType)
    assert context.metadata["trace"]["id"] == "abc"
    assert context.source == "article_extract"
    assert context.stage == "pre_fetch"
    assert context.user_id == "123"
    assert context.request_id == "req-1"


@pytest.mark.unit
def test_fetch_request_normalizes_fields_and_proxy_maps() -> None:
    headers = {"User-Agent": "UA"}
    cookies = {"session": "redacted"}
    proxies = {"https": "http://proxy.example:8080"}
    request = FetchRequest(
        url=" https://example.com/article ",
        method="get",
        headers=headers,
        cookies=cookies,
        timeout=15,
        backend="curl",
        allow_redirects=True,
        impersonate="chrome120",
        proxies=proxies,
    )

    headers["User-Agent"] = "mutated"
    cookies["session"] = "mutated"
    proxies["https"] = "mutated"

    assert request.url == "https://example.com/article"
    assert request.method == "GET"
    assert request.headers["User-Agent"] == "UA"
    assert request.cookies["session"] == "redacted"
    assert request.timeout == 15.0
    assert request.backend == "curl"
    assert request.allow_redirects is True
    assert request.impersonate == "chrome120"
    assert request.proxies["https"] == "http://proxy.example:8080"


@pytest.mark.unit
def test_fetch_request_rejects_missing_url() -> None:
    with pytest.raises(ValueError, match="url is required"):
        FetchRequest(url=" ")


@pytest.mark.unit
def test_fetch_response_normalizes_mapping_response() -> None:
    response = FetchResponse.from_raw(
        {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "text": "<html>ok</html>",
            "url": "https://example.com/final",
            "backend": "curl",
        },
        fallback_url="https://example.com/article",
        fallback_backend="httpx",
        elapsed_seconds=0.25,
    )

    assert response.status == 200
    assert response.headers["Content-Type"] == "text/html"
    assert response.text == "<html>ok</html>"
    assert response.url == "https://example.com/final"
    assert response.backend == "curl"
    assert response.elapsed_seconds == 0.25


@pytest.mark.unit
def test_fetch_response_normalizes_object_response_status_code() -> None:
    raw = SimpleNamespace(
        status_code=204,
        headers={"X-Test": "true"},
        text="",
        url="https://example.com/no-content",
    )

    response = FetchResponse.from_raw(
        raw,
        fallback_url="https://example.com/article",
        fallback_backend="httpx",
    )

    assert response.status == 204
    assert response.headers["X-Test"] == "true"
    assert response.text == ""
    assert response.url == "https://example.com/no-content"
    assert response.backend == "httpx"


@pytest.mark.unit
def test_policy_decision_matches_legacy_policy_fields() -> None:
    decision = PolicyDecision(
        allowed=False,
        mode="strict",
        reason="robots_disallowed",
        stage="pre_fetch",
        source="article_extract",
        details={"sanitized": True},
    )

    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_disallowed"
    assert decision.stage == "pre_fetch"
    assert decision.source == "article_extract"
    assert decision.details["sanitized"] is True


@pytest.mark.unit
def test_runtime_package_does_not_import_legacy_wrappers_or_policy_modules() -> None:
    forbidden_roots = {
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
        "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
        "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy",
        "tldw_Server_API.app.core.Security.egress",
    }
    runtime_dir = Path("tldw_Server_API/app/core/Web_Scraping/runtime")

    for path in runtime_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name for alias in node.names}
                assert imports.isdisjoint(forbidden_roots), (path, imports & forbidden_roots)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in forbidden_roots, (path, node.module)
```

- [ ] **Step 2: Run the tests and verify they fail because runtime does not exist**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
```

Expected: FAIL with `ModuleNotFoundError` for `tldw_Server_API.app.core.Web_Scraping.runtime`.

- [ ] **Step 3: Create runtime request and response files**

Create `tldw_Server_API/app/core/Web_Scraping/runtime/requests.py`:

```python
"""Low-level runtime request contracts for Web_Scraping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in dict(value or {}).items()})


def _freeze_proxy_value(value: Mapping[str, str] | str | None) -> Mapping[str, str] | str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): str(item) for key, item in dict(value).items()})
    return str(value)


@dataclass(frozen=True, slots=True)
class RuntimeRequestContext:
    """Context metadata carried into low-level runtime operations."""

    source: str = "web_scraping"
    stage: str = "runtime"
    user_id: str | int | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source or "web_scraping"))
        object.__setattr__(self, "stage", str(self.stage or "runtime"))
        if self.user_id is not None:
            object.__setattr__(self, "user_id", str(self.user_id))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """Low-level HTTP fetch request used by Web_Scraping runtime adapters."""

    url: str
    method: str = "GET"
    headers: Mapping[str, str] = field(default_factory=dict)
    cookies: Mapping[str, str] = field(default_factory=dict)
    timeout: float | None = None
    backend: str = "httpx"
    allow_redirects: bool = True
    impersonate: str | None = None
    proxies: Mapping[str, str] | str | None = None
    context: RuntimeRequestContext = field(default_factory=RuntimeRequestContext)

    def __post_init__(self) -> None:
        normalized_url = str(self.url or "").strip()
        if not normalized_url:
            raise ValueError("url is required")
        object.__setattr__(self, "url", normalized_url)
        object.__setattr__(self, "method", str(self.method or "GET").strip().upper() or "GET")
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
        object.__setattr__(self, "cookies", _freeze_mapping(self.cookies))
        if self.timeout is not None:
            object.__setattr__(self, "timeout", float(self.timeout))
        object.__setattr__(self, "backend", str(self.backend or "httpx").strip().lower() or "httpx")
        object.__setattr__(self, "allow_redirects", bool(self.allow_redirects))
        if self.impersonate is not None:
            object.__setattr__(self, "impersonate", str(self.impersonate))
        object.__setattr__(self, "proxies", _freeze_proxy_value(self.proxies))
```

Create `tldw_Server_API/app/core/Web_Scraping/runtime/responses.py`:

```python
"""Low-level runtime response contracts for Web_Scraping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in dict(value or {}).items()})


def _raw_get(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    try:
        return raw[key]  # type: ignore[index]
    except (AttributeError, KeyError, LookupError, TypeError):
        pass
    value = getattr(raw, key, None)
    if value is not None:
        return value
    data = getattr(raw, "data", None)
    if isinstance(data, Mapping):
        return data.get(key, default)
    return default


@dataclass(frozen=True, slots=True)
class FetchResponse:
    """Normalized response from a runtime fetch adapter."""

    url: str
    status: int = 0
    headers: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    backend: str = "httpx"
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", str(self.url or ""))
        object.__setattr__(self, "status", int(self.status or 0))
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "backend", str(self.backend or "httpx"))
        object.__setattr__(self, "elapsed_seconds", max(0.0, float(self.elapsed_seconds or 0.0)))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        fallback_url: str,
        fallback_backend: str | None = None,
        elapsed_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FetchResponse":
        status = _raw_get(raw, "status")
        if status is None:
            status = _raw_get(raw, "status_code", 0)
        return cls(
            url=str(_raw_get(raw, "url", fallback_url) or fallback_url),
            status=int(status or 0),
            headers=dict(_raw_get(raw, "headers", {}) or {}),
            text=str(_raw_get(raw, "text", "") or ""),
            backend=str(_raw_get(raw, "backend", fallback_backend or "httpx") or fallback_backend or "httpx"),
            elapsed_seconds=float(elapsed_seconds or 0.0),
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy decision shape consumed by runtime-aware scrape code."""

    allowed: bool
    mode: str
    reason: str
    stage: str
    source: str
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed", bool(self.allowed))
        object.__setattr__(self, "mode", str(self.mode or "compat"))
        object.__setattr__(self, "reason", str(self.reason or "allowed"))
        object.__setattr__(self, "stage", str(self.stage or "runtime"))
        object.__setattr__(self, "source", str(self.source or "web_scraping"))
        if self.details is not None:
            object.__setattr__(self, "details", _freeze_mapping(self.details))
```

Create `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`:

```python
"""Runtime contracts and adapters for the staged Web_Scraping refactor."""

from __future__ import annotations

from .requests import FetchRequest, RuntimeRequestContext
from .responses import FetchResponse, PolicyDecision

__all__ = [
    "FetchRequest",
    "FetchResponse",
    "PolicyDecision",
    "RuntimeRequestContext",
]
```

- [ ] **Step 4: Run the runtime contract tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
```

Expected: PASS for the request/response tests. Import boundary may still pass because only request/response files exist.

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py tldw_Server_API/app/core/Web_Scraping/runtime/requests.py tldw_Server_API/app/core/Web_Scraping/runtime/responses.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
git commit -m "feat: add web scraping runtime contracts"
```

## Task 2: Runtime Protocol Contract Modules

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/policy.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/sessions.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/timeouts.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/cancellation.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py`

- [ ] **Step 1: Extend tests for protocol contract-module behavior**

Append these tests to `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py`:

```python
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    BrowserLaunchOptions,
    RuntimeCookie,
    RuntimeSessionState,
    RuntimeTimeouts,
    is_cancellation,
)


@pytest.mark.unit
def test_runtime_session_state_freezes_cookies_and_headers() -> None:
    cookies = [RuntimeCookie(name="session", value="abc", domain="example.com")]
    headers = {"User-Agent": "UA"}
    state = RuntimeSessionState(cookies=cookies, headers=headers)

    headers["User-Agent"] = "mutated"

    assert state.cookies[0].name == "session"
    assert state.cookies[0].value == "abc"
    assert state.cookies[0].domain == "example.com"
    assert state.headers["User-Agent"] == "UA"


@pytest.mark.unit
def test_runtime_timeout_contract_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="fetch_timeout_s must be non-negative"):
        RuntimeTimeouts(fetch_timeout_s=-1)


@pytest.mark.unit
def test_browser_launch_options_normalize_viewport() -> None:
    options = BrowserLaunchOptions(headless=True, viewport_width=1280, viewport_height=720)

    assert options.headless is True
    assert options.viewport == {"width": 1280, "height": 720}


@pytest.mark.unit
def test_cancellation_helper_preserves_asyncio_cancelled_error() -> None:
    import asyncio

    assert is_cancellation(asyncio.CancelledError()) is True
    assert is_cancellation(RuntimeError("not cancelled")) is False
```

- [ ] **Step 2: Run the tests and verify they fail for missing exports**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
```

Expected: FAIL with `ImportError` for `BrowserLaunchOptions`, `RuntimeCookie`, `RuntimeSessionState`, `RuntimeTimeouts`, or `is_cancellation`.

- [ ] **Step 3: Add protocol contract modules**

Create `tldw_Server_API/app/core/Web_Scraping/runtime/policy.py`:

```python
"""Protocol-only policy boundary for Web_Scraping runtime callers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .requests import RuntimeRequestContext
from .responses import PolicyDecision


class OutboundPolicyChecker(Protocol):
    """Async scrape-level outbound policy checker."""

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Return the outbound policy decision for a scrape request."""
```

Create `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`:

```python
"""Contract-only browser runtime boundaries for later Web_Scraping phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class RuntimeBrowserPage(Protocol):
    async def goto(self, url: str, **kwargs: Any) -> Any:
        """Navigate to a URL."""

    async def content(self) -> str:
        """Return current page content."""

    async def close(self) -> None:
        """Close the page."""


class RuntimeBrowserContext(Protocol):
    async def new_page(self) -> RuntimeBrowserPage:
        """Create a page in this context."""

    async def close(self) -> None:
        """Close the context."""


class RuntimeBrowserLauncher(Protocol):
    async def new_context(self, options: "BrowserLaunchOptions") -> RuntimeBrowserContext:
        """Create a browser context with the supplied options."""


@dataclass(frozen=True, slots=True)
class BrowserLaunchOptions:
    """Browser launch/context options captured without launching a browser."""

    headless: bool = True
    user_agent: str | None = None
    viewport_width: int = 1280
    viewport_height: int = 720

    @property
    def viewport(self) -> dict[str, int]:
        return {"width": int(self.viewport_width), "height": int(self.viewport_height)}
```

Create `tldw_Server_API/app/core/Web_Scraping/runtime/sessions.py`:

```python
"""Session and cookie contracts for Web_Scraping runtime adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): item for key, item in dict(value or {}).items()})


@dataclass(frozen=True, slots=True)
class RuntimeCookie:
    """Normalized cookie state for fetch and browser adapters."""

    name: str
    value: str
    domain: str | None = None
    path: str = "/"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value", str(self.value))
        if self.domain is not None:
            object.__setattr__(self, "domain", str(self.domain))
        object.__setattr__(self, "path", str(self.path or "/"))


@dataclass(frozen=True, slots=True)
class RuntimeSessionState:
    """Immutable session state passed into runtime adapters."""

    cookies: tuple[RuntimeCookie, ...] = ()
    headers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cookies", tuple(self.cookies or ()))
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
```

Create `tldw_Server_API/app/core/Web_Scraping/runtime/timeouts.py`:

```python
"""Timeout and budget contracts for Web_Scraping runtime operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeTimeouts:
    """Runtime timeout values in seconds."""

    fetch_timeout_s: float | None = None
    browser_timeout_s: float | None = None
    preflight_timeout_s: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("fetch_timeout_s", "browser_timeout_s", "preflight_timeout_s"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if normalized < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, normalized)
```

Create `tldw_Server_API/app/core/Web_Scraping/runtime/cancellation.py`:

```python
"""Cancellation helpers for Web_Scraping runtime boundaries."""

from __future__ import annotations

import asyncio


def is_cancellation(exc: BaseException) -> bool:
    """Return True when an exception represents task cancellation."""

    return isinstance(exc, asyncio.CancelledError)
```

Update `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`:

```python
"""Runtime contracts and adapters for the staged Web_Scraping refactor."""

from __future__ import annotations

from .browser import BrowserLaunchOptions, RuntimeBrowserContext, RuntimeBrowserLauncher, RuntimeBrowserPage
from .cancellation import is_cancellation
from .policy import OutboundPolicyChecker
from .requests import FetchRequest, RuntimeRequestContext
from .responses import FetchResponse, PolicyDecision
from .sessions import RuntimeCookie, RuntimeSessionState
from .timeouts import RuntimeTimeouts

__all__ = [
    "BrowserLaunchOptions",
    "FetchRequest",
    "FetchResponse",
    "OutboundPolicyChecker",
    "PolicyDecision",
    "RuntimeBrowserContext",
    "RuntimeBrowserLauncher",
    "RuntimeBrowserPage",
    "RuntimeCookie",
    "RuntimeRequestContext",
    "RuntimeSessionState",
    "RuntimeTimeouts",
    "is_cancellation",
]
```

- [ ] **Step 4: Run runtime contract tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Web_Scraping/runtime tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
git commit -m "feat: add web scraping runtime protocol contracts"
```

## Task 3: Default Fetch Adapter

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py`

- [ ] **Step 1: Write failing fetch adapter tests**

Create `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py` with this initial content:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Web_Scraping.runtime.fetch as runtime_fetch
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchRequest
from tldw_Server_API.app.core.Web_Scraping.runtime.fetch import DefaultFetchClient


@pytest.mark.unit
def test_default_fetch_client_uses_simplified_get_path_without_method(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fetch(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "text": "<html>ok</html>",
            "url": url,
            "backend": "curl",
        }

    monkeypatch.setattr(runtime_fetch, "http_fetch", fake_fetch)

    response = DefaultFetchClient().fetch(
        FetchRequest(
            url="https://example.com/article",
            headers={"User-Agent": "UA"},
            cookies={"session": "abc"},
            timeout=15.0,
            backend="curl",
            allow_redirects=True,
            impersonate="chrome120",
            proxies={"https": "http://proxy.example:8080"},
        )
    )

    assert calls["url"] == "https://example.com/article"
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert "method" not in kwargs
    assert kwargs["backend"] == "curl"
    assert kwargs["follow_redirects"] is True
    assert kwargs["impersonate"] == "chrome120"
    assert kwargs["headers"] == {"User-Agent": "UA"}
    assert kwargs["cookies"] == {"session": "abc"}
    assert kwargs["proxies"] == {"https": "http://proxy.example:8080"}
    assert response.status == 200
    assert response.backend == "curl"


@pytest.mark.unit
def test_default_fetch_client_normalizes_object_like_response(monkeypatch) -> None:
    def fake_fetch(url, **kwargs):
        return SimpleNamespace(
            status_code=201,
            headers={"X-Test": "true"},
            text="<html>created</html>",
            url="https://example.com/final",
        )

    monkeypatch.setattr(runtime_fetch, "http_fetch", fake_fetch)

    response = DefaultFetchClient().fetch(
        FetchRequest(url="https://example.com/article", backend="httpx", timeout=15.0)
    )

    assert response.status == 201
    assert response.headers["X-Test"] == "true"
    assert response.text == "<html>created</html>"
    assert response.url == "https://example.com/final"
    assert response.backend == "httpx"


@pytest.mark.unit
def test_default_fetch_client_rejects_non_get_method() -> None:
    with pytest.raises(ValueError, match="only supports GET"):
        DefaultFetchClient().fetch(FetchRequest(url="https://example.com/article", method="POST"))
```

- [ ] **Step 2: Run adapter tests and verify missing fetch adapter failure**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
```

Expected: FAIL because `runtime.fetch` does not exist.

- [ ] **Step 3: Implement default fetch adapter**

Create `tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py`:

```python
"""Fetch runtime adapter for Web_Scraping."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

from tldw_Server_API.app.core.http_client import fetch as http_fetch

from .requests import FetchRequest
from .responses import FetchResponse


class FetchClient(Protocol):
    """Synchronous fetch client used by runtime-aware scrape code."""

    def fetch(self, request: FetchRequest) -> FetchResponse:
        """Fetch a URL and return a normalized response."""


def _mutable_mapping_or_none(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not value:
        return None
    return {str(key): item for key, item in dict(value).items()}


def _mutable_proxies(value: Mapping[str, str] | str | None) -> dict[str, str] | str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in dict(value).items()}
    return str(value)


class DefaultFetchClient:
    """Default Web_Scraping fetch adapter over the central HTTP helper."""

    def fetch(self, request: FetchRequest) -> FetchResponse:
        if request.method != "GET":
            raise ValueError("DefaultFetchClient only supports GET requests in Phase 2")

        started = time.time()
        raw = http_fetch(
            request.url,
            headers=_mutable_mapping_or_none(request.headers),
            cookies=_mutable_mapping_or_none(request.cookies),
            timeout=request.timeout,
            backend=request.backend,
            follow_redirects=request.allow_redirects,
            impersonate=request.impersonate,
            proxies=_mutable_proxies(request.proxies),
        )
        elapsed = max(0.0, time.time() - started)
        return FetchResponse.from_raw(
            raw,
            fallback_url=request.url,
            fallback_backend=request.backend,
            elapsed_seconds=elapsed,
        )
```

Update `tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py` by adding exports:

```python
from .fetch import DefaultFetchClient, FetchClient
```

Add both names to `__all__`:

```python
"DefaultFetchClient",
"FetchClient",
```

- [ ] **Step 4: Run fetch adapter tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
```

Expected: PASS.

- [ ] **Step 5: Run runtime import-boundary tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py::test_runtime_package_does_not_import_legacy_wrappers_or_policy_modules
```

Expected: PASS. `runtime.fetch` may import `tldw_Server_API.app.core.http_client`, but it must not import Web_Scraping legacy wrappers or policy modules.

- [ ] **Step 6: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Web_Scraping/runtime/__init__.py tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
git commit -m "feat: add web scraping runtime fetch adapter"
```

## Task 4: Concrete Outbound Policy Adapter

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/policy/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/policy/adapters.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py`

- [ ] **Step 1: Add failing policy adapter tests**

Append these tests to `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py`:

```python
import tldw_Server_API.app.core.Web_Scraping.policy.adapters as policy_adapters
from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext


@pytest.mark.asyncio
async def test_default_policy_checker_delegates_to_existing_outbound_policy(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def fake_decide_web_outbound_policy(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            allowed=False,
            mode="strict",
            reason="robots_disallowed",
            stage="pre_fetch",
            source="article_extract",
            details={"policy": "test"},
        )

    monkeypatch.setattr(
        policy_adapters,
        "decide_web_outbound_policy",
        fake_decide_web_outbound_policy,
    )

    decision = await DefaultWebOutboundPolicyChecker().decide(
        "https://example.com/article",
        respect_robots=True,
        user_agent="UA",
        context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        config={"web_scraper": {"web_outbound_policy_mode": "strict"}},
    )

    assert calls["url"] == "https://example.com/article"
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["respect_robots"] is True
    assert kwargs["user_agent"] == "UA"
    assert kwargs["source"] == "article_extract"
    assert kwargs["stage"] == "pre_fetch"
    assert kwargs["config"] == {"web_scraper": {"web_outbound_policy_mode": "strict"}}
    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_disallowed"
    assert decision.details["policy"] == "test"


@pytest.mark.asyncio
async def test_default_policy_checker_defaults_context_source_and_stage(monkeypatch) -> None:
    async def fake_decide_web_outbound_policy(url, **kwargs):
        return SimpleNamespace(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage=kwargs["stage"],
            source=kwargs["source"],
            details=None,
        )

    monkeypatch.setattr(
        policy_adapters,
        "decide_web_outbound_policy",
        fake_decide_web_outbound_policy,
    )

    decision = await DefaultWebOutboundPolicyChecker().decide(
        "https://example.com/article",
        respect_robots=False,
        user_agent=None,
        context=RuntimeRequestContext(),
        config=None,
    )

    assert decision.allowed is True
    assert decision.stage == "runtime"
    assert decision.source == "web_scraping"
```

- [ ] **Step 2: Run adapter tests and verify missing policy package failure**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
```

Expected: FAIL with `ModuleNotFoundError` for `Web_Scraping.policy`.

- [ ] **Step 3: Implement concrete policy adapter outside runtime**

Create `tldw_Server_API/app/core/Web_Scraping/policy/adapters.py`:

```python
"""Concrete outbound policy adapters for Web_Scraping callers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy
from tldw_Server_API.app.core.Web_Scraping.runtime import PolicyDecision, RuntimeRequestContext


class DefaultWebOutboundPolicyChecker:
    """Default adapter for the existing Web_Scraping outbound policy helper."""

    async def decide(
        self,
        url: str,
        *,
        respect_robots: bool,
        user_agent: str | None,
        context: RuntimeRequestContext,
        config: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        raw = await decide_web_outbound_policy(
            url,
            respect_robots=respect_robots,
            user_agent=user_agent,
            source=context.source,
            stage=context.stage,
            config=dict(config or {}),
        )
        return PolicyDecision(
            allowed=bool(raw.allowed),
            mode=str(raw.mode),
            reason=str(raw.reason),
            stage=str(raw.stage),
            source=str(raw.source),
            details=getattr(raw, "details", None),
        )
```

Create `tldw_Server_API/app/core/Web_Scraping/policy/__init__.py`:

```python
"""Concrete policy adapters for Web_Scraping."""

from __future__ import annotations

from .adapters import DefaultWebOutboundPolicyChecker

__all__ = ["DefaultWebOutboundPolicyChecker"]
```

- [ ] **Step 4: Run adapter and runtime boundary tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py
```

Expected: PASS. Runtime import-boundary test must still pass because concrete policy imports live under `Web_Scraping/policy/`, not `Web_Scraping/runtime/`.

- [ ] **Step 5: Commit Task 4**

```bash
git add tldw_Server_API/app/core/Web_Scraping/policy tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
git commit -m "feat: add web scraping policy adapter"
```

## Task 5: Wire Article Lightweight Policy And Fetch Path

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`

- [ ] **Step 1: Write failing article runtime-boundary tests**

Create `tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, PolicyDecision


class FakePolicyChecker:
    def __init__(self, decision: PolicyDecision):
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    async def decide(self, url, *, respect_robots, user_agent, context, config):
        self.calls.append(
            {
                "url": url,
                "respect_robots": respect_robots,
                "user_agent": user_agent,
                "context": context,
                "config": config,
            }
        )
        return self.decision


class FakeFetchClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _install_article_defaults(monkeypatch, *, backend="httpx", web_scraper_config=None):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as ael

    config = {"web_scraper": web_scraper_config or {}}
    monkeypatch.setattr(ael, "load_and_log_configs", lambda: config)
    monkeypatch.setattr(ael, "_js_required", lambda *args, **kwargs: False)

    rules = {
        "domains": {
            "example.com": {
                "backend": backend,
                "handler": "tldw_Server_API.app.core.Web_Scraping.handlers:handle_generic_html",
            }
        }
    }
    monkeypatch.setattr(ael.ScraperRouter, "load_rules_from_yaml", lambda path: rules)

    def fake_handler(html, url):
        return {
            "url": url,
            "title": "handled",
            "author": "n/a",
            "date": "n/a",
            "content": "handled-content",
            "extraction_successful": True,
        }

    monkeypatch.setattr(ael, "resolve_handler", lambda _: fake_handler)
    monkeypatch.setattr(ael, "observe_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(ael, "increment_counter", lambda *args, **kwargs: None)
    return ael


@pytest.mark.asyncio
async def test_scrape_article_uses_runtime_policy_before_preflight(monkeypatch) -> None:
    ael = _install_article_defaults(
        monkeypatch,
        backend="httpx",
        web_scraper_config={"web_scraper_preflight_analyzers": True},
    )
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=False,
            mode="strict",
            reason="robots_disallowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient([])
    run_analysis = AsyncMock()

    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.run_analysis",
        run_analysis,
        raising=False,
    )

    result = await ael.scrape_article("https://example.com/path")

    assert result["extraction_successful"] is False
    assert result["policy_reason"] == "robots_disallowed"
    assert policy_checker.calls[0]["url"] == "https://example.com/path"
    assert policy_checker.calls[0]["context"].source == "article_extract"
    assert policy_checker.calls[0]["context"].stage == "pre_fetch"
    assert fetch_client.requests == []
    run_analysis.assert_not_called()


@pytest.mark.asyncio
async def test_scrape_article_uses_runtime_fetch_client_for_httpx_success(monkeypatch) -> None:
    ael = _install_article_defaults(monkeypatch, backend="httpx")
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient(
        [
            FetchResponse(
                url="https://example.com/path",
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>ok</body></html>",
                backend="httpx",
            )
        ]
    )
    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert len(fetch_client.requests) == 1
    request = fetch_client.requests[0]
    assert request.url == "https://example.com/path"
    assert request.method == "GET"
    assert request.backend == "httpx"
    assert request.allow_redirects is True


@pytest.mark.asyncio
async def test_scrape_article_preserves_curl_to_httpx_fallback(monkeypatch) -> None:
    ael = _install_article_defaults(monkeypatch, backend="curl")
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient(
        [
            RuntimeError("curl unavailable"),
            FetchResponse(
                url="https://example.com/path",
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>ok</body></html>",
                backend="httpx",
            ),
        ]
    )
    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert [request.backend for request in fetch_client.requests] == ["curl", "httpx"]


@pytest.mark.asyncio
async def test_scrape_article_preflight_tls_advice_still_selects_curl(monkeypatch) -> None:
    ael = _install_article_defaults(
        monkeypatch,
        backend="auto",
        web_scraper_config={
            "web_scraper_preflight_analyzers": True,
            "web_scraper_preflight_include_results": True,
        },
    )
    policy_checker = FakePolicyChecker(
        PolicyDecision(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage="pre_fetch",
            source="article_extract",
        )
    )
    fetch_client = FakeFetchClient(
        [
            FetchResponse(
                url="https://example.com/path",
                status=200,
                headers={"Content-Type": "text/html"},
                text="<html><body>ok</body></html>",
                backend="curl",
            )
        ]
    )

    def fake_run_analysis(*args, **kwargs):
        return {"results": {"tls": {"status": "active"}, "js": {"status": "success"}}}

    monkeypatch.setattr(ael, "_ARTICLE_POLICY_CHECKER", policy_checker)
    monkeypatch.setattr(ael, "_ARTICLE_FETCH_CLIENT", fetch_client)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.scraper_analyzers.run_analysis",
        fake_run_analysis,
        raising=False,
    )

    result = await ael.scrape_article("https://example.com/path")

    assert result["content"] == "handled-content"
    assert fetch_client.requests[0].backend == "curl"
    assert result["preflight_analysis"]["advice"]["backend"] == "curl"
    assert "tls_active" in result["preflight_analysis"]["advice"]["notes"]
```

- [ ] **Step 2: Run article runtime-boundary tests and verify missing seam failure**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
```

Expected: FAIL because `Article_Extractor_Lib` has no `_ARTICLE_POLICY_CHECKER` or `_ARTICLE_FETCH_CLIENT`, and still calls legacy policy/fetch directly.

- [ ] **Step 3: Update Article_Extractor_Lib imports**

In `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`, replace the outbound policy import block:

```python
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    WebOutboundPolicyDecision,
    decide_web_outbound_policy,
    decide_web_outbound_policy_sync,
)
```

with:

```python
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy_sync
from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    DefaultFetchClient,
    FetchClient,
    FetchRequest,
    FetchResponse,
    OutboundPolicyChecker,
    PolicyDecision,
    RuntimeRequestContext,
)
```

Expected: synchronous policy call sites still compile because `decide_web_outbound_policy_sync` remains imported.

- [ ] **Step 4: Add article default adapters and private helpers**

Add this block near `_fetch_with_curl` and `_blocked_article_result`, before `scrape_article`:

```python
_ARTICLE_POLICY_CHECKER: OutboundPolicyChecker = DefaultWebOutboundPolicyChecker()
_ARTICLE_FETCH_CLIENT: FetchClient = DefaultFetchClient()
```

Change `_blocked_article_result` signature from the old outbound policy type to:

```python
def _blocked_article_result(
    url: str,
    decision: PolicyDecision,
) -> dict[str, Any]:
```

Add these helpers before `scrape_article`:

```python
async def _decide_article_pre_fetch_policy(
    url: str,
    *,
    respect_robots: bool,
    user_agent: str,
    config: dict[str, Any],
    policy_checker: OutboundPolicyChecker | None = None,
) -> PolicyDecision:
    checker = policy_checker or _ARTICLE_POLICY_CHECKER
    return await checker.decide(
        url,
        respect_robots=respect_robots,
        user_agent=user_agent,
        context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        config={"web_scraper": config},
    )


def _fetch_article_lightweight(
    url: str,
    *,
    backend_choice: str,
    headers: dict[str, str],
    cookies: dict[str, str] | None,
    timeout: float,
    impersonate: str | None,
    proxies: dict[str, str] | None,
    fetch_client: FetchClient | None = None,
) -> tuple[FetchResponse, str]:
    client = fetch_client or _ARTICLE_FETCH_CLIENT

    def _fetch_with_backend(backend: str) -> FetchResponse:
        return client.fetch(
            FetchRequest(
                url=url,
                method="GET",
                headers=headers,
                cookies=cookies or {},
                timeout=timeout,
                backend=backend,
                allow_redirects=True,
                impersonate=impersonate,
                proxies=proxies,
                context=RuntimeRequestContext(source="article_extract", stage="fetch"),
            )
        )

    if backend_choice == "curl":
        try:
            response = _fetch_with_backend("curl")
            return response, response.backend or "curl"
        except _ARTICLE_EXTRACTOR_NONCRITICAL_EXCEPTIONS as exc:
            logging.debug(f"curl backend failed; falling back to httpx: {exc}")

    response = _fetch_with_backend("httpx")
    return response, response.backend or "httpx"
```

- [ ] **Step 5: Replace the scrape_article policy call**

Inside `scrape_article`, replace:

```python
        decision = await decide_web_outbound_policy(
            url,
            respect_robots=bool(getattr(plan, "respect_robots", True)),
            user_agent=effective_ua,
            source="article_extract",
            stage="pre_fetch",
            config={"web_scraper": ws_cfg},
        )
```

with:

```python
        decision = await _decide_article_pre_fetch_policy(
            url,
            respect_robots=bool(getattr(plan, "respect_robots", True)),
            user_agent=effective_ua,
            config=ws_cfg,
        )
```

Expected: policy still runs before the preflight analyzer block.

- [ ] **Step 6: Replace only the lightweight fetch block**

In `scrape_article`, keep cookie merge and `t0 = time.time()` as today. Replace the `if backend_choice == "curl": ... else: ...` fetch section with:

```python
            resp, backend_used = await asyncio.to_thread(
                _fetch_article_lightweight,
                url,
                backend_choice=backend_choice,
                headers=ua_headers,
                cookies=cookies_map or None,
                timeout=15.0,
                impersonate=getattr(plan, "impersonate", None),
                proxies=getattr(plan, "proxies", None) or None,
            )
```

Keep all code after `elapsed = max(0.0, time.time() - t0)` unchanged, including `_resp_get`, JS-required fallback, extraction pipeline, markdown conversion, metrics, and fallback counters.

- [ ] **Step 7: Run article runtime-boundary tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
```

Expected: PASS.

- [ ] **Step 8: Run existing router backend tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py
```

Expected: PASS. If `test_scrape_article_backend_curl_uses_curl` still monkeypatches `_fetch_with_curl`, update that test in the same commit to monkeypatch `_ARTICLE_FETCH_CLIENT` instead, preserving the assertion that the first runtime fetch request uses backend `curl`.

- [ ] **Step 9: Commit Task 5**

```bash
git add tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py
git commit -m "feat: route article fetch through web scraping runtime"
```

## Task 6: Verification And Finalization

**Files:**
- Modify: `backlog/tasks/task-12160 - Plan-Web-Scraping-refactor-Phase-2-runtime-and-policy-boundary-implementation.md` only if Backlog finalization is part of the implementation branch.

- [ ] **Step 1: Run focused Phase 2 tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
```

Expected: PASS.

- [ ] **Step 2: Run compatibility and hardening tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py \
  tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py \
  tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py
```

Expected: PASS.

- [ ] **Step 3: Run formatting whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Run Bandit on touched Python scope**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping/runtime \
  tldw_Server_API/app/core/Web_Scraping/policy \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  -f json -o /tmp/bandit_web_scraping_phase2_runtime_policy.json
```

Expected: no new findings in touched code. If Bandit reports existing unrelated findings in `Article_Extractor_Lib.py`, record them in Backlog notes and fix any finding introduced by this phase.

- [ ] **Step 5: Record final Backlog notes**

Use the Backlog CLI:

```bash
backlog task edit TASK-12160 --notes "Implementation plan executed. Runtime contracts, policy adapter, fetch adapter, article lightweight fetch wiring, compatibility tests, and verification completed." --status Done
```

Expected: `TASK-12160` records verification commands and final status.

- [ ] **Step 6: Commit verification notes if the task file changed**

```bash
git add "backlog/tasks/task-12160 - Plan-Web-Scraping-refactor-Phase-2-runtime-and-policy-boundary-implementation.md"
git commit -m "docs: finalize web scraping phase 2 runtime task"
```

Expected: commit is created only if the Backlog task file changed after verification.

## Plan Self-Review

Spec coverage:

- Runtime contracts are covered by Tasks 1 and 2.
- Concrete policy adapter outside `runtime/` is covered by Task 4.
- Fetch adapter, simplified GET call mode, response normalization, and curl support are covered by Task 3.
- `scrape_article` production wiring, public signature preservation, policy-before-preflight ordering, curl-to-httpx fallback, and preflight payload behavior are covered by Task 5.
- Browser, session, timeout, and cancellation contract-only modules are covered by Task 2.
- Verification, Bandit, and Backlog finalization are covered by Task 6.

Placeholder scan:

- The plan avoids open-ended implementation gaps. Each code-changing task includes concrete file paths, code snippets, commands, and expected results.

Type consistency:

- `FetchRequest`, `FetchResponse`, `PolicyDecision`, `RuntimeRequestContext`, `FetchClient`, and `OutboundPolicyChecker` are introduced before they are used by later tasks.
- `DefaultFetchClient` and `DefaultWebOutboundPolicyChecker` are implemented before `Article_Extractor_Lib.py` imports them.
- Article tests use the same `_ARTICLE_POLICY_CHECKER` and `_ARTICLE_FETCH_CLIENT` names introduced in Task 5.
