# Standalone MCP Docs Stage 4B Source Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded `docs.discover_source` support plus `url_sitemap` refresh for the standalone MCP docs corpus.

**Architecture:** Add a runtime-neutral `DocsSourceDiscoveryService` that reuses Stage 2 URL policy/fetch seams and Stage 4A source registry/sync helpers. Discovery is explicit, synchronous, dry-run first, bounded by config, and independent from `tldw_Server_API`; apply mode either registers sitemap sources or ingests accepted pages through `DocsAcquisitionService.ingest_url()`.

**Tech Stack:** Python 3.10+, dataclasses, stdlib `html.parser`, stdlib XML parsing, optional lazy `beautifulsoup4`/`trafilatura`, stdlib `sqlite3`, pytest fake resolver/transport tests, Bandit on touched Python paths.

---

## Source References

- Design spec: `Docs/superpowers/specs/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-design.md`
- Stage 2 URL acquisition spec: `Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md`
- Stage 4A source sync spec: `Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md`
- Stage 4A implementation plan: `Docs/superpowers/plans/2026-07-02-standalone-mcp-docs-stage4a-source-sync-implementation-plan.md`
- Backlog planning task: `TASK-12122`
- Backlog implementation task: `TASK-12124`

## Scope

Included in Stage 4B.1:

- `DocsSettings` discovery flags and caps.
- `DiscoverSourceRequest` and discovery result shape helpers.
- `docs.status` source discovery status.
- `docs.discover_source` advertisement, validation, and execution.
- Sitemap `urlset` parsing from explicit sitemap URLs.
- One-hop HTML page-link discovery from explicit seed pages.
- Optional lazy BeautifulSoup link extraction with stdlib fallback.
- Dry-run discovery with no docs-store mutation.
- Apply register mode for `url_sitemap` sources.
- Apply ingest and register-and-ingest modes through `DocsAcquisitionService.ingest_url()`.
- `docs.sync_source` support for registered `url_sitemap` sources when `sitemap_sync_enabled=true`.
- Query redaction, same-origin/prefix policy, fake transport/resolver tests, and import-boundary coverage.

Excluded from Stage 4B.1:

- Broad recursive crawling.
- Sitemap index support unless implementation stays small after `urlset` support.
- Browser automation, JavaScript rendering, cookies, login/session scraping, Playwright.
- New required dependencies.
- Jobs/Scheduler wrappers, background recurrence, Media DB, ChromaDB, host RAG bridge.
- New crawler graph tables.

## File Structure

Create:

- `apps/mcp-unified/src/mcp_unified/docs/discovery.py` - candidate dataclasses, sitemap parser, page-link extractor, `DocsSourceDiscoveryService`, public response shaping.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py` - unit and integration coverage for discovery helpers, service dry-run/apply, redaction, and metadata propagation.

Modify:

- `apps/mcp-unified/src/mcp_unified/docs/settings.py` - discovery settings and coercion.
- `apps/mcp-unified/src/mcp_unified/docs/models.py` - discovery literals and request dataclass.
- `apps/mcp-unified/src/mcp_unified/docs/mcp_module.py` - discovery status, tool definition, argument parsing, execution.
- `apps/mcp-unified/src/mcp_unified/docs/sync.py` - `url_sitemap` sync branch using shared discovery helpers.
- `apps/mcp-unified/src/mcp_unified/docs/__init__.py` - export `DiscoverSourceRequest` if needed by tests.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py` - settings defaults/parsing.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py` - provider advertisement/status/argument validation.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py` - `url_sitemap` sync coverage.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py` - host shim advertisement and config default checks.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py` - include discovery module in optional dependency/import-boundary assertions.
- `backlog/tasks/task-12122 - Plan-standalone-MCP-docs-Stage-4B-source-discovery-implementation.md` - task tracking and verification.

Do not modify:

- `pyproject.toml` for new required dependencies.
- `tldw_Server_API` scraping/media/RAG services.
- Jobs/Scheduler code.
- Browser or Playwright tooling.

## Shared Contracts

Use these names consistently.

```python
DiscoveryKind = Literal["auto", "sitemap", "page_links"]
DiscoveryMode = Literal["dry_run", "apply"]
DiscoveryApplyAction = Literal["register", "ingest", "register_and_ingest"]
DiscoveryCandidateStatus = Literal["accepted", "duplicate", "denied", "skipped", "ingested", "failed"]


@dataclass(frozen=True)
class DiscoverSourceRequest:
    url: str
    kind: DiscoveryKind = "auto"
    mode: DiscoveryMode = "dry_run"
    apply_action: DiscoveryApplyAction | None = None
    max_pages: int | None = None
    max_depth: int | None = None
    collections: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    title: str | None = None
    include_seed: bool = False
```

Stable reason codes to add where missing:

```python
SOURCE_DISCOVERY_DISABLED = "source_discovery_disabled"
SOURCE_DISCOVERY_REQUEST_INVALID = "source_discovery_request_invalid"
SOURCE_DISCOVERY_KIND_UNSUPPORTED = "source_discovery_kind_unsupported"
SOURCE_DISCOVERY_LIMIT_EXCEEDED = "source_discovery_limit_exceeded"
SOURCE_DISCOVERY_NO_CANDIDATES = "source_discovery_no_candidates"
SITEMAP_CONTENT_TYPE_DENIED = "sitemap_content_type_denied"
SITEMAP_FETCH_FAILED = "sitemap_fetch_failed"
SITEMAP_PARSE_FAILED = "sitemap_parse_failed"
SITEMAP_INDEX_UNSUPPORTED = "sitemap_index_unsupported"
SITEMAP_XML_FORBIDDEN_DOCTYPE = "sitemap_xml_forbidden_doctype"
SITEMAP_XML_FORBIDDEN_ENTITY = "sitemap_xml_forbidden_entity"
CANDIDATE_OUT_OF_SCOPE = "candidate_out_of_scope"
CANDIDATE_QUERY_NOT_PERSISTED = "candidate_query_not_persisted"
CANDIDATE_DUPLICATE = "candidate_duplicate"
PAGE_LINK_CONTENT_TYPE_DENIED = "page_link_content_type_denied"
PAGE_LINK_REGISTRATION_UNSUPPORTED = "page_link_registration_unsupported"
ROBOTS_UNAVAILABLE = "robots_unavailable"
```

## Test Command Conventions

Run commands from the worktree root:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-docs-source-discovery-design
```

Use the project virtual environment:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q
```

If the virtual environment cannot import the project, stop and report the environment problem. Do not switch to global Python.

---

### Task 1: Settings, Models, And Status Surface

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/settings.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/models.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/mcp_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [ ] **Step 1: Write failing settings tests**

Add to `test_docs_settings.py`:

```python
def test_from_mapping_uses_safe_source_discovery_defaults() -> None:
    settings = DocsSettings.from_mapping({})

    assert settings.enable_source_discovery is False  # nosec B101
    assert settings.max_discovery_pages == 25  # nosec B101
    assert settings.max_discovery_depth == 1  # nosec B101
    assert settings.max_discovery_sitemaps == 3  # nosec B101
    assert settings.discovery_apply_default == "register"  # nosec B101
    assert settings.discovery_same_origin_only is True  # nosec B101


def test_from_mapping_parses_source_discovery_values() -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_source_discovery": "true",
            "max_discovery_pages": "7",
            "max_discovery_depth": "1",
            "max_discovery_sitemaps": "2",
            "discovery_apply_default": "register_and_ingest",
            "discovery_same_origin_only": "false",
        }
    )

    assert settings.enable_source_discovery is True  # nosec B101
    assert settings.max_discovery_pages == 7  # nosec B101
    assert settings.max_discovery_depth == 1  # nosec B101
    assert settings.max_discovery_sitemaps == 2  # nosec B101
    assert settings.discovery_apply_default == "register_and_ingest"  # nosec B101
    assert settings.discovery_same_origin_only is False  # nosec B101


@pytest.mark.parametrize("value", ["", "crawl", "register+ingest"])
def test_from_mapping_rejects_unknown_discovery_apply_default(value: str) -> None:
    with pytest.raises(ValueError, match="discovery_apply_default"):
        DocsSettings.from_mapping({"discovery_apply_default": value})
```

- [ ] **Step 2: Write failing status tests**

Add to the existing provider status test in `test_docs_mcp_provider.py`:

```python
assert status["source_discovery"]["enabled"] is False  # nosec B101
assert status["source_discovery"]["available"] is False  # nosec B101
assert status["source_discovery"]["disabled_reason"] == "source_discovery_disabled"  # nosec B101
assert status["source_discovery"]["supported_kinds"] == ["sitemap", "page_links"]  # nosec B101
```

Add a new enabled case:

```python
def test_provider_status_reports_source_discovery_when_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "enable_source_discovery": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/docs/",),
        }
    )
    provider = DocsMCPToolProvider(settings=settings)

    status = provider.execute("docs.status", {}, scope=AccessScope())

    assert status["source_discovery"]["enabled"] is True  # nosec B101
    assert status["source_discovery"]["available"] is True  # nosec B101
    assert status["source_discovery"]["max_discovery_pages"] == 25  # nosec B101
```

- [ ] **Step 3: Run red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_status_reports_web_acquisition_disabled \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_status_reports_source_discovery_when_enabled \
  -q
```

Expected: FAIL with missing discovery settings/status fields.

- [ ] **Step 4: Implement settings, models, and status**

In `settings.py`, add:

```python
DiscoveryApplyDefault = Literal["register", "ingest", "register_and_ingest"]


def _coerce_discovery_apply_default(value: object, field_name: str) -> DiscoveryApplyDefault:
    text = "register" if value is None else str(value).strip().lower()
    if text not in {"register", "ingest", "register_and_ingest"}:
        raise ValueError(f"{field_name} must be register, ingest, or register_and_ingest")
    return cast(DiscoveryApplyDefault, text)
```

Extend `DocsSettings` and `from_mapping()`:

```python
enable_source_discovery: bool = False
max_discovery_pages: int = 25
max_discovery_depth: int = 1
max_discovery_sitemaps: int = 3
discovery_apply_default: DiscoveryApplyDefault = "register"
discovery_same_origin_only: bool = True
```

In `models.py`, add the shared `DiscoverSourceRequest` dataclass and literals from the Shared Contracts section.

In `mcp_module.py`, add:

```python
def _source_discovery_status(settings: DocsSettings) -> dict[str, Any]:
    enabled = settings.enable_source_discovery
    available = enabled and settings.enable_web_acquisition
    disabled_reason = None if available else (
        "web_acquisition_disabled" if enabled else "source_discovery_disabled"
    )
    return {
        "enabled": enabled,
        "available": available,
        "disabled_reason": disabled_reason,
        "supported_kinds": ["sitemap", "page_links"],
        "max_discovery_pages": settings.max_discovery_pages,
        "max_discovery_depth": settings.max_discovery_depth,
        "max_discovery_sitemaps": settings.max_discovery_sitemaps,
        "discovery_apply_default": settings.discovery_apply_default,
        "discovery_same_origin_only": settings.discovery_same_origin_only,
    }
```

Set `status["source_discovery"] = _source_discovery_status(self.settings)`.

- [ ] **Step 5: Run green tests**

Run the same command from Step 3.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/settings.py \
  apps/mcp-unified/src/mcp_unified/docs/models.py \
  apps/mcp-unified/src/mcp_unified/docs/mcp_module.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
git commit -m "feat: add docs source discovery settings"
```

### Task 2: Discovery Parser And Candidate Helpers

**Files:**

- Create: `apps/mcp-unified/src/mcp_unified/docs/discovery.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py`
- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`

- [ ] **Step 1: Write failing pure-helper tests**

Create `test_docs_source_discovery.py` with these first tests:

```python
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from mcp_unified.docs.discovery import (
    DiscoveredURLCandidate,
    extract_page_links,
    parse_sitemap_urlset,
    public_candidate,
)

pytestmark = pytest.mark.unit


def test_parse_sitemap_urlset_rejects_doctype() -> None:
    body = b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><urlset />'

    result = parse_sitemap_urlset(body, max_pages=10)

    assert result.reason_code == "sitemap_xml_forbidden_doctype"  # nosec B101
    assert result.candidates == []  # nosec B101


def test_parse_sitemap_urlset_enforces_page_limit_before_candidates() -> None:
    body = b"""
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/a</loc></url>
      <url><loc>https://example.com/b</loc></url>
      <url><loc>https://example.com/c</loc></url>
    </urlset>
    """

    result = parse_sitemap_urlset(body, max_pages=2)

    assert result.reason_code == "ok"  # nosec B101
    assert [item.url for item in result.candidates] == [  # nosec B101
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert result.skipped == 1  # nosec B101


def test_extract_page_links_prefers_beautifulsoup_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    original_import = importlib.import_module

    def fake_import(name: str):
        if name == "bs4":
            seen.append(name)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    html = b'<a href="/docs/a">A</a><a rel="nofollow" href="/docs/b">B</a>'

    links = extract_page_links("https://example.com/docs/index.html", html)

    assert "https://example.com/docs/a" in links  # nosec B101
    assert "https://example.com/docs/b" not in links  # nosec B101
    assert seen == ["bs4"] or seen == []  # nosec B101


def test_public_candidate_redacts_query_bearing_url() -> None:
    candidate = DiscoveredURLCandidate(
        url="https://example.com/page?token=secret",
        display_url="https://example.com/page",
        status="accepted",
        reason_code="ok",
        source_kind="sitemap",
        parent_url="https://example.com/sitemap.xml?token=secret",
        parent_display_url="https://example.com/sitemap.xml",
        safe_argument_hash="hash",
    )

    serialized = json.dumps(public_candidate(candidate), sort_keys=True)

    assert "token=secret" not in serialized  # nosec B101
    assert "https://example.com/page" in serialized  # nosec B101
    assert "hash" in serialized  # nosec B101
```

- [ ] **Step 2: Run red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  -q
```

Expected: FAIL because `mcp_unified.docs.discovery` does not exist.

- [ ] **Step 3: Implement pure helpers**

Create `discovery.py` with:

```python
from __future__ import annotations

import importlib
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from .acquisition.policy import safe_argument_hash
from .source_utils import redacted_url_for_display


CandidateStatus = Literal["accepted", "duplicate", "denied", "skipped", "ingested", "failed"]


@dataclass(frozen=True)
class DiscoveredURLCandidate:
    url: str
    display_url: str
    status: CandidateStatus
    reason_code: str
    source_kind: str
    parent_url: str
    parent_display_url: str
    safe_argument_hash: str


@dataclass(frozen=True)
class SitemapParseResult:
    status: str
    reason_code: str
    candidates: list[DiscoveredURLCandidate]
    skipped: int = 0


def parse_sitemap_urlset(body: bytes, *, max_pages: int, parent_url: str = "") -> SitemapParseResult:
    upper = body[:4096].upper()
    if b"<!DOCTYPE" in upper:
        return SitemapParseResult(status="denied", reason_code="sitemap_xml_forbidden_doctype", candidates=[])
    if b"<!ENTITY" in upper:
        return SitemapParseResult(status="denied", reason_code="sitemap_xml_forbidden_entity", candidates=[])
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return SitemapParseResult(status="failed", reason_code="sitemap_parse_failed", candidates=[])
    if _local_name(root.tag) == "sitemapindex":
        return SitemapParseResult(status="denied", reason_code="sitemap_index_unsupported", candidates=[])
    if _local_name(root.tag) != "urlset":
        return SitemapParseResult(status="failed", reason_code="sitemap_parse_failed", candidates=[])
    locs = [
        (loc.text or "").strip()
        for url_node in root.iter()
        if _local_name(url_node.tag) == "url"
        for loc in list(url_node)
        if _local_name(loc.tag) == "loc" and (loc.text or "").strip()
    ]
    selected = locs[:max_pages]
    parent_display = redacted_url_for_display(parent_url) if parent_url else ""
    candidates = [
        DiscoveredURLCandidate(
            url=url,
            display_url=redacted_url_for_display(url),
            status="accepted",
            reason_code="ok",
            source_kind="sitemap",
            parent_url=parent_url,
            parent_display_url=parent_display,
            safe_argument_hash=safe_argument_hash(url),
        )
        for url in selected
    ]
    return SitemapParseResult(status="completed", reason_code="ok", candidates=candidates, skipped=max(0, len(locs) - len(selected)))
```

Also add `extract_page_links()` with lazy BeautifulSoup preference and stdlib fallback. Use `_strip_fragment(url)` before dedupe and skip unsupported schemes.

- [ ] **Step 4: Preserve import boundaries**

Do not import `bs4` at module import time. If `test_docs_package_does_not_import_optional_web_acquisition_dependencies` flags `bs4`, move the import behind `importlib.import_module("bs4")` inside the extractor.

- [ ] **Step 5: Run green tests**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/discovery.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py
git commit -m "feat: add docs source discovery parsers"
```

### Task 3: Discovery Service Dry-Run

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/discovery.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py`

- [ ] **Step 1: Write failing dry-run tests**

Add:

```python
from mcp_unified.docs.acquisition.models import FetchResponse
from mcp_unified.docs.discovery import DocsSourceDiscoveryService
from mcp_unified.docs.models import AccessScope, DiscoverSourceRequest
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore
from tldw_Server_API.tests.MCP_unified.docs.helpers import FakeResolver, FakeTransport


def _store(tmp_path: Path) -> DocsCatalogStore:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    return store


def _settings(tmp_path: Path, **overrides: object) -> DocsSettings:
    values: dict[str, object] = {
        "db_path": str(tmp_path / "docs.db"),
        "enable_web_acquisition": True,
        "enable_source_discovery": True,
        "web_source_profile": "locked_down",
        "allowed_url_prefixes": ("https://example.com/docs/", "https://example.com/sitemap.xml"),
    }
    values.update(overrides)
    return DocsSettings.from_mapping(values)


def test_discover_source_disabled_returns_without_fetch(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enable_source_discovery=False)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/xml"}, body_chunks=[b"never"])])
    service = DocsSourceDiscoveryService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.discover_source(scope=AccessScope(), request=DiscoverSourceRequest(url="https://example.com/sitemap.xml"))

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_discovery_disabled"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_discover_source_requires_web_acquisition_without_fetch(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enable_web_acquisition=False)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/xml"}, body_chunks=[b"never"])])
    service = DocsSourceDiscoveryService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.discover_source(scope=AccessScope(), request=DiscoverSourceRequest(url="https://example.com/sitemap.xml"))

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "web_acquisition_disabled"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_discover_sitemap_dry_run_returns_candidates_without_mutation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"],
            )
        ]
    )
    service = DocsSourceDiscoveryService(settings=_settings(tmp_path), store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(url="https://example.com/sitemap.xml", kind="sitemap"),
    )

    assert result["status"] == "completed"  # nosec B101
    assert result["counts"]["accepted"] == 1  # nosec B101
    assert result["candidates"][0]["url"] == "https://example.com/docs/a"  # nosec B101
    assert store.status()["counts"]["documents"] == 0  # nosec B101
    assert store.list_sources(scope=AccessScope()) == []  # nosec B101


def test_discover_source_approval_required_does_not_resolve_or_fetch(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        web_source_profile="local_first",
        allowed_url_prefixes=(),
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[b"never"])])
    service = DocsSourceDiscoveryService(settings=settings, store=_store(tmp_path), resolver=resolver, transport=transport)

    result = service.discover_source(scope=AccessScope(), request=DiscoverSourceRequest(url="https://example.com/sitemap.xml"))

    assert result["status"] == "approval_required"  # nosec B101
    assert result["reason_code"] == "source_approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101


def test_discover_sitemap_filters_scope_duplicates_and_query_leaks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[
                    b"""
                    <urlset>
                      <url><loc>https://example.com/docs/a</loc></url>
                      <url><loc>https://example.com/docs/a#fragment</loc></url>
                      <url><loc>https://evil.example/docs/a</loc></url>
                      <url><loc>https://example.com/docs/b?token=secret</loc></url>
                    </urlset>
                    """
                ],
            )
        ]
    )
    service = DocsSourceDiscoveryService(settings=_settings(tmp_path), store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(url="https://example.com/sitemap.xml", kind="sitemap"),
    )
    serialized = json.dumps(result, sort_keys=True)

    assert result["counts"]["accepted"] == 1  # nosec B101
    assert result["counts"]["duplicates"] == 1  # nosec B101
    assert result["counts"]["denied"] == 1  # nosec B101
    assert result["counts"]["skipped"] == 1  # nosec B101
    assert result["candidates"][0]["url"] == "https://example.com/docs/a"  # nosec B101
    assert "token=secret" not in serialized  # nosec B101
    assert "?token" not in serialized  # nosec B101
```

- [ ] **Step 2: Run red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_source_disabled_returns_without_fetch \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_source_requires_web_acquisition_without_fetch \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_sitemap_dry_run_returns_candidates_without_mutation \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_source_approval_required_does_not_resolve_or_fetch \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_sitemap_filters_scope_duplicates_and_query_leaks \
  -q
```

Expected: FAIL with missing `DocsSourceDiscoveryService`.

- [ ] **Step 3: Implement dry-run service**

In `discovery.py`, add `DocsSourceDiscoveryService`:

```python
class DocsSourceDiscoveryService:
    def __init__(self, *, settings: DocsSettings, store: DocsCatalogStore, resolver: object | None = None, transport: object | None = None) -> None:
        self.settings = settings
        self.store = store
        self.policy = SourcePolicy(
            web_source_profile=settings.web_source_profile,
            preapproved_domains=settings.preapproved_domains,
            allowed_url_prefixes=settings.allowed_url_prefixes,
            denied_domains=settings.denied_domains,
            allow_arbitrary_public_domains=settings.allow_arbitrary_public_domains,
        )
        self.fetcher = URLFetcher(settings=settings, policy=self.policy, resolver=resolver, transport=transport)
        self.acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)

    def discover_source(self, *, scope: AccessScope, request: DiscoverSourceRequest) -> dict[str, Any]:
        if not self.settings.enable_source_discovery:
            return {"status": "denied", "reason_code": "source_discovery_disabled", "counts": _zero_discovery_counts(), "candidates": [], "warnings": []}
        if not self.settings.enable_web_acquisition:
            return {"status": "denied", "reason_code": "web_acquisition_disabled", "counts": _zero_discovery_counts(), "candidates": [], "warnings": []}
        validation = _validate_discovery_request(request, self.settings)
        if validation is not None:
            return validation
        fetched = self.fetcher.fetch(request.url)
        if fetched.status != "fetched":
            return _fetch_failure_response(fetched)
        kind = _resolve_kind(request.kind, request.url, fetched.headers)
        candidates, warnings = self._candidates_for_fetched(kind=kind, request=request, fetched=fetched)
        candidates, filter_warnings = _filter_candidates(
            candidates,
            seed_url=fetched.canonical_url or request.url,
            policy=self.policy,
            settings=self.settings,
            max_pages=_effective_discovery_page_limit(request, self.settings),
        )
        warnings.extend(filter_warnings)
        return _discovery_response(status="completed", reason_code="ok", request=request, source=None, candidates=candidates, warnings=warnings)
```

Keep `_validate_discovery_request()` small: reject non-positive caps, `max_depth != 1`, unknown kind/mode/action, and `page_links + apply_action=register + include_seed=false`.

Add `_filter_candidates()` with these rules:

- normalize fragments away before dedupe;
- call `SourcePolicy.evaluate()` for every candidate;
- deny candidates outside the seed origin unless they match an explicit allowed URL prefix;
- skip query-bearing candidates unless `settings.persist_url_query_strings` is true;
- bound accepted candidates before any candidate-page fetches;
- keep denied/skipped/duplicate candidates in the public response with redacted display URLs and `safe_argument_hash`.

- [ ] **Step 4: Run green tests**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/discovery.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py
git commit -m "feat: add docs source discovery dry run"
```

### Task 4: Apply Register And Ingest Modes

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/discovery.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py`

- [x] **Step 1: Write failing apply tests**

Add:

```python
def test_discover_sitemap_apply_register_creates_source_without_documents(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(
                status_code=200,
                headers={"content-type": "application/xml"},
                body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"],
            )
        ]
    )
    service = DocsSourceDiscoveryService(settings=_settings(tmp_path), store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(
            url="https://example.com/sitemap.xml",
            kind="sitemap",
            mode="apply",
            apply_action="register",
            collections=("Reference",),
            keywords=("docs",),
            title="Example docs sitemap",
        ),
    )

    assert result["status"] == "completed"  # nosec B101
    assert result["source"]["source_type"] == "url_sitemap"  # nosec B101
    assert "sitemap_sync_disabled" in result["warnings"]  # nosec B101
    assert store.status()["counts"]["documents"] == 0  # nosec B101
    sources = store.list_sources(scope=AccessScope())
    assert sources[0]["metadata"]["default_collections"] == ["Reference"]  # nosec B101
    assert sources[0]["metadata"]["default_keywords"] == ["docs"]  # nosec B101


def test_discover_sitemap_apply_ingests_candidates_and_links_sitemap_source(tmp_path: Path) -> None:
    store = _store(tmp_path)
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"alpha reference body"]),
        ]
    )
    settings = _settings(tmp_path, sitemap_sync_enabled=True)
    service = DocsSourceDiscoveryService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.discover_source(
        scope=AccessScope(),
        request=DiscoverSourceRequest(
            url="https://example.com/sitemap.xml",
            kind="sitemap",
            mode="apply",
            apply_action="register_and_ingest",
            collections=("Reference",),
            keywords=("docs",),
        ),
    )

    assert result["counts"]["ingested"] == 1  # nosec B101
    assert result["candidates"][0]["document_id"]  # nosec B101
    assert store.search_chunks(scope=AccessScope(), query="alpha", limit=10)  # nosec B101
    links = store.source_document_links(scope=AccessScope(), source_id=result["source"]["id"])
    assert links[0]["source_item_uri"] == "https://example.com/docs/a"  # nosec B101
```

- [x] **Step 2: Run red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_sitemap_apply_register_creates_source_without_documents \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py::test_discover_sitemap_apply_ingests_candidates_and_links_sitemap_source \
  -q
```

Expected: FAIL because apply mode is not implemented.

- [x] **Step 3: Implement apply mode**

In `DocsSourceDiscoveryService.discover_source()`:

- compute `apply_action = request.apply_action or settings.discovery_apply_default`;
- for sitemap register/register-and-ingest, call `store.upsert_source()` with `source_type="url_sitemap"`;
- store metadata:

```python
metadata = source_defaults_metadata(keywords=request.keywords, collection_names=request.collections)
metadata.update({"discovery_kind": "sitemap", "same_origin_only": self.settings.discovery_same_origin_only})
```

- warn with `sitemap_sync_disabled` when `settings.sitemap_sync_enabled is False`;
- for `ingest` and `register_and_ingest`, call `self.acquisition.ingest_url()` for accepted candidates, passing `keywords=request.keywords` and `collection_names=request.collections`;
- when a candidate ingests and a sitemap source exists, load the document via `store.get_document(scope, document_id, mode="metadata")` and link it with `store.link_source_document()`;
- never pass query-bearing raw URLs into public response fields.

- [x] **Step 4: Run green tests**

Run the command from Step 2.

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/discovery.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_discovery.py
git commit -m "feat: apply docs source discovery"
```

### Task 5: MCP Provider Tool Wiring

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/mcp_module.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/__init__.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [x] **Step 1: Write failing provider tests**

Add:

```python
def test_provider_advertises_discover_source_when_enabled(tmp_path: Path) -> None:
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "enable_source_discovery": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/docs/",),
        }
    )
    provider = DocsMCPToolProvider(settings=settings)

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    assert "docs.discover_source" in tools  # nosec B101
    assert tools["docs.discover_source"]["metadata"]["category"] == "ingestion"  # nosec B101
    assert tools["docs.discover_source"]["metadata"]["readOnlyHint"] is False  # nosec B101


def test_provider_omits_discover_source_when_disabled(tmp_path: Path) -> None:
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"))

    names = {tool["name"] for tool in provider.tool_definitions()}

    assert "docs.discover_source" not in names  # nosec B101


def test_provider_stale_discover_source_call_is_disabled(tmp_path: Path) -> None:
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"))

    result = provider.execute("docs.discover_source", {"url": "https://example.com/sitemap.xml"}, scope=AccessScope())

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "source_discovery_disabled"  # nosec B101
```

- [x] **Step 2: Run red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_advertises_discover_source_when_enabled \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_omits_discover_source_when_disabled \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_stale_discover_source_call_is_disabled \
  -q
```

Expected: FAIL with missing provider support.

- [x] **Step 3: Wire provider**

In `DocsMCPToolProvider.__init__`, instantiate:

```python
self.discovery = (
    DocsSourceDiscoveryService(settings=settings, store=self.store)
    if settings.enable_source_discovery and settings.enable_web_acquisition
    else None
)
```

Add tool definition:

```python
_tool(
    "docs.discover_source",
    "Discover bounded sitemap or page-link URL candidates and optionally register or ingest them.",
    {
        "url": {"type": "string"},
        "kind": {"type": "string"},
        "mode": {"type": "string"},
        "apply_action": {"type": "string"},
        "max_pages": {"type": "integer"},
        "max_depth": {"type": "integer"},
        "collections": {"type": "array"},
        "keywords": {"type": "array"},
        "title": {"type": "string"},
        "include_seed": {"type": "boolean"},
    },
    ["url"],
    "ingestion",
)
```

Add `_discover_source_request_from_args()` mirroring `_sync_source_request_from_args()` with explicit validation.

- [x] **Step 4: Run green tests**

Run the command from Step 2.

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/mcp_module.py \
  apps/mcp-unified/src/mcp_unified/docs/__init__.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
git commit -m "feat: expose docs source discovery tool"
```

### Task 6: `url_sitemap` Sync Source Refresh

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/sync.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/discovery.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py`

- [x] **Step 1: Write failing sitemap sync tests**

Add:

```python
def test_url_sitemap_sync_apply_ingests_current_candidates(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope()
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_sitemap",
        canonical_uri="https://example.com/sitemap.xml",
        display_name="Example sitemap",
        source_path=None,
        source_url="https://example.com/sitemap.xml",
        redacted_source_url="https://example.com/sitemap.xml",
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={"default_keywords": ["docs"], "default_collections": ["Reference"]},
    )
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "sitemap_sync_enabled": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/sitemap.xml", "https://example.com/docs/"),
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"alpha sync body"]),
        ]
    )
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="apply"))

    assert result["status"] == "completed"  # nosec B101
    assert result["counts"]["created"] == 1  # nosec B101
    assert store.search_chunks(scope=scope, query="alpha", limit=10)  # nosec B101
    assert _document_count(store.list_keywords(scope), "docs") == 1  # nosec B101
    assert _document_count(store.list_collections(scope), "Reference") == 1  # nosec B101


def test_url_sitemap_sync_dry_run_does_not_mutate(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope()
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_sitemap",
        canonical_uri="https://example.com/sitemap.xml",
        display_name="Example sitemap",
        source_path=None,
        source_url="https://example.com/sitemap.xml",
        redacted_source_url="https://example.com/sitemap.xml",
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={"default_keywords": ["docs"], "default_collections": ["Reference"]},
    )
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "sitemap_sync_enabled": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": ("https://example.com/sitemap.xml", "https://example.com/docs/"),
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[b"<urlset><url><loc>https://example.com/docs/a</loc></url></urlset>"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"alpha sync body"]),
        ]
    )
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="dry_run"))

    assert result["status"] == "completed"  # nosec B101
    assert result["counts"]["created"] == 1  # nosec B101
    assert store.status()["counts"]["documents"] == 0  # nosec B101
    assert store.source_document_links(scope=scope, source_id=source_id) == []  # nosec B101
    assert store.status()["counts"]["sync_runs"] == 0  # nosec B101
```

Add a stale test:

```python
def test_url_sitemap_sync_tombstones_missing_links_only_in_apply_mode(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "sitemap_sync_enabled": True,
            "web_source_profile": "locked_down",
            "allowed_url_prefixes": (
                "https://example.com/sitemap.xml",
                "https://example.com/docs/old",
                "https://example.com/docs/new",
            ),
        }
    )
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_sitemap",
        canonical_uri="https://example.com/sitemap.xml",
        display_name="Example sitemap",
        source_path=None,
        source_url="https://example.com/sitemap.xml",
        redacted_source_url="https://example.com/sitemap.xml",
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    acquisition = DocsAcquisitionService(
        settings=settings,
        store=store,
        resolver=resolver,
        transport=FakeTransport(
            [FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"old sitemap body"])]
        ),
    )
    old_ingested = acquisition.ingest_url(scope=scope, url="https://example.com/docs/old")
    old_doc = store.get_document(scope, old_ingested["document"]["id"], mode="metadata")
    store.link_source_document(
        scope=scope,
        source_id=source_id,
        document_id=old_ingested["document"]["id"],
        source_item_uri="https://example.com/docs/old",
        status="active",
        last_hash=old_doc["content_hash"],
        metadata={"importer": "url_sitemap"},
    )
    service = DocsSourceSyncService(
        settings=settings,
        store=store,
        resolver=resolver,
        transport=FakeTransport(
            [
                FetchResponse(status_code=200, headers={"content-type": "application/xml"}, body_chunks=[b"<urlset><url><loc>https://example.com/docs/new</loc></url></urlset>"]),
                FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"new sitemap body"]),
            ]
        ),
    )

    result = service.sync_source(
        scope=scope,
        request=SyncSourceRequest(source_id=source_id, mode="apply", stale_policy="tombstone"),
    )
    links = store.source_document_links(scope=scope, source_id=source_id)
    old_links = [link for link in links if link["source_item_uri"] == "https://example.com/docs/old"]

    assert result["counts"]["created"] == 1  # nosec B101
    assert result["counts"]["tombstoned"] == 1  # nosec B101
    assert old_links[0]["status"] == "tombstoned"  # nosec B101
    assert store.search_chunks(scope=scope, query="old", limit=10) == []  # nosec B101
    assert store.search_chunks(scope=scope, query="new", limit=10)  # nosec B101
```

- [x] **Step 2: Run red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py::test_url_sitemap_sync_apply_ingests_current_candidates \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py::test_url_sitemap_sync_dry_run_does_not_mutate \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py::test_url_sitemap_sync_tombstones_missing_links_only_in_apply_mode \
  -q
```

Expected: FAIL because sitemap sync currently returns `sitemap_sync_disabled` or unsupported.

- [x] **Step 3: Implement shared sitemap sync**

Prefer adding a helper on `DocsSourceDiscoveryService`, for example:

```python
def discover_sitemap_candidates(self, *, url: str, max_pages: int) -> tuple[list[DiscoveredURLCandidate], list[str], dict[str, Any]]:
    fetched = self.fetcher.fetch(url)
    if fetched.status != "fetched":
        return [], [str(fetched.reason or "sitemap_fetch_failed")], {
            "status": fetched.status,
            "reason_code": fetched.reason or "sitemap_fetch_failed",
            "safe_argument_hash": fetched.safe_argument_hash,
        }
    parsed = parse_sitemap_urlset(fetched.body, max_pages=max_pages, parent_url=url)
    if parsed.reason_code != "ok":
        return [], [parsed.reason_code], {"status": parsed.status, "reason_code": parsed.reason_code}
    candidates, warnings = _filter_candidates(
        parsed.candidates,
        seed_url=fetched.canonical_url or url,
        policy=self.policy,
        settings=self.settings,
        max_pages=max_pages,
    )
    return candidates, warnings, {"status": "completed", "reason_code": "ok"}
```

Then in `DocsSourceSyncService.sync_source()` replace the disabled-only `url_sitemap` path with:

- deny with `sitemap_sync_disabled` when `settings.sitemap_sync_enabled` is false;
- fetch and parse sitemap candidates;
- enforce `min(request.max_pages, settings.max_sync_pages, settings.max_sync_run_items)`;
- compare candidates to `store.source_document_links()`;
- in dry-run, return counts/items only;
- in apply, call `DocsAcquisitionService.ingest_url()` for new/updated candidates with source defaults from `source["metadata"]`;
- link successful documents to the sitemap source;
- report or tombstone stale links according to `request.stale_policy`;
- record sync run only in apply mode.

- [x] **Step 4: Run green tests**

Run the command from Step 2.

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/sync.py \
  apps/mcp-unified/src/mcp_unified/docs/discovery.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py
git commit -m "feat: sync docs sitemap sources"
```

### Task 7: Host Shim And Config Safety

**Files:**

- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml` only if explicit defaults need to be visible.

- [x] **Step 1: Write host shim tests**

Add:

```python
@pytest.mark.asyncio
async def test_docs_module_exposes_discover_source_when_enabled(tmp_path: Path) -> None:
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={
                "db_path": str(tmp_path / "docs.db"),
                "enable_web_acquisition": True,
                "enable_source_discovery": True,
                "web_source_profile": "locked_down",
                "allowed_url_prefixes": ["https://example.com/docs/"],
            },
        )
    )
    await module.on_initialize()

    tools = {tool["name"]: tool for tool in await module.get_tools()}

    assert "docs.discover_source" in tools  # nosec B101
    assert tools["docs.discover_source"]["metadata"]["category"] == "ingestion"  # nosec B101
```

Extend `test_repo_docs_mcp_config_keeps_web_acquisition_disabled`:

```python
assert settings.get("enable_source_discovery", False) is False  # nosec B101
```

- [x] **Step 2: Run red/green**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  -q
```

Expected first run: FAIL if host/provider wiring is incomplete. After provider wiring, PASS. If config lacks explicit `enable_source_discovery`, either keep the `.get(..., False)` assertion or add the explicit disabled config key with docs-only rationale.

- [x] **Step 3: Commit**

```bash
git add tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  tldw_Server_API/Config_Files/mcp_modules.yaml
git commit -m "test: cover docs discovery host shim"
```

Only include `mcp_modules.yaml` if it actually changed.

### Task 8: Full Verification And Plan Closeout

**Files:**

- Modify: `backlog/tasks/task-12124 - Implement-standalone-MCP-docs-Stage-4B-source-discovery.md`

- [x] **Step 1: Run focused docs tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q
```

Expected: PASS.

- [x] **Step 2: Run Bandit on touched Python paths**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/mcp-unified/src/mcp_unified/docs tldw_Server_API/tests/MCP_unified/docs -f json -o /tmp/bandit_mcp_docs_stage4b_source_discovery.json
```

Expected: exit 0 or only accepted test-file assert findings already annotated with `# nosec B101`. Fix new findings in touched production code before continuing.

- [x] **Step 3: Run import-boundary test explicitly**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -q
```

Expected: PASS; no top-level optional web dependency imports and no `tldw_Server_API` imports from `mcp_unified.docs`.

- [x] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 5: Update Backlog task**

Record:

- implementation plan path;
- touched files;
- verification results;
- Bandit result or docs-only skip if this plan task stops before code implementation;
- final summary.

- [x] **Step 6: Commit plan closeout**

```bash
git add Docs/superpowers/plans/2026-07-03-standalone-mcp-docs-stage4b-source-discovery-implementation-plan.md \
  backlog/tasks/task-12124\ -\ Implement-standalone-MCP-docs-Stage-4B-source-discovery.md
git commit -m "chore: close mcp docs discovery implementation task"
```

For the current planning-only PR, Bandit can be skipped with the rationale "documentation and Backlog metadata only." For the later implementation PR, Bandit is required on touched Python paths.
