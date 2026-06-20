# Phase 1 Research Discovery Chokepoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first backend slice of a shared research source discovery chokepoint: source catalog, source router, normalized discovery search, OA candidate enrichment, sanitized persisted snapshots, and standalone search API.

**Architecture:** Add a focused `tldw_Server_API.app.core.Research.discovery` package that owns catalog selection, provider routing, normalization, dedupe, OA candidate handling, and snapshot-safe response construction. Persist short-lived, sanitized discovery snapshots in the existing per-user `ResearchSessionsDB` so later standalone ingest can load server-owned result state without trusting client-resubmitted metadata. Expose the first UI/API surface through `GET /api/v1/research/sources` and `POST /api/v1/research/discovery/search` while leaving existing `/api/v1/paper-search/*` endpoints unchanged.

**Tech Stack:** FastAPI, Pydantic schemas, SQLite via `ResearchSessionsDB`, existing `Third_Party` provider helpers, pytest/httpx TestClient, Loguru.

---

Spec: `Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md`
Planning Backlog task: `TASK-2337`
Implementation Backlog task: `TASK-2338`

## Scope Boundary

This plan implements Phase 1 only:

- Source catalog and capability metadata.
- Source/category selection validation with hard cap errors.
- API-backed provider adapter routing for OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed, Zenodo, Figshare, OSF, and Unpaywall-style OA resolution.
- Normalized deduped discovery results with merged provenance.
- OA/full-text candidates as advisory metadata only.
- Safe URL redaction for API responses, persisted snapshots, and candidate ids.
- User-owned persisted discovery snapshots for later ingest handoff.
- Standalone search API.

This plan intentionally does not implement standalone ingest, Deep Research broker migration, compatibility endpoint delegation, or fallback site search rollout. Fallback metadata is exposed but disabled by default.

## Snapshot Storage Decision

Use the existing per-user research sessions database path: `DatabasePaths.get_research_sessions_db_path(owner_user_id)`.

Add `research_discovery_snapshots` to `ResearchSessionsDB`:

- `id TEXT PRIMARY KEY`
- `owner_user_id TEXT NOT NULL`
- `query TEXT NOT NULL`
- `request_json TEXT NOT NULL`
- `response_json TEXT NOT NULL`
- `effective_config_json TEXT NOT NULL`
- `catalog_version TEXT NOT NULL`
- `created_at TEXT NOT NULL`
- `expires_at TEXT NOT NULL`

Default retention is 24 hours. Phase 1 creates snapshots and rejects expired/mismatched-owner reads in the DB helper. A later cleanup job can call `delete_expired_discovery_snapshots`; Phase 1 adds the helper and unit coverage but does not schedule recurring cleanup.

Only sanitized response data is stored. Raw signed, expiring, token-bearing, or otherwise secret-bearing OA URLs must not appear in `response_json`, `request_json`, `effective_config_json`, API responses, logs, or `candidate_id` derivation.

## File Structure

Create:

- `tldw_Server_API/app/core/Research/discovery/__init__.py` - package exports.
- `tldw_Server_API/app/core/Research/discovery/catalog.py` - first-slice source catalog and selection resolution.
- `tldw_Server_API/app/core/Research/discovery/models.py` - core dataclasses for catalog entries, raw provider records, normalized results, OA candidates, source statuses, and discovery responses.
- `tldw_Server_API/app/core/Research/discovery/identity.py` - canonicalization, fingerprinting, dedupe, deterministic ranking, URL safety, and candidate-id helpers.
- `tldw_Server_API/app/core/Research/discovery/oa.py` - OA/full-text candidate extraction and Unpaywall DOI resolver wrapper.
- `tldw_Server_API/app/core/Research/discovery/router.py` - source router and provider adapter protocol.
- `tldw_Server_API/app/core/Research/discovery/adapters.py` - wrappers around existing `Third_Party` search helpers.
- `tldw_Server_API/app/core/Research/discovery/service.py` - orchestration and snapshot persistence boundary.
- `tldw_Server_API/app/api/v1/schemas/research_discovery_schemas.py` - request/response Pydantic models for the standalone discovery API.
- `tldw_Server_API/app/api/v1/endpoints/research_discovery.py` - FastAPI routes for source listing and discovery search.
- `tldw_Server_API/tests/Research/test_research_discovery_catalog.py`
- `tldw_Server_API/tests/Research/test_research_discovery_identity.py`
- `tldw_Server_API/tests/Research/test_research_discovery_router.py`
- `tldw_Server_API/tests/Research/test_research_discovery_adapters.py`
- `tldw_Server_API/tests/Research/test_research_discovery_service.py`
- `tldw_Server_API/tests/Research/test_research_discovery_endpoint.py`

Modify:

- `tldw_Server_API/app/core/DB_Management/ResearchSessionsDB.py` - add discovery snapshot row/dataclass/table/helpers.
- `tldw_Server_API/app/api/v1/router_groups/content.py` - register `research_discovery` under `/api/v1/research`.
- `tldw_Server_API/tests/Research/test_research_sessions_db.py` - add snapshot persistence/ownership/expiry tests.

Do not modify:

- Existing `/api/v1/paper-search/*` endpoints.
- `ResearchBroker` or Deep Research job flow.
- Media DB ingestion helpers.

## Task 1: Source Catalog And Selection Resolution

**Files:**
- Create: `tldw_Server_API/app/core/Research/discovery/__init__.py`
- Create: `tldw_Server_API/app/core/Research/discovery/catalog.py`
- Create: `tldw_Server_API/app/core/Research/discovery/models.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_catalog.py`

- [x] **Step 1: Write failing catalog tests**

Add tests that assert:

```python
def test_catalog_lists_first_slice_sources_with_capabilities():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog()
    source_ids = {source.source_id for source in catalog.list_sources()}

    assert {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    }.issubset(source_ids)
    assert catalog.get_source("openalex").capabilities.searchable is True
    assert catalog.get_source("openalex").capabilities.fallback_search_allowed is False
    assert catalog.catalog_version


def test_catalog_resolves_category_and_rejects_over_cap_selection():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog(max_selected_sources=2)

    resolved, error = catalog.resolve_selection(source_ids=[], categories=["open_research_graph"])

    assert resolved == []
    assert error is not None
    assert error.code == "source_selection_over_cap"
    assert error.selected_count > error.limit


def test_catalog_dedupes_explicit_and_category_selected_sources():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog

    catalog = default_source_catalog(max_selected_sources=10)
    resolved, error = catalog.resolve_selection(source_ids=["openalex"], categories=["open_research_graph"])

    assert error is None
    assert resolved[0].source_id == "openalex"
    assert len({entry.source_id for entry in resolved}) == len(resolved)
```

- [x] **Step 2: Run catalog tests and verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_catalog.py -v
```

Expected: FAIL because the discovery package does not exist.

- [x] **Step 3: Implement catalog dataclasses and defaults**

In `models.py`, define immutable dataclasses:

```python
@dataclass(frozen=True)
class SourceCapabilities:
    searchable: bool
    full_text_resolvable: bool
    ingestable: bool
    requires_credentials: bool
    fallback_search_allowed: bool
    rate_limited: bool


@dataclass(frozen=True)
class ResearchSourceCatalogEntry:
    source_id: str
    display_name: str
    category: str
    subcategory: str | None
    content_types: tuple[str, ...]
    access_level: str
    enabled: bool
    configured: bool
    default_discovery_mode: str
    fallback_enabled: bool
    priority: int
    provider_adapter: str | None
    site_hosts: tuple[str, ...]
    trust_notes: str
    capabilities: SourceCapabilities
    catalog_version: str


@dataclass(frozen=True)
class SourceSelectionError:
    code: str
    message: str
    selected_count: int
    limit: int
```

In `catalog.py`, create `ResearchSourceCatalog` with:

- `list_sources() -> list[ResearchSourceCatalogEntry]`
- `get_source(source_id: str) -> ResearchSourceCatalogEntry`
- `resolve_selection(source_ids: list[str], categories: list[str]) -> tuple[list[ResearchSourceCatalogEntry], SourceSelectionError | None]`
- constructor support for custom `entries` so router/service tests can create disabled or credential-required sources without polluting the default catalog
- hard cap validation after category expansion
- deterministic ordering by `priority`, then `source_id`

Set `CATALOG_VERSION = "research-discovery-v1"`.

Include first-slice entries:

- `openalex`, category `open_research_graph`, adapter `openalex`
- `semantic_scholar`, category `open_research_graph`, adapter `semantic_scholar`
- `crossref`, category `open_research_graph`, adapter `crossref`
- `arxiv`, category `preprints`, adapter `arxiv`
- `pubmed`, category `biomedical`, adapter `pubmed`
- `zenodo`, category `repositories`, adapter `zenodo`
- `figshare`, category `repositories`, adapter `figshare`
- `osf`, category `repositories`, adapter `osf`

All first-slice entries should have `fallback_search_allowed=False` and `default_discovery_mode="api"`.
All default first-slice entries should be `enabled=True`, `configured=True`, and `fallback_enabled=False`.

- [x] **Step 4: Run catalog tests and verify green**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_catalog.py -v
```

Expected: PASS.

- [x] **Step 5: Commit catalog slice**

```bash
git add tldw_Server_API/app/core/Research/discovery/__init__.py tldw_Server_API/app/core/Research/discovery/catalog.py tldw_Server_API/app/core/Research/discovery/models.py tldw_Server_API/tests/Research/test_research_discovery_catalog.py
git commit -m "feat: add research discovery source catalog"
```

## Task 2: Identity, Dedupe, Ranking, And OA Candidate Sanitization

**Files:**
- Modify: `tldw_Server_API/app/core/Research/discovery/models.py`
- Create: `tldw_Server_API/app/core/Research/discovery/identity.py`
- Create: `tldw_Server_API/app/core/Research/discovery/oa.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_identity.py`

- [x] **Step 1: Write failing identity and sanitization tests**

Cover:

```python
def test_fingerprint_prefers_doi_over_url_and_title():
    from tldw_Server_API.app.core.Research.discovery.identity import build_fingerprint

    first = build_fingerprint({"doi": "10.1000/Example", "url": "https://a.test/paper", "title": "A"})
    second = build_fingerprint({"doi": "https://doi.org/10.1000/example", "url": "https://b.test/other", "title": "B"})

    assert first == second
    assert first.startswith("doi:")


def test_merge_records_preserves_all_provenance_and_primary_source():
    from tldw_Server_API.app.core.Research.discovery.identity import normalize_and_merge_records

    results = normalize_and_merge_records(
        [
            {"source_id": "openalex", "provider": "openalex", "doi": "10.1000/example", "title": "Paper"},
            {"source_id": "crossref", "provider": "crossref", "doi": "10.1000/example", "title": "Paper"},
        ],
        catalog_version="research-discovery-v1",
    )

    assert len(results) == 1
    assert results[0].primary_source_id == "openalex"
    assert {item.source_id for item in results[0].merged_provenance} == {"openalex", "crossref"}


def test_signed_oa_url_is_redacted_from_response_snapshot_and_candidate_id():
    from tldw_Server_API.app.core.Research.discovery.oa import build_oa_candidates

    raw_url = "https://repo.example/files/paper.pdf?X-Amz-Signature=SECRET&Expires=999"
    candidates = build_oa_candidates(
        result_fingerprint="doi:10.1000/example",
        source_id="openalex",
        provider="openalex",
        doi="10.1000/example",
        raw_urls=[raw_url],
    )

    candidate = candidates[0]
    assert candidate.url_redacted is True
    assert candidate.safe_url == "https://repo.example/files/paper.pdf"
    assert "SECRET" not in candidate.candidate_id
    assert "X-Amz-Signature" not in candidate.candidate_id
    assert candidate.resolver_reference is not None
    assert candidate.requires_reresolution is True


def test_unpaywall_resolver_wraps_doi_lookup_and_sanitizes_signed_pdf_url():
    from tldw_Server_API.app.core.Research.discovery.oa import ResearchOAResolver

    calls = []

    def fake_resolve_oa_pdf(doi):
        calls.append(doi)
        return "https://repo.example/paper.pdf?token=SECRET", None

    resolver = ResearchOAResolver(resolve_oa_pdf_fn=fake_resolve_oa_pdf)
    candidates = resolver.resolve_for_result(
        result_fingerprint="doi:10.1000/example",
        source_id="unpaywall",
        provider="unpaywall",
        doi="10.1000/example",
        provider_ids={"doi": "10.1000/example"},
        raw_urls=[],
    )

    assert calls == ["10.1000/example"]
    assert candidates[0].url_redacted is True
    assert candidates[0].safe_url == "https://repo.example/paper.pdf"
    assert "SECRET" not in candidates[0].candidate_id
```

- [x] **Step 2: Run identity tests and verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_identity.py -v
```

Expected: FAIL because identity/OA modules are missing.

- [x] **Step 3: Add normalized discovery models**

Extend `models.py` with dataclasses:

```python
@dataclass(frozen=True)
class DiscoveryOACandidate:
    candidate_id: str
    candidate_type: str
    safe_url: str | None
    resolver_reference: str | None
    url_redacted: bool
    requires_reresolution: bool
    provider: str
    access_status: str | None
    license_hint: str | None
    content_type_hint: str | None
    rank: int
    confidence: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryProvenance:
    source_id: str
    provider: str
    discovery_mode: str
    provider_ids: dict[str, str]
    url: str | None
    source_rank: int | None
    status: str
    warnings: tuple[str, ...]
    safe_metadata: dict[str, Any]
    adapter_version: str


@dataclass(frozen=True)
class DiscoveryResult:
    result_id: str
    fingerprint: str
    primary_source_id: str
    primary_provider: str
    discovery_mode: str
    title: str
    authors: tuple[str, ...]
    abstract: str | None
    doi: str | None
    pmid: str | None
    pmcid: str | None
    arxiv_id: str | None
    provider_ids: dict[str, str]
    canonical_url: str | None
    published_at: str | None
    updated_at: str | None
    source_category: str | None
    oa_candidates: tuple[DiscoveryOACandidate, ...]
    recommended_candidate_id: str | None
    ingest_eligible: bool
    dedupe_confidence: float
    ranking_signals: dict[str, Any]
    warnings: tuple[str, ...]
    merged_provenance: tuple[DiscoveryProvenance, ...]
    safe_metadata: dict[str, Any]
    adapter_version: str
    catalog_version: str
```

- [x] **Step 4: Implement identity helpers**

In `identity.py`, implement:

- `normalize_doi(value: Any) -> str | None`
- `canonicalize_url(value: Any) -> str | None`
- `build_fingerprint(raw: dict[str, Any]) -> str`
- `stable_result_id(fingerprint: str, primary_source_id: str, primary_provider: str) -> str`
- `safe_provider_metadata(raw: dict[str, Any]) -> dict[str, Any]`
- `normalize_and_merge_records(records: list[dict[str, Any]], catalog_version: str) -> list[DiscoveryResult]`

Fingerprint priority:

1. DOI
2. PMID or PMCID
3. arXiv id
4. provider ids
5. canonical URL
6. normalized title plus author/date hints

Ranking should be deterministic: source priority when present, identifier strength, title length/non-empty, OA availability, then fingerprint.

Keep `safe_provider_metadata` conservative: remove raw `pdf_url`, `download_url`, `files`, `links`, `headers`, `token`, `api_key`, and any key containing `secret`, `signature`, `credential`, or `authorization`.

- [x] **Step 5: Implement OA candidate sanitizer**

In `oa.py`, implement:

- `sanitize_candidate_url(raw_url: str | None) -> tuple[str | None, bool]`
- `build_resolver_reference(source_id: str, provider: str, doi: str | None, provider_ids: dict[str, str], candidate_type: str) -> str`
- `build_candidate_id(result_fingerprint: str, candidate_type: str, provider: str, safe_url: str | None, resolver_reference: str | None) -> str`
- `build_oa_candidates(...) -> list[DiscoveryOACandidate]`
- `ResearchOAResolver`, with constructor `resolve_oa_pdf_fn: Callable[[str], tuple[str | None, str | None]] = Unpaywall.resolve_oa_pdf`
- `ResearchOAResolver.resolve_for_result(...) -> list[DiscoveryOACandidate]`, which combines provider-supplied raw URLs with DOI-based Unpaywall lookup results

Treat query keys as sensitive when lowercased keys include:

```python
{
    "access_token",
    "api_key",
    "authorization",
    "expires",
    "signature",
    "sig",
    "token",
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-signature",
    "x-goog-algorithm",
    "x-goog-credential",
    "x-goog-signature",
}
```

For sensitive URLs, return a display URL with query and fragment removed, set `url_redacted=True`, set `requires_reresolution=True`, and derive `candidate_id` from the safe URL or resolver reference only.

`ResearchOAResolver` should call Unpaywall only when a DOI is available. It should return provider-supplied candidates even when Unpaywall is not configured, and should convert Unpaywall errors into candidate warnings instead of raising unless all candidate construction itself fails.

- [x] **Step 6: Run identity tests and verify green**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_identity.py -v
```

Expected: PASS.

- [x] **Step 7: Commit identity/OA slice**

```bash
git add tldw_Server_API/app/core/Research/discovery/models.py tldw_Server_API/app/core/Research/discovery/identity.py tldw_Server_API/app/core/Research/discovery/oa.py tldw_Server_API/tests/Research/test_research_discovery_identity.py
git commit -m "feat: normalize research discovery identities"
```

## Task 3: Provider Router And First-Slice Adapters

**Files:**
- Create: `tldw_Server_API/app/core/Research/discovery/router.py`
- Create: `tldw_Server_API/app/core/Research/discovery/adapters.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/__init__.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_router.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_adapters.py`

- [ ] **Step 1: Write failing router tests**

Add tests for fake adapters, provider errors, timeout handling, rate limiting, and concurrency. The test module should import `asyncio` and `pytest`.

```python
@pytest.mark.asyncio
async def test_router_calls_adapter_for_selected_source():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class FakeAdapter:
        async def search(self, *, query, source, limit, filters):
            return [{"source_id": source.source_id, "provider": "openalex", "title": query, "doi": "10.1000/example"}]

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": FakeAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records[0]["source_id"] == "openalex"
    assert statuses[0].source_id == "openalex"
    assert statuses[0].status == "ok"


@pytest.mark.asyncio
async def test_router_records_provider_error_without_leaking_exception_details():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class FailingAdapter:
        async def search(self, **_kwargs):
            raise RuntimeError("secret token /private/key")

    catalog = default_source_catalog()
    router = ResearchSourceRouter(catalog=catalog, adapters={"openalex": FailingAdapter()})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "provider_error"
    assert "secret token" not in statuses[0].message


@pytest.mark.asyncio
async def test_router_marks_source_timeout_without_blocking_other_sources():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class SlowAdapter:
        async def search(self, **_kwargs):
            await asyncio.sleep(1)
            return []

    class FastAdapter:
        async def search(self, *, source, **_kwargs):
            return [{"source_id": source.source_id, "provider": "crossref", "title": "Fast"}]

    catalog = default_source_catalog()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={"openalex": SlowAdapter(), "crossref": FastAdapter()},
        per_source_timeout_seconds=0.01,
        max_concurrency=2,
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex"), catalog.get_source("crossref")],
        per_source_limit=3,
        filters={},
    )

    assert [record["source_id"] for record in records] == ["crossref"]
    assert {status.source_id: status.status for status in statuses} == {
        "openalex": "timeout",
        "crossref": "ok",
    }


@pytest.mark.asyncio
async def test_router_respects_rate_limiter_without_calling_adapter():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    class Adapter:
        async def search(self, **_kwargs):
            raise AssertionError("rate-limited source should not call adapter")

    async def deny_openalex(source_id):
        return source_id != "openalex"

    catalog = default_source_catalog()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={"openalex": Adapter()},
        rate_limiter=deny_openalex,
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[catalog.get_source("openalex")],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert statuses[0].status == "rate_limited"


@pytest.mark.asyncio
async def test_router_reports_policy_and_configuration_blocked_sources():
    from tldw_Server_API.app.core.Research.discovery.catalog import ResearchSourceCatalog
    from tldw_Server_API.app.core.Research.discovery.models import (
        ResearchSourceCatalogEntry,
        SourceCapabilities,
    )
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    def entry(source_id, *, enabled=True, configured=True, requires_credentials=False, adapter="adapter"):
        return ResearchSourceCatalogEntry(
            source_id=source_id,
            display_name=source_id,
            category="test",
            subcategory=None,
            content_types=("paper",),
            access_level="credentialed_api" if requires_credentials else "public_api",
            enabled=enabled,
            configured=configured,
            default_discovery_mode="api" if enabled else "disabled",
            fallback_enabled=False,
            priority=1,
            provider_adapter=adapter,
            site_hosts=(),
            trust_notes="test",
            capabilities=SourceCapabilities(
                searchable=True,
                full_text_resolvable=False,
                ingestable=False,
                requires_credentials=requires_credentials,
                fallback_search_allowed=False,
                rate_limited=False,
            ),
            catalog_version="test-v1",
        )

    catalog = ResearchSourceCatalog(
        entries=[
            entry("disabled_repo", enabled=False),
            entry("credentialed_index", configured=False, requires_credentials=True),
            entry("missing_adapter", adapter="missing"),
        ],
        max_selected_sources=3,
    )
    router = ResearchSourceRouter(catalog=catalog, adapters={})

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[
            catalog.get_source("disabled_repo"),
            catalog.get_source("credentialed_index"),
            catalog.get_source("missing_adapter"),
        ],
        per_source_limit=3,
        filters={},
    )

    assert records == []
    assert {status.source_id: status.status for status in statuses} == {
        "disabled_repo": "policy_blocked",
        "credentialed_index": "credentials_missing",
        "missing_adapter": "provider_not_configured",
    }


@pytest.mark.asyncio
async def test_router_enforces_bounded_concurrency():
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.router import ResearchSourceRouter

    active = 0
    max_seen = 0

    class CountingAdapter:
        async def search(self, *, source, **_kwargs):
            nonlocal active, max_seen
            active += 1
            max_seen = max(max_seen, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [{"source_id": source.source_id, "provider": source.source_id, "title": source.source_id}]

    catalog = default_source_catalog(max_selected_sources=10)
    adapter = CountingAdapter()
    router = ResearchSourceRouter(
        catalog=catalog,
        adapters={
            "openalex": adapter,
            "semantic_scholar": adapter,
            "crossref": adapter,
        },
        max_concurrency=1,
    )

    records, statuses = await router.search_sources(
        query="machine learning",
        sources=[
            catalog.get_source("openalex"),
            catalog.get_source("semantic_scholar"),
            catalog.get_source("crossref"),
        ],
        per_source_limit=3,
        filters={},
    )

    assert len(records) == 3
    assert all(status.status == "ok" for status in statuses)
    assert max_seen == 1
```

- [ ] **Step 2: Run router tests and verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_router.py -v
```

Expected: FAIL because router/adapters are missing.

- [ ] **Step 3: Implement router models and protocol**

Add to `models.py`:

```python
@dataclass(frozen=True)
class SourceStatus:
    source_id: str
    provider: str | None
    status: str
    message: str | None
    result_count: int
    elapsed_ms: float | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryExecutionPolicy:
    per_source_timeout_seconds: float
    total_timeout_seconds: float
    max_concurrency: int
```

In `router.py`, define:

```python
class DiscoveryProviderAdapter(Protocol):
    async def search(
        self,
        *,
        query: str,
        source: ResearchSourceCatalogEntry,
        limit: int,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class SourceRateLimiter(Protocol):
    def __call__(self, source_id: str) -> bool | Awaitable[bool]: ...
```

`ResearchSourceRouter.search_sources(...)` should:

- iterate selected sources deterministically using bounded `asyncio` tasks and `asyncio.Semaphore(max_concurrency)`
- return `policy_blocked` without calling an adapter when `source.enabled` is false or `source.default_discovery_mode == "disabled"`
- return `credentials_missing` without calling an adapter when `source.capabilities.requires_credentials` is true and `source.configured` is false
- return `provider_not_configured` without calling an adapter when `source.provider_adapter` is missing or not present in the adapter registry
- call `source.provider_adapter` for runnable sources
- apply `rate_limiter(source.source_id)` before calling the adapter; a denied source returns `SourceStatus(..., status="rate_limited", result_count=0, ...)`
- enforce `per_source_timeout_seconds` around each adapter call with `asyncio.wait_for`
- return `(records, statuses)`
- add `source_id`, `source_category`, `provider`, `discovery_mode`, `adapter_version`, and `source_priority` to each raw record
- sanitize provider errors to stable messages such as `"Provider request failed."`
- not log raw exception messages that may contain secrets

`ResearchSourceRouter.__init__` should accept:

- `catalog`
- `adapters`
- `per_source_timeout_seconds: float = 10.0`
- `max_concurrency: int = 4`
- `rate_limiter: SourceRateLimiter | None = None`

Default `rate_limiter` is `None`, which allows all sources. This is the configured per-provider rate-limit enforcement point for Phase 1; later work can replace the default with persistent quotas.
Expose a read-only `adapter_names: tuple[str, ...]` property for tests and diagnostics.

- [ ] **Step 4: Implement existing-function provider adapters**

In `adapters.py`, create small adapter classes or a generic function adapter that wraps:

- `OpenAlex.search_openalex(q, offset=0, limit=limit, filter_venue=None, from_year=filters.get("from_year"), to_year=filters.get("to_year"))`
- `Semantic_Scholar.search_papers_semantic_scholar(query, offset=0, limit=limit, fields_of_study=None, publication_types=None, year_range=filters.get("year_range"), venue=None, min_citations=None, open_access_only=False)`
- `Crossref.search_crossref(q, offset=0, limit=limit, filter_venue=None, from_year=filters.get("from_year"), to_year=filters.get("to_year"))`
- `Arxiv.search_arxiv_custom_api(query, author=None, year=filters.get("year"), start_index=0, page_size=limit)`
- `PubMed.search_pubmed(query, offset=0, limit=limit, from_year=filters.get("from_year"), to_year=filters.get("to_year"), free_full_text=False)`
- `Zenodo.search_records(q, page=1, size=limit, type_=None, subtype=None, communities=None)`
- `Figshare.search_articles(q, page=1, page_size=limit, order=None, order_direction=None, search_for=None)`
- `OSF.search_preprints(term=query, page=1, results_per_page=limit, provider=None, from_date=filters.get("from_date"))`

Also implement:

```python
def default_discovery_adapters() -> dict[str, DiscoveryProviderAdapter]:
    return {
        "openalex": OpenAlexDiscoveryAdapter(),
        "semantic_scholar": SemanticScholarDiscoveryAdapter(),
        "crossref": CrossrefDiscoveryAdapter(),
        "arxiv": ArxivDiscoveryAdapter(),
        "pubmed": PubMedDiscoveryAdapter(),
        "zenodo": ZenodoDiscoveryAdapter(),
        "figshare": FigshareDiscoveryAdapter(),
        "osf": OSFDiscoveryAdapter(),
    }
```

Normalize each adapter return shape into raw dictionaries that include at least:

- `title`
- `authors` when present
- `abstract` or `snippet`
- `doi`
- `pmid`/`pmcid`/`arxiv_id` when present
- `url`
- `pdf_url` when present
- `provider`
- `provider_ids`

Use `asyncio.to_thread` for synchronous provider calls. Do not call local FastAPI endpoints.

Add explicit fake-function adapter tests in `tldw_Server_API/tests/Research/test_research_discovery_adapters.py` for each first-slice adapter. Each test should inject the provider function into the adapter constructor, call `.search(...)`, and assert normalized raw fields are present without making network calls. Cover OpenAlex, Semantic Scholar, Crossref, arXiv, PubMed, Zenodo, Figshare, and OSF.

Add a registry test in `test_research_discovery_adapters.py`:

```python
def test_default_discovery_adapters_contains_first_slice_sources():
    from tldw_Server_API.app.core.Research.discovery.adapters import default_discovery_adapters

    adapters = default_discovery_adapters()

    assert {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    }.issubset(adapters)
```

- [ ] **Step 5: Run router tests and provider adapter unit tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Research/test_research_discovery_router.py \
  tldw_Server_API/tests/Research/test_research_discovery_adapters.py \
  tldw_Server_API/tests/Research/test_research_provider_adapters.py \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit router/adapters slice**

```bash
git add tldw_Server_API/app/core/Research/discovery/router.py tldw_Server_API/app/core/Research/discovery/adapters.py tldw_Server_API/app/core/Research/discovery/__init__.py tldw_Server_API/app/core/Research/discovery/models.py tldw_Server_API/tests/Research/test_research_discovery_router.py tldw_Server_API/tests/Research/test_research_discovery_adapters.py
git commit -m "feat: route research discovery providers"
```

## Task 4: Discovery Snapshot Persistence

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ResearchSessionsDB.py`
- Modify: `tldw_Server_API/tests/Research/test_research_sessions_db.py`

- [ ] **Step 1: Write failing snapshot persistence tests**

Add tests:

```python
def test_discovery_snapshot_round_trip_is_owner_scoped(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = db.create_discovery_snapshot(
        owner_user_id="1",
        query="open access batteries",
        request_json={"query": "open access batteries"},
        response_json={"results": [{"result_id": "res_1"}]},
        effective_config_json={"source_ids": ["openalex"]},
        catalog_version="research-discovery-v1",
        retention_hours=24,
    )

    assert db.get_discovery_snapshot(snapshot.id, owner_user_id="1") is not None
    assert db.get_discovery_snapshot(snapshot.id, owner_user_id="2") is None


def test_discovery_snapshot_rejects_expired_rows(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    snapshot = db.create_discovery_snapshot(
        owner_user_id="1",
        query="expired",
        request_json={},
        response_json={},
        effective_config_json={},
        catalog_version="research-discovery-v1",
        retention_hours=-1,
    )

    assert db.get_discovery_snapshot(snapshot.id, owner_user_id="1") is None
```

- [ ] **Step 2: Run DB tests and verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_sessions_db.py -v
```

Expected: FAIL for missing snapshot helpers.

- [ ] **Step 3: Implement schema and row mapper**

In `ResearchSessionsDB.py`:

- add `timedelta` import
- add dataclass:

```python
@dataclass(frozen=True)
class ResearchDiscoverySnapshotRow:
    id: str
    owner_user_id: str
    query: str
    request_json: dict[str, Any]
    response_json: dict[str, Any]
    effective_config_json: dict[str, Any]
    catalog_version: str
    created_at: str
    expires_at: str
```

- create table and indexes in `_ensure_schema()`:

```sql
CREATE TABLE IF NOT EXISTS research_discovery_snapshots (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    request_json TEXT NOT NULL DEFAULT '{}',
    response_json TEXT NOT NULL DEFAULT '{}',
    effective_config_json TEXT NOT NULL DEFAULT '{}',
    catalog_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_research_discovery_snapshots_owner_created
    ON research_discovery_snapshots(owner_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_discovery_snapshots_owner_expires
    ON research_discovery_snapshots(owner_user_id, expires_at);
```

- add `_discovery_snapshot_from_row(...)`
- add `create_discovery_snapshot(...)`
- add `get_discovery_snapshot(snapshot_id: str, *, owner_user_id: str)`
- add `delete_expired_discovery_snapshots(now: str | None = None) -> int`

Use ids shaped as `rd_<12 hex chars>`.

- [ ] **Step 4: Run DB tests and verify green**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_sessions_db.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit snapshot persistence slice**

```bash
git add tldw_Server_API/app/core/DB_Management/ResearchSessionsDB.py tldw_Server_API/tests/Research/test_research_sessions_db.py
git commit -m "feat: persist research discovery snapshots"
```

## Task 5: Discovery Service Orchestration

**Files:**
- Create: `tldw_Server_API/app/core/Research/discovery/service.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/models.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_service.py`

- [ ] **Step 1: Write failing service tests**

Cover successful search, over-cap validation, partial failure, total timeout handling, and sanitized snapshot storage. The test module should import `asyncio` and `pytest`.

```python
@pytest.mark.asyncio
async def test_service_persists_sanitized_discovery_snapshot(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    class FakeRouter:
        async def search_sources(self, *, query, sources, per_source_limit, filters):
            from tldw_Server_API.app.core.Research.discovery.models import SourceStatus
            return [
                {
                    "source_id": "openalex",
                    "provider": "openalex",
                    "title": "OA Paper",
                    "doi": "10.1000/example",
                    "pdf_url": "https://repo.example/paper.pdf?X-Amz-Signature=SECRET",
                }
            ], [SourceStatus("openalex", "openalex", "ok", None, 1, 1.0, ())]

    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchDiscoveryService(
        catalog=default_source_catalog(),
        router=FakeRouter(),
        db_factory=lambda _owner_user_id: db,
    )

    response = await service.search(
        owner_user_id="1",
        query="OA Paper",
        source_ids=["openalex"],
        categories=[],
        per_source_limit=5,
        total_limit=10,
        filters={},
        fallback_policy="disabled",
    )

    stored = db.get_discovery_snapshot(response.discovery_id, owner_user_id="1")
    assert stored is not None
    assert "SECRET" not in str(stored.response_json)
    assert response.results[0].oa_candidates[0].url_redacted is True


@pytest.mark.asyncio
async def test_service_returns_validation_error_for_category_over_cap(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    service = ResearchDiscoveryService(
        catalog=default_source_catalog(max_selected_sources=1),
        router=None,
        db_factory=lambda _owner_user_id: None,
    )

    with pytest.raises(ValueError, match="source_selection_over_cap"):
        await service.search(
            owner_user_id="1",
            query="too broad",
            source_ids=[],
            categories=["open_research_graph"],
            per_source_limit=5,
            total_limit=10,
            filters={},
            fallback_policy="disabled",
        )


@pytest.mark.asyncio
async def test_service_enforces_total_timeout(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    class SlowRouter:
        async def search_sources(self, **_kwargs):
            await asyncio.sleep(1)
            return [], []

    service = ResearchDiscoveryService(
        catalog=default_source_catalog(),
        router=SlowRouter(),
        db_factory=lambda _owner_user_id: None,
        total_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError, match="research_discovery_total_timeout"):
        await service.search(
            owner_user_id="1",
            query="timeout",
            source_ids=["openalex"],
            categories=[],
            per_source_limit=5,
            total_limit=10,
            filters={},
            fallback_policy="disabled",
        )


@pytest.mark.asyncio
async def test_service_hard_fails_when_every_source_fails_with_no_results(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.models import SourceStatus
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    class FailingRouter:
        async def search_sources(self, **_kwargs):
            return [], [SourceStatus("openalex", "openalex", "provider_error", "Provider request failed.", 0, 1.0, ())]

    service = ResearchDiscoveryService(
        catalog=default_source_catalog(),
        router=FailingRouter(),
        db_factory=lambda _owner_user_id: None,
    )

    with pytest.raises(RuntimeError, match="research_discovery_all_sources_failed"):
        await service.search(
            owner_user_id="1",
            query="all failed",
            source_ids=["openalex"],
            categories=[],
            per_source_limit=5,
            total_limit=10,
            filters={},
            fallback_policy="disabled",
        )


@pytest.mark.asyncio
async def test_service_rejects_when_no_selected_sources_are_runnable(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.models import SourceStatus
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    class BlockedRouter:
        async def search_sources(self, **_kwargs):
            return [], [SourceStatus("openalex", "openalex", "policy_blocked", "Source is disabled.", 0, 1.0, ())]

    service = ResearchDiscoveryService(
        catalog=default_source_catalog(),
        router=BlockedRouter(),
        db_factory=lambda _owner_user_id: None,
    )

    with pytest.raises(ValueError, match="research_discovery_no_runnable_sources"):
        await service.search(
            owner_user_id="1",
            query="blocked",
            source_ids=["openalex"],
            categories=[],
            per_source_limit=5,
            total_limit=10,
            filters={},
            fallback_policy="disabled",
        )


def test_default_service_wires_first_slice_adapter_registry():
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    service = ResearchDiscoveryService()

    assert {
        "openalex",
        "semantic_scholar",
        "crossref",
        "arxiv",
        "pubmed",
        "zenodo",
        "figshare",
        "osf",
    }.issubset(service.adapter_names)


@pytest.mark.asyncio
async def test_service_rejects_site_search_fallback_by_default(tmp_path):
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    service = ResearchDiscoveryService(
        catalog=default_source_catalog(),
        router=None,
        db_factory=lambda _owner_user_id: None,
    )

    with pytest.raises(ValueError, match="research_discovery_fallback_disabled"):
        await service.search(
            owner_user_id="1",
            query="fallback attempt",
            source_ids=["openalex"],
            categories=[],
            per_source_limit=5,
            total_limit=10,
            filters={},
            fallback_policy="site_search",
        )


@pytest.mark.asyncio
async def test_service_defaults_empty_source_selection_to_open_research_graph(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.discovery.catalog import default_source_catalog
    from tldw_Server_API.app.core.Research.discovery.models import SourceStatus
    from tldw_Server_API.app.core.Research.discovery.service import ResearchDiscoveryService

    class FakeRouter:
        async def search_sources(self, *, sources, **_kwargs):
            assert {source.source_id for source in sources} == {
                "openalex",
                "semantic_scholar",
                "crossref",
            }
            return [
                {
                    "source_id": "openalex",
                    "provider": "openalex",
                    "title": "Defaulted search",
                    "doi": "10.1000/default",
                }
            ], [SourceStatus("openalex", "openalex", "ok", None, 1, 1.0, ())]

    db = ResearchSessionsDB(tmp_path / "research.db")
    service = ResearchDiscoveryService(
        catalog=default_source_catalog(),
        router=FakeRouter(),
        db_factory=lambda _owner_user_id: db,
    )

    response = await service.search(
        owner_user_id="1",
        query="default source selection",
        source_ids=[],
        categories=[],
        per_source_limit=5,
        total_limit=10,
        filters={},
        fallback_policy="disabled",
    )

    assert response.effective_config["defaulted_categories"] == ["open_research_graph"]
    assert db.get_discovery_snapshot(response.discovery_id, owner_user_id="1") is not None
```

- [ ] **Step 2: Run service tests and verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_service.py -v
```

Expected: FAIL because service is missing.

- [ ] **Step 3: Implement service response models**

Add dataclasses to `models.py`:

```python
@dataclass(frozen=True)
class DiscoveryMetrics:
    selected_source_count: int
    result_count: int
    deduped_result_count: int
    oa_candidate_count: int
    elapsed_ms: float | None


@dataclass(frozen=True)
class DiscoverySearchResponse:
    discovery_id: str
    query: str
    results: tuple[DiscoveryResult, ...]
    source_statuses: tuple[SourceStatus, ...]
    warnings: tuple[str, ...]
    effective_config: dict[str, Any]
    catalog_version: str
    metrics: DiscoveryMetrics
```

- [ ] **Step 4: Implement `ResearchDiscoveryService`**

`ResearchDiscoveryService.__init__` should accept:

- `catalog: ResearchSourceCatalog | None = None`
- `router: ResearchSourceRouter | None = None`
- `oa_resolver: ResearchOAResolver | None = None`
- `db_factory: Callable[[str], ResearchSessionsDB] | None = None`
- `snapshot_retention_hours: int = 24`
- `total_timeout_seconds: float = 30.0`

Default `db_factory` should call `ResearchSessionsDB(DatabasePaths.get_research_sessions_db_path(owner_user_id))`.
Default `oa_resolver` should instantiate `ResearchOAResolver()`.
Default `router` should be `ResearchSourceRouter(catalog=catalog, adapters=default_discovery_adapters())` so the production endpoint supports the first-slice source set without test-only dependency overrides.
Expose a read-only `adapter_names: tuple[str, ...]` property that returns the underlying router adapter names for diagnostics and test coverage.

`search(...)` should:

1. validate non-empty query
2. if both `source_ids` and `categories` are empty, default categories to `["open_research_graph"]` and record `defaulted_categories=["open_research_graph"]` in `effective_config`
3. resolve source/category selection through catalog
4. reject over-cap selections with `ValueError("source_selection_over_cap:<selected_count>:<limit>")`
5. reject fallback policy other than `"disabled"` in Phase 1 with `ValueError("research_discovery_fallback_disabled")` unless all selected catalog entries explicitly allow it and are configured with `fallback_enabled=True`
6. call router under `asyncio.wait_for(..., timeout=total_timeout_seconds)` and raise `TimeoutError("research_discovery_total_timeout")` if the total search exceeds the budget
7. merge and normalize raw records
8. attach OA candidates through `oa_resolver.resolve_for_result(...)`, passing provider URLs from `pdf_url`, `oa_url`, and `download_url` plus DOI/provider ids for Unpaywall re-resolution
9. enforce `total_limit` after ranking
10. if zero records and all source statuses are `policy_blocked`, `credentials_missing`, or `provider_not_configured`, raise `ValueError("research_discovery_no_runnable_sources")`
11. if zero records and every selected source has a failure status, raise `RuntimeError("research_discovery_all_sources_failed")`
12. build source statuses and warnings
13. persist sanitized response through `ResearchSessionsDB.create_discovery_snapshot`
14. return `DiscoverySearchResponse`

Serialize dataclasses to JSON with a small private helper such as `_dataclass_to_json(value: Any) -> Any`; do not store raw provider records.

Use explicit status sets in the service:

```python
_NO_RUNNABLE_STATUSES = {"policy_blocked", "credentials_missing", "provider_not_configured"}
_FAILURE_STATUSES = _NO_RUNNABLE_STATUSES | {"provider_error", "timeout", "rate_limited"}
```

Only raise the all-failed error when there is a status for each selected source and all statuses are in `_FAILURE_STATUSES`.

- [ ] **Step 5: Run service tests and verify green**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_service.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit service slice**

```bash
git add tldw_Server_API/app/core/Research/discovery/service.py tldw_Server_API/app/core/Research/discovery/models.py tldw_Server_API/tests/Research/test_research_discovery_service.py
git commit -m "feat: orchestrate research discovery search"
```

## Task 6: Standalone Discovery API

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/research_discovery_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/research_discovery.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_endpoint.py`

- [ ] **Step 1: Write failing endpoint tests**

Add lightweight FastAPI tests:

```python
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


def test_get_research_sources_endpoint_lists_catalog():
    from tldw_Server_API.app.api.v1.endpoints import research_discovery

    app = FastAPI()
    app.include_router(research_discovery.router, prefix="/api/v1/research")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    with TestClient(app) as client:
        response = client.get("/api/v1/research/sources")

    assert response.status_code == 200
    assert response.json()["catalog_version"]
    openalex = next(item for item in response.json()["sources"] if item["source_id"] == "openalex")
    assert openalex["capabilities"]["searchable"] is True
    assert openalex["configured"] is True
    assert openalex["fallback_enabled"] is False
    assert openalex["fallback_configurable"] is False


def test_discovery_search_endpoint_passes_owner_and_returns_response():
    from tldw_Server_API.app.api.v1.endpoints import research_discovery

    class StubService:
        async def search(self, **kwargs):
            assert kwargs["owner_user_id"] == "1"
            return {
                "discovery_id": "rd_1",
                "query": kwargs["query"],
                "results": [],
                "source_statuses": [],
                "warnings": [],
                "effective_config": {"source_ids": kwargs["source_ids"]},
                "catalog_version": "research-discovery-v1",
                "metrics": {
                    "selected_source_count": 1,
                    "result_count": 0,
                    "deduped_result_count": 0,
                    "oa_candidate_count": 0,
                    "elapsed_ms": 1.0,
                },
            }

    app = FastAPI()
    app.include_router(research_discovery.router, prefix="/api/v1/research")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_discovery.get_research_discovery_service] = lambda: StubService()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/research/discovery/search",
            json={"query": "open access", "source_ids": ["openalex"]},
        )

    assert response.status_code == 200
    assert response.json()["discovery_id"] == "rd_1"
```

- [ ] **Step 2: Run endpoint tests and verify red**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_endpoint.py -v
```

Expected: FAIL because endpoint/schema files are missing.

- [ ] **Step 3: Implement Pydantic schemas**

In `research_discovery_schemas.py`, define request and response schemas that mirror the dataclasses:

- `ResearchSourceCapabilitiesResponse`
- `ResearchSourceResponse`
- `ResearchSourceListResponse`
- `ResearchDiscoverySearchRequest`
- `ResearchDiscoveryOACandidateResponse`
- `ResearchDiscoveryProvenanceResponse`
- `ResearchDiscoveryResultResponse`
- `ResearchDiscoverySourceStatusResponse`
- `ResearchDiscoveryMetricsResponse`
- `ResearchDiscoverySearchResponse`

Use bounded request defaults:

```python
class ResearchDiscoverySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    categories: list[str] = Field(default_factory=list, max_length=20)
    per_source_limit: int = Field(default=5, ge=1, le=20)
    total_limit: int = Field(default=25, ge=1, le=100)
    fallback_policy: str = Field(default="disabled")
    filters: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Implement endpoint**

In `research_discovery.py`:

- define `router = APIRouter(tags=["research-discovery"])`
- import `get_request_user` from `API_Deps.auth_deps`
- `get_research_discovery_service() -> ResearchDiscoveryService`
- `GET /sources`
- `POST /discovery/search`

Map service `ValueError`:

- `source_selection_over_cap` -> HTTP 422 with useful detail
- fallback disabled/policy errors -> HTTP 422
- `research_discovery_no_runnable_sources` -> HTTP 422

Map service runtime failures:

- `research_discovery_all_sources_failed` -> HTTP 502
- `research_discovery_total_timeout` -> HTTP 504

Use `str(current_user.id)` for owner id, matching `research_runs.py`.

- [ ] **Step 5: Register router**

In `router_groups/content.py`, add an `ImportedRouterSpec` near the existing research router:

```python
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.research_discovery",
    log_name="research_discovery",
    prefix=f"{API_V1_PREFIX}/research",
    tags=("research-discovery",),
    route_key="research",
),
```

- [ ] **Step 6: Run endpoint tests and route import smoke**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_endpoint.py -v
```

Expected: PASS.

Run:

```bash
source .venv/bin/activate && python - <<'PY'
from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
specs = list(iter_content_router_specs())
assert any(
    spec.name == "research_discovery"
    and spec.prefix == "/api/v1/research"
    for spec in specs
)
print("content router specs ok")
PY
```

Expected: prints `content router specs ok`.

- [ ] **Step 7: Commit API slice**

```bash
git add tldw_Server_API/app/api/v1/schemas/research_discovery_schemas.py tldw_Server_API/app/api/v1/endpoints/research_discovery.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/tests/Research/test_research_discovery_endpoint.py
git commit -m "feat: expose research discovery search API"
```

## Task 7: Focused Regression, Security Verification, And Backlog Finalization

**Files:**
- Modify: `backlog/tasks/task-2338 - Implement-Phase-1-research-discovery-chokepoint.md` when finalizing implementation.
- No code files unless earlier task verification exposes a defect.

- [ ] **Step 1: Run focused research discovery tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Research/test_research_discovery_catalog.py \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_discovery_router.py \
  tldw_Server_API/tests/Research/test_research_discovery_adapters.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
  tldw_Server_API/tests/Research/test_research_discovery_endpoint.py \
  tldw_Server_API/tests/Research/test_research_sessions_db.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run adjacent existing research tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Research/test_research_provider_adapters.py \
  tldw_Server_API/tests/Research/test_research_provider_config.py \
  tldw_Server_API/tests/Research/test_research_runs_endpoint.py \
  tldw_Server_API/tests/Research/test_unpaywall_sanitizers.py \
  -v
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched code scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/core/DB_Management/ResearchSessionsDB.py \
  tldw_Server_API/app/api/v1/endpoints/research_discovery.py \
  tldw_Server_API/app/api/v1/schemas/research_discovery_schemas.py \
  -f json -o /tmp/bandit_research_discovery_phase1.json
```

Expected: no new high or medium findings in touched code.

- [ ] **Step 4: Run syntax/import check for touched modules**

Run:

```bash
source .venv/bin/activate && python -m compileall \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/api/v1/endpoints/research_discovery.py \
  tldw_Server_API/app/api/v1/schemas/research_discovery_schemas.py
```

Expected: successful compile.

- [ ] **Step 5: Update Backlog task**

Update `TASK-2338` with:

- plan path
- implementation summary
- touched files
- verification commands/results
- known skips or blockers

- [ ] **Step 6: Final implementation commit**

If Task 7 changed only Backlog metadata:

```bash
git add "backlog/tasks/task-2338 - Implement-Phase-1-research-discovery-chokepoint.md"
git commit -m "docs: finalize research discovery implementation tracking"
```

If Task 7 included code fixes, include those files and use a `fix:` commit message describing the defect.

## Completion Checklist

- [ ] Phase 1 endpoints are available at `GET /api/v1/research/sources` and `POST /api/v1/research/discovery/search`.
- [ ] Discovery snapshots are persisted owner-scoped, sanitized, and short-lived.
- [ ] Over-cap category/source selections return validation errors instead of silent truncation.
- [ ] Fallback site search remains disabled by default.
- [ ] Raw signed/token-bearing URLs do not appear in API responses, snapshots, logs, or candidate ids.
- [ ] Existing provider-specific paper-search endpoints are unchanged.
- [ ] Focused pytest commands pass.
- [ ] Bandit touched-scope scan has no new actionable findings.
- [ ] Backlog task is updated with verification.
