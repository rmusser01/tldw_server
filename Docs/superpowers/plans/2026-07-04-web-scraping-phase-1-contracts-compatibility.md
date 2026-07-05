# Web Scraping Phase 1 Contracts And Compatibility Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for implementation checkpoints. The tasks are sequential because they share the same contracts package and test file; do not dispatch implementation tasks in parallel. Use fresh reviewer subagents after each task for spec compliance and code quality.

**Goal:** Add typed internal Web_Scraping contracts and compatibility tests that preserve current public imports, dict-shaped results, and pre-scrape analyzer functionality before any runtime behavior moves.

**Architecture:** Create a new low-level `tldw_Server_API.app.core.Web_Scraping.contracts` package with stdlib-only dataclasses, status enums, failure containers, and conversion helpers. Add no-network tests that lock current legacy entry points and public dictionaries. Runtime wrappers remain untouched in Phase 1.

**Backlog:** `TASK-12158`

---

## Scope

Allowed:

- Add internal contracts, status enums, failure/result dataclasses, and conversion helpers.
- Add compatibility tests that prove current imports from `Docs/Design/web_scraping_refactor_import_inventory.json` remain resolvable.
- Add no-network tests for current article, enhanced scraper, preflight, and WebSearch dictionary shapes.
- Add docs and Backlog notes for Phase 1.

Not allowed:

- Move runtime behavior out of `Article_Extractor_Lib.py`, `enhanced_web_scraping.py`, or `WebSearch_APIs.py`.
- Change public API response shapes.
- Change scraping, WebSearch, preflight analyzer, policy, robots, cookie, crawl, or Playwright behavior.
- Add tests that require real network, real browser execution, or external provider credentials.

## File Map

- Create `tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/statuses.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/errors.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/requests.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/results.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/conversion.py`
- Create `tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py`
- Create `tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py`
- Modify `Docs/Design/WebScraping.md`
- Modify `backlog/tasks/task-12158 - Plan-and-implement-Web-Scraping-refactor-Phase-1-contracts-and-compatibility-tests.md`

## Shared Contract Requirements

The contracts package must not import legacy wrapper modules:

- `Article_Extractor_Lib`
- `enhanced_web_scraping`
- `WebSearch_APIs`

The contracts may use only stdlib imports and local contracts-package imports.

`RuntimeFailure` must store an already-sanitized `public_message`; do not store raw exception text in a field named `message`. Call sites that later adapt exceptions must sanitize before creating the failure object.

Preflight contracts must model the current analyzer attachment shape exactly:

```python
{
    "analysis": {
        "results": {
            "js": {"status": "success", "js_required": True},
            "tls": {"status": "active"},
        },
        "score": {"level": "medium"},
        "recommendations": {"actions": ["use_browser"]},
    },
    "advice": {
        "backend": "curl",
        "method": "playwright",
        "notes": ["js_required", "tls_active"],
    },
}
```

WebSearch contracts must cover the full current `perform_websearch` parameter surface:

```python
search_engine, search_query, content_country, search_lang, output_lang,
result_count, date_range, safesearch, site_blacklist, exactTerms,
excludeTerms, filter, geolocation, search_result_language,
sort_results_by, search_params, site_whitelist, google_domain
```

WebSearch result contracts must distinguish two current public shapes:

- Initialized/search aggregate payloads from `initialize_web_search_results_dict`, which include top-level metadata such as `warnings`.
- Processed provider result items, which are shaped as `{"title", "url", "content", "metadata"}` and preserve provider metadata such as `date_published`, `source`, `relevance_score`, `snippet`, and provider-specific extras.

Compatibility conversion must keep article and enhanced scraper failure shapes separate:

- Article blocked result includes `title`, `author`, `date`, `content`, `extraction_successful`, `error`, and policy fields.
- Enhanced blocked result includes `url`, `error`, `extraction_successful`, and policy fields.

## Task 1: Status And Failure Contracts

**Files:**

- Create `tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/statuses.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/errors.py`
- Create `tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py`

### Steps

- [ ] Add failing tests for `WebScrapingStatus` values and `RuntimeFailure`.
- [ ] Verify the tests fail because the contracts package does not exist.
- [ ] Implement `WebScrapingStatus` with values:
  `ok`, `blocked`, `policy_denied`, `timeout`, `budget_exhausted`, `external_tool_disabled`, `unavailable`, `error`.
- [ ] Implement `RuntimeFailure` as a frozen slots dataclass with:
  `status`, `public_message`, `reason`, `mode`, `stage`, `source`, `backend`, `provider`.
- [ ] Add `RuntimeFailure.as_policy_fields()` returning only present policy keys:
  `policy_reason`, `policy_mode`, `policy_stage`, `policy_source`.
- [ ] Re-export the new types from `contracts.__init__`.
- [ ] Add an AST import-boundary test for `contracts/` that permits only stdlib imports and relative imports inside the contracts package.
- [ ] Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py
```

Expected: status/failure tests pass.

### Required Tests

```python
@pytest.mark.unit
def test_web_scraping_status_values_match_refactor_design() -> None:
    assert [status.value for status in WebScrapingStatus] == [
        "ok",
        "blocked",
        "policy_denied",
        "timeout",
        "budget_exhausted",
        "external_tool_disabled",
        "unavailable",
        "error",
    ]


@pytest.mark.unit
def test_runtime_failure_exposes_sanitized_public_message_and_policy_fields() -> None:
    failure = RuntimeFailure(
        status=WebScrapingStatus.POLICY_DENIED,
        public_message="Blocked by outbound policy",
        reason="robots_unreachable",
        mode="strict",
        stage="pre_fetch",
        source="article_extract",
    )

    assert failure.public_message == "Blocked by outbound policy"
    assert failure.as_policy_fields() == {
        "policy_reason": "robots_unreachable",
        "policy_mode": "strict",
        "policy_stage": "pre_fetch",
        "policy_source": "article_extract",
    }


@pytest.mark.unit
def test_contract_package_import_boundary_stays_stdlib_only() -> None:
    import ast
    from pathlib import Path

    allowed_roots = {
        "__future__",
        "copy",
        "dataclasses",
        "enum",
        "types",
        "typing",
        "collections",
    }
    contracts_dir = Path("tldw_Server_API/app/core/Web_Scraping/contracts")

    for path in contracts_dir.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".", 1)[0] for alias in node.names}
                assert roots <= allowed_roots, (path, roots - allowed_roots)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                root = (node.module or "").split(".", 1)[0]
                assert root in allowed_roots, (path, root)
```

## Task 2: Request And Result Contracts

**Files:**

- Modify `tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/requests.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/results.py`
- Modify `tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py`

### Steps

- [ ] Add failing tests for request normalization, defensive copying, full WebSearch fields, preflight analysis/advice shape, and result defaults.
- [ ] Implement immutable copy helpers for mappings, sequences, and nested public payloads.
- [ ] Implement `ScrapeContext`.
- [ ] Implement `ScrapeRequest` with normalized `url`, `method`, `backend`, immutable headers, immutable custom cookies, `include_preflight`, and context.
- [ ] Implement `CrawlRequest` with normalized `base_url`, `max_pages`, `max_depth`, `include_external`, and context.
- [ ] Implement `SearchRequest` with the full current `perform_websearch` parameter surface listed above.
- [ ] Implement `PreflightAdvice` with `backend`, `method`, and tuple `notes`.
- [ ] Implement `PreflightResult` with top-level `analysis`, `advice`, `status`, and optional `failure`.
- [ ] Implement `ExtractionResult`, `CrawlResult`, `SearchResult`, and `SearchResultsPayload`.
- [ ] Re-export the new types from `contracts.__init__`.
- [ ] Run the contract tests.

Expected: all Task 1 and Task 2 tests pass.

### Required Test Cases

- Mutating input headers, cookies, context metadata, and nested preflight analysis after construction must not mutate the contract objects.
- Empty `ScrapeRequest.url`, `CrawlRequest.base_url`, and `SearchRequest.search_query` must raise `ValueError` with clear messages.
- `PreflightResult` must preserve analyzer keys `results`, `score`, and `recommendations`, and `PreflightAdvice.notes` must become a tuple.
- `SearchRequest` must preserve `site_whitelist`, `site_blacklist`, `exactTerms`, `excludeTerms`, `filter`, `geolocation`, `search_result_language`, `sort_results_by`, `search_params`, and `google_domain`.
- `SearchResultsPayload` defaults must match `initialize_web_search_results_dict`: `safesearch == "active"`, empty `results`, empty `warnings`, `processing_error is None`.
- `SearchResult` must support the current processed provider item shape: `title`, `url`, `content`, and `metadata`.
- `SearchResult.metadata` must preserve arbitrary provider metadata and extras, including `date_published`, `source`, `relevance_score`, `snippet`, and provider-specific keys.
- `SearchResultsPayload` must preserve optional top-level provider extras through an explicit `extra_fields` or equivalent mapping while keeping the initialized payload key set exact when no extras are present.

## Task 3: Compatibility Conversion Helpers

**Files:**

- Modify `tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py`
- Create `tldw_Server_API/app/core/Web_Scraping/contracts/conversion.py`
- Modify `tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py`

### Steps

- [ ] Add failing conversion tests for extraction, preflight, article failure, enhanced failure, search result, and full WebSearch payload shapes.
- [ ] Implement `preflight_result_to_public_dict`.
- [ ] Implement `extraction_result_to_public_dict`.
- [ ] Implement `article_failure_to_public_dict`.
- [ ] Implement `enhanced_failure_to_public_dict`.
- [ ] Implement `search_result_to_public_dict`.
- [ ] Implement `search_results_to_public_dict`.
- [ ] Re-export conversion helpers from `contracts.__init__`.
- [ ] Run the contract tests.

Expected: all contract and conversion tests pass.

### Required Conversion Shapes

Preflight conversion emits only public attachment keys. `status` and `failure` remain internal:

```python
{
    "analysis": {
        "results": {"js": {"status": "success", "js_required": True}},
        "score": {"level": "medium"},
        "recommendations": {"actions": ["use_browser"]},
    },
    "advice": {
        "backend": "curl",
        "method": "playwright",
        "notes": ["js_required", "tls_active"],
    },
}
```

Article policy denial for robots:

```python
{
    "url": "https://example.com/blocked",
    "title": "N/A",
    "author": "N/A",
    "date": "N/A",
    "content": "",
    "extraction_successful": False,
    "error": "Blocked by outbound policy",
    "policy_reason": "robots_unreachable",
    "policy_mode": "strict",
    "policy_stage": "pre_fetch",
    "policy_source": "article_extract",
}
```

Article policy denial for non-robots:

```python
{
    "url": "https://example.com/blocked",
    "title": "N/A",
    "author": "N/A",
    "date": "N/A",
    "content": "",
    "extraction_successful": False,
    "error": "Egress denied: deny_test",
    "policy_reason": "deny_test",
    "policy_mode": "compat",
    "policy_stage": "pre_fetch",
    "policy_source": "article_extract",
}
```

Enhanced policy denial for non-robots:

```python
{
    "url": "https://example.com/blocked",
    "error": "Egress denied: deny_test",
    "extraction_successful": False,
    "policy_reason": "deny_test",
    "policy_mode": "compat",
    "policy_stage": "pre_fetch",
    "policy_source": "enhanced_scrape",
}
```

WebSearch payload conversion must include this exact key set:

```python
{
    "search_engine",
    "search_query",
    "content_country",
    "search_lang",
    "output_lang",
    "result_count",
    "date_range",
    "safesearch",
    "site_whitelist",
    "site_blacklist",
    "exactTerms",
    "excludeTerms",
    "filter",
    "geolocation",
    "search_result_language",
    "sort_results_by",
    "google_domain",
    "results",
    "total_results_found",
    "search_time",
    "error",
    "warnings",
    "processing_error",
}
```

Processed provider result conversion must emit and preserve this item shape:

```python
{
    "title": "Result title",
    "url": "https://example.com/result",
    "content": "Result summary",
    "metadata": {
        "date_published": "2026-07-04",
        "source": "Example",
        "relevance_score": 0.9,
        "snippet": "Result summary",
        "provider_extra": "kept",
    },
}
```

Top-level processed-provider extras such as Brave `city`, `state`, and `more_results_available` are not part of the initialized payload key set. If supplied to the contract through `extra_fields`, conversion must preserve them; if not supplied, conversion must not invent them.

## Task 4: Public Compatibility Contract Tests

**Files:**

- Create `tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py`

### Steps

- [ ] Add a dynamic import test that reads `Docs/Design/web_scraping_refactor_import_inventory.json` and verifies every recorded `module` import and `imported_name` remains resolvable.
- [ ] Add no-network tests for current article blocked dictionaries using `scrape_article` with `decide_web_outbound_policy` monkeypatched to return `WebOutboundPolicyDecision`.
- [ ] Add no-network tests for current enhanced blocked dictionaries using `EnhancedWebScraper.scrape_article` with `decide_web_outbound_policy` monkeypatched to return `WebOutboundPolicyDecision`.
- [ ] Add no-network preflight preservation tests for `Article_Extractor_Lib.scrape_article`:
  - monkeypatch `load_and_log_configs` to enable `web_scraper_preflight_analyzers` and `web_scraper_preflight_include_results`
  - monkeypatch `decide_web_outbound_policy` to allow
  - monkeypatch `scraper_analyzers.run_analysis` to return a concrete analyzer-shaped payload containing `results`, `score`, and `recommendations`
  - for TLS advice, monkeypatch `_fetch_with_curl` and `extract_article_with_pipeline` so the lightweight path returns a local success dict
  - for JS advice, monkeypatch `async_playwright` with a local async fake and monkeypatch `extract_article_with_pipeline` so the Playwright path returns a local success dict
  - assert public `preflight_analysis` contains exactly `analysis` and `advice` for both direct article paths
- [ ] Add no-network preflight preservation tests for `EnhancedWebScraper.scrape_article`:
  - enable `web_scraper_preflight_analyzers`
  - enable `web_scraper_preflight_include_results`
  - monkeypatch `decide_web_outbound_policy` to allow
  - monkeypatch `_run_preflight_analysis` to return a concrete analyzer-shaped payload containing `results`, `score`, and `recommendations`
  - monkeypatch `_scrape_with_playwright` or `_scrape_with_trafilatura` to return a local success dict
  - assert public `preflight_analysis` contains exactly `analysis` and `advice`
  - assert JS advice can set `method == "playwright"` and TLS advice can set `backend == "curl"` when backend setting is `auto`
- [ ] Add WebSearch initialization tests that assert the full initialized key set and the `include_domains` fallback for `site_whitelist`.
- [ ] Add WebSearch processed-result tests that assert provider items keep `{"title", "url", "content", "metadata"}` and preserve arbitrary metadata/provider extras.
- [ ] Add representative no-network public shape tests for `extract_article_with_pipeline` and `ScrapingJob.to_dict`.
- [ ] Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py
```

Expected: compatibility tests pass without production code changes.

### Required Details

The dynamic inventory test must handle both import styles:

- `import tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib`
- `from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib`

For `from` imports, first check `hasattr(imported_module, imported_name)`. If missing, try importing `f"{module}.{imported_name}"` to support package submodule imports. Fail with the inventory record in the assertion message if neither works.

The compatibility test must cover at least these current imported names through the inventory, not a handpicked subset:

- `recursive_scrape`
- `scrape_by_url_level`
- `scrape_from_sitemap`
- `clear_extraction_caches`
- `get_extraction_cache_stats`
- `JobPriority`
- `scraper_analyzers.run_analysis`
- `handlers.resolve_handler`
- `ua_profiles.build_browser_headers`
- `url_utils.normalize_for_crawl`

## Task 5: Documentation, Verification, And Final Review

**Files:**

- Modify `Docs/Design/WebScraping.md`
- Modify `backlog/tasks/task-12158 - Plan-and-implement-Web-Scraping-refactor-Phase-1-contracts-and-compatibility-tests.md`

### Steps

- [ ] Add a short `Phase 1 Contracts` section to `Docs/Design/WebScraping.md` linking this plan and explaining that contracts are additive, stdlib-only, and not wired into runtime behavior yet.
- [ ] Update `TASK-12158` with changed files, implementation notes, and verification evidence.
- [ ] Run focused verification:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -x --tb=short \
  tldw_Server_API/tests/Web_Scraping \
  tldw_Server_API/tests/WebScraping \
  tldw_Server_API/tests/WebSearch
```

- [ ] Run compile:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m py_compile \
  tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/statuses.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/errors.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/requests.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/results.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/conversion.py
```

- [ ] Run Bandit:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping/contracts \
  -f json -o /tmp/bandit_web_scraping_phase1_contracts.json
```

- [ ] Run diff hygiene:

```bash
git diff --check
```

- [ ] Run the Phase 0 import guardrail:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

- [ ] Perform final self-review and subagent review before commit.
- [ ] Commit all Phase 1 changes together if every check passes.

## Final Verification Command Set

Run before considering Phase 1 complete:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -x --tb=short \
  tldw_Server_API/tests/Web_Scraping \
  tldw_Server_API/tests/WebScraping \
  tldw_Server_API/tests/WebSearch

/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m py_compile \
  tldw_Server_API/app/core/Web_Scraping/contracts/__init__.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/statuses.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/errors.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/requests.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/results.py \
  tldw_Server_API/app/core/Web_Scraping/contracts/conversion.py

/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping/contracts \
  -f json -o /tmp/bandit_web_scraping_phase1_contracts.json

git diff --check

/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

## Plan Self-Review

- Spec coverage: Covers Phase 1 only: typed internal contracts plus compatibility tests. Runtime behavior remains in legacy wrappers.
- Pre-scrape analyzer preservation: Corrected from generic `PreflightResult.results` to the current public attachment shape with only `analysis` and `advice` keys. Both direct `Article_Extractor_Lib.scrape_article` and `EnhancedWebScraper.scrape_article` require no-network tests.
- Import coverage: Uses the Phase 0 import inventory dynamically so active helper imports are not missed.
- Import boundary: Adds an AST guard so the new contracts package stays stdlib-only and does not import runtime wrappers, config, HTTP clients, or loggers.
- WebSearch coverage: Models the full current `perform_websearch` input surface, the full `initialize_web_search_results_dict` output key set, `include_domains` fallback to `site_whitelist`, and the processed provider item shape `{"title", "url", "content", "metadata"}`.
- Failure shape coverage: Separates article and enhanced scraper policy-denial conversion to avoid blending incompatible public dictionaries.
- Sanitization: `RuntimeFailure` stores `public_message` explicitly so raw exception text is not implied as safe.
- Risk: Contracts are not wired into runtime code in Phase 1. Later phases must add adapter tests before moving behavior behind these contracts.
