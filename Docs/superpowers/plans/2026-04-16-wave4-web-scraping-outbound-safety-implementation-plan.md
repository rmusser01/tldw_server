# Wave 4 Web Scraping And Outbound Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one explicit shared outbound-policy mode for web scraping and websearch data-plane callers, migrate the in-scope call sites to it, and prove strict-mode behavior without flipping the global default.

**Architecture:** Introduce one thin shared policy helper in the Web Scraping module that wraps raw egress evaluation and reuses the existing robots machinery instead of duplicating it. Migrate scrape callers first, then websearch callers, then finish with a direct-call audit, observability, docs, and final verification so the rollout path is measurable and bounded.

**Tech Stack:** Python, FastAPI, asyncio, httpx, pytest, loguru, Bandit

---

## File Map

### Shared Mode And Decision Layer

- Create: `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py`
  - Thin shared decision layer for scrape and websearch callers. It should resolve `compat` versus `strict`, delegate raw URL/IP checks to `evaluate_url_policy()`, expose both sync and async decision entry points, reuse `RobotsFilter` for robots fetch/cache behavior where robots checks are needed, and emit one consistent decision/result shape.
- Modify: `tldw_Server_API/app/core/config.py`
  - Add a single `WEB_OUTBOUND_POLICY_MODE` / `web_outbound_policy_mode` config surface under the existing web-scraper config load path.
- Modify: `tldw_Server_API/Config_Files/config.txt`
  - Document the new mode with explicit `compat` and `strict` values.
- Modify: `tldw_Server_API/app/core/Web_Scraping/filters.py`
  - Only if needed to expose reusable robots behavior cleanly. Do not fork the robots cache/fetch logic unless the existing filter shape proves insufficient.
- Test: `tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py`
  - New focused tests for mode resolution, compat versus strict robots behavior, reason/stage/source fields, and shared observability labels.
- Test: `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
  - Keep the legacy compat baseline explicit.
- Test: `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`
  - Prove the shared helper and `RobotsFilter` do not diverge on the same URL/mode inputs.

### Scrape Caller Migration

- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
  - Replace branch-local egress and robots handling with the shared policy helper while preserving the established blocked-result response shape.
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
  - Route enhanced scraper entry points, browser-backed navigation, sitemap fetches, and recursive crawl candidate gates through the shared helper.
- Test: `tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py`
  - Prove strict mode blocks before Playwright work starts.
- Test: `tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py`
  - Keep backend-selection and Playwright routing behavior deterministic while strict mode is introduced.
- Test: `tldw_Server_API/tests/WebScraping/test_scraping_module.py`
  - Cover scrape result shape, blocked-result behavior, and preflight interaction.
- Test: `tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py`
  - Prove recursive crawl candidates respect the same shared decision contract.

### Websearch And Research Migration

- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
  - Replace the in-scope provider entry-point `evaluate_url_policy()` branches with the shared helper, keeping provider API requests on raw egress policy only and scrape-style follow-ups on egress plus robots policy.
- Test: `tldw_Server_API/tests/WebSearch/test_websearch_core.py`
  - Add focused provider tests for compat and strict outcomes without requiring real network.
- Test: `tldw_Server_API/tests/Security/test_websearch_egress_guard.py`
  - Preserve a small explicit denial regression against a real provider seam.
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py`
  - Prove the `scrape_url` action inherits the stricter scrape behavior without introducing a branch-local bypass.

### Audit, Documentation, And Final Verification

- Create: `Docs/superpowers/reviews/web-scraping/2026-04-16-wave4-outbound-policy-callsite-audit.md`
  - Record every in-scope direct `evaluate_url_policy()` call site that existed before the migration, note the new shared-helper path used after migration, and explicitly list the out-of-scope exclusions already accepted in the design.
- Modify: `tldw_Server_API/app/core/Web_Scraping/README.md`
  - Document the new shared mode and where it applies.
- Modify: `Docs/Published/User_Guides/Server/Web_Scraping_Ingestion_Guide.md`
  - Document `compat` versus `strict` rollout behavior for server operators.

## Notes

- Keep Wave 4 bounded to the scrape and websearch data-plane:
  - `document_processing_service.py` and `workflows_webhook_dlq_service.py` stay out of scope unless implementation proves they actually depend on the same scrape and robots contract.
- Reuse existing abstractions where they are already correct:
  - `evaluate_url_policy()` remains the raw host/IP/port gate.
  - `http_client.fetch()` remains the redirect-aware fetch path.
  - `RobotsFilter` remains the preferred robots fetch/cache implementation.
- Do not silently change the global default mode in this wave.
- Do not reinterpret `respect_robots=False` into a stricter behavior in this wave.
- The final audit must show that in-scope call sites either route through the shared helper or are explicitly justified as raw preflight checks that still feed the same shared result contract.

### Task 1: Add The Shared Web Outbound Policy Mode

**Files:**
- Create: `tldw_Server_API/app/core/Web_Scraping/outbound_policy.py`
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/app/core/Web_Scraping/filters.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py`

- [x] **Step 1: Write the failing shared-policy tests**

```python
@pytest.mark.asyncio
async def test_web_outbound_policy_strict_blocks_when_robots_fetch_errors(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.outbound_policy import decide_web_outbound_policy

    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")

    async def boom(*_args, **_kwargs):
        raise RuntimeError("robots timeout")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Web_Scraping.filters.RobotsFilter.allowed",
        boom,
    )

    decision = await decide_web_outbound_policy(
        "https://example.com/page",
        respect_robots=True,
        user_agent="UA",
        source="article_extract",
        stage="pre_fetch",
    )

    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_unreachable"
```

- [x] **Step 2: Run the new helper tests to verify they fail**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py -v`
Expected: FAIL with missing shared helper imports and/or missing strict-mode config wiring.

- [x] **Step 3: Implement the minimal shared helper and config wiring**

```python
@dataclass(slots=True)
class WebOutboundPolicyDecision:
    allowed: bool
    mode: Literal["compat", "strict"]
    reason: str
    stage: str
    source: str
    details: dict[str, Any] | None = None


async def decide_web_outbound_policy(... ) -> WebOutboundPolicyDecision:
    raw = evaluate_url_policy(url)
    if not raw.allowed:
        return WebOutboundPolicyDecision(False, mode, raw.reason, stage, source)
    if not respect_robots:
        return WebOutboundPolicyDecision(True, mode, "robots_skipped", stage, source)
    ...


def decide_web_outbound_policy_sync(... ) -> WebOutboundPolicyDecision:
    raw = evaluate_url_policy(url)
    ...
```

Implementation requirements:
- add `WEB_OUTBOUND_POLICY_MODE` / `web_outbound_policy_mode` with allowed values `compat` and `strict`
- resolve mode at call time, not once at import time
- reuse `RobotsFilter` or the existing filter helpers instead of re-implementing robots fetch/cache behavior
- keep provider-call integration synchronous when `respect_robots=False`; do not force the existing sync websearch adapters through ad hoc event-loop shims
- emit structured labels or fields that later tests can assert for compat-allow versus strict-block outcomes
- `filters.py` changes remain unnecessary unless later caller migration exposes a real seam gap

- [x] **Step 4: Re-run the shared-policy tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Web_Scraping/outbound_policy.py tldw_Server_API/app/core/config.py tldw_Server_API/Config_Files/config.txt tldw_Server_API/app/core/Web_Scraping/filters.py tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py
git commit -m "feat: add shared web outbound policy mode"
```

### Task 2: Migrate Legacy And Enhanced Scrape Callers

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Test: `tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py`
- Test: `tldw_Server_API/tests/WebScraping/test_scraping_module.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py`

- [x] **Step 1: Write the failing scrape-path regressions**

```python
@pytest.mark.asyncio
async def test_article_scrape_strict_returns_blocked_result_on_robots_error(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    ...
    result = await scrape_article("https://example.com/page", method="auto")
    assert result["extraction_successful"] is False
    assert result["error"] == "Blocked by outbound policy"
    assert result["policy_reason"] == "robots_unreachable"


@pytest.mark.asyncio
async def test_playwright_path_strict_blocks_before_browser_navigation(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    ...
    assert playwright_used["used"] is False
```

- [x] **Step 2: Run the scrape-path regressions to verify they fail**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py tldw_Server_API/tests/WebScraping/test_scraping_module.py tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py -v`
Expected: FAIL because the scrape branches still use branch-local robots and egress decisions.

- [x] **Step 3: Implement the minimal scrape-call-site migration**

```python
decision = await decide_web_outbound_policy(
    url,
    respect_robots=getattr(plan, "respect_robots", True),
    user_agent=effective_ua,
    source="article_extract",
    stage="pre_fetch",
)
if not decision.allowed:
    return {
        "url": url,
        "content": "",
        "extraction_successful": False,
        "error": "Blocked by outbound policy",
        "policy_reason": decision.reason,
    }
```

Implementation requirements:
- keep the established blocked-result response shape for scrape-style operations
- block Playwright navigation before the browser launches when strict mode denies the URL
- route recursive crawl candidate gating through the same helper instead of mixing helper-based and branch-local logic
- pull the same-code legacy sync neighbors (`scrape_article_blocking()`, `scrape_from_sitemap()`, and `collect_internal_links()`) onto the shared sync helper while preserving their non-scrape result contracts

- [x] **Step 4: Re-run the focused scrape-path tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py tldw_Server_API/tests/WebScraping/test_scraping_module.py tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py tldw_Server_API/tests/WebScraping/test_scraping_module.py tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py
git commit -m "fix: enforce shared outbound policy across scrape paths"
```

### Task 3: Migrate Websearch Providers And Research URL Scraping

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Test: `tldw_Server_API/tests/WebSearch/test_websearch_core.py`
- Test: `tldw_Server_API/tests/Security/test_websearch_egress_guard.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py`

- [x] **Step 1: Write the failing websearch and research regressions**

```python
def test_websearch_provider_strict_returns_shared_policy_reason(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    ...
    with pytest.raises(Exception) as exc:
        web_search.search_web_brave(...)
    assert "robots_unreachable" in str(exc.value) or "outbound policy" in str(exc.value)


@pytest.mark.asyncio
async def test_research_agent_scrape_url_surfaces_strict_policy_block(monkeypatch):
    monkeypatch.setenv("WEB_OUTBOUND_POLICY_MODE", "strict")
    ...
    assert result.success is False
    assert result.error == "Blocked by outbound policy"
```

- [x] **Step 2: Run the websearch and research regressions to verify they fail**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/WebSearch/test_websearch_core.py tldw_Server_API/tests/Security/test_websearch_egress_guard.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py -v`
Expected: FAIL on missing shared mode behavior and missing shared reason propagation.

- [x] **Step 3: Implement the minimal websearch migration**

```python
decision = decide_web_outbound_policy_sync(
    search_url,
    respect_robots=False,
    source="websearch_provider",
    stage="provider_request",
)
if not decision.allowed:
    raise RuntimeError(f"Blocked by outbound policy: {decision.reason}")
```

Implementation requirements:
- provider API calls use the shared helper for raw egress-mode evaluation and reason normalization
- robots policy is only applied to scrape-style URL fetches and follow-up page retrieval, not synthesized for provider API endpoints
- keep `research_agent.py` untouched unless the focused test proves an explicit error-shape bridge is required
- migrate the remaining raw provider-like egress sites in `WebSearch_APIs.py` that share the same contract, including 4chan catalog/archive/thread fetches and provider stubs, instead of leaving mixed helper and non-helper seams in one file

- [x] **Step 4: Re-run the websearch and research tests**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/WebSearch/test_websearch_core.py tldw_Server_API/tests/Security/test_websearch_egress_guard.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/tests/WebSearch/test_websearch_core.py tldw_Server_API/tests/Security/test_websearch_egress_guard.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py
git commit -m "fix: align websearch and research outbound policy"
```

### Task 4: Audit Call Sites, Update Docs, And Run Final Verification

**Files:**
- Create: `Docs/superpowers/reviews/web-scraping/2026-04-16-wave4-outbound-policy-callsite-audit.md`
- Modify: `tldw_Server_API/app/core/Web_Scraping/README.md`
- Modify: `Docs/Published/User_Guides/Server/Web_Scraping_Ingestion_Guide.md`

- [x] **Step 1: Record the call-site audit and explicit exclusions**

```markdown
# Wave 4 Outbound Policy Call-Site Audit

- In scope before migration:
  - Article_Extractor_Lib.py: pre-fetch URL policy and robots gate
  - enhanced_web_scraping.py: scrape entry points, recursive crawl candidates, sitemap fetches
  - WebSearch_APIs.py: provider entry points and scrape follow-ups
- Out of scope for this wave:
  - document_processing_service.py
  - workflows_webhook_dlq_service.py
```

- [x] **Step 2: Update internal and operator-facing documentation**

Document:
- new `WEB_OUTBOUND_POLICY_MODE` / `web_outbound_policy_mode`
- `compat` versus `strict` behavior
- explicit note that strict mode is supported but not the default flip in this wave

- [x] **Step 3: Run the focused final verification set**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py tldw_Server_API/tests/Web_Scraping/test_robots_enforcement.py tldw_Server_API/tests/Web_Scraping/test_filters_and_robots.py tldw_Server_API/tests/WebScraping/test_playwright_guard_and_cookies.py tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py tldw_Server_API/tests/WebScraping/test_scraping_module.py tldw_Server_API/tests/Web_Scraping/test_recursive_crawl_semantics.py tldw_Server_API/tests/WebSearch/test_websearch_core.py tldw_Server_API/tests/Security/test_websearch_egress_guard.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py tldw_Server_API/tests/Web_Scraping/test_process_web_scraping_strategy_validation.py tldw_Server_API/tests/WebScraping/test_webscraping_usage_events.py tldw_Server_API/tests/WebScraping/test_custom_headers_support.py -v`
Expected: PASS

Run: `rg -n "evaluate_url_policy\\(" tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
Expected: only deliberately documented raw preflight checks remain, if any; all other in-scope call sites route through the shared helper.

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Web_Scraping/outbound_policy.py tldw_Server_API/app/core/Web_Scraping/filters.py tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py tldw_Server_API/app/core/config.py -f json -o /tmp/bandit_wave4_outbound_policy.json`
Expected: no new high-severity findings in touched scope.

- [x] **Step 4: Commit**

```bash
git add Docs/superpowers/reviews/web-scraping/2026-04-16-wave4-outbound-policy-callsite-audit.md tldw_Server_API/app/core/Web_Scraping/README.md Docs/Published/User_Guides/Server/Web_Scraping_Ingestion_Guide.md
git commit -m "docs: record wave4 outbound policy rollout and audit"
```
