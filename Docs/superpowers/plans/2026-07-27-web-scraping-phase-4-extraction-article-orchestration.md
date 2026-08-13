# Web Scraping Phase 4 Extraction And Article Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move article extraction and governed single-page scraping into canonical packages while preserving the pre-scrape analyzer, public compatibility, enhanced-scraper behavior, and outbound security controls.

**Architecture:** Build neutral bounded-regex, content, and selector leaves first; move pure HTML extraction above those leaves; then compose policy, optional governed preflight, bounded HTTP/browser acquisition, and bounded extraction execution in `orchestration`. Keep `Article_Extractor_Lib.py` as an explicit compatibility facade plus deferred Phase 5 code, with no copied implementation of Phase 4 responsibilities.

**Tech Stack:** Python 3.10+, asyncio, dataclasses, `regex`, BeautifulSoup, lxml, Trafilatura, Playwright Chromium/CDP, existing Web_Scraping runtime/policy/preflight facades, pytest, Hypothesis, Ruff, Black, Bandit.

**Backlog:** `TASK-12989`

**Design:** `Docs/superpowers/specs/2026-07-26-web-scraping-phase-4-extraction-article-orchestration-design.md`

---

## Delivery Rules

Phase 4 is a merge train, not one large pull request:

1. Phase 4A creates shared leaves and predecessor fixtures.
2. Phase 4B starts from merged 4A and moves extraction.
3. Phase 4C starts from merged 4B and moves article orchestration.
4. Phase 4D starts from merged 4C and runs final integration and documentation gates.

Before editing production code in each unit, create one Backlog child task referencing
`TASK-12989`, set it to `In Progress`, and record the exact unit plan tasks in its
implementation notes. Finish, independently review, rebase, and merge one unit before
starting the next. Do not dispatch production tasks from different delivery units in
parallel.

All Python commands use the project environment explicitly:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
```

## Scope

Included:

- Bounded untrusted/configured regex evaluation.
- Neutral content formatting and metadata helpers.
- Neutral selector/schema DSL implementation and Watchlists compatibility exports.
- Canonical extraction strategies, caches, pipeline, traces, and result assembly.
- Canonical async, blocking, and raw-browser article entry points.
- Optional governed preflight and independent per-dispatch egress decisions.
- HTTP body, browser transfer, and rendered-HTML limits.
- Bounded extraction executor generations and cancellation behavior.
- Internal consumer migrations, compatibility re-exports, and the service keyword fix.

Excluded:

- Moving recursive crawl, sitemap, bookmark, ingestion, progress, or job state.
- Moving WebSearch provider workflows or parsers.
- Removing legacy compatibility imports.
- Unifying direct and enhanced Trafilatura or browser behavior.
- Enabling direct-browser plan headers, plan cookies, or plan proxies.
- Claiming Playwright transport-level DNS pinning.

## File Map

### Phase 4A

- Create: `Helper_Scripts/web_scraping_phase4_fixtures.py`
- Create: `tldw_Server_API/tests/Web_Scraping/fixtures/phase4/manifest.json`
- Create: `tldw_Server_API/tests/Web_Scraping/fixtures/phase4/*.json`
- Create: `tldw_Server_API/app/core/Web_Scraping/safe_regex.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/content/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/content/formatting.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/content/metadata.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/caches.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/engine.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/schema.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
- Modify: `tldw_Server_API/app/core/Watchlists/fetchers.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_predecessor_fixtures.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_safe_regex.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_content.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py`

### Phase 4B

- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/dependencies.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/caches.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/throttles.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/enrichment.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/pipeline.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/jsonld.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/regex.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/cluster.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/llm.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/schema.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/trafilatura.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/observability.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/handlers.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/policy/probe.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_properties.py`

### Phase 4C

- Modify: `tldw_Server_API/app/core/http_client.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/requests.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_models.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/executor.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/article.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/app/core/Collections/reading_service.py`
- Modify: `tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- Modify: `tldw_Server_API/app/core/Watchlists/fetchers.py`
- Modify: `tldw_Server_API/app/core/Workflows/adapters/rag/search.py`
- Modify: `tldw_Server_API/app/core/WebSearch/Web_Search.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py`
- Modify: `tldw_Server_API/app/services/web_scraping_service.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Create: `tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_executor.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_compatibility.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_consumer_imports.py`
- Create: `tldw_Server_API/tests/WebScraping/integration/test_phase4_article_browser_smoke.py`

### Phase 4D

- Modify: `Docs/Design/WebScraping.md`
- Modify: `Docs/Design/WebScraping_Refactor_Import_Inventory.md`
- Modify: `Docs/Design/web_scraping_refactor_import_inventory.json`
- Modify: `tldw_Server_API/app/core/Web_Scraping/README.md`
- Modify: files still importing legacy Phase 4 names when the regenerated inventory
  proves that a canonical import is safe and not a deferred compatibility consumer.

## Stable Contracts

The implementation must bind these exact public signatures:

```python
async def scrape_article(
    url: str,
    custom_cookies: list[dict[str, Any]] | None = None,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]: ...

def scrape_article_blocking(
    url: str,
    custom_cookies: list[dict[str, Any]] | None = None,
    *,
    allow_llm_extraction: bool = True,
) -> dict[str, Any]: ...

def scrape_article_sync(url: str) -> dict[str, Any]: ...
```

The default strategy order remains:

```python
DEFAULT_EXTRACTION_STRATEGY_ORDER = [
    "jsonld",
    "schema",
    "regex",
    "llm",
    "cluster",
    "trafilatura",
]
```

Public boundary failures introduced by Phase 4 use only these codes:

```python
PUBLIC_FAILURE_CODES = frozenset({
    "policy_error",
    "regex_invalid",
    "regex_too_large",
    "regex_timeout",
    "selector_invalid",
    "provider_error",
    "fetch_error",
    "browser_error",
    "response_too_large",
    "extraction_error",
})
```

Boundary-to-field mapping is fixed:

| Boundary | Public field and value |
| --- | --- |
| Target policy evaluation | Article `error="policy_error"` |
| Explicit policy denial | Preserve the safe blocked-result shape and policy fields |
| Lightweight acquisition | Article `error="fetch_error"` |
| Guarded browser acquisition | Article `error="browser_error"` |
| Any acquisition limit | Article `error="response_too_large"` |
| Pipeline boundary | Article `error="extraction_error"` |
| Generated regex validation | Generator `error` set to the matching regex code |
| Generated regex/schema provider | Generator `error="provider_error"` |
| LLM extraction provider | Extraction `llm_error="provider_error"` |
| Selector parse/evaluation | Validation-entry `error="selector_invalid"` |

Direct Playwright receives only the plan/profile user agent and copied caller
`custom_cookies`. Plan cookies, plan headers, and plan proxies remain absent from
the browser context. Headless Chromium, 1280x720 viewport, retries, timeout,
stealth hook/delay, `domcontentloaded`, and `networkidle` behavior remain as
characterized. Route continuation never injects cookie, authorization, or proxy
authorization headers.

## Phase 4A: Shared Leaf Components

### Task 0: Rebase, Baseline, And Create The 4A Child Task

**Files:** No production file edits.

- [x] **Step 1: Start from current `origin/dev`**

```bash
git fetch origin
git rebase origin/dev
git status --short --branch
```

Expected: rebase succeeds and the worktree is clean. If the approved design or
plan conflicts, preserve its approved contracts and rerun focused design checks.

- [x] **Step 2: Create the Phase 4A Backlog child task**

Use Backlog MCP `task_create` with parent/reference `TASK-12989`, title
`Phase 4A shared Web_Scraping leaves`, status `In Progress`, and modified files
from the Phase 4A file map.

- [x] **Step 3: Run the predecessor baseline**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_contracts.py \
  tldw_Server_API/tests/WebScraping/test_selector_validation.py \
  tldw_Server_API/tests/WebScraping/test_schema_dsl_extraction.py
```

Expected: all selected predecessor tests pass. Record exact counts in the 4A task.

### Task 1: Capture Immutable Predecessor Fixtures

**Files:**

- Create: `Helper_Scripts/web_scraping_phase4_fixtures.py`
- Create: `tldw_Server_API/tests/Web_Scraping/fixtures/phase4/manifest.json`
- Create: `tldw_Server_API/tests/Web_Scraping/fixtures/phase4/*.json`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_predecessor_fixtures.py`

- [x] **Step 1: Write the failing manifest test**

```python
def test_phase4_fixture_manifest_is_pinned() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["predecessor_commit"])
    assert set(manifest["cases"]) == {
        "content",
        "metadata",
        "selectors",
        "extraction",
        "article_orchestration_fakes",
    }
```

- [x] **Step 2: Verify RED**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_predecessor_fixtures.py
```

Expected: fail because the manifest and generator do not exist.

- [x] **Step 3: Implement explicit fixture generation**

The helper must require both `--predecessor-commit` and `--output`; it must never
run during normal tests. Build deterministic local cases, normalize timestamps,
cache state, random values, and metric events, and write sorted ASCII JSON:

```python
SCHEMA_VERSION = 1

def build_manifest(predecessor_commit: str, case_files: dict[str, str]) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", predecessor_commit) is None:
        raise ValueError("predecessor_commit must be a full commit id")
    return {
        "schema_version": SCHEMA_VERSION,
        "predecessor_commit": predecessor_commit,
        "cases": dict(sorted(case_files.items())),
    }
```

Fixture tests load expected JSON and call only the canonical implementation under
test. They must not import a copied predecessor module or regenerate files.
The generator must also compare `--predecessor-commit` with `git rev-parse HEAD`
and reject a mismatch, preventing fixtures from claiming false provenance.
The differential helper accepts `behavior_change: int | None`, rejects values
outside `range(1, 12)`, and requires every expected difference to name exactly
one of the eleven approved behavior changes.

- [x] **Step 4: Generate and review fixtures**

```bash
BASE_COMMIT="$(git rev-parse HEAD)"
python Helper_Scripts/web_scraping_phase4_fixtures.py \
  --predecessor-commit "$BASE_COMMIT" \
  --output tldw_Server_API/tests/Web_Scraping/fixtures/phase4
python -m json.tool tldw_Server_API/tests/Web_Scraping/fixtures/phase4/manifest.json >/tmp/phase4-fixture-manifest.json
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_predecessor_fixtures.py
```

Expected: generator exits 0, manifest records the full predecessor commit, and tests pass.

- [x] **Step 5: Commit**

```bash
git add Helper_Scripts/web_scraping_phase4_fixtures.py \
  tldw_Server_API/tests/Web_Scraping/fixtures/phase4 \
  tldw_Server_API/tests/Web_Scraping/test_phase4_predecessor_fixtures.py
git commit -m "test(web-scraping): pin phase 4 predecessor fixtures"
```

### Task 2: Add Bounded Regex And Integrate Router/Generated Patterns

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/safe_regex.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/scraper_router.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_safe_regex.py`

- [x] **Step 1: Write failing bounded-regex tests**

Cover invalid patterns, 4,096-character pattern limit, 8,192-character router
input limit, 1,000,000-character generated sample limit, 100ms timeout, flag
normalization, and router fail-open non-match behavior. Use injected limits and a
deterministic fake compiled object for timeout tests.

```python
def test_router_pattern_timeout_is_a_non_match() -> None:
    result = search_untrusted(
        "(a+)+$",
        "a" * 200 + "!",
        limits=SafeRegexLimits(timeout_s=0.000001),
    )
    assert result.code == "regex_timeout"
    assert result.matched is False
```

- [x] **Step 2: Verify RED**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_safe_regex.py
```

Expected: import failure for `safe_regex`.

- [x] **Step 3: Implement the leaf API**

```python
@dataclass(frozen=True, slots=True)
class SafeRegexLimits:
    max_pattern_chars: int = 4_096
    max_input_chars: int = 8_192
    timeout_s: float = 0.100

@dataclass(frozen=True, slots=True)
class SafeRegexResult:
    matched: bool
    match: Any | None = None
    code: str | None = None

def search_untrusted(
    pattern: str,
    value: str,
    *,
    flags: int = 0,
    limits: SafeRegexLimits = SafeRegexLimits(),
) -> SafeRegexResult: ...
```

Use the installed `regex` package timeout. Map compile failure to `regex_invalid`,
oversize to `regex_too_large`, and timeout to `regex_timeout`; never include the
pattern or raw exception in the result or log.

- [x] **Step 4: Replace only untrusted/configured regex uses**

Route `ScraperRouter` configured patterns and generated-regex sample validation
through `safe_regex`. Keep trusted built-in PII and boilerplate catalogs compiled
with stdlib `re`. Selector transforms and field patterns are integrated in Task 4,
after the neutral selector package exists.

- [x] **Step 5: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_safe_regex.py \
  tldw_Server_API/tests/Web_Scraping/test_router_validation.py \
  tldw_Server_API/tests/WebScraping/test_regex_pattern_generation.py
git add tldw_Server_API/app/core/Web_Scraping/safe_regex.py \
  tldw_Server_API/app/core/Web_Scraping/scraper_router.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_safe_regex.py
git commit -m "feat(web-scraping): bound configured regex execution"
```

Expected: all focused tests pass.

### Task 3: Move Neutral Content Helpers

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/content/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/content/formatting.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/content/metadata.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_content.py`

- [x] **Step 1: Bind formatting and metadata behavior**

Add fixture-driven tests for paragraph formatting, canonical envelopes, the
64-level nesting guard, malformed pass-through, body-only hashing, and direct
legacy export identity:

```python
def test_legacy_content_exports_are_canonical() -> None:
    assert legacy.convert_html_to_markdown is content.convert_html_to_markdown
    assert legacy.ContentMetadataHandler is content.ContentMetadataHandler
```

- [x] **Step 2: Verify RED**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_content.py
```

Expected: canonical content package is missing.

- [x] **Step 3: Move one implementation and re-export it**

```python
# content/__init__.py
from .formatting import convert_html_to_markdown
from .metadata import ContentMetadataHandler

__all__ = ["ContentMetadataHandler", "convert_html_to_markdown"]
```

Move the existing behavior without prose, whitespace, marker, timestamp-format,
or hash changes. Replace the legacy bodies with imports from `content`.

- [x] **Step 4: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_content.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py
git add tldw_Server_API/app/core/Web_Scraping/content \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_content.py
git commit -m "refactor(web-scraping): extract neutral content helpers"
```

### Task 4: Move Selector Engine And Preserve Watchlists Exports

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/caches.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/engine.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/selectors/schema.py`
- Modify: `tldw_Server_API/app/core/Watchlists/fetchers.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py`

- [x] **Step 1: Add differential and architecture tests**

Bind the selector DSL, errors/warnings/counts, XPath/CSS fast paths, transforms,
bounded regex failures, bounded LRU stats/clear, thread safety, endpoint behavior,
and direct Watchlists export identity:

```python
def test_watchlists_selector_exports_are_canonical() -> None:
    assert fetchers.validate_selector_rules is selectors.validate_selector_rules
    assert fetchers.extract_schema_fields is selectors.extract_schema_fields
    assert fetchers.get_selector_cache_stats is selectors.get_selector_cache_stats
    assert fetchers.clear_selector_caches is selectors.clear_selector_caches
```

The AST guard must reject imports from `selectors` to Watchlists, extraction,
orchestration, enhanced scraping, WebSearch, or `Article_Extractor_Lib`.

- [x] **Step 2: Verify RED**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py
```

Expected: canonical selector package is missing.

- [x] **Step 3: Move caches, engine, and schema facade**

```python
# selectors/__init__.py
from .caches import clear_selector_caches, get_selector_cache_stats
from .schema import extract_schema_fields, validate_selector_rules

__all__ = [
    "clear_selector_caches",
    "extract_schema_fields",
    "get_selector_cache_stats",
    "validate_selector_rules",
]
```

`engine.py` owns safety checks, CSS/XPath compilation, contextualization, and
node selection. `schema.py` owns field normalization, DSL walking, extraction,
validation, and transforms. Return `selector_invalid` instead of raw parser or
evaluation messages while preserving `selector_too_complex:*` codes. Route
`regex_replace` transforms and regex field extraction through `safe_regex`.

- [x] **Step 4: Replace Watchlists bodies with direct imports**

Keep private Watchlists-only fetch/network helpers in `fetchers.py`. Move only
the shared selector responsibility and update endpoint imports to the canonical
facade where no compatibility behavior is needed.

- [x] **Step 5: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py \
  tldw_Server_API/tests/WebScraping/test_selector_validation.py \
  tldw_Server_API/tests/WebScraping/test_selector_fast_paths.py \
  tldw_Server_API/tests/WebScraping/test_schema_dsl_extraction.py \
  tldw_Server_API/tests/Watchlists
git add tldw_Server_API/app/core/Web_Scraping/selectors \
  tldw_Server_API/app/core/Watchlists/fetchers.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py
git commit -m "refactor(web-scraping): extract shared selector engine"
```

### Task 5: Complete Phase 4A Gates And Merge

**Files:** Phase 4A files and its Backlog child task.

- [x] **Step 1: Run Phase 4A gates**

```bash
python -m compileall -q \
  Helper_Scripts/web_scraping_phase4_fixtures.py \
  tldw_Server_API/app/core/Web_Scraping/content \
  tldw_Server_API/app/core/Web_Scraping/selectors \
  tldw_Server_API/app/core/Web_Scraping/safe_regex.py
python -m ruff check \
  Helper_Scripts/web_scraping_phase4_fixtures.py \
  tldw_Server_API/app/core/Web_Scraping/content \
  tldw_Server_API/app/core/Web_Scraping/selectors \
  tldw_Server_API/app/core/Web_Scraping/safe_regex.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_*.py
python -m black --check \
  Helper_Scripts/web_scraping_phase4_fixtures.py \
  tldw_Server_API/app/core/Web_Scraping/content \
  tldw_Server_API/app/core/Web_Scraping/selectors \
  tldw_Server_API/app/core/Web_Scraping/safe_regex.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_*.py
python -m bandit -r \
  Helper_Scripts/web_scraping_phase4_fixtures.py \
  tldw_Server_API/app/core/Web_Scraping/content \
  tldw_Server_API/app/core/Web_Scraping/selectors \
  tldw_Server_API/app/core/Web_Scraping/safe_regex.py \
  -f json -o /tmp/bandit_web_scraping_phase4a.json
git diff --check origin/dev...
```

Expected: all commands exit 0 and Bandit has no new medium/high findings.

- [ ] **Step 2: Review, finalize task, rebase, and merge 4A**

Request independent spec-compliance and code-quality reviews. Address findings,
rerun gates, record exact results in the 4A child, set it `Done`, rebase on latest
`origin/dev`, and merge 4A before creating the 4B branch.

## Phase 4B: Extraction Package

### Task 6: Create Extraction Facade, Dependencies, Caches, And Throttles

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/dependencies.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/caches.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/throttles.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py`

- [ ] **Step 1: Start 4B from merged 4A and create its child task**

```bash
git fetch origin
git rebase origin/dev
```

Create `Phase 4B canonical Web_Scraping extraction` under `TASK-12989`.

- [ ] **Step 2: Write failing facade/dependency/cache tests**

Bind the public names, no-new-to-legacy import rule, copied cache reads/writes,
cache stats keys, success-only schema caching, and default dependencies created
at call time.

```python
@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    validate_selector_rules: Callable[..., dict[str, Any]]
    extract_schema_fields: Callable[..., dict[str, Any]]
    perform_chat_api_call: Callable[..., Any]
    increment_counter: Callable[..., None]
    observe_histogram: Callable[..., None]
    log_counter: Callable[..., None]
    perf_counter: Callable[[], float]
    wall_time: Callable[[], float]
    sleep: Callable[[float], None]
    cancellation_checkpoint: Callable[[], None]
```

- [ ] **Step 3: Verify RED**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py
```

Expected: extraction package is missing.

- [ ] **Step 4: Implement package foundations**

Create `build_default_dependencies()` with lazy imports. Keep schema, cluster,
LLM provider, last-call, and strategy-limit state in focused cache/throttle
owners. Copy mutable values on cache read and write. `get_extraction_cache_stats`
must include canonical selector cache keys. Move the cache/throttle state now,
route the still-legacy pipeline through the canonical owners, and re-export the
public cache functions from `Article_Extractor_Lib.py`; do not keep parallel
cache state during later strategy moves.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py \
  tldw_Server_API/tests/WebScraping/test_extraction_caches.py
git add tldw_Server_API/app/core/Web_Scraping/extraction \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py
git commit -m "refactor(web-scraping): establish extraction package foundations"
```

### Task 7: Move JSON-LD And Regex Strategies

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/jsonld.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/regex.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/extraction/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py`

- [ ] **Step 1: Add canonical/legacy identity and fixture tests**

```python
def test_legacy_jsonld_and_regex_exports_are_canonical() -> None:
    assert legacy.extract_jsonld_entities is extraction.extract_jsonld_entities
    assert legacy.extract_regex_entities is extraction.extract_regex_entities
```

Bind JSON-LD graphs/references/microdata, PII catalog limits, overlap, IP checks,
Luhn filtering, masking precedence, and sanitized deterministic errors.

- [ ] **Step 2: Move implementations and remove legacy bodies**

Keep the trusted static PII catalog on stdlib `re`. Generated/configured patterns
continue through `safe_regex`. Strategy modules may import only lower leaves,
their dependency bundle, and local extraction support.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/WebScraping/test_regex_catalog.py \
  tldw_Server_API/tests/WebScraping/test_pii_masking.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py
git add tldw_Server_API/app/core/Web_Scraping/extraction \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py
git commit -m "refactor(web-scraping): move structured extraction strategies"
```

### Task 8: Move Cluster Strategy

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/cluster.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/extraction/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_properties.py`

- [ ] **Step 1: Add cluster parity and cache property tests**

Cover hierarchy, greedy fallback, tags, thresholds, bounded eviction, concurrent
reads, copy isolation, and stable `cluster_error` values. Hypothesis must assert
cache size never exceeds its configured maximum.

- [ ] **Step 2: Move the cluster implementation**

Move tokenization, embeddings, cosine similarity, assignment, tag, and result
assembly helpers as one cohesive strategy. Store and return vector copies.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/WebScraping/test_clustering_fallback.py \
  tldw_Server_API/tests/WebScraping/test_clustering_hierarchical.py \
  tldw_Server_API/tests/WebScraping/test_clustering_tags.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_properties.py
git add tldw_Server_API/app/core/Web_Scraping/extraction \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_properties.py
git commit -m "refactor(web-scraping): move cluster extraction strategy"
```

### Task 9: Move LLM Extraction And Rule Generation

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/llm.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/schema.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/observability.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/policy/probe.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/extraction/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py`

- [ ] **Step 1: Add LLM parity and sanitization tests**

Bind provider resolution, schema/regex generation, chunking, JSON parsing, usage,
retry, throttle, strict mode, and existing safe precondition codes. Inject an
exception containing a credential-bearing URL and assert public fields, traces,
and captured logs contain only `provider_error`, exception class, bounded stage,
and sanitized host.

- [ ] **Step 2: Move LLM and schema strategy code**

All provider calls go through `ExtractionDependencies.perform_chat_api_call`.
Check cooperative cancellation before provider dispatch, between retries, and
before the next chunk. Do not include provider exception text in `llm_error`,
generator `error`, trace detail, or logs.

Create the neutral `observability.py` leaf by moving the bounded host sanitizer
from `policy/probe.py`; expose `sanitized_host(url) -> str` and bounded stage/code
helpers without importing extraction, orchestration, policy, preflight, or legacy
wrappers. Update `policy/probe.py` in the same commit to import that canonical
sanitizer, and make extraction failure logging use the public leaf.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/WebScraping/test_llm_extraction.py \
  tldw_Server_API/tests/WebScraping/test_llm_throttling.py \
  tldw_Server_API/tests/WebScraping/test_regex_pattern_generation.py \
  tldw_Server_API/tests/WebScraping/test_schema_llm_generation.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py
git add tldw_Server_API/app/core/Web_Scraping/extraction \
  tldw_Server_API/app/core/Web_Scraping/observability.py \
  tldw_Server_API/app/core/Web_Scraping/policy/probe.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_failures.py
git commit -m "refactor(web-scraping): move llm extraction strategies"
```

### Task 10: Move Trafilatura, Pipeline, And Enrichment

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/strategies/trafilatura.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/enrichment.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/extraction/pipeline.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/extraction/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py`

- [ ] **Step 1: Add pipeline RED tests**

Cover aliases, duplicates, unknown entries, LLM disabled, `None`, explicit empty,
unknown-only orders, default non-terminal regex, explicit terminal regex, regex
retained on later success/final failure, JSON-LD summary carry-forward, result
copying, enhanced/direct Trafilatura separation, and exact traces/order.

```python
def test_default_regex_enriches_but_does_not_terminate(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "extract_jsonld_entities", failed_jsonld)
    monkeypatch.setattr(pipeline, "extract_regex_entities", matched_regex)
    monkeypatch.setattr(pipeline, "extract_llm_entities", failed_llm)
    monkeypatch.setattr(pipeline, "extract_cluster_entities", failed_cluster)
    monkeypatch.setattr(pipeline, "extract_with_trafilatura", article_result)

    result = pipeline.extract_article_with_pipeline(
        HTML_WITH_EMAIL_AND_ARTICLE,
        URL,
        strategy_order=None,
    )
    assert result["extraction_successful"] is True
    assert result["extraction_strategy"] == "trafilatura"
    assert result["regex_matches"]

def test_explicit_empty_order_preserves_regex_terminal_semantics() -> None:
    result = extract_article_with_pipeline(HTML_WITH_EMAIL_ONLY, URL, strategy_order=[])
    assert result["extraction_strategy"] == "regex"
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py
```

Expected: default regex behavior fails under the predecessor semantics.

- [ ] **Step 3: Implement canonical pipeline**

Keep public `extract_article_with_pipeline` and
`extract_article_data_from_html` signatures exact. Public functions build
default dependencies at call time and delegate to an internal dependency-aware
runner. Copy each strategy result before adding trace, summary, enrichment, or
cache fields. Only literal `strategy_order is None` enables non-terminal regex.

- [ ] **Step 4: Remove moved legacy implementation and re-export facade**

```python
from .extraction import (
    DEFAULT_EXTRACTION_STRATEGY_ORDER,
    clear_extraction_caches,
    extract_article_data_from_html,
    extract_article_with_pipeline,
    extract_cluster_entities,
    extract_jsonld_entities,
    extract_llm_entities,
    extract_regex_entities,
    generate_regex_pattern_from_llm,
    generate_schema_rules_from_llm,
    get_extraction_cache_stats,
)
```

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py \
  tldw_Server_API/tests/WebScraping/test_extraction_pipeline_router.py \
  tldw_Server_API/tests/WebScraping/test_extraction_metrics.py \
  tldw_Server_API/tests/WebScraping/test_extraction_observability.py \
  tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py
git add tldw_Server_API/app/core/Web_Scraping/extraction \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py
git commit -m "refactor(web-scraping): move canonical extraction pipeline"
```

### Task 11: Migrate Extraction Consumers And Sanitize Moved Observability

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/handlers.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py`

- [ ] **Step 1: Add consumer and observability tests**

Assert handlers and enhanced shared extraction import canonical packages; the
crawl-bound `scrape_article_async(context, ...)` uses canonical extraction but
retains ownership and behavior; moved metric labels contain no `url`, `base_url`,
host, raw error, or replacement high-cardinality value.

- [ ] **Step 2: Migrate imports without collapsing enhanced behavior**

```python
from tldw_Server_API.app.core.Web_Scraping.content import convert_html_to_markdown
from tldw_Server_API.app.core.Web_Scraping.extraction import (
    extract_article_data_from_html,
    extract_article_with_pipeline,
)
```

Enhanced JSON-output Trafilatura, tables, DOM fallback, retries, traces, queue,
and jobs remain unchanged.

- [ ] **Step 3: Remove sensitive moved labels and raw exception logging**

Preserve metric names and low-cardinality labels such as `strategy`, `status`,
`backend`, `outcome`, `reason`, and `cache`. Replace raw errors in traces/logs
with stable codes and exception class only. Add an AST assertion that new
extraction modules never place `asyncio.CancelledError` in a recoverable
exception tuple.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_handlers.py \
  tldw_Server_API/tests/Web_Scraping/test_enhanced_web_scraping_guards.py \
  tldw_Server_API/tests/Web_Scraping/test_phase3_enhanced_preflight_facade.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py \
  tldw_Server_API/tests/WebScraping/test_extraction_observability.py
git add tldw_Server_API/app/core/Web_Scraping/handlers.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_architecture.py
git commit -m "refactor(web-scraping): migrate extraction consumers"
```

### Task 12: Complete Phase 4B Gates And Merge

**Files:** Phase 4B files and its Backlog child task.

- [ ] **Step 1: Run complete extraction matrix**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_*.py \
  tldw_Server_API/tests/WebScraping/test_clustering_*.py \
  tldw_Server_API/tests/WebScraping/test_extraction_*.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/WebScraping/test_llm_*.py \
  tldw_Server_API/tests/WebScraping/test_pii_masking.py \
  tldw_Server_API/tests/WebScraping/test_regex_*.py \
  tldw_Server_API/tests/WebScraping/test_schema_*.py
python -m compileall -q tldw_Server_API/app/core/Web_Scraping/extraction
python -m ruff check tldw_Server_API/app/core/Web_Scraping/extraction tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_*.py
python -m black --check tldw_Server_API/app/core/Web_Scraping/extraction tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_*.py
python -m bandit -r tldw_Server_API/app/core/Web_Scraping/extraction -f json -o /tmp/bandit_web_scraping_phase4b.json
git diff --check origin/dev...
```

- [ ] **Step 2: Review, finalize task, rebase, and merge 4B**

Require independent spec and quality reviews. Confirm `Article_Extractor_Lib.py`
contains no moved extraction implementation, finalize the 4B child, and merge
before starting 4C.

## Phase 4C: Governed Article Orchestration

### Task 13: Add Optional Bounded Simple HTTP Fetch

**Files:**

- Modify: `tldw_Server_API/app/core/http_client.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/requests.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py`
- Create: `tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py`

- [ ] **Step 1: Start 4C from merged 4B and create its child task**

```bash
git fetch origin
git rebase origin/dev
```

Create `Phase 4C governed Web_Scraping article orchestration` under `TASK-12989`.

- [ ] **Step 2: Add failing runtime and backend tests**

Bind `FetchRequest.max_response_bytes: int | None = None`, positive normalization,
unchanged callers when omitted, exact-bound success, over-bound failure before a
mapping is returned, compressed-response rejection, redirects, cleanup, and both
simple httpx/curl backends.

```python
request = FetchRequest(url=URL)
assert request.max_response_bytes is None

with pytest.raises(ValueError, match="^Response exceeds max_response_bytes limit$"):
    fetch(URL, backend="httpx", max_response_bytes=5)
```

- [ ] **Step 3: Verify RED**

```bash
python -m pytest -q \
  tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
```

Expected: `FetchRequest` and simple fetch do not support the bound.

- [ ] **Step 4: Implement bounded accumulation**

Use identity encoding and streaming iteration only when the bound is non-`None`.
Share one helper across httpx/curl:

```python
def _read_bounded_chunks(chunks: Iterable[bytes], max_response_bytes: int) -> bytes:
    body = bytearray()
    for chunk in chunks:
        if len(body) + len(chunk) > max_response_bytes:
            raise ValueError("Response exceeds max_response_bytes limit")
        body.extend(chunk)
    return bytes(body)
```

Close every response/session on success and failure. A selected backend that
cannot stream a non-`None` bound fails before unbounded dispatch. `DefaultFetchClient`
forwards the bound; requests with `None` keep the predecessor path.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py \
  tldw_Server_API/tests/http_client/test_http_client.py \
  tldw_Server_API/tests/Web_Scraping/test_http_client_fetch.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
git add tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/app/core/Web_Scraping/runtime/requests.py \
  tldw_Server_API/app/core/Web_Scraping/runtime/fetch.py \
  tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
git commit -m "feat(http): bound simple fetch response bodies"
```

### Task 14: Define Article Plans, Limits, Profiles, And Stable Failures

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/__init__.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_models.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py`

- [ ] **Step 1: Add normalization and compatibility tests**

Cover defaults 16,777,216 and 67,108,864; absent/malformed/boolean/zero/negative
fallback; positive values; immutable request snapshots; browser compatibility
fields; and stable failure dictionaries.

- [ ] **Step 2: Implement immutable models**

```python
@dataclass(frozen=True, slots=True)
class ArticleLimits:
    max_article_bytes: int = 16_777_216
    max_browser_transfer_bytes: int = 67_108_864

@dataclass(frozen=True, slots=True)
class DirectBrowserProfile:
    user_agent: str
    custom_cookies: tuple[Mapping[str, Any], ...]
    retries: int
    timeout_ms: int
    stealth_enabled: bool
    stealth_wait_ms: int
    viewport_width: int = 1280
    viewport_height: int = 720

class ArticleFailure(Exception):
    def __init__(self, code: str, stage: str) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage
```

`ArticlePlan` snapshots routing/extraction settings and normalized limits. It
stores lightweight headers/cookies/proxies separately from direct-browser fields,
so plan headers/cookies/proxies cannot leak into Playwright by accidental reuse.
Copy and recursively freeze caller cookie dictionaries when constructing the
direct-browser profile.

- [ ] **Step 3: Add config defaults**

```ini
web_scraper_max_article_bytes = 16777216
web_scraper_max_browser_transfer_bytes = 67108864
```

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py
git add tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_models.py
git commit -m "feat(web-scraping): define bounded article request profiles"
```

### Task 15: Add Guarded Direct-Browser Egress Routing

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/runtime/browser.py`
- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py`

- [ ] **Step 1: Add route capability and egress RED tests**

Cover target, redirect, subresource, service-worker block, WebSocket conversion,
fresh decisions, deny/guard-error abort, route/WebSocket capability absence,
installation before navigation, cancellation, cleanup, and non-empty
`resolved_ips` treated as URL validation rather than transport pinning.

- [ ] **Step 2: Extend only required runtime protocols**

Add protocol members for Chromium CDP session creation and WebSocket route hooks;
do not import Playwright concrete classes into `runtime`.

- [ ] **Step 3: Implement guarded browser ownership**

Reuse `ProbeEgressGuard` and the Phase 3 guarded-browser routing pattern. The
article adapter must create Chromium with `service_workers="block"`, install HTTP
and WebSocket routes before `page.goto`, and continue routes without adding
headers. Capability failure raises `ArticleFailure("browser_error", "capability")`.

```python
async def _decision_allowed(self, url: str) -> bool:
    decision = await self._egress_guard.decide(
        url,
        context=replace(self._context, stage="fetch"),
    )
    return bool(decision.allowed)
```

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py \
  tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py \
  tldw_Server_API/tests/Web_Scraping/test_phase3_probe_egress.py
git add tldw_Server_API/app/core/Web_Scraping/runtime/browser.py \
  tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py
git commit -m "feat(web-scraping): guard direct browser egress"
```

### Task 16: Enforce Browser Transfer And Rendered-HTML Limits

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py`
- Create: `tldw_Server_API/tests/WebScraping/integration/test_phase4_article_browser_smoke.py`

- [ ] **Step 1: Add transfer/render capability RED tests**

Cover CDP absence, `Network.dataReceived.encodedDataLength`, text and binary
WebSocket frames, exact limit, over-limit page stop, browser-side oversized HTML,
doctype preservation, caller cookies, ignored plan fields, stealth, retries,
waits, viewport, launch mode, and cleanup. No limit failure may retry through an
alternate backend.

- [ ] **Step 2: Install accounting before navigation**

Use a Chromium CDP session with `Network.enable`. Sum HTTP encoded lengths and
WebSocket payload bytes; decode opcode-2 base64 payloads before counting. Set a
latched over-limit state once and close/stop the page. Fail closed when CDP event
registration is unavailable.

- [ ] **Step 3: Serialize and measure in the browser**

```javascript
(maxBytes) => {
  const doctype = document.doctype
    ? new XMLSerializer().serializeToString(document.doctype) + "\n"
    : "";
  const html = doctype + document.documentElement.outerHTML;
  const size = new TextEncoder().encode(html).length;
  return size <= maxBytes ? { ok: true, html } : { ok: false, size };
}
```

Return `response_too_large` without transferring the HTML string when `ok` is
false. Characterize the expression against `page.content()` in the local smoke
test before accepting the serialization.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py
python -m pytest -q tldw_Server_API/tests/WebScraping/integration/test_phase4_article_browser_smoke.py
git add tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py \
  tldw_Server_API/tests/WebScraping/integration/test_phase4_article_browser_smoke.py
git commit -m "feat(web-scraping): bound browser article acquisition"
```

Expected: unit tests pass; smoke passes when Playwright Chromium is available or
skips with the established dependency marker only.

### Task 17: Implement Extraction Executor Generations

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/executor.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_executor.py`

- [ ] **Step 1: Add deterministic concurrency RED tests**

Cover default four workers, positive `EXTRACTOR_MAX_WORKERS`, immutable generation
snapshot, multiple event loops, saturation, queued/running cancellation, discarded
late results, submit failure, permit idempotence, reload, concurrent shutdown,
stale-generation retry, terminal shutdown, PID mismatch, and after-fork reset.

- [ ] **Step 2: Define generation ownership**

```python
@dataclass(slots=True)
class ExecutorGeneration:
    pid: int
    generation_id: int
    worker_count: int
    executor: ThreadPoolExecutor
    permits: BoundedSemaphore
    closed: bool = False

class ManagerState(str, Enum):
    RUNNING = "running"
    RELOADING = "reloading"
    SHUTDOWN = "shutdown"
```

- [ ] **Step 3: Implement locked admission and exact-generation release**

Acquire permits non-blockingly with async backoff from 10ms capped at 100ms.
After acquisition, recheck current PID/generation/open state under the manager
lock and retry stale admission. Submit while replacement is excluded. Attach one
idempotent callback to release only the issuing generation. On caller cancellation,
set the cooperative token, stop awaiting immediately, and discard late results.

- [ ] **Step 4: Implement lifecycle**

Reload closes old admission, drains outside the lock, and installs a new captured
worker count. Shutdown atomically detaches and is terminal. PID mismatch and the
registered after-fork hook discard inherited state without waiting for parent
threads. Register normal process shutdown with `atexit` and expose explicit
reload/test-reset functions. Only explicit test reset/process startup leaves
shutdown.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_executor.py
git add tldw_Server_API/app/core/Web_Scraping/orchestration/executor.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_executor.py
git commit -m "feat(web-scraping): bound extraction executor lifecycle"
```

### Task 18: Move Standard Async Article Orchestration

**Files:**

- Create: `tldw_Server_API/app/core/Web_Scraping/orchestration/article.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py`

- [ ] **Step 1: Add deterministic orchestration RED tests**

Cover plan fallback, policy denial/error, preflight disabled/enabled, successful
advice, explicit precedence, preflight fail-open, payload attachment to success
and final extraction failure, HTTP success/no-content/JS/error fallback, all size
limits, no fallback after oversized acquisition, cancellation at every await,
heartbeat during extraction, result copying, and stable failure mappings. Add an
AST assertion that orchestration modules never place `asyncio.CancelledError` in
a recoverable exception tuple.

- [ ] **Step 2: Define injectable orchestration dependencies**

```python
@dataclass(frozen=True, slots=True)
class ArticleDependencies:
    load_config: Callable[[], Mapping[str, Any]]
    resolve_plan: Callable[[str, Mapping[str, Any]], ArticlePlan]
    evaluate_target: Callable[..., Awaitable[Any]]
    run_preflight: Callable[..., Awaitable[Any]]
    apply_preflight_advice: Callable[..., tuple[str, str, Any]]
    fetch_client: FetchClient
    browser: GuardedArticleBrowser
    executor: ExtractionExecutorManager
    extract: Callable[..., dict[str, Any]]
```

Public `scrape_article` builds defaults at call time and delegates to a private
dependency-aware runner; its signature remains exact.

- [ ] **Step 3: Implement the approved data flow**

Resolve one immutable plan, evaluate target, run optional preflight, apply only
successful advice when selection is auto, fetch with `max_response_bytes`, run
JS detection, use guarded browser when eligible, offload extraction through the
manager, copy/enrich, attach optional successful preflight payload, and return.
`asyncio.CancelledError` is always re-raised.

- [ ] **Step 4: Map and sanitize failures**

Map policy/fetch/browser/limit/extraction boundaries to the exact codes in the
design. Logs include only exception class, stable code, bounded stage, and the
existing sanitized host helper. Metrics keep bounded labels only.

- [ ] **Step 5: Replace legacy body with direct export**

`Article_Extractor_Lib.scrape_article` must be the canonical coroutine object,
not a copied wrapper. Keep crawl/ingestion/sitemap functions in the legacy file.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py \
  tldw_Server_API/tests/Web_Scraping/test_phase3_article_preflight_facade.py \
  tldw_Server_API/tests/Web_Scraping/test_router_backend_selection.py \
  tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py
git add tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py
git commit -m "refactor(web-scraping): move governed article orchestration"
```

### Task 19: Move Blocking And Raw-Browser Sync Compatibility Entry Points

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/__init__.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_compatibility.py`

- [ ] **Step 1: Add active-loop and compatibility RED tests**

Both sync entry points must raise the identical error before config, policy,
metrics, browser, or network fakes are called:

```python
ACTIVE_EVENT_LOOP_ERROR = (
    "Synchronous article scraping cannot run while an event loop is active in this thread"
)
```

Bind blocking robots false, 30-second timeout, cookie reduction, status 200,
content conversion, result fields, and the raw sync `{url,title,content,
extraction_successful}` shape.

- [ ] **Step 2: Implement one guard before side effects**

```python
def _reject_active_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(ACTIVE_EVENT_LOOP_ERROR)
```

- [ ] **Step 3: Implement blocking compatibility profile**

After the guard, use the orchestrator's explicit blocking profile and a fresh
local event loop only where async policy/preflight/browser composition is needed.
Do not use the removed per-analyzer background-loop bridge.

- [ ] **Step 4: Implement governed raw-browser adapter**

After the same guard, call the async guarded article browser through
`asyncio.run`, enforce both limits, and translate success/failure back to the
historical raw HTML dictionary without extraction fields.

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_compatibility.py \
  tldw_Server_API/tests/WebScraping/test_legacy_sync_helpers.py \
  tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py
git add tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_compatibility.py
git commit -m "refactor(web-scraping): govern synchronous article entry points"
```

### Task 20: Migrate Article Consumers And Fix Service Keyword

**Files:**

- Modify: internal consumer files listed in the Phase 4C file map.
- Modify: `tldw_Server_API/app/services/web_scraping_service.py`
- Create: `tldw_Server_API/tests/Web_Scraping/test_phase4_consumer_imports.py`

- [ ] **Step 1: Add import and real-signature RED tests**

Assert Collections, Evaluations, RAG, Watchlists, Workflows, WebSearch, handlers,
enhanced fallback, and services import canonical content/extraction/orchestration
where available. Call the fallback `scrape_and_summarize_multiple` against its
real signature and assert `system_message` is passed.

- [ ] **Step 2: Migrate safe internal imports**

```python
from tldw_Server_API.app.core.Web_Scraping.content import ContentMetadataHandler
from tldw_Server_API.app.core.Web_Scraping.extraction import extract_article_data_from_html
from tldw_Server_API.app.core.Web_Scraping.orchestration import (
    scrape_article,
    scrape_article_blocking,
)
```

Keep legacy imports only for deferred Phase 5 mixed imports or external
compatibility tests. Update monkeypatches to the canonical owning module.

- [ ] **Step 3: Correct the verified keyword mismatch**

```python
result_list = await scrape_and_summarize_multiple(
    urls=url_input,
    custom_prompt_arg=custom_prompt,
    api_name=api_name,
    api_key=api_key,
    keywords=keywords,
    custom_article_titles=custom_titles,
    system_message=system_prompt,
    summarize_checkbox=summarize_checkbox,
    custom_cookies=custom_cookies,
    temperature=temperature,
    allow_llm_extraction=summarize_checkbox,
)
```

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase4_consumer_imports.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  tldw_Server_API/tests/Collections \
  tldw_Server_API/tests/Evaluations \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py \
  tldw_Server_API/tests/Workflows/adapters \
  tldw_Server_API/tests/Web_Scraping/test_websearch_*.py
git add tldw_Server_API/app/core/Collections/reading_service.py \
  tldw_Server_API/app/core/Evaluations/article_extraction_benchmark.py \
  tldw_Server_API/app/core/RAG/rag_service/research_agent.py \
  tldw_Server_API/app/core/Watchlists/fetchers.py \
  tldw_Server_API/app/core/Workflows/adapters/rag/search.py \
  tldw_Server_API/app/core/WebSearch/Web_Search.py \
  tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_consumer_imports.py
git commit -m "refactor(web-scraping): migrate canonical article consumers"
```

### Task 21: Complete Phase 4C Gates And Merge

**Files:** Phase 4C files and its Backlog child task.

- [x] **Step 1: Run orchestration and security gates**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping/test_phase1_contracts.py \
  tldw_Server_API/tests/Web_Scraping/test_phase2_*.py \
  tldw_Server_API/tests/Web_Scraping/test_phase3_*.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_article_*.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_executor.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_consumer_imports.py
python -m compileall -q \
  tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/app/core/Web_Scraping/runtime \
  tldw_Server_API/app/core/http_client.py
python -m ruff check \
  tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/app/core/Web_Scraping/runtime \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_*.py
python -m black --check \
  tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/tests/Web_Scraping/test_phase4_*.py
python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/app/core/Web_Scraping/runtime \
  tldw_Server_API/app/core/http_client.py \
  -f json -o /tmp/bandit_web_scraping_phase4c.json
git diff --check origin/dev...
```

- [ ] **Step 2: Review, finalize task, rebase, and merge 4C**

Require independent spec/security and quality reviews. Confirm preflight remains
optional and in the standard flow; every browser destination is freshly guarded;
both sync guards precede side effects; and no moved observability contains a
full URL/raw error. Finalize and merge 4C before 4D.

Implementation, final gates, independent reviews, and rebase onto
`origin/dev` are complete at `1f9cb427c4`. Merge remains unchecked until the
required human-written Change summary is supplied and the delivery-unit PR is
created and merged.

## Phase 4D: Final Integration And Gates

### Task 22: Regenerate Inventory, Update Docs, And Certify Phase 4

**Files:**

- Modify: `Docs/Design/WebScraping.md`
- Modify: `Docs/Design/WebScraping_Refactor_Import_Inventory.md`
- Modify: `Docs/Design/web_scraping_refactor_import_inventory.json`
- Modify: `tldw_Server_API/app/core/Web_Scraping/README.md`
- Modify: remaining safe internal import consumers identified by inventory.
- Modify: the Phase 4D Backlog child task.

- [ ] **Step 1: Start 4D from merged 4C and create its child task**

```bash
git fetch origin
git rebase origin/dev
```

Create `Phase 4D Web_Scraping integration and certification` under `TASK-12989`.

- [ ] **Step 2: Regenerate import inventory**

```bash
python Helper_Scripts/web_scraping_refactor_inventory.py \
  --root . \
  --json Docs/Design/web_scraping_refactor_import_inventory.json \
  --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
python -m pytest -q tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

Expected: artifacts match byte-for-byte regeneration. Review every remaining
legacy import; migrate only canonical Phase 4 responsibilities and retain mixed
Phase 5 or explicit compatibility consumers.

- [ ] **Step 3: Update architecture documentation**

Document canonical package ownership, optional preflight sequence, limit keys,
executor lifecycle, direct-browser compatibility profile, approved behavior
changes, deferred Phase 5/6/7 scope, and fixture regeneration command. Do not
describe URL route validation as resolved-IP pinning.

- [ ] **Step 4: Run broad Web_Scraping tests**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Web_Scraping \
  tldw_Server_API/tests/WebScraping
```

Expected: all non-environmental tests pass. Playwright/local-service tests may
skip only under their established markers.

- [ ] **Step 5: Run cross-consumer matrix**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Collections \
  tldw_Server_API/tests/Evaluations \
  tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py \
  tldw_Server_API/tests/Watchlists \
  tldw_Server_API/tests/Workflows/adapters
```

Expected: selected cross-consumer suites pass. Any unrelated broad failure must
reproduce on exact `origin/dev` under the same environment before being recorded
as baseline debt.

- [ ] **Step 6: Run final static and security gates**

```bash
python -m compileall -q tldw_Server_API/app/core/Web_Scraping tldw_Server_API/app/core/http_client.py
python -m ruff check \
  tldw_Server_API/app/core/Web_Scraping \
  tldw_Server_API/app/core/http_client.py \
  tldw_Server_API/tests/Web_Scraping \
  tldw_Server_API/tests/WebScraping
python -m black --check \
  Helper_Scripts/web_scraping_phase4_fixtures.py \
  tldw_Server_API/app/core/Web_Scraping/content \
  tldw_Server_API/app/core/Web_Scraping/extraction \
  tldw_Server_API/app/core/Web_Scraping/orchestration \
  tldw_Server_API/app/core/Web_Scraping/selectors \
  tldw_Server_API/app/core/Web_Scraping/observability.py \
  tldw_Server_API/app/core/Web_Scraping/safe_regex.py \
  tldw_Server_API/tests/Web_Scraping/test_phase4_*.py
python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping \
  tldw_Server_API/app/core/http_client.py \
  -f json -o /tmp/bandit_web_scraping_phase4_final.json
git diff --check origin/dev...
```

Expected: compilation, Ruff, Black, and diff checks pass; Bandit reports no new
medium/high findings in touched production code.

Black is intentionally restricted to newly created Phase 4 files. The existing
`Article_Extractor_Lib.py`, `http_client.py`, and runtime files are not Black-clean
on the predecessor commit; formatting those whole files would create unrelated
review churn. Ruff and `git diff --check` remain mandatory for every modified file.

- [ ] **Step 7: Run Python 3.10 compatibility when available**

```bash
if command -v python3.10 >/dev/null 2>&1; then
  python3.10 -m pytest -q \
    tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_contracts.py \
    tldw_Server_API/tests/Web_Scraping/test_phase4_article_compatibility.py
fi
```

Expected: focused contracts pass under Python 3.10, or the missing interpreter is
recorded and CI remains the required Python 3.10 gate.

- [ ] **Step 8: Independent final review and completion**

Request whole-phase spec, security, concurrency, and compatibility review.
Address all findings, rerun affected and final gates, record exact results and
skips, finalize the 4D child, and verify all four delivery-unit pull requests are
merged.

- [ ] **Step 9: Commit final integration records**

```bash
git add Docs/Design/WebScraping.md \
  Docs/Design/WebScraping_Refactor_Import_Inventory.md \
  Docs/Design/web_scraping_refactor_import_inventory.json \
  tldw_Server_API/app/core/Web_Scraping/README.md
git commit -m "docs(web-scraping): certify phase 4 extraction refactor"
```

## Spec Coverage Review

- Shared leaves and dependency direction: Tasks 2-5.
- Deterministic predecessor fixtures and eleven-change allowlist: Tasks 1, 10,
  18, 19, and 22.
- Canonical extraction strategies, caches, traces, and result fields: Tasks 6-12.
- Default regex enrichment and explicit order compatibility: Task 10.
- Enhanced/direct behavior separation: Tasks 10-12.
- Governed preflight and advice precedence: Task 18.
- Independent HTTP/browser egress decisions: Tasks 13, 15, 16, and 18.
- HTTP body, browser transfer, and rendered-HTML limits: Tasks 13, 14, 16, and 18.
- Browser compatibility table and ignored plan fields: Tasks 14-16.
- Executor generations, cancellation, reload, shutdown, and fork: Task 17.
- Stable public failures and sanitized observability: Tasks 2, 4, 9, 11, 14,
  18, and 21.
- Both synchronous active-loop guards and raw result shape: Task 19.
- Internal consumer migration and service keyword correction: Task 20.
- Inventory, documentation, security, broad regression, and independent review:
  Tasks 5, 12, 21, and 22.
- Phase 5-7 exclusions: delivery rules, scope, Tasks 20 and 22.

## Plan Self-Review

- **Spec coverage:** Every architecture, behavior, failure, concurrency,
  compatibility, testing, and completion requirement maps to at least one task
  above.
- **Unfinished-marker scan:** Every production edit has a test-first step, an
  implementation shape, an exact verification command, an expected result, and
  a commit/review boundary.
- **Type consistency:** `SafeRegexLimits`, `SafeRegexResult`,
  `ExtractionDependencies`, `ArticleLimits`, `DirectBrowserProfile`,
  `ArticleFailure`, `ArticleDependencies`, `ExecutorGeneration`, and
  `ExtractionExecutorManager` are defined before downstream tasks use them.
- **Scope:** One implementation plan is appropriate because 4A-4D are sequential
  layers of the same canonical article path. The merge-unit rule prevents the
  plan from becoming one unreviewable change and keeps Phase 5-7 work excluded.
