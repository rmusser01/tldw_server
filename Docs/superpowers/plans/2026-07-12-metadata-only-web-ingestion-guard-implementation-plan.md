# Metadata-Only Web Ingestion Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent web ingestion from storing metadata-only media records while retaining useful JSON-LD summaries when another strategy extracts the page body.

**Architecture:** Tighten the JSON-LD success contract at the shared extraction layer, carry a structured summary through pipeline finalization, and preserve it in the legacy multi-URL helper. Add the same small, explicit body check immediately before each existing persistence loop; reuse `ContentMetadataHandler.has_metadata()` and `strip_metadata()` rather than introducing a new abstraction or Wikipedia-specific behavior.

**Tech Stack:** Python 3.10+, BeautifulSoup/JSON-LD extraction, FastAPI services, pytest/pytest-asyncio, Black, Ruff, Bandit

**Spec:** `Docs/superpowers/specs/2026-07-12-metadata-only-web-ingestion-guard-design.md`

**Backlog:** `TASK-12111`

---

## File map

- `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`: define body-only JSON-LD success, carry a structured summary through later strategies, and stop the legacy multi-URL helper from erasing that summary.
- `tldw_Server_API/app/services/enhanced_web_scraping_service.py`: reject invalid extracted bodies before metadata formatting, chunking, or database writes.
- `tldw_Server_API/app/services/web_scraping_service.py`: apply the same validation in compatibility persistence and report skipped URLs.
- `tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py`: extraction and summary-preservation regressions.
- `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`: enhanced persistence mixed-batch and envelope-only regressions.
- `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`: compatibility persistence mixed-batch and envelope-only regressions.

No new production files, configuration, dependencies, strategy ordering, or Wikipedia-specific rules are needed.

## Stage 0: Record the static-analysis baseline

**Goal**: Capture pre-existing Black and Ruff findings before any implementation edit.

**Success Criteria**: Later verification can prove the task introduced no new formatting or lint findings without reformatting unrelated code in already-noncompliant files.

**Tests**: Normalized Black and Ruff baseline artifacts in `/tmp`.

**Status**: Not Started

### Task 0: Capture baseline findings before code changes

**Files:**

- Read only: all six Python files listed in the file map
- Create outside the repository: `/tmp/task_12111_base_commit.txt`
- Create outside the repository: `/tmp/black_task_12111_baseline.txt`
- Create outside the repository: `/tmp/ruff_task_12111_baseline.json`

- [ ] **Step 1: Record the implementation base commit**

Run before Task 1 makes any code or test edits:

```bash
git rev-parse HEAD > /tmp/task_12111_base_commit.txt
```

Expected: the file contains the commit that includes this approved plan but none of the implementation.

- [ ] **Step 2: Record the normalized Black baseline**

Run:

```bash
source .venv/bin/activate && python -m black --check \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  2>&1 | sed -n 's/^would reformat /would reformat /p' | sort > /tmp/black_task_12111_baseline.txt
```

Expected baseline: only `Article_Extractor_Lib.py` and `test_legacy_fallback_behavior.py` are listed. Do not format either whole file; that cleanup is outside TASK-12111.

- [ ] **Step 3: Record a location-independent Ruff baseline**

```bash
source .venv/bin/activate && python -m ruff check --no-cache --output-format=json \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  | jq 'map({code, filename, message, fix_message: .fix.message}) | sort_by(.filename, .code, .message)' \
  > /tmp/ruff_task_12111_baseline.json
```

Expected baseline: the existing `I001` and `F841` findings in `Article_Extractor_Lib.py`. Because locations are intentionally omitted, later line shifts do not create false differences.

- [ ] **Step 4: Confirm all baseline artifacts exist**

```bash
test -s /tmp/task_12111_base_commit.txt
test -s /tmp/black_task_12111_baseline.txt
test -s /tmp/ruff_task_12111_baseline.json
```

Expected: all commands exit 0. Preserve these files until Stage 4 verification.

## Stage 1: Correct the extraction contract

**Goal**: Make JSON-LD require body text while retaining structured summaries through later extraction and legacy no-summary processing.

**Success Criteria**: Description-only JSON-LD falls through; a later body result keeps the structured summary unless it has its own; optional legacy summarization does not replace the summary with `None`.

**Tests**: Focused JSON-LD and `scrape_and_summarize_multiple` unit regressions.

**Status**: Not Started

### Task 1: Require JSON-LD body content and carry its summary forward

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py:1022-1025`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py:1228`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py:2470-2667`
- Test: `tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py`

- [ ] **Step 1: Write failing extraction regressions**

Add tests with deterministic in-memory HTML and a stub fallback extractor:

```python
def _description_only_html() -> str:
    return """
    <html><head><script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Article",
      "headline": "Structured title",
      "description": "Structured summary"
    }
    </script></head><body><article>Visible page body</article></body></html>
    """


def test_jsonld_description_without_body_is_not_successful():
    result = extract_jsonld_entities(_description_only_html(), "https://example.com/article")

    assert result["extraction_successful"] is False
    assert result["content"] == ""
    assert result["summary"] == "Structured summary"


def test_pipeline_falls_through_and_retains_jsonld_summary():
    def fallback(_html, url):
        return {
            "url": url,
            "title": "Fallback title",
            "content": "Fallback page body",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        _description_only_html(),
        "https://example.com/article",
        strategy_order=["jsonld", "trafilatura"],
        fallback_extractor=fallback,
    )

    assert result["extraction_strategy"] == "trafilatura"
    assert result["content"] == "Fallback page body"
    assert result["summary"] == "Structured summary"


def test_pipeline_does_not_replace_later_nonempty_summary():
    def fallback(_html, url):
        return {
            "url": url,
            "title": "Fallback title",
            "content": "Fallback page body",
            "summary": "Fallback summary",
            "extraction_successful": True,
        }

    result = extract_article_with_pipeline(
        _description_only_html(),
        "https://example.com/article",
        strategy_order=["jsonld", "trafilatura"],
        fallback_extractor=fallback,
    )

    assert result["summary"] == "Fallback summary"
```

- [ ] **Step 2: Run the extraction regressions to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py::test_jsonld_description_without_body_is_not_successful \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py::test_pipeline_falls_through_and_retains_jsonld_summary \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py::test_pipeline_does_not_replace_later_nonempty_summary -v
```

Expected: all three tests fail because JSON-LD currently treats `summary` as extracted content and short-circuits.

- [ ] **Step 3: Implement the minimal extraction correction**

Replace the summary-aware predicate with a body-only predicate and update its only caller:

```python
def _jsonld_has_body_content(result: dict[str, Any]) -> bool:
    content = result.get("content")
    return isinstance(content, str) and bool(content.strip())
```

In `extract_jsonld_entities()`:

```python
result["extraction_successful"] = _jsonld_has_body_content(result)
```

In `extract_article_with_pipeline()`, initialize `jsonld_summary: Optional[str] = None` before `_finalize_result()`. Make finalization fill only a missing or whitespace-only summary:

```python
    jsonld_summary: Optional[str] = None

    def _finalize_result(
        result: dict[str, Any],
        *,
        strategy: Optional[str],
    ) -> dict[str, Any]:
        summary = result.get("summary")
        if jsonld_summary and (not isinstance(summary, str) or not summary.strip()):
            result["summary"] = jsonld_summary
        final = _attach_trace(result, trace, strategy, order)
        if _should_clear_caches("end"):
            clear_extraction_caches()
        return final
```

Immediately after JSON-LD extraction, remember only a non-whitespace string summary:

```python
            summary = result.get("summary")
            if isinstance(summary, str) and summary.strip():
                jsonld_summary = summary
```

- [ ] **Step 4: Run the focused extraction file to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py -v
```

Expected: all tests pass, including existing JSON-LD body short-circuit coverage.

- [ ] **Step 5: Commit the extraction contract**

```bash
git add tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py
git commit -m "fix: require body content for JSON-LD extraction"
```

### Task 2: Preserve a structured summary when legacy summarization is disabled

**Files:**

- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py:3176-3273`
- Test: `tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py`

- [ ] **Step 1: Write the failing legacy summary regression**

Import the module as `extractor_lib` and add:

```python
@pytest.mark.asyncio
async def test_multi_url_scrape_without_llm_keeps_structured_summary(monkeypatch):
    async def fake_scrape_article(url, custom_cookies=None):
        return {
            "url": url,
            "title": "Article",
            "content": "Page body",
            "summary": "Structured summary",
            "extraction_successful": True,
        }

    monkeypatch.setattr(extractor_lib, "scrape_article", fake_scrape_article)
    monkeypatch.setattr(extractor_lib, "RateLimiter", lambda: None)

    results = await extractor_lib.scrape_and_summarize_multiple(
        urls="https://example.com/article",
        custom_prompt_arg=None,
        api_name="",
        api_key=None,
        keywords="",
        custom_article_titles=None,
        summarize_checkbox=False,
    )

    assert results[0]["summary"] == "Structured summary"
```

- [ ] **Step 2: Run the regression to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py::test_multi_url_scrape_without_llm_keeps_structured_summary -v
```

Expected: FAIL because the legacy helper sets `article['summary'] = None`.

- [ ] **Step 3: Implement the minimal preservation rule**

Replace the unconditional no-summary assignment with the native dictionary operation:

```python
                else:
                    article.setdefault("summary", None)
```

Do not change the summarization-enabled branch; a generated summary remains authoritative.

- [ ] **Step 4: Run the focused extraction file to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit summary preservation**

```bash
git add tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py
git commit -m "fix: preserve structured web summaries"
```

## Stage 2: Guard enhanced persistence

**Goal**: Stop enhanced ingestion before metadata formatting or storage when extracted body text is invalid.

**Success Criteria**: Missing, non-string, whitespace-only, and recognized envelope-only bodies are URL-scoped errors; valid siblings persist; response status remains `persist-ok`.

**Tests**: Mixed-batch enhanced persistence regression using the existing fake database.

**Status**: Not Started

### Task 3: Validate body content in enhanced persistence

**Files:**

- Modify: `tldw_Server_API/app/services/enhanced_web_scraping_service.py:697-979`
- Test: `tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py`

- [ ] **Step 1: Write a failing mixed-batch persistence regression**

Add a test that uses the existing `_FakeDB`, `_MetricsStub`, and managed-database monkeypatch pattern. Supply one valid article plus these successful-but-invalid articles:

```python
invalid_articles = [
    {"url": "https://example.com/missing", "content": None, "extraction_successful": True},
    {"url": "https://example.com/non-string", "content": 42, "extraction_successful": True},
    {"url": "https://example.com/blank", "content": "  \n", "extraction_successful": True},
    {
        "url": "https://example.com/envelope",
        "content": '[METADATA]{"url":"https://example.com/envelope"}[/METADATA]\n  ',
        "extraction_successful": True,
    },
]
valid_article = {
    "url": "https://example.com/valid",
    "title": "Valid",
    "content": "Actual article body",
    "extraction_successful": True,
}
```

After `_store_persistent(..., perform_chunking=False, ...)`, assert:

```python
assert persisted["status"] == "persist-ok"
assert persisted["stored_articles"] == 1
assert persisted["media_ids"] == [1]
assert len(fake_db.calls) == 1
assert "Actual article body" in fake_db.calls[0]["content"]
assert persisted["errors"] == [
    "No extracted content: https://example.com/missing",
    "No extracted content: https://example.com/non-string",
    "No extracted content: https://example.com/blank",
    "No extracted content: https://example.com/envelope",
]
```

- [ ] **Step 2: Run the regression to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py::test_store_persistent_skips_articles_without_body_content -v
```

Expected: FAIL because invalid entries currently reach formatting/storage (or become generic storage failures).

- [ ] **Step 3: Add the pre-persistence body guard**

Immediately after the existing `extraction_successful` check and before `try`, metadata formatting, and chunking:

```python
                content_text = article.get("content")
                body_text = (
                    ContentMetadataHandler.strip_metadata(content_text)
                    if isinstance(content_text, str) and ContentMetadataHandler.has_metadata(content_text)
                    else content_text
                )
                if not isinstance(body_text, str) or not body_text.strip():
                    error_msg = f"No extracted content: {article.get('url', 'Unknown URL')}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    continue
```

Pass the already-validated `content_text` to `format_content_with_metadata()` instead of reading the article again. Keep `total_articles`, `stored_articles`, `media_ids`, `errors`, and `persist-ok` response behavior unchanged.

- [ ] **Step 4: Run enhanced persistence tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the enhanced guard**

```bash
git add tldw_Server_API/app/services/enhanced_web_scraping_service.py tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py
git commit -m "fix: skip empty enhanced web ingestion bodies"
```

## Stage 3: Guard compatibility persistence

**Goal**: Apply the same no-empty-body contract to the legacy compatibility database path.

**Success Criteria**: Invalid bodies never reach chunking or the repository; valid siblings persist; skipped URLs appear in an `errors` field without changing `persist-ok`.

**Tests**: Mixed-batch legacy fallback persistence regression using the existing repository fake.

**Status**: Not Started

### Task 4: Validate body content in legacy persistence

**Files:**

- Modify: `tldw_Server_API/app/services/web_scraping_service.py:25-36`
- Modify: `tldw_Server_API/app/services/web_scraping_service.py:465-575`
- Test: `tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py`

- [ ] **Step 1: Write a failing compatibility persistence regression**

Follow `test_fallback_persist_uses_media_repository_api`: force fallback, return one valid article and the same four invalid body variants from the fake `scrape_and_summarize_multiple`, patch the managed database/repository/path, and call `process_web_scraping_task(..., mode="persist", perform_chunking=False)`.

Assert:

```python
assert result["status"] == "persist-ok"
assert result["media_ids"] == [71]
assert result["total_articles"] == 5
assert len(fake_repo.calls) == 1
assert fake_repo.calls[0]["url"] == "https://example.com/valid"
assert result["errors"] == [
    "No extracted content: https://example.com/missing",
    "No extracted content: https://example.com/non-string",
    "No extracted content: https://example.com/blank",
    "No extracted content: https://example.com/envelope",
]
```

- [ ] **Step 2: Run the regression to verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py::test_fallback_persist_skips_articles_without_body_content -v
```

Expected: FAIL because the compatibility loop currently sends every article to the repository and has no `errors` response field.

- [ ] **Step 3: Add the compatibility body guard and error reporting**

Import the existing handler with the fallback scraping functions:

```python
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    ContentMetadataHandler,
    recursive_scrape,
    scrape_and_summarize_multiple,
    scrape_article,
    scrape_by_url_level,
    scrape_from_sitemap,
)
```

Initialize `errors: list[str] = []` beside `media_ids`, then use the same validation before chunk-option resolution:

```python
                        content_text = article.get("content")
                        body_text = (
                            ContentMetadataHandler.strip_metadata(content_text)
                            if isinstance(content_text, str) and ContentMetadataHandler.has_metadata(content_text)
                            else content_text
                        )
                        if not isinstance(body_text, str) or not body_text.strip():
                            error_msg = f"No extracted content: {article.get('url', 'Unknown URL')}"
                            logger.warning(error_msg)
                            errors.append(error_msg)
                            continue
```

Return the optional field without changing status:

```python
                    "errors": errors if errors else None,
```

- [ ] **Step 4: Run compatibility tests to verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  tldw_Server_API/tests/Web_Scraping/test_auto_chunking_web_ingest.py -v
```

Expected: all tests pass, including unchanged repository API and chunking behavior for valid content.

- [ ] **Step 5: Commit the compatibility guard**

```bash
git add tldw_Server_API/app/services/web_scraping_service.py tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py
git commit -m "fix: skip empty legacy web ingestion bodies"
```

## Stage 4: Verify and finalize

**Goal**: Demonstrate the complete fix is regression-safe, formatted, and free of new Bandit findings.

**Success Criteria**: Focused suites, formatting, lint, compilation, Bandit, and diff checks pass; TASK-12111 records evidence and is completed.

**Tests**: Combined focused pytest suites plus static/security checks.

**Status**: Not Started

### Task 5: Run final quality gates and close the tracked task

**Files:**

- Update through Backlog MCP: `backlog/tasks/task-12111 - Prevent-metadata-only-web-ingestion-records.md`
- Update: `Docs/superpowers/plans/2026-07-12-metadata-only-web-ingestion-guard-implementation-plan.md` stage statuses only while work progresses

- [ ] **Step 1: Run the complete focused regression set**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py \
  tldw_Server_API/tests/Web_Scraping/test_auto_chunking_web_ingest.py -v
```

Expected: all tests pass with no unexpected skips or failures.

- [ ] **Step 2: Run baseline-aware formatting and lint checks plus compilation**

```bash
source .venv/bin/activate && python -m black --check \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py
source .venv/bin/activate && python -m black --check \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  2>&1 | sed -n 's/^would reformat /would reformat /p' | sort > /tmp/black_task_12111_after.txt
diff -u /tmp/black_task_12111_baseline.txt /tmp/black_task_12111_after.txt
source .venv/bin/activate && python -m black --diff \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  > /tmp/black_task_12111_after.diff
source .venv/bin/activate && python -m ruff check --no-cache --output-format=json \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py \
  | jq 'map({code, filename, message, fix_message: .fix.message}) | sort_by(.filename, .code, .message)' \
  > /tmp/ruff_task_12111_after.json
diff -u /tmp/ruff_task_12111_baseline.json /tmp/ruff_task_12111_after.json
source .venv/bin/activate && python -m compileall -q \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py
```

Expected: Black exits 0 for the four baseline-clean files; both normalized baseline diffs are empty; compilation exits 0. Compare `/tmp/black_task_12111_after.diff` with `git diff -U0` and confirm no Black proposal overlaps task-added lines. Do not run whole-file formatting on the two baseline-dirty files or fix the pre-existing `I001`/`F841` findings under this task.

Use the captured base commit for that overlap check so earlier implementation commits remain visible:

```bash
BASE_COMMIT="$(cat /tmp/task_12111_base_commit.txt)"
git diff -U0 "$BASE_COMMIT"..HEAD -- \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py
```

- [ ] **Step 3: Run Bandit over touched production scope**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  -f json -o /tmp/bandit_task_12111.json
```

Expected: exit 0 with no new findings. Inspect `/tmp/bandit_task_12111.json` and record the result in TASK-12111.

- [ ] **Step 4: Review the final diff and request code review**

Use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`. Check only task-owned files:

```bash
BASE_COMMIT="$(cat /tmp/task_12111_base_commit.txt)"
git diff --check "$BASE_COMMIT"..HEAD
git diff "$BASE_COMMIT"..HEAD -- \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/tests/WebScraping/test_jsonld_extraction.py \
  tldw_Server_API/tests/Web_Scraping/test_persistence_crawl_metadata.py \
  tldw_Server_API/tests/Web_Scraping/test_legacy_fallback_behavior.py
```

Expected: no whitespace errors; diff matches the spec with no unrelated changes.

- [ ] **Step 5: Finalize TASK-12111 through Backlog MCP and commit records**

Record exact pytest/Black/Ruff/compile/Bandit results, touched files, known skips, and final summary. Mark acceptance criteria and Definition of Done only when supported by evidence, then set the task to Done/complete per the Backlog finalization workflow.

```bash
git add Docs/superpowers/plans/2026-07-12-metadata-only-web-ingestion-guard-implementation-plan.md "backlog/tasks/task-12111 - Prevent-metadata-only-web-ingestion-records.md"
git commit -m "docs: finalize metadata-only ingestion task"
```

Do not stage or modify unrelated worktree changes.
