# Quick Ingest PR 2709 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every actionable PR #2709 review finding without adding a new public ingestion parameter, then validate repeat Quick Ingest behavior through real PDF, web-link, and YouTube Shorts user workflows.

**Architecture:** Preserve the two existing web API contracts and propagate their existing analysis booleans into the internal extraction pipeline as an LLM permission gate. Keep the remaining fixes local to their owning boundaries: Watchpack configuration, restored-session polling, persistence classification/logging, dependency diagnostics, and Quick Ingest UI regression coverage.

**Tech Stack:** FastAPI/Pydantic, Python asyncio, yt-dlp, SQLite repository layer, Next.js/Webpack, React/Ant Design, TypeScript, Vitest, Playwright, pytest, Bandit.

---

## File Map

- `apps/tldw-frontend/next.config.mjs`: preserve existing Webpack ignore entries and append absolute backend runtime globs.
- `apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts`: behavioral Watchpack ignore coverage.
- `apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts`: bounded retries for transient direct-job reads.
- `apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts`: transient/permanent reattachment contracts.
- `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`: one terminal conference-item status helper.
- `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`: existing analysis declaration and result classification coverage.
- `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx`: retain stable AntD modal styles.
- `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`: exact modal style assertions.
- `apps/extension/tests/e2e/live-ux-workflows.spec.ts`: real waiting assertions in changed workflow helpers.
- `apps/extension/tests/e2e/quick-ingest-workflows.spec.ts`: two submissions in one mounted app session with console/page-error capture.
- `apps/extension/tests/e2e/utils/quick-ingest-options.ts`: shared option-toggle helper for the changed Quick Ingest browser tests.
- `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`: restore the established default order and apply an internal request LLM permission gate.
- `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`: carry the analysis intent through queued, sitemap, URL-level, and recursive scraping.
- `tldw_Server_API/app/services/enhanced_web_scraping_service.py`: pass the existing API declaration, classify exact duplicate repository results, and redact logged URLs.
- `tldw_Server_API/app/services/web_scraping_service.py`: forward `perform_analysis`/`summarize_checkbox` to extraction without changing either request schema.
- `tldw_Server_API/tests/WebScraping/test_extraction_pipeline_router.py`: default/custom strategy permission-gate tests.
- `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`: service propagation coverage for analysis intent.
- `tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py`: real duplicate repository, negative, mixed, and log-redaction tests.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/yt_dlp_support.py`: one-shot, non-blocking installed-version diagnostic.
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py`: invoke the diagnostic at yt-dlp request boundaries.
- `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_yt_dlp_support.py`: stale/current/invalid version behavior.
- `Docs/Code_Documentation/Ingestion_Pipeline_Video.md`: existing-environment update guidance.
- `backlog/tasks/task-12946 - Fix-Quick-Ingest-repeat-ingestion-and-queued-job-recovery.md`: maintained only through Backlog MCP/CLI.

### Task 1: Preserve Webpack Watch Ignores

**Files:**
- Modify: `apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts`
- Modify: `apps/tldw-frontend/next.config.mjs`

- [x] **Step 1: Write the failing behavioral test**

Load `next.config.mjs` with each Webpack-valid input shape. For a standalone
`/node_modules/` expression, assert its matching semantics survive in a single
schema-valid expression that also matches the backend roots. For string and
string-array inputs, assert the original strings survive unchanged and the
absolute globs are appended. Resolve representative runtime files from the
repository workspace and assert the final ignore value matches:

```ts
expect(webpackConfig.watchOptions.ignored).toContain(existingRegex)
expect(webpackConfig.watchOptions.ignored).toContain("**/.next/**")
expect(backendPatterns).toEqual(expect.arrayContaining([
  expect.stringMatching(/\/Databases\/\*\*$/),
  expect.stringMatching(/\/tldw_Server_API\/Logs\/\*\*$/),
]))
```

- [x] **Step 2: Run the focused test outside the sandbox and verify RED**

Run: `bunx vitest run apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts`

Expected: FAIL because the current filter drops the regular expression and the appended patterns are relative.

- [x] **Step 3: Implement the minimal normalization**

Preserve the semantics of every Webpack-supported ignore shape while keeping the
final shape schema-valid (`RegExp`, string, or string array). Discard only blank
strings and construct absolute globs from the repository workspace root with
backslashes normalized to `/`. Do not produce a mixed RegExp/string array,
which Watchpack treats as an invalid string array.

- [x] **Step 4: Run the focused test outside the sandbox and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit the isolated config fix**

```bash
git add apps/tldw-frontend/next.config.mjs apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts
git commit -m "fix: preserve webui dev watch ignores"
```

### Task 2: Make Direct-Job Reattachment Tolerate Transient Reads

**Files:**
- Modify: `apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts`

- [x] **Step 1: Write failing transient and permanent tests**

Add one test where `bgRequest` rejects once and then returns `processing`, one where it returns HTTP 503 then `completed`, and retain a 404 test that must interrupt immediately. Use fake timers or inject a zero-delay test seam so retries are deterministic.

- [x] **Step 2: Run the focused test outside the sandbox and verify RED**

Run: `bunx vitest run apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts`

Expected: transient cases return `interrupted` before the fix.

- [x] **Step 3: Add one bounded status-read helper**

Retry network exceptions, 408, 429, and 5xx responses at most three attempts using the existing fixed-delay style. Return permanent 401/403/404 and malformed successful responses without retry. Keep `preferDirect: true` on every attempt.

- [x] **Step 4: Run focused reattachment and session tests outside the sandbox**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/store/__tests__/quick-ingest-session.test.ts
```

Expected: PASS.

- [x] **Step 5: Commit the reattachment fix**

```bash
git add apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts apps/packages/ui/src/services/__tests__/quick-ingest-session-reattach.test.ts
git commit -m "fix: retry transient quick ingest job reads"
```

### Task 3: Correct Duplicate Persistence, Status, and URL Logging

**Files:**
- Modify: `tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py`
- Modify: `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- Modify: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/quick-ingest-batch.ts`

- [x] **Step 1: Write failing backend boundary tests**

Add coverage for:

1. Persist the same successful article twice through the temporary real Media DB. The second call must return `status == "duplicate"`, no `media_ids`, `stored_articles == 0`, and one duplicate/skipped article.
2. An extraction failure such as `"deduplicate worker unavailable"` without `is_duplicate is True` or `error_code == "duplicate_content"` remains an error.
3. A batch containing one exact duplicate and one unrelated failure returns errors and does not collapse to all-duplicate status.
4. A URL containing `?token=secret-value` never writes `secret-value` to captured logs.

- [x] **Step 2: Run the backend test outside the sandbox and verify RED**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py -q`

Expected: FAIL because a repository duplicate ID is counted as stored and broad substring matching classifies unrelated failures as duplicates.

- [x] **Step 3: Implement exact duplicate classification**

Treat extraction output as duplicate only for `is_duplicate is True` or `error_code == "duplicate_content"`. After `add_media_with_keywords`, inspect its repository-owned message and classify only the two exact overwrite-disabled forms as duplicate; do not append that returned existing ID to `media_ids`. Preserve canonicalization, overwrite, and insert/update success behavior. Use `redact_url_for_log` for every touched article URL log.

- [x] **Step 4: Write the failing frontend terminal-status helper test**

Extend batch tests to prove duplicate wins over completed, error wins over completed, and success remains completed for conference collection patches.

- [x] **Step 5: Replace the repeated ternary with one local helper**

```ts
const terminalConferenceStatus = (skipped: boolean, error?: string) =>
  skipped ? "skipped_existing" : error ? "failed" : "completed"
```

- [x] **Step 6: Run backend and frontend focused tests outside the sandbox**

Run the Step 2 command and:

`bunx vitest run apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

Expected: PASS.

- [x] **Step 7: Commit duplicate and logging corrections**

```bash
git add tldw_Server_API/app/services/enhanced_web_scraping_service.py tldw_Server_API/tests/Services/test_enhanced_webscraping_persist.py apps/packages/ui/src/services/tldw/quick-ingest-batch.ts apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
git commit -m "fix: classify persisted web duplicates exactly"
```

### Task 4: Honor Existing API Analysis Declarations

**Files:**
- Modify: `tldw_Server_API/tests/WebScraping/test_extraction_pipeline_router.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Modify: `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- Modify: `tldw_Server_API/app/services/web_scraping_service.py`
- Modify: `apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts`

- [x] **Step 1: Write failing extraction permission tests**

Assert the global `DEFAULT_EXTRACTION_STRATEGY_ORDER` includes `llm`. For both the default order and a custom router order containing `llm`, assert `allow_llm_extraction=False` removes only `llm` while preserving relative order; assert `True` leaves the order intact.

- [x] **Step 2: Write failing API/service propagation tests**

Patch the scraper boundary and verify:

- `/process-web-scraping` with `summarize_checkbox=False` reaches extraction with `allow_llm_extraction=False`;
- the same path with `True` reaches it with `True`;
- `/ingest-web-content` maps `perform_analysis` identically;
- Quick Ingest still sends only its existing `summarize_checkbox` declaration and no `strategy_order`/`extraction_strategy_order` property.

- [x] **Step 3: Run the focused backend/frontend tests outside the sandbox and verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/WebScraping/test_extraction_pipeline_router.py \
  tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py -q
bunx vitest run apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
```

Expected: default-order assertion and propagation assertions fail.

- [x] **Step 4: Implement the internal permission gate**

Restore `"llm"` in `DEFAULT_EXTRACTION_STRATEGY_ORDER`. Add an internal optional `allow_llm_extraction: bool = True` argument at scraper functions only, never to Pydantic request schemas. Resolve the effective order once per article:

```python
order, unknown = _normalize_strategy_order(strategy_order)
if not allow_llm_extraction:
    order = [strategy for strategy in order if strategy != "llm"]
```

Carry the existing `summarize_checkbox`/`perform_analysis` boolean through individual, sitemap, URL-level, recursive, and queued-job metadata paths. Keep direct non-API scraper consumers backward compatible with the default `True`.

- [x] **Step 5: Run focused and adjacent web-scraping tests outside the sandbox**

Run the Step 3 commands plus:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/WebScraping/test_phase1_compatibility_contracts.py \
  tldw_Server_API/tests/WebScraping/test_llm_extraction.py \
  tldw_Server_API/tests/WebScraping/integration/test_llm_extraction_pipeline.py -q
```

Expected: PASS.

- [x] **Step 6: Commit the declaration wiring fix**

```bash
git add tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py tldw_Server_API/app/services/enhanced_web_scraping_service.py tldw_Server_API/app/services/web_scraping_service.py tldw_Server_API/tests/WebScraping/test_extraction_pipeline_router.py tldw_Server_API/tests/Web_Scraping/test_crawl_config_precedence.py apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
git commit -m "fix: honor web analysis intent during extraction"
```

### Task 5: Warn Once for Stale yt-dlp

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/yt_dlp_support.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_yt_dlp_support.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py`
- Modify: `Docs/Code_Documentation/Ingestion_Pipeline_Video.md`

- [x] **Step 1: Write failing version diagnostic tests**

Patch installed-version lookup and logging to prove versions below `2026.7.4` emit exactly one warning containing `pip install -U "yt-dlp>=2026.7.4"`; current versions emit none; missing or malformed metadata never raises.

- [x] **Step 2: Run the focused test outside the sandbox and verify RED**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_yt_dlp_support.py -q`

Expected: FAIL because the helper does not exist.

- [x] **Step 3: Implement a minimal one-shot helper**

Use `importlib.metadata.version("yt-dlp")` and `packaging.version.Version`. Keep module state only for suppressing duplicate warnings. Return normally for every lookup or parse failure. Call it immediately before video yt-dlp request boundaries, after outbound URL validation so unrelated/blocked requests do not create misleading diagnostics.

- [x] **Step 4: Run focused video tests outside the sandbox**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_yt_dlp_support.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_video_boundary_regressions.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_youtube_audio_downloads.py -q
```

Expected: PASS.

- [x] **Step 5: Commit the diagnostic**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/yt_dlp_support.py tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_yt_dlp_support.py Docs/Code_Documentation/Ingestion_Pipeline_Video.md
git commit -m "fix: warn when yt-dlp is below supported version"
```

### Task 6: Strengthen Modal and Browser Regression Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx`
- Modify: `apps/extension/tests/e2e/live-ux-workflows.spec.ts`
- Modify: `apps/extension/tests/e2e/quick-ingest-workflows.spec.ts`
- Create: `apps/extension/tests/e2e/utils/quick-ingest-options.ts`

- [x] **Step 1: Strengthen the modal unit assertion**

Assert the production AntD modal receives:

```ts
expect(modalProps.styles.body).toEqual(expect.objectContaining({
  padding: "0 16px 16px",
  maxHeight: "calc(100vh - 180px)",
  overflowY: "auto",
}))
```

- [x] **Step 2: Replace non-waiting Playwright visibility checks**

For changed transition controls, use `await expect(locator).toBeVisible({ timeout: 5_000 })` before interaction. Use count/immediate visibility only for genuinely optional controls, with a short documented branch. Reuse one option-toggle helper for identical selector/state transitions.

- [x] **Step 3: Add the real repeated-submission browser scenario**

In one mounted WebUI session, open Quick Ingest, submit a persisted URL, wait for terminal success, reset/reopen without reloading the app, submit the same URL again, and assert terminal skipped/existing status. Capture `pageerror` and console errors before the first submission and assert no message contains `Maximum update depth exceeded`.

- [x] **Step 4: Run focused UI unit tests outside the sandbox**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx \
  apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run focused browser tests outside the sandbox**

From `apps/extension`, with the real backend and WebUI already running, run:

`TLDW_LIVE_E2E=1 npx playwright test tests/e2e/quick-ingest-workflows.spec.ts tests/e2e/live-ux-workflows.spec.ts --reporter=line`

The command must use `apps/extension/playwright.config.ts` and its global setup. Expected: both files execute rather than reporting the live workflow as skipped, and all executed tests PASS.

Expected: PASS against the explicitly configured local WebUI/backend test environment.

- [ ] **Step 6: Commit the UI regression coverage**

```bash
git add apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx apps/extension/tests/e2e/live-ux-workflows.spec.ts apps/extension/tests/e2e/quick-ingest-workflows.spec.ts apps/extension/tests/e2e/utils/quick-ingest-options.ts
git commit -m "test: cover repeated quick ingest browser flow"
```

### Task 7: Full Verification, UAT, and PR Closeout

**Files:**
- Modify through Backlog CLI/MCP: `backlog/tasks/task-12946 - Fix-Quick-Ingest-repeat-ingestion-and-queued-job-recovery.md`
- Modify: `Docs/superpowers/plans/2026-07-10-quick-ingest-pr-2709-review-remediation-plan.md` status checkboxes during execution.

- [x] **Step 1: Run all affected automated suites outside the sandbox**

Run the focused commands from Tasks 1-6, then the affected frontend Quick Ingest suite and backend web/video suites. Record exact pass/fail totals in TASK-12946.

- [x] **Step 2: Run Bandit outside the sandbox**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py \
  tldw_Server_API/app/services/enhanced_web_scraping_service.py \
  tldw_Server_API/app/services/web_scraping_service.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/yt_dlp_support.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py \
  -f json -o /tmp/bandit_task_12946.json
```

Expected: no new findings in changed code.

- [x] **Step 3: Perform full user acceptance testing outside the sandbox**

Start the real backend and WebUI. Through the browser, upload a PDF, ingest a reachable local link, ingest `https://www.youtube.com/shorts/6-rf_YXDpPg`, then repeat the link and YouTube submissions in the same mounted app session. Verify visible progress leaves queued/0%, each first submission reaches terminal success with stored media, repeats are visibly skipped/existing, and no maximum-depth console/page error occurs. Inspect corresponding backend job status/results and Media DB entries rather than relying only on UI toasts.

- [ ] **Step 4: Self-review the complete diff against current `origin/dev`**

Check behavior, compatibility, logging safety, exact duplicate boundaries, test quality, and accidental unrelated changes. Run `git diff --check` and inspect `git status --short` without touching the two unrelated untracked watchlist templates.

- [ ] **Step 5: Update Backlog and resolve PR comments**

Use official Backlog MCP/CLI to check acceptance criteria/DoD, replace stale notes with real line breaks, attach plan and verification evidence, and keep status accurate. Reply to and resolve every actionable PR review thread with the implementing commit/test evidence.

- [ ] **Step 6: Rebase, reverify changed surfaces, and push**

Fetch and rebase onto latest `origin/dev`; rerun any conflict-affected focused tests; push with `--force-with-lease`. Confirm PR checks and unresolved-thread count.

- [ ] **Step 7: Enforce the human change-summary merge gate**

Do not claim merge readiness until the human requester has supplied the required change summary explaining what changed and why. Record this as the only blocker if all technical work is complete.
