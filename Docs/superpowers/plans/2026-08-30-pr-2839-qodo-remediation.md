# PR 2839 Qodo Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every validated Qodo finding on PR #2839 without weakening governed HTTP retrieval or changing the human-authored change summary.

**Architecture:** Keep `BrowserTransportDecision` as the single admission contract. Resolve effective auth and transport configuration through existing helpers, then consult the decision before every production Playwright startup and dispatch. Preserve public scraping APIs and replace new private-helper tests with public observable-contract tests.

**Tech Stack:** Python 3.10+, pytest, Loguru, Playwright adapters, Backlog.md.

**Spec:** `Docs/superpowers/plans/2026-08-27-browser-transport-safety-gate.md`

## Global Constraints

- `TASK-13139.13` tracks all repository changes.
- Browser transport remains fail closed for strict, multi-user, explicitly disabled, malformed, or unattested profiles.
- Governed credentialless HTTP extraction remains available when browser transport is denied.
- Public denials and logs must not expose URLs, configuration values, credentials, cookies, headers, proxy URIs, or raw exception text.
- No new dependency, proxy, authenticated browser state, or persistence mechanism is introduced.

---

### Task 1: Correct configuration and decision diagnostics

**Files:**
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/browser_transport.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_browser_transport.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_outbound_policy.py`

**Interfaces:**
- Consumes: `load_auth_config(config_parser, env)` and the existing comprehensive config parser.
- Produces: `default_browser_transport_decision(...) -> BrowserTransportDecision` with canonical auth fallback, explicit invalid-value rejection, and sanitized diagnostics.

- [x] **Step 1:** Write failing tests for absent versus invalid auth configuration, malformed transport-mode classification, and sanitized config-failure logging.
- [x] **Step 2:** Run the focused tests and confirm failures identify the current incorrect branches.
- [x] **Step 3:** Reuse `load_auth_config`, preserve malformed transport input for `decide_browser_transport`, and add fixed safe Loguru events.
- [x] **Step 4:** Run focused tests and confirm both denial and single-user compatibility controls pass.

### Task 2: Gate every production Playwright path

**Files:**
- Modify: `tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/orchestration/article_browser.py`
- Modify: `tldw_Server_API/app/core/Web_Scraping/preflight/adapters/browser.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_phase3_enhanced_preflight_facade.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_phase4_extraction_pipeline.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_phase3_preflight_browser.py`
- Test: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_browser.py`

**Interfaces:**
- Consumes: `default_browser_transport_decision()` and `BrowserTransportDecision.to_capability_metadata()`.
- Produces: denial before `async_playwright().start()`, `chromium.launch()`, browser-context creation, budget reservation, or browser navigation.

- [x] **Step 1:** Write failing tests proving disabled decisions prevent enhanced-scraper startup, enhanced Playwright dispatch, and legacy recursive-crawl launch.
- [x] **Step 2:** Run focused tests and confirm Playwright doubles are reached before the fix.
- [x] **Step 3:** Add the shared decision at narrow lifecycle boundaries and retain HTTP fallback where the public API already supports it.
- [x] **Step 4:** Add sanitized provider-failure logging tests for guarded article and preflight adapters, then implement fixed safe events.
- [x] **Step 5:** Run browser-path focused tests and confirm no direct production Playwright sink bypasses admission.

### Task 3: Satisfy review rules and verify the patch

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/web_retrieval_quality.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_article_extraction_benchmark.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_browser_transport.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_web_retrieval_quality_baseline.py`
- Modify: `tldw_Server_API/tests/Web_Scraping/test_phase4_article_orchestration.py`
- Modify: `backlog/tasks/task-13139.13 - Remediate-Qodo-findings-on-PR-2839.md`

**Interfaces:**
- Consumes: public `scrape_article`, `scrape_article_blocking`, and `scrape_article_sync` contracts.
- Produces: classified/documented tests, documented production helpers, verification evidence, and review responses.

- [x] **Step 1:** Add concise docstrings to newly added helpers and test functions, plus exactly one accepted classification marker per changed test.
- [x] **Step 2:** Replace direct `_run_article` and `_raw_failure_result` assertions with public API assertions using controlled dependency patching at the public boundary.
- [x] **Step 3:** Run focused tests, the affected Wave 0 suite, compile/lint checks, and `git diff --check`.
- [x] **Step 4:** Run Bandit over touched production paths and inspect the final diff for a surviving browser sink or compatibility regression.
- [x] **Step 5:** Update `TASK-13139.13`, commit and push, request Qodo follow-up, and merge only after current-dev, review, and CI gates pass.
