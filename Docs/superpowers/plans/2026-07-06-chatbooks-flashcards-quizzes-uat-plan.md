# Chatbooks Flashcards Quizzes UAT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run live WebUI UAT for Chatbooks, Flashcards, and Quizzes, patch confirmed user-facing root causes, and open a PR against `dev`.

**Architecture:** Reuse existing route components, page objects, and tier-2 e2e specs first. Add only focused tests for confirmed fixes.

**Tech Stack:** Next.js WebUI, React, existing Playwright/Vitest tests, FastAPI backend, live llama.cpp-compatible server at `127.0.0.1:9099`.

---

### Task 1: Live UAT Baseline

**Files:**
- Inspect: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`
- Inspect: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- Inspect: `apps/tldw-frontend/e2e/workflows/tier-2-features/quiz.spec.ts`

- [x] Run the existing tier-2 specs against the live backend.
- [x] Exercise desktop and mobile rendered states for `/chatbooks-playground`, `/flashcards`, and `/quiz`.
- [x] Record only reproducible visual or functional issues.

### Task 2: Root-Cause Patches

**Files:**
- Modify only the route/component files that own confirmed issues.
- Test with the smallest existing or new focused test.

- [x] Trace each issue to the shared owner.
- [x] Add focused regression coverage where practical.
- [x] Apply the shortest root-cause fix.
- [x] Re-run focused tests.

### Task 3: Final PR

**Files:**
- Update: `backlog/tasks/task-12904 - Full-UAT-for-Chatbooks-Flashcards-and-Quizzes-WebUI-pages.md`

- [x] Re-run live e2e/UAT.
- [x] Save screenshot evidence outside the repo.
- [x] Run Bandit only if Python files are touched.
- [x] Commit, push, and open a PR against `dev`.

### UAT Findings And Fixes

- Chatbooks import UAT was blocked by a stale page-object locator after the dropzone copy changed from `.zip chatbook` to `.zip or .chatbook archive`.
- Flashcards search/cleanup returned 500s for plain-text search terms containing punctuation such as hyphens and `front:` because SQLite FTS interpreted user text as operators/columns instead of literal tokens.
- Flashcards review and Quiz live flows hit shared `core.default` ResourceGovernor throttles too quickly for normal page use.
- Flashcards Manage first-entry states hid the primary search/deck controls, making deck-scoped creation and filtering harder to discover.
- Flashcards deck-scoped card creation did not preselect the active Manage deck.
- Flashcards `Test with Quiz` stayed disabled after selecting a review deck when there was no existing quiz handoff.
- Flashcards E2E edit flow targeted the stale `Edit Flashcard` dialog label instead of the current `Edit Card` drawer.

### Verification

- Live tier-2 UAT: `TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://localhost:8080 TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY npx playwright test e2e/workflows/tier-2-features/chatbooks.spec.ts e2e/workflows/tier-2-features/flashcards.spec.ts e2e/workflows/tier-2-features/quiz.spec.ts --reporter=line --workers=1` passed 32/32.
- Screenshot smoke check: desktop/mobile screenshots for Chatbooks, Flashcards, and Quiz saved outside the repo at `/private/tmp/tldw-study-pages-uat-shots-20260706`; the check failed on page errors, visible crash text, and backend 4xx/5xx responses.
- Focused UI tests: `bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.first-time.test.tsx src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage33.link-aware-delete-warning.test.tsx` passed 61/61.
- Focused backend tests: `.venv/bin/python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_db_assets.py::test_list_flashcards_search_accepts_hyphenated_plain_text tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py::test_sqlite_normalization_quotes_plain_tokens_with_punctuation tldw_Server_API/tests/Resource_Governance/test_policy_loader_route_map_db_store.py::test_db_policy_loader_includes_route_map_from_file tldw_Server_API/tests/Resource_Governance/test_slowapi_decorated_routes_mapped.py::test_rg_route_map_covers_rate_limited_paths -q` passed 4/4.
- Bandit: touched Python implementation file had no findings; test-file assert warnings were isolated to pytest code and the non-assert scan passed with `-s B101`.
