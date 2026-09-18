# UAT cycle 2 implementation plan

> For agentic workers: use the systematic-debugging and test-driven-development skills for each bounded repair, with independent domain work under dispatching-parallel-agents and review before integration.

**Goal:** Repair UAT-020–045, then repeat the full fresh single-user/multi-user workflow UAT and continue the cycle for any new issue.
**Architecture:** Keep existing shared frontend/backend boundaries and current authorization policies. Fix the producer of inconsistent state or false-success outcomes, with behavior-level regression tests and real-runtime controls.
**Tech stack:** Next.js/shared React UI, Vitest/Playwright, FastAPI, pytest, SQLite, real llama.cpp.
**Spec:** `Docs/Design/2026-09-15-uat-cycle-2-repairs.md`.

**Integration checkpoint (2026-09-15 06:00 UTC):** Reviewed repairs are committed in `88ecef4a56` (Chat), `2a76d5ac81` (Notes), `6bbe8bda64` (Prompts), `4566479348` (Flashcards), `7639aca719` (ownership), `aa1c542cb7` (setup/defaults), and `a53aa33e58` (Media). Combined parent checks: 60 backend tests; Bandit zero findings across 24 production Python files; unchanged 90 baseline TypeScript diagnostics; five existing lint errors reproduced in Media test fixtures. Targeted live Prompt save succeeds but emitted new UAT-046 notification warning; repair in progress. Playwright session loss interrupted further targeted browser checks; test runtime databases are intact. Stage 5 cannot be marked complete until the full fresh matrix passes.

**Targeted follow-up and fresh-run handoff (2026-09-15):** UAT-046–055 have targeted live verification. UAT-052 corrects the earlier browser-cache persistence false positive: independent server reads prove original-conversation persistence, Note origin, card content and review state. UAT-048 now keeps the other offline Notes tab at its app URL with only the native signed-out screen and `Signed out | tldw` title; reconnect automatically opens login. Bob sees only his own Notes, Alice's queued draft syncs with POST201, and Bob's independent GET of that new note returns404. Offline remote revocation is not claimed. [Retained targeted evidence](output/playwright/cycle2-repair-verification-2026-09-15/README.md) distinguishes the intermediate UAT-048 failure from its final pass. Parent verification records 166 shared, 155 web, and 59 app/login tests passing (overlapping suites, not additive), no new lint findings, the same 90 baseline TypeScript diagnostic signatures, and zero Bandit findings across 24 touched production Python files. The typecheck does not pass outright. Product revision `d40e17dc81` is frozen for fresh cycle 3: single-user API18200/WebUI18280 is initialized and opened; multi-user is initialized with admin bootstrap only. Full fresh-workflow acceptance remains pending. Run WebUIs sequentially: concurrent generated caches caused ENOSPC; retiring only the multi cache freed space and affected research-run polling recovered200.

## Stage 1: Trace and repair account state
**Goal:** Close UAT-034/041/042/045 under TASK-13260.5.
**Success Criteria:** Notes metadata cannot cross server/account boundaries; source labels and titles are correct; manual-key disconnect/re-entry works.
**Tests:** Extend Notes recent-state behavior tests with two authorities and delayed hydration; Notes origin display; connection settings credential clearing; route/title reset. Run existing Notes stage12/stage46 and relevant auth regressions.
**Files:** `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`, Notes tests and provenance components, shared settings/auth components; web title boundary if required.
**Status:** In Progress
- [x] Reproduce the global-setting leak in a behavior test and observe failure.
- [x] Implement authority-scoped persistence and stale-state guards, then verify A/B/reload behavior.
- [x] Trace and test accurate note origin, title reset and manual-key disconnect.
- [x] Review, validate and commit this unit with its Backlog record.
- [x] Finish UAT048: offline Settings logout clears local identity without an overlay; the other Notes tab stays at its app URL with private content unmounted and a native signed-out screen. Reconnect opens login, Bob remains isolated, and Alice recovers/syncs her queued draft. The [final targeted report](output/playwright/cycle2-repair-verification-2026-09-15/uat048-boundary-final-report.md) includes an independent Bob404 control; the intermediate failure remains retained.

## Stage 2: Repair Flashcards and Chat
**Goal:** Close UAT-022/024/026/031/032/035/037/038/039 under TASK-13260.6/.7.
**Success Criteria:** Source-supported cards pass without accepting swapped/unsupported answers; derived cards have usable question/answer; visible Chat context/persistence matches actual requests and tracked multi-user replies succeed.
**Tests:** Flashcard verification boundary including the exact UAT source, unsupported additions and swapped answers; GeneratePanel error UI; Message save/editor validation; ReviewTab initial/completed state; tracked backend dispatch, greeting retry identifiers and active-store character context.
**Files:** Flashcards GeneratePanel/error taxonomy/ReviewTab, Playground Message save handlers, chat knowledge schema/save endpoint, flashcards verification units; Chat state/hooks and tracked completion endpoint with associated tests. Agents coordinate any shared Message/store file before edits.
**Status:** In Progress
- [x] Each domain records exact failing regression commands before production changes.
- [x] Apply minimal fixes described in the design; adapt named shared workflow steps for the explicit card editor without skip branches.
- [x] Run focused frontend/backend regressions and touched Python Bandit; independently review integration.
- [x] Commit each reviewed domain with its Backlog task.

## Stage 3: Repair Media lifecycle and retrieval
**Goal:** Close UAT-027/028/029/030/033/036/040/043/044 under TASK-13260.8/.9.
**Success Criteria:** Ingestion produces real content/analysis or honest failure; source navigation and deletion recover correctly; owned ordinary-user source retrieval succeeds with foreign-data denial preserved.
**Tests:** Scraper terminal status controls, no persistence of rejected fetches, plaintext analysis error handling, empty-library deep link, deletion URL clearing, Trash timestamp, source URL routing, self delete capability, actual owned/foreign Media search and QA contexts.
**Files:** `Plaintext_Files.py`, `enhanced_web_scraping.py`, ingestion service/result mapping, Media view/navigation/Trash, Knowledge QA source actions; `media/search` service/database ownership boundary identified by the positive/negative control trace.
**Status:** In Progress
- [x] Confirm failing boundary tests and real runtime evidence before each repair.
- [x] Repair ingestion/navigation/capability behavior while preserving permissions and remote restrictions.
- [x] Trace normal owned Media search through its database query; reproduce and fix without weakening ownership/ACL filters.
- [x] Run focused tests/Bandit/lint and review each unit before commit.

## Stage 4: Repair setup and prompt transitions
**Goal:** Close UAT-020/021/023/025 under TASK-13260.10/.11.
**Success Criteria:** Readiness includes truthful actionable reasons; endpoint model discovery precedes model selection; QA uses a usable configured default or clearly requires setup; saving a prompt leaves clean navigation state.
**Tests:** Empty setup readiness, valid local endpoint with blank model, configured default Chat→QA request, prompt save/back/new-route behavior and actual prompt reuse.
**Files:** Existing first-run setup schema/readiness and UnifiedSetupWizard/provider validation; Knowledge QA provider-default resolution; prompt create/edit route and form tests.
**Status:** In Progress
- [x] Trace each failing UI/API transition against working adjacent flows.
- [x] Write and observe failing behavior tests, then implement the minimal state/configuration correction.
- [x] Run targeted tests, security/lint checks, review and commit both units.

## Stage 5: Review and repeat fresh UAT
**Goal:** Verify all repairs together, then run the complete original workflow matrix from new profiles in both modes.
**Success Criteria:** No bugs, failures or UX issues encountered across every required step; incomplete/blocked/skipped coverage cannot qualify as completion.
**Tests:** Combined touched regressions; independent review; fresh setup/auth/admin accounts, file and URL ingest/search/actual grounded Chat/QA/citations, Notes→Flashcards, prompt application, Chat-derived Notes/cards, review/re-analysis/delete/restore, cross-account read/write and browser-metadata isolation.
**Status:** In Progress
- [x] Resolve integration review findings; record parent verification of 166 shared, 155 web, and 59 app/login tests (overlapping suites), no new lint findings, unchanged 90 baseline TypeScript diagnostics, and zero Bandit findings across 24 touched production Python files.
- [x] Freeze the repaired product revision at `d40e17dc81` and initialize fresh isolated single/multi data/config profiles. Single-user API18200/WebUI18280 is initialized and opened; multi-user has admin bootstrap only.
- [ ] Open the fresh multi-user WebUI/browser profile and complete its setup before running its full workflow matrix.
- [ ] Execute every named workflow with real inference, inspect actual payloads/provenance and save credential-free evidence.
- [ ] Record every issue in the running tracker; continue repair/review/UAT if any issue remains.
- [ ] Only after a complete issue-free pass, close the Backlog acceptance and active goal; remove only this completed plan.
