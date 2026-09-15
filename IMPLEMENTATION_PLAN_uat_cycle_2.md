# UAT cycle 2 implementation plan

> For agentic workers: use the systematic-debugging and test-driven-development skills for each bounded repair, with independent domain work under dispatching-parallel-agents and review before integration.

**Goal:** Repair UAT-020–045, then repeat the full fresh single-user/multi-user workflow UAT and continue the cycle for any new issue.
**Architecture:** Keep existing shared frontend/backend boundaries and current authorization policies. Fix the producer of inconsistent state or false-success outcomes, with behavior-level regression tests and real-runtime controls.
**Tech stack:** Next.js/shared React UI, Vitest/Playwright, FastAPI, pytest, SQLite, real llama.cpp.
**Spec:** `Docs/Design/2026-09-15-uat-cycle-2-repairs.md`.

## Stage 1: Trace and repair account state
**Goal:** Close UAT-034/041/042/045 under TASK-13260.5.
**Success Criteria:** Notes metadata cannot cross server/account boundaries; source labels and titles are correct; manual-key disconnect/re-entry works.
**Tests:** Extend Notes recent-state behavior tests with two authorities and delayed hydration; Notes origin display; connection settings credential clearing; route/title reset. Run existing Notes stage12/stage46 and relevant auth regressions.
**Files:** `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`, Notes tests and provenance components, shared settings/auth components; web title boundary if required.
**Status:** In Progress
- [x] Reproduce the global-setting leak in a behavior test and observe failure.
- [ ] Implement authority-scoped persistence and stale-state guards, then verify A/B/reload behavior.
- [ ] Trace and test accurate note origin, title reset and manual-key disconnect.
- [ ] Review, validate and commit this unit with its Backlog record.

## Stage 2: Repair Flashcards and Chat
**Goal:** Close UAT-022/024/026/031/032/035/037/038/039 under TASK-13260.6/.7.
**Success Criteria:** Source-supported cards pass without accepting swapped/unsupported answers; derived cards have usable question/answer; visible Chat context/persistence matches actual requests and tracked multi-user replies succeed.
**Tests:** Flashcard verification boundary including the exact UAT source, unsupported additions and swapped answers; GeneratePanel error UI; Message save/editor validation; ReviewTab initial/completed state; tracked backend dispatch, greeting retry identifiers and active-store character context.
**Files:** Flashcards GeneratePanel/error taxonomy/ReviewTab, Playground Message save handlers, chat knowledge schema/save endpoint, flashcards verification units; Chat state/hooks and tracked completion endpoint with associated tests. Agents coordinate any shared Message/store file before edits.
**Status:** In Progress
- [ ] Each domain records exact failing regression commands before production changes.
- [ ] Apply minimal fixes described in the design; adapt named shared workflow steps for the explicit card editor without skip branches.
- [ ] Run focused frontend/backend regressions and touched Python Bandit; independently review integration.
- [ ] Commit each reviewed domain with its Backlog task.

## Stage 3: Repair Media lifecycle and retrieval
**Goal:** Close UAT-027/028/029/030/033/036/040/043/044 under TASK-13260.8/.9.
**Success Criteria:** Ingestion produces real content/analysis or honest failure; source navigation and deletion recover correctly; owned ordinary-user source retrieval succeeds with foreign-data denial preserved.
**Tests:** Scraper terminal status controls, no persistence of rejected fetches, plaintext analysis error handling, empty-library deep link, deletion URL clearing, Trash timestamp, source URL routing, self delete capability, actual owned/foreign Media search and QA contexts.
**Files:** `Plaintext_Files.py`, `enhanced_web_scraping.py`, ingestion service/result mapping, Media view/navigation/Trash, Knowledge QA source actions; `media/search` service/database ownership boundary identified by the positive/negative control trace.
**Status:** In Progress
- [ ] Confirm failing boundary tests and real runtime evidence before each repair.
- [ ] Repair ingestion/navigation/capability behavior while preserving permissions and remote restrictions.
- [ ] Trace normal owned Media search through its database query; reproduce and fix without weakening ownership/ACL filters.
- [ ] Run focused tests/Bandit/lint and review each unit before commit.

## Stage 4: Repair setup and prompt transitions
**Goal:** Close UAT-020/021/023/025 under TASK-13260.10/.11.
**Success Criteria:** Readiness includes truthful actionable reasons; endpoint model discovery precedes model selection; QA uses a usable configured default or clearly requires setup; saving a prompt leaves clean navigation state.
**Tests:** Empty setup readiness, valid local endpoint with blank model, configured default Chat→QA request, prompt save/back/new-route behavior and actual prompt reuse.
**Files:** Existing first-run setup schema/readiness and UnifiedSetupWizard/provider validation; Knowledge QA provider-default resolution; prompt create/edit route and form tests.
**Status:** In Progress
- [ ] Trace each failing UI/API transition against working adjacent flows.
- [ ] Write and observe failing behavior tests, then implement the minimal state/configuration correction.
- [ ] Run targeted tests, security/lint checks, review and commit both units.

## Stage 5: Review and repeat fresh UAT
**Goal:** Verify all repairs together, then run the complete original workflow matrix from new profiles in both modes.
**Success Criteria:** No bugs, failures or UX issues encountered across every required step; incomplete/blocked/skipped coverage cannot qualify as completion.
**Tests:** Combined touched regressions; independent review; fresh setup/auth/admin accounts, file and URL ingest/search/actual grounded Chat/QA/citations, Notes→Flashcards, prompt application, Chat-derived Notes/cards, review/re-analysis/delete/restore, cross-account read/write and browser-metadata isolation.
**Status:** Not Started
- [ ] Resolve integration review findings; record final regression, lint, baseline typecheck comparison and Bandit evidence.
- [ ] Freeze the repaired product revision and initialize fresh isolated single/multi data/config/browser profiles.
- [ ] Execute every named workflow with real inference, inspect actual payloads/provenance and save credential-free evidence.
- [ ] Record every issue in the running tracker; continue repair/review/UAT if any issue remains.
- [ ] Only after a complete issue-free pass, close the Backlog acceptance and active goal; remove only this completed plan.
