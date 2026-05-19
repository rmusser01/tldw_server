# PR 1002 Open Review Items Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the still-open review comments for PR `#1002` by fixing the remaining UI state issues, manuscript validation gaps, and sync/migration omissions without regressing the phase-4 writing suite behavior.

**Architecture:** Keep the fixes localized to the existing writing-suite UI and manuscript backend layers. Add regression tests first for each unresolved behavior, then implement the minimal production changes needed to make those tests pass, preserving existing API shapes and repo conventions.

**Tech Stack:** React, Zustand, TanStack Query, Vitest, FastAPI, Pydantic, SQLite/PostgreSQL migrations, pytest, Bandit

## Progress Update (2026-04-05)

- Completed the header persona default cloning regression fix and its guard coverage.
- Replaced the onboarding success-screen source guard with behavioral coverage and resolved the underlying merge-conflict regression in `OnboardingConnectForm.tsx`.
- Completed the manuscript analysis validation/error-hardening changes and fixed follow-on router import regressions in `writing_manuscripts.py` uncovered by integration tests.
- Verified manuscript DB integrity and sync-related follow-ups via the targeted writing/manuscript backend suite.
- Verification completed:
  - `bunx vitest run src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx src/components/Option/WritingPlayground/__tests__/writing-review-comments.guard.test.ts src/components/Option/WritingPlayground/__tests__/writing-phase2-review-fixes.guard.test.ts`
  - `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_db.py`
  - `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/core/Writing/manuscript_analysis.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py -f json -o /tmp/bandit_pr1002.json`
  - Bandit summary: `0` results (`0` high / `0` medium / `0` low).

---

### Task 1: Lock Down Header Persona Defaults And Persona Sync

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify: `apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts`

- [ ] **Step 1: Write the failing tests**

Update the persona shortcut tests so they assert against `HEADER_SHORTCUT_IDS.length` or exact set equality, and add a regression assertion that `getDefaultShortcutsForPersona("family")` returns a cloned array that can be mutated without changing subsequent calls.

- [ ] **Step 2: Run test to verify it fails**

Run: `bunx vitest run apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts`
Expected: FAIL on the new clone/assertion behavior before the implementation change.

- [ ] **Step 3: Write minimal implementation**

Update `getDefaultShortcutsForPersona()` to always return a fresh array copy, and update the “Show all features” action in `HeaderShortcuts.tsx` to route through the same persona update path so the Settings persona and header shortcut selection do not drift apart.

- [ ] **Step 4: Run test to verify it passes**

Run: `bunx vitest run apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts
git commit -m "fix: keep header persona defaults in sync"
```

### Task 2: Replace Brittle Onboarding Guard Tests With Behavioral Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`
- Read for patterns: `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`

- [ ] **Step 1: Write the failing tests**

Replace source-string assertions with rendered behavior tests that verify success-screen intent selection, family/research guided steps, direct chat navigation, and guided-flow “skip to chat” preserving persona behavior.

- [ ] **Step 2: Run test to verify it fails**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`
Expected: FAIL until the test harness is updated to exercise real behavior.

- [ ] **Step 3: Write minimal implementation**

Only adjust production onboarding code if the new behavioral tests expose a real regression. Otherwise keep the production implementation as-is and land the stronger test coverage.

- [ ] **Step 4: Run test to verify it passes**

Run: `bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.ts
git commit -m "test: replace onboarding source guards with behavior tests"
```

### Task 3: Cover Remaining Writing Playground UI Review Items

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/FeedbackTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingFeedback.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/ResearchTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/modals/ConnectionWebModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/modals/EventLineModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/modals/PlotTrackerModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/modals/StoryPulseModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingAnalysisModalHost.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx`

- [ ] **Step 1: Write the failing tests**

Add or extend focused Vitest coverage for stale mood responses, echo in-flight guarding, research-query/citation coupling, and inspector tab filtering. Where behavior is easiest to prove with typing-only changes, add regression tests around the affected rendered behavior rather than source-text checks.

- [ ] **Step 2: Run test to verify it fails**

Run: `bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx`
Expected: FAIL on the newly added guards before implementation.

- [ ] **Step 3: Write minimal implementation**

Add accessible switch labels, prevent stale mood updates and overlapping echo calls, normalize relationship IDs in `ConnectionWebModal`, replace remaining unsafe modal `any` usage with service-layer types, add error messaging for analysis actions, use Zustand selectors in `WritingAnalysisModalHost`, and build inspector tabs from only the provided content props.

- [ ] **Step 4: Run test to verify it passes**

Run: `bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.inspector-tabs.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/WritingPlayground/FeedbackTab.tsx apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingFeedback.ts apps/packages/ui/src/components/Option/WritingPlayground/hooks/__tests__/useWritingFeedback.test.tsx apps/packages/ui/src/components/Option/WritingPlayground/ResearchTab.tsx apps/packages/ui/src/components/Option/WritingPlayground/modals/ConnectionWebModal.tsx apps/packages/ui/src/components/Option/WritingPlayground/modals/EventLineModal.tsx apps/packages/ui/src/components/Option/WritingPlayground/modals/PlotTrackerModal.tsx apps/packages/ui/src/components/Option/WritingPlayground/modals/StoryPulseModal.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingAnalysisModalHost.tsx apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundInspectorPanel.tsx
git commit -m "fix: harden writing playground review followups"
```

### Task 4: Enforce Manuscript Analysis Validation And Safer Analysis Errors

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- Modify: `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`

- [ ] **Step 1: Write the failing tests**

Add a test that invalid or empty `analysis_types` returns `422`, and a service test that provider/runtime failures return a fixed safe error payload instead of raw exception text.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`
Expected: FAIL on invalid-analysis-type acceptance and raw error leakage before the fix.

- [ ] **Step 3: Write minimal implementation**

Constrain `ManuscriptAnalysisRequest.analysis_types` to the supported literal set with non-empty validation, remove the dead “unknown analysis type” persistence path, and change the structured-analysis service to log full exceptions server-side while returning a fixed non-sensitive payload.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/app/core/Writing/manuscript_analysis.py tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py
git commit -m "fix: validate manuscript analysis requests"
```

### Task 5: Validate Cross-Project And Plot Reference Integrity

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_characters_db.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`

- [ ] **Step 1: Write the failing tests**

Add DB-level regression tests for:
- cross-project character relationships being rejected,
- cross-project scene/world/character links being rejected,
- invalid `parent_id` on world info being rejected across projects,
- impossible plot event/hole combinations (`scene_id` vs `chapter_id`, mismatched `plot_line_id`, mismatched project IDs) being rejected,
- scene updates with only `content_json` marking analyses stale.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`
Expected: FAIL on the new integrity assertions before implementation.

- [ ] **Step 3: Write minimal implementation**

Add internal helper validation in `ManuscriptDBHelper` to assert referenced rows exist and belong to the same project before insert/update, make `link_scene_character()` an upsert for `is_pov`, and expand scene update invalidation to fire when either `content_plain` or `content_json` changes.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py
git commit -m "fix: enforce manuscript reference integrity"
```

### Task 6: Complete Manuscript Sync Metadata And Trigger Payloads

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_characters_db.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`
- Modify: `tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py`

- [ ] **Step 1: Write the failing tests**

Add migration/DB behavior tests that assert:
- scene-link tables include sync metadata columns,
- link create/update/delete activity emits `sync_log` rows,
- sync payloads for manuscript characters/world info/AI analyses include the missing fields,
- `AFTER UPDATE` triggers fire when the previously omitted columns change.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py`
Expected: FAIL on missing schema/trigger coverage before the migration changes.

- [ ] **Step 3: Write minimal implementation**

Update both schema-creation/migration SQL blocks in `ChaChaNotes_DB.py` to add sync metadata for `manuscript_scene_characters` and `manuscript_scene_world_info`, create corresponding sync triggers, and expand the existing `sync_log` payloads and `WHEN` clauses to include the missing manuscript fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py
git commit -m "fix: complete manuscript sync metadata"
```

### Task 7: Verify The Touched Scope

**Files:**
- Modify: `docs/superpowers/plans/2026-04-05-pr-1002-open-review-items.md`

- [x] **Step 1: Run targeted UI verification**

Run: `bunx vitest run src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx src/components/Option/WritingPlayground/__tests__/writing-review-comments.guard.test.ts src/components/Option/WritingPlayground/__tests__/writing-phase2-review-fixes.guard.test.ts`
Result: PASS

- [x] **Step 2: Run targeted backend verification**

Run: `source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py tldw_Server_API/tests/Writing/test_manuscript_analysis_db.py tldw_Server_API/tests/Writing/test_manuscript_characters_db.py tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py tldw_Server_API/tests/Writing/test_manuscript_db.py`
Result: PASS (`217` tests)

- [x] **Step 3: Run Bandit on touched backend scope**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py tldw_Server_API/app/core/Writing/manuscript_analysis.py tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py -f json -o /tmp/bandit_pr1002.json`
Result: exit `0`, `0` findings

- [x] **Step 4: Update the plan status notes**

Mark completed tasks in this plan file before final handoff.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-04-05-pr-1002-open-review-items.md
git commit -m "docs: record verification for pr1002 review followups"
```
