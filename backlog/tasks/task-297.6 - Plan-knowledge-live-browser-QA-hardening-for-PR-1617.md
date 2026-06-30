---
id: TASK-297.6
title: Plan /knowledge live-browser QA hardening for PR 1617
status: Done
assignee: []
created_date: '2026-05-13 03:39'
updated_date: '2026-05-13 06:34'
labels:
  - knowledge
  - ux
  - planning
  - qa
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1617'
documentation:
  - Docs/superpowers/specs/2026-05-13-knowledge-live-browser-qa-hardening-design.md
  - Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-hardening-implementation-plan.md
  - Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md
parent_task_id: TASK-297
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create and commit a design spec for a verification-first /knowledge QA hardening stage inside PR #1617. The stage should broaden live-browser QA across viewports and seeded plus local real-data databases, allow only narrow same-PR fixes for observed /knowledge regressions or low-risk keyboard/power-user friction, and defer saved-view sharing or advanced organization to a later product-expansion issue after QA evidence is collected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec captures the approved verification-first scope, seeded plus local real-data QA strategy, narrow fix gate, and observation-only treatment for saved-view sharing and advanced organization.
- [x] #2 Spec defines staged QA activities, evidence outputs, privacy rules for local real-data QA, and current-PR versus follow-up triage criteria.
- [x] #3 Spec includes a plan to create a follow-up product-expansion GitHub issue after QA synthesis with evidence-based acceptance criteria.
- [x] #4 Spec is reviewed, committed to the current PR branch, and linked from the Backlog task.
- [x] #5 Implementation plan for the approved QA hardening design is written, reviewed, committed, and linked from the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design approved in brainstorming: keep PR #1617 verification-first with a narrow same-PR fix gate; use both repeatable seeded browser QA and one local real-data pass; treat saved-view/profile sharing and advanced organization as observation-only for this PR; create a product-expansion issue after QA synthesis so it can cite evidence.

Spec path: Docs/superpowers/specs/2026-05-13-knowledge-live-browser-qa-hardening-design.md

Spec review passed with no blocking issues. Applied advisory clarifications: evidence-table TBDs are placeholders, existing export/share dialogs are qualified as existing surfaces, and the product-expansion issue is explicitly evidence-based after QA synthesis.

Additional self-review before implementation planning tightened three risks: added a guard against overbuilding seeded fixture infrastructure inside PR #1617, added local real-data safety rules to avoid mutating or leaking private data, and added severity/exit criteria so QA findings have clear merge gates.

Implementation plan path: Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-hardening-implementation-plan.md

Evidence artifact to create during execution: Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md

Implementation plan review approved with no blocking issues. Advisory note for execution: keep blocked fixture/runtime rows in the evidence artifact instead of deleting them so the final QA summary preserves the full required matrix.

Task 1 execution record created the evidence artifact at Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-evidence.md before browser QA began. The artifact records branch codex/knowledge-source-contract, PR #1617, baseline commit 7667cb9da6287dbc6d26fd9cb5b45d12791ebe1e, runtime setup defaults from apps/tldw-frontend/playwright.config.ts, privacy rules, and the predeclared seeded, extension, local real-data, keyboard, and skipped-row matrices.

Task 2 baseline harnesses completed after rerunning the sandbox-blocked WebUI Playwright smoke row outside the sandbox. Shared UI focused Vitest passed 5 files / 29 tests, WebUI route parity Vitest passed 1 file / 4 tests, WebUI `/knowledge` smoke passed 1 Playwright test after initial `listen EPERM` sandbox blocker, and extension knowledge tutorial route coverage passed 2 Playwright tests. Non-blocking harness/build warnings are recorded in the evidence artifact.

Task 3 WebUI seeded live-browser matrix completed. Local backend health was reachable in single-user mode with healthy database/metrics and degraded chacha_notes; the browser matrix itself used synthetic Playwright route mocks, so it is seeded frontend coverage rather than real database/backend coverage.

Task 3 command: `node /private/tmp/knowledge_task3_live_qa.cjs` from `apps/tldw-frontend` against `http://localhost:8080/knowledge` with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`. Results were written to `/private/tmp/knowledge-task3-live-qa/results.json` at `2026-05-13T05:25:20.692Z`.

Task 3 covered `desktop-empty`, `mobile-empty`, `desktop-wide`, `desktop-local-only`, `desktop-constrained`, `tablet`, `mobile`, and `desktop-no-results`. All rows passed with zero console errors and zero failed API requests. Coverage included no-source states, Add Sources reachability, canonical source labels, specific source query/status/recent/workspace filters, generated/workspace artifact hiding, bulk actions, saved profile save/load, local-only web-disabled query, web-enabled query, no-result recovery, Evidence/source cards, and Continue in editor reachability.

Task 3 observation KQA-T3-001: mobile compact toolbar shows web toggle and AI server-default provider, and privacy copy is available in settings/empty state, but privacy copy is not adjacent to the compact toolbar after sources are selected. Classified P3 document-only for now; reassess during keyboard/accessibility/final synthesis before deciding whether a small aria-label or visible disclosure refinement belongs in PR #1617.

Task 4 extension route and viewport pass completed. Shipped extension quick-chat tutorial-card harness passed for `#/knowledge/thread/:threadId` and `#/knowledge/shared/:shareToken` with 2 tests passing in 7.0s.

Task 4 temporary extension route probe command: `bun /private/tmp/extension_knowledge_task4_qa.ts` from `apps/extension`. It used `http://dummy-tldw` seeded synthetic route mocks, so this is extension UI route coverage rather than real backend/database coverage. It generated `/private/tmp/knowledge-task4-extension-qa/results.json` at `2026-05-13T05:34:17.208Z` and verified `#/knowledge`, `#/knowledge/thread/thread-123`, and `#/knowledge/shared/share-token-123` at `390 x 844` with zero console errors and zero failed requests. Each route exposed search input, ready-state title, Add Sources, source scope, web fallback toggle, and all canonical source labels.

Task 4 supplemental legacy harness result: `bunx playwright test tests/e2e/knowledge-rag-ux.spec.ts -g "workflow hub" --reporter=line` failed before reaching `/knowledge` because `#/settings/manageKnowledge` now renders a 404 and the expected `workflow-button` is absent. Recorded as stale supplemental harness coverage with a blocked command result, not a `/knowledge` product defect.

Task 5 privacy-safe local real-data backend-read-only pass completed. Live backend media and notes list endpoints returned 200 when checked with `TLDW_E2E_API_KEY`, with response bodies discarded. The browser probe `TLDW_E2E_API_KEY=<local-e2e-key> node /private/tmp/knowledge_task5_real_readonly.cjs` generated `/private/tmp/knowledge-task5-real-readonly/results.json` at `2026-05-13T05:59:35.851Z`.

Task 5 recorded only counts/control state and captured no screenshots, titles, source excerpts, generated answers, response bodies, or private content. Desktop backend-read-only coverage reached search, ready state, source scope, web fallback toggle, temporary browser-local saved profile create/list/load in isolated Playwright storage, all 8 source-category labels, specific-source filters, source status filter, recent imports filter, workspace scope filter, bulk select, and text filter. It counted 200 media options and 0 note options and recorded read-only GETs to media/notes endpoints with zero failed requests, zero console errors, zero backend mutation requests, and zero local-database mutation attempts. Mobile backend-read-only coverage reached search, ready state, source scope, web fallback toggle, all 8 source-category labels, compact source settings, and compact source categories.

Task 5 did not run weak/no-result query, answer inspection, or citation/source-card inspection against private local data because the current UI query flow creates/persists Knowledge QA chat/thread state and would expose private answer/source content. Seeded synthetic Tasks 2-3 already cover those behaviors without private data.

Task 6 keyboard/power-user pass completed with seeded browser coverage. Desktop keyboard checks reached search/source scope, verified Enter activation across source menu, specific source selector, profiles, evidence, source preview, export, Simple/Detailed, bulk select, and repeated query after changing source scope. Mobile keyboard checks reached search/source scope/Knowledge QA settings, verified settings open/close, search submit, and Evidence with 2 source cards. Playwright did not deliver Escape keydown events to the page in the final diagnostic run, so browser Escape closure is recorded as automation-limited and covered by focused Vitest instead.

Task 7 current-PR fixes completed within the narrow gate. Added capture-phase Escape handling in KnowledgeContextBar and SettingsPanel so nested controls cannot swallow Escape before source/settings overlays close. Renamed the compact toolbar settings control label and tooltip to Open Knowledge QA settings to distinguish it from global app Settings. Added focused regression tests before implementation and verified the red run failed as expected; after the fix, `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx` passed 3 files / 30 tests. Bandit not applicable because no Python production files were touched.

Task 8 product-expansion issue created: https://github.com/rmusser01/tldw_server/issues/1631. The issue keeps saved views, profile sharing/export/import, advanced source organization, workspace grouping, keyboard shortcuts, and mobile disclosure refinements out of PR #1617 unless later user-trial evidence proves the QA workflow needs them. It preserves the non-goals that `/knowledge` remains QA-only, web fallback stays user-toggleable/default-provider based, generated/test/workspace artifacts stay hidden unless workspace-scoped, and older extension route investment requires shipped-versus-legacy confirmation.
<!-- SECTION:NOTES:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the approved plan in `Docs/superpowers/plans/2026-05-13-knowledge-live-browser-qa-hardening-implementation-plan.md` using subagent-driven checkpoints:

1. Create the QA evidence artifact and sync task metadata.
2. Run existing seeded baseline harnesses.
3. Run the WebUI seeded live-browser matrix.
4. Run the extension route and viewport pass.
5. Run the privacy-safe local real-data pass.
6. Run keyboard and power-user friction checks.
7. Fix only current-PR defects proven by QA evidence.
8. Create the evidence-based product-expansion issue.
9. Close out PR QA hardening with evidence, task updates, PR comment, and refreshed PR state.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the PR #1617 `/knowledge` live-browser QA hardening slice. The evidence artifact now records seeded WebUI coverage across desktop/tablet/mobile, extension route coverage for `#/knowledge`, `#/knowledge/thread/:threadId`, and `#/knowledge/shared/:shareToken`, a privacy-safe backend-read-only local real-data pass, and keyboard/power-user coverage.

Current-PR fixes were kept within the approved narrow gate: source/settings overlays now close on Escape using capture-phase handlers even when nested controls intercept key events, and the compact `/knowledge` settings control now uses the distinct label and tooltip `Open Knowledge QA settings`. Product expansion stayed out of PR #1617; follow-up issue https://github.com/rmusser01/tldw_server/issues/1631 tracks saved views, profile sharing/export/import, advanced organization, workspace grouping, keyboard shortcuts, and mobile disclosure refinements only if later QA/user evidence supports them.

Verification: focused red tests failed before the keyboard fixes; after the fix, `bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx` passed 3 files / 30 tests. Broader focused UI verification passed 8 files / 59 tests, and extension route parity passed 1 file / 4 tests. `git diff --check` passed. ASCII and API-key scans on the evidence/task docs returned no matches. Bandit was not applicable because no Python production files were touched. Known blocker: Playwright did not deliver Escape keydown events to the page in the diagnostic browser run, so Escape closure is verified by focused Vitest instead of direct browser observation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
