---
id: TASK-12932
title: Add source-grounded spaced repetition review schedules
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-10 15:22
labels: []
dependencies: []
references:
- 'Design spec: Docs/superpowers/specs/2026-07-09-source-grounded-spaced-repetition-design.md'
- Spec review approved by subagent 019f4a0a-f590-75e0-9fc6-ff1f4d48eba6
- 'Implementation plan: Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md'
- Plan review approved by subagent 019f4a19-7d71-7bf2-8425-db97d8674734
- 'Task 5 Flashcards UI commit: 296cc53a4f'
- 'Task 6 Quiz handoff commit: fb0e1266e8'
- 'Independent code review: subagent 019f4c74-52d1-7c80-8f8a-62d5bfcb76f9'
- 'Review fixes: c669d2b545 and e694a3b08b'
- 'Bandit report: /tmp/bandit_task_12932.json'
- 'Review hardening commit: bd34bb431c'
- 'PR comment fixes: cec28a759c'
- 'Pull request: https://github.com/rmusser01/tldw_server/pull/2705'
- 'Final Bandit report: /tmp/bandit_task_12932_review_final.json'
modified_files:
- Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md
- Docs/superpowers/specs/2026-07-09-source-grounded-spaced-repetition-design.md
- apps/packages/ui/src/assets/locale/en/option.json
- apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx
- apps/packages/ui/src/components/Flashcards/components/SourceReviewDuePanel.tsx
- apps/packages/ui/src/components/Flashcards/components/SourceReviewPlanDrawer.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewDuePanel.test.tsx
- apps/packages/ui/src/components/Flashcards/components/__tests__/SourceReviewPlanDrawer.test.tsx
- apps/packages/ui/src/components/Flashcards/components/index.ts
- apps/packages/ui/src/components/Flashcards/hooks/__tests__/useSourceReviewQueries.test.tsx
- apps/packages/ui/src/components/Flashcards/hooks/index.ts
- apps/packages/ui/src/components/Flashcards/hooks/useSourceReviewQueries.ts
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/shared.ts
- apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx
- apps/packages/ui/src/components/Quiz/QuizPlayground.tsx
- apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx
- apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.source-review.test.tsx
- apps/packages/ui/src/services/__tests__/flashcards-source-review.test.ts
- apps/packages/ui/src/services/flashcards.ts
- apps/packages/ui/src/services/tldw/__tests__/source-review-handoff.test.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/services/tldw/source-review-handoff.ts
- backlog/tasks/task-12932 - Add-source-grounded-spaced-repetition-review-schedules.md
- tldw_Server_API/app/api/v1/endpoints/flashcards.py
- tldw_Server_API/app/api/v1/schemas/flashcards.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py
- tldw_Server_API/app/core/Flashcards/source_review.py
- tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans.py
- tldw_Server_API/tests/ChaChaNotesDB/test_source_review_plans_postgres.py
- tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
- tldw_Server_API/tests/Flashcards/test_source_review_plans_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a first-pass source-grounded spaced repetition workflow so users can schedule future reviews from highlighted or selected source-grounded material and choose the review activity.

Acceptance criteria:
- Users can create a custom review schedule such as Day 1, Day 3, Day 7, Day 14, Day 28, 3 months, and 6 months from selected source-grounded content.
- Scheduled reviews preserve links back to the originating note/source context.
- The first slice supports a small set of reminder activities using existing capabilities where possible: reread note/source excerpt, quiz, flashcards, and fill-in-the-blank/cloze review.
- Existing flashcard/quiz scheduling behavior remains backward compatible.
- WebUI exposes a minimal review-schedule creation and due-review surface.
- Tests cover schedule creation, due-review projection, source linkage, and backward-compatible existing review flows.

Out of scope for the first slice:
- Visual recognition questions.
- Extended matching questions.
- New notification delivery channels beyond an in-app due-review surface.
- Replacing the existing flashcard scheduler model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md task-by-task using TDD and per-task spec/code-quality review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the approved first slice using TDD across scheduling helpers, SQLite/PostgreSQL persistence, Flashcards-owned REST APIs, typed WebUI clients/hooks, Flashcards planner and due-review surfaces, and tokenized Quiz/Flashcards handoffs. Source snapshots remain on the server/sessionStorage rather than in URLs; starting a review does not auto-generate artifacts. Independent review findings were addressed in c669d2b545 and e694a3b08b: nonexistent civil dates are rejected, due rows receive bounded source summaries and 60-second refresh, storage failures remain recoverable, multi-source generation budgets are fair, and ambiguous card provenance is omitted. The approved Quiz first-slice limitation remains explicit: one media source and note IDs are selectable; messages and additional media remain visible as read-only snapshot context.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Reopened after independent PR review 019f4c90-66ed-7f21-ae05-a29786459650. Confirmed findings: missing PostgreSQL RLS for source review tables; oversized timezone OSError; unbounded source IDs, labels, and aggregate bundle; incomplete reread provenance fallback; non-consuming handoff storage; form a11y associations; untranslated source-review UI copy.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered source-grounded spaced-repetition plans with custom day/month schedules and reread, quiz, flashcard, or cloze activities, backed by source snapshots and safe tokenized handoffs. Review hardening added forced PostgreSQL tenant RLS, bounded request payloads, complete reread provenance, translated and accessible controls, one-shot StrictMode-safe handoffs, stable expanded due queues, and datetime-anchor handling. Verification: backend 92 passed and 2 live-PostgreSQL tests skipped because no test service/DSN was available; frontend 58 passed; OpenAPI verified 322 paths and 49 media fields with 10 existing reviewed exceptions; focused Ruff, compileall, locale JSON, and diff checks passed; Bandit reported 0 findings in /tmp/bandit_task_12932_review_final.json. Full shared-UI TypeScript still reports unrelated repository baseline errors, with none in touched source-review files. Browser visual verification could not run because sandbox escalation was unavailable. All six GitHub review threads are resolved; no remaining implementation blocker.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review hardening completed in bd34bb431c and cec28a759c. Added forced PostgreSQL RLS with startup-before-schema guards and two-principal coverage; bounded malformed and oversized source-review input; completed reread provenance, localization, accessibility, and one-shot handoff lifecycle; preserved expanded due queues across refetches; accepted datetime anchors; and hardened malformed reread launch state handling. All six Gemini PR threads were answered and resolved. A fresh subagent review was requested but could not start because the reviewer service reached its usage limit; local review found no remaining critical or important issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
