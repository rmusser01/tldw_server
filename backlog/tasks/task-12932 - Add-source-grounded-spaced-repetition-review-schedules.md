---
id: TASK-12932
title: Add source-grounded spaced repetition review schedules
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-10 15:22'
labels: []
dependencies: []
references:
  - >-
    Design spec:
    Docs/superpowers/specs/2026-07-09-source-grounded-spaced-repetition-design.md
  - Spec review approved by subagent 019f4a0a-f590-75e0-9fc6-ff1f4d48eba6
  - >-
    Implementation plan:
    Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md
  - Plan review approved by subagent 019f4a19-7d71-7bf2-8425-db97d8674734
  - 'Task 5 Flashcards UI commit: 296cc53a4f'
  - 'Task 6 Quiz handoff commit: fb0e1266e8'
  - 'Independent code review: subagent 019f4c74-52d1-7c80-8f8a-62d5bfcb76f9'
  - 'Review fixes: c669d2b545 and e694a3b08b'
  - 'Bandit report: /tmp/bandit_task_12932.json'
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
Delivered source-grounded spaced-repetition plans with custom day/month schedules and reread, quiz, flashcard, or cloze activities. Added atomic plan/occurrence persistence, due/start/complete/skip/delete APIs under /api/v1/flashcards/source-review-plans, a responsive Flashcards planner/due queue, and safe tokenized handoffs into existing Quiz and Flashcards generation forms. Verification: backend focused suite collected 85 tests and exited cleanly (live PostgreSQL fixture skipped because no test Postgres service/DSN was available); frontend consolidated suite 97 passed; shared UI TypeScript passed; OpenAPI guard verified 322 paths and 49 media fields with only 10 existing reviewed exceptions; Bandit reported 0 findings in /tmp/bandit_task_12932.json; working-tree and branch diff checks passed. No unresolved implementation blockers.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
