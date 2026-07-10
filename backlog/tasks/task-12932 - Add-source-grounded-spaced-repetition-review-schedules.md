---
id: TASK-12932
title: Add source-grounded spaced repetition review schedules
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-10 19:48
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

Final PR follow-up: rebased onto dev c588c3b8b521f9411e9ae08c42b3365e886846c8 and addressed all 15 CodeRabbit threads in 3b5f5bb0da. The due poll now stays thin and resume hydration reuses the existing nested start action; optimistic state, handoff validation, error telemetry, provenance cleanup, typed not-found errors, shared caps, schema docs, stored-metadata handling, and PostgreSQL role teardown were also hardened. All 15 threads were answered inline and resolved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered source-grounded spaced-repetition plans with custom day/month schedules and reread, quiz, flashcard, or cloze activities, backed by source snapshots and safe tokenized handoffs. Review hardening added forced PostgreSQL tenant RLS, bounded request payloads, complete reread provenance, translated and accessible controls, one-shot StrictMode-safe handoffs, stable expanded due queues, and datetime-anchor handling. Verification: backend 92 passed and 2 live-PostgreSQL tests skipped because no test service/DSN was available; frontend 58 passed; OpenAPI verified 322 paths and 49 media fields with 10 existing reviewed exceptions; focused Ruff, compileall, locale JSON, and diff checks passed; Bandit reported 0 findings in /tmp/bandit_task_12932_review_final.json. Full shared-UI TypeScript still reports unrelated repository baseline errors, with none in touched source-review files. Browser visual verification could not run because sandbox escalation was unavailable. All six GitHub review threads are resolved; no remaining implementation blocker.

Final rebase verification: backend 95 passed with 2 live-PostgreSQL skips; frontend 114 passed; focused Ruff, critical ChaCha Ruff, compileall, OpenAPI, diff checks, and Bandit passed with 0 findings in /tmp/bandit_task_12932_rebase_review_final.json. Full UI TypeScript reported only unrelated baseline diagnostics.
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
