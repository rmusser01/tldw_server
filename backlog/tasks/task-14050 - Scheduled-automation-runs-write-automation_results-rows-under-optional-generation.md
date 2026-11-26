---
id: TASK-14050
title: Scheduled-automation runs write automation_results rows under optional generation
status: Done
assignee: []
created_date: 2026-10-13 00:00
updated_date: 2026-10-13 00:00
labels:
- scheduled-tasks
- automation
- recurring-question
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/2864
- https://github.com/rmusser01/tldw_server/pull/3637
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
- Docs/User_Guides/WebUI/Automation_Admin_Operations.md
---

## Description (the why)

Filed at chatbook TASK-18940 close-out (its AC#8 record): the live
verification rounds (2026-09-02) showed the scheduled-automation executor
reports `run_succeeded` but writes NO `automation_results` row when the
definition's `generation_mode` is `optional` — the results inbox the
client renders stayed empty for genuinely-executed runs until rows were
seeded by hand. Result generation exists for `required` mode; `optional`
silently produces nothing, so the notification says "succeeded" while the
findings surface shows no finding. That gap is the last unaudited piece of
the ADR-077/184 result pass-back story.

## Acceptance Criteria (the what)

- [ ] A run of a `generation_mode: optional` definition that produces an
      answer writes an `automation_results` row (answer as the finding
      payload, run/definition ids set) exactly once per run
- [ ] A run that produces no answer (skipped/approval_required/failed)
      writes no result row
- [ ] Existing `required`-mode behavior and the results review flow
      (read/dismiss pushback) are unchanged — pinned by the existing
      suites
- [ ] A live round against a real server shows: optional-mode run →
      `run_succeeded` audit → result row present → client Results view
      renders it unread without seeding

## Implementation Plan (the how)

1. Locate the executor's result-write path (`automation_executors` /
   the consumer's `_finish`) and the `required`-mode write; mirror it for
   `optional` when `result_text` is non-empty
2. Tests mirroring the consumer suite's fixture style (optional+answer →
   row; optional+empty → no row; required unchanged)
3. Live verification per lessons-live-verification; record honestly

## Implementation Notes

(added after implementation)

ADR required: no — completes the existing ADR-077 pass-back contract; no
new boundary.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. The Phase 4B visibility policy and scheduled-job compatibility contract govern this fix. The worker uses the existing claim-checked Scheduled Tasks write transaction for the run and finding. The guide documents the Phase 4C no-match rule for the separate RAG API.

The dev baseline reproduced the exact report. A manual optional run emitted run_succeeded and the Results endpoint returned zero rows. The new regression runs preview, create, the compatibility Run Now route, the real consumer, and the Results API. It verifies the full answer, matching IDs, bounded summary, unread state, and one result across review and redelivery. A forced result-write interruption rolls back completion and lets the same job retry. The consumer tests cover empty and failed answers, approval, and visibility exclusions.

Verification against origin/dev 1d8589b7c4 included 234 scheduled-task tests, followed by 40 consumer and Run Now tests on the final visibility policy. Local FastAPI and the Jobs worker used a deterministic model response. Playwright saw the generated result under Unreviewed, opened the answer, and confirmed Mark reviewed persisted read state through the API. Bandit and Ruff reported no new findings.
PR #3637 opened against dev after all 235 affected scheduled-task tests passed on a560526fe7.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Saved the optional-generation answer and completed run in one transaction. The worker records one Results finding across retries, and the WebUI can display and review it. The guide describes no-match run history.
<!-- SECTION:FINAL_SUMMARY:END -->
