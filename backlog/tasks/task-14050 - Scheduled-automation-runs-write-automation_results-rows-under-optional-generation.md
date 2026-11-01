---
id: TASK-14050
title: 'Scheduled-automation runs write automation_results rows under optional generation'
status: To Do
assignee: []
created_date: '2026-10-13 00:00'
updated_date: '2026-10-13 00:00'
labels:
  - scheduled-tasks
  - automation
  - recurring-question
dependencies: []
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
