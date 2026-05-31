---
id: TASK-489
title: Reconcile non-design Watchlists Backlog statuses
status: Done
labels:
- watchlists
- backlog
- tracking
priority: medium
references:
- backlog/tasks
- https://github.com/rmusser01/tldw_server/pulls?q=is%3Apr+watchlists
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit non-design Watchlists Backlog task records against merged implementation work and update stale task statuses/notes so the remaining queue does not look like open product implementation debt. Exclude design-system/product-state migration items because that queue is owned separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify Watchlists Backlog tasks that are still open but have evidence of merged or completed product work.
- [x] #2 Update only non-design-system Watchlists tracking records with clear evidence and final summaries; do not modify design-system/product-state migration tasks.
- [x] #3 Record any tasks intentionally left open with the reason.
- [x] #4 Run metadata hygiene checks and document Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Parse current Watchlists-related Backlog tasks and separate non-design product/PRD tasks from design-system/product-state tasks.
2. Inspect candidate open tasks for implementation notes, PR references, and merged work evidence.
3. Patch stale non-design task metadata to Done only when the file itself contains completion evidence or successor implementation records.
4. Run diff hygiene checks and record the cleanup summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Closed stale non-design Watchlists tracking records with completion/successor evidence: `TASK-440`, `TASK-476`, and `TASK-481`.
- Left `TASK-45.44.3` open intentionally because it is the design-system/product-state Watchlists migration queue owned separately.
- Left `TASK-415` untouched because its Watchlists hit is incidental route text in a main-chat cockpit task, not Watchlists product backlog.
- Left the open design-system PR/task stream untouched because another worker is handling design-system work.
- Verification: current task scan now shows no open non-design Watchlists product implementation records in `backlog/tasks`; remaining hits are design-system/product-state or unrelated route text.
- Verification: `git diff --check` passed.
- Bandit: skipped/not applicable; touched files are Backlog Markdown task metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled stale non-design Watchlists task metadata. `TASK-440`, `TASK-476`, and `TASK-481` now reflect the completed/superseded state documented by their successor planning and implementation records. No design-system/product-state migration tasks or product code were changed.
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
