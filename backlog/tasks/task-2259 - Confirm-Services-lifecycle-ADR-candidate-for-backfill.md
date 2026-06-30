---
id: TASK-2259
title: Confirm Services lifecycle ADR candidate for backfill
status: Done
labels:
- docs
- process
- adr
- services
- lifecycle
modified_files:
- Docs/ADR/inventory/2026-06-04-services-lifecycle-confirmation-audit.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- backlog/tasks/task-2259 - Confirm-Services-lifecycle-ADR-candidate-for-backfill.md
- backlog/tasks/task-2260 - Backfill-Services-lifecycle-startup-and-shutdown-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirm whether INV-031 from Docs/ADR/inventory/2026-06-03-decision-inventory.md is current and bounded enough for ADR backfill. Verify tldw_Server_API/app/services/README.md and representative code/tests for lifespan worker runtime state ownership, cooperative stop events, shutdown drain gates, owned worker shutdown order, caveats, and any scope that should remain inventory-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create a Services lifecycle confirmation audit under `Docs/ADR/inventory/` using current `origin/dev` evidence.
- [x] #2 Classify `INV-031` as ready for bounded ADR backfill, needing code/doc alignment, or inventory-only, with explicit caveats.
- [x] #3 Update the decision inventory only if the confirmation result changes the tracked next action.
- [x] #4 Create a follow-up Backlog task only if the candidate is ready for ADR backfill.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/ADR/inventory/2026-06-04-services-lifecycle-confirmation-audit.md`.
- Classified INV-031 as current governing and ready for one bounded Services lifecycle ADR backfill.
- Scoped the future ADR to FastAPI lifespan Services orchestration, `LifespanWorkerRuntimeState` worker-session ownership, declarative lifecycle worker specs/engine/session, stop-event default strategy, bounded timeout/cancel fallback, and staged shutdown order.
- Documented caveats for callback-only workers, legacy shutdown adapters, bounded lease drain, non-Services-managed background work, and the current runtime-state implementation storing the worker lifecycle session aggregate rather than every individual long-lived handle.
- Updated `Docs/ADR/inventory/2026-06-03-decision-inventory.md` to record TASK-2259 confirmation and queue TASK-2260 for the accepted ADR backfill.
- Created TASK-2260 as the follow-up ADR implementation task.
- Verification:
  - `git diff --check` passed.
  - Scoped ADR/backlog reference grep passed for TASK-2259, TASK-2260, INV-031, the Services confirmation audit, expected ADR-021 path, and Bandit references.
  - Focused Services lifecycle pytest suite passed: 120 passed, 6 warnings.
  - Initial `source .venv/bin/activate` failed because this isolated worktree does not contain its own `.venv`; reran with the shared repo virtualenv interpreter. Reusable command from this worktree: `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Services/test_lifespan_worker_runtime_state.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py tldw_Server_API/tests/Services/test_lifecycle_worker_engine.py tldw_Server_API/tests/Services/test_lifecycle_worker_session.py tldw_Server_API/tests/Services/test_lifecycle_workers.py tldw_Server_API/tests/Services/test_shutdown_transition_handoff.py tldw_Server_API/tests/Services/test_shutdown_job_poller_handoff.py tldw_Server_API/tests/Services/test_shutdown_owned_job_pollers.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`.
  - Removed test-generated untracked artifacts before staging.
- Bandit: skipped because this task touched only Markdown documentation and Backlog task metadata; no Python/code paths were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed INV-031 as ready for a bounded Services lifecycle ADR backfill. The confirmation audit captures current code/test evidence and caveats, the inventory now points to TASK-2259/TASK-2260, and TASK-2260 is queued for the actual ADR. Verification passed for diff hygiene, scoped references, and the focused Services lifecycle pytest suite. Bandit is not applicable because the touched scope is Markdown documentation and Backlog task metadata only.
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
