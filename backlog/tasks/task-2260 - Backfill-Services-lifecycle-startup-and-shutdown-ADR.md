---
id: TASK-2260
title: Backfill Services lifecycle startup and shutdown ADR
status: Done
dependencies:
- TASK-2259
labels:
- docs
- process
- adr
- services
- lifecycle
modified_files:
- Docs/ADR/021-services-lifecycle-startup-and-shutdown.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- tldw_Server_API/app/services/README.md
- backlog/tasks/task-2260 - Backfill-Services-lifecycle-startup-and-shutdown-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded Services lifecycle ADR from TASK-2259 evidence. Scope the accepted decision to FastAPI lifespan startup/shutdown orchestration through Services helpers, LifespanWorkerRuntimeState ownership of the worker lifecycle session, declarative lifecycle worker specs/engine/session ownership, cooperative stop-event workers with bounded timeout/cancel fallback, job-poller drain/quiesce before background worker shutdown, and explicit caveats for callback-only workers, legacy shutdown adapters, and scope limited to lifecycle-managed Services workers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create the next accepted ADR under `Docs/ADR/` using the standard ADR template and TASK-2259 evidence.
- [x] #2 Keep accepted claims scoped to Services lifespan orchestration, worker lifecycle session ownership, declarative worker specs/engine/session, stop-event default strategy, staged shutdown order, and documented caveats.
- [x] #3 Update `Docs/ADR/README.md`, the INV-031 inventory row, and the Services README backlink after ADR creation.
- [x] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/ADR/021-services-lifecycle-startup-and-shutdown.md` as the accepted backfill ADR for INV-031.
- Scoped the ADR to focused Services lifespan orchestration, `LifespanWorkerRuntimeState` worker lifecycle session ownership, declarative `WorkerSpec`/`LifecycleWorkerEngine`/`WorkerLifecycleSession` ownership, stop-event worker defaults, staged shutdown order, and bounded timeout/cancel fallback.
- Preserved caveats for callback-only workers, legacy shutdown adapters, bounded lease drain, non-Services-managed background work, and ADR-003 remaining the Jobs-vs-Scheduler ownership record.
- Updated `Docs/ADR/README.md`, the INV-031 inventory row/default disposition, and `tldw_Server_API/app/services/README.md`.
- Verification:
  - `git diff --check` passed.
  - `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Docs/test_docs_index_path_hygiene_script.py tldw_Server_API/tests/Docs/test_readme_docs_path_hygiene_script.py tldw_Server_API/tests/Docs/test_top_guides_docs_path_hygiene_script.py` passed: 3 passed, 6 warnings.
  - Scoped ADR/reference grep confirmed ADR-021, TASK-2260, INV-031, Services README backlink, and inventory/default references.
- Bandit: skipped because this task touched only Markdown documentation and Backlog task metadata; no Python/code paths were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled INV-031 as ADR-021 for the bounded Services lifecycle startup/shutdown decision. The ADR records Services lifespan orchestration, worker lifecycle session ownership, declarative lifecycle workers, stop-event worker defaults, staged shutdown, bounded timeout/cancel fallback, and explicit caveats for callback-only workers, legacy shutdown adapters, bounded lease drain, and non-Services-managed work. Updated the ADR index, inventory row/defaults, and Services README backlink. Verification passed for diff hygiene, docs path hygiene tests, and scoped reference grep. Bandit is not applicable because the touched scope is Markdown documentation and Backlog task metadata only.
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
