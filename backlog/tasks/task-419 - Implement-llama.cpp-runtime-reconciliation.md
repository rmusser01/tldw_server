---
id: TASK-419
title: Implement llama.cpp runtime reconciliation
status: Done
labels:
- llamacpp
- local-llm
- backend
priority: medium
documentation:
- Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_reconciler.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py
- tldw_Server_API/app/services/startup_heavy_init.py
- tldw_Server_API/app/services/shutdown_resource_cleanup.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_reconciler.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py
- tldw_Server_API/tests/Services/test_startup_heavy_init.py
- tldw_Server_API/tests/Services/test_shutdown_resource_cleanup.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the llama.cpp managed runtime closeout plan: startup/shutdown reconciliation for autostart profiles, bounded restart behavior, and durable last-failure metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add tests covering autostart reconciliation, disabled/non-autostart skips, failed start metadata, max restart suppression, pause suppression, and shutdown behavior.
- [x] #2 Add llama.cpp runtime reconciler service that delegates process ownership to the existing supervisor.
- [x] #3 Persist bounded last runtime failure metadata without storing logs or raw environment.
- [x] #4 Hook reconciliation into startup/shutdown lifecycle following existing service patterns.
- [x] #5 Run focused backend tests, diff checks, and Bandit on touched Python paths before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Used the existing supervisor/store boundary rather than adding a parallel process registry. The reconciler decides eligibility and delegates locking, launch, stop, and persistence through `LlamaCppSupervisor`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1816 review feedback. Reconciler shutdown is isolated from local LLM manager cleanup, supervisor shutdown now attempts every owned runner before reporting stop failures, reconstructed FAILED runtime state prefers the persisted resolved failure model_path, new helper functions have explicit docstrings, and startup/shutdown test helpers now have type hints. Verification: supervisor/reconciler focused suite 27 passed; service lifecycle focused suite 14 passed; broader llama.cpp local/API suite 64 passed; shutdown resource cleanup suite 9 passed after final signature cleanup; git diff --check passed; ASCII scan found no non-ASCII in touched code; Bandit on touched app paths reported errors=0 and results=0.
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
