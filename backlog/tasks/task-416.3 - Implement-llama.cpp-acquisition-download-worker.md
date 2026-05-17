---
id: TASK-416.3
title: Implement llama.cpp acquisition download worker
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-17 22:22
labels:
- llamacpp
- backend
- local-llm
- jobs
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
parent_task_id: TASK-416
priority: high
modified_files:
- backlog/tasks/task-416.3 - Implement-llama.cpp-acquisition-download-worker.md
- tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py
- tldw_Server_API/app/services/llamacpp_acquisition_jobs_worker.py
- tldw_Server_API/app/services/startup_content_jobs_pollers.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py
- tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py
references:
- https://github.com/rmusser01/tldw_server/pull/1833
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the llama.cpp model acquisition/import workflow plan: add the Jobs-backed download worker, startup wiring, progress/cancel/cleanup behavior, and focused worker/startup tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Worker streams downloads to partial files, validates size/checksum, promotes atomically, and registers assets only after validation.
- [x] #2 Worker handles cancellation and validation failures by cleaning partial files and avoiding asset registration.
- [x] #3 Startup content jobs pollers can start/register the worker behind LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED with label llamacpp-acquisition.
- [x] #4 Focused worker/startup tests cover the Task 3 behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py -q --tb=short (82 passed, 5 warnings); source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py -v (21 passed, 5 warnings); source .venv/bin/activate && python -m bandit -r touched service paths -f json -o /tmp/bandit_llamacpp_acquisition_worker.json (0 findings); git diff --check (clean). Docs skip: Task 3 is worker/startup code only; docs are Task 5.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the llama.cpp acquisition download worker and wired it into content jobs startup. Covered successful download/register flow, checksum cleanup, cancellation cleanup, overwrite conflict handling, progress metadata, credential-safe errors, and worker inventory startup registration.
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
