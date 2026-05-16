---
id: TASK-397.2
title: Implement llama.cpp managed runtime stage 1
status: In Progress
assignee: []
created_date: '2026-05-16 01:43'
labels:
  - llamacpp
  - local-llm
  - webui
  - backend
dependencies:
  - TASK-397.1
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-managed-runtime-stage1-implementation-plan.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Stage 1 llama.cpp managed runtime plan: backend profile persistence, process runner, supervisor lifecycle, admin runtime APIs with V1 default-profile compatibility, minimal WebUI runtime panel, and focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime models and JSON profile store support default profile bootstrap and duplicate enabled explicit host/port conflict validation.
- [ ] #2 Single-instance process runner can start, stop, report status, and tail owned logs without per-instance atexit or signal handlers.
- [ ] #3 Supervisor can manage multiple profiles with per-profile locking, explicit lifecycle actions, and synchronous cleanup integration.
- [ ] #4 Admin profile/runtime APIs are admin-only and V1 llama.cpp endpoints remain compatible through the default profile.
- [ ] #5 Minimal WebUI client/types/runtime panel can display multiple instances and lifecycle actions while degrading on unsupported servers.
- [ ] #6 Focused backend/frontend tests, diff checks, and Bandit for touched Python code are run or documented with clear blockers.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Task 1 Notes

- Added `LlamaCppProfile` runtime models, profile store exceptions, and JSON profile persistence.
- Added default profile bootstrap plus enabled explicit host/port conflict validation.
- Added API schemas for profile, runtime, and lifecycle response payloads.
- Verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_profile_store.py -v` -> 4 passed, 5 warnings.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Local_LLM/llamacpp_runtime_models.py tldw_Server_API/app/core/Local_LLM/llamacpp_profile_store.py tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py -f json -o /tmp/bandit_llamacpp_profile_store.json` -> exit 0, results empty.
  - `git diff --check` -> exit 0.
