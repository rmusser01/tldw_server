---
id: TASK-397.2
title: Implement llama.cpp managed runtime stage 1
status: In Progress
assignee: []
created_date: '2026-05-16 01:43'
updated_date: '2026-05-16 02:25'
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
- [x] #2 Single-instance process runner can start, stop, report status, and tail owned logs without per-instance atexit or signal handlers.
- [ ] #3 Supervisor can manage multiple profiles with per-profile locking, explicit lifecycle actions, and synchronous cleanup integration.
- [ ] #4 Admin profile/runtime APIs are admin-only and V1 llama.cpp endpoints remain compatible through the default profile.
- [ ] #5 Minimal WebUI client/types/runtime panel can display multiple instances and lifecycle actions while degrading on unsupported servers.
- [ ] #6 Focused backend/frontend tests, diff checks, and Bandit for touched Python code are run or documented with clear blockers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1: added LlamaCppProfile runtime models, profile store exceptions, JSON profile persistence, default profile bootstrap, enabled explicit host/port conflict validation, and API schemas for profile/runtime/lifecycle payloads. Verification: profile-store pytest initially 4 passed; Bandit on Task 1 Python paths reported no findings; git diff --check passed.

Task 1 quality review fixes: malformed dict-shaped profile stores now fail closed without overwrite, wildcard bind host/port conflicts are rejected, and profile-store tests now cover persistence round-trip, update replacement, get miss, delete true/false, corrupt structure, and wildcard conflicts. Verification: profile-store pytest 9 passed; Bandit profile-store review fix output has no findings; git diff --check passed.

Task 2: added LlamaCppProcessRunner with independent process lifecycle, profile port policy handling, allowlist/path checks, owned log tailing, sync cleanup, and runtime state payloads. Verification: py_compile passed; process runner + management + inventory pytest reported 39 passed; Bandit on runner/runtime models had no findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
