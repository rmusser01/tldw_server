---
id: TASK-397.2
title: Implement llama.cpp managed runtime stage 1
status: In Progress
assignee: []
created_date: '2026-05-16 01:43'
updated_date: '2026-05-16 03:36'
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
- [x] #3 Supervisor can manage multiple profiles with per-profile locking, explicit lifecycle actions, and synchronous cleanup integration.
- [x] #4 Admin profile/runtime APIs are admin-only and V1 llama.cpp endpoints remain compatible through the default profile.
- [ ] #5 Minimal WebUI client/types/runtime panel can display multiple instances and lifecycle actions while degrading on unsupported servers.
- [ ] #6 Focused backend/frontend tests, diff checks, and Bandit for touched Python code are run or documented with clear blockers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1: added LlamaCppProfile runtime models, profile store exceptions, JSON profile persistence, default profile bootstrap, enabled explicit host/port conflict validation, and API schemas for profile/runtime/lifecycle payloads. Verification: profile-store pytest initially 4 passed; Bandit on Task 1 Python paths reported no findings; git diff --check passed.

Task 1 quality review fixes: malformed dict-shaped profile stores now fail closed without overwrite, wildcard bind host/port conflicts are rejected, and profile-store tests now cover persistence round-trip, update replacement, get miss, delete true/false, corrupt structure, and wildcard conflicts. Verification: profile-store pytest 9 passed; Bandit profile-store review fix output has no findings; git diff --check passed.

Task 2: added LlamaCppProcessRunner with independent process lifecycle, profile port policy handling, allowlist/path checks, owned log tailing, sync cleanup, and runtime state payloads. Verification: py_compile passed; process runner + management + inventory pytest reported 39 passed; Bandit on runner/runtime models had no findings; git diff --check passed.

Task 2 review fixes: retained failed-start runtime details with FAILED status and redacted resolved args, expanded runtime response contract fields, rejected occupied explicit ports before spawn, drained stdout/stderr pipes when no log file is configured, validated model_draft/lora_scaled paths, and restored existing LlamaCppHandler server-arg aliases. Verification: py_compile passed; process runner/profile store/management/inventory pytest reported 54 passed; Bandit on runner/runtime/schema had no findings; git diff --check passed.

Task 2 second review fixes: changed default pipe drainers from readline to bounded read(1024) so long/no-newline output cannot stop draining, and filtered None/empty profile server_args before command construction to match existing LlamaCppHandler behavior. Verification: focused regressions passed; py_compile passed; process runner/profile store/management/inventory pytest reported 55 passed; Bandit on runner/runtime/schema had no findings; git diff --check passed.

Task 3: added LlamaCppSupervisor with profile CRUD, per-profile lifecycle locks, independent start/stop/pause/resume/shutdown/cleanup behavior, runtime listing, default-profile bridge helpers, and LLMInferenceManager cleanup integration. Verification: supervisor pytest passed; process runner/profile store/management/inventory regression suite reported 61 passed; py_compile passed; Bandit on supervisor/manager had no findings; git diff --check passed.

Task 3 review fixes: made profile create/update/delete and default profile ensure asynchronous under the per-profile lock, held the default lock across default-profile update plus restart, added a supervisor-wide start lock for autoselect port selection, preserved legacy LlamaCppHandler cleanup while supervisor cleanup is enabled, and removed the core supervisor dependency on API request schemas. Verification: focused supervisor pytest reported 10 passed; touched llama.cpp regression slice reported 65 passed; py_compile passed; Bandit on supervisor/manager had no findings; git diff --check passed.

Task 3 second quality review fixes: validated profile update payloads through LlamaCppProfile before persistence, added supervisor-wide store write serialization, and changed profile deletion to await runner.stop before removing runner/profile ownership. Verification: focused supervisor pytest reported 12 passed; touched llama.cpp regression slice reported 67 passed; py_compile passed; Bandit on supervisor/manager had no findings; git diff --check passed.

Task 4: added admin llama.cpp profile and instance APIs, per-profile lifecycle actions, instance log tailing, supervisor resolver/error mapping, and V1 default-profile routing for start-by-model, stop/status, logs, and use-in-chat while preserving handler/manager fallback compatibility. Verification: runtime API pytest reported 4 passed; Task 4 compatibility set reported 38 passed; broader llama.cpp backend slice reported 84 passed; py_compile passed; Bandit on endpoint/supervisor had no findings; git diff --check passed.
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
