---
id: TASK-397
title: Design llama.cpp managed runtime roadmap
status: Done
assignee: []
created_date: '2026-05-16 01:20'
updated_date: '2026-05-23 11:28'
labels:
  - llamacpp
  - design
  - webui
  - local-llm
dependencies: []
references:
  - 'https://github.com/m94301/llama-studio'
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - Docs/superpowers/plans/2026-05-16-llamacpp-managed-runtime-stage1-implementation-plan.md
  - Docs/superpowers/plans/2026-05-17-llamacpp-managed-runtime-implementation-plan.md
modified_files:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - backlog/tasks/task-397 - Design-llama.cpp-managed-runtime-roadmap.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved roadmap design spec for expanding the merged llama.cpp WebUI management work into a backend-owned managed runtime. The spec must cover multi-instance management, local model import/register workflows, durable instance profiles, supervisor behavior, and multimodal/mmproj/model-family support while preserving the existing V1 single-server API compatibility path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec is added under Docs/superpowers/specs with the approved roadmap architecture and staged delivery plan.
- [x] #2 The spec explicitly documents V1 compatibility wrappers around a default instance profile.
- [x] #3 The spec covers assets, instance profiles, runtime state, supervisor safety behavior, model-family/mmproj support, UX workflow, and testing strategy.
- [x] #4 The spec states that remote model downloads are deferred behind local import/register workflows.
- [x] #5 The written spec is reviewed for internal consistency before implementation planning begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md with the approved managed-runtime roadmap: asset inventory, instance profiles, supervisor lifecycle, V1 wrappers, local import/register first, future download requirements, model-family/mmproj support, WebUI workflow, staged delivery, and testing strategy.

Local consistency review completed: checked the spec for the requested compatibility/default-profile, download deferral, mmproj, supervisor, profile, and testing coverage. `git diff --check -- Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md backlog/tasks/task-397\ -\ Design-llama.cpp-managed-runtime-roadmap.md` passed with no output.

Bandit not run for this task because the current change is documentation/backlog only and touches no Python code.

Spec critique pass completed before implementation planning. Added design clarifications for backend-owned persistence vs config.txt, admin-only/deployment-global scope, multi-user provider-wiring constraints, asset identity and symlink allowlist handling, explicit port policy, default-profile migration behavior, per-profile lifecycle locking, shutdown/orphan behavior, reserved mode flags, local folder import semantics, and extra E2E coverage for duplicate ports and V1 wrapper behavior.

2026-05-23 closeout review:
- Re-reviewed the roadmap spec against all acceptance criteria and current `origin/dev`.
- Updated the spec status from draft to closed for staged implementation tracking.
- Confirmed follow-on planning and implementation evidence exists through `TASK-397.1`, `TASK-397.2`, `TASK-397.5`, `TASK-397.6`, `TASK-397.7`, `TASK-397.8`, `TASK-407`, `TASK-418`, `TASK-418.14`, `TASK-418.15`, `TASK-418.16`, `TASK-419`, and `TASK-423`.
- Confirmed current code contains the expected backend/runtime/WebUI ownership points: profile store, process runner, supervisor service, runtime reconciler startup/shutdown hooks, Admin assets/runtime/profile panels, llama.cpp runtime API tests, and rollout E2E smoke coverage.
- Backlog note: this repository contains a duplicate `TASK-397` ID, so MCP `task_view` resolves `TASK-397` to an unrelated Sync task. This llama.cpp task was updated by exact file path to avoid modifying the wrong record.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the llama.cpp managed runtime roadmap design task. The approved spec covers the requested roadmap architecture, default-profile V1 compatibility, local-first asset/profile/runtime supervision model, model-family and mmproj support, UX workflow, testing strategy, and explicit remote-download deferral. Follow-on work has already carried the roadmap through staged planning, Stage 1 runtime/profile APIs, asset inventory, model-family/mmproj metadata, Admin UI capability/profile surfaces, runtime reconciliation, validation/API hardening, rollout docs, and E2E smoke coverage. Verification for this closeout used exact-path Backlog review because of the duplicate `TASK-397` ID, current-code ownership `rg` checks, unfinished-marker checks, and `git diff --check`; Bandit is not applicable because only Markdown/Backlog metadata changed.
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
