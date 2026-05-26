---
id: TASK-397
title: Design llama.cpp managed runtime roadmap
status: In Progress
assignee: []
created_date: '2026-05-16 01:20'
updated_date: '2026-05-16 01:29'
labels:
  - llamacpp
  - design
  - webui
  - local-llm
dependencies: []
references:
  - 'https://github.com/m94301/llama-studio'
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
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
