---
id: TASK-397
title: Design llama.cpp managed runtime roadmap
status: Done
assignee: []
created_date: '2026-05-16 01:20'
updated_date: '2026-05-29 05:45'
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

Post-rollout closeout: the downstream managed-runtime implementation, asset inventory, model-family/mmproj metadata, capability visibility, profile editor, runtime reconciliation, validation hardening, API compatibility, docs, and smoke coverage have all landed through follow-up tasks. The unrelated completed Sync v2 attachment task that previously reused `TASK-397` was moved to `TASK-490.14`, so `TASK-397` is again the unambiguous llama.cpp managed-runtime roadmap parent.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The llama.cpp managed-runtime roadmap design is complete and has been carried through implementation follow-ups. The design spec covers backend-owned runtime/profile persistence, V1 default-profile compatibility, local asset import/register workflows, supervisor behavior, multimodal/mmproj/model-family boundaries, WebUI workflow, staged delivery, and test strategy. Follow-up tasks implemented and verified the planned runtime, inventory, profile, metadata, capability, reconciliation, compatibility, and rollout slices. Tracker cleanup also resolved the duplicate `TASK-397` ID collision by moving the unrelated Sync v2 attachment record to `TASK-490.14`.
<!-- SECTION:FINAL_SUMMARY:END -->
