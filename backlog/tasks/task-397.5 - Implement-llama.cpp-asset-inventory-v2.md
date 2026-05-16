---
id: TASK-397.5
title: Implement llama.cpp asset inventory v2
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-16 06:37'
labels:
  - llamacpp
  - backend
  - webui
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the merged Asset Inventory V2 plan: local asset schemas, imported folder config parsing, GGUF/mmproj/folder asset discovery, stale-path warnings, candidate mmproj pairing, asset register/import endpoints, legacy inventory compatibility, frontend API/types, and a minimal Admin assets panel. Remote downloads and model-family routing remain deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend exposes asset schemas, imported folder config parsing, and local asset scanning for GGUF, mmproj, folder, and unknown assets.
- [ ] #2 Backend supports admin-only asset list/register-path/import-folder endpoints while preserving legacy inventory and start-by-model compatibility.
- [ ] #3 Asset discovery reports stale-path, allowlist, unknown-capability, and inferred mmproj pairing warnings without remote download behavior.
- [ ] #4 WebUI shared client/types and Admin page expose a minimal assets panel with register/import actions and warnings.
- [ ] #5 Focused backend, frontend, Bandit, and diff verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Executing Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md inline with TDD in worktree .worktrees/llamacpp-asset-inventory-v2.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
