---
id: TASK-596
title: Address snapshot default assignment import follow-up
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:21'
labels:
  - mcp-unified
  - standalone-gateway
  - review-fix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify PR review feedback for GatewayConfigSnapshotManager default assignment imports, fix still-valid paths where snapshot model instances can bypass default assignment id validation, and validate the touched snapshot surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review finding verified against current snapshot import paths.
- [x] #2 Still-valid constructed-model import path rejected before writes.
- [x] #3 Regression test added for non-gateway default assignment id import.
- [x] #4 Snapshot/CLI tests, Bandit, and diff check recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified normal mapping snapshots were already rejected by GatewayConfigSnapshot validation. The still-valid issue was the direct GatewayConfigSnapshot instance path, where model_construct could bypass validation before GatewayConfigSnapshotManager._mutation_actions wrote the assignment. Added manager-level validation and canonical default assignment ids for planned and mutation actions.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the still-valid snapshot import path where constructed GatewayConfigSnapshot instances could carry a non-gateway default assignment id into the manager. Validation now rejects that before mutation, and plan/mutation actions use the canonical gateway default assignment id. Verification: focused regression red/green; 107 snapshot/CLI tests passed; Bandit results empty for mcp_unified/gateway/snapshots.py; git diff --check clean.
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
