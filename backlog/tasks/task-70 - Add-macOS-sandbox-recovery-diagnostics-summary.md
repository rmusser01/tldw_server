---
id: TASK-70
title: Add macOS sandbox recovery diagnostics summary
status: Done
assignee: []
created_date: '2026-05-05 14:07'
labels:
  - sandbox
  - runtime-reliability
  - diagnostics
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow, additive operator-facing recovery_summary block to macOS sandbox diagnostics. The summary should derive from existing reconciliation, image-store, and observability diagnostics so admins can quickly see recovery posture, issue counts, issue codes, and the next safe action without diagnostics mutating state or generalizing repair beyond vz_linux.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /api/v1/sandbox/admin/macos-diagnostics exposes an additive recovery_summary field without removing or renaming existing diagnostics fields.
- [x] #2 The summary is read-only and derived from existing diagnostics payloads rather than re-querying helper/image-store state.
- [x] #3 The summary reports severity/posture, issue codes, stale/unhealthy/orphan/image-store counts, and a recommended next action for operators.
- [x] #4 The summary keeps repair guidance vz_linux-scoped and does not introduce a generic cross-runtime repair contract.
- [x] #5 Focused schema and diagnostics tests cover healthy, unavailable, stale/unhealthy/orphaned, and image-store cleanup candidate cases.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `summarize_recovery()` as a pure projection over already-collected macOS diagnostics blocks.
- Added `SandboxAdminMacOSRecoverySummary` and wired `recovery_summary` into macOS diagnostics responses.
- Updated sandbox README, runtime capability inventory, and macOS operator notes to document read-only recovery-summary semantics.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented an additive macOS sandbox `recovery_summary` diagnostics block for operator visibility. The summary reports healthy/action-recommended/unavailable posture, stable issue codes, counts, recommended action text, and pointers to existing dry-run-first repair or image-store cleanup-plan endpoints when relevant. It derives from existing reconciliation, image-store, and observability diagnostics without re-querying helper state or mutating data, preserving `vz_linux`-scoped repair semantics.

Verification passed: focused diagnostics pytest reported 26 passed; py_compile passed for touched production Python; Ruff passed on touched source/test files; Bandit reported zero findings for touched production Python; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->
