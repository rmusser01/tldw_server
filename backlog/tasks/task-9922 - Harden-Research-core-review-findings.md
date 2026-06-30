---
id: TASK-9922
title: Harden Research core review findings
status: Done
assignee: []
created_date: '2026-06-23 18:28'
updated_date: '2026-06-23 18:58'
labels:
  - research
  - security
  - hardening
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated current-state review findings in tldw_Server_API/app/core/Research: worker per-user store routing, owner scoping, checkpoint replay guards, artifact version durability, cancellation responsiveness, and limits enforcement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings are either fixed with focused tests or documented as not applicable after verification.
- [x] #2 Research service APIs enforce owner scoping for session reads, artifacts, control actions, checkpoint approval, and package builds.
- [x] #3 Research Jobs worker can process API-created per-user sessions without shared-path mismatch.
- [x] #4 Artifact version storage preserves historical versions and avoids version race regressions.
- [x] #5 Pause/cancel and configured research limits are enforced during long-running phases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-06-23-research-core-review-hardening-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification pass confirmed all six review findings were present in current code before fixes. Implemented owner-scoped service reads/actions, checkpoint replay guards, per-user Jobs worker path resolution, unique immutable artifact storage files with latest aliases, cooperative cancellation checks, and collection budget enforcement. Added focused regression coverage in tldw_Server_API/tests/Research/test_research_core_hardening.py. PR review follow-up: async Research job handlers now offload artifact writes through asyncio.to_thread to avoid blocking the event loop. Verification: hardening pytest 10 passed; compatibility pytest slice 10 passed; py_compile passed; git diff --check passed; Bandit JSON reported errors=0 and results=0. Known skip/workaround: default pytest cleanup with the unraisableexception plugin hung during earlier red runs, so focused verification used -p no:unraisableexception.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and fixed all six Research core review findings plus PR review feedback. Owner checks now gate session reads, artifacts, control actions, checkpoint approval, and package builds; checkpoint approvals require the current pending checkpoint in the expected waiting phase; Jobs worker paths resolve per owner from job payloads; artifact rows point at unique immutable files while preserving latest-name aliases; long collection/synthesis work checks cancellation; collection enforces configured search, fetched-doc, and runtime budgets; async job handlers offload artifact writes instead of blocking the event loop. Verification passed: 10 focused hardening tests, 10 compatibility tests, py_compile, git diff --check, and Bandit errors=0 results=0.
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
