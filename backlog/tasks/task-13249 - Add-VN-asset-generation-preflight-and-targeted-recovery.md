---
id: TASK-13249
title: Add VN asset generation preflight and targeted recovery
status: In Progress
assignee: []
created_date: '2026-09-13 18:27'
updated_date: '2026-09-13 18:56'
labels:
  - vn-assets
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
documentation:
  - Docs/Design/2026-09-13-vn-generation-readiness.md
  - Docs/Evidence/VN_Generation_Readiness_2026_09_13.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First productization slice for issue 2021: expose server-owned generation preflight and actionable configuration diagnostics, wire targeted slot retry using existing idempotent API, and verify state recovery. Reconcile issues 2021-2027 and parent links as accompanying tracking work. Full recipe snapshot and worker crash-replay hardening remain in 2021.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pack generation preflight reports effective backend availability and worker configuration without asserting external-worker health.
- [x] #2 The WebUI displays actionable generation failures and supports targeted slot retry with duplicate submission protection.
- [x] #3 Focused backend and frontend tests plus browser QA verify changed behavior; Bandit and relevant type checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented advisory owner-scoped preflight, stable generation keys, targeted retry, active progress polling and pack-switch guards. Mobile browser QA found and fixed implicit grid overflow. Independent review identified four refresh races; failing regressions reproduced each and all are fixed. Final re-review found no remaining issues in those fixes. Verification: 90 backend tests, 31 frontend tests, 3 Chromium smoke tests; touched ESLint and Bandit clean. Full typecheck fails identically on unchanged dev (90 existing diagnostics). See evidence document for commands, before/after observations and limits. Draft PR packaging in progress; human-written Change summary required before merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the bounded generation-readiness and targeted-recovery implementation for #2021. All scoped tests pass. Issues #2021-#2027 were reconciled and registered as children of #1391. Recipe snapshots, reload recovery and worker crash replay remain outside this slice; #2021 stays open. No live provider/GPU/worker deployment or extension build was exercised.
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
