---
id: TASK-394.1
title: Map Quick Ingest active path and legacy tests
status: Done
assignee: []
created_date: '2026-05-16 00:42'
updated_date: '2026-05-29 01:04'
labels:
  - quick-ingest
  - ux
  - audit
  - task-1
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
  - Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md
parent_task_id: TASK-394
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 1: recover and document the active quick-ingest workflow, entry points, services, tests, and legacy/stale surfaces before code changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active path map exists with entry points, modal flow, API services, persistence, extension hooks, and tests
- [x] #2 Legacy/stale tests and surfaces are classified without product-code changes
- [x] #3 Task 1 verification evidence is recorded
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Active path map already exists on origin/dev at Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md and satisfies this slice: it records active WebUI/extension launch paths, active modal/runtime ownership, background/session hooks, legacy QuickIngestModal reachability, stale legacy tests, and active test coverage homes.

Verification rerun on latest origin/dev: `rg -n "Active Launch Paths|Legacy Reachability Decision|Test Classification" Docs/superpowers/plans/2026-05-16-quick-ingest-active-path-map.md` found all required sections; `rg -n "QuickIngestWizardModal|QuickIngestModal|tldw:open-quick-ingest|open-quick-ingest|quick-ingest-run|quick-ingest-cancel|quick-ingest-open-media-primary" apps/tldw-frontend apps/packages/ui/src apps/extension/tests/e2e` succeeded and confirms current evidence still resolves. Bandit is not applicable because this closeout only updates Backlog tracking for an existing documentation artifact.
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
