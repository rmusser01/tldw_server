---
id: TASK-304.10
title: Implement Research Studio docs and release verification
status: Done
assignee: []
created_date: '2026-05-12 23:33'
updated_date: '2026-05-13 00:29'
labels:
  - implementation
  - research-studio
  - webui
  - docs
  - verification
dependencies:
  - TASK-304.9
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs use /research-studio as canonical and legacy paths only as aliases where useful
- [x] #2 Docs use /research-studio?tab=studio for mobile Studio deep links
- [x] #3 Remaining Workspace Playground/Workspace Studio route and label search hits are classified or fixed
- [x] #4 Focused release verification and CDP/browser checks are recorded with blockers separated from completed checks
- [x] #5 Bandit/test skips are recorded when only frontend/docs/task files change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Search apps and Docs for remaining Workspace Playground/Workspace Studio legacy references outside generated Docs/site.
2. Inspect Research Studio docs, tutorial docs, route inventory, and smoke specs before editing.
3. Update only user-facing docs/release inventory references that should now be canonical, preserving intentional internal compatibility names.
4. Run focused docs/search/test checks and CDP/browser smoke if local app can be launched.
5. Record verification, blockers, and final summary in Backlog, then commit the release-verification slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated current Quick Chat/tutorial coverage docs, Published mirrors, extension route inventory, QA checklist, and shared WebUI/extension parity helper to use /research-studio as canonical while preserving workspace-playground internal compatibility names.

Search classification: remaining Workspace Playground/Workspace Studio hits are internal compatibility names, legacy alias route files/tests, tutorial id workspace-playground-basics, deterministic test filenames/helpers, or historical review/plan artifacts. Current release docs and inventories touched by this slice now use /research-studio.

Verification: git diff --check passed; frontend route-registry test passed (2 tests); shared UI workflow-guide/tutorial/route-state/responsive tests passed (42 tests); focused WebUI Playwright parity passed against this worktree on localhost:3002 (1 test).

CDP smoke: /research-studio rendered; /workspace-playground?shared=alias-test canonicalized to /research-studio?shared=alias-test; /workspace-studio?tab=studio canonicalized to /research-studio?tab=studio; mobile /research-studio?tab=studio showed Studio selected with no-source CTA. Screenshots saved under /private/tmp/research-studio-*.png.

Known caveat: manual CDP smoke saw repeated 401s from backend-dependent requests because the running single-user backend rejected the seeded demo key. /api/v1/health itself returned degraded and route rendering/alias/mobile tab checks were still observable.

Bandit not run: this slice changes frontend tests/helpers and documentation/task records only; no backend Python files touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned current Research Studio docs, tutorial coverage, extension route inventory, and deterministic parity helpers with /research-studio as canonical, then recorded focused test and CDP release evidence with the local auth caveat separated.
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
