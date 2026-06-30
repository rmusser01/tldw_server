---
id: TASK-70
title: Phase 2.2 PR 1301 router registry failure handling
status: Done
assignee: []
created_date: '2026-05-05 14:14'
updated_date: '2026-05-05 14:18'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
  - pr-1301
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1301'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR #1301 review feedback that register_router_specs currently swallows every lazy router resolution exception. Add coverage and update the shared registry/conditional spec seam so optional missing imports or missing router attributes are skipped, while unexpected lazy factory or module import runtime errors are re-raised instead of silently omitting routers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optional ImportedRouterSpec ModuleNotFoundError and missing router AttributeError remain skipped with existing diagnostics
- [x] #2 Unexpected RuntimeError from ImportedRouterSpec import resolution is re-raised
- [x] #3 Generic RouterSpec factories can opt into skippable exceptions explicitly
- [x] #4 Focused and full router-group contract tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing registry tests for unexpected lazy resolution failures versus configured skippable exceptions. 2. Add skip_exceptions metadata to RouterSpec/ImportedRouterSpec. 3. Update register_router_specs to skip only configured resolution exceptions and re-raise unexpected ones. 4. Rerun focused/full contract tests and Bandit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented registry failure handling review fix. RouterSpec and ImportedRouterSpec now carry skip_exceptions metadata defaulting to ImportError and AttributeError. register_router_specs re-raises unexpected resolution failures and only skips configured exception types. Updated contract tests for imported router RuntimeError propagation, core chat-loop crash propagation, and explicit opt-in skip behavior.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1301 Qodo reliability finding by preventing unexpected lazy router resolution failures from being silently skipped. Verification: focused red run failed as expected before implementation; focused green passed; full router_groups contract suite passed; main router and OpenAPI contracts passed; Bandit on changed registry/router-group source reported 0 results and 0 errors; git diff --check clean.
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
