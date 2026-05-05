---
id: TASK-67
title: Phase 2.2 minimal auxiliary router conditional cleanup AC
status: Done
assignee: []
created_date: '2026-05-05 05:33'
updated_date: '2026-05-05 05:36'
labels:
  - phase2.2
  - router-cleanup
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1296'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1116 Phase 2.2 by converting the first remaining auxiliary minimal-test optional router registrations from eager try/import RouterSpec blocks to ImportedRouterSpec-backed lazy router specs. Scope is limited to chunking_templates, prompts, claims, text2sql, feedback, vlm, consent, outputs_templates, and outputs in tldw_Server_API/app/api/v1/router_groups/minimal.py. Preserve existing prefixes, tags, route keys, skip context, and current skip-on-import-failure behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected auxiliary minimal optional router specs defer module import and router attribute lookup until registration
- [x] #2 Existing route metadata is preserved for chunking-templates prompts claims text2sql feedback vlm consent outputs-templates and outputs
- [x] #3 Focused router-group test covers the lazy behavior with red/green verification
- [x] #4 Router-group main-router and OpenAPI contract tests pass for the touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a narrow minimal auxiliary router tranche. Added red/green router-group coverage that proves chunking_templates, prompts, claims, text2sql, feedback, vlm, consent, outputs_templates, and outputs defer module import and router attribute access until ImportedRouterSpec resolution. Replaced only those eager try/import RouterSpec blocks in minimal.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Converted the selected minimal auxiliary optional router registrations to ImportedRouterSpec while preserving prefixes and tags. Verification: focused red/green test, full router_groups contract suite, main router contract suite, OpenAPI contracts, Bandit on minimal.py, and git diff --check.
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
