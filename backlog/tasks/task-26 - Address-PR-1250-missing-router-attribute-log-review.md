---
id: TASK-26
title: Address PR 1250 missing router attribute log review
status: Done
assignee: []
created_date: '2026-05-04 02:32'
updated_date: '2026-05-04 02:35'
labels:
  - phase-2.2
  - review
  - router-groups
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1250'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the PR #1250 review finding that ImportedRouterSpec missing-attribute failures only log the attribute name. Preserve lazy router behavior while improving the exception/log message to include the import path and expected attribute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing ImportedRouterSpec router attribute failures include both module import path and attribute name in registration debug logs.
- [x] #2 Existing optional import and lazy lookup behavior is preserved.
- [x] #3 Focused router-group tests pass after the change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review thread verified via gh: Qodo reported missing module path in ImportedRouterSpec missing-attribute logs. Added a red regression by changing missing-attr expectations from bare attr name to module.attr, then updated append_imported_router_spec to raise AttributeError with import_path.attr_name while preserving the original AttributeError as cause.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1250 review finding by making lazy ImportedRouterSpec missing-attribute failures log the full module attribute path. Verification: red focused missing_attr test, green append_imported_router_spec/missing_attr selection, full router_groups contract, main_router contract, OpenAPI contracts, Bandit on conditional.py/content.py, and git diff --check.
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
