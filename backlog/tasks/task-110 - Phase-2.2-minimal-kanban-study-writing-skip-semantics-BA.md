---
id: TASK-110
title: Phase 2.2 minimal kanban study writing skip semantics BA
status: Done
assignee: []
created_date: '2026-05-07 05:23'
updated_date: '2026-05-07 05:30'
labels:
  - phase-2.2
  - router-groups
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1358'
  - 'https://github.com/rmusser01/tldw_server/pull/1360'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Narrow the remaining minimal optional router skip semantics for Kanban, study, and writing/email router specs so missing optional modules/attrs remain skippable but internal runtime ImportError/AttributeError defects are not hidden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal Kanban, study, and writing/email optional router specs use the default OptionalRouterMissingModule/OptionalRouterMissingAttribute skip behavior instead of raw ImportError/AttributeError skips.
- [x] #2 Focused router contract tests cover missing-target skips and propagation of internal ImportError/AttributeError runtime defects for the touched spec groups.
- [x] #3 Existing router group, main router, and OpenAPI contracts still pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green evidence: focused selector 'minimal_optional_router_specs and (kanban or study or writing_email)' failed before production changes with 9 expected failures for raw skip_exceptions and swallowed ImportError/AttributeError defects, then passed after removing the raw skip overrides (15 passed, 121 deselected).

Validation: router group contracts 136 passed; main router contracts 6 passed; OpenAPI contracts 69 passed; Bandit on tldw_Server_API/app/api/v1/router_groups/minimal.py wrote /tmp/bandit_phase2_2_minimal_learning_kanban_skip_semantics_ba.json with results [] and errors []; git diff --check clean; no remaining skip_exceptions=(ImportError, AttributeError) or skip_exceptions=(Exception) matches in router_groups.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the remaining raw ImportError/AttributeError skip overrides from minimal Kanban, study, and writing/email ImportedRouterSpec entries so only target missing-module/missing-attribute sentinel failures are skippable. Updated focused contract tests to use ModuleNotFoundError for missing optional modules, assert default sentinel skip behavior, and verify RuntimeError, ImportError, and AttributeError defects propagate for each touched group.
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
