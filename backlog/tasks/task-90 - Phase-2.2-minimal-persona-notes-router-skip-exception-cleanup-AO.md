---
id: TASK-90
title: Phase 2.2 minimal persona/notes router skip exception cleanup AO
status: Done
assignee: []
created_date: '2026-05-05 23:10'
updated_date: '2026-05-05 23:13'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1329'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tighten minimal-test persona, archetype, and notes optional router registration so missing optional imports remain skippable while runtime defects propagate during lazy router resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal persona, archetype, and notes router specs skip only ImportError and AttributeError during lazy resolution.
- [x] #2 Runtime exceptions raised while importing those router modules propagate instead of being silently skipped.
- [x] #3 Existing prefixes, tags, route keys, and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup, missing import skipping, and runtime error propagation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline persona/notes focused contract tests passed before edits: 2 passed, 92 deselected. RED after test update: 2 failed and 1 passed; failures showed persona/notes still used skip_exceptions=(Exception,) and runtime import errors were still skipped. GREEN after implementation: focused persona/notes contract tests passed with 3 passed, 92 deselected.

Verification: full test_router_groups_contract.py passed with 95 passed; test_main_router_contract.py passed with 6 passed; test_openapi_contracts.py passed with 69 passed; Bandit on minimal.py wrote /tmp/bandit_minimal_persona_notes_ao.json with empty errors/results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed minimal-test persona, archetype, and notes ImportedRouterSpec skip_exceptions from broad Exception to ImportError and AttributeError. Added focused contract coverage for lazy attr lookup metadata, missing optional import skipping, and runtime import error propagation while preserving prefixes, tags, route keys, and default stability.
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
