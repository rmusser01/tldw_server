---
id: TASK-90
title: Phase 2.2 minimal persona/notes router skip exception cleanup AO
status: Done
assignee: []
created_date: '2026-05-05 23:10'
updated_date: '2026-05-05 23:49'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1329'
  - 'https://github.com/rmusser01/tldw_server/pull/1331'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tighten minimal-test persona, archetype, and notes optional router registration so missing optional imports remain skippable while runtime defects propagate during lazy router resolution.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal persona archetype and notes router specs skip only dedicated optional missing module or missing router attribute exceptions.
- [x] #2 Runtime exceptions and internal import-time dependency failures raised while importing those modules propagate instead of being skipped.
- [x] #3 Existing prefixes tags route keys and default stability behavior are preserved.
- [x] #4 Focused router contract tests cover lazy attr lookup true missing optional imports internal import failures and runtime error propagation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline persona/notes focused contract tests passed before edits: 2 passed, 92 deselected. RED after test update: 2 failed and 1 passed; failures showed persona/notes still used skip_exceptions=(Exception,) and runtime import errors were still skipped. GREEN after implementation: focused persona/notes contract tests passed with 3 passed, 92 deselected.

Verification: full test_router_groups_contract.py passed with 95 passed; test_main_router_contract.py passed with 6 passed; test_openapi_contracts.py passed with 69 passed; Bandit on minimal.py wrote /tmp/bandit_minimal_persona_notes_ao.json with empty errors/results.

PR review follow-up: Qodo flagged that ImportError/AttributeError skip types still hide internal import-time bugs. Reopening task to address the still-valid review comment with precise optional-router exceptions.

Review RED: focused helper/persona selection failed with 4 failures before implementation. Failures showed default skip types were still ImportError/AttributeError and nested ModuleNotFoundError plus import-time AttributeError were still swallowed.

Review GREEN: introduced OptionalRouterMissingModule and OptionalRouterMissingAttribute wrappers in conditional.py. The helper now wraps only target module misses and missing router attrs while internal ModuleNotFoundError and import-time AttributeError propagate. Persona/archetype/notes specs inherit the precise default.

Review verification: focused append_imported_router_spec/persona_notes selection passed with 13 passed; full test_router_groups_contract.py passed with 98 passed; test_main_router_contract.py passed with 6 passed; test_openapi_contracts.py passed with 69 passed; Bandit on conditional.py and minimal.py wrote /tmp/bandit_minimal_persona_notes_review_ao.json with empty errors/results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed minimal-test persona, archetype, and notes ImportedRouterSpec skip_exceptions from broad Exception to ImportError and AttributeError. Added focused contract coverage for lazy attr lookup metadata, missing optional import skipping, and runtime import error propagation while preserving prefixes, tags, route keys, and default stability.

PR opened: https://github.com/rmusser01/tldw_server/pull/1331

Review follow-up: addressed Qodo's import-time bug hiding comment by replacing broad optional ImportedRouterSpec defaults with dedicated optional-missing exceptions and updating persona/notes coverage for internal import failures.
<!-- SECTION:FINAL_SUMMARY:END -->
