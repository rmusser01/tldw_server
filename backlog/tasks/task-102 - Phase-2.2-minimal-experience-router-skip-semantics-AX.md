---
id: TASK-102
title: Phase 2.2 minimal experience router skip semantics AX
status: Done
assignee: []
created_date: '2026-05-07 02:15'
updated_date: '2026-05-07 02:31'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1348'
  - 'https://github.com/rmusser01/tldw_server/pull/1352'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Narrow minimal experience router optional skip behavior after PR #1348 merged. The sharing, personalization, and companion ImportedRouterSpec entries should skip only genuinely missing optional router modules or missing router attributes, while runtime import defects propagate during registration instead of being hidden by skip_exceptions=(Exception,).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal experience router specs no longer use skip_exceptions=(Exception,)
- [x] #2 Missing target experience modules are skipped during registration
- [x] #3 Runtime import defects from experience router modules propagate
- [x] #4 Existing prefixes tags route keys and default stability behavior are preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update focused contract tests for precise experience optional skip semantics. 2. Verify RED against current broad skip behavior. 3. Remove broad skip_exceptions from experience ImportedRouterSpec entries. 4. Run focused and broader verification plus Bandit and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline focused selector passed before edits: minimal experience tests 2 passed, confirming current broad skip behavior is covered.

TDD RED after test update failed as intended: experience specs still reported Exception and runtime import failures were skipped. GREEN focused selector passed after removing experience broad skip overrides: 3 passed.

Broader validation passed: router groups 121 passed, lifecycle 54 passed, OpenAPI 69 passed, Bandit results 0, git diff --check clean.

Scope note: remaining skip_exceptions=(Exception,) occurrences are in guardian_safety and utility minimal groups and should be handled in later narrow slices.

Opened PR https://github.com/rmusser01/tldw_server/pull/1352 against dev for this slice.

PR review follow-up: addressed valid test maintainability comments by asserting concrete optional skip exception classes, replacing exact debug-message equality with semantic log checks, and parameterizing runtime propagation across sharing, personalization, and companion.

Review-fix validation: focused minimal experience selector passed 5 selected tests; full router group contract passed 123 tests; Bandit on minimal.py reported 0 results; git diff --check was clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed the minimal experience router ImportedRouterSpec entries for sharing, personalization, and companion by removing broad skip_exceptions=(Exception,) overrides. The experience specs now use the default precise optional-missing skip exceptions, so missing target modules still skip during registration while runtime import defects propagate. Validation: focused experience selector 3 passed after RED verification, router group contracts 121 passed, main lifecycle contracts 54 passed, OpenAPI contracts 69 passed, Bandit on minimal.py found 0 results, and git diff --check was clean.

Review follow-up tightened the test contract without changing production behavior: exception assertions now compare concrete classes, missing-module debug checks are semantic, and runtime propagation is covered for all three experience routers. Review-fix validation: focused selector 5 passed, router group contracts 123 passed, Bandit 0 results, and git diff --check clean.
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
