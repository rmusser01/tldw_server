---
id: TASK-107
title: Phase 2.2 minimal guardian safety router skip semantics AZ
status: Done
assignee: []
created_date: '2026-05-07 04:52'
updated_date: '2026-05-07 05:00'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1355'
  - 'https://github.com/rmusser01/tldw_server/pull/1358'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Narrow minimal guardian/safety router optional skip behavior after PR #1355 merged. The guardian controls, family wizard, and self monitoring ImportedRouterSpec entries should skip genuinely missing optional target modules or missing router attributes, while runtime import defects propagate during registration instead of being hidden by skip_exceptions=(Exception,).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal guardian/safety router specs no longer use skip_exceptions=(Exception,)
- [x] #2 Missing target guardian/safety modules are skipped during registration
- [x] #3 Runtime import defects from guardian/safety router modules propagate
- [x] #4 Existing prefixes tags route keys and default stability behavior are preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update focused contract tests for precise guardian/safety optional skip semantics. 2. Verify RED against current broad skip behavior. 3. Remove broad skip_exceptions from guardian/safety ImportedRouterSpec entries. 4. Run focused and broader verification plus Bandit and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline focused selector before edits passed: minimal guardian/safety tests 2 passed, confirming current broad skip behavior was covered.

TDD RED after test update failed as intended: guardian/safety specs still reported Exception and runtime import failures were skipped. GREEN focused selector passed after removing guardian/safety broad skip overrides: 5 passed.

Broader validation passed: router groups 130 passed, lifecycle 54 passed, OpenAPI 69 passed, Bandit results 0, git diff --check clean.

Scope note: no skip_exceptions=(Exception,) occurrences remain in minimal.py after this slice.

Opened PR https://github.com/rmusser01/tldw_server/pull/1358 against dev for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed the minimal guardian/safety router ImportedRouterSpec entries for guardian_controls, family_wizard, and self_monitoring by removing broad skip_exceptions=(Exception,) overrides. The guardian/safety specs now use the default precise optional-missing skip exceptions, so missing target modules still skip during registration while runtime import defects propagate. Validation: focused guardian/safety selector 5 passed after RED verification, router group contracts 130 passed, main lifecycle contracts 54 passed, OpenAPI contracts 69 passed, Bandit on minimal.py found 0 results, and git diff --check was clean.
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
