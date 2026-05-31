---
id: TASK-105
title: Phase 2.2 minimal utility router skip semantics AY
status: Done
assignee: []
created_date: '2026-05-07 04:32'
updated_date: '2026-05-07 04:39'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1352'
  - 'https://github.com/rmusser01/tldw_server/pull/1355'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Narrow minimal utility router optional skip behavior after PR #1352 merged. The web clipper, skills, translate, and slides ImportedRouterSpec entries should skip genuinely missing optional target modules or missing router attributes, while runtime import defects propagate during registration instead of being hidden by skip_exceptions=(Exception,).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal utility router specs no longer use skip_exceptions=(Exception,)
- [x] #2 Missing target utility modules are skipped during registration
- [x] #3 Runtime import defects from utility router modules propagate
- [x] #4 Existing prefixes tags route keys and default stability behavior are preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update focused contract tests for precise utility optional skip semantics. 2. Verify RED against current broad skip behavior. 3. Remove broad skip_exceptions from utility ImportedRouterSpec entries. 4. Run focused and broader verification plus Bandit and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline focused selector before edits passed: minimal utility tests 2 passed, confirming current broad skip behavior was covered.

TDD RED after test update failed as intended: utility specs still reported Exception and runtime import failures were skipped. GREEN focused selector passed after removing utility broad skip overrides: 6 passed.

Broader validation passed: router groups 127 passed, lifecycle 54 passed, OpenAPI 69 passed, Bandit results 0, git diff --check clean.

Scope note: remaining skip_exceptions=(Exception,) occurrences are the guardian/safety minimal group and should be handled in a later narrow slice.

Opened PR https://github.com/rmusser01/tldw_server/pull/1355 against dev for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed the minimal utility router ImportedRouterSpec entries for web clipper, skills, translate, and slides by removing broad skip_exceptions=(Exception,) overrides. The utility specs now use the default precise optional-missing skip exceptions, so missing target modules still skip during registration while runtime import defects propagate. Validation: focused utility selector 6 passed after RED verification, router group contracts 127 passed, main lifecycle contracts 54 passed, OpenAPI contracts 69 passed, Bandit on minimal.py found 0 results, and git diff --check was clean.
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
