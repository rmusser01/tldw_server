---
id: TASK-101
title: Phase 2.2 minimal workflow router skip semantics AW
status: Done
assignee: []
created_date: '2026-05-07 01:35'
updated_date: '2026-05-07 01:43'
labels:
  - phase2.2
  - router-cleanup
  - minimal
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
  - 'https://github.com/rmusser01/tldw_server/pull/1347'
  - 'https://github.com/rmusser01/tldw_server/pull/1348'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Narrow minimal workflow router optional skip behavior after PR #1347 merged. The workflows, chat_workflows, and scheduler_workflows ImportedRouterSpec entries should skip only genuinely missing optional router modules or missing router attributes, while runtime import defects propagate during registration instead of being hidden by skip_exceptions=(Exception,).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Minimal workflow router specs no longer use skip_exceptions=(Exception,)
- [x] #2 Missing target workflow modules are skipped during registration
- [x] #3 Runtime import defects from workflow router modules propagate
- [x] #4 Existing prefixes tags route keys and default stability behavior are preserved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update focused contract tests for precise workflow optional skip semantics. 2. Verify RED against current broad skip behavior. 3. Remove broad skip_exceptions from workflow ImportedRouterSpec entries. 4. Run focused and broader verification plus Bandit and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline focused selector passed before edits: minimal workflow tests 2 passed, confirming current broad skip behavior is covered.

TDD RED after test update failed as intended: workflow specs still reported Exception and runtime import failures were skipped. GREEN focused selector passed after removing workflow broad skip overrides: 3 passed.

Broader validation passed: router groups 120 passed, lifecycle 54 passed, OpenAPI 69 passed, Bandit results 0, git diff --check clean.

Scope note: remaining skip_exceptions=(Exception,) occurrences are in non-workflow minimal groups and should be handled in later narrow slices.

Opened PR https://github.com/rmusser01/tldw_server/pull/1348 against dev for this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed the minimal workflow router ImportedRouterSpec entries for workflows, chat_workflows, and scheduler_workflows by removing broad skip_exceptions=(Exception,) overrides. The workflow specs now use the default precise optional-missing skip exceptions, so missing target modules still skip during registration while runtime import defects propagate. Validation: focused workflow selector 3 passed after RED verification, router group contracts 120 passed, main lifecycle contracts 54 passed, OpenAPI contracts 69 passed, Bandit on minimal.py found 0 results, and git diff --check was clean.
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
