---
id: TASK-17
title: 'Address PR #1242 review comments'
status: Done
assignee:
  - codex
created_date: '2026-05-03 21:35'
updated_date: '2026-05-03 21:39'
labels:
  - phase-2
  - router-groups
  - pr-review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1242'
  - TASK-10
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up for PR #1242 on the Phase 2.2 router conditional cleanup branch. Address review feedback by narrowing optional router import exception handling and documenting the new helper tests while preserving optional router behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 append_imported_router_spec skips only intended optional missing module or missing router attribute cases and does not suppress unrelated import-time/runtime exceptions.
- [x] #2 New or modified helper tests document the behavior they assert.
- [x] #3 Focused router contract tests pass for the changed helper behavior.
- [x] #4 Bandit on touched router source and git diff --check are run and results recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regression coverage proving append_imported_router_spec still skips missing optional modules and missing router attributes but propagates unrelated import-time exceptions. 2. Add concise docstrings to the new helper tests. 3. Narrow the helper exception handling to ModuleNotFoundError/ImportError and AttributeError only. 4. Run focused router contract tests, Bandit on the touched helper source, and git diff --check; record results before pushing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: test_append_imported_router_spec_propagates_unexpected_import_error failed before the helper change because RuntimeError from importlib.import_module was swallowed as a skipped optional router.

GREEN/focused: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "append_imported_router_spec" -q passed 5 selected tests.

Full focused suite: python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q passed 37 tests.

Bandit: python -m bandit -r tldw_Server_API/app/api/v1/router_groups/conditional.py -f json -o /tmp/bandit_pr1242_review_comments.json reported 0 results and 0 errors.

git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1242 review feedback by replacing the broad helper-level exception catch with explicit optional import handling. ImportError and ModuleNotFoundError still skip unavailable optional routers, AttributeError from lazy router lookup is logged and re-raised into the existing registration skip path, and unrelated import-time runtime failures now propagate. Added docstrings and regression coverage for optional import misses, unexpected import failures, and lazy missing-attribute registration behavior.
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
