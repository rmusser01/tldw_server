---
id: TASK-2244
title: Implement Workspace file inventory ignore policy
status: Done
priority: high
references:
- TASK-2243
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 3 from the Workspace file inventory Jobs implementation plan. Add a side-effect-free ignore policy module for built-in generated/secret skips, bounded gitignore subset parsing, diagnostics, and stable policy fingerprints using TDD.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing ignore-policy tests are written first for built-ins, secret-like files, gitignore subset behavior, malformed/oversized diagnostics, unsupported constructs, and fingerprints.
- [x] #2 Ignore policy helpers skip generated directories and secret-like files without filesystem side effects.
- [x] #3 Conservative gitignore subset parser is bounded and reports diagnostics instead of crashing.
- [x] #4 Policy fingerprint is stable for equivalent rules and changes when effective rules change.
- [x] #5 Focused ignore-policy tests pass and Backlog records verification/completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added red coverage in `tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py`; initial run failed with `ModuleNotFoundError` for the missing `file_inventory_ignore` module.
- Implemented a side-effect-free ignore policy builder with built-in generated directory and secret-like file skips, bounded gitignore text handling, conservative unsupported-pattern diagnostics, and stable effective-rule fingerprints.
- Explicit workspace/gitignore rules are evaluated before built-ins for user-visible reason reporting, while built-ins remain present as default safety coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Workspace file inventory ignore policy module and focused tests. Verification: `pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py -q` passed 7 tests; adjacent `pytest tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_models.py tldw_Server_API/tests/Workspaces/test_workspace_file_inventory_ignore.py -q` passed 21 tests; `compileall` passed for the new module and tests; Bandit reported 0 findings on the new core module.
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
