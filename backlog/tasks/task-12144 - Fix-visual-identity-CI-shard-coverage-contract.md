---
id: TASK-12144
title: Fix visual identity CI shard coverage contract
status: Done
created_date: 2026-07-04 07:10
labels:
- ci
- tests
- visual-identities
priority: High
modified_files:
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
updated_date: 2026-07-04 07:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The CI workflow contract currently treats every root-level Character_Chat test file as covered only by the legacy Character_Chat shards. The visual identity metadata regression test is intentionally assigned to the visual-identities shard with the Visual_Identities suite, causing the CI contract to fail after the latest dev rebase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused CI shard contract test passes without moving visual identity tests into unrelated legacy shards.
- [x] #2 The workflow contract accounts for the visual-identities shard ownership of the Character_Chat visual identity regression test.
- [x] #3 Broader CI contract suite retry outcome is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the failing assertion and workflow shard ownership for the visual identity metadata test.
2. Update the CI contract to include feature-specific visual identity shard ownership while preserving the full Character_Chat coverage guard.
3. Run focused and broader CI contract verification.
4. Record verification, update task status, stage intended files, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: after the latest dev rebase, `tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py` is intentionally owned by the feature-specific `visual-identities` shard alongside `tldw_Server_API/tests/Visual_Identities`, but the CI workflow contract still required every `Character_Chat/test*.py` file to be covered only by the three legacy Character_Chat shards.

Fix: updated the contract to assert the exact `visual-identities` shard paths and include the visual identity Character_Chat regression file in the Character_Chat coverage accounting without moving it into the legacy shards.

Verification:
- RED before fix: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/CI/test_required_workflow_contracts.py::test_full_suite_splits_slow_chat_and_retrieval_shards` failed with the visual identity metadata test as the uncovered right-set item.
- Focused green: same command passed, `1 passed, 8 warnings`.
- Broader CI contract suite: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -x --tb=short tldw_Server_API/tests/CI` passed, `73 passed, 152 warnings`.
- Security: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/tests/CI/test_required_workflow_contracts.py -s B101 -f json -o /tmp/bandit_task_12144.json` produced `errors: []`, `results: []`.
- Whitespace: `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the CI workflow contract so `visual-identities` is recognized as the shard owner for the Character_Chat visual identity metadata regression test. This preserves the feature-specific shard grouping while keeping the full Character_Chat test-file coverage guard active.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Root cause recorded in task notes.
- [x] #2 Verification commands and outcomes recorded.
- [x] #3 Bandit decision recorded for touched scope.
- [x] #4 Final summary explains what changed and why.
<!-- DOD:END -->
