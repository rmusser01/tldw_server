---
id: TASK-3
title: Normalize sandbox runtime reason codes in discovery
status: Done
assignee:
  - codex
created_date: '2026-05-03 17:58'
updated_date: '2026-05-03 18:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an additive normalized reason-code layer to sandbox runtime discovery while preserving raw preflight reasons.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discovery includes normalized_reasons for runtimes with raw reasons.
- [x] #2 Raw reasons remain unchanged for operator diagnostics and compatibility.
- [x] #3 Reason-code vocabulary is centralized in runtime_capabilities.py and exposed through the API schema.
- [x] #4 Sandbox runtime inventory documents the additive normalized reason-code contract.
- [x] #5 Helper protocol error variants normalize to helper_protocol_mismatch and keep raw reasons intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a regression assertion for macOS helper protocol error variants in tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py and verify it fails on the current implementation.
2. Update runtime_capabilities.normalize_runtime_reason to classify macos_virtualization_helper_protocol* raw reason codes as helper_protocol_mismatch while preserving the existing explicit map and image store prefix behavior.
3. Re-run the focused sandbox runtime inventory test, then run py_compile for touched Python, Bandit on touched Python paths, and git diff --check.
4. Update TASK-3 notes/final summary with verification results, then commit and push the PR branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented additive normalized_reasons on runtime discovery while preserving raw reasons. Verification: runtime inventory contract tests passed; runtime capabilities policy tests passed; py_compile passed; Bandit touched Python scan had 0 findings; git diff --check passed. Known skip/caveat: selected TestClient-based feature_discovery_flags group timed out in existing app shutdown/background Jobs teardown after first selected test passed; stack was TestClient/Jobs teardown, not normalized reason code logic.

Reopened for PR #1236 review follow-up: Qodo identified macOS helper protocol error variants that still normalize to unknown.

Review follow-up implemented: helper protocol-prefixed raw reasons plus helper empty-response and invalid-json protocol failures now normalize to helper_protocol_mismatch.
Verification: regression test failed before implementation with an extra unknown normalized reason; after the fix, python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q passed with 6 passed, 2 warnings. python -m py_compile passed for runtime_capabilities.py and test_runtime_inventory_contract.py. git diff --check passed. Bandit on runtime_capabilities.py had 0 findings; combined touched-path Bandit scan still reports the existing low-severity B101 pytest assertion baseline in the test file, with the new assertion explicitly skipped via nosec B101 so it does not add a new finding.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PR #1236 review follow-up so macOS helper protocol error variants normalize to helper_protocol_mismatch, with regression coverage and focused verification.
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
