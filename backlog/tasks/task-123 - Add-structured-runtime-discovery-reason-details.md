---
id: TASK-123
title: Add structured runtime discovery reason details
status: Done
assignee:
  - '@codex'
created_date: '2026-05-08 13:41'
updated_date: '2026-05-08 13:59'
labels:
  - api
  - sandbox
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - tldw_Server_API/app/core/Sandbox/runtime_capabilities.py
  - tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the sandbox Phase 3 runtime discovery gap by exposing structured metadata for normalized runtime readiness reasons while preserving raw reasons and normalized reason codes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime discovery exposes additive structured details for every normalized runtime reason without removing raw reasons or normalized_reasons.
- [x] #2 Runtime reason metadata is centralized, complete for every RuntimeReasonCode, and fails fast on missing or mismatched entries.
- [x] #3 Admin runtime diagnostics can consume the same structured reason metadata without duplicating operator-action logic.
- [x] #4 Sandbox inventory docs describe the reason-details contract and no longer list the gap as current.
- [x] #5 Focused tests cover schema exposure, feature discovery population, metadata completeness, diagnostics projection, and docs contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for runtime reason metadata completeness, schema exposure, feature discovery details, diagnostics projection, and docs contract.
2. Add RuntimeReasonDetails metadata/catalog helpers in runtime_capabilities.py with import-time completeness and code-match validation.
3. Extend SandboxRuntimeInfo and admin runtime diagnostics schemas with additive normalized_reason_details.
4. Populate details in SandboxService.feature_discovery() and reuse metadata-derived operator actions in runtime diagnostics.
5. Update the sandbox runtime capability inventory and run focused pytest, py_compile, Bandit on touched Python, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented structured runtime discovery reason details. Red test run failed with missing normalized_reason_details schema/payload fields, missing RUNTIME_REASON_METADATA, missing runtime_reason_details helper, and docs gap still present. Green verification passed after implementation.

Verification:
- Baseline before edits: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py -q -> 39 passed, 2 warnings.
- Red test run after adding tests -> 8 failed, 35 passed, 2 warnings for the expected missing runtime reason-details contract.
- Green focused test run -> 43 passed, 2 warnings.
- py_compile on touched Python files -> passed.
- Bandit production touched Python scan -> 0 results in /tmp/bandit_runtime_reason_details_prod.json.
- Bandit touched tests with B101 skipped -> 0 results in /tmp/bandit_runtime_reason_details_tests.json.
- git diff --check -> passed.

Known skips/blockers: full repository test suite was not run; focused sandbox runtime/docs contract tests covered the changed surface.

Reopened for PR #1378 review follow-up. Actionable findings verified from Qodo/Gemini: recommended_action should not depend on raw normalized reason order when multiple reason details exist, and SandboxRuntimeInfo.normalized_reason_details should be non-nullable because runtime discovery always emits a list.

PR #1378 review follow-up implemented. Verified two findings: recommended_action was order-dependent, and normalized_reason_details was nullable in the public schema despite runtime discovery always emitting a list. Added failing regression checks for both, then fixed with explicit operator-action priority and a non-nullable list schema field.

Review follow-up verification:
- Red regression run: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py::test_runtime_diagnostics_summary_projects_feature_discovery_rows tldw_Server_API/tests/Docs/test_sandbox_public_docs_contract.py::test_sandbox_runtime_schema_exposes_reason_details -q -> 2 failed as expected.
- Green regression run for same targets -> 2 passed, 2 warnings.
- Focused sandbox/docs contract suite -> 43 passed, 2 warnings.
- py_compile on touched Python files -> passed.
- Bandit production touched Python scan -> 0 results in /tmp/bandit_runtime_reason_details_prod.json.
- Bandit touched tests with B101 skipped -> 0 results in /tmp/bandit_runtime_reason_details_tests.json.
- git diff --check -> passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added centralized runtime reason metadata and additive normalized_reason_details on runtime discovery/admin diagnostics. PR review follow-up made normalized_reason_details non-nullable and made recommended_action selection priority-based rather than reason-order-dependent.
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
