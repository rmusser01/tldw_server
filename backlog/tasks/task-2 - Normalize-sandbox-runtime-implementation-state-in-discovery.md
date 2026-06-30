---
id: TASK-2
title: Normalize sandbox runtime implementation state in discovery
status: Done
assignee: []
created_date: '2026-05-03 17:16'
updated_date: '2026-05-03 17:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a runtime discovery implementation_state field using the sandbox roadmap state vocabulary so clients can distinguish host availability from runtime maturity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox runtime discovery includes implementation_state for every runtime
- [x] #2 Schema documents the supported state vocabulary
- [x] #3 Focused tests cover docker, vz_linux, vz_macos, seatbelt, and worktree state labels
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented runtime implementation_state labels for /api/v1/sandbox/runtimes using the roadmap state vocabulary. Added centralized state mapping in runtime_capabilities.py, surfaced it through SandboxService.feature_discovery(), extended the response schema, and documented the available vs implementation_state split in sandbox docs.

Verification:
- focused red test first failed with KeyError: implementation_state
- python -m pytest tldw_Server_API/tests/sandbox/test_feature_discovery_flags.py -k implementation_state_labels -q: passed
- python -m pytest tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q --timeout=60: passed
- python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capabilities_policy.py -q --timeout=60: passed
- python -m py_compile touched Python modules: passed
- python -m bandit -r touched Python files -f json -o /tmp/bandit_runtime_state.json: passed, 0 findings
- git diff --check: passed

Known skip/blocker:
- Full test_feature_discovery_flags.py timed out in existing TestClient teardown/job-worker startup path at test_egress_allowlist_supported_when_enforced, not in the new implementation_state test.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added normalized runtime implementation_state labels to sandbox runtime discovery so clients can distinguish current host availability from runtime maturity. Updated schema and docs, added focused coverage, and recorded the existing full-file TestClient timeout separately.
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
