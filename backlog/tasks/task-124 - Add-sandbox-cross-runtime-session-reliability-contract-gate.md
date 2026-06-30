---
id: TASK-124
title: Add sandbox cross-runtime session reliability contract gate
status: Done
assignee: []
created_date: '2026-05-08 19:35'
updated_date: '2026-05-09 00:16'
labels:
  - sandbox
  - runtime-reliability
  - testing
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow Phase 4 sandbox reliability slice that turns the current session_contract posture into portable regression coverage. Scope should verify cross-runtime session/recovery claims through runtime discovery and admin diagnostics, preserve the rule that only vz_linux currently advertises repair/warm-VM health requirements, and update the capability inventory to distinguish newly-covered portable contract checks from still-host-gated recovery behavior. Do not generalize repair or change runtime execution semantics in this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Portable sandbox tests verify every runtime's session_contract is projected consistently through public runtime discovery and admin runtime diagnostics where applicable.
- [x] #2 Tests assert only runtimes with supported or host_gated repair_state appear as repair-supported in admin diagnostics, and host-local runtimes remain workspace_only with no live-health or repair claims.
- [x] #3 Docs/Sandbox/sandbox-runtime-capability-inventory.md is updated so the Current Gaps section reflects portable session-contract coverage while preserving remaining host-gated recovery/repair gaps.
- [x] #4 No runtime execution, helper boot, networking, or repair mutation behavior changes are introduced unless a failing test proves an existing contract bug.
- [x] #5 Focused sandbox tests, py_compile for touched Python tests, git diff --check, and Bandit skip/rationale or touched-scope Bandit are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan/design review: kept the slice contract-only. The implementation does not generalize repair, add warm reuse to host-local runtimes, change helper behavior, or change runtime execution/admission.

RED: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q` failed on `test_inventory_documents_portable_session_contract_gate_scope` because the inventory still claimed session behavior tests were incomplete beyond discovery-level `session_contract`.

GREEN: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q` passed with 8 tests.

Verification: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q` passed with 36 tests.

Verification: `python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed.

Verification: `python -m bandit -r tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -s B101 -f json -o /tmp/bandit_sandbox_session_contract_gate_tests.json` reported zero findings after excluding pytest assert-use noise.

Verification: `git diff --check` passed.

PR review fixes: wrapped the overlong session-contract assertion, replaced exact docs-prose checks with case-insensitive semantic regexes, and changed task verification snippets from local absolute interpreter paths to portable `python -m ...` commands.

Review verification: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py -q` passed with 36 tests.

Review verification: `python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed.

Review verification: `python -m ruff check tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed.

Review verification: `python -m bandit -r tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -s B101 -f json -o /tmp/bandit_sandbox_session_contract_gate_tests_review_fixes.json` reported zero findings after excluding pytest assert-use noise.

Review verification: `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a portable session-contract regression gate that validates every runtime's `session_contract` through public runtime discovery and checks that admin diagnostics derive session reuse, live-health, and repair support from the same metadata. The new coverage preserves the rule that host-local runtimes are workspace-only and repair-unsupported while `vz_linux` remains the only warm-VM host-gated repair path. Updated the sandbox capability inventory so it records the new portable contract gate while keeping real host-gated recovery flows and non-`vz_linux` repair ownership as remaining Phase 4 gaps.
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
