---
id: TASK-72
title: Add portable sandbox runtime capability gate
status: Done
assignee: []
created_date: '2026-05-05 14:38'
labels:
  - sandbox
  - runtime-capability
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow sandbox maintenance slice that verifies runtime capability/discovery contracts stay aligned across all RuntimeType values without requiring host-specific runtimes. The gate should exercise the existing runtime metadata, feature discovery, inventory documentation, status taxonomy, and session semantics from portable tests so future sandbox runtime additions cannot drift silently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A focused portable test fails if a RuntimeType lacks required capability/discovery metadata or runtime inventory documentation coverage.
- [x] #2 The gate covers current runtime rows for isolation metadata, implementation state, session semantics, normalized reason/status metadata where applicable, and admin diagnostics/documentation references without requiring Docker, Lima, Firecracker, or Apple Virtualization.framework availability.
- [x] #3 Runtime capability inventory or sandbox docs are updated to document the new gate and remaining non-portable host-gated coverage.
- [x] #4 Focused sandbox tests, syntax/lint checks, Bandit for touched production code when applicable, and git diff checks are run and recorded.
<!-- AC:END -->

## Definition of Done

<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

1. Add a portable runtime capability gate test that injects synthetic preflight rows instead of probing host runtimes.
2. Verify the gate fails on the current branch for the missing cross-runtime capability contract.
3. Add the smallest code/docs/test changes needed to make the gate pass.
4. Run focused sandbox verification and record results.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Scope is intentionally test/docs-first. Production changes should only happen if the red gate exposes a real contract gap.
- RED: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q` failed because `Docs/Sandbox/sandbox-runtime-capability-inventory.md` did not document the Portable Runtime Capability Gate.
- GREEN: Added the inventory section; the focused gate passed with 3 tests.
- Verification: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q` passed with 34 tests.
- Verification: `python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed.
- Verification: `python -m ruff check tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed after import ordering fix.
- Verification: `git diff --check` passed.
- Bandit skipped: this slice touched only tests and documentation, not production Python.
- Review fix RED: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py::test_portable_runtime_capability_gate_covers_emitted_status_reason_aliases -q` failed after adding the Lima `limactl_missing` alias because it normalized to `runtime_error`.
- Review fix: added `limactl_missing` to the runtime-unavailable taxonomy, made runtime discovery assertions use explicit string keys, switched docs lookup to `pytestconfig.rootpath`, and made runtime-name documentation matching format-agnostic.
- Review verification: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/tests/sandbox/test_runtime_inventory_contract.py tldw_Server_API/tests/sandbox/test_run_status_reason_codes.py -q` passed with 34 tests.
- Review verification: `python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py` passed.
- Review verification: `python -m ruff check tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py` passed.
- Review verification: `python -m bandit -r tldw_Server_API/app/core/Sandbox/run_status_taxonomy.py -f json -o /tmp/bandit_sandbox_runtime_capability_gate_review.json` reported 0 findings.
- Review verification: `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a portable sandbox runtime capability gate that injects synthetic preflight rows for every runtime and validates the API-facing discovery projection through the public schema. The gate asserts complete runtime metadata maps, normalized preflight reasons, emitted run-status reason aliases, host-local isolation/session constraints, and inventory documentation coverage without requiring host-specific runtime availability. Updated the runtime capability inventory to document the gate and clarify that real execution remains covered by host-gated smoke paths.

Review follow-up tightened the gate by explicitly covering Lima's `limactl_missing` emitted reason, avoiding brittle docs path/Markdown formatting assumptions, and avoiding implicit enum/string equality in runtime discovery assertions.
<!-- SECTION:FINAL_SUMMARY:END -->
