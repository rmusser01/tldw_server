---
id: TASK-75
title: Refresh sandbox capability inventory after portable gate
status: Done
assignee: []
created_date: '2026-05-05 16:13'
labels:
  - sandbox
  - runtime-capability
  - documentation
dependencies: []
documentation:
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the sandbox runtime capability inventory after the portable runtime capability gate landed so the current gaps section no longer claims there is no single cross-runtime capability gate. Keep this as a narrow documentation/test-maintenance slice that preserves the distinction between portable capability-contract coverage and host-gated real runtime smoke coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The inventory current gaps section no longer says CI has no single cross-runtime capability gate now that the portable gate exists.
- [x] #2 The inventory still clearly states that real runtime execution remains covered by host-gated smoke tests and is not proven by the portable gate.
- [x] #3 A focused docs or sandbox verification command is run and recorded.
- [x] #4 No production runtime behavior is changed in this slice.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Worktree: `<local-worktree-path>`.
- Scope is intentionally docs/test-maintenance only. Do not change runtime behavior.
- RED: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py::test_portable_runtime_capability_gate_inventory_no_longer_lists_gate_as_missing -q` failed because the inventory still contained `CI has no single cross-runtime capability gate`.
- Updated the current gaps table to replace the stale capability-gate gap with a narrower real-execution CI gap: the portable gate covers capability contracts only.
- GREEN: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py::test_portable_runtime_capability_gate_inventory_no_longer_lists_gate_as_missing -q` passed.
- Verification: `python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q` passed with 4 tests.
- Verification: `python -m py_compile tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed.
- Verification: `python -m ruff check tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py` passed.
- Verification: `git diff --check` passed.
- Review fix: replaced the exact positive documentation phrase assertion with a regex-backed section check that tolerates harmless wording and line wrapping while preserving the host-gated real-execution versus portable capability-contract distinction.
- Review fix: removed the committed local absolute worktree path from the task notes.
- Review verification: `python -m bandit -r tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -f json -o /tmp/bandit_sandbox_capability_inventory_refresh_tests.json` ran on the touched test file and reported only pytest assert-use findings (`B101`).
- Review verification: `python -m bandit -r tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -s B101 -f json -o /tmp/bandit_sandbox_capability_inventory_refresh_tests_no_b101.json` reported 0 findings after excluding pytest assert-use noise.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the sandbox runtime capability inventory after the portable capability gate landed. The current gaps table no longer says the project lacks a cross-runtime capability gate; it now records the narrower remaining gap that no single CI job proves real execution for every runtime, while the portable gate covers capability contracts only. Added a regression assertion to keep the stale wording from returning.
<!-- SECTION:FINAL_SUMMARY:END -->
