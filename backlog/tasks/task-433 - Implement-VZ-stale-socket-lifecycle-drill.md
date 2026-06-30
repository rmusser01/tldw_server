---
id: TASK-433
title: Implement VZ stale socket lifecycle drill
status: Done
labels:
- sandbox
- vz_linux
- lifecycle
- hardening
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next sandbox lifecycle slice from the merged drill-gaps design: a bounded stale-socket operator/check path that proves stale helper socket recovery is safe, diagnosable, and refuses symlinks/non-socket/user-controlled paths. Keep it host-independent first, with docs and focused tests; no workflow trigger expansion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Host-independent tests cover stale Unix socket recovery under a private runtime directory.
- [x] #2 Tests cover fail-closed behavior for symlinks, non-socket files, directories, and unsafe parent paths.
- [x] #3 Implementation reuses existing vz-helperctl/helper socket-safety primitives instead of adding a parallel cleanup path.
- [x] #4 Operator docs or drill notes explain how to run the stale socket check manually and what evidence to capture.
- [x] #5 Prepared-host evidence tracker references the new stale socket drill/check without expanding PR/push/scheduled destructive triggers.
- [x] #6 Focused verification, diff hygiene, and Bandit for touched Python scope are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from merged design PR #1850 / `Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md`.

Plan: `Docs/superpowers/plans/2026-05-19-vz-stale-socket-lifecycle-drill.md`.

Baseline verification before implementation:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 134 passed, 1 skipped.
- `swift test` initially failed under the Codex filesystem sandbox because SwiftPM/Clang could not write `~/.cache/clang/ModuleCache`.
- `swift test` passed after host permission for SwiftPM cache access: 88 tests passed.

Implementation notes:
- Added `vz-helperctl.py stale-socket-drill` as a manual operator command.
- The drill validates helper binary, socket path, private runtime/pid/log/serial directories, and active socket state before mutation.
- The drill creates a controlled inactive Unix socket only when the accepted socket path is absent; pre-existing inactive Unix sockets are preserved for `start_helper()` recovery.
- Recovery delegates to existing `start_helper()` and its identity-based stale socket cleanup instead of adding a parallel unlink path.
- If startup returns or raises an operational failure after the drill created a socket, cleanup removes only the captured identity for the drill-created socket.
- Post-start success uses `_managed_helper_running_result()` so all-ok but not-running status rows do not pass the drill.
- Swift direct-launch coverage was reviewed and not changed. Existing `UnixSocketServerTests.swift` already covers regular-file preservation, symlink preservation, stale socket replacement, active socket refusal, stop-time replacement preservation, and unsafe parent refusal. Direct identity-race testing would need production test hooks, so it remains documented rather than added in this slice.
- Updated `tools/macos-vz-helper/README.md` with manual command usage and evidence guidance.
- Updated `Docs/Sandbox/vz-linux-prepared-host-evidence.md` with stale-socket evidence fields and residual-gap guidance.
- PR review follow-up:
  - Added docstrings for the new stale socket helper and CLI entrypoint.
  - Added a fail-closed guard when `socket_creator()` returns without materializing a Unix socket identity.
  - Added a regression test for the no-socket false-positive path.
- Final verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q -k stale_socket_drill` passed: 8 passed, 5 skipped.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tools/macos-vz-helper/Tests/test_vz_helperctl.py -q` passed: 142 passed, 6 skipped.
  - `swift test` under `tools/macos-vz-helper` passed with host SwiftPM cache access: 88 tests passed.
  - `git diff --check` passed.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q tools/macos-vz-helper/scripts/vz-helperctl.py` passed.
  - Full Bandit scan of `tools/macos-vz-helper/scripts/vz-helperctl.py tools/macos-vz-helper/Tests/test_vz_helperctl.py` still exits non-zero on pre-existing helperctl test baseline findings at `test_vz_helperctl.py` imports and older `/tmp`/subprocess examples outside the new stale-socket drill section. The new section's hardcoded `/tmp` tempdir findings were removed.
- Final code-quality subagent review: approved.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a manual `vz-helperctl.py stale-socket-drill` operator command that creates a controlled inactive Unix socket under validated private helper paths, verifies socket identity before reporting creation success, delegates recovery to the existing managed `start_helper()` path, verifies the helper is actually running afterward, and cleans up only the drill-created socket identity on operational startup failure.

Added host-independent Python coverage for successful stale socket recovery, pre-existing inactive socket recovery, fail-closed unsafe path shapes, unsafe parent permissions, dry-run behavior, start failure cleanup, and CLI argument forwarding. Updated the helper README and prepared-host evidence tracker so operators know how to run the drill manually and what evidence to capture.
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
