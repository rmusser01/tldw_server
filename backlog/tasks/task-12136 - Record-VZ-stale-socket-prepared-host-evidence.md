---
id: TASK-12136
title: Record VZ stale socket prepared-host evidence
status: Done
assignee: []
created_date: '2026-07-03 23:55'
updated_date: '2026-07-04 00:00'
labels:
  - sandbox
  - vz_linux
  - evidence
  - lifecycle
dependencies: []
references:
  - Docs/Sandbox/vz-linux-prepared-host-evidence.md
  - Docs/superpowers/specs/2026-05-18-vz-linux-lifecycle-drill-gaps-design.md
  - tools/macos-vz-helper/README.md
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run stale-socket-drill in an isolated private runtime directory on the prepared macOS host.
- [x] #2 Record command, runtime mode, socket path, helper stdout/stderr paths, cleanup state, and pass/fail/skip result in the prepared-host evidence tracker.
- [x] #3 Keep the slice evidence/docs-only and do not expand PR/push/scheduled CI triggers.
- [x] #4 Verification and Bandit applicability are recorded in Backlog.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vz-stale-socket-evidence
Branch: codex/vz-stale-socket-evidence
Base: origin/dev c20013ecce7e3384ec5faa860434d6bdd76d5407
Runtime artifact root: /private/tmp/tldw-vz-stale-socket-stale-socket-20260703-165828

Built the helper with vz-helperctl.py build; the first managed-sandbox build failed because Swift/Clang could not write ~/.cache/clang/ModuleCache, and the same command passed outside the sandbox. Signed the helper with tools/macos-vz-helper/macos-vz-helper.entitlements.

First managed-sandbox stale-socket-drill attempt failed with helper_socket_create_failed / Operation not permitted while creating the controlled Unix socket. Accepted prepared-host evidence reran the same drill outside the sandbox and passed with exit 0. The drill reported stale_socket=ok, start=ok, after_socket=helper_socket_present, after_pid_file/helper_process=helper_pid_running, after_ping=ok, after_protocol_version=1, after_helper_version=0.1.0, and stale_socket_drill=ok.

Cleanup used vz-helperctl.py stop on the same socket/pid paths and returned exit 0. Post-stop status reported socket=helper_socket_absent, pid_file=ok, process=helper_not_running, and ping=helper_not_running. The status command itself exited 1 because of an unrelated default launchd_plist_mismatch on this host, which is documented in the evidence packet. Helper stdout/stderr logs were retained and empty with SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q passed with 23 tests. git diff --check passed. Bandit skipped because the reviewable changes are Markdown/Backlog only; helper build artifacts are untracked evidence setup, not committed source.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recorded a 2026-07-03 prepared-host stale-socket drill evidence packet in Docs/Sandbox/vz-linux-prepared-host-evidence.md. The packet documents controlled inactive socket recovery, helper start/status/protocol checks, explicit stop cleanup, artifact/log pointers, expected skips, and residual follow-ups. Updated the residual-gap table so stale-socket handling is now recorded and remains manual-only for future repeats.
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
