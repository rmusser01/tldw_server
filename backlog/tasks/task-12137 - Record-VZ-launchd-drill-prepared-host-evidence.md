---
id: TASK-12137
title: Record VZ launchd drill prepared-host evidence
status: Done
assignee: []
created_date: '2026-07-04 00:05'
updated_date: '2026-07-04 00:16'
labels:
  - sandbox
  - vz_linux
  - evidence
  - launchd
  - lifecycle
dependencies: []
references:
  - Docs/Sandbox/vz-linux-prepared-host-evidence.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - tools/macos-vz-helper/README.md
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run launchd-drill with a unique LaunchAgent label and private runtime/plist/log paths on the prepared macOS host.
- [x] #2 Record launchd bootstrap/kickstart/status/bootout result, runtime mode, helper stdout/stderr paths, cleanup state, and pass/fail/skip result in the prepared-host evidence tracker.
- [x] #3 Keep the slice evidence/docs-only and do not expand PR/push/scheduled CI triggers.
- [x] #4 Verification and Bandit applicability are recorded in Backlog.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worktree: /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/vz-launchd-drill-evidence
Branch: codex/vz-launchd-drill-evidence
Base: origin/dev f2d9be986499eb1bfda36f566870a98e8dd90d0d
Accepted runtime artifact root: /private/tmp/tldw-vz-launchd-drill-launchd-drill-20260703-171446

Built the helper with vz-helperctl.py build outside the managed filesystem sandbox because Swift/Clang needed access to ~/.cache/clang/ModuleCache. The launchd drill signed the helper with tools/macos-vz-helper/macos-vz-helper.entitlements before bootstrap.

A first diagnostic launchd-drill attempt passed a relative --helper path. The generated LaunchAgent plist preserved that relative ProgramArguments value, so launchd loaded/kicked the service but helper readiness failed with helper_ping_failed. The accepted evidence reran with an absolute helper path and passed with exit 0. Results: launchd_preflight=launchd_service_absent, helper_signing=ok, launchd_bootstrap=ok, launchd_status=ok, launchd_kickstart=ok, helper_status=ok, protocol_version=1, helper_version=0.1.0, launchd_bootout=ok.

Cleanup evidence: after drill-owned bootout, explicit launchd status returned launchd_status_failed=113 and an extra bootout returned No such process, confirming the LaunchAgent was unloaded. Direct helper status reported no pid file, helper_not_running, and helper_ping_failed. The socket file remained as an inactive socket under the private 0700 runtime directory; this is documented as cleanup state and is covered by the separate stale-socket recovery drill. Helper stdout/stderr logs were empty with SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q passed with 23 tests. git diff --check passed. Bandit skipped because the reviewable changes are Markdown/Backlog only; helper build artifacts and launchd artifacts are local evidence setup, not committed source.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Recorded a 2026-07-03 prepared-host launchd-drill evidence packet in Docs/Sandbox/vz-linux-prepared-host-evidence.md. The packet documents isolated LaunchAgent bootstrap/kickstart/helper readiness/protocol checks, drill-owned bootout, artifact/log pointers, expected skips, the relative-helper diagnostic failure, and cleanup state. Updated the residual-gap table so launchd-drill evidence is now recorded while launchd-managed VM smoke remains manual-only if explicitly requested.
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
