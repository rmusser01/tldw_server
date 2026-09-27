---
id: TASK-13243
title: Validate real VZ Linux execution under launchd
status: Done
assignee: []
created_date: '2026-09-13 18:17'
updated_date: '2026-09-13 20:12'
labels:
  - sandbox
  - vz-linux
  - validation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1442'
  - 'https://github.com/rmusser01/tldw_server/pull/2955'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-vz-helper-launchd-validation-drill-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the approved launchd-managed real VM acceptance slice after PR 2628, using a verified current guest and disposable image-store bundle with durable evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Build and sign the current helper; verify the guest contains the VSock buffered-reader fix.
- [x] #2 Run real ephemeral execution, same-session reuse, and diagnostics through the launchd-owned helper using a disposable image-store clone.
- [x] #3 Verify session and LaunchAgent cleanup, preserve source hashes, and retain durable evidence.
- [x] #4 Record exact outcomes and any blockers in the sandbox evidence tracker and this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Inspect source, helper, and host; prepare current guest and isolated evidence. Stage 2: Run launchd VM smoke through the existing managed-socket tests. Stage 3: Verify cleanup and hashes, retain logs, update tracker and commit verified results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current local dev base c70387f496d82fcee92926bf3715bf5cd240ba88. Isolated worktree codex/vz-launchd-vm-validation. Durable evidence: /Users/macbook-dev/Library/Logs/tldw/vz-launchd-vm-validation/20260913-1817. Swift build and signing/Virtualization entitlement verification passed; 22 launchd/helper-smoke contract tests and Go internal/guest tests passed. Original bundle dates June 2026, so preparing a separate ext4 image with a current Linux arm64 guest binary. Builder required preservation of a June 15 stale disk lock after confirming no vmware-vmx process or open disk handle, and a persistent launcher session; current builder IP 192.168.241.128. Original source is not being modified.

Real launchd-drill completed on macOS 26.5.2 arm64 at about 11:29 PDT on 2026-09-13: exit 0; 3 passed, 11 deselected, 0 skipped, 5.83 seconds. JUnit independently confirms 3 tests, no failures/errors/skips. Bootstrap/status/kickstart/helper readiness/protocol 1/helper 0.1.0 and bootout all passed. Test coverage proves expected ephemeral stdout and exit code, identical VM IDs across two session commands, session destruction/control cleanup, and non-mutating diagnostics/dry-run repair. Post-drill launchctl print returned 113 and lsof found no process holding the helper executable. The inactive socket left by bootout was removed explicitly; the temporary runtime directory was removed. Generic helper status also reported an unrelated pre-existing default launchd_plist_mismatch, which was left unchanged. Both original and refreshed source hashes compare identical before and after smoke; disposable rootfs changed as expected. Builder staging files removed; vmrun stop soft completed and vmrun list returned zero running VMs. Pending ext4 journal recovery was completed on the separate source image and e2fsck plus extracted-binary comparison passed. Durable artifacts include the tested helper and guest binaries, refreshed source and run bundles, JUnit, helper and serial logs, plist, source hashes, and artifact-checksums.sha256. git diff --check passed. Bandit not applicable: repository changes are Markdown/Backlog only, with no Python or runtime source edits.

Published together with live-session recovery and the guest output fix in PR #2955 against dev. Evidence remains local and is summarized in the prepared-host ledger; human Change summary remains a merge prerequisite.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed real VZ Linux acceptance through an isolated launchd-managed helper using an image-store disposable bundle with the current buffered-reader guest fix. All three real VM tests passed without skips; source bundles remained unchanged, the LaunchAgent/helper stopped, and operator cleanup removed the inactive socket. Evidence is durable at /Users/macbook-dev/Library/Logs/tldw/vz-launchd-vm-validation/20260913-1817 and documented in Docs/Sandbox/vz-linux-prepared-host-evidence.md. This proves launchd-managed execution/reuse and diagnostics; it does not claim live-VM recovery across helper restart, host reboot, or injected readiness/mismatch failures.
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
