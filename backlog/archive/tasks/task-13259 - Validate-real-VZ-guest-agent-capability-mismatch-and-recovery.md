---
id: TASK-13259
title: Validate real VZ guest-agent capability mismatch and recovery
status: Done
assignee: []
created_date: '2026-09-13 23:39'
updated_date: '2026-09-13 23:41'
labels:
  - sandbox
  - vz-linux
  - host-gated
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow the approved 2026-05-18 lifecycle drill contract after merged PR 2955. Use an image-store disposable Debian arm64 bundle with a test-only guest advertising no exec capability; verify the existing runner rejects it, publishes the stable reason, leaves no reusable control or VM, and a healthy bundle still executes and reuses its session. Do not change production capability policy, mutate the canonical bundle, restart unrelated helpers, or reboot the host.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real missing-exec guest metadata is observed over VSock, not mocked.
- [ ] #2 A session run fails with the explicit required-capability reason, produces no command output, and leaves no persisted VM control or live VM.
- [ ] #3 Healthy-bundle execution and same-session reuse succeed after the rejection on the same isolated helper.
- [ ] #4 Source hashes stay unchanged; VM/helper cleanup and evidence are retained; portable tests and touched-scope Bandit pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This uncommitted top-level ID was allocated concurrently with another task in the primary checkout. Superseded before implementation by TASK-13243.3 to avoid an active task ID collision.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
