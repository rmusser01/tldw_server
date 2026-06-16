---
id: TASK-2364
title: Record VZ Linux manual failure-drill evidence packet
status: Done
labels:
- sandbox
- docs
- vz_linux
references:
- https://github.com/rmusser01/tldw_server/issues/1442
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
- Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
- Docs/Sandbox/macos-runtime-operator-notes.md
modified_files:
- Docs/Sandbox/vz-linux-prepared-host-evidence.md
- backlog/tasks/task-2364 - Record-VZ-Linux-manual-failure-drill-evidence-packet.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the prepared-host VZ Linux smoke path with manual `--include-failure-drills` enabled and update the prepared-host evidence tracker with the actual failure-drill results, artifacts, expected skips, and residual gaps. Scope is evidence/documentation only unless the run exposes a minimal setup/doc issue required to report accurate evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current smoke/failure-drill command path is inspected before execution.
- [x] #2 Manual failure-drill smoke is run on the prepared Apple silicon host, or a precise operator-setup blocker is recorded if prerequisites fail before drill execution.
- [x] #3 Prepared-host evidence tracker records a dated failure-drill evidence packet with commands, results, artifacts/log pointers, expected skips, and residual gaps.
- [x] #4 Verification results and Bandit applicability are recorded in the Backlog task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Inspected `Docs/Sandbox/vz-linux-prepared-host-evidence.md`,
  `tools/macos-vz-helper/scripts/vz-helperctl.py smoke`, and
  `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` before running the
  prepared-host smoke.
- Ran `vz-helperctl.py smoke --include-failure-drills` against the local
  Debian bookworm arm64 bundle on an Apple M4 Pro host. The run built and
  signed the Swift helper, ran helper daemon smoke, default real-host smoke,
  and manual failure-drill tests.
- Recorded evidence in `Docs/Sandbox/vz-linux-prepared-host-evidence.md`,
  including command, host facts, helper signing, runtime paths, artifact/log
  pointers, pass counts, expected skips, and residual gaps.
- Observed that the direct-bundle smoke path updated `rootfs.img` mtime/hash
  during real VM execution; recorded this as a residual gap so future packets
  can prefer disposable clones or a reset source bundle when immutable source
  hashes matter.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added a 2026-06-16 prepared-host evidence packet for manual VZ Linux
  failure drills. Results: helper daemon smoke `2 passed`, real `vz_linux`
  smoke `3 passed, 11 deselected`, failure drills `2 passed, 12 deselected`,
  wrapper ended `smoke: ok`.
- Updated the residual-gap table to mark failure-drill evidence as recorded and
  to track direct-bundle smoke mutability as a follow-up for image-store clone
  hardening.
- Verification: `git diff --check` passed; `python -m pytest
  tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q`
  passed with `18 passed, 6 warnings`.
- Bandit: not run because the change is documentation/Backlog-only and does not
  touch Python/runtime code.
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
