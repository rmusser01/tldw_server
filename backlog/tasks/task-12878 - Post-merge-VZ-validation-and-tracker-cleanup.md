---
id: TASK-12878
title: Post-merge VZ validation and tracker cleanup
status: Done
labels:
- sandbox
- vz-linux
- post-merge
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After PR #2620 merge, validate what the prepared-host VZ smoke path can run from merged origin/dev and clean stale VZ Backlog tracker statuses for already-merged work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verify PR #2620 is present on origin/dev and use origin/dev as the work base.
- [x] #2 Run or dry-run the prepared-host VZ validation path as far as the local host allows, recording exact blockers if real VM smoke cannot run.
- [x] #3 Clean stale VZ tracker tasks whose referenced PRs are already merged, without touching unrelated dirty work.
- [x] #4 Record verification and final summary in Backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Create an isolated worktree from origin/dev.
- Inspect host smoke prerequisites and run the safest available validation command.
- Patch only stale VZ tracker task metadata/final notes needed for already-merged work.
- Run diff checks and commit the cleanup/evidence changes if any.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created clean worktree `codex/vz-post-merge-validation-cleanup` from `origin/dev` at merge commit `3ea6ebb441`.
- Verified PR #2620 is merged into `origin/dev` and `TASK-12138` is present as Done.
- Host facts: macOS 15.6 arm64; Swift/Xcode tooling present (`Apple Swift 6.1.2`, target arm64 macOS 15.0).
- Real VZ smoke selection command ran and selected three `vz_linux_host_smoke` tests, but all skipped because `TLDW_SANDBOX_VZ_LINUX_E2E=1` is not set.
- No configured `TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH` was present; no bootable bundle containing `kernel` and `rootfs.img` was found under `/tmp` or existing `tvz-e2e-*` runtime dirs. Only checked-in fixture bundles were found.
- Host smoke wrapper dry-run succeeded with the fixture bundle when using the repo Python venv explicitly. The first attempt without `--python` failed because system `python3` could not import `dataclass(slots=True)`.
- Helper build succeeded after rerunning outside the filesystem sandbox; the sandboxed build failed only because SwiftPM could not write to the user clang module cache.
- `vz-helperctl.py check --json` after build passed helper binary, path, and entitlement checks; ping remained unavailable because no helper was intentionally left running.
- Non-VM helper daemon Unix-socket smoke passed when run outside the filesystem sandbox. The sandboxed attempt failed with helper `bindFailed(1)`, confirming the app sandbox blocks the helper socket bind path.
- Cleaned stale VZ tracker `TASK-2332 - Upload VZ host-gated smoke evidence artifact` from `In Review` to `Done` after confirming PR #2382 was merged on 2026-06-18.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Post-merge validation confirmed `origin/dev` contains PR #2620 and the guest-agent mismatch task record.
- The host can build and run the macOS VZ helper daemon smoke outside the app sandbox, but full real VZ Linux VM smoke could not run because no bootable `vz_linux` bundle path is configured or present locally.
- Stale VZ tracker cleanup completed for `TASK-2332`; no unrelated dirty checkout state was touched.
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
