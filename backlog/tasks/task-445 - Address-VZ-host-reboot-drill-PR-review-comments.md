---
id: TASK-445
title: Address VZ host reboot drill PR review comments
status: Done
labels:
- sandbox
- vz-linux
- review-fix
priority: high
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
- tools/macos-vz-helper/README.md
- backlog/tasks/task-445 - Address-VZ-host-reboot-drill-PR-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address still-open review comments on PR #1868 after rebasing onto dev. Scope includes host reboot drill README command durability across reboot, bundle metadata normalization, manifest write atomicity, cleanup error handling, and docstrings for new security/operations helpers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Open review findings are verified against current code before fixes are applied.
- [x] #2 README host reboot drill examples remain copy/paste-safe across reboot boundaries.
- [x] #3 Missing bundle metadata remains empty instead of becoming cwd-dependent.
- [x] #4 Manifest writes are private and atomic enough to avoid partial final JSON on interruption.
- [x] #5 Manifest reader cleanup cannot crash the CLI from fd close failures.
- [x] #6 Focused tests and security checks pass after changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Verified live PR #1868 review threads after rebase. The cubic README variable persistence and bundle metadata findings were valid. The Qodo docstring, fd-close cleanup, and atomic manifest write findings were valid.
- Updated README host-reboot drill examples so post-reboot commands rehydrate evidence/socket/launchd variables from durable run-id files instead of relying on shell state across reboot.
- Changed missing bundle metadata normalization so `Path("")` remains an empty manifest value instead of resolving to the current working directory.
- Made private JSON manifest writes use a same-directory temp file, `fsync()`, and `os.replace()` while preserving existing final manifest content on serialization failure.
- Suppressed `OSError` from fd cleanup in pre-manifest reading and added docstrings for new host-reboot evidence helpers.
- Verification: `py_compile` passed for `vz-helperctl.py`; focused helper tests passed with `188 passed, 6 skipped, 2 warnings`; `git diff --check` passed; Bandit reported no errors/results for `vz-helperctl.py`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all currently open PR #1868 review threads from cubic and Qodo with narrow helper/docs/test updates, and verified the focused helper suite plus compile, diff, and Bandit checks.
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
