---
id: TASK-12933
title: Fix visual identity ZIP unsafe path detection on Windows CI
status: In Progress
labels:
- ci
- visual-identities
- windows
- main-followup
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-PR #2692 main CI added a Windows Python 3.12 visual-identities failure. zipfile normalizes backslash member names to forward slashes in ZipInfo.filename on Windows, so visual identity ZIP import must validate the raw ZipInfo.orig_filename to reject unsafe archive paths consistently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] ZIP member validation rejects raw backslash member names even when `zipfile` normalizes `filename`.
- [x] The backslash fixture assertion remains cross-platform.
- [x] Archive import regression tests, diff check, and Bandit pass locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: on Windows, Python's `zipfile` keeps the archive member's raw name in `ZipInfo.orig_filename` but normalizes `ZipInfo.filename` to forward slashes. The importer used `filename`, so a raw `sprites\happy.png` member could be treated as a normal `sprites/happy.png` path and fail later as invalid image content. The fix validates and records errors against `orig_filename` while preserving the normalized safe path for accepted entries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared a second local-only fix for the Windows visual-identities CI failure. Added raw ZIP member-name validation through `ZipInfo.orig_filename`, updated the fixture assertion, and added a regression test that simulates Windows filename normalization. Verified targeted tests, the full archive import file, `git diff --check`, and Bandit on the touched importer. Patch remains unpushed pending completion of the requested CI run.
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
