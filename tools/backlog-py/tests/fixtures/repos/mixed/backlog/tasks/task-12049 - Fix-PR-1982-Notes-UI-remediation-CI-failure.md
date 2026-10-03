---
id: TASK-12049
title: Fix PR 1982 Notes UI remediation CI failure
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-27 06:27'
labels:
  - ci
  - pr-1982
  - notes-ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the current PR #1982 current-head failure in Notes Remediation Targeted Gates / Notes UI Remediation Vitest. Root evidence: job 83795232974 failed because NotesManagerPage.stage39.organization-model test waited for notes-save-notebook to enable after applying a keyword picker selection, but the button remained disabled. Scope: diagnose and address the flaky/actionable Notes UI test path, then re-run focused verification and update PR #1982.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

CI root cause: Stage 39 clicked checkbox label text inside AntD checkbox, then applied filters without confirming the controlled checkbox state updated. On current CI head efb24974, notes-save-notebook remained disabled because no keyword token was applied. Fix: select the stable notes-keyword-picker-option-research checkbox input directly and wait for it to be checked before applying filters. Verification: focused Stage 39 test passed; exact Notes UI remediation seven-file Vitest command passed locally with 7 files / 32 tests. Bandit: skipped, touched file is a frontend TSX test only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the current PR #1982 Notes UI remediation failure by making the Stage 39 notebook-save test select the keyword picker checkbox via its stable test id and wait for checked state before applying filters. Local verification passed for the focused Stage 39 case and the exact seven-file Notes UI remediation Vitest command.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
