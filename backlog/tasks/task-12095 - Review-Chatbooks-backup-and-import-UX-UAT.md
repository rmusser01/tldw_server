---
id: TASK-12095
title: Review Chatbooks backup and import UX/UAT
status: Done
labels:
- ux
- uat
- chatbooks
- webui
- extension
references:
- https://www.nngroup.com/articles/ten-usability-heuristics/
- https://www.nngroup.com/articles/how-to-conduct-a-heuristic-evaluation/
- https://www.nngroup.com/articles/how-to-rate-the-severity-of-usability-problems/
- https://www.nngroup.com/articles/cognitive-walkthroughs/
modified_files:
- Docs/superpowers/specs/2026-07-09-chatbooks-backup-import-uat-ux-design.md
- Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md
- backlog/tasks/task-12095 - Review-Chatbooks-backup-and-import-UX-UAT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform a Nielsen Norman Group-informed UAT and UX/HCI review of Chatbooks backup/import flows across WebUI, browser extension, Settings shortcuts, and OpenWebUI migration paths. Determine whether backup/import is possible, straightforward, and easy; produce severity-ranked findings and a minimal remediation spec.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-07-09-chatbooks-backup-import-uat-ux-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed NN/g-informed UAT/UX review for Chatbooks backup/import. Report written at Docs/Reviews/CHATBOOKS_BACKUP_IMPORT_UAT_UX_REVIEW_2026_07_09.md. Verdict: backup/import is technically possible, but complete backup and archive restore are not straightforward or easy. P0 findings: documented backup-all path conflicts with UI/backend selection semantics; archive restore defaults send unsupported import_media=true. P1 findings: Settings is a conversation-ID-only shortcut, OpenWebUI hydration requires remembered conversation IDs, and the visible Playground naming weakens data-safety trust. Verification: targeted OpenWebUI import/hydration unit test passed 7/7; full live browser UAT skipped because local servers were not running and P0 failures were statically provable. Bandit skipped because this task touched docs/backlog only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
