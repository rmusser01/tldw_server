---
id: TASK-15585
title: Repair and label Document Workspace text-selection actions
status: To Do
labels:
- document-workspace
- annotations
- accessibility
- webui
dependencies:
- TASK-15584
references:
- https://www.nngroup.com/articles/recognition-and-recall/
- https://www.nngroup.com/articles/tooltip-guidelines/
- https://www.nngroup.com/articles/modal-nonmodal-dialog/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate failures in the selected-text popover, dialogs, and related highlight options. Make the available actions clear and usable while the user reads the document.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce the failing action paths and verify the available text-selection actions on synthetic PDF and EPUB fixtures, including nested Highlight and Ask AI menus and the Translate dialog.
- [ ] #2 Preserve the selected passage through menus and dialogs. Ask AI opens the active document conversation and puts the chosen prompt in the composer for review.
- [ ] #3 Give icon-only controls concise labels on mouse hover and keyboard focus. Keep essential actions visibly labeled on touch and verify keyboard dismissal and focus behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The signed-in synthetic PDF displayed an icon-only Highlight control on desktop. With Notes selected, choosing Ask AI and then Explain did not open Chat or populate a draft. No model request or annotation was created by that probe. Check the action event, selection dismissal, and the Chat mount timing before choosing a fix. Use Nielsen Norman Group guidance on recognition, contextual tooltips, and when a modal is warranted. ADR assessment for the planned repair uses the existing document and annotation contracts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
