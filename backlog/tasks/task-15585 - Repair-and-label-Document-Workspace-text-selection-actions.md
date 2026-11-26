---
id: TASK-15585
title: Repair and label Document Workspace text-selection actions
status: In Progress
labels:
- document-workspace
- annotations
- accessibility
- webui
dependencies:
- TASK-15584
references:
- Docs/superpowers/plans/2026-11-20-document-workspace-selection-actions.md
- Docs/ADR/049-chat-history-selection-ownership.md
- https://www.nngroup.com/articles/recognition-and-recall/
- https://www.nngroup.com/articles/tooltip-guidelines/
- https://www.nngroup.com/articles/modal-nonmodal-dialog/
- https://github.com/rmusser01/tldw_server/pull/3534
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate failures in the selected-text popover, dialogs, and related highlight options. Make the available actions clear and usable while the user reads the document.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce the failing action paths and verify the available text-selection actions on synthetic PDF and EPUB fixtures, including nested Highlight and Ask AI menus and the Translate dialog.
- [x] #2 Preserve the selected passage through menus and dialogs. Ask AI opens the active document conversation and puts the chosen prompt in the composer for review.
- [x] #3 Give icon-only controls concise labels on mouse hover and keyboard focus. Keep essential actions visibly labeled on touch and verify keyboard dismissal and focus behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The signed-in synthetic PDF displayed an icon-only Highlight control on desktop. With Notes selected, choosing Ask AI and then Explain did not open Chat or populate a draft. No model request or annotation was created by that probe. Check the action event, selection dismissal, and the Chat mount timing before choosing a fix. Use Nielsen Norman Group guidance on recognition, contextual tooltips, and when a modal is warranted. ADR assessment for the planned repair uses the existing document and annotation contracts.
The requester approved keeping the passage available through selection actions and adding hover and keyboard-focus labels plus visible touch labels. Ask AI opens Chat and appends to the active conversation draft without dispatching. ADR-049 governs the account and document boundaries. No new ADR.
Fixed the Ask AI mount race with an owner-bound workspace handoff and explicit responsive pane activation. The selection stays available in portaled menus and translation. PDF selection retains its page and EPUB retains its CFI and frame position. Reused Hosted patch 059 to open uploaded EPUB archives on Public Dev.
Verified 474 focused WebUI tests, including real Ant Design keyboard menu behavior, and 11 Python translation and request-account tests. Bandit reported zero findings on the changed endpoints. The local real-reader browser checks remain incomplete, so task acceptance still has a pending browser item.
The backend gate for [Dev PR #3534](https://github.com/rmusser01/tldw_server/pull/3534) reproduced OpenAPI drift. The optional expected-account header on Translate and Listen accounts for the entire schema change. Regenerated the shared OpenAPI contract under [ADR-069](../../Docs/ADR/069-frontend-api-contract-package.md). The drift gate and all 16 Translate and account-check tests passed locally.
All eight required GitHub checks passed on the first snapshot repair. When [the commercial model refresh](https://github.com/rmusser01/tldw_server/pull/3535) updated the Dev snapshot, regenerated it from the combined backend. Verified the two account headers are the entire schema difference from current Dev, and 359 focused shared UI and 16 backend tests passed.
Rebased after the Dev sign-in session changes and regenerated the combined OpenAPI fingerprint. The drift check, shared UI typecheck, 181 Document Workspace tests, eight focused Translate, Listen, and account-check tests, and Bandit on the changed endpoints passed locally. The next queue run passed both end-to-end gates and coverage on the updated 45-minute Dev job.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Repaired and labeled PDF and EPUB selection actions. Ask AI appends to the active conversation draft, translation retires obsolete requests, and text generation requests reject an account change before dispatch.
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
