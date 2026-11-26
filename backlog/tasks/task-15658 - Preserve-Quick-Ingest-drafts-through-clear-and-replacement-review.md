---
id: TASK-15658
title: Preserve Quick Ingest drafts through clear and replacement review
status: Done
labels:
- webui
- ux
dependencies:
- TASK-15656
references:
- https://github.com/rmusser01/tldw_server/pull/3638
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved draft preservation and replacement review from PR 3633 on the responsive WebUI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clear all offers a scoped Undo and restores source order, files and metadata.
- [x] #2 Back from replacement review restores the editor, unsent inputs and focus.
- [x] #3 Replacement review identifies affected sources and distinguishes potential replacements from the total submitted.
- [x] #4 Publish a PR to dev with desktop and mobile browser evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design and acceptance walkthrough in PR #3633. No new ADR is needed. Ephemeral editor state and Undo stay in the existing WebUI draft. The submission and persistence contracts stay the same.
Work stages: reproduce loss and incorrect review scope, implement reversible clear and review navigation, then verify the real WebUI at desktop and phone sizes and publish the screenshots.
The editor stays mounted while reviewing. Undo holds the latest cleared queue in this draft, restores the original file objects and preserves newer sources, including same-length pastes. The review shows potential replacements, confirmed playlist matches and the total run count. The projected scope and submitted overwrite flags agree for the mixed playlist walkthrough.
496 focused WebUI tests passed. English JSON messages match their browser mirror, source and published guides match, and scoped ESLint passes. Bandit found no Python files in the touched frontend scope.
PR #3638 targets dev and embeds four verified screenshots. The two viewports cover going Back by keyboard, unsent text and links, preserved scroll, restoring sources, and submitting only after confirmation. All 13 browser scenarios passed, including the equal-length paste regression. WebUI TypeScript passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered Undo for Clear all and a replacement review that preserves the draft, restores focus and shows the run and replacement counts. Published the implementation, user guide and desktop and mobile evidence in PR #3638. The branch builds on PR #3636.
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
