---
id: TASK-15658
title: Preserve Quick Ingest drafts through clear and replacement review
status: In Progress
labels:
- webui
- ux
dependencies:
- TASK-15656
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
- [ ] #4 Publish a PR to dev with desktop and mobile browser evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design and acceptance walkthrough in PR #3633. No new ADR is needed. Ephemeral editor state and Undo stay in the existing WebUI draft. The submission and persistence contracts stay the same.
Work stages: reproduce loss and incorrect review scope, implement reversible clear and review navigation, then verify the real WebUI at desktop and phone sizes and publish the screenshots.
The editor stays mounted while reviewing. Undo holds the latest cleared queue in this draft, restores the original file objects and preserves newer sources, including same-length pastes. The review shows potential replacements, confirmed playlist matches and the total run count. The projected scope and submitted overwrite flags agree for the mixed playlist walkthrough.
496 focused WebUI tests passed. English JSON messages match their browser mirror, source and published guides match, and scoped ESLint passes. Bandit found no Python files in the touched frontend scope.
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
