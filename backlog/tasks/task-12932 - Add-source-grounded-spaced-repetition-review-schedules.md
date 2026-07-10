---
id: TASK-12932
title: Add source-grounded spaced repetition review schedules
status: To Do
references:
- 'Design spec: Docs/superpowers/specs/2026-07-09-source-grounded-spaced-repetition-design.md'
- Spec review approved by subagent 019f4a0a-f590-75e0-9fc6-ff1f4d48eba6
- 'Implementation plan: Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md'
- Plan review approved by subagent 019f4a19-7d71-7bf2-8425-db97d8674734
modified_files:
- Docs/superpowers/specs/2026-07-09-source-grounded-spaced-repetition-design.md
- Docs/superpowers/plans/2026-07-09-source-grounded-spaced-repetition-implementation-plan.md
- backlog/tasks/task-12932 - Add-source-grounded-spaced-repetition-review-schedules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a first-pass source-grounded spaced repetition workflow so users can schedule future reviews from highlighted or selected source-grounded material and choose the review activity.

Acceptance criteria:
- Users can create a custom review schedule such as Day 1, Day 3, Day 7, Day 14, Day 28, 3 months, and 6 months from selected source-grounded content.
- Scheduled reviews preserve links back to the originating note/source context.
- The first slice supports a small set of reminder activities using existing capabilities where possible: reread note/source excerpt, quiz, flashcards, and fill-in-the-blank/cloze review.
- Existing flashcard/quiz scheduling behavior remains backward compatible.
- WebUI exposes a minimal review-schedule creation and due-review surface.
- Tests cover schedule creation, due-review projection, source linkage, and backward-compatible existing review flows.

Out of scope for the first slice:
- Visual recognition questions.
- Extended matching questions.
- New notification delivery channels beyond an in-app due-review surface.
- Replacing the existing flashcard scheduler model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
