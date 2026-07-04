---
id: TASK-12148
title: Apply shared validation gate across Research Workspace generated artifact pipelines
status: To Do
priority: High
references:
- TASK-12147
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add one reusable Research Workspace generated-content validation gate for quizzes, flashcards, audio summaries, adaptive data tables, slides, and mindmaps. The gate should reuse the existing claims/source-pack validation path and asset validators rather than copying bespoke checks into each pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 generated content pipelines can opt into the same source-pack claims validation contract
- [ ] #2 artifact validators reject placeholder or invalid assets using shared checks instead of per-pipeline ad hoc strings
- [ ] #3 user-facing metadata records whether validation ran, which validator/model was used, and any unresolved warnings
- [ ] #4 tests cover at least one happy path and one invalid/placeholder asset for each generated artifact family
<!-- AC:END -->

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
