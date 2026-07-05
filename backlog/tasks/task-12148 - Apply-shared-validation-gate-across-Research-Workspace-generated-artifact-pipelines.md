---
id: TASK-12148
title: >-
  Apply shared validation gate across Research Workspace generated artifact
  pipelines
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 00:54'
labels: []
dependencies: []
references:
  - TASK-12147
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add one reusable Research Workspace generated-content validation gate for quizzes, flashcards, audio summaries, adaptive data tables, slides, and mindmaps. The gate should reuse the existing claims/source-pack validation path and asset validators rather than copying bespoke checks into each pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 generated content pipelines can opt into the same source-pack claims validation contract
- [x] #2 artifact validators reject placeholder or invalid assets using shared checks instead of per-pipeline ad hoc strings
- [x] #3 user-facing metadata records whether validation ran, which validator/model was used, and any unresolved warnings
- [x] #4 tests cover at least one happy path and one invalid/placeholder asset for each generated artifact family
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

- Added shared workspace artifact export validator at tldw_Server_API/app/core/Workspaces/artifact_validation.py.
- Export metadata now includes artifact_validation with gate status, validator id, claims-required flag, claims validator/model, unsupported claim count, and warnings.
- Pipelines opt in by setting producer_metadata.claims_validation_required (or compatible aliases) and storing a claims report under review/version/producer metadata.
- Placeholder checks cover quiz, flashcards, audio_summary, adaptive/data tables, slides/presentations, and mindmaps at the shared export boundary.
- Verification: pytest tldw_Server_API/tests/Workspaces/test_workspace_artifact_validation.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q => 110 passed; Bandit touched Workspaces implementation => 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a shared Workspace artifact export validation gate for generated content families. The gate rejects placeholder/invalid generated artifacts, enforces opt-in claims-validation metadata when requested, rejects unresolved unsupported claims, and records artifact_validation metadata in exports. Covered quizzes, flashcards, audio summaries, adaptive/data tables, slides/presentations, and mindmaps with unit/property-style tests plus an API export regression. Verification passed: 110 Workspaces tests and Bandit on touched Workspaces implementation with 0 findings. No known blockers.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
