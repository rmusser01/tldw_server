---
id: TASK-397.6
title: Implement llama.cpp managed profile model metadata
status: Done
assignee: []
created_date: '2026-05-16 19:34'
updated_date: '2026-05-16 20:00'
labels:
  - llamacpp
  - backend
  - metadata
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the llama.cpp model-family/mmproj profile wiring plan: expose managed llama.cpp profiles through /api/v1/llm/models/metadata with capability and modality metadata while preserving bounded failure behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /api/v1/llm/models/metadata includes managed llama.cpp profile entries with profile IDs, aliases, capability metadata, and modality metadata.
- [x] #2 Metadata type and input_modality filters include or exclude managed profile entries consistently with existing catalog filtering.
- [x] #3 Invalid or stale managed profile assets produce bounded warning metadata instead of failing the entire endpoint.
- [x] #4 Focused metadata/runtime tests, Bandit on touched backend code, and git diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 3 managed profile metadata for /api/v1/llm/models/metadata. Added a public metadata builder backed by the existing llama.cpp profile capability resolver, appended supervisor-managed profile entries through the existing llm_manager path, preserved existing catalog filters, kept stale asset failures as bounded capability warnings, and documented disabled profiles as visible with is_configured=false. PR review fixes offloaded managed profile metadata collection from the async endpoint, bounded Local_LLM scan failures, added scan-truncation warnings, and expanded output_modality test coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Managed llama.cpp profiles now appear in /api/v1/llm/models/metadata with profile IDs, aliases, type, capabilities, modalities, and warning metadata. Review fixes addressed async offload, bounded scan errors, scan-limited warnings, request typing, output_modality coverage, and task metadata cleanup. Verification recorded: 15 metadata/provider tests passed, 77 focused llama.cpp backend tests passed, Bandit on touched backend files reported zero findings, and git diff --check was clean.
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
