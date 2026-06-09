---
id: TASK-2342
title: Design Scheduled Tasks Phase 4 API-first recurring question and agent task
  contract
status: Done
labels:
- scheduled-tasks
- ux
- api-design
priority: High
modified_files:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the API-first Phase 4 Scheduled Tasks design spec for Recurring Question and Agent Task automation families, including WebUI reference-client shell scope, backend dependencies, safety/result contracts, and explicit constraints from existing scheduler/jobs/RAG/ACP surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the API-first Phase 4 Scheduled Tasks product contract for Recurring Question and Agent Task. Incorporated pre-write review findings around additive API evolution, existing ACP schedules, ADR-003 Jobs-vs-Scheduler ownership, Watch/Ingest-specific capability gates, Phase 3 projected results limits, and RAG capability composition. Local spec review approved after adding an explicit Slice 4A capability fallback. Verification: git diff --check passed. Bandit skipped because only documentation and Backlog files were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved Phase 4 API-first design spec is written at Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md. User approved the spec; status was updated to Approved. Verification: git diff --check passed. Bandit skipped because only documentation and Backlog files were changed.
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
