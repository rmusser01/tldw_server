---
id: TASK-15575
title: Batch public Study Pack generation and verify cited answers
status: Done
dependencies:
- TASK-15574
references:
- https://github.com/rmusser01/tldw_server/pull/3512
- Docs/ADR/003-jobs-vs-scheduler-default.md
- Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port output-limit batching and citation verification to public dev on PR 3512. Reconcile the latest dev media picker and add regression coverage before updating the draft.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Truncated batches retry with smaller passages while retaining chapter groups and resource limits.
- [x] #2 Validate source quotes and answer support before saving a deck or returning a completed job.
- [x] #3 Update PR 3512 against current dev with independently reviewed regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. ADR-003 Jobs ownership and ADR-025 gateway routing continue to govern the existing worker and provider calls. This change is internal to the Study Pack generation and validation flow.
Merged dev e656a55f80 and retained its media picker and saved-content evidence handling alongside chapter selection. Independent review approved the port and resolutions. The regressions failed before the port. After the port, 178 Study Pack backend tests and 51 focused UI tests passed. The 44 PostgreSQL cases skipped because the fixture could not connect. OpenAPI contract, Ruff, whitespace checks, and Bandit comparison passed. Model calls were mocked.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The public Study Pack worker splits and retries capped batches while preserving chapter membership. It verifies each source quote and each answer before deck persistence or completion. An unrepaired batch fails the job. Updated the Study Pack guide and PR #3512 against current dev.
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
