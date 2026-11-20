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
updated_date: 2026-11-20 05:06
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
PR #3512 merge follow-through. Rebased the section, chapter and batching commits onto current dev. Reproduced the database surface ratchet before fixing it. Move the chapter lookup into the ChaCha store layer, preserve PostgreSQL owner predicates and SQLite device compatibility, verify the ratchets and Study Pack regressions, then seek independent review. ADR required: no. ADR-003, ADR-025 and ADR-080 continue to govern Jobs, provider routing and database checkout ownership.
The independent review found whole-job retries on terminal generation errors, a book title masking the first chapter, preview authority that differed from the background job, and parent citation bounds persisting across splits. Worker, heading and citation-offset and preview regressions failed before the corresponding fixes and passed afterward (7 focused tests). The chapter query uses the public database API. The original CI mypy step is advisory; matching local checks found no new diagnostics relative to rebased dev. OpenAPI fingerprint, 51 UI tests and unchanged Bandit findings verified.
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
