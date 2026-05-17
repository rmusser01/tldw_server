---
id: TASK-408
title: Add duplicate and failure recovery for conference ingest
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 01:56'
labels:
  - bulk-conference-ingest
  - quick-ingest
  - recovery
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 8 from the bulk conference ingest plan: duplicate policy controls and conservative failure/retry recovery for conference playlist batches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate policy choices cover skip, overwrite, update metadata only, and include existing in collection without surprising overwrite defaults.
- [x] #2 Failure taxonomy classifies common ingest failures into conservative retry/user-action categories.
- [x] #3 Retry selected only targets retryable submit_failed, failed, or cancelled collection items and skips completed items.
- [x] #4 Focused backend/frontend tests cover duplicate policy, failure taxonomy, and selected-subset retry behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Task 8 duplicate/failure recovery slice. Added duplicate policy resolution that only treats duplicate_existing and duplicate_in_batch as confirmed duplicates; unknown/new stay planned. Added Quick Ingest preflight duplicate policy controls and queue metadata propagation. Direct quick-ingest batch now skips non-overwrite duplicate policy items without submitting ingest jobs, forces overwrite only for the overwrite policy, preserves collection item/idempotency metadata in results, and marks submit_failed/failed outcomes with durable collection item IDs. Wizard results now builds retry-all payloads from durable collection item IDs plus retry attempt/idempotency metadata and excludes legacy non-durable failures from durable retry-all.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 8 complete: duplicate policies, conservative failure taxonomy helpers, durable retry request construction, direct duplicate skip/overwrite handling, and focused backend/frontend coverage are in place. Verification: pytest playlist preflight + media ingest worker 26 passed; focused Vitest suites 48 passed; git diff --check passed; Bandit on touched backend playlist preflight module produced zero findings. Full frontend tsc still has pre-existing baseline errors outside this slice in EmbeddingsModelSelectionConfig.tsx, persona-visuals.ts, and lib/api/vnPlay.ts.
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
