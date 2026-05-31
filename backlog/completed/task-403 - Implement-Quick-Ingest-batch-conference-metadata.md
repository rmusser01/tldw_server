---
id: TASK-403
title: Implement Quick Ingest batch conference metadata
status: Done
labels:
- bulk-conference
- quick-ingest
- webui
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 3 for the bulk conference ingestion workflow plan. Add shared conference metadata, per-item overrides, and submission payload support to Quick Ingest for playlist/conference batches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Quick Ingest exposes shared conference metadata fields for likely playlist or multi-video batches.
- [x] Quick Ingest supports per-item title, speaker, track, tag, and selected-item overrides.
- [x] Wizard session persistence carries conference metadata, playlist metadata, and item overrides into processing payloads.
- [x] Direct WebUI and extension-runtime submission paths create durable planned media collection items before job submission.
- [x] Submit failures mark planned collection items as `submit_failed`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ConferenceBatchMetadata` and `ConferenceItemMetadataOverride` to the Quick Ingest queue model.
- Added compact batch and item metadata controls that appear for playlist items and likely multi-video conference batches.
- Persisted conference metadata and playlist/item override metadata through the Quick Ingest session store.
- Added collection payload merge helpers in `conference-collections.ts`.
- Updated direct and background Quick Ingest submission paths to create collections/items, attach planned item IDs to job fields, and patch item status on submit/progress/failure.
- Verification: focused Vitest suite passed, 74 tests across wizard integration/session, conference collection service, and batch submission. `git diff --check` passed. Shared UI `tsc` still fails on the existing repo-wide baseline; a touched-file filter for Quick Ingest/conference/background paths produced no matches.
- Bandit was not run because this slice only touched TypeScript/TSX, plan docs, and Backlog task files.
- Backlog hygiene follow-up on 2026-05-19: moved this already-completed Quick Ingest record from `backlog/tasks` to `backlog/completed` by exact path so active-board checks do not mistake it for remaining sprint work. ID-based completion was intentionally avoided because unrelated `TASK-403` records also exist.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 complete: Quick Ingest can capture shared conference metadata, per-item overrides, carry them through session persistence, and create durable planned collection items before batch job submission for direct and extension-runtime paths.
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
