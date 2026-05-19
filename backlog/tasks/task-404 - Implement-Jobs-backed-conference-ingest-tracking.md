---
id: TASK-404
title: Implement Jobs-backed conference ingest tracking
status: Done
labels:
- bulk-conference-ingest
- quick-ingest
- media-jobs
priority: High
modified_files:
- Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py
- tldw_Server_API/app/services/media_ingest_jobs_worker.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/app/api/v1/schemas/media_request_models.py
- tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_collections_dual_write.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_add_deps_error_mapping.py
- apps/packages/ui/src/components/Common/QuickIngest/types.ts
- apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
- apps/packages/ui/src/services/tldw/quick-ingest-batch.ts
- apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
- apps/packages/ui/src/store/quick-ingest-session.ts
- apps/packages/ui/src/entries/background.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media ingest job submission accepts collection/run metadata, validates per-URL arrays, and preserves planned collection item IDs in job status without storing secrets.
- [x] #2 Media ingest worker maps terminal job outcomes to planned collection item statuses: processing, completed, skipped_existing, failed, cancelled, submit_failed, and missing-media-id failed.
- [x] #3 Frontend persisted Quick Ingest tracking restores collection/run IDs, planned item mapping, durable/degraded mode, counts, and failed URL export affordances; retry remains in the existing Results Panel controls and later selected-subset retry task.
- [x] #4 Focused backend and frontend tests cover planned item binding, worker status updates, synchronous fallback binding, and restored run tracking.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md#task-4-jobs-backed-ingest-run-tracking
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Task 4 against the existing media jobs and collections contracts. Job submission now carries `collection_id`, `planned_item_id`, and `idempotency_key`, validates per-URL planned/idempotency arrays, and marks planned items `submit_failed` if job creation fails.

The worker marks planned items `processing`, `completed`, `skipped_existing`, `failed`, and `cancelled`, and fails closed when a terminal result lacks a media id. Synchronous fallback carries `media_collection_item_id` and `media_ingest_job_id` and resolves planned items.

Quick Ingest persisted tracking now stores `collectionId`, `plannedItemIds`, `jobIdToCollectionItemId`, and `durableMode`; `ProcessingStep` shows durable tracking and can export failed item lists.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 complete. Verification: pytest media job/fallback slice 33 passed; Vitest Quick Ingest slice 73 passed; git diff --check passed; Bandit on touched backend files reported zero findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused pytest and Vitest suites pass for touched media job and Quick Ingest paths.
- [x] #2 git diff --check passes.
- [x] #3 Bandit is run on touched backend files and reports zero findings.
- [x] #4 Plan Task 4 checkboxes and Backlog task are updated before commit.
<!-- DOD:END -->
