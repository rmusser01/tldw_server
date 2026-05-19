---
id: TASK-409
title: Add notifications and QA docs for conference ingest
status: Done
labels:
- bulk-conference-ingest
- quick-ingest
- qa
- docs
priority: High
modified_files:
- Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md
- Docs/User_Guides/Bulk_Conference_Playlist_Ingest.md
- Docs/User_Guides/index.md
- apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/FloatingProgressWidget.test.tsx
- apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
- apps/packages/ui/src/routes/route-paths.ts
- apps/packages/ui/src/routes/option-media-collection.tsx
- apps/packages/ui/src/routes/option-media-review-route-registry.tsx
- apps/tldw-frontend/pages/media-collections/[collectionId].tsx
- apps/tldw-frontend/extension/routes/option-media-collection.tsx
- apps/tldw-frontend/extension/routes/route-registry.tsx
- apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 9 from the bulk conference ingest plan: mocked full-path QA coverage, completion notification affordance, and user documentation for the bulk conference playlist workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deterministic 34-item playlist fixture/test coverage exercises bulk preflight and mixed outcomes without real YouTube/downloads.
- [x] #2 Quick Ingest completion notification summarizes collection name and mixed success/failure/skipped counts without overstating search readiness.
- [x] #3 User documentation covers playlist preflight, conference metadata, durable/degraded modes, failure export/retry, collection review, and scoped Knowledge QA readiness.
- [x] #4 Focused verification, diff check, and Bandit status are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a mocked 34-talk playlist fixture in the media ingest Playwright workflow, including duplicate-existing, duplicate-in-batch, deselection, one mocked processing failure, collection creation, and collection review navigation.
- Added WebUI/extension shared handoff coverage for the active-tab playlist event path without real YouTube requests or downloads.
- Added a minimized Quick Ingest completion summary for conference collections, including collection name and succeeded/skipped/failed/cancelled counts without claiming search readiness.
- Wired the durable collection result CTA to a shared `/media-collections/:collectionId` route for WebUI and extension shells.
- Added the bulk conference playlist user guide and linked it from the user guide index.
- `Docs/API-related/Media_Ingest_Jobs_API.md` and `tldw_Server_API/tests/frontend_e2e/test_quick_ingest_media_workflow.py` did not need edits for this QA/docs slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 9 completed. The conference workflow now has deterministic browser QA for a 34-item playlist through preflight, shared metadata, mixed mocked ingest outcomes, extension handoff, and collection review. The floating completion widget summarizes mixed collection outcomes without overstating readiness, and user documentation covers preflight, duplicate policies, durable/degraded behavior, recovery, review, and scoped QA readiness.

Verification recorded:
- Backend focused pytest: `46 passed, 9 warnings`.
- Frontend focused Vitest: `7 files passed`, `68 tests passed`.
- Additional FloatingProgressWidget focused Vitest after final helper polish: `1 passed`.
- Focused Playwright conference tests: `2 passed`.
- Full `media-ingest.spec.ts` Playwright file: `15 passed`, `12 skipped`, `3 failed` due to existing broader-file drift outside the new conference workflow: missing legacy media search textbox, missing legacy empty-state "open quick ingest" trigger, and unstable legacy review-route link click.
- `tsc --noEmit`: still blocked only by known baseline errors in `EmbeddingsModelSelectionConfig.tsx`, `persona-visuals.ts`, and `lib/api/vnPlay.ts`.
- `git diff --check`: passed.
- Bandit touched backend sweep: zero findings in `/tmp/bandit_bulk_conference_final.json`.
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
