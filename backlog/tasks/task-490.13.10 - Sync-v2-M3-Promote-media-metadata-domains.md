---
id: TASK-490.13.10
title: 'Sync v2 M3: Promote media metadata domains'
status: Done
labels:
- sync
- sync-v2
- m3
- media
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Promote metadata-only media domains (`media.item`, `media.keyword`, and `media.keyword_link`) into Sync v2 M3 with stable identity, conflict/tombstone handling, restore inventory, projection behavior, blob reference compatibility, and personal/workspace scope validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `media.item`, `media.keyword`, and `media.keyword_link` are advertised as supported Sync v2 domains without re-enabling the legacy `media` domain.
- [x] #2 Media metadata envelopes require stable identity and metadata-only payloads, with upsert/tombstone operations and conflict rules for divergent stable IDs.
- [x] #3 Accepted media metadata envelopes materialize into normal Sync object state and appear in restore manifest/preview/repair paths.
- [x] #4 Media domains may be enrolled in personal and workspace datasets while preserving existing personal/workspace metadata boundaries and access checks.
- [x] #5 Blob-bearing media content remains represented by existing `attachment.ref` and M2 blob paths; M3 media metadata does not transfer raw media blobs directly.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Sync v2 media metadata promotion for `media.item`, `media.keyword`, and `media.keyword_link`.

- Added core/API schema domain constants and capabilities while leaving legacy `media` rejected by v2 validation and the default registry.
- Added `MediaMetadataAdapter` with stable identity validation, metadata-only payload rejection for raw media/transcript/summary/embedding fields, upsert/tombstone support, and same-stable-ID hash conflict handling.
- Added `MediaMetadataMaterializer` to project accepted envelopes into `sync_object_state`, including tombstone state and repair replay behavior.
- Registered media metadata adapters/materializers in the default Sync v2 factory and restore inventory domain set.
- Allowed media metadata domains in both personal and workspace dataset enrollment boundaries.
- Added model, store, adapter, service/materializer, restore-preview, and legacy-compat regression coverage.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Promoted metadata-only media library sync domains for M3. The slice supports capability advertisement, dataset enrollment, validation, projection into restoreable object state, restore preview conflict detection, and replay/repair while preserving the legacy `media` compatibility adapter as opt-in only.

Verification:

- `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py -q` -> 240 passed.
- `python -m pytest tldw_Server_API/tests/Sync -q` -> 358 passed.
- `python -m ruff check <touched Sync v2 files>` -> all checks passed.
- `python -m bandit -r <touched production files> -f json -o /tmp/bandit_sync_v2_m3_media_metadata.json` -> 0 results.
- `git diff --check` -> clean.

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
