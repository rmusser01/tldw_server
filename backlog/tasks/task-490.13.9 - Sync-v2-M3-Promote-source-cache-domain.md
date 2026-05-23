---
id: TASK-490.13.9
title: 'Sync v2 M3: Promote source cache domain'
status: Done
labels:
- sync
- sync-v2
- m3
- source-cache
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Promote `source_cache.entry` into Sync v2 M3 as the first broader-domain expansion slice, with stable identity, conflict/tombstone handling, restore inventory, projection behavior, and workspace/personal isolation checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `source_cache.entry` is advertised as a supported Sync v2 domain without re-enabling the legacy `source_cache` domain.
- [x] #2 Source cache envelopes require stable source/cache identity, content hash/provenance metadata, and support upsert/tombstone conflict rules.
- [x] #3 Accepted source cache envelopes materialize into normal Sync object state and are included in restore manifest/preview/repair paths.
- [x] #4 Personal/workspace dataset scope validation and access checks apply to source cache datasets without mixing workspace metadata domains.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `source_cache.entry` to Sync v2 supported domain/operation constants and capability schemas while keeping legacy `source_cache` out of the default registry.
- Promoted the source cache adapter to `source_cache.entry`, requiring `source_id`, `content_hash`, and provenance metadata, with upsert/tombstone conflict handling.
- Added a source-cache materializer that records accepted envelopes in `sync_object_state` so restore preview and repair can operate on normal Sync object state.
- Allowed source-cache datasets in both personal and workspace scopes while leaving personal Notes/Chat and workspace metadata boundaries intact.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Promoted `source_cache.entry` as the first Stage 5 broader-domain slice. The domain is now advertised, accepted in personal/workspace datasets, validated through the adapter, materialized into restoreable object state, and included in restore preview/repair paths. Verification passed with targeted Sync tests, the full `tldw_Server_API/tests/Sync` suite, Ruff, Bandit, and `git diff --check`.
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
