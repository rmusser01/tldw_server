---
id: TASK-13339
title: media_metadata and source_cache materializers are 187 identical lines
status: In Progress
assignee: []
created_date: '2026-09-22 05:00'
updated_date: '2026-09-22 18:58'
labels:
  - duplication
  - sync
dependencies: []
references:
  - tldw_Server_API/app/core/Sync/v2/materializers/media_metadata.py
  - tldw_Server_API/app/core/Sync/v2/materializers/source_cache.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The two files are the same 187 lines. diff reports 28 changed line-pairs and every one is an identifier or message string: the class name, the domain literal (media.item vs source_cache.entry), the _record_*_state function name, and the error prefixes. No structural difference anywhere.

Five projection rules are therefore stated twice: tombstoned objects may not be resurrected by upsert; a reused stable object ID with a different payload hash is a conflict not an overwrite; payload_hash is mandatory; object revision advances by one; the tombstone path preserves the prior object hash.

Live cost is asymmetric coverage - test_sync_v2_media_compat.py exercises media_metadata directly while source_cache has no dedicated materializer suite, so a rule fixed on one side and missed on the other would not be caught. A third metadata-only domain makes it three files.

Destination: materializers/metadata_only.py with one MetadataOnlyMaterializer(domain, error_prefix, label); the two concrete classes become three-line constructions. Error-code strings must stay verbatim because clients match them.

Source: synthesis F38
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One parameterised metadata-only materializer
- [ ] #2 Error code strings unchanged
- [ ] #3 One table-driven suite covers both domains
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE. The two 187-line materializers now construct one shared implementation.

NEW: core/Sync/v2/materializers/metadata_only.py - MetadataOnlyMaterializer(domain, code_prefix, label, lower_label, noun). media_metadata.py (33 lines) and source_cache.py (27 lines) are now thin factories. 374 duplicated lines -> one implementation.

STRINGS PARAMETERISED, NOT UNIFIED - the finding said error codes must stay verbatim; on inspection the MESSAGES also differ in wording, not just label ("tombstoned object" vs "tombstoned entry", "Unsupported media metadata operation" vs "Unsupported source_cache.entry operation"). Four parameters reconstruct every emitted string byte-exactly: {code_prefix}_projection_failed / _tombstoned / _hash_mismatch / _object_id, and the four message forms. Nothing a client matches on has changed.

Kept as FUNCTIONS with the original CapWords names: SyncMaterializer is a structural Protocol, no isinstance check exists anywhere (grepped), so every call site is untouched - including factory.py:145-148, which registers MediaMetadataMaterializer for THREE domains (media.item, media.keyword, media.keyword_link).

METHOD - characterisation test first, green BEFORE and after:
tests/Sync/test_metadata_only_materializer_parity.py, 19 cases covering both domains on every failure path: missing payload_hash, tombstone resurrection, reused object ID, unsupported operation, idempotent same-hash upsert, foreign-domain skip, tombstone recording, and the client-visible conflict metadata dict (including the {prefix}_object_id key). Green at 19 before the refactor, still 19 after.
This is also the first dedicated materializer coverage source_cache has ever had - the asymmetric-coverage risk the finding named.

One branch could NOT be characterised and is preserved verbatim: `envelope.server_cursor is None` is unreachable through the model, because SyncEnvelope.__post_init__ raises "server_cursor is required". Noted in the test.

REGRESSION: test_sync_v2_service 165 passed; media_compat + factory + parity 101 passed. test_sync_v2_domain_adapters shows 3 failed / 48 passed BOTH with and without the refactor (stash-isolated) - those are 3 of the 12 pre-existing failures catalogued in TASK-13344, not caused here.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
