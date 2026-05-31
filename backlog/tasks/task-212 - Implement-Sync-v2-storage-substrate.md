---
id: TASK-212
title: Implement Sync v2 storage substrate
status: Done
assignee: []
created_date: '2026-05-10 02:44'
updated_date: '2026-05-10 03:59'
labels:
  - sync
  - storage
  - api
dependencies:
  - TASK-210
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md
  - tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 internal models, store, errors, and DB helper are added for per-user sync storage.
- [x] #2 Store tests cover device upsert idempotency, dataset enrollment idempotency, envelope insert idempotency, deterministic pull after cursor, conflict lifecycle, and key-record storage without plaintext keys.
- [x] #3 Storage tables cover sync_devices, sync_datasets, sync_domain_state, sync_envelopes, sync_device_cursors, sync_conflicts, and sync_key_records.
- [x] #4 Focused pytest for tldw_Server_API/tests/Sync/test_sync_v2_store.py passes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation commits: 5c7a62cf9 added the storage substrate; a94a14ca1 moved SQL into DB_Management and fixed layering/user_id; c1d53c6b4 hardened store invariants; b5d10e987 made idempotent envelope/key-record inserts atomic with ON CONFLICT DO NOTHING plus stored-row fingerprint checks.

Review notes: local spec review passed after the subagent review hit a usage limit. Local quality review found and fixed the read-then-insert idempotency race risk before closeout.

Verification: python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -v -> 26 passed, 5 warnings. git diff --check -> passed. Bandit touched production scope -> 0 findings, output /tmp/bandit_sync_v2_storage.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Sync v2 storage substrate with DB_Management-owned schema/helper, a thin core store facade, internal models/errors, required storage tables, user-scoped key records, dataset/domain integrity checks, immutable duplicate key records, deterministic pull listing, conflict lifecycle support, and atomic idempotent envelope/key-record inserts.
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
