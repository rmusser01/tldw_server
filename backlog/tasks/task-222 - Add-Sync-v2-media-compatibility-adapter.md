---
id: TASK-222
title: Add Sync v2 media compatibility adapter
status: Done
assignee: []
created_date: '2026-05-10 06:04'
updated_date: '2026-05-10 06:20'
labels:
  - sync
  - media
  - adapter
dependencies:
  - TASK-220
references:
  - tldw_Server_API/app/core/Sync/sync_contract.py
  - tldw_Server_API/app/core/Sync/v2/adapters.py
  - tldw_Server_API/tests/MediaDB2/test_sync_server.py
  - tldw_Server_API/tests/MediaDB2/test_sync_client.py
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from the Chatbook sync engine implementation plan: add the Sync v2 media compatibility domain adapter while preserving existing legacy media sync behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Registers a Sync v2 media adapter that accepts legacy media sync entities through the v2 adapter registry.
- [x] #2 Maps legacy Media, Keywords, and MediaKeywords semantics into Sync v2-compatible envelopes or validation behavior without exposing plaintext private payloads.
- [x] #3 Covers create/update/delete/link/unlink semantics for legacy media sync rows.
- [x] #4 Keeps existing MediaDB2 legacy sync tests passing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect Sync v2 adapter/service validation and legacy media sync contract for allowed entity-operation pairs.
2. Add focused tests for the media compatibility adapter covering legacy Media/Keywords/MediaKeywords acceptance, invalid combinations, link metadata, unsupported versions through the service, and private payload plaintext exclusion in conversion helpers.
3. Implement a Sync v2 media domain adapter and legacy-to-envelope helper in the new domain_adapters package while keeping the existing adapter protocol unchanged.
4. Register the concrete media adapter in the default Sync v2 endpoint registry while preserving static adapters for other domains.
5. Run focused Sync v2, legacy MediaDB2 sync, Bandit, and diff whitespace verification; update TASK-222 and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation in worktree .worktrees/codex-sync-v2-schemas on branch codex/sync-v2-schemas. Using tests-first flow and keeping legacy /sync/send and /sync/get behavior unchanged.

Implemented MediaCompatibilityAdapter with legacy Media/Keywords create-update-delete validation, MediaKeywords link-unlink validation, default Sync v2 media registry wiring, and legacy sync-log to SyncEnvelopeCreate conversion that keeps plaintext fields out of payload_clear for client_private_v1 payloads.

Verification: new media compatibility test red run failed during collection before implementation because domain_adapters.media did not exist; after implementation, focused tests passed. Final runs: pytest tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py tldw_Server_API/tests/MediaDB2/test_sync_server.py tldw_Server_API/tests/MediaDB2/test_sync_client.py -q => 58 passed, 5 warnings. pytest tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q => 16 passed, 5 warnings. Bandit touched production scan => 0 findings, JSON at /tmp/bandit_sync_v2_media_adapter.json. git diff --check => passed.

Review follow-up: fixing decoded dict payload support in legacy_media_sync_log_to_envelope. Root cause verified in media_db/runtime/sync_log_ops.py: get_sync_log_entries json.loads() decodes payload strings into dicts before returning rows. Current adapter coerces payload with str(...), which turns dicts into Python repr strings and breaks json.loads(). Plan: add regression test for decoded dict payload with canonical compact JSON hash/size expectations, then update payload normalization narrowly and rerun focused tests, Bandit, and diff check.

Review fix implemented: legacy payload normalization now accepts decoded mapping payloads, JSON object strings, and empty/None payloads. Mapping payload hash and size are computed from canonical compact JSON with sorted keys; non-object JSON values and non-string/non-mapping payload values raise ValueError. Regression coverage added for MediaDatabase-style decoded dict payloads and non-object payload rejection.

Review fix verification: pytest tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py -q => 20 passed, 5 warnings. pytest tldw_Server_API/tests/Sync/test_sync_v2_media_compat.py tldw_Server_API/tests/MediaDB2/test_sync_server.py tldw_Server_API/tests/MediaDB2/test_sync_client.py -q => 62 passed, 5 warnings. Bandit media adapter scan => 0 findings, JSON at /tmp/bandit_sync_v2_media_payload_fix.json. git diff --check => passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Sync v2 media compatibility adapter and tests. The adapter validates legacy Media, Keywords, and MediaKeywords entity-operation semantics through the existing adapter registry, preserves MediaKeywords link/unlink metadata requirements, and includes a legacy sync-log conversion helper that maps legacy rows into Sync v2 envelopes without copying private plaintext title/content/keyword fields into payload_clear. The default Sync v2 endpoint registry now uses this concrete adapter only for the media domain while leaving other domains on StaticSyncAdapter. Review follow-up: the conversion helper now accepts MediaDatabase decoded dict payloads and computes their hash/size from canonical compact JSON, while rejecting non-object payload values.
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
