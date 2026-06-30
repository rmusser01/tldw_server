---
id: TASK-490.13.13
title: 'Sync v2 M3: Add encryption policy validation models'
status: Done
labels:
- sync
- sync-v2
- m3
- encryption
- tdd
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Start Stage 6 by adding validation models for Sync v2 encryption policies: server_trusted_v1, passphrase_wrapped_v1, device_wrapped_v1, and client_private_v1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failing tests are written first for the four policy modes and watched red before production code changes.
- [x] #2 Core and API schema models validate required policy metadata for server-trusted, passphrase-wrapped, device-wrapped, and client-private policies.
- [x] #3 Invalid or incomplete policy metadata fails closed with clear validation errors and no secret material is required in the model payload.
- [x] #4 Roadmap Stage 6 Step 1 is checked off and Backlog records verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Stage 6 policy validation model slice.

- Added failing tests first in `test_sync_v2_models.py`; red run showed 11 expected failures because `SyncEncryptionPolicyMetadata` did not exist.
- Expanded the shared Sync v2 `EncryptionPolicy` literal to include `server_trusted_v1`, `passphrase_wrapped_v1`, `device_wrapped_v1`, and `client_private_v1`.
- Added core and API `SyncEncryptionPolicyMetadata` models for public, non-secret policy metadata.
- Validates `key_epoch >= 1`.
- Validates `server_trusted_v1` attestation metadata: configured, scope, and covered storage files.
- Validates `passphrase_wrapped_v1` metadata: KDF algorithm, `sha256:` params hash, and recovery key record reference.
- Validates `device_wrapped_v1` metadata: at least one device key record reference.
- Validates `client_private_v1` metadata: server materialization must be `metadata_only`.
- API model forbids extra fields and hides input values in validation errors so rejected secret-like fields such as `wrapped_key_blob` do not leak their values.
- Store and envelope admission intentionally still require `server_trusted_v1`; enabling stricter policies belongs to later Stage 6 storage/service slices.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Sync v2 M3 encryption policy metadata validation for all four planned policy modes without enabling stricter policies for dataset/envelope storage yet.

Verification:

- Red: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py -q` -> 11 failed, 36 passed; failures were expected missing `SyncEncryptionPolicyMetadata`.
- Green targeted: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py -q` -> 185 passed.
- Full Sync: `python -m pytest tldw_Server_API/tests/Sync -q` -> 369 passed.
- Ruff: `python -m ruff check tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/core/Sync/v2/__init__.py tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/tests/Sync/test_sync_v2_models.py` -> all checks passed.
- Bandit: `python -m bandit -r tldw_Server_API/app/api/v1/schemas/sync_v2_models.py tldw_Server_API/app/core/Sync/v2/__init__.py tldw_Server_API/app/core/Sync/v2/models.py -f json -o /tmp/bandit_sync_v2_m3_encryption_policy_models.json` -> 0 results.
- `git diff --check` -> clean.

Next handoff: Stage 6 Step 2 key epoch storage and rotation state.

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
