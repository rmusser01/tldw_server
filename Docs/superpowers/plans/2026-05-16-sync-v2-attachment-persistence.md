## Stage 1: Store Contract
**Goal**: Add durable Sync v2 attachment rows for encrypted client payloads.
**Success Criteria**: Store tests prove schema creation, idempotent insert, duplicate drift rejection, dataset ownership checks, enrolled-domain checks, and manifest aggregate summaries from persisted attachments.
**Tests**: Focused `test_sync_v2_store.py` tests for attachment persistence and restore summaries.
**Status**: Complete

## Stage 2: Service Contract
**Goal**: Expose attachment persistence through `SyncV2Service` with capability and validation gates.
**Success Criteria**: Service tests prove attachments are supported, payloads over the configured byte limit are rejected, dataset/user/domain validation is enforced, and ciphertext is not included in safe error paths.
**Tests**: Focused `test_sync_v2_service.py` tests for `store_attachment`.
**Status**: Complete

## Stage 3: API Contract
**Goal**: Replace the `/api/v1/sync/attachments` 501 feature detector with a real upload endpoint.
**Success Criteria**: Endpoint tests prove successful upload, idempotent dedupe, 400/404/413-style safe failure mapping, and restore-manifest attachment inventory updates.
**Tests**: Focused `test_sync_v2_endpoints.py` tests for attachment upload and capabilities.
**Status**: Complete

## Stage 4: Docs and Verification
**Goal**: Document supported encrypted attachment persistence and complete quality gates.
**Success Criteria**: Sync v2 docs no longer describe attachment persistence as disabled, Backlog acceptance criteria are updated, focused pytest, Bandit, and `git diff --check` pass.
**Tests**: Focused Sync v2 pytest, Bandit on touched production paths, `git diff --check`.
**Status**: Complete
