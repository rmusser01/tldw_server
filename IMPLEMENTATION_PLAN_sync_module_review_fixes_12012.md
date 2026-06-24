## Stage 1: Regression Coverage
**Goal**: Add focused tests for each validated Sync review finding before changing implementation code.
**Success Criteria**: Tests fail for the reviewed defects, not for setup or syntax issues.
**Tests**: Targeted Sync v2 service/domain adapter tests and MediaDB2 legacy sync client tests.
**Status**: Complete

## Stage 2: Sync V2 Hardening
**Goal**: Fix blob chunk idempotency, payload size accounting, and tombstone adapter handling with minimal changes.
**Success Criteria**: Conflicting duplicate chunks cannot overwrite staged data, inline envelope payloads are size-limited, and tombstone envelopes follow delete conflict semantics.
**Tests**: New and existing `tldw_Server_API/tests/Sync` targeted tests.
**Status**: Complete

## Stage 3: Legacy Sync Client Fixes
**Goal**: Restore media FTS update support and prevent outbound sync from stalling behind other clients' log rows.
**Success Criteria**: Media title/content updates refresh FTS without crashing, and skipped non-local sync rows advance the local sent marker.
**Tests**: New and existing `tldw_Server_API/tests/MediaDB2/test_sync_client.py` targeted tests.
**Status**: Complete

## Stage 4: Verification And Task Finalization
**Goal**: Run focused tests and Bandit on touched Sync code, then update Backlog task records.
**Success Criteria**: Targeted tests pass, Bandit results are recorded, and `TASK-12012` has touched files plus final notes.
**Tests**: Targeted pytest commands and Bandit on the touched Sync scope.
**Status**: Complete
