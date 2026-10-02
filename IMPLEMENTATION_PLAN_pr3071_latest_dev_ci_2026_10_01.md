# PR3071 Latest-Dev And CI Reconciliation

Tracking: TASK-13408. Requester supplied Change summary on 2026-10-01.
Baseline: PR head 792b6c4a; new dev 5f3ed81e. Preserve existing services,
credentials, original browser tabs/data, stashes and unrelated working files.

## Stage 1: Reconcile Latest Dev
**Goal**: Preserve both branches' authoritative history, cancellation, retry and
raw-content behavior across seven textual and additional semantic overlaps.
**Success Criteria**: No conflict markers, no discarded upstream fixes or weakened
owner/admission/audit checks; retained runtime environments and original data.
**Tests**: Merge preview, overlap review, TypeScript and focused merged regressions.
**Status**: In Progress

## Stage 2: Repair Exact CI Failures
**Goal**: Reproduce and address total_chunks=0 validation, mandatory moderation
audit fixture failures and stale SSE cache-header expectations at their roots.
**Success Criteria**: Failing cases pass with their safety properties intact;
scoped regression, independent review and Bandit introduce no new findings.
**Tests**: Exact failed CI cases, adjacent source/moderation/streaming suites.
**Status**: In Progress

## Stage 3: Verify And Publish
**Goal**: Restore real services, rerun no-mock Chrome/CDP acceptance for merged
behavior, preserve originals and publish reviewed fixes/evidence to PR3071.
**Success Criteria**: Native receipt/citation/draft/reload and relevant isolation
checks pass; current PR checks and requester summary are verified, without merge.
**Tests**: Actual API/auth/SQLite/IndexedDB/Gemma/embedding UAT, original row/tab
and stash preservation, production bundle gates and final GitHub check snapshot.
**Status**: In Progress

Fresh checks found a captured recovery GET blocked by the transport allowlist,
a protected-image read missing bounded error translation, and valid live RAG
bookkeeping rejected by the source projection. Preserve the failed native trace;
repair the narrow shared paths and rerun real acceptance. The local Docker daemon
is unhealthy, so PostgreSQL qualification is explicitly blocked, not passed.
