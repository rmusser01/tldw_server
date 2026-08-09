# Task 8 Report — Capture direct organization REST mutations

## Status

Complete. Active, ready default-personal Sync datasets now make durable Sync v2
the sole authority for every currently exposed direct Notes organization mutation.
Inactive Sync retains the prior direct ChaChaNotes paths. Task 9-owned note compound
mutations and keyword merge remain fail closed.

## Implementation

- Added a small user-bound `NotesOrganizationCoordinator` with read-only planners
  for keyword, collection, folder, and deterministic relationship mutations.
- Added readiness enforcement for the complete six-domain Notes organization group
  and safe typed HTTP mappings for missing domains, non-ready state, preflight
  conflict, batch idempotency conflict, atomic append failure, and incomplete
  materialization.
- Routed keyword create/rename/delete, collection create/update/delete,
  collection-keyword link/unlink, conversation-keyword link/unlink,
  note-keyword link/unlink, and folder path create/reuse through ordered durable
  mutation groups when Sync is active and ready.
- Kept response shapes, integer REST IDs, route dependencies, optimistic-version
  headers, trailing-slash behavior, and inactive direct mutations intact.
- Covered non-public folder rename/move/delete and note-folder relationships through
  the coordinator seam without adding public endpoints.
- Preserved opaque UUIDv4 stable identities. Plaintext idempotency keys are hashed
  into resource/group identity and are not stored in envelope metadata.
- Made collection plans replay-stable after a partially materialized prefix by
  including every desired keyword upsert with its canonical stable ID and stored
  spelling. This allows the same durable group to resume instead of drifting.
- Added the minimal approved Task 4 support seam,
  `SyncServerOriginBatchAppendError`, solely around atomic insert failures. Existing
  idempotency conflicts retain their dedicated type.
- Updated legacy integration fixtures to provide the stable `sync_id` fields now
  required by keyword and collection response schemas.

## Files

- `tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py` (new)
- `tldw_Server_API/app/api/v1/endpoints/notes.py`
- `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`
- `tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py` (new)
- `tldw_Server_API/tests/Notes/test_notes_api_integration.py`
- `.superpowers/sdd/2026-08-08-notes-organization-sync-implementation-plan/task-8-report.md`

## TDD evidence

All pytest commands used `-p no:cacheprovider` because this managed worktree does
not own a writable pytest cache.

1. First direct keyword route:
   - RED: `1 failed, 2 warnings in 11.83s`; the route returned 201 and wrote
     ChaCha directly but stored no `notes-api` canonical envelope.
   - GREEN: `1 passed, 2 warnings in 10.36s`.
2. First expanded route matrix:
   - RED: `4 failed, 1 passed, 2 warnings in 12.72s`; note link returned the old
     unsupported response and folder/keyword/collection lifecycles lacked canonical
     envelopes.
   - GREEN: `5 passed, 2 warnings in 12.11s`.
3. Atomic append typing:
   - RED: `1 failed, 12 deselected, 2 warnings in 10.26s`; expected
     `sync_server_origin_batch_append_failed`, received the indistinguishable legacy
     `sync_server_origin_append_failed`; no product row was written.
   - GREEN: `1 passed, 12 deselected, 2 warnings in 10.25s` after the approved
     minimal batch append type.
4. Preflight/conflict typing:
   - RED: `1 failed, 15 deselected, 2 warnings in 10.82s`; a duplicate canonical
     name returned generic 503 rather than a safe preflight conflict; no second row
     was written.
   - GREEN: `1 passed, 15 deselected, 2 warnings in 9.72s` with safe 409
     `notes_organization_sync_preflight_failed` and no raw adapter text.
5. Task 9 merge boundary:
   - RED: `1 failed, 15 deselected, 2 warnings in 10.14s`; active Sync keyword
     merge returned 200 and directly mutated ChaCha.
   - GREEN: `1 passed, 15 deselected, 2 warnings in 11.91s`; it returns the stable
     unsupported response and preserves both rows.
6. Mid-group materialization/resume:
   - Strengthened RED first showed a test double returning `failed` without durable
     status. After using the real service exception path, the substantive RED was a
     409 idempotency conflict on replay: the applied keyword prefix changed the
     replanned group shape.
   - GREEN: `1 passed, 16 deselected, 2 warnings in 10.23s`; the stored status vector
     `applied/failed/pending` resumes to fully applied under the same key.

Final focused gates:

```text
pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py -k direct
17 passed, 2 warnings in 17.81s

pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_api_integration.py -k "keyword or collection or folder"
17 passed, 34 deselected, 3 warnings in 10.05s

ruff check <new coordinator, new direct test, minimal batch support seam>
All checks passed!

ruff check --select E9,F63,F7,F82 <legacy modified endpoint and integration test>
All checks passed!

bandit -q <three modified production files>
exit 0, no findings
```

## Warning disposition

Pytest warnings are the established repository/runtime deprecations seen in prior
Tasks 4–7: Passlib imports Python's deprecated `crypt` module and FastAPI warns
about deprecated `on_event` lifecycle registration (the integration selector
counts an additional occurrence). Test startup/teardown also logs the established
single-user API-key format and isolated database-path warnings. None is introduced
by Task 8 or changes an assertion/exit status.

Bandit emits one informational unmatched-`nosec` warning for B608 at the new
coordinator's resource lookup. The interpolated table identifier comes only from
the closed `_RESOURCE_TABLES` domain map and all values remain parameterized;
Bandit reports no finding.

The broad legacy files have existing full-Ruff debt (import ordering and unrelated
unused imports/catches). New files and the support seam pass strict Ruff; the two
legacy modified files pass the runtime-critical selector without mechanically
rewriting unrelated code.

## Decisions, deviations, and self-review

- The only Task 4 edit is the pre-approved typed atomic-append wrapper. No other
  Task 4 behavior changed.
- Planning remains read-only: tests assert folder plans create no projection rows
  before capture, and all loaders only read after successful materialization.
- Active Sync never falls back to direct writes. Partial, initializing, failed,
  preflight, append, and materialization cases all have focused zero-fallback
  assertions.
- Canonical tests verify envelope domain/object/payload, UUIDv4 identity, lineage,
  ordered step/count/hash consistency, apply state, mutation grouping, reloaded
  rows/links, replay, drift, and absence of plaintext idempotency keys.
- Materialization may leave an applied durable prefix, as Task 4 specifies, but no
  unlogged/direct mutation occurs; exact replay resumes the group.
- Collection inline keywords form one complete ordered Task 8 group. Note compound
  create/update/patch/import/bulk work and keyword merge stay owned by Task 9 and
  fail closed under active Sync.
- No new ADR was required: this directly implements ADR-032/ADR-033 and the
  approved Task 8 brief without changing the established authority boundary.

No remaining correctness, security, privacy, compatibility, or Task 9 boundary
concern was found in the scoped self-review.
