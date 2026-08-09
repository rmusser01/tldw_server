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

## Fix Round 1 — replay, restore, and route-contract hardening

### Reviewer findings verified

All four findings were reproduced against commit `050fa346` before their focused
fixes:

- Relationship replans emitted no `restore_intent` after the current Sync head had
  become a tombstone.
- Keyword and collection update/delete routes evaluated mutable projection rows and
  versions before looking for an existing idempotency-group manifest.
- Folder planning omitted already-applied ancestors, and collection membership
  updates could recompute a smaller relationship delta after a materialized prefix.
- Active-Sync missing/deleted keyword planning raised `InputError`, changing the
  existing route contract to 400 instead of 404; version conflicts lacked a stable
  typed error code.

### Round 1 implementation

- Relationship planners now inspect the owner-scoped current Sync head and attach
  `restore_intent=true` only when an upsert restores a tombstoned relationship.
- Added a privacy-safe SHA-256 fingerprint of canonical request identity to each
  stored update/delete/folder plan. The fingerprint covers route operation, local
  IDs, expected version, supplied-field identity, and normalized requested values;
  no plaintext idempotency key or requested value is stored in routing metadata.
- Existing Task 4 manifests are loaded and fingerprint-validated before mutable
  product-row/version checks. Exact keyword rename/delete and collection
  update/delete requests reuse the stored plan; changed requests return the stable
  batch idempotency conflict. Deleted product rows are bypassed only after this
  immutable manifest match.
- Folder creation now plans the complete canonical root-to-leaf shape, including
  existing segments. An active existing-path request establishes a durable group,
  an exact replay keeps the original create response, and a different path under
  the same key conflicts.
- Added stable `NotesOrganizationResourceNotFoundError` and
  `NotesOrganizationVersionConflictError` mappings to preserve 404/409 behavior
  without exposing database details.
- Task 4 batch code, inactive-Sync direct behavior, Task 9 boundaries, and public
  note compound/merge behavior were not changed in this round.

### Round 1 TDD evidence

1. Relationship restore:
   - RED: `1 failed, 17 deselected, 2 warnings in 9.84s`; all relinks had empty
     routing metadata.
   - GREEN: `1 passed, 17 deselected, 2 warnings in 9.94s`; note-keyword,
     conversation-keyword, collection-keyword, and folder-link relinks all carry
     restore intent and materialize present.
2. Exact update/delete replay:
   - RED: `1 failed, 18 deselected, 2 warnings in 10.32s`; an exact keyword rename
     retry failed the now-mutable optimistic version check.
   - GREEN: `1 passed, 18 deselected, 2 warnings in 10.32s`; exact and drift cases
     cover keyword rename/delete and collection name/parent update/delete, including
     exact delete replay after projection deletion.
3. Folder applied-prefix replay:
   - RED: `1 failed, 19 deselected, 2 warnings in 12.69s`; retry after
     `Root=applied, Child=failed` recomputed only the child and returned the stable
     409 plan conflict.
   - GREEN: `1 passed, 19 deselected, 2 warnings in 11.51s`; the original two-step
     IDs and shape resume, and an existing-path group rejects a different path under
     the same key.
4. Collection relationship-prefix replay:
   - `1 passed, 20 deselected, 2 warnings in 11.17s`; a collection update that fails
     after its first relationship applies resumes the stored manifest with identical
     domain/operation/object-ID shape. This was already green after the earlier
     update-manifest seam, so no additional production change was required.
5. Missing/deleted/stale keyword status:
   - RED: `1 failed, 21 deselected, 2 warnings in 10.76s`; never-existing rename and
     delete both returned generic 400.
   - GREEN: `1 passed, 21 deselected, 2 warnings in 10.65s`; missing/deleted are
     stable 404, stale version is stable 409, and failed attempts append no product
     mutation.

Final Round 1 gates:

```text
pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py -k direct
22 passed, 2 warnings in 20.54s

pytest -q -p no:cacheprovider tldw_Server_API/tests/Notes/test_notes_api_integration.py -k "keyword or collection or folder"
17 passed, 34 deselected, 3 warnings in 9.68s

ruff check <coordinator, direct test>
All checks passed!

ruff check <legacy notes endpoint>
Three unchanged baseline findings: I001, BLE001, F841. The same three findings
are present in `git show HEAD:.../notes.py`; Round 1 adds none.

bandit -q <two modified production files>
exit 0; one informational unmatched-no-failed-test warning for the existing B608
closed-table-map `nosec` resource query.

git diff --check
exit 0
```

The pytest warning identities remain the established Passlib `crypt` and FastAPI
`on_event` deprecations described above. No warning affects assertions or exit
status.

### Round 1 self-review

- Manifest lookup remains scoped to the authenticated user's active default
  personal dataset; product-row lookup remains scoped to the per-user ChaCha DB.
- Planners remain write-free. Relationship restore adds an owner-scoped Sync-head
  read; all product and Sync mutations still occur only through capture/materialize.
- Fingerprints are fixed-length digests. Neither the idempotency key nor raw
  requested values are persisted in the added routing field.
- Exact replay bypasses mutable projection preconditions only after the Task 4
  manifest validates and every stored step matches the immutable request
  fingerprint. Absent manifest, missing/deleted resources retain 404 and stale
  versions retain 409.
- Folder and collection failure tests prove stable mutation IDs, step shape,
  durable prefix state, and resumability. Active paths have no direct-write fallback.
- No new correctness, security, privacy, compatibility, or Task 9 boundary concern
  remains after the Round 1 review.
