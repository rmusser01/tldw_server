# Task 7 Report — Bootstrap organization state and isolate device pulls

## Status

Implemented and verified against the Task 7 brief. ADR-032 and ADR-033 remain the governing decisions; no new ADR was required.

## Implementation

- Enrolls the six Notes organization domains atomically and stores CAS-protected `notes_organization_v1` lifecycle metadata.
- Keeps the opaque bootstrap ID out of profile responses while exposing safe state/count/error summaries.
- Adds transaction-local readiness gates for single and grouped appends; only the exact trusted initializing bootstrap ID bypasses the grouped gate.
- Adds a user-bound `NotesOrganizationBootstrapper` in the production factory.
- Captures resources parent-first, then active/dormant relationships, then deleted-resource tombstones in bounded deterministic groups.
- Reuses deterministic group/envelope identities on unchanged resume and derives a new repair plan when the source snapshot changes.
- Verifies every bootstrap step against live ChaCha source state and atomically records it applied without replaying product mutations.
- Requires exact bootstrap ID/state, matching counts, a fresh source verifier, and zero accepted/unapplied organization envelopes before the final ready CAS.
- Derives implicit pull domains from the authenticated device's stored requested capabilities and rejects explicit unsupported domains.

## RED/GREEN evidence

- Profile/enrollment RED: `3 failed, 9 deselected` because the service lacked the injected bootstrapper seam. GREEN: `3 passed, 9 deselected, 2 warnings in 9.61s` (fresh final run).
- Device selection RED: `1 failed, 92 deselected`; a legacy device implicitly received `notes.keyword`. GREEN: `1 passed, 92 deselected, 2 warnings in 9.79s`.
- Atomic gate RED with the production gate removed: `2 failed, 2 warnings`; both single and grouped appends admitted non-ready writes. First reapply exposed the gate on the idempotency lookup rather than the single append transaction (`1 failed, 1 passed`). Corrected GREEN: final strengthened file `3 passed, 2 warnings in 9.79s`.
- Bootstrap RED: collection failed because `notes_organization_bootstrap` did not exist. Initial implementation exposed the storage/model `entity_id`/`object_id` boundary, then passed. Final GREEN: `4 passed, 2 warnings in 12.12s`.
- Exact dormant-link attestation: `2 passed, 35 deselected, 2 warnings in 17.58s`.
- Existing Task 4 coordinator regression: first run `19 passed, 1 failed` because the new optional keyword was forwarded as `None` to a legacy one-argument seam. Preserving the old call shape fixed it: `20 passed, 2 warnings in 9.62s`.
- Scoped Ruff: `All checks passed!` using `--no-cache` because the external worktree denied `.ruff_cache` creation.
- Scoped Bandit: exit 0. It reports an informational pre-existing unmatched `nosec` warning at `Sync_DB.py:3148`; no security findings remain.
- `git diff --check`: exit 0, no output.

## Self-review against the brief

- Complete group enrollment, safe profile shape, CAS transitions, and core-domain availability: covered.
- Initializing/failed fail-closed behavior and in-transaction race gates: covered.
- Final count/source/apply verification and stale-worker rejection: covered.
- User-specific factory binding, deterministic ordering, cycle detection, bounded groups, unchanged resume, changed-snapshot repair, and stable identities: covered.
- No product replay/transient undelete and exact dormant relationship attestation: covered.
- Pre-bootstrap accepted-head drain fails safe when projection cannot be repaired: covered in coordinator behavior.
- Legacy/upgraded device implicit isolation and explicit capability rejection: covered.
- Existing cursor, ownership, encryption, pagination, and non-organization coordinator behavior remain unchanged by focused regression evidence.

## Warnings and concerns

Focused pytest runs emit repository-baseline dependency deprecations and test-environment `system_log_buffer` permission messages. They are unrelated to Task 7 and do not change test outcomes. No open Task 7 contract blocker remains.

## Fix Round 1

Reviewer findings were verified against the implementation before changes. All five findings were confirmed and fixed with focused RED/GREEN coverage.

### Changes

- Generic dataset enrollment now rejects every Notes organization domain and the reserved server-owned metadata keys `notes_organization_v1`, `default_personal`, and `client_family`. The HTTP route returns only `sync_reserved_dataset_enrollment` and a fixed safe message; forged metadata is never echoed.
- Single-envelope and atomic-group appends lock the dataset row used by readiness/final CAS before the readiness check and insert. PostgreSQL emits `SELECT ... FOR UPDATE`; SQLite retains its transactional equivalent.
- Resume compares the fresh source snapshot with bootstrap-captured relationship heads and emits an exact source-absence-attested tombstone only for captured links that disappeared. Pre-history links are never inferred as removals.
- Final ready verification now compares canonical applied resource/relationship heads, operations, and payloads with a final fresh source snapshot and rejects extra active organization heads.
- Retryable post-append verification failures remain `initializing` under the same bootstrap ID and stable safe error code. Repair explicitly reconciles the already preflight-attested historical step and uses a new deterministic plan to supersede it when source changed, without product replay or duplicate envelope IDs.
- The HTTP profile schema now has a typed, extra-forbidden `notes_organization` summary containing only state, non-negative captured/expected counts, and a safe error code. Bootstrap IDs and raw metadata remain excluded.
- API domain typing now includes the six Notes organization domains so profile capabilities/status can represent them; public enrollment remains blocked at the service boundary.

### Focused RED/GREEN evidence

- Forged enrollment service RED: `1 failed, 93 deselected, 2 warnings`; generic enrollment accepted organization/reserved capabilities. GREEN: `1 passed, 93 deselected, 2 warnings`.
- Forged enrollment endpoint RED: `1 failed, 26 deselected, 2 warnings`; HTTP returned 200 and echoed forged metadata. GREEN: `1 passed, 26 deselected, 2 warnings`.
- Transaction lock-order RED: `1 failed, 3 deselected, 2 warnings`; append called readiness and insert without the row-lock seam. GREEN: `1 passed, 4 deselected, 2 warnings`, proving `lock -> readiness -> insert` for single and grouped paths.
- PostgreSQL SQL-intent contract: `1 passed, 20 deselected, 2 warnings`, asserting the exact dataset-row `FOR UPDATE` query. The complete server-free contract file is `21 passed, 2 warnings`.
- Optional real concurrent PostgreSQL regression: `1 skipped, 4 deselected, 7 warnings`; PostgreSQL was not reachable/configured, and Docker fallback was intentionally disabled for the availability probe.
- Changed-snapshot relationship removal RED: `1 failed, 4 deselected, 2 warnings`; the dataset became ready with a stale relationship upsert head. GREEN: `1 passed, 4 deselected, 2 warnings` with exact tombstone/no-stale-head assertions.
- Retryable verification RED: `1 failed, 5 deselected, 2 warnings`; retryable failure transitioned to failed. Initial GREEN was strengthened with source drift; strengthened RED was `1 failed, 5 deselected, 2 warnings` because the pending old step stranded repair. Final GREEN: `1 passed, 5 deselected, 2 warnings`, same bootstrap/group IDs, fresh superseding plan, no duplicates, and every accepted organization envelope applied.
- Typed HTTP profile RED: `1 failed, 27 deselected, 2 warnings`; API typing rejected the organization domains and dropped the safe summary. GREEN: `1 passed, 27 deselected, 2 warnings` across initializing, ready, and failed states with no bootstrap ID/raw metadata.

### Final focused gates

- Task 7 profile core selector: `3 passed, 9 deselected, 2 warnings`.
- Bootstrap file: `6 passed, 2 warnings`.
- Non-live transaction gate file: `4 passed, 1 deselected, 2 warnings`.
- PostgreSQL server-free contract file: `21 passed, 2 warnings`.
- Forged enrollment plus device isolation: `2 passed, 92 deselected, 2 warnings`.
- Forged endpoint plus typed profile lifecycle: `2 passed, 26 deselected, 2 warnings`.
- Task 4 coordinator compatibility: `20 passed, 2 warnings`.
- Task 5 exact bootstrap attestation: `2 passed, 35 deselected, 2 warnings`.
- Scoped Ruff: `All checks passed!`.
- Scoped Bandit: exit 0 with no findings.
- `git diff --check`: exit 0, no output.

### Warning disposition and self-review

- Pytest's two recurring warnings are repository/runtime deprecations outside the Task 7 paths; repeated `system_log_buffer append failed: PermissionError` messages are test-environment logging noise and never changed an assertion or exit status.
- The optional PostgreSQL probe emitted seven baseline/runtime warnings before the fixture skipped because no server was configured.
- Bandit emitted informational unmatched-`nosec` warnings for pre-existing B608 annotations at `endpoints/sync.py:1995, 2025, 2104, 2158` (twice), `2163`, `2194` (twice), and `Sync_DB.py:3164`. Bandit reported no security finding; the warning locations are existing parameterized dynamic-SQL annotations, not new dependency code.
- Self-review found and fixed one additional repair edge: a retryable pending capture whose source changed before retry is now historically reconciled and deterministically superseded rather than left as an accepted-unapplied readiness blocker.
- No Task 8/9 behavior, Task 4 general planning semantics, Task 5 attestation requirements, or Task 6 product materializers were changed. No open blocker remains; live PostgreSQL concurrency remains environment-skipped but has deterministic server-free SQL/ordering coverage.

## Fix Round 2

The three remaining bootstrap-repair findings were verified against the implementation and confirmed. The repair now reasons over complete accepted bootstrap mutation groups rather than current heads alone, preserves removal-plan identity across a pre-ready crash, and never treats a stale preflight attestation as sufficient to mark history applied.

### Changes

- Bootstrap history is paged in bounded chunks and reconstructed as complete mutation groups in server-cursor/group-step order. Exact groups are re-recorded in full step order, including an applied tombstone that shadows an earlier pending upsert, so repair cannot strand accepted history or regress object state.
- Source-drifted pending groups remain pending until a later current correction head is durably applied and exactly matches the final fresh source snapshot. Only then does a transactional store seam reconcile the stale step with the stable audit code `sync_bootstrap_superseded`; reconciliation does not rewrite current object state or mutation fingerprints.
- Applied captured-link removal tombstones recover their original capture payload from the current bootstrap history. A restart under the same source snapshot therefore reconstructs the same removal plan, group shape, and client-envelope identities until the ready CAS succeeds.
- Deleted-resource lineage is omitted only when an applied historical upsert payload exactly matches normalized fresh source state and the applied current tombstone is canonical. If name or parent state changed, repair appends an explicit `restore_intent` correction upsert followed by its tombstone, verifies both without replay, and then reconciles stale history.
- Ordinary unchanged/additive resumes retain the complete deterministic snapshot plan. Resource omission is limited to relationship-removal repair, whose history-derived tombstone makes the adjusted plan stable across interruption.

### Focused RED/GREEN evidence

- Shadowed complete-group repair RED: `1 failed, 6 deselected, 4 warnings`; a deleted-resource upsert remained pending behind its group tombstone. After tracing the adapter result, the first correction attempt correctly failed because restoring over a tombstone lacked explicit restore intent. Final GREEN: `1 passed, 6 deselected, 3 warnings in 10.37s`.
- Removal crash stability RED: `1 failed, 7 deselected, 4 warnings`; restart reached ready without replaying the removal group (`repair_boundaries == [1]`). GREEN: `1 passed, 7 deselected, 3 warnings in 9.66s`, with identical removal-group identities before and after restart.
- Stale-step verification RED: `1 failed, 8 deselected, 4 warnings`; the verified fresh correction was current but the stale accepted step remained pending. GREEN: `1 passed, 8 deselected, 3 warnings in 10.76s`; the spy proves reconciliation occurs only after the correction is applied and exact, and the stale row records `sync_bootstrap_superseded`.
- Changed deleted-resource contrast RED: `1 failed, 9 deselected, 4 warnings`; repair emitted no valid correction upsert. GREEN: `1 passed, 9 deselected, 3 warnings in 10.45s`; the changed normalized payload produces an explicit restore-intent correction lineage and a final applied tombstone head.
- The first combined bootstrap run found an unchanged-resume group-shape regression: `9 passed, 1 failed, 4 warnings`. Restricting lineage omission to the stable removal-repair case fixed it. Fresh combined GREEN: `10 passed, 3 warnings in 14.64s`.

### Final focused gates

- Bootstrap file: `10 passed, 3 warnings in 14.64s`.
- Transaction gate file: `4 passed, 1 skipped, 3 warnings in 40.41s`; the optional live PostgreSQL test skipped because no PostgreSQL server was configured.
- PostgreSQL server-free contract file: `21 passed, 3 warnings in 17.41s`.
- Profile selector: `3 passed, 9 deselected, 3 warnings in 9.94s`.
- Task 4 coordinator compatibility: `20 passed, 3 warnings in 10.35s`.
- Scoped Ruff: `All checks passed!`.
- Scoped Bandit: exit 0 with no findings. It reports one informational pre-existing unmatched `nosec` warning for B608 at `Sync_DB.py:3164`.
- `git diff --check`: exit 0, no output.

### Warning disposition and self-review

- The third recurring pytest warning is `PytestCacheWarning`: the managed external worktree cannot write `.pytest_cache`. The other warnings are repository/runtime deprecations; `system_log_buffer append failed: PermissionError` is test-environment logging noise. None changed an assertion or exit status.
- The live PostgreSQL concurrency test remains honestly environment-skipped. The deterministic server-free lock/ordering contract continues to pass.
- Self-review preserved Task 4 planning, Task 5 attestation, Task 6 materializers, prior enrollment/locking/profile fixes, and Task 8/9 scope. No new ADR is required because Round 2 implements the already-governed durable bootstrap repair contract rather than changing the sync ownership or conflict policy.

## Fix Round 3

The final original-implementer review findings were verified against the Round 2 implementation and confirmed. Repair now avoids every transient object-state regression, resumes from the exact durable manifest for an existing deterministic group, and carries trusted relationship-capture evidence across failed bootstrap attempt IDs.

### Changes

- Exact group repair skips already-applied steps. When an older pending step is shadowed by a later applied current head for the same object, it uses the transactional audit-only reconciliation seam and never calls the verifier seam that rewrites `sync_object_state`.
- The server-origin batch coordinator can load a validated stored mutation group as its authoritative step manifest using the same source/idempotency-derived group ID. Bootstrap replays contiguous stored groups exactly, including canonical routing and lineage, before batching only semantic source steps not already represented by those manifests.
- Stored manifests remain guarded by the existing complete-group shape and materialization-plan-hash validation. Mutable post-apply omission/restore state can no longer change an already durable group shape or its envelope IDs for the same bootstrap ID and source snapshot hash.
- Relationship-removal discovery accepts capture upserts from any non-empty prior bootstrap attempt ID when they belong to accepted, server-origin `notes-organization-bootstrap` history and carry `bootstrap_capture: true`. The emitted verified removal tombstone always belongs to the current attempt. Links without bootstrap capture history remain excluded.

### Focused RED/GREEN evidence

- Transient object-state RED: `1 failed, 10 deselected, 4 warnings in 9.92s`; the shadowed older upsert was recorded through the ordinary verifier rather than with the superseded audit marker, and the SQL hook observed its cursor in an `INSERT INTO sync_object_state` write. GREEN: `1 passed, 10 deselected, 3 warnings in 11.71s`; no traced object-state write references the older cursor and the envelope records `sync_bootstrap_superseded`. The final assertion-order rerun is `1 passed, 12 deselected, 3 warnings in 9.51s`.
- Durable manifest RED: `1 failed, 11 deselected, 4 warnings`; after an applied removal repair also added a resource, restart rederived a smaller group and transitioned failed on idempotency shape drift. GREEN: `1 passed, 11 deselected, 3 warnings in 11.55s`, with the identical group/envelope IDs, no duplicate history, and ready state.
- Cross-attempt relationship RED: `1 failed, 12 deselected, 4 warnings`; attempt ID2 ignored ID1's trusted capture and failed final readiness with the stale link head. GREEN: `1 passed, 12 deselected, 3 warnings in 10.00s`; ID2 emits and verifies its tombstone from ID1 capture history.

### Final focused gates

- Bootstrap file: `13 passed, 3 warnings in 15.62s`.
- Transaction gate file: `4 passed, 1 skipped, 3 warnings in 41.24s`; live PostgreSQL remains unconfigured.
- PostgreSQL server-free contract file: `21 passed, 3 warnings in 17.57s`.
- Profile selector: `3 passed, 9 deselected, 3 warnings in 10.25s`.
- Task 4 coordinator compatibility: `20 passed, 3 warnings in 9.54s`.
- Scoped Ruff: `All checks passed!`.
- Scoped Bandit: exit 0 with no findings or output.
- `git diff --check`: exit 0, no output.

### Warning disposition and self-review

- Pytest warnings remain the managed-worktree `PytestCacheWarning` plus repository/runtime deprecations. `system_log_buffer append failed: PermissionError` remains test-environment logging noise and did not change assertions or exit statuses.
- The durable manifest loader does not weaken validation: it resolves the same deterministic group identity and reuses the existing stored shape/plan-hash verifier before returning steps. Semantic subtraction only prevents already-manifested source mutations from being appended again; replay retains the exact stored routing/base lineage.
- Cross-attempt removal cannot infer pre-history deletion because the candidate must still be an accepted server-origin bootstrap group step with explicit capture metadata. Final exact-head/source verification remains unchanged.
- No Task 8/9 behavior, enrollment/profile surface, PostgreSQL gate, Task 4 planning, Task 5 attestation, or Task 6 materializer was changed. No new ADR is required because Round 3 hardens the already-governed durable bootstrap/resume policy.

## Files

- `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- `tldw_Server_API/app/core/Sync/v2/adapters.py`
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_organization.py`
- `tldw_Server_API/app/core/Sync/v2/factory.py`
- `tldw_Server_API/app/core/Sync/v2/notes_organization_bootstrap.py`
- `tldw_Server_API/app/core/Sync/v2/profile.py`
- `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`
- `tldw_Server_API/app/core/Sync/v2/service.py`
- `tldw_Server_API/app/core/Sync/v2/store.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_gate.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
