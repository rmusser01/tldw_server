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
