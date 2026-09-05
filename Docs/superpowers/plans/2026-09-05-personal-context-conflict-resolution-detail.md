# TASK-13163 implementation seam update

This refines Task 2 of the approved 2026-09-03 server activation/conflict/purge
plan against the merged activation and publication implementation. It changes
no product behavior or wire action vocabulary.

ADR required: no new ADR
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: implement existing encrypted candidate retention, canonical resolution
receipts, narrow freezes, and journaled cross-database recovery.

## Current evidence

- The existing ongoing-contract suite passes 12 tests.
- Batched request fields and endpoint forwarding already exist, but the batch
  service discards the expected candidate IDs and idempotency key before generic
  resolution. Personal Context must have its own canonical dispatch there.
- Both push conflict constructors omit a protected remote candidate.
- Authority publication staging requires a genuine leased journal row. Never
  manufacture a publication row or bypass that validation to stage a candidate.
- Generic retention does not currently fence unresolved conflict references.
- Generic conflict metadata is plaintext; semantic keys and bodies cannot go
  there. Existing ingress receipts do not bind conflict/action/expected IDs.

## File ownership correction

In addition to the original Task 2 file list, include:

- `DB_Management/Personalization_DB.py` and `Personal_Context_Repository.py` for
  durable encrypted conflict state, exact command receipts, and transaction-local
  object/key-slot guards. Follow existing schema initialization/migration patterns
  and verify reopen/upgrade. Include new encrypted owners in purge/key-rotation
  inventories rather than leaving recoverable content outside their lifecycle.
- `DB_Management/Sync_DB.py` for candidate attachment and retention checks in
  both preview and guarded destructive revalidation, using SQLite/PostgreSQL parity.
- `Sync/v2/materializers/personal_context.py` only as needed to preserve structured
  private collision identity rather than flattening every conflict into one code.

## Execution order and tests

### Confirmed user-directed deconfliction

The user clarified: "there should be deconfliction" and "the user chooses",
then authorized continuation. Neither peer wins automatically. Competing
candidates stay unresolved until an explicit user choice: keep shared, keep
local values, reviewed merge, or keep both as deliberately distinct facts.
For same-key creations with different IDs, choosing local values or a merge
updates the established shared canonical identity; it does not silently delete
an unrelated shared record or leave two active facts for the same key. The
conflict receipt accounts for the superseded incoming candidate so retry cannot
resurrect it. The reviewed replacement must name the canonical target explicitly;
do not rewrite caller-supplied IDs invisibly. Keep both remains duplicate_rename
with a new ID and noncolliding key. Tests must cover each explicit choice and
prove no mutation occurs merely from detecting a collision.

### Task 1: Implement TASK-13163

1. Add failing real-repository tests in the new Personal Context conflict suite
   for deterministic encrypted authority candidate creation/replay, expected-ID
   rejection, and interrupted canonical resolution before Sync finalization.
   Reuse production factory/activation fixtures; keep version-one test activation
   explicit, never weaken production exchange proof or advertise v1.
2. Persist exact conflict identity, immutable candidates and freeze ownership
   under the canonical transaction. Ordinary conflicts freeze one object; a key
   collision freezes both IDs and only its contested semantic-key slot. Store
   semantic material encrypted. Reuse existing encrypted storage where its owner
   lifecycle fits; otherwise add narrowly owned tables and migration/reopen tests.
   Authenticate candidate staging against this durable canonical state or a real
   existing publication, never caller-supplied source metadata. Attach and protect
   the exact candidate before a terminal push conflict response; failure remains
   retryable and restart resumes the same journal identity/bytes.
3. Pass expected IDs and command idempotency through batch dispatch into a
   `PersonalContextConflictService`, keeping unrelated Sync domains unchanged.
   Check ownership, registered device, activation, generation, action, exact
   candidates and canonical current heads before mutation. Preserve batch savepoint
   isolation and reject stale items without resolving the generic conflict.
4. Route overwrite/reviewed merge and duplicate_rename through
   `PersonalContextService`. Atomically commit result, authoritative manifest,
   ordered publication batch and an exact conflict/action/candidates/command-digest
   receipt. Replay equality must reject a changed command, not apply twice. Keep
   skip canonical-content-neutral, with a replayable decision/release journal if
   needed for cross-store freeze release. No network/relay wait under a canonical
   write transaction; preserve the existing lock order and after-commit recovery.
   Duplicate requires a new ID and free semantic key. Only the authorized exact
   resolution may bypass/release its own freezes; ordinary direct CRUD and ingress
   must enforce them inside their write transaction.
5. Cover restart before/after candidate attachment and canonical receipt, stale
   reviews, changed idempotency payload, collisions and unrelated writes, partial
   batches, wrong owner/device/proof, purge/recreation and encrypted-owner lifecycle.
   Add retention tests proving both candidates survive dry-run and actual guarded
   compaction until resolved, and plaintext canaries for new storage/log surfaces.
6. Run targeted conflict, Personalization service/repository, Sync service,
   materializer, retention, relay, activation and ongoing-contract regressions.
   Require real PostgreSQL for touched Sync queries. Run affected-code Ruff/Bandit,
   formatter/diff checks, then independent spec and quality reviews. Document
   baseline failures separately; do not replace legitimate assertions to get green.

Do not enable ongoing_sync_version=1, implement TASK-13164/13165, alter the shared
profile contract, introduce a second resolution endpoint, or broaden runtime grants.
