# Notes Link Sync and Graph Lifecycle Implementation Plan

> **For implementation:** REQUIRED SUB-SKILL: use
> `superpowers:test-driven-development` for every behavior change and
> `superpowers:verification-before-completion` before any completion claim.

**Goal:** Make explicit manual note-to-note links durable Sync v2 objects, preserve
them invisibly across note trash/restore, and replace request-time wikilink parsing
with bounded deterministic local projections while preserving the existing graph
API's tag/source behavior.

**Architecture:** Extend `note_edges` in schema v58 and add a strict `notes.link`
domain with a separate resumable `notes_link_v1` enrollment state. One portable
ChaCha store owns link lifecycle and derived graph state; the Sync adapter and
materializer use that store rather than embedding SQL. Active-Sync REST writes go
through a thin link coordinator, while inactive requests retain the legacy route
shape. Graph reads use live endpoint joins, revision-bound caches/cursors, and
persisted derived wikilinks; read paths never perform repair writes.

**Tech stack:** Python 3.11+, FastAPI/Pydantic, SQLite and PostgreSQL, Sync v2
adapters/materializers/restore, pytest/pytest-asyncio, Ruff, Bandit.

## Normative inputs

- `Docs/superpowers/specs/2026-08-10-notes-link-sync-and-graph-lifecycle-design.md`
- `Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md`
- `Docs/ADR/031-notes-capability-sync-domains.md`
- `Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`
- `Docs/ADR/035-canonical-folder-link-suppression-preserves-source-provenance.md`
- `Docs/ADR/036-web-clipper-external-identity-mapping.md`

## ADR check

```text
ADR required: yes
ADR path: Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md
Reason: this changes durable Notes schema, Sync conflict/enrollment contracts,
        PostgreSQL ownership/RLS, public lifecycle APIs, and the boundary between
        canonical and derived graph state.
```

## Global execution rules

- Execute the tasks in order. `ChaChaNotes_DB.py`, Sync registration, and graph
  service are shared seams; do not implement them in parallel.
- Before Task 2 production edits, obtain the user's explicit operational approval
  for the exact v58 PostgreSQL table locks and temporary FORCE-RLS
  suspension/restoration. Record the approved lock set in TASK-13004 notes.
- For each task: add the named RED assertion, run only the focused selector and
  observe the intended failure, make the smallest production change, then rerun the
  same selector GREEN.
- Never add `notes.link` to `NOTES_ORGANIZATION_DOMAINS`; it has independent
  `notes_link_v1` readiness and must not block the six organization domains.
- Product and projection state are authorized only through the one active
  default-personal Chatbook dataset. Do not alias another same-owner dataset onto
  owner-scoped product rows.
- Keep protected label/properties/provenance out of routing metadata, errors, logs,
  and conflict summaries.
- Preserve existing tag/source graph nodes and membership edges. Apply live-two-note
  visibility only to manual/wikilink/backlink edges.
- No read endpoint may parse/repair/queue projections or mutate cache state other
  than ordinary in-memory cache insertion.
- Each task ends with `git diff --check`. Do not describe broad legacy lint findings
  as green; report exact touched-file scopes separately.

---

## Task 1: Define the strict `notes.link` public and internal contract

**Files**

- Create: `tldw_Server_API/app/core/Sync/v2/notes_link.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_contract.py`

- [x] Add RED capability tests asserting `notes.link`, adapter version 1,
  `server_trusted_v1`, and exactly `upsert`/`tombstone`, while confirming the domain
  is absent from `NOTES_ORGANIZATION_DOMAINS`.
- [x] Add RED parser/vector tests for lowercase UUIDv4 edge IDs, canonical
  undirected endpoint order, immutable identity fields, 0–1,000,000 finite weight,
  256-character label, 64-key/depth-4/16-KiB properties, 256-character tombstone
  reason, protected provenance, and `extra="forbid"` behavior.
- [x] Implement frozen strict payload models and one canonical parser. Require
  `created_at`/`last_modified` to match normalized `created_at_client` and
  `created_by` to match the authenticated device, except trusted source-verified
  bootstrap.
- [x] Add public capability schemas without exposing owner, local IDs, raw routing,
  or protected properties in summaries.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_contract.py \
  -k notes_link
git diff --check
```

Expected GREEN: all selected contract/vector tests pass.

---

## Task 2: Add transactional schema v58 and PostgreSQL ownership policy

**Approval gate before production edit**

**Approved 2026-08-10:** the user explicitly approved the operation below,
including the temporary write blocking caused by the transaction-held locks and
the fail-closed rollback behavior.

Ask the user to approve this resolved v57→v58 PostgreSQL operation: lock the
`db_schema_version` authority row with `FOR UPDATE`, then acquire transaction-held
`ACCESS EXCLUSIVE` locks in fixed order on `notes`, `note_edges`,
`chacha_keywords`, `note_keywords`, and `conversations`. Catalog-verify schema ownership and the
existing `notes` RLS state; temporarily run `ALTER TABLE notes NO FORCE ROW LEVEL
SECURITY` only if FORCE was verified active, restore that exact state before the
version bump, and reinstall the full policy set. State that note, manual-link,
keyword, and note-keyword writes may block temporarily. All validation, permitted
column/metadata backfill, DDL, policy state, and version changes roll back together;
the migration never deletes or rekeys a product row.

**Files**

- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_notes_link_migration_v58.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_postgres_contract.py`

- [x] Add RED SQLite v57→v58 and fresh-schema tests covering valid legacy edge
  transformation, label extraction, metadata preservation, soft-delete columns,
  exact version/provenance backfill, stable logical uniqueness, and idempotent
  reopen.
- [x] Add RED rollback tests for every invalid transformed value: identity,
  self-link, missing/cross-owner endpoint, duplicate logical edge, weight, label,
  properties root/key/depth/size, malformed metadata, and constraint/index failure.
- [x] Add server-free PostgreSQL RED contracts for schema-row/table lock ordering,
  catalog/schema-owner/RLS verification, exact temporary unforce/restore set,
  owner/live indexes, endpoint-aware `USING`/`WITH CHECK`, full policy reinstall,
  and initializer serialization.
- [x] Implement schema v58. SQLite rebuilds `note_edges` transactionally.
  PostgreSQL computes and validates every post-transform row before DDL, adds
  nullable columns then constraints, and never guesses/rekeys/deletes. Add local
  derived-link, dirty-generation, projection-state, and graph-revision tables with
  backend-specific owner/database-local keys from the design.
- [x] Install transaction-local graph-revision/dirty-generation triggers for every
  writer path that can bypass the normal store: `notes`, `note_edges`, `keywords`,
  `note_keywords`, and `conversations`. Conversation source/external-reference
  lifecycle changes advance the same owner revision. The
  normal store may update projections eagerly, but the trigger contract guarantees
  cache invalidation and coalesced repair for direct writes.
- [x] Add an optional live-PostgreSQL two-owner migration/RLS test, skipped only when
  no DSN is configured.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_link_migration_v58.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_postgres_contract.py
git diff --check
```

Expected GREEN: fresh/upgraded/rollback/server-free contracts pass; optional live PG
is either green or reported as one explicit environment skip.

---

## Task 3: Implement the portable explicit-link lifecycle store

**Files**

- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_link_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_note_link_store.py`
- Modify: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py`

- [x] Add RED store tests for create/get/list/update/tombstone/restore, canonical
  undirected endpoints, live-only public create, soft-deleted endpoint acceptance
  for trusted historical apply, immutable identity, optimistic version checks,
  stable uniqueness across tombstones, and exact replay with no second write.
- [x] Add RED same-owner SQLite and PostgreSQL query-contract cases, including both
  endpoints, deleted endpoints, cross-owner IDs, and no protected-field leakage.
- [x] Implement `NotesLinkStore` dataclasses plus transaction-aware product methods.
  Make the old `create_manual_note_edge`/`delete_manual_note_edge` methods delegate
  to the new store in inactive compatibility mode rather than retain a second
  authority.
- [x] Advance graph revision in the same product transaction for a real link
  mutation. Exact postcondition replay must not change version, timestamps, or
  revision.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_link_store.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py \
  -k "manual or link_lifecycle or exact_replay or owner"
git diff --check
```

---

## Task 4: Add the domain adapter, materializer, and conflict invariants

**Files**

- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_link.py`
- Create: `tldw_Server_API/app/core/Sync/v2/materializers/notes_link.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_materializer.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

- [x] Add RED adapter tests for first creation, exact replay, stale/divergent mutable
  update, stale delete, exact restore intent, restore-required duplicate object ID,
  immutable retarget/direction/type, missing endpoint deferral, and soft-deleted
  endpoint identity acceptance.
- [x] Add RED real-store materializer tests for SQLite and server-free PostgreSQL SQL
  intent, including crash-after-product-commit replay and safe conflict evidence.
- [x] Implement the strict adapter using canonical prior-head ancestry and bounded
  dependency reads. Implement the user-bound materializer through `NotesLinkStore`;
  do not embed product SQL in the adapter.
- [x] Register and fail closed at factory composition when `notes.link` is advertised
  without its strict adapter/materializer.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py -k notes_link
git diff --check
```

---

## Task 5: Add resumable `notes_link_v1` enrollment and bootstrap

**Files**

- Create: `tldw_Server_API/app/core/Sync/v2/notes_link_bootstrap.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_bootstrap.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`

- [x] Add RED tests proving an existing `notes_organization_v1=ready` dataset can
  atomically add `notes.link` and enter `notes_link_v1=initializing` without changing
  or blocking the six organization domains.
- [x] Add RED fresh/resume/failure/source-drift tests: stable bootstrap ID, immutable
  edge-ID paging across both live and tombstoned current rows, matching upsert versus
  tombstone envelopes, endpoints already enrolled, source verification, exact
  counts and fingerprints, no product reapply, and transition to ready only after
  verification.
- [x] Implement dataset-row-locked begin/transition methods and a separate injected
  `NotesLinkBootstrapper`. Profile bootstrap invokes it only when `notes.link` is
  requested; an old client that omits the domain remains valid.
- [x] Expose safe `notes_link` readiness/status in profile responses. Link writes
  fail closed while initializing/failed; other organization writes remain enabled.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  -k "notes_link or notes_organization"
git diff --check
```

---

## Task 6: Route public link lifecycle through canonical capture

**Files**

- Create: `tldw_Server_API/app/core/Sync/v2/notes_link_coordinator.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_link_capture.py`
- Modify: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py`
- Modify: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_rbac.py`

- [x] Add RED active/inactive tests for optional dataset resolution. Active omission
  selects only the active default-personal dataset; a supplied ID must equal it;
  inactive omission uses legacy product behavior; inactive supplied ID is 409;
  missing/foreign datasets remain indistinguishable.
- [x] Add RED route tests for create, list, detail, PATCH, tombstone, and explicit
  restore. Active update/delete/restore require `expected_version` (428); inactive
  legacy delete remains compatible. Legacy metadata label normalization is tested.
- [x] Add RED capture tests for preflight-before-product-write, exact stable-key
  replay, changed-request 409, not-ready zero-write, append/projection failure 503,
  duplicate logical identity, and no incomplete success.
- [x] Implement a small owner-bound coordinator that builds one canonical plan and
  reuses the existing server-origin append/materialize seam. Route handlers perform
  validation/error mapping only.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_capture.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_rbac.py \
  -k "link or dataset or restore"
git diff --check
```

---

## Task 7: Persist deterministic wikilink projections and bounded maintenance

**Files**

- Create: `tldw_Server_API/app/core/Notes/wikilinks.py`
- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_graph_projection_store.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/wikilink_parser.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/notes.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/projection_service.py`
- Modify: `tldw_Server_API/tests/Notes_Graph/unit/test_wikilink_parser.py`
- Create: `tldw_Server_API/tests/Notes_Graph/unit/test_graph_projection_store.py`
- Create: `tldw_Server_API/tests/Notes_Graph/integration/test_graph_projection_maintenance.py`

- [x] Move the pure canonical UUID parser to neutral Notes code and keep the old
  import as a compatibility re-export. Add RED vectors for duplicates,
  self-links, unresolved targets, 1,024-target first-occurrence truncation, and
  parser-version determinism.
- [x] Add RED transaction tests showing a normal note create/content update updates
  the note row, outgoing projection, dirty generation clear, and graph revision in
  one product transaction. Trash/restore changes visibility/revision without
  deleting projection rows.
- [x] Add RED direct-write/crash tests for trigger-coalesced generations, bounded
  claims, generation-safe deletion, interrupted resume, unresolved target becoming
  visible later, parser-version rebuild, and no read-path writes.
- [x] Implement the projection store and maintenance service through the existing
  task/startup facility. PostgreSQL work requires explicit authenticated owner scope
  and `FOR UPDATE SKIP LOCKED`; SQLite uses database-local transactions. Never run a
  global unscoped PostgreSQL worker.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Notes_Graph/unit/test_wikilink_parser.py \
  tldw_Server_API/tests/Notes_Graph/unit/test_graph_projection_store.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_graph_projection_maintenance.py
git diff --check
```

---

## Task 8: Make graph, backlink, and orphan reads live-only and revision-safe

**Files**

- Modify: `tldw_Server_API/app/core/Notes_Graph/graph_cache.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/graph_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`
- Modify: `tldw_Server_API/tests/Notes_Graph/unit/test_graph_cache.py`
- Create: `tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py`
- Modify: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py`

- [x] Add RED graph tests for hidden incident links on note trash, unchanged link
  head/version, reappearance on restore, derived backlink reversal, unresolved
  targets, manual-only availability during rebuild, and 503 for derived/orphan
  requests while projection is stale.
- [x] Add RED compatibility tests proving tag/source nodes and membership edges
  remain present for live notes.
- [x] Add RED cursor/cache tests: authorization before cache, canonical dataset ID +
  graph revision + parser version + normalized request in the key, 8-KiB encoded and
  4-KiB decoded limits, stale/mismatched failure, and deterministic radius-two
  resume.
- [x] Add RED `/notes/graph/orphans` and link-list keyset pagination tests with
  default 50/max 200, immutable-ID ordering, no full properties in summaries, and
  orphan semantics that ignore tags/sources.
- [x] Replace request-time wikilink parsing with projection reads and live endpoint
  joins. Ensure every graph-visible note/link/keyword/source mutation advances the
  relevant revision transactionally.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Notes_Graph/unit/test_graph_cache.py \
  tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py
git diff --check
```

---

## Task 9: Integrate restore preview, repair, and conflict resolution

**Files**

- Modify: `tldw_Server_API/app/core/Sync/v2/restore.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

- [x] Add RED allowlist/local-inventory tests proving applied live/tombstone
  `notes.link` envelopes appear in restore planning and preserve dataset/group
  metadata in `ordered_actions`.
- [x] Add RED ordering tests: live link after both selected `notes.note` identity
  providers; deleted endpoint identity allowed; link tombstone has no live
  dependency; complete groups stay adjacent; missing providers, contradictions, and
  cycles fail closed.
- [x] Add RED repair/conflict-resolution tests for concurrent edit, stale delete,
  restore/recreate, duplicate logical object ID, exact idempotent replay, and safe
  public conflict evidence.
- [x] Implement the minimal restore-domain allowlist/dependency branch and reuse the
  existing dataset guard, staged conflict planning, whole-group repair, and
  `superseded` terminal semantics. Do not add a second ordering engine.

Run RED/GREEN:

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "notes_link or restore"
git diff --check
```

---

## Task 10: Verify boundedness, regressions, security, docs, and task hygiene

**Files**

- Modify: `Docs/API/Sync_V2_M1.md`
- Modify: relevant Notes graph API/design documentation found during implementation
- Modify: `backlog/tasks/task-13004 - Synchronize-Notes-relationships-graph-and-restore-lifecycle.md`
- Modify only if an incident generalizes: the relevant repository
  `backlog/docs/lessons-*.md`

- [x] Add query-plan/query-count tests for dense/high-degree graphs, link keyset
  pagination, orphan queries, projection batch caps, bootstrap paging, and restore
  action caps. Assert bounded structure rather than fragile wall-clock thresholds.
- [x] Run the focused migration, product-store, Sync adapter/materializer,
  bootstrap/capture, projection, graph, restore/repair, Notes lifecycle, and API/RBAC
  suites serially. Record exact counts and any optional live-PostgreSQL skip.
- [x] Run Ruff on every touched Python file, Ruff format check on touched/new files,
  Bandit on touched production Python, `py_compile`, `git diff --check`, and
  `git diff --cached --check`. Separate inherited whole-file findings from task
  findings.
- [x] Perform a final correctness/security review against all six acceptance
  criteria, ADR-037, dataset/RLS isolation, protected-field privacy, cursor/cache
  authority, crash repair, and migration rollback.
- [x] Update public docs and TASK-13004 Implementation Notes with approach,
  decisions, files, RED/GREEN evidence, static scopes, skips, and review disposition.
  Check AC/DoD only after evidence exists, then set Done through Backlog CLI.

Suggested final gates (adjust only to include newly created files; record exact
commands actually run):

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_link_migration_v58.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_link_store.py \
  tldw_Server_API/tests/Notes_Graph \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_edges.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_rbac.py \
  tldw_Server_API/tests/Notes/test_notes_restore.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_link_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py

git diff --check
git diff --cached --check
git status --short
```

Expected final state: all focused and regression gates pass; the worktree contains
only TASK-13004 changes; the task is Done with AC/DoD and exact evidence recorded.
