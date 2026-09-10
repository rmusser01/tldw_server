# Portable Moodboard Smart-Match Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make moodboard smart-rule results portable and request-bounded through dormant canonical note-time support, versioned normalized dependencies, and disposable owner/dataset-scoped projection generations.

**Architecture:** Schema v62 adds only derived normalized values, dependency epochs, completed smart-match generations, and rebuild state; none is product or Sync authority. Pure rule normalization/evaluation is shared across backends, workers page candidate state under explicit row/time budgets, and request paths read only atomically published generations. `notes.note` materializers learn permanent `canonical_modified_at` parsing/projection now, but production writers do not stamp it until TASK-13007.5.

**Tech Stack:** Python 3.11 `unicodedata`, Pydantic v2, SQLite, PostgreSQL forced RLS, FastAPI, pytest, Ruff, Bandit, OpenAPI exporter, openapi-typescript.

**Design:** `Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md`

**Backlog task:** `TASK-13007.2`

**ADR required:** no
**ADR path:** `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
**Reason:** ADR-040 already defines portable note time, Unicode compatibility, derived projection ownership, bounded rebuilds, and readiness behavior.

---

## File map

**Create**

- `tldw_Server_API/app/core/Notes/moodboard_smart_projection.py` — pure v1 normalization, dependency fingerprinting, literal matching, candidate evaluation, and bounded rebuild orchestration.
- `tldw_Server_API/app/core/DB_Management/chacha/moodboard_smart_projection_store.py` — owner/dataset-scoped epoch, dirty queue, generation, cursor, match, count, and publication storage.
- `tldw_Server_API/app/services/notes_moodboard_projection_worker.py` — bounded scheduler/worker turns; no request-path full scan.
- `tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py` — portable rule/evaluator, budgets, invalidation, crash/resume, and generation tests.
- `tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_migration_v62.py` — SQLite 61→62/fresh/rollback/catalog/index proof.
- `tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_postgres.py` — required live PostgreSQL RLS, parity, and plan proof.
- `tldw_Server_API/tests/Services/test_notes_moodboard_projection_worker.py` — bounded scheduling, continuation, retry, and shutdown tests.

**Modify**

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` — schema v62 derived storage, catalog verification, normalized projection hooks, and smart-store composition.
- `tldw_Server_API/app/core/DB_Management/chacha/note_store.py` — transactional normalized note fields and exact canonical `last_modified` projection support.
- `tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py` — direct keyword, collection, and membership mutations update normalized values and dependency epochs transactionally.
- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` — direct conversation source mutations update normalized source values and dependency epochs transactionally.
- `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py` — Sync-origin portable collection identity translation and normalized keyword/collection dependencies use the same transactional hooks.
- `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py` — forced-RLS policies for derived tables.
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py` — accept only server-bound canonical note-time routing metadata and preserve exact retry semantics.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes.py` — deterministic `canonical_modified_at` selection and projection into existing `notes.last_modified`.
- `tldw_Server_API/app/core/Sync/v2/models.py` — private routing schema description only; no production stamping or new capability.
- `tldw_Server_API/app/api/v1/schemas/notes_moodboards.py` — additive completeness status, generation/revision, nullable total, and portable collection boundary models.
- `tldw_Server_API/app/api/v1/endpoints/notes.py` — translate local collection IDs and serve manual-only pending or completed indexed hybrid results.
- `tldw_Server_API/app/services/startup_optional_workers.py` — register a bounded internal projection worker without changing public capabilities.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py` — canonical time, bootstrap, old-envelope fallback, and exact retry tests.
- `tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py` — client-forged note-time rejection.
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py` — collection/source/keyword/date semantics and no unbounded request scan.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py` — direct keyword/collection/membership normalization and same-transaction epoch invalidation.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py` — direct conversation-source normalization and same-transaction epoch invalidation.
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py` — additive response and portable-boundary validation.
- `tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py` — pending/complete compatibility behavior and pagination.
- `tldw_Server_API/tests/Services/test_startup_optional_workers.py` — worker lifecycle registration.
- `tldw_Server_API/tests/Services/test_openapi_contracts.py` — exact hybrid-response OpenAPI assertions.
- `Docs/Notes/Moodboards.md` — smart projection completeness and portable collection semantics.
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json` — reviewed API-contract fingerprint.

**Generate for verification; do not commit**

- `apps/tldw-frontend/lib/api/generated/openapi.json`
- `apps/tldw-frontend/lib/api/generated/schema.d.ts`

### Task 0: Start the child after TASK-13007.1 lands

- [ ] **Step 1: Verify dependency and attach this plan**

```bash
backlog task 13007.1 --plain
backlog task edit 13007.2 -s "In Progress" \
  --doc Docs/superpowers/plans/2026-08-25-portable-moodboard-smart-match-projection-implementation-plan.md \
  --plan $'1. Add dormant portable notes.note modification-time projection.\n2. Upgrade derived projection storage to schema v62.\n3. Implement backend-independent smart-rule evaluation and portable dependencies.\n4. Add bounded resumable generation workers.\n5. Serve honest hybrid completeness and update API contracts/docs.\n6. Prove SQLite/PostgreSQL parity and close the child.\n\nADR required: no\nADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md\nReason: ADR-040 already governs this child.'
```

Expected: TASK-13007.1 is Done; TASK-13007.2 becomes In Progress before production edits.

### Task 1: Add dormant portable `notes.note` modification-time projection

- [ ] **Step 1: Write RED tests for the time-selection matrix**

Cover server-bound acceptance time, trusted bootstrap `last_modified`, old-envelope immutable `received_at_server` fallback, invalid/client-selected values, exact retry, group-plan fingerprint inclusion, tombstone/restore, and projection into the existing `notes.last_modified` column.

```python
def test_old_note_envelope_projects_received_at_server() -> None:
    envelope = note_envelope(routing_metadata={}, received_at_server="2026-08-25T01:02:03Z")
    apply_note(envelope)
    assert read_note(envelope.object_id)["last_modified"] == "2026-08-25T01:02:03Z"


def test_client_cannot_choose_canonical_modified_at() -> None:
    outcome = push_note(routing_metadata={"canonical_modified_at": "2020-01-01T00:00:00Z"})
    assert outcome.error_code == "notes_note_canonical_modified_at_server_owned"
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py \
  -k 'canonical_modified_at or received_at_server'
```

- [ ] **Step 3: Implement one selector and materializer path**

```python
def canonical_note_modified_at(envelope: SyncEnvelope) -> str:
    value = envelope.routing_metadata.get("canonical_modified_at")
    if value is not None:
        return normalize_server_bound_timestamp(value)
    return normalize_server_bound_timestamp(envelope.received_at_server)
```

Trusted bootstrap conversion supplies the normalized source time through its private server path. All ordinary server acceptance and future production stamping remain disabled in this child. Update every note upsert/tombstone/restore materializer to write the selected value; never add a second product column.

- [ ] **Step 4: Prove production history remains unchanged and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  -k 'note and (canonical_modified_at or routing_metadata or capture)'
git add tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py
git commit -m "feat(sync): support portable note modification time"
```

Expected: harness-created envelopes project portably, while existing REST/server capture still omits the new routing key.

### Task 2: Upgrade derived projection storage to schema v62

- [ ] **Step 1: Write migration/catalog/resume RED tests**

Assert schema 62, fresh/61→62 parity, normalized note/keyword/conversation-source projection fields, exact algorithm ID, owner/dataset dependency epoch, board dirty/rebuild state, completed generations, smart matches, durable keyset cursors, generation uniqueness, publication indexes, forced RLS, rollback checkpoints, and no canonical lineage columns on disposable rows.

For live PostgreSQL, seed a large v61 fixture and assert explicit lock/statement timeouts, deterministic keyset pages, row/wall-clock budgets, durable `chacha_schema_migration_progress` phase/cursor/count/fingerprint state, failure after each page and DDL/index/constraint/RLS/version boundary, resume without full rescan, final source/target verification, and version bump last.

- [ ] **Step 2: Run RED**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_migration_v62.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_postgres.py
```

Expected: fail, never skip; PostgreSQL lacks the durable progress record, configured timeouts, multi-page normalized backfill, boundary-fault recovery, and resume/final-verification behavior.

- [ ] **Step 3: Implement transactional SQLite 61→62 migration**

Add fixed DDL and bounded backfill helpers. Store comparison values produced by Python `unicodedata.normalize("NFC", text).casefold()`, plus the exact `nfc-casefold-ucd-<version>-v1` algorithm ID. Use keyset backfill, deterministic progress, verification, and version-last semantics.

- [ ] **Step 4: Implement bounded resumable PostgreSQL backfill, then DDL/RLS verification**

Reuse the v61 private migration-progress protocol. Normalize legacy notes, keywords, collection memberships, and conversation sources in deterministic `(owner_user_id, source_primary_key)` pages, committing cursor/count/fingerprint after each bounded page. Set lock and statement timeouts for every transaction and resume idempotently after crash or timeout. Final verification recomputes exact counts/fingerprints before schema version 62 is written.

Force owner/dataset RLS on epoch, dirty/rebuild, generation, and match tables. Use exact `USING`/`WITH CHECK`, non-owner test roles, indexed keyset predicates, and fail closed on extra/drifted policies or indexes. The required-live test must execute multiple backfill pages and at least one interruption/resume path rather than only fresh DDL.

- [ ] **Step 5: Run SQLite and required-live GREEN; commit**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_migration_v62.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_postgres.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_smart_projection_store.py \
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_migration_v62.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_postgres.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git commit -m "feat(notes): add bounded smart projection storage"
```

Expected: required-live PostgreSQL executes a multi-page v61→62 normalized backfill, resumes after injected interruption, verifies exact aggregates/catalog/RLS, and only then records schema 62.

### Task 3: Implement portable rules and dependency fingerprints

- [ ] **Step 1: Write rule normalization/evaluation RED tests**

Cover NFC/casefold vectors, exact Unicode-data-version compatibility, literal `%`/`_`, title/content substring, keyword OR, category AND, exact sources, portable collection membership, inclusive UTC bounds, dedup/sort, null/empty categories, tombstoned known collections, unknown/cross-scope rejection, and SQLite/PostgreSQL equivalence.

```python
def test_sql_metacharacters_are_literal() -> None:
    rule = normalize_rule({"query": "%_"}, dependencies())
    assert matches(rule, normalized_note(content="literal %_ token")) is True
    assert matches(rule, normalized_note(content="unrelated")) is False


def test_collection_ids_translate_to_portable_sync_ids() -> None:
    canonical = canonicalize_rest_rule(
        {"notebook_collection_ids": [LOCAL_COLLECTION_ID]},
        owner_id=OWNER,
        dataset_id=DATASET,
    )
    assert canonical["collection_sync_ids"] == [COLLECTION_SYNC_ID]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py \
  -k 'normalize or literal or collection or source or updated'
```

- [ ] **Step 3: Implement pure evaluator and exact dependency fingerprint**

The dependency fingerprint must include board canonical revision/hash, comparison algorithm ID, notes/organization/conversation dependency epochs, and canonical rule bytes. Candidate SQL may narrow by scoped indexed relationships/time/ID only; final literal Unicode semantics run against stored Python-normalized values.

- [ ] **Step 4: Add one common transactional invalidation hook at every authority seam**

Board rule changes dirty one board. Narrow placement changes do not dirty smart results. Direct and Sync-origin note, keyword, collection, collection-membership, and conversation-source mutations call one store-level normalization/invalidation helper using their existing product transaction. Each helper writes the normalized authority value and increments exactly one owner/dataset epoch before commit; rollback leaves both unchanged. No endpoint-only hook is sufficient because DB/service callers may bypass it.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py
git add tldw_Server_API/app/core/Notes/moodboard_smart_projection.py \
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_smart_projection_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py
git commit -m "feat(notes): evaluate portable moodboard rules"
```

### Task 4: Build and publish generations under explicit budgets

- [ ] **Step 1: Write scheduler/worker RED tests**

Cover row and wall-clock limits, stale-board discovery cursors, sparse candidates, high pages, crash after page, resume, dependency invalidation mid-build, atomic final publication, abandoned-generation cleanup bounds, concurrent worker claim, retry/backoff, and no request-path worker execution.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py \
  tldw_Server_API/tests/Services/test_notes_moodboard_projection_worker.py
```

- [ ] **Step 3: Implement one bounded scheduler turn and one bounded rebuild turn**

```python
@dataclass(frozen=True, slots=True)
class SmartProjectionBudget:
    row_limit: int = 500
    wall_clock_ms: int = 100


def run_rebuild_turn(*, store, owner_id, dataset_id, board_id, budget, clock) -> RebuildTurn:
    """Resume one keyset page and publish only after complete verification."""
```

Persist the continuation cursor after each committed page. Write matches into an unpublished generation. Under one final transaction, recheck the dependency fingerprint, mark the generation complete/current, and only then make it readable.

- [ ] **Step 4: Register bounded lifecycle without public capability changes**

Register the worker with existing startup/shutdown services. Startup performs no unbounded synchronous backfill. Failures create privacy-safe retry state and cannot make a dataset appear ready.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py \
  tldw_Server_API/tests/Services/test_notes_moodboard_projection_worker.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py
git add tldw_Server_API/app/services/notes_moodboard_projection_worker.py \
  tldw_Server_API/app/services/startup_optional_workers.py \
  tldw_Server_API/tests/Services/test_notes_moodboard_projection_worker.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py
git commit -m "feat(notes): rebuild smart matches incrementally"
```

### Task 5: Serve honest hybrid completeness and update public contracts

- [ ] **Step 1: Write endpoint/schema/OpenAPI RED tests**

Assert a missing/stale generation returns only manual rows, `smart_results_complete=false`, `smart_projection_status="pending"`, `total=null`, and legacy order `(last_modified DESC,id DESC)`. A current complete generation returns manual/smart/both rows, exact indexed total, generation/revision, stable offset pagination, and no live full scan.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'moodboard'
```

- [ ] **Step 3: Replace request-time smart scans with generation reads**

Delete the request path's use of `_build_moodboard_smart_rule_sql_parts()` for full evaluation. Keep a compatibility helper only if non-request tests need it, clearly marked non-authoritative. Merge manual scoped rows with current-generation rows by note ID and preserve legacy display ordering.

- [ ] **Step 4: Update docs, fingerprint, and generated client proof**

Document pending semantics, nullable total, portable collection translation, dependencies, and local derived ownership in `Docs/Notes/Moodboards.md`.

```bash
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
PYTHON=../../.venv/bin/python bun --cwd apps/tldw-frontend run generate:api-types
git diff -- apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

The first check is expected to fail before the reviewed refresh. Regenerate, inspect the moodboard-only semantic delta and generated `schema.d.ts`, commit only the fingerprint, and rerun the check successfully.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'moodboard'
git add tldw_Server_API/app/api/v1/schemas/notes_moodboards.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  Docs/Notes/Moodboards.md \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat(notes): expose smart projection completeness"
```

### Task 6: Prove parity, bounds, dormancy, and close TASK-13007.2

- [ ] **Step 1: Run the required child matrix with live PostgreSQL**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_migration_v62.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_moodboard_smart_projection_postgres.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_moodboard_smart_projection.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_notes_moodboard_projection_worker.py \
  tldw_Server_API/tests/Services/test_startup_optional_workers.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
```

Expected: no PostgreSQL skip; Unicode, wildcard, timestamp, collection, keyword, source, sparse-query, count, high-page, RLS, and indexed-plan matrices pass.

- [ ] **Step 2: Run static/security/OpenAPI/diff gates**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/Notes/moodboard_smart_projection.py
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_smart_projection_store.py
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py
  tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py
  tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py
  tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes.py
  tldw_Server_API/app/core/Sync/v2/materializers/notes.py
  tldw_Server_API/app/core/Sync/v2/models.py
  tldw_Server_API/app/services/notes_moodboard_projection_worker.py
  tldw_Server_API/app/services/startup_optional_workers.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
  tldw_Server_API/app/api/v1/schemas/notes_moodboards.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13007-2-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
git diff --check
```

- [ ] **Step 3: Self-review authority and bounds**

Confirm smart matches never enter product placement rows or Sync envelopes, request paths never scan all notes, stale/partial generations never claim completeness, production note history remains unstamped, all three new domains remain publicly absent, and every worker/read query binds owner and dataset.

- [ ] **Step 4: Finalize the child**

Record exact SQLite/live PostgreSQL/OpenAPI/generated-type/Ruff/Bandit evidence, check TASK-13007.2 AC/DoD, add concise implementation notes and any real lesson, set it Done, and commit the closeout before TASK-13007.3 begins.
