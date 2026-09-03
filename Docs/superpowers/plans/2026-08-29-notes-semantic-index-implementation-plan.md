# Notes Semantic Index and Graph Edges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly enabled, source-grounded semantic index for Notes and expose verified semantic relationships in the existing Notes Graph workspace without changing legacy graph behavior for existing clients.

**Architecture:** Keep canonical Notes, manual links, Sync, and the synchronous ordinary graph service authoritative. Add server-local semantic configuration, generation, manifest, and work ledgers in ChaChaNotes; store vectors through a dedicated async Notes facade over ChromaDB or bounded pgvector tables; publish generations with compare-and-swap fences; and compose semantic edges through an async projector at the endpoint boundary. The shared React UI manages disclosure, indexing state, semantic filters, evidence, and manual-link conversion for both the WebUI and browser extension.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL, ChaChaNotes stores, Jobs/WorkerSDK, existing embedding orchestration, ChromaDB, pgvector, React 18, TanStack Query, Cytoscape/Dagre, Ant Design, lucide-react, Vitest/Testing Library/axe, Playwright, pytest, Hypothesis, Ruff, Bandit.

**Spec:** `Docs/superpowers/specs/2026-08-29-notes-semantic-index-design.md`

---

## Global Constraints

- Backlog authority is `TASK-13134`. Keep its status, notes, plan link, touched files, verification, and PR link current through the Backlog.md MCP/CLI. Do not edit the Backlog task file manually.
- Rebase on current `origin/dev` before implementation. Recompute the next available ChaChaNotes and AuthNZ migration numbers after the rebase; do not assume the versions visible when this plan was written remain free.
- Use `superpowers:test-driven-development` for every implementation task, `superpowers:systematic-debugging` for unexpected failures, `superpowers:requesting-code-review` after integrated implementation, and `superpowers:verification-before-completion` before commits or PR handoff.
- Activate `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv` before Python, pytest, Ruff, or Bandit commands.
- Keep SQL in `tldw_Server_API/app/core/DB_Management/`. Core services consume typed store methods.
- Keep the existing `NoteGraphService` synchronous. Provider and vector I/O belongs in a new async `SemanticGraphProjector` called by the endpoint/application boundary.
- Omitted `edge_types` must always resolve to the explicit legacy five edge types. Never derive this default from `set(EdgeType)` after adding `semantic`.
- Semantic enablement is explicit and revision-bound. No Note content may reach an embedding provider before capability disclosure, consent, and dimension resolution.
- Provider fallback is disabled. A configured provider/model is pinned for the generation and may use only bounded retries against that same configuration.
- Jobs payloads/results/logs contain opaque owner-bound domain identities and bounded counters/error codes only. They never contain Note text, chunks, vector values, credentials, raw endpoint URLs, or database paths.
- The vector layer persists only opaque IDs and vectors. It must not persist Note text or placeholder documents.
- All evidence offsets are half-open Unicode code-point offsets relative to either the canonical `title` or `content` field. UTF-8 is used only for hashes and byte limits.
- Semantic query failure must fail open to the ordinary graph: return HTTP 200 with the verified ordinary graph and a typed semantic status, never unverified semantic edges.
- Full dataset import/restore and Sync export do not carry semantic configuration, generations, manifests, or vectors. Individual Sync note mutations on an already enabled local dataset only mark local semantic work dirty.
- Notes/account erasure must fence semantic work, physically delete semantic vectors, confirm deletion, then remove manifests/configuration and canonical Notes. Unconfirmed semantic cleanup fails the erasure operation.
- Each task ends in a focused commit after its listed tests pass. Preserve unrelated worktree changes.

## Stable Interfaces

Use these names and ownership boundaries unless a current `origin/dev` conflict requires a mechanically equivalent name:

```python
LEGACY_EDGE_TYPES: frozenset[EdgeType] = frozenset({
    EdgeType.MANUAL,
    EdgeType.WIKILINK,
    EdgeType.BACKLINK,
    EdgeType.TAG_MEMBERSHIP,
    EdgeType.SOURCE_MEMBERSHIP,
})


@dataclass(frozen=True, slots=True)
class SemanticVectorMatch:
    vector_id: str
    cosine_distance: float


class NotesSemanticVectorStore(Protocol):
    async def create_generation(self, binding: SemanticGenerationBinding) -> None: ...
    async def upsert(self, binding: SemanticGenerationBinding, vectors: Sequence[SemanticVector]) -> None: ...
    async def fetch(self, binding: SemanticGenerationBinding, vector_ids: Sequence[str]) -> Mapping[str, tuple[float, ...]]: ...
    async def query(self, binding: SemanticGenerationBinding, query_vectors: Sequence[Sequence[float]], *, limit: int) -> tuple[tuple[SemanticVectorMatch, ...], ...]: ...
    async def delete_ids(self, binding: SemanticGenerationBinding, vector_ids: Sequence[str]) -> None: ...
    async def delete_generation(self, binding: SemanticGenerationBinding) -> None: ...


class SemanticGraphProjector:
    async def project(self, request: NoteGraphRequest, ordinary: NoteGraphResponse, *, user: User) -> NoteGraphResponse: ...
```

The semantic run API is nested under `/api/v1/notes/graph/semantic-index`; no root-level user Jobs route is introduced.

---

## Stage 1: Persistence, Authority, And Lifecycle

**Goal:** Establish owner-scoped semantic state and make every canonical Note mutation update semantic authority atomically.

**Success Criteria:** SQLite/PostgreSQL schemas are equivalent, PostgreSQL RLS is forced, semantic configuration is server-local, and creates/edits/trash/delete/restore/Sync mutations produce fenced dirty or tombstone work without provider/vector calls.

**Tests:** Migration rollback/parity, store state-machine/property tests, lifecycle transaction tests, Sync materializer tests, and AuthNZ permission tests.

**Status:** Not Started

### Task 1: Add Semantic Persistence Records And Migrations

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_semantic_models.py`
- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration.py`
- Test: `tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration_postgres.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_store.py`

- [ ] **Step 1: Write failing migration tests**

Assert fresh schema creation and previous-version migration for `note_semantic_index_configs`, `note_semantic_generations`, `note_semantic_note_state`, `note_semantic_chunks`, and `note_semantic_work`. Verify checks, foreign keys, owner/dataset keys, one active/staging writer constraint, work coalescing indexes, and rollback leaves the old schema version unchanged after an injected statement failure.

- [ ] **Step 2: Write failing PostgreSQL parity/RLS tests**

Use the established live PostgreSQL fixture. Assert equivalent columns and constraints plus `ENABLE ROW LEVEL SECURITY`, `FORCE ROW LEVEL SECURITY`, and owner predicates on every shared table. Verify dimension-specific vector tables are not created by this migration.

- [ ] **Step 3: Write failing typed-store tests**

Cover create/enable/disable CAS, capability/configuration revision mismatch, active-generation switching, monotonically increasing semantic index revisions, dirty generation claims, note-manifest publication, tombstones, cleanup work, bounded retry timestamps, foreign-owner invisibility, and sanitized error/display fields.

Representative invariant:

```python
published = store.publish_note_manifest(
    owner_user_id=owner,
    dataset_id=dataset,
    generation_id=generation,
    note_id=note_id,
    claimed_dirty_generation=3,
    content_version=7,
    manifest=manifest,
)
assert published is False  # concurrent edit advanced dirty_generation to 4
```

- [ ] **Step 4: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration.py tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration_postgres.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_store.py -q
```

Expected: FAIL because the semantic schema/store does not exist.

- [ ] **Step 5: Implement the next available migration and store**

Add closed enums/literals for desired state, generation state, dimension state, note state, work kind, and work claim state. Store only the fields approved in the spec. Instantiate `NoteSemanticStore(self)` in `CharactersRAGDB`. SQLite may rely on the per-user database boundary; PostgreSQL includes explicit owner keys and forced RLS.

- [ ] **Step 6: Run migration/store tests and adjacent migration regressions**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration.py tldw_Server_API/tests/DB_Management/test_chacha_semantic_migration_postgres.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_store.py tldw_Server_API/tests/DB_Management/test_chacha_migration_v64.py -q
```

Expected: PASS. Replace the final adjacent migration filename if `origin/dev` advanced it.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/tests/DB_Management tldw_Server_API/tests/Notes_Graph/unit/test_semantic_store.py
git commit -m "feat: add Notes semantic index persistence (TASK-13134)"
```

### Task 2: Integrate Note And Sync Lifecycle Fences

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Inspect: `tldw_Server_API/app/core/Sync/v2/materializers/notes.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_semantic_lifecycle.py`

- [ ] **Step 1: Write failing lifecycle transaction tests**

For an enabled local dataset, prove add/edit/individual restore increments a note dirty generation and coalesces `index_note`; trash/hard delete publishes a tombstone, increments the semantic index revision, and coalesces vector deletion. Roll back the Note write and assert the semantic ledger also rolls back. For a disabled dataset, assert the same operations create no semantic work.

- [ ] **Step 2: Write failing Sync tests**

Drive `upsert_note_from_sync` and `tombstone_note_from_sync`. Assert enabled local datasets are marked dirty transactionally and provider/vector fakes receive no calls. Exercise Notes JSON export/import against a fresh dataset and prove semantic configuration, manifests, and vectors are absent and indexing remains disabled.

- [ ] **Step 3: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py tldw_Server_API/tests/Sync/test_sync_v2_notes_semantic_lifecycle.py -q
```

Expected: FAIL because Note mutations do not update semantic state.

- [ ] **Step 4: Add transaction-local semantic hooks**

Call no-op-when-disabled store methods from existing `NoteStore` transactions:

```python
self._db.note_semantic_store.mark_note_dirty(
    tx=conn,
    note_id=note_id,
    content_version=version,
    content_fingerprint=fingerprint,
)
```

Use the existing Sync paths through `NoteStore`; do not call Jobs, embeddings, or vector storage in the transaction. Explicitly clear/omit semantic server-local state in whole-dataset restore code only if current restore plumbing would otherwise copy it.

- [ ] **Step 5: Run lifecycle, Sync, and ordinary graph regressions**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py tldw_Server_API/tests/Sync/test_sync_v2_notes_semantic_lifecycle.py tldw_Server_API/tests/Notes_Graph/integration/test_graph_lifecycle_queries.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/note_store.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py tldw_Server_API/tests/Sync/test_sync_v2_notes_semantic_lifecycle.py
git commit -m "feat: fence Notes semantic lifecycle changes (TASK-13134)"
```

### Task 3: Add Capability Policy And Semantic Management Permission

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_capabilities.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_settings.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/rbac_seed.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_settings.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_semantic_permissions.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_notes_graph_semantic_permissions_postgres.py`

- [ ] **Step 1: Write failing capability identity tests**

Assert the compatibility hash changes for provider, model/revision, backend, metric, dimensions, normalization, or chunker version. Assert the disclosure hash changes for endpoint-origin revision, execution/storage boundary, or outbound categories, but not credential rotation. Verify endpoint/storage labels expose only allowlisted sanitized values and unknown boundaries fail closed as external/unavailable. Validate one typed settings object for kill switch, per-Note/chunk/run/provider/query/cleanup caps, bounded retry/backoff, and the operator-controlled pgvector dimension allowlist.

- [ ] **Step 2: Write failing permission tests**

Add `notes.graph.semantic.manage`. Prove approved default roles receive it, revoked roles cannot manage semantic indexing, existing `notes.graph.read` can read status/edges, and existing graph write plus Sync authority remains required for manual conversion.

- [ ] **Step 3: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_settings.py tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_semantic_permissions.py tldw_Server_API/tests/AuthNZ/integration/test_notes_graph_semantic_permissions_postgres.py -q
```

Expected: FAIL with missing capability policy and permission.

- [ ] **Step 4: Implement identities, disclosure, and the next AuthNZ migration**

Return typed capability data including active Note count, bounded processing estimate, provider/model labels, durable credential-source availability, data and storage boundaries, effective limits, availability reason, and a stable capability revision. Request-only credentials are unavailable to background work and must fail capability preflight rather than be persisted. Seed the new manage permission using the next free AuthNZ migration number after rebase.

- [ ] **Step 5: Run capability and AuthNZ tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_settings.py tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_semantic_permissions.py tldw_Server_API/tests/AuthNZ/integration/test_notes_graph_semantic_permissions_postgres.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_capabilities.py tldw_Server_API/app/core/Notes_Graph/semantic_settings.py tldw_Server_API/app/core/AuthNZ/permissions.py tldw_Server_API/app/core/AuthNZ/settings.py tldw_Server_API/app/core/AuthNZ/rbac_seed.py tldw_Server_API/app/core/AuthNZ/migrations.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_settings.py tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_semantic_permissions.py tldw_Server_API/tests/AuthNZ/integration/test_notes_graph_semantic_permissions_postgres.py
git commit -m "feat: gate Notes semantic indexing by capability (TASK-13134)"
```

---

## Stage 2: Canonical Content, Embeddings, And Vector Storage

**Goal:** Produce deterministic, privacy-bounded Note chunks and store/query their vectors through a strict vector-only contract.

**Success Criteria:** Canonical offsets reconstruct exactly, dimensions are pinned before Note transfer, provider fallback and durable caches are disabled, zero-norm/non-finite vectors fail closed, and ChromaDB/pgvector satisfy one shared contract.

**Tests:** Canonicalization/property tests, embedding policy tests, vector contract tests, ChromaDB tests, and live PostgreSQL pgvector tests.

**Status:** Not Started

### Task 4: Implement Canonical Chunking And Strict Embedding Execution

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_content.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_embeddings.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_content.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_embeddings.py`
- Test: `tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py`

- [ ] **Step 1: Write failing canonicalization/property tests**

Reuse the existing canonical title/content normalization behavior from `suggestion_content.py`. Cover CRLF/LF, NFC, empty fields, astral Unicode, field-relative half-open code-point offsets, deterministic chunk IDs/fingerprints, current-field reconstruction, and configured Note/chunk byte/code-point caps. Use Hypothesis to prove chunks never cross field boundaries and reconstruct to the exact canonical slice.

- [ ] **Step 2: Write failing embedding policy tests**

Inject an `EmbeddingRequestOrchestrator` and assert the semantic wrapper supplies an explicit provider header, `allow_fallback_with_header=False`, pinned model/dimensions, a run-scoped in-memory cache, no durable cache, finite consistent vectors, and zero-norm rejection. Resolve only durable owner/server credentials through `resolve_byok_credentials(..., request=None)`; reject request-only credential sources. Test exact declared dimension, fixed non-user probe for unknown dimension, CAS loss, probe failure before Note reads, provider/model revision capture where available, endpoint-origin policy and cross-origin redirect rejection, and existing provider usage accounting without Note content.

Representative boundary:

```python
class NotesSemanticEmbedder:
    async def resolve_dimensions(self, config: PendingSemanticConfig, *, user_id: str) -> ResolvedDimension: ...
    async def embed_chunks(self, chunks: Sequence[SemanticChunkInput], config: ResolvedSemanticConfig, *, user_id: str) -> SemanticEmbeddingBatch: ...
```

- [ ] **Step 3: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_content.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_embeddings.py tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py -q
```

Expected: FAIL because the semantic content/embedder modules do not exist.

- [ ] **Step 4: Implement the smallest endpoint-neutral embedding seam**

Build `NotesSemanticEmbedder` around the existing `EmbeddingRequestOrchestrator` with a Notes-owned executor adapter that resolves durable credentials for the owner and invokes the existing provider primitives. Keep this adapter in `semantic_embeddings.py`; do not import private endpoint helpers. Use the fixed probe string only after consent and before any Note read. Treat unavailable durable credentials, dimension disagreement, non-finite values, zero norm, and unavailable pinned provider as typed systemic failures.

- [ ] **Step 5: Run semantic and existing orchestrator tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_content.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_embeddings.py tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_content.py tldw_Server_API/app/core/Notes_Graph/semantic_embeddings.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_content.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_embeddings.py tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py
git commit -m "feat: add deterministic Notes semantic embeddings (TASK-13134)"
```

### Task 5: Implement The Dedicated Notes Vector Facade

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_vectors.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_vectors_chroma.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_vectors_pg.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py`
- Test: `tldw_Server_API/tests/Notes_Graph/vector_contract.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_vectors.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_chroma.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_pg.py`

- [ ] **Step 1: Write failing facade contract tests**

Build one reusable contract suite for create/upsert/fetch/query/delete IDs/delete generation, owner/dataset/generation isolation, dimension mismatch, non-finite/zero-norm rejection, result shape, unsupported dimension, and idempotent cleanup confirmation. Assert both backends return raw cosine distance and graph-level `max(0, min(1, 1 - distance))` produces equivalent score semantics.

- [ ] **Step 2: Write failing ChromaDB tests**

Assert generation storage is first probed without mutation. Existing storage is validated in place; absent storage is created with `{"hnsw:space": "cosine"}`, and a concurrent creator's winner is re-read and validated without rewriting metadata. This avoids legacy `get_or_create_collection` metadata mutation across the declared ChromaDB range. Also assert direct writes contain `ids` plus `embeddings` only, never documents/metadatas/Note text, never call `store_in_chroma`, and use opaque owner/dataset/generation namespace mapping.

- [ ] **Step 3: Write failing pgvector tests**

Against the established PostgreSQL fixture, assert only operator-allowlisted dimensions at or below the 2,000-dimension `vector` HNSW ceiling map to a bounded fixed `vector(dim)` table, composite owner/dataset/generation/vector ID keys, cosine ANN index, forced RLS, parameterized row operations, and generation deletion by rows rather than tables. Bind all physical operations to the exact schema resolved at capability initialization. Verify filtered iterative HNSW uses independent bounded scan-tuple and total candidate-output controls, and metric labels cannot contain owner/dataset/generation/table names.

- [ ] **Step 4: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_vectors.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_chroma.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_pg.py -q
```

Expected: FAIL because the facade/backends do not exist.

- [ ] **Step 5: Implement vector-only backends and factory**

Keep backend construction behind one factory that returns typed capability failure when vector-only guarantees, pgvector extension/schema, or dimension allowlisting is unavailable. Validate the ChaChaNotes generation binding before every operation and revalidate results later in the graph projector.

- [ ] **Step 6: Run contract and backend tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_vectors.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_chroma.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_pg.py -q
```

Expected: PASS. PostgreSQL skips only through the existing unavailable fixture.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_vectors.py tldw_Server_API/app/core/Notes_Graph/semantic_vectors_chroma.py tldw_Server_API/app/core/Notes_Graph/semantic_vectors_pg.py tldw_Server_API/tests/Notes_Graph/vector_contract.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_vectors.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_chroma.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_vectors_pg.py
git commit -m "feat: add Notes vector-only semantic stores (TASK-13134)"
```

---

## Stage 3: Generation Publication, Jobs, And Management API

**Goal:** Build, activate, update, cancel, disable, and clean semantic generations through bounded owner-scoped Jobs with domain-specific status routes.

**Success Criteria:** Only complete fenced generations activate; incremental manifests publish atomically; failures/cancellation never expose staging vectors; nested API operations are revision/idempotency guarded; cleanup is bounded and recoverable.

**Tests:** Generation service state-machine/property tests, Jobs admission/worker/maintenance tests, endpoint integration tests, route-order tests, and startup lifecycle tests.

**Status:** Not Started

### Task 6: Implement Generation Build And Incremental Publication

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_indexing.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_publication.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_observability.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_indexing.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_publication.py`
- Test: `tldw_Server_API/tests/Notes_Graph/property/test_semantic_publication_invariants.py`

- [ ] **Step 1: Write failing initial-build tests**

Cover pending dimension probe, generation storage creation, bounded Note claims, canonical chunk/vector writes, manifest CAS, convergence, expected/published count verification, vector-ID/dimension/hash/fence verification, atomic active-generation switch, semantic revision increment, publication receipt, previous-generation retirement, and cleanup queueing.

- [ ] **Step 2: Write failing activation-policy tests**

Assert every Note in the fenced snapshot is terminal before activation; systemic provider/configuration/vector failures prevent activation; per-Note exclusions/failures may activate degraded coverage; an eligible non-empty corpus requires at least one indexed Note; and an empty eligible corpus may activate Ready with zero indexed Notes. Edits newer than the snapshot fence remain dirty. Immediately before Note reads and publication, reject missing user/owner authority, revoked semantic-manage permission, disabled desired state, capability/disclosure drift, generation/fence mismatch, provider/model drift, endpoint-policy failure, or vector-capability drift.

- [ ] **Step 3: Write failing incremental/tombstone property tests**

Generate interleavings of edit, vector upsert, manifest publish, new edit, tombstone, and cleanup. Prove an older claim cannot clear new dirtiness; new vector IDs are written before manifest publication; old IDs become invisible immediately after manifest/tombstone publication; and delayed cleanup cannot delete a newer generation.

- [ ] **Step 4: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_indexing.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_publication.py tldw_Server_API/tests/Notes_Graph/property/test_semantic_publication_invariants.py -q
```

Expected: FAIL because indexing/publication services do not exist.

- [ ] **Step 5: Implement bounded indexing and fail-closed activation**

Keep orchestration async and stores sync/transactional. Revalidate the complete execution fence before loading Note content and again before publication. The handler returns the Notes publication receipt; it does not wait for its own terminal Jobs state. Expose only low-cardinality metrics, allowlisted audit fields, and stable error codes. Never cache embeddings outside the current run/batch.

- [ ] **Step 6: Run tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_indexing.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_publication.py tldw_Server_API/tests/Notes_Graph/property/test_semantic_publication_invariants.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_indexing.py tldw_Server_API/app/core/Notes_Graph/semantic_publication.py tldw_Server_API/app/core/Notes_Graph/semantic_observability.py tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: publish fenced Notes semantic generations (TASK-13134)"
```

### Task 7: Add Semantic Jobs, Maintenance, And Nested API Routes

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_jobs.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_api.py`
- Create: `tldw_Server_API/app/services/notes_semantic_index_worker.py`
- Create: `tldw_Server_API/app/services/notes_semantic_maintenance.py`
- Modify: `tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py`
- Create: `tldw_Server_API/app/api/v1/schemas/notes_semantic_index.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/notes_semantic_index.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_endpoints.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_route_order.py`
- Test: `tldw_Server_API/tests/Services/test_notes_semantic_workers.py`

- [ ] **Step 1: Write failing Jobs tests**

Assert one active build/rebuild writer per owner/dataset/configuration, content-free payloads, opaque domain run IDs, Jobs owner-column authority, bounded batch retries against the pinned provider only, cancellation fencing, receipt replay, crash recovery, dirty-work coalescing, failed-Note retry, and generation cleanup confirmation.

- [ ] **Step 2: Write failing endpoint and route-order tests**

Cover all seven nested routes. Assert `PUT`/`DELETE` return 202, mutating routes require `Idempotency-Key` and expected revision, enable binds capability revision, run modes are `rebuild` or `retry_failed`, foreign runs return 404, permission failures return 403, conflicts return typed 409, invalid input returns typed 422, unavailable enable returns sanitized 503, and route registration precedes parameterized Notes routes. Assert no root-level Jobs API is exposed for this feature.

Representative API shape:

```python
@router.post("/graph/semantic-index/runs/{run_id}/cancel", status_code=202)
async def cancel_semantic_run(
    run_id: UUID,
    body: SemanticRunCancelRequest,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    user: User = Depends(require_semantic_manage),
) -> SemanticIndexMutationResponse: ...
```

- [ ] **Step 3: Write failing startup/maintenance tests**

Prove app-managed and standalone workers share one handler, environment flags cannot cause duplicate ownership, maintenance claims bounded work, disabled operator kill switch prevents provider/query work while status/cancel/delete/cleanup remain available, and shutdown drains cleanly.

- [ ] **Step 4: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_endpoints.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_route_order.py tldw_Server_API/tests/Services/test_notes_semantic_workers.py -q
```

Expected: FAIL because the domain Jobs/API/worker routes do not exist.

- [ ] **Step 5: Implement Jobs and derived status projection**

Use `domain="notes"`, a dedicated semantic queue/type, and opaque payload identities. Derive Off, Preparing, Ready/Updating, and Needs attention from desired state, generations, coverage, active Jobs, drift, and cleanup; do not persist one overloaded UI state. Main status returns only a bounded active-run summary and nested run link.

- [ ] **Step 6: Run endpoint/worker tests and OpenAPI smoke coverage**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_endpoints.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_route_order.py tldw_Server_API/tests/Services/test_notes_semantic_workers.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_jobs.py tldw_Server_API/app/core/Notes_Graph/semantic_api.py tldw_Server_API/app/services/notes_semantic_index_worker.py tldw_Server_API/app/services/notes_semantic_maintenance.py tldw_Server_API/app/services/startup_study_privilege_jobs_pollers.py tldw_Server_API/app/api/v1/schemas/notes_semantic_index.py tldw_Server_API/app/api/v1/endpoints/notes_semantic_index.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/tests/Notes_Graph tldw_Server_API/tests/Services/test_notes_semantic_workers.py
git commit -m "feat: add Notes semantic index operations (TASK-13134)"
```

---

## Stage 4: Semantic Graph Projection And Erasure

**Goal:** Add verified semantic edges to first-page focused graph requests while preserving ordinary graph availability and complete data-subject deletion.

**Success Criteria:** Legacy requests remain contract-compatible, semantic parameters bind caches/cursors, deterministic precedence and evidence caps hold, vector failures degrade to ordinary graph 200, and Notes/account erasure confirms physical vector deletion.

**Tests:** Schema/default regression tests, graph projector unit/property tests, endpoint/cache/cursor tests, backend parity tests, DSR tests, and security tests.

**Status:** Not Started

### Task 8: Extend Graph Contracts Without Changing Legacy Defaults

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/graph_service.py`
- Modify: `tldw_Server_API/app/core/Notes_Graph/graph_cache.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_graph_service.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_graph_cache.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_graph_schema.py`

- [ ] **Step 1: Write failing legacy-default regression tests**

Assert omitted `edge_types` resolves exactly to manual, wikilink, backlink, tag membership, and source membership after `semantic` enters `EdgeType`. Assert ordinary requests and cursors do not carry semantic readiness/generation state and produce unchanged nodes/edges/status.

- [ ] **Step 2: Write failing semantic schema/hash tests**

Add bounded `semantic_top_k` and `semantic_threshold`, valid only when semantic is requested. Add typed semantic status/evidence. Assert semantic edge type and every semantic parameter/revision participates in request hashes, outer cache keys, and semantic cursor binding. Assert later pages reject semantic cursor mismatches but continue the ordinary cursor without rerunning semantic search.

- [ ] **Step 3: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_graph_service.py tldw_Server_API/tests/Notes_Graph/unit/test_graph_cache.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_graph_schema.py -q
```

Expected: FAIL because semantic graph contracts are absent.

- [ ] **Step 4: Add explicit defaults and semantic-aware immutable keys**

Keep ordinary projection generation/readiness independent of semantic state. Allow bounded ordinary candidate overfetch only when the async projector requests it, capped by the hard semantic admission allowance. Do not add async calls to `NoteGraphService`.

- [ ] **Step 5: Run schema/cache and existing graph tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_graph_service.py tldw_Server_API/tests/Notes_Graph/unit/test_graph_cache.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_graph_schema.py tldw_Server_API/tests/Notes_Graph/integration/test_graph_endpoint.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/notes_graph.py tldw_Server_API/app/core/Notes_Graph/graph_service.py tldw_Server_API/app/core/Notes_Graph/graph_cache.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: extend Notes graph semantic contracts (TASK-13134)"
```

### Task 9: Implement Async Semantic Projection, Evidence, And Precedence

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_scoring.py`
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_projector.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_scoring.py`
- Test: `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_projector.py`
- Test: `tldw_Server_API/tests/Notes_Graph/property/test_semantic_composition_invariants.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_graph_endpoint.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_manual_conversion.py`

- [ ] **Step 1: Write failing scoring/evidence tests**

Cover finite clamped `1 - cosine_distance`, strongest-current-chunk scoring, stable Note-ID tie breaks, threshold/top-k, at most three deterministic evidence pairs, current fingerprint/version revalidation, 480 code points per excerpt, 2,880 per edge, and stable 256 KiB response evidence truncation that preserves the edge and reports `semantic_evidence_bytes`.

- [ ] **Step 2: Write failing projection/precedence tests**

Assert semantic requires first page plus a current focus Note; graph semantic rate limiting; batch/vector/candidate budgets; self/cross-owner/deleted/stale/unpublished exclusion; graph filter application; no neighbor expansion; manual relationship supersession; wikilink/backlink coexistence; and precedence `focus > manual > wikilink/backlink > semantic > tag/source membership`. Semantic may displace only membership candidates, and unused semantic allowance returns to ordinary candidates.

- [ ] **Step 3: Write failing failure/cache tests**

Provider/vector unavailable, stale configuration, disabled/building state, focus absence, or malformed backend results return ordinary graph 200 plus typed semantic status and no semantic edges. Stable first-page projections reuse `GraphCache` bound to generation/index/configuration/parameters; fresh progress/status is injected after cache retrieval; transient failures are never cached.

Add an optional typed semantic conversion context to `NoteLinkCreate`. It carries only the generation ID and is validated against the owner, dataset, source/target pair, and current semantic relationship before calling the existing Sync-aware link coordinator. It is never persisted as link properties. Audit the actor, Note IDs, generation, and result only; existing callers that omit it remain unchanged.

- [ ] **Step 4: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph/unit/test_semantic_scoring.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_projector.py tldw_Server_API/tests/Notes_Graph/property/test_semantic_composition_invariants.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_graph_endpoint.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_manual_conversion.py -q
```

Expected: FAIL because the async projector does not exist.

- [ ] **Step 5: Implement projector plus pure compositor**

The endpoint awaits the projector after obtaining ordinary candidates. Keep score/evidence/precedence composition pure and independently testable. Validate every vector result against current ChaChaNotes generation, manifest, Note authority, content version, and fingerprint before constructing an edge.

- [ ] **Step 6: Run semantic and complete graph suites**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_scoring.py tldw_Server_API/app/core/Notes_Graph/semantic_projector.py tldw_Server_API/app/api/v1/endpoints/notes_graph.py tldw_Server_API/tests/Notes_Graph
git commit -m "feat: project verified semantic Note edges (TASK-13134)"
```

### Task 10: Integrate Semantic Cleanup With Data-Subject Erasure

**Files:**
- Create: `tldw_Server_API/app/core/Notes_Graph/semantic_erasure.py`
- Modify: `tldw_Server_API/app/services/admin_data_subject_requests_service.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py`
- Test: `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_erasure.py`

- [ ] **Step 1: Write failing erasure-order tests**

Assert Notes erasure fences/admission-blocks semantic jobs, retains cleanup identity, deletes Chroma/pgvector generations, confirms absence, then deletes semantic manifests/configuration and canonical Notes. Assert account erasure includes this path even when the optional generic embeddings category is omitted. Vector cleanup timeout/failure must fail the DSR and preserve enough owner-bound state to retry.

- [ ] **Step 2: Write failing race tests**

Simulate an in-flight worker publishing after erasure starts and prove its fence fails. Simulate delayed old-generation cleanup after a new generation exists and prove it cannot target the new binding.

- [ ] **Step 3: Run the red tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_erasure.py -q
```

Expected: FAIL because Notes erasure does not coordinate semantic vectors.

- [ ] **Step 4: Implement the erasure coordinator**

Invoke `SemanticErasureCoordinator` from the Notes erasure handler before raw Notes database deletion. Keep generic media embeddings erasure unchanged. Emit only bounded operation/backend/error metrics and audit fields.

- [ ] **Step 5: Run DSR and semantic lifecycle tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_erasure.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_note_lifecycle.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Notes_Graph/semantic_erasure.py tldw_Server_API/app/services/admin_data_subject_requests_service.py tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_erasure.py
git commit -m "fix: erase Notes semantic vectors with Notes data (TASK-13134)"
```

---

## Stage 5: Shared UI, Documentation, And Release Gates

**Goal:** Expose understandable semantic setup, progress, filtering, evidence, and manual conversion in the shared Notes Graph experience with equivalent WebUI/extension behavior.

**Success Criteria:** Semantic remains off by default, disclosure is explicit, status is recoverable, graph queries include semantic state in their query identity, evidence is accessible, manual conversion uses canonical graph write/Sync paths, and responsive/accessibility/E2E checks pass.

**Tests:** Service/hook unit tests, component/a11y/i18n tests, WebUI and packaged extension E2E, API docs/OpenAPI checks, Ruff, Bandit, TypeScript, and focused regression suites.

**Status:** Not Started

### Task 11: Add Shared Semantic Client, State, And Setup UI

**Files:**
- Create: `apps/packages/ui/src/services/note-semantic-index.ts`
- Modify: `apps/packages/ui/src/services/note-graph-suggestions.ts`
- Create: `apps/packages/ui/src/components/Notes/hooks/useNotesSemanticIndex.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesGraphWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphInspector.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx`
- Modify: `apps/packages/ui/src/assets/locale/*/option.json`
- Modify: `apps/packages/ui/src/public/_locales/*/option.json`
- Test: `apps/packages/ui/src/services/tldw/__tests__/note-semantic-index.test.ts`
- Test: `apps/packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts`
- Test: `apps/packages/ui/src/components/Notes/__tests__/useNotesSemanticIndex.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.test.tsx`

- [ ] **Step 1: Write failing API client tests**

Assert exact nested routes, typed stable errors, idempotency/revision headers and bodies, 404 foreign-run handling, capability revision enablement, run polling/cancel, and no root Jobs request. Extend graph client types with semantic edge/status/evidence and send the complete edge type set plus threshold/top-k.

- [ ] **Step 2: Write failing hook/query-key tests**

Assert semantic is off by default; capability/status polling uses authority scope; active runs poll and terminal runs stop; offline mode preserves last-good ordinary graph; semantic enabled state, edge set, threshold, top-k, authority, focus, and filters enter the TanStack key; first-page semantic nodes/edges remain de-duplicated while later ordinary pages load.

- [ ] **Step 3: Write failing setup/status UI tests**

Cover Off, Preparing, Ready/Updating, Needs attention, unavailable, consent-required, stale-configuration, cleanup-pending, enable/rebuild/retry/cancel/delete confirmations, provider/model and boundary disclosure, active Note/estimate display, permission-disabled actions, and localized accessible status announcements. Do not expose arbitrary provider/model/endpoint controls.

- [ ] **Step 4: Run the red tests**

```bash
bun --cwd apps/packages/ui run test -- src/services/tldw/__tests__/note-semantic-index.test.ts src/services/tldw/__tests__/note-graph-suggestions.test.ts src/components/Notes/__tests__/useNotesSemanticIndex.test.tsx src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx src/components/Notes/__tests__/NotesGraphInspector.semantic.test.tsx
```

Expected: FAIL because semantic client/hook/UI do not exist.

- [ ] **Step 5: Implement shared client and inspector setup/status**

Use TanStack Query mutations and existing authority/offline patterns. Poll only active domain runs and invalidate both semantic status and graph keys on publication/tombstone. Keep capability disclosures next to the enable action, progress counts compact, errors actionable, and destructive deletion confirmed.

- [ ] **Step 6: Sync locale mirrors and run tests**

```bash
bun --cwd apps/extension run locales:sync
bun --cwd apps/packages/ui run test -- src/services/tldw/__tests__/note-semantic-index.test.ts src/services/tldw/__tests__/note-graph-suggestions.test.ts src/components/Notes/__tests__/useNotesSemanticIndex.test.tsx src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx src/components/Notes/__tests__/NotesGraphInspector.semantic.test.tsx
```

Expected: PASS with asset/public locale parity.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/services apps/packages/ui/src/components/Notes apps/packages/ui/src/assets/locale apps/packages/ui/src/public/_locales
git commit -m "feat: add Notes semantic index controls (TASK-13134)"
```

### Task 12: Add Semantic Graph Visuals, Evidence, And Manual Conversion

**Files:**
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphCanvas.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphRelationshipsView.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphInspector.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Notes/notes-manager-utils.ts`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphToolbar.i18n.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx`
- Create: `apps/tldw-frontend/e2e/workflows/notes-semantic-graph.spec.ts`
- Modify: `apps/extension/tests/e2e/notes-ux.spec.ts`

- [ ] **Step 1: Write failing filter and visual composition tests**

Assert a Similar content filter is off by default, uses a checkbox/toggle plus bounded top-k and threshold controls, and resets to first page when changed. Semantic edges have a distinct non-color-only style and passage-similarity label. Parallel manual/wikilink/semantic relationships group visually but retain all edge IDs/types for filtering and inspection.

- [ ] **Step 2: Write failing evidence/accessibility tests**

Cover qualitative band, numeric passage similarity, provider/model labels, source/target excerpts, truncation messaging, keyboard traversal, screen-reader relationship text, contrast, focus visibility, reduced motion, narrow viewport reflow, and long localized strings without overlap.

- [ ] **Step 3: Write failing manual-conversion tests**

From a semantic edge, invoke the existing canonical manual-link/Sync mutation with source/target Note IDs and its typed semantic generation conversion context. On success, invalidate the graph and show the manual edge as authoritative; do not mutate or accept semantic state. Require existing graph-write/Sync authority and disable the action without it.

- [ ] **Step 4: Run the red component tests**

```bash
bun --cwd apps/packages/ui run test -- src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx src/components/Notes/__tests__/NotesGraphToolbar.i18n.test.tsx src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx
```

Expected: FAIL until semantic visuals/interactions are implemented.

- [ ] **Step 5: Implement controls, canvas/row evidence, and conversion**

Use existing icon libraries and tooltips, stable toolbar dimensions, compact inspector typography, and existing responsive modes. Avoid adding a separate semantic page or duplicating canonical link mutation logic.

- [ ] **Step 6: Add WebUI and packaged-extension E2E**

Mock capabilities/status/runs/graph endpoints and verify enable disclosure, progress, Ready state, Similar content opt-in, first-page query, evidence inspection, conversion to manual link, degraded ordinary graph fallback, cancellation, and deletion. Run the shared WebUI workflow in desktop/mobile and the existing extension Notes fixture in Chromium.

```bash
bun --cwd apps/tldw-frontend run e2e:pw -- e2e/workflows/notes-semantic-graph.spec.ts --reporter=line
bun --cwd apps/extension run test:e2e -- tests/e2e/notes-ux.spec.ts --reporter=line --grep "semantic"
```

Expected: PASS.

- [ ] **Step 7: Run component tests and commit**

```bash
bun --cwd apps/packages/ui run test -- src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx src/components/Notes/__tests__/NotesGraphToolbar.i18n.test.tsx src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx
git add apps/packages/ui/src/components/Notes apps/tldw-frontend/e2e/workflows/notes-semantic-graph.spec.ts apps/extension/tests/e2e/notes-ux.spec.ts
git commit -m "feat: show semantic relationships in Notes Graph (TASK-13134)"
```

### Task 13: Document, Audit, And Verify The Integrated Feature

**Files:**
- Create: `Docs/API/Notes_Semantic_Index.md`
- Create: `Docs/User_Guides/WebUI_Extension/Notes_Semantic_Graph.md`
- Create: `Docs/Published/User_Guides/WebUI_Extension/Notes_Semantic_Graph.md`
- Modify: `Docs/API-related/API_Tags_Index.md`
- Modify: `Docs/Published/API-related/API_Tags_Index.md`
- Test: `tldw_Server_API/tests/Services/test_openapi_contracts.py`
- Modify: Backlog task `TASK-13134` through MCP/CLI only

- [ ] **Step 1: Write operator and user documentation**

Document explicit consent, provider/model and data/storage boundaries, ChromaDB/pgvector requirements, allowed dimensions, kill switch, worker/maintenance settings, nested routes, stable errors, cleanup/DSR behavior, ordinary backup-retention limits for deleted derived vectors, no-fallback policy, coverage semantics, degraded ordinary graph fallback, Similar content controls, passage-similarity interpretation, and manual conversion.

- [ ] **Step 2: Run backend focused suites**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_Graph tldw_Server_API/tests/Admin/test_admin_data_subject_requests_service.py tldw_Server_API/tests/Services/test_notes_semantic_workers.py tldw_Server_API/tests/Services/test_openapi_contracts.py tldw_Server_API/tests/AuthNZ_Unit/test_notes_graph_semantic_permissions.py tldw_Server_API/tests/AuthNZ/integration/test_notes_graph_semantic_permissions_postgres.py -q
```

Expected: PASS with only established unavailable-service skips.

- [ ] **Step 3: Run frontend type, unit, build, and E2E gates**

```bash
bun --cwd apps/packages/ui run test
bun --cwd apps/tldw-frontend run typecheck
bun --cwd apps/tldw-frontend run build
bun --cwd apps/extension run build
bun --cwd apps/tldw-frontend run e2e:pw -- e2e/workflows/notes-semantic-graph.spec.ts --reporter=line
bun --cwd apps/extension run test:e2e -- tests/e2e/notes-ux.spec.ts --reporter=line --grep "semantic"
```

Expected: PASS.

- [ ] **Step 4: Run lint, security, and diff gates**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m ruff check tldw_Server_API/app/core/Notes_Graph tldw_Server_API/app/core/DB_Management/chacha/note_semantic_models.py tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py tldw_Server_API/app/api/v1/endpoints/notes_graph.py tldw_Server_API/app/api/v1/endpoints/notes_semantic_index.py tldw_Server_API/app/services/notes_semantic_index_worker.py tldw_Server_API/app/services/notes_semantic_maintenance.py
python -m bandit -r tldw_Server_API/app/core/Notes_Graph tldw_Server_API/app/core/DB_Management/chacha/note_semantic_models.py tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py tldw_Server_API/app/api/v1/endpoints/notes_semantic_index.py tldw_Server_API/app/services/notes_semantic_index_worker.py tldw_Server_API/app/services/notes_semantic_maintenance.py -f json -o /tmp/bandit_task_13134.json
git diff --check
```

Expected: no new Ruff/Bandit findings and no whitespace errors.

- [ ] **Step 5: Perform final review and update Backlog**

Use `superpowers:requesting-code-review`. Resolve correctness, security, privacy, UX/accessibility, and over-engineering findings; rerun affected gates. Record test results, Bandit output path, touched files, commit hashes, and PR link in `TASK-13134` through Backlog MCP/CLI.

- [ ] **Step 6: Commit documentation and final integration fixes**

```bash
git add Docs/API/Notes_Semantic_Index.md Docs/User_Guides/WebUI_Extension/Notes_Semantic_Graph.md Docs/Published/User_Guides/WebUI_Extension/Notes_Semantic_Graph.md Docs/API-related/API_Tags_Index.md Docs/Published/API-related/API_Tags_Index.md
git commit -m "docs: document Notes semantic graph indexing (TASK-13134)"
```

## Final Acceptance Checklist

- [ ] Omitted graph edge types return exactly the legacy five types.
- [ ] No content transfer occurs before explicit consent and dimension resolution.
- [ ] No provider fallback or durable cross-run embedding cache is possible.
- [ ] Both vector backends are vector-only, owner isolated, cosine normalized, and cleanup-confirmable.
- [ ] Only complete, fenced, integrity-checked generations activate.
- [ ] Concurrent Note edits remain dirty and stale vectors cannot become authoritative.
- [ ] Semantic failure returns the ordinary graph with typed semantic status.
- [ ] Semantic evidence is current, bounded, field-relative, and response-byte capped.
- [ ] Query keys, request hashes, cache keys, and cursors bind semantic parameters and revisions.
- [ ] Notes/account erasure confirms semantic vector deletion before canonical Note deletion.
- [ ] Shared WebUI/extension setup, status, filters, evidence, and manual conversion pass accessibility and responsive checks.
- [ ] Backend/frontend/E2E, Ruff, Bandit, build, and diff gates pass.
- [ ] `TASK-13134` contains final verification and PR metadata.

## Execution Options

1. **Subagent-Driven (recommended):** Execute this plan in the current task with `superpowers:subagent-driven-development`, reviewing each focused task before the next commit.
2. **Inline Execution:** Execute sequentially in this task with `superpowers:executing-plans`, retaining the same red/green/commit checkpoints.
