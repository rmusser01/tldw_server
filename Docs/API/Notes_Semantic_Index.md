# Notes Semantic Index And Graph API

The Notes semantic index is an opt-in, owner- and dataset-scoped projection of
canonical Notes. It embeds deterministic title/body chunks, stores only opaque
vector IDs and vectors in ChromaDB or pgvector, and projects bounded `semantic`
relationships into the existing Notes Graph at query time.

Semantic relationships are derived evidence. They do not mutate Notes, are not
stored as manual links, and are hidden unless a client explicitly requests the
`semantic` edge type. A user can convert a current semantic relationship into
an ordinary manual link through the existing Notes link API.

## Route Ownership And Authentication

All semantic-index management routes are nested below the Notes Graph:

```text
/api/v1/notes/graph/semantic-index
```

There is no root-level semantic-index API and no new user-facing root Jobs API.
The nested `run_id` is the public operation identity; internal Job IDs are not
part of this contract.

Use `X-API-KEY` in single-user mode or `Authorization: Bearer <token>` in
multi-user mode. A token that declares scopes must include `notes`. Every route
accepts an optional `dataset_id` query parameter; omit it to resolve the
canonical personal Notes dataset.

| Permission | Operations |
| --- | --- |
| `notes.graph.read` | Read capabilities, status, runs, and semantic graph edges. |
| `notes.graph.semantic.manage` | Enable, renew consent, rebuild, retry, cancel, and delete the index. Graph read is also required. |
| `notes.graph.write` | Convert a verified semantic relationship into an ordinary manual link. |

The manage permission is the outbound-transfer and provider-cost gate. Clients
cannot submit a provider, model, endpoint, credential, collection, vector
backend, retry policy, or data-boundary declaration.

## Operator Configuration

The feature capability is available by default, but every Notes dataset starts
with indexing off. Configure the default embedding provider/model and a durable
credential through the normal server configuration before enabling a dataset.
Capability preflight fails closed when the endpoint origin, execution boundary,
storage boundary, provider health, credential durability, dimensions, cosine
contract, or vector-only contract cannot be established.

The runtime-specific controls are:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `NOTES_GRAPH_ENABLED` | `true` | Enables the owning Notes Graph route family. |
| `NOTES_SEMANTIC_INDEXING_ENABLED` | `true` when unset | Operator kill switch for enable/rebuild/retry admission, provider work, incremental indexing, and semantic graph queries. |
| `NOTES_SEMANTIC_VECTOR_BACKEND` | `chromadb` | Selects `chromadb` or `pgvector`. The client cannot override it. |
| `NOTES_SEMANTIC_EMBEDDING_DIMENSIONS` | unset | Declares an exact positive output dimension when the configured model is not in the server's known-dimension catalog. It must match actual provider output. |
| `NOTES_SEMANTIC_INDEX_WORKER_ENABLED` | disabled unless truthy | Starts the app-managed dedicated Jobs worker when Notes routes are enabled and the process is not in sidecar mode. |
| `NOTES_SEMANTIC_MAINTENANCE_ENABLED` | disabled unless truthy | Starts bounded recovery, dirty-work admission, failed-work retry, and physical cleanup. |
| `NOTES_SEMANTIC_WORKER_ID` | `notes-semantic-worker-<pid>` | Optional stable worker identity. |
| `NOTES_SEMANTIC_LEASE_SECONDS` | `180` | Dedicated Job/work lease duration. Supply a positive integer. |
| `NOTES_SEMANTIC_MAINTENANCE_INTERVAL_SECONDS` | `60` | Delay between maintenance passes, clamped to at least one second. Each pass has a shared 100-claim budget. |

Set both worker flags in a normal all-in-one deployment. Keep the worker and
maintenance services available while the kill switch is off if cancellation,
delete, cleanup, or data-subject erasure must complete. The kill switch does not
remove routes: status, cancellation, delete, and cleanup remain available.

The typed `SemanticIndexSettings` policy also applies hard server bounds for
per-Note size/chunks, per-run chunks/provider bytes/requests, retries, vector
query calls/candidates, and cleanup batches. These are server policy, not API
parameters.

## Vector Backend Requirements

Both backends are vector-only, cosine-only, owner/dataset/generation isolated,
and cleanup-confirmable. Raw Note text, excerpts, placeholder documents, and
metadata are not written to the vector backend. Finite dimensions and values
are validated, and zero-norm vectors are rejected before storage or query.

### ChromaDB

- Requires a real Chroma client with direct create/get/list/delete collection
  operations and direct vector upsert/get/query/delete operations. The in-memory
  compatibility client is not accepted for this feature.
- Uses one opaque generation collection created with
  `hnsw:space=cosine`. An existing non-cosine collection fails closed and is not
  rewritten.
- Stores only `ids` and `embeddings`.
- Accepts any resolved positive embedding dimension supported consistently by
  the configured Chroma installation.
- Generation deletion is successful only after collection absence is
  confirmed.

### pgvector

- Requires the PostgreSQL backend and pgvector extension version `0.8.0` or
  later. Capability initialization executes `CREATE EXTENSION IF NOT EXISTS
  vector`, so the database role must be able to use an installed extension or
  create it.
- Uses fixed dimension-specific tables in the initialized current schema,
  `vector(dim)` columns, HNSW `vector_cosine_ops` indexes, composite
  owner/dataset/generation/vector keys, and forced row-level security.
- This release allows dimensions `384`, `768`, `1024`, and `1536`. pgvector's
  HNSW `vector` ceiling is 2000 dimensions, but dimensions outside the fixed
  allowlist are intentionally unavailable; there is no `halfvec` substitution.
- Schema and index integrity are rechecked. A partial, non-RLS, wrong-dimension,
  or wrong-operator-class table fails closed.
- Generation cleanup deletes and confirms rows; it does not drop tables.

Changing vector backends for an existing index requires deleting the old index
and confirming cleanup before enabling the new backend.

## Consent And Data Boundaries

Start with the capability response. It discloses:

- active Note count and bounded work estimates;
- provider and model;
- sanitized embedding endpoint origin and local/external execution boundary;
- vector backend/storage label and local/external/unavailable boundary;
- the exact outbound categories: `note_title` and `note_content_chunks`;
- cosine metric, resolved dimensions or probe requirement, availability reason,
  management authority, and revision-bound consent identity.

Enabling sends the exact capability revision back to the server. No Note
content is read for provider execution or transferred before that explicit
consent is accepted. If the provider cannot declare dimensions, the server
sends one fixed non-user probe only after consent and resolves dimensions before
reading any Note text. A failed or unsupported probe prevents Note transfer.

Only normalized Note titles and body chunks are embedded. Attachments, OCR,
tags, linked sources, source excerpts, credentials, endpoint paths/queries, and
other Note metadata are excluded. Jobs payloads/results, logs, metrics, audit
details, retry records, and dead-letter records contain bounded identities,
counts, and stable codes only, never content or vectors.

The selected provider/model/endpoint/configuration is pinned. Provider fallback,
cross-origin redirect routing, and durable cross-run semantic embedding caches
are disabled. Bounded retries use only the same pinned configuration. A worker
may deduplicate within one bounded run in memory and must discard that state
when the run ends.

## Management Routes

### Read Capabilities

```http
GET /api/v1/notes/graph/semantic-index/capabilities
```

This is a content-free preflight; it does not invoke the embedding provider.

```json
{
  "active_note_count": 42,
  "estimated_chunk_count": 84,
  "estimated_run_count": 1,
  "provider_label": "openai",
  "model": "text-embedding-3-small",
  "endpoint_display": "https://api.openai.com",
  "execution_boundary": "external",
  "storage_boundary": "local",
  "storage_label": "chromadb",
  "outbound_data_categories": ["note_content_chunks", "note_title"],
  "capability_revision": "sha256:...",
  "indexing_available": true,
  "unavailable_reason": null,
  "metric": "cosine",
  "resolved_dimensions": 1536,
  "dimension_probe_required": false,
  "renewal_requires_delete": false,
  "manage_authorized": true
}
```

`execution_boundary` is `local` or `external`. `storage_boundary` is `local`,
`external`, or `unavailable`. Unknown or unsafe boundaries are represented as
unavailable and block enablement rather than being guessed.

### Read Status

```http
GET /api/v1/notes/graph/semantic-index
```

Primary states are `off`, `preparing`, `ready`, `updating`, and
`needs_attention`. `detail_reason` distinguishes building, degraded coverage,
stale configuration, consent requirements, cleanup, and unavailability. Status
also returns configuration/index revisions, active generation usability,
indexed/excluded/failed/pending counts, published chunks, cleanup state, and a
bounded active-run summary.

Coverage is measured current state, not a promise that every active Note has a
vector. Empty Notes and Notes over hard limits are excluded. Individual Note
failures reduce coverage while current successful Notes remain queryable.
Concurrent edits remain pending until their newer content version is indexed.
An empty eligible dataset can be Ready with zero indexed Notes; if eligible
Notes existed during an initial build, at least one must publish before that
generation can activate.

### Enable Or Renew Consent

```http
PUT /api/v1/notes/graph/semantic-index
Content-Type: application/json
Idempotency-Key: CLIENT_GENERATED_VALUE

{
  "expected_revision": 0,
  "capability_revision": "sha256:..."
}
```

The response is `202` with the current resource and nested run. Initial
enablement performs a fenced full build. A disclosure-only change requires
renewed consent before more provider calls; a compatibility change requires an
explicit rebuild. A backend change requires delete and confirmed cleanup first.

### Rebuild Or Retry Failed Notes

```http
POST /api/v1/notes/graph/semantic-index/runs
Content-Type: application/json
Idempotency-Key: CLIENT_GENERATED_VALUE

{
  "mode": "rebuild",
  "expected_revision": 4
}
```

Allowed modes are `rebuild` and `retry_failed`. Admission returns `202` and a
`SemanticRunResponse` with counts, revision, cleanup state, safe error code, and
its nested link.

### Read Or Cancel A Run

```http
GET /api/v1/notes/graph/semantic-index/runs/{run_id}
```

```http
POST /api/v1/notes/graph/semantic-index/runs/{run_id}/cancel
Content-Type: application/json
Idempotency-Key: CLIENT_GENERATED_VALUE

{"expected_revision": 2}
```

A missing or foreign run returns the same non-enumerating `404`. Cancellation
never activates a partial generation. Provider work already in flight may
finish or incur cost, but its unpublished vectors remain invisible and are
cleaned later.

### Disable And Delete The Index

```http
DELETE /api/v1/notes/graph/semantic-index
Content-Type: application/json
Idempotency-Key: CLIENT_GENERATED_VALUE

{"expected_revision": 4}
```

The response is `202`. Logical disablement blocks semantic reads immediately,
cancels/fences active publication, retires generations, and queues physical
vector cleanup. `cleanup_pending` remains true until backend deletion is
confirmed. Re-enabling always creates a new opaque generation and fresh build.

## Request Semantic Graph Edges

Semantic edges extend the existing graph endpoint:

```http
GET /api/v1/notes/graph?center_note_id=note:SOURCE&edge_types=manual,wikilink,backlink,tag_membership,source_membership,semantic&semantic_top_k=10&semantic_threshold=0.75
```

The omitted `edge_types` default remains exactly the legacy five types:
`manual`, `wikilink`, `backlink`, `tag_membership`, and `source_membership`.
Omission never enables semantic edges. To retain ordinary relationships while
requesting semantics, send the complete edge-type set as shown above.

`semantic_top_k` is an integer from 1 through the response's `max_top_k`
(server default 10, current hard default cap 50). `semantic_threshold` is a
finite value from 0 through 1 (server default 0.75). Controls are invalid unless
`semantic` is explicitly present. A focus Note is required, and semantic
retrieval runs only on the first page. Later cursors continue the ordinary
graph; clients retain and de-duplicate first-page semantic results.

The response includes `semantic_status` with fresh availability, state,
coverage counts, effective controls, hard evidence/admission caps, revisions,
and typed `truncated_by` reasons. Semantic edges are undirected and include
typed evidence:

```json
{
  "id": "semantic:...",
  "source": "note:SOURCE",
  "target": "note:TARGET",
  "type": "semantic",
  "directed": false,
  "weight": 0.87,
  "evidence": {
    "similarity": 0.87,
    "qualitative_band": "high",
    "source_note_id": "note:SOURCE",
    "target_note_id": "note:TARGET",
    "source_content_version": 3,
    "target_content_version": 8,
    "generation_id": "...",
    "semantic_index_revision": 12,
    "configuration_revision": 4,
    "normalization_version": "...",
    "chunker_version": "...",
    "provider_label": "openai",
    "model_label": "text-embedding-3-small",
    "model_revision": null,
    "excerpt_pairs": []
  }
}
```

Similarity is finite clamped `1 - cosine_distance` in `[0, 1]`. It is the
strongest current passage match between the focused Note and target Note, not
confidence, probability, whole-document agreement, factual correctness, or an
endorsement. Approximate backends may return different candidate sets.

Up to three excerpt pairs are reconstructed from current canonical title/body
fields using half-open Unicode code-point offsets. Each excerpt is at most 480
code points, each edge at most 2,880 excerpt code points, and the response has a
256 KiB evidence cap. When the cap is reached, the edge remains valid with
`evidence_omitted: "response_byte_cap"` and
`semantic_evidence_bytes` truncation.

Semantic failures degrade to the verified ordinary graph. Disabled, preparing,
stale, focus-required, unavailable, or vector-query failure states return HTTP
`200`, no unverified semantic edges, and typed semantic status. Provider
unavailability pauses new work but can leave a compatible active generation
readable. Manual and wikilink/backlink relationships always outrank semantic
admission; a manual link supersedes the semantic edge for the same Note pair.

## Convert To A Manual Link

Use the existing canonical manual-link route, not a semantic-specific route:

```http
POST /api/v1/notes/{source_note_id}/links
Content-Type: application/json

{
  "to_note_id": "note:TARGET",
  "directed": false,
  "weight": 1.0,
  "dataset_id": "optional-canonical-dataset-id",
  "idempotency_key": "CLIENT_GENERATED_VALUE",
  "semantic_conversion": {"generation_id": "ACTIVE_GENERATION_ID"}
}
```

The server revalidates the current owner/dataset, generation, Note pair,
content/manifests, and semantic relationship before using the existing
Sync-aware undirected link coordinator. Excerpts, vectors, similarity, and
model data are not copied into the manual link. An already-current manual link
returns the typed `notes_semantic_conversion_manual_link_exists` conflict so a
client can refresh idempotently.

## Stable Error Contract

Nested semantic-index errors use:

```json
{
  "detail": {
    "error_code": "notes_semantic_configuration_revision_conflict",
    "message": "The semantic index changed; refresh and retry."
  }
}
```

| HTTP | Stable conditions and codes |
| --- | --- |
| `403` | Required graph-read or semantic-manage permission is missing. |
| `404` | `notes_semantic_dataset_not_found`, `notes_semantic_run_not_found`, or a foreign owner/dataset resource. |
| `409` | `notes_semantic_capability_revision_conflict`, `notes_semantic_configuration_revision_conflict`, `notes_semantic_run_revision_conflict`, `notes_semantic_idempotency_conflict`, `notes_semantic_writer_conflict`, `notes_semantic_active_generation_required`, or `notes_semantic_backend_change_requires_delete`. |
| `422` | `notes_semantic_invalid_request`, including malformed/blank idempotency, invalid revisions, modes, or semantic controls. |
| `429` | `notes_semantic_quota_exceeded`. |
| `503` | `notes_semantic_jobs_unavailable`, `notes_semantic_provider_unavailable`, or `notes_semantic_dataset_authority_unavailable`. |

Graph cursor/conversion conflicts use the same typed detail shape. Relevant
codes include `notes_semantic_cursor_mismatch`,
`notes_semantic_conversion_generation_stale`,
`notes_semantic_conversion_pair_mismatch`,
`notes_semantic_conversion_owner_mismatch`, and
`notes_semantic_conversion_manual_link_exists`.

## Cleanup, Erasure, Backup, And Portability

Semantic vectors are sensitive derived Note data. Notes/account data-subject
erasure fences semantic Jobs, captures cleanup identities, deletes and confirms
every Chroma collection or pgvector row set, then removes semantic manifests
and canonical Notes. If the backend is unavailable or absence cannot be
confirmed, the Notes erasure category fails and retains an owner-scoped cleanup
ledger for retry. Selecting Notes for erasure is sufficient; a separate generic
embeddings category is not required.

Logical deletion and live-backend cleanup do not rewrite offline backups.
Deleted derived vectors can remain in ordinary backups until the deployment's
normal backup-retention period expires. Restore procedures must apply that
retention policy; the live API cannot promise immediate destruction of offline
copies.

Notes Sync, exports, full dataset restore, and server movement do not carry
semantic enablement, generations, manifests, or vectors. Individual Sync Note
mutations on an already-enabled local dataset only mark local semantic work
dirty. A restored or imported dataset starts with indexing off and requires a
new local disclosure and explicit enablement before provider transfer.

## Operational Checks

- **Setup unavailable:** verify the configured provider is enabled, uses a
  durable credential, has a sanitized known endpoint origin, and declares or
  can probe a supported dimension. Verify the selected vector backend.
- **Runs stay queued:** enable the dedicated worker and check the Jobs backend.
- **Edits stay pending or cleanup stays pending:** enable maintenance, keep the
  worker running, and inspect only stable codes/counters. Do not log content or
  backend credentials while debugging.
- **pgvector unavailable:** verify PostgreSQL, pgvector 0.8+, extension/schema
  privileges, one of the four allowed dimensions, fixed tables/indexes, and
  forced RLS.
- **ChromaDB unavailable:** verify a persistent real client and direct
  vector-only collection operations; remove or separately migrate conflicting
  non-cosine collections rather than allowing metadata rewrite.
- **Configuration or disclosure drift:** renew consent for boundary-only drift;
  rebuild for compatibility drift; delete first when changing vector backend.
- **Semantic query degraded:** use `semantic_status.detail_reason` and
  `truncated_by`. The ordinary graph remains authoritative and available.
