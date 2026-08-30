# Notes Semantic Index and Graph Edges Design

**Task:** TASK-13134
**Status:** Approved
**Date:** 2026-08-29

## Summary

Add an opt-in, owner- and dataset-scoped semantic index for Notes. The index
embeds deterministic title/body chunks through the configured embedding
provider, stores vectors through a Notes-specific contract backed by the
configured ChromaDB or pgvector backend, and projects bounded semantic
relationships into the existing Notes Graph at query time.

Semantic relationships are derived evidence, not canonical links. They are
hidden by default, never mutate Notes automatically, and can become an ordinary
manual link only through an explicit user action. Raw Note content is not
stored in vector metadata, Jobs payloads, logs, metrics, or audit details.

The design extends the Notes-owned graph projection rather than adapting Notes
to media-ingestion IDs or treating the Notes library as a generic RAG
collection. It reuses existing embedding execution and backend primitives,
Jobs, Notes authority, Sync link coordination, and shared WebUI/extension graph
components without widening the document-bearing generic vector-store
interface.

## Goals

1. Let a user explicitly enable semantic indexing for one canonical Notes
   dataset after reviewing provider, model, storage, and data-boundary details.
2. Index deterministic title/body chunks and maintain them after Note create,
   edit, trash, restore, and delete operations.
3. Add `semantic` as an optional, bounded Notes Graph edge type without
   changing existing graph behavior when it is not requested.
4. Explain each semantic relationship with passage-level similarity, current
   matched excerpts, model provenance, and content versions.
5. Reuse the configured vector backend while preserving Notes ownership,
   tenant isolation, failure recovery, and SQLite/PostgreSQL behavior.
6. Keep graph reads available when semantic infrastructure is disabled,
   building, degraded, stale, or unavailable.
7. Give users an explicit path to create an ordinary manual link from a useful
   semantic relationship.

## Non-Goals

- Automatically accepting or persisting semantic relationships.
- Treating semantic relationships as LLM suggestions or adding them to the
  existing suggestion accept/reject queue.
- Embedding attachments, OCR text, linked source documents, tags, or source
  excerpts.
- Recursive semantic graph expansion or whole-library pairwise comparison.
- Saved semantic filters or graph layouts; TASK-13137 owns that work.
- Automatic background relationship/tag organization; TASK-13135 owns that
  work.
- Library-wide theme extraction; TASK-13136 owns that work.
- Synchronizing vectors, index enablement, or pinned provider settings between
  servers.
- Adding a new vector database or embedding provider.
- Supporting non-cosine semantic scoring in the first slice.

## Existing Constraints

- Notes and manual links are authoritative in ChaChaNotes and the Notes Sync
  coordinators.
- The graph service already owns bounded graph composition, revisions, cursors,
  caching, derived wikilink/tag/source edges, and Cytoscape formatting.
- The shared Notes Graph workspace already exposes edge filters, a responsive
  inspector, a canvas, and an accessible Relationships view in WebUI and the
  browser extension.
- The current embeddings Jobs adapter is media-specific. Notes must reuse
  provider and vector primitives without using synthetic media IDs or the media
  Redis stage contract.
- ChromaDB is the default configured vector backend; pgvector is optional
  through existing backend primitives when the Notes semantic capability
  checks pass.
- Jobs is the default queue for new user-visible background work.

## Approved Product Decisions

- Semantic indexing is off per dataset until a user enables it.
- Enablement performs an initial build, followed by automatic incremental
  maintenance for changed Notes.
- Notes are represented by deterministic title/body chunks.
- Semantic edges are computed at graph-query time and are not materialized.
- The configured vector backend is used through a dedicated Notes semantic
  facade: ChromaDB by default and pgvector when configured and capable.
- The effective embedding provider/model and vector contract are pinned at
  enablement.
- Similar content is a separate graph filter and is off by default.
- Similarity controls are session-only until saved graph views are implemented.
- A semantic edge can be converted explicitly into an ordinary undirected
  manual link.
- Attachments and linked source content are excluded.
- Semantic search expands only from the currently focused Note and never
  recurses through discovered neighbors.
- Turning the feature off disables reads immediately and deletes the index.
- Setup and management live in the Graph inspector.
- The server capability is available by default, but an operator kill switch
  can disable indexing and semantic queries. Dataset indexing still begins off.

## Architecture

### Ownership Boundaries

ChaChaNotes owns:

- dataset enablement and the pinned configuration identity;
- the active semantic generation;
- Note content fingerprints and dirty generations;
- deterministic chunk manifests and canonical evidence offsets;
- generation publication and cleanup state;
- semantic index revisions used by graph cache and cursor binding.

The configured vector backend owns only:

- opaque chunk vector IDs;
- embedding vectors;
- the cosine index required for bounded nearest-neighbor search.

Jobs owns:

- user-visible build, rebuild, retry, cancellation, and delete work;
- idempotency, root status, progress, terminal results, and sanitized errors;
- owner-scoped concurrency and retry admission.

The existing synchronous Notes Graph service remains responsible for ordinary
graph construction, revisions, cursors, and bounded candidate expansion. A
small async `SemanticGraphProjector` at the endpoint/application-service
boundary owns:

- semantic query validation and budgets;
- resolving the active semantic binding before ordinary graph construction and
  supplying it as an external cache/cursor binding;
- awaited vector fetch/query operations and their fail-open-to-ordinary-graph
  error boundary;
- current-authority and manifest validation of vector results;
- note-pair scoring, evidence reconstruction, graph admission, and truncation;
- deterministic merging through a pure graph-composition helper.

The synchronous graph service is not converted to async. A semantic backend
failure is caught by the projector and returns the verified ordinary graph with
typed semantic status rather than failing the graph request. The graph service
accepts semantic request fields for normalized query/cursor identity but
removes `semantic` from ordinary projection-readiness checks and ordinary edge
generation.

Sync continues to own only canonical Notes and manual links. Semantic index
configuration and vectors are server-local projections.

### Minimal Persistence Model

The first slice adds five logical persistence concepts. SQLite may omit a
redundant physical owner column where the database is already per-user;
PostgreSQL includes owner keys and forced RLS on every shared table.

#### `note_semantic_index_configs`

One row per canonical owner/dataset:

- desired state (`enabled` or `disabled`);
- active generation ID, if any;
- monotonically increasing configuration and semantic index revisions;
- compatibility hash once dimensions are resolved, plus the disclosure hash
  and consent-bound capability revision while dimension resolution is pending;
- pinned provider/model, endpoint-origin revision, sanitized endpoint-origin
  display (`scheme://host[:port]` only), data boundary, vector backend/storage
  boundary and sanitized storage label, cosine metric, dimension state
  (`pending` or `resolved`) and resolved dimensions, normalization version, and
  chunker version;
- enable, disable, consent, and update timestamps.

Secrets, raw endpoint query strings, and credentials are never stored.

#### `note_semantic_generations`

One row per server-generated opaque generation:

- generation ID and owning dataset;
- state (`staging`, `active`, `retired`, `failed`, or `deleting`);
- pinned compatibility hash when resolved, dimension state, and root Job UUID;
- expected and published Note/chunk counts;
- manifest hash and publication timestamps;
- bounded sanitized terminal error code, if any.

This row is the Notes-side publication ledger. A separate semantic run-history
or operation-receipt table is not added unless generic Jobs idempotency proves
unable to fence a concrete recreation race.

#### `note_semantic_note_state`

One row per Note in a generation:

- Note ID, current content version, and canonical content fingerprint;
- claimed dirty generation;
- state (`pending`, `indexed`, `excluded`, `failed`, `tombstoned`);
- chunk count, manifest hash, and bounded error code;
- publication timestamp.

#### `note_semantic_chunks`

One row per published chunk:

- opaque chunk/vector ID;
- generation, Note ID, content version, and chunk ordinal;
- canonical field (`title` or `content`), half-open Unicode code-point
  start/end offsets within that field, and chunk fingerprint;
- normalization/chunker versions.

The table stores no raw chunk text and no vector.

#### `note_semantic_work`

A coalescing owner/dataset work ledger:

- work kind (`index_note`, `delete_note_vectors`, or `delete_generation`);
- Note or generation identity;
- dirty generation and fencing token;
- claim state, attempt count, and next eligible time.

It handles Note dirtiness and physical vector cleanup without adding one Job per
chunk.

### Configuration Identity

The server computes two identities:

1. **Compatibility hash:** provider, model, resolved model revision/digest when
   the provider exposes one, vector backend, cosine metric, dimensions,
   normalization version, and chunker version. A change requires a new
   generation and explicit rebuild. The stable model string is the fallback
   identity when no revision/digest is available.
2. **Disclosure hash:** provider/model identity, endpoint-origin revision,
   embedding-execution boundary, vector-storage boundary, and outbound data
   categories. A boundary/origin change requires renewed consent before more
   content is transferred.

Credential rotation alone changes neither identity. Provider fallback is
disabled. Retries use only the pinned provider/model/configuration.

An incompatible global configuration makes semantic reads stale and prevents
incremental indexing until the user approves a rebuild. Disclosure-only drift
prevents new provider calls until renewed consent; already published, current
vectors remain readable with an explicit stale/update-paused status.

Sanitized endpoint/storage display values are persisted with the disclosure
revision so status and audit history can identify where data was sent even
after global configuration changes. They never contain credentials, URL user
info, paths, queries, fragments, database names, or filesystem paths.

### Notes Semantic Vector Contract

Notes uses a dedicated async facade over existing backend primitives. It does
not widen or pass placeholder documents through the generic RAG
`VectorStoreAdapter`, whose interface and results are document-bearing. The
facade must support:

- creating generation storage with cosine distance fixed before the first
  vector is written;
- vector-only upsert by opaque ID;
- fetching current vectors by opaque IDs;
- batched nearest-neighbor queries by precomputed vectors;
- deleting opaque IDs and complete generations;
- reporting backend/storage capability without exposing credentials.

The facade must not persist documents or raw Note text. If the configured
backend implementation cannot guarantee vector-only storage, semantic
capability preflight is unavailable and enablement is rejected.

Generation IDs are server-generated UUIDs mapped through owner/dataset Notes
state. Their opacity is not an authorization mechanism. The facade validates
the owner/dataset/generation mapping on every operation, applies backend
namespace isolation, and the graph projector still revalidates every result
against current ChaChaNotes authority.

The ChromaDB implementation creates a generation-specific collection with
`hnsw:space=cosine` in the initial `get_or_create_collection` call and writes
only `ids` plus `embeddings` through direct collection operations. It never
calls the shared `store_in_chroma` document path and never relies on changing
the metric after collection creation.

The pgvector implementation must not use the generic adapter's physical table
per collection. It uses a bounded set of dimension-specific semantic tables,
with table names derived only from an operator-allowlisted integer dimension.
Each table has a fixed `vector(dim)` column, a composite
owner/dataset/generation/vector-ID key, an ANN cosine index, and forced owner
RLS. Generation deletion removes rows rather than tables. An unsupported
dimension or an installation unable to enforce this schema reports semantic
capability unavailable. Metrics use low-cardinality backend and operation
labels, never generation, collection, owner, dataset, or dimension-table names.

Both implementations return raw cosine distance. The facade validates finite,
consistent dimensions and rejects zero-norm vectors before storage or query;
the graph layer alone computes finite clamped similarity as
`1 - cosine_distance`.

### Cross-Store Publication

ChaChaNotes and the vector backend cannot share a transaction. Publication is
therefore staged and fail closed:

1. Create a generation in `staging` with its dimension unresolved only when the
   selected provider cannot declare an exact output dimension.
2. After consent and before reading any Note content, resolve an unknown
   dimension with one fixed non-user probe string. Validate a finite, non-zero
   vector and compare-and-swap the discovered dimension and final compatibility
   hash into the generation/configuration. A failed probe fails the generation
   without transferring Note content.
3. Create generation storage with the pinned dimension and cosine metric.
4. Claim bounded Note dirty generations and read current Note content through
   owner/dataset authority.
5. Create deterministic chunks and vectors, then upsert them to staging
   generation storage.
6. Publish each Note manifest with compare-and-swap against its claimed dirty
   generation and content version.
7. Run a bounded convergence pass so edits made during the build remain dirty
   and cannot be cleared by an older claim.
8. Verify expected manifest counts, vector IDs, dimensions, compatibility
   hash, and the generation fencing token.
9. Atomically switch `active_generation_id`, record a publication receipt in
   the generation/configuration rows, and increment the semantic index revision
   in ChaChaNotes.
10. Return the publication receipt from the handler so Jobs can record terminal
   success, then retire the previous generation and queue bounded physical
   cleanup.

The handler cannot require its own terminal Jobs state before returning. The
Notes publication receipt is the activation authority; the later terminal Jobs
result is the user-visible confirmation and must reference the same receipt.

Cancelled, failed, incomplete, or unpublished generations are never queryable.
Vectors written before a crash but not represented by a published current
manifest are ignored and later swept.

Incremental updates write new deterministic vector IDs first, then publish the
new current Note manifest and increment the semantic index revision in one
Notes transaction. Tombstones do the same before cleanup. Old IDs remain
invisible once the manifest changes and are deleted asynchronously.

## Canonical Text and Chunking

Only the Note title and body are included. They are normalized independently
with the existing Notes Graph canonicalizer. Evidence ranges identify one
canonical field and use half-open Unicode code-point offsets, matching existing
graph suggestion evidence and Python string slicing.

- Reuse the canonical Notes Graph field normalization, fingerprint, and
  code-point offset rules so evidence reconstruction is consistent with graph
  suggestions. UTF-8 lengths remain authoritative only for fingerprints,
  provider byte limits, and storage budgets.
- The content fingerprint includes normalization version, normalized title,
  normalized body, and Note content version.
- Chunking is deterministic, non-LLM, and bound to a versioned configuration.
- A title-only Note produces a title span. For a Note with a body, chunks are
  spans of the canonical `content` field.
- The normalized title may prefix provider input for each body chunk, but the
  evidence span remains field `content` and identifies only that body range.
  The title prefix is context, not an additional evidence match.
- Empty/whitespace-only Notes are explicitly excluded.
- Per-Note bytes, chunks, provider input, batch bytes, and total-run work have
  hard server limits.
- A Note that exceeds its cap is excluded with actionable status; content is
  never silently truncated.
- If exact provider token counting is unavailable, the chunker uses a
  conservative UTF-8 byte limit from the validated model capability.

Chunk IDs are deterministic hashes of generation, Note ID, content
fingerprint, ordinal, canonical field, and offsets. The vector store receives
the opaque ID and vector only.

## Lifecycle and Jobs

### State Projection

Avoid one overloaded persisted state field. The API derives user-facing state
from:

- desired state (`enabled` or `disabled`);
- active/staging generation state;
- current coverage and dirty/failed work;
- active Jobs status;
- compatibility and disclosure checks;
- cleanup backlog.

The API maps these into four primary UI states: Off, Preparing, Ready/Updating,
and Needs attention. Typed detail reasons include building, degraded,
stale-configuration, consent-required, cleanup-pending, and unavailable.
Disabled with healthy cleanup remains Off with cleanup detail; stalled cleanup
becomes Needs attention.

### Enable and Initial Build

1. The client loads semantic capabilities and status.
2. Capability output discloses active Note count, estimated processing range,
   provider/model, embedding boundary, vector-storage boundary, limits, and a
   capability revision.
3. Enablement submits that revision and an idempotency key. The client does not
   submit arbitrary provider/model/endpoint/collection settings.
4. The server pins the effective configuration and creates one owner/dataset
   root Job with bounded internal batches.
5. Job payloads contain the opaque dataset/run/configuration identities only.
   Owner authority remains in the Jobs owner column, not user-controlled
   payload data.
6. Successful publication activates the staging generation only after every
   Note in the run's fenced snapshot is terminal, manifest/vector integrity
   passes, and no systemic provider, configuration, or vector-store failure
   occurred. Edits newer than that fence remain dirty and cannot be cleared by
   activation. Eligibility is determined before provider work from current,
   non-empty Notes within the per-Note input cap. If the snapshot contains
   eligible Notes, at least one must be indexed. Note-specific exclusions or
   failures may activate a degraded generation with explicit measured
   coverage; a dataset with no eligible Notes may activate as Ready with zero
   indexed Notes.

### Incremental Maintenance

- Note create, edit, and individual restore operations, including mutations
  received through Notes Sync on an already-enabled local dataset, update the
  semantic dirty ledger in the same ChaChaNotes transaction as the Note
  mutation.
- Note write paths never invoke Jobs, vector stores, or providers directly.
- A bounded service admits/coalesces Jobs by owner/dataset dirty watermark.
- Workers claim dirty generations, and compare-and-swap prevents an older
  claim from clearing a concurrent edit.
- One failed Note leaves other current Notes queryable and reduces reported
  coverage.
- Trash/delete tombstones authority and increments the semantic revision
  before physical vector deletion. Old results are immediately filtered.
- Restore creates fresh indexing work rather than reviving old vectors.

Sync never invokes a provider or vector backend inside its transaction. The
bounded maintenance service later admits indexing work. By contrast, importing
or restoring a complete Notes dataset on another server does not carry semantic
configuration or vectors and starts with semantic indexing disabled; no
provider work occurs until new local disclosure and enablement.

### Retry and Cancellation

- Only one build/rebuild writer is active per owner/dataset/configuration.
- Retries are bounded at batch level, idempotent, and pinned to the same
  provider/model/configuration.
- Provider fallback and arbitrary provider-internal retries are disabled for
  this workflow.
- Failed Notes can be retried without rebuilding current successful Notes.
- Cancellation never activates staging state. Partial vectors remain
  unpublished and are swept.
- Progress reports only scanned, indexed, excluded, failed, and pending
  Note/chunk counts plus sanitized error codes.

### Disable and Delete

Disable/delete:

1. verifies owner, dataset, permission, expected configuration revision, and
   idempotency;
2. changes desired state to disabled and blocks semantic reads immediately;
3. cancels active work and fences its publication token;
4. retires/tombstones all generations;
5. queues generation-specific physical cleanup;
6. reports cleanup pending until the vector backend confirms deletion.

Re-enablement creates a new opaque generation, so delayed cleanup cannot delete
new vectors. Re-enablement always performs a fresh build.

## API Design

All routes remain under the existing Notes Graph namespace and are registered
before parameterized Note routes.

### Capability and Management Routes

- `GET /api/v1/notes/graph/semantic-index/capabilities`
- `GET /api/v1/notes/graph/semantic-index`
- `PUT /api/v1/notes/graph/semantic-index`
- `DELETE /api/v1/notes/graph/semantic-index`
- `POST /api/v1/notes/graph/semantic-index/runs`
- `GET /api/v1/notes/graph/semantic-index/runs/{run_id}`
- `POST /api/v1/notes/graph/semantic-index/runs/{run_id}/cancel`

`PUT` and `DELETE` are asynchronous and return `202` with the semantic resource
state and an opaque domain `run_id`. Mutations carry `expected_revision` in the
typed body and require an `Idempotency-Key` header. Initial enablement binds the
capability revision, while operations on an existing resource bind its
configuration revision. Run creation uses an explicit allowlisted mode such as
`rebuild` or `retry_failed`.

The nested semantic run routes are the user-facing detailed status and
cancellation surface; no root-level user Jobs API is added. A run maps to an
internal root Job UUID, but the client needs only the domain `run_id`. Run reads
require graph-read authority and return `404` for a missing or foreign
owner/dataset run. Cancellation additionally requires
`notes.graph.semantic.manage`, matching owner/dataset authority, the run's
expected revision, and an idempotency key. The main semantic resource contains
a bounded active-run summary and link to its nested run route; run history
listing is deferred until a product workflow requires it.

### Graph Request Extension

`GET /api/v1/notes/graph` adds optional fields:

- `edge_types=semantic`;
- `semantic_top_k` within server-provided limits;
- `semantic_threshold` within server-provided limits.

Semantic controls are invalid unless `semantic` is requested. The first slice
uses cosine distance only. The public similarity score is the finite clamped
value `1 - cosine_distance` in `[0, 1]`; ChromaDB and pgvector facade contract
tests must prove equivalent normalization.

The response adds optional typed semantic status with:

- availability and user-facing/detail states;
- active generation/index/configuration revisions;
- current Note coverage, dirty/excluded/failed counts;
- effective threshold/top-k and hard limits;
- semantic-specific truncation reasons.

Existing clients may omit semantic fields and receive unchanged behavior. The
server's omitted `edge_types` default is an explicit frozen legacy set:
`manual`, `wikilink`, `backlink`, `tag_membership`, and `source_membership`.
It must never be derived from all `EdgeType` enum values. The shared client
sends the complete requested edge-type set, and semantic enabled state,
threshold, and top-k participate in the TanStack query key, server request
hash, cache key, and cursor binding.

### Query Algorithm

Semantic retrieval runs only when:

- `semantic` is requested;
- the dataset is enabled and has an active compatible generation;
- a current `center_note_id` exists;
- the caller has graph-read authority;
- the request is the first page (`cursor` is absent). Later pages continue only
  the ordinary graph cursor; the shared client retains and de-duplicates the
  first page's semantic nodes and edges.

The algorithm is:

1. Validate semantic controls and load fresh semantic resource state.
2. Build bounded ordinary graph candidates through the existing synchronous
   graph service. Internal candidate expansion may exceed the public response
   cap only by the hard semantic admission allowance and remains subject to a
   separate server cap.
3. Load the focused Note's current published chunk IDs from ChaChaNotes.
4. Fetch their vectors through the owner/dataset/generation-bound facade.
5. Batch-query nearest chunk vectors under a hard vector-call and candidate
   budget.
6. Exclude self matches and group candidate chunk matches by target Note.
7. Revalidate target owner, dataset, active/deleted state, content version,
   fingerprint, and published manifest.
8. Apply request tag, source, time, and other graph filters to target Notes.
9. Score each Note pair using its strongest current chunk match. Keep up to
   three deterministic matched chunk pairs as evidence.
10. Merge and prune through one deterministic precedence order: focused Note;
    direct manual-link endpoints and edges; wikilink/backlink relationships;
    semantic relationships up to their separate node/edge admission budgets;
    then tag/source membership. Semantic results may displace only lower-priority
    membership candidates, never the focus Note, manual relationships, or
    wikilink/backlink relationships. Unused semantic allowance returns to
    ordinary candidates so the response is not under-filled.
11. Apply threshold, top-k, stable Note-ID tie breaking, public node/edge/degree
    caps, and semantic query/admission caps. Report distinct typed truncation
    reasons for semantic candidates, nodes, edges, and evidence bytes.
12. Admit target Note nodes and one-hop semantic edges. Do not issue semantic
    queries from admitted neighbors.

The score is described as passage similarity, never confidence or probability.
Approximate vector backends may return different candidate sets; inputs,
configuration, validation, score normalization, and final tie ordering remain
revision-bound and reproducible. Cross-backend bit-identical ranking is not a
contract.

### Relationship Composition

- Semantic edges are typed, derived, and directed `false`.
- They contain a typed evidence object rather than arbitrary properties.
- An existing manual link supersedes the semantic edge for the same unordered
  Note pair.
- Wikilink/backlink and semantic relationships may coexist because they convey
  distinct evidence.
- The client groups parallel relationships for canvas display while preserving
  every underlying edge ID/type for filtering and inspection.

Semantic evidence contains:

- clamped similarity and qualitative band;
- source/target Note IDs and content versions;
- generation, configuration, normalization, and chunker revisions;
- sanitized provider/model labels;
- up to three current canonical source/target excerpt pairs.

Excerpt text is reconstructed at read time from current Note content using
fingerprint-bound field-relative Unicode code-point offsets. It is not copied
into chunk or vector records. Each excerpt is at most 480 code points; one edge
contains at most 2,880 excerpt code points across three pairs. A server-wide
hard 256 KiB response evidence-byte cap applies stable edge/evidence ordering,
omits later evidence when reached, and reports `semantic_evidence_bytes`
truncation without dropping the otherwise valid edge.

### Cache and Cursor Binding

When semantic edges are requested, graph cache and cursors bind:

- canonical dataset hash;
- authoritative graph revision;
- active semantic generation and semantic index revision;
- compatibility/configuration, normalization, and chunker revisions;
- focus Note, filters, threshold, top-k, and effective caps.

A stale or mismatched semantic cursor returns a typed conflict. Semantic
generation changes cannot reuse a prior cached graph. The projector reuses
`GraphCache` for the final first-page stable semantic projection with the full
binding above; the ordinary graph service may independently reuse its existing
ordinary-candidate cache. Only stable graph nodes/edges/evidence and immutable
effective configuration are cached. Current build/dirty/failed/cleanup
progress is projected after final-cache retrieval from fresh semantic state,
so the existing graph TTL cannot stale user-visible job status.
Request-specific vector failures are not written into the stable graph cache.

### Error Contract

- Missing/malformed idempotency keys, revisions, or invalid semantic
  parameters: `422` with a stable error code.
- Missing graph/manage/write permission: `403`.
- Stale capability/configuration/run revision, idempotency-key reuse with a
  different operation, an active conflicting writer, or stale/mismatched
  cursor: `409` with a stable typed conflict code.
- Missing or foreign semantic run: `404` without disclosing foreign existence.
- Capability or required backend unavailable during enablement: `503` with a
  sanitized stable reason.
- Disabled, building without an active generation, compatibility-stale, vector
  unavailable, or focus-required semantic state: ordinary graph `200` plus
  typed semantic status and no unverified semantic edges.
- Provider unavailability pauses new indexing work but does not by itself
  suppress reads from a compatible active generation; the response reports the
  paused/degraded maintenance state alongside any verified semantic edges.
- Cross-owner/dataset or unpublished vector results: silently excluded from
  graph data, counted only in low-cardinality security metrics, and never
  returned.

The existing authoritative graph remains available during semantic failures.
Semantic requests must not cause existing wikilink projection readiness checks
to treat semantic state as wikilink projection state.

## Authorization

- Existing `notes.graph.read` permits viewing available semantic status and
  semantic edges.
- New `notes.graph.semantic.manage` permits enable, rebuild, retry, cancel, and
  delete operations.
- Existing `notes.graph.write` and Notes Sync authority are required to convert
  a semantic relationship into a manual link.
- Token scope guards continue to require Notes scope where present.
- Administrative bypass follows the existing verified-principal rules.

The manage permission is the explicit provider-cost/outbound-transfer gate.
Role seeding and PostgreSQL permission tests cover default and revoked roles.

## Graph Workspace UX

### Setup and Status

Before enablement, the edge menu shows an actionable Similar content row with
`Set up`, not a disabled checkbox. It opens the existing Graph inspector.

The primary setup view shows:

- active Notes included and estimated processing range;
- provider/model;
- embedding-execution destination and local/remote boundary;
- vector-storage destination and local/remote boundary;
- exact outbound categories: Note title and body chunks;
- the Enable action.

When the provider cannot declare an exact dimension before execution, setup
states that a fixed non-user probe will resolve it after consent and before any
Note text is read or transferred. A known unsupported dimension blocks the
Enable action.

Vector backend, dimensions, revisions, metric, and chunk counts live under
technical details. Unknown provider or storage boundaries block enablement.

After enablement, the toolbar retains a restrained semantic-index status
affordance even while the graph edge filter is off, because indexing continues
in the background.

Backend states map to four primary user states:

- Off;
- Preparing;
- Ready or Updating;
- Needs attention.

The inspector exposes typed detail, current coverage, progress counts, and one
relevant recovery action. Build/rebuild supports Cancel; failures support Retry
failed or Delete index. Closing WebUI or the extension does not cancel Jobs.
The client polls the nested semantic resource/run route while work is active;
it does not attempt to read the admin Jobs API.

### Semantic Controls

Once ready, Similar content becomes a normal graph edge filter and remains off
by default. Enabling it exposes:

- neighbors per focused Note;
- Minimum passage similarity as an accessible numeric value and slider;
- server-provided minimum, maximum, default, and Reset.

Without a focus Note, the workspace asks the user to focus one and does not run
an arbitrary all-library query. Controls remain session-local until TASK-13137.

Turning the graph filter off does not disable indexing. Disable/delete remains
a distinct management command with destructive confirmation and explicit
cleanup status.

### Canvas and Relationships View

Edge treatment is non-color-dependent:

- solid: authoritative/structural relationships;
- dotted: semantic relationships;
- dashed: provisional LLM suggestions.

A visible legend names the treatments. The client groups edges by unordered
Note pair, preserves each underlying edge ID/type, recalculates groups after
filters change, and allows the inspector to select an individual relationship
type.

The Relationships view exposes every relationship and semantic evidence with
equivalent filtering and actions. The canvas is never the only way to discover
or operate on a semantic relationship.

### Semantic Inspector

Selecting a semantic relationship shows:

- `Similar content`, not `AI suggestion`;
- a qualitative passage-similarity band;
- exact decimal similarity under details, never as confidence;
- up to three matched excerpt pairs;
- freshness/content versions;
- provider/model and semantic generation details;
- `Create manual link` when write authority is present.

Evidence is side-by-side where space permits and stacked on narrow screens.
Creating a manual link uses the existing undirected link mutation path with
weight `1.0`; existing-link conflicts reconcile as an idempotent refresh. The
semantic edge is visually superseded by the manual link. Excerpts/model data
are not copied into the manual link.

Readers without manage permission can view available semantic relationships
but do not see enable, rebuild, retry, cancel, or delete commands. Readers
without write authority do not see conversion controls.

All setup, progress, evidence, confirmation, and completion states preserve
focus, announce asynchronous changes, fit the existing responsive inspector,
avoid nested cards, and remain localized in the shared WebUI/extension package.

## Security and Privacy

Vectors are sensitive derived Note content.

### Execution Fencing

Immediately before loading Note content and again before publication, workers
revalidate:

- user existence and owner/dataset authority;
- semantic manage permission and enabled desired state;
- capability/disclosure revision;
- generation ID and fencing token;
- pinned provider/model and vector compatibility;
- provider endpoint policy and vector-only storage capability.

Clients cannot override providers, models, endpoints, vector collections,
retry policy, or data-boundary declarations.

### Data Handling

- Raw Note text, excerpts, vectors, provider responses, credentials, and
  endpoint query strings never enter Jobs payloads, logs, metrics, audit
  details, retry records, or DLQ records.
- The first slice has no durable cross-run semantic embedding cache. A worker
  may de-duplicate within one bounded batch/run in memory using content hashes
  scoped by owner and pinned configuration, and clears those vectors when the
  batch/run ends.
- Vector metadata contains no source documents.
- Provider exceptions are mapped centrally to stable error codes and
  allowlisted guidance; arbitrary response bodies and exception strings are
  not persisted.
- Audit records for manual conversion contain actor, Note IDs, semantic
  generation, and result only.

### Data Residency and Deletion

Capability disclosure separately describes embedding execution and vector
storage. Unknown boundaries block enablement.

Logical disablement and physical cleanup are reported separately. Delete
removes the live index and retries vector cleanup, but offline backups may
retain derived vectors until ordinary backup retention expires. Setup,
confirmation, and operator documentation state this limitation accurately.

Notes Sync, Notes exports, and server movement do not carry vectors or local
enablement. A restored dataset requires new explicit enablement and disclosure
before any provider transfer.

Notes and account erasure treat semantic state as part of Notes data, not as an
optional generic `embeddings` category. Erasure first fences semantic Jobs,
captures the owner/dataset/generation identities needed for cleanup, deletes
and confirms ChromaDB collections or pgvector rows, and only then removes
semantic manifests/configuration and hard-deletes Notes. If a configured vector
backend is unavailable or physical deletion cannot be confirmed, the Notes DSR
category fails with a retained owner-scoped cleanup ledger; it must not report
success or discard the identities required to retry. The generic embeddings
category may additionally clear unrelated RAG/media embeddings, but selecting
Notes alone is sufficient to remove semantic vectors. Account deletion follows
the same ordering and backend coverage.

### Resource Governance

The server enforces:

- owner-scoped build/rebuild concurrency;
- per-Note bytes/chunks and total-run work;
- provider batch count/bytes and bounded retry;
- graph semantic query rate, source-chunk, candidate, top-k, threshold, edge,
  node, and degree caps;
- cleanup attempts and backoff.

Operator settings are exposed through one typed Notes semantic settings object
and a small number of operational caps, not one environment variable per
internal loop parameter. For pgvector this object includes a bounded allowlist
of dimensions for which forced-RLS semantic tables and ANN indexes may be
created.

## Observability

Low-cardinality metrics cover:

- build/update/rebuild duration and terminal state;
- indexed, excluded, failed, dirty, and pending counts;
- coverage and stale-generation counts;
- vector query latency, candidate/filter/admission counts, and truncation;
- provider/vector failures by stable category;
- cleanup backlog, retries, and oldest pending age;
- capability and permission denials without owner/Note identifiers.

Audit events cover enable, consent renewal, rebuild, retry, cancel, disable,
delete request, physical cleanup completion, and manual-link conversion. Audit
events contain IDs and bounded state only, never content or vectors.

Existing embedding usage accounting records provider work through the pinned
provider/model identity without recording Note content.

## Migration and Rollout

### Database Migration

Use the next available ChaChaNotes schema version at implementation time. The
migration is additive and creates only database tables, indexes, constraints,
and PostgreSQL RLS policies. It must not:

- inspect or chunk Notes;
- contact a provider or vector store;
- create a vector collection;
- enqueue Jobs;
- enable any dataset.

Every existing dataset starts disabled. Extra additive tables remain harmless
to an older binary after rollback. Operators should disable/delete semantic
indexes before a prolonged downgrade; rollback does not destructively remove
vector generations.

### Feature Availability

The semantic capability is available by default while dataset indexing remains
off. An operator kill switch blocks:

- enablement and new rebuild/retry admission;
- provider calls and incremental indexing;
- semantic graph queries.

The kill switch does not remove the router. Status, cancellation, delete, and
cleanup operations remain available so an administrator can remove data during
an incident.

Capability setup reports unavailable when the configured provider/vector
backend cannot satisfy pinned, cosine, vector-only, known-boundary behavior.

### Documentation

Update:

- Notes Graph user documentation;
- nested Notes Graph API documentation and examples;
- embedding/vector operator configuration and compatibility guidance;
- permission/RBAC documentation;
- privacy, data-boundary, backup-retention, deletion, and provider-cost notes;
- troubleshooting for build, stale configuration, degraded coverage, and
  cleanup failures;
- Sync/export documentation identifying semantic indexes as local projections.

## Verification Strategy

### Backend Unit and Property Tests

Unit tests cover:

- canonical field normalization, Unicode code-point offsets, UTF-8 byte caps,
  fingerprints, and chunk IDs;
- deterministic chunking and hard limits;
- compatibility/disclosure identities and dimension enforcement;
- vector distance normalization, zero-norm rejection, and score grouping;
- state projection, initial activation gates, and stable error mapping;
- frozen legacy edge defaults plus semantic opt-in validation;
- graph admission, precedence, truncation, stable-data caching with fresh
  status, first-page semantic behavior, and cursor binding.

Focused property tests cover:

- arbitrary Unicode normalization/offset reconstruction;
- dirty-generation compare-and-swap under random lifecycle sequences;
- idempotent publication and disable/re-enable fencing;
- node/edge/degree/query-cap invariants.

### Backend Integration Tests

SQLite and PostgreSQL tests cover:

- additive migrations and rollback tolerance;
- PostgreSQL forced RLS and owner/dataset isolation;
- enable/build/update/rebuild/cancel/delete lifecycles;
- create/edit/trash/restore/delete and concurrent edits;
- Sync create/edit/individual-restore marks an enabled local dataset dirty
  without provider calls inside the Sync transaction;
- full dataset/export restore starts semantic-disabled and performs no provider
  work before renewed disclosure and enablement;
- permission and token-scope behavior;
- nested semantic run detail/cancel ownership, revision, and idempotency
  behavior without access to admin Jobs routes;
- Notes/account DSR erasure confirms semantic cleanup in each enabled vector
  backend and fails closed when cleanup cannot be confirmed;
- route ordering and OpenAPI contracts.

Notes semantic vector facade contract tests cover ChromaDB and pgvector where
available and prove:

- vector-only persistence with no source documents;
- owner/dataset/generation isolation;
- fetch/query/delete behavior and zero-norm rejection;
- cosine score normalization parity;
- Chroma cosine metadata is fixed at initial collection creation;
- pgvector uses forced-RLS dimension storage rather than per-generation tables;
- wrong-owner/forged metadata results cannot reach graph output.

Fault-injection tests focus on consistency boundaries:

- crash after vector upsert but before manifest publication;
- crash after manifest publication but before generation activation;
- concurrent Note edit after dirty claim;
- cancellation during provider/vector work;
- delayed old-generation cleanup after re-enablement.

Performance verification asserts bounded call/candidate/admission counts rather
than wall-clock timing and proves no whole-library pairwise scan occurs.

### Frontend and E2E Tests

Shared component tests cover the full permission and lifecycle state matrix,
semantic controls, relationship grouping/filtering, evidence, manual
conversion, focus restoration, localization, and destructive confirmation.
Query tests prove edge-type, semantic toggle, threshold, and top-k changes enter
the request and TanStack query key, while omitted semantic controls preserve the
legacy graph request.

Accessibility tests cover keyboard operation, announcements, labels,
non-color edge distinctions, narrow layouts, and Relationships-view parity.

WebUI and packaged-extension E2E cover:

1. enable -> build -> semantic query -> manual conversion;
2. server-side Job continuation across client close/refresh;
3. cancellation or failure recovery;
4. disable/delete and cleanup status.

Visual verification covers representative ready and degraded desktop/mobile
states, dense graphs, long excerpts, and parallel relationship types rather
than snapshotting every backend state.

### Repository Gates

- focused and integration pytest suites;
- focused shared-package Vitest and packaged-extension tests;
- Playwright WebUI/extension workflows;
- SQLite/PostgreSQL migration checks;
- OpenAPI generation/fingerprint check;
- route-order and shard-manifest guards;
- locale coverage/mirror checks;
- Ruff, ESLint, applicable TypeScript checks, Bandit, and `git diff --check`;
- no tracked generated artifacts outside repository policy.

## Failure and Recovery Summary

| Condition | Semantic behavior | Authoritative graph behavior | Recovery |
| --- | --- | --- | --- |
| Dataset disabled | No semantic reads or work | Available | Enable after disclosure |
| Initial build active | Status `Preparing`, no active generation | Available | Wait or cancel |
| Initial build systemic failure | Generation fails and is never activated | Available | Retry after provider/configuration/storage repair |
| One Note update fails | Exclude stale Note, degraded coverage | Available | Retry failed |
| Provider unavailable | No new work; current vectors may remain readable | Available | Retry after readiness |
| Compatibility drift | Semantic reads stale/unavailable | Available | Explicit rebuild |
| Disclosure drift | Updates paused; current vectors labeled stale where affected | Available | Renew consent |
| Vector query failure | No semantic edges, typed unavailable status | Available | Retry request/operator repair |
| Note trashed/deleted | Result immediately filtered | Available | Cleanup retries |
| Build cancelled/crashed | Staging generation unpublished | Available | Retry or delete |
| Cleanup failure | Semantic reads remain disabled for retired data | Available | Durable cleanup retry |
| Operator kill switch | No semantic queries/provider work | Available | Status/delete/cleanup remain available |

## Alternatives Rejected

### Generalize the media embeddings pipeline

Rejected because it would couple Notes to media IDs, Redis ingestion stages,
chunk/storage assumptions, and a much larger regression surface. Provider and
vector primitives are reused without reusing the media resource contract.

### Index Notes as a generic RAG collection

Rejected because generic retrieval does not own Note content-version
authority, graph revisions, explicit semantic edge evidence, disable/delete
semantics, or manual-link conversion.

### Materialize semantic edge rows

Rejected because any Note update or newly indexed Note can change nearest
neighbors for many other Notes. Query-time projection avoids a second stale
relationship lifecycle.

### Store vectors in ChaChaNotes

Rejected because SQLite would require brute-force scans and PostgreSQL would
need a separate implementation. The Notes semantic facade over the configured
vector backend provides the required scalable backend choices.

### Embed one vector per Note or an LLM summary

Rejected because long or multi-topic Notes would be poorly represented, while
LLM summaries add cost and reduce reproducibility. Deterministic chunks provide
bounded matched-excerpt evidence.

## Acceptance Criteria Mapping

1. Owner/dataset scoping, content/model versioning, and bounded Jobs indexing:
   Architecture, Minimal Persistence Model, Lifecycle and Jobs.
2. Complete Note lifecycle and tenant isolation: Incremental Maintenance,
   Disable and Delete, Security and Privacy.
3. Optional semantic edge API with top-k, threshold, caps, truncation, and
   graceful disable behavior: API Design and Query Algorithm.
4. Provenance/evidence and no implicit manual persistence: Relationship
   Composition, Semantic Inspector, Authorization.
5. SQLite/PostgreSQL, RBAC, Sync, recovery, performance, tests, docs, and
   Bandit: Security and Privacy, Migration and Rollout, Verification Strategy.
