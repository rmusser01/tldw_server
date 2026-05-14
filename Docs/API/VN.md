# VN API

The VN API is backend-owned and lives under `/api/v1/vn/vn-*` for custom
frontends. Existing compatibility routers may also expose `/api/v1/vn-play`,
but new clients should discover canonical paths from:

- `GET /api/v1/vn/vn-capabilities`

## Scripted Generation Runtime

Scripted VN play can pause at `generate` opcodes, ask the user to confirm or
cancel the pending generation, store generated revisions, and later regenerate
or activate a previous revision. All command endpoints require:

- `client_scene_version`: the scene version the client rendered.
- `idempotency_key`: a stable caller-generated key for safe retry.

Stale scene versions return `409 stale_scene_version`. Reusing an
idempotency key with different request payload returns
`409 idempotency_key_conflict`. Completed actions replay their stored response.

### Public Commands

- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generation-requests/{generation_request_id}/confirm`
- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generation-requests/{generation_request_id}/cancel`
- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/regenerate`
- `POST /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/activate`
- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations?limit=25&offset=0`
- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions?limit=25&offset=0`
- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}`

Public responses include stable `public_output`, applied visual summaries,
revision status, profile lineage, and pagination metadata. They do not include
raw prompts, raw model output, parser diagnostics, moderation diagnostics, or
provider payloads.

### Debug Detail

Debug detail is owner/admin-only:

- `GET /api/v1/vn/vn-play/sessions/{session_id}/script/generations/{generation_id}/revisions/{revision_id}/debug`

Moderation-blocked raw output is redacted by default. To reveal it, callers
must explicitly pass:

```text
include_blocked_raw=true&confirm=REVEAL_MODERATION_BLOCKED
```

Debug reads use owner/admin authorization and emit structured warning logs in
single-user or no-audit deployments.

### Example

```json
POST /api/v1/vn/vn-play/sessions/12/script/generation-requests/44/confirm
{
  "client_scene_version": 3,
  "idempotency_key": "session-12-confirm-44-v1"
}
```

```json
POST /api/v1/vn/vn-play/sessions/12/script/generations/7/regenerate
{
  "client_scene_version": 4,
  "idempotency_key": "session-12-generation-7-regenerate-1"
}
```

## Setup Metadata

`GET /api/v1/vn/vn-play/setup-options?mode=scripted_story` returns script
version options with generation metadata for custom frontends:

- `generation_profile_key`
- `generation_profile_snapshot_id`
- `generation_profile_snapshot_immutable`
- `provider_class`
- `max_automatic_generation_batch_count`
- `moderation_required`
- `estimated_cost_class`
- `supported_output_schemas`
- `dynamic_choice_support`
- `scene_update_support`
- `confirmation_required`

Setup warnings include missing or unavailable generation profile snapshots and
incompatible generated output schemas.

## Script Authoring Graph

Custom frontends can request a computed authoring graph for VN scripts when
`GET /api/v1/vn/vn-capabilities` returns:

```json
{
  "features": {
    "script_authoring_graph": true
  }
}
```

The graph API is read-only. It exposes backend-owned script structure without
executing the script, calling models, mutating drafts, persisting graph
snapshots, or providing a node-editor implementation.

### Endpoints

- `GET /api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph`
- `POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview`
- `GET /api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/graph`

The draft graph endpoint reads the stored draft and returns
`source: "stored_draft"` with the stored `base_revision`. It computes live
validation diagnostics but does not persist them.

The graph-preview endpoint accepts an unsaved draft and computes a graph
without saving it:

```json
{
  "draft_revision": 4,
  "draft": {
    "schema_version": "vn_script_program.v1",
    "entry_label": "start",
    "labels": {
      "start": [
        {"op": "narrate", "text": "Opening."},
        {"op": "end"}
      ]
    }
  }
}
```

Preview responses use `source: "supplied_draft"` and still include the current
stored `base_revision` for client conflict awareness. A stale
`draft_revision` can produce a graph warning, but preview remains read-only and
does not fail like draft mutation endpoints.

The version graph endpoint reads an immutable published script version and
returns `source: "published_version"` with `version_id`. Version validation
uses published-version snapshot context when available, reported as
`validation_context_source: "published_version_snapshot"`.

### Response Envelope

All graph endpoints return the same envelope:

```json
{
  "schema_version": "vn_script_authoring_graph.v1",
  "graph_semantics_version": "vn_script_authoring_graph_edges.v1",
  "program_schema_version": "vn_script_program.v1",
  "script_id": 12,
  "source": "stored_draft",
  "base_revision": 4,
  "version_id": null,
  "content_hash": "sha256:...",
  "validation_context_source": "current_draft_context",
  "truncated": false,
  "limits": {
    "max_labels": 500,
    "max_ops": 5000,
    "max_edges": 10000,
    "max_supplied_draft_bytes": 1048576
  },
  "outline": {"entry_label": "start", "labels": []},
  "graph": {"nodes": [], "edges": []},
  "diagnostics": {"errors": [], "warnings": []},
  "validation_diagnostics": {"valid": true, "errors": [], "warnings": []}
}
```

Key fields:

- `schema_version` identifies the response shape.
- `graph_semantics_version` identifies static edge and reachability rules.
- `source` is one of `stored_draft`, `supplied_draft`, or
  `published_version`.
- `content_hash` is a SHA-256 hash over the source program,
  `program_schema_version`, and `graph_semantics_version`; it does not hash
  diagnostics wording or the full response.
- `validation_context_source` is `current_draft_context` for stored and
  supplied drafts, or `published_version_snapshot` for version graphs.
- `truncated` is true when a graph limit produced partial output; the
  graph diagnostics explain which limit was reached.

### Outline And Graph Layers

`outline` is the compact layer for simple tree, sidebar, and label-list UIs.
Each outline label includes:

- `id`: stable API ID such as `label:start`.
- `label`: raw display label.
- `source_path`: bracket JSON path such as `$.labels['intro.scene']`.
- `op_count`, `incoming_edge_count`, and `outgoing_edge_count`.
- `reachable`: static reachability from `entry_label`.
- `terminal`: one of `terminal`, `continues`, or `unknown`.
- `summary`: compact text for display.

Reachability is derived from the emitted static graph edges. When a response is
truncated by node or edge limits, labels that would only be reachable through
omitted edges are reported as unreachable in that partial graph.

`graph` is the detailed layer for advanced clients. It contains label nodes,
operation nodes, and static edges. Node IDs are deterministic:

- Labels: `label:<percent-encoded-label>`.
- Operations: `op:<percent-encoded-label>:<zero-based-index>`.

Edge IDs are deterministic but opaque API identifiers. They usually include
the source node, edge type, and target or missing-target key, and they may also
include disambiguators such as a choice index when multiple static edges share
the same operation and target. Example:
`edge:op:start:0:choice:choice:1:label:end`. Clients must use the explicit
edge fields (`type`, `source_id`, `target_id`, `target_label`,
`missing_target`, and `metadata`) instead of parsing the edge ID.

Labels are percent-encoded in IDs because raw labels can contain separators
such as `.`, `:`, `/`, or spaces. Use the `label` field for display. Source
paths always use bracket notation so label names with dots or dashes are not
misread as object paths.

V1 extracts only statically knowable control-flow edges:

- `jump.target`
- `choice.choices[].target`
- `generate.on_generated_choice`
- `generate.on_cancel`

It does not infer random branches, conditionals, model-generated choices, or
runtime fallthrough. Missing targets remain visible as edges with
`target_id: null` and `missing_target: true`.

### Diagnostics

`diagnostics` are graph-construction diagnostics. They cover graph-specific
conditions such as missing edge targets, unreachable labels, invalid label
bodies, omitted edge targets, or output truncation. These diagnostics are
intended to help authoring tools render partial structure from incomplete
drafts.

When a label could rely on implicit fallthrough to the next label,
diagnostics include `graph_fallthrough_not_inferred` with the candidate
`next_label`. Treat this as a static-analysis limitation marker, not a
validation error.

`validation_diagnostics` come from the VN script validator and remain the
authoritative publish/runtime compatibility signal. A graph response can be
useful even when validation is invalid, and graph diagnostics are not a
replacement for validation diagnostics.

### Limits And Truncation

Graph output is bounded by:

- `max_labels`: 500
- `max_ops`: 5000
- `max_edges`: 10000
- `max_supplied_draft_bytes`: 1048576

If a graph limit is reached, the response sets `truncated: true`, returns the
partial graph, and includes a graph warning with the affected limit. Oversized
supplied drafts are rejected before graph construction.

### Custom Frontend Flow

1. Read `GET /api/v1/vn/vn-capabilities` and check
   `features.script_authoring_graph`.
2. Fetch `GET /api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph` for the
   saved draft, or call `draft/graph-preview` while the user edits unsaved JSON.
3. Compare `content_hash` and `graph_semantics_version` before reusing cached
   graph layout state.
4. Render `outline` for simple navigation and `graph` for advanced views.
5. Show `diagnostics` as graph-authoring hints and
   `validation_diagnostics` as validation status.
6. Use existing draft update and publish endpoints for mutations; graph
   endpoints never save or execute script content.

### WebUI Graph Inspector Flow

The WebUI script authoring surface consumes the same contract as custom
frontends:

- Show the graph inspector only when `features.script_authoring_graph` is true.
- Use the saved draft graph for the last persisted draft revision.
- Use graph preview for unsaved editor JSON so authors can inspect structure
  before saving.
- Use version graph from each published-version card for immutable release
  structure.
- Treat `content_hash`, `graph_semantics_version`, `base_revision`, and
  `version_id` as cache/staleness keys for rendered outline state.
- Render `outline` as the primary authoring aid and keep detailed `graph`
  nodes/edges available for richer custom clients.
- Keep graph diagnostics visually separate from validation diagnostics:
  graph diagnostics explain static-analysis limitations or partial structure,
  while validation diagnostics remain the publish/runtime readiness signal.
