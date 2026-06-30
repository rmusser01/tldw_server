# VN Script Authoring Graph API Design

Date: 2026-05-14

## Summary

Add a backend-owned VN script authoring graph API that computes a stable outline and detailed graph from VN script drafts and published versions. This is the next backend-first authoring contract after starter templates and guided snippet editing.

V1 is intentionally static. It does not execute scripts, run model calls, simulate random branches, mutate drafts, persist graph snapshots, or build a visual node editor. Its job is to give custom frontends and the bundled WebUI a server-shaped view of script structure: labels, operation summaries, statically knowable edges, reachability, missing targets, and validation diagnostics.

## Goals

- Support stored drafts, supplied unsaved drafts, and published versions.
- Return both a simple outline layer and an advanced graph layer in one response.
- Keep graph construction tolerant so incomplete drafts return partial structure plus diagnostics.
- Keep validation and publish/runtime checks authoritative.
- Reuse or mirror the validator's reachability rules through tested helper boundaries.
- Give clients stable IDs, bracket JSON paths, compact operation summaries, content hashes, and graph semantics versions.
- Expose capability discovery through `features.script_authoring_graph`.
- Avoid frontend-owned VN traversal rules.

## Non-Goals

- No dry-run or playtest execution in this sprint.
- No model calls, generation jobs, or prompt construction.
- No runtime session creation or mutation.
- No graph persistence tables, cache invalidation, or migrations.
- No visual node editor.
- No text DSL.
- No full operation payload echoing in graph nodes.
- No policy engine replacement.

## API Surface

Add routes under the existing VN Scripts resource:

- `GET /api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph`
- `POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview`
- `GET /api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/graph`

The `GET` draft route computes from the stored draft. The `POST` preview route computes from a supplied draft without persistence. The version route computes from the immutable published version program and its pinned snapshots.

The VN capabilities endpoint adds:

```json
{
  "features": {
    "script_authoring_graph": true
  }
}
```

Clients must treat the feature as optional and fall back to existing JSON draft endpoints when absent.

## Response Shape

Every graph response uses the same envelope:

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
  "outline": {
    "entry_label": "start",
    "labels": []
  },
  "graph": {
    "nodes": [],
    "edges": []
  },
  "diagnostics": {
    "errors": [],
    "warnings": []
  },
  "validation_diagnostics": {
    "valid": false,
    "errors": [],
    "warnings": []
  }
}
```

`schema_version` covers response shape. `graph_semantics_version` covers edge extraction and reachability semantics. This allows future traversal-rule changes without pretending the JSON response shape changed.

`content_hash` is the SHA-256 of a canonical JSON serialization of:

- the source program only
- `program_schema_version`
- `graph_semantics_version`

It must not hash diagnostics wording or the full response. Clients can use it to detect whether a graph result still corresponds to the same source and semantics.

`validation_context_source` tells clients which metadata context was used for `validation_diagnostics`:

- `current_draft_context` for stored and supplied drafts.
- `published_version_snapshot` for published versions.

`truncated` is `true` when the server returns partial graph output because an internal graph limit was reached. Truncated responses must include a graph diagnostic explaining which limit was hit.

Array ordering is deterministic:

- `outline.labels` follows source label order with the entry label first when present.
- `graph.nodes` follows source label order, with each label node immediately followed by operation nodes in index order.
- `graph.edges` follows source operation order, then edge type order for multiple edges from the same operation.
- diagnostics are sorted by source path, severity, and code when they are not naturally emitted in source order.

## Source Modes

### Stored Draft

`GET /scripts/{script_id}/draft/graph` reads the stored draft and returns `source="stored_draft"` plus `base_revision`.

Graph calls must not persist diagnostics as a side effect. Stored draft graph responses should compute `validation_diagnostics` live through the same non-mutating validation helper path used by snippet preview, not reuse potentially stale stored draft diagnostics.

### Supplied Draft Preview

`POST /scripts/{script_id}/draft/graph-preview` accepts:

```json
{
  "draft": {
    "schema_version": "vn_script_program.v1"
  },
  "draft_revision": 4
}
```

Rules:

- The script must exist and belong to the caller.
- The supplied draft is parsed, bounded, graphed, and validated without persistence.
- `draft_revision` is optional metadata for client conflict awareness. V1 should not require it because graph preview is read-only, but the response should still include the current stored `base_revision`. If supplied and different from the stored draft revision, the response should still succeed and include a warning diagnostic such as `graph_preview_revision_stale`; it must not fail like snippet apply because no mutation occurs.
- The response uses `source="supplied_draft"`.
- Supplied draft size and complexity limits are enforced before graph construction.

### Published Version

`GET /scripts/{script_id}/versions/{version_id}/graph` reads the immutable version program and returns `source="published_version"` plus `version_id`.

Validation diagnostics for versions must use pinned version context where available: manifest snapshots, generation profile snapshots, policy profile snapshots, script defaults, content rating, and other published-version metadata. It must not silently use current mutable script settings when version snapshots exist. The response should set `validation_context_source="published_version_snapshot"` when this path is used.

## Outline Layer

The outline layer is optimized for simple UIs:

```json
{
  "entry_label": "start",
  "labels": [
    {
      "id": "label:start",
      "label": "start",
      "source_path": "$.labels['start']",
      "op_count": 4,
      "incoming_edge_count": 0,
      "outgoing_edge_count": 2,
      "reachable": true,
      "terminal": "unknown",
      "summary": "Narration, generated choice set, and 2 outgoing edges."
    }
  ]
}
```

`terminal` is conservative:

- `terminal` when the label mechanically ends with `end` and has no statically extracted outgoing edge after that point.
- `continues` when the graph can mechanically see outgoing control flow.
- `unknown` for labels involving unsupported dynamic semantics, malformed bodies, `return`, `random`, conditions, or ambiguous fallthrough.

V1 must not over-promise dead-end detection. It should report missing targets and malformed structures, but avoid claiming non-terminal dead ends except where mechanically certain.

## Graph Layer

The graph layer is optimized for advanced clients and future node editors:

```json
{
  "nodes": [
    {
      "id": "label:start",
      "type": "label",
      "label": "start",
      "source_path": "$.labels['start']",
      "reachable": true,
      "terminal": "unknown",
      "summary": "4 operations"
    },
    {
      "id": "op:start:1",
      "type": "operation",
      "label": "start",
      "op_index": 1,
      "op": "generate",
      "source_path": "$.labels['start'][1]",
      "summary": "Generate choice_set using profile default."
    }
  ],
  "edges": [
    {
      "id": "edge:op:start:1:on_generated_choice:label:generated_choice",
      "type": "generated_choice_handler",
      "source_id": "op:start:1",
      "target_id": "label:generated_choice",
      "source_path": "$.labels['start'][1].on_generated_choice",
      "target_label": "generated_choice",
      "metadata": {
        "output_schema": "choice_set"
      }
    }
  ]
}
```

Node and edge IDs must be deterministic and stable for the same source program and graph semantics:

- label nodes: `label:<encoded_label_name>`
- operation nodes: `op:<encoded_label_name>:<zero_based_index>`
- edge IDs: `edge:<source_id>:<edge_type>:<target_id_or_missing_key>`

IDs are API identifiers, not database IDs.

Label names in IDs must be encoded because labels can contain separators such as `:`, `/`, `.`, or spaces. V1 should use one explicit encoding helper everywhere, for example URL percent-encoding with uppercase hex. The raw label text remains available in the `label` field; clients must not decode IDs to recover display text.

Edges to missing labels should still be represented when useful for graph rendering:

```json
{
  "id": "edge:op:start:1:jump:missing:missing_target",
  "type": "jump",
  "source_id": "op:start:1",
  "target_id": null,
  "target_label": "missing_target",
  "source_path": "$.labels['start'][1].target",
  "missing_target": true
}
```

Missing-target edges must also emit graph diagnostics.

All source paths use bracket notation such as `$.labels['intro.scene'][0]`. Dot notation must not be used for label names because labels may contain dots, dashes, or other valid non-identifier characters.

Graph nodes contain compact summaries only. They must not echo full operation payloads.

## Edge Semantics

V1 extracts only statically knowable edges:

- `jump.target`
- authored `choice.choices[].target`
- `generate.on_generated_choice`
- `generate.on_cancel`
- terminal markers for `end`

The graph builder must not approximate runtime behavior for `random`, conditional `if`, variable-dependent control flow, model output content, or ambiguous fallthrough. Unsupported or dynamic structures can produce diagnostics or `unknown` terminal state.

Reachability must reuse or mirror the same edge families used by the current script validator's reachability behavior: `jump`, authored `choice`, `generate.on_cancel`, and `generate.on_generated_choice`. Tests must compare graph reachability with validator unreachable-label diagnostics for representative programs so the two do not drift.

## Diagnostics

The response has two diagnostic channels:

- `diagnostics`: graph-specific diagnostics.
- `validation_diagnostics`: the existing script validation result shape.

Graph diagnostics use stable code, severity, message, source path, and details:

```json
{
  "code": "graph_target_missing",
  "severity": "error",
  "message": "Target label was not found.",
  "path": "$.labels['start'][1].target",
  "details": {
    "target_label": "missing",
    "edge_type": "jump"
  }
}
```

Expected graph diagnostic codes include:

- `graph_labels_missing`
- `graph_label_body_invalid`
- `graph_opcode_invalid`
- `graph_target_missing`
- `graph_generated_choice_handler_missing`
- `graph_cancel_target_missing`
- `graph_label_unreachable`
- `graph_edge_limit_exceeded`
- `graph_node_limit_exceeded`
- `graph_preview_revision_stale`
- `graph_unsupported_dynamic_flow`

Existing validator diagnostics remain the authoritative source for publish/runtime validity. Graph diagnostics exist to help clients render structure and recover from malformed-but-parseable drafts.

Graph problems inside a parseable draft should return `200` with diagnostics. Transport errors are reserved for access, missing resources, malformed request bodies, and hard limits.

## Error Model

Expected transport errors:

- `404 script_not_found`
- `404 draft_not_found`
- `404 version_not_found`
- `403 permission_denied`
- `400 supplied_draft_invalid_shape`
- `413 supplied_draft_too_large`
- `422 request_validation_error` for Pydantic request-shape failures

Malformed graph-relevant script content should not become a transport failure if the server can parse the draft as JSON and stay within limits. It should produce partial graph output and graph diagnostics.

## Limits

V1 should define explicit conservative limits:

- `max_supplied_draft_bytes`: 1 MiB
- `max_labels`: 500
- `max_ops`: 5000
- `max_edges`: 10000
- `max_label_length`: align with current script validator/catalog rules
- `max_summary_length`: 240 characters

If supplied draft byte size exceeds `max_supplied_draft_bytes`, return `413 supplied_draft_too_large` before parsing. If label, operation, node, or edge limits are exceeded after parsing, return `200` with `truncated=true`, partial graph output, and `graph_node_limit_exceeded` or `graph_edge_limit_exceeded` diagnostics. Stored drafts and published versions should follow the same truncation behavior because a persisted large script should still be inspectable enough for recovery.

These values can become config later, but V1 should document them and expose them in responses.

## Security And Privacy

- Preserve existing script ownership checks.
- Do not expose raw prompt internals, provider routing, API keys, base URLs, or provider config.
- Do not echo full operation payloads.
- Do not log raw draft text at info level.
- Do not run model calls, file ingestion, jobs, or network operations.
- Do not mutate sessions, branches, drafts, versions, or diagnostics.
- Do not persist graph snapshots.
- Use existing VN error envelopes.

The graph builder may surface existing validator diagnostics for raw provider/model/API-key fields, but it must not become a separate policy engine.

## Backend Components

Suggested implementation boundaries:

- `tldw_Server_API/app/core/VN_Scripts/authoring_graph.py`
  - pure graph builder
  - graph diagnostics
  - stable ID/path helpers
  - canonical hash helper
  - limits
- `tldw_Server_API/app/core/VN_Scripts/service.py`
  - `get_draft_graph(script_id)`
  - `preview_draft_graph(script_id, draft, draft_revision=None)`
  - `get_version_graph(script_id, version_id)`
  - validation context resolution
- `tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py`
  - graph request/response schemas
- `tldw_Server_API/app/api/v1/endpoints/vn_scripts.py`
  - graph endpoints and error mapping
- `tldw_Server_API/app/core/VN_Platform/capabilities.py`
  - `features.script_authoring_graph`

The pure builder should accept parsed program mappings and never call the database. Service methods own script lookup, ownership, source selection, snapshot context, and non-mutating validation.

## Testing

Backend graph-builder tests:

- stored program returns outline and graph layers.
- stable node IDs and bracket JSON paths.
- encoded node IDs for labels containing separators or whitespace.
- deterministic outline, graph node, graph edge, and diagnostics ordering.
- authored choice edges.
- jump edges.
- generated choice handler edges.
- generation cancel edges.
- terminal `end` classification.
- conservative `unknown` classification for `return`, `random`, conditions, and malformed flow.
- missing target diagnostics.
- unreachable label diagnostics.
- graph reachability matches validator unreachable-label diagnostics for representative programs.
- malformed-but-parseable drafts return partial graph plus diagnostics for:
  - non-list label bodies
  - non-object operations
  - missing labels
  - invalid choice arrays
  - malformed generate handlers
- content hash is stable for canonical-equivalent programs and changes when program content or graph semantics version changes.
- graph output does not include raw provider routing secrets or full operation payloads.
- supplied draft limits reject oversized or overly complex drafts.
- stale supplied `draft_revision` returns a graph warning without failing the read-only preview.

Service and endpoint tests:

- `GET /draft/graph` returns stored draft graph and does not persist diagnostics.
- `POST /draft/graph-preview` accepts supplied draft and does not persist draft or diagnostics.
- `GET /versions/{version_id}/graph` uses published-version pinned context.
- missing script, draft, and version errors map to existing VN error envelopes.
- malformed supplied draft shape returns `400`.
- oversized supplied draft returns `413`.
- `vn-capabilities` includes `features.script_authoring_graph = true`.

Docs tests or verification:

- API docs list all graph routes.
- Examples show custom frontend flow and graph diagnostics behavior.
- Non-goals are documented to prevent clients from expecting execution or model calls.

## Rollout Plan

1. Write pure graph builder and focused tests.
2. Add service methods using non-mutating validation and published-version snapshot context.
3. Add Pydantic schemas and endpoints.
4. Add capability flag.
5. Update VN API docs.
6. Run focused VN script tests, compile checks, Bandit for touched Python scope, and `git diff --check`.

No WebUI changes should land in this sprint. A later sprint can consume the graph API in a read-only outline panel or visual editor.

## Risks And Mitigations

- Risk: graph reachability drifts from validator warnings.
  - Mitigation: extract shared edge/reachability helpers or add tests comparing graph and validator behavior.
- Risk: clients treat graph diagnostics as publish authority.
  - Mitigation: keep `validation_diagnostics` separate and document validator/publish authority clearly.
- Risk: terminal/dead-end semantics are wrong.
  - Mitigation: use conservative `terminal | continues | unknown` states and defer dry-run semantics.
- Risk: graph preview becomes expensive on large unsaved drafts.
  - Mitigation: enforce byte, label, op, and edge limits.
- Risk: published-version graph accidentally uses mutable current context.
  - Mitigation: service tests must assert pinned snapshot context is used where available.
- Risk: frontend reimplements traversal while waiting for UI work.
  - Mitigation: document graph as the custom frontend contract and include both outline and detailed graph layers.

## Deferred Decisions

- Dry-run/playtest route shape.
- JSON Patch output for graph deltas.
- Persisted graph snapshots for immutable versions.
- Visual node editor UX.
- Advisory authoring hints beyond graph and validation diagnostics.
