# VN Script Authoring Catalog and Guided Draft Editing Design

Date: 2026-05-12

## Summary

Add an API-first VN script authoring catalog that lets custom frontends and the bundled WebUI discover supported script operations, inspect safe editing snippets, preview snippet insertion, and apply snippet edits to a draft through the backend.

This is the next sprint after starter templates and generated-choice support. The goal is not a new script language, a node editor, or model-written story authoring. The goal is a backend-owned contract that makes the existing JSON draft system easier to edit while preserving the current backend authority for validation, diagnostics, generation profiles, policy, manifests, and publish checks.

## Goals

- Expose preview-safe metadata for `vn_script_program.v1` operations and common guided snippets.
- Let clients preview a snippet against a script draft without saving it.
- Let clients apply a snippet to the stored draft using optimistic revision checks.
- Reuse existing draft validation and diagnostics so the authoring catalog does not become a parallel rule engine.
- Support custom frontends through stable schemas, endpoint discovery, error codes, and additive capability flags.
- Keep WebUI behavior catalog-driven rather than hardcoding VN script rules in React components.

## Non-Goals

- No visual node editor in this sprint.
- No new DSL or migration away from canonical JSON drafts.
- No LLM-generated story-writing endpoint.
- No direct provider/model/API-key fields in snippet parameters or catalog payloads.
- No new database tables unless implementation proves a small audit table is needed later.
- No publish-time bypass. Publish remains the only authority for a runnable script version.

## Existing Baseline

The current backend already has:

- `GET /api/v1/vn/vn-scripts/templates`
- `POST /api/v1/vn/vn-scripts/templates/{template_id}/scripts`
- draft read/write endpoints
- validation and diagnostics endpoints
- publish endpoints
- a pure script validator for `vn_script_program.v1`
- backend checks for asset manifests, audio references, policy, content rating, and generation profile constraints

The new catalog should sit above these pieces. It should describe what can be authored and provide safe draft patches, but it should not duplicate the validator's final authority.

## API Surface

All endpoints live under the existing scripts resource:

- `GET /api/v1/vn/vn-scripts/vn-authoring-catalog`
- `POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview`
- `POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply`

The VN capabilities endpoint should advertise the feature through an additive flag:

```json
{
  "features": {
    "script_authoring_catalog": true,
    "scripted_generation": true
  }
}
```

Clients must treat the feature as optional and fall back to raw JSON editing when the flag or routes are absent.

## Catalog Endpoint

`GET /api/v1/vn/vn-scripts/vn-authoring-catalog` returns preview-safe metadata only.

Example response shape:

```json
{
  "schema_version": "vn_script_authoring_catalog.v1",
  "program_schema_version": "vn_script_program.v1",
  "capability_tokens": [
    "script_authoring_catalog",
    "scripted_generation",
    "scripted_generation.output_schema.choice_set",
    "scripted_generation.output_schema.scene_update",
    "scripted_generation.user_confirmation"
  ],
  "operation_categories": [
    {"id": "story", "label": "Story"},
    {"id": "branching", "label": "Branching"},
    {"id": "visuals", "label": "Visuals"},
    {"id": "audio", "label": "Audio"},
    {"id": "generation", "label": "Generation"},
    {"id": "state", "label": "State"}
  ],
  "operations": [
    {
      "op": "narrate",
      "label": "Narration",
      "category": "story",
      "description": "Show narrator text.",
      "fields": [
        {
          "name": "text",
          "type": "string",
          "required": true,
          "multiline": true,
          "max_length": 8000
        }
      ],
      "supports_condition": true,
      "preview": {"op": "narrate", "text": "The scene opens."},
      "notes": ["Backend diagnostics remain authoritative."]
    },
    {
      "op": "generate",
      "label": "Generate",
      "category": "generation",
      "description": "Request structured model output through a resolved generation profile.",
      "fields": [
        {"name": "scope", "type": "enum", "required": true, "values": ["turn", "scene", "session"]},
        {"name": "output_schema", "type": "enum", "required": false, "values": ["narrative_dialogue", "choice_set", "scene_update"]},
        {"name": "max_choices", "type": "integer", "required": false, "minimum": 1},
        {"name": "requires_user_confirm", "type": "boolean", "required": false}
      ],
      "forbidden_fields": ["api_key", "api_provider", "base_url", "endpoint", "model", "provider", "provider_config"],
      "supports_condition": true,
      "preview": {"op": "generate", "scope": "turn", "max_choices": 2, "output_schema": "choice_set"},
      "notes": ["Generation limits are resolved from the script generation profile during preview, apply, validate, and publish."]
    }
  ],
  "snippets": [
    {
      "id": "generated_choice_set",
      "label": "Generated choice set",
      "category": "generation",
      "description": "Insert a generated choice request and a target handler label.",
      "required_capabilities": ["scripted_generation", "scripted_generation.output_schema.choice_set"],
      "parameters_schema": {
        "type": "object",
        "additionalProperties": false,
        "required": ["handler_label"],
        "properties": {
          "handler_label": {"type": "string", "pattern": "^[a-zA-Z0-9_.-]{1,64}$"},
          "max_choices": {"type": "integer", "minimum": 1},
          "requires_user_confirm": {"type": "boolean"}
        }
      },
      "default_parameters": {
        "handler_label": "generated_choice",
        "max_choices": 2,
        "requires_user_confirm": false
      },
      "anchor_modes": ["after", "before", "append"],
      "creates_labels": true,
      "preview": {
        "inserted_ops": ["generate"],
        "created_labels": ["generated_choice"]
      }
    }
  ],
  "limits": {
    "max_snippet_ops": 25,
    "max_created_labels_per_snippet": 8,
    "max_label_length": 64
  }
}
```

The operation catalog should include every opcode the backend validator recognizes: `choice`, `clear_visuals`, `end`, `generate`, `hide_sprite`, `increment`, `jump`, `label`, `narrate`, `play_bgm`, `play_sfx`, `random`, `return`, `say`, `set`, `set_background`, `show_cg`, `show_sprite`, `stop_bgm`, and `voice_cue`.

Field metadata is advisory and UI-oriented. The backend validator remains authoritative because operation metadata cannot fully express asset availability, generation-profile maps, audio ownership, content-rating policy, or reachability.

Catalog limits are global UI hints only. Script-specific generation limits, supported output schemas, and capability availability are resolved from `vn-capabilities`, the script's generation profile map, and backend validation during preview/apply. Catalog capability tokens must be canonical tokens owned by the backend, such as `scripted_generation` and `scripted_generation.output_schema.choice_set`; clients must not invent or reinterpret capability names. If a snippet lists a capability token that is absent from `vn-capabilities`, clients should hide or disable that snippet and still allow raw JSON editing.

Catalog entries should avoid listing stable validation codes in V1. Diagnostics returned from preview, apply, validate, and publish are the stable source of validation codes. If future catalog versions add validation-code hints, tests must compare those hints against diagnostics emitted by representative invalid programs.

## Snippet Catalog

V1 snippets should cover common JSON authoring tasks:

- `narration`: append narrator text.
- `dialogue`: append a character line.
- `authored_choice`: insert a choice with authored target labels.
- `generated_choice_set`: insert a `generate` op with `output_schema=choice_set`, `on_generated_choice=<handler_label>`, and a handler label body.
- `scene_update_generation`: insert a `generate` op with `output_schema=scene_update`.
- `confirm_gated_generation`: insert a confirmation-gated generation op plus cancel label.
- `set_background`: insert a background visual op.
- `show_sprite`: insert a sprite visual op.
- `play_bgm`: insert a background music op.
- `set_variable`: declare or set a variable.
- `ending`: append an `end` op.

Snippets must be deterministic. The same request against the same draft revision should produce the same patched draft and diagnostics.

The `generated_choice_set` snippet has an exact V1 patch contract:

- Insert `{"op": "generate", "scope": "turn", "output_schema": "choice_set", "max_choices": N, "on_generated_choice": handler_label}` at the requested anchor.
- Include `requires_user_confirm` only when requested.
- Create `labels[handler_label]` with deterministic starter operations such as one `narrate` placeholder followed by `end`.
- Fail with `snippet_label_conflict` when `handler_label` already exists unless a future request field explicitly allows reuse.
- Never set raw provider/model/routing fields.

## Snippet Preview

`POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview` computes a patched draft and diagnostics without saving.

Request:

```json
{
  "snippet_id": "generated_choice_set",
  "anchor": {
    "label": "start",
    "op_index": 1,
    "mode": "after"
  },
  "parameters": {
    "handler_label": "generated_choice",
    "max_choices": 2
  },
  "draft": null
}
```

Rules:

- The script must exist and belong to the caller even when a supplied draft is used.
- If `draft` is omitted or null, preview uses the currently stored draft.
- If `draft` is supplied, preview validates and patches the supplied draft without persisting it. This lets JSON editors preview against unsaved edits.
- Preview returns the patched draft, diagnostics, patch summary, and the stored draft revision it used when applicable.
- `base_revision` is the current stored draft revision when the preview starts. It is informational for supplied-draft previews and must not imply that the supplied draft matched the stored draft.
- Preview must not change `vn_script_drafts`, publish state, version history, manifests, or jobs.
- Preview must use the non-mutating validation helper path. It must not call a service method that stores diagnostics as a side effect.
- Preview validation context comes from the stored script metadata and resolved backend resources, not from client-supplied top-level metadata that would change ownership, policy, generation profiles, content rating, or asset pack selection.

Response:

```json
{
  "script_id": 12,
  "base_revision": 4,
  "snippet_id": "generated_choice_set",
  "draft": {
    "schema_version": "vn_script_program.v1"
  },
  "diagnostics": {
    "valid": true,
    "errors": [],
    "warnings": []
  },
  "patch_summary": {
    "inserted_ops": 1,
    "created_labels": ["generated_choice"],
    "changed_paths": ["$.labels.start[2]", "$.labels.generated_choice"]
  },
  "warnings": []
}
```

## Snippet Apply

`POST /api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply` applies a snippet to the stored draft.

Request:

```json
{
  "snippet_id": "authored_choice",
  "if_revision": 4,
  "anchor": {
    "label": "start",
    "op_index": 2,
    "mode": "before"
  },
  "parameters": {
    "choice_id": "first_branch",
    "choices": [
      {"id": "left", "text": "Take the left path.", "target_label": "left_path"},
      {"id": "right", "text": "Take the right path.", "target_label": "right_path"}
    ]
  }
}
```

Rules:

- `if_revision` is required.
- If the stored draft revision differs, return `409 draft_revision_conflict` with the current revision.
- The server patches the stored draft, runs the same validation/diagnostics path used by draft replacement, and persists the result through the existing atomic draft write path.
- Revision checking must happen in the persistence statement or an equivalent database transaction. A read-then-write check is not sufficient.
- Apply saves the patched draft with returned diagnostics, matching the existing whole-draft replacement behavior. Validation errors do not become transport failures unless patch construction itself fails.
- Apply must not silently publish or mark a script ready.
- Duplicate apply requests with an old `if_revision` fail with `409` instead of inserting the snippet twice.

Response:

```json
{
  "script_id": 12,
  "revision": 5,
  "snippet_id": "authored_choice",
  "draft": {
    "schema_version": "vn_script_program.v1"
  },
  "diagnostics": {
    "valid": true,
    "errors": [],
    "warnings": []
  },
  "patch_summary": {
    "inserted_ops": 1,
    "created_labels": ["left_path", "right_path"],
    "changed_paths": ["$.labels.start[2]", "$.labels.left_path", "$.labels.right_path"]
  }
}
```

## Patch Semantics

The server applies snippets to parsed draft objects, not JSON strings.

Anchors:

- `label`: required existing label name.
- `op_index`: optional zero-based operation index.
- `mode`: `before`, `after`, or `append`.

Insertion rules:

- `append` ignores `op_index` and appends to the label body.
- `before` and `after` require an `op_index` that exists in the label body.
- Snippets that create labels must fail on exact label collisions unless the snippet explicitly supports `reuse_existing_label`.
- Generated label names must be deterministic and sanitized.
- If a snippet declares variables, it must avoid overwriting existing variables unless the request explicitly allows reuse and the type matches.
- Snippets cannot remove existing operations in V1.
- Snippets cannot change script metadata, policy profile IDs, generation profile maps, primary asset pack IDs, content rating, or publish state.

## Validation and Authority

The catalog layer must call the same side-effect-free validation helper used by normal draft editing before persistence. Preview must stop there; apply may then persist through the draft write path. Validation must still resolve:

- approved asset slot keys from the manifest
- accessible audio media refs
- generation profiles and profile maps
- policy and content-rating constraints
- generated output schemas
- reachable labels and target labels
- variable declarations and assignment types
- forbidden raw generation routing keys

The catalog endpoint may expose the known operation names and basic field metadata, but it must not expose secrets, raw provider routing, prompt internals, policy implementation details, or private model configuration.

## Error Model

Use the existing VN error envelope style with stable codes. Error details must be concrete enough for custom frontends to recover without string parsing.

Expected codes:

- `script_not_found`
- `snippet_not_found`
- `snippet_parameter_invalid`
- `snippet_anchor_invalid`
- `snippet_anchor_not_found`
- `snippet_label_conflict`
- `snippet_variable_conflict`
- `snippet_preview_invalid_draft`
- `snippet_patch_validation_failed`
- `draft_revision_conflict`
- `permission_denied`
- `draft_not_found`

`snippet_patch_validation_failed` should be reserved for patch construction failures. Ordinary script validation diagnostics should be returned in `diagnostics` instead of converted into transport errors when the existing draft write behavior allows invalid drafts.

Expected transport statuses:

- `400`: invalid snippet parameters, invalid anchors, label or variable conflicts, malformed supplied draft.
- `403`: caller cannot access the script.
- `404`: script, draft, or snippet does not exist.
- `409`: stale `if_revision`.
- `500`: unexpected patch or persistence failures only.

Required error details:

- `draft_revision_conflict`: `{"current_revision": 5}`
- `snippet_parameter_invalid`: `{"field_path": "$.parameters.max_choices", "message": "Expected integer greater than or equal to 1."}`
- `snippet_anchor_invalid`: `{"anchor": {"label": "start", "op_index": -1, "mode": "after"}}`
- `snippet_anchor_not_found`: `{"anchor": {"label": "missing", "op_index": 0, "mode": "after"}}`
- `snippet_label_conflict`: `{"label": "generated_choice"}`
- `snippet_variable_conflict`: `{"variable": "route_score", "existing_type": "integer"}`

## WebUI Consumption Model

The WebUI should add a guided insert panel beside the existing JSON editor. The panel loads `vn-authoring-catalog`, groups snippets by category, collects snippet parameters, calls preview, then applies with `if_revision` when the user confirms.

The WebUI should not implement a second validator. It may use catalog field metadata for forms, but server diagnostics remain the source of truth. If catalog loading fails, the JSON editor remains available.

Conflict handling:

- On `draft_revision_conflict`, refetch the draft and show a non-destructive conflict message.
- On parameter errors, mark the affected form field using the server error details.
- On diagnostics errors, keep the patched draft visible in preview. The WebUI may offer apply/save only when the backend apply contract permits invalid drafts; publish remains blocked by backend validation.

## Custom Frontend Contract

Custom frontends should be able to:

1. Discover feature availability through `vn-capabilities`.
2. Load `vn-authoring-catalog`.
3. Render snippet forms from `parameters_schema` and `default_parameters`.
4. Preview a patch against a stored or unsaved draft.
5. Apply a patch with `if_revision`.
6. Render diagnostics returned by the backend.
7. Fall back to the canonical JSON draft endpoints.

The contract must remain additive. Adding snippets or optional metadata fields is a minor change. Removing snippets, renaming snippets, changing parameter semantics, or changing patch behavior requires a catalog schema version bump.

## Data Model

V1 should not require a migration. Catalog metadata can be static Python data beside existing VN script templates, and apply should persist through the existing draft storage path.

If auditability becomes necessary later, add a small optional history table in a future sprint. Do not block V1 on edit audit history because script draft revisions already preserve the current persisted state.

## Security and Abuse Controls

- Reject snippet parameters that include raw provider routing keys.
- Snippet request models must reject unknown fields. Pydantic request models should use `extra="forbid"`, and catalog JSON Schemas must set `additionalProperties: false` for every object node.
- Reject forbidden routing keys recursively in nested parameters, and enforce maximum nesting depth, string length, and total parameter payload size.
- Treat every snippet parameter object independently. Nested arrays of objects, such as authored-choice `choices`, must also forbid extra fields and recursive routing keys.
- Validate label names and variable names against strict patterns.
- Enforce maximum snippet size, maximum created labels, and maximum parameter payload size.
- Do not let snippets reference inaccessible audio or asset slots without normal diagnostics.
- Do not run model calls, generation jobs, file ingestion, or external network operations from preview/apply.
- Preserve existing AuthNZ owner checks for scripts and drafts.
- Do not log raw draft text at info level.

## Testing

Backend tests:

- Catalog returns every known validator opcode exactly once.
- Catalog omits forbidden routing fields and secrets.
- Catalog uses canonical backend capability tokens.
- Catalog schema forbids extra snippet parameter fields at every object node.
- Snippet preview against stored draft is non-mutating.
- Snippet preview against supplied draft does not persist.
- Supplied-draft preview still requires script ownership and reports informational `base_revision`.
- Snippet apply increments draft revision and returns diagnostics.
- Preview uses non-mutating validation and does not store diagnostics.
- Preview leaves stored `revision`, `draft_json`, and `diagnostics_json` unchanged for stored-draft and supplied-draft previews.
- Apply with stale `if_revision` returns `409 draft_revision_conflict`.
- Concurrent apply requests against the same revision result in one success and one `409`.
- Invalid anchors return stable errors.
- Error details include `current_revision`, `field_path`, anchor payloads, or conflict identifiers as applicable.
- Transport statuses match the expected error categories.
- Label and variable collisions return stable errors.
- Generated-choice snippet inserts `on_generated_choice` and creates the deterministic handler label body.
- Generated-choice snippet produces a draft that existing validation accepts when the generation profile permits it.
- Existing validation catches asset/audio/generation errors after snippet insertion.
- Raw provider/model/API-key parameters are rejected.

Frontend tests:

- API helper parses catalog, preview, and apply responses.
- Guided insert panel renders snippets from catalog metadata.
- Preview result updates the draft preview without saving.
- Apply updates revision and diagnostics.
- Conflict response triggers refetch-safe UI state.
- Catalog failure leaves raw JSON editing available.

Docs tests:

- `VN_PLATFORM_API.md` documents the new endpoints, feature flag, and custom frontend flow.
- OpenAPI schema includes request/response models for catalog, preview, apply, and errors.

## Rollout Plan

1. Backend catalog data and schemas.
2. Backend preview/apply service using parsed draft objects and existing validation.
3. API endpoints and capabilities flag.
4. Tests for catalog, preview, apply, conflicts, and safety.
5. WebUI API helper/types.
6. WebUI guided insert panel wired to catalog metadata.
7. Docs update and PR review sweep.

Each stage should be independently reviewable. The backend contract should land before WebUI polish so custom frontend support is not coupled to React implementation choices.

## Risks and Mitigations

- Risk: operation metadata drifts from validator behavior.
  - Mitigation: catalog tests compare operation IDs with the validator's known opcode list.
- Risk: snippet apply duplicates edits on retry.
  - Mitigation: require `if_revision` and return `409` for stale applies.
- Risk: custom frontends treat catalog field metadata as authoritative validation.
  - Mitigation: document metadata as advisory and always return backend diagnostics.
- Risk: frontend grows a parallel rule engine.
  - Mitigation: WebUI only renders forms and server diagnostics; preview/apply remain backend-owned.
- Risk: snippet patch code becomes string-based and fragile.
  - Mitigation: require parsed object mutation and changed-path summaries.
- Risk: V1 scope expands into node editing.
  - Mitigation: keep graph visualization and node editors as future work after the API contract proves stable.

## Decisions Deferred

- The first WebUI panel may render all backend snippets, but advanced snippets can be collapsed or grouped behind filters. The API should still expose all V1 snippets.
- V1 returns changed paths only. JSON Patch operations can be added later if external editors need them.
