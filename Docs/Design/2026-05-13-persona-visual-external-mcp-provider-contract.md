# Persona Visual External MCP Provider Contract

Tracks GitHub issue #1682 under the Persona/Buddy reliability epic #1510.

## Decision Summary

External MCP-compatible Persona Visual pack providers should be treated as
review-input sources, not runtime plugins. A provider can offer generated
candidates, manifest patches, draft-pack requests, or portable pack archives,
but the tldw server remains the trust boundary for validation, storage,
preview, commit, and activation.

This contract is design-only. It does not add provider execution, Live2D
runtime support, a marketplace, shared libraries, or any automatic activation
path. The first implementation target after this contract should be an intake
adapter that converts provider output into the existing Persona Visual import
preview or generated-candidate review flows.

## Current Baseline

Persona Visual Packs already have the required local safety model:

1. Packs are user-owned and attached to one persona by default.
2. V1 activation is limited to `sprite_frames`.
3. Renderer support is reported by the renderer capability registry.
4. Manifest V2 design defines the non-sprite renderer envelope and static
   fallback requirement.
5. Archive import preview can report renderer blockers before commit.
6. Personal-library entries are reference-backed and do not store source
   display snapshots.
7. Durable MCP actions create drafts, manifest updates, generation jobs, or
   library draft copies; they do not activate packs.

External providers must reuse these boundaries instead of creating a parallel
asset model.

## Product Invariants

Every external provider integration must preserve these invariants:

1. Provider output is untrusted until the tldw server validates it.
2. Provider output cannot activate or replace an active pack.
3. Provider output cannot write assets directly to Persona Visual storage.
4. Provider output cannot become runtime renderer code.
5. Provider output cannot include remote asset references inside manifests or
   renderer-specific source files.
6. Imported assets remain user-owned and persona-scoped after commit.
7. Review and explicit activation stay separate user decisions.
8. Reference-backed personal-library semantics must not gain display snapshots
   through provider metadata.

## Provider Discovery

An external MCP server that wants to offer Persona Visual packs should advertise
a capability named `tldw.persona_visual_pack_provider.v1`.

Suggested discovery shape:

```json
{
  "capability": "tldw.persona_visual_pack_provider.v1",
  "provider": {
    "id": "local-sprite-pose-maker",
    "display_name": "Local Sprite Pose Maker",
    "version": "1.0.0",
    "homepage": null
  },
  "outputs": [
    "portable_archive",
    "generated_candidate",
    "manifest_patch",
    "draft_pack_request"
  ],
  "renderer_targets": [
    {
      "renderer_type": "sprite_frames",
      "manifest_versions": [1],
      "requires_static_fallback": false
    },
    {
      "renderer_type": "live2d",
      "manifest_versions": [2],
      "renderer_contract_versions": [1],
      "requires_static_fallback": true,
      "runtime_supported_by_provider": false
    }
  ],
  "visual_states": [
    "idle",
    "listening",
    "thinking",
    "speaking",
    "error"
  ],
  "limits": {
    "max_archive_bytes": 52428800,
    "max_asset_count": 128,
    "max_texture_pixels": 16777216
  },
  "review_required": true,
  "activation_allowed": false
}
```

The provider capability is descriptive. tldw must still compare advertised
renderer targets against its own renderer capability registry before preview,
commit, or activation. A provider advertising `live2d` does not make Live2D
supported on the target server.

## Provider Result Envelope

Provider results should use one common envelope so Persona Garden and future
intake adapters can show consistent provenance and diagnostics.

The backend intake boundary for this envelope starts at
`tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py`.
Call `normalize_provider_result_envelope()` before any follow-up resource
retrieval, asset write, import-preview enqueue, draft creation, or runtime
activation. The helper is review-only: it returns sanitized metadata and
machine-readable blockers or warnings, but it does not execute providers or
persist provider output.

```json
{
  "contract_version": 1,
  "result_type": "portable_archive",
  "review_required": true,
  "activation_allowed": false,
  "import_preview_required": true,
  "provider": {
    "id": "local-sprite-pose-maker",
    "display_name": "Local Sprite Pose Maker",
    "version": "1.0.0"
  },
  "pack": {
    "title": "Research Buddy Expressions",
    "renderer_type": "sprite_frames",
    "manifest_version": 1,
    "renderer_contract_version": null,
    "states_offered": ["idle", "thinking", "speaking", "error"],
    "static_fallback_available": true,
    "asset_count": 12,
    "total_bytes": 1843200
  },
  "diagnostics": {
    "status": "ready_for_import_preview",
    "blockers": [],
    "warnings": [
      {
        "code": "state_fallback",
        "message": "listening falls back to idle"
      }
    ]
  },
  "provenance": {
    "source": "mcp_provider",
    "provider_pack_id": "expr-pack-2026-05-13",
    "author": "local-user",
    "license_label": "user-provided"
  },
  "payload": {
    "archive": {
      "mcp_resource_uri": "mcp://local-sprite-pose-maker/resources/expr-pack-2026-05-13.tldw-persona-vpack",
      "sha256": "2f3a6c2c4b0b0c7f9f7ad3e2c0f9f95543e3013d6a45b69822ad0f01f54415be",
      "media_type": "application/vnd.tldw.persona.visual-pack+zip"
    }
  }
}
```

Rules:

1. `contract_version` is the tldw provider-contract version.
2. `result_type` must be one of `portable_archive`, `generated_candidate`,
   `manifest_patch`, or `draft_pack_request`.
3. `review_required` must be true for every durable provider result.
4. `activation_allowed` must be false. Activation is a tldw user action after
   server validation.
5. `import_preview_required` must be true for `portable_archive` results.
6. `mcp_resource_uri` is a retrieval handle for the intake step. It must not be
   copied into the Persona Visual manifest as a remote asset URL. Intake should
   reject URI schemes other than `mcp://` before any resource retrieval.
7. `provenance` is advisory metadata. It must not override user ownership,
   persona scope, validation results, or library source references.
8. `portable_archive` media types should use
   `application/vnd.tldw.persona.visual-pack+zip` for Persona Visual archives.
   Existing exports may still be served as `application/zip`, so intake
   implementations should treat both as zip archive payloads and rely on
   archive validation rather than media type alone.

## Allowed Result Types

### Portable Archive

Use this when the provider can produce a `.tldw-persona-vpack` archive.

Handoff:

1. tldw retrieves the MCP resource through an authenticated MCP client path.
2. tldw runs the existing archive import preview flow.
3. Preview records warnings, blockers, conflicts, quota estimates, renderer
   diagnostics, and proposed commit plan.
4. Commit creates or replaces a reviewed draft only when the preview is
   eligible and required choices are supplied.
5. Activation remains separate.

### Generated Candidate

Use this when the provider proposes new assets or manifest patches for an
existing draft pack.

Minimum shape:

```json
{
  "contract_version": 1,
  "result_type": "generated_candidate",
  "review_required": true,
  "activation_allowed": false,
  "pack": {
    "target_pack_id": "pack-draft-123",
    "target_state": "speaking",
    "renderer_type": "sprite_frames",
    "manifest_version": 1
  },
  "payload": {
    "assets": [
      {
        "provider_asset_id": "speaking-frame-1",
        "media_type": "image/png",
        "sha256": "6cc8f5c53f086c0cde01f76cf1efcc5d01f05c6b49d11a62c0dbff4d61d80b70",
        "byte_size": 204800,
        "mcp_resource_uri": "mcp://local-sprite-pose-maker/resources/speaking-frame-1.png"
      }
    ],
    "manifest_patch": {
      "animations": {
        "speaking_provider_candidate": {
          "frame_rate": 8,
          "frames": [
            {
              "asset_id": "provider_asset:speaking-frame-1",
              "duration_ms": 120
            }
          ]
        }
      },
      "states": {
        "speaking": {
          "animation_id": "speaking_provider_candidate"
        }
      }
    }
  }
}
```

The `provider_asset:` identifier is only an intake placeholder. The server must
replace it with real asset ids after validation and storage. Providers cannot
choose final database asset ids.

### Manifest Patch

Use this when the provider proposes JSON changes for an existing draft pack
without supplying new assets.

Rules:

1. The target pack must already be a draft owned by the current user.
2. The patch cannot target an active pack.
3. The patch must be reviewed through Persona Garden or an equivalent reviewed
   candidate flow.
4. Renderer-specific payloads must remain bounded JSON objects and must be
   validated before candidate acceptance or activation.

### Draft Pack Request

Use this when the provider can describe a pack but still needs the user to
review or fill missing pieces.

Rules:

1. The request can create an inactive draft only.
2. Missing assets, missing fallback, unsupported renderer, or licensing
   blockers must remain visible diagnostics.
3. Draft creation must not imply import-preview success or activation
   eligibility.

## Blocked Diagnostics Example

Providers should return explicit diagnostics when they already know a result is
not ready for import or review.

```json
{
  "contract_version": 1,
  "result_type": "portable_archive",
  "review_required": true,
  "activation_allowed": false,
  "import_preview_required": true,
  "pack": {
    "title": "Expressive Live2D Buddy",
    "renderer_type": "live2d",
    "manifest_version": 2,
    "renderer_contract_version": 1,
    "static_fallback_available": false
  },
  "diagnostics": {
    "status": "blocked_before_import_preview",
    "blockers": [
      {
        "code": "fallback_missing",
        "message": "Manifest V2 provider results require a bounded raster static fallback."
      }
    ],
    "warnings": [
      {
        "code": "runtime_not_claimed",
        "message": "The provider can produce a Live2D archive but does not provide runtime support."
      }
    ]
  },
  "payload": null
}
```

tldw may display this result for review, but it must not commit it until a
future intake step has valid payload data and the server-side import preview
passes.

## Safety Rules

Provider intake must reject or block:

1. runtime JavaScript, HTML, arbitrary web components, executable scripts, or
   dynamic renderer code.
2. remote URLs inside manifests or renderer source files.
3. base64/Data URI embedded binary payloads in manifests.
4. absolute paths, path traversal, duplicate archive members, symlinks,
   hardlinks, or ambiguous case-colliding archive paths.
5. asset writes outside the existing Persona Visual upload/import/generation
   storage paths.
6. provider-selected database ids for assets, packs, personas, users, jobs, or
   library items.
7. cross-user or cross-persona reuse without the existing reviewed duplicate,
   import, or library semantics.
8. source display snapshots in personal-library entries.
9. license or author metadata that attempts to grant support status, activation
   status, or ownership by assertion.
10. unsupported renderer claims that contradict the server renderer capability
    registry.
11. secrets, API keys, bearer tokens, session cookies, local filesystem paths,
    host identifiers, or other sensitive material in provider-supplied
    metadata, diagnostics, provenance, manifests, or archive member names.

Provider intake should require:

1. declared media type, byte size, checksum, and source handle for every
   provider-supplied asset.
2. renderer capability resolution before import preview or candidate review.
3. static fallback for non-sprite renderer proposals.
4. stable diagnostics with machine-readable blocker and warning codes.
5. sanitized provenance that is bounded, free of secrets and host-local
   identifiers, and stored as pack/asset metadata only after server review.

## Relationship To Existing MCP Tools

The internal `persona_visuals` module remains the tldw-owned Persona Visual MCP
surface. It exposes current user/persona-scoped actions such as capabilities,
library item listing, transient state triggers, draft creation, manifest update,
library reuse, and generation enqueue.

External pack providers are different:

1. They describe or supply candidate content.
2. They do not own Persona Visual storage.
3. They do not trigger runtime state.
4. They do not activate packs.
5. They do not replace `persona_visuals` authorization checks.

Future tldw intake tools can consume provider envelopes, but they should route
durable work through the existing import preview, generated candidate, or draft
review paths rather than trusting provider output directly.

## Implementation Slices

Recommended sequence:

1. Contract docs and examples: this slice.
2. Provider discovery/intake adapter: list provider offers and normalize result
   envelopes without persisting assets.
3. Portable archive intake: retrieve a provider archive resource and enqueue the
   existing import-preview job.
4. Generated-candidate intake: retrieve provider assets, validate metadata, and
   create a reviewed candidate for an existing draft.
5. Persona Garden provider review UI: show provider offers, diagnostics,
   provenance, and handoff actions.
6. Optional Live2D provider fixture only after the Live2D runtime spike has its
   own feature gate and fallback rules.

## Non-Goals

This contract does not add:

1. a new MCP server implementation,
2. provider execution inside tldw,
3. Live2D, Rive, Lottie, Spine, or other renderer runtime support,
4. automatic activation,
5. active-pack mutation,
6. direct asset writes,
7. shared marketplace behavior,
8. cross-user sharing,
9. VN/CYOA behavior,
10. live chat response mutation.
