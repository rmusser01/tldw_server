# Persona Visual Manifest V2 Contract Design

Tracks GitHub issue #1623 under the Persona/Buddy reliability epic #1510.

## Decision Summary

Define Manifest V2 as a contract and validation boundary for non-sprite Persona
Visual renderers before implementing Live2D, Rive, Lottie, or external pack
providers. V2 should not replace the current `sprite_frames` V1 path. V1 remains
the only activatable runtime renderer until a future renderer-specific slice
adds implementation, dependency gating, and tests.

The first Manifest V2 slice is design-only. It should establish:

1. renderer-specific asset roles,
2. static fallback requirements,
3. import-preview validation hooks,
4. renderer capability and setup-blocker states,
5. API/UI/MCP behavior for unsupported or feature-gated renderers,
6. security and portability rules for multi-file renderer bundles.

This unblocks a future Live2D spike and external MCP pack-provider contract
without accepting executable runtime plugins, silently activating packs, or
mutating the V1 sprite contract.

## Current Baseline

Persona Visual Packs are currently manifest-backed and user-owned. Assets are
attached to one persona by default, durable import/generation/MCP actions create
draft or reviewed candidates, and activation is explicit.

The current runtime stack is intentionally limited:

1. Backend manifest validation accepts `manifest_version: 1` for
   `renderer_type: "sprite_frames"`.
2. The renderer capability registry reports `sprite_frames` as the only
   activatable and Buddy-runtime renderer.
3. Sprite atlases stay inside V1 as `asset_role: "sprite_sheet"` with
   frame-level `region` rectangles.
4. Import preview reuses the backend manifest validator and therefore fails
   non-sprite manifests before commit.
5. Buddy chooses renderers through the local renderer registry and fails soft to
   text/static fallback when a pack is unavailable or unsupported.

Reserved labels such as `live2d`, `rive`, `lottie`, and `sprite_sheet` may
appear in schemas or capability planning, but they are not support claims until
backend validation, import preview, Buddy runtime rendering, and capability
reporting all agree.

## Product Invariants

Manifest V2 must preserve these invariants:

1. User-owned assets remain scoped to the owning user and persona unless a
   reviewed duplication/import flow creates a new draft.
2. No imported pack, generated candidate, library reuse, or MCP durable action
   activates a pack without explicit user activation.
3. Broken or unsupported packs cannot block Persona Live controls.
4. All non-sprite packs must provide a static fallback asset that Buddy can use
   without the renderer dependency.
5. Runtime renderers are product-owned adapters, not user-uploaded executable
   code.
6. External providers can propose draft packs, manifest patches, archive
   previews, or generated candidates; they cannot become live runtime plugins.
7. V1 `sprite_frames` manifests remain valid and unchanged.

## Manifest Version Boundary

Manifest V2 introduces a renderer contract envelope for non-sprite renderers:

```json
{
  "manifest_version": 2,
  "renderer_type": "live2d",
  "renderer_contract_version": 1,
  "display": {
    "preferred_width": 320,
    "preferred_height": 480,
    "anchor": "bottom_center"
  },
  "renderer_assets": {
    "source_manifest_asset_id": "asset-model3-json",
    "fallback_preview_asset_id": "asset-fallback-png"
  },
  "states": {
    "idle": { "animation_id": "idle" },
    "listening": { "animation_id": "listening" },
    "thinking": { "animation_id": "thinking" },
    "speaking": { "animation_id": "speaking" },
    "error": { "animation_id": "error" }
  },
  "animations": {
    "idle": {
      "renderer_action": {
        "motion_group": "Idle",
        "expression_id": "default"
      }
    }
  },
  "fallbacks": {
    "wake_armed": ["idle"],
    "tool_running": ["thinking"],
    "approval_needed": ["thinking"],
    "offline": ["idle"]
  },
  "metadata": {
    "authoring_tool": "Live2D Cubism",
    "license_notice_asset_id": "asset-license"
  }
}
```

Contract rules:

1. `manifest_version: 2` is required for non-sprite renderers.
2. `renderer_contract_version` identifies the project contract, not the
   upstream renderer SDK version.
3. `renderer_type` must be resolved through the backend renderer capability
   registry.
4. `renderer_assets.fallback_preview_asset_id` is required and must reference a
   bounded raster image.
5. V2 animation payloads are renderer-specific but must be JSON objects, bounded
   in size, and validated by the renderer adapter before activation.
6. States and fallback chains reuse the existing Persona visual state model.
7. V2 manifests must not contain remote URLs, absolute paths, path traversal
   paths, executable HTML, executable JavaScript, unreviewed SVG, or embedded
   data such as base64 payloads or Data URIs.

## Compatibility and Migration

V2 should be additive. There is no automatic migration from V1.

1. Existing V1 `sprite_frames` packs continue to validate, export, import,
   duplicate, save to library, and activate through the existing path.
2. Sprite atlas packs remain V1 because atlas support is just
   `sprite_frames` plus `asset_role: "sprite_sheet"` and frame `region`
   rectangles.
3. A V2 import creates a new draft pack only after preview review and commit.
4. A V2 generated candidate is accepted into an inactive draft or candidate
   pack; activation remains separate.
5. V2 export should preserve renderer metadata and asset-role mapping, but a
   later import on another instance still runs preview validation and capability
   checks.
6. If a server does not support a V2 renderer, it can still display metadata and
   fallback preview information, but it must not mark the pack activatable.

## Renderer Asset Roles

V2 should expand asset roles without changing ownership semantics. Roles are
renderer metadata, not storage permission boundaries.

Required common role categories:

1. `fallback_preview`: bounded raster image used whenever the renderer is
   unsupported, disabled, loading, or failed.
2. `source_manifest`: renderer-native descriptor file such as `.model3.json`,
   `.riv`, or a validated Lottie JSON file.
3. `license_notice`: optional user-visible license/readme file retained for
   portability and review.

These names are cross-renderer categories. A concrete asset row may use a
literal common role when the file is renderer-neutral, or a renderer-specific
role that declares it satisfies the common category. For example, a Live2D
`live2d_model_manifest` asset satisfies the `source_manifest` category rather
than requiring a second duplicate `source_manifest` asset row.

Live2D-oriented roles:

1. `live2d_model_manifest`
2. `live2d_moc`
3. `live2d_texture`
4. `live2d_motion`
5. `live2d_expression`
6. `live2d_physics`
7. `live2d_pose`
8. `live2d_userdata`

Other future renderer roles:

1. `rive_file`
2. `lottie_json`
3. `dotlottie_archive`
4. `renderer_preview`
5. `renderer_metadata`

The exact role list should live in the renderer capability registry so import
preview, export, MCP capabilities, and Persona Garden use one source of truth.

## Import Preview Validation Hooks

Non-sprite archive import must be preview-first and commit-second.

Preview flow:

1. Normalize archive paths and reject absolute paths, path traversal, duplicate
   member ambiguity, hidden executable payloads, remote references, and embedded
   data references.
2. Identify the candidate renderer by manifest `renderer_type` and
   `manifest_version`.
3. Ask the backend renderer registry for an import-preview validator.
4. Validate renderer-specific source-manifest references before creating asset
   rows.
5. Check file counts, total bytes, per-file bytes, texture dimensions, and
   runtime canvas bounds.
6. Require a static fallback raster asset.
7. Return a preview result with blockers, warnings, quota estimates, normalized
   asset-role mapping, and activation eligibility.
8. Commit only reviewed draft packs. Activation remains a separate explicit
   action.

Import preview statuses:

1. `supported`: renderer and validation dependencies are available.
2. `unsupported_renderer`: renderer id is known but not available in this
   server build.
3. `feature_gated`: renderer exists but is disabled by config or feature flag.
4. `dependency_missing`: renderer requires optional SDK/runtime assets that are
   not installed.
5. `license_review_required`: renderer requires user/admin acknowledgement
   before activation can be offered.
6. `invalid_archive`: archive shape or path safety failed before renderer
   validation.
7. `invalid_renderer_assets`: renderer-specific references, checksums, or roles
   failed validation.
8. `fallback_missing`: no valid static fallback asset is available.

The first implemented V2 preview seam is fixture-only:
`preview_renderer_import()` consumes normalized manifest and asset metadata and
returns renderer diagnostics without reading archives, persisting assets, or
activating packs. Archive parsing and commit remain separate future slices.

## Validation Stages

V2 needs different validation strictness at each lifecycle stage:

1. Draft save: permits incomplete non-activatable draft manifests when the
   renderer id is known or reserved, but stores clear diagnostics and never
   changes the active pack.
2. Import preview: validates archive shape, renderer asset references,
   fallback presence, quotas, and capability state before asset rows are
   committed.
3. Candidate accept: validates the proposed manifest and asset references using
   the same rules as reviewed import commit, then creates an inactive pack or
   draft candidate.
4. Activation: requires `can_activate`, `buddy_runtime_supported`, a valid
   static fallback, resolved required visual states, and renderer-specific
   activation validation. For V2 activation, `idle`, `listening`, `thinking`,
   `speaking`, and `error` must resolve either directly or through fallback
   chains. A renderer may support additional states, but it cannot activate with
   less than this baseline unless a future endpoint version explicitly changes
   the Persona visual state contract.
5. Runtime render: treats renderer load failure as a diagnostic and falls back
   to static/text Buddy instead of changing pack state.

## Renderer Capability Contract

The existing `GET /api/v1/persona/visual-renderers` contract should grow before
any V2 renderer is activatable. The current response fields are already
implemented and typed by the backend and WebUI, so V2 must extend them
additively. It must not silently rename existing fields; any rename would need a
versioned endpoint or explicit migration plan.

Existing fields to preserve:

1. `renderer_type`
2. `display_name`
3. `manifest_versions`
4. `can_validate`
5. `can_activate`
6. `buddy_runtime_supported`
7. `import_supported`
8. `export_supported`
9. `disabled_reason`

Additive V2 fields can include:

1. `renderer_contract_versions`
2. `supported_asset_roles`
3. `required_role_categories`
4. `role_category_map`
5. `allowed_mime_types`
6. `allowed_extensions`
7. `max_file_count`
8. `max_total_bytes`
9. `max_texture_width`
10. `max_texture_height`
11. `feature_flag`
12. `setup_status`
13. `setup_blockers`
14. `requires_static_fallback`
15. `requires_license_ack`

Capability status should be explicit. A renderer can be known but not usable.
Clients must treat `can_activate: false` and `buddy_runtime_supported: false` as
hard runtime boundaries.

## Buddy Runtime Behavior

Buddy must never attempt to execute renderer payloads directly from a pack.
Runtime rendering is adapter-owned:

1. Resolve the active pack.
2. Resolve the renderer capability.
3. If unsupported, disabled, blocked, or dependency-missing, render the static
   fallback if available.
4. If the static fallback is unavailable, render the existing text/static Buddy
   fallback.
5. If a renderer loads but fails at runtime, emit diagnostics and fall back
   without affecting Persona Live controls.
6. Runtime renderer adapters cannot load remote pack assets or arbitrary
   user-authored code.

This keeps non-sprite renderers optional and prevents model failures from
becoming live-assistant failures.

## MCP and Provider Boundary

MCP should continue to distinguish runtime state triggers from durable pack
changes.

For Manifest V2:

1. `persona_visuals.capabilities` should expose renderer support and setup
   blockers, not just runtime trigger support.
2. Durable MCP tools may propose draft packs, archive previews, generated
   candidates, or manifest patches.
3. Durable MCP tools must return `review_required: true` for V2 pack creation or
   modification.
4. MCP providers must not activate packs, bypass import preview, mutate active
   packs, or submit runtime code.
5. External MCP-compatible pack providers should target the same import-preview
   and generated-candidate contracts as local providers.

This makes external providers composable while keeping the server as the trust
boundary.

## Security Rules

Manifest V2 import and activation must enforce:

1. No remote URLs in manifests or renderer-specific files.
2. No executable JavaScript, HTML, or arbitrary web components in uploaded
   packs.
3. No embedded binary data in manifests, including base64 payloads and Data
   URIs. Binary assets must flow through asset storage and renderer validation.
4. SVG is rejected unless a future renderer-specific validator explicitly
   sanitizes and rasterizes it before storage.
5. Archive paths are normalized before validation; absolute paths, path
   traversal, duplicate entries, symlinks, and hardlinks are blockers.
6. Renderer source manifests are parsed with renderer-specific schemas, not
   string rewriting.
7. Every renderer-specific file reference must resolve to an archive member and
   then to a created asset row.
8. Checksums should be recorded for renderer-specific files before commit.
9. Renderer adapters must have bounded canvas dimensions and texture limits.
10. Optional dependency and license states must be surfaced before activation.
11. Exports must include enough metadata to reconstruct the draft pack, but
    import into another instance still requires preview and explicit activation.

## Live2D Implications

Manifest V2 does not implement Live2D. It unblocks a later Live2D spike by
making the required gates explicit:

1. `.model3.json` maps to a concrete `live2d_model_manifest` asset role that
   satisfies the common `source_manifest` category.
2. `.moc3`, textures, motions, expressions, physics, and pose files get
   renderer-specific asset roles.
3. Import preview can validate model-relative references before asset rows are
   created.
4. Capability reporting can represent license review, missing SDK/runtime
   pieces, disabled feature flags, and static fallback readiness.
5. Buddy can render a static fallback even if WebGL, SDK setup, or licensing
   gates block runtime rendering.

The first Live2D implementation PR should remain feature-gated, fixture-backed,
and opt-in.

## External MCP Provider Implications

Manifest V2 also unblocks an external provider contract without making
providers runtime plugins.

Allowed provider outputs:

1. generated candidate payload,
2. proposed manifest patch for an existing draft,
3. portable archive for import preview,
4. request to create a new inactive draft pack.

Disallowed provider outputs:

1. runtime JavaScript or renderer code,
2. active-pack mutation,
3. remote asset references,
4. direct asset writes outside reviewed import/generation paths,
5. persona-to-persona sharing without ownership checks.

## Implementation Slices

Recommended follow-up sequence:

1. Manifest V2 schema and renderer capability docs.
2. Backend renderer capability expansion for V2 statuses and asset-role limits.
3. Import-preview validator interface for renderer-specific metadata, with
   fixture-only tests and no new archive parser or runtime renderer.
4. Persona Garden preview UI copy for unsupported, feature-gated,
   dependency-missing, license-review, and fallback-missing V2 packs.
5. MCP capability update to expose renderer support and draft/import boundaries.
6. Feature-gated Live2D fixture spike using local bundled fixtures and static
   fallback.
7. External MCP pack-provider draft/import contract.

Each slice should be independently reviewable and should keep `sprite_frames`
behavior unchanged.

## Non-Goals

This design does not add:

1. a Live2D runtime,
2. a Rive/Lottie runtime,
3. a new activatable renderer,
4. an archive parser implementation,
5. an external MCP provider,
6. a marketplace or shared library,
7. automatic atlas generation,
8. automatic activation,
9. VN/CYOA behavior,
10. live response mutation.

## Acceptance Checklist

Before implementation starts, a Manifest V2 plan should answer:

1. Which renderer capability fields are added first?
2. Which backend schema owns renderer asset roles?
3. Which preview result fields represent blockers versus warnings?
4. Which static fallback asset is required for activation?
5. How does Persona Garden show unsupported or blocked renderers?
6. How do MCP tools expose renderer support without creating runtime plugins?
7. What fixture archives prove path normalization and missing-reference
   validation?
