# Persona Visual Renderer and Provider Adapter Evaluation

Tracks GitHub issue #1497 under the Persona/Buddy visual-pack epic #1449.

## Decision Summary

Do not implement Live2D or another non-sprite renderer as the next code slice.
The current Persona/Buddy system should first add a renderer/provider capability
contract that keeps V1 `sprite_frames` behavior unchanged, then pursue a
feature-flagged Live2D spike only after licensing, bundle validation, and
fallback rules are explicit.

Recommended sequencing:

1. Keep `sprite_frames` as the only activatable runtime renderer for V1.
2. Add a renderer registry/capabilities contract before accepting new
   `renderer_type` values in manifests or imports.
3. Add a sprite atlas/sprite-sheet performance extension inside the existing
   sprite family before adding skeletal/model renderers.
4. Run a Live2D adapter spike behind a feature flag, using local bundled assets
   only and preserving review-before-activation.
5. Treat external providers as draft-pack or generated-candidate producers, not
   as live renderer plugins.

This preserves the existing product invariants: assets are user-owned, packs are
attached to one persona by default, manifests are the portability boundary,
imports/generation/MCP durable changes stay review-first, and activation remains
explicit.

## Current Persona/Buddy Contract

The implementation is already renderer-named but not renderer-pluggable:

1. `tldw_Server_API/app/core/Persona/visuals.py` defines the supported visual
   states, validates manifest version `1`, and only accepts
   `renderer_type == "sprite_frames"` for activatable manifests.
2. `tldw_Server_API/app/core/Persona/visual_service.py` validates uploads as
   raster images, validates manifests during activation and candidate accept,
   and creates duplicated packs as inactive drafts.
3. `tldw_Server_API/app/core/Persona/visual_portability/preview.py` validates
   imported archives by reusing the same manifest validator, so non-sprite
   imports currently fail during preview.
4. `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
   renders state-resolved frame sequences or sprite-sheet regions and falls
   back to text when an animation, asset, or region is invalid.
5. `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
   gates runtime rendering on `visualPack.renderer_type === "sprite_frames"`.
6. `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts`
   resolves named Persona visual states independently of renderer format.

The TypeScript and Pydantic renderer unions already include reserved labels such
as `static_image`, `sprite_sheet`, and `live2d`, but those are not support
claims. Backend validation and Buddy runtime rendering are still sprite/frame
only.

## Evaluation Criteria

Any renderer or provider path must satisfy these constraints:

1. User-owned assets stay scoped to a user and attached to one persona by
   default.
2. Pack data remains manifest-backed and portable through archive export/import.
3. No import, generated candidate, library reuse, or MCP durable action silently
   activates a pack.
4. Runtime failure cannot block live assistant controls.
5. Untrusted packs cannot execute JavaScript, load remote URLs, escape archive
   paths, exceed bounded size/dimension limits, or leak assets across users.
6. Renderer dependencies must be optional, feature-gated where needed, and
   detectable by API/UI capabilities before users queue work.
7. Licensing must be compatible with a local/self-hosted open-source project and
   clear enough for users who import their own assets.

## Candidate Renderer Paths

### 1. Sprite Frames and Sprite Sheets

Status: viable, keep as baseline and first implementation target.

Fit:

1. Matches the current `sprite_frames` manifest, asset upload validation, Buddy
   renderer, import/export preview, generation review, library duplication, and
   MCP draft semantics.
2. Supports still poses, frame sequences, and sprite-sheet regions without new
   runtime dependencies.
3. Keeps generated providers simple: produce raster assets plus manifest patches.

Limits:

1. Large frame sequences can be heavy compared with skeletal/model runtimes.
2. Complex motion, eye tracking, and parameterized expression control require
   many frames or a future renderer.
3. Runtime interpolation is limited to frame timing and fallback chains.

Recommendation:

Extend the existing renderer family before adding new renderer engines:

1. Add optional `sprite_sheet`/atlas capability metadata only after backend
   validation and Buddy rendering agree on the support boundary.
2. Keep it within the same raster-safe validation model: bounded dimensions,
   byte size, region bounds, checksums, no executable content.
3. Treat this as the lowest-risk bridge between current sprite packs and richer
   renderer adapters.

### 2. Live2D Cubism Web

Status: viable future spike, not immediate implementation.

Fit:

1. Live2D is the strongest expressive-persona match among evaluated options:
   parameterized models, textures, motions, expressions, physics, pose files,
   and WebGL rendering align with 2D assistant/persona expectations.
2. The official Web SDK is designed for programmatic use of Cubism models, and
   Live2D model metadata is centered on a `.model3.json` file that references
   model and related assets such as `.moc3`, textures, motions, expressions,
   physics, and pose data.

Risks:

1. Licensing is not a normal MIT dependency path. Live2D states that business
   entities publishing content using the SDK need a release/publication license,
   while individuals and small-scale enterprises have exemptions except for
   expandable applications. The project needs a product/legal gate before any
   default-enabled Live2D support.
2. Cubism Core is not published on GitHub under the same terms as the framework;
   the official docs say it is included in the SDK package from Live2D.
3. Imported model bundles are multi-file and cross-referenced, so archive
   preview must validate every local reference and reject remote URLs, absolute
   paths, path traversal, duplicate member ambiguity, and missing checksums.
4. Runtime memory and GPU/WebGL behavior need explicit bounds and fallback UI.
5. Third-party wrappers can reduce integration effort but should not define the
   project contract. The contract should be compatible with the official SDK
   asset model first.

Minimum manifest shape for a spike:

```json
{
  "manifest_version": 2,
  "renderer_type": "live2d",
  "renderer_contract_version": 1,
  "renderer_assets": {
    "model3_json_asset_id": "asset-model-json",
    "fallback_preview_asset_id": "asset-preview"
  },
  "states": {
    "idle": {
      "animation_id": "idle"
    }
  },
  "animations": {
    "idle": {
      "motion_group": "Idle",
      "expression_id": "default"
    }
  },
  "fallbacks": {
    "thinking": ["idle"]
  }
}
```

Open design points before implementation:

1. Whether Live2D assets are stored as generic `PersonaVisualAsset` rows with
   expanded roles or as a renderer-specific asset manifest section.
2. Whether model-relative file references are remapped to asset ids during
   import, or retained only inside a validated local bundle directory.
3. Whether lip sync and look-at controls belong in the renderer adapter or a
   higher-level Persona Live state bridge.
4. How to surface license/setup status in generation readiness and renderer
   capabilities without blocking sprite/frame users.

Recommendation:

Pursue Live2D only after the registry/capability and archive validation work is
complete. The first Live2D PR should be an opt-in spike with mocked fixtures,
feature gating, no cloud dependency, no automatic activation, and a static
fallback frame for every pack.

### 3. Rive

Status: viable for simple interactive vector personas, defer for now.

Fit:

1. Rive's official web runtime is JavaScript/WASM and supports high-level
   animation/state-machine control as well as lower-level render-loop control.
2. Rive states its official runtimes are MIT-licensed for personal and
   commercial applications.
3. `.riv` files are compact and can map cleanly to named states.

Limits:

1. Rive is a general interactive animation format, not a persona model format.
   It does not directly align with Live2D-style model, texture, expression, and
   motion authoring.
2. The authoring toolchain is external to the current Persona Garden editor.
3. Runtime WASM loading, renderer choice, and asset loading need capability and
   fallback work similar to Live2D.

Recommendation:

Do not implement before Live2D evaluation unless the product goal shifts toward
lightweight vector mascots rather than model-backed persona avatars. Keep it as
a possible future `rive` renderer after the registry exists.

### 4. Lottie or dotLottie

Status: viable for decorative/state animations, not recommended for primary
persona model packs.

Fit:

1. `lottie-web` is MIT-licensed and renders After Effects/Bodymovin JSON with
   SVG, canvas, or HTML renderers.
2. Lottie is easy to use for authored idle/listening/thinking/speaking loops and
   may be useful for simple built-in examples.

Limits:

1. Lottie is not a good user-uploaded persona model boundary by itself. It is an
   animation playback format, not a structured character model format.
2. SVG/HTML rendering paths increase the review burden for untrusted user
   uploads. A future Lottie renderer would need strict JSON schema validation,
   renderer restrictions, remote asset rejection, and possibly canvas-only
   runtime policy.
3. Mapping Persona state to Lottie segments is weaker than the current manifest
   state/animation structure unless the project defines marker conventions.

Recommendation:

Reject as a near-term primary Persona/Buddy renderer. Consider later for bundled
starter assets or simple user-authored loops after renderer registry and import
validation are mature.

### 5. Spine

Status: reject for this effort.

Fit:

1. Spine is a mature 2D skeletal animation ecosystem with web runtimes.

Risks:

1. The official runtime license requires either terms tied to the Spine Editor
   license or each user of products containing the runtimes to have their own
   Spine Editor license unless distribution conditions are satisfied.
2. The licensing and authoring requirements are too heavy for the current
   self-hosted Persona/Buddy visual-pack goal.

Recommendation:

Do not pursue unless a future user base explicitly asks for Spine import and
the licensing implications are accepted.

### 6. Arbitrary HTML, SVG, JavaScript, or Web Components

Status: reject.

Fit:

1. Maximum flexibility for custom avatars and third-party widgets.

Risks:

1. Conflicts with the review-first, local, user-owned safety model because packs
   would become executable UI plugins.
2. Introduces XSS, CSS escape, remote network loading, CSP, extension sidepanel,
   and cross-user data exposure concerns.
3. Hard to preserve deterministic fallback and archive portability.

Recommendation:

Never accept arbitrary executable renderer packs as Persona Visual Packs. If a
future custom-renderer plugin system exists, it should be an admin-installed
extension, not a user-uploaded pack.

## Candidate Provider Paths

Providers should create or transform packs; they should not become runtime
renderers.

### Local Image Generation Provider

Status: viable, already aligned with current Jobs/candidate flow.

The current generation model is the right abstraction: a background job produces
raster assets and a proposed manifest patch, then the user accepts or rejects
the candidate. Extend this path for more states, pose prompts, sprite sheets, or
atlas output before adding renderer engines.

### External MCP-Compatible Pack Provider

Status: viable later, with strict review boundaries.

An MCP provider should return one of:

1. a proposed manifest patch for an existing draft pack,
2. a generated-candidate payload,
3. a portable pack archive for import preview,
4. a request to create a new draft pack.

It must never return executable renderer code, mutate the active pack, or bypass
Persona Garden review.

### Cloud Marketplace or Shared Library Provider

Status: out of scope for this issue.

Cloud/shared-provider semantics would need account, trust, moderation, quota,
license, and cross-user sharing decisions. Keep #1497 focused on local/self-
hosted renderer/provider feasibility.

## Required Extension Points

Add these before enabling non-sprite manifests:

1. Backend renderer registry:
   - renderer id, display name, supported manifest versions, supported asset
     roles, MIME types/extensions, size limits, activatable flag, feature flag,
     validation function, import/export support, and fallback requirements.
2. API capability endpoint or field:
   - expose available renderers and setup blockers to Persona Garden and MCP
     tools, similar to generation readiness diagnostics.
3. Manifest versioning:
   - keep V1 `sprite_frames` intact.
   - introduce `manifest_version: 2` or `renderer_contract_version` before
     accepting renderer-specific payloads.
4. Renderer asset metadata:
   - support model, texture, motion, expression, physics, pose, atlas, preview,
     fallback, license/readme, and source-manifest roles without relaxing
     ownership checks.
5. Import preview validation:
   - renderer-specific archive validation must run during preview and report
     warnings, blockers, quota estimates, fallback availability, and unsupported
     renderer status before commit.
6. Frontend renderer registry:
   - Buddy should choose a renderer by registry lookup and fall back cleanly
     when unsupported, missing, or feature-gated.
7. MCP capability contract:
   - `persona_visuals.capabilities` should distinguish runtime trigger support,
     draft creation support, generation provider readiness, and renderer support.

## Security and Portability Rules

All renderer/provider follow-ups should enforce:

1. No remote asset URLs inside manifests or renderer-specific files.
2. No executable JavaScript, HTML, or unreviewed SVG in user-uploaded packs.
3. Archive path normalization, duplicate-member rejection, size limits, and
   checksum validation before preview succeeds.
4. Renderer-specific file-reference validation before asset rows are created.
5. Per-renderer maximum file count, total bytes, texture dimensions, and runtime
   canvas bounds.
6. Static fallback asset required for every non-sprite pack.
7. Optional dependency failures must degrade to Buddy's existing text/static
   summary without breaking live controls.
8. Export archives must include enough renderer metadata to restore the same
   draft pack on another device, but activation must remain separate.

## Follow-Up Issue Slices

1. Renderer capability registry for Persona Visual Packs:
   define backend/frontend/MCP capability reporting while preserving
   `sprite_frames` as the only activatable renderer.
2. Sprite atlas/sprite-sheet V1.1:
   formalize atlas validation and Buddy rendering as a safe performance
   extension.
3. Non-sprite manifest V2 design:
   define renderer-specific asset roles, fallback requirements, and import
   preview validation hooks.
4. Live2D adapter spike:
   opt-in Web renderer proof of concept using local fixtures, explicit license
   gate, static fallback, and no automatic activation.
5. External MCP pack-provider contract:
   allow providers to submit draft packs, archive previews, or generated
   candidates without runtime code or active-pack mutation.

## Source Notes

1. Live2D Cubism SDK for Web docs describe the Web SDK as a development kit for
   programmatic Cubism model use and note that Cubism Core is included in the
   official SDK package rather than published on GitHub:
   https://docs.live2d.com/en/cubism-sdk-manual/cubism-sdk-for-web/
2. Live2D model docs describe `.model3.json` as the file that tracks model
   references and `.moc3` as the file containing model movement data:
   https://docs.live2d.com/en/cubism-sdk-manual/model-web/
3. Live2D SDK release licensing needs product review before release usage:
   https://www.live2d.com/en/sdk/license/
4. Rive official runtime docs describe the web runtime as JavaScript/WASM and
   state the official runtimes are MIT-licensed:
   https://rive.app/docs/runtimes/web/web-js and
   https://rive.app/docs/runtimes/getting-started
5. `lottie-web` is MIT-licensed and supports SVG, canvas, and HTML renderers:
   https://github.com/airbnb/lottie-web
6. Spine runtime licensing is not a simple permissive dependency fit for this
   feature:
   https://en.esotericsoftware.com/spine-runtimes-license
