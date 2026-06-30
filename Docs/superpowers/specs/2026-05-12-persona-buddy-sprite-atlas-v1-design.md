# Persona Buddy Sprite Atlas V1.1 Design

Date: 2026-05-12
Status: Approved direction; design slice for issue #1611
Owner: Codex brainstorming pass
Backlog: TASK-300

## Summary

Add explicit Persona/Buddy sprite atlas support as a narrow V1.1 hardening slice
under the existing `sprite_frames` renderer. The current renderer already knows
how to crop a `frames[].region` from one image asset, and the backend manifest
validator already checks region bounds when source dimensions are known. This
work turns that behavior into a documented and tested support path instead of a
new renderer.

The implementation should keep `renderer_type: "sprite_frames"`,
`manifest_version: 1`, explicit activation, and the current Buddy text fallback.
Atlas assets are ordinary Persona Visual Pack assets with
`asset_role: "sprite_sheet"` and animation frames that reference the atlas image
plus a region rectangle. "V1.1" is product-contract shorthand for this
hardening slice; it is not a new manifest version.

## Context

PR #1608 added the Persona/Buddy renderer capability registry and kept
`sprite_frames` as the only enabled renderer. The completed renderer-registry
design named sprite atlas support as the next safe follow-up before any non-
sprite manifest V2, Live2D adapter, or external provider work.

Existing code already contains the core pieces:

- Backend renderer capabilities expose only `sprite_frames`.
- `validate_visual_manifest()` validates `frames[].region` bounds when asset
  dimensions are available.
- `SpriteFrameRenderer` renders region-backed frames using CSS background
  cropping.
- Buddy rendering and diagnostics go through the frontend renderer registry.
- `PersonaVisualAssetRole` already includes `sprite_sheet`.

The gap is support clarity. Users and future implementers should not have to
infer that a sprite atlas is represented by `sprite_frames` plus frame regions.

## Goals

1. Define sprite atlas packs as a supported `sprite_frames` V1.1 convention.
2. Preserve the backend renderer capability contract from PR #1608.
3. Add focused backend and frontend coverage for atlas-backed packs.
4. Document a minimal atlas-backed manifest example.
5. Keep Buddy runtime fail-soft when atlas data is missing, malformed, or
   unsupported.

## Non-Goals

1. No new `sprite_sheet` renderer capability.
2. No manifest version bump.
3. No Live2D, Rive, Spine, Lottie, or other non-sprite renderer work.
4. No VN/CYOA runtime changes.
5. No automatic atlas packing, image slicing, or image generation workflow.
6. No Persona Garden atlas authoring UI beyond small copy/help updates if the
   implementation touches an existing editor surface.
7. No marketplace, shared-library, or external MCP provider behavior.

## Approved Model

Sprite atlas support stays under `sprite_frames`.

A supported atlas-backed pack uses:

```json
{
  "manifest_version": 1,
  "renderer_type": "sprite_frames",
  "states": {
    "idle": { "animation_id": "idle" }
  },
  "animations": {
    "idle": {
      "frames": [
        {
          "asset_id": "idle-atlas",
          "region": { "x": 0, "y": 0, "width": 128, "height": 128 },
          "duration_ms": 120
        },
        {
          "asset_id": "idle-atlas",
          "region": { "x": 128, "y": 0, "width": 128, "height": 128 },
          "duration_ms": 120
        }
      ],
      "loop": true,
      "preview_frame": 0
    }
  }
}
```

The referenced asset row should use `asset_role: "sprite_sheet"` when the source
   file is an atlas. Runtime safety still comes from manifest references,
dimensions, and region validation, not from adding a new renderer type.

Atlas-backed animations should use `preview_frame` when they need a specific
preview crop. `preview_asset_id` is still valid for ordinary multi-asset frame
packs, but it is ambiguous for atlas animations where many frames reference the
same asset id.

## Backend Design

Keep the renderer capability registry unchanged for this slice:

- `sprite_frames` remains the only enabled renderer.
- `sprite_sheet` remains a reserved/future label in API/UI type vocabulary, not
  an activatable renderer.
- `validate_visual_manifest()` continues to reject `renderer_type:
  "sprite_sheet"`.

Backend validation should explicitly cover atlas behavior:

1. A `sprite_frames` manifest can reference one `sprite_sheet` asset across
   multiple frames with distinct `region` rectangles.
2. Regions require integer `x`, `y`, `width`, and `height`.
3. Regions reject negative coordinates and non-positive sizes.
4. When asset dimensions are known, regions reject rectangles that exceed the
   source image bounds.
5. When dimensions are missing, validation may remain permissive for finite,
   positive regions. The browser renderer can still crop deterministically from
   the provided URL; known dimensions are an extra bounds proof, not a hard
   prerequisite for every imported or draft atlas.
6. `preview_frame` should be covered for atlas-backed animations so previews do
   not accidentally collapse to the first frame when every frame references the
   same atlas asset.

This slice should not add renderer-level asset-role enforcement. Some existing
packs and imported drafts may have imperfect roles, and role mismatches are less
important than asset existence and region validity for runtime safety. If role
warnings are useful, they should be non-blocking diagnostics in a later editor
polish slice.

## Frontend Design

Keep the current Buddy renderer registry and `SpriteFrameRenderer`.

`SpriteFrameRenderer` should remain the single runtime component for both
separate-frame and atlas-backed `sprite_frames` packs:

- Frames without `region` render as normal images.
- Frames with `region` render as cropped background regions from the referenced
  asset URL.
- Missing assets, invalid regions, or unsupported crop data report the existing
  render errors and fall back to text Buddy.

The implementation should add focused coverage rather than a new component:

1. Registry renderability should remain a coarse mount decision: return true for
   atlas-backed packs with a known asset and at least one referenced frame, then
   let `SpriteFrameRenderer` report `unsupported_region` for malformed crop
   data. This preserves diagnostics instead of hiding the renderer behind the
   text fallback too early.
2. Renderer tests should assert cropped background style for atlas frames.
3. Diagnostics tests should keep `unsupported_region` and `missing_asset`
   behavior stable.

If an existing editor surface displays renderer help or manifest examples, it
can add copy that says atlas sheets are represented by `sprite_frames` plus
frame regions. Do not build a new atlas editor in this slice.

## Data Flow

1. User uploads or imports an atlas image as a Persona Visual Pack asset with
   `asset_role: "sprite_sheet"`.
2. The pack manifest remains `renderer_type: "sprite_frames"` and maps state
   animations to frames that reference the atlas asset.
3. Each atlas-backed frame includes a `region` rectangle.
4. Backend activation/import-preview validation confirms the renderer is
   activatable and checks frame references and known-dimension bounds.
5. Buddy loads the active pack through existing visual-pack APIs.
6. The frontend renderer registry selects `SpriteFrameRenderer`.
7. `SpriteFrameRenderer` crops the atlas region or reports a render error and
   falls back to text Buddy.

## Error Handling

Backend behavior:

- Unknown or future renderer types still fail closed.
- `sprite_sheet` renderer types are rejected until a future issue explicitly
  adds that renderer.
- Bad region shapes fail validation.
- Known-dimension bounds violations fail validation.
- Missing dimensions do not block draft or activation solely because the server
  cannot prove bounds, provided the region shape itself is finite, integer, and
  positive.

Frontend behavior:

- Missing atlas assets produce the existing `missing_asset` diagnostic.
- Out-of-bounds or non-finite region values produce `unsupported_region`.
- Unsupported renderer labels produce `unsupported_renderer`.
- Buddy keeps the text fallback in all failure cases.

## Documentation

Update the current Persona Visual Packs docs to include:

- A short "Sprite Atlas Packs" section.
- The minimal manifest example from this spec or a shorter equivalent.
- A clear statement that `sprite_sheet` is an asset role, not a renderer type in
  this slice.
- A note that `renderer_type: "sprite_sheet"` is still rejected by V1
  capabilities.

If the implementation touches API docs or frontend help copy, it should use the
same language.

## Testing Plan

Backend tests:

1. Valid atlas-backed `sprite_frames` manifest with one sprite-sheet asset and
   multiple region frames validates successfully.
2. Region bounds that exceed known dimensions are rejected.
3. Non-integer or malformed region values are rejected.
4. `renderer_type: "sprite_sheet"` remains unsupported.
5. Activation-required validation preserves required-state behavior for atlas
   packs.
6. Atlas preview behavior uses `preview_frame` rather than relying on
   `preview_asset_id` when all frames share the same atlas asset.

Frontend tests:

1. `SpriteFrameRenderer` renders atlas frames with cropped background style.
2. Renderer registry reports atlas-backed `sprite_frames` packs as renderable
   when the referenced atlas asset exists, without duplicating full region
   validation.
3. Diagnostics keep `unsupported_region` and `missing_asset` fallback behavior.
4. Existing separate-frame sprite tests continue to pass.

Verification should include focused Persona visual backend tests, focused
PersonaBuddy Vitest tests, `git diff --check`, and Bandit on touched backend
code if backend Python changes are made.

## Rollout Notes

This is a low-risk hardening slice because it does not add a new renderer or
change active-pack selection. Existing `sprite_frames` packs keep working, and
malformed atlas data still falls back through the current Buddy diagnostics.

The slice should ship as a small PR with issue #1611 linked. Future work can
then choose from:

1. Persona Garden atlas authoring controls.
2. Automatic atlas packing/slicing jobs.
3. Non-sprite manifest V2 design.
4. Feature-gated Live2D adapter spike.
5. External MCP pack-provider contract.

Those future items should remain separate issues.
