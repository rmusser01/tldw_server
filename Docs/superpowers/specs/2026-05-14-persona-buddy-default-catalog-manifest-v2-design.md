# Persona Buddy Default Catalog and Manifest V2 Visual States Design

Date: 2026-05-14
Status: Approved direction; design slice for TASK-347
Owner: Codex brainstorming pass
Backlog: TASK-347

## Summary

Define the next Persona/Buddy visual-pack design slice: a nine-pack default
catalog, a user-facing asset creation flow based on a neutral identity anchor,
and a manifest V2 extension that supports a large bounded set of custom state
IDs and per-tool animation variants.

This is not a new renderer. It extends the existing Persona Visual Pack contract
that uses `renderer_type: "sprite_frames"`, state-to-animation mappings,
explicit activation, reviewable generated/imported drafts, and text fallback
when Buddy cannot render the active pack. Static expression sheets remain useful
for talking/reaction poses, but animation is generated from a central neutral
pose/model-sheet anchor into strips, frames, or atlases.

## Current Foundations

Current Persona Visual pack behavior already provides the base contract:

- Built-in visual states are `idle`, `wake_armed`, `listening`, `thinking`,
  `speaking`, `tool_running`, `approval_needed`, `error`, and `offline`.
- Activation requires `idle`, `listening`, `thinking`, `speaking`, and `error`.
- `sprite_frames` is the enabled renderer. Sprite atlases are represented as
  `sprite_frames` animations whose frames reference regions of an asset with
  `asset_role: "sprite_sheet"`.
- Authored triggers currently match `live_state`, `tool_category`, and
  `mcp_runtime`.
- Starter packs can be copied to user-owned inactive drafts; activation remains
  explicit.
- Renderer capabilities currently advertise supported manifest versions per
  renderer. `sprite_frames` should advertise manifest V2 only after backend
  validation and Buddy runtime support exist for V2 custom states.

Relevant in-repo references:

- `Docs/Code_Documentation/Persona_Visual_Packs.md`
- `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- `Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md`
- `Docs/superpowers/specs/2026-05-12-persona-buddy-sprite-atlas-v1-design.md`
- `Docs/superpowers/specs/2026-05-12-persona-buddy-renderer-capability-registry-design.md`

Puzzle Attack reference materials in the sibling repo supply the asset-pipeline
model:

- `/Users/macbook-dev/Documents/GitHub/puzzle-attack/content/blog/2026-03-22-ai-asset-pipeline.md`
- `/Users/macbook-dev/Documents/GitHub/puzzle-attack/docs/plans/2026-03-26-anim16-art-pipeline-design.md`
- `/Users/macbook-dev/Documents/GitHub/puzzle-attack/docs/plans/2026-03-29-fighting-animation-system-design.md`
- `/Users/macbook-dev/Documents/GitHub/puzzle-attack/scripts/art-review/README.md`

The reusable idea is not "make a sheet of expressions and call it animation".
The reusable idea is: define a consistent character identity and neutral pose,
generate controlled strips or sheets from that anchor, validate consistency,
review outputs, and only then compile accepted frames into runtime assets.

## Goals

1. Define nine bundled default buddies across basic, intermediate, and intricate
   complexity tiers.
2. Preserve a neutral-pose-first creation workflow for defaults and user-created
   buddies.
3. Distinguish static talking/reaction sheets from real animation strips or
   atlases.
4. Add manifest V2 custom-state semantics for a large bounded set of states and
   per-tool animation variants.
5. Keep manifest V1 packs compatible and keep the V1 required-state activation
   gate.
6. Keep generation/import review-first: background jobs produce drafts or
   candidates, not silently activated runtime packs.
7. Provide staged implementation slices for backend validation, MCP/runtime
   trigger handling, editor UX, default fixtures, docs, and verification.

## Non-Goals

1. No Live2D, Cubism, Spine, Rive, Lottie, or 3D renderer work in this slice.
2. No new `sprite_sheet` renderer type. `sprite_sheet` remains an asset role
   used by the `sprite_frames` renderer.
3. No marketplace or cross-user shared library.
4. No automatic activation of generated or imported visual packs.
5. No direct use of protected character likenesses for bundled defaults. The
   lo-fi default should be an original study companion inspired by the workflow
   and mood, not a copy of the known "Lo-fi Girl" character.
6. No VN/CYOA visual runtime changes.

## User-Confirmed Direction

- The first implementation target is the simple/user-owned asset-pack path, but
  the design should leave room for richer generated animation workflows.
- Assets are user-owned and attached to one persona by default, while still
  being stored as packs with manifests so future duplicate, import/export, or
  shared-library workflows do not need a core format change.
- The expected user creation process should mirror the Puzzle Attack pipeline:
  start from a central neutral pose/state, then create frames and animations
  from that anchor.
- Static expression sheets are for talking and reactive pose selection. They are
  not the animation system.
- There should be nine bundled defaults: three basic, three intermediate, and
  three intricate.
- Manifest V2 should support custom state IDs and per-tool animation variants
  at a large but bounded scale.

## Asset Creation Model

Every default and user-created buddy should be explainable as the same pipeline,
with optional steps for simpler designs.

1. Identity brief
   - Name, silhouette, palette, line weight, personality notes, allowed props,
     forbidden drift, and complexity tier.
   - For uploaded user art, this can be extracted from the neutral image plus a
     short user description.

2. Neutral identity anchor
   - A central neutral pose or model sheet becomes the canonical identity source.
   - This anchor is used for later static talking poses, animation keyframes,
     validation, and regeneration.
   - The anchor should be stored as a normal pack asset and referenced in pack
     metadata or production notes. It does not need to be a rendered runtime
     state.

3. Static talking/reaction sheet
   - A compact sheet of mouth shapes, face changes, or small pose swaps.
   - Used for lightweight speaking and reactive state changes.
   - Useful for basic buddies and low-cost packs.
   - Not treated as frame-by-frame animation unless the manifest explicitly maps
     cells/regions into timed animation frames.

4. Animation strip generation
   - Generate short strips from the neutral anchor for states such as `idle`,
     `listening`, `thinking`, `speaking`, `tool_running`, and custom tool
     variants.
   - Basic packs can use 2 to 6 frame loops; intermediate packs can use 6 to 12
     frame loops; intricate packs can use 12 to 24 frame loops or compiled
     atlases.
   - The current hard cap of 240 frames per animation remains more than enough
     for these loops.

5. Validation and review
   - Check frame counts, grid alignment, transparent background where relevant,
     consistent character identity, one subject per cell, and reasonable bounds.
   - Use a review-before-promote flow similar to Puzzle Attack's art review
     harness: approve, reject, annotate, regenerate, and promote.
   - Generated/imported outputs remain candidates or inactive drafts until the
     user accepts them.

6. Compile to runtime pack
   - Accepted frames become `sprite_frames` animations.
   - Separate frame files and sprite-sheet/atlas regions are both valid under
     the same renderer.
   - The manifest maps required built-in states and any declared custom states
     to animation IDs.

## Complexity Tiers

| Tier | Purpose | Typical Inputs | Runtime Coverage |
| --- | --- | --- | --- |
| Basic | Fast onboarding and user-editable examples | Neutral pose, small expression sheet, 2 to 6 frame loops | Required built-ins plus few optional states |
| Intermediate | Demonstrate richer authoring without high art cost | Neutral model sheet, expression sheet, several state strips | Required built-ins, optional built-ins, several custom tool/reaction states |
| Intricate | Show the ceiling for expressive buddies | Full model sheet, static sheet, keyframes, validated strips, compiled atlas | Required built-ins, optional built-ins, many custom states and per-tool variants |

## Nine Default Starter Buddies

| ID | Tier | Concept | What It Demonstrates |
| --- | --- | --- | --- |
| `research-buddy-basic` | Basic | Clean assistant mascot/robot, close to the earlier mockup direction | Low-risk default with readable silhouette, simple idle/listening/thinking/speaking/error loops |
| `migu-marker-basic` | Basic | Rough marker-line "Migu" inspired buddy with teal twin-tail silhouette and playful wobble | User-art-friendly style; shows simple drawings can become usable buddies |
| `minimal-helper-basic` | Basic | Simple geometric helper or object buddy | Lowest-complexity path for users who want a quick custom pack |
| `study-desk-intermediate` | Intermediate | Original desk/study companion with calm posture and props | Neutral seated anchor, static talking sheet, short writing/listening/thinking loops |
| `tool-helper-intermediate` | Intermediate | Utility-themed assistant with visual tool affordances | Per-tool variants for search, import, summarize, and approval-needed without high art complexity |
| `object-creature-intermediate` | Intermediate | Non-human expressive creature or object | Confirms the format is not limited to humanoid characters |
| `lofi-study-intricate` | Intricate | Original lo-fi-inspired study companion in the Puzzle Attack-style cel-shaded direction | Rich neutral model sheet, static talking sheet, idle/study/speaking loops, and tool variants |
| `action-guide-intricate` | Intricate | Energetic guide with anticipation, reaction, success, and error beats | More dynamic animation timing and expressive state transitions |
| `elaborate-persona-intricate` | Intricate | High-detail fantasy or sci-fi persona buddy | Full high-ceiling pack with many custom states, atlas layout, and strict review/validation |

All nine defaults should be bundled as starter packs and copied into user-owned
inactive drafts when selected. They should not auto-activate on copy.

Starter art production notes:

- Basic defaults should include a neutral anchor, a preview image, and simple
  animations for the five required built-in states. `migu-marker-basic` should
  intentionally preserve the rough user-art feel instead of over-polishing it.
- Intermediate defaults should include a neutral model sheet, a static
  talking/reaction sheet, optional built-in states, and at least one custom
  state example.
- Intricate defaults should include a richer neutral model sheet, static
  talking/reaction sheet, multiple animation strips or atlas regions, and
  several custom tool/reaction variants.
- `lofi-study-intricate` should be an original character in a study/lo-fi mood
  with Puzzle Attack-style production discipline, not a direct recreation of a
  known character.

## Manifest V2 State Model

Manifest V2 should extend state naming while preserving the current runtime
shape:

- Built-in states remain reserved and retain their current meaning.
- Required built-ins still gate activation: `idle`, `listening`, `thinking`,
  `speaking`, and `error`.
- Custom states must be declared in `state_catalog`.
- `states` may reference both built-ins and declared custom states.
- `fallbacks` may reference both built-ins and declared custom states.
- Authored triggers may target both built-ins and declared custom states.
- Unknown custom states fail validation instead of silently becoming runtime
  surprises.
- Manifest V1 remains valid as-is.

Example:

```json
{
  "manifest_version": 2,
  "renderer_type": "sprite_frames",
  "state_catalog": {
    "tool.notes_search": {
      "label": "Searching notes",
      "kind": "tool_variant",
      "description": "Used when the notes search MCP tool is running."
    },
    "reaction.approval_waiting": {
      "label": "Approval waiting",
      "kind": "reaction"
    }
  },
  "states": {
    "idle": { "animation_id": "idle-loop" },
    "listening": { "animation_id": "listen-loop" },
    "thinking": { "animation_id": "think-loop" },
    "speaking": { "animation_id": "talk-loop" },
    "error": { "animation_id": "error-pose" },
    "tool.notes_search": { "animation_id": "tool-notes-search-loop" },
    "reaction.approval_waiting": { "animation_id": "approval-loop" }
  },
  "fallbacks": {
    "tool.notes_search": ["tool_running", "thinking", "idle"],
    "reaction.approval_waiting": ["approval_needed", "idle"]
  },
  "authored_triggers": [
    {
      "id": "notes-search-tool",
      "source": "tool_name",
      "match": "notes.search",
      "state": "tool.notes_search",
      "duration_ms": 2400,
      "priority": 80
    }
  ],
  "animations": {
    "idle-loop": {
      "frames": [{ "asset_id": "idle-0", "duration_ms": 160 }],
      "loop": true
    },
    "listen-loop": {
      "frames": [{ "asset_id": "listen-0", "duration_ms": 160 }],
      "loop": true
    },
    "think-loop": {
      "frames": [{ "asset_id": "think-0", "duration_ms": 160 }],
      "loop": true
    },
    "talk-loop": {
      "frames": [{ "asset_id": "talk-0", "duration_ms": 120 }],
      "loop": true
    },
    "error-pose": {
      "frames": [{ "asset_id": "error-0", "duration_ms": 300 }],
      "loop": false
    },
    "tool-notes-search-loop": {
      "frames": [{ "asset_id": "tool-notes-search-0", "duration_ms": 140 }],
      "loop": true
    },
    "approval-loop": {
      "frames": [{ "asset_id": "approval-0", "duration_ms": 160 }],
      "loop": true
    }
  }
}
```

### State ID Rules

Use a large bounded model rather than unbounded user text:

- Maximum custom states per pack: 256.
- Maximum authored triggers per pack: 512.
- Maximum state ID length: 96 characters.
- Pattern: `^[a-z][a-z0-9_.:-]{0,95}$`.
- Built-in IDs are reserved and cannot be redeclared in `state_catalog`.
- Case-sensitive IDs should be rejected unless already lowercase.
- Newlines, slashes, backslashes, shell-like prefixes, and obvious secret
  markers should be rejected.
- Fallback traversal should cap at depth 8 and detect cycles.

Recommended `state_catalog` fields:

- `label`: user-facing name, length-bounded.
- `kind`: one of `tool_variant`, `reaction`, `live_variant`, `mcp_runtime`,
  `mood`, or `pack_private`.
- `description`: optional, length-bounded helper text.
- `tags`: optional short strings for editor filtering.

Recommended text bounds:

- `label`: 80 characters.
- `description`: 280 characters.
- `tags`: at most 16 tags, 32 characters each.

Keep top-level `fallbacks` as the canonical fallback graph so V2 builds on the
existing resolver shape. The editor can display fallback controls inside custom
state rows, but the serialized manifest should avoid two competing fallback
locations.

### Trigger Matching

Manifest V2 should keep the current trigger sources and add exact tool matching:

- Existing: `live_state`, `tool_category`, `mcp_runtime`.
- New: `tool_name`.

`tool_name` should match a stable sanitized tool key such as
`notes.search`, `media.import`, or `rag.retrieve`. `tool_category` should remain
available for broader variants such as `search`, `ingestion`, `summarization`,
or `approval`.

The Buddy runtime should eventually receive structured active-tool data:

```json
{
  "tool_name": "notes.search",
  "tool_category": "search",
  "status": "running",
  "run_id": "tool-run-id"
}
```

Until that structure exists everywhere, frontend resolution can keep the current
category parser and only enable exact `tool_name` matching when the structured
field is present.

### MCP Behavior

The Persona Visuals MCP module should stop treating target states as a static
enum once manifest V2 is enabled for a pack.

Rules:

1. MCP-triggered states may target any built-in state or any custom state
   declared by the active pack.
2. Unknown custom states should be rejected for direct trigger calls. Runtime
   fallback is for declared-but-unmapped states, not arbitrary strings.
3. Generation jobs may propose new custom states only as draft manifest changes
   that require review before activation.
4. Generated assets for an existing state must target a built-in state or a
   declared custom state.
5. Duration clamping and priority rules should continue to use the existing
   trigger safety bounds.

## Editor UX

Persona Garden and related Buddy visual-pack surfaces should present V2 states
as three groups:

1. Core states
   - Required: `idle`, `listening`, `thinking`, `speaking`, `error`.
   - Always visible and activation-gated.

2. Optional built-ins
   - `wake_armed`, `tool_running`, `approval_needed`, and `offline`.
   - Visible as recommended optional rows.

3. Custom states
   - User-created rows with ID, label, kind, fallback chain, animation, and
     trigger controls.
   - Guided "Add tool variant" flow should create a safe ID, choose `tool_name`
     or `tool_category`, attach an animation, and set fallback to
     `tool_running -> thinking -> idle` by default.

The editor should warn when a custom state has no animation and no fallback, or
when a trigger targets a state that cannot resolve. Draft save can remain
permissive, but activation and import-preview validation should fail closed.

## User Workflows

### Simple Upload

1. User creates or selects a persona.
2. User uploads a neutral image and provides a brief description.
3. The editor creates a basic draft pack with required states mapped to simple
   still poses or tiny loops.
4. User optionally adds a static talking sheet.
5. User previews and activates the pack.

### Guided Generation

1. User starts from a neutral anchor or a default starter pack.
2. User chooses a complexity target: basic, intermediate, or intricate.
3. Background jobs generate static talking sheets and animation strips.
4. User reviews candidates, rejects/regenerates weak outputs, and promotes
   accepted outputs.
5. The editor compiles promoted assets into `sprite_frames` animations.
6. User activates the final pack.

### Advanced Tool Variants

1. User opens custom states.
2. User adds a tool variant, for example `tool.notes_search`.
3. User maps the exact `notes.search` tool to that state.
4. User attaches an animation generated from the neutral anchor.
5. Runtime resolves that variant while the matching tool is active and falls
   back to `tool_running`, `thinking`, then `idle` when unavailable.

## Staged Implementation Plan

### Stage 1: Design Spec

Goal: Land this spec and task record.

Success criteria:

- TASK-347 links the design artifact.
- The spec captures default catalog, asset pipeline, manifest V2 state model,
  staged implementation, and verification.

### Stage 2: Backend Manifest V2 Validation

Goal: Extend backend validation to understand manifest V2 custom states.

Success criteria:

- Manifest V1 remains accepted without `state_catalog`.
- Manifest V2 accepts declared custom states in `states`, `fallbacks`, and
  `authored_triggers`.
- Unknown custom state references fail validation.
- State ID, custom-state count, trigger count, and fallback-cycle limits are
  tested.
- Required built-in activation behavior is unchanged.
- `sprite_frames` renderer capabilities advertise manifest versions `[1, 2]`
  only after V2 validation is implemented.

### Stage 3: Frontend Types, Resolver, and Editor

Goal: Teach WebUI shared types, runtime resolver, and editor surfaces about
custom states.

Success criteria:

- Frontend types represent built-in and custom state IDs without losing the
  known built-in set.
- Runtime resolves exact `tool_name` variants when structured data is present
  and preserves current category fallback behavior.
- Editor shows core, optional, and custom state groups.
- Custom-state creation validates IDs client-side and still relies on backend
  activation/import-preview validation.

### Stage 4: MCP and Generation Jobs

Goal: Let MCP and background generation work with declared custom states.

Success criteria:

- `persona_visuals.trigger_state` can target active-pack custom states.
- Direct MCP trigger calls reject undeclared custom states.
- Generation jobs can produce draft custom-state additions with review before
  activation.
- Existing built-in target-state behavior remains compatible.

### Stage 5: Asset Pipeline Adapter

Goal: Adapt the Puzzle Attack-style pipeline into Persona Visual pack authoring.

Success criteria:

- Identity brief and neutral anchor are first-class generation inputs.
- Static talking/reaction sheets and animation strips are separate job outputs.
- Review/promote flow exists before runtime activation.
- Compiled outputs can be separate frames or atlas regions under
  `sprite_frames`.

### Stage 6: Nine Default Starter Packs

Goal: Add the default catalog as bundled starter packs.

Success criteria:

- Three basic, three intermediate, and three intricate defaults are listed by
  the starter-pack API.
- Starter metadata includes complexity tier, production notes, and preview
  information.
- Copying a starter pack produces an inactive user-owned draft.
- At least one default demonstrates a basic user-art style, one demonstrates
  per-tool variants, and one demonstrates intricate atlas-backed animation.

### Stage 7: Documentation and Verification

Goal: Make the feature understandable and testable.

Success criteria:

- Persona Visual docs explain manifest V2 custom states and neutral-anchor
  asset creation.
- User-facing docs explain simple upload, guided generation, and tool variants.
- Backend, frontend, MCP, and starter-catalog tests cover the behavior above.
- Bandit is run for touched backend code in implementation slices.

## Verification Plan

Backend:

- Validate V1 manifests continue to pass.
- Validate V2 required built-ins, declared custom states, unknown-state
  rejection, unsafe IDs, count caps, trigger caps, and fallback-cycle detection.
- Validate activation still requires the five required built-ins.

Frontend:

- Resolve built-in live states exactly as today.
- Resolve custom exact tool-name triggers when structured tool data exists.
- Resolve category triggers as today.
- Fall back when a custom state is declared but no animation is mapped.
- Editor accepts safe custom state IDs and rejects unsafe ones before submit.

MCP:

- Accept direct trigger calls for built-ins and active-pack custom states.
- Reject undeclared custom state triggers.
- Keep duration clamp behavior.
- Propose generated custom-state additions only as draft/review changes.

Catalog:

- List all nine defaults with stable IDs and complexity metadata.
- Copy each starter into an inactive user-owned draft.
- Ensure bundled defaults do not reuse protected character likenesses directly.

Docs-only design changes should be verified with `git diff --check`. Bandit is
not applicable until backend Python code changes are made.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| State explosion makes packs hard to manage | Hard limits, editor grouping, search/filter by kind/tag, and fallback defaults |
| Custom IDs become unsafe user text | Strict pattern, length cap, rejected slashes/newlines/backslashes/secret markers |
| Tool matching drifts between MCP/runtime/editor | Structured tool event fields and shared trigger validation |
| Generated art loses identity over animation frames | Neutral anchor, validation checks, review/promote gate, regeneration notes |
| Defaults are too complex for normal users to imitate | Three complexity tiers and a basic user-art example |
| Lo-fi default looks like protected IP | Make it an original study companion; document style/mood without copying character design |
| V2 breaks existing packs | V1 compatibility, reserved built-ins, and unchanged activation requirements |

## Open Questions for Implementation

1. Should neutral anchors be represented as a new asset role or as production
   metadata on normal assets? Prefer metadata first to avoid unnecessary schema
   churn.
2. Should exact `tool_name` values come from MCP tool IDs, WebUI display names,
   or a normalized runtime key? Prefer normalized runtime keys.
3. Should starter-pack complexity metadata live only in manifest metadata or in
   starter-pack DB/API fields too? Prefer API fields if filtering by complexity
   becomes part of the UI.
4. Should a generated job be allowed to create a new custom state in the same
   candidate proposal as new frames? Prefer yes, but only as an inactive draft
   patch that the user reviews.
