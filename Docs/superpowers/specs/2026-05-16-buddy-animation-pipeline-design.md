# Buddy Animation Pipeline Design

Date: 2026-05-16
Status: Draft for review
Backlog: TASK-410
GitHub: https://github.com/rmusser01/tldw_server/issues/1787
Parent epic: https://github.com/rmusser01/tldw_server/issues/1510

## Summary

This design defines the Buddy animation pipeline for Persona Visual Packs. It is
scoped to the floating Persona Buddy and live Persona assistant surface, not VN
assets or Persona backend management work.

The key decision is to make final Buddy art a neutral-pose-first production
workflow:

1. establish one approved neutral identity anchor for a Buddy,
2. optionally produce separate static talking and reaction sheets from that anchor,
3. generate timed animation strips or atlas regions from the same anchor,
4. review generated candidates,
5. import or copy the result into an inactive Persona Visual draft pack,
6. activate only after explicit user review.

The current starter catalog contains twelve starter IDs and the runtime
already supports bounded custom visual states through `state_catalog` and
`authored_triggers`. This spec turns those scaffolds into a production-ready
asset pipeline without claiming that finished default animation packs already
exist.

## Current Foundations

The current implementation already provides the foundations this pipeline should
reuse:

- `Docs/Code_Documentation/Persona_Visual_Packs.md` documents user-owned,
  persona-attached, manifest-backed packs, explicit activation, import/export,
  generated-candidate review, personal library reuse, the default-pack
  production tracker, and the current twelve starter IDs.
- `tldw_Server_API/app/core/Persona/visual_starter_fixtures.py` defines the
  immutable starter fixtures, production recipe metadata, complexity tiers,
  expected asset groups, and current production status. The basic tier now
  contains six art-ready defaults; intermediate and intricate starters remain
  scaffolds until their reviewed production assets land.
- `tldw_Server_API/app/core/Persona/visuals.py` validates sprite-frame manifests,
  frame timing, atlas regions, required states, custom state catalog entries,
  authored triggers, fallbacks, and bounds such as 256 custom states and 512
  triggers.
- `apps/packages/ui/src/types/persona-visuals.ts` defines the shared frontend
  manifest shape, including `state_catalog`, `authored_triggers`, custom state
  IDs, and renderer capability metadata.
- `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualState.ts`
  resolves live voice state, tool status, exact `tool_name` triggers, MCP
  runtime reasons, and authored triggers before falling back to built-in states.
- `SpriteFrameRenderer` and `BuddyShellHost` already render active sprite-frame
  packs with fail-soft diagnostics instead of blocking Persona Live.

The correct implementation path is therefore extension and hardening of this
existing stack, not a new avatar runtime.

## Goals

1. Define the final-art production workflow for bundled default Buddies.
2. Preserve the existing Persona Visual pack contract and explicit activation
   model.
3. Adapt the Puzzle Attack art pipeline pattern to Buddy assets: manifest-driven
   production, neutral/model-sheet anchoring, generated strips, compilation,
   review, and deterministic runtime export.
4. Support simple and complex user-created Buddies without forcing every user
   through an intricate art workflow.
5. Support many custom states and per-tool animation variants while keeping IDs,
   labels, triggers, and fallback chains bounded and safe.
6. Keep static talking sheets and static reaction sheets distinct from timed animation frames.

## Non-Goals

1. Do not build a VN/CYOA asset workflow.
2. Do not add executable visual plugins, arbitrary JavaScript, or untrusted
   runtime renderer code.
3. Do not treat the current scaffold PNG fixtures as final art.
4. Do not activate generated, imported, or copied packs automatically.
5. Do not require Live2D, Rive, Lottie, Spine, or another non-sprite renderer
   for the first production Buddy pipeline.
6. Do not use raw prompts, secrets, host-local paths, or provider payloads as
   durable user-facing provenance.

## Bundled Default Starter Catalog

The bundled defaults are organized as three complexity tiers. Earlier tracker
and issue text described three basic defaults, but the current basic tier is the
six Codex Buddy defaults: Search Lens, Index Card, Archive Cube, Paperclip,
Terminal Tile, and Migu. The existing starter IDs stay stable; final art
replaces scaffold assets behind those contracts.

| Tier | Starter ID | Design intent | Production target |
| --- | --- | --- | --- |
| Basic | `search-lens-basic` | Friendly search-lens object Buddy. | Reviewed neutral/preview assets and required-state loops; Codex atlas upgrade path. |
| Basic | `index-card-basic` | Friendly tabbed card object Buddy. | Reviewed neutral/preview assets and required-state loops; Codex atlas upgrade path. |
| Basic | `archive-cube-basic` | Friendly archive/storage cube Buddy. | Reviewed neutral/preview assets and required-state loops; Codex atlas upgrade path. |
| Basic | `paperclip-basic` | Friendly paperclip object Buddy. | Reviewed neutral/preview assets and required-state loops; Codex atlas upgrade path. |
| Basic | `terminal-tile-basic` | Friendly terminal-window tile Buddy. | Reviewed neutral/preview assets and required-state loops; Codex atlas upgrade path. |
| Basic | `migu-marker-basic` | Rough marker-line user-art feel inspired by the supplied Migu image. | Preserve hand-drawn charm; reviewed required-state loops and Codex atlas upgrade path. |
| Intermediate | `study-desk-intermediate` | Calm study companion in the Puzzle Attack clean anime/game-sprite style. | Neutral anchor, static talking sheet, separate required-state loops. |
| Intermediate | `tool-helper-intermediate` | Utility-themed helper with exact tool variant support. | Required states plus at least one exact `tool_name` animation variant. |
| Intermediate | `object-creature-intermediate` | Non-human expressive object companion. | Proves the format is not humanoid-only. |
| Intricate | `lofi-study-intricate` | Original lofi-study companion, not a copyrighted character clone. | Full model sheet, separate talking and reaction sheets, atlas-backed loops, tool variants. |
| Intricate | `action-guide-intricate` | High-motion guide with anticipation and success beats. | Rich reaction states and more expressive motion arcs. |
| Intricate | `elaborate-persona-intricate` | High-detail fantasy/sci-fi assistant. | Multiple custom-state rows, atlas/strip compilation, richer review gates. |

The lofi default must capture the broad product direction of a calm study
companion in the same clean, readable game-art language as Puzzle Attack. It
must not copy a protected character identity. The Migu default should preserve
the user-art spirit of the reference image: playful proportions, visible marker
quality, cyan twin-tail silhouette, and deliberately simple expression language.
The six basic defaults currently ship as art-ready 96x96 Persona Visual frame
packets; cross-app interchange or Codex Buddy reuse should use the documented
Codex/Petdex `pet.json` plus 8x9 atlas path rather than treating those 96x96
packets as the interchange ceiling.

## Complexity Tiers

### Basic

Basic Buddies optimize for quick creation and low art burden. A user should be
able to create a usable Buddy from one neutral pose plus a small number of
derived loops. For bundled defaults, this tier is the six Codex Buddy defaults
listed above. The simple tldw runtime packet may use separate frame assets, but
the portability target for Codex Buddy parity is the Codex/Petdex 8x9 atlas
contract.

Required production assets:

- identity brief
- neutral anchor
- preview image
- required-state loops for `idle`, `listening`, `thinking`, `speaking`, and
  `error`

Optional assets:

- tiny static mouth or expression sheet
- one reaction loop

Basic packs may reuse the same neutral anchor across several states if the
manifest maps those states clearly and fallbacks remain valid.

### Intermediate

Intermediate Buddies add more expression without requiring a full atlas
workflow.

Required production assets:

- identity brief
- neutral anchor or simple model sheet
- static talking and reaction sheets
- separate required-state loops
- one or more custom-state variants

Intermediate packs should demonstrate at least one authored trigger or custom
state category, such as `tool.notes_search` or `reaction.success`, with a clear
fallback chain.

### Intricate

Intricate Buddies demonstrate the high ceiling of the format.

Required production assets:

- identity brief
- approved neutral model sheet
- static talking and reaction sheets
- animation strips or atlas source
- compiled runtime atlas or frame set
- multiple custom-state variants

Intricate packs should use the same neutral identity anchor for every generated
strip. Their review checklist must include identity consistency, silhouette
stability, frame registration, transparent background, atlas region validity,
state/trigger alignment, and fallback behavior.

## Neutral-Pose-First Workflow

The production sequence is intentionally linear:

```text
identity brief
  -> neutral identity anchor
  -> optional static talking and reaction sheets
  -> generated animation strips or atlas frames
  -> validation and compilation
  -> human review
  -> inactive visual-pack draft
  -> explicit activation
```

Rules:

1. The neutral anchor is the identity source of truth for all downstream assets.
2. Do not generate timed animation frames directly from text after the anchor is
   approved.
3. Static talking sheets and static reaction sheets are still images with semantic cells. They are
   not animations until manifest `animations` map cells into timed frames.
4. Animation outputs must be generated from the neutral anchor or a model sheet
   using image-to-image continuity.
5. Generated strips can be compiled into atlas-backed `sprite_frames` manifests
   or separate frame assets.
6. Review must happen before draft commit or candidate acceptance, and
   activation remains separate.

This mirrors the useful Puzzle Attack pattern: model-sheet-first production,
manifest-driven asset IDs, generated strips, compile profiles, release exports,
and review gates. It avoids the failure mode where a single sheet of expressions
is mistaken for an animation set.

## Animation Production Contract

Animation generation should use strip-based production because it scales from
simple to intricate packs.

Recommended source units:

- 4-frame horizontal strips for small loops or previews.
- 16-frame loop targets for polished states, organized as four 4-frame strips.
- Atlas compilation for intricate packs when multiple states share one image.

Recommended runtime forms:

- separate `frame` assets for simple packs.
- one `sprite_sheet` atlas plus frame `region` entries for intricate packs.

Recommended state loop shape:

| State | Intent | Minimum frame behavior |
| --- | --- | --- |
| `idle` | neutral resting presence | soft hold, blink, breathing, or subtle body motion |
| `listening` | attentive input state | lean, eye focus, ear/head cue, or device cue |
| `thinking` | processing state | small loop with concentration or tool-use read |
| `speaking` | output state | mouth/expression movement; can reuse static talking cells as timed frames |
| `error` | recoverable problem | clear but non-alarming fault/retry read |
| `tool_running` | generic tool work | fallback if no exact tool state exists |
| `approval_needed` | waits for user decision | paused, expectant, or prompt-facing loop |
| `wake_armed` | passive listening enabled | subtle readiness cue |
| `offline` | unavailable/degraded | dimmed or inactive loop |

For short-lived states, the first 3 to 4 frames must communicate the state
clearly. Puzzle Attack's 16-frame design has the same concern: many runtime
state durations show only early frames, so key visual meaning must be
front-loaded.

## Manifest State Catalog Semantics

The current runtime already supports custom states through `state_catalog` and
`authored_triggers`. The production design should treat this as the
state-catalog V2 semantics for sprite-frame packs, but the immediate
implementation should not bump the wire-level manifest version. In this slice,
"V2 semantics" means a documented production contract over the existing
`manifest_version: 1` sprite-frame renderer.

The compatibility rule is:

- `manifest_version: 1` plus `renderer_type: "sprite_frames"` remains the only
  activatable V1 Buddy renderer until a deliberate backend capability bump.
- Custom visual states are declared in `state_catalog`.
- `states` may map built-in states and declared custom states to animations.
- `fallbacks` define how custom states degrade to built-in states.
- `authored_triggers` map runtime signals to built-in or custom states.
- `custom_state_variants` means timed runtime loops or frame mappings for
  declared custom states, not static source-sheet cells by themselves.

If a future implementation introduces `manifest_version: 2` for sprite-frame
packs, the renderer capability endpoint must advertise both versions during
migration and all import/export paths must preserve V1 compatibility.

### Bounds

The current backend bounds are acceptable for the first production version:

- up to 256 custom state IDs per pack,
- up to 512 authored triggers per pack,
- up to 240 frames per animation,
- up to 8 fallback-chain depth,
- trigger duration between 100 and 30,000 ms,
- state label up to 80 characters,
- state description up to 280 characters,
- up to 16 tags per custom state,
- custom state ID pattern `^[a-z][a-z0-9_.:-]{0,95}$`.

This is not literally unlimited, but it is large enough for a complex user's
per-tool and per-response variants while staying safe for validation, editor UI,
runtime lookup, and import/export.

### Recommended Custom State ID Namespaces

Use stable, lowercase namespaces:

- `tool.<tool_id>` for exact tool variants.
- `tool.<category>.<variant>` for category-level tool families.
- `reaction.<name>` for short emotional or response beats.
- `mood.<name>` for longer-lived presentation modes.
- `live.<name>` for live-session variants beyond built-ins.
- `pack.<name>` for pack-private states that should not imply global semantics.

Examples:

```json
{
  "state_catalog": {
    "tool.notes_search": {
      "label": "Searching notes",
      "kind": "tool_variant",
      "description": "Shown while the notes search tool runs.",
      "tags": ["tool", "notes"]
    },
    "reaction.success": {
      "label": "Success",
      "kind": "reaction",
      "description": "Short celebratory response after work completes.",
      "tags": ["reaction"]
    }
  },
  "fallbacks": {
    "tool.notes_search": ["tool_running", "thinking", "idle"],
    "reaction.success": ["speaking", "idle"]
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
  ]
}
```

### Trigger Matching

Trigger resolution should keep the current priority order:

1. hard error or recovery state,
2. approval needed,
3. unexpired MCP runtime override,
4. highest-priority matching authored trigger,
5. generic tool running,
6. wake armed,
7. normalized live voice state,
8. offline,
9. idle.

Exact `tool_name` triggers must match the structured active tool name, not a
display string. `tool_category` can use status/category text as a lower-specific
fallback. MCP runtime triggers should use bounded runtime reasons, not arbitrary
payloads.

## Asset Creation And Import

Users should have three production paths:

1. Start from a bundled default scaffold and generate or upload final assets.
2. Upload their own neutral anchor and build a pack from it.
3. Import a `.tldw-persona-vpack` or external MCP provider proposal into review.

All three paths converge on the same draft and review model:

- create or select a persona-attached draft pack,
- add assets through bounded raster storage,
- validate the manifest against available assets and dimensions,
- show preview diagnostics,
- activate only when the user explicitly chooses to activate.

Background jobs are the right execution model for generation, compilation, and
portable import/export because they need progress, retries, review state, and
user-visible failure diagnostics.

## Review Criteria

A candidate Buddy pack is not production-ready until review confirms:

- the final art matches the neutral identity anchor,
- the same character/object is present in every frame,
- transparent background is clean where required,
- cells are sliceable and frame registration is stable,
- no static expression sheet is mislabeled as animation,
- required states resolve,
- custom states are declared in `state_catalog`,
- custom states have fallbacks,
- exact tool variants use structured `tool_name` matches,
- atlas regions fit inside source image dimensions,
- first frames of short loops communicate state,
- provenance is bounded and trace-safe.

## Staged Implementation Plan

### Stage 1: Spec and Tracker

Goal: land this design, link it from the Backlog task, and create or update a
GitHub issue under the live persona/buddy epic.

Verification:

- markdown link/path review,
- `git diff --check`,
- Backlog task records the spec path and non-code validation.

### Stage 2: Catalog Metadata Alignment

Goal: make starter catalog metadata match the current bundled default
production intent without adding fake animation assets.

Likely changes:

- tighten production recipes for the starter IDs,
- ensure final-art status stays `scaffold` until assets exist,
- add doc/API tests proving the catalog is explicit about scaffold status.

Verification:

- focused Persona starter catalog tests,
- schema validation tests,
- no runtime activation claims for missing art.

### Stage 3: Pipeline Contract Fixtures

Goal: add small manifest/fixture examples for neutral anchor, static sheet,
strip, atlas, and compiled runtime manifest outputs.

Likely changes:

- schema fixtures for production packet metadata,
- import-preview examples,
- docs for expected file names and state IDs,
- no generated final art committed unless reviewed assets exist.

Verification:

- manifest validation tests,
- import preview tests,
- asset-reference remapping tests.

### Stage 4: Editor UX For Production Packets

Goal: help users create packs through the neutral-anchor-first workflow in
Persona Garden.

Likely changes:

- visual distinction between neutral anchor, static sheet, generated strips,
  atlas, and active runtime manifest,
- starter catalog copy/generation choices,
- custom-state and exact-tool-variant editing improvements,
- review warnings when static sheets are not wired as timed animation frames.

Verification:

- focused `VisualPackEditor` tests,
- Buddy shell render tests,
- browser screenshot review before calling UI work complete.

### Stage 5: Runtime And MCP Trigger Hardening

Goal: make authored custom states reliable under live tool execution.

Likely changes:

- preserve exact `active_tool_name` propagation,
- add diagnostics for missing custom-state animations,
- ensure MCP runtime triggers stay transient and bounded,
- document external MCP provider handoff for generated animation variants.

Verification:

- `personaVisualState` tests,
- `BuddyShellHost` tests,
- MCP module tests for review-first behavior.

### Stage 6: Final Default Art Production

Goal: produce the actual bundled default Buddy animation packs.

This is an asset-production effort, not a code-only patch. Each default should
go through:

1. identity brief,
2. neutral anchor,
3. static talking and reaction sheets if applicable,
4. animation strip/atlas generation,
5. machine validation,
6. human visual review,
7. inactive draft import/copy,
8. explicit activation test.

Verification:

- visual review artifacts,
- manifest validation,
- runtime Buddy screenshot/video checks,
- catalog status changes from `scaffold` to `art_ready` only for completed
  defaults.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Static expression sheets are mistaken for animations. | Keep `static_sheet` and `animation_outputs` separate in recipes, docs, UI, and review checks. |
| Starter scaffolds are treated as final art. | Preserve `production_status: scaffold` and explicit docs until reviewed assets exist. |
| Custom state IDs become unsafe or unbounded. | Keep current ID regex, unsafe marker checks, and count limits. |
| Per-tool variants become brittle. | Prefer exact structured `tool_name` matches, with `tool_category` only as fallback. |
| Generated frames drift from the character identity. | Require neutral-anchor/image-to-image generation and identity consistency review. |
| Atlas compilation produces partial or out-of-bounds output. | Fail closed when dependencies or regions are invalid. |
| New pipeline duplicates existing Persona Visual behavior. | Reuse draft packs, generated candidates, import preview, personal library, and explicit activation. |

## Open Decisions

1. Whether to represent this as a wire-level `manifest_version: 2` for
   `sprite_frames`, or as documented state-catalog V2 semantics within the
   existing `manifest_version: 1` renderer.
2. Whether final default art should be committed as bundled assets or shipped as
   optional downloadable packs.
3. Whether external MCP providers may submit production packets directly, or
   only portable archives/generated candidates through the existing review
   surfaces.

The recommended first implementation target is to keep the current
`manifest_version: 1` sprite-frame wire format, strengthen the documented
state-catalog V2 semantics, and delay an actual manifest version bump until a
non-compatible renderer or archive change requires one.
