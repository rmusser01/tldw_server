# Persona Live Visual Packs PRD

Product Requirements Document

Feature: User-owned animated 2D visual packs for Persona Buddy and Persona Live
Location: WebUI Persona Garden and shared Persona Buddy shell
Status: Draft product record
Owner: Product / WebUI / Persona
Last Updated: 2026-05-12

---

## 1. Executive Summary

Persona Live already gives users a live assistant surface backed by persona
profiles, voice state, MCP tools, and the floating Persona Buddy shell. Persona
Visual Packs turn that Buddy shell into a user-owned animated 2D assistant:
users can upload, generate, review, import, export, activate, and deactivate
visual packs attached to a persona.

This PRD is the durable product record for the feature. The implementation
brainstorm and staging spec in
`Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md` remains useful
historical context, but this file is the long-standing Product/WebUI source of
truth.

The primary user path is Persona Buddy / Persona Garden, not VN or CYOA play
surfaces. VN asset-pack portability work from PR #1135 informed the background
job and review model for pack import/export, but Persona Visual Packs are a
Persona Live feature and should remain centered on the live assistant.

---

## 2. Problem Statement

Users can define rich persona behavior, voices, commands, tools, and live
sessions, but the visible Buddy assistant is still mostly derived/static unless
a custom visual pack exists. Users need a first-class way to make the live
assistant visually theirs without creating a separate persona identity or
leaving the Persona workflow.

The implementation now covers the core data model, renderer, API, editor, Jobs
flow, MCP module, E2E runtime behavior, Buddy entry point, diagnostics, setup
states, ownership/help copy, duplicate-to-persona drafts, and a reference-backed
personal pack library, import conflict choices, and reusable Persona Garden
affordances. PR #1608 also added the renderer capability contract and Buddy
renderer registry while keeping `sprite_frames` as the only enabled V1 runtime
renderer. The remaining product gap is optional Phase 3 externalization:
external visual providers, shared/cross-device libraries, non-sprite manifest
design, and future renderer adapters.

---

## 3. Goals

1. Let users create, upload, generate, import, export, and activate animated 2D
   visual packs for an existing persona.
2. Keep visual identity attached to Persona Buddy and Persona Live rather than
   creating a parallel avatar/runtime model.
3. Use the existing Persona Garden Visuals tab as the main authoring and review
   surface.
4. Make the floating Persona Buddy shell a direct entry point into the Visuals
   workflow for the selected persona.
5. Preserve a manifest-backed pack format that supports import/export,
   duplicate-to-persona, shared libraries, and future renderer types.
6. Use Jobs for generation and portability work that needs progress, review,
   retry, and user-visible status.
7. Allow MCP to control transient runtime states and create draft/review changes
   without silently replacing the active pack.
8. Ensure broken or missing visual assets never block Persona Live controls.

---

## 4. Non-Goals

1. Do not make VN Play or CYOA scenes the primary visual-assistant path.
2. Do not create a new assistant identity separate from the existing persona.
3. Do not replace profile avatars or character card images across the app.
4. Do not allow executable user-supplied visual plugins, arbitrary JavaScript,
   or unsafe SVG animation logic.
5. Do not ship Live2D as the initial required renderer.
6. Do not let MCP tools silently activate generated or imported packs.
7. Do not require microphone, TTS provider, or external image generation
   availability for baseline runtime tests.

---

## 5. Current Implementation Snapshot

As of 2026-05-12:

1. The durable implementation from PR #1393 is merged into `dev`.
2. The closeout documentation and E2E verification from PR #1400 is merged into
   `dev`.
3. PR #1412 is merged and adds the direct floating Buddy popover action to open
   the selected persona's Visuals workflow.
4. PR #1447 is merged and closes the ordered Product Hardening tracker from
   #1428: reliability diagnostics (#1430), generation/setup UX (#1431), and
   ownership/help copy (#1429).
5. PR #1608 is merged and adds the Persona visual renderer capability registry,
   authenticated renderer capability API, and local Buddy renderer registry.

Implemented foundations include:

1. Backend visual-pack persistence in the persona/ChaChaNotes data layer.
2. `PersonaVisualService` for upload validation, asset storage, activation, and
   deactivation behavior.
3. Persona-scoped visual-pack API routes under
   `/api/v1/persona/profiles/{persona_id}/visual-packs`.
4. Frontend service helpers in `apps/packages/ui/src/services/persona-visuals.ts`.
5. `VisualPackEditor` in Persona Garden for pack creation, asset upload,
   manifest editing, activation, deactivation, export, import preview, and
   generated candidate review.
6. Floating `BuddyShellHost` active-pack loading and `SpriteFrameRenderer`
   rendering for sprite/frame packs.
7. Runtime visual state resolution for live voice, tools, recovery, authored
   triggers, and MCP state overrides.
8. Internal MCP module `persona_visuals` with capabilities, bounded runtime
   state trigger, draft-pack creation, manifest update, and generation enqueue
   tools.
9. Jobs-backed generation and pack portability flows.
10. Persona Live E2E coverage for active-pack rendering and broken-pack
    fallback.
11. An `Open Visuals` action in the floating Persona Buddy popover, with active
    persona id propagation and routing through the existing Persona Garden
    helper to `/persona?persona_id=<id>&tab=visuals`.
12. User-facing reliability diagnostics for broken packs, missing assets,
    invalid manifests, and renderer fallback states.
13. Generation setup diagnostics for disabled Jobs, unavailable image providers,
    adapter failures, selected-backend mismatch, and missing default backends.
14. Persona Visuals editor and code-documentation copy that explains
    user-owned assets, one-persona default attachment, manifest-backed packs,
    active versus available pack behavior, import preview/commit, export, and
    generated-candidate review.
15. Same-user duplicate-to-persona flow for copying a pack to another persona
    as a draft that still requires review and activation.
16. Personal library foundation for saving user-scoped metadata references to
    existing visual packs, listing available/source-changed/unavailable entries,
    editing/removing entries, and using an available entry to create a target
    persona draft through duplicate semantics.
17. Import preview conflict choices for target title matches, including
    create-new-draft and reviewed draft replacement while preserving separate
    activation.
18. Persona Garden reusable visual-pack decision surface that routes create
    draft, personal library, import archive preview, and duplicate-to-persona
    actions into the existing controls while preserving draft/review-before-
    activation semantics.
19. Renderer capability reporting through
    `GET /api/v1/persona/visual-renderers`, with only `sprite_frames` enabled
    for V1 validation, activation, import/export, and Buddy runtime rendering.
20. Buddy rendering and diagnostics route through the local renderer registry
    instead of separate hardcoded renderer checks.

---

## 6. Product Principles

1. Persona-owned: visual packs belong to users and attach to one persona by
   default.
2. Pack-based: assets are stored as manifest-backed packs, not loose profile
   blobs.
3. Review-first: generation and imports produce drafts or review candidates
   before activation.
4. Runtime-safe: Persona Live must remain usable when visual packs fail.
5. Discoverable from Buddy: the visible assistant should lead users directly to
   its visual configuration.
6. Renderer-neutral: V1 is sprite/frame rendering, but the contract should not
   block future Live2D or external renderer adapters.
7. VN-informed, not VN-owned: PR #1135's pack portability model is a precedent,
   but Persona Visual Packs are not a VN runtime feature.

---

## 7. Users and Jobs To Be Done

### Primary Users

1. Local-first research users who rely on a live assistant and want persistent
   visual identity.
2. Power users building multiple personas with different roles, moods, and
   assistant behaviors.
3. Users composing MCP-powered workflows where visual state can reflect tool
   progress, speaking, approvals, or errors.

### Jobs

1. "When I use a live persona, I want the visible assistant to match the persona
   I created."
2. "When I see the floating Buddy, I want an obvious way to customize its
   visuals."
3. "When I generate or import assets, I want to review them before they affect
   my active assistant."
4. "When a tool or live session changes state, I want the assistant visual to
   respond without breaking the session."

---

## 8. User Experience

### 8.1 Primary Navigation

The primary visual-pack path is:

```text
Floating Persona Buddy -> Open Visuals -> /persona?persona_id=<id>&tab=visuals
```

Persona Garden remains the main authoring and review surface. The floating
Buddy should not duplicate the full editor; it should provide a direct action
that preserves persona context and opens the existing Visuals tab.

### 8.2 Visuals Tab

The Visuals tab should support:

1. pack listing and current active/draft state.
2. draft pack creation.
3. raster asset upload.
4. manifest editing for state mappings, animations, timing, loops, fallbacks,
   authored triggers, and frame references.
5. validation feedback.
6. generated candidate review.
7. export job queue, status refresh, and authenticated archive download.
8. import-preview upload, status refresh, preview summary, conflicts, warnings,
   proposed plan, explicit conflict choices, and commit flow.
9. explicit activation and deactivation.

### 8.3 Floating Buddy Runtime

The floating Buddy shell should:

1. render the active pack when one is available and valid.
2. reflect live visual states such as idle, listening, thinking, speaking,
   tool-running, approval-needed, wake-armed, error, and offline.
3. fall back to derived/static Buddy summary when no active pack exists or a
   pack cannot render.
4. include a compact `Open Visuals` action when expanded.
5. avoid becoming a second full visual-pack editor.

### 8.4 First-Run and Empty State

The current system exposes the Visuals tab, the Buddy popover entry point, and
diagnostic/setup copy in the Visuals editor:

1. If no active pack exists, Buddy still shows the derived/static summary and
   exposes `Open Visuals` when the selected persona context is known.
2. The Visuals tab empty state makes draft creation the first step, then points
   users toward uploading frames, mapping states, importing/exporting packs,
   queueing generation, reviewing candidates, and activating a valid pack.
3. Missing generation providers and disabled Jobs are shown as setup or
   unavailable states before users queue generation.
4. Ownership/help copy explains that packs are user-owned, attached to one
   persona by default, manifest-backed, and only rendered after explicit
   activation.

---

## 9. Functional Requirements

### FR-1: Persona-Scoped Pack Ownership

Visual packs MUST be owned by the current user and scoped to one persona by
default.

### FR-2: Manifest-Backed Runtime Contract

Each pack MUST have a structured manifest with renderer type, state mappings,
animation definitions, asset references, frame timing, loop behavior, fallbacks,
and optional authored triggers.

### FR-3: Sprite/Frame V1 Renderer

V1 MUST support sprite/frame rendering through the floating Buddy shell. Still
poses, frame sequences, and sprite-sheet frame references are valid V1 shapes.

### FR-4: Explicit Activation

Users MUST explicitly activate a valid pack. Drafts, generated candidates,
imports, and MCP-created changes MUST NOT silently replace the active pack.

### FR-5: Upload and Asset Validation

Uploads MUST validate MIME type, size, dimensions, storage path, checksum, and
persona/pack ownership before assets are attached to a pack.

### FR-6: Persona Garden Visuals Editor

The Visuals tab MUST be the main user-visible authoring and review interface for
visual packs.

### FR-7: Buddy Direct Entry Point

The expanded floating Buddy popover MUST provide a direct action to the selected
persona's Visuals tab.

### FR-8: Runtime Fallback

Persona Live and Buddy controls MUST remain usable when pack list, pack detail,
asset loading, or rendering fails.

### FR-9: Jobs-Based Generation and Portability

Asset generation, pack export, import preview, and import commit SHOULD use Jobs
when work is user-visible, long-running, reviewable, retryable, or needs status
polling.

### FR-10: PR #1135-Aligned Portability

Persona visual pack export/import SHOULD follow the same product pattern proven
by PR #1135 asset-pack portability:

1. queue background work.
2. expose status and warnings.
3. download completed archives through authenticated clients.
4. upload archives for review-only preview.
5. require explicit commit/activation decisions.

### FR-11: Personal Visual Pack Library

Users SHOULD be able to save an existing same-user visual pack into a personal
library as metadata, not as an immediate asset copy. V1 library entries are
user-scoped references to a source persona and source pack. V1 intentionally
does not store display snapshots; listing derives source display names from live
source rows when those rows still resolve.

Saving the same source pack SHOULD be idempotent. Using a library item SHOULD
duplicate the referenced source pack to the chosen target persona as a draft and
MUST NOT activate the target persona. Stale or deleted source entries SHOULD
remain visible as unavailable, MUST be removable, and MUST NOT be usable.

V1 is not a shared marketplace, public library, or cross-user sharing system.
Future duplicate-to-another-persona, import/export, and shared-library features
should reuse the manifest-backed pack format rather than changing the core
asset model.

### FR-12: MCP Runtime State Control

The internal `persona_visuals` MCP module MAY trigger bounded runtime visual
state overrides. Overrides MUST be named-state only, scoped to persona/session,
duration-limited, and auditable.

### FR-13: MCP Durable Changes

MCP durable actions MUST create drafts, manifest updates, generation jobs, or
review items. They MUST NOT silently activate packs.

---

## 10. Non-Functional Requirements

1. Runtime pack loading failure MUST degrade without blocking live controls.
2. Manifest validation MUST reject missing asset references, unsupported
   renderer types, invalid frame timing, fallback cycles, and activation states
   that cannot resolve.
3. Visual runtime state resolution MUST be deterministic for known live/session
   states.
4. Frontend tests MUST cover active-pack rendering, state transitions, fallback,
   and direct Buddy-to-Visuals routing.
5. Backend tests MUST cover persistence, upload validation, activation gating,
   portability jobs, and MCP tool boundaries.
6. E2E tests MUST not depend on real microphone input, external TTS, or external
   image generation.
7. Pack archive handling MUST avoid leaking assets across users or personas.
8. API and job errors MUST return actionable status/failure reasons where
   available.
9. The UI MUST avoid layout breakage when no pack, a broken pack, or missing
   generation providers are present.

---

## 11. Data and API Model

### Core Entities

1. `PersonaVisualPack`
   - id
   - user id
   - persona id
   - title
   - status: draft, active, archived, failed, or review-oriented variants
   - renderer type
   - manifest
   - version
   - provenance
   - parent pack id where relevant

2. `PersonaVisualAsset`
   - id
   - pack id
   - user id
   - persona id
   - storage key
   - asset role
   - MIME type
   - dimensions
   - byte size
   - checksum
   - provenance

3. `PersonaVisualCandidate`
   - id
   - pack id
   - target state
   - generated assets
   - proposed manifest patch
   - review status
   - warning/failure metadata

4. `PersonaVisualLibraryItem`
   - id
   - user id
   - source persona id and source pack id when still available
   - title, notes, and tags
   - live source persona and pack display names when source rows still resolve
   - source pack version and current source version
   - source available/source changed flags
   - version and timestamps

### Current API Surface

Current persona-scoped API routes include:

1. `GET /api/v1/persona/profiles/{persona_id}/visual-packs`
2. `POST /api/v1/persona/profiles/{persona_id}/visual-packs`
3. `GET /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}`
4. `PATCH /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/manifest`
5. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/assets`
6. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/activate`
7. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/deactivate`
8. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/generation-jobs`
9. `GET /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/generated-candidates`
10. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/candidates/{candidate_id}/review`
11. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/export`
12. `GET /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/exports/{job_id}`
13. `GET /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/exports/{job_id}/download`
14. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/import-previews`
15. `GET /api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview_id}`
16. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview_id}/commit`
17. `GET /api/v1/persona/visual-library`
18. `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/library`
19. `PATCH /api/v1/persona/visual-library/{item_id}`
20. `DELETE /api/v1/persona/visual-library/{item_id}`
21. `POST /api/v1/persona/visual-library/{item_id}/use`
22. `GET /api/v1/persona/visual-renderers`

---

## 12. MCP Contract

The internal `persona_visuals` MCP module is the V1 MCP integration point.

Supported tool families:

1. `persona_visuals.capabilities`
2. `persona_visuals.library_items`
3. `persona_visuals.trigger_state`
4. `persona_visuals.create_draft_pack`
5. `persona_visuals.update_manifest`
6. `persona_visuals.use_library_item`
7. `persona_visuals.enqueue_generation`

Rules:

1. Runtime triggers are transient and duration bounded.
2. Durable changes stay draft/review-only until explicit user activation.
3. Tools must resolve persona scope from context or explicit persona id.
4. Tools must reject unknown states, missing pack ids, invalid manifests, and
   unauthorized persona/pack access.
5. Tool outputs that affect runtime state should be visible in live status and
   audit metadata where existing infrastructure supports it.
6. Personal-library MCP tools are user-scoped and reference-backed. Listing
   derives source display names from live rows, and using a library item creates
   an inactive target-persona draft through duplicate semantics.

---

## 13. Pack Portability

Persona visual packs should be portable as `.tldw-persona-vpack` archives.

The portability model should stay aligned with PR #1135:

1. Export is queued as a background job.
2. Export status includes progress, warnings, archive metadata, and download URL
   when complete.
3. Download uses authenticated binary fetch.
4. Import starts with a review-only preview.
5. Preview reports summary, validation warnings, conflicts, proposed plan, quota
   estimate, required choices, and target warnings.
6. Import commit is explicit. It creates a new draft by default, and may
   replace a reviewed draft-like target pack when preview reported that pack as
   replaceable.
7. Activation remains separate from import commit.

Current and future portability/library work:

1. Duplicate to another persona creates a same-user draft copy (#1450).
2. Personal visual-pack library V1 stores user-scoped references to existing
   source packs and creates target drafts through duplicate semantics (#1468).
3. Import/export conflict choices support target title-match review and
   draft-only replacement (#1490).
4. Persona Garden reusable affordances expose duplicate, personal library,
   import, and draft creation paths without marketplace framing (#1493).
5. MCP reusable-pack tools list personal library entries and create
   target-persona drafts without snapshots, sharing, or automatic activation
   (#1496).
6. Future shared user libraries should be layered on top of the manifest-backed
   pack format rather than replacing it.
7. Future cross-device sync, signed community packs, and external
   MCP-compatible pack providers remain out of scope for V1.

---

## 14. Success Metrics

Initial metrics can be instrumentation-light and test-driven:

1. A user can create or import a draft pack from Persona Garden.
2. A user can reach the selected persona's Visuals workflow from floating Buddy
   in one action.
3. Active visual packs render in the floating Buddy for Persona Live.
4. Broken packs do not block live connection or controls.
5. Export/import preview flows complete without unauthenticated downloads or
   silent activation.
6. MCP runtime state triggers do not outlive configured duration limits.

Suggested future product metrics:

1. Percent of live-persona users who open Visuals from Buddy.
2. Draft pack creation to activation conversion rate.
3. Import preview to commit conversion rate.
4. Visual-pack render failure rate.
5. Generation candidate accept/reject/edit rate.

---

## 15. Rollout Plan

### Phase 0: Foundation - Complete

Merged foundation includes storage, service, API, frontend primitives, Buddy
runtime rendering, Visuals editor, Jobs generation/review, internal MCP module,
portability flows, and E2E coverage.

### Phase 1: Direct Buddy Entry - Complete

PR #1412 merged the floating Buddy `Open Visuals` action and focused tests.

### Phase 2: Product Hardening - Complete

Completed Product Hardening includes:

1. Reliability diagnostics for broken packs, missing assets, invalid manifests,
   and renderer fallback states (#1430).
2. Unavailable-provider and generation setup states for disabled Jobs, missing
   image providers, adapter failures, selected-backend mismatch, and missing
   default backends (#1431).
3. Import commit controls in the Visuals editor, with reviewed import commits
   creating draft packs rather than silently activating them.
4. Docs/help copy explaining ownership, one-persona default attachment,
   manifest-backed packs, active-pack semantics, import preview/commit, export,
   and generated-candidate review (#1429 / PR #1447).
5. Closed hardening tracker #1428.

### Phase 3: Library and Externalization

Phase 3 is optional product work beyond the original Persona/Buddy visual-pack
baseline. The first reference-backed V1 slices are now covered:

1. Duplicate pack to another persona: complete for same-user draft duplication
   (#1450).
2. Personal visual-pack library: complete for user-owned, reference-backed
   save/list/edit/remove/use in Persona Garden (#1468).
3. Import/export conflict choices: complete for preview-backed target title
   conflicts, create-new draft commits, and reviewed draft replacement (#1490).
4. Persona Garden reusable affordances: complete for routing duplicate,
   personal library, import, and draft creation flows without marketplace
   framing (#1493).
5. MCP reusable-pack semantics: complete for listing reference-backed personal
   library entries and creating inactive target-persona drafts (#1496).
6. Shared/cross-device libraries remain future work.
7. External MCP-compatible visual providers remain future work.
8. Renderer/provider adapter evaluation for Live2D and other future paths is
   tracked by #1497 and the 2026-05-10 design evaluation.
9. Renderer capability registry/API and Buddy renderer registry are complete in
   PR #1608, with `sprite_frames` still the only enabled V1 runtime renderer.
10. Live2D or other renderer adapter implementation remains future work after
    non-sprite manifest V2, import-preview validation hooks, dependency gates,
    licensing review, and fallback requirements are separately scoped.

---

## 16. Risks and Mitigations

1. Risk: Users confuse VN assets with Persona Buddy visuals.
   - Mitigation: Keep primary UI, docs, and navigation in Persona Garden and
     Buddy. Treat VN only as portability precedent.

2. Risk: Generated assets silently change the active assistant.
   - Mitigation: Require review and explicit activation.

3. Risk: Broken packs degrade Persona Live.
   - Mitigation: Runtime fallback to derived/static Buddy and test failed pack
     loading.

4. Risk: Future renderer types force a schema rewrite.
   - Mitigation: Keep manifests versioned and route renderer support through
     the capability registry before activation/import/runtime support is
     exposed.

5. Risk: Cross-persona asset leakage.
   - Mitigation: Enforce user/persona/pack ownership checks at API, service, and
     storage levels.

6. Risk: The editor becomes a full image editor.
   - Mitigation: Keep V1 focused on upload, preview, state mapping, timing,
     review, activation, and portability.

---

## 17. Testing and Verification Requirements

Frontend:

1. Buddy shell loads and renders active visual packs.
2. Buddy shell falls back when active visual pack loading fails.
3. Buddy popover links to `/persona?persona_id=<id>&tab=visuals`.
4. Visual state resolver covers live, tool, wake, recovery, authored trigger,
   and MCP override cases.
5. Sprite renderer handles frame selection and invalid frame/asset cases.
6. VisualPackEditor covers creation, upload, manifest edits, activation,
   generation review, export, import preview, and import commit where surfaced.

Backend:

1. Manifest validation rejects broken references, fallback cycles, and invalid
   renderer/state data.
2. Upload validation rejects invalid files and unsafe storage keys.
3. Activation requires a valid renderable baseline.
4. Export/import-preview/import-commit jobs preserve ownership and review
   semantics.
5. MCP tools enforce persona scope, state bounds, duration bounds, and draft-only
   durable edits.

E2E:

1. Persona Live renders an active visual pack through mocked backend fixtures.
2. Persona Live remains usable when pack loading fails.
3. Direct Buddy to Visuals routing is covered in focused frontend tests and can
   be promoted to E2E if routing regressions recur.

---

## 18. Open Product Questions

1. Current behavior: import commit creates a reviewed draft pack only. Future
   question: should the UI add an optional "activate now" step after validation?
2. Should empty Buddy state show a stronger first-run affordance when no active
   pack exists, beyond the current derived/static Buddy summary plus `Open
   Visuals` link?
3. What are the default upload quotas per user, persona, pack, and archive?
4. Which image generation provider path should be the default first-run
   recommendation?
5. Should users be able to mark packs as reusable templates before shared
   libraries exist, or does the personal reference-backed library cover that
   near-term need?
6. Which visual state names should become externally documented for future MCP
   providers?

---

## 19. Durable References

1. Historical implementation spec:
   `Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md`
2. Implementation plan:
   `Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md`
3. Buddy workflow entry plan:
   `Docs/superpowers/plans/2026-05-09-persona-buddy-visual-workflow-entry-plan.md`
4. PR #1393: Add persona visual packs and portability flow.
5. PR #1400: Persona visual closeout docs/E2E.
6. PR #1412: Expose persona visuals from Buddy shell.
7. Issue #1410: Expose Persona Buddy visual packs in live assistant.
8. PR #1447 / issue #1429: Persona visual pack ownership/help copy.
9. Issue #1428: Completed Persona/Buddy visual-pack Product Hardening tracker.
10. Issue #1450: Same-user Persona Visual pack duplicate-to-persona draft flow.
11. Issue #1468: Personal Persona Visual pack library foundation.
12. Issue #1490: Persona visual-pack import conflict choices.
13. Issue #1493: Persona Garden reusable affordances.
14. Issue #1496: MCP reusable-pack semantics for reference-backed personal
    library entries.
15. Issue #1497: Persona visual-pack renderer/provider adapter evaluation.
16. Renderer/provider adapter evaluation:
    `Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md`
17. PR #1608: Persona Buddy renderer capability registry.
18. Issue #1609: Renderer capability docs and tracker refresh after PR #1608.
