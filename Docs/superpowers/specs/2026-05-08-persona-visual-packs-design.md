# Persona Visual Packs Design

Date: 2026-05-08
Status: Approved for spec review
Owner: Codex brainstorming pass
Backlog: TASK-125

## Summary

Extend the existing Persona Buddy and Persona Live assistant surfaces into a user-owned animated 2D assistant system.

V1 treats the current floating Persona Buddy shell as the primary animated assistant surface. Users can upload or generate sprite/frame-based visual packs, map pack assets to named assistant states, preview and edit those mappings, and explicitly activate a reviewed pack for one persona. The pack format is manifest-driven and renderer-neutral enough to support a later Live2D adapter, but the first implementation target is a practical sprite/frame-pack renderer.

The system should also expose an internal `persona_visuals` MCP module. MCP tools may trigger safe transient visual states during a live session and may create or propose durable draft asset changes, but they must not silently replace the active visual pack.

## Goals

- Let users create or upload their own animated 2D assistant/persona visuals.
- Attach visual assets to one persona by default while storing them as portable packs with manifests.
- Upgrade the existing floating Persona Buddy shell into the animated assistant surface.
- Keep Persona Live, Persona Buddy, and MCP tool execution on their existing runtime foundations.
- Provide a pack editor for upload/generation review, preview, timing, loop settings, fallback chains, and required state mapping.
- Use background Jobs for generated assets and heavier asset processing, with a user review step before activation.
- Add an internal MCP module for safe visual asset draft changes and bounded runtime state control.
- Preserve a renderer-neutral state contract so Live2D or external MCP-compatible visual providers can be added later.

## Non-Goals

- Do not create a separate avatar/persona identity outside the existing persona model.
- Do not replace inline persona avatars or profile pictures in chat/list surfaces.
- Do not ship Live2D support in the first implementation target.
- Do not allow user-supplied JavaScript, arbitrary SVG animation logic, or executable visual plugins in V1.
- Do not let MCP tools silently replace the active assistant visual pack.
- Do not build shared visual libraries, marketplace semantics, or cross-persona duplication in V1, beyond preserving a manifest format that can support them later.
- Do not require real microphone, real TTS, or external image-generation services for baseline runtime tests.

## Existing Context

Relevant repo context:

- `Docs/superpowers/specs/2026-03-31-persona-buddy-facet-design.md`
- `Docs/superpowers/specs/2026-03-31-persona-buddy-track-b-floating-shell-design.md`
- `Docs/superpowers/specs/2026-04-30-persona-wake-word-support-design.md`
- `apps/tldw-frontend/pages/persona.tsx`
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- `apps/packages/ui/src/components/PersonaGarden/`
- `apps/packages/ui/src/components/Common/PersonaBuddy/`
- `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
- `apps/packages/ui/src/services/persona-stream.ts`
- `tldw_Server_API/app/api/v1/endpoints/persona.py`
- `tldw_Server_API/app/core/Persona/buddy.py`
- `tldw_Server_API/app/core/MCP_unified/`

The current system already has the foundations this feature should reuse:

- Persona profiles expose `buddy_summary` and avatar-related fields.
- Persona Buddy is a persona-owned visual facet, not a separate companion identity.
- A floating Buddy shell already exists in the shared UI package.
- Persona Live already streams state through `/api/v1/persona/stream`.
- The live controller already distinguishes idle, listening, thinking, speaking, error, wake, action, and tool-related state.
- Persona turns already use existing MCP policy, scope, approvals, audit metadata, and tool execution.
- User-visible long-running work should use Jobs when it needs progress, status, retries, admin controls, or review.

The missing layer is not a new persona runtime. It is a visual asset and animation system that can subscribe to the existing persona/live state and can be safely extended through MCP.

## User-Confirmed Product Rules

1. V1 should support both uploaded assets and AI-generated pose/animation assets.
2. V1 should use a generic animation-state contract now, with pose/frame packs first and Live2D later.
3. The first MCP target is an internal `persona_visuals` module; an external MCP-compatible contract can follow later.
4. MCP should support both durable asset-management actions and transient runtime visual state control, but through named states, permission checks, and audit events.
5. The first render target is the existing floating Persona Buddy shell.
6. Runtime defaults and user-authored triggers should both be supported.
7. Baseline state mapping is required so the assistant never goes blank.
8. Generated assets should run as background Jobs and require review before activation.
9. Assets are user-owned and attached to one persona by default, but stored as packs with manifests so later duplication, import/export, or shared libraries do not require changing the core format.
10. V1 should include a pack editor, not only a minimal mapping form.

## Recommended Approach

Implement a manifest-backed `PersonaVisualPack` layer beneath the existing Persona Buddy shell.

The shell remains the visible surface. Persona Buddy remains the persona-owned identity facet. Persona Live remains the runtime for voice, turn execution, tool calls, and wake behavior. The new visual-pack layer supplies custom animated render data and a resolver that maps existing live/session state to active animations.

This approach is recommended because it:

- builds on existing persona-owned buddy semantics
- keeps one assistant/persona runtime instead of creating a parallel avatar runtime
- lets v1 deliver immediate value through sprite/frame packs
- keeps expensive generation and processing behind Jobs
- allows MCP composability without granting silent control over active assets
- leaves a clean adapter point for Live2D, import/export, shared libraries, and external MCP later

## Architecture

### Persona Visual Pack

A visual pack is a persona-scoped, user-owned bundle of assets plus a manifest.

Day-one pack metadata should include:

- pack id
- user/owner id
- persona id
- status: draft, review, active, archived, failed
- renderer type: `sprite_frames` for V1
- manifest version
- active/draft relationship
- provenance: uploaded, generated, imported, mixed
- timestamps

A pack is attached to one persona by default, but it should not be modeled as an unstructured blob inside the persona profile. It should have stable pack identity so future duplicate-to-persona, import/export, shared libraries, and asset provenance can be added without changing runtime consumers.

### Persona Visual Asset

Individual visual files should be tracked separately from pack metadata.

Day-one asset metadata should include:

- asset id
- pack id
- owner id
- persona id
- filename or storage key
- media type
- byte size
- checksum
- dimensions
- asset role: frame, still pose, sprite sheet, preview, generated candidate
- validation status and failure reason
- provenance metadata

Storage details should sit behind a service boundary. API consumers should not need to know whether the first implementation stores files on disk, in an app-managed media directory, or behind a future object-store adapter.

### Manifest Contract

The manifest is the central runtime contract. It should be structured, versioned, and validated.

V1 manifest fields should cover:

- manifest version
- renderer type
- required state mappings
- optional authored triggers
- animation definitions
- asset references
- frame order
- sprite-sheet regions when applicable
- frame rate
- loop mode
- alignment anchors
- preview metadata
- fallback chains
- capability flags

The manifest should not be a freeform prompt or UI-only document. It is the contract shared by backend validation, frontend preview, the Buddy shell renderer, and MCP tools.

### Visual State Resolver

The visual state resolver maps existing assistant/session signals to named visual states.

Recommended V1 state map:

- `idle`
- `wake_armed`
- `listening`
- `thinking`
- `speaking`
- `tool_running`
- `approval_needed`
- `error`
- `offline`

The resolver should always return a renderable result. If the active pack does not contain an exact animation for a state, it follows the manifest fallback chain, then `idle`, then the existing derived/static Buddy rendering.

Persona Live must not fail because a visual pack is incomplete or broken.

### Renderer

The first renderer is `sprite_frames`.

It should support:

- still image poses
- ordered frame sequences
- sprite-sheet regions
- loop settings
- frame rate
- alignment anchors
- preview frame
- state-specific fallback

Animated WebP/GIF imports may be accepted if they can be validated and normalized into the V1 renderer contract. They should not become a special runtime path that bypasses manifest validation.

Live2D is deferred. The state contract and renderer host should leave room for a future `live2d` renderer type, but V1 should not build that adapter.

## Backend Design

### Service Boundary

Add a `PersonaVisualService` or equivalent persona-domain service that owns:

- pack creation
- draft/revision handling
- asset ingestion and validation
- manifest validation and normalization
- activation and rollback
- generated candidate review
- pack capability projection
- storage adapter access

The service should prevent endpoints, Jobs, and MCP modules from each implementing their own validation paths.

### API Surface

The API should stay persona-scoped.

Recommended endpoint shape, adjusted to match existing route naming conventions during implementation:

- `GET /api/v1/persona/profiles/{persona_id}/visual-packs`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs`
- `GET /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/assets`
- `PUT /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/manifest`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/validate`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/activate`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/generation-jobs`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/generated-candidates/{candidate_id}/accept`
- `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/generated-candidates/{candidate_id}/reject`

Activation should be explicit. A draft pack can be saved while invalid, but activation should require every baseline state to resolve to a valid animation or valid fallback.

### Jobs Integration

Generated asset creation and heavier processing should use Jobs.

A generation job should produce:

- status
- progress where available
- failure reason
- generated assets
- proposed manifest patch
- review metadata

The job should not mutate the active pack. Its output becomes a review item that the user can accept, edit, or reject. Accepting the item merges the proposed assets and manifest changes into a draft pack or active-pack draft revision, depending on the eventual revision policy.

## Frontend Design

### Animated Buddy Shell

The existing Buddy shell host becomes the animated assistant surface for supported desktop layouts.

The shell should:

- consume active persona context as it does today
- load the active visual pack summary when available
- pass Persona Live/session state into the visual state resolver
- render via the active renderer
- fall back to the current derived/static Buddy representation when needed
- avoid taking over the live voice panel or command editor

The shell should remain separate from inline persona avatars. Existing `avatar_url` and profile-picture surfaces continue to render as they do today unless a future plan explicitly changes them.

### Pack Editor

The V1 editor should live in the Persona Garden/persona detail area rather than inside the live voice control card.

It should support:

- pack list and active/draft status
- upload assets
- request generated poses or animations
- preview draft and active pack states
- assign required states
- configure frame order
- configure frame rate and loop mode
- configure alignment anchors
- set fallback chains
- view validation errors
- test named states
- review generated candidates before accepting
- activate a valid pack
- revert/deactivate to derived Buddy rendering

The editor should be stronger than a minimal mapping form, but it should not become a full image editor. Cropping, onion-skin tools, state-machine authoring, and prompt-assisted full visual authoring are deferred.

### Runtime UI Feedback

The live session UI should expose enough information to make visual state control understandable:

- current visual state
- transient state override when active
- tool/action visual state source when relevant
- generated asset job status and review availability, where appropriate
- recovery messaging when a pack fails to render and the shell falls back

This should stay compact. The live panel should not become the main asset-management surface.

## MCP Design

Add an internal `persona_visuals` MCP module first.

Recommended V1 tool families:

- query active pack capabilities
- trigger a named runtime visual state for a bounded duration
- create a draft visual pack
- add uploaded/generated asset references to a draft pack
- propose or update state mappings in a draft manifest
- enqueue a generation request
- query generation/review status

Transient runtime controls should be:

- named-state only
- bounded by duration limits
- scoped to persona and session
- visible in audit/status UI
- lower priority than safety-critical runtime states such as error/recovery

Durable controls should create drafts or review items. They should not silently replace the active pack. A later trust setting may allow narrower automation, but V1 should require user review for generated assets and explicit activation for durable visual replacement.

External MCP compatibility is deferred. The internal module should still document a stable enough contract that future external servers can compose with the same state names, draft semantics, and review model.

## Data Flows

### Live Runtime Flow

1. A supported persona-aware page mounts the Buddy shell.
2. The shell resolves the active persona.
3. The shell loads buddy summary and active visual pack summary.
4. Persona Live emits session state through the existing frontend controller.
5. The visual state resolver normalizes live/session/tool/wake/error state into a named visual state.
6. The renderer asks the active pack for the best animation for that state.
7. If no valid animation exists, the resolver applies fallback and then derived/static Buddy rendering.

### Authoring Flow

1. The user opens the pack editor for a persona.
2. The user creates a draft pack or edits a draft revision.
3. The user uploads assets or accepts generated candidates into the draft.
4. The editor validates and previews the manifest.
5. The user maps required states and configures timing, looping, alignment, and fallback.
6. The user tests named states locally.
7. The user activates the pack after validation succeeds.

### Generation Flow

1. The user requests a pose or animation variation from the editor or through MCP.
2. The backend creates a user-visible Job.
3. The Job generates assets and a proposed manifest patch.
4. The system stores the result as a candidate requiring review.
5. The user previews, accepts, edits, or rejects the candidate.
6. Accepted candidates update a draft pack, not the active pack directly.

### MCP Flow

1. A trusted internal module or persona turn asks `persona_visuals` for capabilities or a transient state trigger.
2. Policy and persona/session scope are checked.
3. Runtime triggers emit bounded state overrides and audit metadata.
4. Durable asset changes create drafts, generation Jobs, or review candidates.
5. The live shell reflects accepted active-pack changes only after explicit activation.

## Error Handling And Safety

Custom visuals are data, not executable extensions.

V1 should accept validated raster/image assets and structured manifests. It should reject user-supplied scripts, arbitrary SVG animation logic, invalid MIME types, oversized files, unsafe dimensions, bad checksums, and unsupported renderer types.

Manifest validation should reject:

- missing asset references
- unsupported renderer types
- impossible frame ranges or sprite coordinates
- missing dimensions
- invalid frame rates
- fallback cycles
- baseline states that cannot resolve during activation
- malformed authored triggers

Draft saves may preserve invalid work so users can continue editing. Activation must be stricter.

Runtime failures should degrade in this order:

1. exact state animation
2. manifest fallback
3. `idle`
4. derived/static Buddy shell
5. dormant shell with clear non-blocking recovery state

Generated jobs should expose failure reasons and leave the active pack untouched. Rejected generated candidates should remain auditable and removable, but should not appear in runtime.

## Security And Privacy

Assets are user-owned. Access checks should enforce owner and persona scope for pack reads, writes, uploads, generation results, and activation.

Upload handling should include:

- MIME validation
- extension checks as supporting evidence, not the only validation
- size limits
- dimension limits
- checksum tracking
- quota hooks
- sanitized filenames or storage keys
- no secret or token logging

MCP runtime state triggers should include persona id, session id, triggering tool/module, state name, duration, and reason in audit metadata where the existing MCP audit model supports it.

The design should avoid cross-persona asset exposure in V1. Future shared libraries or duplicate-to-persona features should be explicit copy/share operations, not accidental visibility through storage layout.

## Testing Strategy

Backend tests should cover:

- manifest parsing
- required state resolution
- fallback chains and cycle rejection
- invalid asset references
- upload validation
- owner/persona access checks
- draft vs active behavior
- activation gating
- generation candidate accept/reject transitions
- rollback/deactivate behavior

Frontend tests should cover:

- pack editor required-state mapping
- frame order, frame rate, loop, and fallback editing
- invalid manifest feedback
- preview/test-trigger behavior
- activation disabled until valid
- shell state changes for listening/thinking/speaking/tool/error fixtures
- shell fallback when the active pack is missing or broken

MCP tests should cover:

- capability queries
- transient state duration limits
- invalid state rejection
- persona/session scoping
- audit metadata
- durable changes becoming drafts or review candidates
- active pack remaining unchanged until explicit activation

E2E tests should extend existing Persona Live coverage with mocked/fixture visual states. They should not depend on real microphone input, external TTS, or external image generation.

## Rollout Plan

1. Add storage, schema, service, and API support for visual packs, assets, manifests, and activation.
2. Add renderer-neutral visual state resolver and `sprite_frames` renderer in the existing Buddy shell.
3. Add the Persona Garden pack editor for upload, preview, state mapping, timing, fallback, and activation.
4. Add Jobs-backed generation and candidate review.
5. Add internal `persona_visuals` MCP tools for draft asset changes and transient runtime state triggers.
6. Later, add import/export, duplicate-to-persona, shared libraries, external MCP compatibility, and Live2D renderer support.

The first implementation plan should stop after a useful v1: uploaded/generated sprite/frame packs, pack editor, shell runtime integration, Jobs review, and internal MCP tools.

## Open Questions For Implementation Planning

- Which storage adapter should visual assets use first: existing app media storage, persona-specific filesystem storage, or a new shared asset-store helper?
- Should active pack edits create immutable revisions, mutable draft revisions, or a hybrid revision model?
- What exact upload limits should V1 enforce for file size, dimensions, frame count, and total per-persona quota?
- Which image generation provider path should be wired first, and how should unavailable provider states appear in the editor?
- How much of the generated candidate review UI should be reusable for future media-generation review flows?
- Should authored triggers bind only to generic visual states in V1, or also to tool categories such as search, notes, media, and web?

## Acceptance Criteria For V1 Planning

- The implementation plan preserves the existing Persona Buddy and Persona Live runtime boundaries.
- The first renderer target is sprite/frame packs.
- The manifest contract is versioned, structured, and validated on the backend.
- The frontend shell can always fall back to derived/static Buddy rendering.
- The pack editor supports preview, mapping, timing, loop settings, fallback chains, and generated candidate review.
- Generated assets run through Jobs and require review before activation.
- Internal MCP tools can trigger bounded runtime states and create draft/review changes, but cannot silently replace the active pack.
- Tests cover backend validation, frontend editor/runtime behavior, MCP policy boundaries, and Persona Live visual state fixtures.
