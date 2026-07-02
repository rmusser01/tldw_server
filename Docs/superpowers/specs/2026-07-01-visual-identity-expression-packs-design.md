# Visual Identity Expression Packs Design

Date: 2026-07-01
Status: Approved for spec review
Owner: Codex brainstorming pass
Backlog: TASK-12089

## Summary

Add shared Visual Identity Packs for character/persona chat. A pack is a reusable, manifest-backed set of expression assets that can be associated with a character or persona by default, resolved automatically in chat, and later bound to a VN role.

V1 focuses on practical chat value: SillyTavern-style expression ZIP import, manual expression asset editing, animated raster image support, chat portrait switching, and a basic single-character VN-style stage panel. It deliberately avoids full VN asset generation and full VN scene composition, but it stores enough actor, expression, anchor, crop, scale, and version metadata for later VN casting.

## Goals

- Let users attach expression/sprite packs to characters and personas.
- Resolve the selected character/persona default visual pack automatically in chat.
- Support SillyTavern-compatible expression names as import aliases while keeping internal canonical expression keys.
- Allow custom expression labels beyond the compatibility baseline.
- Support raster expression assets, including animated GIF and WebP, with guarded AVIF support when the backend can validate it safely.
- Preserve animated source files in V1 instead of destructively transcoding them.
- Add a chat portrait renderer and a basic single-character stage view that use the same expression resolver.
- Support manual expression control through an expression picker and a client-side `/emote <expression>` command.
- Preserve current mood detection as the default automatic expression signal when no stronger expression source is active.
- Keep existing Persona Visual Pack and character mood image behavior working without a required migration.
- Prepare for later VN flows where visual identities can be cast into roles or filled from VN asset generation.

## Non-Goals

- Do not implement VN asset generation as part of this V1.
- Do not implement full multi-character VN scene blocking, backgrounds, CG layers, or script timing in chat.
- Do not replace the existing Persona Visual Pack APIs in V1.
- Do not require migration of current Persona Buddy visual packs.
- Do not store expression packs as unstructured blobs inside character card JSON.
- Do not support SVG, user-supplied JavaScript, executable visual plugins, Live2D, Rive, or Lottie in V1.
- Do not infer model-directed expressions by regex over assistant prose.
- Do not silently mutate historical chat appearance when a pack is edited later.

## Existing Context

The repository already has three relevant foundations:

- Persona Visual Packs are manifest-backed, user-owned visual packs for Persona Buddy and Persona Live. They use the `sprite_frames` renderer, state catalogs, renderer capabilities, fallback behavior, and asset metadata.
- VN Asset Packs are a separate generated/reviewed asset system for future visual-novel workflows. They already model sprites, labels, durable asset records, and runtime manifests.
- Character/persona chat already persists mood metadata and has a lightweight character mood image path stored in character extensions.

The practical gap is not rendering from scratch. The missing layer is a shared identity-level expression pack that can be used by character chat, persona chat, and future VN role binding without creating a third unrelated asset model.

## User-Confirmed Product Rules

1. V1 should start with expression packs for chat rather than VN role binding or generation.
2. Visual packs should be shared identity resources that can attach to characters, personas, and later VN roles.
3. Runtime expression selection should be hybrid: manual UI and `/emote` controls first, current mood detection as automatic fallback, and a schema path for future model-directed or server-classified expressions.
4. SillyTavern-compatible expression slots should be the compatibility baseline, with custom labels and aliases allowed.
5. Pack creation V1 should support ZIP import and manual editing first. VN generation should be prepared for, not implemented.
6. V1 rendering should include both chat portrait integration and a basic VN-style stage view.
7. A selected character or persona should apply its associated default visual pack automatically.

## Recommended Approach

Introduce shared Visual Identity Packs behind a compatibility boundary.

The implementation should reuse the existing Persona Visual Pack concepts where they fit: manifest validation, `sprite_frames` rendering, asset metadata, fallback behavior, renderer capability reporting, and storage service boundaries. It should not force a broad rename or migration of existing Persona Visual APIs in V1. Existing Persona Buddy users keep their current behavior, while new shared APIs can read/write compatible manifests and eventually become the common substrate.

This approach is preferred over a character-only expression layer because it avoids duplicating asset validation, renderer logic, and manifest semantics. It is also preferred over making VN Asset Packs the canonical home in V1 because chat expression packs should not depend on the heavier VN generation/review workflow.

## Core Concepts

### Visual Identity Pack

A user-owned visual pack containing expression slots, raster assets, and a validated manifest.

Important metadata:

- pack id
- owner user id
- title and description
- source type: imported, uploaded, generated, mixed
- renderer type: `sprite_frames` in V1
- current active version id
- status: draft, active, archived, failed
- compatibility metadata for SillyTavern imports
- created and updated timestamps

### Pack Version

Activated pack versions are immutable by default. Editing an active pack creates a draft from the current version. Activating that draft creates a new version.

This prevents accidental changes to historical chat appearance. Bindings can point to "latest active" for new sessions, but message history stores the resolved pack version and asset id.

### Expression Slot

An expression slot is the semantic state the renderer resolves to.

Fields should include:

- canonical key, such as `neutral`, `happy`, `sad`, or a custom key
- display name
- aliases
- optional SillyTavern source name
- optional tags for future VN/model use
- default asset id for the slot
- fallback expression key

SillyTavern names are import aliases, not hard-coded runtime names everywhere. Model output, chat UI, and future VN scripts should resolve through the canonical key and alias table.

### Visual Asset

Each uploaded or imported raster file is tracked separately from pack metadata.

Fields should include:

- asset id
- pack id and pack version or draft id
- owner user id
- storage reference
- source filename
- MIME type
- byte size
- checksum or content hash
- dimensions
- animation flag
- frame metadata when cheaply available
- preview asset id or preview frame metadata when available
- crop, anchor, and scale metadata
- validation status and failure reason

Animated source files should be stored as originals in V1. Crop, anchor, and scale are applied at render time rather than destructively editing the source.

### Visual Binding

A binding connects a visual pack to a target identity.

Supported V1 binding targets:

- character
- persona

Reserved future target:

- VN role

One binding can be marked as default for the target. Chat/VN explicit overrides outrank defaults. Default resolution is:

`chat or VN explicit override -> selected character/persona default pack -> legacy mood image/static avatar -> neutral placeholder`

### Expression State

Expression state records what expression was selected, why, and how confidently.

Fields should include:

- actor identity reference
- pack id
- pack version id when resolved
- expression key
- resolved asset id when available
- source: manual, mood_detection, model_directed, server_classifier, fallback
- confidence
- fallback reason
- created timestamp

The state is actor-scoped even though V1 stage mode shows one character/persona. That avoids a poor migration path for later multi-character VN scenes.

## Import And Asset Handling

### ZIP Import

SillyTavern-style ZIP import should create a draft pack first, not immediately activate a pack.

The importer should:

- scan filenames and folders
- normalize case and separators
- map known expression names through the alias table
- create custom expression slots for unrecognized names
- detect collisions deterministically
- store per-entry warnings and errors
- show the user a draft review screen before activation

Collision handling must be stable. If a ZIP contains multiple candidate files for the same expression, the draft should record the collision and require user choice unless a deterministic configured rule can safely choose one. No file should be silently discarded without a warning.

Default expression selection follows this order:

1. Exact or alias match for `neutral`, `default`, or `normal`.
2. User choice during draft review.
3. Placeholder fallback.

The importer must not choose a random file as the default.

### Manual Editing

Users can upload, replace, clear, and remap individual expression assets in a draft. Manual edits use the same validation path as ZIP import. Active packs are edited by creating a draft version.

### Asset Validation

Allowed V1 asset classes are raster images only.

Baseline formats:

- PNG
- JPEG
- WebP
- GIF

Conditional format:

- AVIF, only when backend capability checks confirm safe MIME validation, dimension extraction, and storage/render handling.

Unsupported formats, including SVG, are rejected in V1.

Validation should enforce:

- per-file byte limit
- total pack byte limit
- maximum dimensions
- MIME sniffing
- allowed extension checks
- image decode/dimension checks
- owner and quota checks
- animation detection where supported
- clear unsupported-media errors

Animated GIF, WebP, and AVIF files should not be resized or transcoded in V1 unless the pipeline explicitly preserves animation. Preview extraction is useful but non-blocking: if preview extraction fails and the original asset validates, the draft can continue with a generic thumbnail warning.

### Archive Safety

ZIP handling must defend against hostile and messy archives.

Reject or quarantine:

- path traversal entries
- absolute paths
- symlink entries
- suspicious duplicate paths
- excessive entry counts
- excessive decompression ratio
- nested archives by default
- files over configured limits
- files with invalid MIME or extension mismatch

Partial import is allowed, but skipped files and warnings are first-class data in the draft result. Invalid files are never referenced by an active manifest.

### Capabilities

The backend should expose capability metadata for supported formats, size limits, renderer support, and conditional AVIF availability. The frontend uses these capabilities to configure upload affordances and messaging. User-specific quota or storage details require normal authentication and should not leak filesystem or deployment internals.

## Runtime Expression Resolution

Runtime resolution should be deterministic and side-effect-light.

Priority order:

1. Session-level manual expression override.
2. Client-side `/emote <expression>` override.
3. Trusted structured message metadata for future model-directed expression output.
4. Future server classifier output.
5. Existing client-side mood detection when confidence meets threshold.
6. Pack default expression.
7. Legacy mood image or static avatar.
8. Neutral placeholder.

Manual picker and `/emote` overrides are sticky for the current session until cleared or changed. Per-message metadata records what displayed at the time, but session overrides control future messages.

Mood detection should use thresholds to prevent expression flicker. Low-confidence mood detection should resolve to the pack default or `neutral`.

Model-directed expression control must use trusted structured response metadata or a tool/schema contract. It should not be inferred from assistant prose.

### `/emote` Command

`/emote <expression>` is a client command parsed before the message is sent to the model. It resolves through canonical keys and aliases. If users need to send literal text beginning with `/emote`, the input should support an escape behavior.

The command should:

- set the current session expression override
- persist an expression event or session state update
- not send command syntax to the model by default
- return a clear UI error when the expression cannot be resolved

## Chat UI

### Chat Portrait

When a selected character/persona has a resolved visual pack, the chat portrait uses the active expression asset instead of the static avatar. If resolution fails, it falls back through the normal chain and records a fallback reason.

### Stage View

V1 stage view is a toggleable single-character focused sprite panel inside chat. It uses the same resolved expression as the portrait path and is intentionally not a full VN compositor.

Out of scope for this panel:

- multiple actors
- backgrounds
- CG/event layers
- blocking or positions for multiple sprites
- script timing

The stage should respect reduced-motion preferences and provide a static-preview path for animated assets when appropriate.

### Expression Picker

The expression picker should show:

- canonical names
- alias search
- preview thumbnails
- custom labels
- missing asset warnings
- unsupported asset warnings
- clear override action

Dense menus should prefer static previews instead of loading every full animated file.

## API And Persistence

The shared API should live under `/api/v1/visual-identities/...`. Existing Persona Visual endpoints remain under their current persona route group for compatibility.

Recommended API areas:

- capabilities
- pack CRUD
- draft import
- draft asset upload/update/delete
- pack activation and archival
- character/persona binding management
- chat-session pack and expression overrides
- resolver responses for active expression assets
- permission-checked asset serving or short-lived signed URLs

ZIP imports should use Jobs for draft creation, validation, preview extraction, progress reporting, cancellation, and retry visibility. Manual single-asset edits can remain synchronous if validation is cheap.

Pack manifests and activated versions are immutable. Drafts are mutable. Activation creates a new version.

Bindings may use "latest active" for new sessions, but message history must persist:

- actor identity
- pack id
- pack version id
- expression key
- resolved asset id when available
- fallback reason when applicable

This lets old chat history replay predictably after pack edits.

Visual pack asset URLs are user-owned content. They should go through authenticated API access or short-lived signed URLs, consistent with existing project storage conventions.

## Compatibility

Existing Persona Visual Pack endpoints remain supported in V1. New shared APIs may read/write compatible manifests, but current Persona Buddy users should not need a migration.

Existing character mood image data remains a legacy fallback. New expression packs should not be stored in character card extension blobs.

Character/persona export in V1 should include binding metadata by default, but not binary assets unless the user explicitly requests an "include visual assets" option. Imports without bundled assets should mark bindings unresolved instead of silently breaking.

## VN Bridge

V1 should store enough metadata for later VN casting:

- actor identity
- expression key
- asset role
- anchor
- crop
- scale
- default sprite behavior
- pack id and version id

Two future flows should be enabled by this shape:

1. VN Asset Pack to Visual Identity draft pack: VN generation can fill missing expression slots or create a draft identity pack for review.
2. Visual Identity Pack to VN actor asset resolver: VN playback can cast a character/persona into a role and resolve expression assets from that pack.

V1 should not imply that VN generation already knows how to fill these slots.

## Error Handling

Errors should be explicit and user-actionable.

Examples:

- unsupported media type
- AVIF unavailable on this server
- image dimensions exceed limit
- archive entry rejected for safety
- duplicate expression candidates need review
- expression alias not found
- active pack version is immutable
- binding target not found or not owned by user
- asset missing or deleted
- asset exists but cannot be rendered in current capability mode

Resolver responses should include fallback reasons rather than silently returning a different asset.

## Testing Strategy

Backend tests:

- manifest validation for canonical expressions, aliases, custom slots, fallback rules, version immutability, and binding resolution
- compatibility fixtures for existing Persona Visual Pack manifests
- compatibility fixtures for legacy character mood images
- ZIP import safety for traversal, absolute paths, symlinks, nested archives, duplicate paths, entry count, decompression ratio, bad MIME, over-limit dimensions, collision handling, and partial-import warnings
- property or fuzz-style importer tests for weird filenames, aliases, duplicate paths, casing, nested folders, and custom labels
- asset validation for PNG, JPEG, WebP, GIF, and conditional AVIF paths
- auth tests for pack CRUD, bindings, draft imports, asset access, user isolation, and no cross-user pack binding
- version activation race coverage for concurrent drafts, binding changes during resolution, and asset deletion while drafts exist
- storage cleanup for cancelled drafts, failed imports, skipped files, preview extraction failures, and deleted pack versions
- Jobs-backed ZIP import path, including progress, cancellation, partial warnings, and retry behavior

Frontend tests:

- pack manager and draft import review flow
- expression picker behavior, alias search, missing asset warnings, and unsupported MIME error display
- chat portrait rendering for normal, missing, animated, and fallback assets
- stage view rendering for normal, missing, animated, reduced-motion, and static-preview modes
- `/emote` parsing before send, escape behavior, sticky override, clear override, and unresolved expression errors
- AVIF unavailable capability degradation

Integration tests:

- character/persona default pack resolution
- chat-session override resolution
- persisted message replay with pack version and asset id
- legacy mood images/static avatars still working when no pack exists
- existing Persona Visual Pack behavior unchanged

Security validation should include Bandit for touched backend code during implementation. This design doc alone does not require Bandit execution.

## Rollout

Roll out incrementally:

1. Add shared data model/service boundaries and capability reporting while leaving Persona Visual APIs intact.
2. Add draft ZIP import and manual editing behind a feature flag or config gate.
3. Add character/persona default binding.
4. Add chat portrait resolution.
5. Add expression picker and `/emote` command.
6. Add the single-character stage view behind a visible setting or feature flag.
7. Preserve VN hooks as metadata and documentation only.

The highest risks are accidental migration churn, unsafe archive import, animation mishandling, stale history replay, and cross-user asset exposure. The design mitigates these through compatibility boundaries, strict archive validation, original-file preservation, immutable activated versions, draft review, fallback reasons, and authenticated asset access.
