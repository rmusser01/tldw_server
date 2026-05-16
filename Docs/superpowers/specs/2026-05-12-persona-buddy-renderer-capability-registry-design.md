# Persona Buddy Renderer Capability Registry Design

## Context

Persona Visual Packs are already the user-owned 2D asset system for Persona
Buddy and Persona Live. The current V1 path is intentionally sprite/frame based:
Persona Garden authors packs, the backend validates pack manifests, activation is
explicit, and the floating Buddy shell renders the active pack through
`SpriteFrameRenderer`.

The next Buddy display slice should make the renderer boundary explicit without
starting Live2D or external provider work. Today the backend manifest validator
accepts only `sprite_frames` through a hardcoded set, the Buddy dock directly
checks `visualPack.renderer_type === "sprite_frames"`, and frontend diagnostics
keep a separate supported-renderer set. The frontend type vocabulary already
mentions future renderer names such as `static_image`, `sprite_sheet`, and
`live2d`, but there is no shared capability contract that says what the server
can validate, activate, import, export, or render.

## Goals

1. Add a renderer capability contract for Persona/Buddy visual packs.
2. Keep `sprite_frames` as the only enabled renderer for activation and Buddy
   runtime rendering.
3. Fail closed for unsupported renderer types during backend validation and
   activation.
4. Expose backend renderer support for WebUI services and future editor
   surfaces instead of duplicating backend assumptions.
5. Refactor Buddy display code so renderer selection goes through a frontend
   renderer registry rather than a hardcoded component check.
6. Preserve all existing sprite/frame pack behavior.

## Non-Goals

1. No Live2D runtime, model loading, Cubism integration, or Live2D editor UI.
2. No new asset generation behavior.
3. No Persona Chat judge or Persona Chat runtime changes.
4. No VN, VN Play, or CYOA asset-runtime changes.
5. No shared marketplace, cross-user library, or external MCP pack-provider
   contract in this slice.
6. No Persona Garden capability panel unless needed for a minimal endpoint
   smoke path.

## Approved Approach

Implement an end-to-end thin slice:

1. Backend renderer capability registry.
2. Read-only Persona API endpoint exposing renderer capabilities.
3. Frontend Buddy renderer registry.
4. Diagnostics and Buddy fallback behavior wired to the same frontend registry.

This gives the Buddy display system a real renderer boundary while keeping the
implementation narrow and reversible.

## Backend Design

Add a small backend renderer capability module near the existing persona visual
manifest validator. The concrete file can be
`tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`, or the
capability records can live in `visuals.py` if the implementation stays small.
Prefer a separate module if the registry would make `visuals.py` harder to scan.

Each capability record should be explicit and serializable:

```python
@dataclass(frozen=True)
class PersonaVisualRendererCapability:
    renderer_type: str
    display_name: str
    manifest_versions: tuple[int, ...]
    can_validate: bool
    can_activate: bool
    buddy_runtime_supported: bool
    import_supported: bool
    export_supported: bool
    disabled_reason: str | None = None
```

The first implementation should register:

1. `sprite_frames`: enabled for validation, activation, Buddy runtime rendering,
   import, and export.

Do not expose disabled future renderer records in this V1 slice. Names such as
`static_image`, `sprite_sheet`, and `live2d` can remain in the frontend type
vocabulary, but the server capability endpoint should list only enabled
`sprite_frames` until a follow-up slice defines the next renderer's real
validation and runtime behavior.

`validate_visual_manifest()` should use the registry when checking
`manifest_version` and `renderer_type`. The validator should continue to validate
only the existing sprite/frame manifest body in this slice. Unsupported renderer
types should raise `PersonaVisualManifestError` with a stable message such as
`unsupported renderer_type: live2d`, or an equivalent stable code if the
implementation introduces structured manifest errors.

Draft manifest updates should stay permissive. The current editor flow can store
incomplete or invalid draft manifests while a user is still authoring a pack.
This slice should fail closed at validation boundaries that already decide
runtime safety, especially activation and import preview, instead of turning
draft save into a full manifest gate.

Do not add new asset-role enforcement in the capability registry. Existing pack
validation is based on manifest asset references and known asset dimensions, and
asset role semantics should stay with the existing upload/import/service paths.
Adding renderer-level role enforcement in this slice risks breaking existing
draft, generated-candidate, or imported asset rows without improving Buddy
runtime safety.

`PersonaVisualService.activate_pack()` should keep validating before activation.
The service should continue mapping validation failures to the existing stable
service error shape, but the message should clearly identify unsupported
renderers when that is the cause.

## API Design

Expose a read-only capability endpoint under the Persona visual-pack surface,
for example:

```text
GET /api/v1/persona/visual-renderers
```

The endpoint should use the same auth/scoping conventions as other Persona
routes. It does not need persona-specific state because renderer support is a
server capability, but keeping it under `/persona` makes the ownership boundary
clear.

Response shape:

```json
{
  "renderers": [
    {
      "renderer_type": "sprite_frames",
      "display_name": "Sprite frames",
      "manifest_versions": [1],
      "can_validate": true,
      "can_activate": true,
      "buddy_runtime_supported": true,
      "import_supported": true,
      "export_supported": true,
      "disabled_reason": null
    }
  ]
}
```

The V1 endpoint should list only enabled renderer records. Unknown renderer names
should not be accepted just because the frontend type union includes them.

Add explicit API schemas near the existing Persona Visual schemas, for example
`PersonaVisualRendererCapabilityResponse` and
`PersonaVisualRendererCapabilitiesResponse`. The endpoint implementation should
live near the current visual-library and visual-pack routes in
`tldw_Server_API/app/api/v1/endpoints/persona.py`, so route ownership stays in
the existing Persona API surface.

## Frontend Design

Add a frontend Buddy renderer registry, likely under:

```text
apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.ts
```

The registry should provide a small API:

```ts
type PersonaVisualRendererRegistration = {
  rendererType: PersonaVisualRendererType
  canRender: (pack: PersonaVisualPack) => boolean
  render: React.ComponentType<PersonaVisualRendererComponentProps>
}

getPersonaVisualRenderer(rendererType): PersonaVisualRendererRegistration | null
canRenderPersonaVisualPack(pack): boolean
```

For this slice, the only registered renderer is `sprite_frames`, backed by
`SpriteFrameRenderer`.

`BuddyShellDock` should stop importing and selecting `SpriteFrameRenderer`
directly. It should ask the registry whether the active pack can render. If it
can, it renders the registered component. If it cannot, it shows the existing
text Buddy fallback.

`personaVisualDiagnostics.ts` should use the same frontend registry or a shared
helper derived from it. The goal is to remove the local
`SUPPORTED_RUNTIME_RENDERERS` set so renderability and diagnostics cannot drift.

The broader `PersonaVisualRendererType` union can remain in
`persona-visuals.ts` because those labels are part of the future manifest
vocabulary. Runtime behavior must still be capability-gated.

Add frontend response types and a service helper, for example
`getPersonaVisualRendererCapabilities()`, in the existing
`apps/packages/ui/src/services/persona-visuals.ts` service module. This service
helper gives future editor surfaces a backend-backed capability source, but
Buddy rendering must still use the local renderer registry in this slice.

## Data Flow

1. The backend owns canonical renderer capabilities.
2. Manifest validation checks renderer type and manifest version through the
   backend registry.
3. Activation remains explicit and requires a valid activatable renderer.
4. The WebUI service layer can fetch renderer capabilities for future UI
   surfaces.
5. Buddy loads the active pack using the existing pack list/detail flow.
6. Buddy asks the frontend renderer registry whether the active pack can render.
7. Renderable sprite/frame packs use `SpriteFrameRenderer`.
8. Unsupported, missing, or invalid packs use the text Buddy fallback and the
   existing diagnostic UI.

The capability endpoint is implemented and tested in this slice, but Buddy
runtime renderability remains local and deterministic. The floating Buddy should
not block render decisions on a capability endpoint fetch in this first slice.

## Error Handling

Backend behavior should fail closed:

1. Unknown renderer types are invalid.
2. Known but disabled renderer types are invalid for activation.
3. Manifest versions outside a renderer capability's supported version list are
   invalid.
4. Draft manifest save can still persist incomplete or invalid drafts, but those
   drafts cannot activate until the manifest passes registry-backed validation.
5. Import-preview and activation paths should not produce active packs for
   unsupported renderers.

Frontend behavior should fail soft:

1. Unsupported active-pack renderer types should not crash Buddy.
2. Unsupported renderer diagnostics should remain visible in the dock/popover.
3. Renderer labels displayed in diagnostics should be treated as plain text.
4. Render errors should stay scoped to the current active pack and visual state.

## Testing Plan

Backend tests:

1. Capability registry lists `sprite_frames` as activatable and Buddy-runtime
   supported.
2. Capability lookup rejects unknown renderer types or returns a disabled/null
   result, depending on the chosen helper API.
3. `validate_visual_manifest()` still accepts valid V1 `sprite_frames`
   manifests.
4. `validate_visual_manifest()` rejects `live2d`, `static_image`,
   `sprite_sheet`, and unknown renderer types unless a future renderer is
   explicitly enabled.
5. Capability endpoint returns the expected `sprite_frames` record.
6. Draft manifest update behavior is either unchanged or explicitly tested if
   the implementation touches that route; unsupported renderers must still fail
   at activation/import-preview validation boundaries.

Frontend tests:

1. Renderer registry resolves `sprite_frames` to the sprite renderer.
2. Renderer registry reports unsupported renderers as not renderable.
3. `BuddyShellDock` renders sprite/frame packs through the registry.
4. `BuddyShellDock` falls back without rendering `SpriteFrameRenderer` for an
   unsupported renderer.
5. `personaVisualDiagnostics` reports `unsupported_renderer` from the shared
   registry path.
6. Persona visual service coverage verifies the capability endpoint helper
   parses the V1 response shape without requiring Buddy runtime code to fetch it.

Existing sprite renderer and visual diagnostics tests should remain valid.

## Rollout Notes

This is an internal contract hardening slice. It should be safe to ship because
the only enabled runtime stays `sprite_frames` and the Buddy fallback already
exists. The main user-visible change is clearer unsupported-renderer behavior if
a future or malformed pack reaches the runtime.

After this lands, follow-up slices can be considered in this order:

1. Sprite atlas or sprite-sheet V1.1 validation and Buddy rendering.
2. Non-sprite manifest V2 design.
3. Feature-gated Live2D adapter spike.
4. External MCP pack-provider contract.

Each follow-up should reuse the registry instead of adding new hardcoded
renderer checks.

## Implementation Planning Decisions

1. The capability endpoint should list only enabled `sprite_frames` in this V1
   slice. Disabled future renderer placeholders are deferred until a renderer
   follow-up has concrete validation/runtime behavior.
2. Should backend manifest errors remain string-only for this slice, or should
   the manifest validator introduce stable structured error codes?
3. The first frontend slice should use a local Buddy renderer registry while the
   endpoint exists for service/API coverage and later editor surfaces. Buddy
   should not fetch renderer capabilities during runtime rendering.

Recommendation: keep manifest errors string-only unless existing API error
mapping makes structured codes cheap. The local frontend registry should keep
runtime rendering deterministic.
