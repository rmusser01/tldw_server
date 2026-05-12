# Persona Buddy Renderer Capability Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a renderer capability registry for Persona/Buddy visual packs, expose it through the Persona API, and route Buddy display rendering/diagnostics through a frontend renderer registry while keeping `sprite_frames` as the only enabled V1 renderer.

**Architecture:** The backend owns canonical renderer capabilities and uses them at existing validation boundaries. The WebUI gets a typed capability service helper for future editor surfaces, while the floating Buddy keeps runtime render decisions local through a frontend registry so it never blocks on a capability fetch.

**Tech Stack:** FastAPI, Pydantic, pytest, ChaChaNotes Persona Visual Pack service, React 18, TypeScript, Vitest, Testing Library, Bandit.

---

## Source Inputs

- Design spec: `Docs/superpowers/specs/2026-05-12-persona-buddy-renderer-capability-registry-design.md`
- Planning task: `TASK-294`
- Prior spec task: `TASK-293`
- Current backend validator: `tldw_Server_API/app/core/Persona/visuals.py`
- Current backend service: `tldw_Server_API/app/core/Persona/visual_service.py`
- Current import preview validator: `tldw_Server_API/app/core/Persona/visual_portability/preview.py`
- Current Persona API schemas: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Current Persona API routes: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Current frontend types/service:
  - `apps/packages/ui/src/types/persona-visuals.ts`
  - `apps/packages/ui/src/services/persona-visuals.ts`
- Current Buddy runtime:
  - `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
  - `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`
  - `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts`

## Scope Boundaries

Included:

- Backend capability registry listing only enabled `sprite_frames` in V1.
- Registry-backed manifest renderer/version validation.
- Read-only Persona API capability endpoint.
- Frontend capability response types and service helper.
- Frontend Buddy renderer registry used by Buddy display and diagnostics.
- Tests proving unsupported renderers fail closed at activation/import-preview validation boundaries.

Not included:

- Live2D runtime, model loading, editor UI, or Cubism integration.
- Disabled future renderer records in the V1 API response.
- New image generation behavior.
- Renderer-level asset-role enforcement.
- Persona Chat, VN, VN Play, or CYOA changes.
- Making Buddy fetch renderer capabilities during runtime rendering.

## Implementation Decisions

1. The backend registry should be a focused new module:
   `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`.
2. The V1 capability endpoint should return only `sprite_frames`.
3. Draft manifest save remains permissive. Do not call full manifest validation
   from `PATCH /profiles/{persona_id}/visual-packs/{pack_id}/manifest`.
4. Activation and import-preview remain fail-closed because they already call
   `validate_visual_manifest()`.
5. Capability records should not include `supported_asset_roles`; asset-role
   behavior stays in existing upload/import/service paths.
6. Frontend renderability should come from a local registry, not the backend
   capability endpoint.

## File Structure

Create:

- `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`
  - Dataclass and pure helper functions for renderer capability lookup/listing.
- `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`
  - Frontend renderer registry mapping `sprite_frames` to `SpriteFrameRenderer`.
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`
  - Focused registry tests.
- `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`
  - Service-helper contract test for the capability endpoint.

Modify:

- `tldw_Server_API/app/core/Persona/visuals.py`
  - Replace hardcoded renderer set with registry-backed validation and remove
    `SUPPORTED_RENDERER_TYPES` from exports.
- `tldw_Server_API/app/api/v1/schemas/persona.py`
  - Add capability response schemas.
- `tldw_Server_API/app/api/v1/endpoints/persona.py`
  - Add read-only `/visual-renderers` endpoint near other visual routes.
- `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`
  - Add registry and unsupported-renderer validation tests.
- `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
  - Add endpoint and draft/activation boundary tests.
- `tldw_Server_API/tests/Persona/test_persona_visual_portability.py`
  - Add import-preview unsupported-renderer rejection test.
- `apps/packages/ui/src/types/persona-visuals.ts`
  - Add capability response interfaces.
- `apps/packages/ui/src/services/persona-visuals.ts`
  - Add `getPersonaVisualRendererCapabilities()`.
- `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
  - Delegate visual rendering to the frontend renderer registry.
- `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts`
  - Reuse the registry for supported-renderer diagnostics.
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`
  - Keep unsupported-renderer diagnostic coverage on the shared path.
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
  - Add/adjust unsupported active-pack renderer fallback coverage if needed.
- `backlog/tasks/task-294 - Create-Persona-Buddy-renderer-capability-implementation-plan.md`
  - Track plan review and verification.

## Task 1: Backend Capability Registry And Core Validation

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`
- Modify: `tldw_Server_API/app/core/Persona/visuals.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`

### Goal

Add a pure backend registry and make manifest renderer/version validation use it without changing valid `sprite_frames` behavior.

### Steps

- [ ] **Step 1: Add failing registry tests**

Append focused tests to `tldw_Server_API/tests/Persona/test_persona_visuals_core.py`:

```python
from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    get_persona_visual_renderer_capability,
    list_persona_visual_renderer_capabilities,
)


def test_renderer_capability_registry_lists_only_sprite_frames_in_v1() -> None:
    capabilities = list_persona_visual_renderer_capabilities()

    assert [cap.renderer_type for cap in capabilities] == ["sprite_frames"]
    capability = capabilities[0]
    assert capability.display_name == "Sprite frames"
    assert capability.manifest_versions == (1,)
    assert capability.can_validate is True
    assert capability.can_activate is True
    assert capability.buddy_runtime_supported is True
    assert capability.import_supported is True
    assert capability.export_supported is True
    assert capability.disabled_reason is None


def test_renderer_capability_lookup_rejects_unknown_or_future_renderers() -> None:
    assert get_persona_visual_renderer_capability("sprite_frames") is not None
    assert get_persona_visual_renderer_capability("live2d") is None
    assert get_persona_visual_renderer_capability("sprite_sheet") is None
    assert get_persona_visual_renderer_capability("not_real") is None
```

Add renderer validation tests:

```python
@pytest.mark.parametrize("renderer_type", ["live2d", "static_image", "sprite_sheet", "not_real"])
def test_manifest_rejects_unsupported_renderer_types(renderer_type: str) -> None:
    manifest = _activatable_manifest()
    manifest["renderer_type"] = renderer_type

    with pytest.raises(PersonaVisualManifestError, match="unsupported renderer_type"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={
                "asset-idle",
                "asset-listen",
                "asset-think",
                "asset-speak",
                "asset-error",
            },
            require_activatable=True,
        )
```

- [ ] **Step 2: Run the failing core tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q --tb=short
```

Expected: FAIL because `visual_renderer_capabilities.py` does not exist.

- [ ] **Step 3: Implement the backend registry**

Create `tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py`:

```python
"""Renderer capability registry for Persona/Buddy visual packs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaVisualRendererCapability:
    """Server-supported renderer behavior for Persona Visual Pack manifests."""

    renderer_type: str
    display_name: str
    manifest_versions: tuple[int, ...]
    can_validate: bool
    can_activate: bool
    buddy_runtime_supported: bool
    import_supported: bool
    export_supported: bool
    disabled_reason: str | None = None


_SPRITE_FRAMES = PersonaVisualRendererCapability(
    renderer_type="sprite_frames",
    display_name="Sprite frames",
    manifest_versions=(1,),
    can_validate=True,
    can_activate=True,
    buddy_runtime_supported=True,
    import_supported=True,
    export_supported=True,
)

_CAPABILITIES: dict[str, PersonaVisualRendererCapability] = {
    _SPRITE_FRAMES.renderer_type: _SPRITE_FRAMES,
}


def list_persona_visual_renderer_capabilities() -> tuple[PersonaVisualRendererCapability, ...]:
    """Return enabled renderer capabilities exposed by this server."""

    return tuple(_CAPABILITIES.values())


def get_persona_visual_renderer_capability(
    renderer_type: str,
) -> PersonaVisualRendererCapability | None:
    """Return the enabled capability for a renderer type, if supported."""

    return _CAPABILITIES.get(str(renderer_type or "").strip())
```

- [ ] **Step 4: Wire `visuals.py` to the registry**

Modify `tldw_Server_API/app/core/Persona/visuals.py`:

```python
from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    get_persona_visual_renderer_capability,
)

# Remove SUPPORTED_RENDERER_TYPES = {"sprite_frames"} and remove it from __all__.
```

In `_normalize_manifest_shape()` replace the hardcoded renderer check:

```python
    renderer_type = normalized.get("renderer_type")
    capability = get_persona_visual_renderer_capability(str(renderer_type or ""))
    if capability is None or not capability.can_validate:
        raise PersonaVisualManifestError(f"unsupported renderer_type: {renderer_type}")
    if normalized.get("manifest_version") not in capability.manifest_versions:
        raise PersonaVisualManifestError(
            f"manifest_version must be one of {', '.join(str(version) for version in capability.manifest_versions)}"
        )
```

Keep the rest of sprite/frame validation unchanged.

- [ ] **Step 5: Run core tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit backend registry slice**

Run:

```bash
git diff --check
git add tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py \
  tldw_Server_API/app/core/Persona/visuals.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_core.py
git commit -m "feat: add persona visual renderer capabilities"
```

## Task 2: Backend Capability API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

### Goal

Expose enabled renderer capabilities through the Persona API without requiring a persona id or database lookup.

### Steps

- [ ] **Step 1: Add failing API tests**

Append to `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`:

```python
def test_list_persona_visual_renderer_capabilities(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        response = client.get("/api/v1/persona/visual-renderers")

    assert response.status_code == 200
    assert response.json() == {
        "renderers": [
            {
                "renderer_type": "sprite_frames",
                "display_name": "Sprite frames",
                "manifest_versions": [1],
                "can_validate": True,
                "can_activate": True,
                "buddy_runtime_supported": True,
                "import_supported": True,
                "export_supported": True,
                "disabled_reason": None,
            }
        ]
    }


def test_persona_visual_renderer_capabilities_respect_persona_feature_flag(
    monkeypatch: pytest.MonkeyPatch,
    persona_db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep

    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: False)
    with _client_for_user(1, persona_db) as client:
        response = client.get("/api/v1/persona/visual-renderers")

    assert response.status_code == 404
```

- [ ] **Step 2: Run the failing API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_list_persona_visual_renderer_capabilities tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_persona_visual_renderer_capabilities_respect_persona_feature_flag -q --tb=short
```

Expected: FAIL because the route and schemas do not exist.

- [ ] **Step 3: Add response schemas**

In `tldw_Server_API/app/api/v1/schemas/persona.py`, add near the existing Persona Visual schemas:

```python
class PersonaVisualRendererCapabilityResponse(BaseModel):
    renderer_type: PersonaVisualRendererType
    display_name: str
    manifest_versions: list[int] = Field(default_factory=list)
    can_validate: bool
    can_activate: bool
    buddy_runtime_supported: bool
    import_supported: bool
    export_supported: bool
    disabled_reason: str | None = None


class PersonaVisualRendererCapabilitiesResponse(BaseModel):
    renderers: list[PersonaVisualRendererCapabilityResponse] = Field(default_factory=list)
```

- [ ] **Step 4: Add the endpoint**

In `tldw_Server_API/app/api/v1/endpoints/persona.py`, import the schema classes and registry helper:

```python
from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    list_persona_visual_renderer_capabilities,
)
```

Add the route near the visual-library/visual-pack routes, before `/profiles/...` visual-pack routes:

```python
@router.get(
    "/visual-renderers",
    response_model=PersonaVisualRendererCapabilitiesResponse,
    tags=["persona"],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_rate_limit)],
)
async def list_persona_visual_renderers(
    _current_user: User = Depends(get_request_user),
) -> PersonaVisualRendererCapabilitiesResponse:
    """List enabled Persona/Buddy visual renderer capabilities for this server."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    _require_current_user_id(_current_user)
    return PersonaVisualRendererCapabilitiesResponse(
        renderers=[
            PersonaVisualRendererCapabilityResponse(
                renderer_type=capability.renderer_type,
                display_name=capability.display_name,
                manifest_versions=list(capability.manifest_versions),
                can_validate=capability.can_validate,
                can_activate=capability.can_activate,
                buddy_runtime_supported=capability.buddy_runtime_supported,
                import_supported=capability.import_supported,
                export_supported=capability.export_supported,
                disabled_reason=capability.disabled_reason,
            )
            for capability in list_persona_visual_renderer_capabilities()
        ]
    )
```

- [ ] **Step 5: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_list_persona_visual_renderer_capabilities tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_persona_visual_renderer_capabilities_respect_persona_feature_flag -q --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit API slice**

Run:

```bash
git diff --check
git add tldw_Server_API/app/api/v1/schemas/persona.py \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py
git commit -m "feat: expose persona visual renderer capabilities"
```

## Task 3: Validation Boundary Regressions

**Files:**
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_portability.py`

### Goal

Prove draft saves remain permissive while activation and import-preview reject unsupported renderers.

### Steps

- [ ] **Step 1: Add draft-save and activation-boundary API tests**

Append to `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`:

```python
def test_draft_manifest_update_accepts_future_renderer_but_activation_rejects_it(
    persona_db: CharactersRAGDB,
) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Renderer Boundary Persona")
        pack = _create_visual_pack(client, persona_id)

        draft_response = client.patch(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/manifest",
            json={
                "manifest": {
                    "manifest_version": 1,
                    "renderer_type": "live2d",
                    "states": {},
                    "animations": {},
                },
                "expected_version": pack["version"],
            },
        )
        assert draft_response.status_code == 200
        assert draft_response.json()["manifest"]["renderer_type"] == "live2d"

        activate_response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/activate"
        )
        assert activate_response.status_code == 400
        assert activate_response.json()["detail"]["code"] == "invalid_manifest"
        assert "unsupported renderer_type" in activate_response.json()["detail"]["message"]
```

- [ ] **Step 2: Add import-preview unsupported-renderer test**

Use existing helpers in `tldw_Server_API/tests/Persona/test_persona_visual_portability.py` rather than duplicating archive builders. Add a test that creates a valid archive payload, changes `metadata/pack.json` `visual_manifest.renderer_type` to `live2d`, and asserts:

```python
with pytest.raises(ValueError, match="malformed_visual_manifest"):
    PersonaVisualPackImportPreviewer().create_preview(
        archive_path=archive_path,
        owner_user_id="user-1",
        target_persona_id="persona-1",
    )
```

If the existing archive helper is not easy to mutate, add a small local helper in the test file that mirrors the existing canonical archive builder and keeps checksums correct.

- [ ] **Step 3: Run boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py::test_draft_manifest_update_accepts_future_renderer_but_activation_rejects_it tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit boundary regression slice**

Run:

```bash
git diff --check
git add tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_portability.py
git commit -m "test: cover persona visual renderer boundaries"
```

## Task 4: Frontend Types, Service Helper, And Renderer Registry

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Create: `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`
- Create: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`

### Goal

Add typed frontend access to the capability endpoint and a local Buddy renderer registry that supports only `sprite_frames`.

### Steps

- [ ] **Step 1: Add failing frontend tests**

Create `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx`:

```tsx
import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { PersonaVisualPack } from "@/types/persona-visuals"
import {
  canRenderPersonaVisualPack,
  getPersonaVisualRenderer,
  PersonaVisualRendererHost
} from "../personaVisualRenderers"

const buildPack = (renderer_type: PersonaVisualPack["renderer_type"] = "sprite_frames"): PersonaVisualPack => ({
  id: "pack-1",
  persona_id: "persona-1",
  title: "Pack",
  renderer_type,
  status: "active",
  manifest: {
    manifest_version: 1,
    renderer_type,
    states: { idle: { animation_id: "idle" } },
    animations: { idle: { frames: [{ asset_id: "asset-1" }] } }
  },
  assets_by_id: {
    "asset-1": {
      id: "asset-1",
      asset_role: "frame",
      url: "/asset.png",
      mime_type: "image/png",
      width: 24,
      height: 24
    }
  }
})

describe("persona visual renderer registry", () => {
  it("resolves sprite_frames and rejects unsupported renderers", () => {
    expect(getPersonaVisualRenderer("sprite_frames")).not.toBeNull()
    expect(getPersonaVisualRenderer("live2d")).toBeNull()
    expect(canRenderPersonaVisualPack(buildPack("sprite_frames"))).toBe(true)
    expect(canRenderPersonaVisualPack(buildPack("live2d"))).toBe(false)
  })

  it("renders sprite frame packs through the registry host", async () => {
    const onRenderError = vi.fn()
    render(
      <PersonaVisualRendererHost
        pack={buildPack()}
        state="idle"
        fallbackLabel="Persona"
        className="h-10 w-10"
        onRenderError={onRenderError}
      />
    )

    expect(screen.getByTestId("persona-visual-frame")).toBeInTheDocument()
    await waitFor(() => expect(onRenderError).toHaveBeenCalledWith(null))
  })
})
```

Create `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts` using the local `tldwClient` mock pattern:

```ts
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchWithAuth: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    fetchWithAuth: (...args: unknown[]) => mocks.fetchWithAuth(...args)
  }
}))

import { getPersonaVisualRendererCapabilities } from "../persona-visuals"

describe("persona visual service", () => {
  beforeEach(() => {
    mocks.fetchWithAuth.mockReset()
  })

  it("fetches renderer capabilities from the Persona API", async () => {
    mocks.fetchWithAuth.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        renderers: [{ renderer_type: "sprite_frames", manifest_versions: [1] }]
      })
    })

    const response = await getPersonaVisualRendererCapabilities()

    expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
      "/api/v1/persona/visual-renderers",
      expect.objectContaining({ method: "GET" })
    )
    expect(response.renderers).toHaveLength(1)
  })
})
```

- [ ] **Step 2: Run failing frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx src/services/__tests__/persona-visuals.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the registry module and service helper do not exist.

- [ ] **Step 3: Add frontend capability types**

In `apps/packages/ui/src/types/persona-visuals.ts` add:

```ts
export interface PersonaVisualRendererCapability {
  renderer_type: PersonaVisualRendererType
  display_name: string
  manifest_versions: number[]
  can_validate: boolean
  can_activate: boolean
  buddy_runtime_supported: boolean
  import_supported: boolean
  export_supported: boolean
  disabled_reason?: string | null
}

export interface PersonaVisualRendererCapabilitiesResponse {
  renderers: PersonaVisualRendererCapability[]
}
```

- [ ] **Step 4: Add frontend service helper**

In `apps/packages/ui/src/services/persona-visuals.ts`, import the new response type and add:

```ts
export async function getPersonaVisualRendererCapabilities(): Promise<
  PersonaVisualRendererCapabilitiesResponse
> {
  const payload = await fetchPersonaVisualJson<PersonaVisualRendererCapabilitiesResponse>(
    "/api/v1/persona/visual-renderers"
  )
  return {
    renderers: Array.isArray(payload?.renderers) ? payload.renderers : []
  }
}
```

- [ ] **Step 5: Implement the frontend renderer registry**

Create `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx`:

```tsx
import React from "react"

import type {
  PersonaVisualPack,
  PersonaVisualRendererType,
  PersonaVisualStateId
} from "@/types/persona-visuals"

import {
  getAssetsById
} from "./personaVisualDiagnostics"
import {
  SpriteFrameRenderer,
  type PersonaVisualRenderError
} from "./SpriteFrameRenderer"

export type PersonaVisualRendererComponentProps = {
  pack: PersonaVisualPack
  state: PersonaVisualStateId
  fallbackLabel: string
  className?: string
  onRenderError?: (error: PersonaVisualRenderError | null) => void
}

export type PersonaVisualRendererRegistration = {
  rendererType: PersonaVisualRendererType
  canRender: (pack: PersonaVisualPack | null | undefined) => boolean
  Component: React.ComponentType<PersonaVisualRendererComponentProps>
}

const SpriteFrameRendererHost: React.FC<PersonaVisualRendererComponentProps> = ({
  pack,
  state,
  fallbackLabel,
  className,
  onRenderError
}) => (
  <SpriteFrameRenderer
    manifest={pack.manifest}
    assets={getAssetsById(pack)}
    state={state}
    fallbackLabel={fallbackLabel}
    className={className}
    onRenderError={onRenderError}
  />
)

const SPRITE_FRAME_REGISTRATION: PersonaVisualRendererRegistration = {
  rendererType: "sprite_frames",
  canRender: (pack) =>
    pack?.renderer_type === "sprite_frames" &&
    Boolean(pack.manifest) &&
    Object.keys(getAssetsById(pack)).length > 0,
  Component: SpriteFrameRendererHost
}

const RENDERERS: Partial<Record<PersonaVisualRendererType, PersonaVisualRendererRegistration>> = {
  sprite_frames: SPRITE_FRAME_REGISTRATION
}

export const getPersonaVisualRenderer = (
  rendererType: PersonaVisualRendererType | string | null | undefined
): PersonaVisualRendererRegistration | null =>
  RENDERERS[rendererType as PersonaVisualRendererType] ?? null

export const canRenderPersonaVisualPack = (
  pack: PersonaVisualPack | null | undefined
): boolean => Boolean(pack && getPersonaVisualRenderer(pack.renderer_type)?.canRender(pack))

export const PersonaVisualRendererHost: React.FC<PersonaVisualRendererComponentProps> = (props) => {
  const renderer = getPersonaVisualRenderer(props.pack.renderer_type)
  if (!renderer || !renderer.canRender(props.pack)) {
    return <span>{props.fallbackLabel}</span>
  }
  const Component = renderer.Component
  return <Component {...props} />
}
```

- [ ] **Step 6: Run frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx src/services/__tests__/persona-visuals.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 7: Commit frontend registry/service slice**

Run:

```bash
git diff --check
git add apps/packages/ui/src/types/persona-visuals.ts \
  apps/packages/ui/src/services/persona-visuals.ts \
  apps/packages/ui/src/services/__tests__/persona-visuals.test.ts \
  apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualRenderers.tsx \
  apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx
git commit -m "feat: add buddy visual renderer registry"
```

## Task 5: Buddy Dock And Diagnostics Integration

**Files:**
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellDock.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualDiagnostics.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`

### Goal

Remove hardcoded renderer checks from Buddy display and diagnostics by using the frontend registry.

### Steps

- [ ] **Step 1: Add/adjust failing Buddy fallback test**

In `BuddyShellHost.test.tsx`, add a test where `listPersonaVisualPacks` returns an active `live2d` pack:

```tsx
it("falls back and reports diagnostics for unsupported active renderer packs", async () => {
  const visualPack = {
    ...buildVisualPack("persona-1"),
    renderer_type: "live2d" as const,
    manifest: {
      ...buildVisualPack("persona-1").manifest,
      renderer_type: "live2d" as const
    }
  }
  visualMocks.listPersonaVisualPacks.mockResolvedValue({
    packs: [visualPack],
    active_pack: visualPack
  })

  renderHost({
    context: {
      surface_id: "persona-garden",
      surface_active: true,
      active_persona_id: "persona-1",
      position_bucket: "sidepanel-desktop",
      persona_source: "route-local",
      buddy_summary: buildBuddySummary("persona-1")
    },
    root: "sidepanel"
  })

  await waitFor(() => {
    expect(screen.getByTestId("persona-buddy-visual-diagnostic")).toHaveTextContent(
      "Visual renderer is not supported here"
    )
  })
  expect(screen.queryByTestId("persona-visual-frame")).not.toBeInTheDocument()
  expect(screen.getByTestId("persona-buddy-dock")).toHaveTextContent("Persona persona-1")
})
```

- [ ] **Step 2: Wire `BuddyShellDock` through the registry**

Replace direct `SpriteFrameRenderer` import/use with:

```tsx
import {
  canRenderPersonaVisualPack,
  PersonaVisualRendererHost
} from "./personaVisualRenderers"
```

Update renderability:

```tsx
const canRenderVisualPack = !isDormant && canRenderPersonaVisualPack(visualPack)
```

Render through:

```tsx
<PersonaVisualRendererHost
  pack={visualPack}
  state={visualState}
  fallbackLabel={buddySummary.persona_name}
  className="max-h-10 max-w-10 object-contain"
  onRenderError={onVisualRenderError}
/>
```

- [ ] **Step 3: Wire diagnostics through the registry**

In `personaVisualDiagnostics.ts`, remove `SUPPORTED_RUNTIME_RENDERERS` and import:

```ts
import { getPersonaVisualRenderer } from "./personaVisualRenderers"
```

Replace the unsupported-renderer check:

```ts
if (!getPersonaVisualRenderer(pack.renderer_type)) {
  return [
    createDiagnostic(
      "unsupported_renderer",
      "warning",
      "Visual renderer is not supported here",
      `The Buddy runtime cannot render ${pack.renderer_type} packs yet.`
    )
  ]
}
```

Avoid circular imports. If importing the registry from diagnostics creates a
cycle because `personaVisualRenderers.tsx` imports `getAssetsById`, split pure
helpers into a tiny module:

```text
apps/packages/ui/src/components/Common/PersonaBuddy/personaVisualAssets.ts
```

Move `getAssetsById()` and `normalizeFrames()` there, and update both diagnostics and renderer imports.
Also update `SpriteFrameRenderer.tsx`, because it currently imports
`normalizeFrames` from `personaVisualDiagnostics.ts`.

- [ ] **Step 4: Run Buddy frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 5: Commit Buddy integration slice**

Run:

```bash
git diff --check
git add apps/packages/ui/src/components/Common/PersonaBuddy
git commit -m "feat: route buddy visuals through renderer registry"
```

## Task 6: Final Verification And PR Readiness

**Files:**
- Modify: `backlog/tasks/<implementation-task>.md` when implementation task exists.
- Modify: this plan only if implementation discovers a valid plan correction.

### Goal

Run focused verification and package the implementation evidence.

### Steps

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Common/PersonaBuddy/__tests__/personaVisualRenderers.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/services/__tests__/persona-visuals.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend production files**

Run from repo root:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Persona/visual_renderer_capabilities.py tldw_Server_API/app/core/Persona/visuals.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_persona_visual_renderer_capabilities.json
```

Expected: command exits 0 and reports no new findings in touched code.

- [ ] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; branch contains only intended files.

- [ ] **Step 5: Optional type-check sanity**

Run if time permits or if TypeScript changes are non-trivial:

```bash
cd apps/packages/ui
bunx tsc --noEmit
```

Expected: either PASS or known unrelated baseline failures documented with a grep showing no touched-file errors.

- [ ] **Step 6: Final commit and task notes**

If any verification-only notes or task records changed:

```bash
git add backlog/tasks/<implementation-task>.md
git commit -m "chore: record renderer capability verification"
```

Do not commit unrelated dirty worktree files. Do not merge or open a PR until the user chooses the execution path.
