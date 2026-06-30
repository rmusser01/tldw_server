# Persona Visual Buddy Setup Choices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first-run Visual Buddy setup choices so a user can start from a bundled default, import a pack, or start blank without auto-activating anything.

**Architecture:** Keep `VisualPackEditor` as the owner of visual-pack mutations and focus behavior. Add one presentational setup-choice component that is reused by `VisualPackEditor` and `AssistantSetupWizard`, then add a thin route-level setup detour so the wizard can temporarily reveal the Visuals tab while assistant setup gating is active.

**Tech Stack:** React, TypeScript, Ant Design, Vitest, Testing Library, existing Persona Visual REST service helpers.

---

## Source Context

Approved spec:
- `Docs/superpowers/specs/2026-05-15-persona-visual-buddy-setup-choices-design.md`

Backlog task:
- `TASK-362.1`

Current relevant files:
- `apps/packages/ui/src/types/persona-visuals.ts`
- `apps/packages/ui/src/services/persona-visuals.ts`
- `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`
- `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- `apps/packages/ui/src/components/PersonaGarden/VisualPackReusePanel.tsx`
- `apps/packages/ui/src/components/PersonaGarden/AssistantSetupWizard.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx`
- `apps/packages/ui/src/routes/personaTypes.ts`
- `apps/packages/ui/src/routes/hooks/usePersonaSetupOrchestrator.ts`
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`

Backend starter response contract already exists:

```ts
type StarterListResponse = {
  starter_packs: Array<{
    id: string
    title: string
    description: string
    renderer_type: "sprite_frames" | "sprite_sheet" | "static_image" | "live2d"
    manifest_version: number
    states_offered: string[]
    asset_count: number
    total_bytes: number
    tags: string[]
    license_label: string
  }>
}
```

Important implementation constraint:
- `sidepanel-persona.tsx` currently renders `AssistantSetupWizard` instead of `PersonaGardenTabs` while setup is required unless an existing detour is active.
- The implementation must add a visual setup detour. A route-only `tab=visuals` change is not enough.
- `VisualPackEditor` currently renders import/portability controls only when `selectedPack` exists. First-run import must be reachable when there are no packs, so split the import preview/commit controls out from upload/export/manifest-editing if needed.

---

## File Structure

Create:
- `apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx`
- `Docs/superpowers/plans/2026-05-15-persona-visual-buddy-setup-choices-implementation-plan.md`

Modify:
- `apps/packages/ui/src/types/persona-visuals.ts`
  - Add starter catalog request/response types.
- `apps/packages/ui/src/services/persona-visuals.ts`
  - Add starter catalog list/detail/copy helpers and list normalizer.
- `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`
  - Add starter catalog service coverage.
- `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
  - Load starter catalog, render setup choices, copy selected starter into inactive draft, select copied draft, and focus import/blank controls.
- `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
  - Add setup-choice integration coverage.
- `apps/packages/ui/src/components/PersonaGarden/AssistantSetupWizard.tsx`
  - Accept and render compact optional visual setup content/handler.
- `apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx`
  - Add compact visual setup rendering/callback coverage.
- `apps/packages/ui/src/routes/personaTypes.ts`
  - Add visual setup detour type.
- `apps/packages/ui/src/routes/hooks/usePersonaSetupOrchestrator.ts`
  - Own the visual setup detour state and return handler.
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Wire wizard compact action to visual detour, render normal tabs during the detour, and add return-to-setup affordance.
- `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
  - Add setup-gating detour coverage.
- `backlog/tasks/task-362.1 - Implement-first-run-Persona-Visual-Buddy-setup-choices.md`
  - Record plan path and verification notes as work progresses.

Do not modify backend starter routes in this slice unless frontend selection is impossible with the current copy response.

---

## Task 1: Starter Catalog Service And Types

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`

- [x] **Step 1: Add failing starter catalog service tests**

Add tests to `apps/packages/ui/src/services/__tests__/persona-visuals.test.ts`.

```ts
import {
  copyPersonaVisualStarterPack,
  getPersonaVisualStarterPack,
  listPersonaVisualStarterPacks
} from "../persona-visuals"

it("lists starter packs from wrapper responses", async () => {
  mocks.fetchWithAuth.mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: async () => ({
      starter_packs: [
        {
          id: "research-buddy-starter",
          title: "Research Buddy Starter",
          description: "A bundled sprite starter.",
          renderer_type: "sprite_frames",
          manifest_version: 1,
          states_offered: ["idle", "thinking"],
          asset_count: 1,
          total_bytes: 512,
          tags: ["starter"],
          license_label: "bundled"
        }
      ]
    })
  })

  await expect(listPersonaVisualStarterPacks()).resolves.toEqual({
    starter_packs: [
      expect.objectContaining({
        id: "research-buddy-starter",
        title: "Research Buddy Starter",
        renderer_type: "sprite_frames"
      })
    ]
  })
  expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
    "/api/v1/persona/visual-starter-packs",
    expect.objectContaining({ method: "GET" })
  )
})

it("normalizes direct-list starter pack responses for defensive tests", async () => {
  mocks.fetchWithAuth.mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: async () => [{ id: "starter-1", title: "Starter 1" }]
  })

  await expect(listPersonaVisualStarterPacks()).resolves.toEqual({
    starter_packs: [expect.objectContaining({ id: "starter-1" })]
  })
})

it("loads starter pack details with an encoded id", async () => {
  mocks.fetchWithAuth.mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: async () => ({
      id: "starter/with space",
      title: "Starter Detail",
      description: "Detail",
      renderer_type: "sprite_frames",
      manifest_version: 1,
      states_offered: ["idle"],
      asset_count: 1,
      total_bytes: 128,
      tags: [],
      license_label: "bundled",
      manifest: { manifest_version: 1, renderer_type: "sprite_frames", states: {}, animations: {} },
      assets: []
    })
  })

  await getPersonaVisualStarterPack("starter/with space")

  expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
    "/api/v1/persona/visual-starter-packs/starter%2Fwith%20space",
    expect.objectContaining({ method: "GET" })
  )
})

it("copies a starter pack to a target persona without activation fields", async () => {
  mocks.fetchWithAuth.mockResolvedValueOnce({
    ok: true,
    status: 201,
    json: async () => ({
      id: "copied-pack",
      persona_id: "persona-1",
      title: "Research Buddy Starter",
      renderer_type: "sprite_frames",
      status: "draft",
      manifest: { manifest_version: 1, renderer_type: "sprite_frames", states: {}, animations: {} }
    })
  })

  await copyPersonaVisualStarterPack("starter-1", {
    target_persona_id: "persona-1"
  })

  expect(mocks.fetchWithAuth).toHaveBeenCalledWith(
    "/api/v1/persona/visual-starter-packs/starter-1/copy",
    expect.objectContaining({ method: "POST" })
  )
  const [, init] = mocks.fetchWithAuth.mock.calls.at(-1) as [string, any]
  expect(init.headers).toEqual(expect.objectContaining({ "Content-Type": "application/json" }))
  expect(JSON.parse(String(init.body))).toEqual({ target_persona_id: "persona-1" })
})
```

- [x] **Step 2: Run the failing service tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/__tests__/persona-visuals.test.ts
```

Expected: FAIL because the starter functions do not exist.

- [x] **Step 3: Add starter types**

Add to `apps/packages/ui/src/types/persona-visuals.ts` near the existing visual pack types:

```ts
export interface PersonaVisualStarterPackAssetSummary {
  asset_key: string
  filename: string
  mime_type: string
  asset_role: PersonaVisualAssetRole | string
  byte_size: number
}

export interface PersonaVisualStarterPackSummary {
  id: string
  title: string
  description: string
  renderer_type: PersonaVisualRendererType
  manifest_version: number
  states_offered: string[]
  asset_count: number
  total_bytes: number
  tags: string[]
  license_label: string
}

export interface PersonaVisualStarterPackDetail
  extends PersonaVisualStarterPackSummary {
  manifest: PersonaVisualManifest | Record<string, unknown>
  assets: PersonaVisualStarterPackAssetSummary[]
}

export interface PersonaVisualStarterPackListResponse {
  starter_packs: PersonaVisualStarterPackSummary[]
}

export interface PersonaVisualStarterPackCopyRequest {
  target_persona_id: string
  title?: string | null
}
```

- [x] **Step 4: Add service helpers**

Update imports in `apps/packages/ui/src/services/persona-visuals.ts`, then add:

```ts
const visualStarterPackPath = (
  starterPackId?: string,
  suffix = ""
): `/api/v1/persona/visual-starter-packs${string}` => {
  if (!starterPackId) return "/api/v1/persona/visual-starter-packs"
  return `/api/v1/persona/visual-starter-packs/${encodeURIComponent(starterPackId)}${suffix}`
}

export const normalizePersonaVisualStarterPackList = (
  payload: PersonaVisualStarterPackSummary[] | PersonaVisualStarterPackListResponse
): PersonaVisualStarterPackListResponse => {
  if (Array.isArray(payload)) return { starter_packs: payload }
  return {
    starter_packs: Array.isArray(payload?.starter_packs)
      ? payload.starter_packs
      : []
  }
}

export async function listPersonaVisualStarterPacks(): Promise<
  PersonaVisualStarterPackListResponse
> {
  const payload = await fetchPersonaVisualJson<
    PersonaVisualStarterPackSummary[] | PersonaVisualStarterPackListResponse
  >(visualStarterPackPath())
  return normalizePersonaVisualStarterPackList(payload)
}

export async function getPersonaVisualStarterPack(
  starterPackId: string
): Promise<PersonaVisualStarterPackDetail> {
  return fetchPersonaVisualJson<PersonaVisualStarterPackDetail>(
    visualStarterPackPath(starterPackId)
  )
}

export async function copyPersonaVisualStarterPack(
  starterPackId: string,
  payload: PersonaVisualStarterPackCopyRequest
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    visualStarterPackPath(starterPackId, "/copy"),
    {
      method: "POST",
      body: payload
    }
  )
}
```

- [x] **Step 5: Run the service tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/__tests__/persona-visuals.test.ts
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add apps/packages/ui/src/types/persona-visuals.ts apps/packages/ui/src/services/persona-visuals.ts apps/packages/ui/src/services/__tests__/persona-visuals.test.ts
git commit -m "feat: add persona visual starter catalog client"
```

---

## Task 2: Reusable Setup Choice Card

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx`

- [x] **Step 1: Write failing component tests**

Create `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx`.

```tsx
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { VisualBuddySetupChoiceCard } from "../VisualBuddySetupChoiceCard"

const starter = {
  id: "research-buddy-starter",
  title: "Research Buddy Starter",
  description: "Starter sprite pack",
  renderer_type: "sprite_frames" as const,
  manifest_version: 1,
  states_offered: ["idle", "thinking"],
  asset_count: 1,
  total_bytes: 512,
  tags: ["starter"],
  license_label: "bundled"
}

describe("VisualBuddySetupChoiceCard", () => {
  it("renders first-run setup actions", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={starter}
        starterCount={1}
        onUseDefault={vi.fn()}
        onImportPack={vi.fn()}
        onStartBlank={vi.fn()}
      />
    )

    expect(screen.getByTestId("visual-buddy-setup-choice-card")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /use default/i })).toBeEnabled()
    expect(screen.getByRole("button", { name: /import pack/i })).toBeEnabled()
    expect(screen.getByRole("button", { name: /start blank/i })).toBeEnabled()
    expect(screen.getByText(/no visual buddy is active/i)).toBeInTheDocument()
  })

  it("frames existing drafts as reviewable but inactive", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={2}
        recommendedStarter={starter}
        starterCount={1}
      />
    )

    expect(screen.getByText(/draft/i)).toBeInTheDocument()
    expect(screen.getByText(/activate/i)).toBeInTheDocument()
  })

  it("disables default copy without blocking import or blank", () => {
    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        recommendedStarter={null}
        starterCatalogError="Starter catalog unavailable"
        onImportPack={vi.fn()}
        onStartBlank={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: /use default/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: /import pack/i })).toBeEnabled()
    expect(screen.getByRole("button", { name: /start blank/i })).toBeEnabled()
  })

  it("invokes compact open visuals action without rendering mutation buttons", () => {
    const onOpenVisuals = vi.fn()

    render(
      <VisualBuddySetupChoiceCard
        selectedPersonaId="persona-1"
        selectedPersonaName="Garden Helper"
        hasActiveVisual={false}
        packCount={0}
        compact
        onOpenVisuals={onOpenVisuals}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /set up visual buddy/i }))

    expect(onOpenVisuals).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("button", { name: /use default/i })).not.toBeInTheDocument()
  })
})
```

- [x] **Step 2: Run the failing component test**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx
```

Expected: FAIL because the component does not exist.

- [x] **Step 3: Implement the presentational component**

Create `apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx`.

Implementation requirements:
- Use existing `antd` `Button`, `Tag`, and `Typography`.
- Use lucide icons for visible actions, for example `Sparkles`, `Upload`, `PenLine`, `Images`.
- Keep the component presentational. It must not import service helpers.
- Full mode renders `Use default`, optional `Choose another default`, `Import pack`, and `Start blank`.
- Compact mode renders generic optional wizard copy and one `Set up visual buddy` action mapped to `onOpenVisuals`.
- Compact mode must not claim that a visual pack is missing because the wizard
  does not fetch active visual state in this slice.
- Use `data-testid="visual-buddy-setup-choice-card"`.
- Avoid rendering a card inside another card during integration. The component may be a bordered panel, but `VisualPackEditor` should mount it as a sibling before the main editor panel.

- [x] **Step 4: Run the component test**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx
git commit -m "feat: add visual buddy setup choice card"
```

---

## Task 3: VisualPackEditor Setup Flow Integration

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Add failing editor tests**

Add focused tests to `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`.

Test cases:
- no active pack and no packs shows the setup choice card.
- active pack hides the setup choice card.
- existing drafts but no active pack shows draft-review copy.
- `Use default` calls list/copy endpoints, selects copied draft, and never calls activate.
- copy selection does not fall back to an older draft after reload.
- starter catalog load failure disables `Use default` but leaves `Import pack` and `Start blank` usable.
- `Start blank` focuses `persona-visual-pack-title-input`.
- `Import pack` focuses `persona-visual-import-preview-input` even when there are no packs.

Use the existing `mocks.fetchWithAuth` pattern. Add a helper in the test file:

```ts
const starterCatalogPayload = {
  starter_packs: [
    {
      id: "research-buddy-starter",
      title: "Research Buddy Starter",
      description: "Bundled starter",
      renderer_type: "sprite_frames",
      manifest_version: 1,
      states_offered: ["idle", "thinking"],
      asset_count: 1,
      total_bytes: 512,
      tags: ["starter"],
      license_label: "bundled"
    }
  ]
}
```

For the copy-selection regression, use this response sequence:

```ts
// initial visual packs list
{ packs: [oldDraft], active_pack: null }

// starter copy response
copiedDraft

// reload visual packs list
{ packs: [oldDraft, copiedDraft], active_pack: null }
```

Expected assertion:

```ts
expect(screen.getByTestId("persona-visual-pack-select")).toHaveValue("copied-draft")
expect(screen.queryByText(/activate/i)).toBeInTheDocument()
expect(
  mocks.fetchWithAuth.mock.calls.some(([path]) => String(path).includes("/activate"))
).toBe(false)
```

- [x] **Step 2: Run the failing editor tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: FAIL because the setup card and starter service wiring are missing.

- [x] **Step 3: Track active visual state and starter catalog state**

In `VisualPackEditor.tsx`:
- Import `VisualBuddySetupChoiceCard`.
- Import `listPersonaVisualStarterPacks` and `copyPersonaVisualStarterPack`.
- Add state:

```ts
const [activePackId, setActivePackId] = React.useState("")
const [starterPacks, setStarterPacks] = React.useState<PersonaVisualStarterPackSummary[]>([])
const [starterCatalogLoading, setStarterCatalogLoading] = React.useState(false)
const [starterCatalogError, setStarterCatalogError] = React.useState<string | null>(null)
const [copyingStarterId, setCopyingStarterId] = React.useState("")
const [starterPickerOpen, setStarterPickerOpen] = React.useState(false)
```

Update `loadPacks` to support preferred selection:

```ts
const loadPacks = React.useCallback(
  async (options: { preferredPackId?: string; fallbackPack?: PersonaVisualPack } = {}): Promise<boolean> => {
    // existing guard stays
    const response = await listPersonaVisualPacks(selectedPersonaId)
    const nextPacks = options.fallbackPack
      ? mergePack(response.packs || [], options.fallbackPack)
      : response.packs || []
    const activePack =
      response.active_pack ??
      nextPacks.find((pack) => pack.status === "active") ??
      null
    setActivePackId(activePack?.id || "")
    setPacks(nextPacks)
    const preferred =
      (options.preferredPackId
        ? nextPacks.find((pack) => pack.id === options.preferredPackId)
        : null) ??
      activePack ??
      nextPacks.find((pack) => pack.id === selectedPackId) ??
      nextPacks[0] ??
      null
    setSelectedPackId(preferred?.id || "")
    // preserve existing manifest reset behavior
  },
  [isActive, selectedPersonaId, selectedPackId]
)
```

If adding `selectedPackId` to the dependency causes extra reloads, keep a `selectedPackIdRef` instead. Do not keep stale selection behavior.

- [x] **Step 4: Load starter catalog defensively**

Add a `loadStarterCatalog` callback and effect:

```ts
const loadStarterCatalog = React.useCallback(async () => {
  if (!isActive) return
  setStarterCatalogLoading(true)
  setStarterCatalogError(null)
  try {
    const response = await listPersonaVisualStarterPacks()
    setStarterPacks(response.starter_packs || [])
  } catch (starterError) {
    setStarterPacks([])
    setStarterCatalogError(
      starterError instanceof Error
        ? starterError.message
        : "Failed to load starter packs."
    )
  } finally {
    setStarterCatalogLoading(false)
  }
}, [isActive])

React.useEffect(() => {
  if (!isActive) return
  void loadStarterCatalog()
}, [isActive, loadStarterCatalog])
```

- [x] **Step 5: Add default-copy handler**

Add:

```ts
const handleCopyStarterPack = React.useCallback(
  async (starterPackId: string) => {
    const normalizedStarterId = String(starterPackId || "").trim()
    if (!selectedPersonaId || !normalizedStarterId) return
    setCopyingStarterId(normalizedStarterId)
    setError(null)
    try {
      const copied = await copyPersonaVisualStarterPack(normalizedStarterId, {
        target_persona_id: selectedPersonaId
      })
      setPacks((current) => mergePack(current, copied))
      setSelectedPackId(copied.id)
      await loadPacks({ preferredPackId: copied.id, fallbackPack: copied })
      setStatusMessage("Default visual copied as an inactive draft. Review it, then activate when ready.")
      setStarterPickerOpen(false)
    } catch (copyError) {
      setError(copyError instanceof Error ? copyError.message : "Failed to copy starter pack.")
    } finally {
      setCopyingStarterId("")
    }
  },
  [loadPacks, selectedPersonaId]
)
```

Do not call `activatePersonaVisualPack` from this handler.

- [x] **Step 6: Make first-run import reachable without an existing pack**

Keep upload, asset editing, manifest editing, export, and activation under `selectedPack`.

Move or split only the import preview/commit UI so `Import pack` can reveal it when `selectedPack` is null. The minimal shape is:
- a top-level `portabilityImportPanel` that contains file picker, preview, commit, and import job controls.
- `export` stays inside the selected-pack-only portability area, or is hidden/disabled with selected-pack copy.

Keep the existing import functions:
- `createPersonaVisualImportPreview`
- `getPersonaVisualImportPreview`
- `startPersonaVisualImportCommit`
- `getPersonaVisualImportCommit`

Do not broaden #1696 upload/import polish in this task.

- [x] **Step 7: Render setup choices**

Compute:

```ts
const hasActiveVisual = Boolean(activePackId)
const showSetupChoices = isActive && !hasActiveVisual
const recommendedStarter = starterPacks[0] ?? null
```

Render `VisualBuddySetupChoiceCard` as a sibling before the existing main editor panel:

```tsx
{showSetupChoices ? (
  <VisualBuddySetupChoiceCard
    selectedPersonaId={selectedPersonaId}
    selectedPersonaName={selectedPersonaName || selectedPersonaId}
    hasActiveVisual={hasActiveVisual}
    packCount={packs.length}
    recommendedStarter={recommendedStarter}
    starterCount={starterPacks.length}
    starterCatalogLoading={starterCatalogLoading}
    starterCatalogError={starterCatalogError}
    copyingDefault={Boolean(copyingStarterId)}
    onUseDefault={
      recommendedStarter
        ? () => void handleCopyStarterPack(recommendedStarter.id)
        : undefined
    }
    onChooseDefault={
      starterPacks.length > 1
        ? () => setStarterPickerOpen(true)
        : undefined
    }
    onImportPack={openImportArchivePicker}
    onStartBlank={focusDraftTitleInput}
  />
) : null}
```

Implement a simple starter picker with existing `antd` `Modal` or inline panel. The picker can show title, description, renderer type, tags, and license label. It calls `handleCopyStarterPack(selectedStarter.id)`.

Update `focusDraftTitleInput` to scroll before focus:

```ts
draftTitleInputRef.current?.scrollIntoView?.({ block: "center" })
draftTitleInputRef.current?.focus()
```

- [x] **Step 8: Run editor tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: PASS.

- [x] **Step 9: Commit**

```bash
git add apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
git commit -m "feat: wire visual buddy setup choices"
```

---

## Task 4: Assistant Setup Wizard Visual Detour

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/AssistantSetupWizard.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx`
- Modify: `apps/packages/ui/src/routes/personaTypes.ts`
- Modify: `apps/packages/ui/src/routes/hooks/usePersonaSetupOrchestrator.ts`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`

- [x] **Step 1: Add failing wizard component test**

In `AssistantSetupWizard.test.tsx`, add:

```tsx
it("renders optional compact visual setup without adding a required setup step", () => {
  const onOpenVisualSetup = vi.fn()

  render(
    <AssistantSetupWizard
      catalog={[{ id: "default_persona", name: "Default Persona" }]}
      selectedPersonaId="default_persona"
      currentStep="voice"
      postSetupTargetTab="profiles"
      progressItems={[
        { step: "persona", label: "Choose persona", status: "completed", summary: null },
        { step: "voice", label: "Voice defaults", status: "current", summary: null }
      ]}
      visualSetupContent={
        <div data-testid="setup-visual-content">
          <button type="button" onClick={onOpenVisualSetup}>
            Set up visual buddy
          </button>
        </div>
      }
      saving={false}
      error={null}
      onUsePersona={vi.fn()}
      onCreatePersona={vi.fn()}
    />
  )

  expect(screen.getByTestId("setup-visual-content")).toBeInTheDocument()
  expect(screen.queryByTestId("assistant-setup-progress-step-visuals")).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: /set up visual buddy/i }))
  expect(onOpenVisualSetup).toHaveBeenCalledTimes(1)
})
```

- [x] **Step 2: Add failing route detour test**

In `sidepanel-persona.test.tsx`, add a focused test that:
- sets `mocks.location.search = "?persona_id=garden-helper&tab=profiles"`.
- returns a persona profile with `setup.is_setup_required` true or the current setup shape used by existing tests.
- renders `SidepanelPersona`.
- confirms `assistant-setup-overlay` is visible.
- clicks `Set up visual buddy`.
- confirms `persona-visual-pack-editor` is visible.
- confirms `assistant-setup-overlay` is no longer visible.
- clicks `Return to setup`.
- confirms `assistant-setup-overlay` is visible again.

Mock minimum fetch responses for visual tab load:

```ts
if (path === "/api/v1/persona/profiles/garden-helper/visual-packs") {
  return okResponse({ packs: [], active_pack: null })
}
if (path === "/api/v1/persona/visual-starter-packs") {
  return okResponse({ starter_packs: [] })
}
if (path.includes("/visual-library")) {
  return okResponse({ items: [] })
}
if (path === "/api/v1/persona/catalog") {
  return okResponse([])
}
```

Use the existing helper patterns in `sidepanel-persona.test.tsx`; do not add broad mocks that hide real route behavior.

- [x] **Step 3: Run failing wizard/route tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx \
  src/routes/__tests__/sidepanel-persona.test.tsx
```

Expected: FAIL because visual setup content and detour state do not exist.

- [x] **Step 4: Add wizard extension point**

In `AssistantSetupWizard.tsx`:
- Add optional prop `visualSetupContent?: React.ReactNode`.
- Render it near the top of the wizard after progress/error and before current step-specific content.
- Keep it outside the required progress model.

Minimal render:

```tsx
{visualSetupContent ? (
  <div data-testid="assistant-setup-visual-optional">
    {visualSetupContent}
  </div>
) : null}
```

- [x] **Step 5: Add visual detour type and orchestrator state**

In `personaTypes.ts`:

```ts
export type SetupVisualDetourState = {
  source: "wizard_optional_card"
  returnStep: PersonaSetupStep
}
```

In `usePersonaSetupOrchestrator.ts`:
- Import `SetupVisualDetourState`.
- Add `setupVisualDetour` and `setSetupVisualDetour` to the return type.
- Clear visual detour in setup reset/start paths where command/live detours are cleared.
- Clear visual detour on persona switch or route bootstrap changes that select a
  different persona, matching the existing detour safety model.
- Add handlers:

```ts
const handleOpenVisualSetupDetour = React.useCallback(() => {
  setSetupVisualDetour({
    source: "wizard_optional_card",
    returnStep: personaSetupWizard.currentStep
  })
  setActiveTab("visuals")
}, [personaSetupWizard.currentStep, setActiveTab])

const handleReturnToSetupFromVisualDetour = React.useCallback(() => {
  setSetupVisualDetour(null)
}, [])
```

If the orchestrator already centralizes analytics for detours, emit a small setup analytics event. Keep analytics optional; do not block the detour on analytics.

- [x] **Step 6: Wire sidepanel route rendering**

In `sidepanel-persona.tsx`:
- Pass `visualSetupContent` to `AssistantSetupWizard`.
- Use `VisualBuddySetupChoiceCard` in compact mode with generic unknown-state copy:

```tsx
visualSetupContent={
  <VisualBuddySetupChoiceCard
    selectedPersonaId={selectedPersonaId}
    selectedPersonaName={selectedPersonaName}
    hasActiveVisual={false}
    packCount={0}
    compact
    onOpenVisuals={setupOrch.handleOpenVisualSetupDetour}
  />
}
```

- Update the setup gate:

```tsx
const shouldShowSetupWizard =
  setupOrch.personaSetupWizard.isSetupRequired &&
  !setupOrch.setupCommandDetour &&
  !setupOrch.setupLiveDetour &&
  !setupOrch.setupVisualDetour
```

- Render a route-level return affordance while `setupVisualDetour` is active. Place it above `PersonaGardenTabs`, similar to the existing live detour notice:

```tsx
{setupOrch.setupVisualDetour ? (
  <div className="mb-3 rounded-lg border border-sky-500/30 bg-sky-500/10 p-3 text-sm text-sky-100">
    <div>Review visual setup, then return to assistant setup.</div>
    <button type="button" onClick={setupOrch.handleReturnToSetupFromVisualDetour}>
      Return to setup
    </button>
  </div>
) : null}
```

Do not make visuals a setup progress step.

- [x] **Step 7: Run wizard/route tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx \
  src/routes/__tests__/sidepanel-persona.test.tsx
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/PersonaGarden/AssistantSetupWizard.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx apps/packages/ui/src/routes/personaTypes.ts apps/packages/ui/src/routes/hooks/usePersonaSetupOrchestrator.ts apps/packages/ui/src/routes/sidepanel-persona.tsx apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx
git commit -m "feat: add visual setup detour from assistant setup"
```

---

## Task 5: Verification, Docs, And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-362.1 - Implement-first-run-Persona-Visual-Buddy-setup-choices.md`
- Modify if needed: `Docs/Code_Documentation/Persona_Visual_Packs.md`

- [x] **Step 1: Run focused UI tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/persona-visuals.test.ts \
  src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx \
  src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx \
  src/components/PersonaGarden/__tests__/AssistantSetupWizard.test.tsx \
  src/routes/__tests__/sidepanel-persona.test.tsx
```

Expected: PASS.

- [x] **Step 2: Run design-system state verification**

Run:

```bash
cd apps/packages/ui
bun run verify:design-system-state
```

Expected: PASS. If it fails because new copy introduced unregistered product-state wording, either revise the copy to existing labels or update the registry according to existing design-system instructions.

- [x] **Step 3: Run diff hygiene**

Run from repo root:

```bash
git diff --check
```

Expected: no output.

- [x] **Step 4: Run Bandit only if backend Python changed**

Expected for this frontend-only plan: no backend Python changes, so record Bandit as not applicable.

If backend Python changes become necessary, run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_persona_visual_setup_choices.json
```

Expected: no new findings in touched Python files.

- [x] **Step 5: Update Backlog task**

Update `TASK-362.1`:
- Check acceptance criteria that passed.
- Add verification commands and results.
- Add a final summary only after implementation is complete.
- Note any skipped checks with reasons.

- [x] **Step 6: Commit verification/task updates**

```bash
git add backlog/tasks/task-362.1\ -\ Implement-first-run-Persona-Visual-Buddy-setup-choices.md Docs/Code_Documentation/Persona_Visual_Packs.md
git commit -m "docs: record visual buddy setup verification"
```

Skip `Docs/Code_Documentation/Persona_Visual_Packs.md` from this commit if no docs changes were needed.

---

## Implementation Order

1. Starter service/types.
2. Presentational setup card.
3. `VisualPackEditor` behavior.
4. Assistant setup detour.
5. Verification and task closeout.

This order keeps each commit reviewable and prevents route/wizard work from masking lower-level setup-card or starter-copy failures.
