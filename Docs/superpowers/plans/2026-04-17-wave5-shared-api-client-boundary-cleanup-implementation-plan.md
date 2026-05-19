# Wave 5 Shared API Client Boundary Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared UI API client boundary explicit and reviewable by adding a tested ownership inventory, extracting runtime helpers out of `TldwApiClient.ts`, de-shadowing the bounded Wave 5 slices, and fixing the maintainership workflow around `verify:openapi`.

**Architecture:** Keep the current mixin architecture, but stop treating the monolithic `TldwApiClient.ts` class body as an accidental second implementation of domain methods. Introduce a pre-mixin base export plus an authoritative overlap manifest, move the runtime helpers that domains currently import from `TldwApiClient.ts` into focused helper modules with transitional re-exports, then de-shadow only the bounded Wave 5 slices: `admin` (5 overlaps), `workspace-api` (11 overlaps), and `presentations` (3 overlaps). Close by adding the missing UI-package `verify:openapi` entry point, a maintainer note near the code, and an explicit backlog handoff for the deferred higher-coupling domains.

**Tech Stack:** TypeScript, Vitest, React Testing Library, Node.js package scripts, Markdown docs

---

## File Structure

### Ownership Inventory And Collision Guard

- Create: `Docs/superpowers/reviews/shared-api-client/README.md`
  - Index Wave 5 ownership artifacts and follow-on backlog handoffs for the shared UI client boundary.
- Create: `Docs/superpowers/reviews/shared-api-client/2026-04-17-wave5-ownership-inventory.md`
  - Record the measured overlap counts, the Wave 5 in-scope slices (`admin`, `workspace-api`, `presentations`), and the explicit follow-on backlog for deferred overlapping domains (`models-audio`, `characters`, `chat-rag`, `collections`, `media`).
- Create: `apps/packages/ui/src/services/tldw/client-ownership.ts`
  - Authoritative ownership manifest: in-scope slice list, deferred overlap domains, transitional overlap allowlist for every currently overlapping domain, and helpers for the collision guard test.
- Create: `apps/packages/ui/src/services/__tests__/tldw-api-client.ownership-guard.test.ts`
  - Compare `TldwApiClientBase.prototype` against the domain method objects and fail if overlaps differ from the manifest.
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Export `TldwApiClientBase`, keep `TldwApiClient` as the mixed facade, and keep `TldwApiClientCore` compatible with current domain typing.
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.exports.test.ts`
  - Lock `TldwApiClientBase` plus the transitional helper re-exports.

### Shared Runtime Helper Extraction

- Create: `apps/packages/ui/src/services/tldw/collections-normalizers.ts`
  - Move `normalizeReadingDigestSchedule`, `toFiniteNumber`, `toOptionalString`, `toRecord`, and ingestion-source normalization helpers into a focused collections-side helper module.
- Create: `apps/packages/ui/src/services/tldw/presentation-style.ts`
  - Move visual-style snapshot cloning helpers out of `TldwApiClient.ts`.
- Create: `apps/packages/ui/src/services/tldw/persona-normalizers.ts`
  - Move persona normalization helpers out of `TldwApiClient.ts`.
- Create: `apps/packages/ui/src/services/tldw/__tests__/shared-normalizers.test.ts`
  - Direct regression coverage for the extracted helper modules.
- Modify: `apps/packages/ui/src/services/tldw/domains/collections.ts`
  - Import collections-side runtime helpers from the new helper module instead of `../TldwApiClient`.
- Modify: `apps/packages/ui/src/services/tldw/domains/presentations.ts`
  - Import visual-style snapshot helpers from the new helper module instead of `../TldwApiClient`.
- Modify: `apps/packages/ui/src/services/tldw/domains/characters.ts`
  - Import persona normalization helpers from the new helper module instead of `../TldwApiClient`.
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Replace inline helper definitions with imports and transitional re-exports.
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
  - Keep the presentation normalization contract green after helper extraction.

### De-Shadow Slice A: `admin` And `workspace-api`

- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Remove the overlapping `admin` and `workspace-api` class methods from the base class while leaving the domain mixin runtime surface intact.
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
  - Remove the `admin` and `workspace-api` overlaps from the transitional allowlist once the class methods are deleted.
- Create: `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
  - Instantiate `TldwApiClient`, call representative `admin` and `workspace-api` methods, and assert the request paths still come from the mixed-in domain behavior.
- Test:
  - `apps/packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.media-budget.test.tsx`
  - `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

### De-Shadow Slice B: `presentations`

- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Remove the overlapping `presentations` class methods from the base class.
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
  - Remove the `presentations` overlaps from the transitional allowlist.
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
  - Add direct smoke coverage for `generateSlidesFromMedia()`, `getPresentation()`, and `exportPresentation()`.
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
  - Keep the visual-style normalization path green after the class-method removal.
- Test:
  - `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioBootstrap.test.tsx`
  - `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioCreatePayload.test.tsx`
  - `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioPage.test.tsx`

### Maintainership Workflow And Final Handoff

- Create: `apps/packages/ui/src/services/tldw/README.md`
  - Short maintainer note: what belongs in `TldwApiClient.ts`, what belongs in `domains/`, when a focused helper module is justified, and which overlaps are explicitly deferred after Wave 5.
- Modify: `apps/packages/ui/package.json`
  - Add a UI-package `verify:openapi` alias that points to the existing script under `apps/extension/scripts/verify-openapi-client-paths.mjs`.
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Update the maintenance comment to reference the runnable UI-package command.
- Modify: `apps/packages/ui/src/services/tldw/fallback-schemas.ts`
  - Keep the verification instructions aligned with the new alias.
- Modify: `Docs/superpowers/reviews/shared-api-client/2026-04-17-wave5-ownership-inventory.md`
  - Finalize the deferred-overlap handoff with the remaining overlap counts and next action for the follow-on backlog.
- Verify:
  - `cd apps/packages/ui && npm run verify:openapi`
  - focused Vitest slice for ownership, helper extraction, request-core, and slice regressions

## Notes

- The measured class-vs-domain overlap counts at plan-authoring time are:
  - `admin`: 5 overlaps
  - `workspace-api`: 11 overlaps
  - `presentations`: 3 overlaps
  - `models-audio`: 24 overlaps
  - `characters`: 20 overlaps
  - `collections`: 45 overlaps
  - `media`: 39 overlaps
  - `chat-rag`: 98 overlaps
- This plan intentionally keeps Wave 5 bounded to:
  - shared ownership inventory and guard rails
  - runtime helper extraction
  - de-shadowing `admin`, `workspace-api`, and `presentations`
  - maintainership workflow and backlog handoff
- `models-audio` is not in scope for this implementation plan even though the design listed it as a candidate early slice. The measured overlap count is larger and the surface mixes models, embeddings, llama.cpp, MLX, transcription, TTS, and image behavior. Record it as a deferred follow-on backlog in the Stage 0 inventory instead of stretching this wave.
- Bandit is not applicable to the expected Wave 5 TypeScript/package.json/Markdown touched scope. If the implementation later introduces Python helpers or Python verification wrappers, add a Bandit step for those touched Python paths before completion and call that out explicitly in the verification summary.

### Task 1: Record The Ownership Inventory And Add A Collision Guard

**Files:**
- Create: `Docs/superpowers/reviews/shared-api-client/README.md`
- Create: `Docs/superpowers/reviews/shared-api-client/2026-04-17-wave5-ownership-inventory.md`
- Create: `apps/packages/ui/src/services/tldw/client-ownership.ts`
- Create: `apps/packages/ui/src/services/__tests__/tldw-api-client.ownership-guard.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.exports.test.ts`

- [ ] **Step 1: Write the failing ownership-guard test and inventory manifest shape**

Create the manifest and the red test first.

```ts
// apps/packages/ui/src/services/tldw/client-ownership.ts
export const WAVE5_IN_SCOPE_SLICES = [
  "admin",
  "workspace-api",
  "presentations"
] as const

export const DEFERRED_OVERLAP_DOMAINS = [
  "models-audio",
  "characters",
  "collections",
  "media",
  "chat-rag"
] as const

export const TRANSITIONAL_DOMAIN_OVERLAPS = {
  admin: [
    "createAdminRole",
    "deleteAdminRole",
    "listAdminRoles",
    "listAdminUsers",
    "updateAdminUser"
  ],
  "workspace-api": [
    "createSkill",
    "deleteSkill",
    "executeSkill",
    "exportSkill",
    "getSkill",
    "getSkillsContext",
    "importSkill",
    "importSkillFile",
    "listSkills",
    "seedSkills",
    "updateSkill"
  ],
  presentations: [
    "exportPresentation",
    "generateSlidesFromMedia",
    "getPresentation"
  ],
  // Also include the exact current overlap names for:
  // - models-audio
  // - characters
  // - collections
  // - media
  // - chat-rag
  // Copy those names from the Stage 0 inventory so the guard fails on any
  // unexpected overlap drift outside the three in-scope slices too.
} as const
```

```ts
// apps/packages/ui/src/services/__tests__/tldw-api-client.ownership-guard.test.ts
import { describe, expect, it } from "vitest"
import { TldwApiClientBase } from "@/services/tldw/TldwApiClient"
import {
  adminMethods,
  characterMethods,
  chatRagMethods,
  collectionsMethods,
  mediaMethods,
  modelsAudioMethods,
  presentationsMethods,
  workspaceApiMethods
} from "@/services/tldw/domains"
import { TRANSITIONAL_DOMAIN_OVERLAPS } from "@/services/tldw/client-ownership"

const baseMethodNames = new Set(
  Object.getOwnPropertyNames(TldwApiClientBase.prototype).filter(
    (name) => name !== "constructor"
  )
)

const actualOverlaps = {
  admin: Object.keys(adminMethods).filter((name) => baseMethodNames.has(name)).sort(),
  characters: Object.keys(characterMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort(),
  "chat-rag": Object.keys(chatRagMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort(),
  collections: Object.keys(collectionsMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort(),
  media: Object.keys(mediaMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort(),
  "models-audio": Object.keys(modelsAudioMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort(),
  "workspace-api": Object.keys(workspaceApiMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort(),
  presentations: Object.keys(presentationsMethods)
    .filter((name) => baseMethodNames.has(name))
    .sort()
}

it("matches the recorded overlap baseline", () => {
  expect(actualOverlaps).toEqual(TRANSITIONAL_DOMAIN_OVERLAPS)
})
```

- [ ] **Step 2: Run the ownership guard to verify it fails**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/tldw-api-client.ownership-guard.test.ts src/services/__tests__/tldw-api-client.exports.test.ts`
Expected: FAIL because `TldwApiClientBase` does not exist yet and the overlap inventory is not yet wired into the test surface.

- [ ] **Step 3: Export a pre-mixin base class and wire the inventory artifact**

Update `TldwApiClient.ts` so the class body has an explicit pre-mixin export.

```ts
export class TldwApiClientBase {
  // existing class body moves here unchanged
}

export class TldwApiClient extends TldwApiClientBase {}

Object.assign(
  TldwApiClient.prototype,
  adminMethods,
  mediaMethods,
  characterMethods,
  chatRagMethods,
  collectionsMethods,
  modelsAudioMethods,
  presentationsMethods,
  workspaceApiMethods,
  webClipperMethods
)
```

Write the inventory artifact with the overlap counts and the Wave 5 in-scope slice decision.

```markdown
## Wave 5 Ownership Inventory

- In-scope slices:
  - admin (5 overlaps)
  - workspace-api (11 overlaps)
  - presentations (3 overlaps)

- Deferred follow-on slices:
  - models-audio (24 overlaps)
  - characters (20 overlaps)
  - collections (45 overlaps)
  - media (39 overlaps)
  - chat-rag (98 overlaps)

- Reason for boundary:
  - the in-scope slices are the smallest coherent overlap groups with clear consumer coverage
  - the deferred slices remain too broad for the same reviewable pass
```

- [ ] **Step 4: Re-run the ownership guard and export coverage**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/tldw-api-client.ownership-guard.test.ts src/services/__tests__/tldw-api-client.exports.test.ts`
Expected: PASS, with the explicit overlap allowlist matching the current baseline for all currently overlapping domains and the Wave 5 in-scope slices called out separately in the manifest.

- [ ] **Step 5: Commit**

```bash
git add Docs/superpowers/reviews/shared-api-client/README.md \
  Docs/superpowers/reviews/shared-api-client/2026-04-17-wave5-ownership-inventory.md \
  apps/packages/ui/src/services/tldw/client-ownership.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.ownership-guard.test.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.exports.test.ts
git commit -m "test: lock wave5 client ownership inventory"
```

### Task 2: Extract Runtime Helpers Out Of `TldwApiClient.ts`

**Files:**
- Create: `apps/packages/ui/src/services/tldw/collections-normalizers.ts`
- Create: `apps/packages/ui/src/services/tldw/presentation-style.ts`
- Create: `apps/packages/ui/src/services/tldw/persona-normalizers.ts`
- Create: `apps/packages/ui/src/services/tldw/__tests__/shared-normalizers.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/collections.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/presentations.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/characters.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.exports.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`

- [ ] **Step 1: Write the failing helper-module tests**

Start with a direct test that imports the new helper modules and checks the existing normalization behavior.

```ts
import {
  normalizeReadingDigestSchedule,
  normalizeIngestionSourceListResponse
} from "@/services/tldw/collections-normalizers"
import { clonePresentationVisualStyleSnapshot } from "@/services/tldw/presentation-style"
import { normalizePersonaProfile } from "@/services/tldw/persona-normalizers"

it("normalizes reading digest schedules with string ids and boolean flags", () => {
  expect(
    normalizeReadingDigestSchedule({
      id: 12,
      enabled: 1,
      require_online: 0,
      format: "html"
    })
  ).toMatchObject({
    id: "12",
    enabled: true,
    require_online: false,
    format: "html"
  })
})
```

- [ ] **Step 2: Run the helper tests to verify failure**

Run: `cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__/shared-normalizers.test.ts src/services/__tests__/tldw-api-client.exports.test.ts src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
Expected: FAIL because the helper modules do not exist yet and the domains still import runtime helpers from `../TldwApiClient`.

- [ ] **Step 3: Extract focused helper modules and keep transitional re-exports**

Move the runtime helpers without changing their behavior.

```ts
// apps/packages/ui/src/services/tldw/persona-normalizers.ts
export const normalizePersonaProfile = <T extends Record<string, unknown>>(...)
export const normalizePersonaExemplar = (...)

// apps/packages/ui/src/services/tldw/presentation-style.ts
export const clonePresentationVisualStyleSnapshot = (...)
export const buildPresentationVisualStyleSnapshot = (...)

// apps/packages/ui/src/services/tldw/collections-normalizers.ts
export const normalizeReadingDigestSchedule = (...)
export const normalizeIngestionSource = (...)
export const normalizeIngestionSourceListResponse = (...)
```

Then update `TldwApiClient.ts` to re-export those helpers for compatibility.

```ts
export {
  normalizeReadingDigestSchedule,
  normalizeIngestionSource,
  normalizeIngestionSourceItem,
  normalizeIngestionSourceItemsListResponse,
  normalizeIngestionSourceListResponse,
  normalizeIngestionSourceSyncTrigger
} from "./collections-normalizers"

export {
  clonePresentationVisualStyleSnapshot,
  buildPresentationVisualStyleSnapshot
} from "./presentation-style"

export {
  normalizePersonaProfile,
  normalizePersonaExemplar
} from "./persona-normalizers"
```

- [ ] **Step 4: Re-run helper and export coverage**

Run: `cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__/shared-normalizers.test.ts src/services/__tests__/tldw-api-client.exports.test.ts src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
Expected: PASS, with `collections.ts`, `presentations.ts`, and `characters.ts` no longer importing runtime helpers from `../TldwApiClient`.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/collections-normalizers.ts \
  apps/packages/ui/src/services/tldw/presentation-style.ts \
  apps/packages/ui/src/services/tldw/persona-normalizers.ts \
  apps/packages/ui/src/services/tldw/__tests__/shared-normalizers.test.ts \
  apps/packages/ui/src/services/tldw/domains/collections.ts \
  apps/packages/ui/src/services/tldw/domains/presentations.ts \
  apps/packages/ui/src/services/tldw/domains/characters.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.exports.test.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts
git commit -m "refactor: extract shared tldw client helpers"
```

### Task 3: De-Shadow The `admin` And `workspace-api` Slices

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
- Create: `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
- Test:
  - `apps/packages/ui/src/components/Option/Admin/__tests__/ServerAdminPage.media-budget.test.tsx`
  - `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

- [ ] **Step 1: Write the failing slice smoke tests**

Add direct client smoke coverage for representative `admin` and `workspace-api` methods.

```ts
it("keeps listAdminUsers on the mixed admin domain path after class cleanup", async () => {
  mocks.bgRequest.mockResolvedValue({ users: [], total: 0, page: 1, limit: 20, pages: 0 })
  const client = new TldwApiClient()
  await client.listAdminUsers({ limit: 20 })
  expect(mocks.bgRequest).toHaveBeenCalledWith(
    expect.objectContaining({
      path: "/api/v1/admin/users?limit=20",
      method: "GET"
    })
  )
})

it("keeps listSkills on the mixed workspace-api domain path after class cleanup", async () => {
  mocks.bgRequest.mockResolvedValue({ skills: [], count: 0, total: 0, limit: 10, offset: 0 })
  const client = new TldwApiClient()
  await client.listSkills({ limit: 10 })
  expect(mocks.bgRequest).toHaveBeenCalledWith(
    expect.objectContaining({
      path: "/api/v1/skills?limit=10",
      method: "GET"
    })
  )
})
```

Also add base-class assertions:

```ts
expect(Object.getOwnPropertyNames(TldwApiClientBase.prototype)).not.toContain("listAdminUsers")
expect(Object.getOwnPropertyNames(TldwApiClientBase.prototype)).not.toContain("listSkills")
```

- [ ] **Step 2: Run the slice smoke tests to verify failure**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
Expected: FAIL because the base class still defines the overlapping `admin` and `workspace-api` methods.

- [ ] **Step 3: Delete the overlapping class methods and shrink the overlap allowlist**

Remove only the Wave 5 `admin` and `workspace-api` overlapping methods from `TldwApiClientBase`.

Delete these `admin` methods from the base class:

- `listAdminUsers`
- `updateAdminUser`
- `listAdminRoles`
- `createAdminRole`
- `deleteAdminRole`

Delete these `workspace-api` methods from the base class:

- `listSkills`
- `getSkill`
- `createSkill`
- `updateSkill`
- `deleteSkill`
- `importSkill`
- `importSkillFile`
- `seedSkills`
- `exportSkill`
- `executeSkill`
- `getSkillsContext`

Then update `client-ownership.ts` so `admin` and `workspace-api` are no longer listed under transitional overlaps.

- [ ] **Step 4: Re-run direct and consumer-facing regressions**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/tldw-api-client.ownership-guard.test.ts src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/components/Option/Admin/__tests__/ServerAdminPage.media-budget.test.tsx src/components/Option/Skills/__tests__/Manager.test.tsx`
Expected: PASS, with the ownership guard showing no remaining Wave 5 overlap entries for `admin` or `workspace-api`.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/tldw/client-ownership.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts
git commit -m "refactor: de-shadow admin and workspace api client slices"
```

### Task 4: De-Shadow The `presentations` Slice

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
- Test:
  - `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioBootstrap.test.tsx`
  - `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioCreatePayload.test.tsx`
  - `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioPage.test.tsx`

- [ ] **Step 1: Extend the slice smoke tests for presentations**

Add direct client smoke coverage for the overlapping `presentations` methods.

```ts
it("keeps generateSlidesFromMedia on the mixed presentations domain path", async () => {
  mocks.bgRequest.mockResolvedValue({ id: "pres-1", title: "Deck", theme: "black", slides: [], version: 1, created_at: "2026-04-17T00:00:00Z" })
  const client = new TldwApiClient()
  await client.generateSlidesFromMedia(7, { titleHint: "Deck" })
  expect(mocks.bgRequest).toHaveBeenCalledWith(
    expect.objectContaining({
      path: "/api/v1/slides/generate/from-media",
      method: "POST"
    })
  )
})
```

- [ ] **Step 2: Run the presentations slice tests to verify failure**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
Expected: FAIL because `TldwApiClientBase` still defines the overlapping `presentations` methods.

- [ ] **Step 3: Delete the overlapping presentations class methods and update the overlap manifest**

Remove only these `presentations` methods from the base class:

- `generateSlidesFromMedia`
- `getPresentation`
- `exportPresentation`

Then remove the `presentations` overlap entries from `client-ownership.ts`.

- [ ] **Step 4: Re-run service and consumer-facing presentation regressions**

Run: `cd apps/packages/ui && bunx vitest run src/services/__tests__/tldw-api-client.ownership-guard.test.ts src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/services/__tests__/tldw-api-client.presentations-normalization.test.ts src/components/Option/PresentationStudio/__tests__/PresentationStudioBootstrap.test.tsx src/components/Option/PresentationStudio/__tests__/PresentationStudioCreatePayload.test.tsx src/components/Option/PresentationStudio/__tests__/PresentationStudioPage.test.tsx`
Expected: PASS, with no remaining Wave 5 overlap entries for `presentations`.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/services/tldw/client-ownership.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts
git commit -m "refactor: de-shadow presentations client slice"
```

### Task 5: Fix The `verify:openapi` Workflow And Write The Maintainer Handoff

**Files:**
- Create: `apps/packages/ui/src/services/tldw/README.md`
- Modify: `apps/packages/ui/package.json`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/fallback-schemas.ts`
- Modify: `Docs/superpowers/reviews/shared-api-client/2026-04-17-wave5-ownership-inventory.md`

- [ ] **Step 1: Prove the current UI-package workflow is still broken**

Run: `cd apps/packages/ui && npm run verify:openapi`
Expected: FAIL with a missing-script error because the runnable `verify:openapi` entry point only exists under `apps/extension/package.json`.

- [ ] **Step 2: Add the UI-package alias and update maintenance comments**

Add a local script alias in `apps/packages/ui/package.json`.

```json
{
  "scripts": {
    "verify:openapi": "node ../../extension/scripts/verify-openapi-client-paths.mjs"
  }
}
```

Update the maintenance comments in `openapi-guard.ts` and `fallback-schemas.ts` to point at the UI package command instead of an ambiguous workspace.

- [ ] **Step 3: Write the maintainer note and backlog handoff**

Add `apps/packages/ui/src/services/tldw/README.md` with:

```md
## Ownership Rules

- `TldwApiClient.ts` owns transport/bootstrap/caches/path helpers.
- `domains/*.ts` own feature-facing API methods.
- Focused helper modules are allowed only for runtime helpers shared across domains or needed to break class-file coupling.

## Deferred Follow-On Overlaps

- models-audio
- characters
- collections
- media
- chat-rag
```

Extend the inventory artifact with the explicit next action for the deferred slices.

- [ ] **Step 4: Run the final Wave 5 verification pack**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/__tests__/tldw-api-client.ownership-guard.test.ts \
  src/services/__tests__/tldw-api-client.exports.test.ts \
  src/services/__tests__/tldw-api-client.boundary-slices.test.ts \
  src/services/__tests__/tldw-api-client.presentations-normalization.test.ts \
  src/services/tldw/__tests__/shared-normalizers.test.ts \
  src/services/__tests__/request-core.path-normalization.test.ts \
  src/services/tldw/__tests__/request-core.hosted.test.ts \
  src/components/Option/Admin/__tests__/ServerAdminPage.media-budget.test.tsx \
  src/components/Option/Skills/__tests__/Manager.test.tsx \
  src/components/Option/PresentationStudio/__tests__/PresentationStudioBootstrap.test.tsx \
  src/components/Option/PresentationStudio/__tests__/PresentationStudioCreatePayload.test.tsx \
  src/components/Option/PresentationStudio/__tests__/PresentationStudioPage.test.tsx

cd apps/packages/ui && npm run verify:openapi
```

Expected:

- the ownership guard passes with only the deferred out-of-wave overlaps still recorded in the inventory artifact
- the helper extraction and slice regressions are green
- the consumer-facing admin, skills, and presentation tests remain green
- `verify:openapi` runs from the UI package successfully

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/README.md \
  apps/packages/ui/package.json \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  apps/packages/ui/src/services/tldw/fallback-schemas.ts \
  Docs/superpowers/reviews/shared-api-client/2026-04-17-wave5-ownership-inventory.md
git commit -m "docs: finalize wave5 client boundary handoff"
```
