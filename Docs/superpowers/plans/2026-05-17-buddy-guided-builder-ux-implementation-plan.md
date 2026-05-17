# Buddy Guided Builder UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full guided Buddy builder in Persona Garden Visuals so users can choose bundled Basic defaults, import Codex/Petdex pets, review draft readiness, configure states/triggers, and explicitly activate a Persona Visual pack.

**Architecture:** Keep `VisualPackEditor` as the integration owner for existing service calls and pack lifecycle state, but extract the guided builder into focused Persona Garden components. All mutations continue to use existing Persona Visual endpoints and draft-first semantics. Movement states remain normal Persona Visual custom states, with a small runtime follow-through slice that maps Buddy drag direction into short-lived visual overrides when supported by the active pack.

**Tech Stack:** React 18, TypeScript, Vitest, Testing Library, Zustand runtime stores, existing Persona Visual services/types, sidepanel i18n JSON, existing `SpriteFrameRenderer`, existing `BuddyShellHost`.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-17-buddy-guided-builder-ux-design.md`
- Backlog: `TASK-420`
- Parent epic: `https://github.com/rmusser01/tldw_server/issues/1510`
- Related trackers: `https://github.com/rmusser01/tldw_server/issues/1787`, `https://github.com/rmusser01/tldw_server/issues/1803`

## Scope Decisions

- Keep this in `apps/packages/ui`; do not add backend endpoints for this plan.
- Implement the full guided builder in stages, but keep each task independently testable.
- Render the full builder in `VisualPackEditor` for the Visuals tab. Keep `VisualBuddySetupChoiceCard` as the compact Assistant Setup entry point that opens the Visuals detour.
- Use `search-lens-basic` as the current default fixture in new/updated tests. Keep `research-buddy-starter` only in explicit legacy compatibility tests.
- Start blank should continue to route to the existing draft-title/create-draft path for this plan. Do not invent a new blank-pack endpoint.
- Codex import review should show source semantics, atlas metadata, state coverage, and draft semantics. Defer a visual atlas row table unless the existing preview data already exposes it cleanly.
- Include movement runtime follow-through in this plan after the configuration UI lands.

## Plan Review Findings

This critique pass tightened the implementation plan before code starts:

- Task 1 introduces new archive-admission copy, so it must also add the English
  locale keys in the same commit. Do not wait until the builder shell task.
- The builder must become the primary Visuals-tab workflow for no-active,
  draft, review, and active-pack states. It is not only a first-run replacement
  for `VisualBuddySetupChoiceCard`.
- Codex/native source labels and atlas details must be derived from the current
  `PersonaVisualImportPreviewResponse` contract, especially `schema_version`,
  `bundle_summary.assets`, `asset_group`, `asset_role`, `width`, and `height`.
  Do not assume a future `source_format` field exists.
- Example code should use existing typed Persona Visual helpers such as
  `asPersonaVisualStateId()` and `asPersonaVisualCustomStateId()` instead of
  `as any`.
- Movement runtime work should consider pointer capture and stale-closure risk
  in `BuddyShellHost`, because drag handlers are window-level listeners.

## File Structure

Create:

- `apps/packages/ui/src/components/PersonaGarden/buddyBuilderArchive.ts`
  - Shared import archive admission constants and helper functions.
- `apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts`
  - Source-step types, step ordering, state reset helpers, tier grouping helpers, and review summarizers that do not render JSX.
- `apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx`
  - Top-level guided builder component. Receives existing `VisualPackEditor` handlers and data as props.
- `apps/packages/ui/src/components/PersonaGarden/BuddySourcePicker.tsx`
  - Source choice buttons and step navigation.
- `apps/packages/ui/src/components/PersonaGarden/BuddyStarterCatalogPicker.tsx`
  - Tier-aware starter catalog display for Basic, Intermediate, and Intricate packs.
- `apps/packages/ui/src/components/PersonaGarden/BuddyImportFormatPanel.tsx`
  - Native Persona Visual and Codex/Petdex archive admission copy and file input.
- `apps/packages/ui/src/components/PersonaGarden/BuddyDraftReviewPanel.tsx`
  - Draft/import review diagnostics, state coverage, source semantics, warnings, blockers, and optional sprite-frame preview.
- `apps/packages/ui/src/components/PersonaGarden/BuddyStateConfigurationPanel.tsx`
  - Common state, movement state, custom state, and authored-trigger configuration shell.
- `apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx`
- `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx`

Modify:

- `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
  - Render the guided builder as the primary Visuals-tab surface for first-run, draft, review, and active-pack states, while wiring existing handlers into it.
- `apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx`
  - Keep compact Assistant Setup behavior, but update copy if needed so it opens the builder rather than promising the complete setup UX.
- `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
  - Add integration coverage for the builder and update stale starter fixtures.
- `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx`
  - Keep compact-card tests aligned with the builder entry behavior.
- `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
  - Confirm Assistant Setup visual detour lands in the builder.
- `apps/packages/ui/src/routes/__tests__/sidepanel-persona-locale-keys.test.ts`
  - Add any new sidepanel i18n keys if this guard covers Persona Garden keys.
- `apps/packages/ui/src/assets/locale/en/sidepanel.json`
  - Add English keys for new builder labels.
- `apps/packages/ui/src/public/_locales/en/sidepanel.json`
  - Mirror English public locale keys if this repo keeps extension public assets in sync for sidepanel text.
- Other locale files only if current repo conventions require placeholder propagation. Do not hand-translate.
- `apps/packages/ui/src/store/persona-visual-runtime.ts`
  - Add a clear override helper if needed for drag release.
- `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
  - Set short-lived movement overrides during drag when supported.
- `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`
  - Add movement override tests.
- `apps/packages/ui/src/store/__tests__/persona-visual-runtime.test.ts`
  - Add store coverage if a new clear helper is introduced.

Do not touch:

- Backend Persona Visual import/copy/runtime code.
- VN/CYOA surfaces.
- Asset production files for intermediate or intricate Buddies.

## Task 1: Archive Admission And Starter Fixture Cleanup

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/buddyBuilderArchive.ts`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- Modify: `apps/packages/ui/src/public/_locales/en/sidepanel.json`

**Status:** Complete

- [x] **Step 1: Write archive admission helper tests**

Create `buddyBuilderArchive.test.ts` with cases for native pack, Codex ZIP, generic ZIP MIME, unsupported extension, and unsupported MIME.

```ts
import { describe, expect, it, vi } from "vitest"

import {
  BUDDY_IMPORT_ARCHIVE_ACCEPT,
  getBuddyImportArchiveFileError,
  isBuddyImportArchiveFile
} from "../buddyBuilderArchive"

const t = vi.fn((key: string, options?: { defaultValue?: string }) =>
  options?.defaultValue ?? key
)

describe("buddyBuilderArchive", () => {
  it("accepts native Persona Visual pack archives with normal zip media types", () => {
    const file = new File(["zip"], "pack.tldw-persona-vpack", {
      type: "application/zip"
    })

    expect(isBuddyImportArchiveFile(file)).toBe(true)
    expect(getBuddyImportArchiveFileError(file, t)).toBeNull()
  })

  it("accepts Codex and Petdex zip archives with generic browser media types", () => {
    const zip = new File(["zip"], "pet.zip", { type: "application/octet-stream" })
    const compressed = new File(["zip"], "pet.zip", {
      type: "application/x-zip-compressed"
    })

    expect(isBuddyImportArchiveFile(zip)).toBe(true)
    expect(isBuddyImportArchiveFile(compressed)).toBe(true)
  })

  it("rejects unsupported extensions before preview", () => {
    const file = new File(["image"], "pet.png", { type: "image/png" })

    expect(isBuddyImportArchiveFile(file)).toBe(false)
    expect(getBuddyImportArchiveFileError(file, t)).toContain(
      ".tldw-persona-vpack"
    )
  })

  it("keeps the file input accept string explicit", () => {
    expect(BUDDY_IMPORT_ARCHIVE_ACCEPT).toContain(".tldw-persona-vpack")
    expect(BUDDY_IMPORT_ARCHIVE_ACCEPT).toContain(".zip")
    expect(BUDDY_IMPORT_ARCHIVE_ACCEPT).toContain("application/zip")
  })
})
```

- [x] **Step 2: Run helper tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts
```

Expected: FAIL because `buddyBuilderArchive.ts` does not exist.

- [x] **Step 3: Implement the archive helper**

Create `buddyBuilderArchive.ts` with shared constants and helpers.

```ts
export type BuddyBuilderTranslate = (
  key: string,
  options?: Record<string, unknown>
) => string

export const NATIVE_PERSONA_VISUAL_PACK_EXTENSION = ".tldw-persona-vpack"
export const CODEX_PET_ARCHIVE_EXTENSION = ".zip"

export const BUDDY_IMPORT_ARCHIVE_EXTENSIONS = [
  NATIVE_PERSONA_VISUAL_PACK_EXTENSION,
  CODEX_PET_ARCHIVE_EXTENSION
] as const

export const BUDDY_IMPORT_ARCHIVE_MIME_TYPES = new Set([
  "application/octet-stream",
  "application/vnd.tldw.persona.visual-pack+zip",
  "application/x-zip-compressed",
  "application/zip"
])

export const BUDDY_IMPORT_ARCHIVE_ACCEPT = [
  ...BUDDY_IMPORT_ARCHIVE_EXTENSIONS,
  ...BUDDY_IMPORT_ARCHIVE_MIME_TYPES
].join(",")

export const hasBuddyImportArchiveExtension = (file: File | null): boolean => {
  if (!file) return true
  const fileName = file.name.toLowerCase()
  return BUDDY_IMPORT_ARCHIVE_EXTENSIONS.some((extension) =>
    fileName.endsWith(extension)
  )
}

export const hasBuddyImportArchiveMediaType = (file: File | null): boolean => {
  if (!file) return true
  const mediaType = file.type.trim().toLowerCase()
  return !mediaType || BUDDY_IMPORT_ARCHIVE_MIME_TYPES.has(mediaType)
}

export const isBuddyImportArchiveFile = (file: File | null): boolean =>
  hasBuddyImportArchiveExtension(file) && hasBuddyImportArchiveMediaType(file)

export const getBuddyImportArchiveFileError = (
  file: File | null,
  t: BuddyBuilderTranslate
): string | null => {
  if (isBuddyImportArchiveFile(file)) return null
  if (!hasBuddyImportArchiveExtension(file)) {
    return t("sidepanel:personaGarden.visuals.builder.importUnsupportedExtension", {
      defaultValue:
        "Choose a .tldw-persona-vpack or Codex/Petdex .zip archive."
    })
  }
  return t("sidepanel:personaGarden.visuals.builder.importUnsupportedMimeType", {
    defaultValue:
      "Choose a Persona Visual or Codex/Petdex archive with a supported zip media type."
  })
}
```

- [x] **Step 4: Wire `VisualPackEditor` to the helper**

Replace the local `PORTABLE_VISUAL_PACK_EXTENSION`, MIME set, `isPortableVisualPackFile`, and `getImportPreviewFileError` logic with imports from `buddyBuilderArchive.ts`. Keep export filename generation using `.tldw-persona-vpack`.

The upload input should use:

```tsx
accept={BUDDY_IMPORT_ARCHIVE_ACCEPT}
```

- [x] **Step 5: Add archive admission i18n keys**

Add the new archive error keys used by `getBuddyImportArchiveFileError()` in:

- `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- `apps/packages/ui/src/public/_locales/en/sidepanel.json`

Required keys:

```json
{
  "personaGarden": {
    "visuals": {
      "builder": {
        "importUnsupportedExtension": "Choose a .tldw-persona-vpack or Codex/Petdex .zip archive.",
        "importUnsupportedMimeType": "Choose a Persona Visual or Codex/Petdex archive with a supported zip media type."
      }
    }
  }
}
```

- [x] **Step 6: Update stale VisualPackEditor starter fixtures**

In `VisualPackEditor.test.tsx`, use `search-lens-basic` for default starter tests and expected copy calls. Keep `research-buddy-starter` only where the test name explicitly says it is testing legacy alias compatibility.

- [x] **Step 7: Run focused tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/services/__tests__/persona-visuals.test.ts src/routes/__tests__/sidepanel-persona-locale-keys.test.ts --testTimeout=30000
```

Expected: PASS.

- [x] **Step 8: Commit Task 1**

```bash
git add apps/packages/ui/src/components/PersonaGarden/buddyBuilderArchive.ts apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/packages/ui/src/assets/locale/en/sidepanel.json apps/packages/ui/src/public/_locales/en/sidepanel.json
git commit -m "feat: admit Codex buddy archives in visual import"
```

## Task 2: Guided Source And Draft Builder Shell

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts`
- Create: `apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/BuddySourcePicker.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/BuddyStarterCatalogPicker.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/BuddyImportFormatPanel.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- Modify: `apps/packages/ui/src/public/_locales/en/sidepanel.json`

**Status:** Complete

- [x] **Step 1: Write builder state helper tests**

Create tests for source reset behavior and starter tier grouping.

```ts
import { describe, expect, it } from "vitest"

import {
  BASIC_BUDDY_STARTER_IDS,
  groupBuddyStarterPacksByTier,
  resetBuddyBuilderForSource
} from "../buddyBuilderState"

describe("buddyBuilderState", () => {
  it("keeps the six Basic defaults in the expected order", () => {
    expect(BASIC_BUDDY_STARTER_IDS).toEqual([
      "search-lens-basic",
      "index-card-basic",
      "archive-cube-basic",
      "paperclip-basic",
      "terminal-tile-basic",
      "migu-marker-basic"
    ])
  })

  it("resets downstream import and draft state when the source changes", () => {
    const next = resetBuddyBuilderForSource(
      {
        source: "bundled",
        selectedStarterId: "search-lens-basic",
        selectedImportFile: new File(["zip"], "pet.zip"),
        importPreview: { job_id: "job-1" },
        selectedDraftPackId: "pack-1",
        activationReady: true
      },
      "codex_import"
    )

    expect(next.source).toBe("codex_import")
    expect(next.selectedStarterId).toBeNull()
    expect(next.selectedImportFile).toBeNull()
    expect(next.importPreview).toBeNull()
    expect(next.selectedDraftPackId).toBeNull()
    expect(next.activationReady).toBe(false)
  })

  it("groups starter packs by tier and treats only art-ready Basic packs as recommended", () => {
    const groups = groupBuddyStarterPacksByTier([
      {
        id: "search-lens-basic",
        title: "Search Lens",
        description: "",
        renderer_type: "sprite_frames",
        manifest_version: 1,
        states_offered: [],
        asset_count: 1,
        total_bytes: 1,
        tags: [],
        license_label: "bundled",
        complexity_tier: "basic",
        production_status: "art_ready",
        neutral_anchor_required: true,
        expected_asset_groups: [],
        animation_coverage_notes: []
      },
      {
        id: "lofi-study-intricate",
        title: "Lofi Study",
        description: "",
        renderer_type: "sprite_frames",
        manifest_version: 1,
        states_offered: [],
        asset_count: 1,
        total_bytes: 1,
        tags: [],
        license_label: "bundled",
        complexity_tier: "intricate",
        production_status: "scaffold",
        neutral_anchor_required: true,
        expected_asset_groups: [],
        animation_coverage_notes: []
      }
    ])

    expect(groups.basic[0].recommended).toBe(true)
    expect(groups.intricate[0].recommended).toBe(false)
  })
})
```

- [x] **Step 2: Run state tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts
```

Expected: FAIL because `buddyBuilderState.ts` does not exist.

- [x] **Step 3: Implement `buddyBuilderState.ts`**

Implement source IDs, step IDs, Basic default ID constants, reset helpers, and tier grouping. Keep this file JSX-free so tests stay cheap.

```ts
import type {
  PersonaVisualImportPreviewResponse,
  PersonaVisualImportPreviewStartResponse,
  PersonaVisualStarterPackSummary
} from "@/types/persona-visuals"

export type BuddyBuilderSource =
  | "bundled"
  | "codex_import"
  | "native_import"
  | "library"
  | "duplicate"
  | "blank"

export type BuddyBuilderStep =
  | "source"
  | "draft"
  | "review"
  | "configure"
  | "activate"

export const BUDDY_BUILDER_STEPS: BuddyBuilderStep[] = [
  "source",
  "draft",
  "review",
  "configure",
  "activate"
]

export const BASIC_BUDDY_STARTER_IDS = [
  "search-lens-basic",
  "index-card-basic",
  "archive-cube-basic",
  "paperclip-basic",
  "terminal-tile-basic",
  "migu-marker-basic"
] as const

const BASIC_BUDDY_STARTER_ID_SET = new Set<string>(BASIC_BUDDY_STARTER_IDS)

export type BuddyBuilderState = {
  source: BuddyBuilderSource | null
  selectedStarterId: string | null
  selectedImportFile: File | null
  importPreview:
    | PersonaVisualImportPreviewStartResponse
    | PersonaVisualImportPreviewResponse
    | Record<string, unknown>
    | null
  selectedDraftPackId: string | null
  activationReady: boolean
}

export const resetBuddyBuilderForSource = (
  state: BuddyBuilderState,
  source: BuddyBuilderSource
): BuddyBuilderState => ({
  ...state,
  source,
  selectedStarterId: null,
  selectedImportFile: null,
  importPreview: null,
  selectedDraftPackId: null,
  activationReady: false
})

export type BuddyStarterCatalogItem = PersonaVisualStarterPackSummary & {
  recommended: boolean
}

export const groupBuddyStarterPacksByTier = (
  packs: PersonaVisualStarterPackSummary[]
): Record<"basic" | "intermediate" | "intricate", BuddyStarterCatalogItem[]> => {
  const groups = {
    basic: [] as BuddyStarterCatalogItem[],
    intermediate: [] as BuddyStarterCatalogItem[],
    intricate: [] as BuddyStarterCatalogItem[]
  }
  for (const pack of packs) {
    groups[pack.complexity_tier].push({
      ...pack,
      recommended:
        pack.complexity_tier === "basic" &&
        pack.production_status === "art_ready" &&
        BASIC_BUDDY_STARTER_ID_SET.has(pack.id)
    })
  }
  return groups
}
```

- [x] **Step 4: Write builder render tests**

Create `BuddyGuidedBuilder.test.tsx` that renders the source step, verifies the six Basic defaults, verifies scaffold copy is visually distinct or primary-disabled, verifies Codex and native import choices are separate, and verifies accessible step navigation labels. Also cover an existing active-pack state so the builder does not regress into a first-run-only surface.

- [x] **Step 5: Run builder tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx
```

Expected: FAIL because the builder components do not exist.

- [x] **Step 6: Implement source, catalog, and import panels**

Implement the components with existing design-system classes/patterns used by Persona Garden panels. Requirements:

- no modal for the main flow,
- compact top stepper for narrow layouts,
- no nested cards,
- source buttons have accessible names,
- Basic recommended packs have primary copy action,
- Intermediate/Intricate scaffolds are visible but clearly labeled as production packets,
- Codex/Petdex `.zip` and native `.tldw-persona-vpack` copy are distinct.

- [x] **Step 7: Integrate builder into `VisualPackEditor`**

Wire the builder to existing state and handlers:

- starter packs and copy handler,
- import preview file state and preview handler,
- import commit handler,
- duplicate/library/blank handlers already in the editor,
- selected draft pack and active pack,
- validation/activation state.

Keep `VisualPackEditor` as the data/mutation owner. The builder should receive props and emit callbacks rather than importing services directly.

- [x] **Step 8: Update compact setup card semantics**

Keep `VisualBuddySetupChoiceCard` compact mode in Assistant Setup. Update labels/help so it says it opens Buddy visuals or starts the visual builder, not that the compact card itself is the complete builder.

- [x] **Step 9: Add i18n keys**

Add English keys under `personaGarden.visuals.builder` in:

- `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- `apps/packages/ui/src/public/_locales/en/sidepanel.json`

Use key names that map to actual UI sections, for example:

```json
{
  "personaGarden": {
    "visuals": {
      "builder": {
        "heading": "Buddy builder",
        "sourceStep": "Choose a source",
        "draftStep": "Create a draft",
        "reviewStep": "Review readiness",
        "configureStep": "Configure states",
        "activateStep": "Activate",
        "bundledSource": "Bundled Buddy",
        "codexImportSource": "Import Codex/Petdex pet",
        "nativeImportSource": "Import Persona Visual pack"
      }
    }
  }
}
```

Preserve existing JSON ordering conventions as much as practical.

- [x] **Step 10: Run focused source/draft tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx --testTimeout=30000
```

Expected: PASS.

- [x] **Step 11: Commit Task 2**

```bash
git add apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx apps/packages/ui/src/components/PersonaGarden/BuddySourcePicker.tsx apps/packages/ui/src/components/PersonaGarden/BuddyStarterCatalogPicker.tsx apps/packages/ui/src/components/PersonaGarden/BuddyImportFormatPanel.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/VisualBuddySetupChoiceCard.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx apps/packages/ui/src/assets/locale/en/sidepanel.json apps/packages/ui/src/public/_locales/en/sidepanel.json
git commit -m "feat: add guided Buddy source and draft builder"
```

## Task 3: Review Diagnostics And Draft Preview

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/BuddyDraftReviewPanel.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- Modify: `apps/packages/ui/src/public/_locales/en/sidepanel.json`

**Status:** Complete

- [x] **Step 1: Write review summarizer tests**

Add tests in `buddyBuilderState.test.ts` for a helper like `summarizeBuddyDraftReadiness()`.

Cases:

- Codex/Petdex preview displays source semantics and atlas dimensions from current backend preview metadata. Use `schema_version === "codex.pet.v1"` for Codex source semantics and derive atlas dimensions from `bundle_summary.assets[]` entries with `asset_group === "animation_atlas"` or `asset_role === "sprite_sheet"`.
- Native archive and Codex archive are distinguished by preview payload, not MIME.
- Missing required states produce activation blockers.
- `moving_right` and `moving_left` appear under movement states when in `state_catalog`.

- [x] **Step 2: Run summarizer tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts
```

Expected: FAIL because the summarizer does not exist.

- [x] **Step 3: Implement review summarizer helpers**

Add pure helpers to `buddyBuilderState.ts` for required state coverage, movement state extraction, custom state extraction, warnings/blockers extraction, and source-format labels. Do not parse ZIP data in the frontend.

Suggested shape:

```ts
export type BuddyDraftReadinessSummary = {
  sourceLabel: string
  atlasSummary: Array<{ assetId?: string; width: number | null; height: number | null }>
  requiredStates: Array<{ id: PersonaVisualBuiltinStateId; resolved: boolean }>
  movementStates: Array<{ id: "moving_right" | "moving_left"; resolved: boolean }>
  customStates: Array<{ id: string; label: string; kind: string; fallback?: string }>
  blockers: string[]
  warnings: string[]
  canActivate: boolean
}
```

- [x] **Step 4: Write panel render tests**

Create `BuddyDraftReviewPanel.test.tsx` to verify:

- Codex import shows "imported as a Persona Visual draft" semantics.
- Backend source type is shown from preview/result data.
- Blockers disable the activation path.
- Existing `SpriteFrameRenderer` is used when a draft has renderable `sprite_frames` assets.
- Missing preview bytes falls back to diagnostics and does not use a handcrafted HTML mockup.

- [x] **Step 5: Run panel tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx
```

Expected: FAIL because the panel does not exist.

- [x] **Step 6: Implement `BuddyDraftReviewPanel`**

Render:

- status/source summary,
- starter production status and tier where available,
- import preview blockers/warnings/conflicts,
- Codex/Petdex atlas dimensions from `bundle_summary.assets` if present in backend preview,
- built-in state coverage,
- custom states,
- movement states,
- asset role/group summary,
- activation blockers,
- optional `SpriteFrameRenderer` preview for the selected draft and `idle` state.

- [x] **Step 7: Integrate review panel into the builder**

In `BuddyGuidedBuilder`, make the Review step the canonical place to commit imports, inspect copied drafts, and proceed to Configure or Activate.

- [x] **Step 8: Add i18n keys**

Add review-specific keys under `personaGarden.visuals.builder.review`.

- [x] **Step 9: Run focused review tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testTimeout=30000
```

Expected: PASS.

- [x] **Step 10: Commit Task 3**

```bash
git add apps/packages/ui/src/components/PersonaGarden/BuddyDraftReviewPanel.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/packages/ui/src/assets/locale/en/sidepanel.json apps/packages/ui/src/public/_locales/en/sidepanel.json
git commit -m "feat: add Buddy draft review diagnostics"
```

## Task 4: State And Trigger Configuration Panels

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/BuddyStateConfigurationPanel.tsx`
- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/sidepanel.json`
- Modify: `apps/packages/ui/src/public/_locales/en/sidepanel.json`

**Status:** Complete

- [x] **Step 1: Write configuration panel tests**

Create `BuddyStateConfigurationPanel.test.tsx` covering:

- core states render in the documented order,
- `moving_right` and `moving_left` render as movement states, not task-running states,
- custom states from `state_catalog` render with label, kind, description, tags, and fallback,
- exact `tool_name` triggers render separately from `tool_category`,
- controls have accessible names,
- saving delegates to the existing manifest save callback.

- [x] **Step 2: Run configuration panel tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx
```

Expected: FAIL because the panel does not exist.

- [x] **Step 3: Implement `BuddyStateConfigurationPanel`**

Use the existing manifest data. Do not create a new manifest schema. Render:

- Core states,
- Movement states,
- Custom states,
- Tool/runtime triggers,
- Advanced manifest details.

Prefer read/edit controls that call existing `VisualPackEditor` manifest update handlers. If full inline editing is too large for one task, keep low-risk controls read-only and route edits to existing advanced sections, but tests must prove the grouped sections and save path are present.

- [x] **Step 4: Integrate Configure step**

Wire `BuddyStateConfigurationPanel` into the builder Configure step. Preserve existing raw manifest editor controls under an advanced section.

- [x] **Step 5: Add i18n keys**

Add configuration keys under `personaGarden.visuals.builder.configure`.

- [x] **Step 6: Run focused configuration tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx --testTimeout=30000
```

Expected: PASS.

- [x] **Step 7: Commit Task 4**

```bash
git add apps/packages/ui/src/components/PersonaGarden/BuddyStateConfigurationPanel.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx apps/packages/ui/src/components/PersonaGarden/BuddyGuidedBuilder.tsx apps/packages/ui/src/components/PersonaGarden/buddyBuilderState.ts apps/packages/ui/src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/packages/ui/src/assets/locale/en/sidepanel.json apps/packages/ui/src/public/_locales/en/sidepanel.json
git commit -m "feat: configure Buddy visual states in builder"
```

## Task 5: Movement Runtime Follow-Through

**Files:**
- Modify: `apps/packages/ui/src/store/persona-visual-runtime.ts`
- Modify: `apps/packages/ui/src/store/__tests__/persona-visual-runtime.test.ts`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx`
- Modify: `apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx`

**Status:** Not Started

- [ ] **Step 1: Write runtime store tests if a clear helper is needed**

If `BuddyShellHost` needs explicit clearing beyond `clearExpired`, add `clearOverride()` and test it.

```ts
import { describe, expect, it } from "vitest"

import { usePersonaVisualRuntimeStore } from "../persona-visual-runtime"
import { asPersonaVisualStateId } from "@/types/persona-visuals"

describe("persona visual runtime override clearing", () => {
  it("clears the current runtime override without touching diagnostics", () => {
    usePersonaVisualRuntimeStore.getState().setOverride({
      personaId: "persona-1",
      sessionId: null,
      state: asPersonaVisualStateId("moving_right"),
      reason: "buddy_drag",
      expiresAt: Date.now() + 500
    })

    usePersonaVisualRuntimeStore.getState().clearOverride()

    expect(usePersonaVisualRuntimeStore.getState().override).toBeNull()
  })
})
```

- [ ] **Step 2: Write Buddy drag movement tests**

In `BuddyShellHost.test.tsx`, add tests for:

- active pack declares `moving_right` and drag to the right sets a runtime override,
- active pack declares `moving_left` and drag to the left sets a runtime override,
- pointerup clears movement override,
- packs without movement states continue to move the dock without setting override.

- [ ] **Step 3: Run runtime tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/store/__tests__/persona-visual-runtime.test.ts src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx --testTimeout=30000
```

Expected: FAIL until movement override logic exists.

- [ ] **Step 4: Implement runtime store helper if needed**

Add:

```ts
clearOverride: () => void
```

to `PersonaVisualRuntimeStore` and implementation:

```ts
clearOverride: () => set({ override: null })
```

- [ ] **Step 5: Implement directional drag override**

In `BuddyShellHost.tsx`:

- keep existing dock position clamping,
- track last pointer X while dragging,
- when horizontal movement exceeds a small threshold, call `setOverride`,
- only set `moving_right` or `moving_left` if the active pack's manifest declares that state in `state_catalog` or `states`,
- use reason `buddy_drag`,
- use short expiry, for example `Date.now() + 300`,
- clear the override on pointerup.
- preserve existing window-level pointer listener cleanup and account for stale
  closures with refs or carefully scoped dependencies.
- keep or improve pointer capture on the drag handle so release handling stays
  reliable outside the dock element.

Suggested helper:

```ts
import { asPersonaVisualCustomStateId } from "@/types/persona-visuals"

const hasVisualState = (
  manifest: PersonaVisualManifest | null | undefined,
  state: "moving_right" | "moving_left"
) =>
  Boolean(
    manifest?.states?.[asPersonaVisualCustomStateId(state)] ||
      manifest?.state_catalog?.[asPersonaVisualCustomStateId(state)]
  )
```

- [ ] **Step 6: Run runtime tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/store/__tests__/persona-visual-runtime.test.ts src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx --testTimeout=30000
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add apps/packages/ui/src/store/persona-visual-runtime.ts apps/packages/ui/src/store/__tests__/persona-visual-runtime.test.ts apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx
git commit -m "feat: animate Buddy movement states while dragging"
```

## Task 6: Integrated Verification, Browser QA, And Closeout

**Files:**
- Modify if needed: `apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaGardenPanels.i18n.test.tsx`
- Modify if needed: `apps/packages/ui/src/routes/__tests__/sidepanel-persona-locale-keys.test.ts`
- Modify: `backlog/tasks/task-420 - Plan-Buddy-default-selection-and-Codex-import-UX.md`

**Status:** Not Started

- [ ] **Step 1: Run full focused frontend suite for this surface**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/buddyBuilderArchive.test.ts src/components/PersonaGarden/__tests__/buddyBuilderState.test.ts src/components/PersonaGarden/__tests__/BuddyGuidedBuilder.test.tsx src/components/PersonaGarden/__tests__/BuddyDraftReviewPanel.test.tsx src/components/PersonaGarden/__tests__/BuddyStateConfigurationPanel.test.tsx src/components/PersonaGarden/__tests__/VisualBuddySetupChoiceCard.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/store/__tests__/persona-visual-runtime.test.ts src/routes/__tests__/sidepanel-persona.test.tsx src/services/__tests__/persona-visuals.test.ts --testTimeout=30000
```

Expected: PASS.

- [ ] **Step 2: Run locale/design-state guards**

Run:

```bash
cd apps/packages/ui
bun run verify:design-system-state
bunx vitest run src/components/PersonaGarden/__tests__/PersonaGardenPanels.i18n.test.tsx src/routes/__tests__/sidepanel-persona-locale-keys.test.ts --testTimeout=30000
```

Expected: PASS or existing documented baseline exceptions only for `verify:design-system-state`.

- [ ] **Step 3: Run TypeScript check if practical**

Use the repo's current frontend typecheck command if one exists in the active branch. If only broad repo-wide `tsc` is available and fails on known unrelated baseline errors, capture the log and confirm no diagnostics mention touched files.

Suggested command:

```bash
cd apps/packages/ui
bunx tsc --noEmit --pretty false
```

Expected: PASS, or documented unrelated baseline failures with no touched-file diagnostics.

- [ ] **Step 4: Run browser QA**

Start or reuse the WebUI dev server, then verify:

- Persona Garden Visuals no-active-pack state shows the builder,
- Basic catalog shows six art-ready defaults,
- Codex `.zip` selection reaches preview path instead of file-gate rejection,
- draft review diagnostics are readable,
- Assistant Setup visual detour lands in the builder,
- narrow viewport uses compact stepper/accordion instead of cramped rail.

Use the Browser plugin when available for local target verification. Otherwise use the repo's existing Playwright flow and record the reason.

- [ ] **Step 5: Run diff checks**

Run:

```bash
git diff --check
git diff --cached --check
```

Expected: PASS.

- [ ] **Step 6: Run Bandit or document skip**

This plan is frontend TypeScript plus Markdown. Bandit is not applicable unless backend Python files are touched unexpectedly. If any Python files are touched, run:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_buddy_builder.json
```

Expected: zero new findings in touched Python scope.

- [ ] **Step 7: Update TASK-420 final notes**

Update the Backlog task with:

- implementation summary,
- verification commands and outcomes,
- known skips or blockers,
- PR link if opened.

- [ ] **Step 8: Commit closeout**

```bash
git add backlog/tasks/task-420\ -\ Plan-Buddy-default-selection-and-Codex-import-UX.md
git commit -m "docs: record Buddy builder verification"
```

## PR Preparation

- [ ] Confirm `git status --short` is clean after final commit.
- [ ] Confirm branch is based on latest `origin/dev`; rebase if needed.
- [ ] Open a PR against `dev`.
- [ ] In the PR body, include a human-owned `Change summary` placeholder for the requester rather than fabricating it.
- [ ] Include verification results, known skips, and screenshots or Browser QA notes.
