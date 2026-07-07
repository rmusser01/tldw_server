# Character Expression Editor Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared Character Editor expression-image editor, chat discovery nudge, and browser-extension handoff for the existing `Emote: <state>` character-chat feature.

**Architecture:** Keep the feature in shared UI code. Reuse the existing character metadata pipeline, avatar image validation/generation behavior, and Characters route helpers; save expression images only to `extensions.tldw.mood_images` while reading legacy aliases. Chat and extension surfaces only discover or link to the shared editor, so there is one editor and one storage contract.

**Tech Stack:** React 18, TypeScript, Ant Design form controls, lucide-react icons, Vitest + Testing Library, WXT/browser runtime helpers, existing `tldwClient` image artifact APIs.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-07-07-character-expression-editor-onboarding-design.md`
- Backlog: `TASK-12906`
- Related parser/runtime follow-up: `TASK-12164`

## Scope Check

This plan covers one vertical slice:

- Shared Character Editor expression image rows.
- Canonical metadata read/write helpers.
- Chat nudge for characters with no configured expression images.
- Browser-extension handoff link to the shared editor.

It intentionally does not build a sidepanel editor, a chat setup wizard, backend persistence for nudge dismissal, a `set_emote` tool, or Persona Visual runtime unification.

## File Map

- Modify `apps/packages/ui/src/utils/character-mood.ts`: keep existing mood detection behavior, but align expression image helpers with the backend arbitrary slug rule and canonical metadata contract.
- Create or modify `apps/packages/ui/src/utils/character-emotes.ts`: provide `normalizeCharacterEmoteState()` if it is not already present after rebasing onto latest `dev`.
- Modify `apps/packages/ui/src/utils/__tests__/character-mood.test.ts`: cover arbitrary custom states, canonical precedence, alias cleanup, empty-map cleanup, and legacy reads.
- Create `apps/packages/ui/src/components/Option/Characters/character-expression-images.ts`: pure row/value helpers for the editor.
- Test `apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts`: cover row initialization, validation, duplicate detection, and payload conversion.
- Create `apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx`: compact editable section with URL/upload/generate image source controls, thumbnails, preview picker, and copy action.
- Test `apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx`: cover visible starter rows, add/remove, validation, preview fallback, and copy feedback.
- Modify `apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx`: replace the “Mood images coming soon” placeholder with `CharacterExpressionImagesSection`.
- Modify `apps/packages/ui/src/components/Option/Characters/utils.ts`: merge validated expression images into `extensions.tldw.mood_images`, preserve unrelated metadata, and block invalid raw metadata when expression rows must be merged.
- Modify `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx`: seed edit/duplicate forms from existing mood image metadata.
- Test or extend `apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx`: verify editor save payload stores canonical mood images.
- Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundComposerNotices.tsx`: add the dismissible expression setup nudge near existing chat notices.
- Modify `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx`: cover nudge visibility, scoped dismissal, and no-nudge when images exist.
- Modify `apps/packages/ui/src/utils/characters-route.ts`: add optional `focus` and `characterId` query parameters, and extract a reusable open helper only if it avoids copying existing sidepanel tab-opening logic.
- Modify `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`: reuse the route helper changes without changing current create behavior.
- Modify `apps/packages/ui/src/components/Sidepanel/Chat/CharacterControlsSheet.tsx`: add “Edit character expressions” handoff action.
- Test `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.expression-handoff.test.tsx`: verify extension handoff opens the Characters route/options hash with `focus=expressions`.
- Update `backlog/tasks/task-12168 - Implement-character-expression-editor-onboarding.md`: record touched files and verification as tasks complete.

## Implementation Tasks

### Task 0: Reconcile Against Latest `dev`

**Files:**
- Inspect: `apps/packages/ui/src/utils/character-mood.ts`
- Inspect: `apps/packages/ui/src/utils/character-emotes.ts`
- Inspect: `tldw_Server_API/app/core/Character_Chat/emote_directives.py`
- Modify only if missing after rebase: `apps/packages/ui/src/utils/character-emotes.ts`

- [ ] **Step 1: Fetch and rebase before code work**

Run:

```bash
git fetch origin
git rebase origin/dev
```

Expected: current implementation branch is based on latest `dev`, including the merged `Emote: <state>` parser/runtime work.

- [ ] **Step 2: Confirm frontend and backend slug rules match**

Run:

```bash
rg -n "CHARACTER_EMOTE_STATE_PATTERN|normalizeCharacterEmoteState|\\^\\[a-z0-9\\]" tldw_Server_API/app/core/Character_Chat apps/packages/ui/src/utils
```

Expected:

- Backend accepts `^[a-z0-9][a-z0-9_-]{0,39}$`.
- Frontend has or will get a helper with the same rule.
- Do not reintroduce built-in-only expression state restrictions for image maps.

- [ ] **Step 3: If missing, add the small frontend normalizer**

Use this exact behavior in `apps/packages/ui/src/utils/character-emotes.ts` if latest `dev` does not already provide it:

```ts
export const CHARACTER_EMOTE_STATE_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/

export const normalizeCharacterEmoteState = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.trim().toLowerCase()
  return CHARACTER_EMOTE_STATE_PATTERN.test(normalized) ? normalized : null
}
```

- [ ] **Step 4: Commit only if this task changed files**

```bash
git add apps/packages/ui/src/utils/character-emotes.ts
git commit -m "chore: align frontend emote state normalization"
```

Skip the commit if latest `dev` already has the helper.

### Task 1: Metadata Helper Contract

**Files:**
- Modify: `apps/packages/ui/src/utils/character-mood.ts`
- Test: `apps/packages/ui/src/utils/__tests__/character-mood.test.ts`

- [ ] **Step 1: Write failing helper tests**

Add tests that prove the storage contract:

```ts
it("reads arbitrary safe expression states from canonical mood images", () => {
  const images = getCharacterMoodImagesFromExtensions({
    tldw: { mood_images: { smirk: TINY_PNG_BASE64, "joy-soft": "https://example.test/joy.png" } }
  })

  expect(images.smirk).toMatch(/^data:image\/png;base64,/)
  expect(images["joy-soft"]).toBe("https://example.test/joy.png")
})

it("prefers canonical mood images as a whole map over legacy aliases", () => {
  const images = getCharacterMoodImagesFromExtensions({
    tldw: {
      mood_images: { happy: "https://example.test/happy.png" },
      moodImages: { sad: "https://example.test/sad.png" }
    },
    mood_images: { angry: "https://example.test/angry.png" }
  })

  expect(Object.keys(images)).toEqual(["happy"])
})

it("writes canonical mood images and removes legacy aliases", () => {
  const merged = mergeCharacterMoodImagesIntoExtensions(
    {
      tldw: { moodImages: { sad: "https://example.test/sad.png" }, prompt_preset: "roleplay" },
      mood_images: { angry: "https://example.test/angry.png" },
      moodImages: { confused: "https://example.test/confused.png" }
    },
    { smirk: "https://example.test/smirk.png" }
  )

  expect((merged as any).tldw.mood_images).toEqual({ smirk: "https://example.test/smirk.png" })
  expect((merged as any).tldw.moodImages).toBeUndefined()
  expect((merged as any).mood_images).toBeUndefined()
  expect((merged as any).moodImages).toBeUndefined()
  expect((merged as any).tldw.prompt_preset).toBe("roleplay")
})

it("removes mood image keys when saving an empty map", () => {
  const merged = mergeCharacterMoodImagesIntoExtensions(
    { tldw: { mood_images: { happy: "https://example.test/happy.png" } } },
    {}
  )

  expect((merged as any).tldw).toBeUndefined()
})
```

- [ ] **Step 2: Run the failing tests**

```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/character-mood.test.ts
```

Expected: at least the arbitrary-state test fails if the branch still has the old `CharacterMoodLabel`-only implementation.

- [ ] **Step 3: Implement the minimal helper changes**

Keep `detectCharacterMood()` and `normalizeCharacterMoodLabel()` for legacy automatic mood detection. Change only image-map helpers to use arbitrary emote states:

```ts
import { normalizeCharacterEmoteState } from "./character-emotes"

export type CharacterMoodImages = Record<string, string>

const normalizeMoodImageSource = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  if (!trimmed) return null
  if (
    trimmed.startsWith("data:image/") ||
    trimmed.startsWith("http://") ||
    trimmed.startsWith("https://")
  ) {
    return trimmed
  }
  return createImageDataUrl(trimmed)
}
```

In `getCharacterMoodImagesFromExtensions()`, normalize keys with `normalizeCharacterEmoteState(rawMood)` instead of `normalizeCharacterMoodLabel(rawMood)`.

In `mergeCharacterMoodImagesIntoExtensions()`, normalize keys the same way, write only `tldw.mood_images`, and always delete these aliases:

```ts
delete tldw.moodImages
delete parsed.mood_images
delete parsed.moodImages
```

In `resolveCharacterMoodImageUrl()`, first try `normalizeCharacterEmoteState(moodLabel)` and fall back to `normalizeCharacterMoodLabel(moodLabel)` only if runtime callers still pass legacy aliases.

- [ ] **Step 4: Run tests until green**

```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/character-mood.test.ts
```

Expected: all tests in `character-mood.test.ts` pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/utils/character-mood.ts apps/packages/ui/src/utils/__tests__/character-mood.test.ts apps/packages/ui/src/utils/character-emotes.ts
git commit -m "fix: support custom character expression image states"
```

### Task 2: Pure Editor Row Helpers

**Files:**
- Create: `apps/packages/ui/src/components/Option/Characters/character-expression-images.ts`
- Modify: `apps/packages/ui/src/components/Option/Characters/utils.ts`
- Test: `apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts`

- [ ] **Step 1: Write failing pure-helper tests**

Cover starter rows, legacy metadata loading, invalid states, duplicates, incomplete custom rows, and canonical payload conversion:

```ts
import {
  EXPRESSION_IMAGE_STARTER_STATES,
  expressionRowsFromExtensions,
  normalizeExpressionImageRows,
  expressionRowsToMoodImages
} from "../character-expression-images"
import { DEFAULT_CHARACTER_PROMPT_PRESET } from "@/data/character-prompt-presets"
import { applyCharacterMetadataToExtensions } from "../utils"

it("creates starter rows plus configured custom rows", () => {
  const rows = expressionRowsFromExtensions({
    tldw: {
      mood_images: {
        happy: "https://example.test/happy.png",
        smirk: "https://example.test/smirk.png"
      }
    }
  })

  expect(rows.map((row) => row.state)).toEqual([
    ...EXPRESSION_IMAGE_STARTER_STATES,
    "smirk"
  ])
  expect(rows.find((row) => row.state === "happy")?.starter).toBe(true)
  expect(rows.find((row) => row.state === "happy")?.image.url).toBe(
    "https://example.test/happy.png"
  )
  expect(rows.find((row) => row.state === "smirk")?.image.url).toBe(
    "https://example.test/smirk.png"
  )
})

it("blocks duplicate and incomplete custom rows", () => {
  const result = normalizeExpressionImageRows([
    { id: "1", state: "happy", image: { mode: "url", url: "https://example.test/happy.png", base64: "" }, starter: true },
    { id: "2", state: "happy", image: { mode: "url", url: "https://example.test/other.png", base64: "" }, starter: false },
    { id: "3", state: "smirk", image: { mode: "url", url: "", base64: "" }, starter: false }
  ])

  expect(result.errors).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ id: "2", reason: "duplicate" }),
      expect.objectContaining({ id: "3", reason: "missing-image" })
    ])
  )
})

it("drops empty starter rows and returns mood image map", () => {
  expect(
    expressionRowsToMoodImages([
      { id: "neutral", state: "neutral", image: { mode: "url", url: "", base64: "" }, starter: true },
      { id: "thinking", state: "thinking", image: { mode: "url", url: "https://example.test/thinking.png", base64: "" }, starter: true }
    ])
  ).toEqual({ thinking: "https://example.test/thinking.png" })
})

it("preserves invalid raw extensions when only empty starter rows exist", () => {
  const rawExtensions = "{not valid json"
  const result = applyCharacterMetadataToExtensions(rawExtensions, {
    preset: DEFAULT_CHARACTER_PROMPT_PRESET,
    expressionRows: expressionRowsFromExtensions({})
  })

  expect(result).toBe(rawExtensions)
})

it("blocks invalid raw extensions when expression rows need a metadata write", () => {
  const result = applyCharacterMetadataToExtensions("{not valid json", {
    preset: DEFAULT_CHARACTER_PROMPT_PRESET,
    expressionRows: [
      { id: "thinking", state: "thinking", image: { mode: "url", url: "https://example.test/thinking.png", base64: "" }, starter: true }
    ]
  })

  expect(result).toBeNull()
})
```

- [ ] **Step 2: Run failing helper tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts
```

Expected: test file fails because the helper module does not exist.

- [ ] **Step 3: Implement the helper module**

Create `apps/packages/ui/src/components/Option/Characters/character-expression-images.ts` with these exported shapes:

```ts
import { extractAvatarValues, createAvatarValue, type AvatarFieldValue } from "./AvatarField"
import {
  getCharacterMoodImagesFromExtensions,
  type CharacterMoodImages
} from "@/utils/character-mood"
import { normalizeCharacterEmoteState } from "@/utils/character-emotes"

export const EXPRESSION_IMAGE_STARTER_STATES = [
  "neutral",
  "happy",
  "sad",
  "angry",
  "thinking",
  "surprised"
] as const

export type ExpressionImageRow = {
  id: string
  state: string
  image: AvatarFieldValue
  starter: boolean
}

export type ExpressionImageRowErrorReason =
  | "invalid-state"
  | "duplicate"
  | "missing-state"
  | "missing-image"

export type ExpressionImageRowError = {
  id: string
  reason: ExpressionImageRowErrorReason
}
```

Required functions:

- `expressionRowsFromExtensions(extensions: unknown): ExpressionImageRow[]`
- `createEmptyCustomExpressionRow(): ExpressionImageRow`
- `normalizeExpressionImageRows(rows: ExpressionImageRow[]): { rows: ExpressionImageRow[]; errors: ExpressionImageRowError[] }`
- `expressionRowsToMoodImages(rows: ExpressionImageRow[]): CharacterMoodImages`

Implementation rules:

- Starter rows always appear in the fixed starter order.
- Configured custom states appear after starter rows, sorted by insertion/read order from metadata.
- `extractAvatarValues(row.image)` is the source of truth for URL/base64 extraction.
- Empty starter rows are ignored.
- Empty custom rows are invalid after creation.
- A valid custom state with no image is invalid and blocks save.
- Duplicate normalized states are invalid.
- This helper module must not import from `./utils`; keep dependencies one-way so `utils.ts` can import row helpers without a circular save-path dependency.
- Canonical extension merging is owned by `utils.ts`, not this helper.

- [ ] **Step 4: Extend character payload helper**

In `apps/packages/ui/src/components/Option/Characters/utils.ts`, extend the imports:

```ts
import {
  expressionRowsToMoodImages,
  normalizeExpressionImageRows,
  type ExpressionImageRow
} from "./character-expression-images"
import { mergeCharacterMoodImagesIntoExtensions } from "@/utils/character-mood"
```

Then update the params and return type for `applyCharacterMetadataToExtensions()`:

```ts
export const applyCharacterMetadataToExtensions = (
  rawExtensions: unknown,
  params: {
    preset: CharacterPromptPresetId
    defaultAuthorNote?: unknown
    generation?: CharacterGenerationSettings
    expressionRows?: ExpressionImageRow[]
  }
): Record<string, any> | string | undefined | null => {
```

Inside `applyCharacterMetadataToExtensions()`, change the existing invalid-raw-JSON guard to account for expression rows:

```ts
  const parsed = parseExtensionsObject(rawExtensions)
  const normalizedExpressionRows = Array.isArray(params.expressionRows)
    ? normalizeExpressionImageRows(params.expressionRows)
    : null
  const expressionMoodImages = normalizedExpressionRows
    ? expressionRowsToMoodImages(normalizedExpressionRows.rows)
    : {}
  const parsedTldw = parsed && isPlainObject(parsed.tldw) ? parsed.tldw : null
  const hadMoodImageKeys =
    Boolean(parsedTldw && ("mood_images" in parsedTldw || "moodImages" in parsedTldw)) ||
    Boolean(parsed && ("mood_images" in parsed || "moodImages" in parsed))
  const shouldMergeExpressionImages =
    Boolean(normalizedExpressionRows) &&
    (normalizedExpressionRows!.errors.length > 0 ||
      Object.keys(expressionMoodImages).length > 0 ||
      hadMoodImageKeys)
  const hadRawString =
    typeof rawExtensions === "string" &&
    rawExtensions.trim().length > 0 &&
    parsed === null

  if (hadRawString && shouldMergeExpressionImages) return null
  if (hadRawString) {
    return rawExtensions as string
  }
```

After the existing preset/default-author-note/generation code has updated `next`, and before the existing final `if (Object.keys(next).length > 0)` return block, insert:

```ts
  if (shouldMergeExpressionImages) {
    if (!normalizedExpressionRows || normalizedExpressionRows.errors.length > 0) {
      return null
    }
    next = mergeCharacterMoodImagesIntoExtensions(
      next,
      expressionMoodImages
    )
  }
```

In `buildCharacterPayload(values)`, pass `expressionRows: values.expression_images`. Empty starter-only rows are allowed to be present on normal create/edit forms; they must not force an expression metadata merge unless there is a configured image, invalid expression row, or existing mood-image metadata key to clean up.

If `applyCharacterMetadataToExtensions()` returns `null`, throw a validation error with a stable message that the form can display. This covers invalid raw metadata JSON when expression rows must be merged and the defensive case where invalid row data reaches the payload builder:

```ts
throw new Error("Invalid extensions JSON. Fix metadata before saving expression images.")
```

- [ ] **Step 5: Run helper tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts apps/packages/ui/src/utils/__tests__/character-mood.test.ts
```

Expected: both helper suites pass.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Characters/character-expression-images.ts apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts apps/packages/ui/src/components/Option/Characters/utils.ts
git commit -m "feat: add character expression image form helpers"
```

### Task 3: Character Editor Expression Section

**Files:**
- Create: `apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx`
- Test: `apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx`

- [ ] **Step 1: Write failing component tests**

Test only the section behavior, not the full manager:

```tsx
it("renders starter expression rows and adds a custom row", async () => {
  render(
    <Form initialValues={{ expression_images: expressionRowsFromExtensions({}) }}>
      <CharacterExpressionImagesSection characterName="Mira" characterDescription="Archivist" />
    </Form>
  )

  expect(screen.getByDisplayValue("neutral")).toBeInTheDocument()
  expect(screen.getByDisplayValue("thinking")).toBeInTheDocument()

  await userEvent.click(screen.getByRole("button", { name: /add expression/i }))
  expect(screen.getByLabelText(/custom expression state/i)).toBeInTheDocument()
})

it("copies the selected preview emote directive", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined)
  Object.assign(navigator, { clipboard: { writeText } })

  render(
    <Form
      initialValues={{
        expression_images: [
          { id: "thinking", state: "thinking", starter: true, image: { mode: "url", url: "https://example.test/thinking.png", base64: "" } }
        ]
      }}
    >
      <CharacterExpressionImagesSection characterName="Mira" />
    </Form>
  )

  await userEvent.click(screen.getByRole("button", { name: /copy emote directive/i }))
  expect(writeText).toHaveBeenCalledWith("Emote: thinking")
})
```

- [ ] **Step 2: Run failing component tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx
```

Expected: test file fails because the component does not exist.

- [ ] **Step 3: Implement compact row UI**

Create a small, local row component inside `CharacterExpressionImagesSection.tsx`; do not build a new generic asset editor. Required UI:

- Section title: `Expression images`.
- Short help text: `Map Emote: <state> directives to character images.`
- `Form.List name="expression_images"` rows.
- State slug `Input` with `aria-label`.
- Source mode `Radio.Group` or segmented control with URL, Upload, Generate.
- URL `Input` for URL mode.
- `Upload` with the same `MAX_AVATAR_IMAGE_BYTES`, MIME validation, and base64 validation behavior as `AvatarField`.
- Generate controls using `tldwClient.getImageBackends()` and `tldwClient.createImageArtifact()` with per-row loading/error state.
- Thumbnail preview and remove button.
- Add expression button.
- Preview picker, portrait preview, and copy action.

Keep helpers close to the component unless they are pure and already covered by Task 2.

- [ ] **Step 4: Use avatar image extraction rules**

Use `createImageDataUrl()` for base64 preview values and keep saved row values as `AvatarFieldValue`:

```ts
const getRowImageUrl = (value?: AvatarFieldValue): string => {
  if (!value) return ""
  if (value.mode === "url") return value.url?.trim() || ""
  return value.base64 ? createImageDataUrl(value.base64) || "" : ""
}
```

Remote preview image load failure should set row-local preview error only; it must not invalidate the row or clear the saved URL.

- [ ] **Step 5: Run component tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx
```

Expected: section tests pass.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx
git commit -m "feat: add character expression image editor section"
```

### Task 4: Wire Character Editor Save/Load

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx`
- Modify: `apps/packages/ui/src/components/Option/Characters/utils.ts`
- Test: `apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx` or a smaller existing form test if available

- [ ] **Step 1: Write failing integration test for edit/save payload**

Add or update a test that opens a character with legacy expression metadata, edits/saves, and asserts canonical payload:

```ts
expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
  expect.anything(),
  expect.objectContaining({
    extensions: expect.objectContaining({
      tldw: expect.objectContaining({
        mood_images: expect.objectContaining({
          smirk: "https://example.test/smirk.png"
        })
      }),
      mood_images: undefined,
      moodImages: undefined
    })
  })
)
```

Use a real object assertion instead of relying on `undefined` keys if the payload serializer omits them:

```ts
const payload = tldwClientMock.updateCharacter.mock.calls.at(-1)?.[1]
expect(payload.extensions.tldw.mood_images.smirk).toBe("https://example.test/smirk.png")
expect(payload.extensions.mood_images).toBeUndefined()
expect(payload.extensions.moodImages).toBeUndefined()
expect(payload.extensions.tldw.moodImages).toBeUndefined()
```

- [ ] **Step 2: Run failing test**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "expression"
```

Expected: fails until the editor is wired.

- [ ] **Step 3: Replace placeholder with the real section**

In `CharacterEditorForm.tsx`, import and render:

```tsx
<Form.Item
  noStyle
  shouldUpdate={(prev, cur) =>
    prev?.name !== cur?.name ||
    prev?.description !== cur?.description ||
    prev?.avatar !== cur?.avatar ||
    prev?.extensions !== cur?.extensions
  }
>
  {({ getFieldValue }) => (
    <CharacterExpressionImagesSection
      characterName={getFieldValue("name")}
      characterDescription={getFieldValue("description")}
      baseAvatar={getFieldValue("avatar")}
      rawExtensions={getFieldValue("extensions")}
    />
  )}
</Form.Item>
```

Remove only the “Mood images (coming soon)” placeholder block.

- [ ] **Step 4: Seed create/edit/duplicate form values**

In `useCharacterCrud.tsx`, import `expressionRowsFromExtensions()` and set:

```ts
expression_images: expressionRowsFromExtensions(record.extensions),
```

for edit and duplicate. For create/default form state, ensure `CharacterEditorForm` or the create modal initializes `expression_images` to `expressionRowsFromExtensions({})`.

- [ ] **Step 5: Validate expression rows before save**

In `CharacterExpressionImagesSection`, add a hidden `Form.Item` validator or list-level validation that uses `normalizeExpressionImageRows()` and returns visible inline errors. Save must be blocked for:

- invalid slug,
- duplicate slug,
- custom row with no state,
- custom row with valid state but no image,
- invalid raw `extensions` JSON when expression rows must be merged.

Do not block save for empty starter rows or broken remote preview loads.

- [ ] **Step 6: Run editor tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx -t "expression|advanced"
```

Expected: new expression tests pass; any existing advanced-field tests still pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx apps/packages/ui/src/components/Option/Characters/utils.ts apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx
git commit -m "feat: wire expression images into character editor"
```

### Task 5: Chat Setup Nudge

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundComposerNotices.tsx`
- Modify if stable server/user scope values are available at the call site: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx`

- [ ] **Step 1: Write failing nudge tests**

Add tests for:

- nudge appears when a selected character has no expression images,
- nudge does not appear when `extensions.tldw.mood_images` is non-empty,
- dismiss stores a scoped key containing every available stable value in order: server, user, character,
- the same character id on different server/user scopes does not share dismissal,
- no broad localStorage key is used when the character has no stable id.

Example assertion:

```ts
expect(screen.getByText(/add expression images/i)).toBeInTheDocument()
await userEvent.click(screen.getByRole("button", { name: /dismiss expression image setup/i }))
expect(
  localStorage.getItem(
    "character-expression-nudge:server:http://localhost:8000:user:7:character:42"
  )
).toBe("true")
```

- [ ] **Step 2: Run failing nudge tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx -t "expression"
```

Expected: fails because the nudge does not exist.

- [ ] **Step 3: Implement nudge helpers in the notice file**

Keep this local unless another file already has equivalent selected-character identity helpers:

```ts
type CharacterExpressionNudgeScopeInput = {
  server?: unknown
  user?: unknown
  character: unknown
}

const normalizeStableScopeValue = (value: unknown): string | null => {
  if (typeof value === "number" && Number.isFinite(value)) return String(value)
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

const readFirstStableRecordValue = (
  record: Record<string, unknown>,
  keys: string[]
): string | null => {
  for (const key of keys) {
    const normalized = normalizeStableScopeValue(record[key])
    if (normalized) return normalized
  }
  return null
}

const getStableCharacterNudgeScope = ({
  server,
  user,
  character
}: CharacterExpressionNudgeScopeInput): string | null => {
  if (!character || typeof character !== "object") return null
  const record = character as Record<string, unknown>
  const characterId = readFirstStableRecordValue(record, [
    "id",
    "character_id",
    "characterId",
    "slug"
  ])
  if (!characterId) return null

  const serverId =
    normalizeStableScopeValue(server) ||
    readFirstStableRecordValue(record, [
      "server_id",
      "serverId",
      "server_url",
      "serverUrl",
      "api_base_url",
      "apiBaseUrl"
    ])
  const userId =
    normalizeStableScopeValue(user) ||
    readFirstStableRecordValue(record, [
      "user_id",
      "userId",
      "owner_user_id",
      "ownerUserId"
    ])

  const parts: string[] = []
  if (serverId) parts.push("server", serverId)
  if (userId) parts.push("user", userId)
  parts.push("character", characterId)
  return parts.join(":")
}

const buildCharacterExpressionNudgeDismissKey = (
  scope: string | null
): string | null => {
  if (!scope) {
    return null
  }
  return `character-expression-nudge:${scope}`
}

const hasExpressionImages = (character: unknown): boolean => {
  if (!character || typeof character !== "object") return false
  const extensions = (character as Record<string, unknown>).extensions
  return Object.keys(getCharacterMoodImagesFromExtensions(extensions)).length > 0
}
```

Dismissal behavior:

- Include every available stable scope value in this order: server, user, character.
- If a stable character scope exists, persist to `localStorage` key `character-expression-nudge:${scope}`.
- If no stable scope exists, keep dismissal in React state for the current page session only.
- If `PlaygroundForm.tsx` has stable server/user values already in local state, storage, or config, add optional props to `PlaygroundComposerNotices` and pass them through. If it does not, derive server/user only from `selectedCharacter` fields and do not invent a new data source.

- [ ] **Step 4: Render the nudge near existing notices**

Render after `<ChatFirstRunNudge />` and before other transient notices:

```tsx
<CharacterExpressionSetupNudge
  selectedCharacter={selectedCharacter}
  selectedCharacterName={selectedCharacterName}
  serverScope={serverScope}
  userScope={userScope}
  t={t}
/>
```

The primary action should link to:

```tsx
<Link to={buildCharactersRoute({ from: "chat-emote-nudge", focus: "expressions" })}>
  {t("playground:composer.expressionNudgeAction", "Edit expressions")}
</Link>
```

Fallback to `/characters?from=chat-emote-nudge` if `focus` is not yet supported.

`focus=expressions` is a route hint. If the Characters page does not yet consume it, the link still satisfies the V1 fallback by opening the Characters page; direct section anchoring can be added only where the existing route layer supports it.

- [ ] **Step 5: Run nudge tests**

```bash
bunx vitest run apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx
```

Expected: existing first-run notice tests and new expression nudge tests pass.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/PlaygroundComposerNotices.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx
git commit -m "feat: nudge chat users to configure expression images"
```

### Task 6: Extension Handoff Route

**Files:**
- Modify: `apps/packages/ui/src/utils/characters-route.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/CharacterControlsSheet.tsx`
- Test: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.expression-handoff.test.tsx`

- [ ] **Step 1: Write failing route helper tests if none exist**

If `characters-route.ts` has no direct test, create `apps/packages/ui/src/utils/__tests__/characters-route.test.ts`:

```ts
it("builds characters route with expression focus", () => {
  expect(
    buildCharactersRoute({
      from: "sidepanel-character-controls",
      focus: "expressions",
      characterId: 42
    })
  ).toBe("/characters?from=sidepanel-character-controls&focus=expressions&characterId=42")
})
```

- [ ] **Step 2: Write failing sidepanel handoff test**

Mount `CharacterControlsSheet`, mock browser runtime where needed, click `Edit character expressions`, and assert the opened URL contains:

```text
/options.html#/characters?from=sidepanel-character-controls&focus=expressions
```

or the web fallback route:

```text
/characters?from=sidepanel-character-controls&focus=expressions
```

- [ ] **Step 3: Run failing tests**

```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/characters-route.test.ts apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.expression-handoff.test.tsx
```

Expected: route helper or action does not yet exist.

- [ ] **Step 4: Extend route helper narrowly**

In `characters-route.ts`:

```ts
type BuildCharactersRouteOptions = {
  from: string
  create?: boolean
  focus?: "expressions"
  characterId?: string | number | null
}

export const buildCharactersRoute = ({
  from,
  create = false,
  focus,
  characterId
}: BuildCharactersRouteOptions): string => {
  const params = new URLSearchParams({ from })
  if (create) params.set("create", "true")
  if (focus) params.set("focus", focus)
  if (characterId !== null && typeof characterId !== "undefined") {
    params.set("characterId", String(characterId))
  }
  return `/characters?${params.toString()}`
}
```

Preserve existing create-route output for `CharacterSelect`.

- [ ] **Step 5: Add handoff action**

In `CharacterControlsSheet.tsx`, add a small section or action near the selected assistant summary:

```tsx
<Button variant="outline" onClick={() => void handleOpenExpressionEditor()}>
  {t("playground:characterRail.editExpressions", "Edit character expressions")}
</Button>
```

`handleOpenExpressionEditor()` should reuse the same runtime/options-tab behavior as `CharacterSelect`. If copying more than about 20 lines, extract a tiny helper in `characters-route.ts` or a new `open-characters-workspace.ts` utility and update `CharacterSelect` to use it too.

- [ ] **Step 6: Run sidepanel tests**

```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/characters-route.test.ts apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.expression-handoff.test.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx
```

Expected: route helper, expression handoff, and existing role-play handoff tests pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/utils/characters-route.ts apps/packages/ui/src/utils/__tests__/characters-route.test.ts apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx apps/packages/ui/src/components/Sidepanel/Chat/CharacterControlsSheet.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.expression-handoff.test.tsx
git commit -m "feat: link extension users to character expression editor"
```

### Task 7: Final Verification and Backlog Update

**Files:**
- Update: `backlog/tasks/task-12168 - Implement-character-expression-editor-onboarding.md`

- [ ] **Step 1: Run targeted frontend tests**

```bash
bunx vitest run apps/packages/ui/src/utils/__tests__/character-mood.test.ts apps/packages/ui/src/components/Option/Characters/__tests__/character-expression-images.test.ts apps/packages/ui/src/components/Option/Characters/__tests__/CharacterExpressionImagesSection.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/CharacterControlsSheet.expression-handoff.test.tsx
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run package typecheck if feasible**

Prefer:

```bash
bun -C apps/tldw-frontend run typecheck
```

If the package script is unavailable or too broad for the current environment, run the closest existing frontend typecheck script and record the exact command/output in `TASK-12906`.

- [ ] **Step 3: Run final changed-scope diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional task files and feature files are modified relative to the starting dirty worktree.

- [ ] **Step 4: Bandit scope decision**

This implementation should touch TypeScript/frontend files only. Record in `TASK-12906`:

```text
Bandit: skipped; TASK-12906 touched frontend TypeScript/docs only and no Python code.
```

If Python files are touched despite the plan, run:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_task_12168.json
```

- [ ] **Step 5: Update Backlog task**

Record:

- plan path,
- touched files,
- verification commands and outcomes,
- known skips,
- final summary.

- [ ] **Step 6: Final commit**

```bash
git add backlog/tasks/task-12168\ -\ Implement-character-expression-editor-onboarding.md
git commit -m "chore: record character expression editor verification"
```

If all previous commits already include the task update, skip this commit.

## Implementation Notes

- Keep UI copy short. Do not add a tutorial modal or visible explanation of internals.
- Use icon buttons with `aria-label`/tooltips for row remove, upload clear, generate/regenerate, and copy.
- Use stable row dimensions for thumbnails and buttons so rows do not jump when image status changes.
- Do not put nested cards inside the Character Editor metadata card; use a simple bordered section and rows.
- Preserve existing `extensions` data unless it is one of the legacy mood-image aliases being intentionally removed during save.
- Do not store empty starter rows.
- Treat invalid metadata JSON as a save blocker only when expression rows need to be merged; keep existing behavior for unrelated metadata-only edits unless the final form-level validation deliberately tightens it.
- Do not use backend persistence for the chat nudge dismissal.
- Do not add new dependencies.

## Review Checkpoints

- After Task 1: verify arbitrary emote slugs still match backend parser behavior.
- After Task 4: manually inspect one saved payload to confirm canonical `extensions.tldw.mood_images` and no legacy aliases.
- After Task 6: inspect generated URLs for WebUI route, extension options hash, and in-place options navigation.
- Before PR: run the targeted tests and typecheck above, then request code review with `superpowers:requesting-code-review`.
