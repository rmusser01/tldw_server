# Writing Playground Document-First Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a document-first proposed-edit workflow to the shared Writing Playground so users can request creative writing changes, review diffs, and apply or reject safe proposals.

**Architecture:** Keep the feature inside the existing shared `WritingPlayground` surface used by WebUI and extension. Add pure revision utilities, a structured proposal parser/prompt builder, a small persistence bridge into the existing writing session save path, and focused UI components for actions, diffs, and the revision queue. Stage the implementation so plain-text proposals work first; rich editor behavior can honestly fall back to copy/manual apply when preserving rich structure is unsafe.

**Tech Stack:** React, TypeScript, Ant Design, lucide-react, TipTap, TanStack Query, existing `TldwChatService`, existing `/api/v1/writing/*` session persistence, Vitest, Testing Library, Playwright extension smoke tests.

---

## Source Documents

- Design spec: `Docs/superpowers/specs/2026-05-22-writing-playground-document-first-revisions-design.md`
- Design task: `backlog/tasks/task-443 - Design-document-first-Writing-Playground-revision-workflow.md`
- Planning task: `backlog/tasks/task-458 - Plan-document-first-Writing-Playground-revision-implementation.md`

## Scope Check

This plan is a single client-first implementation slice. It deliberately avoids:

- backend revision-history APIs
- full rich-text operational transforms
- persistent comment threads or annotations
- a new route or chat-first writing page
- broad `WritingPlayground/index.tsx` restructuring beyond the local seams needed for this feature

Before executing Task 1, create a Backlog.md implementation task for the code work and record this
plan path on that task. The planning task is `TASK-458`; do not use it as the implementation task
once source files are being changed.

## File Structure

Create focused files under the existing Writing Playground directory:

- Create `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts`
  - Shared revision action, operation, preset, target, status, proposal, schema-versioned
    payload, and apply result types.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-presets.ts`
  - Defines inspectable workflow presets, labels, and instruction text.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-utils.ts`
  - Pure helpers for word counts, action-aware target resolution, document fingerprints,
    insertion anchors, destructive-target confirmation, drift detection, retargeting, and
    plain-text apply plans.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-prompt-utils.ts`
  - Builds non-streaming proposed-edit prompts from the document, target, preset, and existing
    Writing Playground context; validates model responses into proposal drafts or raw/advisory
    fallbacks.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts`
  - Owns revision queue state, session-payload serialization, proposal create/apply/reject/regenerate state transitions, and delegates text mutation through existing editor/session callbacks.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx`
  - Compact action controls for Continue, Rewrite, Expand, Tighten, Tone, Outline, Custom, preset
    selection, target summary, and destructive-target confirmation.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionDiff.tsx`
  - Small text diff view for before/after proposal review.
- Create `apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionQueue.tsx`
  - Proposal cards with Apply, Reject, Copy, Regenerate, conflict, raw-suggestion, and advisory states.

Modify existing files:

- Modify `apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts`
  - Add schema-versioned revision payload support to `WritingSessionPayload`.
  - Add payload helpers for `getRevisionsFromPayload`, `mergeRevisionsIntoPayload`, and a stable revision signature.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingSessionManagement.ts`
  - Track revision payload signatures or expose a narrow payload-patch save helper so proposal-only changes persist and do not overwrite unsaved prompt/settings changes.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
  - Wire action bar, queue, non-streaming proposal generation, proposal apply, and status bar counts into the existing editor layout.
- Modify locale files only if the implementation chooses translation keys instead of inline fallback labels:
  - `apps/packages/ui/src/assets/locale/en/option.json`
  - `apps/packages/ui/src/public/_locales/en/option.json`

Add focused tests:

- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx`
- Create `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx`
- Modify or add integration coverage near:
  - `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx`
  - `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts`
  - `apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts`
  - `apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts`

## Implementation Tasks

### Task 1: Add Revision Types And Pure Apply Utilities

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-utils.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts`

- [ ] **Step 1: Write failing utility tests**

Cover these cases:

```ts
import {
  buildInsertionAnchor,
  confirmRevisionTarget,
  countWords,
  createDocumentFingerprint,
  findParagraphRange,
  planRevisionApply,
  resolveRevisionTarget
} from "../writing-revision-utils"

describe("writing revision utilities", () => {
  it("counts words and selected words deterministically", () => {
    expect(countWords("One two\nthree.")).toBe(3)
    expect(countWords("   ")).toBe(0)
  })

  it("resolves the current paragraph around a cursor", () => {
    const text = "Alpha one.\n\nBeta two.\nBeta three."
    expect(findParagraphRange(text, 14)).toEqual({ start: 12, end: 33 })
  })

  it("plans a direct replacement when beforeText still matches", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: "Omega"
    })
    expect(planRevisionApply("Alpha beta", proposal)).toEqual({
      type: "apply",
      start: 0,
      end: 5,
      nextText: "Omega beta"
    })
  })

  it("conflicts when replacement target drift is ambiguous", () => {
    const proposal = makeReplacementProposal({
      start: 0,
      end: 5,
      beforeText: "Alpha",
      replacementText: "Omega"
    })
    expect(planRevisionApply("Intro Alpha beta Alpha", proposal).type).toBe("conflict")
  })

  it("retargets a zero-length insertion by unique prefix and suffix anchor", () => {
    const original = "Alpha beta"
    const anchor = buildInsertionAnchor(original, 5)
    const proposal = makeInsertionProposal({
      start: 5,
      end: 5,
      beforeText: "",
      replacementText: " brave",
      anchor
    })
    expect(planRevisionApply("Intro. Alpha beta", proposal)).toMatchObject({
      type: "retarget"
    })
  })

  it("does not treat empty beforeText as a safe insertion match after drift", () => {
    const proposal = makeInsertionProposal({
      start: 5,
      end: 5,
      beforeText: "",
      replacementText: " brave",
      anchor: {
        documentFingerprint: createDocumentFingerprint("Alpha beta"),
        prefix: "Alpha",
        suffix: " beta"
      }
    })
    expect(planRevisionApply("Completely different", proposal).type).toBe("conflict")
  })

  it("targets the whole document for advisory outline requests", () => {
    const text = "First paragraph.\n\nSecond paragraph."
    const target = resolveRevisionTarget({
      text,
      action: "outline",
      operation: "advisory",
      cursor: 3
    })
    expect(target).toMatchObject({
      mode: "document",
      start: 0,
      end: text.length,
      requiresConfirmation: false
    })
  })

  it("requires confirmation before large text-changing document targets", () => {
    const target = resolveRevisionTarget({
      text: "First paragraph.\n\nSecond paragraph.",
      action: "rewrite",
      operation: "replace",
      cursor: 3,
      preferredTargetMode: "document"
    })
    expect(target).toMatchObject({
      mode: "document",
      requiresConfirmation: true
    })
  })

  it("allows apply after a whole-document text-changing target is confirmed", () => {
    const target = resolveRevisionTarget({
      text: "First paragraph.\n\nSecond paragraph.",
      action: "rewrite",
      operation: "replace",
      cursor: 3,
      preferredTargetMode: "document"
    })
    const confirmed = confirmRevisionTarget(target)
    expect(confirmed.requiresConfirmation).toBe(false)
    expect(confirmed.confirmationReason).toBeUndefined()
  })

  it("surfaces the resolved target for custom requests before generation", () => {
    const target = resolveRevisionTarget({
      text: "First paragraph.\n\nSecond paragraph.",
      action: "custom",
      operation: "replace",
      cursor: 20
    })
    expect(target.mode).toBe("paragraph")
    expect(target.label).toContain("paragraph")
  })
})
```

The helper factories can live inside the test file.

- [ ] **Step 2: Run the failing tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts
```

Expected: FAIL because the new utility module does not exist.

- [ ] **Step 3: Implement the types**

Create `writing-revision-types.ts` with:

```ts
export type WritingRevisionAction =
  | "continue"
  | "rewrite"
  | "expand"
  | "tighten"
  | "tone"
  | "outline"
  | "custom"

export type WritingRevisionOperation = "insert" | "replace" | "advisory"

export type WritingRevisionPresetId =
  | "draft_freely"
  | "polish_prose"
  | "developmental_edit"
  | "preserve_voice"
  | "make_concise"
  | "expand_sensory_detail"

export type WritingRevisionStatus =
  | "pending"
  | "applied"
  | "rejected"
  | "conflict"
  | "raw_suggestion"
  | "advisory"

export type WritingRevisionAnchor = {
  documentFingerprint: string
  prefix: string
  suffix: string
}

export type WritingRevisionTarget = {
  mode: "selection" | "paragraph" | "cursor" | "document"
  start: number
  end: number
  beforeText: string
  anchor: WritingRevisionAnchor
  label: string
  requiresConfirmation: boolean
  confirmationReason?: string
}

export type WritingRevisionProposal = {
  id: string
  sessionId: string
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  presetId?: WritingRevisionPresetId | null
  presetInstruction?: string | null
  instruction: string
  target: WritingRevisionTarget
  replacementText?: string
  rawText?: string
  rationale?: string
  title?: string
  notes?: string[]
  regeneratedFromId?: string
  createdAt: string
  status: WritingRevisionStatus
}

export type WritingRevisionApplyPlan =
  | { type: "apply"; start: number; end: number; nextText: string }
  | { type: "retarget"; start: number; end: number; nextText: string }
  | { type: "conflict"; reason: string }
  | { type: "noop"; reason: string }

export type WritingRevisionPayload = {
  schemaVersion: 1
  items: WritingRevisionProposal[]
}
```

- [ ] **Step 4: Implement pure utility functions**

Create `writing-revision-utils.ts` with:

```ts
import type {
  WritingRevisionAction,
  WritingRevisionAnchor,
  WritingRevisionApplyPlan,
  WritingRevisionOperation,
  WritingRevisionProposal,
  WritingRevisionTarget
} from "./writing-revision-types"

const DEFAULT_ANCHOR_WINDOW = 80
const DEFAULT_LARGE_TARGET_CHARS = 1800

export const countWords = (text: string): number => {
  const matches = text.trim().match(/\S+/g)
  return matches ? matches.length : 0
}

export const createDocumentFingerprint = (text: string): string => {
  let hash = 2166136261
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(16)
}

export const buildInsertionAnchor = (
  text: string,
  offset: number,
  windowSize = DEFAULT_ANCHOR_WINDOW
): WritingRevisionAnchor => {
  const safeOffset = Math.max(0, Math.min(text.length, offset))
  return {
    documentFingerprint: createDocumentFingerprint(text),
    prefix: text.slice(Math.max(0, safeOffset - windowSize), safeOffset),
    suffix: text.slice(safeOffset, Math.min(text.length, safeOffset + windowSize))
  }
}

export const findParagraphRange = (
  text: string,
  cursor: number
): { start: number; end: number } => {
  const safeCursor = Math.max(0, Math.min(text.length, cursor))
  const before = text.lastIndexOf("\n\n", safeCursor - 1)
  const after = text.indexOf("\n\n", safeCursor)
  return {
    start: before === -1 ? 0 : before + 2,
    end: after === -1 ? text.length : after
  }
}

const makeRevisionTarget = (input: {
  text: string
  mode: WritingRevisionTarget["mode"]
  start: number
  end: number
  operation: WritingRevisionOperation
  label: string
  confirmationReason?: string
}): WritingRevisionTarget => {
  const start = Math.max(0, Math.min(input.text.length, input.start))
  const end = Math.max(start, Math.min(input.text.length, input.end))
  const isTextChanging = input.operation !== "advisory"
  return {
    mode: input.mode,
    start,
    end,
    beforeText: input.text.slice(start, end),
    anchor: buildInsertionAnchor(input.text, start),
    label: input.label,
    requiresConfirmation: Boolean(isTextChanging && input.confirmationReason),
    confirmationReason: isTextChanging ? input.confirmationReason : undefined
  }
}

export const confirmRevisionTarget = (
  target: WritingRevisionTarget
): WritingRevisionTarget => ({
  ...target,
  requiresConfirmation: false,
  confirmationReason: undefined
})

export const resolveRevisionTarget = (input: {
  text: string
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  selection?: { start: number; end: number } | null
  cursor?: number | null
  preferredTargetMode?: WritingRevisionTarget["mode"] | null
  maxAutomaticTargetCharacters?: number
}): WritingRevisionTarget => {
  const { text, action, operation, selection } = input
  const maxAutomaticTargetCharacters =
    input.maxAutomaticTargetCharacters ?? DEFAULT_LARGE_TARGET_CHARS
  if (selection && selection.start !== selection.end) {
    const start = Math.max(0, Math.min(text.length, selection.start))
    const end = Math.max(start, Math.min(text.length, selection.end))
    return makeRevisionTarget({
      text,
      mode: "selection",
      start,
      end,
      operation,
      label: "selection"
    })
  }

  const cursor = Math.max(0, Math.min(text.length, input.cursor ?? text.length))
  if (action === "continue") {
    return makeRevisionTarget({
      text,
      mode: "cursor",
      start: cursor,
      end: cursor,
      operation: "insert",
      label: cursor === text.length ? "document end" : "cursor"
    })
  }

  if (action === "outline" || operation === "advisory") {
    return makeRevisionTarget({
      text,
      mode: "document",
      start: 0,
      end: text.length,
      operation: "advisory",
      label: "whole document"
    })
  }

  if (input.preferredTargetMode === "document") {
    return makeRevisionTarget({
      text,
      mode: "document",
      start: 0,
      end: text.length,
      operation,
      label: "whole document",
      confirmationReason: "Confirm before applying a whole-document text-changing request."
    })
  }

  const paragraph = findParagraphRange(text, cursor)
  const paragraphLength = paragraph.end - paragraph.start
  if (paragraphLength > 0 && paragraphLength <= maxAutomaticTargetCharacters) {
    return makeRevisionTarget({
      text,
      mode: "paragraph",
      start: paragraph.start,
      end: paragraph.end,
      operation,
      label: "current paragraph"
    })
  }

  return makeRevisionTarget({
    text,
    mode: "document",
    start: 0,
    end: text.length,
    operation,
    label: "whole document",
    confirmationReason: "The current paragraph could not be resolved safely."
  })
}
```

Then add `planRevisionApply()` with the exact-target, unique-retarget, insertion-anchor, advisory,
large-target confirmation, and conflict behavior from the spec. Applying code must refuse
text-changing proposals whose target still requires confirmation.

- [ ] **Step 5: Run tests to verify utilities pass**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-utils.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts
git commit -m "feat: add writing revision utilities"
```

### Task 2: Add Workflow Presets

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-presets.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts`

- [ ] **Step 1: Write failing preset tests**

Cover:

- all six spec presets exist: Draft freely, Polish prose, Developmental edit, Preserve voice,
  Make concise, Expand sensory detail
- each preset has a stable `id`, user-facing `label`, and visible `instruction`
- preset ids match `WritingRevisionPresetId`
- preset instructions are not empty and do not silently override the user's custom instruction

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts
```

Expected: FAIL because the preset module does not exist.

- [ ] **Step 2: Implement visible preset definitions**

Create `writing-revision-presets.ts`:

```ts
import type { WritingRevisionPresetId } from "./writing-revision-types"

export type WritingRevisionPreset = {
  id: WritingRevisionPresetId
  label: string
  instruction: string
}

export const WRITING_REVISION_PRESETS: WritingRevisionPreset[] = [
  {
    id: "draft_freely",
    label: "Draft freely",
    instruction: "Prioritize momentum, vivid continuation, and useful new material."
  },
  {
    id: "polish_prose",
    label: "Polish prose",
    instruction: "Improve clarity, rhythm, word choice, and sentence flow without changing intent."
  },
  {
    id: "developmental_edit",
    label: "Developmental edit",
    instruction: "Focus on structure, stakes, pacing, continuity, and what the passage needs next."
  },
  {
    id: "preserve_voice",
    label: "Preserve voice",
    instruction: "Keep the author's diction, cadence, point of view, and stylistic fingerprints."
  },
  {
    id: "make_concise",
    label: "Make concise",
    instruction: "Reduce redundancy and sharpen phrasing while preserving meaning and voice."
  },
  {
    id: "expand_sensory_detail",
    label: "Expand sensory detail",
    instruction: "Add concrete sensory detail grounded in the existing scene and tone."
  }
]

export const getWritingRevisionPreset = (
  id?: WritingRevisionPresetId | null
): WritingRevisionPreset | null =>
  WRITING_REVISION_PRESETS.find((preset) => preset.id === id) ?? null
```

- [ ] **Step 3: Run preset tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts
```

Expected: PASS.

- [ ] **Step 4: Commit Task 2**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-presets.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts
git commit -m "feat: add writing revision presets"
```

### Task 3: Add Structured Proposal Prompt And Validation Utilities

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-prompt-utils.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts`

- [ ] **Step 1: Write failing prompt utility tests**

Cover:

- rewrite action creates a text-changing prompt that asks for JSON only
- outline defaults to advisory operation
- prompt includes selected preset instruction when present
- prompt includes existing Writing Playground context: selected template/theme, composed context
  prompt or context messages, memory block, author note, world info entries, active provider/model,
  and generation settings summary
- valid JSON with `replacement` becomes a pending insert or replace proposal
- valid advisory JSON without replacement becomes advisory
- malformed JSON becomes `raw_suggestion`
- streamed or partial-looking JSON is not applyable until complete parsing succeeds

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts
```

Expected: FAIL because the new module does not exist.

- [ ] **Step 2: Implement prompt and parser utilities**

Create exports shaped like:

```ts
export const buildRevisionUserPrompt = (input: {
  action: WritingRevisionAction
  operation: WritingRevisionOperation
  instruction: string
  documentText: string
  target: WritingRevisionTarget
  presetInstruction?: string | null
  writingContext: {
    selectedTemplateName?: string | null
    selectedThemeName?: string | null
    chatMode: boolean
    contextComposedPrompt?: string | null
    contextMessages?: Array<{ role: string; content: string }> | null
    memoryBlock?: unknown
    authorNote?: unknown
    worldInfoEntries?: unknown[]
    provider?: string | null
    model?: string | null
    generationSettingsSummary: Record<string, unknown>
  }
}): string => {
  return [
    "You are helping revise a creative writing document.",
    "Return only valid JSON. Do not include markdown fences.",
    `Action: ${input.action}`,
    `Operation: ${input.operation}`,
    `Instruction: ${input.instruction}`,
    input.presetInstruction ? `Workflow preset: ${input.presetInstruction}` : null,
    `Template: ${input.writingContext.selectedTemplateName || "(none)"}`,
    `Theme: ${input.writingContext.selectedThemeName || "(none)"}`,
    `Chat mode: ${input.writingContext.chatMode ? "enabled" : "disabled"}`,
    `Provider: ${input.writingContext.provider || "(default)"}`,
    `Model: ${input.writingContext.model || "(unset)"}`,
    "Generation settings:",
    JSON.stringify(input.writingContext.generationSettingsSummary),
    "Composed writing context:",
    input.writingContext.contextComposedPrompt ||
      JSON.stringify(input.writingContext.contextMessages ?? []),
    "Memory / author note / world info:",
    JSON.stringify({
      memoryBlock: input.writingContext.memoryBlock,
      authorNote: input.writingContext.authorNote,
      worldInfoEntries: input.writingContext.worldInfoEntries ?? []
    }),
    "Target text:",
    input.target.beforeText || "(insertion point)",
    `Target summary: ${input.target.label}`,
    "Full document:",
    input.documentText,
    "JSON shape:",
    input.operation === "advisory"
      ? '{"title":"...","rawText":"...","rationale":"...","notes":["..."]}'
      : '{"title":"...","replacement":"...","rationale":"...","notes":["..."]}'
  ].filter(Boolean).join("\n\n")
}
```

Implement `parseRevisionModelResponse()` so:

- text-changing proposals require a string `replacement`
- advisory proposals accept `rawText`, `rationale`, `title`, or `notes`
- malformed JSON returns a `raw_suggestion` draft with `rawText`
- parser never marks partial output applyable
- parser copies `presetId`, `presetInstruction`, target metadata, and model response metadata into
  the created proposal draft so regeneration can reuse the same context

- [ ] **Step 3: Run tests to verify parser behavior**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts
```

Expected: PASS.

- [ ] **Step 4: Commit Task 3**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-prompt-utils.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts
git commit -m "feat: validate writing revision proposals"
```

### Task 4: Extend Session Payload Persistence For Revision State

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingSessionManagement.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts`

- [ ] **Step 1: Write failing persistence tests**

Add tests proving:

- `mergeRevisionsIntoPayload()` preserves prompt/settings and writes `revisions`
- `mergeRevisionsIntoPayload()` writes `{ schemaVersion: 1, items: [...] }`, not a bare array
- `getRevisionsFromPayload()` ignores malformed revisions
- selected workflow preset id persists in session payload and rejects unknown preset ids
- revision signatures change when proposal status changes
- clearing all revisions removes or normalizes the payload field

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts
```

Expected: FAIL for missing revision helpers.

- [ ] **Step 2: Add payload helpers**

In `hooks/utils.ts`:

```ts
import type {
  WritingRevisionPresetId,
  WritingRevisionPayload,
  WritingRevisionProposal
} from "../writing-revision-types"

export const WRITING_REVISION_PAYLOAD_SCHEMA_VERSION = 1

export type WritingSessionPayload = Record<string, unknown> & {
  prompt?: string
  prompt_rich?: JSONContent
  revisions?: WritingRevisionPayload
  revision_preset_id?: WritingRevisionPresetId | null
  // existing fields...
}

export const getRevisionsFromPayload = (
  payload?: Record<string, unknown> | null
): WritingRevisionProposal[] => {
  if (!isRecord(payload) || !isRecord(payload.revisions)) return []
  if (payload.revisions.schemaVersion !== WRITING_REVISION_PAYLOAD_SCHEMA_VERSION) return []
  if (!Array.isArray(payload.revisions.items)) return []
  return payload.revisions.items.filter(isWritingRevisionProposal)
}

export const mergeRevisionsIntoPayload = (
  payload: Record<string, unknown> | null | undefined,
  revisions: WritingRevisionProposal[]
): WritingSessionPayload => {
  const base = isRecord(payload) ? payload : {}
  const next: WritingSessionPayload = { ...base }
  if (revisions.length > 0) {
    next.revisions = {
      schemaVersion: WRITING_REVISION_PAYLOAD_SCHEMA_VERSION,
      items: revisions
    }
  } else {
    delete next.revisions
  }
  return next
}

export const getRevisionPayloadSignature = (
  payload?: Record<string, unknown> | null
): string => JSON.stringify(getRevisionsFromPayload(payload))

export const getRevisionPresetIdFromPayload = (
  payload?: Record<string, unknown> | null
): WritingRevisionPresetId | null => {
  const value = isRecord(payload) ? payload.revision_preset_id : null
  return isWritingRevisionPresetId(value) ? value : null
}
```

Keep validation intentionally structural and conservative.

- [ ] **Step 3: Add a safe payload-patch save seam**

In `useWritingSessionManagement.ts`, add a returned helper such as:

```ts
const applySessionPayloadPatch = React.useCallback(
  (patcher: (payload: WritingSessionPayload) => WritingSessionPayload) => {
    if (!activeSessionDetail) return
    const basePayload =
      pendingSaveMapRef.current[activeSessionDetail.id] ??
      mergePayloadIntoSession(
        activeSessionDetail.payload,
        editorText,
        settings,
        selectedTemplateName,
        selectedThemeName,
        chatMode,
        { promptRich: editorPromptRichRef.current }
      )
    const nextPayload = patcher(basePayload)
    pendingSaveMapRef.current[activeSessionDetail.id] = nextPayload
    setIsDirty(true)
    scheduleSave(activeSessionDetail.id, nextPayload)
  },
  [
    activeSessionDetail,
    chatMode,
    editorText,
    scheduleSave,
    selectedTemplateName,
    selectedThemeName,
    settings
  ]
)
```

This helper is required so proposal-only saves merge with pending editor changes instead of stale
`activeSessionDetail.payload`.

- [ ] **Step 4: Run session payload tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingSessionManagement.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts
git commit -m "feat: persist writing revision state"
```

### Task 5: Add Revision State Hook

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx`

- [ ] **Step 1: Write failing hook tests**

Cover:

- loads revisions from active session payload
- rejects a proposal without changing text
- applies a plain replacement through the provided text callback
- marks conflict without mutating text
- advisory proposals do not call apply text callback
- proposal state is passed to `applySessionPayloadPatch`
- regenerate marks the source proposal rejected, appends a new pending proposal with
  `regeneratedFromId`, and preserves the source target/instruction/preset
- rich-editor unsupported apply responses mark the proposal conflict/manual-apply instead of
  pretending the patch was applied

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx
```

Expected: FAIL because the hook does not exist.

- [ ] **Step 2: Implement hook state transitions**

The hook should accept dependencies rather than importing global editor state:

```ts
export function useWritingRevisions(deps: {
  activeSessionId: string | null
  activeSessionPayload?: Record<string, unknown> | null
  editorText: string
  applyEditorText: (nextText: string) => { applied: true } | { applied: false; reason: string }
  applySessionPayloadPatch: (patcher: (payload: WritingSessionPayload) => WritingSessionPayload) => void
}) {
  // load, update status, persist revisions, apply plans
}
```

Keep the hook deterministic and side-effect-light. Do not call the LLM from this hook.

Regeneration is modeled as a callback-driven state transition, not an internal LLM call:

```ts
const regenerateRevision = async (
  proposalId: string,
  createReplacement: (source: WritingRevisionProposal) => Promise<WritingRevisionProposal>
) => {
  const source = findProposal(proposalId)
  if (!source) return
  const replacement = await createReplacement(source)
  setRevisions((current) => [
    ...current.map((proposal) =>
      proposal.id === source.id ? { ...proposal, status: "rejected" } : proposal
    ),
    {
      ...replacement,
      regeneratedFromId: source.id,
      target: source.target,
      instruction: source.instruction,
      presetId: source.presetId,
      presetInstruction: source.presetInstruction,
      status: "pending"
    }
  ])
}
```

- [ ] **Step 3: Run hook tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Commit Task 5**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx
git commit -m "feat: manage writing revision queue state"
```

### Task 6: Add Action Bar, Diff, And Queue Components

**Files:**
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionDiff.tsx`
- Create: `apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionQueue.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx`
- Test: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx`

- [ ] **Step 1: Write failing component tests**

Cover:

- action bar disables actions when generation is unavailable
- preset selector renders the six workflow presets and shows the selected preset instruction
- action bar shows the resolved target summary before sending Custom requests
- action bar requires explicit confirmation before sending whole-document text-changing requests
- Tone exposes a direction/custom instruction input
- Outline defaults to advisory copy
- queue renders pending proposal diff and Apply/Reject/Copy/Regenerate
- queue hides Apply for advisory proposals
- conflict state shows copy/manual-apply guidance
- raw suggestion state shows raw text and Copy only

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx
```

Expected: FAIL because components do not exist.

- [ ] **Step 2: Implement components with stable dimensions**

Use Ant Design and existing styling conventions. Keep controls dense:

- icon+label buttons for actions where labels matter
- `Select` or segmented menu for workflow presets; the selected preset instruction must remain
  visible or inspectable near the controls
- target summary text from `WritingRevisionTarget.label`
- confirmation control for `target.requiresConfirmation` before text-changing generation is sent
- compact custom instruction input
- `Tag` for status
- simple text diff based on line or word chunks
- no nested cards inside cards unless the existing layout forces it

Do not add a new icon library.

- [ ] **Step 3: Run component tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Commit Task 6**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionDiff.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionQueue.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx
git commit -m "feat: add writing revision queue UI"
```

### Task 7: Wire Plain-Text Proposal Generation Into WritingPlayground

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx`
- Optional modify: `apps/packages/ui/src/assets/locale/en/option.json`
- Optional modify: `apps/packages/ui/src/public/_locales/en/option.json`

- [ ] **Step 1: Write failing integration tests**

Extend `WritingPlayground.phase1-baseline.test.tsx` with tests that:

- render the action bar when an active session exists
- click Rewrite with selected text and receive a mocked structured response
- show a pending proposal instead of immediately mutating editor text
- click Apply and mutate editor text
- generate malformed output and show raw suggestion without Apply
- generate Outline and show advisory proposal without Apply
- select a workflow preset and verify its visible instruction is included in the proposed-edit prompt
- verify memory block, author note, world info, selected template/theme, provider/model, and
  generation settings summary are passed into `buildRevisionUserPrompt`
- regenerate a proposal and verify the old proposal is rejected while the regenerated proposal is
  appended with `regeneratedFromId`
- simulate rich-editor apply unsupported and verify the queue shows copy/manual-apply guidance
  without mutating rich content
- confirm a whole-document text-changing target, generate a proposal, and verify Apply is enabled
  for the confirmed target

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
```

Expected: FAIL before integration.

- [ ] **Step 2: Wire hook and components**

In `index.tsx`:

- import `WritingActionBar`, `WritingRevisionQueue`, revision prompt utilities, and `useWritingRevisions`
- take `applySessionPayloadPatch` from `useWritingSessionManagement`
- derive current selection from `getCurrentEditorAdapter()`
- build target with `resolveRevisionTarget()`
- show `target.label` before Custom generation and block whole-document text-changing generation
  until `target.requiresConfirmation` is acknowledged
- acknowledge destructive target confirmation with `confirmRevisionTarget()` before creating the
  proposal metadata; do not persist confirmed proposals that still carry `requiresConfirmation: true`
- read the selected workflow preset from `WRITING_REVISION_PRESETS` and pass
  `presetId`/`presetInstruction` into prompt generation and proposal metadata
- initialize the selected workflow preset from `getRevisionPresetIdFromPayload()` and persist changes
  through `applySessionPayloadPatch()` without overwriting pending prompt/settings edits
- reuse the existing Writing Playground context pipeline:
  - `contextComposedPrompt` and `contextMessages` from `useWritingContextComposition`
  - `memoryBlock`, `authorNote`, and `worldInfoEntries`
  - `selectedTemplateName`, `selectedThemeName`, `chatMode`, `apiProviderOverride`, and
    `selectedModel`
  - the same generation settings fields used by the existing Generate path, excluding secrets
- call `TldwChatService.sendMessage()` for proposed-edit generation, not stream-to-editor generation
- force proposed-edit calls to complete-response parsing; do not display partial JSON as applyable
- parse the complete response before adding proposal state
- wire `regenerateRevision()` by rebuilding the request from the source proposal's action,
  instruction, target, preset, and current Writing Playground context
- render the queue below the editor toolbar/content area for the first slice

Do not replace the existing Generate button. This feature adds a reviewable edit path beside the
existing direct generation path.

- [ ] **Step 3: Preserve rich editor honesty**

When `editorMode === "tiptap"`:

- allow proposal generation from canonical plain text
- apply only when the active adapter can safely replace plain text
- otherwise mark or render copy/manual apply guidance
- do not pretend rich structure is preserved when it is not

- [ ] **Step 4: Run integration tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit Task 7**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/index.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx \
  apps/packages/ui/src/assets/locale/en/option.json \
  apps/packages/ui/src/public/_locales/en/option.json
git commit -m "feat: wire writing revision proposals"
```

If locale files were not changed, omit them from `git add`.

### Task 8: Add Status Bar Counts And Route/Extension Parity Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts`
- Modify: `apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts`

- [ ] **Step 1: Write failing status and parity assertions**

Add assertions for:

- word count in status bar
- selected word count when a selection exists and can be observed in component tests
- pending revisions count
- route parity still uses shared `WritingPlayground`
- extension options route can find the revision action bar or queue test id

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx \
  apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
```

Expected: FAIL before status/parity wiring.

- [ ] **Step 2: Implement status counts and stable test ids**

Add stable test ids:

- `writing-revision-action-bar`
- `writing-revision-queue`
- `writing-revision-pending-count`
- `writing-status-word-count`
- `writing-status-selected-word-count`

Keep status text compact and do not displace save/generation status.

- [ ] **Step 3: Run focused Vitest coverage**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx \
  apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
```

Expected: PASS.

- [ ] **Step 4: Run extension smoke if environment is already prepared**

Run:

```bash
cd apps/extension
bunx playwright test tests/e2e/writing-playground-mode-parity.spec.ts
```

Expected: PASS. If the extension build is blocked by known local build noise, record the exact
failure and keep the Vitest route parity evidence.

- [ ] **Step 5: Commit Task 8**

```bash
git add \
  apps/packages/ui/src/components/Option/WritingPlayground/index.tsx \
  apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts \
  apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts
git commit -m "test: cover writing revision parity"
```

### Task 9: Final Verification And Handoff

**Files:**
- Modify if needed: `backlog/tasks/task-458 - Plan-document-first-Writing-Playground-revision-implementation.md`

- [ ] **Step 1: Run focused frontend unit suite**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run route parity guard**

Run:

```bash
bunx vitest run apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run extension smoke when available**

Run:

```bash
cd apps/extension
bunx playwright test tests/e2e/writing-playground-mode-parity.spec.ts
```

Expected: PASS, or document environment/build blocker with exact output.

- [ ] **Step 4: Run formatting checks**

Run:

```bash
git diff --check
rg -nP "[^\x00-\x7F]" \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-types.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-presets.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-utils.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/writing-revision-prompt-utils.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingRevisions.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/hooks/useWritingSessionManagement.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionDiff.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/WritingRevisionQueue.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-presets.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-revision-prompt-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-session-payload-utils.test.ts \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/useWritingRevisions.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingRevisionQueue.test.tsx \
  apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx \
  apps/tldw-frontend/extension/__tests__/writing-playground-route-parity.guard.test.ts \
  apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts
```

Expected: no trailing whitespace and no accidental non-ASCII in implementation-owned touched files.
If locale files are changed, append those exact locale file paths to the `rg` command before running
it.

- [ ] **Step 5: Record Bandit skip**

Bandit is not required if the implementation remains frontend TypeScript only. If any backend
Python is touched, run:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_writing_revisions.json
```

Expected: no new findings.

- [ ] **Step 6: Update Backlog task and commit**

Update implementation notes on the implementation task created for the code work, including:

- files touched
- verification commands and outcomes
- extension smoke status
- Bandit skip or results

Commit:

```bash
git add <touched files> backlog/tasks/<implementation-task-file>.md
git commit -m "feat: add writing revision workflow"
```

## Follow-Up Tasks Not In This Plan

- Design persistent comments/annotations.
- Add backend revision-history APIs if session-payload persistence proves insufficient.
- Improve TipTap structural patching beyond safe plain-text replacement.
- Add provider-native structured output support when the selected chat provider exposes it through the existing backend contract.
