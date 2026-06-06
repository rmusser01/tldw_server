# Chat Media Context Full-Content Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `/chat` Knowledge panel media-library context actions so they insert, ask, pin, and copy full media content instead of title-only fallback snippets.

**Architecture:** Keep `/api/v1/media/search` lightweight and fix the frontend conversion boundary. Media-library results receive a narrow origin marker during normalization, `toPinnedResult` preserves that marker on `RagPinnedResult`, and the shared full-media resolver fetches full content only for pinned results with both `mediaId` and `contextOrigin === "media-library"`.

**Tech Stack:** React, TypeScript, Zustand, Vitest, Testing Library, tldw frontend API client.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-06-chat-media-context-full-content-design.md`
- Backlog: `TASK-527`

## File Structure

- Modify `apps/packages/ui/src/utils/rag-format.ts`
  - Add a narrow optional `contextOrigin?: "media-library"` field to `RagPinnedResult`.
  - Keep formatting output unchanged; do not print or serialize the origin marker into prompts.
- Modify `apps/packages/ui/src/components/Knowledge/hooks/useKnowledgeSearch.ts`
  - Add `metadata.origin = "media-library"` in `normalizeMediaSearchResults`.
  - Preserve that marker as `contextOrigin` in `toPinnedResult`.
  - Gate `withFullMediaTextIfAvailable` on `contextOrigin === "media-library"`.
  - Resolve full media text before Knowledge Search Insert, Ask, Pin/Save, and copy actions.
- Modify `apps/packages/ui/src/components/Knowledge/hooks/useFileSearch.ts`
  - Resolve full media text before file-search copy actions. Attach already routes through the shared resolver and should continue to work after the origin guard.
- Modify `apps/packages/ui/src/components/Knowledge/KnowledgePanel.tsx`
  - Resolve full media text before component-level Ask confirmation paths and Preview modal Ask.
  - Preserve existing Preview modal Insert behavior.
- Modify `apps/packages/ui/src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts`
  - Add helper and hook regression tests for origin propagation, guarded fetching, ask, pin, copy, and RAG chunk non-expansion.
- Modify `apps/packages/ui/src/components/Knowledge/hooks/__tests__/useFileSearch.test.ts`
  - Add/adjust media-library origin expectations and file-search copy coverage.
- Modify `apps/packages/ui/src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx`
  - Add Preview modal Ask coverage that verifies the formatted prompt uses the resolver result.

## Stage 1: Marker and Resolver Contract

**Goal:** Make media-library origin explicit and preserve it through conversion without broad metadata propagation.

**Success Criteria:** Media search rows normalize with `metadata.origin = "media-library"`, `toPinnedResult` emits `contextOrigin: "media-library"`, and `withFullMediaTextIfAvailable` fetches only for pinned media-library results.

**Tests:** `useKnowledgeSearch.test.ts`

**Status:** Not Started

### Task 1: Add failing helper tests for origin propagation and guarded fetching

**Files:**
- Modify: `apps/packages/ui/src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts`

- [ ] **Step 1: Add test setup mocks**

Add mocked `tldwClient` before importing `../useKnowledgeSearch`:

```ts
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    getMediaDetails: vi.fn(),
    searchMedia: vi.fn()
  }
}))
```

Import `beforeEach`, `vi`, `waitFor`, and the resolver:

```ts
import { beforeEach, describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  extractContentFromMediaDetail,
  extractMediaId,
  normalizeMediaSearchResults,
  toPinnedResult,
  withFullMediaTextIfAvailable,
  type RagResult
} from "../useKnowledgeSearch"
```

- [ ] **Step 2: Add failing tests**

Add these tests:

```ts
it("marks normalized media-library rows and carries the marker into pinned results", () => {
  const [result] = normalizeMediaSearchResults({
    items: [{ id: 42, title: "Quarterly Report", type: "pdf", url: "/api/v1/media/42" }]
  })

  expect(result.metadata?.media_id).toBe(42)
  expect(result.metadata?.origin).toBe("media-library")
  expect(toPinnedResult(result).contextOrigin).toBe("media-library")
})

it("fetches full text only for pinned media-library results", async () => {
  vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
    content: { text: "Full media body content" }
  })

  const resolved = await withFullMediaTextIfAvailable({
    id: "media-42",
    title: "Quarterly Report",
    snippet: "Library item: Quarterly Report",
    mediaId: 42,
    contextOrigin: "media-library"
  })

  expect(resolved.snippet).toBe("Full media body content")
  expect(tldwClient.getMediaDetails).toHaveBeenCalledWith(
    42,
    expect.objectContaining({
      include_content: true,
      include_versions: false,
      include_version_content: false
    })
  )
})

it("does not fetch full media details for chunk-scoped pinned results", async () => {
  const resolved = await withFullMediaTextIfAvailable({
    id: "chunk-42",
    title: "Chunk",
    snippet: "Retrieved chunk only",
    mediaId: 42
  })

  expect(resolved.snippet).toBe("Retrieved chunk only")
  expect(tldwClient.getMediaDetails).not.toHaveBeenCalled()
})

it("keeps fallback snippet when full media detail fetch fails or returns empty content", async () => {
  vi.mocked(tldwClient.getMediaDetails).mockRejectedValueOnce(new Error("offline"))

  const failedFetch = await withFullMediaTextIfAvailable({
    id: "media-42",
    title: "Quarterly Report",
    snippet: "Library item: Quarterly Report",
    mediaId: 42,
    contextOrigin: "media-library"
  })

  vi.mocked(tldwClient.getMediaDetails).mockResolvedValueOnce({
    content: { text: "" }
  })

  const emptyContent = await withFullMediaTextIfAvailable({
    id: "media-42",
    title: "Quarterly Report",
    snippet: "Library item: Quarterly Report",
    mediaId: 42,
    contextOrigin: "media-library"
  })

  expect(failedFetch.snippet).toBe("Library item: Quarterly Report")
  expect(emptyContent.snippet).toBe("Library item: Quarterly Report")
})
```

- [ ] **Step 3: Run tests and verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts
```

Expected: FAIL because `metadata.origin`, `contextOrigin`, and the guarded resolver behavior are not implemented yet.

### Task 2: Implement origin marker and guarded resolver

**Files:**
- Modify: `apps/packages/ui/src/utils/rag-format.ts`
- Modify: `apps/packages/ui/src/components/Knowledge/hooks/useKnowledgeSearch.ts`

- [ ] **Step 1: Extend the pinned result type narrowly**

In `rag-format.ts`:

```ts
export type RagPinnedResult = {
  id: string
  title?: string
  source?: string
  url?: string
  snippet: string
  type?: string
  mediaId?: number
  contextOrigin?: "media-library"
}
```

- [ ] **Step 2: Mark normalized media-library results**

In `normalizeMediaSearchResults`, add the origin marker to the metadata object:

```ts
const metadata: Record<string, unknown> = {
  title,
  type,
  source: title,
  url: itemUrl,
  origin: "media-library",
  created_at: getFirstString(rawItem, ["created_at", "date", "added_at"]) || undefined
}
```

- [ ] **Step 3: Preserve only the narrow origin marker in pinned results**

In `toPinnedResult`:

```ts
const contextOrigin =
  getMetadataValue(item.metadata, "origin") === "media-library"
    ? "media-library"
    : undefined

return {
  id: buildPinnedResultId(item, text),
  title: title || undefined,
  source: getResultSource(item) || undefined,
  url: url || undefined,
  snippet,
  type: getResultType(item) || undefined,
  mediaId: extractMediaId(item) ?? undefined,
  contextOrigin
}
```

- [ ] **Step 4: Gate full-media fetching on the pinned origin marker**

In `withFullMediaTextIfAvailable`:

```ts
if (!pinned.mediaId || pinned.contextOrigin !== "media-library") return pinned
```

Keep the existing fetch and fallback behavior unchanged after this guard.

- [ ] **Step 5: Run helper tests and verify they pass**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit marker and resolver contract**

```bash
git add apps/packages/ui/src/utils/rag-format.ts apps/packages/ui/src/components/Knowledge/hooks/useKnowledgeSearch.ts apps/packages/ui/src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts
git commit -m "fix: gate chat media context expansion by origin"
```

## Stage 2: Media Context Action Paths

**Goal:** Ensure every `/chat` media-library action path that creates context resolves full content before formatting.

**Success Criteria:** Knowledge Search Insert, Ask, Pin/Save, copy, Preview Insert, Preview Ask, File Search Attach, and File Search copy all format full media content when detail content is available. RAG/QA chunks remain chunk-scoped.

**Tests:** `useKnowledgeSearch.test.ts`, `useFileSearch.test.ts`, `KnowledgePanelQAPreview.test.tsx`

**Status:** Not Started

### Task 3: Add failing action-path tests

**Files:**
- Modify: `apps/packages/ui/src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts`
- Modify: `apps/packages/ui/src/components/Knowledge/hooks/__tests__/useFileSearch.test.ts`
- Modify: `apps/packages/ui/src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx`

- [ ] **Step 1: Add hook-level tests for Ask, Pin/Save, and copy**

In `useKnowledgeSearch.test.ts`, add mocks for `react-i18next`, `@plasmohq/storage/hook`, and reset the Zustand store:

```ts
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback?: string) => fallback ?? _key })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false, vi.fn()] as const
}))
```

Import and reset `useStoreMessageOption` between hook tests so pinned state does not leak:

```ts
import { useStoreMessageOption } from "@/store/option"

beforeEach(() => {
  vi.clearAllMocks()
  useStoreMessageOption.setState({
    ragPinnedResults: [],
    ragMediaIds: null
  })
})
```

Use `renderHook`, `act`, and `waitFor` to test:

```ts
const mediaResult = normalizeMediaSearchResults({
  items: [{ id: 42, title: "Quarterly Report", type: "pdf", url: "/api/v1/media/42" }]
})[0]

vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
  content: { text: "Full media body content" }
})
```

Required cases:

- `handleAsk(mediaResult)` eventually calls `onAsk` with `"Full media body content"` and `{ ignorePinnedResults: true }`.
- `handlePin(mediaResult)` eventually stores a pinned result whose `snippet` is `"Full media body content"` and keeps `ragMediaIds` as `[42]`.
- `copyResult(mediaResult, "markdown")` writes full media content to the clipboard.
- A chunk-style `RagResult` with `metadata.media_id = 42` and no `metadata.origin` does not call `getMediaDetails`.

- [ ] **Step 2: Add file-search copy coverage**

In `useFileSearch.test.ts`, add or adjust tests so:

- Search results include `metadata.origin = "media-library"`.
- `copyResult(result, "markdown")` writes full media content when `getMediaDetails` returns `content.text`.
- Attach continues to insert full media content with the new origin guard.

- [ ] **Step 3: Add Preview modal Ask coverage**

In `KnowledgePanelQAPreview.test.tsx`, keep existing QA preview expectations and add a test that verifies the Preview modal Ask path uses the resolved pinned result:

```ts
mockWithFullMediaTextIfAvailable.mockResolvedValueOnce({
  id: "pin-doc-1",
  title: "QA Doc Title",
  snippet: "Resolved preview context"
})
```

Click Preview, then Ask, and assert `onAsk` receives `"Resolved preview context"`.

- [ ] **Step 4: Run tests and verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts src/components/Knowledge/hooks/__tests__/useFileSearch.test.ts src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx
```

Expected: FAIL because Ask, Pin/Save, copy, and Preview Ask do not all resolve through `withFullMediaTextIfAvailable` yet.

### Task 4: Resolve full media text in all action paths

**Files:**
- Modify: `apps/packages/ui/src/components/Knowledge/hooks/useKnowledgeSearch.ts`
- Modify: `apps/packages/ui/src/components/Knowledge/hooks/useFileSearch.ts`
- Modify: `apps/packages/ui/src/components/Knowledge/KnowledgePanel.tsx`

- [ ] **Step 1: Resolve Knowledge Search copy**

In `copyResult`:

```ts
const pinned = toPinnedResult(item)
const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
await navigator.clipboard.writeText(formatRagResult(resolvedPinned, format))
```

- [ ] **Step 2: Resolve Knowledge Search Ask**

Wrap the Ask path in an async IIFE while keeping the public handler type as `(item) => void`:

```ts
void (async () => {
  const pinned = toPinnedResult(item)
  const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
  onAsk(formatRagResult(resolvedPinned, "markdown"), { ignorePinnedResults: true })
})()
```

- [ ] **Step 3: Resolve Pin/Save before storing**

In `handlePin`, preserve the duplicate check using the original pinned ID, then resolve before storing:

```ts
const pinned = toPinnedResult(item)
if (pinnedResults.some((result) => result.id === pinned.id)) return
void (async () => {
  const resolvedPinned = await withFullMediaTextIfAvailable(pinned)
  const nextPinned = [...pinnedResults, resolvedPinned]
  setRagPinnedResults(nextPinned)
  const mediaIds = collectPinnedMediaIds(nextPinned)
  setRagMediaIds(mediaIds.length > 0 ? mediaIds : null)
})()
```

- [ ] **Step 4: Resolve File Search copy**

In `useFileSearch.copyResult`, use the same `resolvedPinned` pattern as Knowledge Search copy.

- [ ] **Step 5: Resolve component-level Ask and Preview Ask**

In `KnowledgePanel.tsx`, add a local helper:

```ts
const askWithResolvedContext = React.useCallback(
  (pinned: RagPinnedResult) => {
    void (async () => {
      const resolved = await withFullMediaTextIfAvailable(pinned)
      onAsk(formatRagResult(resolved, "markdown"), {
        ignorePinnedResults: true
      })
    })()
  },
  [onAsk]
)
```

Use it in:

- The immediate `handleAsk` path.
- The `Modal.confirm` `onOk` path.
- The Preview modal Ask button before `setPreviewItem(null)`.

- [ ] **Step 6: Run action-path tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts src/components/Knowledge/hooks/__tests__/useFileSearch.test.ts src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit action-path fixes**

```bash
git add apps/packages/ui/src/components/Knowledge/hooks/useKnowledgeSearch.ts apps/packages/ui/src/components/Knowledge/hooks/useFileSearch.ts apps/packages/ui/src/components/Knowledge/KnowledgePanel.tsx apps/packages/ui/src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts apps/packages/ui/src/components/Knowledge/hooks/__tests__/useFileSearch.test.ts apps/packages/ui/src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx
git commit -m "fix: resolve chat media context actions"
```

## Stage 3: Submit-Path and Regression Verification

**Goal:** Verify the submit path receives full pinned media content without changing file-retrieval behavior.

**Success Criteria:** `ragPinnedResults` still flow through `formatPinnedResults` on submit/raw preview, now with full snippets from Pin/Save; file retrieval continues to skip pinned inline expansion.

**Tests:** Targeted Vitest tests plus static checks.

**Status:** Not Started

### Task 5: Verify submit-path assumptions and update tests only if needed

**Files:**
- Inspect: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`
- Inspect: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundRawPreview.ts`
- Modify only if existing tests need expectation updates.

- [ ] **Step 1: Confirm no submit-path code change is needed**

Check that both submit and raw preview still call:

```ts
formatPinnedResults(ragPinnedResults, "markdown")
```

when `fileRetrievalEnabled` is false.

- [ ] **Step 2: Run targeted tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Knowledge/hooks/__tests__/useKnowledgeSearch.test.ts src/components/Knowledge/hooks/__tests__/useFileSearch.test.ts src/components/Knowledge/__tests__/KnowledgePanelQAPreview.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.pinned-fallback.test.tsx src/components/Option/Playground/__tests__/usePlaygroundRawPreview.mcp-tools.test.tsx
```

Expected: PASS. If the Playground tests are unrelatedly flaky, document the exact failure and rerun only once before deciding whether it is unrelated baseline noise.

- [ ] **Step 3: Run diff and formatting checks**

Run from repo root:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Record Bandit applicability**

No Python files should be touched in this implementation. Record in `TASK-527` that Bandit is not applicable for this TypeScript-only frontend change. If any Python file is touched, run:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_chat_media_context.json
```

- [ ] **Step 5: Commit final verification metadata if needed**

If only Backlog task metadata changes after verification:

```bash
git add "backlog/tasks/task-527 - Design-chat-media-context-full-content-insertion-fix.md"
git commit -m "chore: record chat media context verification"
```

## Stage 4: Finalization

**Goal:** Leave the branch with documented verification, focused commits, and no staged unrelated changes.

**Success Criteria:** Backlog task has final notes, verification commands are recorded, unrelated dirty files are not staged, and the user receives the exact changed-file summary.

**Tests:** Git status and targeted test evidence.

**Status:** Not Started

### Task 6: Update task tracking and final status

**Files:**
- Modify: `backlog/tasks/task-527 - Design-chat-media-context-full-content-insertion-fix.md`

- [ ] **Step 1: Update Backlog task with implementation notes**

Record:

- The origin-marker guard.
- The action paths covered.
- The targeted Vitest commands and results.
- Bandit applicability.
- Any skipped or failed checks with reasons.

- [ ] **Step 2: Inspect final status**

Run from repo root:

```bash
git status --short
```

Expected: only intended files for this task are staged or modified by this work; unrelated pre-existing dirty files remain unstaged.

- [ ] **Step 3: Commit Backlog update with implementation if not already committed**

```bash
git add "backlog/tasks/task-527 - Design-chat-media-context-full-content-insertion-fix.md"
git commit -m "chore: finalize chat media context task"
```

- [ ] **Step 4: Final response**

Report:

- Files changed.
- Tests run and results.
- Bandit status or skip reason.
- Any residual risk, especially if a verification command could not run.
