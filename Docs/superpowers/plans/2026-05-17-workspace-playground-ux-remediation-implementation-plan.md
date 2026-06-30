# Workspace Playground UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the shared `/workspace-playground` WebUI/extension UX defects around hidden pane recovery, offscreen chat input, source intake, My Media library loading, and model selection.

**Architecture:** Keep the existing shared `WorkspacePlayground` stack and harden only the route/component contracts that are failing. Add body-level restore rails for collapsed panes, make the route wrappers provide a complete bounded-height chain, normalize media-library response shapes in the Add Sources modal, and move the workspace chat model picker onto the existing chat-model service.

**Tech Stack:** React 18, TypeScript, Ant Design, lucide-react, Zustand stores, Vitest + Testing Library, Playwright for route verification.

---

## Spec And Tracking

- Spec: `Docs/superpowers/specs/2026-05-17-workspace-playground-ux-remediation-design.md`
- Planning task: `TASK-408`
- Completed design task: `TASK-407`

## File Structure

- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
  - Owns desktop layout, pane collapsed state, focus helpers, and restore rail rendering.
- Modify: `apps/packages/ui/src/routes/option-workspace-playground.tsx`
  - Shared options route wrapper; must pass a bounded height contract to `WorkspacePlayground`.
- Modify: `apps/tldw-frontend/extension/routes/option-workspace-playground.tsx`
  - Extension options route wrapper; must pass the same bounded height contract through `PageShell`.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx`
  - Left-pane header source intake label.
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/media-library-normalization.ts`
  - Pure helper for media-library item/total/id normalization.
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts`
  - Unit coverage for response shapes and item id normalization.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx`
  - My Media tab uses the helper, distinguishes empty/all-added/error states, and keeps pagination.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx`
  - Model picker uses shared chat-model service and non-interfering field markup.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx`
  - Desktop layout guardrails for restore rails and height contract.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx`
  - Add Sources label guard.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx`
  - My Media response-shape, all-added, and error-state coverage. Also update stale tab-order expectation if it conflicts with the current fixed tab order.
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx`
  - Model picker service and selectable Auto/model behavior.
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts`
  - Page-object helpers for restore rails, composer visibility, Add Sources, My Media, and model picker.
- Modify: `apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts`
  - Browser-level regression checks for the reported route behavior.
- Modify: `apps/test-utils/workspace-playground/page.ts`
  - Shared WebUI/extension parity page object for composer viewport and collapsed-pane restore behavior.
- Modify: `apps/test-utils/workspace-playground/contract.ts`
  - Shared parity contract assertions that run on both WebUI and extension options routes.
- Modify: `backlog/tasks/task-408 - Plan-Workspace-Playground-UX-remediation-implementation.md`
  - Track final verification notes and Bandit skip if this remains frontend-only.

## Task 1: Layout Guard Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts`
- Modify: `apps/test-utils/workspace-playground/page.ts`
- Modify: `apps/test-utils/workspace-playground/contract.ts`

- [ ] **Step 1: Write failing unit tests for restore rails**

Add tests near the existing desktop layout guardrails:

```tsx
it("keeps a visible restore control when the sources pane is collapsed", () => {
  testState.leftPaneCollapsed = true

  render(<WorkspacePlayground />)

  expect(screen.queryByTestId("workspace-sources-pane")).not.toBeInTheDocument()
  const restore = screen.getByRole("button", { name: /show sources/i })
  expect(restore).toHaveAttribute("data-testid", "workspace-restore-sources")

  fireEvent.click(restore)
  expect(testState.setLeftPaneCollapsed).toHaveBeenCalledWith(false)
})

it("keeps a visible restore control when the studio pane is collapsed", () => {
  testState.rightPaneCollapsed = true

  render(<WorkspacePlayground />)

  expect(screen.queryByTestId("workspace-studio-pane")).not.toBeInTheDocument()
  const restore = screen.getByRole("button", { name: /show studio/i })
  expect(restore).toHaveAttribute("data-testid", "workspace-restore-studio")

  fireEvent.click(restore)
  expect(testState.setRightPaneCollapsed).toHaveBeenCalledWith(false)
})
```

Ensure `fireEvent` is imported from `@testing-library/react` if it is not already.

- [ ] **Step 2: Write failing unit test for route wrapper height contract**

Add a static or render guard in `WorkspacePlayground.desktop-layout.test.tsx` if route-wrapper imports are awkward in the current test harness:

```tsx
import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const testDirname = dirname(fileURLToPath(import.meta.url))

it("documents the bounded-height route wrapper contract", () => {
  const sharedRoute = readFileSync(
    resolve(testDirname, "../../../../routes/option-workspace-playground.tsx"),
    "utf8"
  )
  const extensionRoute = readFileSync(
    resolve(
      process.cwd(),
      "../../tldw-frontend/extension/routes/option-workspace-playground.tsx"
    ),
    "utf8"
  )

  expect(sharedRoute).toContain("min-h-0")
  expect(sharedRoute).toContain("flex-1")
  expect(extensionRoute).toContain("min-h-0")
  expect(extensionRoute).toContain("flex-1")
})
```

If the relative path is brittle in Vitest, move this to a dedicated static guard test beside the route tests and keep the same assertion intent.

- [ ] **Step 3: Add E2E page-object helpers**

Add helpers to `WorkspacePlaygroundPage`:

```ts
readonly chatInput: Locator
readonly restoreSourcesButton: Locator
readonly restoreStudioButton: Locator
readonly modelSelect: Locator

// in constructor
this.chatInput = page.locator("#workspace-main-content textarea").first()
this.restoreSourcesButton = page.getByTestId("workspace-restore-sources")
this.restoreStudioButton = page.getByTestId("workspace-restore-studio")
this.modelSelect = page.getByRole("combobox", { name: /select model/i }).first()

async expectComposerVisibleWithoutPageScroll(): Promise<void> {
  await expect(this.chatInput).toBeVisible({ timeout: 10_000 })
  const box = await this.chatInput.boundingBox()
  const viewport = this.page.viewportSize()
  expect(box).not.toBeNull()
  expect(viewport).not.toBeNull()
  expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(
    (viewport?.height ?? 0) + 1
  )
}
```

- [ ] **Step 4: Add WebUI failing E2E assertions**

Extend the existing `collapses and restores sources + studio panes` test:

```ts
await workspacePage.hideSourcesPane()
await expect(workspacePage.restoreSourcesButton).toBeVisible()
await workspacePage.restoreSourcesButton.click()
await expect(workspacePage.sourcesPanel).toBeVisible()

await workspacePage.hideStudioPane()
await expect(workspacePage.restoreStudioButton).toBeVisible()
await workspacePage.restoreStudioButton.click()
await expect(workspacePage.studioPanel).toBeVisible()
```

Add a focused composer visibility test:

```ts
test("keeps the chat composer visible on first load", async ({ authedPage, diagnostics }) => {
  const workspacePage = new WorkspacePlaygroundPage(authedPage)
  await workspacePage.goto()
  await workspacePage.waitForReady()

  await workspacePage.expectComposerVisibleWithoutPageScroll()

  await assertNoCriticalErrors(diagnostics)
})
```

- [ ] **Step 5: Add shared extension parity assertions**

Update `apps/test-utils/workspace-playground/page.ts` so the parity contract can verify the same regressions in the extension options route:

```ts
readonly chatInput: Locator
readonly restoreSourcesButton: Locator
readonly restoreStudioButton: Locator

// in constructor
this.chatInput = this.chatPanel.locator("textarea").first()
this.restoreSourcesButton = page.getByTestId("workspace-restore-sources")
this.restoreStudioButton = page.getByTestId("workspace-restore-studio")

async expectComposerVisibleWithoutPageScroll(): Promise<void> {
  await expect(this.chatInput).toBeVisible({ timeout: 10_000 })
  const box = await this.chatInput.boundingBox()
  const viewport = this.page.viewportSize()
  expect(box).not.toBeNull()
  expect(viewport).not.toBeNull()
  expect((box?.y ?? 0) + (box?.height ?? 0)).toBeLessThanOrEqual(
    (viewport?.height ?? 0) + 1
  )
}

async hideSourcesPane(): Promise<void> {
  await this.sourcesPanel.getByRole("button", { name: /hide sources/i }).click()
  await expect(this.sourcesPanel).toBeHidden({ timeout: 10_000 })
}

async restoreSourcesPane(): Promise<void> {
  await expect(this.restoreSourcesButton).toBeVisible({ timeout: 10_000 })
  await this.restoreSourcesButton.click()
  await expect(this.sourcesPanel).toBeVisible({ timeout: 10_000 })
}

async hideStudioPane(): Promise<void> {
  await this.studioPanel.getByRole("button", { name: /hide studio/i }).click()
  await expect(this.studioPanel).toBeHidden({ timeout: 10_000 })
}

async restoreStudioPane(): Promise<void> {
  await expect(this.restoreStudioButton).toBeVisible({ timeout: 10_000 })
  await this.restoreStudioButton.click()
  await expect(this.studioPanel).toBeVisible({ timeout: 10_000 })
}
```

Update `apps/test-utils/workspace-playground/contract.ts` immediately after `assertBaselinePanesVisible()`:

```ts
await workspacePage.expectComposerVisibleWithoutPageScroll()

await workspacePage.hideSourcesPane()
await workspacePage.restoreSourcesPane()

await workspacePage.hideStudioPane()
await workspacePage.restoreStudioPane()
```

This is required because `apps/extension/tests/e2e/workspace-playground.parity.spec.ts` already calls this shared contract against `${optionsUrl}#/workspace-playground`; adding the assertions here makes the extension options route fail until it gets the same visible composer and restore-rail behavior.

- [ ] **Step 6: Run tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because `workspace-restore-sources`, `workspace-restore-studio`, and/or bounded route wrapper classes do not exist yet.

Optional E2E failure check after unit failure is confirmed:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/workspace-playground.spec.ts --grep "collapses and restores|chat composer" --reporter=line
```

Expected: FAIL on the new restore rail/composer assertions.

Optional extension parity failure check if the extension build is available:

```bash
cd apps/extension
bunx playwright test tests/e2e/workspace-playground.parity.spec.ts --reporter=line
```

Expected: FAIL on the shared contract's new restore rail/composer assertions.

- [ ] **Step 7: Commit failing tests**

```bash
git add apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts apps/test-utils/workspace-playground/page.ts apps/test-utils/workspace-playground/contract.ts
git commit -m "test: cover workspace playground layout recovery"
```

## Task 2: Layout Shell Implementation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- Modify: `apps/packages/ui/src/routes/option-workspace-playground.tsx`
- Modify: `apps/tldw-frontend/extension/routes/option-workspace-playground.tsx`

- [ ] **Step 1: Add restore rail component**

In `WorkspacePlayground/index.tsx`, import the needed icons:

```ts
import {
  AlertTriangle,
  FileText,
  MessageSquare,
  Sparkles,
  Search,
  Command,
  Loader2,
  PanelLeftOpen,
  PanelRightOpen
} from "lucide-react"
```

Add a small local component above `WorkspacePlaygroundBody`:

```tsx
type WorkspacePaneRestoreRailProps = {
  side: "left" | "right"
  label: string
  testId: string
  onClick: () => void
}

const WorkspacePaneRestoreRail: React.FC<WorkspacePaneRestoreRailProps> = ({
  side,
  label,
  testId,
  onClick
}) => {
  const Icon = side === "left" ? PanelLeftOpen : PanelRightOpen
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      aria-label={label}
      className={`hidden shrink-0 items-center gap-1.5 rounded-lg border border-border/80 bg-surface/95 px-2 py-2 text-xs font-medium text-primary shadow-card transition hover:border-primary/40 hover:bg-primary/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus lg:flex ${
        side === "left" ? "mr-1" : "ml-1"
      }`}
    >
      <Icon className="h-4 w-4" aria-hidden="true" />
      <span className="sr-only xl:not-sr-only">{label}</span>
    </button>
  )
}
```

Keep this component local unless reuse emerges elsewhere.

- [ ] **Step 2: Render restore rails in desktop shell**

Inside the desktop layout flex row, render the left rail where the left pane normally appears:

```tsx
{!leftPaneOpen && (
  <WorkspacePaneRestoreRail
    side="left"
    testId="workspace-restore-sources"
    label={t("playground:workspace.showSources", "Show sources")}
    onClick={() => {
      setLeftPaneCollapsed(false)
      window.setTimeout(() => focusWorkspacePane("sources"), 0)
    }}
  />
)}
```

Render the right rail symmetrically after `<main>` and before the right pane/drawer:

```tsx
{!rightPaneOpen && (
  <WorkspacePaneRestoreRail
    side="right"
    testId="workspace-restore-studio"
    label={t("playground:workspace.showStudio", "Show studio")}
    onClick={() => {
      setRightPaneCollapsed(false)
      window.setTimeout(() => focusWorkspacePane("studio"), 0)
    }}
  />
)}
```

If the focus helper already calls `setLeftPaneCollapsed(false)`, keep the click handler simple:

```tsx
onClick={() => focusWorkspacePane("sources")}
```

Use whichever version avoids duplicate state writes in tests.

- [ ] **Step 3: Harden route wrapper height**

In `apps/packages/ui/src/routes/option-workspace-playground.tsx`, change the wrapper div:

```tsx
<OptionLayout>
  <div className="flex min-h-0 w-full flex-1">
    <WorkspacePlayground />
  </div>
</OptionLayout>
```

In `apps/tldw-frontend/extension/routes/option-workspace-playground.tsx`, update `PageShell` classes:

```tsx
<PageShell
  className="flex min-h-0 flex-1"
  maxWidthClassName="max-w-full"
>
  <WorkspacePlayground />
</PageShell>
```

If `PageShell` horizontal padding causes workspace clipping in browser verification, the implementation may add `px-0` support through `className` or a narrow `PageShell` prop. Do not change global `PageShell` defaults.

- [ ] **Step 4: Verify unit tests pass**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 5: Commit layout implementation**

```bash
git add apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx apps/packages/ui/src/routes/option-workspace-playground.tsx apps/tldw-frontend/extension/routes/option-workspace-playground.tsx
git commit -m "fix: keep workspace playground panes recoverable"
```

## Task 3: Source Intake Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx`
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx`

- [ ] **Step 1: Add Add Sources label test**

In `SourcesPane.stage2.test.tsx`, add:

```tsx
it("labels the primary source intake action as Add Sources", () => {
  render(<SourcesPane />)

  fireEvent.click(screen.getByRole("button", { name: "Add Sources" }))

  expect(mockOpenAddSourceModal).toHaveBeenCalledTimes(1)
})
```

Expected initial failure: the current button is named `Add`.

- [ ] **Step 2: Add media-library normalizer tests**

Create `media-library-normalization.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import {
  getMediaLibraryItemKey,
  normalizeMediaLibraryResponse
} from "../SourcesPane/media-library-normalization"

describe("media-library-normalization", () => {
  it.each([
    ["media", { media: [{ id: 1, title: "Media" }], total_count: 9 }],
    ["results", { results: [{ id: 2, title: "Result" }], total: 8 }],
    ["items", { items: [{ id: 3, title: "Item" }], count: 7 }],
    ["data", { data: [{ id: 4, title: "Data" }], pagination: { total: 6 } }]
  ])("normalizes %s response shape", (_label, response) => {
    const normalized = normalizeMediaLibraryResponse(response)

    expect(normalized.items).toHaveLength(1)
    expect(normalized.totalCount).toBeGreaterThan(normalized.items.length)
  })

  it("normalizes nested data.items response shape", () => {
    expect(
      normalizeMediaLibraryResponse({
        data: { items: [{ media_id: 5, title: "Nested" }], total: 5 }
      })
    ).toMatchObject({
      items: [{ media_id: 5, title: "Nested" }],
      totalCount: 5
    })
  })

  it("returns stable string keys for numeric and string ids", () => {
    expect(getMediaLibraryItemKey({ media_id: 0 })).toBe("0")
    expect(getMediaLibraryItemKey({ id: "abc" })).toBe("abc")
    expect(getMediaLibraryItemKey({ title: "missing" })).toBeNull()
  })
})
```

- [ ] **Step 3: Add My Media UI tests**

In `AddSourceModal.stage2.intake.test.tsx`, add:

```tsx
it("renders My Media items from items response shape", async () => {
  workspaceStoreState.addSourceModalTab = "existing"
  mockListMedia.mockResolvedValueOnce({
    items: [{ id: 701, title: "Library Item", type: "pdf" }],
    total: 1
  })

  render(<AddSourceModal />)

  expect(await screen.findByText("Library Item")).toBeInTheDocument()
  expect(screen.getByText("Showing 1 of 1")).toBeInTheDocument()
})

it("distinguishes all-added media from an empty media library", async () => {
  workspaceStoreState.addSourceModalTab = "existing"
  workspaceStoreState.sources = [{ mediaId: 701 }]
  mockListMedia.mockResolvedValueOnce({
    items: [{ id: 701, title: "Already Added", type: "pdf" }],
    total: 1
  })

  render(<AddSourceModal />)

  expect(
    await screen.findByText(/already in this workspace/i)
  ).toBeInTheDocument()
})

it("shows a load error when My Media cannot load", async () => {
  workspaceStoreState.addSourceModalTab = "existing"
  mockListMedia.mockRejectedValueOnce(new Error("offline"))

  render(<AddSourceModal />)

  expect(await screen.findByText(/unable to load media/i)).toBeInTheDocument()
})

it("toggles a My Media checkbox once when clicked directly", async () => {
  workspaceStoreState.addSourceModalTab = "existing"
  mockListMedia.mockResolvedValueOnce({
    items: [{ id: 701, title: "Library Item", type: "pdf" }],
    total: 1
  })

  render(<AddSourceModal />)

  const checkbox = await screen.findByRole("checkbox", {
    name: /select library item/i
  })
  await userEvent.click(checkbox)
  expect(checkbox).toBeChecked()

  await userEvent.click(checkbox)
  expect(checkbox).not.toBeChecked()
})
```

Import `userEvent` from `@testing-library/user-event` if this test file does not already use it.

- [ ] **Step 4: Update stale fixed-tab-order test if needed**

If `AddSourceModal.stage2.intake.test.tsx` still contains a test expecting usage-frequency tab reordering, update it to match the current fixed-order implementation:

```tsx
it("keeps Add Sources tabs in a stable order despite prior usage frequency", async () => {
  window.localStorage.setItem(
    ADD_SOURCE_TAB_USAGE_STORAGE_KEY,
    JSON.stringify({ upload: 0, existing: 2, url: 5, paste: 1, search: 9 })
  )

  render(<AddSourceModal />)

  const tabLabels = screen
    .getAllByRole("tab")
    .map((tab) => tab.textContent?.replace(/\s+/g, " ").trim())

  expect(tabLabels).toEqual([
    "Upload",
    "My Media",
    "URL",
    "Paste",
    "Search Server"
  ])
})
```

This is not disabling coverage; it aligns the test with the current code comment that tab usage is tracked but no longer drives rendered order.

- [ ] **Step 5: Run tests and verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the helper does not exist, the Sources button says `Add`, and My Media does not normalize `items`/`data` or distinguish all-added/error states yet.

- [ ] **Step 6: Commit failing source-intake tests**

```bash
git add apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx
git commit -m "test: cover workspace source intake regressions"
```

## Task 4: Source Intake Implementation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx`
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/media-library-normalization.ts`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts`

- [ ] **Step 1: Implement media-library helper**

Create `media-library-normalization.ts`:

```ts
type MediaLibraryRecord = Record<string, unknown>

const isRecord = (value: unknown): value is MediaLibraryRecord =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const toFiniteCount = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return value
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null
  }
  return null
}

const firstArray = (...values: unknown[]): unknown[] => {
  for (const value of values) {
    if (Array.isArray(value)) return value
  }
  return []
}

export const getMediaLibraryItemKey = (item: unknown): string | null => {
  if (!isRecord(item)) return null
  const id = item.media_id ?? item.id
  if (typeof id === "number" && Number.isFinite(id)) return String(id)
  if (typeof id === "string" && id.trim()) return id.trim()
  return null
}

export const normalizeMediaLibraryResponse = (
  response: unknown,
  fallbackTotal?: number
): { items: unknown[]; totalCount: number } => {
  if (Array.isArray(response)) {
    return { items: response, totalCount: response.length }
  }
  if (!isRecord(response)) {
    return { items: [], totalCount: fallbackTotal ?? 0 }
  }

  const nestedData = isRecord(response.data) ? response.data : null
  const items = firstArray(
    response.media,
    response.results,
    response.items,
    response.data,
    nestedData?.items,
    nestedData?.media,
    nestedData?.results
  )

  const pagination = isRecord(response.pagination) ? response.pagination : null
  const nestedPagination =
    nestedData && isRecord(nestedData.pagination) ? nestedData.pagination : null
  const total =
    toFiniteCount(response.total_count) ??
    toFiniteCount(response.total) ??
    toFiniteCount(response.count) ??
    toFiniteCount(response.results_count) ??
    toFiniteCount(pagination?.total) ??
    toFiniteCount(nestedData?.total_count) ??
    toFiniteCount(nestedData?.total) ??
    toFiniteCount(nestedData?.count) ??
    toFiniteCount(nestedPagination?.total) ??
    fallbackTotal ??
    items.length

  return { items, totalCount: Math.max(total, items.length) }
}
```

- [ ] **Step 2: Rename left pane button**

In `SourcesPane/index.tsx`, change:

```tsx
{t("playground:sources.add", "Add")}
```

to:

```tsx
{t("playground:sources.addSources", "Add Sources")}
```

In `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts`, update the existing Add Source modal trigger from the old exact `Add` label to the new label:

```ts
this.sourcesPanel.getByRole("button", { name: /^add sources$/i })
```

- [ ] **Step 3: Update ExistingTab state and id handling**

In `AddSourceModal.tsx`, import the helper:

```ts
import {
  getMediaLibraryItemKey,
  normalizeMediaLibraryResponse
} from "./media-library-normalization"
```

Change selected media state to string keys:

```ts
const [selectedMediaKeys, setSelectedMediaKeys] = React.useState<Set<string>>(
  new Set()
)
```

Update existing media ids:

```ts
const existingMediaIds = React.useMemo(
  () => new Set(sources.map((s) => String(s.mediaId))),
  [sources]
)
```

Use the normalizer after list/search responses:

```ts
const normalized = normalizeMediaLibraryResponse(
  response,
  append ? media.length : 0
)
const items = normalized.items
const normalizedTotal = normalized.totalCount
```

When deduping:

```ts
const dedupedItems = Array.from(
  new Map(
    nextItems
      .map((item: any) => [getMediaLibraryItemKey(item), item] as const)
      .filter(([key]) => key !== null)
  ).values()
)
```

- [ ] **Step 4: Distinguish empty/all-added/error states**

Add local tab error state or reuse modal `setError`. Prefer local tab state so Upload/URL/Paste errors are not conflated with a My Media load failure:

```ts
const [loadError, setLoadError] = React.useState<string | null>(null)
```

In `fetchMediaFromServer`, clear/set it:

```ts
setLoadError(null)
// catch
setLoadError(t("playground:sources.mediaLoadError", "Unable to load media library."))
setError(mapSourceIngestionError(err))
```

Derive all-added state:

```ts
const responseHadItems = media.length > 0
const allVisibleAlreadyAdded = responseHadItems && availableMedia.length === 0
```

Render in order:

```tsx
{loadError ? (
  <Alert type="error" message={loadError} />
) : isLoading ? (
  <div className="flex justify-center py-8"><Spin /></div>
) : allVisibleAlreadyAdded ? (
  <Empty
    image={Empty.PRESENTED_IMAGE_SIMPLE}
    description={t(
      "playground:sources.allMediaAlreadyAdded",
      "All visible media are already in this workspace."
    )}
  />
) : availableMedia.length === 0 ? (
  <Empty
    image={Empty.PRESENTED_IMAGE_SIMPLE}
    description={t("playground:sources.noMediaFound", "No media found")}
  />
) : (
  // existing list
)}
```

- [ ] **Step 5: Update selected add behavior**

Use stable item keys when selecting and adding:

```ts
const selectedItems = media.filter((item) => {
  const key = getMediaLibraryItemKey(item)
  return key && selectedMediaKeys.has(key) && !existingMediaIds.has(key)
})
```

For rendering rows:

```tsx
const id = getMediaLibraryItemKey(item)
if (!id) return null
```

Use `selectedMediaKeys` for checkbox state and `setSelectedMediaKeys` for toggles. Give each checkbox an accessible label and stop direct checkbox clicks from bubbling back to the row click handler:

```tsx
<Checkbox
  aria-label={t("playground:sources.selectMediaItem", "Select {{title}}", {
    title
  })}
  checked={selectedMediaKeys.has(id)}
  onClick={(event) => event.stopPropagation()}
  onChange={() => toggleMedia(id)}
/>
```

Keep the row click handler so clicking the rest of the row remains a fast selection target:

```tsx
<List.Item onClick={() => toggleMedia(id)}>
```

- [ ] **Step 6: Run tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 7: Commit source-intake implementation**

```bash
git add apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/media-library-normalization.ts apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts
git commit -m "fix: load workspace media library sources"
```

## Task 5: Model Picker Tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx`

- [ ] **Step 1: Replace raw model mock with shared service mock**

In `ChatPane.stage2.test.tsx`, update hoisted mocks:

```ts
const hoistedMocks = vi.hoisted(() => ({
  setSelectedModel: vi.fn(),
  fetchChatModels: vi.fn()
}))
```

Add a mock for `@/services/tldw-server`:

```ts
vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: hoistedMocks.fetchChatModels
}))
```

Leave `tldwClient` mocked for lorebook diagnostics but remove `getModels` from that mock.

- [ ] **Step 2: Update existing model picker test**

Change the model response shape to match `fetchChatModels`:

```tsx
hoistedMocks.fetchChatModels.mockResolvedValue([
  {
    model: "tldw:gpt-4o",
    name: "tldw:gpt-4o",
    nickname: "GPT-4o",
    provider: "openai"
  },
  {
    model: "tldw:claude-3-5-sonnet",
    name: "tldw:claude-3-5-sonnet",
    nickname: "Claude 3.5 Sonnet",
    provider: "anthropic"
  }
])
```

Assert:

```tsx
const modelSelect = await screen.findByRole("combobox", {
  name: "Select model"
})

fireEvent.change(modelSelect, { target: { value: "tldw:gpt-4o" } })

expect(hoistedMocks.setSelectedModel).toHaveBeenCalledWith("tldw:gpt-4o")
expect(screen.getByRole("option", { name: /openai/i })).toBeInTheDocument()
```

- [ ] **Step 3: Add Auto fallback test**

Add:

```tsx
it("keeps Auto selectable when chat model loading fails", async () => {
  hoistedMocks.fetchChatModels.mockRejectedValueOnce(new Error("offline"))

  renderChatPane()

  const modelSelect = await screen.findByRole("combobox", {
    name: "Select model"
  })
  expect(modelSelect).not.toBeDisabled()
  expect(screen.getByRole("option", { name: "Auto" })).toBeInTheDocument()

  fireEvent.change(modelSelect, { target: { value: "" } })
  expect(hoistedMocks.setSelectedModel).toHaveBeenCalledWith(null)
})
```

- [ ] **Step 4: Run test and verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL until ChatPane uses `fetchChatModels` and keeps Auto available on failure.

- [ ] **Step 5: Commit failing model picker tests**

```bash
git add apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx
git commit -m "test: cover workspace chat model picker"
```

## Task 6: Model Picker Implementation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx`

- [ ] **Step 1: Import shared model service**

In `ChatPane/index.tsx`, import:

```ts
import { fetchChatModels } from "@/services/tldw-server"
```

Remove the model picker dependency on `tldwClient.getModels()`.

- [ ] **Step 2: Update model option mapping**

Keep `ChatModelOption`, but map from shared UI model shape:

```ts
type ChatModelOption = {
  id: string
  label: string
  provider?: string | null
}
```

Replace the existing model-loading effect with:

```ts
React.useEffect(() => {
  if (modelsFetchedRef.current) return
  modelsFetchedRef.current = true

  let isMounted = true
  setLoadingModels(true)
  void fetchChatModels({ returnEmpty: true })
    .then((models) => {
      if (!isMounted) return
      const uniqueById = new Map<string, ChatModelOption>()
      for (const model of models || []) {
        const modelId = extractStringCandidate((model as { model?: unknown }).model)
        if (!modelId) continue
        const provider = extractStringCandidate(
          (model as { provider?: unknown }).provider
        )
        const nickname = extractStringCandidate(
          (model as { nickname?: unknown }).nickname
        )
        const label = nickname && nickname !== modelId ? `${nickname} (${modelId})` : modelId
        uniqueById.set(modelId, {
          id: modelId,
          label: provider ? `${provider} · ${label}` : label,
          provider
        })
      }
      setAvailableModels(
        Array.from(uniqueById.values()).sort((a, b) =>
          a.label.localeCompare(b.label, undefined, { sensitivity: "base" })
        )
      )
    })
    .catch(() => {
      if (!isMounted) return
      setAvailableModels([])
    })
    .finally(() => {
      if (isMounted) setLoadingModels(false)
    })

  return () => {
    isMounted = false
  }
}, [])
```

- [ ] **Step 3: Keep picker visible and Auto available**

Change the render condition from:

```tsx
{provenanceEnabled && (loadingModels || availableModels.length > 0) && (
```

to:

```tsx
{provenanceEnabled && (
```

Replace the wrapping `<label>` with a non-label field container to avoid native select activation issues:

```tsx
<div
  className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface/80 px-2 py-1 text-[11px] text-text-muted"
  title={...}
>
  <Cpu className="h-3 w-3" />
  <span id="workspace-chat-model-picker-label">
    {t("playground:chat.modelPickerLabel", "Model")}
  </span>
  <select
    aria-labelledby="workspace-chat-model-picker-label"
    aria-label={t("playground:chat.modelPickerAria", "Select model")}
    value={selectedModel ?? ""}
    onChange={(event) => {
      const value = event.target.value.trim()
      setSelectedModel(value.length > 0 ? value : null)
    }}
    className="min-w-[9rem] max-w-[220px] truncate bg-transparent text-[11px] text-text focus:outline-none"
    disabled={false}
  >
    <option value="">{t("playground:chat.modelPickerAuto", "Auto")}</option>
    {availableModels.map((model) => (
      <option key={model.id} value={model.id}>
        {model.label}
      </option>
    ))}
  </select>
  {loadingModels && <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" />}
</div>
```

If the double accessible naming causes test ambiguity, keep only `aria-label`.

- [ ] **Step 4: Run model tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 5: Commit model implementation**

```bash
git add apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx
git commit -m "fix: repair workspace chat model picker"
```

## Task 7: Integration Verification And Closeout

**Files:**
- Modify: `backlog/tasks/task-408 - Plan-Workspace-Playground-UX-remediation-implementation.md`

- [ ] **Step 1: Run focused UI unit tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.desktop-layout.test.tsx src/components/Option/WorkspacePlayground/__tests__/SourcesPane.stage2.test.tsx src/components/Option/WorkspacePlayground/__tests__/media-library-normalization.test.ts src/components/Option/WorkspacePlayground/__tests__/AddSourceModal.stage2.intake.test.tsx src/components/Option/WorkspacePlayground/__tests__/ChatPane.stage2.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 2: Run route/client guard checks**

Run:

```bash
cd apps/packages/ui
bun run verify:openapi
```

Expected: PASS.

Run:

```bash
cd apps/extension
bun run verify:openapi
```

Expected: PASS.

- [ ] **Step 3: Run workspace E2E checks**

Run WebUI workspace coverage:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/workspace-playground.spec.ts --reporter=line
```

Expected: PASS or documented unrelated baseline failure. If it fails because the test server setup is unavailable, record the exact blocker in `TASK-408`.

Run extension parity if the environment supports it:

```bash
cd apps/extension
bunx playwright test tests/e2e/workspace-playground.parity.spec.ts --reporter=line
```

Expected: PASS or documented environment blocker.

- [ ] **Step 4: Browser-observe the route**

Start the WebUI if needed:

```bash
cd apps/tldw-frontend
bun run dev -- -p 3000
```

Use Browser or Playwright to visit `http://127.0.0.1:3000/workspace-playground` and verify:

- chat input is visible without page scrolling
- Sources collapse leaves `Show sources` visible
- Studio collapse leaves `Show studio` visible
- Add Sources opens the modal
- My Media displays seeded or real media-library items when available
- model picker can be opened and changed

- [ ] **Step 5: Document Bandit status**

No Python files should be touched in this slice. Update `TASK-408` with:

```text
Bandit not run: implementation touched only frontend TypeScript/TSX tests and route wrappers.
```

If implementation unexpectedly touches Python, run:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_workspace_playground_ux.json
```

- [ ] **Step 6: Final status and commit**

Update `TASK-408` final summary with tests run and any skips. Then commit the final task update:

```bash
git add "backlog/tasks/task-408 - Plan-Workspace-Playground-UX-remediation-implementation.md"
git commit -m "docs: close workspace playground ux plan"
```

If implementation was done in separate task commits, this final commit should only contain Backlog closeout metadata.
