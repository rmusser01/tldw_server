# Flashcards Narrow UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved narrow flashcards UX remediation as two reviewable PRs: PR 1 fixes the direct extension route, import/export copy, re-rate recovery, selected-deck create prefill, and completion-action regression; PR 2 makes all-deck Study dashboard-first and preserves user-facing deck names in session history.

**Architecture:** Keep PR 1 frontend-only unless current branch state proves otherwise. Keep PR 2 frontend-first for dashboard behavior, but start the session-history work with a payload inspection gate; add backend/API schema only if the current review-session payload cannot preserve renamed/deleted deck names. Preserve existing component boundaries and avoid broad flashcards redesign.

**Tech Stack:** Next.js WebUI, WXT extension routes, shared `apps/packages/ui` React components, Ant Design, React Query flashcard hooks, Vitest/Testing Library, Playwright e2e where the existing harness can seed cards, FastAPI/Pydantic/ChaChaNotes DB only for PR 2 session-name snapshots if needed.

---

## Source Spec

- Design: `Docs/superpowers/specs/2026-05-29-flashcards-narrow-ux-remediation-design.md`
- Backlog planning task: `TASK-484`
- Prior design task: `TASK-483`

## Branching And Backlog

Do not implement in the dirty main checkout. Each PR gets its own implementation task and clean worktree.

Suggested branch/worktree names:

- PR 1 branch: `codex/flashcards-entry-review-recovery`
- PR 1 worktree: `.worktrees/flashcards-entry-review-recovery`
- PR 2 branch: `codex/flashcards-dashboard-session-history`
- PR 2 worktree: `.worktrees/flashcards-dashboard-session-history`

Before PR 1 edits:

- Create a Backlog task such as `Implement flashcards entry and review recovery`.
- Link this plan and the design spec in implementation notes.

Before PR 2 edits:

- Create a Backlog task such as `Implement flashcards dashboard and session history`.
- Start from latest `dev` after PR 1 is merged, or rebase on PR 1 if explicitly continuing before merge.

## File Responsibility Map

### PR 1 Files

- Modify: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
  - Add the real extension sidepanel `/flashcards` route and lazy import.
- Create: `apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx`
  - Provide the app-extension handoff component imported by the registry.
- Modify: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`
  - Assert the app extension sidepanel registry itself includes `/flashcards`.
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
  - Add one-shot Study to Manage/Create deck handoff state.
  - Keep `Import / Export` as visible tab label.
- Modify: `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
  - Rename stale Transfer test wording.
  - Add selected Study deck to Create handoff coverage.
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
  - Accept create-drawer initial deck props separately from Manage filter initial props.
  - Open Create drawer from `openCreateSignal` with the one-shot deck context.
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
  - Accept `initialDeckId`.
  - Apply it after open-time `form.resetFields()`.
- Test: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx`
  - Add or extend coverage for `initialDeckId`.
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  - Move visible `Re-rate last card` control outside the active-card answer branch.
  - Keep copy accurate: do not say `Undo rating`.
  - Verify or gate `Practice again` absence.
- Test: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`
  - Add visible re-rate and completion regression coverage if enough existing mocks can support it.
- Optional Test: create `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx`
  - Use this if adding the re-rate flow to `ReviewTab.create-cta.test.tsx` makes that file too large.
- Modify: `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
  - Keep tab label assertions on `Import / Export`; rename test title.
- Modify as needed: `apps/packages/ui/src/public/_locales/en/option.json`
  - Update remaining user-facing `Transfer summary` copy to `Import/export summary` or more specific labels.

### PR 2 Files

- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  - Add explicit `allDeckReviewStarted` state and dashboard-first all-deck behavior.
- Test: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`
  - Add dashboard-first, Review all due, scope-reset, and selected-deck fast-path coverage.
- Modify: `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
  - Resolve deck labels by snapshot, then current deck lookup, then graceful fallback.
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  - Pass the current deck list into recent session history for legacy sessions without snapshots.
- Test: `apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx`
  - Add preserved deck name, current-deck lookup, deleted/unavailable fallback, and no raw `Deck 1` assertions.
- Modify: `apps/packages/ui/src/services/flashcards.ts`
  - Add optional deck-name snapshot type only if payload inspection shows it is needed.
- Backend if needed:
  - Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
  - Test: `tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py`
  - Test: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`

## PR 1: Entry And Review Recovery

### Task 1: Create PR 1 Worktree And Tracking

**Files:**
- Modify: Backlog task created for PR 1
- No code files

- [ ] **Step 1: Fetch latest dev**

Run:

```bash
git fetch origin dev
```

Expected: command completes without errors.

- [ ] **Step 2: Create clean worktree**

Run:

```bash
git worktree add .worktrees/flashcards-entry-review-recovery -b codex/flashcards-entry-review-recovery origin/dev
```

Expected: worktree created on a new branch.

- [ ] **Step 3: Confirm clean worktree**

Run:

```bash
git -C .worktrees/flashcards-entry-review-recovery status --short --ignored=no
```

Expected: no output.

- [ ] **Step 4: Create PR 1 Backlog task**

Use Backlog MCP or CLI. Include:

- Title: `Implement flashcards entry and review recovery`
- References: this plan and the design spec
- Modified files from the PR 1 file map

Expected: task id recorded in the PR 1 work notes.

### Task 2: Register Actual Extension `/flashcards` Sidepanel Route

**Files:**
- Create: `apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx`
- Modify: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`

- [ ] **Step 1: Write failing app-extension registry test**

Add a test to `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`:

```ts
it("registers the app extension /flashcards sidepanel route", () => {
  expect(extensionSidepanelRegistrySource).toMatch(/path:\s*"\/flashcards"/)
  expect(extensionSidepanelRegistrySource).toContain("SidepanelFlashcards")
  expect(extensionSidepanelRegistrySource).toContain("sidepanel-flashcards")
})

const extensionSidepanelFlashcardsCandidates = [
  "apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx",
  "../../tldw-frontend/extension/routes/sidepanel-flashcards.tsx",
  "../tldw-frontend/extension/routes/sidepanel-flashcards.tsx"
]
const extensionSidepanelFlashcardsPath =
  extensionSidepanelFlashcardsCandidates.find((candidate) => existsSync(candidate))

it("has an app extension sidepanel flashcards handoff component", () => {
  expect(extensionSidepanelFlashcardsPath).toBeDefined()
})
```

- [ ] **Step 2: Run route tests and verify failure**

Run from repo root or package root:

```bash
cd apps/packages/ui && bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
```

Expected: FAIL because `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx` does not register `/flashcards` and the local handoff component file is missing.

- [ ] **Step 3: Create app-extension flashcards handoff component**

Create `apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx`:

```tsx
import React from "react"
import { useTranslation } from "react-i18next"
import { Layers } from "lucide-react"
import { Button, Typography } from "antd"
import { browser } from "wxt/browser"

const { Text, Title } = Typography

export default function SidepanelFlashcards() {
  const { t } = useTranslation()
  const hasAutoOpenedRef = React.useRef(false)

  const openFlashcards = React.useCallback(() => {
    const url = browser.runtime.getURL("/options.html#/flashcards")
    if (browser.tabs?.create) {
      browser.tabs.create({ url }).catch(() => {
        window.open(url, "_blank")
      })
      return
    }
    window.open(url, "_blank")
  }, [])

  React.useEffect(() => {
    if (hasAutoOpenedRef.current) return
    hasAutoOpenedRef.current = true
    openFlashcards()
  }, [openFlashcards])

  return (
    <div className="flex flex-col items-center justify-center gap-4 p-6 text-center">
      <Layers className="size-10 text-text-muted" aria-hidden="true" />
      <Title level={5}>{t("sidepanel:flashcards.title", "Flashcards")}</Title>
      <Text type="secondary">
        {t(
          "sidepanel:flashcards.openedInTab",
          "Flashcards opens in a full tab for the best study experience."
        )}
      </Text>
      <Button type="primary" onClick={openFlashcards}>
        {t("sidepanel:flashcards.openAgain", "Open Flashcards")}
      </Button>
    </div>
  )
}
```

- [ ] **Step 4: Implement route registration**

In `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`, add the lazy import:

```ts
const SidepanelFlashcards = lazy(() => import("./sidepanel-flashcards"))
```

Add the route before settings:

```tsx
{
  kind: "sidepanel",
  path: "/flashcards",
  element: <SidepanelFlashcards />,
  targets: ALL_TARGETS
},
```

- [ ] **Step 5: Run route tests and verify pass**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit route fix**

Run:

```bash
git add apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx apps/tldw-frontend/extension/routes/sidepanel-flashcards.tsx apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts
git commit -m "fix: register flashcards sidepanel route"
```

Expected: commit succeeds.

### Task 3: Clean Remaining User-Facing Transfer Copy

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx`
- Modify as needed: `apps/packages/ui/src/public/_locales/en/option.json`

- [ ] **Step 1: Write failing copy assertions**

In `FlashcardsManager.consistency.test.tsx`, rename the stale test title:

```ts
it("uses Study/Manage/Import Export/Templates/Scheduler tab labels", () => {
  // existing assertions stay on "Import / Export"
})
```

Add or update an import/export tab test to assert user-facing summary copy no longer says `Transfer summary`. If no focused test exists, add a small assertion in an existing ImportExportTab test:

```ts
expect(screen.queryByText("Transfer summary")).not.toBeInTheDocument()
expect(screen.getByText("Import/export summary")).toBeInTheDocument()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
```

Expected: FAIL only on the newly changed copy assertion if current code still renders `Transfer summary`.

- [ ] **Step 3: Update visible copy**

Keep the tab default at `Import / Export`:

```tsx
t("option:flashcards.tabImportExport", { defaultValue: "Import / Export" })
```

Change user-facing summary default copy:

```tsx
t("option:flashcards.transferSummary", {
  defaultValue: "Import/export summary"
})
```

If updating locale JSON, preserve key names unless a rename is already required. Do not rename internal `TransferActionSummary` types.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit copy cleanup**

Run:

```bash
git add apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx apps/packages/ui/src/public/_locales/en/option.json
git commit -m "fix: clarify flashcards import export copy"
```

Expected: commit succeeds. Omit unchanged files from `git add`.

### Task 4: Add Study Selected-Deck Create Handoff

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx`

- [ ] **Step 1: Write failing manager handoff test**

In `FlashcardsManager.consistency.test.tsx`, extend the mocked `ManageTab` props:

```ts
ManageTab: (props: {
  onNavigateToImport: () => void
  openCreateSignal?: number
  initialDeckId?: number
  initialShowWorkspaceDecks?: boolean
  createInitialDeckId?: number | null
  createInitialShowWorkspaceDecks?: boolean
  onCreateHandoffConsumed?: () => void
}) => (
  <div data-testid="mock-manage-tab">
    <button onClick={props.onNavigateToImport}>Route Import</button>
    <span data-testid="mock-open-create-signal">{String(props.openCreateSignal ?? 0)}</span>
    <span data-testid="mock-manage-initial-deck-id">{String(props.initialDeckId ?? "")}</span>
    <span data-testid="mock-manage-show-workspace">{String(props.initialShowWorkspaceDecks ?? false)}</span>
    <span data-testid="mock-create-initial-deck-id">{String(props.createInitialDeckId ?? "")}</span>
    <span data-testid="mock-create-show-workspace">{String(props.createInitialShowWorkspaceDecks ?? false)}</span>
    <button onClick={props.onCreateHandoffConsumed}>Consume Create Handoff</button>
  </div>
)
```

Add test:

```ts
it("passes the selected Study deck into the Create drawer handoff", () => {
  window.history.replaceState({}, "", "/flashcards?tab=review&deck_id=12")
  render(<FlashcardsManager />)

  fireEvent.click(screen.getByText("Route Create"))

  expect(screen.getByTestId("mock-manage-tab")).toBeInTheDocument()
  expect(screen.getByTestId("mock-open-create-signal")).toHaveTextContent("1")
  expect(screen.getByTestId("mock-create-initial-deck-id")).toHaveTextContent("12")
})
```

Add a leakage test:

```ts
it("clears the selected Study deck create handoff after it is consumed", () => {
  window.history.replaceState({}, "", "/flashcards?tab=review&deck_id=12")
  render(<FlashcardsManager />)

  fireEvent.click(screen.getByText("Route Create"))
  expect(screen.getByTestId("mock-create-initial-deck-id")).toHaveTextContent("12")

  fireEvent.click(screen.getByText("Consume Create Handoff"))
  expect(screen.getByTestId("mock-create-initial-deck-id")).toHaveTextContent("")
})
```

- [ ] **Step 2: Write failing create drawer prefill test**

In `FlashcardCreateDrawer.deck-reference.test.tsx`, add a test that renders with decks and `initialDeckId={12}`:

```tsx
render(
  <FlashcardCreateDrawer
    open
    onClose={vi.fn()}
    onSuccess={vi.fn()}
    decks={[{ id: 12, name: "Biology" } as any]}
    initialDeckId={12}
  />
)

expect(screen.getByText("Biology")).toBeInTheDocument()
```

If Ant Design Select rendering makes `getByText` ambiguous, assert the form value through a create submit or by opening the Select and checking the selected option.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx
```

Expected: FAIL due missing `createInitialDeckId` / `initialDeckId`.

- [ ] **Step 4: Implement manager one-shot handoff**

In `FlashcardsManager.tsx`, add state:

```ts
const [createDeckHandoff, setCreateDeckHandoff] = React.useState<{
  deckId?: number | null
  showWorkspaceDecks?: boolean
} | null>(null)
```

Update `routeToCreateEntryPoint`:

```ts
const routeToCreateEntryPoint = React.useCallback(() => {
  setCreateDeckHandoff({
    deckId: reviewDeckId ?? currentStudyIntent?.deckId ?? null,
    showWorkspaceDecks: currentStudyIntent?.forceShowWorkspaceItems ?? false
  })
  setActiveTab("cards")
  setOpenCreateSignal((prev) => prev + 1)
}, [currentStudyIntent?.deckId, currentStudyIntent?.forceShowWorkspaceItems, reviewDeckId])
```

Pass separate props to `ManageTab`:

```tsx
createInitialDeckId={createDeckHandoff?.deckId ?? null}
createInitialShowWorkspaceDecks={createDeckHandoff?.showWorkspaceDecks ?? false}
onCreateHandoffConsumed={() => setCreateDeckHandoff(null)}
```

Do not remove existing `initialDeckId`; it still controls direct Manage route filtering.

- [ ] **Step 5: Implement ManageTab drawer handoff props**

Extend `ManageTabProps` with:

```ts
createInitialDeckId?: number | null
createInitialShowWorkspaceDecks?: boolean
onCreateHandoffConsumed?: () => void
```

In the `openCreateSignal` effect, apply one-shot context:

```ts
const [createDrawerInitialDeckId, setCreateDrawerInitialDeckId] = React.useState<number | null>(null)
const createHandoffRef = React.useRef({
  deckId: createInitialDeckId,
  showWorkspaceDecks: createInitialShowWorkspaceDecks
})
const lastOpenCreateSignalRef = React.useRef(openCreateSignal ?? 0)

React.useEffect(() => {
  createHandoffRef.current = {
    deckId: createInitialDeckId,
    showWorkspaceDecks: createInitialShowWorkspaceDecks
  }
}, [createInitialDeckId, createInitialShowWorkspaceDecks])

React.useEffect(() => {
  if (!openCreateSignal || openCreateSignal === lastOpenCreateSignalRef.current) return
  lastOpenCreateSignalRef.current = openCreateSignal
  const handoff = createHandoffRef.current
  if (handoff.showWorkspaceDecks) {
    setShowWorkspaceDecks(true)
  }
  if (handoff.deckId !== undefined) {
    setMDeckId(handoff.deckId ?? null)
  }
  setCreateDrawerInitialDeckId(handoff.deckId ?? null)
  setCreateOpen(true)
  onCreateHandoffConsumed?.()
}, [onCreateHandoffConsumed, openCreateSignal])
```

Pass to drawer:

```tsx
<FlashcardCreateDrawer
  open={createOpen}
  onClose={() => {
    setCreateOpen(false)
    setCreateDrawerInitialDeckId(null)
    onCreateHandoffConsumed?.()
  }}
  decks={decksQuery.data || []}
  decksLoading={decksQuery.isLoading}
  includeWorkspaceItems={workspaceVisibilityOptions.includeWorkspaceItems}
  workspaceId={workspaceVisibilityOptions.workspaceId}
  initialDeckId={createDrawerInitialDeckId ?? undefined}
/>
```

- [ ] **Step 6: Implement drawer initial deck prop after reset**

Extend `FlashcardCreateDrawerProps`:

```ts
initialDeckId?: number | null
```

Destructure it and update the open effect:

```ts
React.useEffect(() => {
  if (open) {
    form.resetFields()
    if (initialDeckId != null) {
      form.setFieldsValue({ deck_id: initialDeckId })
    }
    setShowPreview(false)
    setShowInlineCreate(false)
    setTemplateValueModalOpen(false)
    setSaveTemplateModalOpen(false)
    setSaveTemplateInitialValues(null)
    setInlineDeckName("")
    inlineSchedulerDraft.resetToDefaults()
  }
}, [form, initialDeckId, inlineSchedulerDraft.resetToDefaults, open])
```

- [ ] **Step 7: Run focused tests and verify pass**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit selected-deck create handoff**

Run:

```bash
git add apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx
git commit -m "fix: preselect study deck when creating flashcards"
```

Expected: commit succeeds.

### Task 5: Keep Re-rate Visible After Rating

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify or create: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx`
- Modify as needed: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`

- [ ] **Step 1: Write failing visible re-rate test**

Prefer a new focused test file if `ReviewTab.create-cta.test.tsx` becomes unwieldy. Mock the same hooks as `ReviewTab.create-cta.test.tsx`.

Test intent:

```tsx
it("keeps Re-rate last card visible after rating advances away from the answer branch", async () => {
  const mutateAsync = vi.fn().mockResolvedValue({
    uuid: "card-1",
    ef: 2.6,
    interval_days: 2,
    repetitions: 2,
    lapses: 0,
    due_at: "2026-02-20T09:30:00.000Z",
    version: 2,
    review_session_id: 101
  })
  // mock useReviewQuery with active card, then due counts as needed
  render(<ReviewTab onNavigateToCreate={vi.fn()} onNavigateToImport={vi.fn()} reviewDeckId={11} onReviewDeckChange={vi.fn()} isActive />)

  fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
  fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))

  expect(await screen.findByRole("button", { name: /Re-rate last card/i })).toBeInTheDocument()
})
```

Add a second assertion in the same test or separate test:

```tsx
fireEvent.click(screen.getByRole("button", { name: /Re-rate last card/i }))
expect(screen.getByText("Question")).toBeInTheDocument()
expect(screen.getByText("Answer")).toBeInTheDocument()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx
```

Expected: FAIL because the current visible re-rate button only renders in the active-card answer branch.

- [ ] **Step 3: Extract shared re-rate control**

In `ReviewTab.tsx`, add a local render helper near other derived UI:

```tsx
const rerateControl = showUndoButton && lastReviewedCard ? (
  <div className="mt-3 rounded border border-border bg-surface2/60 p-3">
    <Button
      type="text"
      icon={<Undo2 className="size-4" />}
      onClick={handleUndoReview}
      className="min-h-11 focus:ring-2 focus:ring-primary focus:ring-offset-2"
      aria-label={t("option:flashcards.rerateLastCardAria", {
        defaultValue: "Re-rate last card, {{seconds}} seconds remaining",
        seconds: undoCountdown
      })}
    >
      <span className="flex items-center gap-2">
        {t("option:flashcards.rerateLastCard", {
          defaultValue: "Re-rate last card"
        })}
        <span className="inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-surface px-1.5 text-xs font-medium tabular-nums" role="timer" aria-live="polite">
          {undoCountdown}s
        </span>
      </span>
    </Button>
  </div>
) : null
```

Use this helper in both active-card and no-active-card/completion areas, outside the `showAnswer` branch. Remove the old branch-local duplicate.

- [ ] **Step 4: Preserve shortcut behavior**

Keep:

```ts
onUndo: showUndoButton ? handleUndoReview : undefined
```

Do not rename shortcut chip text unless product copy is intentionally changing; `Ctrl+Z Re-rate` is accurate.

- [ ] **Step 5: Run focused re-rate test**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Run adjacent ReviewTab tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx
```

Expected: PASS. Update snapshots only if visible UI intentionally changed.

- [ ] **Step 7: Commit visible re-rate**

Run:

```bash
git add apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.rerate.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
git commit -m "fix: keep flashcard re-rate visible after rating"
```

Expected: commit succeeds. Omit unchanged files.

### Task 6: Verify Practice Again Is Absent When No Cram Cards Exist

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx` only if current branch has a `Practice again` button
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx` or `ReviewTab.rerate.test.tsx`

- [ ] **Step 1: Search current branch behavior**

Run:

```bash
rg -n "Practice again|practiceAgain|cram.*again" apps/packages/ui/src/components/Flashcards apps/packages/ui/src/public/_locales/en/option.json
```

Expected: determine whether the branch currently renders a `Practice again` control.

- [ ] **Step 2: Write regression test**

Add a test for caught-up completion with no cram queue:

```tsx
it("does not offer Practice again when no cram cards exist", () => {
  vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
  vi.mocked(useDueCountsQuery).mockReturnValue({
    data: { due: 0, new: 0, learning: 0, total: 0 }
  } as any)
  vi.mocked(useCramQueueQuery).mockReturnValue({ data: [] } as any)

  render(<ReviewTab onNavigateToCreate={vi.fn()} onNavigateToImport={vi.fn()} reviewDeckId={11} onReviewDeckChange={vi.fn()} isActive />)

  expect(screen.queryByRole("button", { name: /Practice again/i })).not.toBeInTheDocument()
})
```

- [ ] **Step 3: Run test and verify result**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
```

Expected: PASS if button is already absent. FAIL if current branch renders it unconditionally.

- [ ] **Step 4: Implement only if failing**

If failing, derive availability from current cram data:

```ts
const hasCramPracticeCards = cramQueue.length > 0
```

Render `Practice again` only when `hasCramPracticeCards` is true. Prefer not adding a new button if no button exists.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit regression or fix**

Run:

```bash
git add apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
git commit -m "test: cover flashcards practice again availability"
```

Expected: commit succeeds. If implementation changed behavior, use `fix:` instead of `test:`.

### Task 7: PR 1 Verification And Browser Pass

**Files:**
- Modify: PR 1 Backlog task
- No new code unless verification finds issues

- [ ] **Step 1: Run PR 1 focused suites**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.deck-reference.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run TypeScript/package gate if available**

Inspect `apps/packages/ui/package.json`, then run the narrowest existing typecheck command. Common candidate:

```bash
cd apps/packages/ui && bun run typecheck
```

Expected: PASS, or document if no such script exists.

- [ ] **Step 3: Run browser verification**

Start the app according to current repo instructions, then verify:

- `/flashcards` loads.
- Study selected deck -> Create opens drawer with that deck selected.
- Rating a card shows `Re-rate last card`.
- Clicking `Re-rate last card` returns to the last card for another rating.
- Caught-up state with no cram cards does not show `Practice again`.

Record URL, seeded data assumptions, and screenshots only if useful.

- [ ] **Step 4: Bandit decision**

PR 1 should be frontend-only. Document:

```text
Bandit skipped: frontend-only changes, no Python touched.
```

- [ ] **Step 5: Update PR 1 Backlog task**

Include:

- Changed files
- Test commands and pass/fail status
- Browser verification notes
- Bandit skip reason
- Any known skips

- [ ] **Step 6: Create PR 1**

Push branch and create PR with a human-written Change summary section that explains what changed and why.

## PR 2: Dashboard-First Study And Session History

Start PR 2 only after PR 1 is merged, or explicitly rebase PR 2 on the finished PR 1 branch.

### Task 8: Create PR 2 Worktree And Inspect Session Payload Contract

**Files:**
- Modify: Backlog task created for PR 2
- No code files yet

- [ ] **Step 1: Fetch latest dev after PR 1**

Run:

```bash
git fetch origin dev
```

Expected: command completes without errors.

- [ ] **Step 2: Create PR 2 worktree**

Run:

```bash
git worktree add .worktrees/flashcards-dashboard-session-history -b codex/flashcards-dashboard-session-history origin/dev
```

Expected: worktree created on new branch.

- [ ] **Step 3: Create PR 2 Backlog task**

Use Backlog MCP or CLI. Include:

- Title: `Implement flashcards dashboard and session history`
- References: this plan and the design spec
- Modified files from the PR 2 file map

- [ ] **Step 4: Inspect current session payload code**

Read:

- `apps/packages/ui/src/services/flashcards.ts`
- `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`

Expected current state on the inspected branch: no deck-name field in `FlashcardReviewSessionSummary`.

- [ ] **Step 5: Record schema decision**

If current payload still lacks a user-facing deck name, write in PR 2 task notes:

```text
Session payload inspection: no preserved deck-name snapshot exists. PR 2 will add nullable deck_name_snapshot to DB/API/client.
```

If the payload already has one, write:

```text
Session payload inspection: existing field <name> is sufficient. PR 2 will not change backend schema.
```

### Task 9A: Add Backend Deck Name Snapshot If Payload Inspection Requires It

Skip this task only if Task 8 proves an existing session payload field is sufficient.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`

- [ ] **Step 1: Write failing DB test for session snapshot**

In `test_flashcard_review_sessions.py`, add:

```py
def test_review_session_preserves_deck_name_snapshot_after_deck_rename(db: CharactersRAGDB):
    deck_id, card_uuid = _create_card(db, deck_name="Original Deck Name")

    updated = db.review_flashcard(card_uuid, rating=4, answer_time_ms=700)
    session_id = int(updated["review_session_id"])
    db.update_deck(deck_id, {"name": "Renamed Deck"})

    sessions = db.list_flashcard_review_sessions(deck_id=deck_id)

    assert int(sessions[0]["id"]) == session_id  # nosec B101
    assert sessions[0]["deck_name_snapshot"] == "Original Deck Name"  # nosec B101
```

Adjust `db.update_deck` call to match the actual deck update API if needed.

- [ ] **Step 2: Write failing API test**

In `test_study_suggestions_endpoints_api.py`, extend `test_review_sessions_list_route_returns_db_sessions_with_filters`:

```py
assert body[0]["deck_name_snapshot"] == "Review Route Deck"  # nosec B101
```

- [ ] **Step 3: Run backend tests and verify failure**

Run with venv:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py::test_review_session_preserves_deck_name_snapshot_after_deck_rename tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py::test_review_sessions_list_route_returns_db_sessions_with_filters -v
```

Expected: FAIL due missing `deck_name_snapshot`.

- [ ] **Step 4: Add nullable DB column and migrations**

In `ChaChaNotes_DB.py` table definitions, add:

```sql
deck_name_snapshot TEXT,
```

Add SQLite schema ensure:

```py
if "deck_name_snapshot" not in session_cols:
    conn.execute("ALTER TABLE flashcard_review_sessions ADD COLUMN deck_name_snapshot TEXT")
```

Add PostgreSQL ensure:

```py
self.backend.execute(
    "ALTER TABLE flashcard_review_sessions ADD COLUMN IF NOT EXISTS deck_name_snapshot TEXT",
    connection=conn,
)
```

- [ ] **Step 5: Add snapshot to selects/deserializer**

Update every `SELECT` on `flashcard_review_sessions` used by:

- `list_flashcard_review_sessions`
- `get_or_create_flashcard_review_session`
- `get_flashcard_review_session`
- `mark_flashcard_review_session_completed`

Include `deck_name_snapshot` and return it from `_deserialize_flashcard_review_session_row`.

- [ ] **Step 6: Populate snapshot at session creation**

Before inserting a session, resolve the current deck name when `deck_id` is not null. Keep it nullable:

```py
deck_name_snapshot = self._get_deck_name_snapshot(deck_id)
```

If no helper exists, add a small private helper near review-session helpers:

```py
def _get_deck_name_snapshot(self, deck_id: int | None) -> str | None:
    if deck_id is None:
        return None
    row = self.execute_query(
        "SELECT name FROM decks WHERE id = ?",
        (int(deck_id),),
    ).fetchone()
    if not row:
        return None
    name = str(row["name"] if isinstance(row, dict) else row[0]).strip()
    return name or None
```

Use the project row access pattern if this helper needs adjustment.

- [ ] **Step 7: Update API and frontend types**

In Pydantic schema:

```py
deck_name_snapshot: Optional[str] = None
```

In TypeScript:

```ts
deck_name_snapshot?: string | null
```

- [ ] **Step 8: Run backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py::test_review_sessions_list_route_returns_db_sessions_with_filters -v
```

Expected: PASS. If the full file is too slow or unrelated failures appear, record exact failures and run the focused new tests.

- [ ] **Step 9: Commit backend snapshot work**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/schemas/flashcards.py apps/packages/ui/src/services/flashcards.ts tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py
git commit -m "feat: preserve flashcard review deck names"
```

Expected: commit succeeds.

### Task 9B: Frontend Session History Labels

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx`
- Modify: `apps/packages/ui/src/services/flashcards.ts` if not already changed in Task 9A

- [ ] **Step 1: Write failing preserved-name tests**

Add tests:

```tsx
it("shows preserved deck names instead of raw deck ids", () => {
  vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
    data: [{
      id: 81,
      deck_id: 12,
      deck_name_snapshot: "Renal Biology",
      review_mode: "due",
      tag_filter: null,
      scope_key: "due:deck:12",
      status: "completed",
      started_at: "2026-04-05T18:00:00Z",
      last_activity_at: "2026-04-05T18:10:00Z",
      completed_at: "2026-04-05T18:12:00Z",
      client_id: "test"
    }],
    isLoading: false,
    isFetching: false
  } as any)

  render(<RecentStudySessions deckId={12} selectedSessionId={null} onOpenSession={vi.fn()} isActive />)

  expect(screen.getByText("Renal Biology")).toBeInTheDocument()
  expect(screen.getByText("Due review")).toBeInTheDocument()
  expect(screen.queryByText("Deck 12")).not.toBeInTheDocument()
  expect(screen.queryByText("due:deck:12")).not.toBeInTheDocument()
})
```

Add fallback test:

```tsx
it("uses non-technical fallback when no deck name can be resolved", () => {
  // data has deck_id but no snapshot
  expect(screen.getByText("Deck unavailable")).toBeInTheDocument()
  expect(screen.queryByText("Deck 12")).not.toBeInTheDocument()
  expect(screen.queryByText(/due:deck/)).not.toBeInTheDocument()
})
```

Add current-deck lookup test:

```tsx
it("uses the current deck name when a legacy session has no snapshot", () => {
  vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
    data: [{
      id: 82,
      deck_id: 12,
      review_mode: "due",
      tag_filter: null,
      scope_key: "due:deck:12",
      status: "completed",
      started_at: "2026-04-05T18:00:00Z",
      last_activity_at: "2026-04-05T18:10:00Z",
      completed_at: "2026-04-05T18:12:00Z",
      client_id: "test"
    }],
    isLoading: false,
    isFetching: false
  } as any)

  render(
    <RecentStudySessions
      deckId={12}
      decks={[{ id: 12, name: "Current Biology" } as any]}
      selectedSessionId={null}
      onOpenSession={vi.fn()}
      isActive
    />
  )

  expect(screen.getByText("Current Biology")).toBeInTheDocument()
  expect(screen.queryByText("Deck 12")).not.toBeInTheDocument()
  expect(screen.queryByText("due:deck:12")).not.toBeInTheDocument()
})
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
```

Expected: FAIL because current UI renders `Deck {id}`.

- [ ] **Step 3: Implement label resolver**

In `RecentStudySessions.tsx`, add optional `decks` prop and helper:

```ts
function getSessionDeckLabel(
  session: FlashcardReviewSessionSummary,
  decksById: Map<number, string>
): string | null {
  const snapshot = session.deck_name_snapshot?.trim()
  if (snapshot) return snapshot
  if (session.deck_id != null) {
    const currentName = decksById.get(session.deck_id)?.trim()
    return currentName || "Deck unavailable"
  }
  return null
}

function getSessionScopeLabel(session: FlashcardReviewSessionSummary): string {
  const modeLabel = session.review_mode === "cram" ? "Cram review" : "Due review"
  const tag = session.tag_filter?.trim()
  return tag ? `${modeLabel} - ${tag}` : modeLabel
}
```

Build the map:

```ts
const decksById = React.useMemo(
  () => new Map((decks ?? []).map((deck) => [deck.id, deck.name])),
  [decks]
)
```

In `ReviewTab.tsx`, pass the current deck list:

```tsx
<RecentStudySessions
  deckId={reviewDeckId ?? null}
  decks={availableDecks}
  selectedSessionId={selectedStudySessionId}
  onOpenSession={...}
  isActive={isActive}
/>
```

If Task 8 found an existing sufficient deck-name field with a different name, normalize that field into the helper before using `deck_name_snapshot`.

Replace:

```tsx
{session.deck_id != null ? <Tag>Deck {session.deck_id}</Tag> : null}
```

with:

```tsx
{deckLabel ? <Tag>{deckLabel}</Tag> : null}
```

Replace any raw scope-key display such as:

```tsx
<Text>{session.scope_key}</Text>
```

with:

```tsx
<Text>{getSessionScopeLabel(session)}</Text>
```

The tests must fail if `due:deck:12`, `cram:deck:12`, or another raw `scope_key` appears in visible session history copy.

- [ ] **Step 4: Improve action copy only within scope**

Change snapshot action labels from:

```tsx
Viewing snapshot
Reopen snapshot for session ${session.id}
```

to:

```tsx
Viewing completed session
View completed session
```

Do not build resume behavior in this slice.

- [ ] **Step 5: Run frontend session tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit session labels**

Run:

```bash
git add apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx apps/packages/ui/src/services/flashcards.ts
git commit -m "fix: show deck names in flashcard session history"
```

Expected: commit succeeds. Omit unchanged files.

### Task 10: All-Deck Study Dashboard First

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx`

- [ ] **Step 1: Write failing dashboard-first test**

Add to `ReviewTab.create-cta.test.tsx`:

```tsx
it("shows all-deck dashboard before starting all due review", () => {
  vi.mocked(useDecksQuery).mockReturnValue({
    data: [{ id: 11, name: "Biology" }],
    isLoading: false
  } as any)
  vi.mocked(useReviewQuery).mockReturnValue({
    data: {
      uuid: "global-due-card",
      deck_id: 11,
      front: "Global due question",
      back: "Global due answer",
      notes: null,
      extra: null,
      is_cloze: false,
      tags: [],
      ef: 2.5,
      interval_days: 1,
      repetitions: 1,
      lapses: 0,
      due_at: null,
      last_reviewed_at: null,
      last_modified: null,
      deleted: false,
      client_id: "test",
      version: 1,
      model_type: "basic",
      reverse: false
    }
  } as any)
  vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as any)
  vi.mocked(useDueCountsQuery).mockReturnValue({
    data: { due: 1, new: 0, learning: 0, total: 1 }
  } as any)

  render(<ReviewTab onNavigateToCreate={vi.fn()} onNavigateToImport={vi.fn()} reviewDeckId={undefined} onReviewDeckChange={vi.fn()} isActive />)

  expect(screen.getByTestId("flashcards-review-deck-dashboard")).toBeInTheDocument()
  expect(screen.queryByText("Global due question")).not.toBeInTheDocument()
})
```

- [ ] **Step 2: Write failing Review all due test**

Add:

```tsx
it("starts all-deck review from Review all due", () => {
  render(/* same setup as previous test */)

  fireEvent.click(screen.getByRole("button", { name: /Review all due/i }))

  expect(screen.getByText("Global due question")).toBeInTheDocument()
})
```

- [ ] **Step 3: Write selected-deck fast-path test**

Add:

```tsx
it("keeps selected-deck review on the fast path", () => {
  render(<ReviewTab onNavigateToCreate={vi.fn()} onNavigateToImport={vi.fn()} reviewDeckId={11} onReviewDeckChange={vi.fn()} isActive />)

  expect(screen.queryByTestId("flashcards-review-deck-dashboard")).not.toBeInTheDocument()
  expect(screen.getByText("Global due question")).toBeInTheDocument()
})
```

- [ ] **Step 4: Write scope-reset test**

Add:

```tsx
it("returns to the all-deck dashboard when review scope changes", () => {
  const props = {
    onNavigateToCreate: vi.fn(),
    onNavigateToImport: vi.fn(),
    onReviewDeckChange: vi.fn(),
    isActive: true
  }
  const { rerender } = render(
    <ReviewTab {...props} reviewDeckId={undefined} />
  )

  fireEvent.click(screen.getByRole("button", { name: /Review all due/i }))
  expect(screen.getByText("Global due question")).toBeInTheDocument()

  rerender(<ReviewTab {...props} reviewDeckId={22} />)
  rerender(<ReviewTab {...props} reviewDeckId={undefined} />)

  expect(screen.getByTestId("flashcards-review-deck-dashboard")).toBeInTheDocument()
})
```

- [ ] **Step 5: Run tests and verify failure**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
```

Expected: FAIL because all-deck review currently consumes `reviewQuery.data` immediately.

- [ ] **Step 6: Add all-deck state and derived active card**

In `ReviewTab.tsx`, add state:

```ts
const [allDeckReviewStarted, setAllDeckReviewStarted] = React.useState(false)
```

Derive:

```ts
const isAllDeckDueReview = reviewMode === "due" && reviewDeckId == null
const canShowAllDeckDashboard = isAllDeckDueReview && !allDeckReviewStarted && !reviewOverrideCard && !localOverrideCard
const activeCard =
  localOverrideCard ??
  reviewOverrideCard ??
  (reviewMode === "cram"
    ? cramQueueCard
    : canShowAllDeckDashboard
      ? null
      : reviewQuery.data)
```

- [ ] **Step 7: Reset all-deck start state on scope changes**

Add effect:

```ts
React.useEffect(() => {
  setAllDeckReviewStarted(false)
}, [reviewDeckId, reviewMode, cramTagFilter, forceShowWorkspaceItems])
```

If this resets too often during implementation, replace dependencies with a stable `reviewScopeKey`.

- [ ] **Step 8: Render dashboard in no-card branch**

Add a dashboard block before existing empty-state branches:

```tsx
{canShowAllDeckDashboard ? (
  <Card data-testid="flashcards-review-deck-dashboard">
    <Space direction="vertical" size="middle" className="w-full">
      <Text strong>{t("option:flashcards.deckDashboardTitle", { defaultValue: "Choose what to study" })}</Text>
      <Text type="secondary">
        {t("option:flashcards.deckDashboardDescription", {
          defaultValue: "Review all due cards or pick a deck before starting."
        })}
      </Text>
      <Space wrap>
        <Button
          type="primary"
          onClick={() => setAllDeckReviewStarted(true)}
          data-testid="flashcards-review-all-due"
        >
          {t("option:flashcards.reviewAllDue", { defaultValue: "Review all due" })}
        </Button>
        {decks.map((deck) => (
          <Button key={deck.id} onClick={() => onReviewDeckChange(deck.id)}>
            {formatDeckDisplayName(deck, `Deck ${deck.id}`)}
          </Button>
        ))}
      </Space>
    </Space>
  </Card>
) : activeCard ? (
  // existing active card
) : (
  // existing empty/completion card
)}
```

Use the existing deck label helper/imports already available in `ReviewTab.tsx`; do not introduce a large new dashboard component unless the branch becomes hard to read.

- [ ] **Step 9: Run focused tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
```

Expected: PASS.

- [ ] **Step 10: Run adjacent ReviewTab suites**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.analytics-summary.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.study-suggestions.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.study-pack-remediation.test.tsx
```

Expected: PASS. Update snapshots only when dashboard-first behavior intentionally changes the no-card branch.

- [ ] **Step 11: Commit dashboard-first behavior**

Run:

```bash
git add apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/__snapshots__/ReviewTab.create-cta.test.tsx.snap
git commit -m "feat: show flashcard deck dashboard before all-deck review"
```

Expected: commit succeeds. Omit unchanged snapshot file if not updated.

### Task 11: PR 2 Verification

**Files:**
- Modify: PR 2 Backlog task
- No new code unless verification finds issues

- [ ] **Step 1: Run frontend focused suites**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx src/components/Flashcards/components/__tests__/RecentStudySessions.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run backend tests if backend changed**

If Task 9A ran, run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py::test_review_sessions_list_route_returns_db_sessions_with_filters -v
```

Expected: PASS or documented unrelated baseline failures.

- [ ] **Step 3: Run Bandit if backend changed**

If Task 9A ran, run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/api/v1/schemas/flashcards.py -f json -o /tmp/bandit_flashcards_session_history.json
```

Expected: no new findings in touched code. If no backend changed, document Bandit skip.

- [ ] **Step 4: Browser verification**

Verify in browser:

- `/flashcards` Study with no deck selected shows dashboard first.
- `Review all due` starts all-deck review without selecting a deck.
- Selecting a deck starts that deck review.
- Recent sessions show preserved or resolved deck names, not raw `Deck 1`.
- Deleted/unavailable deck fallback is non-technical.

- [ ] **Step 5: Update PR 2 Backlog task**

Record:

- Payload inspection decision
- Changed files
- Test commands and results
- Browser verification notes
- Bandit result or skip reason

- [ ] **Step 6: Create PR 2**

Push branch and create PR with a human-written Change summary section that explains what changed and why.

## Cross-PR Final Checks

- [ ] PR 1 and PR 2 both reference this plan and the design spec.
- [ ] PR 2 is rebased on latest `dev` after PR 1 merges.
- [ ] No unrelated files from the dirty main checkout are included.
- [ ] Each PR includes focused tests for its changed behavior.
- [ ] Each PR documents skipped verification, if any.
- [ ] AI-generated PR merge gate is satisfied by a human-authored Change summary.
