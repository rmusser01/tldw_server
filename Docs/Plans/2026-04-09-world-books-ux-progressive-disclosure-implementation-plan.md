# World Books UX Progressive Disclosure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the 2,602-line Manager.tsx monolith into focused components implementing a two-panel layout with progressive disclosure, improving the NNG heuristic score from ~2.85/5 to ~4/5.

**Architecture:** Extract the monolith into 5 new components (EmptyState, Toolbar, ListPanel, DetailPanel, BudgetBar) while preserving all existing hooks, utilities, and API client patterns. The existing hooks (`useWorldBookFiltering`, `useWorldBookBulkActions`, `useWorldBookImportExport`, `useWorldBookTestMatching`) remain unchanged — they're lifted into the new `Manager.tsx` shell and passed down via props. The Manager becomes a ~200-line orchestrator.

**Tech Stack:** React 18, TypeScript, Ant Design, Tailwind CSS, React Query, Vitest + React Testing Library, react-i18next, Zustand (layout store), lucide-react icons.

**Design doc:** `Docs/Plans/2026-04-09-world-books-ux-progressive-disclosure-design.md`

**Test runner:** `vitest run` from `apps/packages/ui/`
**Test command (WorldBooks only):** `npm run test:worldbooks` (or `vitest run src/components/Option/WorldBooks/__tests__ --maxWorkers=1`)
**Test file location:** `apps/packages/ui/src/components/Option/WorldBooks/__tests__/`
**Test naming convention:** `[Component].[descriptor].test.tsx`

**Existing test files (53):** Must remain passing throughout. Run the full WorldBooks test suite after each task.

---

## Task 1: WorldBookEmptyState — First-Run Experience

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookEmptyState.tsx`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEmptyState.test.tsx`
- Reference: `apps/packages/ui/src/components/Common/FeatureEmptyState.tsx` (pattern to follow)
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/worldBookFormUtils.ts` (templates)

This is a standalone component with no dependencies on Manager.tsx state. Safe to build first.

**Step 1: Write the failing test**

Create `__tests__/WorldBookEmptyState.test.tsx`:

```tsx
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { WorldBookEmptyState } from "../WorldBookEmptyState"

describe("WorldBookEmptyState", () => {
  it("renders the 3-step visual flow", () => {
    render(
      <WorldBookEmptyState
        onCreateNew={vi.fn()}
        onCreateFromTemplate={vi.fn()}
        onImport={vi.fn()}
      />
    )
    expect(screen.getByText(/create a world book/i)).toBeInTheDocument()
    expect(screen.getByText(/add entries/i)).toBeInTheDocument()
    expect(screen.getByText(/attach to/i)).toBeInTheDocument()
  })

  it("renders the keyword matching example", () => {
    render(
      <WorldBookEmptyState
        onCreateNew={vi.fn()}
        onCreateFromTemplate={vi.fn()}
        onImport={vi.fn()}
      />
    )
    expect(screen.getByText(/magic system/i)).toBeInTheDocument()
  })

  it("renders template quick-start buttons", () => {
    render(
      <WorldBookEmptyState
        onCreateNew={vi.fn()}
        onCreateFromTemplate={vi.fn()}
        onImport={vi.fn()}
      />
    )
    expect(screen.getByRole("button", { name: /fantasy/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /sci-fi/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /product/i })).toBeInTheDocument()
  })

  it("calls onCreateNew when primary CTA is clicked", async () => {
    const user = userEvent.setup()
    const onCreateNew = vi.fn()
    render(
      <WorldBookEmptyState
        onCreateNew={onCreateNew}
        onCreateFromTemplate={vi.fn()}
        onImport={vi.fn()}
      />
    )
    await user.click(screen.getByRole("button", { name: /create your first/i }))
    expect(onCreateNew).toHaveBeenCalledTimes(1)
  })

  it("calls onCreateFromTemplate with template key", async () => {
    const user = userEvent.setup()
    const onCreateFromTemplate = vi.fn()
    render(
      <WorldBookEmptyState
        onCreateNew={vi.fn()}
        onCreateFromTemplate={onCreateFromTemplate}
        onImport={vi.fn()}
      />
    )
    await user.click(screen.getByRole("button", { name: /fantasy/i }))
    expect(onCreateFromTemplate).toHaveBeenCalledWith("fantasy")
  })

  it("calls onImport when import link is clicked", async () => {
    const user = userEvent.setup()
    const onImport = vi.fn()
    render(
      <WorldBookEmptyState
        onCreateNew={vi.fn()}
        onCreateFromTemplate={vi.fn()}
        onImport={onImport}
      />
    )
    await user.click(screen.getByRole("button", { name: /import/i }))
    expect(onImport).toHaveBeenCalledTimes(1)
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookEmptyState.test.tsx`
Expected: FAIL — module `../WorldBookEmptyState` not found.

**Step 3: Write the component**

Create `WorldBookEmptyState.tsx`. Requirements:
- Props: `onCreateNew: () => void`, `onCreateFromTemplate: (key: string) => void`, `onImport: () => void`
- 3-step flow: "1. Create a world book" → "2. Add entries with keywords" → "3. Attach to a character or chat"
- Concrete example paragraph mentioning "magic system"
- Primary CTA: "Create your first world book" button (`type="primary"`)
- Template buttons: one per `WORLD_BOOK_STARTER_TEMPLATES` from `worldBookFormUtils.ts` — use `template.key` and `template.label`
- Secondary: "Import from JSON" button (`type="default"`)
- Styling: Follow FeatureEmptyState patterns — centered card, `rounded-2xl border border-border bg-surface`, Tailwind spacing
- Use `useTranslation(["option"])` for all user-visible text with `defaultValue` fallbacks

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookEmptyState.test.tsx`
Expected: PASS (6 tests).

**Step 5: Run existing WorldBooks tests to confirm no regressions**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All 53 existing test files pass.

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookEmptyState.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEmptyState.test.tsx
git commit -m "feat(world-books): add WorldBookEmptyState with 3-step onboarding flow"
```

---

## Task 2: WorldBookBudgetBar — Reusable Budget Indicator

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookBudgetBar.tsx`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookBudgetBar.test.tsx`
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/worldBookStatsUtils.ts` (utilization functions)

Standalone component that wraps existing `getBudgetUtilizationPercent`, `getBudgetUtilizationBand`, and `getBudgetUtilizationColor` utils.

**Step 1: Write the failing test**

Create `__tests__/WorldBookBudgetBar.test.tsx`:

```tsx
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { WorldBookBudgetBar } from "../WorldBookBudgetBar"

describe("WorldBookBudgetBar", () => {
  it("renders current and max token values", () => {
    render(<WorldBookBudgetBar estimatedTokens={285} tokenBudget={700} />)
    expect(screen.getByText(/285/)).toBeInTheDocument()
    expect(screen.getByText(/700/)).toBeInTheDocument()
  })

  it("renders the meter role with correct aria attributes", () => {
    render(<WorldBookBudgetBar estimatedTokens={285} tokenBudget={700} />)
    const meter = screen.getByRole("meter")
    expect(meter).toHaveAttribute("aria-valuenow", "285")
    expect(meter).toHaveAttribute("aria-valuemax", "700")
  })

  it("shows warning when usage exceeds budget", () => {
    render(<WorldBookBudgetBar estimatedTokens={800} tokenBudget={700} />)
    expect(screen.getByText(/exceeds/i)).toBeInTheDocument()
  })

  it("renders nothing when tokenBudget is null or zero", () => {
    const { container } = render(<WorldBookBudgetBar estimatedTokens={100} tokenBudget={0} />)
    expect(container.querySelector("[role='meter']")).not.toBeInTheDocument()
  })

  it("shows projected state when projectedTokens is provided", () => {
    render(
      <WorldBookBudgetBar
        estimatedTokens={285}
        tokenBudget={700}
        projectedTokens={340}
      />
    )
    expect(screen.getByText(/after save/i)).toBeInTheDocument()
    expect(screen.getByText(/340/)).toBeInTheDocument()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookBudgetBar.test.tsx`
Expected: FAIL — module not found.

**Step 3: Write the component**

Create `WorldBookBudgetBar.tsx`. Requirements:
- Props: `estimatedTokens: number`, `tokenBudget: number`, `projectedTokens?: number`, `className?: string`
- Uses `getBudgetUtilizationPercent`, `getBudgetUtilizationBand`, `getBudgetUtilizationColor` from `worldBookStatsUtils.ts`
- Renders Ant Design `<Progress>` bar with `role="meter"`, `aria-valuenow`, `aria-valuemax`, `aria-label="Token budget usage"`
- Shows `{estimatedTokens}/{tokenBudget} tokens` label
- When `projectedTokens` is provided, show secondary label: "After save: {projectedTokens}/{tokenBudget}"
- When utilization > 100%, show warning text: "Estimated usage exceeds the configured budget."
- When `tokenBudget` is 0 or not a finite number, render nothing (return `null`)
- Compact styling: `text-xs`, inline with the entries list header

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookBudgetBar.test.tsx`
Expected: PASS (5 tests).

**Step 5: Run existing WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All existing tests pass.

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookBudgetBar.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookBudgetBar.test.tsx
git commit -m "feat(world-books): add WorldBookBudgetBar with projected-state and a11y meter"
```

---

## Task 3: WorldBookToolbar — Reorganized Toolbar with Tools Dropdown

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookToolbar.tsx`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookToolbar.test.tsx`
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/worldBookManagerUtils.ts` (LOREBOOK_DEBUG_ENTRYPOINT_HREF)

**Step 1: Write the failing test**

Create `__tests__/WorldBookToolbar.test.tsx`:

```tsx
import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { WorldBookToolbar } from "../WorldBookToolbar"

const defaultProps = {
  listSearch: "",
  onSearchChange: vi.fn(),
  enabledFilter: "all" as const,
  onEnabledFilterChange: vi.fn(),
  attachmentFilter: "all" as const,
  onAttachmentFilterChange: vi.fn(),
  onNewWorldBook: vi.fn(),
  onOpenTestMatching: vi.fn(),
  onOpenMatrix: vi.fn(),
  onOpenGlobalStats: vi.fn(),
  onImport: vi.fn(),
  onExportAll: vi.fn(),
  hasWorldBooks: true,
  hasSelection: false,
  globalStatsFetching: false,
  bulkExportAllLoading: false,
}

describe("WorldBookToolbar", () => {
  it("renders search input and filter dropdowns", () => {
    render(<WorldBookToolbar {...defaultProps} />)
    expect(screen.getByLabelText("Search world books")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter by enabled status")).toBeInTheDocument()
    expect(screen.getByLabelText("Filter by attachment state")).toBeInTheDocument()
  })

  it("renders 'New World Book' as primary button", () => {
    render(<WorldBookToolbar {...defaultProps} />)
    const btn = screen.getByRole("button", { name: /new world book/i })
    expect(btn).toBeInTheDocument()
  })

  it("renders a Tools dropdown that contains analysis and I/O actions", async () => {
    const user = userEvent.setup()
    render(<WorldBookToolbar {...defaultProps} />)
    const toolsBtn = screen.getByRole("button", { name: /tools/i })
    await user.click(toolsBtn)

    // Analysis section
    expect(screen.getByText(/test matching/i)).toBeInTheDocument()
    expect(screen.getByText(/relationship matrix/i)).toBeInTheDocument()
    expect(screen.getByText(/global statistics/i)).toBeInTheDocument()

    // I/O section
    expect(screen.getByText(/import json/i)).toBeInTheDocument()
    expect(screen.getByText(/export all/i)).toBeInTheDocument()
  })

  it("disables analysis tools when no world books exist", async () => {
    const user = userEvent.setup()
    render(<WorldBookToolbar {...defaultProps} hasWorldBooks={false} />)
    const toolsBtn = screen.getByRole("button", { name: /tools/i })
    await user.click(toolsBtn)

    // Menu items should be disabled (Ant Design renders aria-disabled)
    const testMatchingItem = screen.getByText(/test matching/i).closest("[role='menuitem']")
    expect(testMatchingItem).toHaveAttribute("aria-disabled", "true")
  })

  it("calls onNewWorldBook when primary button is clicked", async () => {
    const user = userEvent.setup()
    const onNewWorldBook = vi.fn()
    render(<WorldBookToolbar {...defaultProps} onNewWorldBook={onNewWorldBook} />)
    await user.click(screen.getByRole("button", { name: /new world book/i }))
    expect(onNewWorldBook).toHaveBeenCalledTimes(1)
  })

  it("shows 'Export Selected' in Tools menu when hasSelection is true", async () => {
    const user = userEvent.setup()
    render(<WorldBookToolbar {...defaultProps} hasSelection={true} onExportSelected={vi.fn()} />)
    const toolsBtn = screen.getByRole("button", { name: /tools/i })
    await user.click(toolsBtn)
    expect(screen.getByText(/export selected/i)).toBeInTheDocument()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookToolbar.test.tsx`
Expected: FAIL — module not found.

**Step 3: Write the component**

Create `WorldBookToolbar.tsx`. Requirements:
- Props: search state, filter state, callback props for each action, loading/disabled flags
- Layout: flex-wrap row with search + filters on left, Tools dropdown + New button on right
- Tools dropdown uses Ant Design `<Dropdown menu={{ items }}>` with grouped items:
  - Group 1 (Analysis): Test Matching, Relationship Matrix, Global Statistics
  - Group 2 (I/O): Import JSON, Export All, Export Selected (conditional on `hasSelection`)
  - Group 3 (Debug): Chat Injection Panel (links to `LOREBOOK_DEBUG_ENTRYPOINT_HREF`)
  - Use `type: "divider"` between groups
- "New World Book" is `<Button type="primary">`
- Analysis/export items disabled when `!hasWorldBooks`
- Search uses `<Input allowClear>` with `min-w-[220px] md:w-72`
- Filters use `<Select>` matching existing options from Manager.tsx lines 1363-1391
- Use `useTranslation(["option"])` for labels

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookToolbar.test.tsx`
Expected: PASS (6 tests).

**Step 5: Run existing WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All existing tests pass.

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookToolbar.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookToolbar.test.tsx
git commit -m "feat(world-books): add WorldBookToolbar with Tools dropdown and visual hierarchy"
```

---

## Task 4: WorldBookListPanel — Simplified Table

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookListPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookListPanel.test.tsx`
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx:1264-1558` (current table + columns)

**Step 1: Write the failing test**

Create `__tests__/WorldBookListPanel.test.tsx`:

```tsx
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { WorldBookListPanel } from "../WorldBookListPanel"

const sampleBooks = [
  { id: 1, name: "Fantasy Lore", description: "Core lore", entry_count: 12, enabled: true, last_modified: new Date().toISOString(), token_budget: 700 },
  { id: 2, name: "Product KB", description: "Reusable facts", entry_count: 8, enabled: true, last_modified: new Date().toISOString(), token_budget: 500 },
  { id: 3, name: "Old Campaign", description: "Archived", entry_count: 3, enabled: false, last_modified: new Date().toISOString(), token_budget: 500 },
]

const defaultProps = {
  worldBooks: sampleBooks,
  selectedWorldBookId: null as number | null,
  onSelectWorldBook: vi.fn(),
  selectedRowKeys: [] as React.Key[],
  onSelectedRowKeysChange: vi.fn(),
  pendingDeleteIds: [] as number[],
  onEditWorldBook: vi.fn(),
  onRowAction: vi.fn(),
  tableSort: {} as any,
  onTableSortChange: vi.fn(),
  loading: false,
}

describe("WorldBookListPanel", () => {
  it("renders Name+Description merged column", () => {
    render(<WorldBookListPanel {...defaultProps} />)
    expect(screen.getByText("Fantasy Lore")).toBeInTheDocument()
    expect(screen.getByText("Core lore")).toBeInTheDocument()
  })

  it("does NOT render BookOpen icon column, Attached To, or Budget columns", () => {
    render(<WorldBookListPanel {...defaultProps} />)
    // These column headers should not exist
    expect(screen.queryByText("Attached To")).not.toBeInTheDocument()
    expect(screen.queryByText("Budget")).not.toBeInTheDocument()
  })

  it("renders only Edit and overflow menu action buttons per row", () => {
    render(<WorldBookListPanel {...defaultProps} />)
    const editButtons = screen.getAllByLabelText(/edit/i)
    // One per row
    expect(editButtons.length).toBe(3)
    const overflowButtons = screen.getAllByLabelText(/more actions/i)
    expect(overflowButtons.length).toBe(3)
  })

  it("calls onSelectWorldBook when a row is clicked", async () => {
    const user = userEvent.setup()
    const onSelectWorldBook = vi.fn()
    render(<WorldBookListPanel {...defaultProps} onSelectWorldBook={onSelectWorldBook} />)
    await user.click(screen.getByText("Fantasy Lore"))
    expect(onSelectWorldBook).toHaveBeenCalledWith(1)
  })

  it("highlights the selected row", () => {
    render(<WorldBookListPanel {...defaultProps} selectedWorldBookId={1} />)
    // The selected row should have a distinct class
    const row = screen.getByText("Fantasy Lore").closest("tr")
    expect(row?.className).toMatch(/bg-primary|selected|ring/)
  })

  it("shows disabled status with icon alongside color tag", () => {
    render(<WorldBookListPanel {...defaultProps} />)
    const disabledTags = screen.getAllByText("Disabled")
    expect(disabledTags.length).toBe(1)
  })

  it("renders overflow menu with correct actions", async () => {
    const user = userEvent.setup()
    render(<WorldBookListPanel {...defaultProps} />)
    const overflowButtons = screen.getAllByLabelText(/more actions/i)
    await user.click(overflowButtons[0])

    expect(screen.getByText(/manage entries/i)).toBeInTheDocument()
    expect(screen.getByText(/duplicate/i)).toBeInTheDocument()
    expect(screen.getByText(/quick attach/i)).toBeInTheDocument()
    expect(screen.getByText(/export json/i)).toBeInTheDocument()
    expect(screen.getByText(/statistics/i)).toBeInTheDocument()
    expect(screen.getByText(/delete/i)).toBeInTheDocument()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookListPanel.test.tsx`
Expected: FAIL — module not found.

**Step 3: Write the component**

Create `WorldBookListPanel.tsx`. Requirements:
- Props: `worldBooks`, `selectedWorldBookId`, `onSelectWorldBook`, `selectedRowKeys`, `onSelectedRowKeysChange`, `pendingDeleteIds`, `onEditWorldBook`, `onRowAction` (callback for overflow menu items with action key + record), `tableSort`, `onTableSortChange`, `loading`
- Columns: Checkbox (via rowSelection), Name+Description (merged), Entries count, Status (with CircleCheck/CirclePause icon + color tag), Last Modified (relative with absolute tooltip), Actions (Edit button + overflow Dropdown)
- Row click calls `onSelectWorldBook(record.id)` — use `onRow={{ onClick }}` on the Table
- Selected row gets `bg-primary/5 ring-1 ring-primary/20` class via `rowClassName`
- Overflow menu items: Manage Entries, Duplicate, Quick Attach Characters, Export JSON, Statistics, divider, Delete (danger)
- `onRowAction` receives `(action: string, record: any)` where action is one of: `"entries"`, `"duplicate"`, `"attach"`, `"export"`, `"stats"`, `"delete"`
- Pending delete rows show `<Tag color="orange">Pending delete</Tag>` next to name
- Use `formatWorldBookLastModified` from `worldBookListUtils.ts`
- Wrap in `<nav aria-label="World books list">` landmark

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookListPanel.test.tsx`
Expected: PASS (7 tests).

**Step 5: Run existing WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All existing tests pass.

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookListPanel.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookListPanel.test.tsx
git commit -m "feat(world-books): add WorldBookListPanel with simplified columns and overflow menu"
```

---

## Task 5: WorldBookDetailPanel — Tabbed Detail View

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx`
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx` (entries tab content)
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookForm.tsx` (settings tab content)
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookBudgetBar.tsx` (Task 2)

This is the most complex new component. It replaces the entries Drawer, edit Modal, stats Modal, and quick-attach Modal with a tabbed panel.

**Step 1: Write the failing test**

Create `__tests__/WorldBookDetailPanel.test.tsx`:

```tsx
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { WorldBookDetailPanel } from "../WorldBookDetailPanel"

const sampleWorldBook = {
  id: 1,
  name: "Fantasy Lore",
  description: "Core lore for a fantasy world",
  entry_count: 12,
  enabled: true,
  token_budget: 700,
  scan_depth: 3,
  recursive_scanning: false,
  last_modified: new Date().toISOString(),
  version: 1,
}

const defaultProps = {
  worldBook: sampleWorldBook,
  attachedCharacters: [
    { id: 1, name: "Gandalf" },
    { id: 2, name: "Aria" },
  ],
  allWorldBooks: [sampleWorldBook],
  allCharacters: [
    { id: 1, name: "Gandalf" },
    { id: 2, name: "Aria" },
    { id: 3, name: "Mordecai" },
  ],
  onUpdateWorldBook: vi.fn(),
  onDeleteEntry: vi.fn(),
  onAttachCharacter: vi.fn(),
  onDetachCharacter: vi.fn(),
  onOpenTestMatching: vi.fn(),
  maxRecursiveDepth: 10,
  updating: false,
  entryFormInstance: null as any,
}

describe("WorldBookDetailPanel", () => {
  it("renders the summary bar with key metadata", () => {
    render(<WorldBookDetailPanel {...defaultProps} />)
    expect(screen.getByText("Fantasy Lore")).toBeInTheDocument()
    expect(screen.getByText(/12 entries/i)).toBeInTheDocument()
    expect(screen.getByText(/enabled/i)).toBeInTheDocument()
    expect(screen.getByText(/2 characters/i)).toBeInTheDocument()
  })

  it("renders Entries tab as default active tab", () => {
    render(<WorldBookDetailPanel {...defaultProps} />)
    const entriesTab = screen.getByRole("tab", { name: /entries/i })
    expect(entriesTab).toHaveAttribute("aria-selected", "true")
  })

  it("renders all four tabs", () => {
    render(<WorldBookDetailPanel {...defaultProps} />)
    expect(screen.getByRole("tab", { name: /entries/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /attachments/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /stats/i })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: /settings/i })).toBeInTheDocument()
  })

  it("switches to Settings tab on click", async () => {
    const user = userEvent.setup()
    render(<WorldBookDetailPanel {...defaultProps} />)
    await user.click(screen.getByRole("tab", { name: /settings/i }))
    expect(screen.getByRole("tab", { name: /settings/i })).toHaveAttribute("aria-selected", "true")
  })

  it("has correct landmark role", () => {
    render(<WorldBookDetailPanel {...defaultProps} />)
    expect(screen.getByRole("main", { name: /world book detail/i })).toBeInTheDocument()
  })
})

describe("WorldBookDetailPanel — no selection", () => {
  it("renders placeholder when worldBook is null", () => {
    render(<WorldBookDetailPanel {...defaultProps} worldBook={null} />)
    expect(screen.getByText(/select a world book/i)).toBeInTheDocument()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx`
Expected: FAIL — module not found.

**Step 3: Write the component**

Create `WorldBookDetailPanel.tsx`. Requirements:

- Props:
  - `worldBook: any | null` — the selected world book record (null = no selection)
  - `attachedCharacters: any[]` — characters attached to this world book
  - `allWorldBooks: any[]` — for duplicate-name validation in settings tab
  - `allCharacters: any[]` — for attachment tab character selector
  - `onUpdateWorldBook: (values: any) => void` — save settings
  - `onAttachCharacter: (characterId: number) => Promise<void>`
  - `onDetachCharacter: (characterId: number) => Promise<void>`
  - `onOpenTestMatching: (worldBookId?: number) => void`
  - `maxRecursiveDepth: number`
  - `updating: boolean`
  - `entryFormInstance: any` — Ant Design form instance for entries (passed through)
  - `entryFilterPreset?: EntryFilterPreset`

- **No-selection state:** When `worldBook` is null, render a centered placeholder: "Select a world book to view its entries and settings" with a simplified 3-step visual. Wrap in `<main aria-label="World book detail">`.

- **Summary bar:** At top, show: world book name (as heading), enabled/disabled status tag with icon, `{entry_count} entries`, `Budget: {token_budget} tok`, `{attachedCharacters.length} characters`, relative last_modified. Use `text-xs text-text-muted` for metadata.

- **Tabs:** Use Ant Design `<Tabs>` with 4 items:
  1. **Entries** (default): Render `<EntryManager>` from `WorldBookEntryManager.tsx` with `<WorldBookBudgetBar>` above it. Pass through `worldBookId`, `worldBookName`, `tokenBudget`, `worldBooks`, `entryFilterPreset`, `form`.
  2. **Attachments:** List of attached characters with detach buttons + a `<Select>` to attach new characters. Reuse the quick-attach modal content from Manager.tsx lines 2489-2553.
  3. **Stats:** Fetch and display per-book statistics using `tldwClient.worldBookStatistics()`. Reuse the Descriptions layout from Manager.tsx lines 1779-1916.
  4. **Settings:** Render `<WorldBookForm mode="edit">` with the two-tier labels from Design Section 5. Include inline `<WorldBookBudgetBar>`.

- Focus management: The heading (`<h2>`) gets `tabIndex={-1}` and `ref` for programmatic focus when a world book is selected.

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx`
Expected: PASS (6 tests).

**Step 5: Run existing WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All existing tests pass.

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx
git commit -m "feat(world-books): add WorldBookDetailPanel with tabbed entries/attachments/stats/settings"
```

---

## Task 6: Human-Readable Label Mapping

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorldBooks/worldBookLabelUtils.ts`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/worldBookLabelUtils.test.ts`
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookForm.tsx` (apply labels)

**Step 1: Write the failing test**

Create `__tests__/worldBookLabelUtils.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { getSettingLabel, getSettingDescription, getSettingTechnicalNote } from "../worldBookLabelUtils"

describe("worldBookLabelUtils", () => {
  describe("getSettingLabel", () => {
    it("returns friendly label by default", () => {
      expect(getSettingLabel("scan_depth", false)).toBe("Messages to search")
      expect(getSettingLabel("token_budget", false)).toBe("Context size limit")
      expect(getSettingLabel("recursive_scanning", false)).toBe("Chain matching")
    })

    it("returns technical label when showTechnical is true", () => {
      expect(getSettingLabel("scan_depth", true)).toBe("Scan Depth")
      expect(getSettingLabel("token_budget", true)).toBe("Token Budget")
      expect(getSettingLabel("recursive_scanning", true)).toBe("Recursive Scanning")
    })

    it("returns the key as fallback for unknown settings", () => {
      expect(getSettingLabel("unknown_field", false)).toBe("unknown_field")
    })
  })

  describe("getSettingDescription", () => {
    it("returns user-friendly description by default", () => {
      const desc = getSettingDescription("scan_depth", false)
      expect(desc).toMatch(/how far back/i)
    })

    it("returns technical description when showTechnical is true", () => {
      const desc = getSettingDescription("scan_depth", true)
      expect(desc).toMatch(/scan_depth/i)
    })
  })

  describe("getSettingTechnicalNote", () => {
    it("returns API field name and range", () => {
      expect(getSettingTechnicalNote("scan_depth")).toMatch(/1-20/)
      expect(getSettingTechnicalNote("token_budget")).toMatch(/50-5000/)
    })
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/worldBookLabelUtils.test.ts`
Expected: FAIL — module not found.

**Step 3: Write the utility**

Create `worldBookLabelUtils.ts`:

```ts
type SettingKey = "scan_depth" | "token_budget" | "recursive_scanning"

const FRIENDLY_LABELS: Record<SettingKey, string> = {
  scan_depth: "Messages to search",
  token_budget: "Context size limit",
  recursive_scanning: "Chain matching",
}

const TECHNICAL_LABELS: Record<SettingKey, string> = {
  scan_depth: "Scan Depth",
  token_budget: "Token Budget",
  recursive_scanning: "Recursive Scanning",
}

const FRIENDLY_DESCRIPTIONS: Record<SettingKey, string> = {
  scan_depth:
    "How far back in the conversation to look for keyword matches. Higher = more matches found, slower processing.",
  token_budget:
    "Maximum amount of world info added to each response. Higher = more lore available to the AI, but uses more of the conversation window.",
  recursive_scanning:
    "When a matched entry contains keywords from other entries, also include those. Useful for interconnected lore.",
}

const TECHNICAL_DESCRIPTIONS: Record<SettingKey, string> = {
  scan_depth: "scan_depth: 1-20. Number of recent messages to search for keyword matches.",
  token_budget:
    "token_budget: 50-5000 (~4 chars ≈ 1 token). Maximum characters of world info injected into context.",
  recursive_scanning:
    "recursive_scanning: Also search matched content for additional keyword matches. Max depth configurable server-side.",
}

const TECHNICAL_NOTES: Record<SettingKey, string> = {
  scan_depth: "scan_depth: 1-20",
  token_budget: "token_budget: 50-5000 (~4 chars ≈ 1 token)",
  recursive_scanning: "recursive_scanning: max depth configurable",
}

const isSettingKey = (key: string): key is SettingKey =>
  key in FRIENDLY_LABELS

export const getSettingLabel = (key: string, showTechnical: boolean): string => {
  if (!isSettingKey(key)) return key
  return showTechnical ? TECHNICAL_LABELS[key] : FRIENDLY_LABELS[key]
}

export const getSettingDescription = (key: string, showTechnical: boolean): string => {
  if (!isSettingKey(key)) return ""
  return showTechnical ? TECHNICAL_DESCRIPTIONS[key] : FRIENDLY_DESCRIPTIONS[key]
}

export const getSettingTechnicalNote = (key: string): string => {
  if (!isSettingKey(key)) return ""
  return TECHNICAL_NOTES[key]
}
```

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/worldBookLabelUtils.test.ts`
Expected: PASS (7 tests).

**Step 5: Update WorldBookForm.tsx to use label utils**

Modify `WorldBookForm.tsx`:
- Import `getSettingLabel`, `getSettingDescription` from `./worldBookLabelUtils`
- Add a `showTechnicalLabels?: boolean` prop
- Replace hardcoded `"Scan Depth"` with `getSettingLabel("scan_depth", showTechnicalLabels)`
- Replace hardcoded `"Token Budget"` with `getSettingLabel("token_budget", showTechnicalLabels)`
- Replace hardcoded `"Recursive Scanning"` with `getSettingLabel("recursive_scanning", showTechnicalLabels)`
- Update `<LabelWithHelp>` help text to use `getSettingDescription()`
- Rename `<details>` summary from "Advanced Settings" to "Matching & Budget"

**Step 6: Run WorldBookForm tests to verify no regressions**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookForm.test.tsx`
Expected: PASS. If any tests assert on the exact text "Advanced Settings" or "Scan Depth", update those assertions to match the new friendly labels.

**Step 7: Run all WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All pass (update any broken assertions for renamed labels).

**Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/worldBookLabelUtils.ts
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/worldBookLabelUtils.test.ts
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookForm.tsx
git commit -m "feat(world-books): add two-tier label system with friendly defaults and technical toggle"
```

---

## Task 7: Wire Up — Replace Manager.tsx with Orchestrator

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx` (major rewrite)
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBooksWorkspace.tsx` (minor layout update)
- Test: Run all existing 53 test files — they test Manager.tsx behavior

This is the integration task. The existing Manager.tsx becomes a ~200-300 line orchestrator that:
1. Holds all React Query queries and mutations (unchanged)
2. Calls all existing hooks (unchanged)
3. Manages which world book is selected (`selectedWorldBookId` state)
4. Renders the new components instead of inline JSX

**Step 1: Create a backup branch checkpoint**

```bash
git stash  # if any uncommitted changes
git checkout -b world-books-ux/pre-integration
git checkout -  # back to working branch
```

**Step 2: Rewrite Manager.tsx**

Replace the 2,602-line render section with the new component tree. The logic (hooks, queries, mutations, callbacks) stays. The JSX changes to:

```tsx
return (
  <div className="space-y-4" data-testid="world-books-manager">
    <WorldBookToolbar
      listSearch={listSearch}
      onSearchChange={setListSearch}
      enabledFilter={enabledFilter}
      onEnabledFilterChange={setEnabledFilter}
      attachmentFilter={attachmentFilter}
      onAttachmentFilterChange={(value) => {
        if (value !== "all") requestAttachmentHydration()
        setAttachmentFilter(value)
      }}
      onNewWorldBook={() => setOpen(true)}
      onOpenTestMatching={() => openTestMatchingModal()}
      onOpenMatrix={handleOpenMatrix}
      onOpenGlobalStats={() => setOpenGlobalStats(true)}
      onImport={openImportModal}
      onExportAll={() => void exportWorldBookBundle("all")}
      onExportSelected={() => void exportWorldBookBundle("selected")}
      hasWorldBooks={Array.isArray(data) && data.length > 0}
      hasSelection={selectedWorldBookKeys.length > 0}
      globalStatsFetching={openGlobalStats && globalStatsFetching}
      bulkExportAllLoading={bulkExportMode === "all"}
      bulkExportSelectedLoading={bulkExportMode === "selected"}
    />

    {/* Bulk action bar — preserved as-is */}
    {pendingDeleteIds.length > 0 && (/* existing pending-delete banner */)}
    {selectedWorldBookKeys.length > 0 && (/* existing bulk action bar */)}

    {status === "pending" && <Skeleton active paragraph={{ rows: 6 }} />}

    {status === "success" && (
      Array.isArray(data) && data.length === 0 && !hasActiveListFilters ? (
        <WorldBookEmptyState
          onCreateNew={() => setOpen(true)}
          onCreateFromTemplate={(key) => {
            createForm.setFieldsValue({ template_key: key })
            setOpen(true)
          }}
          onImport={openImportModal}
        />
      ) : (
        <div className="flex gap-4">
          {/* Left panel */}
          <div className="w-[35%] min-w-[280px] shrink-0">
            <WorldBookListPanel
              worldBooks={filteredWorldBooks}
              selectedWorldBookId={selectedWorldBookId}
              onSelectWorldBook={setSelectedWorldBookId}
              selectedRowKeys={selectedWorldBookKeys}
              onSelectedRowKeysChange={setSelectedWorldBookKeys}
              pendingDeleteIds={pendingDeleteIds}
              onEditWorldBook={(record) => {
                setSelectedWorldBookId(record.id)
                // Switch detail panel to settings tab
              }}
              onRowAction={handleRowAction}
              tableSort={tableSort}
              onTableSortChange={handleTableSortChange}
              loading={false}
            />
          </div>
          {/* Right panel */}
          <div className="flex-1 min-w-0">
            <WorldBookDetailPanel
              worldBook={selectedWorldBookRecord}
              attachedCharacters={selectedWorldBookAttached}
              allWorldBooks={(data || []) as any[]}
              allCharacters={(characters || []) as any[]}
              onUpdateWorldBook={updateWB}
              onAttachCharacter={/* ... */}
              onDetachCharacter={/* ... */}
              onOpenTestMatching={openTestMatchingModal}
              maxRecursiveDepth={maxRecursiveDepth}
              updating={updating}
              entryFormInstance={entryForm}
              entryFilterPreset={entryFilterPreset}
            />
          </div>
        </div>
      )
    )}

    {/* Modals that stay: Create, Import, Matrix, Test Matching, Global Stats */}
    {/* These are preserved from the current Manager.tsx */}
  </div>
)
```

Key changes in the logic section:
- Add `const [selectedWorldBookId, setSelectedWorldBookId] = React.useState<number | null>(null)`
- Add `selectedWorldBookRecord` memo: find the selected book from `data`
- Add `selectedWorldBookAttached` memo: `getAttachedCharacters(selectedWorldBookId)`
- Add `handleRowAction` callback that dispatches overflow menu actions
- Remove: `openEdit`/`setOpenEdit`, `editId`/`setEditId` state (replaced by detail panel settings tab)
- Remove: `openEntries`/`setOpenEntries` state (replaced by detail panel entries tab)
- Remove: `statsFor`/`setStatsFor` state (replaced by detail panel stats tab)
- Remove: `openAttach`/`setOpenAttach` state (replaced by detail panel attachments tab)
- Keep: `open` (create modal), `openImport`, `openMatrix`, `openTestMatching`, `openGlobalStats`

**Step 3: Update WorldBooksWorkspace.tsx**

The PageShell `maxWidthClassName` should be `"max-w-none"` when the two-panel layout is active (when there are world books), to give the panels room. When empty state is showing, keep the original max-width.

**Step 4: Run ALL WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`

Expected: Many tests will fail because they assert on the old UI structure (modal titles, button labels at specific positions, table column headers). This is expected — the tests were written for the old layout.

**Step 5: Update failing tests**

For each failing test file:
1. Read the test to understand what behavior it's testing
2. If the behavior is preserved (just in a different location), update the selector/assertion
3. If the behavior was removed (e.g., edit modal replaced by settings tab), update the test to verify the new equivalent
4. If the test is no longer relevant, add a comment explaining why and update the assertion

Common updates:
- Tests looking for `"Edit World Book"` modal title → look for Settings tab content instead
- Tests looking for entries Drawer → look for Entries tab content instead
- Tests looking for the stats Modal → look for Stats tab content instead
- Tests looking for 7 action buttons → look for 2 (Edit + overflow)
- Tests looking for the debug banner → look inside Tools dropdown

**Step 6: Verify ALL tests pass**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: ALL tests pass.

**Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBooksWorkspace.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/
git commit -m "feat(world-books): replace monolith Manager with two-panel orchestrator layout"
```

---

## Task 8: Responsive Behavior — Tablet & Mobile Layouts

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx` (breakpoint logic)
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookListPanel.tsx` (collapsible mode)
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx` (back button)
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookToolbar.tsx` (mobile layout)
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBooksManager.responsiveStage1.test.tsx` (update)
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBooksManager.responsiveStage2.test.tsx` (update)

**Step 1: Add responsive logic to Manager.tsx**

Use `Grid.useBreakpoint()` (already imported) to determine layout mode:

```tsx
const screens = Grid.useBreakpoint()
const layoutMode = screens.lg ? "desktop" : screens.md ? "tablet" : "mobile"
```

- **Desktop (`lg+`):** Side-by-side panels as wired in Task 7
- **Tablet (`md`):** Stacked layout with collapsible list. Pass `collapsible={true}` to WorldBookListPanel. List renders as an accordion that collapses when a world book is selected.
- **Mobile (`sm`):** Navigation stack. When `selectedWorldBookId` is null, show only WorldBookListPanel full-width. When a world book is selected, show only WorldBookDetailPanel full-width with a back button.

**Step 2: Add collapsible mode to WorldBookListPanel**

Add prop `collapsible?: boolean`. When true, wrap the table in a `<details>` element that is open when `selectedWorldBookId` is null and closed after selection. The summary shows the selected world book name.

**Step 3: Add back button to WorldBookDetailPanel**

Add prop `onBack?: () => void`. When provided, render a `"← World Books"` button above the summary bar. Clicking it calls `onBack()` which sets `selectedWorldBookId` to null in the parent.

**Step 4: Add mobile toolbar layout to WorldBookToolbar**

Add prop `compact?: boolean`. When true:
- Search goes full-width on its own row
- Filters become a single "Filters" popover button
- Tools remains as dropdown icon button
- "New World Book" becomes a text button (FAB is handled separately by the Manager)

**Step 5: Update responsive test files**

Update `responsiveStage1` and `responsiveStage2` tests to verify:
- Desktop shows side-by-side panels
- Mobile shows only list or only detail (not both)
- Back button appears on mobile detail view
- Toolbar adapts per breakpoint

**Step 6: Run all WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All pass.

**Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookListPanel.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookToolbar.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/
git commit -m "feat(world-books): add responsive tablet/mobile layouts with nav stack and collapsible list"
```

---

## Task 9: Budget Feedback in Entry Creation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookBudgetBar.tsx` (if needed)
- Reference: `apps/packages/ui/src/components/Option/WorldBooks/worldBookEntryUtils.ts` (`estimateEntryTokens`)
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx`

**Step 1: Write the failing test**

Create `__tests__/WorldBookEntryManager.budget.test.tsx`:

```tsx
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

// Test that BudgetBar is rendered in the entries view
// and that the add-entry form shows projected budget
describe("WorldBookEntryManager budget feedback", () => {
  it("renders WorldBookBudgetBar at the top of the entries tab", () => {
    // This test needs the full EntryManager setup with mocked queries.
    // Verify that a role="meter" element exists when tokenBudget is provided.
    // Implementation: render EntryManager with tokenBudget=700, mock entries response
    // with estimatedTokens sum, assert meter is present.
  })

  it("shows per-entry token estimate inline", () => {
    // Verify each entry card shows "~N tokens" text
  })

  it("shows projected budget when adding a new entry", () => {
    // Type content into the add-entry form, verify "After save:" text appears
  })

  it("shows soft warning when projected budget exceeds limit", () => {
    // Type long content, verify warning text appears but Save button is NOT disabled
  })
})
```

Note: The exact test implementation depends on the EntryManager's mock setup pattern. Follow the pattern from `WorldBooksManager.entryStage1.test.tsx` for mocking.

**Step 2: Run test to verify it fails**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx`
Expected: FAIL.

**Step 3: Modify WorldBookEntryManager.tsx**

Key changes:
1. Accept `tokenBudget?: number` prop (already exists, just needs to be used more)
2. Add `<WorldBookBudgetBar>` at the top of the component, above the entry list. Pass `estimatedTokens` computed by summing `estimateEntryTokens()` across all entries.
3. In the add/edit entry form section, compute `projectedTokens = currentTotal + estimateEntryTokens(formContentValue)`. Pass this to a second `<WorldBookBudgetBar projectedTokens={projectedTokens}>`.
4. Show per-entry token estimate: below each entry's content in the list, add `<span className="text-xs text-text-muted">~{estimateEntryTokens(entry)} tokens</span>`
5. Soft warning: when `projectedTokens > tokenBudget`, show a warning div below the budget bar. The Save button remains enabled.

**Step 4: Run test to verify it passes**

Run: `cd apps/packages/ui && npx vitest run src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx`
Expected: PASS.

**Step 5: Run all WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All pass.

**Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx
git commit -m "feat(world-books): add live budget feedback and per-entry token estimates in entry manager"
```

---

## Task 10: Accessibility Polish

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookListPanel.tsx` (aria-labels, status icons)
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx` (focus management, landmarks)
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx` (keyboard shortcuts, Escape handler)
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBooksManager.accessibilityStage1.test.tsx` (update)

**Step 1: Add status icons alongside color tags**

In `WorldBookListPanel.tsx`, update the Enabled/Disabled column renderer:
- Enabled: `<Tag color="green"><CircleCheck className="w-3 h-3 inline mr-1" />Enabled</Tag>`
- Disabled: `<Tag color="volcano"><CirclePause className="w-3 h-3 inline mr-1" />Disabled</Tag>`

Import `CircleCheck`, `CirclePause` from `lucide-react`.

**Step 2: Add specific aria-labels to action buttons**

In `WorldBookListPanel.tsx`, update Edit button:
```tsx
aria-label={`Edit ${record?.name || "world book"}`}
```

Update overflow menu button:
```tsx
aria-label={`More actions for ${record?.name || "world book"}`}
```

**Step 3: Add focus management to WorldBookDetailPanel**

- Add `const headingRef = React.useRef<HTMLHeadingElement>(null)` 
- On the `<h2>` element: `ref={headingRef} tabIndex={-1}`
- Expose an `imperativeRef` or use `useEffect` to focus the heading when `worldBook` changes from null to a value (or when `worldBook.id` changes)

**Step 4: Add Escape key handler to Manager.tsx**

In the Manager orchestrator, add a `useEffect` for keyboard:
```tsx
React.useEffect(() => {
  const handler = (e: KeyboardEvent) => {
    // Only when no modal/drawer is open and focus is in the detail panel
    if (e.key === "Escape" && selectedWorldBookId != null && !open && !openImport && !openMatrix) {
      setSelectedWorldBookId(null)
    }
  }
  document.addEventListener("keydown", handler)
  return () => document.removeEventListener("keydown", handler)
}, [selectedWorldBookId, open, openImport, openMatrix])
```

**Step 5: Add reduced motion support**

In Manager.tsx and WorldBookListPanel.tsx, any `animate-pulse` classes should be wrapped:
```tsx
className={`... ${pulse && !prefersReducedMotion ? "animate-pulse" : pulse ? "ring-2 ring-blue-400" : ""}`}
```

Add `const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches` as a hook or inline check.

**Step 6: Update accessibility test files**

Update `accessibilityStage1-4` tests to verify:
- Status tags have icons (check for `CircleCheck`/`CirclePause` in rendered output)
- Action buttons have specific aria-labels containing the world book name
- Detail panel heading receives focus on selection
- Escape key clears selection

**Step 7: Run all WorldBooks tests**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All pass.

**Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookListPanel.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx
git add apps/packages/ui/src/components/Option/WorldBooks/__tests__/
git commit -m "feat(world-books): add a11y polish — status icons, specific aria-labels, focus management, reduced motion"
```

---

## Task 11: E2E Smoke Test Update

**Files:**
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/WorldBooksPage.ts` (update selectors)
- Modify: `apps/tldw-frontend/e2e/workflows/world-books.spec.ts` (update flow)

**Step 1: Update WorldBooksPage page object selectors**

The page object needs to reflect the new UI structure:
- Table selectors: update if column structure changed
- Action buttons: now Edit + overflow menu instead of 7 buttons
- Entries: now in the detail panel tab instead of a drawer
- Edit: now in the detail panel settings tab instead of a modal

Read the existing page object, update selectors to match new `data-testid` attributes.

**Step 2: Update E2E workflow**

The create → edit → delete flow should now:
1. Click "New World Book" (unchanged — still primary button)
2. Fill form and submit (unchanged — still a modal)
3. Click the created world book in the list (new — selects it in list panel)
4. Verify detail panel shows entries tab (new)
5. Switch to Settings tab to edit (new — instead of edit modal)
6. Delete via overflow menu (new — instead of direct delete button)

**Step 3: Run E2E tests**

Run: `cd apps/tldw-frontend && npm run e2e:pw -- --grep "world-books"`
Expected: PASS (may need adjustments based on server availability).

**Step 4: Commit**

```bash
git add apps/tldw-frontend/e2e/
git commit -m "test(world-books): update e2e page object and workflow for two-panel layout"
```

---

## Task 12: Final Cleanup & Documentation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/Manager.tsx` (remove dead code)
- Modify: `Docs/Plans/2026-04-09-world-books-ux-progressive-disclosure-design.md` (mark complete)

**Step 1: Remove dead code from Manager.tsx**

After the rewrite, scan for:
- Unused imports (old modal state variables, old render functions)
- Functions that were inlined into new components
- Commented-out code

Run: `npx tsc --noEmit` from `apps/packages/ui/` to catch any type errors.

**Step 2: Run full test suite one last time**

Run: `cd apps/packages/ui && npm run test:worldbooks`
Expected: All tests pass.

**Step 3: Run TypeScript check**

Run: `cd apps/packages/ui && npx tsc --noEmit`
Expected: No errors.

**Step 4: Update design doc status**

Add to the top of `Docs/Plans/2026-04-09-world-books-ux-progressive-disclosure-design.md`:

```markdown
**Status:** Implementation complete
**Implementation plan:** `Docs/Plans/2026-04-09-world-books-ux-progressive-disclosure-implementation-plan.md`
```

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/WorldBooks/
git add Docs/Plans/
git commit -m "chore(world-books): clean up dead code and mark design doc as complete"
```

---

## Summary of New Files Created

| File | Task | Purpose |
|------|------|---------|
| `WorldBookEmptyState.tsx` | 1 | First-run 3-step onboarding |
| `WorldBookBudgetBar.tsx` | 2 | Reusable token budget indicator |
| `WorldBookToolbar.tsx` | 3 | Reorganized toolbar with Tools dropdown |
| `WorldBookListPanel.tsx` | 4 | Simplified table with overflow actions |
| `WorldBookDetailPanel.tsx` | 5 | Tabbed detail view (entries/attach/stats/settings) |
| `worldBookLabelUtils.ts` | 6 | Two-tier label mapping |

## Summary of Modified Files

| File | Tasks | Changes |
|------|-------|---------|
| `Manager.tsx` | 7,8,10 | Rewritten from 2,602-line monolith to ~300-line orchestrator |
| `WorldBooksWorkspace.tsx` | 7 | Updated max-width for two-panel layout |
| `WorldBookForm.tsx` | 6 | Two-tier labels via `worldBookLabelUtils` |
| `WorldBookEntryManager.tsx` | 9 | Budget bar, per-entry token estimates, projected budget |
| 53 existing test files | 7,8,10 | Updated selectors/assertions for new layout |
| E2E page object + spec | 11 | Updated for two-panel workflow |

## Dependency Order

```
Task 1 (EmptyState) ──────────────────────┐
Task 2 (BudgetBar) ───────────────────────┤
Task 3 (Toolbar) ─────────────────────────┤
Task 4 (ListPanel) ───────────────────────┼──> Task 7 (Wire Up) ──> Task 8 (Responsive)
Task 5 (DetailPanel) ─────────────────────┤                              │
Task 6 (Labels) ──────────────────────────┘                              v
                                                               Task 9 (Budget in Entries)
                                                                         │
                                                                         v
                                                               Task 10 (A11y Polish)
                                                                         │
                                                                         v
                                                               Task 11 (E2E Update)
                                                                         │
                                                                         v
                                                               Task 12 (Cleanup)
```

Tasks 1-6 can be executed in parallel (no dependencies between them).
Tasks 7-12 must be sequential.
