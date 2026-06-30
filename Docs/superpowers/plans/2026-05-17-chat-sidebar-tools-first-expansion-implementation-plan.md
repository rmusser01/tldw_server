# Chat Sidebar Tools-First Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared chat sidebar open with tools/shortcuts expanded and recent conversations collapsed every time it is opened or foregrounded.

**Architecture:** Keep sidebar presentation state in `ChatSidebar`, with parent layouts passing only a monotonic open-reset signal. Use a local `recentCollapsed` state, derive one `recentHistoryVisible` boolean, and gate history rendering, coordinator visibility, selection controls, and server overview queries from that boolean.

**Tech Stack:** React, TypeScript, Zustand coordinator store, `@plasmohq/storage` settings hooks, Vitest, React Testing Library.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md`
- Backlog design task: `TASK-401`
- Backlog planning task: `TASK-404`

## File Structure

Modify:

- `apps/packages/ui/src/components/Common/ChatSidebar.tsx`
  - Owns `recentCollapsed`, `recentHistoryVisible`, reset-on-open behavior, recent disclosure rendering, and selection-control gating.
- `apps/packages/ui/src/components/Layouts/Layout.tsx`
  - Passes `openResetKey` to shared `ChatSidebar` for desktop, mobile drawer, and `tldw:open-chat-sidebar` opens.
- `apps/tldw-frontend/components/layout/WebLayout.tsx`
  - Mirrors the shared layout reset signal for the Next.js shell.
- `apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx`
  - Updates lazy history expectations for collapsed recent history and manual recent expansion.
- `apps/packages/ui/src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx`
  - Updates coordinator visibility assertions to use recent-history visibility, not merely sidebar expansion.
- `apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx`
  - Updates the `ChatSidebar` mock if needed to capture/assert the new `openResetKey` prop.

Create:

- `apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx`
  - Focused component tests for tools-first open/reset, recent disclosure, search reachability, selection reset, and `openResetKey`.
- `apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts`
  - Lightweight source guard or focused layout test proving the shared layout passes `openResetKey` into both desktop and mobile `ChatSidebar` mounts.

Do not modify:

- `apps/packages/ui/src/components/Sidepanel/Chat/Sidebar.tsx`
  - This is the separate sidepanel active-chat/history drawer and is out of scope unless a later task confirms it needs the same shortcuts/recent disclosure model.
- `apps/packages/ui/src/services/settings/ui-settings.ts`
  - V1 should not persist `recentCollapsed`.

## Task 1: Add Tests For Tools-First Sidebar Reset

**Files:**

- Create: `apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx`
- Modify later: `apps/packages/ui/src/components/Common/ChatSidebar.tsx`

- [ ] **Step 1: Write the component test harness**

Create a local mutable `useSetting` mock so tests can observe shortcut-collapse writes and rerender after state changes.

```tsx
// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ChatSidebar } from "../../ChatSidebar"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

const settingState = vi.hoisted(() => ({
  activeTab: "server",
  shortcutsCollapsed: true,
  shortcutSelection: ["quick-ingest", "chat"],
  setActiveTab: vi.fn((next: string) => {
    settingState.activeTab = next
  }),
  setShortcutsCollapsed: vi.fn(async (next: boolean) => {
    settingState.shortcutsCollapsed = next
  }),
  setShortcutSelection: vi.fn()
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { key?: string }) => {
    if (setting.key === "tldw:sidebar:activeTab") {
      return [settingState.activeTab, settingState.setActiveTab]
    }
    if (setting.key === "tldw:sidebar:shortcutsCollapsed") {
      return [
        settingState.shortcutsCollapsed,
        settingState.setShortcutsCollapsed
      ]
    }
    if (setting.key === "tldw:sidebar:shortcutSelection") {
      return [settingState.shortcutSelection, settingState.setShortcutSelection]
    }
    return [null, vi.fn()]
  }
}))
```

Also mock the same dependencies used by existing `ChatSidebar` tests:

```tsx
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@/hooks/useDebounce", () => ({
  useDebounce: <T,>(value: T) => value
}))

vi.mock("@/hooks/useServerChatHistory", () => ({
  SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE: 25,
  useServerChatHistory: () => ({ data: [], total: 0 })
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => vi.fn()
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector?: (state: { temporaryChat: boolean }) => unknown) =>
    typeof selector === "function" ? selector({ temporaryChat: false }) : { temporaryChat: false }
}))

vi.mock("@/store/folder", () => ({
  useFolderStore: (selector?: (state: { conversationKeywordLinks: never[] }) => unknown) =>
    typeof selector === "function"
      ? selector({ conversationKeywordLinks: [] })
      : { conversationKeywordLinks: [] }
}))

vi.mock("@/store/route-transition", () => ({
  useRouteTransitionStore: (selector?: (state: { start: ReturnType<typeof vi.fn> }) => unknown) =>
    typeof selector === "function" ? selector({ start: vi.fn() }) : { start: vi.fn() }
}))

vi.mock("../ServerChatList", () => ({
  ServerChatList: () => <div data-testid="server-chat-list" />
}))

vi.mock("../FolderChatList", () => ({
  FolderChatList: () => <div data-testid="folder-chat-list" />
}))

vi.mock("../../QuickChatHelper", () => ({
  QuickChatHelperButton: () => null
}))

vi.mock("../../NotesDock", () => ({
  NotesDockButton: () => null
}))

vi.mock("@/components/Sidepanel/Chat/ModeToggle", () => ({
  ModeToggle: () => null
}))
```

- [ ] **Step 2: Add reset-on-expand and direct-mount tests**

```tsx
const renderSidebar = (props?: Partial<React.ComponentProps<typeof ChatSidebar>>) =>
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} {...props} />
    </MemoryRouter>
  )

beforeEach(() => {
  settingState.activeTab = "server"
  settingState.shortcutsCollapsed = true
  settingState.shortcutSelection = ["quick-ingest", "chat"]
  settingState.setShortcutsCollapsed.mockClear()
  useChatSurfaceCoordinatorStore.setState({
    routeId: null,
    surface: null,
    visiblePanels: {
      "server-history": false,
      "mcp-tools": false,
      "audio-health": false,
      "model-catalog": false
    },
    engagedPanels: {
      "server-history": false,
      "mcp-tools": false,
      "audio-health": false,
      "model-catalog": false
    }
  })
})

it("direct expanded mount opens shortcuts and keeps recent conversations collapsed", async () => {
  const { rerender } = renderSidebar()

  await waitFor(() => {
    expect(settingState.setShortcutsCollapsed).toHaveBeenCalledWith(false)
  })
  rerender(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )
  expect(screen.getByRole("button", { name: /Shortcuts/i })).toHaveAttribute(
    "aria-expanded",
    "true"
  )
  expect(
    screen.getByRole("button", { name: /Recent conversations/i })
  ).toHaveAttribute("aria-expanded", "false")
  expect(screen.getByTestId("chat-sidebar-shortcut-quick-ingest")).toBeInTheDocument()
  expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
})

it("resets to tools-first when collapsed sidebar is expanded", async () => {
  const { rerender } = render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed />
    </MemoryRouter>
  )

  rerender(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )

  await waitFor(() => {
    expect(settingState.setShortcutsCollapsed).toHaveBeenCalledWith(false)
  })
  rerender(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )
  expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
})
```

- [ ] **Step 3: Run the new test to verify it fails**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
```

Expected: FAIL because `Recent conversations` disclosure and reset behavior do not exist yet.

- [ ] **Step 4: Implement the minimal reset API and local recent state**

In `apps/packages/ui/src/components/Common/ChatSidebar.tsx`, extend props and derive reset state:

```tsx
interface ChatSidebarProps {
  collapsed?: boolean
  onToggleCollapse?: () => void
  className?: string
  openResetKey?: number
}
```

Inside `ChatSidebar`:

```tsx
const [recentCollapsed, setRecentCollapsed] = useState(true)
const hasSearchQuery = normalizedSearchQuery.length > 0
const recentHistoryVisible = !recentCollapsed || hasSearchQuery
const previousCollapsedRef = React.useRef<boolean | null>(null)
const previousOpenResetKeyRef = React.useRef(openResetKey)

const resetToolsFirst = React.useCallback(() => {
  if (shortcutsCollapsed === true) {
    void setShortcutsCollapsed(false)
  }
  setRecentCollapsed(true)
  setSelectionMode(false)
}, [setShortcutsCollapsed, shortcutsCollapsed])

React.useEffect(() => {
  const wasCollapsed = previousCollapsedRef.current
  if (!collapsed && (wasCollapsed === null || wasCollapsed === true)) {
    resetToolsFirst()
  }
  previousCollapsedRef.current = collapsed
}, [collapsed, resetToolsFirst])

React.useEffect(() => {
  if (previousOpenResetKeyRef.current === openResetKey) return
  previousOpenResetKeyRef.current = openResetKey
  if (!collapsed) {
    resetToolsFirst()
  }
}, [collapsed, openResetKey, resetToolsFirst])
```

- [ ] **Step 5: Add the recent disclosure shell**

Add a disclosure button before the search/tabs/list block:

```tsx
const showRecentBody = recentHistoryVisible
const recentConversationsExpanded = showRecentBody

const toggleRecentConversations = React.useCallback(() => {
  setRecentCollapsed((prev) => {
    const next = !prev
    if (next) {
      setSelectionMode(false)
    } else if (currentTab === "server") {
      markPanelEngaged("server-history")
    }
    return next
  })
}, [currentTab, markPanelEngaged])
```

Render:

```tsx
<button
  type="button"
  aria-expanded={recentConversationsExpanded}
  aria-controls="chat-sidebar-recent-conversations"
  onClick={toggleRecentConversations}
  className={cn(
    "group flex w-full items-center justify-between px-3 py-2 text-left hover:bg-surface",
    focusRingClasses
  )}
>
  <span className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
    {t("common:chatSidebar.recentConversations", "Recent conversations")}
  </span>
  <ChevronDown
    className={cn(
      "size-4 text-text-muted transition-transform group-hover:text-text",
      recentConversationsExpanded ? "rotate-0" : "-rotate-90"
    )}
  />
</button>
```

Wrap search/tabs/list with:

```tsx
{showRecentBody && (
  <div id="chat-sidebar-recent-conversations">
    {/* existing search, tabs, and list content */}
  </div>
)}
```

- [ ] **Step 6: Run the new test to verify it passes**

Run:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
```

Expected: PASS for the first two tests.

- [ ] **Step 7: Commit Task 1**

```bash
git add apps/packages/ui/src/components/Common/ChatSidebar.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
git commit -m "feat: reset chat sidebar to tools first"
```

## Task 2: Gate Recent Conversations, Search, And Selection Controls

**Files:**

- Modify: `apps/packages/ui/src/components/Common/ChatSidebar.tsx`
- Modify: `apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx`

- [ ] **Step 1: Add tests for manual expansion and selection reset**

Append tests:

```tsx
it("shows recent conversation controls when the user expands recent conversations", () => {
  renderSidebar()

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

  expect(screen.getByTestId("chat-sidebar-search")).toBeInTheDocument()
  expect(screen.getByLabelText("Chat view")).toBeInTheDocument()
  expect(screen.getByTestId("server-chat-list")).toBeInTheDocument()
})

it("hides server selection controls while recent conversations are collapsed", () => {
  renderSidebar()

  expect(
    screen.queryByRole("button", { name: /Select chats/i })
  ).not.toBeInTheDocument()

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

  expect(screen.getByRole("button", { name: /Select chats/i })).toBeInTheDocument()
})

it("exits selection mode when recent conversations collapse", () => {
  renderSidebar()

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
  fireEvent.click(screen.getByRole("button", { name: /Select chats/i }))
  expect(screen.getByRole("button", { name: /Exit selection/i })).toBeInTheDocument()

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

  expect(screen.getByRole("button", { name: /Select chats/i })).toBeInTheDocument()
})
```

- [ ] **Step 2: Add a test for search reachability**

The user can expand Recent conversations to start a search. If a query is active
and an open reset occurs, the search/results region must stay reachable.

```tsx
it("keeps search controls reachable when a query is active across reset", () => {
  const { rerender } = renderSidebar({ openResetKey: 1 })

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
  fireEvent.change(screen.getByTestId("chat-sidebar-search"), {
    target: { value: "alpha" }
  })

  rerender(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} openResetKey={2} />
    </MemoryRouter>
  )

  expect(screen.getByTestId("chat-sidebar-search")).toBeInTheDocument()
  expect(screen.getByTestId("server-chat-list")).toBeInTheDocument()
})
```

- [ ] **Step 3: Run tests to verify failures**

Run:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
```

Expected: FAIL until selection controls and search reset behavior are fully wired.

- [ ] **Step 4: Scope selection controls to recent visibility**

In `ChatSidebar.tsx`, change the header selection button condition from:

```tsx
{currentTab === "server" && (
```

to:

```tsx
{recentHistoryVisible && currentTab === "server" && (
```

Ensure collapsing recent conversations calls `setSelectionMode(false)`.

- [ ] **Step 5: Make active search override recent collapse**

Keep the body rendered when `hasSearchQuery` is true:

```tsx
const recentHistoryVisible = !recentCollapsed || hasSearchQuery
const recentConversationsExpanded = recentHistoryVisible
```

Leave `searchQuery` untouched in `resetToolsFirst`.

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add apps/packages/ui/src/components/Common/ChatSidebar.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
git commit -m "fix: gate chat history controls behind recent disclosure"
```

## Task 3: Preserve Lazy History Loading

**Files:**

- Modify: `apps/packages/ui/src/components/Common/ChatSidebar.tsx`
- Modify: `apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx`

- [ ] **Step 1: Update lazy-history tests for collapsed recent history**

In `ChatSidebar.lazy-history.test.tsx`, replace the existing expectation with
three scenarios:

```tsx
it("keeps server history overview disabled while recent conversations are collapsed", () => {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )

  expect(useServerChatHistoryMock).toHaveBeenCalledWith(
    "",
    expect.objectContaining({
      enabled: false,
      mode: "overview"
    })
  )
})

it("enables server history overview after the user expands recent conversations", async () => {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

  await waitFor(() => {
    expect(useServerChatHistoryMock).toHaveBeenLastCalledWith(
      "",
      expect.objectContaining({
        enabled: true,
        mode: "overview"
      })
    )
  })
})

it("does not fetch a server count badge on first expanded mount", () => {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )

  expect(useServerChatHistoryMock).toHaveBeenCalledTimes(1)
  expect(useServerChatHistoryMock).toHaveBeenLastCalledWith(
    "",
    expect.objectContaining({ enabled: false })
  )
})
```

Add imports if missing:

```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
```

- [ ] **Step 2: Update coordinator visibility test**

In `ChatSidebar.coordinator.test.tsx`, rename the current test and assert
visibility remains false until recent is expanded:

```tsx
it("marks server history visible only when recent conversations are expanded", () => {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} />
    </MemoryRouter>
  )

  expect(
    useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
  ).toBe(false)

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))

  expect(
    useChatSurfaceCoordinatorStore.getState().visiblePanels["server-history"]
  ).toBe(true)
})
```

- [ ] **Step 3: Run tests to verify failures**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx
```

Expected: FAIL until coordinator and query gating use `recentHistoryVisible`.

- [ ] **Step 4: Gate coordinator visibility and overview count query**

In `ChatSidebar.tsx`, update the server count query:

```tsx
const serverHistoryPanelVisible = recentHistoryVisible && currentTab === "server"
const serverHistoryOverviewEnabled = useChatSurfaceCoordinatorStore(
  (state) => shouldEnableOptionalResource(state, "server-history")
)

const { total: serverChatCount = 0 } = useServerChatHistory("", {
  enabled: serverHistoryPanelVisible && serverHistoryOverviewEnabled,
  mode: "overview",
  page: 1,
  limit: SERVER_CHAT_HISTORY_OVERVIEW_PAGE_SIZE,
  filterMode: "all"
})
```

Update the panel visibility effect:

```tsx
React.useEffect(() => {
  setPanelVisible("server-history", serverHistoryPanelVisible)

  return () => {
    setPanelVisible("server-history", false)
  }
}, [serverHistoryPanelVisible, setPanelVisible])
```

When opening recent conversations on the server tab, mark the panel engaged:

```tsx
if (!next && currentTab === "server") {
  markPanelEngaged("server-history")
}
```

Also engage server history if the recent body is already visible and the user
switches from Folders to Server:

```tsx
React.useEffect(() => {
  if (recentHistoryVisible && currentTab === "server") {
    markPanelEngaged("server-history")
  }
}, [currentTab, markPanelEngaged, recentHistoryVisible])
```

When search becomes non-empty, keep the existing `markPanelEngaged` call. This
effect covers tab switching; the search handler covers typing while already on
the server tab.

- [ ] **Step 5: Do not mount `ServerChatList` while hidden**

Render list content only inside the recent body:

```tsx
{recentHistoryVisible && (
  <div id="chat-sidebar-recent-conversations">
    {/* search */}
    {/* tabs */}
    <div className={cn("flex-1 overflow-y-auto", temporaryChat ? "pointer-events-none opacity-50" : "")}>
      {currentTab === "server" && (
        <ServerChatList
          searchQuery={debouncedSearchQuery}
          selectionMode={selectionMode}
        />
      )}
      {currentTab === "folders" && <FolderChatList />}
    </div>
  </div>
)}
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add apps/packages/ui/src/components/Common/ChatSidebar.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx apps/packages/ui/src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx
git commit -m "fix: keep chat history loading behind recent disclosure"
```

## Task 4: Wire Open Reset Signals From Layout Shells

**Files:**

- Modify: `apps/packages/ui/src/components/Layouts/Layout.tsx`
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify: `apps/packages/ui/src/components/Common/ChatSidebar.tsx`
- Create: `apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts`
- Modify: `apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx`

- [ ] **Step 1: Add a ChatSidebar `openResetKey` behavior test**

In `ChatSidebar.tools-first.test.tsx`:

```tsx
it("resets to tools-first when openResetKey changes while already expanded", async () => {
  const { rerender } = renderSidebar({ openResetKey: 1 })

  fireEvent.click(screen.getByRole("button", { name: /Recent conversations/i }))
  expect(screen.getByTestId("server-chat-list")).toBeInTheDocument()

  rerender(
    <MemoryRouter initialEntries={["/chat"]}>
      <ChatSidebar collapsed={false} openResetKey={2} />
    </MemoryRouter>
  )

  await waitFor(() => {
    expect(screen.queryByTestId("server-chat-list")).not.toBeInTheDocument()
  })
  expect(screen.getByRole("button", { name: /Shortcuts/i })).toHaveAttribute(
    "aria-expanded",
    "true"
  )
})
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx
```

Expected: FAIL until `openResetKey` is implemented.

- [ ] **Step 3: Implement `openResetKey` in `ChatSidebar`**

Use the prop and effect from Task 1. Ensure the effect compares previous and
current keys so it does not loop on initial render.

- [ ] **Step 4: Wire shared layout reset key**

In `apps/packages/ui/src/components/Layouts/Layout.tsx`:

```tsx
const [chatSidebarOpenResetKey, setChatSidebarOpenResetKey] = React.useState(0)
const signalChatSidebarOpen = React.useCallback(() => {
  setChatSidebarOpenResetKey((value) => value + 1)
}, [])
```

When opening from the explicit event handler:

```tsx
if (showChatSidebar) {
  signalChatSidebarOpen()
  if (isMobile) {
    setSidebarOpen(true)
    return
  }
  setChatSidebarCollapsed(false)
  return
}
```

When toggling:

```tsx
if (showChatSidebar) {
  if (isMobile) {
    if (!sidebarOpen) signalChatSidebarOpen()
    setSidebarOpen((prev) => !prev)
    return
  }
  if (chatSidebarCollapsed) signalChatSidebarOpen()
  setChatSidebarCollapsed((prev) => !prev)
  return
}
```

Pass the prop in both mounts:

```tsx
<ChatSidebar
  collapsed={chatSidebarCollapsed}
  openResetKey={chatSidebarOpenResetKey}
  onToggleCollapse={() => {
    if (chatSidebarCollapsed) signalChatSidebarOpen()
    setChatSidebarCollapsed((prev) => !prev)
  }}
  className="sticky top-0 shrink-0 border-r border-border"
/>
```

```tsx
<ChatSidebar
  collapsed={false}
  openResetKey={chatSidebarOpenResetKey}
  onToggleCollapse={() => setSidebarOpen(false)}
/>
```

- [ ] **Step 5: Mirror reset key in Next.js WebLayout**

Apply the same pattern to
`apps/tldw-frontend/components/layout/WebLayout.tsx`. Keep naming identical if
possible: `chatSidebarOpenResetKey` and `signalChatSidebarOpen`.

- [ ] **Step 6: Add layout guard coverage**

If runtime layout tests become too heavy, add a narrow source guard:

```ts
import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

describe("Layout chat sidebar reset signal", () => {
  it("passes openResetKey to desktop and mobile ChatSidebar mounts", () => {
    const source = readFileSync(
      "src/components/Layouts/Layout.tsx",
      "utf8"
    )
    expect(source).toContain("chatSidebarOpenResetKey")
    expect(source.match(/openResetKey=\\{chatSidebarOpenResetKey\\}/g)).toHaveLength(2)
  })
})
```

For `WebLayout.chat-scroll-contract.test.tsx`, update the `ChatSidebar` mock to
capture props and assert the prop exists when the feature flag mock is changed
to `useChatSidebar: () => [true]`.

- [ ] **Step 7: Run layout and sidebar tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
```

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add apps/packages/ui/src/components/Common/ChatSidebar.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx apps/packages/ui/src/components/Layouts/Layout.tsx apps/packages/ui/src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts apps/tldw-frontend/components/layout/WebLayout.tsx apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
git commit -m "fix: reset chat sidebar on explicit open events"
```

## Task 5: Final Verification And Browser Check

**Files:**

- Modify: `backlog/tasks/task-404 - Plan-chat-sidebar-tools-first-expansion-implementation.md`
- Possibly modify: `backlog/tasks/task-401 - Design-chat-sidebar-tools-first-expansion-behavior.md`

- [ ] **Step 1: Run focused package UI tests**

From `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx src/components/Layouts/__tests__/Layout.chat-sidebar-reset-signal.guard.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run focused WebLayout test**

From `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run lint/type-adjacent checks if available and cheap**

Check available scripts first:

```bash
bun run
```

If a targeted typecheck or lint script exists and is already used by this repo,
run the smallest relevant one. Do not introduce new tooling.

- [ ] **Step 4: Run Bandit policy check**

This change is frontend TypeScript only. Record Bandit as not applicable in
`TASK-404` unless Python files were touched unexpectedly.

- [ ] **Step 5: Browser-check if the app runs cleanly**

If the dev server can start from the current checkout without unrelated
blockers:

```bash
bun run dev
```

Use Browser to verify:

- Desktop `/chat` with persistent sidebar enabled:
  - collapsed icon rail expands to shortcuts visible;
  - recent conversations collapsed;
  - expanding recent shows search/tabs/history;
  - collapse/reopen resets to tools-first.
- Mobile/narrow drawer:
  - drawer open resets to tools-first;
  - recent remains manually expandable.

If the server cannot run because of unrelated checkout state, record the blocker
instead of broadening scope.

- [ ] **Step 6: Update Backlog**

Update `TASK-404` with:

- files changed;
- focused tests run and results;
- browser check result or blocker;
- Bandit frontend-only skip;
- final summary.

- [ ] **Step 7: Commit final task/verification updates**

```bash
git add backlog/tasks/task-404\\ -\\ Plan-chat-sidebar-tools-first-expansion-implementation.md
git commit -m "docs: record chat sidebar verification"
```

Only include `TASK-401` if implementation work also updates the design task.
