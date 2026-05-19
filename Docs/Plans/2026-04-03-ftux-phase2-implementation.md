# FTUX Phase 2 Implementation Plan — MCPHub, Media, Quizzes

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the 12 highest-severity FTUX issues across the MCPHub, Media, and Quizzes pages so first-time users can reach productivity without dead ends.

**Architecture:** Each task is self-contained with its own test and commit. Tutorials follow the existing Joyride infrastructure (definition file + registry import + i18n strings + data-testid targets). Empty states and cross-navigation fixes are inline component edits. No new libraries or architectural changes.

**Tech Stack:** React, TypeScript, Ant Design, react-joyride (via existing TutorialRunner), react-i18next, Zustand (tutorial store), localStorage, CustomEvent API.

**Design doc:** `Docs/Plans/2026-04-03-ftux-phase2-mcphub-media-quiz-design.md`

---

## Stage 1: MCPHub FTUX (4 tasks)

### Task 1: Add page subtitle and dismissible explainer to MCPHub

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx`:

```tsx
import { render, screen, fireEvent } from "@testing-library/react"
import { McpHubPage } from "../McpHubPage"

// Mock all child tabs to avoid deep rendering
vi.mock("../PermissionProfilesTab", () => ({ PermissionProfilesTab: () => <div data-testid="mock-profiles" /> }))
vi.mock("../ToolCatalogsTab", () => ({ ToolCatalogsTab: () => <div data-testid="mock-catalog" /> }))
vi.mock("../ExternalServersTab", () => ({ ExternalServersTab: () => <div data-testid="mock-servers" /> }))
vi.mock("../PolicyAssignmentsTab", () => ({ PolicyAssignmentsTab: () => <div data-testid="mock-assignments" /> }))
vi.mock("../PathScopesTab", () => ({ PathScopesTab: () => <div data-testid="mock-scopes" /> }))
vi.mock("../CapabilityMappingsTab", () => ({ CapabilityMappingsTab: () => <div data-testid="mock-capabilities" /> }))
vi.mock("../WorkspaceSetsTab", () => ({ WorkspaceSetsTab: () => <div data-testid="mock-workspaces" /> }))
vi.mock("../SharedWorkspacesTab", () => ({ SharedWorkspacesTab: () => <div data-testid="mock-shared" /> }))
vi.mock("../GovernanceAuditTab", () => ({ GovernanceAuditTab: () => <div data-testid="mock-audit" /> }))
vi.mock("../GovernancePacksTab", () => ({ GovernancePacksTab: () => <div data-testid="mock-packs" /> }))
vi.mock("../ApprovalPoliciesTab", () => ({ ApprovalPoliciesTab: () => <div data-testid="mock-approvals" /> }))

describe("McpHubPage FTUX", () => {
  beforeEach(() => localStorage.clear())

  it("shows subtitle explaining what MCP Hub is", () => {
    render(<McpHubPage />)
    expect(screen.getByText(/Model Context Protocol/i)).toBeInTheDocument()
  })

  it("shows dismissible explainer card on first visit", () => {
    render(<McpHubPage />)
    expect(screen.getByTestId("mcp-hub-explainer")).toBeInTheDocument()
  })

  it("hides explainer after dismiss and persists", () => {
    render(<McpHubPage />)
    fireEvent.click(screen.getByRole("button", { name: /dismiss|got it/i }))
    expect(screen.queryByTestId("mcp-hub-explainer")).not.toBeInTheDocument()
    expect(localStorage.getItem("tldw_mcp_hub_explainer_dismissed")).toBe("true")
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps && npx vitest run packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx`
Expected: FAIL — subtitle and explainer don't exist yet.

**Step 3: Implement**

Edit `McpHubPage.tsx`:

```tsx
import { useRef, useState } from "react"
import { Alert, Tabs, Typography } from "antd"
// ... existing imports ...

const EXPLAINER_DISMISSED_KEY = "tldw_mcp_hub_explainer_dismissed"

export const McpHubPage = () => {
  const [activeTab, setActiveTab] = useState<McpHubGovernanceAuditTabKey>("tool-catalogs")
  // ... existing state ...

  const [explainerDismissed, setExplainerDismissed] = useState(() => {
    try {
      return localStorage.getItem(EXPLAINER_DISMISSED_KEY) === "true"
    } catch {
      return false
    }
  })

  const handleDismissExplainer = () => {
    setExplainerDismissed(true)
    try {
      localStorage.setItem(EXPLAINER_DISMISSED_KEY, "true")
    } catch { /* ignore */ }
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 p-4" data-testid="mcp-hub-shell">
      <div>
        <Typography.Title level={3} style={{ margin: 0 }}>
          MCP Hub
        </Typography.Title>
        <Typography.Text type="secondary">
          Manage external tool servers and governance policies for the Model Context Protocol (MCP).
        </Typography.Text>
      </div>

      {!explainerDismissed && (
        <Alert
          data-testid="mcp-hub-explainer"
          type="info"
          showIcon
          closable
          onClose={handleDismissExplainer}
          message="What is MCP Hub?"
          description="MCP lets AI assistants use external tools (web search, code execution, etc.). This page controls which tools are available, who can use them, and what permissions they have. Start with the Tool Catalog to see available tools, then add external servers under Servers & Credentials."
        />
      )}

      <Tabs
        data-testid="mcp-hub-tabs"
        activeKey={activeTab}
        // ... rest unchanged ...
```

Key changes:
- Default tab changed from `"profiles"` to `"tool-catalogs"`
- Added subtitle under title
- Added dismissible explainer Alert with data-testid
- Added data-testid to shell and tabs

**Step 4: Run tests**

Run: `cd apps && npx vitest run packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx`
Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx
git commit -m "feat(mcp-hub): add page subtitle, explainer card, default to Tool Catalog tab"
```

---

### Task 2: Create MCPHub Joyride tutorial

**Files:**
- Create: `apps/packages/ui/src/tutorials/definitions/mcp-hub.ts`
- Modify: `apps/packages/ui/src/tutorials/registry.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/tutorials.json`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx` (add data-testid to tab items)

**Step 1: Write the test**

Create `apps/packages/ui/src/tutorials/__tests__/mcp-hub-tutorial.test.ts`:

```ts
import { getTutorialsForRoute, getTutorialById } from "../registry"

describe("MCPHub tutorial registration", () => {
  it("is registered for /mcp-hub route", () => {
    const tutorials = getTutorialsForRoute("/mcp-hub")
    expect(tutorials.length).toBeGreaterThanOrEqual(1)
    expect(tutorials[0].id).toBe("mcp-hub-basics")
  })

  it("has at least 4 steps", () => {
    const tutorial = getTutorialById("mcp-hub-basics")
    expect(tutorial).toBeDefined()
    expect(tutorial!.steps.length).toBeGreaterThanOrEqual(4)
  })

  it("every step has a valid target selector", () => {
    const tutorial = getTutorialById("mcp-hub-basics")!
    for (const step of tutorial.steps) {
      expect(step.target).toMatch(/\[data-testid=/)
    }
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps && npx vitest run packages/ui/src/tutorials/__tests__/mcp-hub-tutorial.test.ts`
Expected: FAIL — no tutorial registered for /mcp-hub.

**Step 3: Create tutorial definition**

Create `apps/packages/ui/src/tutorials/definitions/mcp-hub.ts`:

```ts
/**
 * MCP Hub Tutorial Definitions
 */

import { Plug } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const mcpHubBasics: TutorialDefinition = {
  id: "mcp-hub-basics",
  routePattern: "/mcp-hub",
  labelKey: "tutorials:mcpHub.basics.label",
  labelFallback: "MCP Hub Basics",
  descriptionKey: "tutorials:mcpHub.basics.description",
  descriptionFallback:
    "Learn how to browse tools, connect servers, and manage permissions.",
  icon: Plug,
  priority: 1,
  steps: [
    {
      target: '[data-testid="mcp-hub-shell"]',
      titleKey: "tutorials:mcpHub.basics.welcomeTitle",
      titleFallback: "Welcome to MCP Hub",
      contentKey: "tutorials:mcpHub.basics.welcomeContent",
      contentFallback:
        "MCP Hub manages external AI tools. Start here to see what tools are available and connect new servers.",
      placement: "center",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-tab-tool-catalogs"]',
      titleKey: "tutorials:mcpHub.basics.catalogTitle",
      titleFallback: "Tool Catalog",
      contentKey: "tutorials:mcpHub.basics.catalogContent",
      contentFallback:
        "Browse all registered tools here. Each tool shows its capabilities and risk level.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-tab-credentials"]',
      titleKey: "tutorials:mcpHub.basics.credentialsTitle",
      titleFallback: "Servers & Credentials",
      contentKey: "tutorials:mcpHub.basics.credentialsContent",
      contentFallback:
        "Connect external MCP servers here. Each server provides additional tools for your AI assistant.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-tab-profiles"]',
      titleKey: "tutorials:mcpHub.basics.profilesTitle",
      titleFallback: "Permission Profiles",
      contentKey: "tutorials:mcpHub.basics.profilesContent",
      contentFallback:
        "Create permission profiles to control which tools each user or persona can access.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="mcp-hub-tab-audit"]',
      titleKey: "tutorials:mcpHub.basics.auditTitle",
      titleFallback: "Governance Audit",
      contentKey: "tutorials:mcpHub.basics.auditContent",
      contentFallback:
        "Review policy findings and configuration issues across all your MCP Hub settings.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const mcpHubTutorials: TutorialDefinition[] = [mcpHubBasics]
```

**Step 4: Register in registry**

In `apps/packages/ui/src/tutorials/registry.ts`, add import and spread:

```ts
import { mcpHubTutorials } from "./definitions/mcp-hub"

export const TUTORIAL_REGISTRY: TutorialDefinition[] = [
  ...gettingStartedTutorials,
  // ... existing ...
  ...moderationTutorials,
  ...mcpHubTutorials
]
```

**Step 5: Add i18n strings**

In `apps/packages/ui/src/assets/locale/en/tutorials.json`, add `mcpHub` section:

```json
"mcpHub": {
  "basics": {
    "label": "MCP Hub Basics",
    "description": "Learn how to browse tools, connect servers, and manage permissions.",
    "welcomeTitle": "Welcome to MCP Hub",
    "welcomeContent": "MCP Hub manages external AI tools. Start here to see what tools are available and connect new servers.",
    "catalogTitle": "Tool Catalog",
    "catalogContent": "Browse all registered tools here. Each tool shows its capabilities and risk level.",
    "credentialsTitle": "Servers & Credentials",
    "credentialsContent": "Connect external MCP servers here. Each server provides additional tools for your AI assistant.",
    "profilesTitle": "Permission Profiles",
    "profilesContent": "Create permission profiles to control which tools each user or persona can access.",
    "auditTitle": "Governance Audit",
    "auditContent": "Review policy findings and configuration issues across all your MCP Hub settings."
  }
}
```

**Step 6: Add data-testid to tab items in McpHubPage.tsx**

Each tab item needs a `data-testid` on its label. Wrap each label with a span:

```tsx
items={[
  {
    key: "tool-catalogs",
    label: <span data-testid="mcp-hub-tab-tool-catalogs">Tool Catalog</span>,
    children: <ToolCatalogsTab />
  },
  {
    key: "credentials",
    label: <span data-testid="mcp-hub-tab-credentials">Servers & Credentials</span>,
    children: ( /* ExternalServersTab */ )
  },
  {
    key: "profiles",
    label: <span data-testid="mcp-hub-tab-profiles">Profiles</span>,
    children: <PermissionProfilesTab />
  },
  // ... remaining tabs with data-testid="mcp-hub-tab-{key}" pattern
```

Note: Also rename "Credentials" label to "Servers & Credentials" and "Catalog" to "Tool Catalog" (addresses MH11, MH12).

**Step 7: Run tests**

Run: `cd apps && npx vitest run packages/ui/src/tutorials/__tests__/mcp-hub-tutorial.test.ts`
Expected: PASS

**Step 8: Commit**

```bash
git add apps/packages/ui/src/tutorials/definitions/mcp-hub.ts \
  apps/packages/ui/src/tutorials/registry.ts \
  apps/packages/ui/src/tutorials/__tests__/mcp-hub-tutorial.test.ts \
  apps/packages/ui/src/assets/locale/en/tutorials.json \
  apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
git commit -m "feat(mcp-hub): add Joyride tutorial, rename tab labels for clarity"
```

---

### Task 3: Reorder MCPHub tabs with visual grouping

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`

**Step 1: Write the test**

Add to `McpHubPage.ftux.test.tsx`:

```tsx
it("renders Tool Catalog as the first tab", () => {
  render(<McpHubPage />)
  const tabs = screen.getAllByRole("tab")
  expect(tabs[0]).toHaveTextContent(/Tool Catalog/i)
})

it("renders Servers & Credentials as the second tab", () => {
  render(<McpHubPage />)
  const tabs = screen.getAllByRole("tab")
  expect(tabs[1]).toHaveTextContent(/Servers/i)
})
```

**Step 2: Run test to verify it fails**

Expected: FAIL — current first tab is "Profiles".

**Step 3: Reorder tab items array**

In `McpHubPage.tsx`, reorder the `items` array to group logically:

```
Getting Started: Tool Catalog, Servers & Credentials
Policies:        Profiles, Assignments, Approvals
Advanced:        Path Scopes, Capability Mappings, Workspace Sets, Shared Workspaces, Governance Packs, Audit
```

Move `tool-catalogs` and `credentials` to the front. Keep remaining order for policies then advanced.

**Step 4: Run tests**

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx \
  apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx
git commit -m "feat(mcp-hub): reorder tabs — Tool Catalog and Servers first"
```

---

### Task 4: Review checkpoint — Stage 1

**Step 1: Run all MCPHub tests**

Run: `cd apps && npx vitest run packages/ui/src/components/Option/MCPHub/__tests__/`
Expected: ALL PASS

**Step 2: Run tutorial tests**

Run: `cd apps && npx vitest run packages/ui/src/tutorials/__tests__/`
Expected: ALL PASS

**Step 3: Visual verification**

Open `/mcp-hub` in browser. Verify:
- [ ] Subtitle visible below "MCP Hub" title
- [ ] Blue explainer card visible, dismissible, stays dismissed on refresh
- [ ] Tool Catalog is the first/active tab
- [ ] Servers & Credentials is the second tab
- [ ] Tab labels updated (no more bare "Catalog" or "Credentials")
- [ ] Tutorial prompt appears (toast) offering a guided tour

---

## Stage 2: Media Page FTUX (4 tasks)

### Task 5: Add first-ingest tutorial to primary `/media` ResultsList

**Files:**
- Modify: `apps/packages/ui/src/components/Media/ResultsList.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Media/__tests__/ResultsList.ftux.test.tsx`:

```tsx
import { render, screen, fireEvent } from "@testing-library/react"
import { ResultsList } from "../ResultsList"

// i18n mock
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: { defaultValue?: string }) => opts?.defaultValue ?? key
  })
}))

const emptyProps = {
  results: [],
  selectedId: null,
  onSelect: vi.fn(),
  totalCount: 0,
  loadedCount: 0,
  isLoading: false,
  hasActiveFilters: false,
  searchQuery: "",
  onOpenQuickIngest: vi.fn()
}

describe("ResultsList FTUX", () => {
  beforeEach(() => localStorage.clear())

  it("shows first-ingest tutorial when library is empty", () => {
    render(<ResultsList {...emptyProps} />)
    expect(screen.getByTestId("first-ingest-tutorial")).toBeInTheDocument()
    expect(screen.getByText(/Get started/i)).toBeInTheDocument()
  })

  it("has inline URL input and Ingest button", () => {
    render(<ResultsList {...emptyProps} />)
    expect(screen.getByPlaceholderText(/youtube/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /ingest/i })).toBeInTheDocument()
  })

  it("hides tutorial after dismiss and shows fallback with Quick Ingest button", () => {
    render(<ResultsList {...emptyProps} />)
    fireEvent.click(screen.getByRole("button", { name: /dismiss|skip/i }))
    expect(screen.queryByTestId("first-ingest-tutorial")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /quick ingest/i })).toBeInTheDocument()
  })

  it("does not show tutorial when filters are active", () => {
    render(<ResultsList {...emptyProps} hasActiveFilters />)
    expect(screen.queryByTestId("first-ingest-tutorial")).not.toBeInTheDocument()
  })
})
```

**Step 2: Run test to verify it fails**

Run: `cd apps && npx vitest run packages/ui/src/components/Media/__tests__/ResultsList.ftux.test.tsx`
Expected: FAIL — no `first-ingest-tutorial` testid exists.

**Step 3: Implement first-ingest tutorial in ResultsList.tsx**

Add to `ResultsList.tsx` — in the `results.length === 0 && !isLoading` branch, before the existing empty states, add a first-ingest tutorial check:

```tsx
import { Upload } from "lucide-react"

const FIRST_INGEST_DISMISSED_KEY = "tldw_first_ingest_tutorial_dismissed"

// Inside the component:
const [tutorialDismissed, setTutorialDismissed] = useState(() => {
  try {
    return localStorage.getItem(FIRST_INGEST_DISMISSED_KEY) === "true"
  } catch {
    return false
  }
})

const handleDismissTutorial = useCallback(() => {
  setTutorialDismissed(true)
  try {
    localStorage.setItem(FIRST_INGEST_DISMISSED_KEY, "true")
  } catch { /* ignore */ }
}, [])

const [ingestUrl, setIngestUrl] = useState("")

// In the render — replace the `results.length === 0 && !isLoading` branch:
) : results.length === 0 && !isLoading ? (
  <div className="px-4 py-6 text-center">
    {!hasActiveFilters && !hasSearchQuery && !tutorialDismissed ? (
      <div data-testid="first-ingest-tutorial" className="space-y-3">
        <Upload className="mx-auto h-10 w-10 text-text-subtle" />
        <p className="text-text font-medium">
          {t('mediaPage.firstIngest.title', 'Get started — ingest your first content')}
        </p>
        <p className="text-xs text-text-subtle">
          {t('mediaPage.firstIngest.hint', 'Paste a YouTube URL, or use Quick Ingest for PDFs, audio, EPUBs, web pages, and more.')}
        </p>
        <div className="flex items-center gap-2 max-w-md mx-auto">
          <Input
            placeholder={t('mediaPage.firstIngest.placeholder', 'Paste a YouTube URL...')}
            value={ingestUrl}
            onChange={(e) => setIngestUrl(e.target.value)}
            onPressEnter={() => onOpenQuickIngest?.()}
          />
          <Button type="primary" onClick={() => onOpenQuickIngest?.()}>
            {t('mediaPage.firstIngest.ingestButton', 'Ingest')}
          </Button>
        </div>
        <Button type="link" size="small" onClick={handleDismissTutorial}>
          {t('mediaPage.firstIngest.dismiss', 'Skip for now')}
        </Button>
      </div>
    ) : hasActiveFilters ? (
      // ... existing hasActiveFilters branch ...
    ) : (
      // ... existing no-results branch (with Quick Ingest button) ...
    )}
  </div>
```

Add `Input` to the antd imports.

**Step 4: Run tests**

Run: `cd apps && npx vitest run packages/ui/src/components/Media/__tests__/ResultsList.ftux.test.tsx`
Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Media/ResultsList.tsx \
  apps/packages/ui/src/components/Media/__tests__/ResultsList.ftux.test.tsx
git commit -m "feat(media): add first-ingest tutorial to primary /media page"
```

---

### Task 6: Fix URL passthrough in first-ingest tutorial

**Files:**
- Modify: `apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx`
- Modify: `apps/packages/ui/src/components/Media/ResultsList.tsx` (same fix)

**Step 1: Write the test**

Add to `ResultsList.ftux.test.tsx`:

```tsx
it("passes typed URL to onOpenQuickIngest when clicking Ingest", () => {
  const onOpen = vi.fn()
  render(<ResultsList {...emptyProps} onOpenQuickIngest={onOpen} />)
  const input = screen.getByPlaceholderText(/youtube/i)
  fireEvent.change(input, { target: { value: "https://youtube.com/watch?v=test" } })
  fireEvent.click(screen.getByRole("button", { name: /ingest/i }))
  expect(onOpen).toHaveBeenCalledWith({ source: "https://youtube.com/watch?v=test" })
})
```

**Step 2: Run test to verify it fails**

Expected: FAIL — `onOpenQuickIngest` is called with no arguments currently.

**Step 3: Implement**

In `ResultsList.tsx`, update the Ingest button and Enter handler:

```tsx
<Button type="primary" onClick={() => onOpenQuickIngest?.({ source: ingestUrl.trim() || undefined })}>
```

And update the `onPressEnter`:
```tsx
onPressEnter={() => onOpenQuickIngest?.({ source: ingestUrl.trim() || undefined })}
```

Note: `onOpenQuickIngest` prop type needs to accept optional detail. Update the prop type:

```tsx
onOpenQuickIngest?: (detail?: { source?: string }) => void
```

Similarly fix in `MediaReviewResultsList.tsx` — wire the input value through `requestQuickIngestOpen(inputValue)`. The `requestQuickIngestOpen` function already accepts `detail?: unknown` as its first parameter.

**Step 4: Run tests**

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Media/ResultsList.tsx \
  apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx \
  apps/packages/ui/src/components/Media/__tests__/ResultsList.ftux.test.tsx
git commit -m "fix(media): pass typed URL through to Quick Ingest modal"
```

---

### Task 7: Auto-refresh Media results after Quick Ingest completion

**Files:**
- Modify: `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.ingest-refresh.test.ts`:

```ts
describe("ViewMediaPage ingest auto-refresh", () => {
  it("listens for tldw:quick-ingest-complete event", () => {
    // Verify the event listener is registered
    const addSpy = vi.spyOn(window, "addEventListener")
    // render component (with necessary providers)
    // assert addEventListener was called with "tldw:quick-ingest-complete"
    expect(addSpy).toHaveBeenCalledWith(
      "tldw:quick-ingest-complete",
      expect.any(Function)
    )
    addSpy.mockRestore()
  })
})
```

Note: This test may need to be adapted based on how ViewMediaPage is rendered (it likely needs router + query client providers). If full render is impractical, test the event wiring in isolation.

**Step 2: Run test to verify it fails**

Expected: FAIL — no such event listener exists.

**Step 3: Implement**

In `ViewMediaPage.tsx`, add a `useEffect` that listens for a completion event and triggers refetch:

```tsx
// Near other useEffect hooks:
React.useEffect(() => {
  const handleIngestComplete = () => {
    // Trigger a fresh search after a short delay to let the server index
    setTimeout(() => {
      search.refetch()
    }, 1500)
  }
  window.addEventListener("tldw:quick-ingest-complete", handleIngestComplete)
  return () => window.removeEventListener("tldw:quick-ingest-complete", handleIngestComplete)
}, [search.refetch])
```

Then ensure the Quick Ingest modal dispatches this event on completion. Search for where ingestion results are shown (likely in the QuickIngestModal or ResultsPanel). Add:

```tsx
window.dispatchEvent(new CustomEvent("tldw:quick-ingest-complete"))
```

The event dispatch location needs investigation — look for `data-testid="quick-ingest-complete"` in the codebase which indicates where the success state is rendered. Add the event dispatch there.

**Step 4: Run tests**

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Review/ViewMediaPage.tsx \
  apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.ingest-refresh.test.ts
git commit -m "feat(media): auto-refresh results after Quick Ingest completion"
```

---

### Task 8: Auto-link batch ID from Quick Ingest to jobs panel

**Files:**
- Modify: `apps/packages/ui/src/components/Media/MediaIngestJobsPanel.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Media/__tests__/MediaIngestJobsPanel.autolink.test.ts`:

```ts
describe("MediaIngestJobsPanel auto-link", () => {
  it("picks up batch ID from tldw:quick-ingest-complete event detail", () => {
    // The panel should listen for the completion event and extract the batch ID
    // When a batch ID is received, savedBatchId state should update
    // and the panel should auto-expand
  })
})
```

Note: Exact test structure depends on how the panel renders and what providers it needs. The key behavior: when `tldw:quick-ingest-complete` fires with `detail: { batchId: "abc-123" }`, the panel should auto-set `savedBatchId` and auto-expand.

**Step 2: Implement**

In `MediaIngestJobsPanel.tsx`, add an event listener:

```tsx
React.useEffect(() => {
  const handleIngestComplete = (e: Event) => {
    const detail = (e as CustomEvent)?.detail
    if (detail?.batchId) {
      setBatchDraft(detail.batchId)
      void setSavedBatchId(detail.batchId)
      setPanelCollapsed(false)
    }
  }
  window.addEventListener("tldw:quick-ingest-complete", handleIngestComplete)
  return () => window.removeEventListener("tldw:quick-ingest-complete", handleIngestComplete)
}, [setSavedBatchId])
```

Update the Quick Ingest completion dispatch to include batch ID:

```tsx
window.dispatchEvent(new CustomEvent("tldw:quick-ingest-complete", {
  detail: { batchId: response.batch_id }
}))
```

**Step 3: Run tests**

Expected: PASS

**Step 4: Commit**

```bash
git add apps/packages/ui/src/components/Media/MediaIngestJobsPanel.tsx \
  apps/packages/ui/src/components/Media/__tests__/MediaIngestJobsPanel.autolink.test.ts
git commit -m "feat(media): auto-link batch ID from Quick Ingest to jobs panel"
```

---

### Task 9: Review checkpoint — Stage 2

**Step 1: Run all Media tests**

Run: `cd apps && npx vitest run packages/ui/src/components/Media/__tests__/`
Expected: ALL PASS

**Step 2: Visual verification**

Open `/media` in browser with empty library. Verify:
- [ ] First-ingest tutorial visible with upload icon, title, inline URL input
- [ ] Typing URL and clicking Ingest opens Quick Ingest with URL pre-filled
- [ ] "Skip for now" dismisses tutorial, shows Quick Ingest button
- [ ] After completing an ingest, results auto-refresh
- [ ] Jobs panel auto-expands with batch ID populated

---

## Stage 3: Quizzes Page FTUX (4 tasks)

### Task 10: Create Quiz Joyride tutorial

**Files:**
- Create: `apps/packages/ui/src/tutorials/definitions/quiz.ts`
- Modify: `apps/packages/ui/src/tutorials/registry.ts`
- Modify: `apps/packages/ui/src/assets/locale/en/tutorials.json`
- Modify: `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx` (add data-testid)

**Step 1: Write the test**

Create `apps/packages/ui/src/tutorials/__tests__/quiz-tutorial.test.ts`:

```ts
import { getTutorialsForRoute, getTutorialById } from "../registry"

describe("Quiz tutorial registration", () => {
  it("is registered for /quiz route", () => {
    const tutorials = getTutorialsForRoute("/quiz")
    expect(tutorials.length).toBeGreaterThanOrEqual(1)
    expect(tutorials[0].id).toBe("quiz-basics")
  })

  it("has at least 4 steps", () => {
    const tutorial = getTutorialById("quiz-basics")
    expect(tutorial).toBeDefined()
    expect(tutorial!.steps.length).toBeGreaterThanOrEqual(4)
  })
})
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Create tutorial definition**

Create `apps/packages/ui/src/tutorials/definitions/quiz.ts`:

```ts
/**
 * Quiz Playground Tutorial Definitions
 */

import { GraduationCap } from "lucide-react"
import type { TutorialDefinition } from "../registry"

const quizBasics: TutorialDefinition = {
  id: "quiz-basics",
  routePattern: "/quiz",
  labelKey: "tutorials:quiz.basics.label",
  labelFallback: "Quiz Basics",
  descriptionKey: "tutorials:quiz.basics.description",
  descriptionFallback:
    "Learn how to generate, take, and review quizzes from your media content.",
  icon: GraduationCap,
  priority: 1,
  steps: [
    {
      target: '[data-testid="quiz-playground-tabs"]',
      titleKey: "tutorials:quiz.basics.tabsTitle",
      titleFallback: "Quiz Playground",
      contentKey: "tutorials:quiz.basics.tabsContent",
      contentFallback:
        "The Quiz Playground has five tabs. Generate quizzes from your media, create them manually, take quizzes, manage your library, and review results.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-generate"]',
      titleKey: "tutorials:quiz.basics.generateTitle",
      titleFallback: "Generate from Media",
      contentKey: "tutorials:quiz.basics.generateContent",
      contentFallback:
        "Start here. Select a video, article, or document from your media library, and the AI will generate quiz questions for you.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-create"]',
      titleKey: "tutorials:quiz.basics.createTitle",
      titleFallback: "Create Manually",
      contentKey: "tutorials:quiz.basics.createContent",
      contentFallback:
        "Prefer to write your own? Create quizzes with multiple choice, true/false, fill-in-the-blank, and matching questions.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-take"]',
      titleKey: "tutorials:quiz.basics.takeTitle",
      titleFallback: "Take a Quiz",
      contentKey: "tutorials:quiz.basics.takeContent",
      contentFallback:
        "Once you have quizzes, come here to take them. Your answers are saved and scored automatically.",
      placement: "bottom",
      disableBeacon: true
    },
    {
      target: '[data-testid="quiz-tab-results"]',
      titleKey: "tutorials:quiz.basics.resultsTitle",
      titleFallback: "Review Results",
      contentKey: "tutorials:quiz.basics.resultsContent",
      contentFallback:
        "See your scores, review incorrect answers, and track your progress over time.",
      placement: "bottom",
      disableBeacon: true
    }
  ]
}

export const quizTutorials: TutorialDefinition[] = [quizBasics]
```

**Step 4: Register in registry.ts**

```ts
import { quizTutorials } from "./definitions/quiz"

// In TUTORIAL_REGISTRY:
  ...moderationTutorials,
  ...mcpHubTutorials,
  ...quizTutorials
```

**Step 5: Add i18n strings to tutorials.json**

```json
"quiz": {
  "basics": {
    "label": "Quiz Basics",
    "description": "Learn how to generate, take, and review quizzes from your media content.",
    "tabsTitle": "Quiz Playground",
    "tabsContent": "The Quiz Playground has five tabs. Generate quizzes from your media, create them manually, take quizzes, manage your library, and review results.",
    "generateTitle": "Generate from Media",
    "generateContent": "Start here. Select a video, article, or document from your media library, and the AI will generate quiz questions for you.",
    "createTitle": "Create Manually",
    "createContent": "Prefer to write your own? Create quizzes with multiple choice, true/false, fill-in-the-blank, and matching questions.",
    "takeTitle": "Take a Quiz",
    "takeContent": "Once you have quizzes, come here to take them. Your answers are saved and scored automatically.",
    "resultsTitle": "Review Results",
    "resultsContent": "See your scores, review incorrect answers, and track your progress over time."
  }
}
```

**Step 6: Add data-testid to QuizPlayground.tsx tabs**

In the Tabs `items` array, wrap each label:

```tsx
{
  key: "take",
  label: <span data-testid="quiz-tab-take">{renderTabLabel(...)}</span>,
  // ...
}
```

Add `data-testid="quiz-playground-tabs"` to the `<Tabs>` component.

**Step 7: Run tests**

Expected: PASS

**Step 8: Commit**

```bash
git add apps/packages/ui/src/tutorials/definitions/quiz.ts \
  apps/packages/ui/src/tutorials/registry.ts \
  apps/packages/ui/src/tutorials/__tests__/quiz-tutorial.test.ts \
  apps/packages/ui/src/assets/locale/en/tutorials.json \
  apps/packages/ui/src/components/Quiz/QuizPlayground.tsx
git commit -m "feat(quiz): add Joyride tutorial with 5-step guided tour"
```

---

### Task 11: Default Quiz to Generate tab when no quizzes exist

**Files:**
- Modify: `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.default-tab.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

vi.mock("../hooks", () => ({
  useQuizzesQuery: () => ({ data: { count: 0 } }),
  useAttemptsQuery: () => ({ data: { count: 0 } })
}))

vi.mock("../tabs/TakeQuizTab", () => ({
  TakeQuizTab: () => <div data-testid="take-tab-content">Take Tab</div>
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_k: string, opts?: any) => opts?.defaultValue ?? _k })
}))

describe("QuizPlayground default tab", () => {
  it("defaults to Generate tab when totalQuizzes is 0", () => {
    const qc = new QueryClient()
    render(
      <QueryClientProvider client={qc}>
        <QuizPlayground />
      </QueryClientProvider>
    )
    // The Generate tab should be active
    const generateTab = screen.getByTestId("quiz-tab-generate")
    expect(generateTab.closest(".ant-tabs-tab-active")).toBeTruthy()
  })
})
```

Note: Test structure may need adjustment based on mocking requirements.

**Step 2: Run test to verify it fails**

Expected: FAIL — defaults to "take".

**Step 3: Implement**

In `QuizPlayground.tsx`, change the initial tab logic:

```tsx
const { data: quizCounts } = useQuizzesQuery({ limit: 1, offset: 0 })
const totalQuizzes = quizCounts?.count ?? 0

// After quizCounts loads, switch to generate if empty
const defaultTabResolved = React.useRef(false)
React.useEffect(() => {
  if (defaultTabResolved.current) return
  if (quizCounts === undefined) return  // still loading
  defaultTabResolved.current = true
  if (totalQuizzes === 0 && !initialAssessmentIntent) {
    setActiveTab("generate")
  }
}, [quizCounts, totalQuizzes, initialAssessmentIntent])
```

Keep the initial `useState("take")` so users with quizzes land on Take. The effect overrides to "generate" only for zero-quiz users on first load.

**Step 4: Run tests**

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Quiz/QuizPlayground.tsx \
  apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.default-tab.test.tsx
git commit -m "feat(quiz): default to Generate tab when no quizzes exist"
```

---

### Task 12: Add empty-media alert in GenerateTab with link to /media

**Files:**
- Modify: `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.empty-media.test.tsx`:

```tsx
describe("GenerateTab empty media guidance", () => {
  it("shows alert with link to /media when no media items loaded", () => {
    // Render GenerateTab with empty media list
    // Assert an Alert is visible with text about importing content
    // Assert a link/button to /media exists
  })
})
```

**Step 2: Run test to verify it fails**

Expected: FAIL

**Step 3: Implement**

In `GenerateTab.tsx`, find the source selector section (around line 953). Before the Media Select dropdown, add a conditional alert:

```tsx
{!isLoadingList && loadedMediaItems.length === 0 && (
  <Alert
    type="info"
    showIcon
    data-testid="quiz-generate-no-media"
    message={t("option:quiz.generate.noMedia", {
      defaultValue: "No media content found"
    })}
    description={
      <span>
        {t("option:quiz.generate.noMediaHint", {
          defaultValue: "Import videos, articles, or documents in your"
        })}{" "}
        <a href="/media" onClick={(e) => { e.preventDefault(); navigate("/media") }}>
          {t("option:quiz.generate.mediaLibrary", { defaultValue: "Media Library" })}
        </a>
        {t("option:quiz.generate.noMediaSuffix", {
          defaultValue: ", then return here to generate quizzes from them."
        })}
      </span>
    }
    className="mb-4"
  />
)}
```

Use `useNavigate` from react-router (or the project's routing system) for the navigation.

**Step 4: Run tests**

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.empty-media.test.tsx
git commit -m "feat(quiz): show guidance with link to /media when no media content exists"
```

---

### Task 13: Add navigation CTA in Results tab empty state

**Files:**
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx`

**Step 1: Write the test**

Create `apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.empty-cta.test.tsx`:

```tsx
describe("ResultsTab empty state CTA", () => {
  it("shows 'Take a Quiz' button when no attempts exist", () => {
    // Render ResultsTab with empty attempts
    // Assert a button with text "Take a Quiz" is visible
  })

  it("calls onRetakeQuiz when 'Take a Quiz' is clicked", () => {
    // Render with onRetakeQuiz mock
    // Click the button
    // Assert onRetakeQuiz was called
  })
})
```

**Step 2: Run test to verify it fails**

Expected: FAIL — no CTA button in empty state.

**Step 3: Implement**

In `ResultsTab.tsx`, find the empty state (around line 1311). Add a CTA button:

```tsx
if (attempts.length === 0 && !hasActiveFilters) {
  return (
    <>
      {contextHolder}
      <Empty
        description={
          <div className="space-y-2">
            <p className="text-text-muted">
              {t("option:quiz.noAttempts", { defaultValue: "No quiz attempts yet" })}
            </p>
            <p className="text-sm text-text-subtle">
              {t("option:quiz.noAttemptsHint", {
                defaultValue: "Complete a quiz to see your results here"
              })}
            </p>
          </div>
        }
      >
        <Button
          type="primary"
          onClick={() => onRetakeQuiz?.({
            startQuizId: null,
            highlightQuizId: null,
            forceShowWorkspaceItems: false,
            sourceTab: "results",
            attemptId: null,
            assignmentMode: null,
            assignmentDueAt: null,
            assignmentNote: null,
            assignedByRole: null
          })}
        >
          {t("option:quiz.takeAQuiz", { defaultValue: "Take a Quiz" })}
        </Button>
      </Empty>
    </>
  )
}
```

**Step 4: Run tests**

Expected: PASS

**Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx \
  apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.empty-cta.test.tsx
git commit -m "feat(quiz): add 'Take a Quiz' CTA in Results empty state"
```

---

### Task 14: Review checkpoint — Stage 3

**Step 1: Run all Quiz tests**

Run: `cd apps && npx vitest run packages/ui/src/components/Quiz/__tests__/`
Expected: ALL PASS

**Step 2: Run tutorial tests**

Run: `cd apps && npx vitest run packages/ui/src/tutorials/__tests__/`
Expected: ALL PASS

**Step 3: Visual verification**

Open `/quiz` in browser with no quizzes. Verify:
- [ ] Lands on Generate tab (not Take)
- [ ] Generate tab shows blue alert: "No media content found" with link to /media
- [ ] Clicking "Media Library" link navigates to /media
- [ ] Results tab shows "Take a Quiz" button
- [ ] Tutorial prompt appears offering guided tour
- [ ] Tutorial walks through all 5 tabs with descriptions

---

## Stage 4: Final Verification

### Task 15: Run full test suite and integration check

**Step 1: Run all modified component tests**

```bash
cd apps && npx vitest run \
  packages/ui/src/components/Option/MCPHub/__tests__/ \
  packages/ui/src/components/Media/__tests__/ \
  packages/ui/src/components/Quiz/__tests__/ \
  packages/ui/src/tutorials/__tests__/
```

Expected: ALL PASS

**Step 2: Run lint check**

```bash
cd apps && npx eslint packages/ui/src/components/Option/MCPHub/McpHubPage.tsx \
  packages/ui/src/components/Media/ResultsList.tsx \
  packages/ui/src/components/Quiz/QuizPlayground.tsx \
  packages/ui/src/tutorials/definitions/mcp-hub.ts \
  packages/ui/src/tutorials/definitions/quiz.ts \
  packages/ui/src/tutorials/registry.ts
```

Expected: No errors

**Step 3: Full walkthrough per persona**

Clear localStorage, then walk through each page as each persona:

**Content Consumer path:**
1. `/media` → sees first-ingest tutorial → pastes URL → Ingest → content appears → auto-refresh
2. `/quiz` → lands on Generate → sees "go to Media Library" alert → navigates → ingests → returns → generates quiz → takes quiz → sees results

**Technical User path:**
1. `/mcp-hub` → sees explainer card → dismisses → browses Tool Catalog → adds server in Credentials → creates profile
2. `/quiz` → Generate tab → selects media → generates quiz → manages in Manage tab

**Family Safety Parent path:**
1. `/media` → sees friendly tutorial → ingests educational video
2. `/quiz` → Generate tab → guidance about needing media → navigates to media → ingests → generates quiz for child

**Step 4: Final commit**

If any issues found during walkthrough, fix and commit individually.
