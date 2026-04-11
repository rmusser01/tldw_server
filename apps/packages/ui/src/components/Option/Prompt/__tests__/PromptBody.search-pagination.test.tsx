import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { notification } from "antd"
import { MemoryRouter, useLocation } from "react-router-dom"
import { PromptBody } from "../index"

const state = vi.hoisted(() => ({
  isOnline: true,
  privateMode: false,
  prompts: [] as any[]
}))

const promptStudioStore = vi.hoisted(() => ({
  setActiveSubTab: vi.fn(),
  setSelectedProjectId: vi.fn(),
  setSelectedPromptId: vi.fn(),
  setExecutePlaygroundOpen: vi.fn()
}))

const mocks = vi.hoisted(() => ({
  getAllPrompts: vi.fn(),
  getAllCopilotPrompts: vi.fn(),
  setAllCopilotPrompts: vi.fn(async () => []),
  getDeletedPrompts: vi.fn(),
  searchPromptsServer: vi.fn(),
  exportPromptsServer: vi.fn(),
  listPromptCollectionsServer: vi.fn(),
  createPromptCollectionServer: vi.fn(),
  updatePromptCollectionServer: vi.fn(),
  exportPrompts: vi.fn(),
  importPromptsV2: vi.fn(),
  deletePromptById: vi.fn(),
  restorePrompt: vi.fn(async (_id?: string | number) => undefined),
  permanentlyDeletePrompt: vi.fn(async () => undefined),
  emptyTrash: vi.fn(async () => 0),
  autoSyncPrompt: vi.fn(
    async (_id?: string | number) =>
      ({ success: true, syncStatus: "synced" }) as {
        success: boolean
        syncStatus: string
        error?: string
      }
  ),
  pushToStudio: vi.fn(async () => ({ success: true })),
  pullFromStudio: vi.fn(
    async (_id?: string | number) =>
      ({ success: true, localId: "local-1", syncStatus: "synced" }) as {
        success: boolean
        localId?: string
        syncStatus: string
        error?: string
      }
  ),
  getStudioPromptById: vi.fn(),
  getLlmProviders: vi.fn(),
  updatePrompt: vi.fn(async (_payload?: any) => "updated-id"),
  incrementPromptUsage: vi.fn(async () => ({
    usageCount: 1,
    lastUsedAt: Date.now()
  })),
  tldwInitialize: vi.fn(async () => undefined),
  createChatCompletion: vi.fn(),
  hasPromptStudio: vi.fn(),
  navigate: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  setSelectedSystemPrompt: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => mocks.navigate
  }
})

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    Skeleton: () => <div data-testid="prompts-loading-skeleton" />,
    Table: (props: any) => {
      const rows = Array.isArray(props?.dataSource) ? props.dataSource : []
      const currentPage = props?.pagination?.current || 1
      const pageSize = props?.pagination?.pageSize || 20
      const columns = Array.isArray(props?.columns) ? props.columns : []
      const selectedRowKeys = Array.isArray(props?.rowSelection?.selectedRowKeys)
        ? props.rowSelection.selectedRowKeys
        : []
      return (
        <div data-testid={props["data-testid"] || "mock-table"}>
          <div data-testid="table-row-count">{rows.length}</div>
          <div data-testid="table-selected-count">{selectedRowKeys.length}</div>
          {rows.map((row: any, index: number) => {
            const rowId = row?.id || row?.key || `row-${index}`
            const rowProps =
              typeof props?.onRow === "function" ? props.onRow(row, index) || {} : {}
            return (
              <div
                key={rowId}
                {...rowProps}
                data-testid={rowProps?.["data-testid"] || `table-row-${rowId}`}
              >
                <div data-testid="table-row-name">{row?.name || row?.title || row?.key}</div>
                {columns.map((column: any, columnIndex: number) => {
                  const columnKey = column?.key || column?.dataIndex || columnIndex
                  const value = column?.dataIndex ? row?.[column.dataIndex] : undefined
                  if (typeof column?.render === "function") {
                    return (
                      <div key={`col-${columnKey}`} data-testid={`table-cell-${columnKey}`}>
                        {column.render(value, row, index)}
                      </div>
                    )
                  }
                  return (
                    <div key={`col-${columnKey}`} data-testid={`table-cell-${columnKey}`}>
                      {value}
                    </div>
                  )
                })}
              </div>
            )
          })}
          <button
            type="button"
            data-testid="table-next-page"
            onClick={() =>
              props?.onChange?.({
                current: currentPage + 1,
                pageSize
              })
            }
          >
            next page
          </button>
          <button
            type="button"
            data-testid="table-select-all"
            onClick={() =>
              props?.rowSelection?.onChange?.(rows.map((row: any) => row.id))
            }
          >
            select all
          </button>
          <button
            type="button"
            data-testid="table-sort-title-asc"
            onClick={() =>
              props?.onChange?.(
                { current: currentPage, pageSize },
                {},
                { columnKey: "title", order: "ascend" }
              )
            }
          >
            sort title asc
          </button>
          <button
            type="button"
            data-testid="table-sort-title-desc"
            onClick={() =>
              props?.onChange?.(
                { current: currentPage, pageSize },
                {},
                { columnKey: "title", order: "descend" }
              )
            }
          >
            sort title desc
          </button>
          <button
            type="button"
            data-testid="table-sort-modified-desc"
            onClick={() =>
              props?.onChange?.(
                { current: currentPage, pageSize },
                {},
                { columnKey: "modifiedAt", order: "descend" }
              )
            }
          >
            sort modified desc
          </button>
        </div>
      )
    }
  }
})

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => state.isOnline
}))

vi.mock("@/utils/is-private-mode", () => ({
  get isFireFoxPrivateMode() {
    return state.privateMode
  },
  get isFireFox() {
    return state.privateMode
  }
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    setSelectedQuickPrompt: mocks.setSelectedQuickPrompt,
    setSelectedSystemPrompt: mocks.setSelectedSystemPrompt
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(async () => true)
}))

vi.mock("@/services/application", () => ({
  getAllCopilotPrompts: (...args: unknown[]) =>
    (mocks.getAllCopilotPrompts as (...args: unknown[]) => unknown)(...args),
  upsertCopilotPrompts: (...args: unknown[]) =>
    (mocks.setAllCopilotPrompts as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/prompts-api", () => ({
  exportPromptsServer: (...args: unknown[]) =>
    (mocks.exportPromptsServer as (...args: unknown[]) => unknown)(...args),
  searchPromptsServer: (...args: unknown[]) =>
    (mocks.searchPromptsServer as (...args: unknown[]) => unknown)(...args),
  listPromptCollectionsServer: (...args: unknown[]) =>
    (mocks.listPromptCollectionsServer as (...args: unknown[]) => unknown)(...args),
  createPromptCollectionServer: (...args: unknown[]) =>
    (mocks.createPromptCollectionServer as (...args: unknown[]) => unknown)(...args),
  updatePromptCollectionServer: (...args: unknown[]) =>
    (mocks.updatePromptCollectionServer as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/prompt-studio", () => ({
  hasPromptStudio: (...args: unknown[]) =>
    (mocks.hasPromptStudio as (...args: unknown[]) => unknown)(...args),
  getPrompt: (...args: unknown[]) =>
    (mocks.getStudioPromptById as (...args: unknown[]) => unknown)(...args),
  getLlmProviders: (...args: unknown[]) =>
    (mocks.getLlmProviders as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) =>
      (mocks.tldwInitialize as (...args: unknown[]) => unknown)(...args),
    createChatCompletion: (...args: unknown[]) =>
      (mocks.createChatCompletion as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/store/prompt-studio", () => ({
  usePromptStudioStore: (selector: (state: typeof promptStudioStore) => unknown) =>
    selector(promptStudioStore)
}))

vi.mock("@/services/prompt-sync", () => ({
  autoSyncPrompt: (...args: unknown[]) =>
    (mocks.autoSyncPrompt as (...args: unknown[]) => unknown)(...args),
  pushToStudio: (...args: unknown[]) =>
    (mocks.pushToStudio as (...args: unknown[]) => unknown)(...args),
  pullFromStudio: (...args: unknown[]) =>
    (mocks.pullFromStudio as (...args: unknown[]) => unknown)(...args),
  shouldAutoSyncWorkspacePrompts: vi.fn(async () => false),
  unlinkPrompt: vi.fn(async () => ({ success: true })),
  getConflictInfo: vi.fn(async () => null),
  resolveConflict: vi.fn(async () => ({ success: true })),
  getAllPromptsWithSyncStatus: vi.fn(async () => [])
}))

vi.mock("@/db/dexie/helpers", () => ({
  deletePromptById: (...args: unknown[]) =>
    (mocks.deletePromptById as (...args: unknown[]) => unknown)(...args),
  getAllPrompts: (...args: unknown[]) =>
    (mocks.getAllPrompts as (...args: unknown[]) => unknown)(...args),
  savePrompt: vi.fn(async (payload: any) => ({
    id: "saved-id",
    ...payload
  })),
  updatePrompt: (...args: unknown[]) =>
    (mocks.updatePrompt as (...args: unknown[]) => unknown)(...args),
  incrementPromptUsage: (...args: unknown[]) =>
    (mocks.incrementPromptUsage as (...args: unknown[]) => unknown)(...args),
  exportPrompts: (...args: unknown[]) =>
    (mocks.exportPrompts as (...args: unknown[]) => unknown)(...args),
  importPromptsV2: (...args: unknown[]) =>
    (mocks.importPromptsV2 as (...args: unknown[]) => unknown)(...args),
  getDeletedPrompts: (...args: unknown[]) =>
    (mocks.getDeletedPrompts as (...args: unknown[]) => unknown)(...args),
  restorePrompt: (...args: unknown[]) =>
    (mocks.restorePrompt as (...args: unknown[]) => unknown)(...args),
  permanentlyDeletePrompt: (...args: unknown[]) =>
    (mocks.permanentlyDeletePrompt as (...args: unknown[]) => unknown)(...args),
  emptyTrash: (...args: unknown[]) =>
    (mocks.emptyTrash as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("../PromptDrawer", () => ({
  PromptDrawer: (props: any) =>
    props.open ? (
      <div data-testid="mock-prompt-drawer">
        {props?.initialValues?.name || props?.initialValues?.id || "open"}
      </div>
    ) : null
}))

vi.mock("../PromptActionsMenu", () => ({
  PromptActionsMenu: (props: any) => (
    <div data-testid="mock-prompt-actions-menu">
      <button
        type="button"
        data-testid={`mock-use-in-chat-${props?.promptId || "unknown"}`}
        onClick={() => props?.onUseInChat?.()}
      >
        use in chat
      </button>
      <button
        type="button"
        data-testid={`mock-quick-test-${props?.promptId || "unknown"}`}
        onClick={() => props?.onQuickTest?.()}
      >
        quick test
      </button>
    </div>
  )
}))

vi.mock("../SyncStatusBadge", () => ({
  SyncStatusBadge: () => <span data-testid="mock-sync-status-badge" />
}))

vi.mock("../ConflictResolutionModal", () => ({
  ConflictResolutionModal: () => null
}))

vi.mock("../ProjectSelector", () => ({
  ProjectSelector: () => null
}))

vi.mock("../Studio/StudioTabContainer", () => ({
  StudioTabContainer: () => <div data-testid="mock-studio-tab-container" />
}))

const LocationProbe = () => {
  const location = useLocation()
  return <div data-testid="prompt-location-search">{location.search}</div>
}

const renderPromptBody = (initialEntries: string[] = ["/prompts"]) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries}>
        <LocationProbe />
        <PromptBody />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe("PromptBody server search and pagination", () => {
  const setViewportWidth = (width: number) => {
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      writable: true,
      value: width
    })
    window.dispatchEvent(new Event("resize"))
  }

  const getVisibleRowNames = () =>
    screen
      .getAllByTestId("table-row-name")
      .map((node) => node.textContent?.trim() || "")

  it("renders tab-selected routes without crashing during initial URL sync", () => {
    renderPromptBody(["/prompts?tab=trash"])

    expect(screen.getByTestId("prompt-location-search")).toHaveTextContent("?tab=trash")
  })

  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    mocks.navigate.mockReset()
    mocks.setSelectedQuickPrompt.mockReset()
    mocks.setSelectedSystemPrompt.mockReset()
    state.privateMode = false
    state.isOnline = true
    setViewportWidth(1280)
    state.prompts = [
      {
        id: "local-1",
        name: "Alpha One",
        title: "Alpha One",
        content: "alpha content",
        is_system: false,
        createdAt: 100,
        serverId: 101,
        keywords: ["alpha"]
      },
      {
        id: "local-2",
        name: "Alpha Two",
        title: "Alpha Two",
        content: "alpha content two",
        is_system: false,
        createdAt: 90,
        serverId: 102,
        keywords: ["alpha"]
      },
      {
        id: "local-3",
        name: "Alpha Three",
        title: "Alpha Three",
        content: "alpha content three",
        is_system: false,
        createdAt: 80,
        serverId: 103,
        keywords: ["alpha"]
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)
    mocks.getDeletedPrompts.mockResolvedValue([])
    mocks.getAllCopilotPrompts.mockResolvedValue([])
    mocks.hasPromptStudio.mockResolvedValue(false)
    mocks.pullFromStudio.mockResolvedValue({
      success: true,
      localId: "local-1",
      syncStatus: "synced"
    })
    mocks.getStudioPromptById.mockResolvedValue({
      data: {
        data: {
          id: 101,
          project_id: 10
        }
      }
    })
    mocks.getLlmProviders.mockResolvedValue({
      providers: [
        {
          name: "openai",
          display_name: "OpenAI",
          default_model: "gpt-4o-mini",
          models: ["gpt-4o-mini"]
        }
      ],
      default_provider: "openai"
    })
    mocks.createChatCompletion.mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: "Quick test output" } }]
        })
      )
    )
    mocks.exportPrompts.mockResolvedValue([])
    mocks.incrementPromptUsage.mockResolvedValue({
      usageCount: 1,
      lastUsedAt: Date.now()
    })
    mocks.listPromptCollectionsServer.mockResolvedValue([])
    mocks.createPromptCollectionServer.mockResolvedValue({ collection_id: 1 })
    mocks.updatePromptCollectionServer.mockImplementation(async (_id: number, payload: any) => ({
      collection_id: 1,
      name: "Collection",
      description: null,
      prompt_ids: payload?.prompt_ids || []
    }))
    mocks.importPromptsV2.mockResolvedValue({
      imported: 0,
      skipped: 0,
      failed: 0
    })
    mocks.deletePromptById.mockResolvedValue(undefined)
    mocks.searchPromptsServer.mockImplementation(async (params: any) => {
      if (params.page === 2) {
        return {
          items: [{ id: 103, uuid: "s-103", name: "Alpha Three" }],
          total_matches: 3,
          page: 2,
          per_page: 20
        }
      }
      return {
        items: [
          { id: 101, uuid: "s-101", name: "Alpha One" },
          { id: 102, uuid: "s-102", name: "Alpha Two" }
        ],
        total_matches: 3,
        page: 1,
        per_page: 20
      }
    })
  })

  afterEach(() => {
    setViewportWidth(1280)
    vi.clearAllMocks()
  })

  it("exposes the screen-reader status live region", async () => {
    renderPromptBody()

    const announcer = await screen.findByRole("status")
    expect(announcer).toHaveAttribute("id", "prompts-status-announcer")
    expect(announcer).toHaveAttribute("aria-live", "polite")
    expect(announcer).toHaveAttribute("aria-atomic", "true")
  })

  it("preserves favorite toggle semantics and keyboard row activation for inspector", async () => {
    renderPromptBody()

    const favoriteButton = await screen.findByTestId("prompt-favorite-local-1")
    expect(favoriteButton).toHaveAttribute("aria-pressed", "false")

    const firstRow = screen.getByTestId("prompt-row-local-1")
    expect(firstRow).toHaveAttribute("tabindex", "0")
    fireEvent.keyDown(firstRow, { key: "Enter", code: "Enter" })

    await waitFor(() => {
      expect(
        screen.getByTestId("prompts-inspector-panel-scaffold")
      ).toBeInTheDocument()
    })
    expect(screen.queryByTestId("mock-prompt-drawer")).not.toBeInTheDocument()
  })

  it("uses full-width responsive controls on compact viewports", async () => {
    setViewportWidth(480)
    renderPromptBody()

    const searchInput = await screen.findByTestId("prompts-search")
    const searchWrapper = searchInput.closest(".ant-input-affix-wrapper")
    expect(searchWrapper).not.toBeNull()
    expect((searchWrapper as HTMLElement).style.width).toBe("100%")

    const typeFilter = screen.getByTestId("prompts-type-filter")
    const tagFilter = screen.getByTestId("prompts-tag-filter")
    const tagMode = screen.getByTestId("prompts-tag-match-mode")

    expect(typeFilter.style.width).toBe("100%")
    expect(tagFilter.style.width).toBe("100%")
    expect(tagMode.style.width).toBe("100%")
  })

  it("shows mobile overflow affordance and collapses lower-priority columns", async () => {
    setViewportWidth(480)
    renderPromptBody()

    await screen.findByTestId("prompts-table")
    expect(
      screen.getByTestId("prompts-table-overflow-indicator")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("table-cell-keywords")).not.toBeInTheDocument()
    expect(screen.queryByTestId("table-cell-type")).not.toBeInTheDocument()
    expect(screen.queryByTestId("table-cell-modifiedAt")).not.toBeInTheDocument()
    expect(screen.queryByTestId("table-cell-syncStatus")).not.toBeInTheDocument()
  })

  it("applies touch-friendly bulk action target sizes on compact viewports", async () => {
    setViewportWidth(480)
    renderPromptBody()

    await screen.findByTestId("prompts-table")
    fireEvent.click(screen.getByTestId("table-select-all"))

    const bulkExportButton = await screen.findByTestId("prompts-bulk-export")
    expect(bulkExportButton.className).toContain("min-h-[44px]")
  })

  it("debounces online server search and requests next page", async () => {
    renderPromptBody()

    const search = await screen.findByRole("textbox", {
      name: "Search prompts..."
    })

    fireEvent.change(search, { target: { value: "alpha" } })

    await new Promise((resolve) => setTimeout(resolve, 150))
    expect(mocks.searchPromptsServer).not.toHaveBeenCalled()

    await new Promise((resolve) => setTimeout(resolve, 220))

    await waitFor(() => expect(mocks.searchPromptsServer).toHaveBeenCalledTimes(1))
    expect(mocks.searchPromptsServer.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        searchQuery: "alpha",
        page: 1,
        resultsPerPage: 20
      })
    )
    expect(screen.getByTestId("table-row-count")).toHaveTextContent("2")

    fireEvent.click(screen.getByTestId("table-next-page"))

    await waitFor(() => {
      expect(mocks.searchPromptsServer).toHaveBeenCalledTimes(2)
    })
    expect(mocks.searchPromptsServer.mock.calls[1]?.[0]).toEqual(
      expect.objectContaining({
        searchQuery: "alpha",
        page: 2
      })
    )
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })
  })

  it("falls back to local filtering when offline", async () => {
    state.isOnline = false
    state.prompts = [
      {
        id: "local-a",
        name: "Alpha Prompt",
        title: "Alpha Prompt",
        content: "first content",
        is_system: false,
        createdAt: 100,
        keywords: ["alpha"]
      },
      {
        id: "local-b",
        name: "Beta Prompt",
        title: "Beta Prompt",
        content: "second content",
        is_system: false,
        createdAt: 90,
        keywords: ["beta"]
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    const search = await screen.findByRole("textbox", {
      name: "Search prompts..."
    })
    fireEvent.change(search, { target: { value: "beta" } })

    await new Promise((resolve) => setTimeout(resolve, 320))

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })
    expect(getVisibleRowNames()).toEqual(["Beta Prompt"])
    expect(mocks.searchPromptsServer).not.toHaveBeenCalled()
  })

  it("applies title sort override and persists sort state in session storage", async () => {
    state.prompts = [
      {
        id: "prompt-c",
        name: "Charlie",
        title: "Charlie",
        content: "c",
        is_system: false,
        favorite: true,
        createdAt: 1,
        keywords: []
      },
      {
        id: "prompt-b",
        name: "Bravo",
        title: "Bravo",
        content: "b",
        is_system: false,
        createdAt: 3,
        keywords: []
      },
      {
        id: "prompt-a",
        name: "Alpha",
        title: "Alpha",
        content: "a",
        is_system: false,
        createdAt: 2,
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })
    expect(getVisibleRowNames()).toEqual(["Charlie", "Bravo", "Alpha"])

    fireEvent.click(screen.getByTestId("table-sort-title-asc"))

    await waitFor(() => {
      expect(getVisibleRowNames()).toEqual(["Alpha", "Bravo", "Charlie"])
    })
    await waitFor(() => {
      const saved = window.sessionStorage.getItem(
        "tldw-prompts-custom-sort-v1"
      )
      expect(saved).toContain('"key":"title"')
      expect(saved).toContain('"order":"ascend"')
    })
  })

  it("sorts by modified timestamp when modified column sort is selected", async () => {
    const now = Date.now()
    state.prompts = [
      {
        id: "prompt-old",
        name: "Old Prompt",
        title: "Old Prompt",
        content: "old",
        is_system: false,
        createdAt: now - 60_000,
        updatedAt: now - 30_000,
        keywords: []
      },
      {
        id: "prompt-new",
        name: "New Prompt",
        title: "New Prompt",
        content: "new",
        is_system: false,
        createdAt: now - 5_000,
        updatedAt: now - 1_000,
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("2")
    })

    fireEvent.click(screen.getByTestId("table-sort-modified-desc"))

    await waitFor(() => {
      expect(getVisibleRowNames()).toEqual(["New Prompt", "Old Prompt"])
    })
  })

  it("shows bulk actions after selection and runs bulk push", async () => {
    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))

    expect(screen.getByTestId("prompts-bulk-add-keyword")).toBeInTheDocument()
    expect(screen.getByTestId("prompts-bulk-toggle-favorite")).toBeInTheDocument()
    expect(screen.getByTestId("prompts-bulk-push-server")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("prompts-bulk-push-server"))

    await waitFor(() => {
      expect(mocks.autoSyncPrompt).toHaveBeenCalledTimes(3)
    })
  })

  it("exports JSON locally by default", async () => {
    mocks.exportPrompts.mockResolvedValue([
      {
        id: "prompt-export",
        name: "Export Me",
        title: "Export Me",
        content: "content",
        is_system: false,
        createdAt: 123
      }
    ])

    renderPromptBody()
    fireEvent.click(await screen.findByTestId("prompts-export"))

    await waitFor(() => {
      expect(mocks.exportPrompts).toHaveBeenCalledTimes(1)
    })
    expect(mocks.exportPromptsServer).not.toHaveBeenCalled()
  })

  it("keeps only failed rows selected when bulk favorite has partial failures", async () => {
    mocks.updatePrompt.mockImplementation(async (payload: any) => {
      if (payload?.id === "local-2") {
        throw new Error("update failed")
      }
      return payload?.id || "updated-id"
    })

    renderPromptBody()
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))
    expect(screen.getByTestId("table-selected-count")).toHaveTextContent("3")

    fireEvent.click(screen.getByTestId("prompts-bulk-toggle-favorite"))

    await waitFor(() => {
      expect(mocks.updatePrompt).toHaveBeenCalledTimes(3)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("1")
    })
  })

  it("applies bulk keyword updates with partial failure handling", async () => {
    state.prompts = [
      {
        id: "local-1",
        name: "Alpha One",
        title: "Alpha One",
        content: "alpha content",
        is_system: false,
        createdAt: 100,
        keywords: ["alpha"]
      },
      {
        id: "local-2",
        name: "Alpha Two",
        title: "Alpha Two",
        content: "alpha content two",
        is_system: false,
        createdAt: 90,
        keywords: ["alpha"]
      },
      {
        id: "local-3",
        name: "Alpha Three",
        title: "Alpha Three",
        content: "alpha content three",
        is_system: false,
        createdAt: 80,
        keywords: ["bulk-tag"]
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)
    mocks.updatePrompt.mockImplementation(async (payload: any) => {
      if (payload?.id === "local-2") {
        throw new Error("keyword update failed")
      }
      return payload?.id || "updated-id"
    })

    renderPromptBody()
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))
    fireEvent.click(screen.getByTestId("prompts-bulk-add-keyword"))

    const input = await screen.findByTestId("prompts-bulk-keyword-input")
    fireEvent.change(input, { target: { value: "bulk-tag" } })
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" })

    await waitFor(() => {
      expect(mocks.updatePrompt).toHaveBeenCalledTimes(2)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("1")
    })
  })

  it("keeps failed rows selected when bulk push has partial failures", async () => {
    mocks.autoSyncPrompt.mockImplementation(async (id?: string | number) => {
      if (id === "local-2") {
        return {
          success: false,
          error: "project resolution failed",
          syncStatus: "local"
        }
      }
      return {
        success: true,
        syncStatus: "synced"
      }
    })

    renderPromptBody()
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))
    expect(screen.getByTestId("table-selected-count")).toHaveTextContent("3")
    fireEvent.click(screen.getByTestId("prompts-bulk-push-server"))

    await waitFor(() => {
      expect(mocks.autoSyncPrompt).toHaveBeenCalledTimes(3)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("1")
    })
  })

  it("preserves use-in-chat modal behavior and navigates with both prompt parts", async () => {
    state.prompts = [
      {
        id: "local-1",
        name: "Dual Prompt",
        title: "Dual Prompt",
        content: "fallback",
        is_system: false,
        createdAt: 100,
        system_prompt: "System instruction body",
        user_prompt: "User template body",
        keywords: ["alpha"]
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("mock-use-in-chat-local-1"))
    expect(await screen.findByTestId("prompt-insert-both")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("prompt-insert-both"))

    await waitFor(() => {
      expect(mocks.incrementPromptUsage).toHaveBeenCalledWith("local-1")
    })
    expect(mocks.setSelectedSystemPrompt).toHaveBeenCalledWith("local-1")
    expect(mocks.setSelectedQuickPrompt).toHaveBeenCalledWith("User template body")
    expect(mocks.navigate).toHaveBeenCalledWith("/chat")
  })

  it("tracks usage when quick prompt inserts directly without opening modal", async () => {
    state.prompts = [
      {
        id: "local-quick",
        name: "Quick Prompt",
        title: "Quick Prompt",
        content: "Quick insert content",
        is_system: false,
        createdAt: 100,
        user_prompt: "Quick insert content",
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("mock-use-in-chat-local-quick"))

    await waitFor(() => {
      expect(mocks.incrementPromptUsage).toHaveBeenCalledWith("local-quick")
    })
    expect(mocks.setSelectedQuickPrompt).toHaveBeenCalledWith("Quick insert content")
    expect(mocks.setSelectedSystemPrompt).toHaveBeenCalledWith(undefined)
    expect(mocks.navigate).toHaveBeenCalledWith("/chat")
  })

  it("opens local quick test modal for unsynced prompts and renders response output", async () => {
    state.prompts = [
      {
        id: "local-only",
        name: "Local Prompt",
        title: "Local Prompt",
        content: "Summarize {{text}}",
        user_prompt: "Summarize {{text}}",
        is_system: false,
        createdAt: 100,
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("mock-quick-test-local-only"))

    const input = await screen.findByTestId("prompts-local-quick-test-input")
    fireEvent.change(input, { target: { value: "draft the release note" } })
    fireEvent.click(screen.getByTestId("prompts-local-quick-test-run"))

    await waitFor(() => {
      expect(mocks.createChatCompletion).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(screen.getByTestId("prompts-local-quick-test-output")).toHaveTextContent(
        "Quick test output"
      )
    })
  })

  it("opens studio execute context for synced prompts when quick test is clicked", async () => {
    state.prompts = [
      {
        id: "synced-1",
        name: "Synced Prompt",
        title: "Synced Prompt",
        content: "content",
        is_system: false,
        createdAt: 100,
        serverId: 501,
        studioProjectId: 77,
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)
    mocks.hasPromptStudio.mockResolvedValue(true)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("mock-quick-test-synced-1"))

    await waitFor(() => {
      expect(promptStudioStore.setSelectedProjectId).toHaveBeenCalledWith(77)
    })
    expect(promptStudioStore.setSelectedPromptId).toHaveBeenCalledWith(501)
    expect(promptStudioStore.setActiveSubTab).toHaveBeenCalledWith("prompts")
    expect(promptStudioStore.setExecutePlaygroundOpen).toHaveBeenCalledWith(true)
  })

  it("opens the edit drawer from prompt deep-link query parameter", async () => {
    state.prompts = [
      {
        id: "local-2",
        name: "Linked Prompt",
        title: "Linked Prompt",
        content: "linked content",
        is_system: false,
        createdAt: 100,
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody(["/prompts?prompt=local-2"])

    await waitFor(() => {
      expect(screen.getByTestId("mock-prompt-drawer")).toBeInTheDocument()
    })
    expect(screen.getByTestId("mock-prompt-drawer")).toHaveTextContent(
      "Linked Prompt"
    )
  })

  it("pulls and opens a shared prompt deep-link when only server id is provided", async () => {
    const importedPrompt = {
      id: "imported-1",
      name: "Imported Shared Prompt",
      title: "Imported Shared Prompt",
      content: "imported content",
      is_system: false,
      createdAt: 200,
      serverId: 101,
      keywords: []
    }
    mocks.getAllPrompts
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([importedPrompt])
    mocks.pullFromStudio.mockResolvedValue({
      success: true,
      localId: "imported-1",
      syncStatus: "synced"
    })

    renderPromptBody(["/prompts?prompt=101&source=studio"])

    await waitFor(() => {
      expect(mocks.pullFromStudio).toHaveBeenCalledWith(101)
    })

    await waitFor(() => {
      expect(screen.getByTestId("mock-prompt-drawer")).toBeInTheDocument()
    })
    expect(screen.getByTestId("mock-prompt-drawer")).toHaveTextContent(
      "Imported Shared Prompt"
    )
  })

  it("warns when a shared prompt deep-link cannot be pulled from server", async () => {
    const warningSpy = vi.spyOn(notification, "warning")
    mocks.getAllPrompts.mockResolvedValue([])
    mocks.pullFromStudio.mockResolvedValue({
      success: false,
      error: "Server prompt not found",
      syncStatus: "local"
    })

    renderPromptBody(["/prompts?prompt=404&source=studio"])

    await waitFor(() => {
      expect(mocks.pullFromStudio).toHaveBeenCalledWith(404)
    })
    await waitFor(() => {
      expect(warningSpy).toHaveBeenCalled()
    })
    expect(screen.queryByTestId("mock-prompt-drawer")).not.toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByTestId("prompt-location-search").textContent).toBe("")
    })
    warningSpy.mockRestore()
  })

  it("shows access guidance when a shared prompt deep-link fails with authorization errors", async () => {
    const warningSpy = vi.spyOn(notification, "warning")
    mocks.getAllPrompts.mockResolvedValue([])
    mocks.pullFromStudio.mockResolvedValue({
      success: false,
      error: "HTTP 403 Forbidden",
      syncStatus: "local"
    })

    renderPromptBody(["/prompts?prompt=402&source=studio"])

    await waitFor(() => {
      expect(warningSpy).toHaveBeenCalled()
    })
    const latestWarning = warningSpy.mock.calls.at(-1)?.[0] as
      | { message?: string; description?: string }
      | undefined
    expect(latestWarning?.message).toBe("Prompt not found")
    expect(latestWarning?.description).toBe(
      "You don't have permission to open this shared prompt. Check your server login and project access."
    )
    await waitFor(() => {
      expect(screen.getByTestId("prompt-location-search").textContent).toBe("")
    })
    warningSpy.mockRestore()
  })

  it("retains row selection while paging through large local datasets", async () => {
    state.isOnline = false
    state.prompts = Array.from({ length: 45 }, (_, index) => ({
      id: `prompt-${index + 1}`,
      name: `Prompt ${index + 1}`,
      title: `Prompt ${index + 1}`,
      content: `content ${index + 1}`,
      is_system: false,
      createdAt: 1_000 - index,
      keywords: []
    }))
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("20")
    })
    fireEvent.click(screen.getByTestId("table-select-all"))
    expect(screen.getByTestId("table-selected-count")).toHaveTextContent("20")

    fireEvent.click(screen.getByTestId("table-next-page"))
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("20")
    })
    expect(screen.getByTestId("table-selected-count")).toHaveTextContent("20")
  })

  it("handles search, sort, and pagination on large prompt fixtures without crashing", async () => {
    state.isOnline = false
    state.prompts = Array.from({ length: 500 }, (_, index) => ({
      id: `perf-${index + 1}`,
      name: `Performance Prompt ${index + 1}`,
      title: `Performance Prompt ${index + 1}`,
      content: `Prompt performance content ${index + 1}`,
      is_system: false,
      createdAt: 10_000 - index,
      updatedAt: 20_000 - index,
      keywords: [`tag-${index % 7}`]
    }))
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody()

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("20")
    })

    const search = screen.getByRole("textbox", {
      name: "Search prompts..."
    })
    fireEvent.change(search, { target: { value: "Performance Prompt 49" } })

    await new Promise((resolve) => setTimeout(resolve, 320))
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("11")
    })

    fireEvent.click(screen.getByTestId("table-sort-title-asc"))
    await waitFor(() => {
      expect(getVisibleRowNames()[0]).toContain("Performance Prompt 49")
    })

    fireEvent.click(screen.getByTestId("table-next-page"))
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("0")
    })
  })

  it("shows source-specific partial-load alerts when prompt and copilot data fail", async () => {
    mocks.getAllPrompts.mockRejectedValue(new Error("prompt load failed"))
    mocks.getAllCopilotPrompts.mockRejectedValue(new Error("copilot load failed"))

    renderPromptBody()

    expect(
      await screen.findByText("Some prompt data isn't available")
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Custom prompts couldn/)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Copilot prompts couldn/)
    ).toBeInTheDocument()
  })

  it("blocks mutating actions in Firefox private mode", async () => {
    state.privateMode = true
    const errorSpy = vi.spyOn(notification, "error")

    renderPromptBody()

    expect(
      await screen.findByText(/Firefox Private Mode doesn't support IndexedDB/)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("prompts-add"))
    expect(screen.queryByTestId("mock-prompt-drawer")).not.toBeInTheDocument()

    const importFile = new File(["[]"], "prompts.json", {
      type: "application/json"
    })
    fireEvent.change(screen.getByTestId("prompts-import-file"), {
      target: { files: [importFile] }
    })

    await waitFor(() => {
      expect(mocks.importPromptsV2).not.toHaveBeenCalled()
    })
    expect(errorSpy).toHaveBeenCalled()
    errorSpy.mockRestore()
  })

  it("warns on missing deep-link prompt and clears the prompt query parameter", async () => {
    const warningSpy = vi.spyOn(notification, "warning")
    state.prompts = [
      {
        id: "existing-prompt",
        name: "Existing Prompt",
        title: "Existing Prompt",
        content: "x",
        is_system: false,
        createdAt: 1,
        keywords: []
      }
    ]
    mocks.getAllPrompts.mockResolvedValue(state.prompts)

    renderPromptBody(["/prompts?prompt=missing-prompt"])

    await waitFor(() => {
      expect(warningSpy).toHaveBeenCalled()
    })
    const latestWarning = warningSpy.mock.calls.at(-1)?.[0] as
      | { message?: string; description?: string }
      | undefined
    expect(latestWarning?.message).toBe("Prompt not found")
    expect(latestWarning?.description).toBe(
      "The requested prompt could not be found. It may have been deleted."
    )
    expect(screen.queryByTestId("mock-prompt-drawer")).not.toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByTestId("prompt-location-search").textContent).toBe("")
    })
    warningSpy.mockRestore()
  })

  it("retains failed bulk-delete rows for retry and clears selection after successful retry", async () => {
    mocks.deletePromptById.mockImplementation(async (id: string) => {
      if (id === "local-2") {
        throw new Error("delete failed")
      }
      return undefined
    })

    renderPromptBody()
    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))
    fireEvent.click(screen.getByTestId("prompts-bulk-delete"))

    await waitFor(() => {
      expect(mocks.deletePromptById).toHaveBeenCalledTimes(3)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("1")
    })

    mocks.deletePromptById.mockResolvedValue(undefined)
    fireEvent.click(screen.getByTestId("prompts-bulk-delete"))

    await waitFor(() => {
      expect(mocks.deletePromptById).toHaveBeenCalledTimes(4)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("0")
    })
  })

  it("filters trash prompts by name and renders content preview", async () => {
    const now = Date.now()
    mocks.getDeletedPrompts.mockResolvedValue([
      {
        id: "trash-1",
        name: "Deprecated Summary Prompt",
        content: "Summarize meeting notes with action items.",
        deletedAt: now - 2 * 24 * 60 * 60 * 1000
      },
      {
        id: "trash-2",
        name: "Old Translation Prompt",
        content: "Translate this text into Spanish.",
        deletedAt: now - 5 * 24 * 60 * 60 * 1000
      }
    ])

    renderPromptBody(["/prompts?tab=trash"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("2")
    })
    expect(
      screen.getByText("Summarize meeting notes with action items.")
    ).toBeInTheDocument()

    fireEvent.change(screen.getByTestId("prompts-trash-search"), {
      target: { value: "translation" }
    })

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })
    expect(getVisibleRowNames()).toEqual(["Old Translation Prompt"])
  })

  it("shows trash remaining-days indicators with severity styling", async () => {
    const now = Date.now()
    const dayMs = 24 * 60 * 60 * 1000
    mocks.getDeletedPrompts.mockResolvedValue([
      {
        id: "trash-danger",
        name: "Expiring Prompt",
        content: "x",
        deletedAt: now - 25 * dayMs
      },
      {
        id: "trash-warning",
        name: "Soon Prompt",
        content: "x",
        deletedAt: now - 18 * dayMs
      },
      {
        id: "trash-normal",
        name: "Recent Prompt",
        content: "x",
        deletedAt: now - 3 * dayMs
      }
    ])

    renderPromptBody(["/prompts?tab=trash"])

    const danger = await screen.findByTestId("prompts-trash-remaining-trash-danger")
    const warning = await screen.findByTestId("prompts-trash-remaining-trash-warning")
    const normal = await screen.findByTestId("prompts-trash-remaining-trash-normal")

    expect(danger).toHaveTextContent("5 days left")
    expect(danger.className).toContain("text-danger")
    expect(warning).toHaveTextContent("12 days left")
    expect(warning.className).toContain("text-warn")
    expect(normal).toHaveTextContent("27 days left")
    expect(normal.className).toContain("text-text-muted")
  })

  it("restores selected trash prompts in bulk and keeps failed items selected for retry", async () => {
    mocks.getDeletedPrompts.mockResolvedValue([
      { id: "trash-1", name: "Trash 1", content: "one", deletedAt: Date.now() - 1000 },
      { id: "trash-2", name: "Trash 2", content: "two", deletedAt: Date.now() - 1000 },
      { id: "trash-3", name: "Trash 3", content: "three", deletedAt: Date.now() - 1000 }
    ])
    mocks.restorePrompt.mockImplementation(async (id: string) => {
      if (id === "trash-2") {
        throw new Error("restore failed")
      }
      return undefined
    })

    renderPromptBody(["/prompts?tab=trash"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))
    expect(screen.getByTestId("table-selected-count")).toHaveTextContent("3")

    fireEvent.click(screen.getByTestId("prompts-trash-bulk-restore"))

    await waitFor(() => {
      expect(mocks.restorePrompt).toHaveBeenCalledTimes(3)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("1")
    })

    mocks.restorePrompt.mockResolvedValue(undefined)
    fireEvent.click(screen.getByTestId("prompts-trash-bulk-restore"))

    await waitFor(() => {
      expect(mocks.restorePrompt).toHaveBeenCalledTimes(4)
    })
    await waitFor(() => {
      expect(screen.getByTestId("table-selected-count")).toHaveTextContent("0")
    })
  })

  it("keeps row-level trash restore action functional with row selection enabled", async () => {
    mocks.getDeletedPrompts.mockResolvedValue([
      { id: "trash-row", name: "Row Restore Prompt", content: "abc", deletedAt: Date.now() - 1000 }
    ])

    renderPromptBody(["/prompts?tab=trash"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("table-select-all"))
    fireEvent.click(screen.getByTestId("prompts-trash-restore-trash-row"))

    await waitFor(() => {
      expect(mocks.restorePrompt).toHaveBeenCalled()
      expect(mocks.restorePrompt.mock.calls.at(-1)?.[0]).toBe("trash-row")
    })
  })

  it("shows a specific error for empty import files", async () => {
    const errorSpy = vi.spyOn(notification, "error")

    renderPromptBody()
    const emptyFile = {
      name: "empty.json",
      text: vi.fn(async () => "   ")
    } as unknown as File

    fireEvent.change(screen.getByTestId("prompts-import-file"), {
      target: { files: [emptyFile] }
    })

    await waitFor(() => {
      expect(mocks.importPromptsV2).not.toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalled()
    })
    const errorCalls = errorSpy.mock.calls.map((call) =>
      JSON.stringify(call?.[0] || {})
    )
    expect(
      errorCalls.some((call) => call.includes("File is empty"))
    ).toBe(true)
    errorSpy.mockRestore()
  })

  it("shows a specific error for malformed JSON imports", async () => {
    const errorSpy = vi.spyOn(notification, "error")

    renderPromptBody()
    const badJsonFile = {
      name: "broken.json",
      text: vi.fn(async () => '{"prompts":[}')
    } as unknown as File

    fireEvent.change(screen.getByTestId("prompts-import-file"), {
      target: { files: [badJsonFile] }
    })

    await waitFor(() => {
      expect(mocks.importPromptsV2).not.toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalled()
    })
    const errorCalls = errorSpy.mock.calls.map((call) =>
      JSON.stringify(call?.[0] || {})
    )
    expect(
      errorCalls.some((call) => call.includes("Invalid JSON"))
    ).toBe(true)
    errorSpy.mockRestore()
  })

  it("shows a specific error for invalid import schema", async () => {
    const errorSpy = vi.spyOn(notification, "error")

    renderPromptBody()
    const wrongSchemaFile = {
      name: "wrong-schema.json",
      text: vi.fn(async () => '{"foo":"bar"}')
    } as unknown as File

    fireEvent.change(screen.getByTestId("prompts-import-file"), {
      target: { files: [wrongSchemaFile] }
    })

    await waitFor(() => {
      expect(mocks.importPromptsV2).not.toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalled()
    })
    const errorCalls = errorSpy.mock.calls.map((call) =>
      JSON.stringify(call?.[0] || {})
    )
    expect(
      errorCalls.some((call) => call.includes("File format not recognized"))
    ).toBe(true)
    errorSpy.mockRestore()
  })

  it("filters copilot prompts by search text", async () => {
    state.privateMode = false
    mocks.getAllCopilotPrompts.mockResolvedValue([
      { key: "summary", prompt: "Summarize text: {text}" },
      { key: "translate", prompt: "Translate to French: {text}" },
      { key: "custom", prompt: "Rewrite as concise bullets: {text}" }
    ])

    renderPromptBody(["/prompts?tab=copilot"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("3")
    })

    fireEvent.change(screen.getByTestId("copilot-search"), {
      target: { value: "french" }
    })

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })
  })

  it("copies a copilot prompt into custom drawer defaults", async () => {
    state.privateMode = false
    mocks.getAllCopilotPrompts.mockResolvedValue([
      { key: "summary", prompt: "Summarize this text: {text}" }
    ])

    renderPromptBody(["/prompts?tab=copilot"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("copilot-action-copy-custom-summary"))

    await waitFor(() => {
      expect(screen.getByTestId("mock-prompt-drawer")).toHaveTextContent(
        "Copilot: summary"
      )
    })
  })

  it("copies copilot prompt text to clipboard", async () => {
    state.privateMode = false
    const writeText = vi.fn(async () => undefined)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })

    mocks.getAllCopilotPrompts.mockResolvedValue([
      { key: "translate", prompt: "Translate to English: {text}" }
    ])

    renderPromptBody(["/prompts?tab=copilot"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("copilot-action-copy-clipboard-translate"))

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith("Translate to English: {text}")
    })
  })

  it("shows copilot placeholder helper state and enforces validator before save", async () => {
    state.privateMode = false
    mocks.getAllCopilotPrompts.mockResolvedValue([
      { key: "custom", prompt: "Prompt without placeholder" }
    ])

    renderPromptBody(["/prompts?tab=copilot"])

    await waitFor(() => {
      expect(screen.getByTestId("table-row-count")).toHaveTextContent("1")
    })

    fireEvent.click(screen.getByTestId("copilot-action-edit-custom"))
    expect(screen.getByTestId("copilot-text-placeholder-status")).toHaveTextContent(
      "missing placeholder"
    )

    fireEvent.click(screen.getByTestId("copilot-edit-save"))

    await waitFor(() => {
      expect(mocks.setAllCopilotPrompts).not.toHaveBeenCalled()
    })

    fireEvent.change(screen.getByTestId("copilot-edit-prompt-input"), {
      target: { value: "Now includes {text}" }
    })

    await waitFor(() => {
      expect(screen.getByTestId("copilot-text-placeholder-status")).toHaveTextContent(
        "placeholder detected"
      )
    })

    fireEvent.click(screen.getByTestId("copilot-edit-save"))

    await waitFor(() => {
      expect(mocks.setAllCopilotPrompts).toHaveBeenCalledWith([
        {
          key: "custom",
          prompt: "Now includes {text}"
        }
      ])
    })
  })

  it("keeps copilot edit control focus styles and minimum touch target classes", async () => {
    state.privateMode = false
    mocks.getAllCopilotPrompts.mockResolvedValue([
      { key: "custom", prompt: "Prompt with {text}" }
    ])

    renderPromptBody(["/prompts?tab=copilot"])

    const editButton = await screen.findByTestId("copilot-action-edit-custom")
    expect(editButton.className).toContain("focus:ring-2")
    expect(editButton.className).toContain("focus:ring-primary")
    expect(editButton.className).toContain("min-h-8")
    expect(editButton.className).toContain("min-w-8")
  })

  it("opens keyboard shortcuts help from button and from '?' shortcut", async () => {
    renderPromptBody()

    fireEvent.click(screen.getByTestId("prompts-shortcuts-help-button"))
    expect(
      await screen.findByText("Keyboard shortcuts")
    ).toBeInTheDocument()
    expect(screen.getByText("Create new prompt")).toBeInTheDocument()
    expect(screen.getByText("Open shortcut help")).toBeInTheDocument()

    fireEvent.keyDown(document, { key: "Escape", code: "Escape" })
    fireEvent.keyDown(document, { key: "?", code: "Slash", shiftKey: true })
    expect(await screen.findByText("Keyboard shortcuts")).toBeInTheDocument()
  })
})
