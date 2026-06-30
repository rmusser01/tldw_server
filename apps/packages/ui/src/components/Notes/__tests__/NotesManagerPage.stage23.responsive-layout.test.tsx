import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  responsiveState
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  responsiveState: {
    width: 1024,
    isMobile: false
  }
}))

const setViewportWidth = (width: number) => {
  responsiveState.width = width
  responsiveState.isMobile = width < 768
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => responsiveState.isMobile
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: vi.fn()
  })
}))

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: vi.fn(async () => []),
  searchNoteKeywords: vi.fn(async () => [])
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn()
    })
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mockGetSetting,
    clearSetting: mockClearSetting
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: { onSelectNote: (id: string) => void }) => (
    <div data-testid="notes-list-panel">
      <button
        type="button"
        data-testid="notes-list-select-note-1"
        onClick={() => onSelectNote("note-1")}
      >
        Select note 1
      </button>
    </div>
  )
}))

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

describe("NotesManagerPage stage 23 responsive mobile layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setViewportWidth(1024)
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 1, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      if (path === "/api/v1/notes/note-1" && method === "GET") {
        return {
          id: "note-1",
          title: "Selected note title",
          content: "Selected note content",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T10:00:00.000Z"
        }
      }
      return {}
    })
  })

  it("uses mobile single-panel mode below 768 and desktop split mode at 768+", async () => {
    const { rerender } = renderPage()

    expect(screen.getByTestId("notes-desktop-sidebar-toggle")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-mobile-open-list-button")).not.toBeInTheDocument()

    for (const width of [320, 375]) {
      setViewportWidth(width)
      rerender(
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false }
              }
            })
          }
        >
          <NotesManagerPage />
        </QueryClientProvider>
      )

      await waitFor(() => {
        expect(screen.getByTestId("notes-mobile-open-list-button")).toBeInTheDocument()
      })
      expect(screen.queryByTestId("notes-desktop-sidebar-toggle")).not.toBeInTheDocument()
    }

    for (const width of [768, 1024]) {
      setViewportWidth(width)
      rerender(
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false }
              }
            })
          }
        >
          <NotesManagerPage />
        </QueryClientProvider>
      )

      await waitFor(() => {
        expect(screen.getByTestId("notes-desktop-sidebar-toggle")).toBeInTheDocument()
      })
      expect(screen.queryByTestId("notes-mobile-open-list-button")).not.toBeInTheDocument()
    }
  })

  it("restores desktop sidebar collapse state after returning from mobile", async () => {
    const { rerender } = renderPage()
    const listRegion = screen.getByTestId("notes-list-region")
    expect(listRegion.className).toContain("w-[300px]")

    fireEvent.click(screen.getByTestId("notes-desktop-sidebar-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("notes-list-region").className).toContain("w-0 overflow-hidden")
    })

    setViewportWidth(375)
    rerender(
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: {
              queries: { retry: false },
              mutations: { retry: false }
            }
          })
        }
      >
        <NotesManagerPage />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("notes-list-region").className).toContain("-translate-x-full")
    })

    setViewportWidth(1024)
    rerender(
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: {
              queries: { retry: false },
              mutations: { retry: false }
            }
          })
        }
      >
        <NotesManagerPage />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId("notes-list-region").className).toContain("w-0 overflow-hidden")
    })
  })

  it("keeps selected note loaded across mobile/desktop mode switches and closes mobile list on select", async () => {
    setViewportWidth(375)
    const { rerender } = renderPage()

    fireEvent.click(screen.getByTestId("notes-mobile-open-list-button"))
    await waitFor(() => {
      expect(screen.getByTestId("notes-list-region").className).toContain("translate-x-0")
    })

    fireEvent.click(screen.getByTestId("notes-list-select-note-1"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue("Selected note title")
    })
    expect(screen.getByTestId("notes-list-region").className).toContain("-translate-x-full")

    setViewportWidth(1024)
    rerender(
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: {
              queries: { retry: false },
              mutations: { retry: false }
            }
          })
        }
      >
        <NotesManagerPage />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue("Selected note title")
    })
  })

  it("keeps mobile controls non-overlapping and honors reduced-motion users", async () => {
    setViewportWidth(375)
    renderPage()

    await waitFor(() => {
      expect(screen.getByTestId("notes-mobile-open-list-button")).toBeInTheDocument()
    })

    expect(screen.queryByTestId("notes-create-study-pack-button")).not.toBeInTheDocument()
    expect(screen.getByTestId("notes-list-region").className).toContain(
      "motion-reduce:transition-none"
    )
  })

  it("keeps the first-draft title field usable on mobile", async () => {
    setViewportWidth(375)
    renderPage()

    fireEvent.click(screen.getByTestId("notes-editor-empty-create"))

    const titleInput = await screen.findByRole("textbox", { name: "Note title" })
    expect(screen.getByTestId("notes-title-row")).toHaveClass("flex-wrap")
    expect(titleInput).toHaveClass("!min-w-full")
    expect(titleInput).toHaveClass("sm:!min-w-[18rem]")
    expect(titleInput).toHaveClass("flex-1")
  })
})
