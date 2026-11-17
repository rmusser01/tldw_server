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
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockSetSetting: vi.fn(),
  mockClearSetting: vi.fn()
}))

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

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: mockMessageInfo
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
    setSetting: mockSetSetting,
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
  default: () => <div data-testid="notes-list-panel" />
}))

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: ({ onExport }: any) => (
    <div>
      <button data-testid="print-note-action" onClick={() => onExport("print")}>
        Print note
      </button>
    </div>
  )
}))

const createPrintWindow = () => {
  const printWindow = {
    opener: {} as unknown,
    document: {
      open: vi.fn(),
      write: vi.fn((_html: string) => expect(printWindow.opener).toBeNull()),
      close: vi.fn()
    },
    focus: vi.fn(),
    print: vi.fn()
  }
  return printWindow
}

const openPrintWindow = (printWindow: ReturnType<typeof createPrintWindow>, features = "") => {
  const options = new Map(features.split(",").map((feature) => {
    const [name, value] = feature.trim().toLowerCase().split("=")
    return [name, value] as const
  }))
  if (["noopener", "noreferrer"].some((name) => options.has(name) && !["0", "no", "false"].includes(options.get(name) ?? ""))) return null
  return printWindow as unknown as Window
}

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

describe("NotesManagerPage stage 37 print export", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      return {}
    })
  })

  it("opens a print window with print-safe html for single-note export", async () => {
    const mockPrintWindow = createPrintWindow()
    const openSpy = vi.spyOn(window, "open").mockImplementation((_url, _target, features) => openPrintWindow(mockPrintWindow, features))

    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: 'Printable note title <img src=x onerror="alert(1)">' }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Printable **body** content <script>alert(1)</script>" }
    })
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(openSpy).toHaveBeenCalled()
      expect(mockPrintWindow.document.write).toHaveBeenCalledTimes(1)
    })

    const features = String(openSpy.mock.calls[0]?.[2]).split(",")
    expect(features).toEqual(expect.arrayContaining(["width=1024", "height=768"]))
    expect(features.some((feature) => /^(noopener|noreferrer)(=|$)/i.test(feature))).toBe(false)

    const html = String(mockPrintWindow.document.write.mock.calls[0]?.[0] || "")
    expect(html).toContain("@media print")
    expect(html).toContain("Printable note title")
    expect(html).toContain("<strong>body</strong>")
    expect(html).toContain("Printable note title &lt;img")
    expect(html).not.toContain("<script>")
    expect(new DOMParser().parseFromString(html, "text/html").querySelector("[onerror]")).toBeNull()
    expect(mockPrintWindow.focus).toHaveBeenCalled()
    expect(mockPrintWindow.print).toHaveBeenCalled()
    expect(mockMessageSuccess).toHaveBeenCalledWith(
      "Opened print view. Use Save as PDF in your browser to export PDF."
    )
  })

  it("shows an actionable error when the print pop-up cannot be opened", async () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    const printSpy = vi.spyOn(window, "print")

    renderPage()
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(mockMessageError).toHaveBeenCalledWith(
        "Unable to open print view. Please allow pop-ups and try again."
      )
    })
    expect(mockMessageSuccess).not.toHaveBeenCalled()
    expect(printSpy).not.toHaveBeenCalled()
  })
})
