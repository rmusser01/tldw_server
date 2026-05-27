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
  default: ({ onImportNotes, importInProgress }: any) => (
    <div>
      <button data-testid="trigger-import-picker" onClick={() => onImportNotes?.()}>
        Open import picker
      </button>
      <div data-testid="import-in-progress">{importInProgress ? "loading" : "idle"}</div>
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

describe("NotesManagerPage stage 36 import workflow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("previews selected files and submits import payload with duplicate strategy", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
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
      if (path === "/api/v1/notes/import" && method === "POST") {
        return {
          created_count: 2,
          updated_count: 0,
          skipped_count: 0,
          failed_count: 0
        }
      }
      return {}
    })

    renderPage()

    const importInput = screen.getByTestId("notes-import-input") as HTMLInputElement
    const jsonFile = new File(
      [JSON.stringify([{ title: "Imported title", content: "Imported content", keywords: ["research"] }])],
      "import-batch.json",
      { type: "application/json" }
    )

    fireEvent.change(importInput, {
      target: { files: [jsonFile] }
    })

    await waitFor(() => {
      expect(screen.getByTestId("notes-import-modal")).toBeInTheDocument()
    })
    expect(screen.getByTestId("notes-import-preview-summary")).toHaveTextContent("Files: 1")
    expect(screen.getByTestId("notes-import-preview-summary")).toHaveTextContent("Estimated notes: 1")

    fireEvent.change(screen.getByTestId("notes-import-duplicate-strategy"), {
      target: { value: "skip" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Import notes" }))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/import",
          method: "POST",
          body: expect.objectContaining({ duplicate_strategy: "skip" })
        })
      )
    })
    const importCall = mockBgRequest.mock.calls
      .map(([request]) => request)
      .find((request) => request.path === "/api/v1/notes/import")
    expect(importCall?.body?.items?.[0]).toMatchObject({
      file_name: "import-batch.json",
      format: "json"
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith("Imported 2 notes.")
  })

  it("shows warning toast for partial import results", async () => {
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
      if (path === "/api/v1/notes/import" && method === "POST") {
        return {
          created_count: 1,
          updated_count: 0,
          skipped_count: 1,
          failed_count: 1
        }
      }
      return {}
    })

    renderPage()

    const importInput = screen.getByTestId("notes-import-input") as HTMLInputElement
    const importFile = new File([JSON.stringify([{ title: "Imported title", content: "Imported body" }])], "partial-import.json", {
      type: "application/json"
    })
    fireEvent.change(importInput, {
      target: { files: [importFile] }
    })

    expect(await screen.findByTestId("notes-import-modal")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Import notes" }))

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith(
        expect.stringContaining("Import completed with partial results")
      )
    })
  })

  it("blocks submit when selected files have parse errors and preserves the preview", async () => {
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
      if (path === "/api/v1/notes/import" && method === "POST") {
        throw new Error("invalid import should not submit")
      }
      return {}
    })

    renderPage()

    const importInput = screen.getByTestId("notes-import-input") as HTMLInputElement
    const invalidJsonFile = new File(["{not-json"], "broken-import.json", {
      type: "application/json"
    })
    fireEvent.change(importInput, {
      target: { files: [invalidJsonFile] }
    })

    expect(await screen.findByText("Could not parse notes from this JSON file.")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Import notes" }))

    expect(mockMessageWarning).toHaveBeenCalledWith(
      "Fix or reselect files with import errors before importing."
    )
    expect(screen.getByTestId("notes-import-modal")).toBeInTheDocument()
    expect(
      mockBgRequest.mock.calls.some(([request]) => request.path === "/api/v1/notes/import")
    ).toBe(false)
  })
})
