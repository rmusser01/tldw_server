import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"

const {
  mockWebSearch,
  mockAddMedia,
  mockAddSource,
  mockMessageWarning
} = vi.hoisted(() => ({
  mockWebSearch: vi.fn(),
  mockAddMedia: vi.fn(),
  mockAddSource: vi.fn(),
  mockMessageWarning: vi.fn()
}))

const workspaceStoreState = {
  addSourceModalOpen: true,
  addSourceModalTab: "search" as const,
  addSourceProcessing: false,
  addSourceError: null as string | null,
  sources: [] as Array<{ mediaId: number }>,
  closeAddSourceModal: vi.fn(),
  setAddSourceModalTab: vi.fn(),
  setAddSourceProcessing: vi.fn(),
  setAddSourceError: vi.fn(),
  addSource: mockAddSource,
  workspaceTag: "workspace:test"
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          },
      interpolationOptions?: Record<string, unknown>
    ) => {
      const renderTemplate = (value: string) =>
        value.replace(/\{\{(\w+)\}\}/g, (_m, token) =>
          String(interpolationOptions?.[token] ?? "")
        )

      if (typeof defaultValueOrOptions === "string") {
        return renderTemplate(defaultValueOrOptions)
      }
      if (defaultValueOrOptions?.defaultValue) {
        return renderTemplate(defaultValueOrOptions.defaultValue)
      }
      return _key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    uploadMedia: vi.fn(),
    addMedia: mockAddMedia,
    webSearch: mockWebSearch,
    searchMedia: vi.fn().mockResolvedValue({ results: [] }),
    listMedia: vi.fn().mockResolvedValue({ media: [] }),
    updateMediaKeywords: vi.fn().mockResolvedValue(undefined)
  }
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: {
      ...actual.message,
      warning: mockMessageWarning
    }
  }
})

describe("AddSourceModal Stage 1 error surfaces", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.addSourceModalOpen = true
    workspaceStoreState.addSourceModalTab = "search"
    workspaceStoreState.addSourceProcessing = false
    workspaceStoreState.addSourceError = null

    mockWebSearch.mockResolvedValue({
      results: [
        {
          title: "Result One",
          url: "https://example.com/one"
        },
        {
          title: "Result Two",
          url: "https://example.com/two"
        }
      ]
    })

    mockAddMedia
      .mockResolvedValueOnce({ results: [{ media_id: 9001, title: "One" }] })
      .mockRejectedValueOnce(new Error("timeout"))
  })

  it("reports partial batch URL ingestion failures with actionable summary", async () => {
    render(<AddSourceModal />)

    fireEvent.change(screen.getByPlaceholderText("Search the web..."), {
      target: { value: "example" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Search" }))

    expect(await screen.findByText("Result One")).toBeInTheDocument()
    expect(screen.getByText("Result Two")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Result One"))
    fireEvent.click(screen.getByText("Result Two"))
    fireEvent.click(screen.getByRole("button", { name: "Add 2 selected" }))

    await waitFor(() => {
      expect(mockAddSource).toHaveBeenCalledTimes(1)
      expect(mockMessageWarning).toHaveBeenCalledWith(
        expect.stringContaining("Added 1 of 2 sources")
      )
      expect(mockMessageWarning).toHaveBeenCalledWith(
        expect.stringContaining("https://example.com/two")
      )
    })
  })
})
