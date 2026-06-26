import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useQuery, type UseQueryResult } from "@tanstack/react-query"
import { SourceSelector } from "../SourceSelector"
import type { DataTableSource } from "@/types/data-tables"

const {
  sourceState,
  useQueryMock,
  refetchMock
} = vi.hoisted(() => ({
  sourceState: {
    selectedSources: [
      {
        type: "document" as const,
        id: "doc-1",
        title: "Annual report",
        snippet: "Selected before the refresh failed"
      }
    ],
    activeSourceType: "chat" as "chat" | "document" | "rag_query",
    sourceSearchQuery: "budget",
    addSource: vi.fn(),
    removeSource: vi.fn(),
    setActiveSourceType: vi.fn(),
    setSourceSearchQuery: vi.fn()
  },
  useQueryMock: vi.fn(),
  refetchMock: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock
}))

vi.mock("@/store/data-tables", () => ({
  useDataTablesStore: (selector: (state: typeof sourceState) => unknown) =>
    selector(sourceState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listChats: vi.fn(),
    listMedia: vi.fn(),
    searchMedia: vi.fn()
  }
}))

describe("SourceSelector recovery states", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sourceState.selectedSources = [
      {
        type: "document",
        id: "doc-1",
        title: "Annual report",
        snippet: "Selected before the refresh failed"
      }
    ]
    sourceState.activeSourceType = "chat"
    sourceState.sourceSearchQuery = "budget"
    refetchMock.mockReset()
  })

  it("renders a shared recovery state when chat sources fail to load", () => {
    const rawError = "/api/v1/chats?limit=50 returned 404"
    vi.mocked(useQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isFetching: false,
      isError: true,
      error: new Error(rawError),
      errorUpdatedAt: 1,
      refetch: refetchMock
    } as unknown as UseQueryResult<DataTableSource[], Error>)
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    try {
      render(<SourceSelector />)
    } finally {
      errorSpy.mockRestore()
    }

    expect(screen.getByText(/Selected Sources/)).toBeInTheDocument()
    expect(screen.getByText("Annual report")).toBeInTheDocument()
    expect(screen.getByDisplayValue("budget")).toBeInTheDocument()

    const recovery = screen.getByTestId("data-tables-source-load-recovery")
    expect(recovery).toHaveAttribute("data-ds-component", "StatePanel")
    expect(
      within(recovery).getByText("Data sources could not load")
    ).toBeInTheDocument()
    expect(within(recovery).getByText(rawError).closest("dl")).toHaveAttribute(
      "aria-label",
      "Diagnostics"
    )

    fireEvent.click(within(recovery).getByRole("button", { name: "Try again" }))
    expect(refetchMock).toHaveBeenCalledTimes(1)
  })
})
