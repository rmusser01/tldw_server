import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  undoNotificationMock,
  confirmDangerMock,
  tldwClientMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  notificationMock: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  },
  undoNotificationMock: {
    showUndoNotification: vi.fn()
  },
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    createWorldBook: vi.fn(async () => ({ id: 1 })),
    updateWorldBook: vi.fn(async () => ({})),
    deleteWorldBook: vi.fn(async () => ({})),
    listCharacters: vi.fn(async () => []),
    listCharacterWorldBooks: vi.fn(async () => []),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    addWorldBookEntry: vi.fn(async () => ({})),
    updateWorldBookEntry: vi.fn(async () => ({})),
    deleteWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 1, failed_ids: [] })),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({})),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({}))
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => undoNotificationMock
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => confirmDangerMock
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: null,
  status: "success",
  isLoading: false,
  isFetching: false,
  isPending: false,
  error: null,
  refetch: vi.fn(),
  ...value
})

const makeUseMutationResult = (opts: any) => ({
  mutate: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    }
  },
  mutateAsync: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    }
  },
  isPending: false
})

const entryData = Array.from({ length: 12 }, (_, index) => ({
  entry_id: index + 1,
  keywords: [`keyword-${index + 1}`],
  content: `Entry content ${index + 1}`,
  priority: index % 100,
  enabled: true,
  case_sensitive: false,
  regex_match: false,
  whole_word_match: true,
  appendable: false
}))

describe("WorldBooksManager bulk operations stage-1 selection UX", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({
          data: [
            {
              id: 1,
              name: "Arcana",
              description: "Main lore",
              enabled: true,
              entry_count: entryData.length
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
      }
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({ data: entryData, status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("shows the contextual action bar only after at least one row is selected", async () => {
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    fireEvent.click(screen.getByText("Arcana"))
    const keywordsHeader = await screen.findByRole("columnheader", { name: "Keywords" })
    expect(screen.queryByText("1 selected")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Enable" })).not.toBeInTheDocument()

    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    const checkboxes = within(tableWrapper as HTMLElement).getAllByRole("checkbox")
    fireEvent.click(checkboxes[1])

    expect(screen.getByText("1 selected")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Enable" })).toBeInTheDocument()
  }, 30000)

  it("supports select-all escalation and clear selection", async () => {
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    fireEvent.click(screen.getByText("Arcana"))
    const keywordsHeader = await screen.findByRole("columnheader", { name: "Keywords" })

    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    const checkboxes = within(tableWrapper as HTMLElement).getAllByRole("checkbox")
    fireEvent.click(checkboxes[1])
    fireEvent.click(screen.getByRole("button", { name: "Select all 12 entries" }))

    await waitFor(() => {
      expect(screen.getByText("12 selected")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Clear selected entries" }))
    await waitFor(() => {
      expect(screen.queryByText("12 selected")).not.toBeInTheDocument()
      expect(screen.queryByRole("button", { name: "Enable" })).not.toBeInTheDocument()
    })
  }, 45000)

  it("allows keyboard activation for the select-all escalation action", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    const keywordsHeader = await screen.findByRole("columnheader", { name: "Keywords" })

    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    const checkboxes = within(tableWrapper as HTMLElement).getAllByRole("checkbox")
    await user.click(checkboxes[1])

    const selectAllButton = screen.getByRole("button", { name: "Select all 12 entries" })
    selectAllButton.focus()
    await user.keyboard("{Enter}")

    expect(screen.getByText("12 selected")).toBeInTheDocument()
  }, 30000)
})
