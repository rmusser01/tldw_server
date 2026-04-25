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
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 2, failed_ids: [] })),
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

const entryData = [
  {
    entry_id: 1,
    keywords: ["wizard"],
    content: "Entry content 1",
    priority: 50,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  },
  {
    entry_id: 2,
    keywords: ["alchemy"],
    content: "Entry content 2",
    priority: 40,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  },
  {
    entry_id: 3,
    keywords: ["history"],
    content: "Entry content 3",
    priority: 30,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  }
]

describe("WorldBooksManager bulk operations stage-2 set-priority action", () => {
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

  const selectTwoEntries = async (user: ReturnType<typeof userEvent.setup>) => {
    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    const keywordsHeader = await screen.findByRole("columnheader", { name: "Keywords" })
    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    const checkboxes = within(tableWrapper as HTMLElement).getAllByRole("checkbox")
    await user.click(checkboxes[1])
    await user.click(checkboxes[2])
  }

  it("submits set_priority with clamped value and selected ids", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)
    await selectTwoEntries(user)

    await user.click(screen.getByRole("button", { name: "Set Priority" }))
    await user.click(screen.getByRole("button", { name: "Apply Priority" }))

    await waitFor(() => {
      expect(tldwClientMock.bulkWorldBookEntries).toHaveBeenCalledWith({
        entry_ids: [1, 2],
        operation: "set_priority",
        priority: 50
      })
    })
    expect(notificationMock.success).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Bulk priority updated" })
    )
  }, 30000)

  it("shows warning when set-priority returns partial failures", async () => {
    const user = userEvent.setup()
    tldwClientMock.bulkWorldBookEntries.mockResolvedValueOnce({
      success: false,
      affected_count: 1,
      failed_ids: [2]
    } as any)

    render(<WorldBooksManager />)
    await selectTwoEntries(user)

    await user.click(screen.getByRole("button", { name: "Set Priority" }))
    fireEvent.change(screen.getByLabelText("Bulk priority value"), { target: { value: "80" } })
    await user.click(screen.getByRole("button", { name: "Apply Priority" }))

    await waitFor(() => {
      expect(notificationMock.warning).toHaveBeenCalledWith(
        expect.objectContaining({ message: "Bulk priority completed with errors" })
      )
    })
  }, 30000)

  it("surfaces error messaging when set-priority request fails", async () => {
    const user = userEvent.setup()
    tldwClientMock.bulkWorldBookEntries.mockRejectedValueOnce(new Error("Network unavailable"))

    render(<WorldBooksManager />)
    await selectTwoEntries(user)

    await user.click(screen.getByRole("button", { name: "Set Priority" }))
    await user.click(screen.getByRole("button", { name: "Apply Priority" }))

    await waitFor(() => {
      expect(notificationMock.error).toHaveBeenCalledWith(
        expect.objectContaining({ description: "Network unavailable" })
      )
    })
  }, 30000)
})
