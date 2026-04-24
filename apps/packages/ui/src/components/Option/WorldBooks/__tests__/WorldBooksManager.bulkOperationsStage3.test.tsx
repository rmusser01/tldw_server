import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor, within } from "@testing-library/react"
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

const sourceEntries = [
  {
    entry_id: 1,
    keywords: ["alpha"],
    content: "Duplicate entry",
    priority: 70,
    enabled: false,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  },
  {
    entry_id: 2,
    keywords: ["beta"],
    content: "Move me",
    priority: 23,
    enabled: true,
    case_sensitive: true,
    regex_match: false,
    whole_word_match: false,
    appendable: true
  }
]

describe("WorldBooksManager bulk operations stage-3 move workflow", () => {
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
              description: "Source",
              enabled: true,
              entry_count: sourceEntries.length
            },
            {
              id: 2,
              name: "Archive",
              description: "Destination",
              enabled: true,
              entry_count: 1
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
        return makeUseQueryResult({ data: sourceEntries, status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("moves selected entries with skip-existing strategy and preserves metadata", async () => {
    const user = userEvent.setup()
    tldwClientMock.listWorldBookEntries.mockResolvedValueOnce({
      entries: [
        {
          entry_id: 101,
          keywords: ["alpha"],
          content: "Duplicate entry"
        }
      ]
    })

    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    const keywordsHeader = await screen.findByRole("columnheader", { name: "Keywords" })
    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    const checkboxes = within(tableWrapper as HTMLElement).getAllByRole("checkbox")
    await user.click(checkboxes[1])
    await user.click(checkboxes[2])

    await user.click(screen.getByRole("button", { name: "Move To" }))
    await user.click(screen.getByRole("combobox", { name: "Bulk move destination" }))
    await user.click(await screen.findByText("Archive", { selector: ".ant-select-item-option-content" }))
    await user.click(screen.getByRole("button", { name: "Move Entries" }))

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalled()
    })
    expect(tldwClientMock.listWorldBookEntries).toHaveBeenCalledWith(2, false)
    expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledTimes(1)
    expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledWith(
      2,
      expect.objectContaining({
        keywords: ["beta"],
        content: "Move me",
        priority: 23,
        enabled: true,
        case_sensitive: true,
        whole_word_match: false,
        appendable: true
      })
    )
    expect(tldwClientMock.bulkWorldBookEntries).toHaveBeenCalledWith({
      entry_ids: [2],
      operation: "delete"
    })
    expect(notificationMock.warning).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Move completed with warnings" })
    )
  }, 60000)
})
