import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  undoNotificationMock,
  confirmDangerMock,
  tldwClientMock,
  mockBreakpoints
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
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 0, failed_ids: [] })),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({})),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({})),
    processWorldBookContext: vi.fn(async () => ({
      injected_content: "",
      entries_matched: 0,
      tokens_used: 0,
      books_used: 1,
      entry_ids: [],
      diagnostics: []
    }))
  },
  mockBreakpoints: { md: false }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  return {
    ...actual,
    Grid: {
      ...actual.Grid,
      useBreakpoint: () => mockBreakpoints
    }
  }
})

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

describe("WorldBooksManager responsive stage-2 entry drawer mobile ergonomics", () => {
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
              entry_count: 1,
              token_budget: 500
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {} })
      }
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({
          data: [
            {
              entry_id: 10,
              keywords: ["seed-keyword"],
              content: "Seed entry",
              priority: 75,
              enabled: true,
              case_sensitive: false,
              regex_match: false,
              whole_word_match: true,
              appendable: false
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it(
    "hides priority/enabled columns and applies larger touch targets on mobile",
    async () => {
      const user = userEvent.setup()
      mockBreakpoints.md = false
      render(<WorldBooksManager />)

      await user.click(screen.getByText("Arcana"))
      const detailPanel = await screen.findByRole("main", { name: "World book detail" })

      expect(within(detailPanel).queryAllByRole("columnheader", { name: "Priority" })).toHaveLength(0)
      expect(within(detailPanel).queryAllByRole("columnheader", { name: "Enabled" })).toHaveLength(0)

      const editEntryButton = within(detailPanel).getByRole("button", { name: "Edit entry" })
      const deleteEntryButton = within(detailPanel).getByRole("button", { name: "Delete entry" })
      expect(editEntryButton).toHaveClass("min-h-11")
      expect(editEntryButton).toHaveClass("min-w-11")
      expect(deleteEntryButton).toHaveClass("min-h-11")
      expect(deleteEntryButton).toHaveClass("min-w-11")
    },
    30000
  )

  it(
    "keeps priority/enabled columns visible on desktop",
    async () => {
      const user = userEvent.setup()
      mockBreakpoints.md = true
      render(<WorldBooksManager />)

      await user.click(screen.getByText("Arcana"))
      const detailPanel = await screen.findByRole("main", { name: "World book detail" })

      expect(within(detailPanel).queryAllByRole("columnheader", { name: "Priority" })).toHaveLength(1)
      expect(within(detailPanel).queryAllByRole("columnheader", { name: "Enabled" })).toHaveLength(1)

      const editEntryButton = within(detailPanel).getByRole("button", { name: "Edit entry" })
      expect(editEntryButton.className).not.toContain("min-h-11")
      expect(editEntryButton.className).not.toContain("min-w-11")
    },
    15000
  )
})
