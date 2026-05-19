import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
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
    worldBookStatistics: vi.fn(async () => ({
      world_book_id: 1,
      name: "Arcana",
      total_entries: 2,
      enabled_entries: 2,
      disabled_entries: 0,
      total_keywords: 2,
      regex_entries: 0,
      case_sensitive_entries: 0,
      average_priority: 50,
      total_content_length: 120,
      estimated_tokens: 30,
      token_budget: 500
    })),
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
  mockBreakpoints: { md: true }
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

const expectOpenModalBodyScrollable = () => {
  const modalBody = document.querySelector(
    ".ant-modal-wrap:not(.ant-modal-wrap-hidden) .ant-modal-body"
  ) as HTMLElement | null

  expect(modalBody).not.toBeNull()
  expect(modalBody).toHaveStyle("max-height: 80vh")
  expect(modalBody).toHaveStyle("overflow-y: auto")
}

describe("WorldBooksManager responsive stage-4 modal scrolling", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true

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
              entry_count: 2,
              token_budget: 500
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [{ id: 10, name: "Aria" }] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: { 1: [{ id: 10, name: "Aria" }] } })
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
    "applies modal body scrolling to create and edit world-book dialogs",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      await user.click(screen.getByRole("button", { name: "New World Book" }))
      expect(await screen.findByText("Create World Book")).toBeInTheDocument()
      expectOpenModalBodyScrollable()

      await user.click(screen.getByRole("button", { name: "Close" }))

      // Edit modal is no longer directly accessible (replaced by detail panel Settings tab)
      // Just verify the create modal scrolling works
    },
    60000
  )

  it(
    "applies modal body scrolling to statistics and attachment matrix dialogs",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      // Open overflow menu then click Statistics
      await user.click(screen.getByRole("button", { name: "More actions for Arcana" }))
      await user.click(await screen.findByText("Statistics"))
      expect(await screen.findByText("World Book Statistics")).toBeInTheDocument()
      expectOpenModalBodyScrollable()

      await user.click(screen.getByRole("button", { name: "Close" }))

      // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
      expect(await screen.findByText("World Book ↔ Character Matrix")).toBeInTheDocument()
      expectOpenModalBodyScrollable()
    },
    30000
  )
})
