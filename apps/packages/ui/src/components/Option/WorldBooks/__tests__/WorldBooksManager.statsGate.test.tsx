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
    listWorldBooks: vi.fn(async () => ({
      world_books: [{ id: 1, name: "Arcana", entry_count: 2 }]
    })),
    listCharacters: vi.fn(async () => []),
    listCharacterWorldBooks: vi.fn(async () => []),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    worldBookStatistics: vi.fn(async () => ({ world_book_id: 1, name: "Arcana" }))
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

const latestQueryOptions = (key: string) => {
  const matches = useQueryMock.mock.calls
    .map(([options]: [any]) => options)
    .filter(
      (options: any) =>
        Array.isArray(options?.queryKey) && options.queryKey[0] === key
    )
  return matches[matches.length - 1]
}

const globalStatsQueryOptions = () =>
  latestQueryOptions("tldw:worldBookGlobalStatistics")

const selectedBookStatsQueryOptions = () =>
  latestQueryOptions("tldw:selectedWorldBookStatistics")

describe("WorldBooksManager stats fetch gating (WP-47)", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => ({
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      isPending: false
    }))
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
              token_budget: 100
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:worldBookGlobalStatistics") {
        return makeUseQueryResult({
          data: {
            totalBooks: 1,
            totalEntries: 2,
            totalKeywords: 3,
            totalEstimatedTokens: 40,
            totalTokenBudget: 100,
            sharedKeywordCount: 0,
            conflictKeywordCount: 0,
            conflicts: []
          },
          status: "success"
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("does not enable the worldbook stats fetch-all until the stats panel opens", async () => {
    render(<WorldBooksManager />)

    // Books are listed, but the per-book-entries fetch-all stays gated off.
    expect(await screen.findByText("Arcana")).toBeInTheDocument()
    const gated = globalStatsQueryOptions()
    expect(gated).toBeDefined()
    expect(gated.enabled).toBe(false)
    expect(tldwClientMock.listWorldBookEntries).not.toHaveBeenCalled()
    // Per-book stats stay gated until the Stats tab is opened too.
    const bookStats = selectedBookStatsQueryOptions()
    expect(bookStats).toBeDefined()
    expect(bookStats.enabled).toBe(false)
    expect(tldwClientMock.worldBookStatistics).not.toHaveBeenCalled()
  })

  it("enables the fetch-all only after Global Statistics is opened", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(await screen.findByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Global Statistics"))

    const enabledOptions = globalStatsQueryOptions()
    expect(enabledOptions).toBeDefined()
    expect(enabledOptions.enabled).toBe(true)
  }, 15000)
})
