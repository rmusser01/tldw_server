import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
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
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 0, failed_ids: [] })),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({ world_book_id: 99, entries_imported: 1, merged: false })),
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

describe("WorldBooksManager statistics stage-3 global statistics view", () => {
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
              entry_count: 2,
              token_budget: 100
            },
            {
              id: 2,
              name: "Bestiary",
              description: "Creatures",
              enabled: true,
              entry_count: 1,
              token_budget: 150
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
      if (key === "tldw:worldBookGlobalStatistics") {
        return makeUseQueryResult({
          data: {
            totalBooks: 2,
            totalEntries: 3,
            totalKeywords: 5,
            totalEstimatedTokens: 90,
            totalTokenBudget: 250,
            sharedKeywordCount: 1,
            conflictKeywordCount: 1,
            conflicts: [
              {
                keyword: "dragon",
                worldBookIds: [1, 2],
                worldBookNames: ["Arcana", "Bestiary"],
                affectedBooks: [
                  { id: 1, name: "Arcana" },
                  { id: 2, name: "Bestiary" }
                ],
                variantCount: 2,
                occurrenceCount: 2
              }
            ]
          },
          status: "success"
        })
      }
      if (key === "tldw:listWorldBookEntries") {
        const worldBookId = Number(queryKey[1])
        if (worldBookId === 1) {
          return makeUseQueryResult({
            data: [
              {
                entry_id: 11,
                keywords: ["dragon"],
                content: "Arcana dragon lore",
                enabled: true,
                regex_match: false
              },
              {
                entry_id: 12,
                keywords: ["city"],
                content: "Arcana city lore",
                enabled: true,
                regex_match: false
              }
            ],
            status: "success"
          })
        }
        return makeUseQueryResult({ data: [], status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("renders aggregate metrics and conflict listings in global stats modal", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Global Statistics
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Global Statistics"))

    await waitFor(() => {
      expect(screen.getByText("Global World Book Statistics")).toBeInTheDocument()
    })
    expect(screen.getByText("90/250 (36.0%)")).toBeInTheDocument()
    expect(screen.getByText("dragon")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open conflict keyword dragon in Arcana" })
    ).toBeInTheDocument()
  }, 15000)

  it("drills down from conflict row into entries filtered by keyword", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Global Statistics
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Global Statistics"))
    await user.click(await screen.findByRole("button", { name: "Open conflict keyword dragon in Arcana" }))

    await waitFor(() => {
      expect(screen.getByText("Entries: Arcana")).toBeInTheDocument()
    })
    expect(screen.getByLabelText("Search entries")).toHaveValue("dragon")
    expect(screen.getByText("Arcana dragon lore")).toBeInTheDocument()
    expect(screen.queryByText("Arcana city lore")).not.toBeInTheDocument()
  }, 15000)
})
