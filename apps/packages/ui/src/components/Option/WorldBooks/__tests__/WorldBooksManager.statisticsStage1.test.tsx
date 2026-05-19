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
    listWorldBookEntries: vi.fn(async () => ({
      entries: [
        {
          entry_id: 11,
          keywords: ["disabled"],
          content: "Disabled lore",
          enabled: false,
          regex_match: false
        },
        {
          entry_id: 12,
          keywords: ["regex"],
          content: "Regex lore",
          enabled: true,
          regex_match: true
        }
      ]
    })),
    addWorldBookEntry: vi.fn(async () => ({})),
    updateWorldBookEntry: vi.fn(async () => ({})),
    deleteWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 0, failed_ids: [] })),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({
      world_book_id: 1,
      name: "Arcana",
      total_entries: 2,
      enabled_entries: 1,
      disabled_entries: 1,
      total_keywords: 3,
      regex_entries: 1,
      case_sensitive_entries: 0,
      average_priority: 42,
      total_content_length: 128,
      estimated_tokens: 32
    })),
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

describe("WorldBooksManager statistics stage-1 actionable metrics", () => {
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
              entry_count: 2
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
        return makeUseQueryResult({
          data: [
            {
              entry_id: 11,
              keywords: ["disabled"],
              content: "Disabled lore",
              enabled: false,
              regex_match: false
            },
            {
              entry_id: 12,
              keywords: ["regex"],
              content: "Regex lore",
              enabled: true,
              regex_match: true
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:selectedWorldBookStatistics") {
        return makeUseQueryResult({
          data: {
            total_entries: 2,
            enabled_entries: 1,
            disabled_entries: 1,
            total_keywords: 3,
            estimated_tokens: 32
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

  it("opens entries drawer with disabled filter when disabled metric is clicked", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open overflow menu then click Statistics
    await user.click(screen.getByRole("button", { name: "More actions for Arcana" }))
    await user.click(await screen.findByText("Statistics"))
    await user.click(await screen.findByRole("button", { name: "Open disabled entries" }))

    expect(await screen.findByText("Entries: Arcana")).toBeInTheDocument()
    expect(screen.getByText("Disabled")).toBeInTheDocument()
    expect(screen.getByText("Disabled lore")).toBeInTheDocument()
    expect(screen.queryByText("Regex lore")).not.toBeInTheDocument()
  }, 15000)

  it("supports keyboard activation for regex drill-down metric", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open overflow menu then click Statistics
    await user.click(screen.getByRole("button", { name: "More actions for Arcana" }))
    await user.click(await screen.findByText("Statistics"))
    const regexButton = await screen.findByRole("button", { name: "Open regex entries" })
    regexButton.focus()
    await user.keyboard("{Enter}")

    await waitFor(() => {
      expect(screen.getByText("Entries: Arcana")).toBeInTheDocument()
    })
    expect(screen.getByText("Regex only")).toBeInTheDocument()
    expect(screen.getByText("Regex lore")).toBeInTheDocument()
    expect(screen.queryByText("Disabled lore")).not.toBeInTheDocument()
  }, 15000)

  it("keeps zero-value drill-down rows non-interactive", async () => {
    const user = userEvent.setup()
    tldwClientMock.worldBookStatistics.mockResolvedValueOnce({
      world_book_id: 1,
      name: "Arcana",
      total_entries: 2,
      enabled_entries: 2,
      disabled_entries: 0,
      total_keywords: 3,
      regex_entries: 0,
      case_sensitive_entries: 0,
      average_priority: 42,
      total_content_length: 128,
      estimated_tokens: 32
    })

    render(<WorldBooksManager />)

    // Open overflow menu then click Statistics
    await user.click(screen.getByRole("button", { name: "More actions for Arcana" }))
    await user.click(await screen.findByText("Statistics"))

    await waitFor(() => {
      expect(screen.getByText("World Book Statistics")).toBeInTheDocument()
    })
    expect(screen.queryByRole("button", { name: "Open disabled entries" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Open regex entries" })).not.toBeInTheDocument()
  }, 15000)

  it("renders live statistics in the detail-panel stats tab", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByText("Arcana"))
    await user.click(screen.getByRole("tab", { name: "Stats" }))

    expect(await screen.findByText("Enabled entries")).toBeInTheDocument()
    expect(screen.getByText("Estimated using ~4 characters per token.")).toBeInTheDocument()
  }, 15000)
})
