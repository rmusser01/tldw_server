import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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
    importWorldBook: vi.fn(async () => ({})),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({})),
    processWorldBookContext: vi.fn()
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

const worldBooks = [
  {
    id: 1,
    name: "Arcana",
    description: "Main lore",
    scan_depth: 4,
    token_budget: 600,
    recursive_scanning: true,
    enabled: true,
    entry_count: 2
  }
]

describe("WorldBooksManager cross-feature integration stage-2 test matching", () => {
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
        return makeUseQueryResult({ data: worldBooks, status: "success" })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [], status: "success" })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
      }
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({ data: [], status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it(
    "runs test matching from manager modal and supports iterative reruns",
    async () => {
      const user = userEvent.setup()
    tldwClientMock.processWorldBookContext
      .mockResolvedValueOnce({
        injected_content: "",
        entries_matched: 1,
        tokens_used: 42,
        books_used: 1,
        entry_ids: [100],
        token_budget: 600,
        budget_exhausted: false,
        skipped_entries_due_to_budget: 0,
        diagnostics: [
          {
            entry_id: 100,
            world_book_id: 1,
            activation_reason: "keyword_match",
            keyword: "castle",
            token_cost: 42,
            priority: 80,
            regex_match: false,
            content_preview: "Castle lore details"
          }
        ]
      })
      .mockResolvedValueOnce({
        injected_content: "",
        entries_matched: 0,
        tokens_used: 8,
        books_used: 1,
        entry_ids: [],
        token_budget: 600,
        budget_exhausted: false,
        skipped_entries_due_to_budget: 0,
        diagnostics: []
      })

    render(<WorldBooksManager />)

    // Open Tools dropdown then click Test Matching
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Test Matching"))
    fireEvent.change(screen.getByRole("textbox", { name: "Sample text for keyword test" }), {
      target: { value: "castle walls and siege history" }
    })
    await user.click(screen.getByRole("button", { name: "Run keyword test" }))

    await waitFor(() => {
      expect(tldwClientMock.processWorldBookContext).toHaveBeenCalledWith({
        text: "castle walls and siege history",
        world_book_ids: [1],
        scan_depth: 4,
        token_budget: 600,
        recursive_scanning: true
      })
    })

    expect(await screen.findByText("Entries matched")).toBeInTheDocument()
    expect(screen.getByText("Books used")).toBeInTheDocument()
    expect(screen.getByText("Tokens used")).toBeInTheDocument()
    expect(screen.getByText("Token budget")).toBeInTheDocument()
    expect(screen.getByText("Keyword match: castle")).toBeInTheDocument()
    expect(screen.getByText("42")).toBeInTheDocument()
    expect(screen.getByText("600")).toBeInTheDocument()

    fireEvent.change(screen.getByRole("textbox", { name: "Sample text for keyword test" }), {
      target: { value: "this text should match nothing" }
    })
    await user.click(screen.getByRole("button", { name: "Run keyword test" }))

    await waitFor(() => {
      expect(tldwClientMock.processWorldBookContext).toHaveBeenLastCalledWith({
        text: "this text should match nothing",
        world_book_ids: [1],
        scan_depth: 4,
        token_budget: 600,
        recursive_scanning: true
      })
    })

    expect(await screen.findByText("No entries matched for this sample text.")).toBeInTheDocument()
    expect(screen.getByText("No entries matched for this sample text.")).toBeInTheDocument()
  },
    15000
  )

  it(
    "opens test keywords from Tools menu and surfaces API errors",
    async () => {
      const user = userEvent.setup()
      tldwClientMock.processWorldBookContext.mockRejectedValueOnce(
        new Error("Processing failed on server")
      )

      render(<WorldBooksManager />)

      await user.click(screen.getByRole("button", { name: "Tools" }))
      await user.click(await screen.findByRole("menuitem", { name: "Test Matching" }))
      fireEvent.change(screen.getByRole("textbox", { name: "Sample text for keyword test" }), {
        target: { value: "castle walls" }
      })
      await user.click(screen.getByRole("button", { name: "Run keyword test" }))

      expect(await screen.findByText("Processing failed on server")).toBeInTheDocument()
    },
    30000
  )
})
