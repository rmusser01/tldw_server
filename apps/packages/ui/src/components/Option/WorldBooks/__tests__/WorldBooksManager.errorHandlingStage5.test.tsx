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
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({ world_book_id: 99, name: "Arcana", entries_imported: 1, merged: false })),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({})),
    getWorldBookRuntimeConfig: vi.fn(async () => ({ max_recursive_depth: 12 })),
    processWorldBookContext: vi.fn(async () => ({
      injected_content: "",
      entries_matched: 0,
      books_used: 0,
      tokens_used: 0,
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
      return undefined
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

describe("WorldBooksManager error-handling stage-5 edge warnings", () => {
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
              scan_depth: 3,
              token_budget: 500,
              recursive_scanning: false,
              entry_count: 2
            }
          ],
          status: "success"
        })
      }
      if (key === "tldw:worldBookRuntimeConfig") {
        return makeUseQueryResult({ data: { max_recursive_depth: 12 }, status: "success" })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [{ id: 1, name: "Active Character" }] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({
          data: {
            1: [
              {
                id: 2,
                name: "Deleted Character",
                attachment_enabled: true,
                attachment_priority: 0
              }
            ]
          },
          isLoading: false
        })
      }
      if (key === "tldw:worldBookGlobalStatistics") {
        return makeUseQueryResult({ data: null, status: "success" })
      }
      if (key === "tldw:worldBookPreviewEntries") {
        return makeUseQueryResult({ data: [], status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  it.each([
    [new Error("Catalog read unavailable"), "Catalog read unavailable"],
    [null, "Could not load world books right now. Check your server connection and try again."]
  ] as const)("shows a distinct catalog read error and explicit retry (%s)", async (error, message) => {
    const user = userEvent.setup()
    const refetch = vi.fn(async () => makeUseQueryResult({}))
    const normalQuery = useQueryMock.getMockImplementation()
    if (!normalQuery) throw new Error("Expected the existing worldbook query fixture")
    useQueryMock.mockImplementation((options: { queryKey: readonly unknown[] }) =>
      options.queryKey[0] === "tldw:listWorldBooks"
        ? makeUseQueryResult({ data: undefined, status: "error", error, refetch })
        : normalQuery(options)
    )
    render(<WorldBooksManager />)

    expect(screen.getByRole("alert")).toHaveTextContent(message)
    expect(screen.getByRole("heading", { name: "Unable to load world books" })).toBeVisible()
    expect(screen.queryByRole("button", { name: "Create your first world book", exact: true })).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Retry", exact: true }))
    expect(refetch).toHaveBeenCalledTimes(1)
  })

  it("blocks duplicate catalog retries while the read is in flight", async () => {
    const user = userEvent.setup()
    const refetch = vi.fn()
    const normalQuery = useQueryMock.getMockImplementation()
    if (!normalQuery) throw new Error("Expected the existing worldbook query fixture")
    useQueryMock.mockImplementation((options: { queryKey: readonly unknown[] }) =>
      options.queryKey[0] === "tldw:listWorldBooks"
        ? makeUseQueryResult({ data: undefined, status: "error", error: new Error("Unavailable"), isFetching: true, refetch })
        : normalQuery(options)
    )
    render(<WorldBooksManager />)

    const retry = screen.getByRole("button", { name: "Retry", exact: true })
    expect(retry).toBeDisabled()
    await user.click(retry)
    expect(refetch).not.toHaveBeenCalled()
  })

  it("restores the actual catalog after an explicit successful retry", async () => {
    const user = userEvent.setup()
    const normalQuery = useQueryMock.getMockImplementation()
    if (!normalQuery) throw new Error("Expected the existing worldbook query fixture")
    let failed = true
    const refetch = vi.fn(async () => { failed = false })
    useQueryMock.mockImplementation((options: { queryKey: readonly unknown[] }) =>
      options.queryKey[0] === "tldw:listWorldBooks" && failed
        ? makeUseQueryResult({ data: undefined, status: "error", error: new Error("Unavailable"), refetch })
        : normalQuery(options)
    )
    const view = render(<WorldBooksManager />)

    await user.click(screen.getByRole("button", { name: "Retry", exact: true }))
    view.rerender(<WorldBooksManager />)
    expect(refetch).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole("heading", { name: "Unable to load world books" })).not.toBeInTheDocument()
    expect(screen.getByRole("region", { name: "World books table" })).toHaveTextContent("Arcana")
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it(
    "filters stale attachment rows when attached characters no longer exist",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      await user.click(screen.getByText("Arcana"))
      const detailPanel = await screen.findByRole("main", { name: "World book detail" })
      await user.click(screen.getByRole("tab", { name: "Attachments" }))

      expect(screen.queryByText("Deleted Character")).not.toBeInTheDocument()
      expect(detailPanel).toHaveTextContent("No characters attached.")
    },
    20000
  )

  it(
    "shows recursive scanning warning with backend max depth",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      await user.click(screen.getByRole("button", { name: "New World Book" }))
      await user.click(await screen.findByRole("combobox", { name: "Starter Template (optional)" }))
      await user.click(await screen.findByText("Fantasy Setting"))

      const warning = await screen.findByTestId("recursive-scanning-warning-create")
      expect(warning).toHaveTextContent("limited to 12 levels")
    },
    20000
  )
})
