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
    bulkWorldBookEntries: vi.fn(async () => ({
      success: true,
      affected_count: 0,
      failed_ids: []
    })),
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
  mockBreakpoints: { lg: true, md: true, sm: true } as Record<string, boolean>
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

const WORLD_BOOKS_DATA = [
  {
    id: 1,
    name: "Arcana",
    description: "Main lore",
    enabled: true,
    entry_count: 2,
    token_budget: 500
  },
  {
    id: 2,
    name: "Geography",
    description: "Maps and places",
    enabled: false,
    entry_count: 5,
    token_budget: 300
  }
]

function setupQueryMocks() {
  useQueryClientMock.mockReturnValue({
    invalidateQueries: vi.fn()
  })
  useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))

  useQueryMock.mockImplementation((opts: any) => {
    const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
    const key = queryKey[0]

    if (key === "tldw:listWorldBooks") {
      return makeUseQueryResult({
        data: WORLD_BOOKS_DATA,
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
      return makeUseQueryResult({ data: [] })
    }
    return makeUseQueryResult({})
  })
}

describe("WorldBooksManager responsive layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupQueryMocks()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it(
    "desktop shows two-panel layout",
    () => {
      mockBreakpoints.lg = true
      mockBreakpoints.md = true
      mockBreakpoints.sm = true

      render(<WorldBooksManager />)

      expect(screen.getByTestId("world-books-two-panel")).toBeInTheDocument()
      expect(screen.queryByTestId("world-books-stacked")).not.toBeInTheDocument()
      expect(screen.queryByTestId("world-books-mobile")).not.toBeInTheDocument()
    },
    15000
  )

  it(
    "mobile shows only list when no selection",
    () => {
      mockBreakpoints.lg = false
      mockBreakpoints.md = false
      mockBreakpoints.sm = true

      render(<WorldBooksManager />)

      expect(screen.getByTestId("world-books-mobile")).toBeInTheDocument()
      // List panel should be visible
      expect(screen.getByRole("navigation", { name: "World books list" })).toBeInTheDocument()
      // Detail panel should NOT be rendered (no world book detail landmark)
      expect(screen.queryByRole("main", { name: "World book detail" })).not.toBeInTheDocument()
    },
    15000
  )

  it(
    "mobile shows only detail with back button when world book is selected",
    async () => {
      mockBreakpoints.lg = false
      mockBreakpoints.md = false
      mockBreakpoints.sm = true

      const user = userEvent.setup()
      render(<WorldBooksManager />)

      // Select a world book by clicking on it in the list
      await user.click(screen.getByText("Arcana"))

      // Now the list should be hidden and detail visible
      expect(screen.queryByRole("navigation", { name: "World books list" })).not.toBeInTheDocument()
      expect(screen.getByRole("main", { name: "World book detail" })).toBeInTheDocument()

      // Back button should be present
      expect(
        screen.getByRole("button", { name: "Back to world books list" })
      ).toBeInTheDocument()
    },
    15000
  )

  it(
    "back button clears selection and shows list again on mobile",
    async () => {
      mockBreakpoints.lg = false
      mockBreakpoints.md = false
      mockBreakpoints.sm = true

      const user = userEvent.setup()
      render(<WorldBooksManager />)

      // Select a world book
      await user.click(screen.getByText("Arcana"))

      // Verify detail is showing
      expect(screen.getByRole("main", { name: "World book detail" })).toBeInTheDocument()

      // Click back button
      await user.click(
        screen.getByRole("button", { name: "Back to world books list" })
      )

      // List should be visible again, detail gone
      expect(screen.getByRole("navigation", { name: "World books list" })).toBeInTheDocument()
      expect(screen.queryByRole("main", { name: "World book detail" })).not.toBeInTheDocument()
    },
    15000
  )

  it(
    "tablet shows stacked layout",
    () => {
      mockBreakpoints.lg = false
      mockBreakpoints.md = true
      mockBreakpoints.sm = true

      render(<WorldBooksManager />)

      expect(screen.getByTestId("world-books-stacked")).toBeInTheDocument()
      expect(screen.queryByTestId("world-books-two-panel")).not.toBeInTheDocument()
      expect(screen.queryByTestId("world-books-mobile")).not.toBeInTheDocument()
    },
    15000
  )
})
