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
  mutate: (variables: any) => {
    void Promise.resolve(opts?.mutationFn?.(variables))
      .then((result) => {
        opts?.onSuccess?.(result, variables, undefined)
      })
      .catch((error) => {
        opts?.onError?.(error, variables, undefined)
      })
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

describe("WorldBooksManager error-handling stage-3 optimistic concurrency", () => {
  let currentWorldBooks: any[]

  beforeEach(() => {
    vi.clearAllMocks()

    currentWorldBooks = [
      {
        id: 1,
        name: "Arcana",
        description: "Main lore",
        scan_depth: 3,
        token_budget: 500,
        recursive_scanning: false,
        enabled: true,
        version: 7,
        entry_count: 2
      }
    ]

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({ data: currentWorldBooks, status: "success" })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
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

  afterEach(() => {
    vi.clearAllMocks()
  })

  it(
    "passes expectedVersion when saving edits from the detail-panel settings tab",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      await user.click(screen.getByRole("button", { name: "Edit Arcana" }))

      const detailPanel = await screen.findByRole("main", { name: "World book detail" })
      const settingsTab = within(detailPanel).getByRole("tab", { name: "Settings" })
      await waitFor(() => {
        expect(settingsTab).toHaveAttribute("aria-selected", "true")
      })

      const nameInput = within(detailPanel).getByRole("textbox", { name: "Name" })
      await user.clear(nameInput)
      await user.type(nameInput, "Arcana Updated")
      await user.click(within(detailPanel).getByRole("button", { name: "Save" }))

      await waitFor(() => {
        expect(tldwClientMock.updateWorldBook).toHaveBeenCalledWith(
          1,
          expect.objectContaining({ name: "Arcana Updated" }),
          { expectedVersion: 7 }
        )
      })
    },
    30000
  )

  it(
    "shows conflict recovery actions in the detail-panel settings tab and retries with latest version",
    async () => {
      const user = userEvent.setup()

      tldwClientMock.updateWorldBook.mockImplementationOnce(async () => {
        currentWorldBooks = [
          {
            ...currentWorldBooks[0],
            description: "Server-side updated",
            version: 8
          }
        ]
        throw {
          status: 409,
          message: "Version mismatch. Expected 7, found 8. Please refresh and try again."
        }
      })

      render(<WorldBooksManager />)

      await user.click(screen.getByRole("button", { name: "Edit Arcana" }))

      const detailPanel = await screen.findByRole("main", { name: "World book detail" })
      const settingsTab = within(detailPanel).getByRole("tab", { name: "Settings" })
      await waitFor(() => {
        expect(settingsTab).toHaveAttribute("aria-selected", "true")
      })

      const descriptionInput = within(detailPanel).getByRole("textbox", {
        name: "Description (optional)"
      })
      await user.clear(descriptionInput)
      await user.type(descriptionInput, "My local edit")
      await user.click(within(detailPanel).getByRole("button", { name: "Save" }))

      await waitFor(() => {
        expect(within(detailPanel).getByTestId("world-book-edit-conflict")).toBeInTheDocument()
      })

      await user.click(within(detailPanel).getByRole("button", { name: "Reapply my edits" }))

      await waitFor(() => {
        expect(notificationMock.info).toHaveBeenCalledWith(
          expect.objectContaining({ message: "Edits reapplied" })
        )
      })

      const descriptionAfterMerge = within(detailPanel).getByRole("textbox", {
        name: "Description (optional)"
      })
      expect(descriptionAfterMerge).toHaveValue("My local edit")

      await user.click(within(detailPanel).getByRole("button", { name: "Save" }))

      await waitFor(() => {
        expect(tldwClientMock.updateWorldBook).toHaveBeenNthCalledWith(
          1,
          1,
          expect.objectContaining({ description: "My local edit" }),
          { expectedVersion: 7 }
        )
        expect(tldwClientMock.updateWorldBook).toHaveBeenNthCalledWith(
          2,
          1,
          expect.objectContaining({ description: "My local edit" }),
          { expectedVersion: 8 }
        )
      })
    },
    30000
  )
})
