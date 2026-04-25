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
    detachWorldBookFromCharacter: vi.fn(async () => ({}))
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

const worldBooks = [
  {
    id: 1,
    name: "Arcana",
    description: "Main lore",
    enabled: true,
    entry_count: 2
  }
]

describe("WorldBooksManager attachment stage-3 metadata controls", () => {
  let currentAttachmentsByBook: Record<number, any[]>

  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true

    currentAttachmentsByBook = {
      1: [
        {
          id: 1,
          name: "Alice",
          attachment_enabled: false,
          attachment_priority: 7
        }
      ]
    }

    tldwClientMock.attachWorldBookToCharacter.mockImplementation(
      (async (characterId: number, worldBookId: number, options?: { enabled?: boolean; priority?: number }) => {
        const row = currentAttachmentsByBook[worldBookId] || []
        currentAttachmentsByBook[worldBookId] = row.map((item) =>
          item.id === characterId
            ? {
                ...item,
                attachment_enabled:
                  typeof options?.enabled === "boolean"
                    ? options.enabled
                    : item.attachment_enabled,
                attachment_priority:
                  typeof options?.priority === "number"
                    ? options.priority
                    : item.attachment_priority
              }
                : item
        )
      }) as any
    )

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
        return makeUseQueryResult({
          data: [
            { id: 1, name: "Alice" },
            { id: 2, name: "Bob" }
          ],
          status: "success"
        })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: currentAttachmentsByBook, isLoading: false })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it(
    "edits per-attachment enabled/priority metadata from a matrix cell",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

    // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
    expect(await screen.findByText("Matrix view active (2 characters).")).toBeInTheDocument()
    expect(screen.getByText("P7")).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", {
        name: "Edit attachment settings Arcana / Alice"
      })
    )

    const enabledSwitch = await screen.findByLabelText(
      "Attachment enabled Arcana / Alice"
    )
    await user.click(enabledSwitch)

    const priorityInput = screen.getByLabelText("Attachment priority Arcana / Alice")
    fireEvent.change(priorityInput, { target: { value: "12" } })

    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(tldwClientMock.attachWorldBookToCharacter).toHaveBeenCalledWith(1, 1, {
        enabled: true,
        priority: 12
      })
    })
      expect(screen.getByRole("status")).toHaveTextContent(
        "Updated attachment settings for Alice in Arcana."
      )
    },
    20000
  )
})
