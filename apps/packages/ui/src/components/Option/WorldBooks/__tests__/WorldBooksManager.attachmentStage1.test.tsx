import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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

type CharacterRecord = { id: number; name: string }

describe("WorldBooksManager attachment stage-1 scalable views", () => {
  let currentCharacters: CharacterRecord[]
  let currentAttachmentsByBook: Record<number, CharacterRecord[]>
  let attachmentQueryError: Error | null
  let invalidateQueriesMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true
    currentCharacters = []
    currentAttachmentsByBook = { 1: [] }
    attachmentQueryError = null
    invalidateQueriesMock = vi.fn()

    useQueryClientMock.mockReturnValue({
      invalidateQueries: invalidateQueriesMock
    })

    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({ data: worldBooks, status: "success" })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: currentCharacters, status: "success" })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({
          data: currentAttachmentsByBook,
          isLoading: false,
          isError: attachmentQueryError != null,
          error: attachmentQueryError
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("switches to list view when character count exceeds threshold", async () => {
    const user = userEvent.setup()
    currentCharacters = Array.from({ length: 11 }, (_, index) => ({
      id: index + 1,
      name: `Character ${index + 1}`
    }))

    render(<WorldBooksManager />)

    // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))

    expect(await screen.findByText("List view active (11 characters).")).toBeInTheDocument()
    expect(
      screen.getByLabelText("Attachment selector for Arcana")
    ).toBeInTheDocument()
  })

  it("keeps matrix mode for small desktop character sets and supports attach/detach toggles", async () => {
    const user = userEvent.setup()
    currentCharacters = [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" }
    ]
    currentAttachmentsByBook = {
      1: [{ id: 1, name: "Alice" }]
    }

    render(<WorldBooksManager />)

    // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
    expect(await screen.findByText("Matrix view active (2 characters).")).toBeInTheDocument()

    await user.click(screen.getByLabelText("Toggle attachment Arcana / Bob"))
    await waitFor(() => {
      expect(tldwClientMock.attachWorldBookToCharacter).toHaveBeenCalledWith(2, 1)
    })

    await user.click(screen.getByLabelText("Toggle attachment Arcana / Alice"))
    await waitFor(() => {
      expect(tldwClientMock.detachWorldBookFromCharacter).toHaveBeenCalledWith(1, 1)
    })
  }, 30000)

  it(
    "uses list view on mobile and supports list-mode attach + detach-all",
    async () => {
      const user = userEvent.setup()
      mockBreakpoints.md = false
      currentCharacters = [
        { id: 1, name: "Alice" },
        { id: 2, name: "Bob" }
      ]
      currentAttachmentsByBook = {
        1: [{ id: 1, name: "Alice" }]
      }

      render(<WorldBooksManager />)

      // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
      expect(await screen.findByText("List view active (2 characters).")).toBeInTheDocument()

      await user.click(screen.getByLabelText("Attachment selector for Arcana"))
      await user.click(await screen.findByText("Bob"))
      await waitFor(() => {
        expect(tldwClientMock.attachWorldBookToCharacter).toHaveBeenCalledWith(2, 1)
      })

      await user.click(screen.getByRole("button", { name: "Detach all characters from Arcana" }))
      await waitFor(() => {
        expect(tldwClientMock.detachWorldBookFromCharacter).toHaveBeenCalledWith(1, 1)
      })
    },
    15000
  )

  it("hydrates attachment relationships sequentially instead of bursting every character request at once", async () => {
    currentCharacters = [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" },
      { id: 3, name: "Cara" }
    ]
    tldwClientMock.listCharacterWorldBooks.mockImplementation(
      () => new Promise(() => undefined)
    )

    render(<WorldBooksManager />)

    const attachmentQueryCall = useQueryMock.mock.calls.find((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:worldBookAttachments"
    })
    const attachmentQuery = attachmentQueryCall?.[0]

    expect(attachmentQuery?.queryFn).toBeTypeOf("function")

    void attachmentQuery.queryFn()
    await Promise.resolve()
    await Promise.resolve()

    expect(tldwClientMock.listCharacterWorldBooks).toHaveBeenCalledTimes(1)
    expect(tldwClientMock.listCharacterWorldBooks).toHaveBeenCalledWith(1, true)
  })

  it("retains disabled attachment metadata when the relationship reader includes disabled links", async () => {
    currentCharacters = [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" }
    ]
    tldwClientMock.listCharacterWorldBooks.mockImplementation(
      async (characterId: number, includeDisabled?: boolean) =>
        includeDisabled
          ? [{ world_book_id: 1, attachment_enabled: characterId === 1, attachment_priority: characterId }]
          : []
    )

    render(<WorldBooksManager />)

    const attachmentQueryCall = useQueryMock.mock.calls.find((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:worldBookAttachments"
    })

    await expect(attachmentQueryCall?.[0]?.queryFn()).resolves.toEqual({
      1: [
        { id: 1, name: "Alice", attachment_enabled: true, attachment_priority: 1 },
        { id: 2, name: "Bob", attachment_enabled: false, attachment_priority: 2 }
      ]
    })
  })

  it("surfaces a relationship transport failure instead of reporting no attachments", async () => {
    currentCharacters = [{ id: 1, name: "Alice" }]
    const transportError = Object.assign(new Error("service unavailable"), { status: 503 })
    tldwClientMock.listCharacterWorldBooks.mockRejectedValue(transportError)

    render(<WorldBooksManager />)

    const attachmentQueryCall = useQueryMock.mock.calls.find((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:worldBookAttachments"
    })

    await expect(attachmentQueryCall?.[0]?.queryFn()).rejects.toBe(transportError)
  })

  it("treats a missing character relationship as no association", async () => {
    currentCharacters = [{ id: 1, name: "Alice" }]
    tldwClientMock.listCharacterWorldBooks.mockRejectedValue(
      Object.assign(new Error("not found"), { status: 404 })
    )

    render(<WorldBooksManager />)

    const attachmentQueryCall = useQueryMock.mock.calls.find((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:worldBookAttachments"
    })

    await expect(attachmentQueryCall?.[0]?.queryFn()).resolves.toEqual({})
  })

  it("keeps attachment hydration disabled until attachment tooling is opened", async () => {
    const user = userEvent.setup()
    currentCharacters = [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" }
    ]

    render(<WorldBooksManager />)

    const attachmentQueryCalls = useQueryMock.mock.calls.filter((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:worldBookAttachments"
    })

    expect(attachmentQueryCalls.at(-1)?.[0]?.enabled).toBe(false)

    const characterQueryCalls = useQueryMock.mock.calls.filter((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:listCharactersForWB"
    })
    expect(characterQueryCalls.at(-1)?.[0]?.enabled).toBe(false)

    // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))

    const updatedAttachmentQueryCalls = useQueryMock.mock.calls.filter((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:worldBookAttachments"
    })

    expect(updatedAttachmentQueryCalls.at(-1)?.[0]?.enabled).toBe(true)

    const updatedCharacterQueryCalls = useQueryMock.mock.calls.filter((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:listCharactersForWB"
    })
    expect(updatedCharacterQueryCalls.at(-1)?.[0]?.enabled).toBe(true)
  })

  it("loads attachment candidates through the corrected character collection client", async () => {
    currentCharacters = [{ id: 6, name: "Alice" }]
    tldwClientMock.listCharacters.mockResolvedValue(currentCharacters)

    render(<WorldBooksManager />)

    const characterQueryCall = useQueryMock.mock.calls.find((call) => {
      const queryKey = Array.isArray(call?.[0]?.queryKey) ? call[0].queryKey : []
      return queryKey[0] === "tldw:listCharactersForWB"
    })
    const characterQuery = characterQueryCall?.[0]

    expect(characterQuery?.queryFn).toBeTypeOf("function")
    await expect(characterQuery.queryFn()).resolves.toEqual(currentCharacters)
    expect(tldwClientMock.listCharacters).toHaveBeenCalledTimes(1)
  })

  it("retries failed attachment hydration through the existing query client", async () => {
    attachmentQueryError = new Error("service unavailable")

    render(<WorldBooksManager />)
    fireEvent.click(screen.getByText("Arcana"))
    fireEvent.click(await screen.findByRole("tab", { name: "Attachments" }))
    const attachmentsPanel = within(
      await screen.findByRole("tabpanel", { name: "Attachments" })
    )
    expect(await attachmentsPanel.findByRole("alert")).toHaveTextContent(
      "Unable to load character attachments. Try again."
    )
    const retryButton = await attachmentsPanel.findByRole("button", { name: "Try again" })
    expect(retryButton).toBeVisible()
    expect(retryButton).toBeEnabled()
    fireEvent.click(retryButton)

    await waitFor(() => {
      expect(invalidateQueriesMock).toHaveBeenCalledWith({
        queryKey: ["tldw:listCharactersForWB"]
      })
      expect(invalidateQueriesMock).toHaveBeenCalledWith({
        queryKey: ["tldw:worldBookAttachments"]
      })
    })
  })
})
