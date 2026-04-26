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

describe("WorldBooksManager attachment stage-2 toggle feedback", () => {
  let currentCharacters: CharacterRecord[]
  let currentAttachmentsByBook: Record<number, CharacterRecord[]>

  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true
    currentCharacters = [
      { id: 1, name: "Alice" },
      { id: 2, name: "Bob" }
    ]
    currentAttachmentsByBook = {
      1: [{ id: 1, name: "Alice" }]
    }

    tldwClientMock.attachWorldBookToCharacter.mockImplementation(
      (async (characterId: number, worldBookId: number) => {
        const row = currentAttachmentsByBook[worldBookId] || []
        if (!row.some((character) => character.id === characterId)) {
          const character =
            currentCharacters.find((item) => item.id === characterId) || {
              id: characterId,
              name: `Character ${characterId}`
            }
          currentAttachmentsByBook[worldBookId] = [...row, character]
        }
      }) as any
    )

    tldwClientMock.detachWorldBookFromCharacter.mockImplementation(
      (async (characterId: number, worldBookId: number) => {
        const row = currentAttachmentsByBook[worldBookId] || []
        currentAttachmentsByBook[worldBookId] = row.filter(
          (character) => character.id !== characterId
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
        return makeUseQueryResult({ data: currentCharacters, status: "success" })
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
    "highlights newly attached/detached cells and shows success micro-feedback",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
      expect(await screen.findByText("Matrix view active (2 characters).")).toBeInTheDocument()

      await user.click(screen.getByLabelText("Toggle attachment Arcana / Bob"))
      await waitFor(() => {
        expect(tldwClientMock.attachWorldBookToCharacter).toHaveBeenCalledWith(2, 1)
      })
      await waitFor(() => {
        expect(screen.getByTestId("matrix-cell-1-2")).toHaveAttribute(
          "data-delta-state",
          "attached"
        )
      })
      expect(screen.getByRole("status")).toHaveTextContent("Attached Bob to Arcana.")

      await user.click(screen.getByLabelText("Toggle attachment Arcana / Alice"))
      await waitFor(() => {
        expect(tldwClientMock.detachWorldBookFromCharacter).toHaveBeenCalledWith(1, 1)
      })
      await waitFor(() => {
        expect(screen.getByTestId("matrix-cell-1-1")).toHaveAttribute(
          "data-delta-state",
          "detached"
        )
      })
      expect(screen.getByRole("status")).toHaveTextContent("Detached Alice from Arcana.")
    },
    15000
  )

  it("shows explicit failure feedback and leaves delta state unchanged when a toggle fails", async () => {
    const user = userEvent.setup()
    tldwClientMock.attachWorldBookToCharacter.mockRejectedValueOnce(new Error("Network down"))

    render(<WorldBooksManager />)

    // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
    expect(await screen.findByText("Matrix view active (2 characters).")).toBeInTheDocument()

    await user.click(screen.getByLabelText("Toggle attachment Arcana / Bob"))
    await waitFor(() => {
      expect(screen.getByRole("status")).toHaveTextContent("Changes were reverted.")
    })
    expect(screen.getByRole("status")).toHaveTextContent("Network down")
    expect(screen.getByTestId("matrix-cell-1-2")).toHaveAttribute(
      "data-delta-state",
      "none"
    )
  }, 30000)

  it(
    "shows list-mode delta chips for mobile attachment changes",
    async () => {
      const user = userEvent.setup()
      mockBreakpoints.md = false

      render(<WorldBooksManager />)

      // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
      expect(await screen.findByText("List view active (2 characters).")).toBeInTheDocument()

      await user.click(screen.getByLabelText("Attachment selector for Arcana"))
      await user.click(await screen.findByText("Bob"))

      await waitFor(() => {
        expect(screen.getByText("+1 new")).toBeInTheDocument()
      })

      await user.click(screen.getByRole("button", { name: "Detach all characters from Arcana" }))
      await waitFor(() => {
        expect(screen.getByText("-1 removed")).toBeInTheDocument()
      })
    },
    15000
  )
})
