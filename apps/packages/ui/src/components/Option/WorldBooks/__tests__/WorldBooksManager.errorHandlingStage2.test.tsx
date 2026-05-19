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
    importWorldBook: vi.fn(async () => ({ world_book_id: 99, name: "Arcana", entries_imported: 1, merged: false })),
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

const uploadFile = async (
  user: ReturnType<typeof userEvent.setup>,
  file: File
) => {
  const modalTitles = await screen.findAllByText("Import World Book (JSON)")
  const modalTitle = modalTitles[modalTitles.length - 1]
  const modal = modalTitle.closest(".ant-modal") as HTMLElement | null
  expect(modal).not.toBeNull()
  const input = (modal as HTMLElement).querySelector('input[type="file"]')
  expect(input).not.toBeNull()
  await user.upload(input as HTMLInputElement, file)
}

describe("WorldBooksManager error-handling stage-2 import diagnostics", () => {
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
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("shows expandable raw parse details while keeping friendly primary error copy", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Import JSON
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Import JSON"))

    const parseErrorFile = new File(["{}"], "broken.json", { type: "application/json" })
    ;(parseErrorFile as any).text = async () => {
      throw new Error("synthetic parse error detail")
    }
    await uploadFile(user, parseErrorFile)

    await waitFor(() => {
      expect(screen.getByText("File is not valid JSON.")).toBeInTheDocument()
    })

    await user.click(screen.getByText("More details"))
    expect(screen.getByText(/synthetic parse error detail/i)).toBeInTheDocument()
  }, 30000)

  it("surfaces conversion context in a more-details section for validation errors", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Import JSON
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Import JSON"))

    const missingWorldBookPayload = JSON.stringify({
      entries: [{ keywords: ["k"], content: "c" }]
    })
    const missingWorldBookFile = new File([missingWorldBookPayload], "missing-world-book.json", {
      type: "application/json"
    })
    ;(missingWorldBookFile as any).text = async () => missingWorldBookPayload
    await uploadFile(user, missingWorldBookFile)

    await waitFor(() => {
      expect(screen.getByText("File is missing the 'world_book' field.")).toBeInTheDocument()
    })

    await user.click(screen.getByText("More details"))
    expect(screen.getByText(/Detected format:/i)).toBeInTheDocument()
  }, 30000)
})
