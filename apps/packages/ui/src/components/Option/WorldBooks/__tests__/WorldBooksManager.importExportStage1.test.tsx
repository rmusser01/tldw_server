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

const uploadJson = async (user: ReturnType<typeof userEvent.setup>, text: string, name = "import.json") => {
  const modalTitles = await screen.findAllByText("Import World Book (JSON)")
  const modalTitle = modalTitles[modalTitles.length - 1]
  const modal = modalTitle.closest(".ant-modal") as HTMLElement | null
  expect(modal).not.toBeNull()
  const input = (modal as HTMLElement).querySelector('input[type="file"]')
  expect(input).not.toBeNull()
  const file = new File([text], name, { type: "application/json" })
  ;(file as any).text = async () => text
  await user.upload(input as HTMLInputElement, file)
}

describe("WorldBooksManager import/export stage-1 guidance and validation", () => {
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

  it(
    "shows format help and merge-on-conflict tooltip guidance",
    async () => {
      const user = userEvent.setup()
      render(<WorldBooksManager />)

      // Open Tools dropdown then click Import JSON
      await user.click(screen.getByRole("button", { name: "Tools" }))
      await user.click(await screen.findByText("Import JSON"))
      await user.click(await screen.findByText("Format help"))

      expect(await screen.findByText("Expected tldw JSON shape:")).toBeInTheDocument()

      await user.hover(screen.getByRole("img", { name: "Merge on conflict help" }))
      expect(
        await screen.findByText(/Existing entries are not removed or modified/i)
      ).toBeInTheDocument()
    },
    15000
  )

  it("shows user-friendly malformed JSON error copy", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Import JSON
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Import JSON"))
    await uploadJson(user, '{"world_book": }')

    await waitFor(() => {
      expect(
        screen.getByText("File is not valid JSON (check for trailing commas or invalid characters).")
      ).toBeInTheDocument()
    })
  })

  it("surfaces required-field validation for missing world_book and empty entries", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Import JSON
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Import JSON"))
    await uploadJson(user, JSON.stringify({ entries: [{ keywords: ["k"], content: "c" }] }))
    await waitFor(() => {
      expect(screen.getByText("File is missing the 'world_book' field.")).toBeInTheDocument()
    })

    await uploadJson(user, JSON.stringify({ world_book: { name: "Arcana" }, entries: [] }))
    await waitFor(() => {
      expect(screen.getByText("File is missing entries (found 0 entries).")).toBeInTheDocument()
    })
  }, 15000)
})
