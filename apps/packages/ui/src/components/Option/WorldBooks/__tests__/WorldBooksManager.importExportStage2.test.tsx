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

const uploadJson = async (
  user: ReturnType<typeof userEvent.setup>,
  text: string,
  name = "import.json"
) => {
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

describe("WorldBooksManager import/export stage-2 preview depth", () => {
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

  it("shows world-book settings and expandable entry preview details", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Import JSON
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Import JSON"))
    await uploadJson(
      user,
      JSON.stringify({
        world_book: {
          name: "Arcana Import",
          scan_depth: 7,
          token_budget: 900,
          recursive_scanning: true,
          enabled: false
        },
        entries: [
          { keywords: ["alpha", "beta"], content: "First lore entry" },
          { keywords: ["gamma"], content: "Second lore entry" }
        ]
      })
    )

    await waitFor(() => {
      expect(screen.getByText("Scan depth: 7")).toBeInTheDocument()
    })
    expect(screen.getByText("Token budget: 900")).toBeInTheDocument()
    expect(screen.getByText("Recursive scanning: Enabled")).toBeInTheDocument()
    expect(screen.getByText("World book enabled: Disabled")).toBeInTheDocument()

    const summary = screen.getByText("Preview first 2 entries")
    await user.click(summary)
    expect(screen.getByTestId("import-preview-entries")).toHaveAttribute("open")
    expect(screen.getByText("First lore entry")).toBeInTheDocument()
    expect(screen.getByText("Second lore entry")).toBeInTheDocument()
  }, 30000)

  it(
    "limits preview rendering to first five entries and truncates long content",
    async () => {
      const user = userEvent.setup()
      const longContent = "L".repeat(220)

      const entries = Array.from({ length: 12 }, (_, index) => ({
        keywords: [`kw-${index + 1}`],
        content: index === 0 ? longContent : `entry-${index + 1}`
      }))

      render(<WorldBooksManager />)

      // Open Tools dropdown then click Import JSON
      await user.click(screen.getByRole("button", { name: "Tools" }))
      await user.click(await screen.findByText("Import JSON"))
      await uploadJson(
        user,
        JSON.stringify({
          world_book: { name: "Large Import" },
          entries
        })
      )

      const summary = await screen.findByText("Preview first 5 entries")
      await user.click(summary)

      const previewDetails = screen.getByTestId("import-preview-entries")
      expect(previewDetails).toHaveAttribute("open")

      const previewNodes = within(previewDetails).getAllByTestId(
        /import-preview-entry-/
      )
      expect(previewNodes).toHaveLength(5)
      expect(screen.getByText("Showing first 5 of 12 entries.")).toBeInTheDocument()
      expect(screen.getByText(`${"L".repeat(137)}...`)).toBeInTheDocument()
    },
    15000
  )
})
