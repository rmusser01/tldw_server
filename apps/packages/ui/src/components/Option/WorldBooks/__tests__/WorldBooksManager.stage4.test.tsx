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
    createWorldBook: vi.fn(async () => ({ id: 999 })),
    updateWorldBook: vi.fn(async () => ({})),
    deleteWorldBook: vi.fn(async () => ({})),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    addWorldBookEntry: vi.fn(async () => ({}))
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

const baseWorldBooks = [
  {
    id: 1,
    name: "Arcana",
    description: "Main lorebook",
    scan_depth: 3,
    token_budget: 500,
    recursive_scanning: false,
    enabled: true,
    entry_count: 2
  }
]

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

describe("WorldBooksManager stage-4 flows", () => {
  let currentWorldBooks: any[]

  beforeEach(() => {
    vi.clearAllMocks()
    currentWorldBooks = [...baseWorldBooks]
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
      if (key === "tldw:worldBookPreviewEntries") {
        return makeUseQueryResult({
          data: [
            {
              entry_id: 88,
              keywords: ["preview"],
              content: "Preview entry content"
            }
          ],
          status: "success"
        })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("renders the manager action bar without throwing", () => {
    render(<WorldBooksManager />)

    // New layout: Edit + More actions overflow buttons per row
    expect(screen.getByRole("button", { name: "Edit Arcana" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "More actions for Arcana" })).toBeInTheDocument()
  })

  it("duplicates a world book and clones its entries", async () => {
    const user = userEvent.setup()
    const sourceEntries = [
      {
        entry_id: 10,
        keywords: ["alpha", "beta"],
        content: "Arcana entry one",
        priority: 65,
        enabled: true,
        case_sensitive: false,
        regex_match: false,
        whole_word_match: true,
        appendable: false
      },
      {
        entry_id: 11,
        keywords: ["gamma"],
        content: "Arcana entry two",
        priority: 40,
        enabled: true,
        case_sensitive: false,
        regex_match: false,
        whole_word_match: true,
        appendable: true
      }
    ]

    tldwClientMock.createWorldBook.mockResolvedValueOnce({ id: 101 })
    tldwClientMock.listWorldBookEntries.mockResolvedValueOnce({ entries: sourceEntries })
    tldwClientMock.addWorldBookEntry.mockResolvedValue({})

    render(<WorldBooksManager />)

    // Open overflow menu and click Duplicate
    await user.click(screen.getByRole("button", { name: "More actions for Arcana" }))
    await user.click(await screen.findByText("Duplicate"))

    await waitFor(() => {
      expect(tldwClientMock.createWorldBook).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Copy of Arcana",
          description: "Main lorebook",
          scan_depth: 3,
          token_budget: 500,
          recursive_scanning: false,
          enabled: true
        })
      )
    })
    await waitFor(() => {
      expect(tldwClientMock.listWorldBookEntries).toHaveBeenCalledWith(1, false)
    })
    await waitFor(() => {
      expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledWith(
      101,
      expect.objectContaining({
        keywords: ["alpha", "beta"],
        content: "Arcana entry one"
      })
    )
    expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledWith(
      101,
      expect.objectContaining({
        keywords: ["gamma"],
        content: "Arcana entry two"
      })
    )
    expect(notificationMock.success).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Duplicated" })
    )
  })

  it("creates from starter template and seeds template entries", async () => {
    const user = userEvent.setup()
    tldwClientMock.createWorldBook.mockResolvedValueOnce({ id: 202 })
    tldwClientMock.addWorldBookEntry.mockResolvedValue({})

    render(<WorldBooksManager />)

    await user.click(screen.getByRole("button", { name: "New World Book" }))
    await user.click(await screen.findByRole("combobox", { name: "Starter Template (optional)" }))
    await user.click(await screen.findByText("Fantasy Setting"))
    await user.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(tldwClientMock.createWorldBook).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Fantasy Lore",
          scan_depth: 4,
          token_budget: 700,
          recursive_scanning: true
        })
      )
    })
    await waitFor(() => {
      expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.addWorldBookEntry).toHaveBeenNthCalledWith(
      1,
      202,
      expect.objectContaining({
        keywords: ["magic system", "mana"]
      })
    )
    expect(tldwClientMock.addWorldBookEntry).toHaveBeenNthCalledWith(
      2,
      202,
      expect.objectContaining({
        keywords: ["capital city", "high council"]
      })
    )
  }, 15000)

  it("shows custom empty-state guidance with create CTA", async () => {
    const user = userEvent.setup()
    currentWorldBooks = []

    render(<WorldBooksManager />)

    // New empty state uses WorldBookEmptyState component with different text
    expect(screen.getByText("World Books")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Create your first world book" }))
    expect(await screen.findByRole("combobox", { name: "Starter Template (optional)" })).toBeInTheDocument()
  })

  it("supports bulk enable actions from row selection", async () => {
    const user = userEvent.setup()
    currentWorldBooks = [
      ...baseWorldBooks,
      {
        id: 2,
        name: "Archive",
        description: "Old entries",
        scan_depth: 3,
        token_budget: 300,
        recursive_scanning: false,
        enabled: false,
        entry_count: 1
      }
    ]

    render(<WorldBooksManager />)

    const checkboxes = screen.getAllByRole("checkbox")
    await user.click(checkboxes[1])
    await user.click(checkboxes[2])
    await user.click(screen.getByRole("button", { name: "Enable" }))

    await waitFor(() => {
      expect(tldwClientMock.updateWorldBook).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.updateWorldBook).toHaveBeenNthCalledWith(
      1,
      1,
      expect.objectContaining({ enabled: true })
    )
    expect(tldwClientMock.updateWorldBook).toHaveBeenNthCalledWith(
      2,
      2,
      expect.objectContaining({ enabled: true })
    )
  }, 20000)

  it("opens the detail panel entries view when a world book row is selected", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByText("Arcana"))

    const detailPanel = await screen.findByRole("main", { name: "World book detail" })
    expect(within(detailPanel).getByRole("tab", { name: "Entries" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
    expect(detailPanel).toHaveTextContent("No entries yet")
  })
})
