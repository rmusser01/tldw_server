import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"
import { DEFAULT_BULK_ADD_CONCURRENCY } from "../worldBookBulkUtils"

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
    bulkWorldBookEntries: vi.fn(async () => ({})),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({})),
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

describe("WorldBooksManager entry drawer stage-4 bulk add workflow", () => {
  const getBulkModeSwitch = (): HTMLElement => {
    const label = screen.getByText("Bulk add mode")
    const toggleContainer = label.closest("div")
    if (!toggleContainer) throw new Error("Bulk add mode toggle container not found")
    return within(toggleContainer).getByRole("switch")
  }

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
              entry_count: 1
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
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({
          data: [
            {
              entry_id: 10,
              keywords: ["seed-keyword"],
              content: "Seed entry",
              priority: 50,
              enabled: true,
              case_sensitive: false,
              regex_match: false,
              whole_word_match: true,
              appendable: false
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

  it("documents all supported bulk separators in the UI", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    await screen.findByRole("columnheader", { name: "Keywords" })
    await user.click(getBulkModeSwitch())
    await user.click(screen.getByText("Supported formats"))

    expect(screen.getByText("keyword1, keyword2 => content")).toBeInTheDocument()
    expect(screen.getByText("keyword1, keyword2 -> content")).toBeInTheDocument()
    expect(screen.getByText("keyword1, keyword2 | content")).toBeInTheDocument()
    expect(screen.getByText("keyword1, keyword2<TAB>content")).toBeInTheDocument()
  }, 15000)

  it("runs bounded-concurrency bulk add and reports progress with per-entry failures", async () => {
    const user = userEvent.setup()
    let inFlight = 0
    let maxInFlight = 0

    tldwClientMock.addWorldBookEntry.mockImplementation((async (_worldBookId: number, entry: any) => {
      inFlight += 1
      maxInFlight = Math.max(maxInFlight, inFlight)
      await new Promise((resolve) => setTimeout(resolve, 80))
      inFlight -= 1
      if (String(entry?.content || "").includes("FAIL")) {
        throw new Error("Simulated failure")
      }
      return {}
    }) as any)

    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    await screen.findByRole("columnheader", { name: "Keywords" })
    await user.click(getBulkModeSwitch())

    fireEvent.change(screen.getByLabelText("Bulk entry input"), {
      target: {
        value: [
          "alpha => success-1",
          "bravo -> success-2",
          "charlie | FAIL-3",
          "delta\tsuccess-4",
          "echo => FAIL-5",
          "foxtrot -> success-6",
          "golf | success-7",
          "hotel\tsuccess-8"
        ].join("\n")
      }
    })

    await user.click(screen.getByRole("button", { name: "Add Entries" }))

    expect(await screen.findByText(/Bulk progress: \d+ \/ 8/)).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText(/Bulk progress: [1-7] \/ 8/)).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByText("Bulk progress: 8 / 8 (6 succeeded, 2 failed)")).toBeInTheDocument()
    })

    expect(screen.getByText("Failed entries (2)")).toBeInTheDocument()
    expect(screen.getByText(/Line 3 \(charlie\): Simulated failure/)).toBeInTheDocument()
    expect(screen.getByText(/Line 5 \(echo\): Simulated failure/)).toBeInTheDocument()

    expect(maxInFlight).toBeGreaterThan(1)
    expect(maxInFlight).toBeLessThanOrEqual(DEFAULT_BULK_ADD_CONCURRENCY)
    expect(notificationMock.warning).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Bulk add completed with errors"
      })
    )
  }, 90000)
})
