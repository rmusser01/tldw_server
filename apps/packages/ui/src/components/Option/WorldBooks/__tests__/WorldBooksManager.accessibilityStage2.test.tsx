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
    entry_count: 1
  }
]

const entryData = [
  {
    entry_id: 1,
    keywords: ["castle"],
    content: "Castle historical facts.",
    priority: 60,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  }
]

describe("WorldBooksManager accessibility stage-2 semantics", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true

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
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
      }
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({ data: entryData, status: "success" })
      }
      if (key === "tldw:worldBookRuntimeConfig") {
        return makeUseQueryResult({ data: { max_recursive_depth: 10 }, status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("updates advanced settings summary aria-expanded state in world book form", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByRole("button", { name: "New World Book" }))
    const advancedSummary = await screen.findByText("Matching & Budget")

    expect(advancedSummary).toHaveAttribute("aria-expanded", "false")
    await user.click(advancedSummary)
    await waitFor(() => {
      expect(advancedSummary).toHaveAttribute("aria-expanded", "true")
    })
  }, 30000)

  it("updates matching options aria state and exposes switch semantics in entry authoring", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    const matchingSummary = await screen.findByText("Matching Options")
    expect(matchingSummary).toHaveAttribute("aria-expanded", "false")

    await user.click(matchingSummary)
    await waitFor(() => {
      expect(matchingSummary).toHaveAttribute("aria-expanded", "true")
    })

    const bulkModeSwitch = screen.getByRole("switch", {
      name: "Toggle bulk add mode"
    })
    expect(bulkModeSwitch).toHaveTextContent("Off")

    await user.click(bulkModeSwitch)
    await waitFor(() => {
      expect(bulkModeSwitch).toHaveTextContent("On")
    })
  }, 30000)

  it("keeps icon-only action controls explicitly labeled", async () => {
    render(<WorldBooksManager />)

    // New two-panel layout: Edit button + More actions overflow per row
    expect(await screen.findByRole("button", { name: "Edit Arcana" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "More actions for Arcana" })).toBeInTheDocument()
  })
})
