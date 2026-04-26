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
    bulkWorldBookEntries: vi.fn(async () => ({})),
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

describe("WorldBooksManager entry drawer stage-2 authoring controls", () => {
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
              appendable: false,
              group: "History"
            },
            {
              entry_id: 11,
              keywords: ["seed-generic"],
              content: "Ungrouped note",
              priority: 30,
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

  it("uses tag keywords with preview in add and edit flows", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    await screen.findByRole("columnheader", { name: "Keywords" })

    const addKeywordInput = screen.getByRole("combobox", { name: "Keywords" })
    await user.type(addKeywordInput, "entry-alpha{enter}entry-beta{enter}")
    expect(screen.getByText("entry-alpha", { selector: ".ant-tag" })).toBeInTheDocument()
    expect(screen.getByText("entry-beta", { selector: ".ant-tag" })).toBeInTheDocument()

    const keywordsHeader = screen.getByRole("columnheader", { name: "Keywords" })
    const tableWrapper = keywordsHeader.closest(".ant-table-wrapper")
    expect(tableWrapper).not.toBeNull()
    await user.click(
      within(tableWrapper as HTMLElement).getAllByRole("button", { name: "Edit entry" })[0]
    )
    expect(await screen.findByRole("button", { name: "Save Changes" })).toBeInTheDocument()
    expect(screen.getAllByText("seed-keyword", { selector: ".ant-tag" }).length).toBeGreaterThan(1)
  }, 60000)

  it("supports entry group field in table and add payload", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    expect(await screen.findByRole("columnheader", { name: "Group" })).toBeInTheDocument()
    expect(screen.getByText("History", { selector: ".ant-tag" })).toBeInTheDocument()
    expect(screen.getByText("Ungrouped")).toBeInTheDocument()

    await user.click(screen.getByLabelText("Filter entries by group"))
    await user.click(
      await screen.findByText("History", { selector: ".ant-select-item-option-content" })
    )
    expect(screen.getByText("Seed entry")).toBeInTheDocument()
    expect(screen.queryByText("Ungrouped note")).not.toBeInTheDocument()

    await user.type(screen.getByRole("textbox", { name: "Content" }), "Harbor overview")
    await user.type(screen.getByRole("textbox", { name: "Group (optional)" }), "  Geography  ")
    await user.click(screen.getByRole("button", { name: "Add Entry" }))

    await waitFor(() => {
      expect(tldwClientMock.addWorldBookEntry).toHaveBeenCalledWith(
        1,
        expect.objectContaining({
          content: "Harbor overview",
          group: "Geography"
        })
      )
    })
  }, 30000)

  it("blocks add submit on invalid regex keyword patterns", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    await screen.findByRole("columnheader", { name: "Keywords" })
    await user.click(screen.getByText("Matching Options"))

    const regexSwitch = screen.getAllByRole("switch", { name: "Regex Match" })[0]
    await user.click(regexSwitch)

    const addKeywordInput = screen.getByRole("combobox", { name: "Keywords" })
    await user.type(addKeywordInput, "(broken{enter}")
    await user.type(screen.getByRole("textbox", { name: "Content" }), "Body content")
    await user.click(screen.getByRole("button", { name: "Add Entry" }))

    expect(await screen.findByText(/Invalid regex pattern:/)).toBeInTheDocument()
    expect(tldwClientMock.addWorldBookEntry).not.toHaveBeenCalled()
  }, 45000)
})
