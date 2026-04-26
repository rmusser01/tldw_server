import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import axe from "axe-core"
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

const runA11yBaselineRules = async (context: Element) =>
  axe.run(context, {
    runOnly: {
      type: "rule",
      values: [
        "button-name",
        "link-name",
        "label",
        "aria-valid-attr",
        "aria-valid-attr-value",
        "aria-required-attr"
      ]
    },
    resultTypes: ["violations"]
  })

const expectNoInvalidAriaViolations = (
  violations: Array<{
    id: string
  }>
) => {
  const disallowedIds = new Set([
    "aria-valid-attr",
    "aria-valid-attr-value",
    "aria-required-attr"
  ])

  const disallowedViolations = violations.filter((violation) =>
    disallowedIds.has(violation.id)
  )

  expect(disallowedViolations).toEqual([])
}

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
    content: "Castle reference lore.",
    priority: 60,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  }
]

type CharacterRecord = { id: number; name: string }

describe("WorldBooksManager accessibility stage-1 baseline harness", () => {
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

  it("captures baseline label gap while enforcing valid aria/name rules in list view", async () => {
    const { container } = render(<WorldBooksManager />)
    expect(await screen.findByText("Arcana")).toBeInTheDocument()

    const results = await runA11yBaselineRules(container)
    expectNoInvalidAriaViolations(results.violations)
    expect(results.violations.map((violation) => violation.id)).not.toContain("button-name")
    expect(results.violations.map((violation) => violation.id)).not.toContain("link-name")
    expect(results.violations.map((violation) => violation.id)).toContain("label")
  }, 15000)

  it("enforces valid aria/name rules in the detail-panel entries workflow", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByText("Arcana"))
    const entriesTab = await screen.findByRole("tab", { name: /Entries/i })
    const detailScope =
      entriesTab.closest("main[aria-label='World book detail']") ||
      screen.getByRole("main", { name: "World book detail" })

    const results = await runA11yBaselineRules(detailScope)
    expectNoInvalidAriaViolations(results.violations)
    expect(results.violations.map((violation) => violation.id)).not.toContain("button-name")
  }, 30000)

  it("enforces valid aria/name rules in relationship matrix mode", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Open Tools dropdown then click Relationship Matrix
    await user.click(screen.getByRole("button", { name: "Tools" }))
    await user.click(await screen.findByText("Relationship Matrix"))
    const matrixStatus = await screen.findByText("Matrix view active (2 characters).")
    const matrixScope = matrixStatus.closest(".ant-modal-content") || document.body

    const results = await runA11yBaselineRules(matrixScope)
    expectNoInvalidAriaViolations(results.violations)
  }, 30000)
})
