import { dictionaryOwnerFixture, dictionarySnapshotFixture, scopedDictionaryQueryFixture, scopedDictionaryQueryClientFixture, selectDictionaryChatFixture } from "./dictionary-owner.fixture"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionariesManager } from "../Manager"

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches:
      /min-width:\s*576px/.test(query) ||
      /min-width:\s*768px/.test(query) ||
      /min-width:\s*992px/.test(query),
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false
  })
})

if (typeof window.ResizeObserver === "undefined") {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
  ;(globalThis as any).ResizeObserver = ResizeObserverMock
}

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  tldwClientMock,
  scrollIntoViewMock
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
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    listDictionaries: vi.fn(async () => ({ dictionaries: [] })),
    getDictionary: vi.fn(async () => ({})),
    listDictionaryEntries: vi.fn(async () => ({ entries: [] })),
    createDictionary: vi.fn(async () => ({ id: 1000 })),
    updateDictionary: vi.fn(async () => ({})),
    deleteDictionary: vi.fn(async () => ({})),
    duplicateDictionary: vi.fn(async () => ({})),
    addDictionaryEntry: vi.fn(async () => ({ id: 901 })),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    bulkDictionaryEntries: vi.fn(async () => ({ success: true })),
    reorderDictionaryEntries: vi.fn(async () => ({ success: true })),
    importDictionaryJson: vi.fn(async () => ({})),
    importDictionary: vi.fn(async () => ({})),
    exportDictionaryJson: vi.fn(async () => "{}"),
    exportDictionaryMarkdown: vi.fn(async () => "# dictionary"),
    getDictionaryStatistics: vi.fn(async () => ({})),
    validateDictionary: vi.fn(async () => ({
      ok: true,
      schema_version: 1,
      errors: [],
      warnings: [],
      entry_stats: {
        total: 1,
        literal: 1,
        regex: 0
      }
    })),
    processDictionary: vi.fn(async () => ({
      processed_text: "preview output",
      replacements: 1,
      iterations: 1,
      entries_used: [1],
      token_budget_exceeded: false
    }))
  },
  scrollIntoViewMock: vi.fn()
}))

window.HTMLElement.prototype.scrollIntoView = scrollIntoViewMock

vi.mock("@/hooks/useChatDraftOwner", () => ({ useChatDraftOwner: () => dictionaryOwnerFixture }))
vi.mock("@/hooks/chat/useSelectServerChat", () => ({ useSelectServerChat: () => selectDictionaryChatFixture }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: (...args: Parameters<typeof dictionarySnapshotFixture>) => dictionarySnapshotFixture(...args) }))

vi.mock("@tanstack/react-query", () => ({
  useQuery: (options: { queryKey?: readonly unknown[] }) => scopedDictionaryQueryFixture(options, useQueryMock(options)),
  useMutation: useMutationMock,
  useQueryClient: () => scopedDictionaryQueryClientFixture(useQueryClientMock())
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

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: {
      hasChatDictionaries: true
    }
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({ title, description }: any) => (
    <div>
      <h2>{title}</h2>
      {description ? <p>{description}</p> : null}
    </div>
  )
}))

vi.mock("@/components/Common/LabelWithHelp", () => ({
  LabelWithHelp: ({ label }: { label: React.ReactNode }) => <span>{label}</span>
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(async () => true)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: undefined,
  status: "success",
  error: null,
  isPending: false,
  isFetching: false,
  isLoading: false,
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
    } finally {
      opts?.onSettled?.(undefined, undefined, variables, undefined)
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
    } finally {
      opts?.onSettled?.(undefined, undefined, variables, undefined)
    }
  },
  isPending: false
})

describe("DictionariesManager validation stage-1 discoverability", () => {
  let entryRows: any[] = []

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    entryRows = [
      {
        id: 1,
        dictionary_id: 77,
        pattern: "BP",
        replacement: "blood pressure",
        type: "literal",
        probability: 1,
        group: "Clinical",
        enabled: true,
        case_sensitive: false
      }
    ]

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn(),
      getQueryData: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 77,
              name: "Medical Terms",
              description: "Clinical substitutions",
              is_active: true,
              entry_count: entryRows.length
            }
          ]
        })
      }

      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: {
            id: 77,
            name: "Medical Terms",
            description: "Clinical substitutions"
          }
        })
      }

      if (key === "tldw:listDictionaryEntries" || key === "tldw:listDictionaryEntriesAll") {
        return makeUseQueryResult({
          status: "success",
          data: entryRows
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("exposes header validation/preview actions and keeps inline entry testing", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    expect(
      screen.getByRole("button", { name: "Run validation" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Run preview" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Test entry BP" })
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Run validation" }))

    await waitFor(() => {
      expect(tldwClientMock.validateDictionary).toHaveBeenCalledTimes(1)
    })
    expect(screen.getByText("Valid")).toBeInTheDocument()
  }, 120000)

  it("preserves empty-entry validation guard in the new header action bar", async () => {
    const user = userEvent.setup()
    entryRows = []
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    expect(
      screen.getByRole("button", { name: "Run validation" })
    ).toBeDisabled()
    expect(
      screen.getAllByText("Add at least one entry to validate.").length
    ).toBeGreaterThan(0)
  }, 120000)

  it("allows validation findings to jump to and highlight matching entry rows", async () => {
    const user = userEvent.setup()
    tldwClientMock.validateDictionary.mockResolvedValueOnce({
      ok: false,
      schema_version: 1,
      errors: [
        {
          code: "entry_pattern_conflict",
          field: "entries[0].pattern",
          message: "Pattern conflicts with another rule."
        }
      ],
      warnings: [],
      entry_stats: {
        total: 1,
        literal: 1,
        regex: 0
      }
    })

    render(<DictionariesManager />)
    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await user.click(screen.getByRole("button", { name: "Run validation" }))

    const findingButton = await screen.findByRole("button", {
      name: /entry_pattern_conflict: Pattern conflicts with another rule\./i
    })
    await user.click(findingButton)

    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled()
    })
    expect(screen.getByText("BP").closest("tr")).toHaveClass("bg-warn/10")
  }, 60000)

  it("renders a diff preview for changed text after running preview", async () => {
    const user = userEvent.setup()
    tldwClientMock.processDictionary.mockResolvedValueOnce({
      processed_text: "Doctor Smith has high blood pressure",
      replacements: 2,
      iterations: 1,
      entries_used: [1],
      token_budget_exceeded: false
    })

    render(<DictionariesManager />)
    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    await user.click(screen.getByRole("button", { name: "Run preview" }))
    await user.type(
      screen.getByPlaceholderText("Paste text to preview dictionary substitutions."),
      "Dr. Smith has high BP"
    )
    await user.click(screen.getAllByRole("button", { name: "Run preview" })[0])

    await waitFor(() => {
      expect(tldwClientMock.processDictionary).toHaveBeenCalledTimes(1)
    })
    expect(screen.getByText("Diff preview")).toBeInTheDocument()
    expect(screen.getByText("Original (with removals)")).toBeInTheDocument()
    expect(screen.getByText("Processed (with additions)")).toBeInTheDocument()
    expect(screen.getByText("Doctor")).toBeInTheDocument()
  }, 60000)

  it("persists preview draft text across manager reopen for the same dictionary", async () => {
    const user = userEvent.setup()
    const firstRender = render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await user.click(screen.getByRole("button", { name: "Run preview" }))
    const previewInput = await screen.findByPlaceholderText(
      "Paste text to preview dictionary substitutions."
    )
    await user.type(previewInput, "Persistent preview draft")

    firstRender.unmount()

    render(<DictionariesManager />)
    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await user.click(screen.getByRole("button", { name: "Run preview" }))

    expect(
      await screen.findByDisplayValue("Persistent preview draft")
    ).toBeInTheDocument()
  }, 60000)

  it("supports save/load/delete lifecycle for dictionary-scoped preview test cases", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await user.click(screen.getByRole("button", { name: "Run preview" }))

    const previewInput = await screen.findByPlaceholderText(
      "Paste text to preview dictionary substitutions."
    )
    await user.type(previewInput, "Case A text")
    await user.type(screen.getByLabelText("Test case name"), "Case A")
    await user.click(screen.getByRole("button", { name: "Save test case" }))

    expect(screen.getByText("Case A")).toBeInTheDocument()
    expect(
      window.localStorage.getItem("tldw:dictionaries:preview-cases:77")
    ).toContain("Case A")

    await user.clear(previewInput)
    await user.type(previewInput, "Temporary text")
    await user.click(
      screen.getByRole("button", { name: "Load test case Case A" })
    )
    expect(
      await screen.findByDisplayValue("Case A text")
    ).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: "Delete test case Case A" })
    )
    expect(screen.queryByText("Case A")).not.toBeInTheDocument()
  }, 60000)

  it("runs validation from Ctrl/Cmd+Shift+V when entry manager is focused on non-input content", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    await user.keyboard("{Control>}{Shift>}v{/Shift}{/Control}")

    await waitFor(() => {
      expect(tldwClientMock.validateDictionary).toHaveBeenCalledTimes(1)
    })
  }, 60000)

  it("does not hijack Ctrl/Cmd+Shift+V while typing in entry inputs", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await user.click(screen.getByLabelText("Search dictionary entries"))
    await user.keyboard("{Control>}{Shift>}v{/Shift}{/Control}")

    expect(tldwClientMock.validateDictionary).not.toHaveBeenCalled()
  }, 60000)
})
