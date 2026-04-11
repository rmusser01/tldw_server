import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionariesManager } from "../Manager"

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
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    getDictionary: vi.fn(async () => ({
      id: 302,
      name: "Mobile Preview",
      description: "Dictionary for preview controls"
    })),
    listDictionaryEntries: vi.fn(async () => ({
      entries: [
        {
          id: 12,
          dictionary_id: 302,
          pattern: "BP",
          replacement: "blood pressure",
          type: "literal",
          probability: 1,
          enabled: true,
          case_sensitive: false,
          group: "clinical",
          max_replacements: 0
        }
      ]
    })),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    reorderDictionaryEntries: vi.fn(async () => ({})),
    bulkDictionaryEntries: vi.fn(async () => ({
      success: true,
      affected_count: 1,
      failed_ids: []
    })),
    addDictionaryEntry: vi.fn(async () => ({})),
    updateDictionary: vi.fn(async () => ({})),
    deleteDictionary: vi.fn(async () => ({})),
    importDictionaryJSON: vi.fn(async () => ({})),
    importDictionaryMarkdown: vi.fn(async () => ({})),
    exportDictionaryJSON: vi.fn(async () => ({ name: "Mobile Preview", entries: [] })),
    exportDictionaryMarkdown: vi.fn(async () => ({ content: "# dictionary" })),
    validateDictionary: vi.fn(async () => ({ errors: [], warnings: [] })),
    processDictionaryText: vi.fn(async () => ({
      original_text: "BP reading",
      processed_text: "blood pressure reading",
      replacements: 1,
      iterations: 1,
      entries_used: ["BP"]
    })),
    listChats: vi.fn(async () => [])
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
  default: ({ title }: { title: string }) => <div>{title}</div>
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

vi.mock("@/store/option", () => ({
  useStoreMessageOption: () => ({
    setHistoryId: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTitle: vi.fn()
  })
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

describe("DictionariesManager responsive stage-3", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: (query: string) => ({
        matches: /max-width:\s*767px/.test(query),
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false
      })
    })

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
              id: 302,
              name: "Mobile Preview",
              description: "Responsive preview controls",
              is_active: true,
              entry_count: 1
            }
          ]
        })
      }

      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: {
            id: 302,
            name: "Mobile Preview",
            description: "Responsive preview controls"
          }
        })
      }

      if (key === "tldw:listDictionaryEntries" || key === "tldw:listDictionaryEntriesAll") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 12,
              dictionary_id: 302,
              pattern: "BP",
              replacement: "blood pressure",
              type: "literal",
              probability: 1,
              enabled: true,
              case_sensitive: false,
              group: "clinical",
              max_replacements: 0
            }
          ]
        })
      }

      if (key === "tldw:listChatsForDictionaryAssign") {
        return makeUseQueryResult({
          status: "success",
          data: []
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("stacks preview settings controls on mobile widths", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Mobile Preview" })
    )
    await screen.findByText("Manage Entries: Mobile Preview")

    await user.click(screen.getByRole("button", { name: "Run preview" }))
    await screen.findByText("Processing limit")

    const previewCaseControls = screen.getByTestId("dictionary-preview-case-controls")
    expect(previewCaseControls).toHaveClass("flex-col")
    expect(previewCaseControls).toHaveClass("sm:flex-row")

    const previewControlsGrid = screen.getByTestId("dictionary-preview-controls-grid")
    expect(previewControlsGrid).toHaveClass("grid-cols-1")
    expect(previewControlsGrid).toHaveClass("md:grid-cols-2")
  }, 40000)
})
