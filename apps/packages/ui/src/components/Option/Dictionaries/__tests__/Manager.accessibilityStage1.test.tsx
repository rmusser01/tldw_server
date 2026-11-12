import { dictionaryOwnerFixture, dictionarySnapshotFixture, scopedDictionaryQueryFixture, scopedDictionaryQueryClientFixture, selectDictionaryChatFixture } from "./dictionary-owner.fixture"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
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
      id: 401,
      name: "Accessible Dictionary",
      description: "Accessibility checks"
    })),
    listDictionaryEntries: vi.fn(async () => ({
      entries: [
        {
          id: 21,
          dictionary_id: 401,
          pattern: "BP",
          replacement: "blood pressure",
          type: "literal",
          probability: 1,
          enabled: true,
          case_sensitive: false,
          max_replacements: 0
        }
      ]
    })),
    updateDictionary: vi.fn(async () => ({})),
    validateDictionary: vi.fn(async () => ({ errors: [], warnings: [] })),
    processDictionaryText: vi.fn(async () => ({
      original_text: "BP",
      processed_text: "blood pressure",
      replacements: 1,
      iterations: 1,
      entries_used: ["BP"]
    })),
    addDictionaryEntry: vi.fn(async () => ({})),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    reorderDictionaryEntries: vi.fn(async () => ({})),
    bulkDictionaryEntries: vi.fn(async () => ({
      success: true,
      affected_count: 1,
      failed_ids: []
    })),
    importDictionaryJSON: vi.fn(async () => ({})),
    importDictionaryMarkdown: vi.fn(async () => ({})),
    exportDictionaryJSON: vi.fn(async () => ({ name: "Accessible Dictionary", entries: [] })),
    exportDictionaryMarkdown: vi.fn(async () => ({ content: "# dictionary" })),
    dictionaryStatistics: vi.fn(async () => ({})),
    dictionaryActivity: vi.fn(async () => ({ events: [], total: 0, limit: 10, offset: 0 })),
    listChats: vi.fn(async () => [])
  }
}))

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

describe("DictionariesManager accessibility stage-1", () => {
  beforeEach(() => {
    vi.clearAllMocks()

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
              id: 401,
              name: "Accessible Dictionary",
              description: "Accessibility checks",
              is_active: true,
              entry_count: 1,
              version: 3
            }
          ]
        })
      }

      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: {
            id: 401,
            name: "Accessible Dictionary",
            description: "Accessibility checks"
          }
        })
      }

      if (key === "tldw:listDictionaryEntries" || key === "tldw:listDictionaryEntriesAll") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 21,
              dictionary_id: 401,
              pattern: "BP",
              replacement: "blood pressure",
              type: "literal",
              probability: 1,
              enabled: true,
              case_sensitive: false,
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

  it("supports keyboard toggling of active dictionary switch from list view", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const activeSwitch = screen.getByRole("switch", {
      name: "Set dictionary Accessible Dictionary inactive"
    })
    activeSwitch.focus()

    await user.keyboard("{Enter}")

    await waitFor(() => {
      expect(tldwClientMock.updateDictionary).toHaveBeenCalledWith(
        401,
        expect.objectContaining({
          is_active: false,
          version: 3
        })
      )
    })
  })

  it("exposes collapse panel semantics and labelled regions for validation/preview", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "Manage entries for Accessible Dictionary"
      })
    )
    await screen.findByText("Manage Entries: Accessible Dictionary")

    const validateToggle = screen
      .getAllByRole("button", { name: /Validate dictionary/i })
      .find((element) => element.hasAttribute("aria-expanded"))
    expect(validateToggle).toBeDefined()
    expect(validateToggle).toHaveAttribute("aria-expanded", "false")

    await user.click(validateToggle!)
    expect(validateToggle).toHaveAttribute("aria-expanded", "true")

    expect(screen.getByTestId("dictionary-validation-panel")).toHaveAttribute("role", "region")

    const previewToggle = screen
      .getAllByRole("button", { name: /Preview transforms/i })
      .find((element) => element.hasAttribute("aria-expanded"))
    expect(previewToggle).toBeDefined()
    await user.click(previewToggle!)
    expect(previewToggle).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByTestId("dictionary-preview-panel")).toHaveAttribute("role", "region")
  }, 40000)
})
