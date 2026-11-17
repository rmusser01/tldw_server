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
  tldwClientMock,
  confirmDangerMock
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
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    dictionaryStatistics: vi.fn(async () => ({
      dictionary_id: 101,
      name: "Mobile Dictionary",
      total_entries: 1,
      regex_entries: 0,
      literal_entries: 1,
      groups: [],
      average_probability: 1,
      total_usage_count: 0
    })),
    dictionaryActivity: vi.fn(async () => ({
      dictionary_id: 101,
      events: [],
      total: 0,
      limit: 10,
      offset: 0
    })),
    exportDictionaryJSON: vi.fn(async () => ({ name: "Mobile Dictionary", entries: [] })),
    exportDictionaryMarkdown: vi.fn(async () => ({ content: "# dictionary" })),
    updateDictionary: vi.fn(async () => ({})),
    deleteDictionary: vi.fn(async () => ({})),
    getDictionary: vi.fn(async () => ({})),
    importDictionaryJSON: vi.fn(async () => ({})),
    importDictionaryMarkdown: vi.fn(async () => ({})),
    listChats: vi.fn(async () => []),
    getChatSettings: vi.fn(async () => ({ settings: {} })),
    updateChatSettings: vi.fn(async () => ({})),
    listDictionaryEntries: vi.fn(async () => ({ entries: [] })),
    validateDictionary: vi.fn(async () => ({ errors: [], warnings: [] }))
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
  useConfirmDanger: () => confirmDangerMock
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

describe("DictionariesManager responsive stage-1", () => {
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
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 101,
              name: "Mobile Dictionary",
              description: "compact actions",
              is_active: true,
              entry_count: 1
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

  it("keeps primary actions visible and moves secondary actions into overflow on mobile", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    expect(
      screen.getByRole("button", { name: "Edit dictionary Mobile Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Manage entries for Mobile Dictionary" })
    ).toBeInTheDocument()

    const overflowButton = screen.getByRole("button", {
      name: "More actions for Mobile Dictionary"
    })
    expect(overflowButton).toHaveAttribute("aria-haspopup", "menu")

    expect(
      screen.queryByRole("button", { name: "Export Mobile Dictionary as JSON" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Delete dictionary Mobile Dictionary" })
    ).not.toBeInTheDocument()

    overflowButton.focus()
    await user.keyboard("{Enter}")
    await waitFor(() => {
      expect(screen.getByRole("menuitem", { name: "View statistics" })).toBeInTheDocument()
    })

    await user.click(screen.getByRole("menuitem", { name: "View statistics" }))
    await waitFor(() => {
      expect(tldwClientMock.dictionaryStatistics).toHaveBeenCalledWith(101)
    })
  }, 30000)
})
