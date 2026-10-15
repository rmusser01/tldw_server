import { dictionaryOwnerFixture, dictionarySnapshotFixture, scopedDictionaryQueryFixture, scopedDictionaryQueryClientFixture, selectDictionaryChatFixture } from "./dictionary-owner.fixture"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
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

const { useQueryMock, useMutationMock, useQueryClientMock, notificationMock } =
  vi.hoisted(() => ({
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
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    listDictionaries: vi.fn(async () => ({ dictionaries: [] })),
    getDictionary: vi.fn(async () => ({})),
    listDictionaryEntries: vi.fn(async () => ({ entries: [] })),
    addDictionaryEntry: vi.fn(async () => ({ id: 100 })),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    reorderDictionaryEntries: vi.fn(async () => ({})),
    bulkDictionaryEntries: vi.fn(async () => ({
      success: true,
      affected_count: 1,
      failed_ids: []
    }))
  }
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

describe("DictionariesManager entry stage-4 container architecture", () => {
  beforeEach(() => {
    vi.clearAllMocks()
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
              entry_count: 1
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
          data: [
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
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("opens entry manager inside a drawer instead of a parent modal", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )

    expect(await screen.findByText("Manage Entries: Medical Terms")).toBeInTheDocument()
    expect(document.querySelector(".ant-drawer")).not.toBeNull()
    expect(
      document.querySelector(".ant-modal .ant-modal-title")?.textContent
    ).not.toBe("Manage Entries")
  }, 30000)

  it("still allows opening the entry edit modal while drawer remains mounted", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", { name: "Manage entries for Medical Terms" })
    )
    await screen.findByText("Manage Entries: Medical Terms")

    await user.click(screen.getByRole("button", { name: "Edit entry BP" }))

    expect(await screen.findByText("Edit Entry")).toBeInTheDocument()
    expect(document.querySelector(".ant-drawer")).not.toBeNull()
  }, 30000)
})
