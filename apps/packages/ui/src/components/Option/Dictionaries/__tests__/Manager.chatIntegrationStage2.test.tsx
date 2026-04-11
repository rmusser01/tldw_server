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
  confirmDangerMock,
  notificationMock,
  tldwClientMock,
  storeActions
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  confirmDangerMock: vi.fn(async () => false),
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
    updateDictionary: vi.fn(async () => ({})),
    deleteDictionary: vi.fn(async () => ({}))
  },
  storeActions: {
    setHistoryId: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTitle: vi.fn()
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
  useConfirmDanger: () => confirmDangerMock
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: any) => {
    const state = {
      setHistoryId: storeActions.setHistoryId,
      setServerChatId: storeActions.setServerChatId,
      setServerChatState: storeActions.setServerChatState,
      setServerChatTitle: storeActions.setServerChatTitle
    }
    return selector ? selector(state) : state
  }
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

describe("DictionariesManager chat integration stage-2", () => {
  beforeEach(() => {
    vi.clearAllMocks()

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
              id: 11,
              name: "Alpha Dictionary",
              description: "First",
              is_active: true,
              processing_priority: 1,
              used_by_chat_count: 3,
              used_by_active_chat_count: 1,
              entry_count: 2
            },
            {
              id: 22,
              name: "Beta Dictionary",
              description: "Second",
              is_active: true,
              processing_priority: 2,
              used_by_chat_count: 0,
              used_by_active_chat_count: 0,
              entry_count: 1
            },
            {
              id: 33,
              name: "Gamma Dictionary",
              description: "Disabled",
              is_active: false,
              processing_priority: null,
              used_by_chat_count: 0,
              used_by_active_chat_count: 0,
              entry_count: 1
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  it("shows processing-priority guidance and values in the list", () => {
    render(<DictionariesManager />)

    expect(
      screen.getByText(
        "Processing order for active dictionaries uses Priority (alphabetical by dictionary name), then each dictionary's entry order."
      )
    ).toBeInTheDocument()
    expect(screen.getByRole("columnheader", { name: "Priority" })).toBeInTheDocument()
    expect(screen.getByText("P1")).toBeInTheDocument()
    expect(screen.getByText("P2")).toBeInTheDocument()
    expect(screen.getByText("inactive")).toBeInTheDocument()
  })

  it("shows linked-chat warnings for deactivate and delete actions", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("switch", {
        name: "Set dictionary Alpha Dictionary inactive"
      })
    )

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Deactivate dictionary?"
        })
      )
    })

    const confirmDangerCalls = confirmDangerMock.mock
      .calls as unknown as Array<[Record<string, unknown>]>
    const deactivateCall = confirmDangerCalls[0]?.[0]
    expect(String(deactivateCall?.content || "")).toContain("1 active chat session")
    expect(String(deactivateCall?.content || "")).toContain("3 linked chat sessions")

    const overflowButton = screen.getByRole("button", {
      name: "More actions for Alpha Dictionary"
    })
    overflowButton.focus()
    await user.keyboard("{Enter}")
    await waitFor(() => {
      expect(screen.getByRole("menuitem", { name: "Delete dictionary" })).toBeInTheDocument()
    })
    await user.click(screen.getByRole("menuitem", { name: "Delete dictionary" }))

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalledTimes(2)
    })

    const deleteCall = confirmDangerCalls[1]?.[0]
    expect(String(deleteCall?.content || "")).toContain(
      "linked to 3 chat session(s), including 1 active session(s)"
    )
  }, 30000)
})
