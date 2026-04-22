// @vitest-environment jsdom
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
  storeActions,
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
  storeActions: {
    setHistoryId: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTitle: vi.fn()
  },
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    listDictionaries: vi.fn(async () => ({ dictionaries: [] })),
    listChats: vi.fn(async () => []),
    getChatSettings: vi.fn(async () => ({ settings: {} })),
    updateChatSettings: vi.fn(async () => ({ settings: {} }))
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

describe("DictionariesManager chat integration stage-1", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.history.pushState(null, "", "#/settings/dictionaries")

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn(),
      getQueryData: vi.fn()
    })

    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))

    tldwClientMock.getChatSettings.mockImplementation(async (...args: [string?]) => {
      const chatId = args[0]
      if (chatId === "chat-001") {
        return {
          settings: {
            chat_dictionary_ids: [7]
          }
        }
      }
      throw new Error("404 chat settings not found")
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 42,
              name: "Medical Terms",
              description: "Clinical substitutions",
              is_active: true,
              entry_count: 3,
              used_by_chat_count: 2,
              used_by_active_chat_count: 1,
              used_by_chat_refs: [
                {
                  chat_id: "chat-001",
                  title: "ER Intake",
                  state: "in-progress"
                }
              ]
            }
          ]
        })
      }
      if (key === "tldw:listChatsForDictionaryAssign") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: "chat-001",
              title: "ER Intake",
              state: "in-progress",
              created_at: "2026-02-18T10:00:00Z"
            },
            {
              id: "chat-002",
              title: "ICU Follow-up",
              state: "resolved",
              created_at: "2026-02-18T11:00:00Z"
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  it("opens linked chat context from the Used by column", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "Open most recent linked chat for Medical Terms"
      })
    )

    expect(storeActions.setHistoryId).toHaveBeenCalledWith(null, {
      preserveServerChatId: true
    })
    expect(storeActions.setServerChatId).toHaveBeenCalledWith("chat-001")
    expect(storeActions.setServerChatState).toHaveBeenCalledWith("in-progress")
    expect(storeActions.setServerChatTitle).toHaveBeenCalledWith("ER Intake")
    expect(window.location.hash).toBe("#/")
  }, 30000)

  it("assigns the dictionary to selected chat sessions", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "Quick assign Medical Terms to chats"
      })
    )

    await user.click(screen.getByRole("checkbox", { name: "Select chat ICU Follow-up" }))

    await user.click(
      screen.getByRole("button", {
        name: "Assign to 2 chats"
      })
    )

    await waitFor(() => {
      expect(tldwClientMock.updateChatSettings).toHaveBeenCalledTimes(2)
    })

    const updateCalls = tldwClientMock.updateChatSettings.mock.calls.map(
      (call: any[]) => ({
        chatId: call[0],
        patch: call[1]
      })
    )

    expect(updateCalls).toEqual(
      expect.arrayContaining([
        {
          chatId: "chat-001",
          patch: {
            chat_dictionary_ids: [7, 42]
          }
        },
        {
          chatId: "chat-002",
          patch: {
            chat_dictionary_ids: [42],
            chat_dictionary_id: 42
          }
        }
      ])
    )

    const queryClient = useQueryClientMock.mock.results[0]?.value
    expect(queryClient.invalidateQueries).toHaveBeenCalledWith({
      queryKey: ["tldw:listDictionaries"]
    })
    expect(notificationMock.success).toHaveBeenCalled()
  }, 45000)
})
