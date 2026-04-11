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
    exportDictionaryJSON: vi.fn(async () => ({
      name: "Accessible Dictionary",
      entries: []
    })),
    exportDictionaryMarkdown: vi.fn(async () => ({ content: "# dictionary" })),
    dictionaryStatistics: vi.fn(async () => ({
      dictionary_id: 401,
      name: "Accessible Dictionary",
      total_entries: 1,
      regex_entries: 0,
      literal_entries: 1
    })),
    dictionaryActivity: vi.fn(async () => ({
      events: [],
      total: 0,
      limit: 10,
      offset: 0
    })),
    listChats: vi.fn(async () => [
      { id: "chat-1", title: "Alpha Chat", state: "in-progress" }
    ])
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

describe("DictionariesManager accessibility stage-2 focus management", () => {
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
          data: [{ id: "chat-1", title: "Alpha Chat", state: "in-progress" }]
        })
      }

      return makeUseQueryResult({})
    })
  })

  const closeTopMostModal = async (user: ReturnType<typeof userEvent.setup>) => {
    const closeButtons = Array.from(
      document.querySelectorAll<HTMLButtonElement>(".ant-modal-close")
    )
    expect(closeButtons.length).toBeGreaterThan(0)
    await user.click(closeButtons[closeButtons.length - 1] as HTMLButtonElement)
  }

  it("returns focus to the create trigger after closing Create Dictionary modal", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const createButton = screen.getByRole("button", { name: "New Dictionary" })
    createButton.focus()
    await user.click(createButton)

    await screen.findByText("Create Dictionary")
    await closeTopMostModal(user)

    await waitFor(() => {
      expect(createButton).toHaveFocus()
    })
  }, 60000)

  it("returns focus to the import trigger after closing Import Dictionary modal", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const importButton = screen.getByRole("button", { name: "Import" })
    importButton.focus()
    await user.click(importButton)

    await screen.findByText("Import Dictionary")
    await closeTopMostModal(user)

    await waitFor(() => {
      expect(importButton).toHaveFocus()
    })
  }, 60000)

  it("returns focus to manage-entries trigger after closing the entries drawer", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const manageEntriesButton = screen.getByRole("button", {
      name: "Manage entries for Accessible Dictionary"
    })
    manageEntriesButton.focus()
    await user.click(manageEntriesButton)
    await screen.findByText("Manage Entries: Accessible Dictionary")

    const closeDrawerButton = document.querySelector<HTMLButtonElement>(".ant-drawer-close")
    expect(closeDrawerButton).not.toBeNull()
    await user.click(closeDrawerButton as HTMLButtonElement)

    await waitFor(() => {
      expect(
        screen.queryByText("Manage Entries: Accessible Dictionary")
      ).not.toBeInTheDocument()
    })
    await waitFor(() => {
      expect(manageEntriesButton).toHaveFocus()
    })
  }, 30000)

  it("opens quick-assign modal via compact dropdown", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const overflowButton = screen.getByRole("button", {
      name: "More actions for Accessible Dictionary"
    })
    overflowButton.focus()
    await user.keyboard("{Enter}")
    await waitFor(() => {
      expect(screen.getByRole("menuitem", { name: "Quick assign to chats" })).toBeInTheDocument()
    })
    await user.click(screen.getByRole("menuitem", { name: "Quick assign to chats" }))

    expect(
      await screen.findByText("Quick assign: Accessible Dictionary")
    ).toBeInTheDocument()

    await closeTopMostModal(user)

    await waitFor(() => {
      expect(overflowButton).toHaveFocus()
    })
  }, 60000)

  it("opens statistics modal via compact dropdown", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const overflowButton = screen.getByRole("button", {
      name: "More actions for Accessible Dictionary"
    })
    overflowButton.focus()
    await user.keyboard("{Enter}")
    await waitFor(() => {
      expect(screen.getByRole("menuitem", { name: "View statistics" })).toBeInTheDocument()
    })
    await user.click(screen.getByRole("menuitem", { name: "View statistics" }))

    expect(
      await screen.findByText("Dictionary Statistics", {}, { timeout: 15000 })
    ).toBeInTheDocument()

    await closeTopMostModal(user)

    await waitFor(() => {
      expect(overflowButton).toHaveFocus()
    })
  }, 90000)

  it("returns focus to entry edit trigger after closing nested Edit Entry modal", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "Manage entries for Accessible Dictionary"
      })
    )
    await screen.findByText("Manage Entries: Accessible Dictionary")

    const editEntryButton = await screen.findByRole("button", { name: "Edit entry BP" })
    editEntryButton.focus()
    await user.click(editEntryButton)

    await screen.findByText("Edit Entry", {}, { timeout: 15000 })
    await closeTopMostModal(user)

    await waitFor(() => {
      expect(editEntryButton).toHaveFocus()
    })
    expect(
      screen.getByText("Manage Entries: Accessible Dictionary")
    ).toBeInTheDocument()
  }, 60000)
})
