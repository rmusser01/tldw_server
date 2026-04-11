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
  confirmDangerMock: vi.fn(async () => true),
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
    createDictionary: vi.fn(async () => ({ id: 99 })),
    addDictionaryEntry: vi.fn(async () => ({})),
    updateDictionary: vi.fn(async () => ({})),
    deleteDictionary: vi.fn(async () => ({})),
    dictionaryStatistics: vi.fn(async () => ({
      dictionary_id: 7,
      name: "Activity Dictionary",
      total_entries: 2,
      regex_entries: 0,
      literal_entries: 2,
      enabled_entries: 2,
      disabled_entries: 0,
      probabilistic_entries: 0,
      timed_effect_entries: 0,
      zero_usage_entries: 0,
      pattern_conflict_count: 0,
      groups: [],
      average_probability: 1,
      created_at: "2026-02-18T10:00:00Z",
      updated_at: "2026-02-18T10:30:00Z",
      last_used: "2026-02-18T11:00:00Z",
      total_usage_count: 2,
      entry_usage: []
    })),
    dictionaryActivity: vi.fn(async () => ({
      dictionary_id: 7,
      total: 1,
      limit: 10,
      offset: 0,
      events: [
        {
          id: 1,
          dictionary_id: 7,
          chat_id: "chat-123",
          entries_used: [11, 12],
          replacements: 2,
          iterations: 1,
          token_budget_used: 320,
          original_text_preview: "foo bar",
          processed_text_preview: "FOO BAR",
          created_at: "2026-02-18T11:00:00Z"
        }
      ]
    })),
    dictionaryVersions: vi.fn(async () => ({
      dictionary_id: 7,
      total: 2,
      limit: 30,
      offset: 0,
      versions: [
        {
          revision: 3,
          source_dictionary_version: 3,
          change_type: "entry_update",
          summary: "Updated abbreviation",
          entry_count: 2,
          created_at: "2026-02-18T12:00:00Z"
        },
        {
          revision: 2,
          source_dictionary_version: 2,
          change_type: "metadata_update",
          summary: "Updated tags",
          entry_count: 2,
          created_at: "2026-02-18T11:00:00Z"
        }
      ]
    })),
    dictionaryVersionSnapshot: vi.fn(async () => ({
      dictionary_id: 7,
      revision: 3,
      source_dictionary_version: 3,
      change_type: "entry_update",
      summary: "Updated abbreviation",
      created_at: "2026-02-18T12:00:00Z",
      dictionary: {
        id: 7,
        name: "Activity Dictionary",
        description: "Tracks replacements",
        category: "Medical",
        tags: ["clinical", "urgent"],
        is_active: true,
        default_token_budget: 320,
        version: 3,
        created_at: "2026-02-18T10:00:00Z",
        updated_at: "2026-02-18T12:00:00Z"
      },
      entries: [
        {
          id: 11,
          dictionary_id: 7,
          pattern: "foo",
          replacement: "FOO",
          type: "literal",
          probability: 1,
          group: null,
          timed_effects: { sticky: 0, cooldown: 0, delay: 0 },
          max_replacements: 0,
          enabled: true,
          case_sensitive: true,
          created_at: "2026-02-18T10:00:00Z",
          updated_at: "2026-02-18T12:00:00Z"
        }
      ]
    })),
    revertDictionaryVersion: vi.fn(async () => ({
      dictionary_id: 7,
      reverted_to_revision: 3,
      current_dictionary_version: 4,
      current_revision: 4,
      message: "Dictionary reverted to revision 3."
    }))
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

describe("DictionariesManager chat integration stage-3", () => {
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
              id: 7,
              name: "Activity Dictionary",
              description: "Tracks replacements",
              category: "Medical",
              tags: ["clinical", "urgent"],
              is_active: true,
              default_token_budget: 320,
              entry_count: 2
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  it("submits default token budget when creating dictionaries", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(screen.getByRole("button", { name: "New Dictionary" }))
    await user.type(screen.getByRole("textbox", { name: "Name" }), "Clinical Terms")
    await user.type(
      screen.getByRole("spinbutton", { name: "Processing limit" }),
      "450"
    )
    await user.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(tldwClientMock.createDictionary).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Clinical Terms",
          default_token_budget: 450
        })
      )
    })
  }, 30000)

  it("supports keyboard shortcuts for creating and submitting dictionary forms", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.keyboard("{Control>}n{/Control}")

    const nameInput = await screen.findByRole("textbox", { name: "Name" })
    await user.type(nameInput, "Shortcut Dictionary")
    await user.keyboard("{Control>}{Enter}{/Control}")

    await waitFor(() => {
      expect(tldwClientMock.createDictionary).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Shortcut Dictionary"
        })
      )
    })
  }, 30000)

  it("submits category and tags when creating dictionaries", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(screen.getByRole("button", { name: "New Dictionary" }))
    await user.type(screen.getByRole("textbox", { name: "Name" }), "Tagged Dictionary")
    await user.type(screen.getByRole("textbox", { name: "Category" }), "Operations")
    await user.click(screen.getByRole("combobox", { name: "Tags" }))
    await user.keyboard("oncall{Enter}triage{Enter}")
    await user.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(tldwClientMock.createDictionary).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Tagged Dictionary",
          category: "Operations",
          tags: ["oncall", "triage"]
        })
      )
    })
  }, 30000)

  it("applies starter template entries when selected during create", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(screen.getByRole("button", { name: "New Dictionary" }))
    await user.type(screen.getByRole("textbox", { name: "Name" }), "Medical Glossary")
    await user.click(screen.getByRole("combobox", { name: "Starter Template" }))
    await user.click(
      await screen.findByText(/Medical Abbreviations/i, {
        selector: ".ant-select-item-option-content"
      })
    )
    await user.click(screen.getByRole("button", { name: "Create" }))

    await waitFor(() => {
      expect(tldwClientMock.createDictionary).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Medical Glossary"
        })
      )
      expect(tldwClientMock.createDictionary).toHaveBeenCalledWith(
        expect.not.objectContaining({
          starter_template: expect.anything()
        })
      )
    })

    await waitFor(() => {
      expect(tldwClientMock.addDictionaryEntry).toHaveBeenCalledTimes(3)
    })
    expect(tldwClientMock.addDictionaryEntry).toHaveBeenNthCalledWith(
      1,
      99,
      expect.objectContaining({
        pattern: "BP",
        replacement: "blood pressure"
      })
    )
  }, 30000)

  it("renders recent activity and default token budget in the statistics modal", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    const overflowButton = screen.getByRole("button", {
      name: "More actions for Activity Dictionary"
    })
    overflowButton.focus()
    await user.keyboard("{Enter}")
    await waitFor(() => {
      expect(screen.getByRole("menuitem", { name: "View statistics" })).toBeInTheDocument()
    })
    await user.click(screen.getByRole("menuitem", { name: "View statistics" }))

    await waitFor(() => {
      expect(tldwClientMock.dictionaryStatistics).toHaveBeenCalledWith(7)
    })
    await waitFor(() => {
      expect(tldwClientMock.dictionaryActivity).toHaveBeenCalledWith(7, {
        limit: 10,
        offset: 0
      })
    })

    expect(
      await screen.findByText("Dictionary Statistics", undefined, { timeout: 20000 })
    ).toBeInTheDocument()
    expect(
      await screen.findByText("Recent activity", undefined, { timeout: 20000 })
    ).toBeInTheDocument()
    expect(
      await screen.findByText("Chat: chat-123", undefined, { timeout: 20000 })
    ).toBeInTheDocument()
    expect(
      await screen.findByText("Entries: 11, 12", undefined, { timeout: 20000 })
    ).toBeInTheDocument()

    // Processing limit is inside the collapsed "Advanced" section
    await user.click(screen.getByText("Advanced"))
    expect(
      await screen.findByText("320 tokens", undefined, { timeout: 20000 })
    ).toBeInTheDocument()
  }, 30000)

  it("paginates recent activity and fetches the next offset", async () => {
    const user = userEvent.setup()
    tldwClientMock.dictionaryActivity
      .mockResolvedValueOnce({
        dictionary_id: 7,
        total: 12,
        limit: 10,
        offset: 0,
        events: [
          {
            id: 1,
            dictionary_id: 7,
            chat_id: "chat-123",
            entries_used: [11],
            replacements: 1,
            iterations: 1,
            token_budget_used: 220,
            original_text_preview: "foo",
            processed_text_preview: "FOO",
            created_at: "2026-02-18T11:00:00Z",
          },
        ],
      })
      .mockResolvedValueOnce({
        dictionary_id: 7,
        total: 12,
        limit: 10,
        offset: 10,
        events: [
          {
            id: 11,
            dictionary_id: 7,
            chat_id: "chat-456",
            entries_used: [12],
            replacements: 1,
            iterations: 1,
            token_budget_used: 180,
            original_text_preview: "bar",
            processed_text_preview: "BAR",
            created_at: "2026-02-18T11:05:00Z",
          },
        ],
      })

    render(<DictionariesManager />)

    {
      const overflowBtn = screen.getByRole("button", {
        name: "More actions for Activity Dictionary",
      })
      overflowBtn.focus()
      await user.keyboard("{Enter}")
      await waitFor(() => {
        expect(screen.getByRole("menuitem", { name: "View statistics" })).toBeInTheDocument()
      })
      await user.click(screen.getByRole("menuitem", { name: "View statistics" }))
    }

    expect(
      await screen.findByText("Page 1 of 2", undefined, { timeout: 20000 })
    ).toBeInTheDocument()
    expect(
      await screen.findByText("Chat: chat-123", undefined, { timeout: 20000 })
    ).toBeInTheDocument()

    await user.click(await screen.findByTestId("dictionary-activity-next-page"))

    await waitFor(() => {
      expect(tldwClientMock.dictionaryActivity).toHaveBeenCalledWith(7, {
        limit: 10,
        offset: 10,
      })
    })
    expect(
      await screen.findByText("Chat: chat-456", undefined, { timeout: 20000 })
    ).toBeInTheDocument()
    expect(screen.queryByText("Chat: chat-123")).not.toBeInTheDocument()
  }, 30000)

  it("filters dictionaries by category and tags in the list toolbar", async () => {
    const user = userEvent.setup()
    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 7,
              name: "Activity Dictionary",
              description: "Tracks replacements",
              category: "Medical",
              tags: ["clinical", "urgent"],
              is_active: true,
              entry_count: 2
            },
            {
              id: 8,
              name: "Casual Dictionary",
              description: "Everyday language",
              category: "Social",
              tags: ["casual"],
              is_active: true,
              entry_count: 1
            }
          ]
        })
      }
      return makeUseQueryResult({})
    })

    render(<DictionariesManager />)

    expect(screen.getByText("Activity Dictionary")).toBeInTheDocument()
    expect(screen.getByText("Casual Dictionary")).toBeInTheDocument()

    await user.click(
      screen.getByRole("combobox", { name: "Filter dictionaries by category" })
    )
    await user.click(
      await screen.findByText("Medical", {
        selector: ".ant-select-item-option-content"
      })
    )

    await waitFor(() => {
      expect(screen.getByText("Activity Dictionary")).toBeInTheDocument()
      expect(screen.queryByText("Casual Dictionary")).not.toBeInTheDocument()
    })

    await user.click(
      screen.getByRole("combobox", { name: "Filter dictionaries by tags" })
    )
    await user.click(
      await screen.findByText("urgent", {
        selector: ".ant-select-item-option-content"
      })
    )

    await waitFor(() => {
      expect(screen.getByText("Activity Dictionary")).toBeInTheDocument()
      expect(screen.queryByText("Casual Dictionary")).not.toBeInTheDocument()
    })
  }, 30000)

  it("opens dictionary version history and reverts the selected revision", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    {
      const overflowBtn = screen.getByRole("button", {
        name: "More actions for Activity Dictionary"
      })
      overflowBtn.focus()
      await user.keyboard("{Enter}")
      await waitFor(() => {
        expect(screen.getByRole("menuitem", { name: "Version history" })).toBeInTheDocument()
      })
      await user.click(screen.getByRole("menuitem", { name: "Version history" }))
    }

    expect(
      await screen.findByText("Dictionary Version History - Activity Dictionary")
    ).toBeInTheDocument()

    await waitFor(() => {
      expect(tldwClientMock.dictionaryVersions).toHaveBeenCalledWith(7, {
        limit: 30,
        offset: 0
      })
    })
    await waitFor(() => {
      expect(tldwClientMock.dictionaryVersionSnapshot).toHaveBeenCalledWith(7, 3)
    })

    await user.click(screen.getByRole("button", { name: "Revert to revision 3" }))

    await waitFor(() => {
      expect(tldwClientMock.revertDictionaryVersion).toHaveBeenCalledWith(7, 3)
    })
  }, 30000)
})
