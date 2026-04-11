import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { CharactersManager, withCharacterNameInLabel } from "../Manager"
import { DEFAULT_CHARACTER_STORAGE_KEY } from "@/utils/default-character-preference"
import { ensureLocalStorageApi } from "./testUtils"

const TEMPLATE_CHOOSER_SEEN_KEY = "characters-template-chooser-seen"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  useNavigateMock,
  useCharacterShortcutsMock,
  generateFullCharacterMock,
  generateFieldMock,
  cancelGenerationMock,
  clearGenerationErrorMock,
  useStorageMock,
  exportCharacterToJSONMock,
  exportCharacterToPNGMock,
  exportCharactersToJSONMock,
  confirmDangerMock,
  navigateMock,
  setSelectedCharacterMock,
  focusComposerMock,
  notificationMock,
  storeSetters,
  tldwClientMock,
  templateData
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  useNavigateMock: vi.fn(),
  useCharacterShortcutsMock: vi.fn((_options?: any) => undefined),
  generateFullCharacterMock: vi.fn(),
  generateFieldMock: vi.fn(
    async (
      _field?: string,
      _values?: Record<string, unknown>,
      _options?: Record<string, unknown>
    ) => "Generated value"
  ),
  cancelGenerationMock: vi.fn(),
  clearGenerationErrorMock: vi.fn(),
  useStorageMock: vi.fn(),
  exportCharacterToJSONMock: vi.fn(),
  exportCharacterToPNGMock: vi.fn(),
  exportCharactersToJSONMock: vi.fn(),
  confirmDangerMock: vi.fn(async (_config?: Record<string, unknown>) => true),
  navigateMock: vi.fn(),
  setSelectedCharacterMock: vi.fn(),
  focusComposerMock: vi.fn(),
  storeSetters: {
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setHistoryId: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTopic: vi.fn(),
    setServerChatClusterId: vi.fn(),
    setServerChatSource: vi.fn(),
    setServerChatExternalRef: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatAssistantKind: vi.fn(),
    setServerChatAssistantId: vi.fn(),
    setServerChatMetaLoaded: vi.fn()
  },
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
    getDefaultCharacterPreference: vi.fn(async () => null),
    setDefaultCharacterPreference: vi.fn(async () => ({
      applied: [],
      skipped: []
    })),
    listAllCharacters: vi.fn(async () => []),
    listCharacters: vi.fn(async () => []),
    listCharactersPage: vi.fn(async () => ({
      items: [],
      total: 0,
      page: 1,
      page_size: 25,
      has_more: false
    })),
    listChats: vi.fn(async () => []),
    createChat: vi.fn(async () => ({ id: "quick-chat-session-default" })),
    createCharacter: vi.fn(
      async (_payload?: Record<string, unknown>) =>
        ({ id: "char-1" }) as { id: string | number }
    ),
    updateCharacter: vi.fn(async () => ({})),
    deleteCharacter: vi.fn(async () => ({})),
    deleteChat: vi.fn(async () => undefined),
    restoreCharacter: vi.fn(async () => ({})),
    listCharacterVersions: vi.fn(async () => ({ items: [], total: 0 })),
    diffCharacterVersions: vi.fn(async () => ({
      character_id: 0,
      from_entry: { change_id: 0, version: 1, operation: "update", payload: {} },
      to_entry: { change_id: 0, version: 1, operation: "update", payload: {} },
      changed_fields: [],
      changed_count: 0
    })),
    revertCharacter: vi.fn(async () => ({})),
    importCharacterFile: vi.fn(
      async (
        _file?: File,
        _options?: { allowImageOnly?: boolean }
      ) =>
        ({ success: true }) as { success: boolean; message?: string }
    ),
    exportCharacter: vi.fn(async () => ({})),
    completeCharacterChatTurn: vi.fn(async () => ({
      assistant_content: "Quick chat response"
    })),
    listChatMessages: vi.fn(async () => []),
    getChat: vi.fn(async () => null),
    listWorldBooks: vi.fn(async () => ({ world_books: [] })),
    listCharacterWorldBooks: vi.fn(async () => []),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({})),
    fetchWithAuth: vi.fn()
  },
  templateData: [
    {
      id: "writer-coach",
      name: "Writer Coach",
      description: "Helps with structure and tone.",
      system_prompt: "You are a writing coach.",
      greeting: "Ready to improve your draft?",
      tags: ["writing", "coach"]
    },
    {
      id: "interviewer",
      name: "Interview Trainer",
      description: "Runs realistic interview practice.",
      system_prompt: "You are an interview coach.",
      greeting: "Let's practice your interview skills.",
      tags: ["career"]
    },
    {
      id: "study-buddy",
      name: "Study Buddy",
      description: "Explains difficult concepts simply.",
      system_prompt: "You teach with examples.",
      greeting: "What are we studying today?",
      tags: ["education"]
    }
  ]
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
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const template = fallbackOrOptions.defaultValue || key
        return template.replace(/\{\{(\w+)\}\}/g, (_, token: string) => {
          const value = fallbackOrOptions[token]
          return value == null ? `{{${token}}}` : String(value)
        })
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: useNavigateMock
  }
})

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => confirmDangerMock
}))

vi.mock("@/hooks/useCharacterShortcuts", () => ({
  useCharacterShortcuts: (options: any) => useCharacterShortcutsMock(options)
}))

vi.mock("@/hooks/useCharacterGeneration", () => ({
  useCharacterGeneration: () => ({
    isGenerating: false,
    generatingField: null,
    error: null,
    generateFullCharacter: generateFullCharacterMock,
    generateField: generateFieldMock,
    cancel: cancelGenerationMock,
    clearError: clearGenerationErrorMock
  })
}))

vi.mock("@/hooks/useFormDraft", () => ({
  useFormDraft: () => ({
    hasDraft: false,
    draftData: null,
    saveDraft: vi.fn(),
    clearDraft: vi.fn(),
    applyDraft: vi.fn(() => null),
    dismissDraft: vi.fn(),
    lastSaved: null
  })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, setSelectedCharacterMock]
}))

vi.mock("@/hooks/useComposerFocus", () => ({
  focusComposer: focusComposerMock
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: any) =>
    selector(storeSetters)
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (...args: any[]) => useStorageMock(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(async () => [])
}))

vi.mock("@/utils/character-export", () => ({
  exportCharacterToJSON: (...args: any[]) => exportCharacterToJSONMock(...args),
  exportCharacterToPNG: (...args: any[]) => exportCharacterToPNGMock(...args),
  exportCharactersToJSON: (...args: any[]) => exportCharactersToJSONMock(...args)
}))

vi.mock("@/data/character-templates", () => ({
  CHARACTER_TEMPLATES: templateData
}))

vi.mock("../GenerateCharacterPanel", () => ({
  GenerateCharacterPanel: () => <div data-testid="generate-character-panel" />,
  GenerationPreviewModal: () => null
}))

const makeUseQueryResult = (value: Record<string, unknown>) => ({
  data: undefined,
  status: "success",
  error: null,
  isPending: false,
  isFetching: false,
  isLoading: false,
  refetch: vi.fn(),
  ...value
})

const resolveStorageKey = (key: unknown): string => {
  if (typeof key === "string") return key
  if (key && typeof key === "object" && "key" in key) {
    const nested = (key as { key?: unknown }).key
    return typeof nested === "string" ? nested : ""
  }
  return ""
}

const openAdvancedFilters = async (
  user: ReturnType<typeof userEvent.setup>
) => {
  const advancedFiltersToggle = screen.queryByRole("button", {
    name: /Filters/i
  })
  if (advancedFiltersToggle) {
    await user.click(advancedFiltersToggle)
  }
}

const findEditSubmitButton = async (timeout = 15000) =>
  waitFor(() => {
    const candidate = screen
      .getAllByRole("button", { name: "Save changes" })
      .find((button) => button.getAttribute("type") === "submit")
    expect(candidate).toBeDefined()
    return candidate as HTMLElement
  }, { timeout })

describe("CharactersManager first-use onboarding", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    ensureLocalStorageApi().clear()
    window.history.replaceState({}, "", "/")
    useNavigateMock.mockReturnValue(navigateMock)
    confirmDangerMock.mockResolvedValue(true)
    tldwClientMock.fetchWithAuth.mockReset()
    useStorageMock.mockImplementation(
      (_key: unknown, defaultValue: unknown) => [
        defaultValue ?? null,
        vi.fn(),
        { isLoading: false }
      ]
    )

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn()
    })

    useMutationMock.mockReturnValue({
      mutate: vi.fn(),
      mutateAsync: vi.fn(),
      isPending: false
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })
  })

  const selectTagManagerSourceTag = async (
    user: ReturnType<typeof userEvent.setup>,
    dialog: HTMLElement,
    tagLabelWithCount: RegExp
  ) => {
    const combobox = within(dialog).getByRole("combobox")
    fireEvent.mouseDown(combobox)
    await user.click(await screen.findByText(tagLabelWithCount))
  }

  const selectCharacterWorldBook = async (
    user: ReturnType<typeof userEvent.setup>,
    scope: ReturnType<typeof within>,
    worldBookName: string
  ) => {
    const field = scope
      .getByText("World book attachments")
      .closest(".ant-form-item")
    expect(field).not.toBeNull()

    const placeholder = within(field as HTMLElement).queryByText(
      "Select world book to attach"
    )
    if (placeholder) {
      fireEvent.mouseDown(placeholder)
      await user.click(await screen.findByText(worldBookName))
      await within(field as HTMLElement).findByText(worldBookName)
      return
    }

    const selector =
      (field as HTMLElement).querySelector(".ant-select-selector") ??
      within(field as HTMLElement).getByRole("combobox")

    fireEvent.mouseDown(selector as HTMLElement)
    await user.click(
      await screen.findByText(worldBookName, {
        selector: ".ant-select-item-option-content"
      })
    )
    await within(field as HTMLElement).findByText(worldBookName)
  }

  const getListCharactersQueryOptions = () => {
    const call = useQueryMock.mock.calls.find(([opts]) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      return key === "tldw:listCharacters"
    })
    expect(call).toBeDefined()
    return call?.[0] as any
  }

  const getLatestListCharactersQueryOptions = () => {
    const calls = [...useQueryMock.mock.calls].reverse()
    const call = calls.find(([opts]) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      return key === "tldw:listCharacters"
    })
    expect(call).toBeDefined()
    return call?.[0] as any
  }

  const getDeletedScopeListCharactersQueryOptions = () => {
    const call = useQueryMock.mock.calls.find(([opts]) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      const params = Array.isArray(opts?.queryKey) ? opts?.queryKey?.[1] : undefined
      return key === "tldw:listCharacters" && params?.deleted_only === true
    })
    expect(call).toBeDefined()
    return call?.[0] as any
  }

  it("ensures aria labels include the current character name when localization is stale", () => {
    expect(
      withCharacterNameInLabel(
        "Edit character Compare Alpha",
        "Edit character {{name}}",
        "Default Assistant"
      )
    ).toBe("Edit character Default Assistant")
    expect(
      withCharacterNameInLabel(
        "Chat as {{name}}",
        "Chat as {{name}}",
        "Persona 2"
      )
    ).toBe("Chat as Persona 2")
  })

  it("requests character list payloads with image data for avatar rendering", async () => {
    render(<CharactersManager />)

    const listQuery = getListCharactersQueryOptions()
    expect(listQuery.queryKey[1]).toMatchObject({
      include_image_base64: true
    })

    await listQuery.queryFn()

    expect(tldwClientMock.listCharactersPage).toHaveBeenCalledWith(
      expect.objectContaining({
        include_image_base64: true
      })
    )
    expect(tldwClientMock.listAllCharacters).not.toHaveBeenCalled()
  })

  it("serializes created/updated date filters into query params and clears them", async () => {
    const user = userEvent.setup()
    render(<CharactersManager />)
    await openAdvancedFilters(user)

    fireEvent.change(
      screen.getByLabelText("Filter characters created on or after"),
      { target: { value: "2026-02-01" } }
    )
    fireEvent.change(
      screen.getByLabelText("Filter characters created on or before"),
      { target: { value: "2026-02-03" } }
    )
    fireEvent.change(
      screen.getByLabelText("Filter characters updated on or after"),
      { target: { value: "2026-02-10" } }
    )
    fireEvent.change(
      screen.getByLabelText("Filter characters updated on or before"),
      { target: { value: "2026-02-11" } }
    )

    await waitFor(() => {
      const latestListQuery = getLatestListCharactersQueryOptions()
      expect(latestListQuery.queryKey[1]).toMatchObject({
        created_from: "2026-02-01T00:00:00.000Z",
        created_to: "2026-02-03T23:59:59.999Z",
        updated_from: "2026-02-10T00:00:00.000Z",
        updated_to: "2026-02-11T23:59:59.999Z"
      })
    })

    const latestListQuery = getLatestListCharactersQueryOptions()
    await latestListQuery.queryFn()

    expect(tldwClientMock.listCharactersPage).toHaveBeenCalledWith(
      expect.objectContaining({
        created_from: "2026-02-01T00:00:00.000Z",
        created_to: "2026-02-03T23:59:59.999Z",
        updated_from: "2026-02-10T00:00:00.000Z",
        updated_to: "2026-02-11T23:59:59.999Z"
      })
    )

    const clearButtons = screen.getAllByRole("button", { name: "Clear filters" })
    await user.click(clearButtons[0])

    expect(
      screen.getByLabelText("Filter characters created on or after")
    ).toHaveValue("")
    expect(
      screen.getByLabelText("Filter characters created on or before")
    ).toHaveValue("")
    expect(
      screen.getByLabelText("Filter characters updated on or after")
    ).toHaveValue("")
    expect(
      screen.getByLabelText("Filter characters updated on or before")
    ).toHaveValue("")

    await waitFor(() => {
      const clearedListQuery = getLatestListCharactersQueryOptions()
      expect(clearedListQuery.queryKey[1].created_from).toBeUndefined()
      expect(clearedListQuery.queryKey[1].created_to).toBeUndefined()
      expect(clearedListQuery.queryKey[1].updated_from).toBeUndefined()
      expect(clearedListQuery.queryKey[1].updated_to).toBeUndefined()
    })
  })

  it("forces server query mode for date filters when rollout flag is disabled", async () => {
    const user = userEvent.setup()
    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === "ff_characters_server_query") {
        return [false, vi.fn(), { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    fireEvent.change(
      screen.getByLabelText("Filter characters created on or after"),
      { target: { value: "2026-02-01" } }
    )

    await waitFor(() => {
      const latestListQuery = getLatestListCharactersQueryOptions()
      expect(latestListQuery.queryKey[2]).toBe("server")
      expect(latestListQuery.queryKey[1]).toMatchObject({
        created_from: "2026-02-01T00:00:00.000Z"
      })
    })
  })

  it("maps persisted last-used sort state to server sort params", async () => {
    const records = [
      {
        id: "last-used-1",
        name: "Recently Used Character",
        system_prompt: "Prompt text",
        last_used_at: "2026-02-18T12:00:00.000Z",
        version: 1
      }
    ]

    window.localStorage.setItem("characters-sort-column", "lastUsedAt")
    window.localStorage.setItem("characters-sort-order", "descend")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    expect(await screen.findByRole("columnheader", { name: /Activity/i })).toBeInTheDocument()

    const listQuery = getLatestListCharactersQueryOptions()
    expect(listQuery.queryKey[1]).toMatchObject({
      sort_by: "last_used_at",
      sort_order: "desc"
    })

    await listQuery.queryFn()
    expect(tldwClientMock.listCharactersPage).toHaveBeenCalledWith(
      expect.objectContaining({
        sort_by: "last_used_at",
        sort_order: "desc"
      })
    )
  })

  it("falls back to legacy list loading when /characters/query hits path.character_id route conflict", async () => {
    const routeConflictError = Object.assign(
      new Error(
        "Input should be a valid integer, unable to parse string as an integer (path.character_id) (GET /api/v1/characters/query?page=1&page_size=10)"
      ),
      { status: 422 }
    )
    tldwClientMock.listCharactersPage.mockRejectedValueOnce(routeConflictError)
    tldwClientMock.listCharacters.mockResolvedValueOnce([
      { id: "legacy-1", name: "Legacy Character" }
    ])

    render(<CharactersManager />)

    const listQuery = getListCharactersQueryOptions()
    const response = await listQuery.queryFn()

    expect(tldwClientMock.listCharactersPage).toHaveBeenCalled()
    expect(tldwClientMock.listCharacters).toHaveBeenCalledWith(
      expect.objectContaining({
        limit: 10,
        offset: 0,
        include_image_base64: true
      })
    )
    expect(response.items).toEqual([
      expect.objectContaining({ id: "legacy-1", name: "Legacy Character" })
    ])
    expect(notificationMock.error).not.toHaveBeenCalled()
  })

  it("falls back when route-conflict marker is only present on error details payload", async () => {
    const routeConflictError = Object.assign(new Error("Request failed"), {
      details: {
        detail:
          "Input should be a valid integer, unable to parse string as an integer (path.character_id)"
      }
    })
    tldwClientMock.listCharactersPage.mockRejectedValueOnce(routeConflictError)
    tldwClientMock.listCharacters.mockResolvedValueOnce([
      { id: "legacy-2", name: "Legacy Details Character" }
    ])

    render(<CharactersManager />)

    const listQuery = getListCharactersQueryOptions()
    const response = await listQuery.queryFn()

    expect(tldwClientMock.listCharactersPage).toHaveBeenCalled()
    expect(tldwClientMock.listCharacters).toHaveBeenCalledWith(
      expect.objectContaining({
        limit: 10,
        offset: 0,
        include_image_base64: true
      })
    )
    expect(response.items).toEqual([
      expect.objectContaining({
        id: "legacy-2",
        name: "Legacy Details Character"
      })
    ])
    expect(notificationMock.error).not.toHaveBeenCalled()
  })

  it("degrades to an empty list with notification when query and legacy fallbacks both fail", async () => {
    const routeConflictError = Object.assign(
      new Error(
        "Input should be a valid integer, unable to parse string as an integer (path.character_id)"
      ),
      { status: 422 }
    )
    tldwClientMock.listCharactersPage.mockRejectedValueOnce(routeConflictError)
    tldwClientMock.listCharacters.mockRejectedValueOnce(
      new Error("Legacy character listing unavailable")
    )
    tldwClientMock.listAllCharacters.mockRejectedValueOnce(
      new Error("Legacy all-characters listing unavailable")
    )

    render(<CharactersManager />)

    const listQuery = getListCharactersQueryOptions()
    await expect(listQuery.queryFn()).resolves.toMatchObject({
      items: [],
      total: 0,
      page: 1,
      page_size: 10,
      has_more: false
    })
    expect(notificationMock.error).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Error"
      })
    )
  })

  it("pins character list query throwOnError to false", async () => {
    render(<CharactersManager />)
    const listQuery = getListCharactersQueryOptions()
    expect(listQuery.throwOnError).toBe(false)
  })

  it("falls back to legacy client-side query path when rollout flag is disabled", async () => {
    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === "ff_characters_server_query") {
        return [false, vi.fn(), { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })
    tldwClientMock.listAllCharacters.mockResolvedValueOnce([
      { id: "2", name: "Zeta", tags: ["alpha"] },
      { id: "1", name: "Alpha", tags: ["alpha"] }
    ])

    render(<CharactersManager />)

    const listQuery = getListCharactersQueryOptions()
    expect(listQuery.queryKey[2]).toBe("legacy")

    const response = await listQuery.queryFn()

    expect(tldwClientMock.listAllCharacters).toHaveBeenCalledWith({
      pageSize: 250,
      maxPages: 50
    })
    expect(tldwClientMock.listCharactersPage).not.toHaveBeenCalled()
    expect(response.items.map((item: any) => item.name)).toEqual([
      "Alpha",
      "Zeta"
    ])
    expect(response.total).toBe(2)
  })

  it("renders enriched empty state with guidance and import CTA", async () => {
    render(<CharactersManager />)

    expect(screen.getByRole("heading", { name: "Get started with your first character" })).toBeInTheDocument()
    expect(
      screen.getByText(
        "Create reusable personas you can chat with. Each character keeps its own conversation history."
      )
    ).toBeInTheDocument()
    expect(screen.getByText("Create from scratch")).toBeInTheDocument()
    expect(screen.getByText("Start from a template")).toBeInTheDocument()
    expect(screen.getByText("Import existing")).toBeInTheDocument()
  })

  it("shows template option in empty state and opens create modal with templates", async () => {
    const user = userEvent.setup()
    render(<CharactersManager />)

    const templateButton = screen.getByText("Start from a template").closest("button")!
    expect(templateButton).toBeInTheDocument()

    await user.click(templateButton)

    // Should open the create modal with template chooser expanded
    await waitFor(() => {
      expect(screen.getByText("Choose a template")).toBeInTheDocument()
    })
    expect(window.localStorage.getItem(TEMPLATE_CHOOSER_SEEN_KEY)).toBe("true")
  })

  it("expands template chooser on first create-modal open and keeps it collapsed once seen", async () => {
    const user = userEvent.setup()
    render(<CharactersManager />)

    await user.click(screen.getByRole("button", { name: "New character" }))

    expect(await screen.findByText("Choose a template")).toBeInTheDocument()
    expect(window.localStorage.getItem(TEMPLATE_CHOOSER_SEEN_KEY)).toBe("true")
  }, 60000)

  it("keeps template chooser collapsed when seen flag already exists", async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    expect(screen.queryByText("Choose a template")).not.toBeInTheDocument()
    expect(
      await screen.findByRole("button", { name: "Start from a template..." })
    ).toBeInTheDocument()
  }, 60000)

  it("shows and applies the system prompt example from the create form", async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    await user.click(await screen.findByRole("button", { name: "Show example" }))
    expect(await screen.findByText("Writing Assistant example")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Use this example" }))

    expect(
      await screen.findByDisplayValue(/You are a skilled writing assistant/i)
    ).toBeInTheDocument()
  }, 60000)

  it("promotes prompt preset and groups advanced fields into named sections in create mode", async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    const createFormElement = createSubmitButton.closest("form")
    expect(createFormElement).not.toBeNull()
    const createScope = within(createFormElement as HTMLElement)

    expect(
      createScope.getByText(
        /System prompt: full behavioral instructions sent to the model/i
      )
    ).toBeInTheDocument()
    expect(
      createScope.getByText("Description: brief blurb shown in character lists and cards.")
    ).toBeInTheDocument()
    expect(createScope.getByText("Prompt preset")).toBeInTheDocument()
    expect(createScope.queryByText("Generation temperature")).not.toBeInTheDocument()

    await user.click(createScope.getByRole("button", { name: "Show advanced fields" }))

    expect(createScope.getByRole("button", { name: "Prompt control" })).toBeInTheDocument()
    expect(createScope.getByRole("button", { name: "Generation settings" })).toBeInTheDocument()
    expect(createScope.getByRole("button", { name: "Metadata" })).toBeInTheDocument()
    expect(
      createScope.getByText(
        "Personality: adjectives and traits injected into context to shape voice and behavior."
      )
    ).toBeInTheDocument()
    expect(createScope.getByRole("button", { name: "Add alternate greeting" })).toBeInTheDocument()

    await user.click(createScope.getByRole("button", { name: "Generation settings" }))
    expect(createScope.getByText("Generation temperature")).toBeInTheDocument()

    await user.click(createScope.getByRole("button", { name: "Metadata" }))
    expect(createScope.getByText("Extensions (JSON)")).toBeInTheDocument()
    expect(createScope.getByText("Mood images (coming soon)")).toBeInTheDocument()
  }, 60000)

  it("renders the same advanced section structure in edit mode", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-edit-1",
      name: "Existing Character",
      system_prompt: "Existing system prompt.",
      greeting: "Hi",
      description: "Existing description",
      tags: ["existing"],
      version: 1
    }

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /Edit character/i }))

    const saveButton = await findEditSubmitButton()
    const editFormElement = saveButton.closest("form")
    expect(editFormElement).not.toBeNull()
    const editScope = within(editFormElement as HTMLElement)

    expect(
      editScope.getByText(
        /System prompt: full behavioral instructions sent to the model/i
      )
    ).toBeInTheDocument()
    expect(
      editScope.getByText("Description: brief blurb shown in character lists and cards.")
    ).toBeInTheDocument()
    expect(editScope.getByText("Prompt preset")).toBeInTheDocument()
    await user.click(editScope.getByRole("button", { name: "Show advanced fields" }))

    expect(editScope.getByRole("button", { name: "Prompt control" })).toBeInTheDocument()
    expect(editScope.getByRole("button", { name: "Generation settings" })).toBeInTheDocument()
    expect(editScope.getByRole("button", { name: "Metadata" })).toBeInTheDocument()
    expect(
      editScope.getByText(
        "Personality: adjectives and traits injected into context to shape voice and behavior."
      )
    ).toBeInTheDocument()

    await user.click(editScope.getByRole("button", { name: "Metadata" }))
    expect(editScope.getByText("Mood images (coming soon)")).toBeInTheDocument()
  }, 60000)

  it("preloads world-book attachments in edit mode and syncs attachments on save", async () => {
    const characterRecord = {
      id: 101,
      name: "Worldbook Character",
      system_prompt: "Existing system prompt.",
      greeting: "Hi",
      description: "Existing description",
      tags: ["existing"],
      version: 4
    }

    tldwClientMock.listCharacterWorldBooks.mockResolvedValue([
      { world_book_id: 11, world_book_name: "Lore Atlas" }
    ])
    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any, callbacks?: any) => {
        try {
          const result = await opts?.mutationFn?.(variables)
          opts?.onSuccess?.(result, variables, undefined)
          callbacks?.onSuccess?.(result)
        } catch (error) {
          opts?.onError?.(error, variables, undefined)
          callbacks?.onError?.(error)
        }
      },
      mutateAsync: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "tldw:characterEditWorldBooks") {
        return makeUseQueryResult({
          data: {
            options: [
              { id: 11, name: "Lore Atlas", enabled: true },
              { id: 22, name: "Ship Registry", enabled: true }
            ],
            attachedIds: [11, 22]
          }
        })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    fireEvent.click(await screen.findByRole("button", { name: /Edit character/i }))

    const saveButton = await findEditSubmitButton()
    const editFormElement = saveButton.closest("form")
    expect(editFormElement).not.toBeNull()
    const editScope = within(editFormElement as HTMLElement)

    fireEvent.click(editScope.getByRole("button", { name: "Show advanced fields" }))
    fireEvent.click(editScope.getByRole("button", { name: "Metadata" }))
    expect(editScope.getByText("Lore Atlas")).toBeInTheDocument()
    expect(editScope.getByText("Ship Registry")).toBeInTheDocument()
    fireEvent.click(saveButton)

    await waitFor(() => {
      expect(tldwClientMock.updateCharacter).toHaveBeenCalled()
    })
  }, 60000)

  it("syncs selected world-book attachments after creating a character", async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")
    tldwClientMock.createCharacter.mockResolvedValueOnce({ id: 205 })
    tldwClientMock.listCharacterWorldBooks.mockResolvedValue([])
    let submittedVariables: any = null

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any, callbacks?: any) => {
        submittedVariables = variables
        try {
          const result = await opts?.mutationFn?.(variables)
          opts?.onSuccess?.(result, variables, undefined)
          callbacks?.onSuccess?.(result)
        } catch (error) {
          opts?.onError?.(error, variables, undefined)
          callbacks?.onError?.(error)
        }
      },
      mutateAsync: async (variables: any) => {
        submittedVariables = variables
        const result = await opts?.mutationFn?.(variables)
        opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [], status: "success" })
      }
      if (key === "tldw:characterEditWorldBooks") {
        return makeUseQueryResult({
          data: {
            options: [
              { id: 11, name: "Lore Atlas" },
              { id: 22, name: "Ship Registry" }
            ],
            attachedIds: []
          }
        })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    const createFormElement = createSubmitButton.closest("form")
    expect(createFormElement).not.toBeNull()
    const createScope = within(createFormElement as HTMLElement)

    fireEvent.change(createScope.getByPlaceholderText("e.g. Writing coach"), {
      target: { value: "Worldbook Builder" }
    })
    fireEvent.change(
      createScope.getByPlaceholderText(
        "E.g., You are a patient math teacher who explains concepts step by step and checks understanding with short examples."
      ),
      { target: { value: "You are a grounded assistant." } }
    )

    await user.click(createScope.getByRole("button", { name: "Show advanced fields" }))
    await user.click(createScope.getByRole("button", { name: "Metadata" }))
    await selectCharacterWorldBook(user, createScope, "Lore Atlas")

    await user.click(createScope.getByRole("button", { name: "Create character" }))

    await waitFor(() => {
      expect(tldwClientMock.createCharacter).toHaveBeenCalled()
    })
    expect(submittedVariables?.world_book_ids).toEqual([11])
    await waitFor(() => {
      expect(tldwClientMock.attachWorldBookToCharacter).toHaveBeenCalledWith(205, 11)
    })
    expect(tldwClientMock.detachWorldBookFromCharacter).not.toHaveBeenCalled()
  }, 60000)

  it("shows a permission error when world-book attachment sync is forbidden", async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")
    tldwClientMock.createCharacter.mockResolvedValueOnce({ id: 206 })
    tldwClientMock.listCharacterWorldBooks.mockResolvedValue([])
    const forbiddenError = Object.assign(new Error("Forbidden"), { status: 403 })
    tldwClientMock.attachWorldBookToCharacter.mockRejectedValueOnce(forbiddenError)
    let submittedVariables: any = null

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any, callbacks?: any) => {
        submittedVariables = variables
        try {
          const result = await opts?.mutationFn?.(variables)
          opts?.onSuccess?.(result, variables, undefined)
          callbacks?.onSuccess?.(result)
        } catch (error) {
          opts?.onError?.(error, variables, undefined)
          callbacks?.onError?.(error)
        }
      },
      mutateAsync: async (variables: any) => {
        submittedVariables = variables
        const result = await opts?.mutationFn?.(variables)
        opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [], status: "success" })
      }
      if (key === "tldw:characterEditWorldBooks") {
        return makeUseQueryResult({
          data: {
            options: [{ id: 11, name: "Lore Atlas" }],
            attachedIds: []
          }
        })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    const createFormElement = createSubmitButton.closest("form")
    expect(createFormElement).not.toBeNull()
    const createScope = within(createFormElement as HTMLElement)

    fireEvent.change(createScope.getByPlaceholderText("e.g. Writing coach"), {
      target: { value: "Forbidden Worldbook Builder" }
    })
    fireEvent.change(
      createScope.getByPlaceholderText(
        "E.g., You are a patient math teacher who explains concepts step by step and checks understanding with short examples."
      ),
      { target: { value: "You are a grounded assistant." } }
    )

    await user.click(createScope.getByRole("button", { name: "Show advanced fields" }))
    await user.click(createScope.getByRole("button", { name: "Metadata" }))
    await selectCharacterWorldBook(user, createScope, "Lore Atlas")

    await user.click(createScope.getByRole("button", { name: "Create character" }))

    expect(submittedVariables?.world_book_ids).toEqual([11])
    await waitFor(() => {
      expect(notificationMock.error).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "You do not have permission to modify world-book attachments."
        })
      )
    })
  }, 60000)

  it("persists gallery density preference and applies compact rendering", async () => {
    const user = userEvent.setup()
    const galleryRecord = {
      id: "gallery-1",
      name: "Gallery Dense",
      description: "This description should be hidden in compact mode.",
      tags: ["one", "two", "three"],
      version: 1
    }

    window.localStorage.setItem("characters-view-mode", "gallery")
    window.localStorage.setItem("characters-gallery-density", "compact")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [galleryRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    expect(await screen.findByText("Gallery Dense")).toBeInTheDocument()
    expect(
      screen.queryByText("This description should be hidden in compact mode.")
    ).not.toBeInTheDocument()

    // Open the Display dropdown and click Rich
    await user.click(screen.getByRole("button", { name: "Display options" }))
    await user.click(await screen.findByText("Rich"))

    expect(
      await screen.findByText("This description should be hidden in compact mode.")
    ).toBeInTheDocument()
    expect(window.localStorage.getItem("characters-gallery-density")).toBe("rich")
  }, 30000)

  it("applies persisted page size for table pagination", async () => {
    const records = Array.from({ length: 12 }, (_, index) => ({
      id: `char-${index + 1}`,
      name: `Character ${index + 1}`,
      system_prompt: "Prompt text",
      version: 1
    }))

    window.localStorage.setItem("characters-page-size", "25")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    expect(await screen.findByText("Character 12")).toBeInTheDocument()
    expect(window.localStorage.getItem("characters-page-size")).toBe("25")
  }, 30000)

  it("enforces the explicit 500-character name limit in create mode", async () => {
    const user = userEvent.setup()
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    const createFormElement = createSubmitButton.closest("form")
    expect(createFormElement).not.toBeNull()
    const createScope = within(createFormElement as HTMLElement)

    const nameInput = createScope.getByPlaceholderText("e.g. Writing coach")
    expect(nameInput).toHaveAttribute("maxlength", "500")

    fireEvent.change(nameInput, { target: { value: "A".repeat(501) } })
    fireEvent.change(
      createScope.getByPlaceholderText(
        "E.g., You are a patient math teacher who explains concepts step by step and checks understanding with short examples."
      ),
      { target: { value: "System prompt with enough text." } }
    )

    await user.click(createScope.getByRole("button", { name: "Create character" }))

    expect(
      await screen.findByText("Name must be 500 characters or fewer")
    ).toBeInTheDocument()
    expect(tldwClientMock.createCharacter).not.toHaveBeenCalled()
  }, 30000)

  it("supports has-conversations filter scaffold and clear reset", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "char-a",
        name: "With Chats",
        creator: "alice",
        system_prompt: "Prompt text",
        version: 1
      },
      {
        id: "char-b",
        name: "No Chats",
        creator: "bob",
        system_prompt: "Prompt text",
        version: 1
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: { "char-a": 3, "char-b": 0 } })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    expect(await screen.findByText("With Chats")).toBeInTheDocument()
    expect(screen.getByText("No Chats")).toBeInTheDocument()
    expect(
      screen.getByLabelText("Filter characters by creator")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("checkbox", { name: "Has conversations" }))

    await waitFor(() => {
      expect(screen.queryByText("No Chats")).not.toBeInTheDocument()
    })

    await user.click(screen.getByRole("button", { name: "Clear filters" }))

    expect(await screen.findByText("No Chats")).toBeInTheDocument()
  }, 30000)

  it("toggles advanced filters panel while keeping primary controls visible", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "char-a",
        name: "With Chats",
        creator: "alice",
        system_prompt: "Prompt text",
        version: 1
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    expect(screen.getByPlaceholderText("Search characters")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Filters/i })
    ).toBeInTheDocument()

    await openAdvancedFilters(user)

    expect(
      await screen.findByLabelText("Filter characters by creator")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /Filters/i }))

    await waitFor(() => {
      expect(
        screen.queryByLabelText("Filter characters by creator")
      ).not.toBeInTheDocument()
    })
    expect(screen.getByPlaceholderText("Search characters")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /Filters/i })
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: /Filters/i }))

    expect(
      await screen.findByLabelText("Filter characters by creator")
    ).toBeInTheDocument()
  }, 30000)

  it("serializes folder filter into reserved folder tag query params and clears it", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "folder-1",
        name: "Folder Candidate",
        system_prompt: "Prompt text",
        version: 1
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "tldw:characterFolders") {
        return makeUseQueryResult({
          data: [{ id: 12, name: "Research" }],
          status: "success"
        })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    fireEvent.mouseDown(screen.getByLabelText("Filter characters by folder"))
    await user.click(await screen.findByText("Research"))

    await waitFor(() => {
      const latestListQuery = getLatestListCharactersQueryOptions()
      expect(latestListQuery.queryKey[1]).toMatchObject({
        tags: ["__tldw_folder_id:12"],
        match_all_tags: true
      })
    })

    await user.click(screen.getByRole("button", { name: "Clear filters" }))

    await waitFor(() => {
      const latestListQuery = getLatestListCharactersQueryOptions()
      expect(latestListQuery.queryKey[1].tags).toBeUndefined()
      expect(latestListQuery.queryKey[1].match_all_tags).toBeUndefined()
    })
  }, 30000)

  it("hides reserved folder tokens from tag table and tag-manager surfaces", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "folder-token-1",
        name: "Folder Tagged",
        tags: ["visible", "__tldw_folder_id:12"],
        system_prompt: "Prompt text",
        version: 2
      }
    ]

    tldwClientMock.listCharactersPage.mockResolvedValue({
      items: records,
      total: records.length,
      page: 1,
      page_size: 100,
      has_more: false
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    expect(await screen.findByText("visible")).toBeInTheDocument()
    expect(screen.queryByText("__tldw_folder_id:12")).not.toBeInTheDocument()
    await openAdvancedFilters(user)

    await user.click(screen.getByRole("button", { name: "Manage tags" }))
    const tagDialog = await screen.findByRole("dialog")
    expect(within(tagDialog).getByText("visible")).toBeInTheDocument()
    expect(
      within(tagDialog).queryByText("__tldw_folder_id:12")
    ).not.toBeInTheDocument()
  }, 30000)

  it("replaces existing folder token when reassigning folder in edit mode", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "folder-edit-1",
      name: "Folder Reassign",
      system_prompt: "Prompt text",
      description: "Description",
      tags: ["alpha", "__tldw_folder_id:2"],
      version: 5
    }

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any, callbacks?: any) => {
        try {
          const result = await opts?.mutationFn?.(variables)
          opts?.onSuccess?.(result, variables, undefined)
          callbacks?.onSuccess?.(result)
        } catch (error) {
          opts?.onError?.(error, variables, undefined)
          callbacks?.onError?.(error)
        }
      },
      mutateAsync: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "tldw:characterFolders") {
        return makeUseQueryResult({
          data: [
            { id: 2, name: "Old Folder" },
            { id: 9, name: "New Folder" }
          ],
          status: "success"
        })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /Edit character/i }))

    const saveButton = await findEditSubmitButton()
    const editFormElement = saveButton.closest("form")
    expect(editFormElement).not.toBeNull()
    const editScope = within(editFormElement as HTMLElement)

    await user.click(editScope.getByRole("button", { name: "Show advanced fields" }))
    await user.click(editScope.getByRole("button", { name: "Metadata" }))

    const folderField = editScope
      .getByText("Folder")
      .closest(".ant-form-item")
    expect(folderField).not.toBeNull()
    const folderCombobox = within(folderField as HTMLElement).getByRole("combobox")
    const folderSelectContent = (folderField as HTMLElement).querySelector(
      ".ant-select-content"
    )
    expect(folderSelectContent).not.toBeNull()
    fireEvent.mouseDown(folderSelectContent as HTMLElement)
    await waitFor(() => {
      expect(folderCombobox).toHaveAttribute("aria-expanded", "true")
    })
    await user.click(
      await screen.findByText("New Folder", {
        selector: ".ant-select-item-option-content"
      })
    )
    await waitFor(() => expect(saveButton).toBeEnabled())
    await user.click(saveButton)

    await waitFor(() => {
      const latestCall = tldwClientMock.updateCharacter.mock.calls.at(-1)
      expect(latestCall?.[0]).toBe("folder-edit-1")
      expect(latestCall?.[1]).toEqual(
        expect.objectContaining({
          tags: ["alpha", "__tldw_folder_id:9"]
        })
      )
    })
  }, 60_000)

  it("filters to favorited characters when favorites-only is enabled", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "fav-1",
        name: "Favorited Character",
        system_prompt: "Prompt text",
        extensions: { tldw: { favorite: true } },
        version: 3
      },
      {
        id: "fav-2",
        name: "Regular Character",
        system_prompt: "Prompt text",
        version: 2
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    expect(await screen.findByText("Favorited Character")).toBeInTheDocument()
    expect(screen.getByText("Regular Character")).toBeInTheDocument()

    await user.click(screen.getByRole("checkbox", { name: "Favorites only" }))

    await waitFor(() => {
      expect(screen.queryByText("Regular Character")).not.toBeInTheDocument()
    })
    expect(screen.getByText("Favorited Character")).toBeInTheDocument()
  }, 30000)

  it("toggles favorite state from table row actions", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "toggle-fav-1",
        name: "Toggle Favorite",
        system_prompt: "Prompt text",
        extensions: {},
        version: 9
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      await screen.findByRole("button", {
        name: "Add Toggle Favorite to favorites"
      })
    )

    await waitFor(() => {
      expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
        "toggle-fav-1",
        { extensions: { tldw: { favorite: true } } },
        9
      )
    })
  }, 30000)

  it("switches to recently-deleted scope and queries deleted-only records", async () => {
    const user = userEvent.setup()

    render(<CharactersManager />)

    await user.click(screen.getByText("Trash"))

    await waitFor(() => {
      const deletedQuery = getDeletedScopeListCharactersQueryOptions()
      expect(deletedQuery.queryKey[1]).toMatchObject({
        include_deleted: true,
        deleted_only: true
      })
    })

    const deletedQuery = getDeletedScopeListCharactersQueryOptions()
    await deletedQuery.queryFn()

    expect(tldwClientMock.listCharactersPage).toHaveBeenCalledWith(
      expect.objectContaining({
        include_deleted: true,
        deleted_only: true
      })
    )
  }, 30000)

  it("restores characters from recently-deleted scope actions", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "deleted-1",
        name: "Deleted Character",
        system_prompt: "Prompt text",
        version: 12
      }
    ]

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        await opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      mutateAsync: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        await opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      screen.getByText("Trash", {
        selector: ".ant-segmented-item-label"
      })
    )
    await waitFor(() => {
      const deletedQuery = getDeletedScopeListCharactersQueryOptions()
      expect(deletedQuery.queryKey[1]).toMatchObject({
        include_deleted: true,
        deleted_only: true
      })
    })
    await user.click(await screen.findByRole("button", { name: /Restore character/i }))

    await waitFor(() => {
      expect(tldwClientMock.restoreCharacter).toHaveBeenCalledWith("deleted-1", 12)
    })
  }, 45000)

  it("shows actionable restore-window errors and emits recovery telemetry when restore fails", async () => {
    const user = userEvent.setup()
    const dispatchSpy = vi.spyOn(window, "dispatchEvent")
    const records = [
      {
        id: "deleted-2",
        name: "Deleted Character",
        system_prompt: "Prompt text",
        version: 8
      }
    ]

    tldwClientMock.restoreCharacter.mockRejectedValueOnce(
      new Error(
        "Restore window expired for character ID 7. This character was deleted at 2026-02-10T10:00:00Z and could only be restored until 2026-02-11T10:00:00Z UTC."
      )
    )

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any) => {
        try {
          const result = await opts?.mutationFn?.(variables)
          await opts?.onSuccess?.(result, variables, undefined)
          return result
        } catch (error) {
          await opts?.onError?.(error, variables, undefined)
          return undefined
        }
      },
      mutateAsync: async (variables: any) => {
        try {
          const result = await opts?.mutationFn?.(variables)
          await opts?.onSuccess?.(result, variables, undefined)
          return result
        } catch (error) {
          await opts?.onError?.(error, variables, undefined)
          return undefined
        }
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      screen.getByText("Trash", {
        selector: ".ant-segmented-item-label"
      })
    )
    await waitFor(() => {
      const deletedQuery = getDeletedScopeListCharactersQueryOptions()
      expect(deletedQuery.queryKey[1]).toMatchObject({
        include_deleted: true,
        deleted_only: true
      })
    })

    await user.click(await screen.findByRole("button", { name: /Restore character/i }))

    await waitFor(() => {
      expect(notificationMock.error).toHaveBeenCalled()
    })
    const payload = notificationMock.error.mock.calls.at(-1)?.[0]
    expect(payload?.message).toBe("Failed to restore character")
    expect(payload?.description).toContain("Restore window expired")
    expect(payload?.description).toContain("could only be restored until")
    expect(payload?.description).toContain("check server logs")

    const recoveryEvent = dispatchSpy.mock.calls
      .map((args) => args[0])
      .find(
        (event): event is CustomEvent<Record<string, unknown>> =>
          event instanceof CustomEvent &&
          event.type === "tldw:characters-recovery" &&
          event.detail?.action === "restore_failed"
      )

    expect(recoveryEvent).toBeDefined()
    expect(recoveryEvent?.detail?.action).toBe("restore_failed")

    dispatchSpy.mockRestore()
  }, 45000)

  it("uses soft-delete copy and undo semantics for bulk delete", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "bulk-1",
        name: "Bulk One",
        system_prompt: "Prompt text",
        version: 10
      },
      {
        id: "bulk-2",
        name: "Bulk Two",
        system_prompt: "Prompt text",
        version: 20
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      await screen.findByRole("checkbox", { name: "Select all on page" })
    )

    const clearSelectionButton = await screen.findByRole("button", {
      name: "Clear selection"
    })
    const toolbar = clearSelectionButton.closest("div")?.parentElement
    expect(toolbar).not.toBeNull()
    await user.click(
      within(toolbar as HTMLElement).getByRole("button", { name: "Delete" })
    )

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalledTimes(1)
    })
    expect(confirmDangerMock.mock.calls[0]?.[0]).toMatchObject({
      content: expect.stringContaining(
        "This will soft-delete 2 characters. You can undo for 10 seconds."
      )
    })

    await waitFor(() => {
      expect(tldwClientMock.deleteCharacter).toHaveBeenCalledTimes(2)
    })
    await waitFor(() => {
      expect(notificationMock.info).toHaveBeenCalledTimes(1)
    })

    const infoPayload = notificationMock.info.mock.calls[0]?.[0]
    expect(infoPayload?.duration).toBe(10)

    const undoHandler = infoPayload?.description?.props?.onClick
    expect(typeof undoHandler).toBe("function")
    await undoHandler()

    await waitFor(() => {
      expect(tldwClientMock.restoreCharacter).toHaveBeenCalledWith("bulk-1", 11)
    })
    await waitFor(() => {
      expect(tldwClientMock.restoreCharacter).toHaveBeenCalledWith("bulk-2", 21)
    })
  }, 30000)

  it("enables compare only when exactly two characters are selected", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "cmp-1",
        name: "Compare One",
        description: "Description one",
        system_prompt: "Prompt one",
        tags: ["alpha"],
        version: 1
      },
      {
        id: "cmp-2",
        name: "Compare Two",
        description: "Description two",
        system_prompt: "Prompt two",
        tags: ["beta"],
        version: 2
      },
      {
        id: "cmp-3",
        name: "Compare Three",
        description: "Description three",
        system_prompt: "Prompt three",
        tags: ["gamma"],
        version: 3
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("checkbox", { name: "Select Compare One" }))
    expect(await screen.findByRole("button", { name: "Compare" })).toBeDisabled()

    await user.click(screen.getByRole("checkbox", { name: "Select Compare Two" }))
    expect(screen.getByRole("button", { name: "Compare" })).toBeEnabled()

    await user.click(screen.getByRole("checkbox", { name: "Select Compare Three" }))
    expect(screen.getByRole("button", { name: "Compare" })).toBeDisabled()
  }, 30000)

  it("opens a compare modal for two selected characters and shows field differences", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "cmp-a",
        name: "Compare Alpha",
        description: "Alpha description",
        system_prompt: "Prompt alpha",
        tags: ["alpha"],
        version: 11
      },
      {
        id: "cmp-b",
        name: "Compare Beta",
        description: "Beta description",
        system_prompt: "Prompt beta",
        tags: ["beta"],
        version: 12
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("checkbox", { name: "Select all on page" }))
    const compareButton = await screen.findByRole("button", { name: "Compare" })
    await waitFor(() => {
      expect(compareButton).toBeEnabled()
    })
    await user.click(compareButton)

    const compareTitle = await screen.findByText("Compare characters")
    const dialog = compareTitle.closest(".ant-modal")
    expect(dialog).not.toBeNull()
    expect(within(dialog as HTMLElement).getByText("Prompt alpha")).toBeInTheDocument()
    expect(within(dialog as HTMLElement).getByText("Prompt beta")).toBeInTheDocument()
    expect(
      within(dialog as HTMLElement).getByText(/tracked fields differ/i)
    ).toBeInTheDocument()

    const closeButtons = within(dialog as HTMLElement).getAllByRole("button", {
      name: "Close"
    })
    const footerCloseButton = closeButtons.find(
      (button) => !button.classList.contains("ant-modal-close")
    )
    expect(footerCloseButton).toBeDefined()
    await user.click(footerCloseButton as HTMLElement)
  }, 30000)

  it("copies and exports comparison summaries from the compare modal", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "cmp-copy-1",
        name: "Copy Left",
        description: "Left description",
        system_prompt: "Prompt left",
        tags: ["left"],
        version: 31
      },
      {
        id: "cmp-copy-2",
        name: "Copy Right",
        description: "Right description",
        system_prompt: "Prompt right",
        tags: ["right"],
        version: 32
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    const writeTextMock = vi.fn(async (_text: string) => undefined)
    const originalClipboard = (navigator as Navigator & { clipboard?: Clipboard }).clipboard
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: writeTextMock }
    })

    const originalCreateObjectURL = URL.createObjectURL
    const originalRevokeObjectURL = URL.revokeObjectURL
    const createObjectURLMock = vi.fn(() => "blob:compare-summary")
    const revokeObjectURLMock = vi.fn()
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: createObjectURLMock
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: revokeObjectURLMock
    })
    const anchorClickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined)

    try {
      render(<CharactersManager />)

      await user.click(await screen.findByRole("checkbox", { name: "Select all on page" }))
      const compareButton = await screen.findByRole("button", { name: "Compare" })
      await waitFor(() => {
        expect(compareButton).toBeEnabled()
      })
      await user.click(compareButton)

      const compareTitle = await screen.findByText("Compare characters")
      const dialog = compareTitle.closest(".ant-modal")
      expect(dialog).not.toBeNull()

      await user.click(
        within(dialog as HTMLElement).getByRole("button", { name: "Copy summary" })
      )
      await waitFor(() => {
        expect(writeTextMock).toHaveBeenCalledTimes(1)
      })
      expect(writeTextMock.mock.calls[0]?.[0]).toContain("Character comparison summary")
      expect(writeTextMock.mock.calls[0]?.[0]).toContain("Copy Left")
      expect(writeTextMock.mock.calls[0]?.[0]).toContain("Copy Right")

      await user.click(
        within(dialog as HTMLElement).getByRole("button", { name: "Export summary" })
      )
      expect(createObjectURLMock).toHaveBeenCalledTimes(1)
      expect(anchorClickSpy).toHaveBeenCalledTimes(1)
      expect(revokeObjectURLMock).toHaveBeenCalledWith("blob:compare-summary")
    } finally {
      anchorClickSpy.mockRestore()
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: originalClipboard
      })
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        writable: true,
        value: originalCreateObjectURL
      })
      Object.defineProperty(URL, "revokeObjectURL", {
        configurable: true,
        writable: true,
        value: originalRevokeObjectURL
      })
    }
  }, 30000)

  it("supports keyboard Enter to trigger and commit inline name editing", async () => {
    const records = [
      {
        id: "inline-1",
        name: "Inline Name",
        description: "Original description",
        system_prompt: "Prompt text",
        version: 7
      }
    ]

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        await opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      mutateAsync: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        await opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    const nameInlineButton = await screen.findByRole("button", {
      name: /Edit name inline/i
    })
    nameInlineButton.focus()
    fireEvent.keyDown(nameInlineButton, { key: "Enter" })

    const inlineInput = await screen.findByDisplayValue("Inline Name")
    fireEvent.change(inlineInput, { target: { value: "Inline Name Updated" } })
    fireEvent.keyDown(inlineInput, { key: "Enter" })

    await waitFor(() => {
      expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
        "inline-1",
        { name: "Inline Name Updated" },
        7
      )
    })
  }, 30000)

  it("supports F2 and Escape for inline description edit with focus return", async () => {
    const records = [
      {
        id: "inline-2",
        name: "Inline Description Character",
        description: "Inline description text",
        system_prompt: "Prompt text",
        version: 5
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    const descriptionInlineButton = await screen.findByRole("button", {
      name: /Edit description inline/i
    })
    descriptionInlineButton.focus()
    fireEvent.keyDown(descriptionInlineButton, { key: "F2" })

    const inlineInput = await screen.findByDisplayValue("Inline description text")
    fireEvent.keyDown(inlineInput, { key: "Escape" })

    await waitFor(() => {
      expect(
        screen.queryByDisplayValue("Inline description text")
      ).not.toBeInTheDocument()
    })
    expect(tldwClientMock.updateCharacter).not.toHaveBeenCalled()
  }, 30000)

  it("supports Space key variants for inline name and description editing", async () => {
    const records = [
      {
        id: "inline-space",
        name: "Inline Space Name",
        description: "Inline Space Description",
        system_prompt: "Prompt text",
        version: 3
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    const nameInlineButton = await screen.findByRole("button", {
      name: /Edit name inline/i
    })
    const nameSpaceEvent = new KeyboardEvent("keydown", {
      key: " ",
      bubbles: true,
      cancelable: true
    })
    nameInlineButton.dispatchEvent(nameSpaceEvent)
    expect(nameSpaceEvent.defaultPrevented).toBe(true)

    const nameInlineInput = await screen.findByDisplayValue("Inline Space Name")
    fireEvent.keyDown(nameInlineInput, { key: "Escape" })

    await waitFor(() => {
      expect(screen.queryByDisplayValue("Inline Space Name")).not.toBeInTheDocument()
    })

    const descriptionInlineButton = await screen.findByRole("button", {
      name: /Edit description inline/i
    })
    const descriptionSpacebarEvent = new KeyboardEvent("keydown", {
      key: "Spacebar",
      bubbles: true,
      cancelable: true
    })
    descriptionInlineButton.dispatchEvent(descriptionSpacebarEvent)
    expect(descriptionSpacebarEvent.defaultPrevented).toBe(true)

    await screen.findByDisplayValue("Inline Space Description")
  }, 30000)

  it("renames tags across affected characters from the manage tags modal", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "char-1",
        name: "Legacy One",
        tags: ["legacy", "shared"],
        system_prompt: "Prompt text",
        version: 1
      },
      {
        id: "char-2",
        name: "Legacy Two",
        tags: ["legacy"],
        system_prompt: "Prompt text",
        version: 3
      },
      {
        id: "char-3",
        name: "Shared Only",
        tags: ["shared"],
        system_prompt: "Prompt text",
        version: 2
      }
    ]

    tldwClientMock.listCharactersPage.mockResolvedValue({
      items: records,
      total: records.length,
      page: 1,
      page_size: 100,
      has_more: false
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    await user.click(screen.getByRole("button", { name: "Manage tags" }))
    const renameDialog = await screen.findByRole("dialog")
    expect(within(renameDialog).getByText("Manage tags")).toBeInTheDocument()

    await selectTagManagerSourceTag(user, renameDialog, /^legacy \(2\)$/i)
    fireEvent.change(within(renameDialog).getByPlaceholderText("Destination tag"), {
      target: { value: "modern" }
    })
    await user.click(within(renameDialog).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(tldwClientMock.updateCharacter).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
      "char-1",
      { tags: ["modern", "shared"] },
      1
    )
    expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
      "char-2",
      { tags: ["modern"] },
      3
    )
  }, 30000)

  it("merges source tags into destination tags from the manage tags modal", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "char-a",
        name: "Alpha Beta",
        tags: ["alpha", "beta", "misc"],
        system_prompt: "Prompt text",
        version: 4
      },
      {
        id: "char-b",
        name: "Alpha Only",
        tags: ["alpha"],
        system_prompt: "Prompt text",
        version: 6
      },
      {
        id: "char-c",
        name: "Beta Only",
        tags: ["beta"],
        system_prompt: "Prompt text",
        version: 5
      }
    ]

    tldwClientMock.listCharactersPage.mockResolvedValue({
      items: records,
      total: records.length,
      page: 1,
      page_size: 100,
      has_more: false
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    await user.click(screen.getByRole("button", { name: "Manage tags" }))
    const mergeDialog = await screen.findByRole("dialog")
    expect(within(mergeDialog).getByText("Manage tags")).toBeInTheDocument()

    await user.click(within(mergeDialog).getByText("Merge"))
    await selectTagManagerSourceTag(user, mergeDialog, /^alpha \(2\)$/i)
    fireEvent.change(within(mergeDialog).getByPlaceholderText("Destination tag"), {
      target: { value: "beta" }
    })
    await user.click(within(mergeDialog).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(tldwClientMock.updateCharacter).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
      "char-a",
      { tags: ["beta", "misc"] },
      4
    )
    expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
      "char-b",
      { tags: ["beta"] },
      6
    )
  }, 30000)

  it("deletes tags across affected characters from the manage tags modal", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "char-x",
        name: "Obsolete One",
        tags: ["obsolete", "active"],
        system_prompt: "Prompt text",
        version: 7
      },
      {
        id: "char-y",
        name: "Obsolete Two",
        tags: ["obsolete"],
        system_prompt: "Prompt text",
        version: 8
      },
      {
        id: "char-z",
        name: "Active",
        tags: ["active"],
        system_prompt: "Prompt text",
        version: 9
      }
    ]

    tldwClientMock.listCharactersPage.mockResolvedValue({
      items: records,
      total: records.length,
      page: 1,
      page_size: 100,
      has_more: false
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await openAdvancedFilters(user)

    await user.click(screen.getByRole("button", { name: "Manage tags" }))
    const deleteDialog = await screen.findByRole("dialog")
    expect(within(deleteDialog).getByText("Manage tags")).toBeInTheDocument()

    await user.click(within(deleteDialog).getByText("Delete"))
    await selectTagManagerSourceTag(user, deleteDialog, /^obsolete \(2\)$/i)
    expect(
      within(deleteDialog).queryByPlaceholderText("Destination tag")
    ).not.toBeInTheDocument()
    await user.click(within(deleteDialog).getByRole("button", { name: "Apply" }))

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(tldwClientMock.updateCharacter).toHaveBeenCalledTimes(2)
    })
    expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
      "char-x",
      { tags: ["active"] },
      7
    )
    expect(tldwClientMock.updateCharacter).toHaveBeenCalledWith(
      "char-y",
      { tags: [] },
      8
    )
  }, 30000)

  it("opens version history modal and renders field-level diffs", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "101",
        name: "Versioned Character",
        description: "Latest",
        system_prompt: "Prompt text",
        version: 4
      }
    ]

    const versionItems = [
      {
        change_id: 40,
        version: 4,
        operation: "update",
        timestamp: "2026-02-18T10:10:00Z",
        payload: {
          name: "Versioned Character",
          description: "Latest",
          tags: ["alpha", "beta"]
        }
      },
      {
        change_id: 39,
        version: 3,
        operation: "update",
        timestamp: "2026-02-18T09:00:00Z",
        payload: {
          name: "Versioned Character",
          description: "Earlier",
          tags: ["alpha"]
        }
      }
    ]

    tldwClientMock.listCharacterVersions.mockResolvedValue({
      items: versionItems,
      total: 2
    })
    tldwClientMock.diffCharacterVersions.mockResolvedValue({
      character_id: 101,
      from_entry: versionItems[1],
      to_entry: versionItems[0],
      changed_fields: [
        {
          field: "description",
          old_value: "Earlier",
          new_value: "Latest"
        },
        {
          field: "tags",
          old_value: ["alpha"],
          new_value: ["alpha", "beta"]
        }
      ],
      changed_count: 2
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      if (key === "tldw:characterVersions") {
        void opts?.queryFn?.()
        return makeUseQueryResult({
          data: {
            items: versionItems,
            total: 2
          },
          status: "success"
        })
      }
      if (key === "tldw:characterVersionDiff") {
        void opts?.queryFn?.()
        return makeUseQueryResult({
          data: {
            character_id: 101,
            from_entry: versionItems[1],
            to_entry: versionItems[0],
            changed_fields: [
              {
                field: "description",
                old_value: "Earlier",
                new_value: "Latest"
              },
              {
                field: "tags",
                old_value: ["alpha"],
                new_value: ["alpha", "beta"]
              }
            ],
            changed_count: 2
          },
          status: "success"
        })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      await screen.findByLabelText("More actions for Versioned Character")
    )
    await user.click(await screen.findByText("Version history"))

    await waitFor(() => {
      expect(tldwClientMock.listCharacterVersions).toHaveBeenCalledWith(101, {
        limit: 100
      })
    })
    await waitFor(() => {
      expect(tldwClientMock.diffCharacterVersions).toHaveBeenCalledWith(
        101,
        3,
        4
      )
    })

    expect(
      await screen.findByText("Version history: Versioned Character")
    ).toBeInTheDocument()
    const versionDialog = await screen.findByRole("dialog")
    expect(
      within(versionDialog).getByText("Differences: v3 -> v4")
    ).toBeInTheDocument()
    expect(within(versionDialog).getAllByText("Description").length).toBeGreaterThan(0)
    expect(within(versionDialog).getAllByText("Tags").length).toBeGreaterThan(0)
  }, 30000)

  it("reverts character from selected version in version history modal", async () => {
    const user = userEvent.setup()
    const records = [
      {
        id: "101",
        name: "Versioned Character",
        description: "Latest",
        system_prompt: "Prompt text",
        version: 4
      }
    ]

    const versionItems = [
      {
        change_id: 40,
        version: 4,
        operation: "update",
        timestamp: "2026-02-18T10:10:00Z",
        payload: {
          name: "Versioned Character",
          description: "Latest"
        }
      },
      {
        change_id: 39,
        version: 3,
        operation: "update",
        timestamp: "2026-02-18T09:00:00Z",
        payload: {
          name: "Versioned Character",
          description: "Earlier"
        }
      }
    ]

    tldwClientMock.listCharacterVersions.mockResolvedValue({
      items: versionItems,
      total: 2
    })
    tldwClientMock.diffCharacterVersions.mockResolvedValue({
      character_id: 101,
      from_entry: versionItems[1],
      to_entry: versionItems[0],
      changed_fields: [
        {
          field: "description",
          old_value: "Earlier",
          new_value: "Latest"
        }
      ],
      changed_count: 1
    })
    tldwClientMock.revertCharacter.mockResolvedValueOnce({
      id: 101,
      name: "Versioned Character",
      version: 5
    })
    useMutationMock.mockImplementation((opts: any) => ({
      mutate: (variables: any) => opts?.mutationFn?.(variables),
      mutateAsync: (variables: any) => opts?.mutationFn?.(variables),
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      if (key === "tldw:characterVersions") {
        void opts?.queryFn?.()
        return makeUseQueryResult({
          data: {
            items: versionItems,
            total: 2
          },
          status: "success"
        })
      }
      if (key === "tldw:characterVersionDiff") {
        void opts?.queryFn?.()
        return makeUseQueryResult({
          data: {
            character_id: 101,
            from_entry: versionItems[1],
            to_entry: versionItems[0],
            changed_fields: [
              {
                field: "description",
                old_value: "Earlier",
                new_value: "Latest"
              }
            ],
            changed_count: 1
          },
          status: "success"
        })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      await screen.findByLabelText("More actions for Versioned Character")
    )
    await user.click(await screen.findByText("Version history"))
    await screen.findByText("Version history: Versioned Character")

    await user.click(
      screen.getByRole("button", { name: "Revert to selected version" })
    )

    await waitFor(() => {
      expect(confirmDangerMock).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(tldwClientMock.revertCharacter).toHaveBeenCalledWith(101, 3)
    })
    await waitFor(() => {
      expect(tldwClientMock.listCharacterVersions).toHaveBeenCalledWith(101, {
        limit: 100
      })
    })
  }, 30000)

  it("saves alternate greetings in reordered list order", async () => {
    window.localStorage.setItem(TEMPLATE_CHOOSER_SEEN_KEY, "true")

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: (variables: any) => opts?.mutationFn?.(variables),
      mutateAsync: (variables: any) => opts?.mutationFn?.(variables),
      isPending: false
    }))

    render(<CharactersManager />)
    fireEvent.click(screen.getByRole("button", { name: "New character" }))
    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    const createFormElement = createSubmitButton.closest("form")
    expect(createFormElement).not.toBeNull()
    const createScope = within(createFormElement as HTMLElement)

    fireEvent.change(createScope.getByPlaceholderText("e.g. Writing coach"), {
      target: { value: "Tester" }
    })
    fireEvent.change(
      createScope.getByPlaceholderText(
        "E.g., You are a patient math teacher who explains concepts step by step and checks understanding with short examples."
      ),
      {
        target: { value: "Helpful guide." }
      }
    )

    fireEvent.click(createScope.getByRole("button", { name: "Show advanced fields" }))
    fireEvent.click(createScope.getByRole("button", { name: "Add alternate greeting" }))
    fireEvent.click(createScope.getByRole("button", { name: "Add alternate greeting" }))

    const greetingInputs = createScope.getAllByPlaceholderText(
      "Enter an alternate greeting message"
    )
    fireEvent.change(greetingInputs[0], { target: { value: "First" } })
    fireEvent.change(greetingInputs[1], { target: { value: "Second" } })

    const moveUpButtons = createScope.getAllByLabelText("Move greeting up")
    fireEvent.click(moveUpButtons[1])

    fireEvent.click(createScope.getByRole("button", { name: "Create character" }))

    await waitFor(() => {
      expect(tldwClientMock.createCharacter).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Tester",
          alternate_greetings: ["Second", "First"]
        })
      )
    })
  }, 60000)

  it("supports first-run template -> create -> chat handoff", async () => {
    const user = userEvent.setup()
    let listCharactersData: any[] = []

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: listCharactersData, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    useMutationMock.mockImplementation((opts: any) => {
      return {
        mutate: async (variables: any) => {
          const result = await opts?.mutationFn?.(variables)
          if (
            variables &&
            typeof variables.name === "string" &&
            typeof variables.system_prompt === "string"
          ) {
            listCharactersData = [
              {
                id: result?.id ?? "char-1",
                name: variables?.name ?? "Writer Coach",
                description: variables?.description ?? "Helps with structure and tone.",
                system_prompt: variables?.system_prompt ?? "You are a writing coach.",
                greeting: variables?.greeting ?? "Ready to improve your draft?",
                tags: variables?.tags ?? ["writing", "coach"],
                version: 1
              }
            ]
          }
          await opts?.onSuccess?.(result, variables, undefined)
          return result
        },
        mutateAsync: async (variables: any) => {
          const result = await opts?.mutationFn?.(variables)
          if (
            variables &&
            typeof variables.name === "string" &&
            typeof variables.system_prompt === "string"
          ) {
            listCharactersData = [
              {
                id: result?.id ?? "char-1",
                name: variables?.name ?? "Writer Coach",
                description: variables?.description ?? "Helps with structure and tone.",
                system_prompt: variables?.system_prompt ?? "You are a writing coach.",
                greeting: variables?.greeting ?? "Ready to improve your draft?",
                tags: variables?.tags ?? ["writing", "coach"],
                version: 1
              }
            ]
          }
          await opts?.onSuccess?.(result, variables, undefined)
          return result
        },
        isPending: false
      }
    })

    render(<CharactersManager />)

    // Click "Start from a template" onboarding card to open drawer with templates
    const templateCard = screen.getByText("Start from a template").closest("button")!
    await user.click(templateCard)

    // Click the Writer Coach template in the drawer
    const writerCoachButton = await screen.findByRole("button", { name: /Writer Coach/i })
    await user.click(writerCoachButton)
    expect(await screen.findByDisplayValue("Writer Coach")).toBeInTheDocument()

    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    fireEvent.click(createSubmitButton)

    // After creation the drawer closes and the table re-renders with the new character.
    // Wait until "Writer Coach" appears inside a table row (not in the drawer).
    const tableRow = await waitFor(() => {
      const candidates = screen.getAllByText("Writer Coach")
      const row = candidates.map((el) => el.closest("tr")).find(Boolean)
      if (!row) throw new Error("Writer Coach not found inside a <tr> yet")
      return row as HTMLElement
    })
    const chatButton = within(tableRow).getByRole("button", {
      name: /Chat/i
    })
    await user.click(chatButton)

    expect(setSelectedCharacterMock).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "char-1",
        name: "Writer Coach",
        system_prompt: expect.any(String),
        greeting: expect.any(String)
      })
    )
    expect(navigateMock).toHaveBeenCalledWith("/")
    expect(focusComposerMock).toHaveBeenCalled()
  }, 30000)

  it("submits edit flow through the shared form component", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-edit-flow",
      name: "Edit Flow Character",
      system_prompt: "Stay helpful.",
      greeting: "Hello",
      description: "Original description",
      tags: ["editing"],
      version: 3
    }

    useMutationMock.mockImplementation((opts: any) => ({
      mutate: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        await opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      mutateAsync: async (variables: any) => {
        const result = await opts?.mutationFn?.(variables)
        await opts?.onSuccess?.(result, variables, undefined)
        return result
      },
      isPending: false
    }))

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /Edit character/i }))
    const saveButton = await findEditSubmitButton()
    const editFormElement = saveButton.closest("form")
    expect(editFormElement).not.toBeNull()
    const editScope = within(editFormElement as HTMLElement)

    fireEvent.change(
      editScope.getByPlaceholderText("Short description"),
      { target: { value: "Updated description from edit flow test." } }
    )

    fireEvent.submit(editFormElement as HTMLFormElement)

    await waitFor(() => {
      const latestCall = tldwClientMock.updateCharacter.mock.calls.at(-1)
      expect(latestCall?.[0]).toBe("char-edit-flow")
      expect(latestCall?.[1]).toEqual(
        expect.objectContaining({
          description: "Updated description from edit flow test."
        })
      )
    })
  }, 30000)

  it("exports a character from row actions as JSON", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-export-flow",
      name: "Export Flow Character",
      system_prompt: "You are exportable.",
      greeting: "Hi",
      description: "Export me",
      tags: ["export"],
      version: 1
    }

    tldwClientMock.exportCharacter.mockResolvedValueOnce({
      id: "char-export-flow",
      name: "Export Flow Character"
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)
    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByText("Export as JSON"))

    await waitFor(() => {
      expect(tldwClientMock.exportCharacter).toHaveBeenCalledWith(
        "char-export-flow",
        { format: "v3" }
      )
    })
    expect(exportCharacterToJSONMock).toHaveBeenCalledTimes(1)
  }, 30000)

  it("sets a character as default from row actions", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-default",
      name: "Default Candidate",
      system_prompt: "Default system prompt",
      greeting: "Default greeting",
      version: 1
    }
    const setDefaultCharacterMock = vi.fn(async () => undefined)

    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === DEFAULT_CHARACTER_STORAGE_KEY) {
        return [null, setDefaultCharacterMock, { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Set as default" }))

    await waitFor(() => {
      expect(setDefaultCharacterMock).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-default",
          name: "Default Candidate",
          system_prompt: "Default system prompt",
          greeting: "Default greeting"
        })
      )
    })
    expect(tldwClientMock.setDefaultCharacterPreference).toHaveBeenCalledWith(
      "char-default"
    )
  }, 30000)

  it("clears default character from row actions when selected record is default", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-default-clear",
      name: "Default Clear Candidate",
      system_prompt: "Default system prompt",
      greeting: "Default greeting",
      version: 1
    }
    const setDefaultCharacterMock = vi.fn(async () => undefined)

    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === DEFAULT_CHARACTER_STORAGE_KEY) {
        return [
          {
            id: "char-default-clear",
            name: "Default Clear Candidate",
            system_prompt: "Default system prompt",
            greeting: "Default greeting"
          },
          setDefaultCharacterMock,
          { isLoading: false }
        ]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Clear default" }))

    await waitFor(() => {
      expect(setDefaultCharacterMock).toHaveBeenCalledWith(null)
    })
    expect(tldwClientMock.setDefaultCharacterPreference).toHaveBeenCalledWith(
      null
    )
  }, 30000)

  it("shows conversation insights summary with last active and average message count", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-conv-insights",
      name: "Conversation Insights Character",
      system_prompt: "Conversation insights prompt",
      greeting: "Conversation insights greeting",
      version: 1
    }

    tldwClientMock.listChats.mockResolvedValueOnce([
      {
        id: "chat-1",
        title: "First",
        character_id: "char-conv-insights",
        created_at: "2026-02-10T10:00:00.000Z",
        updated_at: "2026-02-12T12:00:00.000Z",
        message_count: 4
      },
      {
        id: "chat-2",
        title: "Second",
        character_id: "char-conv-insights",
        created_at: "2026-02-11T10:00:00.000Z",
        updated_at: "2026-02-14T14:00:00.000Z",
        message_count: 6
      }
    ])

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(
      await screen.findByRole("menuitem", { name: "View conversations" })
    )

    await waitFor(() => {
      expect(tldwClientMock.listChats).toHaveBeenCalledWith(
        expect.objectContaining({
          character_id: "char-conv-insights"
        })
      )
    })

    expect(await screen.findByText(/Last active:/)).toBeInTheDocument()
    expect(await screen.findByText("Avg messages: 5")).toBeInTheDocument()
  }, 30000)

  it("opens quick chat from table actions and sends a character-scoped prompt", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "101",
      name: "Quick Chat Character",
      system_prompt: "Stay concise and roleplay as this character.",
      greeting: "Hello from quick chat",
      description: "Quick test",
      version: 1
    }

    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === "selectedModel") {
        return ["mock-chat-model", vi.fn(), { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    tldwClientMock.createChat.mockResolvedValueOnce({
      id: "quick-chat-session-1"
    })
    tldwClientMock.completeCharacterChatTurn.mockResolvedValueOnce({
      assistant_content: "Character quick reply"
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({
          data: [{ model: "mock-chat-model", provider: "openai" }]
        })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Test in popup" }))

    const input = await screen.findByPlaceholderText(
      "Ask this character a quick question..."
    )

    await user.type(input, "How do you help?")
    await user.click(screen.getByRole("button", { name: "Send" }))

    await waitFor(() => {
      expect(tldwClientMock.createChat).toHaveBeenCalledWith(
        expect.objectContaining({
          character_id: 101
        })
      )
      expect(tldwClientMock.completeCharacterChatTurn).toHaveBeenCalledWith(
        "quick-chat-session-1",
        expect.objectContaining({
          model: "mock-chat-model",
          append_user_message: "How do you help?"
        })
      )
    })
    expect(navigateMock).not.toHaveBeenCalled()
    expect(await screen.findByText("Character quick reply")).toBeInTheDocument()
  }, 30000)

  it("opens quick chat from gallery preview actions", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "202",
      name: "Gallery Quick Chat Character",
      system_prompt: "Gallery quick chat system prompt",
      greeting: "Hi from gallery quick chat",
      description: "Gallery entry",
      version: 1
    }

    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === "selectedModel") {
        return ["mock-chat-model", vi.fn(), { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    window.localStorage.setItem("characters-view-mode", "gallery")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({
          data: [{ model: "mock-chat-model", provider: "openai" }]
        })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(
      await screen.findByRole("button", {
        name: "Click to preview Gallery Quick Chat Character"
      })
    )
    // The gallery preview action menu lives inside the preview modal, so target
    // its explicit aria-label instead of the tooltip text node from the card.
    const moreButton = await screen.findByLabelText(
      "More actions for Gallery Quick Chat Character"
    )
    await user.click(moreButton)
    await user.click(
      await screen.findByRole("menuitem", {
        name: "Test in popup"
      })
    )

    await screen.findByPlaceholderText(
      "Ask this character a quick question..."
    )
    expect(screen.getByText("Hi from gallery quick chat")).toBeInTheDocument()
  }, 30000)

  it("shows attached world books in gallery preview context", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "preview-worldbook",
      name: "Preview Worldbook Character",
      system_prompt: "Preview world-book prompt",
      greeting: "Preview greeting",
      description: "Preview world-book description",
      version: 1
    }

    window.localStorage.setItem("characters-view-mode", "gallery")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "tldw:characterPreviewWorldBooks") {
        return makeUseQueryResult({
          data: [
            { id: 11, name: "Lore Atlas" },
            { id: 22, name: "Chronicle Index" }
          ]
        })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByText("Preview Worldbook Character"))

    const previewTitle = await screen.findByText("Character Preview")
    const previewModal = previewTitle.closest(".ant-modal")
    expect(previewModal).not.toBeNull()

    const modalScope = within(previewModal as HTMLElement)
    expect(modalScope.getByText("World Books")).toBeInTheDocument()
    expect(modalScope.getByRole("link", { name: "Open World Books workspace" })).toHaveAttribute(
      "href",
      expect.stringContaining("focusCharacterId=preview-worldbook")
    )

    const loreLink = modalScope.getByRole("link", { name: "Open world book Lore Atlas" })
    expect(loreLink).toHaveAttribute("href", expect.stringContaining("focusWorldBookId=11"))
    expect(loreLink).toHaveAttribute(
      "href",
      expect.stringContaining("focusCharacterId=preview-worldbook")
    )

    const chronicleLink = modalScope.getByRole("link", {
      name: "Open world book Chronicle Index"
    })
    expect(chronicleLink).toHaveAttribute("href", expect.stringContaining("focusWorldBookId=22"))
  }, 30000)

  it("opens a full-size image modal when the preview avatar is clicked", async () => {
    const user = userEvent.setup()
    const avatarUrl = "https://example.com/preview-avatar.png"
    const characterRecord = {
      id: "preview-avatar-click",
      name: "Preview Avatar Character",
      avatar_url: avatarUrl,
      system_prompt: "Avatar preview prompt",
      greeting: "Avatar preview greeting",
      description: "Avatar preview description",
      version: 1
    }

    window.localStorage.setItem("characters-view-mode", "gallery")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "tldw:characterPreviewWorldBooks") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByText("Preview Avatar Character"))
    await screen.findByText("Character Preview")

    await user.click(await screen.findByTestId("character-preview-avatar-button"))

    const fullImage = await screen.findByTestId("character-preview-full-image")
    expect(fullImage).toHaveAttribute("src", avatarUrl)
    expect(await screen.findByText("Character image")).toBeInTheDocument()
  }, 30000)

  it("shows world-book empty state in gallery preview when no attachments exist", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "preview-no-worldbook",
      name: "Preview Empty Worldbooks",
      system_prompt: "No world books",
      greeting: "No linked books",
      description: "Empty preview",
      version: 1
    }

    window.localStorage.setItem("characters-view-mode", "gallery")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "tldw:characterPreviewWorldBooks") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByText("Preview Empty Worldbooks"))
    await screen.findByText("Character Preview")
    expect(
      await screen.findByText("No world books attached to this character.")
    ).toBeInTheDocument()
  }, 30000)

  it("shows world-book loading state in gallery preview while attachments fetch", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "preview-loading-worldbook",
      name: "Preview Loading Worldbooks",
      system_prompt: "Loading world books",
      greeting: "Loading linked books",
      description: "Loading preview",
      version: 1
    }

    window.localStorage.setItem("characters-view-mode", "gallery")

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "tldw:characterPreviewWorldBooks") {
        return makeUseQueryResult({ data: [], isFetching: true })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByText("Preview Loading Worldbooks"))
    await screen.findByText("Character Preview")
    expect(
      await screen.findByText("Loading attached world books...")
    ).toBeInTheDocument()
  }, 30000)

  it("promotes quick chat into full chat flow without route changes until promoted", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "303",
      name: "Promote Quick Chat Character",
      system_prompt: "Promotion prompt",
      greeting: "Promotion greeting",
      description: "Promotion test",
      alternate_greetings: ["Backup greeting"],
      extensions: { tone: "steady" },
      version: 1
    }

    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === "selectedModel") {
        return ["mock-chat-model", vi.fn(), { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    tldwClientMock.createChat.mockResolvedValueOnce({
      id: "quick-chat-session-promote"
    })
    tldwClientMock.completeCharacterChatTurn.mockResolvedValueOnce({
      assistant_content: "Promotion reply"
    })

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({
          data: [{ model: "mock-chat-model", provider: "openai" }]
        })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Test in popup" }))

    const quickChatInput = await screen.findByPlaceholderText(
      "Ask this character a quick question..."
    )

    await user.type(quickChatInput, "Promote this thread")
    await user.click(screen.getByRole("button", { name: "Send" }))
    await screen.findByText("Promotion reply")
    expect(navigateMock).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: "Open full chat" }))

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith("/")
    })
    expect(setSelectedCharacterMock).toHaveBeenCalled()
    expect(setSelectedCharacterMock).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "Promote Quick Chat Character",
        alternate_greetings: expect.any(Array),
        extensions: expect.anything()
      })
    )
    expect(storeSetters.setServerChatCharacterId).toHaveBeenCalledWith("303")
    expect(storeSetters.setServerChatAssistantKind).toHaveBeenCalledWith("character")
    expect(storeSetters.setServerChatAssistantId).toHaveBeenCalledWith("303")
    expect(storeSetters.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(tldwClientMock.deleteChat).not.toHaveBeenCalled()
  }, 30000)

  it("opens chat in a new tab from row actions without replacing current-tab navigation", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-new-tab",
      name: "New Tab Character",
      system_prompt: "Use new tab flow.",
      greeting: "Hi from new tab",
      version: 1
    }

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    const openSpy = vi
      .spyOn(window, "open")
      .mockImplementation(() => ({ closed: false } as unknown as Window))

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Chat in new tab" }))

    await waitFor(() => {
      expect(setSelectedCharacterMock).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-new-tab",
          name: "New Tab Character"
        })
      )
      expect(openSpy).toHaveBeenCalledWith(
        expect.any(String),
        "_blank",
        "noopener,noreferrer"
      )
    })
    expect(navigateMock).not.toHaveBeenCalled()

    openSpy.mockRestore()
  }, 30000)

  it("opens chat in a new tab from actions menu", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-new-tab-gallery",
      name: "Gallery New Tab Character",
      system_prompt: "Gallery new-tab prompt.",
      greeting: "Gallery greeting",
      version: 1
    }

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    const openSpy = vi
      .spyOn(window, "open")
      .mockImplementation(() => ({ closed: false } as unknown as Window))

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Chat in new tab" }))

    await waitFor(() => {
      expect(setSelectedCharacterMock).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "char-new-tab-gallery",
          name: "Gallery New Tab Character"
        })
      )
      expect(openSpy).toHaveBeenCalledWith(
        expect.any(String),
        "_blank",
        "noopener,noreferrer"
      )
    })
    expect(navigateMock).not.toHaveBeenCalled()

    openSpy.mockRestore()
  }, 30000)

  it("shows a warning when chat-in-new-tab is blocked by the browser", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: "char-popup-blocked",
      name: "Popup Blocked Character",
      system_prompt: "Popup test prompt.",
      greeting: "Popup test greeting",
      version: 1
    }

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null)

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(await screen.findByRole("menuitem", { name: "Chat in new tab" }))

    await waitFor(() => {
      expect(notificationMock.warning).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Popup blocked"
        })
      )
    })

    openSpy.mockRestore()
  }, 30000)

  it("keeps AI field generation affordances wired in create mode", async () => {
    const user = userEvent.setup()
    useStorageMock.mockImplementation((key: unknown, defaultValue: unknown) => {
      if (resolveStorageKey(key) === "characterGenModel") {
        return ["mock-generation-model", vi.fn(), { isLoading: false }]
      }
      return [defaultValue ?? null, vi.fn(), { isLoading: false }]
    })

    render(<CharactersManager />)
    await user.click(screen.getByRole("button", { name: "New character" }))

    const createSubmitButton = await waitFor(() => {
      const candidate = screen
        .getAllByRole("button", { name: "Create character" })
        .find((button) => button.getAttribute("type") === "submit")
      expect(candidate).toBeDefined()
      return candidate as HTMLElement
    })
    const createFormElement = createSubmitButton.closest("form")
    expect(createFormElement).not.toBeNull()
    const createScope = within(createFormElement as HTMLElement)

    const generateButtons = createScope.getAllByLabelText("Generate with AI")
    expect(generateButtons.length).toBeGreaterThan(0)
    await user.click(generateButtons[0])

    await waitFor(() => {
      expect(generateFieldMock).toHaveBeenCalled()
    })

    const generateCall = generateFieldMock.mock.calls[0]
    expect(generateCall?.[2]).toMatchObject({
      model: "mock-generation-model"
    })
  }, 30000)

  it("registers keyboard shortcut callbacks for core manager actions", async () => {
    render(<CharactersManager />)

    expect(useCharacterShortcutsMock).toHaveBeenCalled()
    const shortcutOptions = useCharacterShortcutsMock.mock.calls[0]?.[0]
    expect(shortcutOptions?.enabled).toBe(true)
    expect(typeof shortcutOptions?.onNewCharacter).toBe("function")
    expect(typeof shortcutOptions?.onFocusSearch).toBe("function")
    expect(typeof shortcutOptions?.onCloseModal).toBe("function")
  }, 45000)

  it("adds skip-link, main landmark, and shortcut summary semantics", async () => {
    render(<CharactersManager />)

    const skipLink = screen.getByRole("link", {
      name: "Skip to characters content"
    })
    expect(skipLink).toHaveAttribute("href", "#characters-main-content")

    const mainRegion = screen.getByRole("main")
    expect(mainRegion).toHaveAttribute("id", "characters-main-content")
    expect(mainRegion).toHaveAttribute(
      "aria-describedby",
      "characters-shortcuts-summary"
    )
    expect(mainRegion.parentElement).toHaveClass("characters-page")

    const shortcutSummary = document.getElementById("characters-shortcuts-summary")
    expect(shortcutSummary).not.toBeNull()
    expect(shortcutSummary).toHaveTextContent("Keyboard shortcuts")

    expect(
      screen.getByRole("button", { name: "Display options" })
    ).toBeInTheDocument()
  })

  it("allows keyboard focus on display options button", async () => {
    render(<CharactersManager />)

    const displayButton = screen.getByRole("button", {
      name: "Display options"
    })
    // Verify the button is accessible and focusable (not disabled, not hidden)
    expect(displayButton).toBeEnabled()
    expect(displayButton).toBeVisible()
    expect(displayButton.tabIndex).not.toBe(-1)
  })

  it("renders reduced-motion utility classes on row action controls", async () => {
    const records = [
      {
        id: "motion-1",
        name: "Motion Ready Character",
        system_prompt: "Prompt text",
        version: 1
      }
    ]

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: records, status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    render(<CharactersManager />)

    const chatButton = await screen.findByRole("button", { name: /Chat/i })
    expect(chatButton.className).toContain("motion-reduce:transition-none")
  })

  it("imports a character file through the upload control", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const file = new File(
      [JSON.stringify({ name: "Uploaded Character" })],
      "uploaded-character.json",
      { type: "application/json" }
    )
    await user.upload(input, file)

    expect(tldwClientMock.importCharacterFile).not.toHaveBeenCalled()
    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("Uploaded Character")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalled()
    })
    expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
      expect.objectContaining({ name: "uploaded-character.json" }),
      expect.objectContaining({ allowImageOnly: false })
    )
  })

  it("shows an OK action after successful import completion", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const file = new File(
      [JSON.stringify({ name: "Uploaded Character" })],
      "uploaded-character.json",
      { type: "application/json" }
    )
    await user.upload(input, file)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalled()
    })

    const okButton = await screen.findByRole("button", { name: /ok/i })
    expect(okButton).toBeEnabled()
    expect(screen.queryByRole("button", { name: "Confirm import" })).not.toBeInTheDocument()
  })

  it("creates a persona from a character and opens Persona Garden on the new profile", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: 7,
      name: "Captain A",
      description: "Command strategist",
      system_prompt: "Lead with confidence.",
      greeting: "Ready for orders?",
      version: 1
    }

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    tldwClientMock.fetchWithAuth.mockImplementation((path: string, init?: any) => {
      if (path === "/api/v1/persona/profiles?limit=200") {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/profiles" && init?.method === "POST") {
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: "persona-captain-a" })
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Create Persona from Character" })
    )

    await waitFor(() => {
      expect(tldwClientMock.fetchWithAuth).toHaveBeenCalledWith(
        "/api/v1/persona/profiles",
        expect.objectContaining({
          method: "POST",
          body: expect.objectContaining({
            name: "Captain A Persona",
            character_card_id: 7,
            mode: "persistent_scoped"
          })
        })
      )
    })
    expect(navigateMock).toHaveBeenCalledWith(
      "/persona?persona_id=persona-captain-a&tab=profiles"
    )
  }, 15000)

  it("disables create persona while a create request is already pending for the character", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: 7,
      name: "Captain A",
      description: "Command strategist",
      system_prompt: "Lead with confidence.",
      greeting: "Ready for orders?",
      version: 1
    }
    let resolveCreate: ((value: any) => void) | null = null

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    tldwClientMock.fetchWithAuth.mockImplementation((path: string, init?: any) => {
      if (path === "/api/v1/persona/profiles?limit=200") {
        return Promise.resolve({
          ok: true,
          json: async () => []
        })
      }
      if (path === "/api/v1/persona/profiles" && init?.method === "POST") {
        return new Promise((resolve) => {
          resolveCreate = resolve
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Create Persona from Character" })
    )

    await waitFor(() => {
      expect(
        tldwClientMock.fetchWithAuth.mock.calls.filter(
          ([path, init]) =>
            path === "/api/v1/persona/profiles" && init?.method === "POST"
        )
      ).toHaveLength(1)
    })

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    const pendingCreateItem = await screen.findByRole("menuitem", {
      name: "Creating Persona..."
    })
    expect(pendingCreateItem).toHaveAttribute("aria-disabled", "true")

    resolveCreate?.({
      ok: true,
      json: async () => ({ id: "persona-captain-a" })
    })

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith(
        "/persona?persona_id=persona-captain-a&tab=profiles"
      )
    })
  }, 15000)

  it("opens the existing derived persona in Persona Garden from row actions", async () => {
    const user = userEvent.setup()
    const characterRecord = {
      id: 9,
      name: "Archivist",
      description: "Knows the stacks.",
      system_prompt: "Preserve institutional memory.",
      greeting: "What record do you need?",
      version: 1
    }

    useQueryMock.mockImplementation((opts: any) => {
      const key = Array.isArray(opts?.queryKey) ? opts.queryKey[0] : undefined
      if (key === "tldw:listCharacters") {
        return makeUseQueryResult({ data: [characterRecord], status: "success" })
      }
      if (key === "getModelsForFieldGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "getAllModelsForGeneration") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:characterConversationCounts") {
        return makeUseQueryResult({ data: {} })
      }
      return makeUseQueryResult({})
    })

    tldwClientMock.fetchWithAuth.mockImplementation((path: string) => {
      if (path === "/api/v1/persona/profiles?limit=200") {
        return Promise.resolve({
          ok: true,
          json: async () => [
            {
              id: "persona-archivist",
              name: "Archivist Persona",
              origin_character_id: 9
            }
          ]
        })
      }
      return Promise.resolve({
        ok: false,
        error: `unhandled path: ${path}`,
        json: async () => ({})
      })
    })

    render(<CharactersManager />)

    await user.click(await screen.findByRole("button", { name: /More actions/i }))
    await user.click(
      await screen.findByRole("menuitem", { name: "Open in Persona Garden" })
    )

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith(
        "/persona?persona_id=persona-archivist&tab=profiles"
      )
    })
  }, 15000)

  it("opens import preview when files are dropped on the import drop zone", async () => {
    render(<CharactersManager />)

    const dropZone = screen.getByTestId("character-import-dropzone")
    const droppedFile = new File(
      [JSON.stringify({ name: "Dropped Character" })],
      "dropped-character.json",
      { type: "application/json" }
    )
    const dataTransfer = {
      files: [droppedFile]
    }

    fireEvent.dragEnter(dropZone, { dataTransfer })
    fireEvent.dragOver(dropZone, { dataTransfer })
    fireEvent.drop(dropZone, { dataTransfer })

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("Dropped Character")).toBeInTheDocument()
    expect(
      screen.getByTestId("character-import-progress-summary")
    ).toHaveTextContent("Queued 1")
  })

  it("renders per-file runtime status transitions during batch import", async () => {
    const user = userEvent.setup()
    let resolveFirstImport: ((value: { success: boolean; message: string }) => void) | null =
      null
    const firstImport = new Promise<{ success: boolean; message: string }>((resolve) => {
      resolveFirstImport = resolve
    })
    tldwClientMock.importCharacterFile
      .mockImplementationOnce(() => firstImport)
      .mockRejectedValueOnce(new Error("Invalid character payload"))

    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const fileOne = new File(
      [JSON.stringify({ name: "Status One" })],
      "status-one.json",
      { type: "application/json" }
    )
    const fileTwo = new File(
      [JSON.stringify({ name: "Status Two" })],
      "status-two.json",
      { type: "application/json" }
    )
    await user.upload(input, [fileOne, fileTwo])
    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(screen.getAllByTestId("character-import-status-queued")).toHaveLength(2)

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(screen.getByTestId("character-import-status-processing")).toBeInTheDocument()
    })

    resolveFirstImport?.({
      success: true,
      message: "First import ok"
    })

    await waitFor(() => {
      expect(screen.getByTestId("character-import-status-success")).toBeInTheDocument()
      expect(screen.getByTestId("character-import-status-failure")).toBeInTheDocument()
    })

    const summary = screen.getByTestId("character-import-progress-summary")
    expect(summary).toHaveTextContent("Success 1")
    expect(summary).toHaveTextContent("Failed 1")
  })

  it("retries only failed files from import preview without duplicating successful imports", async () => {
    const user = userEvent.setup()
    tldwClientMock.importCharacterFile
      .mockResolvedValueOnce({ success: true, message: "First import ok" })
      .mockRejectedValueOnce(new Error("Second import failed"))
      .mockResolvedValueOnce({ success: true, message: "Second retry ok" })

    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const fileOne = new File(
      [JSON.stringify({ name: "Retry One" })],
      "retry-one.json",
      { type: "application/json" }
    )
    const fileTwo = new File(
      [JSON.stringify({ name: "Retry Two" })],
      "retry-two.json",
      { type: "application/json" }
    )

    await user.upload(input, [fileOne, fileTwo])
    expect(await screen.findByText("Import preview")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledTimes(2)
      expect(screen.getByTestId("character-import-status-failure")).toBeInTheDocument()
    })

    await user.click(screen.getByRole("button", { name: "Retry failed" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledTimes(3)
      expect(screen.queryByTestId("character-import-status-failure")).not.toBeInTheDocument()
    })

    const importCharacterFileCalls = tldwClientMock.importCharacterFile.mock
      .calls as unknown as Array<[File, { allowImageOnly?: boolean }?]>
    const thirdCallFile = importCharacterFileCalls[2]?.[0]
    expect(thirdCallFile?.name).toBe("retry-two.json")
  })

  it("imports a batch of files and reports per-file failures", async () => {
    const user = userEvent.setup()
    tldwClientMock.importCharacterFile
      .mockResolvedValueOnce({ success: true, message: "First import ok" })
      .mockRejectedValueOnce(new Error("Invalid character payload"))

    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()
    expect(input).toHaveAttribute("multiple")

    const fileOne = new File(
      [JSON.stringify({ name: "Batch One" })],
      "batch-one.json",
      { type: "application/json" }
    )
    const fileTwo = new File(
      [JSON.stringify({ name: "Batch Two" })],
      "batch-two.json",
      { type: "application/json" }
    )

    await user.upload(input, [fileOne, fileTwo])
    expect(await screen.findByText("Import preview")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledTimes(2)
    })
    await waitFor(() => {
      expect(notificationMock.warning).toHaveBeenCalled()
    })

    const warningPayload = notificationMock.warning.mock.calls.at(-1)?.[0]
    expect(warningPayload?.description).toContain("succeeded")
    expect(warningPayload?.description).toContain("failed")
    expect(warningPayload?.description).toContain(
      "batch-two.json: Invalid character payload"
    )
  })

  it("parses YAML metadata in preview before confirming import", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const yamlFile = new File(
      [
        [
          "name: YAML Character",
          "description: Imported from YAML preview",
          "tags: [assistant, yaml]"
        ].join("\n")
      ],
      "yaml-character.yaml",
      { type: "text/yaml" }
    )

    await user.upload(input, yamlFile)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("YAML Character")).toBeInTheDocument()
    expect(
      await screen.findByText("Imported from YAML preview")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
        expect.objectContaining({ name: "yaml-character.yaml" }),
        expect.objectContaining({ allowImageOnly: false })
      )
    })
  })

  it("imports markdown metadata files through preview + confirm flow", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const markdownFile = new File(
      [
        [
          "name: Markdown Character",
          "description: Imported from markdown metadata",
          "tags:",
          "  - markdown",
          "  - import"
        ].join("\n")
      ],
      "markdown-character.md",
      { type: "text/markdown" }
    )

    await user.upload(input, markdownFile)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("Markdown Character")).toBeInTheDocument()
    expect(
      await screen.findByText("Imported from markdown metadata")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
        expect.objectContaining({ name: "markdown-character.md" }),
        expect.objectContaining({ allowImageOnly: false })
      )
    })
  })

  it("imports text files containing JSON card data through preview + confirm flow", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const textJsonFile = new File(
      [
        JSON.stringify({
          spec: "chara_card_v3",
          spec_version: "3.0",
          data: {
            name: "Text JSON Character",
            description: "Imported from text file JSON payload",
            tags: ["text-json", "import"]
          }
        })
      ],
      "text-json-card.txt",
      { type: "text/plain" }
    )

    await user.upload(input, textJsonFile)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("Text JSON Character")).toBeInTheDocument()
    expect(
      await screen.findByText("Imported from text file JSON payload")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
        expect.objectContaining({ name: "text-json-card.txt" }),
        expect.objectContaining({ allowImageOnly: false })
      )
    })
  })

  it("imports PNG character files through preview + confirm flow", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const pngSignature = new Uint8Array([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a
    ])
    const pngFile = new File([pngSignature], "png-character.png", {
      type: "image/png"
    })

    await user.upload(input, pngFile)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("png-character")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
        expect.objectContaining({ name: "png-character.png" }),
        expect.objectContaining({ allowImageOnly: false })
      )
    })
  })

  it("round-trips exported v3 JSON shape through import preview + confirm", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const exportedV3Payload = {
      spec: "chara_card_v3",
      spec_version: "3.0",
      data: {
        name: "Round Trip Character",
        description: "Round-trip export/import payload",
        tags: ["round-trip", "v3"]
      }
    }
    const exportedFile = new File(
      [JSON.stringify(exportedV3Payload, null, 2)],
      "round-trip-character.json",
      { type: "application/json" }
    )

    await user.upload(input, exportedFile)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText("Round Trip Character")).toBeInTheDocument()
    expect(
      await screen.findByText("Round-trip export/import payload")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Confirm import" }))

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
        expect.objectContaining({ name: "round-trip-character.json" }),
        expect.objectContaining({ allowImageOnly: false })
      )
    })
  })

  it("surfaces malformed JSON preview errors and blocks confirm import", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const broken = new File(["{ invalid"], "broken-character.json", {
      type: "application/json"
    })
    await user.upload(input, broken)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText(/Invalid JSON syntax/i)).toBeInTheDocument()

    const confirmButton = screen.getByRole("button", { name: "Confirm import" })
    expect(confirmButton).toBeDisabled()
    expect(tldwClientMock.importCharacterFile).not.toHaveBeenCalled()
  }, 15000)

  it("surfaces malformed YAML preview errors and blocks confirm import", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const brokenYaml = new File(
      [
        [
          "name: Broken YAML Character",
          "description: This one has malformed list syntax",
          "tags: [assistant, yaml"
        ].join("\n")
      ],
      "broken-character.yaml",
      { type: "text/yaml" }
    )
    await user.upload(input, brokenYaml)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText(/Malformed YAML content:/i)).toBeInTheDocument()

    const confirmButton = screen.getByRole("button", { name: "Confirm import" })
    expect(confirmButton).toBeDisabled()
    expect(tldwClientMock.importCharacterFile).not.toHaveBeenCalled()
  }, 15000)

  it("allows canceling import preview without persisting files", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const file = new File(
      [JSON.stringify({ name: "Cancel Me" })],
      "cancel-me.json",
      { type: "application/json" }
    )
    await user.upload(input, file)

    const previewTitle = await screen.findByText("Import preview")
    const modalRoot = previewTitle.closest(".ant-modal") as HTMLElement | null
    const cancelButton = modalRoot
      ? within(modalRoot).getByRole("button", { name: "Cancel" })
      : screen.getByRole("button", { name: "Cancel" })
    await user.click(cancelButton)

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).not.toHaveBeenCalled()
    })
  }, 15000)

  it("imports valid files and warns when malformed previews are skipped", async () => {
    const user = userEvent.setup()
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const valid = new File([JSON.stringify({ name: "Valid One" })], "valid-one.json", {
      type: "application/json"
    })
    const broken = new File(["{ invalid"], "broken-one.json", {
      type: "application/json"
    })
    await user.upload(input, [valid, broken])

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    const confirmButton = screen.getByRole("button", { name: "Confirm import" })
    expect(confirmButton).toBeEnabled()

    await user.click(confirmButton)

    await waitFor(() => {
      expect(tldwClientMock.importCharacterFile).toHaveBeenCalledTimes(1)
    })
    expect(tldwClientMock.importCharacterFile).toHaveBeenCalledWith(
      expect.objectContaining({ name: "valid-one.json" }),
      expect.objectContaining({ allowImageOnly: false })
    )
    await waitFor(() => {
      expect(notificationMock.warning).toHaveBeenCalled()
    })
    const warningPayload = notificationMock.warning.mock.calls.at(-1)?.[0]
    expect(warningPayload?.description).toContain("files were skipped")
  })

  it("shows unsupported extension errors in preview and blocks confirm", async () => {
    const user = userEvent.setup({ applyAccept: false })
    const { container } = render(<CharactersManager />)
    const input = container.querySelector("input[type='file']") as HTMLInputElement
    expect(input).not.toBeNull()

    const unsupportedFile = new File(["name,description"], "unsupported.csv", {
      type: "text/csv"
    })
    await user.upload(input, unsupportedFile)

    expect(await screen.findByText("Import preview")).toBeInTheDocument()
    expect(await screen.findByText(/Unsupported file type:/i)).toBeInTheDocument()

    const confirmButton = screen.getByRole("button", { name: "Confirm import" })
    expect(confirmButton).toBeDisabled()
    expect(tldwClientMock.importCharacterFile).not.toHaveBeenCalled()
  })
})
