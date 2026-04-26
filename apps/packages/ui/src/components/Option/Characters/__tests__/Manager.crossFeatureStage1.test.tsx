import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { CharactersManager } from "../Manager"
import { ensureLocalStorageApi } from "./testUtils"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  useNavigateMock,
  useStorageMock,
  confirmDangerMock,
  navigateMock,
  setSelectedCharacterMock,
  focusComposerMock,
  notificationMock,
  tldwClientMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  useNavigateMock: vi.fn(),
  useStorageMock: vi.fn(),
  confirmDangerMock: vi.fn(async () => true),
  navigateMock: vi.fn(),
  setSelectedCharacterMock: vi.fn(),
  focusComposerMock: vi.fn(),
  notificationMock: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  },
  tldwClientMock: {
    initialize: vi.fn(async () => undefined)
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
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
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
  useCharacterShortcuts: () => undefined
}))

vi.mock("@/hooks/useCharacterGeneration", () => ({
  useCharacterGeneration: () => ({
    isGenerating: false,
    generatingField: null,
    error: null,
    generateFullCharacter: vi.fn(),
    generateField: vi.fn(),
    cancel: vi.fn(),
    clearError: vi.fn()
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
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn()
    })
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

vi.mock("@/data/character-templates", () => ({
  CHARACTER_TEMPLATES: [
    {
      id: "writer-coach",
      name: "Writer Coach",
      description: "Helps with structure and tone.",
      system_prompt: "You are a writing coach.",
      greeting: "Ready to improve your draft?",
      tags: ["writing", "coach"]
    }
  ]
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

describe("CharactersManager cross-feature integration stage-1", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    ensureLocalStorageApi().clear()
    window.history.replaceState(
      {},
      "",
      "/characters?from=world-books&focusCharacterId=7&focusWorldBookId=11"
    )

    useNavigateMock.mockReturnValue(navigateMock)
    confirmDangerMock.mockResolvedValue(true)
    useStorageMock.mockImplementation(
      (_key: string, defaultValue: unknown) => [
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
        return makeUseQueryResult({
          data: {
            items: [
              {
                id: 7,
                slug: "captain-a",
                name: "Captain A",
                description: "Command strategist"
              }
            ],
            total: 1,
            page: 1,
            page_size: 25,
            has_more: false
          },
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
        return makeUseQueryResult({ data: { "7": 0 } })
      }
      if (key === "tldw:characterPreviewWorldBooks") {
        return makeUseQueryResult({
          data: [
            { id: 21, name: "Lore Atlas" },
            { id: 22, name: "Ship Registry" }
          ]
        })
      }
      return makeUseQueryResult({})
    })
  })

  it("opens focused character preview from world-books route context with attached world-book links", async () => {
    render(<CharactersManager />)

    expect(await screen.findByText("Character Preview")).toBeInTheDocument()
    const backLink = screen.getByRole("link", { name: "Back to World Books" })
    expect(backLink).toHaveAttribute("href", "/world-books?focusWorldBookId=11")

    const openWorkspaceLink = screen.getByRole("link", {
      name: "Open World Books workspace"
    })
    expect(openWorkspaceLink).toHaveAttribute(
      "href",
      "/world-books?from=characters&focusCharacterId=7"
    )

    const loreAtlasLink = screen.getByText("Lore Atlas")
    expect(loreAtlasLink).toHaveAttribute(
      "href",
      "/world-books?from=characters&focusCharacterId=7&focusWorldBookId=21"
    )
  })
})
