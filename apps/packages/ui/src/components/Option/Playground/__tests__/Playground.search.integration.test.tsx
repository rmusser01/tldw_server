// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useHistorySelection, useHistorySelectionContext, type HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { resolveHistorySelection } from "@/utils/history-selection"
import { Playground } from "../Playground"
import {
  decodeSidepanelChatWebUiHandoff,
  encodeSidepanelChatWebUiHandoff,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE
} from "@/services/tldw/sidepanel-chat-webui-handoff"

const h1 = vi.hoisted(() => ({ enabled: false, legacyNative: false, controller: null as HistorySelectionController | null, bookmarks: new Map<string, any>(), confirm: vi.fn() }))
const h1Key = (scope: any, owner: any) => JSON.stringify([scope.profile_id, scope.client_session_id, owner.owner_key, owner.conversation_id])

const forkPresentation = vi.hoisted(() => ({controller: null as HistorySelectionController | null, mode: null as "ordinary" | "pending" | "fork" | null, settings: null as {authorNote: string} | null}))
vi.mock("@/hooks/chat/useHistorySelection", async importOriginal => {
  const actual = await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>()
  return {...actual, useHistorySelectionContext: () => {
    const controller = actual.useHistorySelectionContext()
    return forkPresentation.controller ?? (forkPresentation.mode ? {...controller, settingsMode: () => forkPresentation.mode!, forkSettings: forkPresentation.settings} : controller)
  }}
})

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [
      { id: "m-1", message: "alpha message", isBot: false, role: "user" },
      { id: "m-2", message: "beta response", isBot: true, role: "assistant" }
    ],
    history: [],
    historyId: "history-1",
    serverChatId: "chat-1",
    isLoading: false,
    selectedModel: "model-1",
    selectedSystemPrompt: "prompt-1",
    selectedQuickPrompt: "quick-1",
    chatMode: "normal",
    webSearch: false,
    toolChoice: "none",
    temporaryChat: false,
    useOCR: false,
    fileRetrievalEnabled: false,
    ragMediaIds: null as number[] | null,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedQuickPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    setChatMode: vi.fn(),
    setWebSearch: vi.fn(),
    setToolChoice: vi.fn(),
    setTemporaryChat: vi.fn(),
    setUseOCR: vi.fn(),
    setFileRetrievalEnabled: vi.fn(),
    setRagMediaIds: vi.fn(),
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false
  }
}))

const artifactsState = vi.hoisted(() => ({
  value: {
    isOpen: false,
    active: null,
    isPinned: false,
    history: [],
    unreadCount: 0,
    setOpen: vi.fn(),
    closeArtifact: vi.fn(),
    markRead: vi.fn()
  }
}))

const smartScrollState = vi.hoisted(() => ({
  value: {
    containerRef: { current: null } as React.MutableRefObject<HTMLDivElement | null>,
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn()
  }
}))

const mobileViewportState = vi.hoisted(() => ({
  value: false
}))

const desktopViewportState = vi.hoisted(() => ({
  value: true
}))

const storageState = vi.hoisted(() => ({
  value: new Map<string, unknown>()
}))

const artifactFixture = vi.hoisted(() => ({
  id: "artifact-1",
  title: "Generated table",
  content: "a,b\n1,2",
  kind: "table" as const
}))

const storeOptionState = vi.hoisted(() => ({
  value: {
    compareParentByHistory: {} as Record<
      string,
      { parentHistoryId: string; clusterId?: string }
    >,
    uploadedFiles: [] as Array<Record<string, unknown>>,
    contextFiles: [] as Array<Record<string, unknown>>,
    setUploadedFiles: vi.fn(),
    setContextFiles: vi.fn()
  }
}))

const tldwClientState = vi.hoisted(() => ({
  getConfig: vi.fn(async () => ({})),
  getProvidersStatus: vi.fn(async () => ({})),
  initialize: vi.fn(async () => null),
  getResearchBundle: vi.fn(async () => null),
  getDocumentUploadDraft: vi.fn(async () => ({ payload: {} })),
  deleteDocumentUploadDraft: vi.fn(async () => undefined)
}))

const routerState = vi.hoisted(() => ({
  hashRouter: false,
  navigate: vi.fn()
}))

type ChatSettingsSyncParams = {
  historyId: string | null
  serverChatId: string | null
}

type ChatSettingsPatchParams = ChatSettingsSyncParams & {
  patch: Record<string, unknown>
}

const chatSettingsState = vi.hoisted(() => ({
  syncChatSettingsForServerChat: vi.fn(
    async (_params: ChatSettingsSyncParams): Promise<unknown> => null
  ),
  applyChatSettingsPatch: vi.fn(
    async (_params: ChatSettingsPatchParams): Promise<unknown> => null
  )
}))

const loadLocalConversationMock = vi.hoisted(() => vi.fn(async () => {}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string, options?: Record<string, unknown>) => {
      const template = defaultValue || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: () => <div data-testid="playground-form" />
}))

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: React.forwardRef(function MockPlaygroundChat(
    props: {
      searchQuery?: string
      matchedMessageIndices?: Set<number>
      activeSearchMessageIndex?: number | null
    },
    _ref
  ) {
    const selection = useHistorySelectionContext()
    if (h1.enabled) h1.controller = selection
    return (
      <div
        data-selected-history={h1.enabled ? selection?.capture?.status === "captured" ? selection.capture.selected_content.map(row => row.message).join(" / ") : selection?.status : ""}
        data-testid="playground-chat"
        data-search-query={props.searchQuery || ""}
        data-search-count={props.matchedMessageIndices?.size || 0}
        data-search-active-index={
          props.activeSearchMessageIndex == null ? "" : props.activeSearchMessageIndex
        }
      />
    )
  })
}))

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => <div data-testid="artifacts-panel" />
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => h1.enabled ? { ...messageOptionState.value, ...useStoreMessageOption() } : messageOptionState.value
}))

vi.mock("@/hooks/usePlaygroundSessionPersistence", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/hooks/usePlaygroundSessionPersistence")>()
  return { usePlaygroundSessionPersistence: () => h1.enabled ? actual.usePlaygroundSessionPersistence() : ({
    restoreSession: vi.fn(async () => false), sessionScopeReady: true, hasPersistedSession: false, persistedHistoryId: null, persistedServerChatId: null
  }) }
})

vi.mock("@/hooks/playground-session-restore", () => ({
  shouldRestorePersistedPlaygroundSession: () => false
}))

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
}))

vi.mock("@/db/dexie/helpers", () => ({
  getFullChatData: vi.fn(async (id: string) => ({ historyInfo: { id }, messages: [] })),
  getSessionFiles: vi.fn(async () => []),
  formatSelectedHistory: (capture: any) => ({ history: capture.selected_content.map((row: any) => ({ role: "assistant", content: row.message })), messages: capture.selected_content.map((row: any) => ({ id: row.id, message: row.message, isBot: true, name: "Assistant", sources: [] })) }),
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getPromptById: vi.fn(async () => null),
  getRecentChatFromWebUI: vi.fn(async () => null)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ setSystemPrompt: vi.fn() })
}))

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => smartScrollState.value
}))

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage",
  THEME_SETTING: {
    key: "theme",
    defaultValue: "dark"
  },
  HEADER_SHORTCUT_IDS: [],
  SIDEBAR_SHORTCUT_IDS: [],
  CHAT_WINDOW_OPACITY_SETTING: "chatWindowOpacity",
  CHAT_MESSAGE_OPACITY_SETTING: "chatMessageOpacity",
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING: "chatCharacterImageOpacity",
  resolveOpacityAlpha: (value: unknown, fallback = 35) =>
    typeof value === "number" && Number.isFinite(value)
      ? value / 100
      : fallback / 100
}))

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: []
}))

vi.mock("@/store/option", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/store/option")>()
  const useStoreMessageOption = (selector: any = (state: any) => state) => h1.enabled ? actual.useStoreMessageOption(selector) : selector(storeOptionState.value)
  useStoreMessageOption.getState = () => h1.enabled ? actual.useStoreMessageOption.getState() : storeOptionState.value
  useStoreMessageOption.setState = actual.useStoreMessageOption.setState
  return { useStoreMessageOption }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientState
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (selector: (state: typeof artifactsState.value) => unknown) =>
    selector(artifactsState.value)
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: string) => {
    if (setting === "chatWindowOpacity") return [35]
    if (setting === "chatMessageOpacity") return [60]
    if (setting === "chatCharacterImageOpacity") return [100]
    return [""]
  }
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.value.has(key) ? storageState.value.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => mobileViewportState.value,
  useDesktop: () => desktopViewportState.value
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: (params: ChatSettingsSyncParams) =>
    chatSettingsState.syncChatSettingsForServerChat(params),
  applyChatSettingsPatch: (params: ChatSettingsPatchParams) =>
    chatSettingsState.applyChatSettingsPatch(params)
}))

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => loadLocalConversationMock
}))

vi.mock("../playground-shortcuts", () => ({
  resolvePlaygroundShortcutAction: () => null
}))

vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: () => undefined
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => routerState.navigate,
    useLocation: () => routerState.hashRouter && window.location.hash.startsWith("#/") ? { ...(() => { const route = new URL(window.location.hash.slice(1), window.location.origin); return { pathname: route.pathname, search: route.search, hash: route.hash } })(), state: null, key: "hash-location" } : ({
      pathname: window.location.pathname || "/chat",
      search: window.location.search || "",
      hash: window.location.hash || "",
      state: null,
      key: "test-location"
    })
  }
})

describe("Playground thread search integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    h1.confirm.mockClear()
    h1.enabled = false
    forkPresentation.controller = null
    h1.legacyNative = false
    forkPresentation.mode = null
    forkPresentation.settings = null
    routerState.hashRouter = false
    storageState.value.clear()
    mobileViewportState.value = false
    desktopViewportState.value = true
    loadLocalConversationMock.mockClear()
    messageOptionState.value.historyId = "history-1"
    messageOptionState.value.serverChatId = "chat-1"
    messageOptionState.value.selectedModel = "model-1"
    messageOptionState.value.selectedSystemPrompt = "prompt-1"
    messageOptionState.value.selectedQuickPrompt = "quick-1"
    messageOptionState.value.chatMode = "normal"
    messageOptionState.value.webSearch = false
    messageOptionState.value.toolChoice = "none"
    messageOptionState.value.temporaryChat = false
    messageOptionState.value.useOCR = false
    messageOptionState.value.fileRetrievalEnabled = false
    messageOptionState.value.ragMediaIds = null
    artifactsState.value.isOpen = false
    artifactsState.value.active = null
    artifactsState.value.history = []
    artifactsState.value.unreadCount = 0
    storeOptionState.value.compareParentByHistory = {}
    storeOptionState.value.uploadedFiles = []
    storeOptionState.value.contextFiles = []
    storeOptionState.value.setUploadedFiles.mockImplementation((files) => {
      storeOptionState.value.uploadedFiles = files
    })
    storeOptionState.value.setContextFiles.mockImplementation((files) => {
      storeOptionState.value.contextFiles = files
    })
    tldwClientState.getDocumentUploadDraft.mockResolvedValue({ payload: {} })
    window.history.replaceState(null, "", "/chat")
  })

  it("does not import ambient attachment settings while a copied child is pending or verified", async () => {
    forkPresentation.mode = "pending"
    messageOptionState.value.serverChatId = "child"
    const mounted = render(<Playground />)
    await act(async () => {})
    expect(chatSettingsState.syncChatSettingsForServerChat).not.toHaveBeenCalled()
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled()
    forkPresentation.mode = "fork"
    forkPresentation.settings = {authorNote: "server"}
    mounted.rerender(<Playground />)
    await act(async () => {})
    expect(chatSettingsState.syncChatSettingsForServerChat).not.toHaveBeenCalled()
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled()
    forkPresentation.mode = "ordinary"
    forkPresentation.settings = null
    mounted.rerender(<Playground />)
    await waitFor(() => expect(chatSettingsState.syncChatSettingsForServerChat).toHaveBeenCalled())
  })

  it("restores ordinary attachments through a real owner-qualified legacy controller without accepting ancestry", async () => {
    h1.legacyNative = true
    const actual = renderHook(() => useHistorySelection())
    await act(async () => {await actual.result.current.open({kind: "native", owner_key: "local-owner", conversation_id: "chat-1", validate_lease: () => true} as any)})
    forkPresentation.controller = actual.result.current
    render(<Playground />)
    await waitFor(() => expect(chatSettingsState.syncChatSettingsForServerChat).toHaveBeenCalledWith({historyId: "history-1", serverChatId: "chat-1"}))
    expect(actual.result.current.status).toBe("legacy_review_required")
    expect(h1.confirm).not.toHaveBeenCalled()
  })

  it("forks timeline messages using stable IDs rather than rendered positions", async () => {
    render(<Playground />)
    act(() => window.dispatchEvent(new CustomEvent("tldw:timeline-action", {detail: {
      action: "branch", historyId: "history-1", messageId: "m-2"
    }})))
    await waitFor(() => expect(messageOptionState.value.createChatBranch).toHaveBeenCalledWith("m-2"))
  })

  it("opens in-thread search on Cmd/Ctrl+F and forwards query to PlaygroundChat", async () => {
    render(<Playground />)

    fireEvent.keyDown(window, { key: "f", ctrlKey: true })

    const input = screen.getByPlaceholderText(
      "Search messages in this conversation"
    ) as HTMLInputElement
    expect(input).toBeInTheDocument()

    fireEvent.change(input, { target: { value: "beta" } })

    expect(screen.getByTestId("playground-chat")).toHaveAttribute(
      "data-search-query",
      "beta"
    )
    await waitFor(() => {
      expect(screen.getByTestId("playground-chat")).toHaveAttribute(
        "data-search-count",
        "1"
      )
    })
  })

  it("opens shortcut help from the header and closes with Escape", () => {
    render(<Playground />)

    fireEvent.click(screen.getByTestId("playground-shortcuts-help-trigger"))
    expect(
      screen.getByTestId("playground-shortcuts-help-panel")
    ).toBeInTheDocument()

    fireEvent.keyDown(window, { key: "Escape" })
    expect(
      screen.queryByTestId("playground-shortcuts-help-panel")
    ).not.toBeInTheDocument()
  })

  it("opens shortcut help when a global open-shortcuts event is dispatched", async () => {
    render(<Playground />)

    window.dispatchEvent(new CustomEvent("tldw:open-playground-shortcuts"))
    await waitFor(() => {
      expect(
        screen.getByTestId("playground-shortcuts-help-panel")
      ).toBeInTheDocument()
    })
  })

  it("does not render the deprecated chat workflows header action", () => {
    render(<Playground />)

    expect(
      screen.queryByTestId("playground-chat-workflows-trigger")
    ).not.toBeInTheDocument()
    expect(routerState.navigate).not.toHaveBeenCalled()
  })

  it("shows a desktop right-edge artifacts expand button only when an artifact is active and the rail is closed", () => {
    artifactsState.value.active = artifactFixture
    artifactsState.value.isOpen = false

    render(<Playground />)

    expect(
      screen.getByRole("button", { name: "Expand artifacts rail" })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("artifacts-panel")).not.toBeInTheDocument()
  })

  it("does not show the right-edge artifacts expand button without an active artifact", () => {
    artifactsState.value.active = null
    artifactsState.value.isOpen = false

    render(<Playground />)

    expect(
      screen.queryByRole("button", { name: "Expand artifacts rail" })
    ).not.toBeInTheDocument()
  })

  it("opens artifacts from the right edge and marks them read", () => {
    artifactsState.value.active = artifactFixture
    artifactsState.value.isOpen = false

    render(<Playground />)
    fireEvent.click(screen.getByRole("button", { name: "Expand artifacts rail" }))

    expect(artifactsState.value.setOpen).toHaveBeenCalledWith(true)
    expect(artifactsState.value.markRead).toHaveBeenCalledTimes(1)
  })

  it("routes artifact focus events to the edge button when the rail is closed", async () => {
    artifactsState.value.active = artifactFixture
    artifactsState.value.isOpen = false

    render(<Playground />)
    const edgeButton = screen.getByRole("button", {
      name: "Expand artifacts rail"
    })

    window.dispatchEvent(new CustomEvent("tldw:focus-artifacts-trigger"))

    await waitFor(() => {
      expect(document.activeElement).toBe(edgeButton)
    })
  })

  it("shows mobile artifacts sheet context and returns focus to trigger when closing", async () => {
    mobileViewportState.value = true
    desktopViewportState.value = false
    storageState.value.set("playgroundChatLayoutMode", "cockpit")
    artifactsState.value.isOpen = true

    render(<Playground />)

    expect(
      screen.getByTestId("playground-mobile-artifacts-sheet")
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("playground-mobile-artifacts-title")
    ).toHaveTextContent("Artifacts panel")

    fireEvent.click(screen.getByTestId("playground-mobile-artifacts-return"))
    expect(artifactsState.value.closeArtifact).toHaveBeenCalledTimes(1)

    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByTestId("playground-artifacts-trigger")
      )
    })
  })

  it("shows branch fork context and returns to parent history in one action", () => {
    storeOptionState.value.compareParentByHistory = {
      "history-1": {
        parentHistoryId: "history-parent",
        clusterId: "cluster-a"
      },
      "history-parent": {
        parentHistoryId: "history-root"
      }
    }

    const openHistorySpy = vi.fn()
    const onOpenHistory = ((event: Event) => {
      openHistorySpy((event as CustomEvent).detail)
    }) as EventListener
    window.addEventListener("tldw:open-history", onOpenHistory)

    render(<Playground />)

    expect(screen.getByTestId("playground-branch-fork-point")).toHaveTextContent(
      "Fork point: cluster-a"
    )
    expect(screen.getByTestId("playground-branch-depth")).toHaveTextContent(
      "Depth 2"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Back to comparison chat" })
    )

    expect(openHistorySpy).toHaveBeenCalledWith({
      historyId: "history-parent"
    })

    window.removeEventListener("tldw:open-history", onOpenHistory)
  })

  it("restores sidepanel WebUI handoff state from the URL fragment and clears the fragment", async () => {
    messageOptionState.value.historyId = "history-1"
    messageOptionState.value.serverChatId = null
    messageOptionState.value.selectedSystemPrompt = "stale-system"
    messageOptionState.value.selectedQuickPrompt = "stale-quick"

    const encodedHandoff = encodeSidepanelChatWebUiHandoff({
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: Date.now(),
      draft: "continue from the sidepanel",
      historyId: "history-handoff",
      serverChatId: "server-handoff",
      selectedSystemPrompt: null,
      selectedQuickPrompt: "",
      chatMode: "rag",
      webSearch: true,
      toolChoice: "auto",
      temporaryChat: true,
      useOCR: true,
      ragMediaIds: [101, 202],
      fileRetrievalEnabled: true
    })
    const hashParams = new URLSearchParams()
    hashParams.set(SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM, encodedHandoff)
    window.history.replaceState(null, "", `/chat#${hashParams.toString()}`)
    expect(decodeSidepanelChatWebUiHandoff(encodedHandoff)).toMatchObject({
      historyId: "history-handoff"
    })

    render(<Playground />)

    await waitFor(
      () => {
        expect(loadLocalConversationMock).toHaveBeenCalledWith(
          "history-handoff"
        )
      },
      { timeout: 5_000 }
    )
    await waitFor(() => {
      expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(
        "server-handoff"
      )
    })
    expect(
      messageOptionState.value.setSelectedSystemPrompt
    ).toHaveBeenCalledWith(null)
    expect(messageOptionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(
      null
    )
    expect(messageOptionState.value.setChatMode).toHaveBeenCalledWith("rag")
    expect(messageOptionState.value.setWebSearch).toHaveBeenCalledWith(true)
    expect(messageOptionState.value.setToolChoice).toHaveBeenCalledWith("auto")
    expect(messageOptionState.value.setTemporaryChat).toHaveBeenCalledWith(true)
    expect(messageOptionState.value.setUseOCR).toHaveBeenCalledWith(true)
    expect(messageOptionState.value.setRagMediaIds).toHaveBeenCalledWith([
      101,
      202
    ])
    expect(
      messageOptionState.value.setFileRetrievalEnabled
    ).toHaveBeenCalledWith(true)
    expect(window.location.hash).toBe("")
  })

  it("imports sidepanel document draft files into chat attachments", async () => {
    const existingFile = {
      id: "existing-file",
      filename: "existing.md",
      type: "text/markdown",
      content: "existing",
      size: 8,
      uploadedAt: 1,
      processed: false,
      processingMode: "add_to_chat"
    }
    const draftFile = {
      id: "draft-file",
      filename: "draft.pdf",
      type: "application/pdf",
      content: "draft",
      size: 16,
      uploadedAt: 2,
      processed: false,
      processingMode: "add_to_chat"
    }
    storeOptionState.value.uploadedFiles = [existingFile]
    storeOptionState.value.contextFiles = [existingFile]
    tldwClientState.getDocumentUploadDraft.mockResolvedValueOnce({
      payload: { files: [draftFile] }
    })

    const encodedHandoff = encodeSidepanelChatWebUiHandoff({
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: Date.now(),
      chatDocumentDraftId: "document-draft-1"
    })
    const hashParams = new URLSearchParams()
    hashParams.set(SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM, encodedHandoff)
    window.history.replaceState(null, "", `/chat#${hashParams.toString()}`)

    render(<Playground />)

    await waitFor(() => {
      expect(tldwClientState.getDocumentUploadDraft).toHaveBeenCalledWith(
        "document-draft-1"
      )
    })
    expect(storeOptionState.value.setUploadedFiles).toHaveBeenCalledWith([
      existingFile,
      draftFile
    ])
    expect(messageOptionState.value.setContextFiles).toHaveBeenCalledWith([
      existingFile,
      draftFile
    ])
    expect(tldwClientState.deleteDocumentUploadDraft).toHaveBeenCalledWith(
      "document-draft-1"
    )
    expect(messageOptionState.value.setRagMediaIds).not.toHaveBeenCalled()
  })
})

vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: async () => "profile",
  getLocalHistoryOwner: async (id: string) => ({
    kind: "local",
    profile_id: "profile",
    owner_key: "local-owner",
    conversation_id: id
  }),
  loadHistoryBookmark: async (scope: any, owner: any) =>
    h1.bookmarks.get(h1Key(scope, owner)) || null,
  saveHistoryBookmark: async (scope: any, view: any) =>
    h1.bookmarks.set(h1Key(scope, view), { ...scope, view })
}))
vi.mock("@/db/dexie/fork-operations", () => ({findForkCandidate: async () => null, loadForkOperations: async () => []}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getHistoryInfo = async () => null
  }
}))
vi.mock("@/services/chat-history-selection", () => ({
  confirmLegacyHistoryProjection: (...args: any[]) => h1.confirm(...args),
  captureHistorySnapshot: async (owner: any, view: any) => {
    const bound = { ...view, owner_key: "local-owner" }
    const snapshot: any = {
      version: 1,
      owner_key: "local-owner",
      conversation_id: owner.conversation_id,
      source_digest: "source",
      storage_context_digest: "storage",
      fences: {},
      interpretation_status: { kind: "parent_graph_v1" },
      nodes: ["A", "B"].map((id) => ({
        id,
        revision: "1",
        role: "assistant",
        parent_id: null,
        settled: true,
        preview: owner.conversation_id + " answer " + id
      }))
    }
    if (h1.legacyNative) return {status: "legacy_review_required", code: "legacy_review_required", snapshot, view: bound}
    const result = resolveHistorySelection(snapshot, bound, "send", "")
    if (result.status !== "ready") return { ...result, snapshot, view: bound }
    return {
      status: "captured",
      snapshot,
      view: bound,
      rows: result.rows,
      selected_content: result.rows.map((row) => ({
        id: row.id,
        revision: row.revision,
        message: row.preview,
        images: []
      })),
      storage_context_digest: "storage",
      purpose: "send"
    }
  }
}))

describe("Playground H1 URL initialization with the mounted session/controller", () => {
  beforeEach(() => {
    routerState.hashRouter = false
    h1.enabled = true
    forkPresentation.controller = null
    h1.legacyNative = false
    forkPresentation.mode = null
    forkPresentation.settings = null
    h1.controller = null
    h1.bookmarks.clear()
    h1.confirm.mockClear()
    localStorage.clear()
    sessionStorage.clear()
    storageState.value.clear()
    window.history.replaceState(null, "", "/chat")
    usePlaygroundSessionStore.getState().clearSession()
    useStoreMessageOption.setState({
      historyId: null,
      serverChatId: null,
      history: [],
      messages: [],
      temporaryChat: false,
      queuedMessages: [],
      compareMode: false,
      compareSelectedModels: [],
      contextFiles: [],
      serverChatAssistantId: null,
      serverChatAssistantKind: null,
      serverChatCharacterId: null,
      serverChatMetaLoaded: false
    })
  })
  function seed(pending = false) {
    const reference = {
      profile_id: "profile",
      client_session_id: "frozen-source",
      owner_key: "local-owner",
      conversation_id: "chat-one",
      owner_kind: "local"
    }
    const view = {
      owner_key: reference.owner_key,
      conversation_id: reference.conversation_id,
      view_session_id: "source-view",
      selection_revision: 3,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "A" }
    }
    const record: any = { ...reference, view }
    if (pending)
      Object.assign(record, {
        pending_confirmation: {
          version: 1,
          projection_id: "pending-original",
          selection_revision: 3,
          ordered_path_ids: ["A"],
          cursor: view.cursor
        },
        pending_view_session_id: "source-view",
        pending_dispatch_started: true
      })
    h1.bookmarks.set(h1Key(reference, view), record)
    return reference
  }
  it.each(["query", "hash", "hash-router"])(
    "consumes a %s handoff once and restores destination choice, conversation switch and empty reset",
    async (route) => {
      routerState.hashRouter = route === "hash-router"
      const reference = seed()
      const query =
        "keep=value&historySelection=" +
        encodeURIComponent(JSON.stringify(reference))
      window.history.replaceState(
        { preserved: true },
        "",
        route === "query"
          ? "/chat?" + query + "#anchor"
          : "/options.html?outer=value#/chat?" + query + "#anchor"
      )
      let page = render(<Playground />)
      await waitFor(() =>
        expect(screen.getByTestId("playground-chat")).toHaveAttribute(
          "data-selected-history",
          "chat-one answer A"
        )
      )
      await act(async () => {
        await h1.controller!.choose({ kind: "after_message", message_id: "B" })
      })
      page.unmount()
      page = render(<Playground />)
      await waitFor(() =>
        expect(screen.getByTestId("playground-chat")).toHaveAttribute(
          "data-selected-history",
          "chat-one answer B"
        )
      )
      expect(window.location.href).not.toContain("historySelection")
      expect(window.location.href).toContain("keep=value")
      if (route === "query") expect(window.location.hash).toBe("#anchor")
      else {
        expect(window.location.search).toBe("?outer=value")
        expect(window.location.hash).toContain("#anchor")
      }
      expect(window.history.state).toEqual({ preserved: true })
      await act(async () => {
        await h1.controller!.loadConversation({ historyId: "chat-two" }, null)
      })
      page.unmount()
      page = render(<Playground />)
      await waitFor(() =>
        expect(h1.controller?.view?.conversation_id).toBe("chat-two")
      )
      await act(async () => {
        h1.controller!.reset()
        usePlaygroundSessionStore.getState().clearSession()
        useStoreMessageOption.setState({
          historyId: null,
          serverChatId: null,
          messages: [],
          history: []
        })
      })
      page.unmount()
      page = render(<Playground />)
      await waitFor(() => expect(h1.controller?.status).toBe("idle"))
      expect(useStoreMessageOption.getState().messages).toEqual([])
      page.unmount()
      // A different destination URL still represents a new initialization.
      window.history.replaceState(null, "", "/chat?" + query)
      page = render(<Playground />)
      await waitFor(() =>
        expect(h1.controller?.view?.conversation_id).toBe("chat-one")
      )
      page.unmount()
    }
  )
  it("retains the exact pending origin after consuming its URL and reloading", async () => {
    const reference = seed(true)
    window.history.replaceState(
      null,
      "",
      "/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference))
    )
    let page = render(<Playground />)
    await waitFor(() => expect(h1.controller?.status).toBe("pending_unknown"))
    page.unmount()
    page = render(<Playground />)
    await waitFor(() =>
      expect(h1.controller?.pending?.intent.projection_id).toBe(
        "pending-original"
      )
    )
    expect(window.location.search).toBe("")
    expect(h1.controller?.pending?.scope.client_session_id).toBe(
      "frozen-source"
    )
    expect(h1.controller?.pending?.view.view_session_id).toBe("source-view")
    expect(
      h1.bookmarks.get(h1Key(reference, reference)).pending_dispatch_started
    ).toBe(true)
    expect(h1.confirm).not.toHaveBeenCalled()
    page.unmount()
  })
})
