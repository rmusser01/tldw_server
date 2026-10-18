import "./native-plasmo-storage-fixture"
// @vitest-environment jsdom
import React from "react"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useOriginalSdkSceneFixture } from "./sdk-original-scene-fixture"

const {
  createChatMock,
  normalChatModeMock,
  streamCharacterChatCompletionMock,
  syncChatSettingsForServerChatMock,
  saveLocalSuccessMock
} =
  vi.hoisted(() => ({
    createChatMock: vi.fn(),
    normalChatModeMock: vi.fn(),
    streamCharacterChatCompletionMock: vi.fn(),
    syncChatSettingsForServerChatMock: vi.fn(async () => null),
    saveLocalSuccessMock: vi.fn()
  }))

vi.mock("@/hooks/chat-modes/normalChatMode", async original => ({
  ...await original<typeof import("@/hooks/chat-modes/normalChatMode")>(),
  normalChatMode: normalChatModeMock
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    definitions: {}, scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: entryAccount.scopeInvalidatedSignal,
    requestScope: { config: { serverUrl: "https://chat.test", authMode: "single-user" }, userId: null },
    release: vi.fn()
  })
}))

vi.mock("@/hooks/chat-modes/continueChatMode", () => ({
  continueChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/ragMode", () => ({
  ragMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/tabChatMode", () => ({
  tabChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/documentChatMode", () => ({
  documentChatMode: vi.fn()
}))

vi.mock("@/hooks/utils/messageHelpers", () => ({
  validateBeforeSubmit: vi.fn(() => true),
  createSaveMessageOnSuccess: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-overlay"
  ),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-overlay"
  )
}))

vi.mock("@/hooks/chat-helper", async original => ({
  ...await original<typeof import("@/hooks/chat-helper")>(),
  saveMessageOnSuccess: saveLocalSuccessMock
}))

vi.mock("@/hooks/handlers/messageHandlers", () => ({
  createNativeDeleteMessage: vi.fn(() => vi.fn(async () => false)),
  createRegenerateLastMessage: vi.fn(() => vi.fn()),
  createEditMessage: vi.fn(() => vi.fn()),
  createStopStreamingRequest: vi.fn(() => vi.fn()),
  createBranchMessage: vi.fn(() => vi.fn())
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: vi.fn(() => "generated-id"),
  saveHistory: vi.fn(),
  saveMessage: vi.fn(),
  updateHistory: vi.fn(),
  updateMessage: vi.fn(),
  updateMessageMedia: vi.fn(async () => null),
  removeMessageByIndex: vi.fn(),
  formatToChatHistory: vi.fn((items: unknown) => items),
  formatToMessage: vi.fn((items: unknown) => items),
  getSessionFiles: vi.fn(async () => []),
  getPromptById: vi.fn(async () => null)
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => null)
}))

vi.mock("@/db/dexie/branch", () => ({
  generateBranchFromMessageIds: vi.fn(async () => null)
}))

vi.mock("@/services/actor-settings", async original => ({
  ...await original<typeof import("@/services/actor-settings")>()
}))

vi.mock("@/utils/selected-character-storage", () => ({
  SELECTED_CHARACTER_STORAGE_KEY: "selected_character",
  selectedCharacterStorage: {
    get: vi.fn(async () => null),
    set: vi.fn(async () => null)
  },
  selectedCharacterSyncStorage: {
    get: vi.fn(async () => null)
  },
  parseSelectedCharacterValue: vi.fn(() => null)
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: {},
    updateSettings: vi.fn()
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => {
    const [value] = React.useState(defaultValue)
    return [value, vi.fn()] as const
  }
}))

vi.mock("@/store/option", async original => {
  const actual = await original<typeof import("@/store/option")>()
  return {
    ...actual,
    useStoreMessageOption: {
      getState: actual.useStoreMessageOption.getState,
      setState: actual.useStoreMessageOption.setState
    }
  }
})

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: syncChatSettingsForServerChatMock
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    initialize: vi.fn(async () => null)
  }
}))

const createHookOptions = () => ({
  t: (_key: string, fallback?: string) => fallback || _key,
  notification: {
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  },
  abortController: null,
  setAbortController: vi.fn(),
  messages: [],
  setMessages: vi.fn(),
  history: [],
  setHistory: vi.fn(),
  historyId: "history-overlay",
  setHistoryId: vi.fn(),
  temporaryChat: false,
  selectedModel: "deepseek-chat",
  useOCR: false,
  selectedSystemPrompt: null,
  selectedKnowledge: null,
  toolChoice: "auto" as const,
  webSearch: false,
  currentChatModelSettings: {
    apiProvider: "openai",
    setSystemPrompt: vi.fn()
  },
  setIsSearchingInternet: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setActionInfo: vi.fn(),
  fileRetrievalEnabled: false,
  ragMediaIds: null,
  ragSearchMode: "hybrid" as const,
  ragTopK: 8,
  ragEnableGeneration: true,
  ragEnableCitations: true,
  ragSources: [],
  ragAdvancedOptions: {},
  serverChatId: null,
  serverChatTitle: null,
  serverChatCharacterId: null,
  serverChatAssistantKind: null,
  serverChatAssistantId: null,
  serverChatPersonaMemoryMode: null,
  serverChatMetaLoaded: false,
  serverChatState: "in-progress" as const,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  setServerChatId: vi.fn(),
  setServerChatTitle: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatAssistantKind: vi.fn(),
  setServerChatAssistantId: vi.fn(),
  setServerChatPersonaMemoryMode: vi.fn(),
  setServerChatMetaLoaded: vi.fn(),
  setServerChatState: vi.fn(),
  setServerChatVersion: vi.fn(),
  setServerChatTopic: vi.fn(),
  setServerChatClusterId: vi.fn(),
  setServerChatSource: vi.fn(),
  setServerChatExternalRef: vi.fn(),
  ensureServerChatHistoryId: vi.fn(async () => "history-overlay"),
  contextFiles: [],
  setContextFiles: vi.fn(),
  documentContext: null,
  setDocumentContext: vi.fn(),
  uploadedFiles: [],
  compareModeActive: false,
  compareSelectedModels: [],
  compareMaxModels: 3,
  compareFeatureEnabled: false,
  markCompareHistoryCreated: vi.fn(),
  replyTarget: null,
  clearReplyTarget: vi.fn(),
  messageSteeringPrompts: null,
  setSelectedQuickPrompt: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  invalidateServerChatHistory: vi.fn(),
  selectedCharacter: {
    id: "char-overlay",
    name: "Overlay Guide",
    system_prompt: "Overlay guide prompt"
  },
  selectedAssistant: {
    kind: "character" as const,
    id: "char-overlay",
    name: "Overlay Guide",
    system_prompt: "Overlay guide prompt",
    metadata: {
      selectionMode: "overlay"
    }
  },
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

const entryAccount = { requestScope: { config: { serverUrl: "https://chat.test", authMode: "single-user" as const }, userId: null }, scopeInvalidatedSignal: new AbortController().signal }
const useChatActions = (options: Parameters<typeof useOriginalSdkSceneFixture>[0]) =>
  useOriginalSdkSceneFixture(options, () => entryAccount)

describe("useChatActions overlay integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    createChatMock.mockResolvedValue({ id: "server-chat-42", title: "Overlay hello" })
    normalChatModeMock.mockResolvedValue(undefined)
    streamCharacterChatCompletionMock.mockImplementation(async function* () {
      yield {
        choices: [
          {
            delta: {
              content: "Tracked reply"
            }
          }
        ]
      }
    })
  })

  it("routes overlay character selections through normal chat mode without creating tracked chats", async () => {
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Overlay hello",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.not.objectContaining({ character_id: expect.anything() }),
      expect.objectContaining({ requestScope: expect.anything() })
    )
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).toHaveBeenCalledTimes(1)
    expect(normalChatModeMock.mock.calls[0]?.[6]).toEqual(
      expect.objectContaining({
        assistantIdentity: {
          name: "Overlay Guide",
          avatarUrl: undefined
        },
        overlaySystemPrompt: "Overlay guide prompt"
      })
    )
  })

  it.each(["current", "new", "same-id", "unmount"] as const)(
    "keeps a held overlay save attached to its rendered source across %s",
    async (boundary) => {
      const store = (await import("@/store/option")).useStoreMessageOption
      const { createSaveMessageOnSuccess } = await import(
        "@/hooks/utils/messageHelpers"
      )
      const previous = store.getState()
      previous.startNewChatDraft()
      const options = createHookOptions()
      options.ensureServerChatHistoryId.mockImplementationOnce(async (_id, _title, _signal, _snapshot, scene) => {
        expect(scene.acceptHistoryMirror("saved-overlay")).toBe(true)
        store.getState().setHistoryId("saved-overlay", { preserveServerChatId: true })
        return "saved-overlay"
      })
      let finishPersistence!: () => void
      const held = new Promise<void>((resolve) => {
        finishPersistence = resolve
      })
      const persist = vi.fn(
        async (payload: { setHistoryId: (id: string, options?: { preserveServerChatId: boolean }) => void }) => {
          await held
          payload.setHistoryId("saved-overlay", { preserveServerChatId: true })
          return "saved-overlay"
        }
      )
      vi.mocked(createSaveMessageOnSuccess).mockReturnValueOnce(persist)
      normalChatModeMock.mockImplementationOnce(
        async (...args: Parameters<typeof normalChatModeMock>) => {
          await args[6].saveMessageOnSuccess({
            requestScope: args[6].servicePromptSnapshot?.requestScope,
            scopeSignal: args[6].servicePromptSnapshot?.scopeSignal,
            scopeInvalidatedSignal: args[6].servicePromptSnapshot?.scopeInvalidatedSignal,
            historyId: null,
            setHistoryId: args[6].setHistoryId,
            conversationId: "server-chat-42",
            saveToDb: true,
            message: "Overlay request",
            fullText: "Overlay reply"
          })
        }
      )
      const hook = renderHook(() =>
        useChatActions(options as Parameters<typeof useChatActions>[0])
      )
      const submission = hook.result.current.onSubmit({
        message: "Overlay request",
        image: ""
      })
      try {
        await waitFor(() => expect(persist).toHaveBeenCalledOnce())
        if (boundary === "new" || boundary === "same-id") {
          store.getState().startNewChatDraft()
          if (boundary === "same-id")
            store.setState({
              historyId: "saved-overlay",
              serverChatId: "server-chat-42"
            })
        }
        if (boundary === "unmount") hook.unmount()
        options.setHistoryId.mockClear()
        options.setServerChatId.mockClear()
        syncChatSettingsForServerChatMock.mockClear()
        await act(async () => {
          finishPersistence()
          await submission
        })
        if (boundary === "current") {
          expect(options.setHistoryId).toHaveBeenCalledWith("saved-overlay", { preserveServerChatId: true })
          expect(options.setServerChatId).not.toHaveBeenCalled()
          expect(store.getState().serverChatId).toBe("server-chat-42")
          expect(syncChatSettingsForServerChatMock).toHaveBeenCalledOnce()
        } else {
          expect(options.setHistoryId).not.toHaveBeenCalled()
          expect(options.setServerChatId).not.toHaveBeenCalled()
          expect(syncChatSettingsForServerChatMock).not.toHaveBeenCalled()
        }
        expect(persist).toHaveBeenCalledOnce()
      } finally {
        finishPersistence()
        await submission
        hook.unmount()
        store.setState(previous)
      }
    }
  )

  it.each([false, true])(
    "preserves actual server metadata at first local-ID acknowledgement (retired=%s)",
    async (retired) => {
      const store = (await import("@/store/option")).useStoreMessageOption
      const previous = store.getState()
      previous.startNewChatDraft()
      store.setState({
        historyId: null,
        serverChatId: "server-chat-42",
        serverChatTitle: "Saved identity",
        serverChatCharacterId: "saved-character",
        serverChatVersion: 7,
        serverChatMetaLoaded: true
      })
      const { createSaveMessageOnSuccess } = await import(
        "@/hooks/utils/messageHelpers"
      )
      const actual = await vi.importActual<
        typeof import("@/hooks/utils/messageHelpers")
      >("@/hooks/utils/messageHelpers")
      vi.mocked(createSaveMessageOnSuccess).mockImplementationOnce(
        actual.createSaveMessageOnSuccess
      )
      let finishPersistence!: () => void
      const held = new Promise<void>((resolve) => {
        finishPersistence = resolve
      })
      saveLocalSuccessMock.mockImplementationOnce(async (payload) => {
        await held
        payload.setHistoryId("first-local-history")
        return "first-local-history"
      })
      const options = createHookOptions()
      Object.assign(options, {
        historyId: null,
        serverChatId: "server-chat-42",
        serverChatCharacterId: "saved-character",
        setHistoryId: store.getState().setHistoryId
      })
      options.ensureServerChatHistoryId.mockImplementationOnce(async (_id, _title, _signal, _snapshot, scene) => {
        expect(scene.acceptHistoryMirror("first-local-history")).toBe(true)
        store.getState().setHistoryId("first-local-history", { preserveServerChatId: true })
        return "first-local-history"
      })
      normalChatModeMock.mockImplementationOnce(
        async (...args: Parameters<typeof normalChatModeMock>) => {
          await args[6].saveMessageOnSuccess({
            requestScope: args[6].servicePromptSnapshot?.requestScope,
            scopeSignal: args[6].servicePromptSnapshot?.scopeSignal,
            scopeInvalidatedSignal: args[6].servicePromptSnapshot?.scopeInvalidatedSignal,
            historyId: null,
            setHistoryId: args[6].setHistoryId,
            conversationId: "server-chat-42",
            message: "Current question",
            fullText: "Current reply"
          })
        }
      )
      const hook = renderHook(() =>
        useChatActions(options as Parameters<typeof useChatActions>[0])
      )
      const submission = hook.result.current.onSubmit({
        message: "Current question",
        image: ""
      })
      try {
        await waitFor(() => expect(saveLocalSuccessMock).toHaveBeenCalledOnce())
        if (retired) {
          store.getState().startNewChatDraft()
          store.setState({ historyId: "newer-local-history" })
        }
        await act(async () => {
          finishPersistence()
          await submission
        })
        expect(store.getState()).toMatchObject({
          historyId: retired ? "newer-local-history" : "first-local-history",
          serverChatId: "server-chat-42",
          serverChatTitle: "Saved identity",
          serverChatCharacterId: "saved-character",
          serverChatVersion: 7,
          serverChatMetaLoaded: true
        })
      } finally {
        finishPersistence()
        await submission
        hook.unmount()
        store.setState(previous)
      }
    }
  )

  it("reconciles scratch overlay settings only when a send links a new server conversation", async () => {
    const options = createHookOptions()
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        servicePromptSnapshot?: ServicePromptSnapshot
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
      }
      await params.saveMessageOnSuccess({
        requestScope: params.servicePromptSnapshot?.requestScope,
        scopeSignal: params.servicePromptSnapshot?.scopeSignal,
        scopeInvalidatedSignal: params.servicePromptSnapshot?.scopeInvalidatedSignal,
        historyId: null,
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Overlay hello",
        image: "",
        fullText: "Overlay reply",
        source: [],
        conversationId: "server-chat-42"
      })
    })

    const { result } = renderHook(() => useChatActions(options as Parameters<typeof useChatActions>[0]))

    await act(async () => {
      await result.current.onSubmit({
        message: "Overlay hello",
        image: ""
      })
    })

    const store = (await import("@/store/option")).useStoreMessageOption
    const source = store.getState()
    expect(syncChatSettingsForServerChatMock).toHaveBeenCalledWith({
      historyId: "history-overlay",
      serverChatId: "server-chat-42",
      allowScratchFallback: true,
      requestScope: entryAccount.requestScope,
      signal: normalChatModeMock.mock.calls[0][6].servicePromptSnapshot.scopeSignal,
      sourceDraftAddress: source.chatDraftAddress,
      validate_lease: expect.any(Function)
    })
    const lease = syncChatSettingsForServerChatMock.mock.calls[0][0].validate_lease
    expect(lease()).toBe(true)
    try {
      source.startNewChatDraft()
      expect(lease()).toBe(false)
    } finally {
      store.setState(source)
    }
  })

})
