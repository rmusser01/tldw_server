// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"

const {
  createChatMock,
  normalChatModeMock,
  streamCharacterChatCompletionMock,
  baseSaveMessageOnSuccessMock,
  syncChatSettingsForServerChatMock,
  getConfigMock,
  savePlaygroundSessionMock,
  buildChatSurfaceScopeKeyFromConfigMock
} = vi.hoisted(() => ({
  createChatMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  baseSaveMessageOnSuccessMock: vi.fn(
    async (_payload?: unknown): Promise<string | null> => "history-persona"
  ),
  syncChatSettingsForServerChatMock: vi.fn(async () => null),
  getConfigMock: vi.fn(),
  savePlaygroundSessionMock: vi.fn(),
  buildChatSurfaceScopeKeyFromConfigMock: vi.fn()
}))

vi.mock("@/hooks/chat-modes/normalChatMode", () => ({
  normalChatMode: normalChatModeMock
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
  createSaveMessageOnSuccess: vi.fn(() => baseSaveMessageOnSuccessMock),
  createSaveMessageOnError: vi.fn(
    () =>
      async (_payload?: unknown): Promise<string | null> =>
        "history-persona"
  )
}))

vi.mock("@/hooks/handlers/messageHandlers", () => ({
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

vi.mock("@/services/actor-settings", () => ({
  getActorSettingsForChat: vi.fn(async () => null)
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

vi.mock("@/store/option", () => ({
  useStoreMessageOption: {
    getState: () => ({ selectedModel: "deepseek-chat" as string | null })
  }
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: syncChatSettingsForServerChatMock
}))

vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: buildChatSurfaceScopeKeyFromConfigMock
}))

vi.mock("@/store/playground-session", () => ({
  usePlaygroundSessionStore: {
    getState: () => ({
      saveSession: savePlaygroundSessionMock
    })
  }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    initialize: vi.fn(async () => null),
    getConfig: getConfigMock
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
  historyId: "history-persona",
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
  ensureServerChatHistoryId: vi.fn(async () => "history-persona"),
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
  selectedCharacter: null,
  selectedAssistant: {
    kind: "persona" as const,
    id: "garden-helper",
    name: "Garden Helper",
    metadata: {
      selectionMode: "tracked"
    }
  },
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

describe("useChatActions persona integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    syncChatSettingsForServerChatMock.mockResolvedValue(null)
    getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    buildChatSurfaceScopeKeyFromConfigMock.mockReturnValue("scope:chat")
    createChatMock.mockResolvedValue({
      id: "persona-chat-1",
      title: "Persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only"
    })
    normalChatModeMock.mockResolvedValue(undefined)
  })

  it("creates a persona-backed chat with assistant_kind=persona", async () => {
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello persona",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith({
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only",
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    })
    expect(options.setServerChatId).toHaveBeenCalledWith("persona-chat-1")
    expect(options.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith("persona")
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith("garden-helper")
    expect(options.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(
      "read_only"
    )
    expect(
      options.setServerChatId.mock.invocationCallOrder[0]
    ).toBeLessThan(
      options.setServerChatAssistantKind.mock.invocationCallOrder.at(-1) ?? 0
    )
    expect(options.setServerChatMetaLoaded).toHaveBeenCalledWith(true)
    expect(savePlaygroundSessionMock).toHaveBeenCalledWith(
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "persona-chat-1",
        trackedAssistantKind: "persona",
        trackedAssistantId: "garden-helper",
        trackedCharacterId: null,
        trackedAssistantDisplayName: "Garden Helper",
        trackedAssistantAvatarUrl: null,
        serverChatPersonaMemoryMode: "read_only",
        scopeKey: "scope:chat",
        trackedAssistantSelection: expect.objectContaining({
          kind: "persona",
          id: "garden-helper",
          name: "Garden Helper",
          metadata: expect.objectContaining({
            selectionMode: "tracked"
          })
        })
      })
    )
    expect(
      savePlaygroundSessionMock.mock.invocationCallOrder[0]
    ).toBeLessThan(normalChatModeMock.mock.invocationCallOrder[0])
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Hello persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Garden Helper",
          avatarUrl: undefined
        },
        historyId: "history-persona",
        serverChatId: "persona-chat-1"
      })
    )
  })

  it("does not forward stale character state into the first persona send", async () => {
    const options = {
      ...createHookOptions(),
      serverChatId: "character-chat-1",
      serverChatTitle: "Old character chat",
      serverChatCharacterId: 42,
      serverChatAssistantKind: "character" as const,
      serverChatAssistantId: "42",
      serverChatPersonaMemoryMode: "read_write" as const,
      serverChatMetaLoaded: true,
      serverChatState: "resolved" as const,
      serverChatTopic: "Old topic",
      serverChatClusterId: "old-cluster",
      serverChatSource: "old-source",
      serverChatExternalRef: "old-ref"
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello persona",
        image: ""
      })
    })

    expect(options.setServerChatId).toHaveBeenCalledWith(null)
    expect(options.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith(null)
    expect(options.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(null)
    expect(options.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(options.setServerChatTitle).toHaveBeenCalledWith(null)
    expect(options.setServerChatState).toHaveBeenCalledWith("in-progress")
    expect(options.setServerChatVersion).toHaveBeenCalledWith(null)
    expect(options.setServerChatTopic).toHaveBeenCalledWith(null)
    expect(options.setServerChatClusterId).toHaveBeenCalledWith(null)
    expect(options.setServerChatSource).toHaveBeenCalledWith(null)
    expect(options.setServerChatExternalRef).toHaveBeenCalledWith(null)
    expect(createChatMock).toHaveBeenCalledWith({
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only",
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    })
    expect(options.setServerChatAssistantKind).toHaveBeenLastCalledWith("persona")
    expect(options.setServerChatAssistantId).toHaveBeenLastCalledWith(
      "garden-helper"
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Hello persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Garden Helper",
          avatarUrl: undefined
        },
        serverChatId: "persona-chat-1"
      })
    )
  })

  it("keeps persona routing ahead of character fallback when persona chats carry a character id", async () => {
    const options = {
      ...createHookOptions(),
      serverChatId: "persona-chat-existing",
      serverChatAssistantKind: "persona" as const,
      serverChatAssistantId: "garden-helper",
      serverChatCharacterId: "char-shadow",
      selectedCharacter: {
        id: "char-stale",
        name: "Stale Character",
        system_prompt: "Stale prompt"
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Stay persona",
        image: ""
      })
    })

    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: expect.anything()
      })
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Stay persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Garden Helper",
          avatarUrl: undefined
        },
        serverChatId: "persona-chat-existing"
      })
    )
  })

  it("preserves tracked persona server linkage when local history is assigned", async () => {
    const setHistoryId = vi.fn()
    const options = {
      ...createHookOptions(),
      historyId: null,
      setHistoryId,
      serverChatId: null
    }
    baseSaveMessageOnSuccessMock.mockImplementationOnce(
      async (payload?: { setHistoryId?: (id: string) => void }) => {
        payload?.setHistoryId?.("history-persona")
        return "history-persona"
      }
    )
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
      }
      await params.saveMessageOnSuccess({
        historyId: null,
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Hello persona",
        image: "",
        fullText: "Persona reply",
        source: [],
        assistantMessageId: "assistant-persona-1",
        reasoning_time_taken: 0,
        conversationId: "persona-chat-1"
      })
    })

    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello persona",
        image: ""
      })
    })

    expect(setHistoryId).toHaveBeenCalledWith("history-persona", {
      preserveServerChatId: true
    })
  })
})
