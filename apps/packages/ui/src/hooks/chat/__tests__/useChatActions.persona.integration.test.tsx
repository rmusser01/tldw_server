// @vitest-environment jsdom
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"

const {
  createChatMock,
  getChatMock,
  addChatMessageMock,
  getServerCapabilitiesMock,
  pageAssistModelMock,
  normalChatModeMock,
  ragModeMock,
  streamCharacterChatCompletionMock,
  baseSaveMessageOnSuccessMock,
  syncChatSettingsForServerChatMock,
  getConfigMock,
  savePlaygroundSessionMock,
  buildChatSurfaceScopeKeyFromConfigMock,
  loadServicePromptSnapshotMock,
  releaseServicePromptSnapshotMock
} = vi.hoisted(() => ({
  createChatMock: vi.fn(),
  getChatMock: vi.fn(),
  addChatMessageMock: vi.fn(),
  getServerCapabilitiesMock: vi.fn(),
  pageAssistModelMock: vi.fn(),
  normalChatModeMock: vi.fn(),
  ragModeMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  baseSaveMessageOnSuccessMock: vi.fn(
    async (_payload?: unknown): Promise<string | null> => "history-persona"
  ),
  syncChatSettingsForServerChatMock: vi.fn(async () => null),
  getConfigMock: vi.fn(),
  savePlaygroundSessionMock: vi.fn(),
  buildChatSurfaceScopeKeyFromConfigMock: vi.fn(),
  loadServicePromptSnapshotMock: vi.fn(),
  releaseServicePromptSnapshotMock: vi.fn()
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: loadServicePromptSnapshotMock
}))

vi.mock("@/hooks/chat-modes/normalChatMode", () => ({
  normalChatMode: normalChatModeMock
}))

vi.mock("@/hooks/chat-modes/continueChatMode", () => ({
  continueChatMode: vi.fn()
}))

vi.mock("@/hooks/chat-modes/ragMode", () => ({
  ragMode: ragModeMock
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
  getServerCapabilities: getServerCapabilitiesMock
}))

vi.mock("@/models", () => ({ pageAssistModel: pageAssistModelMock }))

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
    getChat: getChatMock,
    addChatMessage: addChatMessageMock,
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

const servicePromptSnapshot = {
  scopeKey: "scope:captured-user",
  requestScope: {
    config: {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user" as const,
      authSource: "manual" as const
    },
    userId: 42
  },
  capability: "supported" as const,
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  definitions: {
    "chat.rag.answer": {
      definition: { id: "chat.rag.answer", parts: [] },
      parts: { template: "Answer" },
      source: "packaged" as const,
      revision: null
    },
    "chat.rag.question_rewrite": {
      definition: { id: "chat.rag.question_rewrite", parts: [] },
      parts: { template: "Rewrite" },
      source: "packaged" as const,
      revision: null
    },
    "chat.web_search.answer": {
      definition: { id: "chat.web_search.answer", parts: [] },
      parts: { template: "Web answer" },
      source: "packaged" as const,
      revision: null
    }
  },
  release: releaseServicePromptSnapshotMock
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe("useChatActions persona integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getServerCapabilitiesMock.mockResolvedValue({ hasChatSaveToDb: false })
    syncChatSettingsForServerChatMock.mockResolvedValue(null)
    getConfigMock.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
    buildChatSurfaceScopeKeyFromConfigMock.mockReturnValue("scope:chat")
    loadServicePromptSnapshotMock.mockResolvedValue(servicePromptSnapshot)
    createChatMock.mockResolvedValue({
      id: "persona-chat-1",
      title: "Persona chat",
      assistant_kind: "persona",
      assistant_id: "garden-helper",
      persona_memory_mode: "read_only"
    })
    getChatMock.mockResolvedValue({
      id: "workspace-existing-chat",
      scope_type: "workspace",
      workspace_id: "workspace-plain"
    })
    addChatMessageMock.mockImplementation(async (_chatId, payload) => ({
      id: `server-message-${payload?.role ?? "message"}`,
      version: 1
    }))
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
    }, { scope: undefined, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal })
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
        scopeKey: servicePromptSnapshot.scopeKey,
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
        serverChatId: "persona-chat-1",
        conversationId: "persona-chat-1"
      })
    )
  })

  it.each(["turn", "reply"])("captures the tracked %s owner before mirror linking and releases it", async operation => {
    const pending = deferred<typeof servicePromptSnapshot>()
    loadServicePromptSnapshotMock.mockReturnValueOnce(pending.promise)
    const options = { ...createHookOptions(), serverChatId: "persona-chat-1", serverChatMetaLoaded: true, serverChatAssistantKind: "persona", serverChatAssistantId: "garden-helper", compareFeatureEnabled: true }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    let request!: Promise<unknown>
    act(() => { request = operation === "turn" ? result.current.onSubmit({ message: "Owned follow-up", image: "" }) : result.current.sendPerModelReply({ clusterId: "cluster", modelId: "openai:gpt-4.1", message: "Owned reply" }) })
    await vi.waitFor(() => expect(loadServicePromptSnapshotMock).toHaveBeenCalledTimes(1))
    expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled()
    await act(async () => { pending.resolve(servicePromptSnapshot); await request })
    expect(options.ensureServerChatHistoryId).toHaveBeenCalledWith("persona-chat-1", options.serverChatTitle || undefined, servicePromptSnapshot.scopeInvalidatedSignal, servicePromptSnapshot)
    expect(servicePromptSnapshot.release).toHaveBeenCalledTimes(1)
  })

  it("persists the canonical captured scope for a prompt-backed persona turn", async () => {
    const options = {
      ...createHookOptions(),
      webSearch: true
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Search with this persona",
        image: ""
      })
    })

    expect(savePlaygroundSessionMock).toHaveBeenCalledWith(
      expect.objectContaining({ scopeKey: "scope:captured-user" })
    )
    expect(buildChatSurfaceScopeKeyFromConfigMock).not.toHaveBeenCalled()
    expect(releaseServicePromptSnapshotMock).toHaveBeenCalledTimes(1)
  })

  it("passes workspace scope through when creating a persona-backed chat", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-1" } as const
    const options = {
      ...createHookOptions(),
      scope
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello scoped persona",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      {
        assistant_kind: "persona",
        assistant_id: "garden-helper",
        persona_memory_mode: "read_only",
        state: "in-progress",
        topic_label: undefined,
        cluster_id: undefined,
        source: undefined,
        external_ref: undefined
      },
      { scope, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal }
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Hello scoped persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "persona-chat-1"
      })
    )
  })

  it("routes an inherited workspace persona default when no explicit assistant is selected", async () => {
    const inheritedWorkspaceAssistant = {
      kind: "persona" as const,
      id: "workspace-helper",
      name: "Workspace Helper",
      metadata: {
        selectionMode: "tracked",
        source: "workspace",
        personaMemoryMode: "read_write"
      }
    }
    const options = {
      ...createHookOptions(),
      selectedAssistant: inheritedWorkspaceAssistant,
      inheritedAssistant: inheritedWorkspaceAssistant,
      inheritedPersonaMemoryMode: "read_write" as const
    }
    createChatMock.mockResolvedValueOnce({
      id: "workspace-persona-chat",
      title: "Workspace persona chat",
      assistant_kind: "persona",
      assistant_id: "workspace-helper",
      persona_memory_mode: "read_write"
    })
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Use workspace persona",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith({
      assistant_kind: "persona",
      assistant_id: "workspace-helper",
      persona_memory_mode: "read_write",
      state: "in-progress",
      topic_label: undefined,
      cluster_id: undefined,
      source: undefined,
      external_ref: undefined
    }, { scope: undefined, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal })
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith("persona")
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith(
      "workspace-helper"
    )
    expect(savePlaygroundSessionMock).toHaveBeenCalledWith(
      expect.objectContaining({
        serverChatId: "workspace-persona-chat",
        trackedAssistantKind: "persona",
        trackedAssistantId: "workspace-helper",
        trackedAssistantDisplayName: "Workspace Helper",
        serverChatPersonaMemoryMode: "read_write"
      })
    )
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Use workspace persona",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        assistantIdentity: {
          name: "Workspace Helper",
          avatarUrl: undefined
        },
        serverChatId: "workspace-persona-chat"
      })
    )
  })

  it("creates a workspace-scoped server chat for plain normal sends", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-plain" } as const
    createChatMock.mockResolvedValueOnce({
      id: "workspace-plain-chat",
      title: "Workspace plain chat",
      state: "in-progress",
      version: 3
    })
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        historyId: string | null
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
        serverChatId?: string | null
      }
      expect(params.historyId).toBe("history-persona")
      expect(params.serverChatId).toBe("workspace-plain-chat")
      await params.saveMessageOnSuccess({
        historyId: "history-persona",
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Plain workspace hello",
        image: "",
        fullText: "Plain workspace reply",
        source: []
      })
    })
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: null,
      serverChatTitle: null,
      selectedAssistant: null
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Plain workspace hello",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "in-progress"
      }),
      expect.objectContaining({
        scope
      })
    )
    expect(options.setServerChatId).toHaveBeenCalledWith("workspace-plain-chat")
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith(null)
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith(null)
    expect(normalChatModeMock).toHaveBeenCalledWith(
      "Plain workspace hello",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "workspace-plain-chat"
      })
    )
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "workspace-plain-chat"
      })
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "workspace-plain-chat",
      expect.objectContaining({
        role: "user",
        content: "Plain workspace hello"
      }),
      { scope }
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "workspace-plain-chat",
      expect.objectContaining({
        role: "assistant",
        content: "Plain workspace reply"
      }),
      { scope }
    )
  })

  it("adopts the first persisted plain chat so server message actions become available", async () => {
    createChatMock.mockResolvedValueOnce({ id: "first-saved-chat", title: "Hello" })
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as { saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null> }
      await params.saveMessageOnSuccess({
        historyId: null, selectedModel: "deepseek-chat", message: "Hello", image: "",
        fullText: "Saved answer", source: [], saveToDb: true, conversationId: "first-saved-chat"
      })
    })
    const options = {
      ...createHookOptions(), serverChatId: null, serverChatTitle: null,
      serverChatAssistantKind: null, serverChatAssistantId: null,
      serverChatCharacterId: null, selectedAssistant: null, historyId: null
    }
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => { await result.current.onSubmit({ message: "Hello", image: "" }) })
    expect(options.setServerChatId).toHaveBeenCalledWith("first-saved-chat")
  })

  it("rejects a stale server chat id from another scope before workspace sends", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-fresh" } as const
    getChatMock.mockResolvedValueOnce({
      id: "stale-global-chat",
      scope_type: "global",
      workspace_id: null
    })
    createChatMock.mockResolvedValueOnce({
      id: "workspace-fresh-chat",
      title: "Workspace fresh chat",
      state: "in-progress"
    })
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        serverChatId?: string | null
        conversationId?: string | null
      }
      expect(params.serverChatId).toBe("workspace-fresh-chat")
      expect(params.conversationId).toBe("workspace-fresh-chat")
    })
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: "stale-global-chat",
      serverChatTitle: "Stale global chat",
      serverChatMetaLoaded: true,
      selectedAssistant: null
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Use the active workspace",
        image: ""
      })
    })

    expect(getChatMock).toHaveBeenCalledWith("stale-global-chat", {
      scope,
      requestScope: servicePromptSnapshot.requestScope,
      signal: servicePromptSnapshot.scopeSignal
    })
    expect(options.setServerChatId).toHaveBeenCalledWith(null)
    expect(options.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "in-progress"
      }),
      expect.objectContaining({ scope })
    )
    expect(options.setServerChatId).toHaveBeenLastCalledWith(
      "workspace-fresh-chat"
    )
  })

  it.each([false, true].flatMap(durable => ["reject", "resolve"].map(settlement => ({ durable, settlement }))))(
    "preserves the binding after Stop during workspace conversation preflight (durable $durable, $settlement)",
    async ({ durable, settlement }) => {
    const controller = new AbortController()
    loadServicePromptSnapshotMock.mockResolvedValueOnce({
      ...servicePromptSnapshot,
      scopeSignal: controller.signal
    })
    const lookup = deferred<Record<string, unknown>>()
    if (settlement === "resolve") {
      getChatMock.mockReturnValueOnce(lookup.promise)
    } else {
      getChatMock.mockImplementationOnce((_id, requestOptions) => new Promise((_resolve, reject) => {
        requestOptions.signal.addEventListener("abort", () => {
          reject(new DOMException("Request cancelled", "AbortError"))
        }, { once: true })
      }))
    }
    const options = {
      ...createHookOptions(),
      scope: { type: "workspace", workspaceId: "workspace-plain" } as const,
      serverChatId: "bound-original-chat",
      selectedAssistant: null
    }
    const { result, unmount } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    let submission!: ReturnType<typeof result.current.onSubmit>
    act(() => {
      submission = result.current.onSubmit({
        message: "Keep this conversation",
        image: "",
        controller,
        ...(durable ? { requestOverrides: { tldwTurn: {
          user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739"
        } } } : {})
      })
    })
    await vi.waitFor(() => expect(getChatMock).toHaveBeenCalledTimes(1))
    const historyLinksBeforeStop = options.ensureServerChatHistoryId.mock.calls.length
    controller.abort()
    lookup.resolve({ id: "bound-original-chat", scope_type: "workspace", workspace_id: "workspace-plain" })
    await act(async () => {
      expect(await submission).toMatchObject({ status: "skipped", reason: "Request cancelled" })
    })
    expect(options.setServerChatId).not.toHaveBeenCalled()
    expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalled()
    expect(options.ensureServerChatHistoryId).toHaveBeenCalledTimes(historyLinksBeforeStop)
    expect(normalChatModeMock).not.toHaveBeenCalled()
    unmount()
  })

  describe.each(["normal", "rag"] as const)("%s durable workspace binding", (mode) => {
    const scope = { type: "workspace", workspaceId: "workspace-plain" } as const
    const tldwTurn = {
      user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739"
    }
    const modeOverrides =
      mode === "rag" ? { fileRetrievalEnabled: true, ragMediaIds: [101] } : {}
    const generationMock = mode === "rag" ? ragModeMock : normalChatModeMock

    it.each([
      { isRegenerate: false, status: 503 },
      { isRegenerate: true, status: 503 },
      { isRegenerate: false, status: 404 },
      { isRegenerate: true, status: 404 }
    ])("propagates $status without replacing a bound turn (regenerate=$isRegenerate)", async ({ isRegenerate, status }) => {
      const error = new Error(`Conversation lookup failed (${status})`)
      getChatMock.mockRejectedValueOnce(error)
      const options = {
        ...createHookOptions(),
        scope,
        selectedAssistant: null,
        serverChatId: "different-current-chat"
      }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      await act(async () => {
        expect(await result.current.onSubmit({
          message: "Retry the saved turn",
          image: "",
          isRegenerate,
          serverChatIdOverride: "bound-original-chat",
          requestOverrides: { ...modeOverrides, tldwTurn }
        })).toEqual({ status: "failed", errorMessage: error.message })
      })

      expect(getChatMock).toHaveBeenCalledWith("bound-original-chat", {
        scope,
        requestScope: servicePromptSnapshot.requestScope,
        signal: servicePromptSnapshot.scopeSignal
      })
      expect(createChatMock).not.toHaveBeenCalled()
      expect(normalChatModeMock).not.toHaveBeenCalled()
      expect(ragModeMock).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
      expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
    })

    it.each([
      { label: "another workspace", metadata: { scope_type: "workspace", workspace_id: "other-workspace" } },
      { label: "global scope", metadata: { scope_type: "global" } },
      { label: "missing scope", metadata: {} }
    ])("rejects $label without replacing the bound conversation", async ({ metadata }) => {
      getChatMock.mockResolvedValueOnce({ id: "bound-original-chat", ...metadata })
      const options = {
        ...createHookOptions(),
        scope,
        selectedAssistant: null,
        serverChatId: "bound-original-chat"
      }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      await act(async () => {
        expect(await result.current.onSubmit({
          message: "Retry the saved turn",
          image: "",
          isRegenerate: true,
          serverChatIdOverride: "bound-original-chat",
          requestOverrides: { ...modeOverrides, tldwTurn }
        })).toEqual({
          status: "failed",
          errorMessage: "The bound conversation is unavailable in this workspace."
        })
      })

      expect(createChatMock).not.toHaveBeenCalled()
      expect(normalChatModeMock).not.toHaveBeenCalled()
      expect(ragModeMock).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
      expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
    })

    it.each([false, true])("keeps the explicit conversation and turn identity (regenerate=%s)", async (isRegenerate) => {
      getChatMock.mockResolvedValueOnce({
        id: "bound-original-chat",
        scope_type: "workspace",
        workspace_id: scope.workspaceId
      })
      const options = {
        ...createHookOptions(),
        scope,
        selectedAssistant: null,
        serverChatId: "different-current-chat"
      }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      await act(async () => {
        expect(await result.current.onSubmit({
          message: "Retry the saved turn",
          image: "",
          isRegenerate,
          serverChatIdOverride: "bound-original-chat",
          requestOverrides: { ...modeOverrides, tldwTurn }
        })).toEqual({ status: "submitted" })
      })

      expect(getChatMock).toHaveBeenCalledWith("bound-original-chat", {
        scope,
        requestScope: servicePromptSnapshot.requestScope,
        signal: servicePromptSnapshot.scopeSignal
      })
      expect(createChatMock).not.toHaveBeenCalled()
      expect(generationMock).toHaveBeenCalledWith(
        "Retry the saved turn", "", isRegenerate, [], [], expect.any(AbortSignal),
        expect.objectContaining({
          serverChatId: "bound-original-chat",
          conversationId: "bound-original-chat",
          tldwTurn
        })
      )
    })

    it("creates a conversation for an unbound durable first send", async () => {
      const options = { ...createHookOptions(), scope, selectedAssistant: null }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      await act(async () => {
        expect(await result.current.onSubmit({
          message: "Start a new turn",
          image: "",
          requestOverrides: { ...modeOverrides, tldwTurn }
        })).toEqual({ status: "submitted" })
      })

      expect(getChatMock).not.toHaveBeenCalled()
      expect(createChatMock).toHaveBeenCalledTimes(1)
      expect(generationMock.mock.calls[0][6]).toEqual(expect.objectContaining({
        conversationId: "persona-chat-1",
        tldwTurn
      }))
    })

    it("preserves replacement bootstrap for a legacy override without durable identity", async () => {
      getChatMock.mockRejectedValueOnce(new Error("Conversation lookup failed (503)"))
      const options = { ...createHookOptions(), scope, selectedAssistant: null }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      await act(async () => {
        expect(await result.current.onSubmit({
          message: "Legacy send",
          image: "",
          serverChatIdOverride: "stale-legacy-chat",
          requestOverrides: modeOverrides
        })).toEqual({ status: "submitted" })
      })

      expect(options.setServerChatId).toHaveBeenCalledWith(null)
      expect(createChatMock).toHaveBeenCalledTimes(1)
      expect(generationMock.mock.calls[0][6]).toEqual(expect.objectContaining({
        conversationId: "persona-chat-1"
      }))
    })
  })

  describe.each([
    { name: "existing normal chat", persona: false, chatId: "bound-original-chat" },
    { name: "new normal chat", persona: false, chatId: null },
    { name: "new persona chat", persona: true, chatId: null }
  ])("durable scope for $name", ({ persona, chatId }) => {
    const scope = { type: "workspace", workspaceId: "workspace-plain" } as const
    const tldwTurn = {
      user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739"
    }
    const bindingMock = chatId ? getChatMock : createChatMock

    it("captures scope before history or server binding and forwards the same snapshot", async () => {
      const base = createHookOptions()
      const options = {
        ...base,
        scope,
        serverChatId: chatId,
        selectedAssistant: persona ? base.selectedAssistant : null
      }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      await act(async () => {
        expect(await result.current.onSubmit({
          message: "Keep the captured scope",
          image: "",
          isRegenerate: Boolean(chatId),
          serverChatIdOverride: chatId,
          requestOverrides: { tldwTurn }
        })).toEqual({ status: "submitted" })
      })

      expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith([], {
        signal: expect.any(AbortSignal)
      })
      expect(loadServicePromptSnapshotMock.mock.invocationCallOrder[0]).toBeLessThan(
        options.ensureServerChatHistoryId.mock.invocationCallOrder[0]
      )
      expect(loadServicePromptSnapshotMock.mock.invocationCallOrder[0]).toBeLessThan(
        bindingMock.mock.invocationCallOrder[0]
      )
      expect(options.ensureServerChatHistoryId).toHaveBeenCalledWith(
        expect.any(String),
        persona || chatId ? undefined : "Persona chat",
        servicePromptSnapshot.scopeInvalidatedSignal,
        servicePromptSnapshot
      )
      expect(normalChatModeMock.mock.calls[0][6].servicePromptSnapshot).toBe(
        servicePromptSnapshot
      )
      expect(releaseServicePromptSnapshotMock).toHaveBeenCalledTimes(1)
    })

    it("does not dispatch generation after scope changes during server binding", async () => {
      const scopeController = new AbortController()
      const scopedSnapshot = {
        ...servicePromptSnapshot,
        scopeSignal: scopeController.signal,
        scopeInvalidatedSignal: scopeController.signal
      }
      loadServicePromptSnapshotMock.mockResolvedValue(scopedSnapshot)
      const binding = deferred<Record<string, unknown>>()
      bindingMock.mockImplementationOnce(() => binding.promise)
      const base = createHookOptions()
      const options = {
        ...base,
        scope,
        serverChatId: chatId,
        selectedAssistant: persona ? base.selectedAssistant : null
      }
      const { result } = renderHook(() =>
        useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
      )

      let submission!: ReturnType<typeof result.current.onSubmit>
      act(() => {
        submission = result.current.onSubmit({
          message: "Do not cross scopes",
          image: "",
          isRegenerate: Boolean(chatId),
          serverChatIdOverride: chatId,
          requestOverrides: { tldwTurn }
        })
      })
      await vi.waitFor(() => expect(bindingMock).toHaveBeenCalledTimes(1))
      scopeController.abort()
      binding.resolve({
        id: chatId ?? "newly-created-chat",
        scope_type: "workspace",
        workspace_id: scope.workspaceId,
        assistant_kind: persona ? "persona" : null,
        assistant_id: persona ? "garden-helper" : null
      })

      await act(async () => {
        expect(await submission).toMatchObject({ status: "failed" })
      })

      expect(normalChatModeMock).not.toHaveBeenCalled()
      expect(ragModeMock).not.toHaveBeenCalled()
      expect(options.setServerChatId).not.toHaveBeenCalled()
      expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
      expect(releaseServicePromptSnapshotMock).toHaveBeenCalledTimes(1)
    })
  })

  it("does not look up a bound conversation after scope invalidation during history loading", async () => {
    const invalidated = new AbortController()
    loadServicePromptSnapshotMock.mockResolvedValue({
      ...servicePromptSnapshot,
      scopeInvalidatedSignal: invalidated.signal
    })
    const options = {
      ...createHookOptions(),
      scope: { type: "workspace", workspaceId: "workspace-plain" } as const,
      serverChatId: "bound-original-chat",
      selectedAssistant: null
    }
    options.ensureServerChatHistoryId.mockImplementationOnce(async () => {
      invalidated.abort()
      return "history-persona"
    })
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )

    await act(async () => {
      expect(await result.current.onSubmit({
        message: "Keep the original target",
        image: "",
        serverChatIdOverride: "bound-original-chat",
        requestOverrides: {
          tldwTurn: { user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739" }
        }
      })).toMatchObject({ status: "failed" })
    })
    expect(getChatMock).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).not.toHaveBeenCalled()
  })

  it("does not replace a current conversation when the request guard rejects its captured scope", async () => {
    getChatMock.mockRejectedValueOnce(Object.assign(new Error("Captured account changed"), {
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    }))
    const options = {
      ...createHookOptions(),
      scope: { type: "workspace", workspaceId: "workspace-plain" } as const,
      serverChatId: "bound-original-chat",
      selectedAssistant: null
    }
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
    await act(async () => {
      expect(await result.current.onSubmit({
        message: "Stay in the original conversation", image: "",
        requestOverrides: {
          tldwTurn: { user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739" }
        }
      })).toMatchObject({ status: "failed" })
    })
    expect(createChatMock).not.toHaveBeenCalled()
    expect(options.setServerChatId).not.toHaveBeenCalled()
    expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
    expect(normalChatModeMock).not.toHaveBeenCalled()
  })

  it.each([
    { capsUnavailable: false, receipt: undefined },
    { capsUnavailable: true, receipt: undefined },
    { capsUnavailable: false, receipt: false },
    { capsUnavailable: true, receipt: false }
  ])("keeps one durable user without receipts (capsUnavailable=$capsUnavailable, receipt=$receipt)", async ({ capsUnavailable, receipt }) => {
    const { runChatPipeline } = await import("@/hooks/chat-modes/chatModePipeline")
    const rows: Array<{ role: string; content: string }> = []
    const tldwTurn = { user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739" }
    const scope = { type: "workspace", workspaceId: "workspace-plain" } as const
    if (capsUnavailable) {
      getServerCapabilitiesMock.mockRejectedValueOnce(new Error("Capability cache unavailable"))
    }
    pageAssistModelMock.mockResolvedValue({
      conversationId: "bound-original-chat",
      saveToDb: true,
      serverMessagesAlreadyPersisted: receipt,
      stream: async function* () {
        rows.push({ role: "user", content: "Durable question" })
        yield "Fallback assistant answer"
      }
    })
    addChatMessageMock.mockImplementation(async (_chatId, payload) => {
      rows.push(payload)
      return { id: `mirror-${rows.length}` }
    })
    normalChatModeMock.mockImplementationOnce(async (message, image, regenerate, messages, history, signal, params) =>
      runChatPipeline({
        id: "normal",
        setupMessages: () => ({ targetMessageId: "assistant-1" }),
        preparePrompt: async () => ({
          chatHistory: [],
          humanMessage: { role: "user", content: message },
          sources: []
        })
      }, message, image, regenerate, messages, history, signal, params)
    )
    const options = {
      ...createHookOptions(), scope, selectedAssistant: null,
      serverChatId: "bound-original-chat"
    }
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )
    await act(async () => {
      expect(await result.current.onSubmit({
        message: "Durable question", image: "", requestOverrides: { tldwTurn }
      })).toEqual({ status: "submitted" })
    })

    expect(rows.filter((row) => row.role === "user")).toEqual([
      { role: "user", content: "Durable question" }
    ])
    expect(rows.filter((row) => row.role === "assistant")).toHaveLength(1)
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(expect.objectContaining({
      serverOwnsUserMessage: true,
      serverMessagesAlreadyPersisted: false
    }))
  })

  it.each([
    { rag: false, chatId: null },
    { rag: true, chatId: null },
    { rag: false, chatId: "bound-original-chat" },
    { rag: true, chatId: "bound-original-chat" }
  ])("checks the failed turn's expected scope before any binding or generation (%j)", async ({ rag, chatId }) => {
    const expectedScope = servicePromptSnapshot.requestScope
    const scopeError = Object.assign(new Error("Request scope changed"), {
      code: "service_prompt_scope_changed"
    })
    loadServicePromptSnapshotMock.mockRejectedValueOnce(scopeError)
    const options = {
      ...createHookOptions(),
      scope: { type: "workspace", workspaceId: "workspace-plain" },
      selectedAssistant: null,
      serverChatId: chatId
    }
    const { result } = renderHook(() =>
      useChatActions(options as unknown as Parameters<typeof useChatActions>[0])
    )

    await act(async () => {
      expect(await result.current.onSubmit({
        message: "Original question", image: "", isRegenerate: Boolean(chatId),
        serverChatIdOverride: chatId,
        requestOverrides: {
          tldwTurn: { user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739" },
          requestScope: expectedScope,
          ...(rag ? { ragMediaIds: [7], fileRetrievalEnabled: true } : {})
        }
      })).toEqual({ status: "failed", errorMessage: "Request scope changed" })
    })

    expect(loadServicePromptSnapshotMock).toHaveBeenCalledWith(
      rag ? ["chat.rag.answer", "chat.rag.question_rewrite"] : [],
      { signal: expect.any(AbortSignal), requestScope: expectedScope }
    )
    expect(options.ensureServerChatHistoryId).not.toHaveBeenCalled()
    expect(getChatMock).not.toHaveBeenCalled()
    expect(createChatMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(ragModeMock).not.toHaveBeenCalled()
    expect(options.setStreaming).toHaveBeenLastCalledWith(false)
  })

  it("creates a workspace-scoped server chat for staged RAG sends", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-rag" } as const
    createChatMock.mockResolvedValueOnce({
      id: "workspace-rag-chat",
      title: "Workspace RAG chat",
      state: "in-progress"
    })
    ragModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        historyId: string | null
        saveMessageOnSuccess: (payload: Record<string, unknown>) => Promise<string | null>
        serverChatId?: string | null
      }
      expect(params.historyId).toBe("history-persona")
      expect(params.serverChatId).toBe("workspace-rag-chat")
      await params.saveMessageOnSuccess({
        historyId: "history-persona",
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Use staged source",
        image: "",
        fullText: "RAG reply",
        source: [],
        saveToDb: false
      })
    })
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: null,
      serverChatTitle: null,
      selectedAssistant: null
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Use staged source",
        image: "",
        requestOverrides: {
          fileRetrievalEnabled: true,
          ragMediaIds: [101]
        }
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "in-progress"
      }),
      expect.objectContaining({
        scope,
        requestScope: servicePromptSnapshot.requestScope,
        signal: servicePromptSnapshot.scopeSignal
      })
    )
    expect(ragModeMock).toHaveBeenCalledWith(
      "Use staged source",
      "",
      false,
      [],
      [],
      expect.any(AbortSignal),
      expect.objectContaining({
        historyId: "history-persona",
        serverChatId: "workspace-rag-chat",
        ragMediaIds: [101]
      })
    )
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "workspace-rag-chat"
      })
    )
  })

  it("does not publish workspace chat metadata when scope changes during history linking", async () => {
    const scope = {
      type: "workspace",
      workspaceId: "workspace-scope-race"
    } as const
    const scopeController = new AbortController()
    const scopedSnapshot = {
      ...servicePromptSnapshot,
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: scopeController.signal
    }
    loadServicePromptSnapshotMock.mockResolvedValueOnce(scopedSnapshot)
    createChatMock.mockResolvedValueOnce({
      id: "workspace-scope-race-chat",
      title: "Workspace scope race",
      state: "resolved",
      version: 5,
      topic_label: "Race topic",
      cluster_id: "race-cluster",
      source: "workspace",
      external_ref: "race-ref"
    })
    const historyLink = deferred<string | null>()
    const options = {
      ...createHookOptions(),
      scope,
      selectedAssistant: null,
      ensureServerChatHistoryId: vi.fn(() => historyLink.promise)
    }
    const { result } = renderHook(() => useChatActions(options as any))

    let submission!: ReturnType<typeof result.current.onSubmit>
    act(() => {
      submission = result.current.onSubmit({
        message: "Keep workspace state scoped",
        image: "",
        requestOverrides: {
          fileRetrievalEnabled: true,
          ragMediaIds: [101]
        }
      })
    })
    await vi.waitFor(() =>
      expect(options.ensureServerChatHistoryId).toHaveBeenCalledTimes(1)
    )
    scopeController.abort()
    historyLink.resolve("history-stale")

    await act(async () => {
      await submission
    })

    expect(options.ensureServerChatHistoryId).toHaveBeenCalledWith(
      "workspace-scope-race-chat",
      "Workspace scope race",
      scopeController.signal,
      scopedSnapshot
    )
    expect(options.setServerChatId).not.toHaveBeenCalled()
    expect(options.setServerChatTitle).not.toHaveBeenCalled()
    expect(options.setServerChatCharacterId).not.toHaveBeenCalled()
    expect(options.setServerChatAssistantKind).not.toHaveBeenCalled()
    expect(options.setServerChatAssistantId).not.toHaveBeenCalled()
    expect(options.setServerChatPersonaMemoryMode).not.toHaveBeenCalled()
    expect(options.setServerChatMetaLoaded).not.toHaveBeenCalled()
    expect(options.setServerChatState).not.toHaveBeenCalled()
    expect(options.setServerChatVersion).not.toHaveBeenCalled()
    expect(options.setServerChatTopic).not.toHaveBeenCalled()
    expect(options.setServerChatClusterId).not.toHaveBeenCalled()
    expect(options.setServerChatSource).not.toHaveBeenCalled()
    expect(options.setServerChatExternalRef).not.toHaveBeenCalled()
    expect(options.invalidateServerChatHistory).not.toHaveBeenCalled()
    expect(ragModeMock).not.toHaveBeenCalled()
  })

  it("reuses an existing global conversation through inference and local persistence", async () => {
    let capturedParams:
      | {
          conversationId?: string | null
          serverChatId?: string | null
        }
      | null = null
    normalChatModeMock.mockImplementationOnce(async (...args: unknown[]) => {
      const params = args[6] as {
        conversationId?: string | null
        saveMessageOnSuccess: (
          payload: Record<string, unknown>
        ) => Promise<string | null>
        serverChatId?: string | null
      }
      capturedParams = params
      await params.saveMessageOnSuccess({
        historyId: "history-persona",
        isRegenerate: false,
        selectedModel: "deepseek-chat",
        message: "Plain global hello",
        image: "",
        fullText: "Plain global reply",
        source: []
      })
    })
    const options = {
      ...createHookOptions(),
      selectedAssistant: null,
      serverChatId: "existing-global-chat",
      serverChatTitle: "Existing global chat"
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Plain global hello",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(capturedParams?.serverChatId).toBe("existing-global-chat")
    expect(capturedParams?.conversationId).toBe("existing-global-chat")
    expect(baseSaveMessageOnSuccessMock).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "existing-global-chat"
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
    }, { scope: undefined, requestScope: servicePromptSnapshot.requestScope, signal: servicePromptSnapshot.scopeSignal })
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
        image: "",
        requestOverrides: {
          tldwTurn: {
            user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739"
          }
        }
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
        serverChatId: "persona-chat-existing",
        conversationId: "persona-chat-existing",
        tldwTurn: {
          user_message_id: "5bbd7b2f-a92b-427c-8062-b039d47de739"
        }
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
