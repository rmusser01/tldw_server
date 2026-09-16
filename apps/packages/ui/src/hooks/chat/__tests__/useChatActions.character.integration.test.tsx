// @vitest-environment jsdom
import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useChatActions } from "../useChatActions"
import type { Message } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"

const {
  bgStreamMock,
  saveLocalSuccessMock,
  saveLocalErrorMock,
  addChatMessageMock,
  createChatMock,
  detectCharacterMoodMock,
  streamCharacterChatCompletionMock,
  persistCharacterCompletionMock,
  normalChatModeMock,
  resolveVisualIdentityBindingMock
} = vi.hoisted(() => ({
  bgStreamMock: vi.fn(),
  saveLocalErrorMock: vi.fn(async (_payload: unknown) => "history-character"),
  saveLocalSuccessMock: vi.fn(async (_payload: unknown) => "history-character"),
  addChatMessageMock: vi.fn(async () => ({ id: "user-server-1", version: 1 })),
  createChatMock: vi.fn(),
  detectCharacterMoodMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  persistCharacterCompletionMock: vi.fn(async () => ({
    assistant_message_id: "assistant-server-1",
    version: 1
  })),
  normalChatModeMock: vi.fn(),
  resolveVisualIdentityBindingMock: vi.fn(async () => ({
    actor_kind: "character",
    actor_id: 12,
    pack_id: 1,
    pack_version_id: 2,
    expression_key: "surprised",
    requested_expression_key: "surprised",
    asset_id: 9,
    storage_relpath: null,
    fallback_reason: "manual_override",
    is_animated: false,
    content_type: "image/png",
    asset_url: "/api/v1/visual-identities/packs/1/assets/9/content"
  }))
}))

const recoveryAuthority = vi.hoisted(() => ({ controller: new AbortController() }))

const messageStoreState = vi.hoisted(() => ({
  value: {
    selectedModel: "deepseek-chat" as string | null,
    serverChatId: null as string | null,
    serverChatCharacterId: null as string | number | null,
    serverChatAssistantKind: null as "character" | "persona" | null,
    serverChatSource: null as string | null
  }
}))

vi.mock("@/services/background-proxy", () => ({ bgStream: bgStreamMock, bgRequest: vi.fn(), bgUpload: vi.fn() }))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
    scopeKey: "scope:test-chat", requestScope: { config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user" }, userId: null },
    scopeSignal: signal, scopeInvalidatedSignal: recoveryAuthority.controller.signal, definitions: {}, capability: "unchecked", release: vi.fn()
  })
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
  createSaveMessageOnSuccess: vi.fn(
    () => saveLocalSuccessMock
  ),
  createSaveMessageOnError: vi.fn(
    () => saveLocalErrorMock
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

vi.mock("@/utils/character-mood", () => ({
  detectCharacterMood: detectCharacterMoodMock
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
  parseSelectedCharacterValue: vi.fn((value: unknown) => value)
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
    getState: () => messageStoreState.value
  }
}))

vi.mock("@/services/tldw/server-capabilities", () => ({
  getServerCapabilities: vi.fn(async () => ({ hasChatSaveToDb: false }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    createChat: createChatMock,
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock,
    addChatMessage: addChatMessageMock,
    getChatSettings: vi.fn(async () => ({ settings: null })),
    initialize: vi.fn(async () => null),
    resolveVisualIdentityBinding: resolveVisualIdentityBindingMock
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
  historyId: "history-character",
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
  serverChatId: "tracked-chat-1",
  serverChatTitle: "Tracked character chat",
  serverChatCharacterId: "char-tracked",
  serverChatAssistantKind: "character" as const,
  serverChatAssistantId: null,
  serverChatPersonaMemoryMode: null,
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
  ensureServerChatHistoryId: vi.fn(async () => "history-character"),
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
    id: "char-stale",
    name: "Stale Character",
    system_prompt: "Stale prompt"
  },
  selectedAssistant: null,
  messageSteeringMode: "none" as const,
  messageSteeringForceNarrate: false,
  clearMessageSteering: vi.fn()
})

describe("useChatActions character integration", () => {
  beforeEach(() => {
    recoveryAuthority.controller = new AbortController()
    vi.clearAllMocks()
    useStoreChatModelSettings.getState().reset()
    normalChatModeMock.mockResolvedValue(undefined)
    createChatMock.mockResolvedValue({
      id: "unexpected-new-chat",
      character_id: "char-stale",
      title: "Wrong chat"
    })
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
    detectCharacterMoodMock.mockReturnValue({
      label: "neutral",
      confidence: 0.2,
      topic: "reply"
    })
    messageStoreState.value = {
      selectedModel: "deepseek-chat",
      serverChatId: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatSource: null
    }
    resolveVisualIdentityBindingMock.mockClear()
  })


  const useRealCharacterTransport = () => {
    streamCharacterChatCompletionMock.mockImplementation(chatRagMethods.streamCharacterChatCompletion)
    bgStreamMock.mockImplementation(async function* () {
      yield JSON.stringify({ choices: [{ delta: { content: "Configured reply" } }] })
    })
  }

  const scopedOptions = (scope: string, values: Record<string, number>) => {
    const store = useStoreChatModelSettings.getState()
    store.setActiveSettingsScope(scope)
    for (const [key, value] of Object.entries(values)) {
      store.updateScopedSetting(scope, key as "numPredict", value)
    }
    return { ...createHookOptions(), currentChatModelSettings: useStoreChatModelSettings.getState() }
  }

  it.each([0, 0.5])("forwards scoped current-chat controls through the real Character transport (sampling=%s)", async sampling => {
    useRealCharacterTransport()
    const options = scopedOptions("openai:deepseek-chat", { numPredict: 16, temperature: sampling, topP: sampling, repeatPenalty: sampling })
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    await act(async () => { await result.current.onSubmit({ message: "Configured question", image: "" }) })
    expect(bgStreamMock).toHaveBeenCalledWith(expect.objectContaining({
      path: expect.stringContaining("/chats/tracked-chat-1/complete-v2"),
      body: expect.objectContaining({ max_tokens: 16, temperature: sampling, top_p: sampling, repetition_penalty: sampling, stream: true })
    }))
  })

  it("omits unset controls from the real Character request after switching to an unconfigured model scope", async () => {
    useRealCharacterTransport()
    scopedOptions("openai:previous-model", { numPredict: 16, temperature: 0.5 })
    const options = scopedOptions("openai:deepseek-chat", {})
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    await act(async () => { await result.current.onSubmit({ message: "Default question", image: "" }) })
    const body = bgStreamMock.mock.calls[0][0].body
    for (const key of ["max_tokens", "temperature", "top_p", "repetition_penalty"]) expect(body).not.toHaveProperty(key)
  })

  it("retains captured model controls while a pending Character user save outlives a settings-scope change", async () => {
    useRealCharacterTransport()
    let release!: (value: { id: string; version: number }) => void
    addChatMessageMock.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    const original = scopedOptions("openai:deepseek-chat", { numPredict: 16, temperature: 0.25 })
    const { result, rerender } = renderHook(({ options }) => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]), { initialProps: { options: original } })
    let pending!: ReturnType<typeof result.current.onSubmit>
    await act(async () => { pending = result.current.onSubmit({ message: "Captured question", image: "" }); await Promise.resolve() })
    await waitFor(() => expect(addChatMessageMock).toHaveBeenCalled())
    const next = { ...scopedOptions("openai:new-model", { numPredict: 128, temperature: 0.75 }), selectedModel: "new-model" }
    messageStoreState.value.selectedModel = "new-model"
    rerender({ options: next })
    await act(async () => { release({ id: "captured-user", version: 1 }); await pending })
    expect(bgStreamMock).toHaveBeenCalledWith(expect.objectContaining({ body: expect.objectContaining({ model: "deepseek-chat", max_tokens: 16, temperature: 0.25 }) }))
    expect(useStoreChatModelSettings.getState().numPredict).toBe(128)
  })

  it("does not publish recovered IDs or save an old owner's partial after the lease invalidates", async () => {
    streamCharacterChatCompletionMock.mockImplementationOnce(async function* () { yield "<think>Only reasoning</think>" })
    addChatMessageMock.mockResolvedValueOnce({ id: "user-server-1", version: 1 })
    let release!: (value: { id: string; version: number }) => void
    const held = new Promise<{ id: string; version: number }>(resolve => { release = resolve })
    addChatMessageMock.mockImplementationOnce(() => held)
    const options = scopedOptions("owner-a:deepseek-chat", { numPredict: 16 })
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    let pending!: ReturnType<typeof result.current.onSubmit>
    await act(async () => { pending = result.current.onSubmit({ message: "Question", image: "" }); await new Promise(resolve => setTimeout(resolve, 0)) })
    await waitFor(() => expect(addChatMessageMock).toHaveBeenCalledTimes(2))
    recoveryAuthority.controller.abort()
    scopedOptions("owner-b:deepseek-chat", { numPredict: 128 })
    options.setMessages.mockClear()
    await act(async () => { release({ id: "old-assistant", version: 1 }); await pending })
    expect(saveLocalErrorMock).not.toHaveBeenCalled()
    expect(options.setMessages).not.toHaveBeenCalled()
    expect(useStoreChatModelSettings.getState().numPredict).toBe(128)
  })

  it.each([
    ["closed", "<think>Only reasoning</think>"],
    ["unclosed", "<think>Only reasoning"],
    ["structured", { choices: [{ delta: { reasoning_content: "Only reasoning" } }] }]
  ])("preserves %s reasoning as a recoverable tracked completion with acknowledged IDs", async (_label, chunk) => {
    addChatMessageMock.mockResolvedValueOnce({ id: "user-server-1", version: 1 })
    addChatMessageMock.mockResolvedValueOnce({ id: "assistant-server-1", version: 1 })
    streamCharacterChatCompletionMock.mockImplementationOnce(async function* () { yield chunk })
    let rows: Message[] = []
    const options = { ...createHookOptions(), setMessages: vi.fn((next: Message[] | ((previous: Message[]) => Message[])) => { rows = typeof next === "function" ? next(rows) : next }) }
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    await act(async () => { await result.current.onSubmit({ message: "Question", image: "" }) })
    expect(saveLocalSuccessMock).not.toHaveBeenCalled()
    expect(saveLocalErrorMock).toHaveBeenCalledWith(expect.objectContaining({
      botMessage: expect.stringContaining("Only reasoning"), userServerMessageId: "user-server-1", assistantServerMessageId: "assistant-server-1"
    }))
    expect(rows.find(row => row.isBot)).toMatchObject({ generationInfo: { interrupted: true, interruptionReason: expect.stringContaining("final answer") } })
  })

  it.each(["fallback", "degraded"])("forwards the confirmed assistant ID from %s success to local persistence", async (outcome) => {
    persistCharacterCompletionMock.mockRejectedValueOnce(outcome === "fallback"
      ? new Error("Persist endpoint unavailable")
      : Object.assign(new Error("Saved with validation warning"), {
          status: 503,
          detail: { code: "persist_validation_degraded", saved: true, assistant_message_id: "fallback-assistant" }
        }))
    addChatMessageMock.mockResolvedValueOnce({ id: "user-server-1", version: 1 })
    if (outcome === "fallback") addChatMessageMock.mockResolvedValueOnce({ id: "fallback-assistant", version: 1 })
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as unknown as Parameters<typeof useChatActions>[0]))
    await act(async () => { await result.current.onSubmit({ message: "Question", image: "" }) })
    expect(addChatMessageMock).toHaveBeenCalledTimes(outcome === "fallback" ? 2 : 1)
    expect(saveLocalSuccessMock).toHaveBeenCalledWith(expect.objectContaining({
      userServerMessageId: "user-server-1", assistantServerMessageId: "fallback-assistant",
      serverMessagesAlreadyPersisted: true
    }))
  })

  it("keeps tracked character routing anchored to current chat metadata when global character state is stale", async () => {
    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello tracked character",
        image: ""
      })
    })

    expect(saveLocalSuccessMock).toHaveBeenCalledWith(expect.objectContaining({
      userServerMessageId: "user-server-1", assistantServerMessageId: "assistant-server-1"
    }))
    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledTimes(1)
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "tracked-chat-1",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(options.setServerChatAssistantKind).toHaveBeenCalledWith("character")
    expect(options.setServerChatAssistantId).toHaveBeenCalledWith("char-tracked")
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(options.setServerChatId).not.toHaveBeenCalledWith(null)
  })

  it("rebinds a greeting to the newly created conversation after a failed session", async () => {
    let visibleMessages: any[] = [{
      id: "local-greeting", isBot: true, message: "Welcome back",
      messageType: "character:greeting", serverMessageId: "old-chat-greeting"
    }]
    addChatMessageMock.mockImplementation(async (_id, payload) => ({
      id: payload.role === "assistant" ? "new-chat-greeting" : "new-user", version: 1
    }))
    const character = { id: "char-stale", name: "Guide", greeting: "Welcome back", system_prompt: "Help the gardener." }
    const options = {
      ...createHookOptions(), serverChatId: null, serverChatCharacterId: null,
      serverChatAssistantKind: null, selectedCharacter: character,
      selectedAssistant: { ...character, kind: "character", metadata: { selectionMode: "tracked" } },
      messages: visibleMessages,
      setMessages: (next: any) => { visibleMessages = typeof next === "function" ? next(visibleMessages) : next }
    }
    const { result } = renderHook(() => useChatActions(options as any))
    await act(async () => { await result.current.onSubmit({ message: "Try again", image: "" }) })
    expect(visibleMessages.find(message => message.messageType === "character:greeting")?.serverMessageId).toBe("new-chat-greeting")
  })

  it("handles emote commands without sending chat", async () => {
    const options = {
      ...createHookOptions(),
      setVisualIdentityManualExpressionOverride: vi.fn()
    }
    const { result } = renderHook(() => useChatActions(options as any))

    let submitResult: unknown
    await act(async () => {
      submitResult = await result.current.onSubmit({
        message: "/emote surprised",
        image: ""
      })
    })

    expect(options.setVisualIdentityManualExpressionOverride).toHaveBeenCalledWith(
      "surprised"
    )
    expect(streamCharacterChatCompletionMock).not.toHaveBeenCalled()
    expect(normalChatModeMock).not.toHaveBeenCalled()
    expect(options.setStreaming).not.toHaveBeenCalledWith(true)
    expect(submitResult).toEqual({
      status: "skipped",
      reason: "Visual identity expression updated"
    })
  })

  it("uses manual visual identity override when resolving assistant message metadata", async () => {
    const options = {
      ...createHookOptions(),
      serverChatCharacterId: 12,
      selectedCharacter: {
        id: "12",
        name: "Numeric Character",
        system_prompt: "Prompt"
      },
      visualIdentityManualExpressionOverride: "surprised"
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "React to this",
        image: ""
      })
    })

    expect(resolveVisualIdentityBindingMock).toHaveBeenCalledWith(
      expect.objectContaining({
        actor_kind: "character",
        actor_id: 12,
        expression_key: "surprised",
        manual_override_expression_key: "surprised"
      })
    )
  })

  it("does not let stale overlay assistant state override the active character chat", async () => {
    const options = {
      ...createHookOptions(),
      serverChatId: "other-character-chat",
      serverChatTitle: "Other character chat",
      serverChatCharacterId: 42,
      serverChatAssistantId: "42",
      selectedCharacter: {
        id: 99,
        name: "Miku",
        system_prompt: "Stale Miku prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "99",
        name: "Miku",
        system_prompt: "Stale Miku prompt",
        metadata: { selectionMode: "overlay" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Continue the other character chat",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "other-character-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "other-character-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 42
      }),
      undefined
    )
    const lastPersistCall = persistCharacterCompletionMock.mock.calls.at(-1) as
      | unknown[]
      | undefined
    const lastPersistRequest = lastPersistCall?.[1] as
      | Record<string, unknown>
      | undefined
    expect(lastPersistRequest?.speaker_character_name).toBeUndefined()
    expect(options.setServerChatId).not.toHaveBeenCalledWith(null)
    expect(options.setServerChatCharacterId).not.toHaveBeenCalledWith(null)
  })

  it("honors an explicit tracked character switch over the current chat metadata", async () => {
    createChatMock.mockResolvedValueOnce({
      id: "new-miku-chat",
      character_id: 99,
      title: "Miku chat"
    })
    const options = {
      ...createHookOptions(),
      serverChatId: "other-character-chat",
      serverChatTitle: "Other character chat",
      serverChatCharacterId: 42,
      serverChatAssistantId: "42",
      selectedCharacter: {
        id: 42,
        name: "Other Character",
        system_prompt: "Other character prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "99",
        name: "Miku",
        system_prompt: "Miku prompt",
        metadata: { selectionMode: "tracked" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Start talking with Miku",
        image: ""
      })
    })

    expect(options.setServerChatId).toHaveBeenCalledWith(null)
    expect(options.setServerChatCharacterId).toHaveBeenCalledWith(null)
    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: "99"
      })
    )
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "new-miku-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "new-miku-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 99,
        speaker_character_name: "Miku"
      }),
      undefined
    )
  })

  it("reuses the latest store character chat when greeting persistence updates before the send closure", async () => {
    messageStoreState.value = {
      selectedModel: "deepseek-chat",
      serverChatId: "ashley-greeting-chat",
      serverChatCharacterId: 4,
      serverChatAssistantKind: "character",
      serverChatSource: "webui-character-chat"
    }
    const options = {
      ...createHookOptions(),
      serverChatId: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatSource: null,
      selectedCharacter: {
        id: 4,
        name: "Ashley",
        system_prompt: "Ashley prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "4",
        name: "Ashley",
        system_prompt: "Ashley prompt",
        metadata: { selectionMode: "tracked" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Continue Ashley from her greeting",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "ashley-greeting-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "ashley-greeting-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 4,
        speaker_character_name: "Ashley"
      }),
      undefined
    )
  })

  it("uses the latest store character id when the hook prop still points at another character", async () => {
    messageStoreState.value = {
      selectedModel: "deepseek-chat",
      serverChatId: "miku-current-chat",
      serverChatCharacterId: 99,
      serverChatAssistantKind: "character",
      serverChatSource: "webui-character-chat"
    }
    const options = {
      ...createHookOptions(),
      serverChatId: "old-character-chat",
      serverChatTitle: "Old character chat",
      serverChatCharacterId: 42,
      serverChatAssistantKind: "character",
      serverChatAssistantId: "42",
      serverChatSource: "webui-character-chat",
      selectedCharacter: {
        id: 99,
        name: "Miku",
        system_prompt: "Miku prompt"
      },
      selectedAssistant: {
        kind: "character",
        id: "99",
        name: "Miku",
        system_prompt: "Miku prompt",
        metadata: { selectionMode: "tracked" }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Continue Miku from the selected greeting",
        image: ""
      })
    })

    expect(createChatMock).not.toHaveBeenCalled()
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "miku-current-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.any(Object)
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "miku-current-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply",
        speaker_character_id: 99,
        speaker_character_name: "Miku"
      }),
      undefined
    )
  })

  it("passes workspace scope through when creating a character-backed chat", async () => {
    const scope = { type: "workspace", workspaceId: "workspace-1" } as const
    const options = {
      ...createHookOptions(),
      scope,
      serverChatId: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      selectedCharacter: null,
      selectedAssistant: {
        kind: "character" as const,
        id: "char-scoped",
        name: "Scoped Character",
        system_prompt: "Scoped prompt",
        metadata: {
          selectionMode: "tracked" as const
        }
      }
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Hello workspace character",
        image: ""
      })
    })

    expect(createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: "char-scoped",
        state: "in-progress"
      }),
      { scope }
    )
    expect(addChatMessageMock).toHaveBeenCalledWith(
      "unexpected-new-chat",
      expect.objectContaining({
        role: "user",
        content: "Hello workspace character"
      }),
      { scope }
    )
    expect(streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "unexpected-new-chat",
      expect.objectContaining({
        include_character_context: true,
        model: "deepseek-chat"
      }),
      expect.objectContaining({ scope })
    )
    expect(persistCharacterCompletionMock).toHaveBeenCalledWith(
      "unexpected-new-chat",
      expect.objectContaining({
        assistant_content: "Tracked reply"
      }),
      { scope }
    )
    expect(normalChatModeMock).not.toHaveBeenCalled()
  })

  it("strips explicit streaming emote directives and persists emote events", async () => {
    streamCharacterChatCompletionMock.mockImplementationOnce(async function* () {
      yield "Em"
      yield "ote: smug\n"
      yield "Hello "
      yield "there.\n"
      yield "Emote: annoyed\n"
      yield "Fine."
    })
    detectCharacterMoodMock.mockReturnValueOnce({
      label: "happy",
      confidence: 0.9,
      topic: "classifier"
    })

    let messagesState: any[] = []
    const messageSnapshots: any[][] = []
    const moodLabels: unknown[] = []
    const options = {
      ...createHookOptions(),
      setMessages: vi.fn((next) => {
        messagesState = typeof next === "function" ? next(messagesState) : next
        messageSnapshots.push(messagesState)
        const assistant = messagesState.find((message) => message?.isBot)
        moodLabels.push(assistant?.moodLabel)
      })
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Test explicit emotes",
        image: ""
      })
    })

    const persistCall = persistCharacterCompletionMock.mock.calls.at(-1) as
      | unknown[]
      | undefined
    const persistPayload = persistCall?.[1] as
      | Record<string, unknown>
      | undefined
    expect(persistPayload).toMatchObject({
      assistant_content: "Hello there.\nFine.",
      mood_label: "annoyed",
      emote_events: [
        { state: "smug", at_char: 0 },
        { state: "annoyed", at_char: 13 }
      ]
    })
    expect(persistPayload).not.toHaveProperty("mood_confidence")
    expect(persistPayload).not.toHaveProperty("mood_topic")
    expect(detectCharacterMoodMock).not.toHaveBeenCalled()

    const renderedAssistantMessages = messageSnapshots
      .flatMap((snapshot) => snapshot.filter((message) => message?.isBot))
      .map((message) => String(message?.message ?? ""))
    expect(
      renderedAssistantMessages.some((message) => message.includes("Emote:"))
    ).toBe(false)
    expect(renderedAssistantMessages.at(-1)).toBe("Hello there.\nFine.")
    expect(moodLabels).toContain("smug")
    expect(moodLabels).toContain("annoyed")
  })

  it("preserves explicit emote metadata when recovering a failed stream", async () => {
    streamCharacterChatCompletionMock.mockImplementationOnce(async function* () {
      yield "Em"
      yield "ote: smug\n"
      yield "Em"
      throw new Error("stream failed")
    })

    let messagesState: any[] = []
    const messageSnapshots: any[][] = []
    const options = {
      ...createHookOptions(),
      setMessages: vi.fn((next) => {
        messagesState = typeof next === "function" ? next(messagesState) : next
        messageSnapshots.push(messagesState)
      })
    }
    const { result } = renderHook(() => useChatActions(options as any))

    await act(async () => {
      await result.current.onSubmit({
        message: "Recover explicit emote",
        image: ""
      })
    })

    const persistCall = persistCharacterCompletionMock.mock.calls.at(-1) as
      | unknown[]
      | undefined
    const persistPayload = persistCall?.[1] as
      | Record<string, unknown>
      | undefined
    expect(persistPayload).toMatchObject({
      assistant_content: "Em",
      mood_label: "smug",
      emote_events: [{ state: "smug", at_char: 0 }]
    })
    expect(persistPayload).not.toHaveProperty("mood_confidence")
    expect(persistPayload).not.toHaveProperty("mood_topic")
    expect(detectCharacterMoodMock).not.toHaveBeenCalled()
    expect(saveLocalErrorMock).toHaveBeenCalledWith(expect.objectContaining({ assistantServerMessageId: "assistant-server-1" }))

    const assistantMessages = messageSnapshots
      .flatMap((snapshot) => snapshot.filter((message) => message?.isBot))
      .map((message) => ({
        message: String(message?.message ?? ""),
        moodLabel: message?.moodLabel,
        metadataExtra: message?.metadataExtra
      }))
    expect(assistantMessages.some((entry) => entry.message.includes("Emote:"))).toBe(
      false
    )
    expect(assistantMessages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          message: "Em",
          moodLabel: "smug",
          metadataExtra: expect.objectContaining({
            emote_events: [{ state: "smug", at_char: 0 }]
          })
        })
      ])
    )
  })

  it(
    "does not duplicate recovery persistence when saved-degraded includes emote metadata",
    async () => {
      streamCharacterChatCompletionMock.mockImplementationOnce(
        async function* () {
          yield "Em"
          yield "ote: smug\n"
          yield "Em"
          throw new Error("stream failed")
        }
      )
      persistCharacterCompletionMock.mockRejectedValueOnce(
        Object.assign(new Error("saved degraded"), {
          status: 503,
          detail: {
            code: "persist_validation_degraded",
            saved: true,
            assistant_message_id: "assistant-degraded-1",
            version: 7
          }
        })
      )

      let messagesState: any[] = []
      const options = {
        ...createHookOptions(),
        setMessages: vi.fn((next) => {
          messagesState =
            typeof next === "function" ? next(messagesState) : next
        })
      }
      const { result } = renderHook(() => useChatActions(options as any))

      await act(async () => {
        await result.current.onSubmit({
          message: "Recover saved degraded emote",
          image: ""
        })
      })

      const tldwAddCalls = addChatMessageMock.mock.calls.filter(
        ([, payload]: [unknown, any]) => payload?.role === "assistant"
      )
      expect(tldwAddCalls).toHaveLength(0)
      expect(saveLocalErrorMock).toHaveBeenCalledWith(expect.objectContaining({ assistantServerMessageId: "assistant-degraded-1" }))
      expect(messagesState).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            isBot: true,
            serverMessageId: "assistant-degraded-1",
            serverMessageVersion: 7,
            moodLabel: "smug",
            metadataExtra: expect.objectContaining({
              emote_events: [{ state: "smug", at_char: 0 }]
            })
          })
        ])
      )
    }
  )

  it("sends emote metadata when normal persist falls back to addChatMessage", async () => {
    streamCharacterChatCompletionMock.mockImplementationOnce(async function* () {
      yield "Em"
      yield "ote: smug\n"
      yield "Hello."
    })
    persistCharacterCompletionMock.mockRejectedValueOnce(
      Object.assign(new Error("persist failed"), { status: 500 })
    )
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined)

    const options = createHookOptions()
    const { result } = renderHook(() => useChatActions(options as any))

    try {
      await act(async () => {
        await result.current.onSubmit({
          message: "Fallback explicit emote",
          image: ""
        })
      })
    } finally {
      consoleErrorSpy.mockRestore()
    }

    const tldwAddCalls = addChatMessageMock.mock.calls
    const assistantAddCall = tldwAddCalls.find(
      ([, payload]: [unknown, any]) => payload?.role === "assistant"
    )
    expect(assistantAddCall?.[1]).toMatchObject({
      role: "assistant",
      content: "Hello.",
      metadata_extra: expect.objectContaining({
        mood_label: "smug",
        mood_confidence: null,
        mood_topic: null,
        emote_events: [{ state: "smug", at_char: 0 }]
      })
    })
  })
})
