import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  addChatMessageMock: vi.fn(),
  createChatMock: vi.fn(),
  detectCharacterMoodMock: vi.fn(),
  generateIDMock: vi.fn(),
  getModelNicknameByIDMock: vi.fn(),
  initializeMock: vi.fn(),
  persistCharacterCompletionMock: vi.fn(),
  resolveApiProviderForModelMock: vi.fn(),
  resolveExplicitProviderForSelectedModelMock: vi.fn(),
  streamCharacterChatCompletionMock: vi.fn(),
  consumeStreamingChunkMock: vi.fn()
}))

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => mocks.generateIDMock()
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: (modelId: string) => mocks.getModelNicknameByIDMock(modelId)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    addChatMessage: (...args: unknown[]) => mocks.addChatMessageMock(...args),
    createChat: (...args: unknown[]) => mocks.createChatMock(...args),
    initialize: () => mocks.initializeMock(),
    persistCharacterCompletion: (...args: unknown[]) =>
      mocks.persistCharacterCompletionMock(...args),
    streamCharacterChatCompletion: (...args: unknown[]) =>
      mocks.streamCharacterChatCompletionMock(...args)
  }
}))

vi.mock("@/utils/character-mood", () => ({
  detectCharacterMood: (...args: unknown[]) => mocks.detectCharacterMoodMock(...args)
}))

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: (...args: unknown[]) =>
    mocks.resolveApiProviderForModelMock(...args),
  resolveExplicitProviderForSelectedModel: (...args: unknown[]) =>
    mocks.resolveExplicitProviderForSelectedModelMock(...args)
}))

vi.mock("@/utils/streaming-chunks", () => ({
  consumeStreamingChunk: (...args: unknown[]) => mocks.consumeStreamingChunkMock(...args)
}))

import { createCharacterChatMode } from "../useCharacterChatMode"

const createSetterBundle = () => ({
  setAbortController: vi.fn(),
  setHistory: vi.fn(),
  setHistoryId: vi.fn(),
  setIsProcessing: vi.fn(),
  setMessages: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatClusterId: vi.fn(),
  setServerChatExternalRef: vi.fn(),
  setServerChatId: vi.fn(),
  setServerChatMetaLoaded: vi.fn(),
  setServerChatSource: vi.fn(),
  setServerChatState: vi.fn(),
  setServerChatTitle: vi.fn(),
  setServerChatTopic: vi.fn(),
  setServerChatVersion: vi.fn(),
  setStreaming: vi.fn()
})

const translate = (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
  if (typeof fallbackOrOptions === "string") return fallbackOrOptions
  return fallbackOrOptions?.defaultValue || key
}

describe("createCharacterChatMode contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    let id = 0
    mocks.generateIDMock.mockImplementation(() => {
      id += 1
      return `generated-${id}`
    })
    mocks.initializeMock.mockResolvedValue(null)
    mocks.getModelNicknameByIDMock.mockResolvedValue(null)
    mocks.resolveExplicitProviderForSelectedModelMock.mockReturnValue("openai")
    mocks.resolveApiProviderForModelMock.mockResolvedValue("openai")
    mocks.createChatMock.mockResolvedValue({
      id: "chat-77",
      title: "Mira session",
      character_id: 42,
      state: "in-progress",
      version: 3
    })
    mocks.addChatMessageMock.mockImplementation(async (_chatId, payload) => {
      const role = (payload as { role?: string })?.role
      return role === "user"
        ? { id: "msg-user-1", version: 1 }
        : { id: "msg-assistant-fallback", version: 2 }
    })
    mocks.persistCharacterCompletionMock.mockResolvedValue({
      assistant_message_id: "msg-assistant-1",
      version: 2
    })
    mocks.streamCharacterChatCompletionMock.mockImplementation(async function* () {
      yield { delta: "Pong" }
    })
    mocks.consumeStreamingChunkMock.mockImplementation((state, chunk) => {
      const token = String((chunk as { delta?: string })?.delta || "")
      return {
        fullText: `${state.fullText}${token}`,
        contentToSave: `${state.contentToSave}${token}`,
        token,
        apiReasoning: state.apiReasoning
      }
    })
    mocks.detectCharacterMoodMock.mockReturnValue({
      label: "neutral",
      confidence: 0.2,
      topic: "reply"
    })
  })

  it("creates a character chat and streams complete-v2 with character context", async () => {
    const setters = createSetterBundle()
    let messagesState: unknown[] = []
    setters.setMessages.mockImplementation((next) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const scope = { type: "workspace", workspaceId: "workspace-1" } as const
    const controller = new AbortController()
    const mode = createCharacterChatMode({
      ...setters,
      t: translate as any,
      notification: { error: vi.fn() },
      selectedCharacter: {
        id: 42,
        name: "Mira",
        avatar_url: "https://example.test/mira.png"
      } as any,
      temporaryChat: false,
      historyId: "history-1",
      serverChatId: null,
      serverChatCharacterId: null,
      serverChatState: "in-progress",
      serverChatTopic: "first-class-roleplay",
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      currentChatModelSettings: {
        apiProvider: "openai",
        setSystemPrompt: vi.fn()
      },
      invalidateServerChatHistory: vi.fn(),
      greetingEnabled: false,
      greetingSelectionId: null,
      greetingsChecksum: null,
      useCharacterDefault: false,
      directedCharacterId: 88,
      resolvedMessageSteeringPrompts: null,
      getEffectiveSelectedModel: vi.fn(() => "tldw:test-model"),
      saveMessageOnSuccess: vi.fn(async () => "history-1"),
      saveMessageOnError: vi.fn(async () => "history-1"),
      discardCurrentTurnOnAbortRef: { current: false },
      scope
    } as any)

    await mode({
      message: "Hello Mira",
      image: "",
      isRegenerate: false,
      messages: [],
      history: [],
      signal: controller.signal,
      model: "tldw:test-model",
      controller,
      messageSteering: {
        continueAsUser: false,
        impersonateUser: false,
        forceNarrate: false
      }
    })

    expect(mocks.createChatMock).toHaveBeenCalledWith(
      expect.objectContaining({
        character_id: 42,
        state: "in-progress",
        topic_label: "first-class-roleplay"
      }),
      { scope }
    )
    expect(mocks.addChatMessageMock).toHaveBeenCalledWith(
      "chat-77",
      {
        role: "user",
        content: "Hello Mira"
      },
      { scope }
    )
    expect(mocks.streamCharacterChatCompletionMock).toHaveBeenCalledWith(
      "chat-77",
      expect.objectContaining({
        include_character_context: true,
        model: "test-model",
        provider: "openai",
        save_to_db: true,
        directed_character_id: 88,
        continue_as_user: false,
        impersonate_user: false,
        force_narrate: false
      }),
      { signal: controller.signal, scope }
    )
    expect(mocks.persistCharacterCompletionMock).toHaveBeenCalledWith(
      "chat-77",
      expect.objectContaining({
        assistant_content: "Pong",
        assistant_message_id: "generated-1",
        speaker_character_id: 88,
        speaker_character_name: "Mira"
      }),
      { scope }
    )
    expect(setters.setServerChatId).toHaveBeenCalledWith("chat-77")
    expect(setters.setServerChatCharacterId).toHaveBeenCalledWith(42)
  })
})
