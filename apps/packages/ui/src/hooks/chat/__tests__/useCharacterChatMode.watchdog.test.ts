import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

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

// Must match the watchdog value baked into useCharacterChatMode.ts and the two
// live inline copies (useChatActions.ts, useMessage.tsx).
const STREAM_INACTIVITY_TIMEOUT_MS = 60_000

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

const translate = (
  key: string,
  fallbackOrOptions?: string | { defaultValue?: string; name?: string }
) =>
  typeof fallbackOrOptions === "string"
    ? fallbackOrOptions
    : fallbackOrOptions?.defaultValue || key

describe("createCharacterChatMode stream-inactivity watchdog", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()

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
    mocks.addChatMessageMock.mockResolvedValue({ id: "msg-user-1", version: 1 })
    mocks.persistCharacterCompletionMock.mockResolvedValue({
      assistant_message_id: "msg-assistant-1",
      version: 2
    })
    mocks.consumeStreamingChunkMock.mockImplementation((state) => ({
      fullText: state.fullText,
      contentToSave: state.contentToSave,
      token: "",
      apiReasoning: state.apiReasoning
    }))
    mocks.detectCharacterMoodMock.mockReturnValue({
      label: "neutral",
      confidence: 0.2,
      topic: "reply"
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("aborts a stalled stream after the inactivity timeout instead of hanging", async () => {
    // A stream that never yields a chunk. It only settles once the watchdog
    // aborts the shared controller, mirroring how a real fetch-backed stream
    // unwinds on abort.
    mocks.streamCharacterChatCompletionMock.mockImplementation(
      async function* (
        _chatId: unknown,
        _options: unknown,
        transport: { signal: AbortSignal }
      ) {
        const { signal } = transport
        await new Promise<void>((resolve) => {
          if (signal.aborted) {
            resolve()
            return
          }
          signal.addEventListener("abort", () => resolve(), { once: true })
        })
        // Never yields: the stalled stream ends only because it was aborted.
      }
    )

    const setters = createSetterBundle()
    let messagesState: any[] = []
    setters.setMessages.mockImplementation((next) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const saveMessageOnError = vi.fn(async (_payload: any) => null)
    const notification = { error: vi.fn() }
    const controller = new AbortController()

    const mode = createCharacterChatMode({
      ...setters,
      t: translate as any,
      notification,
      selectedCharacter: { id: 42, name: "Mira" } as any,
      temporaryChat: false,
      historyId: "history-1",
      serverChatId: null,
      serverChatCharacterId: null,
      serverChatState: "in-progress",
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null,
      currentChatModelSettings: { apiProvider: "openai", setSystemPrompt: vi.fn() },
      invalidateServerChatHistory: vi.fn(),
      greetingEnabled: false,
      greetingSelectionId: null,
      greetingsChecksum: null,
      useCharacterDefault: false,
      directedCharacterId: null,
      resolvedMessageSteeringPrompts: null,
      getEffectiveSelectedModel: vi.fn(() => "tldw:test-model"),
      saveMessageOnSuccess: vi.fn(async () => "history-1"),
      saveMessageOnError,
      discardCurrentTurnOnAbortRef: { current: false }
    } as any)

    const runPromise = mode({
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

    // Let the async setup (initialize/createChat/addChatMessage) settle and the
    // watchdog timer get scheduled before we advance fake time.
    await vi.advanceTimersByTimeAsync(0)
    expect(controller.signal.aborted).toBe(false)
    expect(saveMessageOnError).not.toHaveBeenCalled()

    // Fast-forward past the inactivity window: the watchdog must fire.
    await vi.advanceTimersByTimeAsync(STREAM_INACTIVITY_TIMEOUT_MS)
    await runPromise

    expect(controller.signal.aborted).toBe(true)
    expect(saveMessageOnError).toHaveBeenCalledTimes(1)
    const errorArg = saveMessageOnError.mock.calls[0][0]?.e
    expect((errorArg as any)?.name).toBe("StreamInactivityTimeout")
    expect(notification.error).toHaveBeenCalledWith(
      expect.objectContaining({ message: "Stream timed out" })
    )
    expect(setters.setStreaming).toHaveBeenLastCalledWith(false)
  })
})
