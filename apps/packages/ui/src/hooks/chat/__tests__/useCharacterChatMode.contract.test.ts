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

import { decodeChatErrorPayload } from "@/utils/chat-error-message"

import {
  classifyCharacterChatFailureRecovery,
  createCharacterChatMode
} from "../useCharacterChatMode"

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
  fallbackOrOptions?: string | { defaultValue?: string; name?: string },
  options?: { name?: string }
) => {
  const value =
    typeof fallbackOrOptions === "string"
      ? fallbackOrOptions
      : fallbackOrOptions?.defaultValue || key
  const name =
    options?.name ||
    (typeof fallbackOrOptions === "object" ? fallbackOrOptions.name : undefined)
  return typeof name === "string" ? value.replace("{{name}}", name) : value
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
        topic_label: "first-class-roleplay",
        source: "webui-character-chat",
        title: "Mira: Hello Mira"
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

  it("classifies structured provider setup failures without treating every 503 as configuration", () => {
    const providerSetupError = Object.assign(
      new Error("provider_not_configured: OpenAI API key is missing"),
      {
        status: 503,
        response: {
          data: {
            code: "provider_not_configured",
            detail: "OpenAI API key is missing"
          }
        }
      }
    )

    expect(classifyCharacterChatFailureRecovery(providerSetupError)).toMatchObject({
      kind: "provider_unconfigured",
      action: "open-model-settings"
    })

    const transient503 = Object.assign(new Error("Service unavailable"), {
      status: 503
    })

    expect(classifyCharacterChatFailureRecovery(transient503)).toMatchObject({
      kind: "transient",
      action: "retry"
    })
  })

  it("bounds and redacts persisted character chat failure details", () => {
    const largeProviderBody = [
      "provider_not_configured",
      "Bearer sk-secret-token-1234567890",
      "x".repeat(6000)
    ].join(" ")
    const providerSetupError = Object.assign(
      new Error("provider_not_configured: OpenAI API key is missing"),
      {
        status: 503,
        response: {
          data: {
            code: "provider_not_configured",
            body: largeProviderBody
          }
        }
      }
    )

    const recovery = classifyCharacterChatFailureRecovery(providerSetupError)

    expect(recovery.kind).toBe("provider_unconfigured")
    expect(recovery.detail.length).toBeLessThanOrEqual(3000)
    expect(recovery.detail).toContain("provider_not_configured")
    expect(recovery.detail).toContain("Bearer [redacted]")
    expect(recovery.detail).not.toContain("sk-secret-token-1234567890")
  })

  it("uses translated character chat recovery copy when a translator is supplied", () => {
    const providerSetupError = Object.assign(
      new Error("provider_not_configured"),
      {
        response: {
          data: {
            code: "provider_not_configured"
          }
        }
      }
    )
    const t = vi.fn((key: string, options?: { defaultValue?: string }) =>
      key.endsWith("providerUnconfigured.summary")
        ? "Translated provider setup summary"
        : options?.defaultValue ?? key
    )

    const recovery = classifyCharacterChatFailureRecovery(
      providerSetupError,
      t as any
    )

    expect(recovery.summary).toBe("Translated provider setup summary")
    expect(t).toHaveBeenCalledWith(
      "playground:characterChatFailure.providerUnconfigured.summary",
      expect.objectContaining({
        defaultValue: "Character chat model setup needs attention."
      })
    )
  })

  it("maps provider setup stream failures to model-settings recovery copy", async () => {
    const setters = createSetterBundle()
    let messagesState: any[] = []
    setters.setMessages.mockImplementation((next) => {
      messagesState = typeof next === "function" ? next(messagesState) : next
    })
    const providerSetupError = Object.assign(
      new Error("provider_not_configured: OpenAI API key is missing"),
      {
        status: 503,
        response: {
          data: {
            code: "provider_not_configured",
            detail: "OpenAI API key is missing"
          }
        }
      }
    )
    mocks.streamCharacterChatCompletionMock.mockImplementation(async function* () {
      yield* []
      throw providerSetupError
    })
    const saveMessageOnError = vi.fn(async () => "history-1")
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
      getEffectiveSelectedModel: vi.fn(() => "tldw:gpt-4o"),
      saveMessageOnSuccess: vi.fn(async () => "history-1"),
      saveMessageOnError,
      discardCurrentTurnOnAbortRef: { current: false }
    } as any)

    await mode({
      message: "Hello Mira",
      image: "",
      isRegenerate: false,
      messages: [],
      history: [],
      signal: controller.signal,
      model: "tldw:gpt-4o",
      controller,
      messageSteering: {
        continueAsUser: false,
        impersonateUser: false,
        forceNarrate: false
      }
    })

    const assistantError = messagesState.find(
      (entry) => entry?.isBot && entry?.id === "generated-1"
    )
    const payload = decodeChatErrorPayload(String(assistantError?.message || ""))

    expect(payload).toMatchObject({
      summary: "Character chat model setup needs attention.",
      recoveryAction: "open-model-settings",
      recoveryLabel: "Open model settings"
    })
    expect(payload?.hint).toContain("Open model settings")
    expect(payload?.detail).toContain("provider_not_configured")
    expect(saveMessageOnError).toHaveBeenCalledWith(
      expect.objectContaining({
        botMessage: assistantError?.message,
        userMessage: "Hello Mira",
        selectedModel: "tldw:gpt-4o"
      })
    )
    expect(messagesState).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ isBot: false, message: "Hello Mira" }),
        expect.objectContaining({ isBot: true, id: "generated-1" })
      ])
    )
  })
})
