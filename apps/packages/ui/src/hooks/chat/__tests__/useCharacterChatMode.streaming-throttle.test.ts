// Web character-chat streaming throttle guard — createCharacterChatMode
// (TASK-13521, Batch W1 Stage 1).
//
// useCharacterChatMode.ts (and useChatActions.ts) previously carried a private
// inline copy of the 80ms streaming throttle. Stage 1 extracts that throttle
// into the shared `hooks/chat/streamingThrottle.ts` helper; these tests pin
// the observable contract across the extraction:
//
//   - 500 chunks paced <80ms apart produce at most ceil(elapsed/80)+1
//     streaming message updates (function-form setMessages that change the
//     array) — the path must not regress to per-chunk updates;
//   - a stream ending inside an open window still lands the complete text
//     (trailing flush before the final commit);
//   - aborting mid-stream still lands the partial streamed text as the final
//     message content (error content is the streamed fullText).
//
// Run: cd apps/packages/ui && bun run vitest run src/hooks/chat/__tests__/useCharacterChatMode.streaming-throttle.test.ts
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
import type { Message } from "@/store/option"

const STREAMING_UPDATE_INTERVAL_MS = 80

const metrics = {
  windowActive: false,
  chunks: 0,
  streamingMessageUpdates: 0
}

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

const tokenForChunk = (index: number) => `token-${index} `

const fullTextForChunks = (chunks: number) =>
  Array.from({ length: chunks }, (_, index) => tokenForChunk(index)).join("")

describe("createCharacterChatMode streaming throttle (shared helper adoption)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    metrics.windowActive = false
    metrics.chunks = 0
    metrics.streamingMessageUpdates = 0

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
    mocks.consumeStreamingChunkMock.mockImplementation(
      (state: any, chunk: any) => {
        const token: string = chunk?.choices?.[0]?.delta?.content ?? ""
        const fullText = `${state.fullText}${token}`
        return {
          fullText,
          contentToSave: fullText,
          token,
          apiReasoning: state.apiReasoning
        }
      }
    )
    mocks.detectCharacterMoodMock.mockReturnValue({
      label: "neutral",
      confidence: 0.2,
      topic: "reply"
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  type StreamShape = {
    chunks: number
    paceMs: number
    abortAfterChunk?: { chunk: number; controller: AbortController }
  }

  const installStream = ({ chunks, paceMs, abortAfterChunk }: StreamShape) => {
    mocks.streamCharacterChatCompletionMock.mockImplementation(
      async function* (
        _chatId: unknown,
        _options: unknown,
        transport: { signal: AbortSignal }
      ) {
        metrics.windowActive = true
        try {
          for (let index = 0; index < chunks; index += 1) {
            if (paceMs > 0) await vi.advanceTimersByTimeAsync(paceMs)
            if (
              abortAfterChunk &&
              index === abortAfterChunk.chunk &&
              !transport.signal.aborted
            ) {
              // Abort before serving the next chunk: an aborted transport
              // stops streaming, so the consumer sees the stream end right
              // after the chunk that tripped the abort.
              abortAfterChunk.controller.abort()
              return
            }
            metrics.chunks += 1
            yield { choices: [{ delta: { content: tokenForChunk(index) } }] }
          }
        } finally {
          metrics.windowActive = false
        }
      }
    )
  }

  const mountMode = () => {
    const setters = createSetterBundle()
    let messagesState: Message[] = []
    setters.setMessages.mockImplementation(
      (next: Message[] | ((prev: Message[]) => Message[])) => {
        const resolved =
          typeof next === "function"
            ? (next as (prev: Message[]) => Message[])(messagesState)
            : next
        if (metrics.windowActive && resolved !== messagesState) {
          metrics.streamingMessageUpdates += 1
        }
        messagesState = resolved
      }
    )
    const mode = createCharacterChatMode({
      ...setters,
      t: translate as any,
      notification: { error: vi.fn() },
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
      saveMessageOnError: vi.fn(async (_payload: any) => null),
      discardCurrentTurnOnAbortRef: { current: false }
    } as any)
    return {
      setters,
      mode,
      messagesState: () => messagesState
    }
  }

  const runTurn = (mode: unknown, controller: AbortController) =>
    (mode as any)({
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

  it("coalesces 500 chunks <80ms apart into at most ceil(elapsed/80)+1 streaming updates", async () => {
    const CHUNKS = 500
    const PACE_MS = 8
    const elapsedMs = CHUNKS * PACE_MS
    installStream({ chunks: CHUNKS, paceMs: PACE_MS })

    const { mode, messagesState } = mountMode()
    await vi.advanceTimersByTimeAsync(0)
    await runTurn(mode, new AbortController())

    const maxUpdates =
      Math.ceil(elapsedMs / STREAMING_UPDATE_INTERVAL_MS) + 1
    expect(metrics.chunks).toBe(CHUNKS)
    expect(metrics.streamingMessageUpdates).toBeLessThanOrEqual(maxUpdates)
    expect(metrics.streamingMessageUpdates).toBeGreaterThanOrEqual(
      Math.floor(elapsedMs / STREAMING_UPDATE_INTERVAL_MS) - 1
    )
    expect(messagesState().at(-1)?.message).toBe(fullTextForChunks(CHUNKS))
  })

  it("lands the complete text when the stream ends inside an open window", async () => {
    const CHUNKS = 120
    // No clock advance: every chunk lands inside the first 80ms window, so
    // the complete text only lands through the trailing flush.
    installStream({ chunks: CHUNKS, paceMs: 0 })

    const { mode, messagesState } = mountMode()
    await vi.advanceTimersByTimeAsync(0)
    await runTurn(mode, new AbortController())

    expect(metrics.chunks).toBe(CHUNKS)
    // No throttled flush fired mid-stream; the final content still complete.
    expect(metrics.streamingMessageUpdates).toBe(0)
    expect(messagesState().at(-1)?.message).toBe(fullTextForChunks(CHUNKS))
  })

  it("aborts mid-stream and still lands the partial streamed text", async () => {
    const ABORT_AFTER = 250
    const controller = new AbortController()
    installStream({ chunks: 500, paceMs: 8, abortAfterChunk: { chunk: ABORT_AFTER, controller } })

    const { mode, messagesState } = mountMode()
    await vi.advanceTimersByTimeAsync(0)
    await runTurn(mode, controller)

    expect(metrics.chunks).toBe(ABORT_AFTER)
    // buildCharacterChatAssistantErrorContent(fullText) === fullText when the
    // partial text is non-empty: the abort must not lose streamed content.
    expect(messagesState().at(-1)?.message).toBe(
      fullTextForChunks(ABORT_AFTER)
    )
  })
})
