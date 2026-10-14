// Sidepanel streaming throttle tests — real `useMessage` hook (TASK-13521, Batch W1 Stage 1).
//
// Drives the production character streaming path in `hooks/useMessage.tsx`
// (same harness shape as `streaming-render.bench.test.ts`, the W0 baseline
// harness) and pins the Stage-1 behavior the plan requires:
//
//   (a) 500 chunks paced <80ms apart produce at most ceil(elapsed/80)+1
//       message-array store updates during the streaming window, the final
//       message content equals the exact concatenation of all chunks, and
//       non-streaming messages keep referential identity;
//   (b) a stream that ends inside an open 80ms window still lands the complete
//       text (trailing flush before the final commit);
//   (c) a second stream after the first still updates correctly.
//
// Additionally asserts the narrowed update itself: no transcript-array
// `.map` clone may run while chunks are streaming (last-index replacement
// via `slice` instead of `prev.map`).
//
// Run: cd apps/packages/ui && bun run vitest run src/hooks/__tests__/useMessage.streaming-throttle.test.ts
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useMessage } from "../useMessage"
import { useStoreMessageOption, type Message } from "~/store/option"

const { SEED_MESSAGES, metrics, benchCharacter } = vi.hoisted(() => ({
  SEED_MESSAGES: 200,
  metrics: {
    windowActive: false,
    chunks: 0,
    messageUpdates: 0,
    transcriptArrayMapClones: 0
  },
  benchCharacter: { id: 42, name: "Bench Character" }
}))

const {
  streamCharacterChatCompletionMock,
  persistCharacterCompletionMock
} = vi.hoisted(() => ({
  streamCharacterChatCompletionMock: vi.fn(),
  persistCharacterCompletionMock: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({ invalidateQueries: vi.fn() })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@/context", () => ({
  usePageAssist: () => ({
    controller: null,
    setController: vi.fn(),
    embeddingController: null,
    setEmbeddingController: vi.fn()
  })
}))

vi.mock("~/store", () => ({
  useStoreMessage: () => ({ currentURL: "", setCurrentURL: vi.fn() })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => ({
    selectedModel: "bench-model",
    setSelectedModel: vi.fn()
  })
}))

vi.mock("@/store/model", () => {
  const state = {
    apiProvider: "provider-1",
    reset: vi.fn(),
    activeSettingsScope: "global",
    getEffectiveSettings: () => ({ apiProvider: "provider-1" })
  }
  return {
    useStoreChatModelSettings: Object.assign(() => state, {
      getState: () => state
    })
  }
})

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [benchCharacter, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn()],
  getSelectedAssistantOperationRevision: () => 0
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    success: vi.fn()
  })
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null })
}))

// Force the send-mode router onto the tracked-character sidepanel path so the
// production character streaming loop in useMessage.tsx runs.
vi.mock("@/hooks/chat/effective-assistant-state", () => ({
  resolveEffectiveAssistantState: () => ({
    mode: "tracked_character",
    kind: "character",
    id: "42",
    displayName: "Bench Character",
    avatarUrl: null,
    systemPromptSnapshot: null
  })
}))

vi.mock("@/services/chat-loop/hooks", () => ({
  useChatLoopState: () => ({ state: {}, dispatch: vi.fn(), reset: vi.fn() })
}))

vi.mock("@/services/chat-loop/bridge", () => ({
  subscribeChatLoopEvents: () => vi.fn(),
  publishChatLoopEvent: vi.fn()
}))

vi.mock("@/utils/chat-model-validation", () => ({
  validateSelectedChatModelAvailability: vi.fn(async () => ({ status: "valid" }))
}))

// No image backends: keeps onSubmit on the character streaming path.
vi.mock("@/utils/image-backends", () => ({
  resolveImageBackendCandidates: () => []
}))

vi.mock("@/db/dexie/nickname", () => ({
  getModelNicknameByID: vi.fn(async () => null)
}))

// The persistence tail runs after the measured window; keep it deterministic.
vi.mock("@/hooks/utils/messageHelpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/hooks/utils/messageHelpers")>()),
  createSaveMessageOnSuccess: () => vi.fn(async () => "bench-history"),
  createSaveMessageOnError: () => vi.fn(async () => "bench-history")
}))

vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<any>()),
  useHistorySelectionContext: () => null
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    scopeKey: "scope:bench",
    requestScope: {
      config: { serverUrl: "http://127.0.0.1:8000", authMode: "single-user" },
      userId: null
    },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    definitions: {},
    capability: "unchecked",
    release: vi.fn()
  }),
  renderServicePromptPart: () => ""
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => ({ id: "bench-chat", character_id: 42 })),
    getCharacter: vi.fn(async () => ({ id: 42, name: "Bench Character" })),
    createChat: vi.fn(async () => ({ id: "bench-chat", character_id: 42 })),
    addChatMessage: vi.fn(async () => ({ id: "bench-user-1", version: 1 })),
    streamCharacterChatCompletion: streamCharacterChatCompletionMock,
    persistCharacterCompletion: persistCharacterCompletionMock
  }
}))

const STREAMING_UPDATE_INTERVAL_MS = 80

const seedMessages = (): Message[] =>
  Array.from({ length: SEED_MESSAGES }, (_, index) => {
    const isBot = index % 2 === 1
    return {
      id: `seed-${index}`,
      isBot,
      role: isBot ? ("assistant" as const) : ("user" as const),
      name: isBot ? "Bench Character" : "You",
      message: `Seed message ${index}: ${"lorem ipsum dolor sit amet ".repeat(4)}`,
      sources: [],
      createdAt: 1_700_000_000_000 + index,
      parentMessageId: index > 0 ? `seed-${index - 1}` : null
    }
  })

const tokenForChunk = (index: number) => `token-${index} `

const fullTextForChunks = (chunks: number) =>
  Array.from({ length: chunks }, (_, index) => tokenForChunk(index)).join("")

type StreamShape = { chunks: number; paceMs: number }

const installStream = ({ chunks, paceMs }: StreamShape) => {
  streamCharacterChatCompletionMock.mockImplementation(async function* () {
    metrics.windowActive = true
    try {
      for (let index = 0; index < chunks; index += 1) {
        if (paceMs > 0) await vi.advanceTimersByTimeAsync(paceMs)
        metrics.chunks += 1
        yield { choices: [{ delta: { content: tokenForChunk(index) } }] }
      }
    } finally {
      // The streaming window closes when the generator is exhausted, before
      // the post-loop trailing flush and final commit run.
      metrics.windowActive = false
    }
  })
}

describe("useMessage character streaming throttle (sidepanel path)", () => {
  let seeds: Message[]
  let unsubscribeStore: () => void
  let arrayMapSpy: ReturnType<typeof vi.spyOn>
  let unmountHook: () => void

  beforeEach(() => {
    vi.useFakeTimers()
    metrics.windowActive = false
    metrics.chunks = 0
    metrics.messageUpdates = 0
    metrics.transcriptArrayMapClones = 0
    persistCharacterCompletionMock.mockImplementation(async () => {
      return { assistant_message_id: "bench-assistant-1", version: 1 }
    })
    seeds = seedMessages()
    useStoreMessageOption.setState({
      messages: seeds,
      history: [
        { role: "user", content: "bench question" },
        { role: "assistant", content: "bench answer" }
      ],
      historyId: "bench-history",
      isFirstMessage: false,
      chatMode: "normal",
      selectedModel: "bench-model",
      serverChatId: "bench-chat",
      serverChatCharacterId: 42,
      serverChatAssistantKind: "character",
      serverChatMetaLoaded: true,
      temporaryChat: false,
      webSearch: false,
      streaming: false,
      isProcessing: false
    })
  })

  afterEach(() => {
    unsubscribeStore?.()
    unmountHook?.()
    arrayMapSpy?.mockRestore()
    vi.useRealTimers()
  })

  const mountInstrumentedHook = () => {
    // Live transcript length during streaming: 200 seeded + user + assistant.
    const liveTranscriptLength = SEED_MESSAGES + 2

    unsubscribeStore = useStoreMessageOption.subscribe((state, previous) => {
      if (!metrics.windowActive) return
      if (state.messages !== previous.messages) metrics.messageUpdates += 1
    })

    const unmodifiedArrayMap = Array.prototype.map as unknown as (
      this: unknown[],
      mapper: (value: never, index: number, array: unknown[]) => unknown,
      thisArg?: unknown
    ) => unknown[]
    arrayMapSpy = vi
      .spyOn(Array.prototype, "map")
      .mockImplementation(function (this: unknown[], mapper, thisArg) {
        if (
          metrics.windowActive &&
          this.length === liveTranscriptLength
        ) {
          metrics.transcriptArrayMapClones += 1
        }
        return unmodifiedArrayMap.call(this, mapper as never, thisArg)
      })

    const hook = renderHook(() => useMessage())
    unmountHook = hook.unmount
    return hook
  }

  it("(a) coalesces 500 chunks <80ms apart into at most ceil(elapsed/80)+1 updates", async () => {
    const CHUNKS = 500
    const PACE_MS = 8
    const elapsedMs = CHUNKS * PACE_MS
    installStream({ chunks: CHUNKS, paceMs: PACE_MS })

    const { result } = mountInstrumentedHook()
    await act(async () => {
      await result.current.onSubmit({ message: "bench question", image: "" })
    })

    const maxUpdates =
      Math.ceil(elapsedMs / STREAMING_UPDATE_INTERVAL_MS) + 1
    expect(metrics.chunks).toBe(CHUNKS)
    expect(metrics.messageUpdates).toBeLessThanOrEqual(maxUpdates)
    // The text must still stream (not batch into nothing) mid-stream.
    expect(metrics.messageUpdates).toBeGreaterThanOrEqual(
      Math.floor(elapsedMs / STREAMING_UPDATE_INTERVAL_MS) - 1
    )

    const finalMessages = useStoreMessageOption.getState().messages
    expect(finalMessages).toHaveLength(SEED_MESSAGES + 2)
    expect(finalMessages.at(-1)?.message).toBe(fullTextForChunks(CHUNKS))
    // Non-streaming messages keep referential identity.
    for (let index = 0; index < SEED_MESSAGES; index += 1) {
      expect(finalMessages[index]).toBe(seeds[index])
    }
  })

  it("(a-narrowed) performs zero transcript-array .map clones while chunks stream", async () => {
    const CHUNKS = 500
    const PACE_MS = 8
    installStream({ chunks: CHUNKS, paceMs: PACE_MS })

    const { result } = mountInstrumentedHook()
    await act(async () => {
      await result.current.onSubmit({ message: "bench question", image: "" })
    })

    expect(metrics.chunks).toBe(CHUNKS)
    expect(metrics.transcriptArrayMapClones).toBe(0)
  })

  it("(b) a stream ending inside an open window still lands the complete text", async () => {
    // All chunks arrive without advancing the clock: the 80ms timer never
    // fires mid-stream, so the complete text only lands through the trailing
    // flush before the final commit.
    const CHUNKS = 100
    installStream({ chunks: CHUNKS, paceMs: 0 })

    const { result } = mountInstrumentedHook()
    await act(async () => {
      await result.current.onSubmit({ message: "bench question", image: "" })
    })

    expect(metrics.chunks).toBe(CHUNKS)
    expect(metrics.messageUpdates).toBe(0)
    expect(metrics.transcriptArrayMapClones).toBe(0)
    const finalMessages = useStoreMessageOption.getState().messages
    expect(finalMessages.at(-1)?.message).toBe(fullTextForChunks(CHUNKS))
    expect(finalMessages).toHaveLength(SEED_MESSAGES + 2)
  })

  it("(c) a second stream after the first still updates correctly", async () => {
    const FIRST_CHUNKS = 40
    const SECOND_CHUNKS = 60
    installStream({ chunks: FIRST_CHUNKS, paceMs: 0 })

    const { result } = mountInstrumentedHook()
    await act(async () => {
      await result.current.onSubmit({ message: "first question", image: "" })
    })
    const afterFirst = useStoreMessageOption.getState().messages
    expect(afterFirst).toHaveLength(SEED_MESSAGES + 2)
    const firstAssistant = afterFirst.at(-1)
    expect(firstAssistant?.message).toBe(fullTextForChunks(FIRST_CHUNKS))

    installStream({ chunks: SECOND_CHUNKS, paceMs: 0 })
    await act(async () => {
      await result.current.onSubmit({ message: "second question", image: "" })
    })

    const afterSecond = useStoreMessageOption.getState().messages
    expect(afterSecond).toHaveLength(SEED_MESSAGES + 4)
    // The first assistant message keeps its identity and content after the
    // second stream (its object reference may legitimately change once, for
    // the post-persist server-id attach).
    expect(afterSecond[SEED_MESSAGES + 1]?.id).toBe(firstAssistant?.id)
    expect(afterSecond[SEED_MESSAGES + 1]?.message).toBe(
      fullTextForChunks(FIRST_CHUNKS)
    )
    expect(afterSecond.at(-1)?.message).toBe(fullTextForChunks(SECOND_CHUNKS))
  })
})
