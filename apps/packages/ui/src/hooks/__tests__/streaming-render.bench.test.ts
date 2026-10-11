// Streaming-render benchmark — sidepanel chat path (TASK-13520, Batch W0 Stage 1).
//
// Measures how much store work each streamed token triggers when the sidepanel
// chat (`useMessage.tsx` character streaming loop) renders a live completion:
// the chunk handler currently runs `setMessages((prev) => prev.map(...))` per
// token, cloning the whole transcript array and notifying every
// `useStoreMessageOption` subscriber on every chunk.
//
// Method:
//   - Seeds the REAL `useStoreMessageOption` zustand store with N=200 messages.
//   - Renders the real `useMessage` hook and submits one character-chat turn.
//   - `tldwClient.streamCharacterChatCompletion` yields 500 deterministic
//     token chunks; everything else in the hook is production code.
//   - The measurement window opens at the first chunk pull and closes when
//     post-stream persistence starts (`persistCharacterCompletion`).
//   - Counts (a) store-update invocations (subscriber notifications on
//     `useStoreMessageOption` where the `messages` array identity changed) and
//     (b) transcript array clones (spy on `Array.prototype.map` invoked on an
//     array of the live transcript length).
//
// This is a measurement harness, not a regression gate: assertions only
// sanity-check that the counters are finite, non-negative and non-zero and
// that the harness drove the intended path. Perf thresholds live in
// Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md (Stage 3 baseline document);
// batch W1 re-runs this bench after fixing the per-token work and records the
// before/after delta there.
//
// Run: cd apps/packages/ui && bun run vitest run src/hooks/__tests__/streaming-render.bench.test.ts
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useMessage } from "../useMessage"
import { useStoreMessageOption, type Message } from "~/store/option"

const { SEED_MESSAGES, CHUNKS, bench, benchCharacter } = vi.hoisted(() => ({
  // Plan constants (Stage 1): N=200 seeded messages, 500 streamed token chunks.
  SEED_MESSAGES: 200,
  CHUNKS: 500,
  bench: {
    counting: false,
    chunks: 0,
    storeNotifications: 0,
    messageUpdateNotifications: 0,
    transcriptArrayMapClones: 0,
    streamStartedAt: 0,
    streamEndedAt: 0
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
// production `characterChatMode` streaming loop in useMessage.tsx runs.
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

describe("streaming render bench (sidepanel path)", () => {
  beforeEach(() => {
    bench.counting = false
    bench.chunks = 0
    bench.storeNotifications = 0
    bench.messageUpdateNotifications = 0
    bench.transcriptArrayMapClones = 0
    bench.streamStartedAt = 0
    bench.streamEndedAt = 0
    streamCharacterChatCompletionMock.mockImplementation(async function* () {
      bench.streamStartedAt = performance.now()
      bench.counting = true
      for (let index = 0; index < CHUNKS; index += 1) {
        bench.chunks += 1
        yield { choices: [{ delta: { content: tokenForChunk(index) } }] }
      }
    })
    persistCharacterCompletionMock.mockImplementation(async () => {
      // Post-stream persistence starts here: close the measurement window.
      bench.counting = false
      bench.streamEndedAt = performance.now()
      return { assistant_message_id: "bench-assistant-1", version: 1 }
    })
    useStoreMessageOption.setState({
      messages: seedMessages(),
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

  it("records store-update and array-clone counts for 500 streamed tokens over 200 seeded messages", async () => {
    // Live transcript length during streaming: 200 seeded + user turn + assistant stub.
    const liveTranscriptLength = SEED_MESSAGES + 2

    const unsubscribe = useStoreMessageOption.subscribe((state, previous) => {
      if (!bench.counting) return
      bench.storeNotifications += 1
      if (state.messages !== previous.messages) {
        bench.messageUpdateNotifications += 1
      }
    })

    const unmodifiedArrayMap = Array.prototype.map as unknown as (
      this: unknown[],
      mapper: (value: never, index: number, array: unknown[]) => unknown,
      thisArg?: unknown
    ) => unknown[]
    const arrayMapSpy = vi
      .spyOn(Array.prototype, "map")
      .mockImplementation(function (this: unknown[], mapper, thisArg) {
        if (bench.counting && this.length === liveTranscriptLength) {
          bench.transcriptArrayMapClones += 1
        }
        return unmodifiedArrayMap.call(this, mapper as never, thisArg)
      })

    const { result, unmount } = renderHook(() => useMessage())
    try {
      await act(async () => {
        await result.current.onSubmit({ message: "bench question", image: "" })
      })

      const streamPhaseMs = bench.streamEndedAt - bench.streamStartedAt
      const finalMessages = useStoreMessageOption.getState().messages
      const finalAssistantMessage = finalMessages.at(-1)
      const finalAssistantChars =
        typeof finalAssistantMessage?.message === "string"
          ? finalAssistantMessage.message.length
          : 0
      const metrics = {
        seedMessages: SEED_MESSAGES,
        chunksStreamed: bench.chunks,
        storeSubscriberNotifications: bench.storeNotifications,
        messageUpdateNotifications: bench.messageUpdateNotifications,
        transcriptArrayMapClones: bench.transcriptArrayMapClones,
        streamPhaseMs: Number(streamPhaseMs.toFixed(2)),
        msPerChunk: Number((streamPhaseMs / bench.chunks).toFixed(4)),
        finalAssistantChars
      }
      console.log(`[streaming-render.bench] ${JSON.stringify(metrics)}`)

      // Harness-correctness assertions (not perf thresholds).
      expect(streamCharacterChatCompletionMock).toHaveBeenCalledTimes(1)
      expect(bench.chunks).toBe(CHUNKS)
      expect(finalMessages).toHaveLength(liveTranscriptLength)
      expect(finalAssistantChars).toBe(
        Array.from({ length: CHUNKS }, (_, index) => tokenForChunk(index))
          .join("")
          .length
      )

      // Measurement-recorded sanity assertions: finite, non-negative, non-zero.
      expect(Number.isFinite(bench.storeNotifications)).toBe(true)
      expect(Number.isFinite(bench.messageUpdateNotifications)).toBe(true)
      expect(Number.isFinite(bench.transcriptArrayMapClones)).toBe(true)
      expect(Number.isFinite(streamPhaseMs)).toBe(true)
      expect(bench.storeNotifications).toBeGreaterThan(0)
      expect(bench.messageUpdateNotifications).toBeGreaterThan(0)
      expect(bench.transcriptArrayMapClones).toBeGreaterThan(0)
      expect(streamPhaseMs).toBeGreaterThanOrEqual(0)
      expect(arrayMapSpy).toHaveBeenCalled()
    } finally {
      unsubscribe()
      unmount()
      arrayMapSpy.mockRestore()
    }
  })
})
