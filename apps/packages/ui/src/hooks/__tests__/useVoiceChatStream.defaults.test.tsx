import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useVoiceChatStream } from "@/hooks/useVoiceChatStream"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const DEFAULT_KITTEN_MODEL = "KittenML/kitten-tts-nano-0.8"
const DEFAULT_KITTEN_VOICE = "Bella"

const storageState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))

const audioCatalogState = vi.hoisted(() => ({
  devices: [] as Array<{ deviceId: string; label: string }>,
  isSettled: true
}))

const audioSourcePreferenceState = vi.hoisted(() => ({
  preference: {
    featureGroup: "live_voice" as const,
    sourceKind: "default_mic" as const,
    deviceId: null,
    lastKnownLabel: null
  },
  isLoading: false
}))

const audioPlayerState = vi.hoisted(() => ({
  start: vi.fn(() => {}),
  append: vi.fn(() => {}),
  finish: vi.fn(() => {}),
  stop: vi.fn(() => {}),
  state: { playing: false }
}))

const micState = vi.hoisted(() => ({
  useMicStream: vi.fn(),
  start: vi.fn(async () => {}),
  stop: vi.fn(() => {}),
  active: false
}))

const voiceChatState = vi.hoisted(() => ({
  voiceChatModel: "",
  voiceChatPauseMs: 700,
  voiceChatTriggerPhrases: [],
  voiceChatAutoResume: true,
  voiceChatBargeIn: false,
  voiceChatTtsMode: "stream"
}))

const sttState = vi.hoisted(() => ({
  model: "whisper-1"
}))

const selectedModelState = vi.hoisted(() => ({
  selectedModel: null as string | null
}))

const buildState = vi.hoisted(() => ({
  buildVoiceConversationPreflight: vi.fn(async () => ({
    websocketUrl: "ws://example.test/voice",
    llm: {},
    tts: {
      provider: "tldw",
      format: "mp3"
    }
  })),
  normalizeVoiceConversationRuntimeError: vi.fn((error: Error) => error)
}))

const resolveProviderState = vi.hoisted(() => ({
  resolveApiProviderForModel: vi.fn(async () => "stub")
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.values.has(key) ? storageState.values.get(key) : defaultValue,
    vi.fn(),
    {}
  ]
}))

vi.mock("@/hooks/useAudioSourceCatalog", () => ({
  useAudioSourceCatalog: () => audioCatalogState
}))

vi.mock("@/hooks/useAudioSourcePreferences", () => ({
  useAudioSourcePreferences: () => audioSourcePreferenceState
}))

vi.mock("@/hooks/useSttSettings", () => ({
  useSttSettings: () => ({
    model: sttState.model,
    temperature: 0,
    task: "transcribe",
    responseFormat: "json",
    timestampGranularities: "word",
    prompt: "",
    useSegmentation: false,
    segK: 2,
    segMinSegmentSize: 20,
    segLambdaBalance: 0.5,
    segUtteranceExpansionWidth: 2,
    segEmbeddingsProvider: "",
    segEmbeddingsModel: ""
  })
}))

vi.mock("@/hooks/useVoiceChatSettings", () => ({
  useVoiceChatSettings: () => voiceChatState
}))

vi.mock("@/hooks/chat/useSelectedModel", () => ({
  useSelectedModel: () => selectedModelState
}))

vi.mock("@/hooks/useMicStream", () => ({
  useMicStream: micState.useMicStream
}))

vi.mock("@/hooks/useStreamingAudioPlayer", () => ({
  useStreamingAudioPlayer: () => ({
    start: audioPlayerState.start,
    append: audioPlayerState.append,
    finish: audioPlayerState.finish,
    stop: audioPlayerState.stop,
    state: audioPlayerState.state
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(async () => ({
      serverUrl: "http://localhost:8000",
      authMode: "single_user",
      apiKey: "test-key"
    }))
  }
}))

vi.mock("@/services/tldw/voice-conversation", () => buildState)

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: (...args: unknown[]) =>
    (resolveProviderState.resolveApiProviderForModel as unknown as (
      ...args: unknown[]
    ) => unknown)(...args)
}))

class MockWebSocket {
  static instances: MockWebSocket[] = []

  static OPEN = 1
  static CLOSED = 3

  readyState = 0
  binaryType = "blob"
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  send(payload: string) {
    this.sent.push(payload)
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  }

  triggerOpen() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }
}

describe("useVoiceChatStream defaults", () => {
  beforeEach(() => {
    storageState.values.clear()
    audioCatalogState.devices = []
    audioCatalogState.isSettled = true
    audioSourcePreferenceState.preference = {
      featureGroup: "live_voice",
      sourceKind: "default_mic",
      deviceId: null,
      lastKnownLabel: null
    }
    audioSourcePreferenceState.isLoading = false
    audioPlayerState.start.mockClear()
    audioPlayerState.append.mockClear()
    audioPlayerState.finish.mockClear()
    audioPlayerState.stop.mockClear()
    micState.useMicStream.mockReset()
    micState.useMicStream.mockReturnValue({
      start: micState.start,
      stop: micState.stop,
      active: micState.active
    })
    micState.start.mockClear()
    micState.stop.mockClear()
    voiceChatState.voiceChatModel = ""
    voiceChatState.voiceChatPauseMs = 700
    voiceChatState.voiceChatTriggerPhrases = []
    voiceChatState.voiceChatAutoResume = true
    voiceChatState.voiceChatBargeIn = false
    voiceChatState.voiceChatTtsMode = "stream"
    sttState.model = "whisper-1"
    selectedModelState.selectedModel = null
    vi.mocked(buildState.buildVoiceConversationPreflight).mockClear()
    vi.mocked(buildState.normalizeVoiceConversationRuntimeError).mockClear()
    resolveProviderState.resolveApiProviderForModel.mockReset()
    resolveProviderState.resolveApiProviderForModel.mockResolvedValue("stub")
    vi.mocked(tldwClient.getConfig).mockReset()
    vi.mocked(tldwClient.getConfig).mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single_user",
      apiKey: "test-key"
    } as any)
    MockWebSocket.instances = []
    ;(globalThis as any).WebSocket = MockWebSocket
  })

  it("feeds the canonical Kitten defaults into voice-conversation preflight", async () => {
    renderHook(() =>
      useVoiceChatStream({
        active: true
      })
    )

    await waitFor(() => {
      expect(buildState.buildVoiceConversationPreflight).toHaveBeenCalled()
    })

    expect(buildState.buildVoiceConversationPreflight).toHaveBeenCalledWith(
      expect.objectContaining({
        ttsProvider: "tldw",
        tldwTtsModel: DEFAULT_KITTEN_MODEL,
        tldwTtsVoice: DEFAULT_KITTEN_VOICE
      })
    )
  })

  it("uses the voice_chat mic owner with default PCM16 capture", () => {
    renderHook(() =>
      useVoiceChatStream({
        active: false
      })
    )

    const lastCall = micState.useMicStream.mock.calls.at(-1)
    expect(lastCall?.[1]).toEqual({ owner: "voice_chat" })
  })

  it("sends strict v1 voice chat config after auth", async () => {
    renderHook(() =>
      useVoiceChatStream({
        active: true
      })
    )

    await waitFor(() => {
      expect(MockWebSocket.instances[0]).toBeDefined()
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.triggerOpen()
      await Promise.resolve()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(ws.sent.length).toBeGreaterThan(1)
    })

    const frames = ws.sent.map((raw) => JSON.parse(raw))
    expect(frames[0]).toEqual({ type: "auth", token: "test-key" })
    expect(frames.find((frame) => frame.type === "config")).toMatchObject({
      type: "config",
      protocol_version: 1,
      mode: "voice_chat",
      audio_format: "pcm16",
      sample_rate: 16000,
      channels: 1
    })
  })

  it("can start a push-to-talk stream and send the release control frame", async () => {
    const { result } = renderHook(() =>
      useVoiceChatStream({
        active: true,
        mode: "push_to_talk"
      })
    )

    await waitFor(() => {
      expect(MockWebSocket.instances[0]).toBeDefined()
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.triggerOpen()
      await Promise.resolve()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(ws.sent.length).toBeGreaterThan(1)
    })

    expect(micState.useMicStream.mock.calls.at(-1)?.[1]).toEqual({
      owner: "push_to_talk"
    })
    expect(
      ws.sent.map((raw) => JSON.parse(raw)).find((frame) => frame.type === "config")
    ).toMatchObject({
      type: "config",
      mode: "push_to_talk"
    })

    act(() => {
      result.current.sendPushToTalkRelease()
    })

    expect(JSON.parse(ws.sent.at(-1) || "{}")).toEqual({
      type: "push_to_talk_release"
    })
  })
})
