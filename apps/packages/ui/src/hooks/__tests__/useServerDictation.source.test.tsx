import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mockNotificationError = vi.hoisted(() => vi.fn())

const clientState = vi.hoisted(() => ({
  getConfig: vi.fn(async () => ({
    serverUrl: "http://localhost:8000",
    authMode: "single_user",
    apiKey: "test-key"
  }))
}))

const micState = vi.hoisted(() => ({
  callback: null as ((chunk: ArrayBuffer) => void) | null,
  options: undefined as unknown,
  start: vi.fn(async () => {}),
  stop: vi.fn(() => {}),
  active: false
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: mockNotificationError
  })
}))

vi.mock("@/hooks/useMicStream", () => ({
  useMicStream: (callback: (chunk: ArrayBuffer) => void, options: unknown) => {
    micState.callback = callback
    micState.options = options
    return {
      start: micState.start,
      stop: micState.stop,
      active: micState.active
    }
  }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: clientState.getConfig
  }
}))

import { useServerDictation } from "../useServerDictation"
import type { SttSettings } from "../useSttSettings"

const defaultSttSettings: SttSettings = {
  model: "whisper-1",
  temperature: 0,
  task: "transcribe",
  responseFormat: "json",
  timestampGranularities: "segment",
  prompt: "",
  useSegmentation: false,
  segK: 3,
  segMinSegmentSize: 10,
  segLambdaBalance: 0.5,
  segUtteranceExpansionWidth: 1,
  segEmbeddingsProvider: "",
  segEmbeddingsModel: ""
}

class MockWebSocket {
  static OPEN = 1
  static CLOSED = 3
  static instances: MockWebSocket[] = []

  readyState = 0
  sentMessages: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string | ArrayBuffer }) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor(public url: string) {
    MockWebSocket.instances.push(this)
  }

  send(payload: string) {
    this.sentMessages.push(payload)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.()
  }

  message(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }

  error() {
    this.onerror?.()
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.()
  }
}

const buildHook = (
  overrides?: Partial<Parameters<typeof useServerDictation>[0]>
) =>
  useServerDictation({
    canUseServerStt: true,
    speechToTextLanguage: "en-US",
    sttSettings: defaultSttSettings,
    onTranscript: vi.fn(),
    ...overrides
  })

const sentFrames = (ws: MockWebSocket) =>
  ws.sentMessages.map((message) => JSON.parse(message))

describe("useServerDictation selected source handling", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.instances = []
    ;(globalThis as any).WebSocket = MockWebSocket
    clientState.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "single_user",
      apiKey: "test-key"
    })
    micState.callback = null
    micState.options = undefined
    micState.start.mockResolvedValue(undefined)
    micState.active = false
  })

  it("uses the dictation mic owner", () => {
    renderHook(() => buildHook())

    expect(micState.options).toEqual({ owner: "dictation" })
  })

  it("sends strict dictate config before audio frames", async () => {
    const { result } = renderHook(() => buildHook())

    await act(async () => {
      await result.current.startServerDictation()
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.open()
      await Promise.resolve()
      await Promise.resolve()
      micState.callback?.(new ArrayBuffer(2))
    })

    const sent = sentFrames(ws)
    expect(sent[0]).toMatchObject({ type: "auth", token: "test-key" })
    expect(sent[1]).toMatchObject({
      type: "config",
      protocol_version: 1,
      mode: "dictate",
      audio_format: "pcm16",
      sample_rate: 16000,
      channels: 1
    })
    expect(sent[2]).toMatchObject({ type: "audio" })
  })

  it("starts the mic with the selected device after websocket config", async () => {
    const { result } = renderHook(() => buildHook())

    await act(async () => {
      await result.current.startServerDictation({
        sourceKind: "mic_device",
        deviceId: "usb-1"
      })
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.open()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(micState.start).toHaveBeenCalledWith({ deviceId: "usb-1" })
  })

  it("emits partial preview separately and final transcript once", async () => {
    const onPartialTranscript = vi.fn()
    const onTranscript = vi.fn()
    const { result } = renderHook(() =>
      buildHook({ onPartialTranscript, onTranscript })
    )

    await act(async () => {
      await result.current.startServerDictation()
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.open()
      await Promise.resolve()
      ws.message({ type: "partial", text: "hel" })
      ws.message({ type: "full_transcript", text: "hello" })
    })

    expect(onPartialTranscript).toHaveBeenCalledWith("hel")
    expect(onTranscript).toHaveBeenCalledTimes(1)
    expect(onTranscript).toHaveBeenCalledWith("hello")
  })

  it("reports microphone startup failures through onError", async () => {
    const onError = vi.fn()
    const startupError = new Error("Requested microphone unavailable")
    micState.start.mockRejectedValueOnce(startupError)
    const { result } = renderHook(() => buildHook({ onError }))

    await act(async () => {
      await result.current.startServerDictation({
        sourceKind: "mic_device",
        deviceId: "usb-missing"
      })
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.open()
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(onError).toHaveBeenCalledWith(startupError)
    expect(mockNotificationError).toHaveBeenCalledTimes(1)
    expect(result.current.isServerDictating).toBe(false)
  })

  it("does not create a second websocket on a rapid double-start", async () => {
    const { result } = renderHook(() => buildHook())

    await act(async () => {
      await Promise.all([
        result.current.startServerDictation(),
        result.current.startServerDictation()
      ])
    })

    expect(MockWebSocket.instances).toHaveLength(1)
    expect(micState.start).not.toHaveBeenCalled()
  })

  it("stops mic and websocket on stop", async () => {
    const { result } = renderHook(() => buildHook())

    await act(async () => {
      await result.current.startServerDictation()
    })

    const ws = MockWebSocket.instances[0]
    await act(async () => {
      ws.open()
      await Promise.resolve()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(result.current.isServerDictating).toBe(true)
    })

    act(() => {
      result.current.stopServerDictation()
    })

    expect(sentFrames(ws).at(-1)).toMatchObject({ type: "stop" })
    expect(micState.stop).toHaveBeenCalled()
    expect(result.current.isServerDictating).toBe(false)
  })
})
