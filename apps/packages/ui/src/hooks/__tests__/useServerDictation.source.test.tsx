import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { createAudioCaptureSessionCoordinator } from "@/audio"

const {
  mockNotificationError,
  mockGetUserMedia,
  mockTrackStop,
  mockTranscribeAudio
} = vi.hoisted(() => ({
  mockNotificationError: vi.fn(),
  mockGetUserMedia: vi.fn(),
  mockTrackStop: vi.fn(),
  mockTranscribeAudio: vi.fn()
}))

const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

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

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    transcribeAudio: mockTranscribeAudio
  }
}))

class MockMediaRecorder {
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onstop: (() => void | Promise<void>) | null = null
  mimeType = "audio/webm"
  state = "inactive"

  start = vi.fn(() => {
    this.state = "recording"
  })

  stop = vi.fn(() => {
    this.state = "inactive"
    this.onstop?.()
  })
}

vi.stubGlobal("MediaRecorder", MockMediaRecorder)
vi.stubGlobal("navigator", {
  mediaDevices: { getUserMedia: mockGetUserMedia }
})

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

const buildHook = (overrides?: Partial<Parameters<typeof useServerDictation>[0]>) =>
  useServerDictation({
    canUseServerStt: true,
    speechToTextLanguage: "en-US",
    sttSettings: defaultSttSettings,
    onTranscript: vi.fn(),
    ...overrides
  })

const createMockStream = () =>
  ({
    getTracks: () => [{ stop: mockTrackStop }]
  }) as unknown as MediaStream

describe("useServerDictation selected source handling", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetUserMedia.mockResolvedValue(createMockStream())
    mockTranscribeAudio.mockResolvedValue({ text: "transcript" })
    delete (
      globalThis as typeof globalThis & {
        [AUDIO_CAPTURE_COORDINATOR_KEY]?: unknown
      }
    )[AUDIO_CAPTURE_COORDINATOR_KEY]
  })

  it("records server dictation from the selected mic device", async () => {
    const { result } = renderHook(() => buildHook())

    await act(async () => {
      await result.current.startServerDictation({
        sourceKind: "mic_device",
        deviceId: "usb-1"
      })
    })

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: { deviceId: { exact: "usb-1" } }
    })
  })

  it("falls back to the default microphone when no deviceId is provided", async () => {
    const { result } = renderHook(() => buildHook())

    await act(async () => {
      await result.current.startServerDictation({
        sourceKind: "default_mic",
        deviceId: null
      })
    })

    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true })
  })

  it("surfaces a busy-owner error instead of silently returning when another low-level capture is active", async () => {
    ;(
      globalThis as typeof globalThis & {
        [AUDIO_CAPTURE_COORDINATOR_KEY]?: unknown
      }
    )[AUDIO_CAPTURE_COORDINATOR_KEY] =
      createAudioCaptureSessionCoordinator("live_voice")
    const onError = vi.fn()
    const { result } = renderHook(() => buildHook({ onError }))

    await act(async () => {
      await result.current.startServerDictation({
        sourceKind: "mic_device",
        deviceId: "usb-1"
      })
    })

    expect(mockGetUserMedia).not.toHaveBeenCalled()
    expect(onError).toHaveBeenCalledTimes(1)
    expect(mockNotificationError).toHaveBeenCalledTimes(1)
  })

  it("reports microphone startup failures through onError", async () => {
    const onError = vi.fn()
    const startupError = new Error("Requested microphone unavailable")
    mockGetUserMedia.mockRejectedValueOnce(startupError)
    const { result } = renderHook(() => buildHook({ onError }))

    await act(async () => {
      await result.current.startServerDictation({
        sourceKind: "mic_device",
        deviceId: "usb-missing"
      })
    })

    expect(onError).toHaveBeenCalledWith(startupError)
    expect(mockNotificationError).toHaveBeenCalledTimes(1)
  })
})
