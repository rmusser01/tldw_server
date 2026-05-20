// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createAudioCaptureSessionCoordinator } from "@/audio"

type QueryOptions = {
  queryKey?: readonly unknown[]
}

type MediaDevicesMock = {
  getUserMedia: ReturnType<typeof vi.fn>
  enumerateDevices: ReturnType<typeof vi.fn>
  addEventListener: ReturnType<typeof vi.fn>
  removeEventListener: ReturnType<typeof vi.fn>
  ondevicechange: ((event: Event) => void) | null
}

const updateStoredValue = (
  storageValues: Map<string, unknown>,
  key: string,
  defaultValue: unknown,
  nextValue: unknown
) => {
  const currentValue = storageValues.has(key)
    ? storageValues.get(key)
    : defaultValue
  const resolvedValue =
    typeof nextValue === "function"
      ? (nextValue as (current: unknown) => unknown)(currentValue)
      : nextValue
  storageValues.set(key, resolvedValue)
}

const originalMediaDevices = navigator.mediaDevices
const AUDIO_CAPTURE_COORDINATOR_KEY = Symbol.for(
  "tldw.audioCaptureSessionCoordinator"
)

const {
  storageValues,
  mockTrackStop,
  mockGetUserMedia,
  mockEnumerateDevices,
  invalidateQueriesMock,
  transcribeAudioMock,
  getTranscriptionModelsMock
} = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  mockTrackStop: vi.fn(),
  mockGetUserMedia: vi.fn(),
  mockEnumerateDevices: vi.fn(),
  invalidateQueriesMock: vi.fn(),
  transcribeAudioMock: vi.fn(),
  getTranscriptionModelsMock: vi.fn(async () => ({
    all_models: ["whisper-1"]
  }))
}))

class MockMediaRecorder {
  stream: MediaStream
  ondataavailable: ((e: BlobEvent) => void) | null = null
  onstop: (() => void) | null = null
  onerror: ((e: Event) => void) | null = null
  mimeType = "audio/webm"
  state = "inactive" as "inactive" | "recording"

  constructor(stream: MediaStream) {
    this.stream = stream
  }

  start = vi.fn(() => {
    this.state = "recording"
  })

  stop = vi.fn(() => {
    this.state = "inactive"
    this.onstop?.()
  })
}

vi.stubGlobal("MediaRecorder", MockMediaRecorder)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageValues.has(key) ? storageValues.get(key) : defaultValue,
    (nextValue: unknown) =>
      updateStoredValue(storageValues, key, defaultValue, nextValue),
    { isLoading: false }
  ]
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: invalidateQueriesMock
  }),
  useQuery: vi.fn((options: QueryOptions | undefined) => {
    if (options?.queryKey?.[0] === "fetchTTSSettings") {
      return {
        data: {
          ttsProvider: "browser",
          ttsEnabled: true,
          tldwTtsSpeed: 1,
          tldwTtsStreaming: false,
          responseSplitting: "punctuation"
        },
        isLoading: false
      }
    }
    return {
      data: [],
      isLoading: false
    }
  })
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="speech-page-shell">{children}</div>
  )
}))

vi.mock("@/components/Common/WaveformCanvas", () => ({
  default: () => <div data-testid="waveform-canvas" />
}))

vi.mock("@/components/Common/TtsJobProgress", () => ({
  TtsJobProgress: () => <div data-testid="tts-job-progress" />
}))

vi.mock("@/components/Common/LongformDraftEditor", () => ({
  LongformDraftEditor: () => <div data-testid="longform-draft-editor" />
}))

vi.mock("@/components/Common/CharacterProgressBar", () => ({
  CharacterProgressBar: () => <div data-testid="character-progress-bar" />
}))

vi.mock("@/components/Option/Speech/TtsProviderStrip", () => ({
  TtsProviderStrip: () => <div data-testid="tts-provider-strip" />
}))

vi.mock("@/components/Option/Speech/TtsStickyActionBar", () => ({
  TtsStickyActionBar: () => <div data-testid="tts-sticky-action-bar" />
}))

vi.mock("@/components/Option/Speech/TtsInspectorPanel", () => ({
  TtsInspectorPanel: () => <div data-testid="tts-inspector-panel" />
}))

vi.mock("@/components/Option/Speech/TtsVoiceTab", () => ({
  TtsVoiceTab: () => <div data-testid="tts-voice-tab" />
}))

vi.mock("@/components/Option/Speech/TtsOutputTab", () => ({
  TtsOutputTab: () => <div data-testid="tts-output-tab" />
}))

vi.mock("@/components/Option/Speech/TtsAdvancedTab", () => ({
  TtsAdvancedTab: () => <div data-testid="tts-advanced-tab" />
}))

vi.mock("@/components/Option/TTS/VoiceCloningManager", () => ({
  VoiceCloningManager: () => <div data-testid="voice-cloning-manager" />
}))

vi.mock("@/components/Option/Speech/RenderStrip", () => ({
  RenderStrip: () => <div data-testid="render-strip" />
}))

vi.mock("@/components/Option/Audio/AudioPresetControls", () => ({
  AudioPresetControls: () => <div data-testid="audio-preset-controls" />
}))

vi.mock("@/components/Option/Speech/VoicePickerModal", () => ({
  VoicePickerModal: () => <div data-testid="voice-picker-modal" />
}))

vi.mock("@/hooks/useTtsPlayground", () => ({
  TTS_PRESETS: {
    balanced: {
      label: "Balanced",
      value: "balanced"
    }
  },
  useTtsPlayground: () => ({
    segments: [],
    isGenerating: false,
    generateSegments: vi.fn(async () => []),
    clearSegments: vi.fn(),
    setSegments: vi.fn()
  })
}))

vi.mock("@/hooks/useStreamingAudioPlayer", () => ({
  useStreamingAudioPlayer: () => ({
    start: vi.fn(),
    append: vi.fn(),
    finish: vi.fn(),
    stop: vi.fn(),
    state: "idle",
    getBufferedBlob: vi.fn(() => null)
  })
}))

vi.mock("@/hooks/useTtsProviderData", () => ({
  OPENAI_TTS_MODELS: ["tts-1"],
  OPENAI_TTS_VOICES: {
    "tts-1": [{ label: "Alloy", value: "alloy" }]
  },
  useTtsProviderData: () => ({
    hasAudio: true,
    providersInfo: {
      providers: {
        browser: {
          supports_streaming: false
        }
      },
      voices: {}
    },
    tldwTtsModels: [],
    tldwVoiceCatalog: [],
    elevenLabsData: null,
    elevenLabsLoading: false,
    elevenLabsError: null,
    refetchElevenLabs: vi.fn()
  })
}))

vi.mock("@/hooks/useMultiRenderState", () => ({
  useMultiRenderState: () => ({
    renders: [],
    hasReady: false,
    hasIdle: false,
    playingId: null,
    addRender: vi.fn(),
    updateConfig: vi.fn(),
    generateAll: vi.fn(async () => undefined),
    playAllSequentially: vi.fn(),
    clearAll: vi.fn(),
    generateRender: vi.fn(async () => undefined),
    removeRender: vi.fn(),
    startPlaying: vi.fn(),
    stopPlaying: vi.fn(),
    handleStripEnded: vi.fn()
  })
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: vi.fn(() => null),
  resolveTtsProviderContext: vi.fn(async (text: string) => ({
    utterance: text
  }))
}))

vi.mock("@/services/tts-providers", () => ({
  getTtsProviderLabel: vi.fn((provider?: string) => provider || "browser")
}))

vi.mock("@/services/tts", () => ({
  DEFAULT_TTS_PROVIDER: "tldw",
  DEFAULT_TLDW_TTS_MODEL: "KittenML/kitten-tts-nano-0.8",
  DEFAULT_TLDW_TTS_VOICE: "Bella",
  getTTSProvider: vi.fn(async () => "browser"),
  getTTSSettings: vi.fn(async () => ({
    ttsProvider: "browser",
    ttsEnabled: true,
    responseSplitting: "punctuation",
    tldwTtsSpeed: 1,
    tldwTtsStreaming: false
  })),
  setTTSSettings: vi.fn(async () => undefined),
  SUPPORTED_TLDW_TTS_FORMATS: ["mp3"],
  setTldwTTSSpeed: vi.fn(),
  setTldwTTSResponseFormat: vi.fn(),
  setTldwTTSStreamingEnabled: vi.fn(),
  setResponseSplitting: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getTranscriptionModels: getTranscriptionModelsMock,
    transcribeAudio: transcribeAudioMock,
    createNote: vi.fn(async () => undefined),
    getConfig: vi.fn(async () => ({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single_user",
      apiKey: "test-key"
    })),
    createTtsJob: vi.fn(),
    streamAudioJobProgress: vi.fn(),
    getTtsJobArtifacts: vi.fn(),
    downloadOutput: vi.fn()
  }
}))

vi.mock("@/utils/clipboard", () => ({
  copyToClipboard: vi.fn(async () => true)
}))

vi.mock("@/utils/tts", () => ({
  estimateTtsDurationSeconds: vi.fn(() => 1),
  splitMessageContent: vi.fn((text: string) => [text])
}))

vi.mock("@/utils/markdown-to-text", () => ({
  markdownToText: vi.fn((text: string) => text)
}))

vi.mock("@/utils/request-timeout", () => ({
  isTimeoutLikeError: vi.fn(() => false)
}))

vi.mock("@/utils/template-guards", () => ({
  withTemplateFallback: vi.fn((_template: string, fallback: string) => fallback)
}))

vi.mock("@/services/tldw/voice-cloning", () => ({
  listCustomVoices: vi.fn(async () => [])
}))

vi.mock("@/services/tldw/tts-provider-keys", () => ({
  normalizeTtsProviderKey: vi.fn((value?: string) => value || ""),
  toServerTtsProviderKey: vi.fn((value?: string) => value || "")
}))

import SpeechPlaygroundPage from "../SpeechPlaygroundPage"

const buildAudioInput = (
  overrides: Partial<MediaDeviceInfo>
): MediaDeviceInfo =>
  ({
    deviceId: "device-1",
    groupId: "group-1",
    kind: "audioinput",
    label: "USB microphone",
    toJSON: () => ({}),
    ...overrides
  }) as MediaDeviceInfo

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

describe("SpeechPlaygroundPage audio source", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    storageValues.clear()
    storageValues.set("speechPlaygroundMode", "roundtrip")
    storageValues.set("speechPlaygroundHistory", [])

    mockGetUserMedia.mockResolvedValue({
      getTracks: () => [{ stop: mockTrackStop }]
    })
    mockEnumerateDevices.mockResolvedValue([
      buildAudioInput({ deviceId: "default", label: "Default microphone" }),
      buildAudioInput({ deviceId: "usb-1", label: "USB microphone" })
    ])

    const mediaDevicesMock: MediaDevicesMock = {
      getUserMedia: mockGetUserMedia,
      enumerateDevices: mockEnumerateDevices,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      ondevicechange: null
    }

    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: mediaDevicesMock
    })
    delete (
      globalThis as typeof globalThis & {
        [AUDIO_CAPTURE_COORDINATOR_KEY]?: unknown
      }
    )[AUDIO_CAPTURE_COORDINATOR_KEY]
  })

  afterEach(() => {
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: originalMediaDevices
    })
  })

  it("uses the remembered speech_playground mic preference when recording starts", async () => {
    storageValues.set("speechPlaygroundAudioSourcePreference", {
      featureGroup: "speech_playground",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })

    const user = userEvent.setup()

    render(<SpeechPlaygroundPage />)

    await user.click(
      screen.getByRole("button", { name: /start dictation/i })
    )

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: { deviceId: { exact: "usb-1" } }
    })
  })

  it("keeps the remembered mic device active while device enumeration is still loading", async () => {
    const deferredDevices = createDeferred<MediaDeviceInfo[]>()
    mockEnumerateDevices.mockImplementation(() => deferredDevices.promise)
    storageValues.set("speechPlaygroundAudioSourcePreference", {
      featureGroup: "speech_playground",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })

    const user = userEvent.setup()

    render(<SpeechPlaygroundPage />)

    await user.click(
      screen.getByRole("button", { name: /start dictation/i })
    )

    expect(mockGetUserMedia).toHaveBeenCalledWith({
      audio: { deviceId: { exact: "usb-1" } }
    })

    deferredDevices.resolve([
      buildAudioInput({ deviceId: "default", label: "Default microphone" }),
      buildAudioInput({ deviceId: "usb-1", label: "USB microphone" })
    ])
  })

  it("shows fallback active and records from the default mic after catalog settles without the remembered device", async () => {
    storageValues.set("speechPlaygroundAudioSourcePreference", {
      featureGroup: "speech_playground",
      sourceKind: "mic_device",
      deviceId: "usb-missing",
      lastKnownLabel: "Missing microphone"
    })
    mockEnumerateDevices.mockResolvedValue([
      buildAudioInput({ deviceId: "default", label: "Default microphone" }),
      buildAudioInput({ deviceId: "usb-2", label: "Desk microphone" })
    ])

    const user = userEvent.setup()

    render(<SpeechPlaygroundPage />)

    await waitFor(() => {
      expect(screen.getByText("Source fallback active")).toBeInTheDocument()
    })

    await user.click(
      screen.getByRole("button", { name: /start dictation/i })
    )

    expect(mockGetUserMedia).toHaveBeenCalledWith({ audio: true })
  })

  it("blocks recording start when another low-level capture owner is active", async () => {
    ;(
      globalThis as typeof globalThis & {
        [AUDIO_CAPTURE_COORDINATOR_KEY]?: unknown
      }
    )[AUDIO_CAPTURE_COORDINATOR_KEY] =
      createAudioCaptureSessionCoordinator("live_voice")

    const user = userEvent.setup()

    render(<SpeechPlaygroundPage />)

    await user.click(
      screen.getByRole("button", { name: /start dictation/i })
    )

    expect(mockGetUserMedia).not.toHaveBeenCalled()
    await waitFor(() => {
      expect(
        screen.getAllByText(
          "Stop the active capture session in live_voice and try again."
        ).length
      ).toBeGreaterThanOrEqual(1)
    })
  })
})
