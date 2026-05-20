// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

type QueryOptions = {
  queryKey?: readonly unknown[]
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

const createDeferred = <T,>(): {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (reason?: unknown) => void
} => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

const {
  invalidateQueriesMock,
  transcribeAudioMock,
  getTranscriptionModelsMock,
  getElevenLabsVoicesMock,
  getElevenLabsModelsMock,
  addRenderMock,
  setTTSSettingsMock,
  ttsSettingsRef,
  audioPresetControlPropsRef,
} =
  vi.hoisted(() => ({
    invalidateQueriesMock: vi.fn(),
    transcribeAudioMock: vi.fn(),
    getTranscriptionModelsMock: vi.fn(async () => ({
      all_models: ["whisper-1"],
    })),
    getElevenLabsVoicesMock: vi.fn(),
    getElevenLabsModelsMock: vi.fn(),
    addRenderMock: vi.fn(),
    setTTSSettingsMock: vi.fn(async () => undefined),
    ttsSettingsRef: {
      current: {
        ttsProvider: "",
        ttsEnabled: true,
        tldwTtsSpeed: 1,
        tldwTtsStreaming: false,
        responseSplitting: "punctuation",
      } as any,
    },
    audioPresetControlPropsRef: {
      current: null as null | {
        kind: string
        currentConfig: Record<string, unknown>
        onApply: (config: Record<string, unknown>, preset: any) => void
      },
    },
  }))

const { storageValues, setSpeechModeMock, setSpeechHistoryMock } = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  setSpeechModeMock: vi.fn(),
  setSpeechHistoryMock: vi.fn(),
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key,
  }),
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    if (key === "speechPlaygroundMode") {
      return [
        storageValues.has(key) ? storageValues.get(key) : defaultValue,
        (nextValue: unknown) => {
          setSpeechModeMock(nextValue)
          updateStoredValue(storageValues, key, defaultValue, nextValue)
        },
        { isLoading: false }
      ] as const
    }
    if (key === "speechPlaygroundHistory") {
      return [
        storageValues.has(key) ? storageValues.get(key) : defaultValue,
        (nextValue: unknown) => {
          setSpeechHistoryMock(nextValue)
          updateStoredValue(storageValues, key, defaultValue, nextValue)
        },
        { isLoading: false }
      ] as const
    }
    return [
      storageValues.has(key) ? storageValues.get(key) : defaultValue,
      (nextValue: unknown) =>
        updateStoredValue(storageValues, key, defaultValue, nextValue),
      { isLoading: false }
    ] as const
  },
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: invalidateQueriesMock,
  }),
  useQuery: vi.fn((options: QueryOptions | undefined) => {
    if (options?.queryKey?.[0] === "fetchTTSSettings") {
      return {
        data: ttsSettingsRef.current,
        isLoading: false,
      }
    }
    return {
      data: [],
      isLoading: false,
    }
  }),
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="speech-page-shell">{children}</div>
  ),
}))

vi.mock("@/components/Common/WaveformCanvas", () => ({
  default: () => <div data-testid="waveform-canvas" />,
}))

vi.mock("@/components/Common/TtsJobProgress", () => ({
  TtsJobProgress: () => <div data-testid="tts-job-progress" />,
}))

vi.mock("@/components/Common/LongformDraftEditor", () => ({
  LongformDraftEditor: () => <div data-testid="longform-draft-editor" />,
}))

vi.mock("@/components/Common/CharacterProgressBar", () => ({
  CharacterProgressBar: () => <div data-testid="character-progress-bar" />,
}))

vi.mock("@/components/Option/Speech/TtsProviderStrip", () => ({
  TtsProviderStrip: (props: { provider: string }) => (
    <div data-testid="tts-provider-strip" data-provider={props.provider} />
  ),
}))

vi.mock("@/components/Option/Speech/TtsStickyActionBar", () => ({
  TtsStickyActionBar: ({ onAddRender }: { onAddRender?: () => void }) => (
    <button type="button" data-testid="tts-sticky-action-bar" onClick={onAddRender}>
      Add render
    </button>
  ),
}))

vi.mock("@/components/Option/Speech/TtsInspectorPanel", () => ({
  TtsInspectorPanel: () => <div data-testid="tts-inspector-panel" />,
}))

vi.mock("@/components/Option/Speech/TtsVoiceTab", () => ({
  TtsVoiceTab: () => <div data-testid="tts-voice-tab" />,
}))

vi.mock("@/components/Option/Speech/TtsOutputTab", () => ({
  TtsOutputTab: () => <div data-testid="tts-output-tab" />,
}))

vi.mock("@/components/Option/Speech/TtsAdvancedTab", () => ({
  TtsAdvancedTab: () => <div data-testid="tts-advanced-tab" />,
}))

vi.mock("@/components/Option/TTS/VoiceCloningManager", () => ({
  VoiceCloningManager: () => <div data-testid="voice-cloning-manager" />,
}))

vi.mock("@/components/Option/Speech/RenderStrip", () => ({
  RenderStrip: () => <div data-testid="render-strip" />,
}))

vi.mock("@/components/Option/Audio/AudioPresetControls", () => ({
  AudioPresetControls: (props: {
    kind: string
    currentConfig: Record<string, unknown>
    onApply: (config: Record<string, unknown>, preset: any) => void
  }) => {
    audioPresetControlPropsRef.current = props
    return (
      <button
        type="button"
        data-testid={`${props.kind}-preset-controls`}
        onClick={() =>
          props.onApply(
            {
              provider: " openai ",
              model: " gpt-4o-mini-tts ",
              voice: " verse ",
              response_format: " opus ",
              speed: 1.2,
              response_splitting: "paragraph",
              streaming: false
            },
            { id: "preset-1", name: "OpenAI verse" }
          )
        }
      >
        Apply mock {props.kind} preset
      </button>
    )
  },
}))

vi.mock("@/components/Option/Speech/VoicePickerModal", () => ({
  VoicePickerModal: () => <div data-testid="voice-picker-modal" />,
}))

vi.mock("@/hooks/useTtsPlayground", () => ({
  TTS_PRESETS: {
    balanced: {
      label: "Balanced",
      value: "balanced",
    },
  },
  useTtsPlayground: () => ({
    segments: [],
    isGenerating: false,
    generateSegments: vi.fn(async () => []),
    clearSegments: vi.fn(),
    setSegments: vi.fn(),
  }),
}))

vi.mock("@/hooks/useStreamingAudioPlayer", () => ({
  useStreamingAudioPlayer: () => ({
    start: vi.fn(),
    append: vi.fn(),
    finish: vi.fn(),
    stop: vi.fn(),
    state: "idle",
    getBufferedBlob: vi.fn(() => null),
  }),
}))

vi.mock("@/hooks/useTtsProviderData", () => ({
  OPENAI_TTS_MODELS: ["tts-1"],
  OPENAI_TTS_VOICES: {
    "tts-1": [{ label: "Alloy", value: "alloy" }],
  },
  useTtsProviderData: () => ({
    hasAudio: true,
    providersInfo: {
      providers: {
        browser: {
          supports_streaming: false,
        },
      },
      voices: {},
    },
    tldwTtsModels: [],
    tldwVoiceCatalog: [],
    elevenLabsData: null,
    elevenLabsLoading: false,
    elevenLabsError: null,
    refetchElevenLabs: vi.fn(),
  }),
}))

vi.mock("@/hooks/useMultiRenderState", () => ({
  useMultiRenderState: () => ({
    renders: [],
    hasReady: false,
    hasIdle: false,
    playingId: null,
    addRender: addRenderMock,
    updateConfig: vi.fn(),
    generateAll: vi.fn(async () => undefined),
    playAllSequentially: vi.fn(),
    clearAll: vi.fn(),
    generateRender: vi.fn(async () => undefined),
    removeRender: vi.fn(),
    startPlaying: vi.fn(),
    stopPlaying: vi.fn(),
    handleStripEnded: vi.fn(),
  }),
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: vi.fn(() => null),
  resolveTtsProviderContext: vi.fn(async (text: string) => ({
    utterance: text,
  })),
}))

vi.mock("@/services/tts-providers", () => ({
  getTtsProviderLabel: vi.fn((provider?: string) => provider || "browser"),
}))

vi.mock("@/services/tts", () => ({
  getTTSProvider: vi.fn(async () => "tldw"),
  getTTSSettings: vi.fn(async () => ttsSettingsRef.current),
  setTTSSettings: setTTSSettingsMock,
  SUPPORTED_TLDW_TTS_FORMATS: ["mp3"],
  setTldwTTSSpeed: vi.fn(),
  setTldwTTSResponseFormat: vi.fn(),
  setTldwTTSStreamingEnabled: vi.fn(),
  setResponseSplitting: vi.fn(),
  DEFAULT_TTS_PROVIDER: "tldw",
  DEFAULT_TLDW_TTS_MODEL: "KittenML/kitten-tts-nano-0.8",
  DEFAULT_TLDW_TTS_VOICE: "Bella",
}))

vi.mock("@/services/elevenlabs", () => ({
  getVoices: getElevenLabsVoicesMock,
  getModels: getElevenLabsModelsMock,
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getTranscriptionModels: getTranscriptionModelsMock,
    transcribeAudio: transcribeAudioMock,
    createNote: vi.fn(async () => undefined),
    getConfig: vi.fn(async () => ({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single_user",
      apiKey: "test-key",
    })),
    createTtsJob: vi.fn(),
    streamAudioJobProgress: vi.fn(),
    getTtsJobArtifacts: vi.fn(),
    downloadOutput: vi.fn(),
  },
}))

vi.mock("@/utils/clipboard", () => ({
  copyToClipboard: vi.fn(async () => true),
}))

vi.mock("@/utils/tts", () => ({
  estimateTtsDurationSeconds: vi.fn(() => 1),
  splitMessageContent: vi.fn((text: string) => [text]),
}))

vi.mock("@/utils/markdown-to-text", () => ({
  markdownToText: vi.fn((text: string) => text),
}))

vi.mock("@/utils/request-timeout", () => ({
  isTimeoutLikeError: vi.fn(() => false),
}))

vi.mock("@/utils/template-guards", () => ({
  withTemplateFallback: vi.fn((_template: string, fallback: string) => fallback),
}))

vi.mock("@/services/tldw/voice-cloning", () => ({
  listCustomVoices: vi.fn(async () => []),
}))

vi.mock("@/services/tldw/tts-provider-keys", () => ({
  normalizeTtsProviderKey: vi.fn((value?: string) => value || ""),
  toServerTtsProviderKey: vi.fn((value?: string) => value || ""),
}))

import SpeechPlaygroundPage from "../SpeechPlaygroundPage"

describe("SpeechPlaygroundPage", () => {
  beforeEach((): void => {
    vi.clearAllMocks()
    invalidateQueriesMock.mockReset()
    transcribeAudioMock.mockReset()
    getTranscriptionModelsMock.mockClear()
    getElevenLabsVoicesMock.mockReset()
    getElevenLabsModelsMock.mockReset()
    addRenderMock.mockReset()
    setTTSSettingsMock.mockClear()
    audioPresetControlPropsRef.current = null
    ttsSettingsRef.current = {
      ttsProvider: "",
      ttsEnabled: true,
      tldwTtsSpeed: 1,
      tldwTtsStreaming: false,
      responseSplitting: "punctuation",
    }
    storageValues.clear()
    localStorage.clear()
    storageValues.set("speechPlaygroundMode", "roundtrip")
    storageValues.set("speechPlaygroundHistory", [])
    setSpeechModeMock.mockReset()
    setSpeechHistoryMock.mockReset()
  })

  it("renders without triggering a temporal dead zone error", (): void => {
    render(<SpeechPlaygroundPage />)

    expect(screen.getByTestId("speech-page-shell")).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { level: 1, name: "Speech Playground" })
    ).toBeInTheDocument()
  })

  it("passes the resolved provider to the tldw strip when the stored provider is empty", (): void => {
    render(<SpeechPlaygroundPage />)

    expect(screen.getByTestId("tts-provider-strip")).toHaveAttribute(
      "data-provider",
      "tldw"
    )
  })

  it("hides the mode switcher and STT region when locked to listen mode", (): void => {
    storageValues.set("speechPlaygroundMode", "speak")

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    expect(screen.queryByText("Mode")).not.toBeInTheDocument()
    expect(screen.queryByText("Current transcription model")).not.toBeInTheDocument()
    expect(screen.getByTestId("tts-provider-strip")).toBeInTheDocument()
    expect(getTranscriptionModelsMock).not.toHaveBeenCalled()
  })

  it("adds OpenAI render strips with OpenAI model and voice defaults", (): void => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "openai",
      openAITTSModel: "gpt-4o-mini-tts",
      openAITTSVoice: "alloy",
      tldwTtsModel: "KittenML/kitten-tts-nano-0.8",
      tldwTtsVoice: "Bella",
      tldwTtsResponseFormat: "mp3",
    }

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    fireEvent.click(screen.getByTestId("tts-sticky-action-bar"))

    expect(addRenderMock).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: "openai",
        model: "gpt-4o-mini-tts",
        voice: "alloy",
      })
    )
    expect(addRenderMock).not.toHaveBeenCalledWith(
      expect.objectContaining({
        provider: "openai",
        model: "KittenML/kitten-tts-nano-0.8",
        voice: "Bella",
      })
    )
  })

  it("uses TTS-specific page copy and history controls when locked to listen mode", (): void => {
    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    expect(
      screen.getByRole("heading", { level: 1, name: "Text to Speech" })
    ).toBeInTheDocument()
    expect(screen.getByRole("status", { name: "TTS readiness" })).toHaveTextContent(
      "Browser preview: Ready"
    )
    expect(
      screen.getByText("Draft text, choose a voice, and generate audio in one place.")
    ).toBeInTheDocument()
    expect(screen.getByText("TTS history")).toBeInTheDocument()
    expect(screen.getByText("Generate audio to see TTS history here.")).toBeInTheDocument()
    expect(screen.queryByTestId("speech-history-type-filter")).not.toBeInTheDocument()
  })

  it("renders TTS preset controls and applies a saved preset without starting generation", async (): Promise<void> => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "browser",
      tldwTtsResponseFormat: "mp3",
      tldwTtsSpeed: 1,
    }

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    expect(screen.getByTestId("tts-preset-controls")).toBeInTheDocument()
    expect(audioPresetControlPropsRef.current?.currentConfig).toEqual(
      expect.objectContaining({
        provider: "browser",
        response_format: "mp3",
        speed: 1,
      })
    )

    fireEvent.click(screen.getByRole("button", { name: "Apply mock tts preset" }))

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          ttsProvider: "openai",
          openAITTSModel: "gpt-4o-mini-tts",
          openAITTSVoice: "verse",
          tldwTtsResponseFormat: "opus",
          tldwTtsSpeed: 1.2,
          responseSplitting: "paragraph",
        })
      )
    })
    expect(addRenderMock).not.toHaveBeenCalled()
  })

  it("does not overwrite stored mode when locked mode is provided", (): void => {
    storageValues.set("speechPlaygroundMode", "speak")

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    expect(setSpeechModeMock).not.toHaveBeenCalled()
  })

  it("filters stored history down to TTS entries when locked to listen mode", (): void => {
    storageValues.set("speechPlaygroundHistory", [
      {
        id: "stt-1",
        type: "stt",
        createdAt: "2026-03-11T00:00:00.000Z",
        text: "Recorded transcript",
      },
      {
        id: "tts-1",
        type: "tts",
        createdAt: "2026-03-11T00:01:00.000Z",
        text: "Synthesized narration",
      },
    ])

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    expect(screen.getByText("Synthesized narration")).toBeInTheDocument()
    expect(screen.queryByText("Recorded transcript")).not.toBeInTheDocument()
  })

  it("keeps the mode switcher visible when unlocked", (): void => {
    render(<SpeechPlaygroundPage />)

    expect(screen.getByText("Speech Playground")).toBeInTheDocument()
    expect(screen.getByText("Mode")).toBeInTheDocument()
    expect(screen.getByTestId("speech-history-type-filter")).toBeInTheDocument()
    expect(getTranscriptionModelsMock).toHaveBeenCalled()
  })

  it("shows the shared audio source picker in the speech playground", (): void => {
    render(<SpeechPlaygroundPage />)

    expect(
      screen.getByLabelText("Speech playground input source")
    ).toBeInTheDocument()
  })

  it("offers inline ElevenLabs API key entry when ElevenLabs is selected without a key", (): void => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "elevenlabs",
      elevenLabsApiKey: "",
    }

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    expect(screen.getByLabelText("ElevenLabs API key")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Enter your ElevenLabs API key below to load voices and models. You can also manage it in Settings."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Test & Save" })
    ).toBeDisabled()
    expect(screen.queryByText("Set API key in Settings")).not.toBeInTheDocument()
  })

  it("validates and saves the inline ElevenLabs API key to local TTS settings", async (): Promise<void> => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "elevenlabs",
      elevenLabsApiKey: "",
    }
    getElevenLabsVoicesMock.mockResolvedValue([
      { voice_id: "voice-1", name: "Voice 1" },
    ])
    getElevenLabsModelsMock.mockResolvedValue([
      { model_id: "model-1", name: "Model 1" },
    ])

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    fireEvent.change(screen.getByLabelText("ElevenLabs API key"), {
      target: { value: "sk_test_inline" },
    })
    fireEvent.click(screen.getByRole("button", { name: "Test & Save" }))

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          ttsProvider: "elevenlabs",
          elevenLabsApiKey: "sk_test_inline",
        })
      )
    })
    expect(getElevenLabsVoicesMock).toHaveBeenCalledWith("sk_test_inline", {
      timeoutMs: 10_000,
    })
    expect(getElevenLabsModelsMock).toHaveBeenCalledWith("sk_test_inline", {
      timeoutMs: 10_000,
    })
    expect(invalidateQueriesMock).toHaveBeenCalledWith({
      queryKey: ["fetchTTSSettings"],
    })
  })

  it("keeps the inline key details collapsed across input rerenders", (): void => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "elevenlabs",
      elevenLabsApiKey: "",
    }

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    const details = screen.getByText("Enter API key").closest("details") as HTMLDetailsElement
    expect(details.open).toBe(true)

    fireEvent.click(screen.getByText("Enter API key"))
    expect(details.open).toBe(false)

    fireEvent.change(screen.getByLabelText("ElevenLabs API key"), {
      target: { value: "sk_test_inline" },
    })
    expect(details.open).toBe(false)
  })

  it("does not force ElevenLabs as the provider if provider changes while validation is pending", async (): Promise<void> => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "elevenlabs",
      elevenLabsApiKey: "",
    }
    const voices = createDeferred<Array<{ voice_id: string; name: string }>>()
    const models = createDeferred<Array<{ model_id: string; name: string }>>()
    getElevenLabsVoicesMock.mockReturnValue(voices.promise)
    getElevenLabsModelsMock.mockReturnValue(models.promise)

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    fireEvent.change(screen.getByLabelText("ElevenLabs API key"), {
      target: { value: "sk_test_inline" },
    })
    fireEvent.click(screen.getByRole("button", { name: "Test & Save" }))
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "browser",
    }
    voices.resolve([{ voice_id: "voice-1", name: "Voice 1" }])
    models.resolve([{ model_id: "model-1", name: "Model 1" }])

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          ttsProvider: "browser",
          elevenLabsApiKey: "sk_test_inline",
        })
      )
    })
    expect(setTTSSettingsMock).not.toHaveBeenCalledWith(
      expect.objectContaining({
        ttsProvider: "elevenlabs",
        elevenLabsApiKey: "sk_test_inline",
      })
    )
  })

  it("shows stable validation errors without exposing raw ElevenLabs details", async (): Promise<void> => {
    ttsSettingsRef.current = {
      ...ttsSettingsRef.current,
      ttsProvider: "elevenlabs",
      elevenLabsApiKey: "",
    }
    getElevenLabsVoicesMock.mockRejectedValue(
      Object.assign(new Error("Request failed for sk_secret_inline"), {
        response: { status: 401 },
      })
    )
    getElevenLabsModelsMock.mockResolvedValue([
      { model_id: "model-1", name: "Model 1" },
    ])

    render(<SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />)

    fireEvent.change(screen.getByLabelText("ElevenLabs API key"), {
      target: { value: "sk_secret_inline" },
    })
    fireEvent.click(screen.getByRole("button", { name: "Test & Save" }))

    expect(
      await screen.findByText("Invalid ElevenLabs API key. Check the key and try again.")
    ).toBeInTheDocument()
    expect(screen.queryByText(/sk_secret_inline/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Request failed/)).not.toBeInTheDocument()
  })
})
