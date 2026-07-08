import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TTSModeSettings } from "../TTSModeSettings"

const {
  getTTSSettingsMock,
  setTTSSettingsMock,
  setElevenLabsApiKeyMock,
  setElevenLabsKeyValidMock,
  setElevenLabsKeyTestedAtMock,
  synthesizeSpeechMock,
  generateSpeechMock,
  generateOpenAITTSMock,
  getVoicesMock,
  getModelsMock,
  messageSuccessMock,
  messageErrorMock,
  setTTSEnabledMock,
  audioPlayMock,
  audioPauseMock,
  revokeObjectURLMock,
  createObjectURLMock,
  speechSynthesisSpeakMock,
  speechSynthesisCancelMock,
  speechSynthesisGetVoicesMock,
} = vi.hoisted(() => ({
  getTTSSettingsMock: vi.fn(),
  setTTSSettingsMock: vi.fn(async () => undefined),
  setElevenLabsApiKeyMock: vi.fn(async () => undefined),
  setElevenLabsKeyValidMock: vi.fn(async () => undefined),
  setElevenLabsKeyTestedAtMock: vi.fn(async () => undefined),
  synthesizeSpeechMock: vi.fn(async () => new ArrayBuffer(8)),
  generateSpeechMock: vi.fn(async () => new ArrayBuffer(8)),
  generateOpenAITTSMock: vi.fn(async () => new ArrayBuffer(8)),
  getVoicesMock: vi.fn(),
  getModelsMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageErrorMock: vi.fn(),
  setTTSEnabledMock: vi.fn(),
  audioPlayMock: vi.fn(async () => undefined),
  audioPauseMock: vi.fn(),
  revokeObjectURLMock: vi.fn(),
  createObjectURLMock: vi.fn(() => "blob:preview"),
  speechSynthesisSpeakMock: vi.fn(),
  speechSynthesisCancelMock: vi.fn(),
  speechSynthesisGetVoicesMock: vi.fn(() => []),
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallback?: string,
      values?: Record<string, string | number>
    ) => {
      if (!fallback) return _key
      if (!values) return fallback
      return Object.entries(values).reduce(
        (text, [key, value]) => text.replace(`{{${key}}}`, String(value)),
        fallback
      )
    },
  }),
}))

vi.mock("@/services/tts", () => ({
  DEFAULT_TLDW_TTS_MODEL: "KittenML/kitten-tts-nano-0.8",
  SUPPORTED_TLDW_TTS_FORMATS: ["mp3"],
  getTTSSettings: getTTSSettingsMock,
  setTTSSettings: setTTSSettingsMock,
  setElevenLabsApiKey: setElevenLabsApiKeyMock,
  setElevenLabsKeyValid: setElevenLabsKeyValidMock,
  setElevenLabsKeyTestedAt: setElevenLabsKeyTestedAtMock,
}))

vi.mock("@/services/elevenlabs", () => ({
  getVoices: getVoicesMock,
  getModels: getModelsMock,
  generateSpeech: generateSpeechMock,
}))

vi.mock("@/services/openai-tts", () => ({
  generateOpenAITTS: generateOpenAITTSMock,
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    synthesizeSpeech: synthesizeSpeechMock,
  },
}))

vi.mock("@/services/tldw/audio-voices", () => ({
  fetchTldwVoiceCatalog: vi.fn(async () => []),
  fetchTldwVoices: vi.fn(async () => []),
}))

vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: vi.fn(async () => null),
}))

vi.mock("@/services/tldw/audio-models", () => ({
  fetchTldwTtsModels: vi.fn(async () => []),
}))

vi.mock("@/services/tldw/voice-cloning", () => ({
  listCustomVoices: vi.fn(async () => []),
}))

vi.mock("@/services/tldw/tts-provider-keys", () => ({
  normalizeTtsProviderKey: vi.fn((value?: string) => value || ""),
  toServerTtsProviderKey: vi.fn((value?: string) => value || ""),
}))

vi.mock("@/services/tts-provider", () => ({
  inferTldwProviderFromModel: vi.fn(() => null),
}))

vi.mock("@/store/webui", () => ({
  useWebUI: () => ({
    setTTSEnabled: setTTSEnabledMock,
  }),
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: messageSuccessMock,
    error: messageErrorMock,
  }),
}))

const buildTtsSettings = (overrides: Record<string, unknown> = {}) => ({
  ttsEnabled: true,
  ttsProvider: "elevenlabs",
  browserTTSVoices: [],
  voice: "",
  ssmlEnabled: false,
  removeReasoningTagTTS: true,
  elevenLabsApiKey: "sk_existing",
  elevenLabsVoiceId: "",
  elevenLabsModel: "",
  elevenLabsKeyValid: null,
  elevenLabsKeyTestedAt: "",
  responseSplitting: "punctuation",
  openAITTSBaseUrl: "https://api.openai.com/v1",
  openAITTSApiKey: "",
  openAITTSModel: "tts-1",
  openAITTSVoice: "alloy",
  openAITTSKeyValid: null,
  openAITTSKeyTestedAt: "",
  ttsAutoPlay: false,
  playbackSpeed: 1,
  tldwTtsModel: "KittenML/kitten-tts-nano-0.8",
  tldwTtsVoice: "Bella",
  tldwTtsResponseFormat: "mp3",
  tldwTtsSpeed: 1,
  tldwTtsLanguage: "",
  tldwTtsStreaming: false,
  tldwTtsEmotion: "",
  tldwTtsEmotionIntensity: 1,
  tldwTtsNormalize: true,
  tldwTtsNormalizeUnits: false,
  tldwTtsNormalizeUrls: true,
  tldwTtsNormalizeEmails: true,
  tldwTtsNormalizePhones: true,
  tldwTtsNormalizePlurals: true,
  ...overrides,
})

const renderSettings = () => {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={client}>
      <TTSModeSettings />
    </QueryClientProvider>
  )
}

describe("TTSModeSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    class MockAudio {
      src: string
      onended: (() => void) | null = null
      onerror: (() => void) | null = null

      constructor(src: string) {
        this.src = src
      }

      play = audioPlayMock
      pause = audioPauseMock
    }

    vi.stubGlobal("Audio", MockAudio)
    vi.stubGlobal("URL", {
      createObjectURL: createObjectURLMock,
      revokeObjectURL: revokeObjectURLMock,
    })
    vi.stubGlobal(
      "SpeechSynthesisUtterance",
      function SpeechSynthesisUtterance(text: string) {
        return { text, voice: null, rate: 1 }
      }
    )
    const speechSynthesis = {
      speak: speechSynthesisSpeakMock,
      cancel: speechSynthesisCancelMock,
      getVoices: speechSynthesisGetVoicesMock,
    }
    vi.stubGlobal("speechSynthesis", speechSynthesis)
    Object.defineProperty(window, "speechSynthesis", {
      configurable: true,
      value: speechSynthesis,
    })
    getTTSSettingsMock.mockResolvedValue(buildTtsSettings())
    getVoicesMock.mockResolvedValue([{ voice_id: "voice-1", name: "Voice 1" }])
    getModelsMock.mockResolvedValue([{ model_id: "model-1", name: "Model 1" }])
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("shows persisted ElevenLabs validation status and last tested timestamp", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        elevenLabsKeyValid: true,
        elevenLabsKeyTestedAt: "2026-04-30T12:00:00.000Z",
      })
    )

    renderSettings()

    expect(await screen.findByText("Valid")).toBeInTheDocument()
    expect(screen.getByLabelText("valid")).toBeInTheDocument()
    expect(screen.getByText(/Last tested/)).toBeInTheDocument()
  })

  it("persists validation status when the user tests an ElevenLabs key", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        elevenLabsApiKey: "",
        elevenLabsKeyValid: null,
        elevenLabsKeyTestedAt: "",
      })
    )

    renderSettings()

    fireEvent.change(await screen.findByPlaceholderText("sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"), {
      target: { value: "sk_valid" },
    })
    fireEvent.click(screen.getByRole("button", { name: "Test" }))

    await waitFor(() => {
      expect(setElevenLabsApiKeyMock).toHaveBeenCalledWith("sk_valid")
      expect(setElevenLabsKeyValidMock).toHaveBeenCalledWith(true)
      expect(setElevenLabsKeyTestedAtMock).toHaveBeenCalledWith(expect.any(String))
    })
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
    expect(await screen.findByText("Valid")).toBeInTheDocument()
  })

  it("auto-revalidates a saved ElevenLabs key without saving unrelated settings", async () => {
    renderSettings()

    await waitFor(() => {
      expect(setElevenLabsApiKeyMock).toHaveBeenCalledWith("sk_existing")
      expect(setElevenLabsKeyValidMock).toHaveBeenCalledWith(true)
      expect(setElevenLabsKeyTestedAtMock).toHaveBeenCalledWith(expect.any(String))
    })
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
  })

  it("shows persisted OpenAI validation status and last tested timestamp", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "openai",
        openAITTSKeyValid: false,
        openAITTSKeyTestedAt: "2026-04-30T12:00:00.000Z",
      })
    )

    renderSettings()

    expect(await screen.findByText("Failed")).toBeInTheDocument()
    expect(screen.getByLabelText("failed")).toBeInTheDocument()
    expect(screen.getByText(/Last tested/)).toBeInTheDocument()
  })

  it("previews browser TTS with the selected voice and playback speed", async () => {
    const browserVoice = { name: "System Voice" }
    speechSynthesisGetVoicesMock.mockReturnValue([browserVoice])
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "browser",
        browserTTSVoices: [{ voiceName: "System Voice", lang: "en-US" }],
        voice: "System Voice",
        playbackSpeed: 1.25,
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))

    expect(speechSynthesisSpeakMock).toHaveBeenCalledTimes(1)
    expect(speechSynthesisSpeakMock.mock.calls[0][0]).toMatchObject({
      text: expect.stringContaining("preview"),
      voice: browserVoice,
      rate: 1.25,
    })
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
  })

  it("previews tldw TTS with current form values and never opens a websocket", async () => {
    const webSocketSpy = vi.fn()
    vi.stubGlobal("WebSocket", webSocketSpy)
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsModel: "kitten_tts",
        tldwTtsVoice: "Luna",
        tldwTtsResponseFormat: "wav",
        tldwTtsSpeed: 1.15,
        tldwTtsLanguage: "en",
        tldwTtsStreaming: true,
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))

    await waitFor(() => {
      expect(synthesizeSpeechMock).toHaveBeenCalledWith(
        expect.stringContaining("preview"),
        expect.objectContaining({
          model: "kitten_tts",
          voice: "Luna",
          responseFormat: "wav",
          speed: 1.15,
          language: "en",
          stream: false,
          signal: expect.any(AbortSignal),
        })
      )
    })
    expect(webSocketSpy).not.toHaveBeenCalled()
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
  })

  it("previews ElevenLabs with the typed API key instead of the saved key", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        elevenLabsApiKey: "sk_saved",
        elevenLabsVoiceId: "voice-1",
        elevenLabsModel: "model-1",
      })
    )

    renderSettings()

    fireEvent.change(await screen.findByPlaceholderText("sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"), {
      target: { value: "sk_unsaved" },
    })
    fireEvent.click(await screen.findByTestId("tts-preview-button"))

    await waitFor(() => {
      expect(generateSpeechMock).toHaveBeenCalledWith(
        "sk_unsaved",
        expect.stringContaining("preview"),
        "voice-1",
        "model-1",
        undefined,
        expect.objectContaining({ signal: expect.any(AbortSignal) })
      )
    })
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
  })

  it("previews OpenAI-compatible TTS through the existing speech helper", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "openai",
        openAITTSModel: "tts-1-hd",
        openAITTSVoice: "nova",
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))

    await waitFor(() => {
      expect(generateOpenAITTSMock).toHaveBeenCalledWith({
        text: expect.stringContaining("preview"),
        model: "tts-1-hd",
        voice: "nova",
        signal: expect.any(AbortSignal),
      })
    })
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
  })

  it("blocks preview before synthesis when required provider fields are missing", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsModel: "",
        tldwTtsVoice: "",
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))

    expect(await screen.findByText("Select a tldw TTS model first.")).toBeInTheDocument()
    expect(synthesizeSpeechMock).not.toHaveBeenCalled()
  })

  it("stops active server preview by aborting playback and revoking the object URL", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsModel: "kitten_tts",
        tldwTtsVoice: "Luna",
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))
    await screen.findByRole("button", { name: /Stop preview/i })
    fireEvent.click(screen.getByTestId("tts-preview-button"))

    expect(audioPauseMock).toHaveBeenCalled()
    expect(revokeObjectURLMock).toHaveBeenCalledWith("blob:preview")
  })
})
