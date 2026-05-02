import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TTSModeSettings } from "../TTSModeSettings"

const {
  getTTSSettingsMock,
  setTTSSettingsMock,
  setElevenLabsApiKeyMock,
  setElevenLabsKeyValidMock,
  setElevenLabsKeyTestedAtMock,
  getVoicesMock,
  getModelsMock,
  messageSuccessMock,
  messageErrorMock,
  setTTSEnabledMock,
} = vi.hoisted(() => ({
  getTTSSettingsMock: vi.fn(),
  setTTSSettingsMock: vi.fn(async () => undefined),
  setElevenLabsApiKeyMock: vi.fn(async () => undefined),
  setElevenLabsKeyValidMock: vi.fn(async () => undefined),
  setElevenLabsKeyTestedAtMock: vi.fn(async () => undefined),
  getVoicesMock: vi.fn(),
  getModelsMock: vi.fn(),
  messageSuccessMock: vi.fn(),
  messageErrorMock: vi.fn(),
  setTTSEnabledMock: vi.fn(),
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

describe("TTSModeSettings ElevenLabs key validation status", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getTTSSettingsMock.mockResolvedValue(buildTtsSettings())
    getVoicesMock.mockResolvedValue([{ voice_id: "voice-1", name: "Voice 1" }])
    getModelsMock.mockResolvedValue([{ model_id: "model-1", name: "Model 1" }])
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
})
