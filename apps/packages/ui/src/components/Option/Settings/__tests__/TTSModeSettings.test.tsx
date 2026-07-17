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
  audioInstancesMock,
  fetchTldwVoiceCatalogMock,
  fetchTldwVoicesMock,
  fetchTtsProvidersMock,
  fetchTldwTtsModelsMock,
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
  audioInstancesMock: [] as Array<{ playbackRate: number; src: string }>,
  fetchTldwVoiceCatalogMock: vi.fn(),
  fetchTldwVoicesMock: vi.fn(),
  fetchTtsProvidersMock: vi.fn(),
  fetchTldwTtsModelsMock: vi.fn(),
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
  SUPPORTED_TLDW_TTS_FORMATS: ["mp3", "wav", "opus"],
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
  fetchTldwVoiceCatalog: fetchTldwVoiceCatalogMock,
  fetchTldwVoices: fetchTldwVoicesMock,
}))

vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: fetchTtsProvidersMock,
}))

vi.mock("@/services/tldw/audio-models", () => ({
  fetchTldwTtsModels: fetchTldwTtsModelsMock,
}))

vi.mock("@/services/tldw/voice-cloning", () => ({
  listCustomVoices: vi.fn(async () => []),
}))

vi.mock("@/services/tldw/tts-provider-keys", () => ({
  normalizeTtsProviderKey: vi.fn((value?: string) => value || ""),
  toServerTtsProviderKey: vi.fn((value?: string) => value || ""),
}))

vi.mock("@/services/tts-provider", () => ({
  formatToMimeType: vi.fn((format: string) => {
    if (format === "wav") return "audio/wav"
    if (format === "opus") return "audio/opus"
    return "audio/mpeg"
  }),
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
  tldwTtsBackend: "",
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
    audioInstancesMock.length = 0
    synthesizeSpeechMock.mockResolvedValue(new ArrayBuffer(8))
    generateSpeechMock.mockResolvedValue(new ArrayBuffer(8))
    generateOpenAITTSMock.mockResolvedValue(new ArrayBuffer(8))
    class MockAudio {
      src: string
      playbackRate = 1
      onended: (() => void) | null = null
      onerror: (() => void) | null = null

      constructor(src: string) {
        this.src = src
        audioInstancesMock.push(this)
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
    fetchTldwVoiceCatalogMock.mockResolvedValue([])
    fetchTldwVoicesMock.mockResolvedValue([])
    fetchTtsProvidersMock.mockResolvedValue(null)
    fetchTldwTtsModelsMock.mockResolvedValue([])
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
        playbackSpeed: 1.35,
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
    await waitFor(() => {
      expect(audioPlayMock).toHaveBeenCalled()
    })
    expect(audioInstancesMock.at(-1)?.playbackRate).toBe(1.35)
    expect(webSocketSpy).not.toHaveBeenCalled()
    expect(setTTSSettingsMock).not.toHaveBeenCalled()
  })

  it("labels tldw preview blobs with the selected response format MIME type", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsModel: "kitten_tts",
        tldwTtsVoice: "Luna",
        tldwTtsResponseFormat: "opus",
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))

    await waitFor(() => {
      expect(createObjectURLMock).toHaveBeenCalled()
    })
    expect((createObjectURLMock.mock.calls[0][0] as Blob).type).toBe("audio/opus")
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

  it("shows that OpenAI-compatible previews still depend on server speech credentials", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "openai",
      })
    )

    renderSettings()

    expect(
      await screen.findByText(/server speech API/i)
    ).toBeInTheDocument()
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

  it("lets the user stop a pending server preview request", async () => {
    let capturedSignal: AbortSignal | undefined
    synthesizeSpeechMock.mockImplementation((_text, options) => {
      capturedSignal = options?.signal
      return new Promise<ArrayBuffer>(() => {})
    })
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsModel: "kitten_tts",
        tldwTtsVoice: "Luna",
      })
    )

    renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))
    fireEvent.click(await screen.findByRole("button", { name: /Stop preview/i }))

    expect(capturedSignal?.aborted).toBe(true)
  })

  it("cleans up active preview playback on unmount", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsModel: "kitten_tts",
        tldwTtsVoice: "Luna",
      })
    )

    const { unmount } = renderSettings()

    fireEvent.click(await screen.findByTestId("tts-preview-button"))
    await screen.findByRole("button", { name: /Stop preview/i })
    unmount()

    expect(audioPauseMock).toHaveBeenCalled()
    expect(revokeObjectURLMock).toHaveBeenCalledWith("blob:preview")
  })

  it("cleans up active preview playback when the provider changes", async () => {
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
    fireEvent.mouseDown(
      screen.getByLabelText("generalSettings.tts.ttsProvider.label")
    )
    fireEvent.click(await screen.findByText("Browser TTS (no setup needed)"))

    expect(audioPauseMock).toHaveBeenCalled()
    expect(revokeObjectURLMock).toHaveBeenCalledWith("blob:preview")
  })
})

const explicitBackendCatalog = {
  supports_explicit_backend: true,
  providers: {
    "gateway:Company": {
      display_name: "Company Speech",
      models: ["Vendor/Exact-Case", "Vendor/Free-Form"],
      default_model: "Vendor/Exact-Case",
      model_capabilities: {
        "Vendor/Exact-Case": {
          default_voice: "Narrator",
          voices: ["Narrator", "Guide"],
          requires_freeform_voice: false,
          formats: ["wav", "mp3"],
          default_format: "wav",
          native_formats: ["mp3"]
        },
        "Vendor/Free-Form": {
          default_voice: null,
          voices: [],
          requires_freeform_voice: true,
          formats: ["wav"],
          native_formats: ["wav"]
        }
      },
      fallback: {
        available: true,
        targets: ["openrouter", "gateway:backup"]
      },
      base_url: "https://private-gateway.invalid",
      credential_source: "user-api-key"
    },
    "gateway:Backup": {
      display_name: "Backup Speech",
      models: ["Backup/Only"],
      default_model: "Backup/Only",
      model_capabilities: {
        "Backup/Only": {
          default_voice: "BackupVoice",
          voices: ["BackupVoice"],
          requires_freeform_voice: false,
          formats: ["flac"],
          native_formats: ["flac"]
        }
      },
      fallback: { available: false, targets: [] }
    }
  },
  voices: {}
}

const selectAntOption = async (label: string, option: string) => {
  fireEvent.mouseDown(await screen.findByLabelText(label))
  fireEvent.click(
    await screen.findByText(option, {
      selector: ".ant-select-item-option-content"
    })
  )
}

describe("TTSModeSettings explicit backend discovery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsBackend: "",
        tldwTtsModel: "legacy-model",
        tldwTtsVoice: "legacy-voice"
      })
    )
    fetchTldwVoiceCatalogMock.mockResolvedValue([])
    fetchTldwVoicesMock.mockResolvedValue([])
    fetchTtsProvidersMock.mockResolvedValue(explicitBackendCatalog)
    fetchTldwTtsModelsMock.mockImplementation(async (backend?: string) => {
      if (!backend) return [{ id: "legacy-model", label: "legacy-model" }]
      const provider =
        explicitBackendCatalog.providers[
          backend as keyof typeof explicitBackendCatalog.providers
        ]
      return (provider?.models || []).map((id) => ({ id, label: id }))
    })
  })

  it("keeps legacy automatic mode when the server lacks explicit backend support", async () => {
    fetchTtsProvidersMock.mockResolvedValue({
      supports_explicit_backend: false,
      providers: { openai: { models: ["tts-1"] } },
      voices: {}
    })

    renderSettings()

    expect(await screen.findByLabelText("tldw TTS model")).toBeInTheDocument()
    expect(screen.queryByLabelText("tldw TTS backend")).not.toBeInTheDocument()
    expect(fetchTldwTtsModelsMock).toHaveBeenCalledWith(undefined)
  })

  it("ignores a saved explicit backend when connected to an older server", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsBackend: "gateway:company",
        tldwTtsModel: "Vendor/Exact-Case",
        tldwTtsVoice: "Narrator"
      })
    )
    fetchTtsProvidersMock.mockResolvedValue({
      supports_explicit_backend: false,
      providers: { openai: { models: ["tts-1"] } },
      voices: {}
    })

    renderSettings()

    expect(await screen.findByLabelText("tldw TTS model")).toBeInTheDocument()
    await waitFor(() => {
      expect(fetchTldwTtsModelsMock).toHaveBeenCalledWith(undefined)
    })
    expect(
      fetchTldwTtsModelsMock.mock.calls.some(
        ([backend]) => backend === "gateway:company"
      )
    ).toBe(false)

    const saveButton = screen.getByRole("button", { name: "saved" })
    expect(saveButton).toBeDisabled()

    fireEvent.click(
      screen.getByLabelText("generalSettings.tts.ttsAutoPlay.label")
    )
    fireEvent.click(screen.getByRole("button", { name: "save" }))

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          tldwTtsBackend: "gateway:company",
          ttsAutoPlay: true
        })
      )
    })
  })

  it("shows display names while retaining canonical backend values", async () => {
    renderSettings()

    await selectAntOption("tldw TTS backend", "Company Speech")

    expect(screen.getAllByText("Company Speech").length).toBeGreaterThan(0)
    expect(fetchTldwTtsModelsMock).toHaveBeenCalledWith("gateway:Company")

    fireEvent.click(screen.getByRole("button", { name: /save/i }))
    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          tldwTtsBackend: "gateway:Company",
          tldwTtsModel: "Vendor/Exact-Case",
          tldwTtsVoice: "Narrator",
          tldwTtsResponseFormat: "mp3"
        })
      )
    })
  })

  it("resets an unsupported response format in the backend selection update", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsResponseFormat: "ogg"
      })
    )
    renderSettings()

    await selectAntOption("tldw TTS backend", "Company Speech")
    fireEvent.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenLastCalledWith(
        expect.objectContaining({
          tldwTtsBackend: "gateway:Company",
          tldwTtsModel: "Vendor/Exact-Case",
          tldwTtsVoice: "Narrator",
          tldwTtsResponseFormat: "wav"
        })
      )
    })
  })

  it("resets an unsupported response format in the model selection update", async () => {
    getTTSSettingsMock.mockResolvedValue(
      buildTtsSettings({
        ttsProvider: "tldw",
        tldwTtsBackend: "gateway:Company",
        tldwTtsModel: "Vendor/Exact-Case",
        tldwTtsVoice: "Narrator",
        tldwTtsResponseFormat: "mp3"
      })
    )
    renderSettings()

    await selectAntOption("tldw TTS model", "Vendor/Free-Form")
    const freeformVoice = await screen.findByLabelText("tldw TTS voice")
    expect(freeformVoice).toBeRequired()
    expect(freeformVoice).toHaveValue("")
    fireEvent.change(freeformVoice, { target: { value: "ManualVoice" } })
    fireEvent.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenLastCalledWith(
        expect.objectContaining({
          tldwTtsBackend: "gateway:Company",
          tldwTtsModel: "Vendor/Free-Form",
          tldwTtsVoice: "ManualVoice",
          tldwTtsResponseFormat: "wav"
        })
      )
    })
  })

  it("resets model and voice atomically without hidden per-backend history", async () => {
    renderSettings()

    await selectAntOption("tldw TTS backend", "Company Speech")
    await selectAntOption("tldw TTS model", "Vendor/Free-Form")

    const freeformVoice = await screen.findByLabelText("tldw TTS voice")
    expect(freeformVoice).toBeRequired()
    expect(freeformVoice).toHaveValue("")
    fireEvent.change(freeformVoice, { target: { value: "ManualVoice" } })

    await selectAntOption("tldw TTS backend", "Backup Speech")
    await selectAntOption("tldw TTS backend", "Company Speech")

    fireEvent.click(screen.getByRole("button", { name: /save/i }))
    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenLastCalledWith(
        expect.objectContaining({
          tldwTtsBackend: "gateway:Company",
          tldwTtsModel: "Vendor/Exact-Case",
          tldwTtsVoice: "Narrator"
        })
      )
    })
  })

  it("shows sanitized possible fallback targets without authority details", async () => {
    fetchTtsProvidersMock.mockResolvedValue({
      ...explicitBackendCatalog,
      providers: {
        ...explicitBackendCatalog.providers,
        "gateway:Company": {
          ...explicitBackendCatalog.providers["gateway:Company"],
          fallback: {
            available: true,
            targets: [
              "openrouter",
              "gateway:backup",
              "https://private-gateway.invalid",
              "credential:user-api-key",
              " gateway:spaced",
              "gateway:spaced ",
              "OpenRouter",
              "gateway:Backup",
              "gateway:bad_slug",
              `gateway:${"a".repeat(64)}`
            ]
          }
        }
      }
    })
    renderSettings()

    await selectAntOption("tldw TTS backend", "Company Speech")

    const disclosure = await screen.findByText(/^Possible fallback targets:/)
    expect(disclosure).toHaveTextContent(
      /^Possible fallback targets: openrouter, gateway:backup$/
    )
    expect(disclosure).not.toHaveTextContent("private-gateway")
    expect(disclosure).not.toHaveTextContent("credential")
    expect(disclosure).not.toHaveTextContent("spaced")
    expect(disclosure).not.toHaveTextContent("OpenRouter")
    expect(disclosure).not.toHaveTextContent("Backup")
    expect(disclosure).not.toHaveTextContent("bad_slug")
  })

  it("renders normalized malformed discovery without exposing authority fields", async () => {
    const { normalizeTtsProvidersResponse } = await vi.importActual<
      typeof import("@/services/tldw/audio-providers")
    >("@/services/tldw/audio-providers")
    const normalized = normalizeTtsProvidersResponse({
      providers: {
        "gateway:safe": {
          display_name: "Safe Speech",
          models: ["Safe/Model", null, 7],
          default_model: "Safe/Model",
          model_capabilities: {
            "Safe/Model": {
              voices: { unsafe: true },
              formats: "mp3",
              default_voice: { unsafe: true }
            }
          },
          fallback: {
            available: true,
            targets: ["openrouter", 42]
          },
          base_url: "https://private-gateway.invalid",
          credential_source: "user-api-key"
        }
      },
      voices: {},
      supports_explicit_backend: true
    })
    fetchTtsProvidersMock.mockResolvedValue(normalized)

    renderSettings()
    await selectAntOption("tldw TTS backend", "Safe Speech")

    expect(await screen.findByLabelText("tldw TTS model")).toBeInTheDocument()
    expect(await screen.findByLabelText("tldw TTS voice")).toBeInTheDocument()
    expect(document.body).not.toHaveTextContent("private-gateway")
    expect(document.body).not.toHaveTextContent("user-api-key")
  })

  it("returns to automatic inference and persists an empty backend", async () => {
    renderSettings()

    await selectAntOption("tldw TTS backend", "Company Speech")
    await selectAntOption(
      "tldw TTS backend",
      "Automatic (legacy model inference)"
    )
    fireEvent.click(screen.getByRole("button", { name: /save/i }))

    await waitFor(() => {
      expect(setTTSSettingsMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ tldwTtsBackend: "" })
      )
    })
  })
})
