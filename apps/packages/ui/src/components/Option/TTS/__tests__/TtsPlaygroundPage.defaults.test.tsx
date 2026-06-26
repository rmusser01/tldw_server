import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import TtsPlaygroundPage from "@/components/Option/TTS/TtsPlaygroundPage"

const DEFAULT_KITTEN_MODEL = "KittenML/kitten-tts-nano-0.8"

const ttsSettingsState = vi.hoisted(() => ({
  data: {
    ttsProvider: "",
    tldwTtsModel: "",
    tldwTtsVoice: "",
    tldwTtsResponseFormat: "mp3",
    tldwTtsSpeed: 1,
    elevenLabsApiKey: "",
    elevenLabsVoiceId: "",
    elevenLabsModel: "",
    openAITTSModel: "tts-1",
    openAITTSVoice: "alloy",
    playbackSpeed: 1,
    voice: ""
  } as Record<string, unknown>
}))

const providerDataState = vi.hoisted(() => ({
  hasAudio: true,
  tldwVoiceCatalog: [
    { id: "Bella", name: "Bella", language: "en" }
  ]
}))

vi.mock("@/services/tts", () => ({
  getTTSSettings: vi.fn(async () => ttsSettingsState.data),
  getTTSProvider: vi.fn(async () => "tldw"),
  setTTSSettings: vi.fn(async () => {}),
  DEFAULT_TTS_PROVIDER: "tldw",
  DEFAULT_TLDW_TTS_MODEL: "KittenML/kitten-tts-nano-0.8",
  DEFAULT_TLDW_TTS_VOICE: "Bella"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  }),
  Trans: ({ defaults }: { defaults: string }) => <>{defaults}</>
}))

vi.mock("@/hooks/useTtsProviderData", () => ({
  useTtsProviderData: () => ({
    hasAudio: providerDataState.hasAudio,
    providersInfo: null,
    tldwTtsModels: [],
    tldwVoiceCatalog: providerDataState.tldwVoiceCatalog,
    elevenLabsData: null,
    elevenLabsLoading: false,
    elevenLabsError: null,
    refetchElevenLabs: vi.fn()
  }),
  OPENAI_TTS_MODELS: [
    { value: "tts-1", label: "tts-1" },
    { value: "tts-1-hd", label: "tts-1-hd" }
  ],
  OPENAI_TTS_VOICES: {
    "tts-1": [{ value: "alloy", label: "Alloy" }],
    "tts-1-hd": [{ value: "alloy", label: "Alloy" }]
  }
}))

vi.mock("@/hooks/useTtsPlayground", () => ({
  useTtsPlayground: () => ({
    segments: [],
    isGenerating: false,
    generateSegments: vi.fn(async () => []),
    clearSegments: vi.fn(() => {})
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      ffmpegAvailable: true
    }
  })
}))

vi.mock("@/components/Option/TTS/TtsProviderCatalog", () => ({
  TtsProviderCatalog: () => <div data-testid="provider-catalog" />
}))

vi.mock("@/components/Option/Settings/TTSModeSettings", () => ({
  TTSModeSettings: () => <div data-testid="tts-mode-settings" />
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

describe("TtsPlaygroundPage defaults", () => {
  beforeEach(() => {
    providerDataState.hasAudio = true
    providerDataState.tldwVoiceCatalog = [
      { id: "Bella", name: "Bella", language: "en" }
    ]
    ttsSettingsState.data = {
      ttsProvider: "",
      tldwTtsModel: "",
      tldwTtsVoice: "",
      tldwTtsResponseFormat: "mp3",
      tldwTtsSpeed: 1,
      elevenLabsApiKey: "",
      elevenLabsVoiceId: "",
      elevenLabsModel: "",
      openAITTSModel: "tts-1",
      openAITTSVoice: "alloy",
      playbackSpeed: 1,
      voice: ""
    }
  })

  it("materializes the Kitten baseline when tldw settings are otherwise empty", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false
        }
      }
    })

    render(
      <QueryClientProvider client={queryClient}>
        <TtsPlaygroundPage />
      </QueryClientProvider>
    )

    await waitFor(() => {
      expect(screen.getByLabelText("tldw server model")).toHaveValue(
        DEFAULT_KITTEN_MODEL
      )
    })

    expect(screen.getByText("Bella (en)")).toBeInTheDocument()
    expect(screen.getByText(/Current provider:/).textContent).toContain(
      "tldw Server (Local)"
    )
  })

  it("renders the Text to Speech page heading", () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false
        }
      }
    })

    render(
      <QueryClientProvider client={queryClient}>
        <TtsPlaygroundPage />
      </QueryClientProvider>
    )

    expect(
      screen.getByRole("heading", { level: 1, name: "Text to Speech" })
    ).toBeInTheDocument()
  })

  it("renders a single shared setup state when the tldw server has no TTS provider", async () => {
    providerDataState.hasAudio = false
    providerDataState.tldwVoiceCatalog = []
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false
        }
      }
    })

    const { container } = render(
      <QueryClientProvider client={queryClient}>
        <TtsPlaygroundPage />
      </QueryClientProvider>
    )

    const statePanel = await screen.findByTestId("tts-no-provider-recovery")
    expect(statePanel).toHaveAttribute("data-ds-component", "StatePanel")
    expect(
      screen.getAllByText("No TTS provider detected on your server")
    ).toHaveLength(1)
    expect(screen.getByText(/Open Speech Settings/)).toBeInTheDocument()
    expect(screen.getByText(/Browser/)).toBeInTheDocument()
    expect(container.querySelectorAll(".ant-alert")).toHaveLength(0)
  })
})
