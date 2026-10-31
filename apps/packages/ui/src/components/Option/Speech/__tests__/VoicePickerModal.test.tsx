import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { VoicePickerModal, type VoiceSelection } from "../VoicePickerModal"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/services/tldw/audio-providers", () => ({
  fetchTtsProviders: vi.fn().mockResolvedValue(null)
}))

const { synthesizeSpeechDetailedMock } = vi.hoisted(() => ({
  synthesizeSpeechDetailedMock: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    synthesizeSpeechDetailed: synthesizeSpeechDetailedMock
  }
}))

vi.mock("@/services/tldw/tts-provider-keys", () => ({
  normalizeTtsProviderKey: (k: string) => k.toLowerCase()
}))

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } }
})

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
)

const mockProvidersInfo = {
  providers: {
    kokoro: {
      provider_name: "Kokoro",
      formats: ["mp3", "wav"],
      supports_streaming: true
    }
  },
  voices: {
    kokoro: [
      { id: "af_heart", name: "Heart", language: "en" },
      { id: "am_adam", name: "Adam", language: "en" }
    ]
  }
}

const gatewayProvidersInfo = {
  supports_explicit_backend: true,
  providers: {
    "gateway:Primary": {
      provider_name: "openai-compatible",
      display_name: "Primary gateway",
      default_model: "SpeechModel",
      models: ["SpeechModel"],
      formats: ["mp3"]
    }
  },
  voices: {
    "gateway:Primary": [{ id: "narrator", name: "Narrator" }]
  }
}

describe("VoicePickerModal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    synthesizeSpeechDetailedMock.mockResolvedValue({
      buffer: new ArrayBuffer(100),
      fallbackUsed: false
    })
    localStorage.clear()
  })

  it("renders modal with search input when open", () => {
    render(
      <VoicePickerModal
        open
        onClose={vi.fn()}
        onSelect={vi.fn()}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    expect(screen.getByText("Choose a Voice")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Search voices, providers...")).toBeInTheDocument()
  })

  it("does not render content when closed", () => {
    render(
      <VoicePickerModal
        open={false}
        onClose={vi.fn()}
        onSelect={vi.fn()}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    expect(screen.queryByText("Choose a Voice")).not.toBeInTheDocument()
  })

  it("displays server provider voices", () => {
    render(
      <VoicePickerModal
        open
        onClose={vi.fn()}
        onSelect={vi.fn()}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    expect(screen.getByText("Heart")).toBeInTheDocument()
    expect(screen.getByText("Adam")).toBeInTheDocument()
  })

  it("displays OpenAI voices", () => {
    render(
      <VoicePickerModal
        open
        onClose={vi.fn()}
        onSelect={vi.fn()}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    expect(screen.getByText("alloy")).toBeInTheDocument()
  })

  it("filters voices by search query", () => {
    render(
      <VoicePickerModal
        open
        onClose={vi.fn()}
        onSelect={vi.fn()}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    const searchInput = screen.getByPlaceholderText("Search voices, providers...")
    fireEvent.change(searchInput, { target: { value: "Heart" } })
    expect(screen.getByText("Heart")).toBeInTheDocument()
    expect(screen.queryByText("alloy")).not.toBeInTheDocument()
  })

  it("calls onSelect and onClose when a voice is clicked", () => {
    const onSelect = vi.fn()
    const onClose = vi.fn()
    render(
      <VoicePickerModal
        open
        onClose={onClose}
        onSelect={onSelect}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    fireEvent.click(screen.getByText("Heart"))
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: "tldw",
        voice: "af_heart"
      })
    )
    expect(onClose).toHaveBeenCalled()
  })

  it("saves and displays recent voices", () => {
    const storedVoices: VoiceSelection[] = [
      { provider: "tldw", voice: "af_heart", model: "kokoro" }
    ]
    localStorage.setItem("tts-recent-voices", JSON.stringify(storedVoices))

    render(
      <VoicePickerModal
        open
        onClose={vi.fn()}
        onSelect={vi.fn()}
        providersInfo={mockProvidersInfo}
      />,
      { wrapper }
    )
    expect(screen.getByText("Recent")).toBeInTheDocument()
    expect(screen.getByText("kokoro/af_heart")).toBeInTheDocument()
  })

  it("keeps the exact named backend for selection and generated previews", async () => {
    const onSelect = vi.fn()
    render(
      <VoicePickerModal
        open
        onClose={vi.fn()}
        onSelect={onSelect}
        providersInfo={gatewayProvidersInfo}
      />,
      { wrapper }
    )

    fireEvent.click(screen.getByRole("button", { name: "Preview Narrator" }))
    await waitFor(() => {
      expect(synthesizeSpeechDetailedMock).toHaveBeenCalledWith(
        "Hello, this is a voice preview.",
        expect.objectContaining({
          backend: "gateway:Primary",
          model: "SpeechModel",
          voice: "narrator"
        })
      )
    })

    fireEvent.click(screen.getByText("Narrator"))
    expect(onSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: "tldw",
        backend: "gateway:Primary",
        model: "SpeechModel",
        voice: "narrator"
      })
    )
  })
})
