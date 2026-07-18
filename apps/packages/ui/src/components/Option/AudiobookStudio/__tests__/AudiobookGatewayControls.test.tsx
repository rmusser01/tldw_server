// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { GenerationPanel } from "../Generation/GenerationPanel"
import { ChapterVoiceSelector } from "../ChapterEditor/ChapterVoiceSelector"
import { useAudiobookStudioStore } from "@/store/audiobook-studio"

const ttsProviderDataRef = vi.hoisted(() => ({
  current: {
    hasAudio: true,
    providersInfo: {
      supports_explicit_backend: true,
      providers: {
        "gateway:primary": {
          display_name: "Primary gateway",
          default_model: "SpeechModel",
          models: ["SpeechModel"]
        }
      },
      voices: {}
    },
    tldwTtsModels: [{ id: "SpeechModel", label: "SpeechModel" }],
    tldwVoiceCatalog: [{ id: "narrator", name: "Narrator" }],
    elevenLabsData: null,
    elevenLabsLoading: false
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: {
      ttsProvider: "tldw",
      tldwTtsBackend: "gateway:primary",
      tldwTtsModel: "SpeechModel",
      tldwTtsVoice: "narrator"
    }
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [null]
}))

vi.mock("@/hooks/useAudiobookGeneration", () => ({
  useAudiobookGeneration: () => ({
    generateAllChapters: vi.fn(),
    cancelGeneration: vi.fn()
  })
}))

vi.mock("@/hooks/useTtsProviderData", () => ({
  OPENAI_TTS_MODELS: [],
  OPENAI_TTS_VOICES: {},
  useTtsProviderData: () => ttsProviderDataRef.current
}))

describe("audiobook gateway controls", () => {
  beforeEach(() => {
    ttsProviderDataRef.current.tldwVoiceCatalog = [
      { id: "narrator", name: "Narrator" }
    ]
    useAudiobookStudioStore.setState({
      chapters: [
        {
          id: "chapter-1",
          title: "Chapter 1",
          content: "Narration",
          order: 0,
          voiceConfig: {},
          status: "pending"
        }
      ],
      defaultVoiceConfig: {
        provider: "tldw",
        tldwBackend: "gateway:primary",
        tldwAllowFallback: false,
        tldwModel: "SpeechModel",
        tldwVoice: "narrator"
      },
      isGenerating: false,
      currentGeneratingId: null
    })
  })

  it("shows the persisted default backend and per-request fallback policy", () => {
    render(<GenerationPanel />)

    expect(screen.getByLabelText("Audiobook backend")).toHaveTextContent(
      "Primary gateway"
    )
    expect(screen.getByLabelText("Allow configured fallback")).not.toBeChecked()
  })

  it("round-trips chapter fallback changes without dropping the backend", () => {
    const onChange = vi.fn()
    render(
      <ChapterVoiceSelector
        voiceConfig={{
          provider: "tldw",
          tldwBackend: "gateway:primary",
          tldwAllowFallback: true,
          tldwModel: "SpeechModel",
          tldwVoice: "narrator"
        }}
        onChange={onChange}
        compact={false}
      />
    )

    expect(screen.getByLabelText("Chapter TTS backend")).toHaveTextContent(
      "Primary gateway"
    )
    fireEvent.click(screen.getByLabelText("Chapter allow configured fallback"))
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        tldwBackend: "gateway:primary",
        tldwAllowFallback: false
      })
    )
  })

  it("keeps an explicit automatic override instead of inheriting the global backend", () => {
    useAudiobookStudioStore.setState({
      defaultVoiceConfig: {
        provider: "tldw",
        tldwBackend: "",
        tldwAllowFallback: true
      }
    })

    render(<GenerationPanel />)

    expect(screen.getByLabelText("Audiobook backend")).toHaveTextContent(
      "Automatic (legacy model inference)"
    )
  })

  it("keeps backend-scoped models selectable when no unfiltered voices are returned", () => {
    ttsProviderDataRef.current.tldwVoiceCatalog = []

    render(<GenerationPanel />)

    expect(screen.getByText("SpeechModel")).toBeInTheDocument()
  })
})
