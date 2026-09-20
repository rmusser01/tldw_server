import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { TtsVoiceTab } from "../TtsVoiceTab"
import { TtsOutputTab } from "../TtsOutputTab"
import { TtsAdvancedTab } from "../TtsAdvancedTab"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))

describe("TtsVoiceTab", () => {
  it("renders provider, model, and voice selectors", () => {
    render(
      <TtsVoiceTab
        provider="tldw"
        model="kokoro"
        voice="af_heart"
        onProviderChange={vi.fn()}
        onModelChange={vi.fn()}
        onVoiceChange={vi.fn()}
        modelOptions={[{ label: "kokoro", value: "kokoro" }]}
        voiceOptions={[{ label: "af_heart", value: "af_heart" }]}
        focusField={null}
        onFocusHandled={vi.fn()}
      />
    )
    expect(screen.getByText("Provider")).toBeInTheDocument()
    expect(screen.getByText("Model")).toBeInTheDocument()
    expect(screen.getByText("Voice")).toBeInTheDocument()
  })
})

describe("TtsOutputTab", () => {
  it("renders format, speed, and splitting controls", () => {
    render(
      <TtsOutputTab
        format="mp3"
        synthesisSpeed={1}
        playbackSpeed={1}
        responseSplitting="punctuation"
        streaming={false}
        canStream={true}
        streamFormatSupported={true}
        onFormatChange={vi.fn()}
        onSynthesisSpeedChange={vi.fn()}
        onPlaybackSpeedChange={vi.fn()}
        onResponseSplittingChange={vi.fn()}
        onStreamingChange={vi.fn()}
        formatOptions={[{ label: "mp3", value: "mp3" }]}
        normalize={true}
        onNormalizeChange={vi.fn()}
        normalizeUnits={false}
        onNormalizeUnitsChange={vi.fn()}
        normalizeUrls={true}
        onNormalizeUrlsChange={vi.fn()}
        normalizeEmails={true}
        onNormalizeEmailsChange={vi.fn()}
        normalizePhones={true}
        onNormalizePhonesChange={vi.fn()}
        normalizePlurals={true}
        onNormalizePluralsChange={vi.fn()}
        focusField={null}
        onFocusHandled={vi.fn()}
      />
    )
    expect(screen.getByText("Format")).toBeInTheDocument()
    expect(screen.getByText("Synthesis Speed")).toBeInTheDocument()
    expect(screen.getByText("Playback Speed")).toBeInTheDocument()
    expect(screen.getByText("Response Splitting")).toBeInTheDocument()
  })
})

describe("TtsAdvancedTab", () => {
  it("renders draft editor and SSML toggles", () => {
    render(
      <TtsAdvancedTab
        useDraftEditor={false}
        onDraftEditorChange={vi.fn()}
        useTtsJob={false}
        onTtsJobChange={vi.fn()}
        ssmlEnabled={false}
        onSsmlChange={vi.fn()}
        removeReasoning={true}
        onRemoveReasoningChange={vi.fn()}
        allowFallback={true}
        onAllowFallbackChange={vi.fn()}
        isTldw={true}
        onOpenVoiceCloning={vi.fn()}
      />
    )
    expect(screen.getByText("Draft editor")).toBeInTheDocument()
    expect(screen.getByText(/SSML/)).toBeInTheDocument()
  })
})
