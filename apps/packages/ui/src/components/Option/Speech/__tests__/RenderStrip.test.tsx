import React from "react"
import { describe, expect, it, vi, beforeEach } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { RenderStrip, type RenderStripConfig } from "../RenderStrip"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/components/Common/UnifiedAudioPlayer", () => ({
  UnifiedAudioPlayer: ({ label }: { label?: string }) => (
    <div data-testid="unified-audio-player" aria-label={label} />
  )
}))

vi.mock("@/components/Common/TtsJobProgress", () => ({
  TtsJobProgress: ({ title }: { title?: string }) => (
    <div data-testid="tts-job-progress">{title}</div>
  )
}))

const baseConfig: RenderStripConfig = {
  provider: "tldw",
  voice: "af_heart",
  model: "kokoro",
  format: "mp3",
  speed: 1
}

describe("RenderStrip", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders config tags for provider, voice, format", () => {
    render(<RenderStrip id="r1" state="idle" config={baseConfig} />)
    expect(screen.getByText("kokoro")).toBeInTheDocument()
    expect(screen.getByText("af_heart")).toBeInTheDocument()
    expect(screen.getByText("MP3")).toBeInTheDocument()
  })

  it("labels browser render strips as Browser preview", () => {
    render(
      <RenderStrip
        id="r1"
        state="idle"
        config={{ provider: "browser", format: "mp3", speed: 1 }}
      />
    )

    expect(screen.getByText("Browser preview")).toBeInTheDocument()
    expect(screen.queryByText("browser")).not.toBeInTheDocument()
  })

  it("shows Generate button in idle state", () => {
    render(<RenderStrip id="r1" state="idle" config={baseConfig} />)
    expect(screen.getByRole("button", { name: "Generate audio" })).toBeInTheDocument()
  })

  it("does not show Generate button in ready state", () => {
    render(
      <RenderStrip
        id="r1"
        state="ready"
        config={baseConfig}
        audioUrl="blob:test"
      />
    )
    expect(screen.queryByRole("button", { name: "Generate audio" })).not.toBeInTheDocument()
  })

  it("calls onGenerate when Generate is clicked", () => {
    const onGenerate = vi.fn()
    render(
      <RenderStrip id="r1" state="idle" config={baseConfig} onGenerate={onGenerate} />
    )
    fireEvent.click(screen.getByRole("button", { name: "Generate audio" }))
    expect(onGenerate).toHaveBeenCalledWith("r1")
  })

  it("shows TtsJobProgress in generating state", () => {
    render(<RenderStrip id="r1" state="generating" config={baseConfig} />)
    expect(screen.getByTestId("tts-job-progress")).toBeInTheDocument()
  })

  it("shows UnifiedAudioPlayer in ready state", () => {
    render(
      <RenderStrip
        id="r1"
        state="ready"
        config={baseConfig}
        audioUrl="blob:test"
      />
    )
    expect(screen.getByTestId("unified-audio-player")).toBeInTheDocument()
  })

  it("shows error message and retry button in error state", () => {
    const onRetry = vi.fn()
    render(
      <RenderStrip
        id="r1"
        state="error"
        config={baseConfig}
        errorMessage="Connection failed"
        onRetry={onRetry}
      />
    )
    expect(screen.getByText("Connection failed")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Retry"))
    expect(onRetry).toHaveBeenCalledWith("r1")
  })

  it("shows a settings link for recoverable credential errors", () => {
    render(
      <RenderStrip
        id="r1"
        state="error"
        config={baseConfig}
        errorMessage="Credentials need attention. Open Settings -> Speech, check the selected provider credentials, and retry."
        errorSettingsHref="/settings/speech"
      />
    )

    expect(screen.getByRole("link", { name: "Open Settings" })).toHaveAttribute(
      "href",
      "/settings/speech"
    )
  })

  it("does not show speed tag when speed is 1", () => {
    render(<RenderStrip id="r1" state="idle" config={{ ...baseConfig, speed: 1 }} />)
    expect(screen.queryByText("1x")).not.toBeInTheDocument()
  })

  it("shows speed tag when speed is not 1", () => {
    render(<RenderStrip id="r1" state="idle" config={{ ...baseConfig, speed: 1.5 }} />)
    expect(screen.getByText("1.5x")).toBeInTheDocument()
  })

  it("calls onConfigTagClick when voice tag is clicked", () => {
    const onConfigTagClick = vi.fn()
    render(
      <RenderStrip
        id="r1"
        state="idle"
        config={baseConfig}
        onConfigTagClick={onConfigTagClick}
      />
    )
    fireEvent.click(screen.getByText("af_heart"))
    expect(onConfigTagClick).toHaveBeenCalledWith("r1", "voice")
  })

  it("has correct aria-label", () => {
    render(<RenderStrip id="r1" state="idle" config={baseConfig} />)
    expect(
      screen.getByRole("region", { name: "Render strip: kokoro af_heart" })
    ).toBeInTheDocument()
  })
})
