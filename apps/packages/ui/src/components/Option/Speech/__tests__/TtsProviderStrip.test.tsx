import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { TtsProviderStrip } from "../TtsProviderStrip"
import type { TtsPresetKey } from "@/hooks/useTtsPlayground"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

const defaults = {
  provider: "openai",
  model: "tts-1",
  voice: "alloy",
  format: "mp3",
  speed: 1,
  presetValue: "balanced" as TtsPresetKey,
  onPresetChange: vi.fn(),
  onLabelClick: vi.fn(),
  onGearClick: vi.fn()
}

describe("TtsProviderStrip", () => {
  it("renders model, voice, format as visible text", () => {
    render(<TtsProviderStrip {...defaults} />)

    expect(screen.getByText("tts-1")).toBeInTheDocument()
    expect(screen.getByText("alloy")).toBeInTheDocument()
    expect(screen.getByText("mp3")).toBeInTheDocument()
  })

  it('calls onLabelClick("voice", "model") when model tag is clicked', () => {
    const onLabelClick = vi.fn()
    render(<TtsProviderStrip {...defaults} onLabelClick={onLabelClick} />)

    fireEvent.click(screen.getByText("tts-1"))
    expect(onLabelClick).toHaveBeenCalledWith("voice", "model")
  })

  it("simplifies display for browser provider", () => {
    render(<TtsProviderStrip {...defaults} provider="browser" />)

    expect(screen.getByText("Browser local output")).toBeInTheDocument()
    expect(screen.queryByText("mp3")).not.toBeInTheDocument()
  })
})
