import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import OptionTts from "../option-tts"
import OptionStt from "../option-stt"
import OptionSpeech from "../option-speech"

vi.mock("~/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    children,
    routeId,
    routeLabel
  }: {
    children: React.ReactNode
    routeId: string
    routeLabel: string
  }) => (
    <div data-testid="route-boundary" data-route-id={routeId} data-route-label={routeLabel}>
      {children}
    </div>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

vi.mock("@/components/Option/TTS/TtsPlaygroundPage", () => ({
  __esModule: true,
  default: () => <div data-testid="tts-playground">TTS</div>
}))

vi.mock("@/components/Option/STT/SttPlaygroundPage", () => ({
  __esModule: true,
  default: () => <div data-testid="stt-playground">STT</div>
}))

vi.mock("@/components/Option/Speech/SpeechPlaygroundPage", () => ({
  __esModule: true,
  default: ({
    initialMode,
    lockedMode,
    hideModeSwitcher
  }: {
    initialMode?: string
    lockedMode?: string
    hideModeSwitcher?: boolean
  }) => (
    <div
      data-testid="speech-playground"
      data-mode={initialMode ?? "roundtrip"}
      data-locked-mode={lockedMode ?? ""}
      data-hide-mode-switcher={hideModeSwitcher ? "true" : "false"}
    >
      Speech
    </div>
  )
}))

describe("audio option routes", () => {
  it("routes /tts into the shared speech playground locked to TTS mode", async () => {
    render(<OptionTts />)

    const speech = await screen.findByTestId("speech-playground")
    const boundary = screen.getByTestId("route-boundary")

    expect(screen.getByTestId("option-layout")).toBeVisible()
    expect(boundary).toHaveAttribute("data-route-id", "tts")
    expect(boundary).toHaveAttribute("data-route-label", "TTS Playground")
    expect(speech).toBeVisible()
    expect(speech).toHaveAttribute("data-locked-mode", "listen")
    expect(speech).toHaveAttribute("data-hide-mode-switcher", "true")
    expect(screen.queryByTestId("tts-playground")).not.toBeInTheDocument()
  })

  it("renders dedicated STT playground on /stt route component", async () => {
    render(<OptionStt />)

    const boundary = screen.getByTestId("route-boundary")

    expect(screen.getByTestId("option-layout")).toBeVisible()
    expect(boundary).toHaveAttribute("data-route-id", "stt")
    expect(boundary).toHaveAttribute("data-route-label", "STT Playground")
    expect(await screen.findByTestId("stt-playground")).toBeVisible()
    expect(screen.queryByTestId("speech-playground")).not.toBeInTheDocument()
  })

  it("keeps /speech route mapped to the unified speech playground", () => {
    render(<OptionSpeech />)

    const speech = screen.getByTestId("speech-playground")
    const boundary = screen.getByTestId("route-boundary")

    expect(boundary).toHaveAttribute("data-route-id", "speech")
    expect(boundary).toHaveAttribute("data-route-label", "Speech Playground")
    expect(speech).toBeVisible()
    expect(speech).toHaveAttribute("data-mode", "roundtrip")
    expect(screen.queryByTestId("tts-playground")).not.toBeInTheDocument()
    expect(screen.queryByTestId("stt-playground")).not.toBeInTheDocument()
  })
})
