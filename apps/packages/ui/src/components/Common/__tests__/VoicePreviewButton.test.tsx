import React from "react"
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest"
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react"
import { VoicePreviewButton } from "../VoicePreviewButton"

const mockSynthesizeSpeechDetailed = vi.fn()

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    synthesizeSpeechDetailed: (...args: any[]) => mockSynthesizeSpeechDetailed(...args),
  },
}))

vi.mock("antd", () => ({
  Button: ({ children, disabled, onClick, ...props }: any) => (
    <button disabled={disabled} onClick={onClick} aria-label={props["aria-label"]}>
      {children}
    </button>
  ),
  Tooltip: ({ children }: any) => <>{children}</>,
}))

vi.mock("lucide-react", () => ({
  Play: (props: any) => <svg data-testid="play-icon" {...props} />,
  Square: (props: any) => <svg data-testid="square-icon" {...props} />,
  Loader2: (props: any) => <svg data-testid="loader-icon" {...props} />,
}))

describe("VoicePreviewButton", () => {
  let mockPlay: ReturnType<typeof vi.fn>
  let mockPause: ReturnType<typeof vi.fn>
  let mockAudioInstance: any

  beforeEach(() => {
    mockPlay = vi.fn().mockResolvedValue(undefined)
    mockPause = vi.fn()
    mockAudioInstance = {
      play: mockPlay,
      pause: mockPause,
      src: "",
      onended: null as (() => void) | null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }

    vi.stubGlobal(
      "Audio",
      function MockAudio() {
        return mockAudioInstance
      }
    )
    vi.stubGlobal("URL", {
      ...globalThis.URL,
      createObjectURL: vi.fn(() => "blob:mock-url"),
      revokeObjectURL: vi.fn(),
    })

    mockSynthesizeSpeechDetailed.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders a button with 'Preview' text and aria-label", () => {
    render(
      <VoicePreviewButton model="tts-1" voice="alloy" provider="openai" />
    )

    const button = screen.getByRole("button", { name: "Preview voice" })
    expect(button).toBeTruthy()
    expect(button.textContent).toContain("Preview")
  })

  it("is disabled when voice is empty", () => {
    render(
      <VoicePreviewButton model="tts-1" voice="" provider="openai" />
    )

    const button = screen.getByRole("button", { name: "Preview voice" })
    expect(button).toBeDisabled()
  })

  it("calls detailed synthesis with gateway scope and plays audio on click", async () => {
    const fakeAudioData = new ArrayBuffer(8)
    mockSynthesizeSpeechDetailed.mockResolvedValue({
      buffer: fakeAudioData,
      fallbackUsed: false
    })

    render(
      <VoicePreviewButton
        model="SpeechModel"
        voice="narrator"
        provider="tldw"
        backend="gateway:primary"
        allowFallback={false}
      />
    )

    const button = screen.getByRole("button", { name: "Preview voice" })

    await act(async () => {
      fireEvent.click(button)
    })

    await waitFor(() => {
      expect(mockSynthesizeSpeechDetailed).toHaveBeenCalledWith(
        "Hello, this is a preview of the selected voice.",
        {
          model: "SpeechModel",
          voice: "narrator",
          responseFormat: "mp3",
          backend: "gateway:primary",
          allowFallback: false
        }
      )
    })

    expect(URL.createObjectURL).toHaveBeenCalled()
    expect(mockPlay).toHaveBeenCalled()
  })
})
