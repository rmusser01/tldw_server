import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mockStartRecording = vi.fn()
const mockStopRecording = vi.fn()
const mockClearRecording = vi.fn()
const mockLoadBlob = vi.fn()
let mockRecordingBlob: Blob | null = null
let mockRecordingDuration = 0
const notificationErrorMock = vi.fn()

vi.mock("@/hooks/useAudioRecorder", () => ({
  useAudioRecorder: () => ({
    status: "idle",
    blob: mockRecordingBlob,
    durationMs: mockRecordingDuration,
    startRecording: mockStartRecording,
    stopRecording: mockStopRecording,
    clearRecording: mockClearRecording,
    loadBlob: mockLoadBlob
  })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    error: notificationErrorMock,
    success: vi.fn()
  })
}))

import { RecordingStrip } from "../RecordingStrip"

describe("RecordingStrip", () => {
  const mockOnBlobReady = vi.fn()
  const mockOnSettingsToggle = vi.fn()

  beforeEach(() => {
    vi.resetAllMocks()
    mockRecordingBlob = null
    mockRecordingDuration = 0
  })

  afterEach(() => vi.unstubAllGlobals())

  it("B05 history restore uses the existing recorder once and does not reload after Clear", async () => {
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = vi.fn(() => "blob:b05-history")
      static revokeObjectURL = vi.fn()
    })
    const restoredRecording = { blob: new Blob(["saved audio"]), durationMs: 12800, isCurrent: () => true }
    mockLoadBlob.mockImplementation((blob: Blob, duration: number) => {
      mockRecordingBlob = blob
      mockRecordingDuration = duration
      return true
    })
    const view = render(<RecordingStrip onBlobReady={mockOnBlobReady} restoredRecording={restoredRecording} />)
    await waitFor(() => expect(mockLoadBlob).toHaveBeenCalledWith(restoredRecording.blob, 12800))
    view.rerender(<RecordingStrip onBlobReady={mockOnBlobReady} restoredRecording={restoredRecording} />)
    expect(view.container.querySelector("audio[controls]")).not.toBeNull()
    expect(screen.getByText("00:12")).toBeInTheDocument()
    mockClearRecording.mockImplementationOnce(() => { mockRecordingBlob = null; mockRecordingDuration = 0 })
    fireEvent.click(screen.getByRole("button", { name: "Clear recording" }))
    view.rerender(<RecordingStrip onBlobReady={mockOnBlobReady} restoredRecording={restoredRecording} />)
    expect(view.container.querySelector("audio")).toBeNull()
    expect(mockLoadBlob).toHaveBeenCalledOnce()
  })

  it("B05 history restore refuses a stale source lease", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} restoredRecording={{ blob: new Blob(["old"]), durationMs: 5000, isCurrent: () => false }} />)
    expect(mockLoadBlob).not.toHaveBeenCalled()
  })

  it("B05 history restore consumes a busy refusal without replaying after Stop", () => {
    mockLoadBlob.mockReturnValueOnce(false)
    const restoredRecording = { blob: new Blob(["history"]), durationMs: 5000, isCurrent: () => true }
    const view = render(<RecordingStrip onBlobReady={mockOnBlobReady} restoredRecording={restoredRecording} />)
    expect(notificationErrorMock).toHaveBeenCalledWith(expect.objectContaining({ description: "Stop recording before loading a saved recording." }))
    view.rerender(<RecordingStrip onBlobReady={mockOnBlobReady} restoredRecording={restoredRecording} />)
    expect(mockLoadBlob).toHaveBeenCalledOnce()
  })

  it.each(["Record", "Clear", "Upload"])("B05 history restore retires pending reads before %s intent changes the recorder", async (action) => {
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = vi.fn(() => "blob:b05-intent")
      static revokeObjectURL = vi.fn()
    })
    const order: string[] = []
    const onSourceChange = () => { order.push("intent") }
    mockStartRecording.mockImplementationOnce(() => { order.push("Record") })
    mockClearRecording.mockImplementationOnce(() => { order.push("Clear") })
    mockLoadBlob.mockImplementationOnce(() => { order.push("Upload"); return true })
    if (action === "Clear") mockRecordingBlob = new Blob(["current"])
    const view = render(<RecordingStrip onBlobReady={mockOnBlobReady} onSourceChange={onSourceChange} />)
    if (action === "Upload") {
      fireEvent.change(view.container.querySelector('input[type="file"]')!, { target: { files: [new File(["new"], "new.wav", { type: "audio/wav" })] } })
      await waitFor(() => expect(mockLoadBlob).toHaveBeenCalled())
    } else {
      fireEvent.click(screen.getByRole("button", { name: action === "Record" ? "Start recording (Space)" : "Clear recording", exact: true }))
    }
    expect(order).toEqual(["intent", action])
  })

  it("B05 clearing notifies the parent that the source is empty", async () => {
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = vi.fn(() => "blob:b05-recording")
      static revokeObjectURL = vi.fn()
    })
    mockRecordingBlob = new Blob(["recorded audio"], { type: "audio/wav" })
    mockRecordingDuration = 1200
    const { rerender } = render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    await waitFor(() => {
      expect(mockOnBlobReady).toHaveBeenLastCalledWith(mockRecordingBlob, 1200)
    })

    mockClearRecording.mockImplementationOnce(() => {
      mockRecordingBlob = null
      mockRecordingDuration = 0
    })
    fireEvent.click(screen.getByRole("button", { name: "Clear recording" }))
    rerender(<RecordingStrip onBlobReady={mockOnBlobReady} />)

    await waitFor(() => {
      expect(mockOnBlobReady).toHaveBeenLastCalledWith(null, 0)
    })
  })

  it.each([10.538667, Infinity, NaN, 0, -1])("B05 uploaded audio metadata retains truthful duration %s", (seconds) => {
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = vi.fn(() => "blob:b05-duration")
      static revokeObjectURL = vi.fn()
    })
    mockRecordingBlob = new Blob(["uploaded audio"], { type: "audio/mp4" })
    const { container } = render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    const audio = container.querySelector("audio")!
    Object.defineProperty(audio, "duration", { value: seconds, configurable: true })
    fireEvent.loadedMetadata(audio)
    if (Number.isFinite(seconds) && seconds > 0) {
      expect(mockLoadBlob).toHaveBeenCalledWith(mockRecordingBlob, seconds * 1000)
    } else {
      expect(mockLoadBlob).not.toHaveBeenCalled()
    }
  })

  it("B05 uploaded audio metadata preserves an existing measured duration", () => {
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = vi.fn(() => "blob:b05-measured")
      static revokeObjectURL = vi.fn()
    })
    mockRecordingBlob = new Blob(["recorded audio"], { type: "audio/webm" })
    mockRecordingDuration = 1200
    const { container } = render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    const audio = container.querySelector("audio")!
    Object.defineProperty(audio, "duration", { value: 1.5, configurable: true })
    fireEvent.loadedMetadata(audio)
    expect(mockLoadBlob).not.toHaveBeenCalled()
  })

  it("renders record button in idle state", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    const btn = screen.getByRole("button", { name: /start recording/i })
    expect(btn).toBeInTheDocument()
  })

  it("calls startRecording when record clicked", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    const btn = screen.getByRole("button", { name: /start recording/i })
    fireEvent.click(btn)
    expect(mockStartRecording).toHaveBeenCalledOnce()
  })

  it("shows upload button", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    const uploadBtn = screen.getByRole("button", { name: "Upload audio file" })
    expect(uploadBtn).toBeInTheDocument()
  })

  it("shows the current audio source state before recording starts", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)

    const status = screen
      .getByText("Audio source: no audio selected")
      .closest('[role="status"]')

    expect(status).toBeInTheDocument()
    expect(status).not.toHaveAttribute("aria-label")
  })

  it("disables recording and upload when route readiness blocks transcription", () => {
    render(
      <RecordingStrip
        onBlobReady={mockOnBlobReady}
        disabled
        disabledReason="No transcription models are available."
      />
    )

    expect(screen.getByRole("button", { name: /start recording/i })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Upload audio file" })).toBeDisabled()
    expect(screen.getByText("No transcription models are available."))
      .toHaveAttribute("role", "status")
  })

  it("uses non-danger styling for temporary disabled recording states", () => {
    render(
      <RecordingStrip
        onBlobReady={mockOnBlobReady}
        disabled
        disabledReason="Loading transcription model catalog."
        disabledStatusType="secondary"
      />
    )

    const status = screen.getByText("Loading transcription model catalog.")

    expect(status).toHaveClass("ant-typography-secondary")
    expect(status).not.toHaveClass("ant-typography-danger")
  })

  it("only cancels the Space shortcut event when it handles recording", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)

    const toggleEvent = new CustomEvent("stt-toggle-record", {
      cancelable: true
    })
    window.dispatchEvent(toggleEvent)

    expect(toggleEvent.defaultPrevented).toBe(true)
    expect(mockStartRecording).toHaveBeenCalledOnce()
  })

  it("does not cancel the Space shortcut event while recording is disabled", () => {
    render(
      <RecordingStrip
        onBlobReady={mockOnBlobReady}
        disabled
        disabledReason="No transcription models are available."
      />
    )

    const toggleEvent = new CustomEvent("stt-toggle-record", {
      cancelable: true
    })
    window.dispatchEvent(toggleEvent)

    expect(toggleEvent.defaultPrevented).toBe(false)
    expect(mockStartRecording).not.toHaveBeenCalled()
  })

  it("shows settings toggle button when prop provided", () => {
    render(
      <RecordingStrip
        onBlobReady={mockOnBlobReady}
        onSettingsToggle={mockOnSettingsToggle}
      />
    )
    const settingsBtn = screen.getByRole("button", { name: "Toggle settings" })
    expect(settingsBtn).toBeInTheDocument()
    fireEvent.click(settingsBtn)
    expect(mockOnSettingsToggle).toHaveBeenCalledOnce()
  })

  it("has correct aria-label on record button", () => {
    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    const btn = screen.getByRole("button", { name: /start recording/i })
    expect(btn.getAttribute("aria-label")).toContain("Start recording")
  })

  it("shows classified microphone permission notification when recording cannot start", async () => {
    mockStartRecording.mockRejectedValueOnce(
      new DOMException("Permission denied", "NotAllowedError")
    )

    render(<RecordingStrip onBlobReady={mockOnBlobReady} />)
    fireEvent.click(screen.getByRole("button", { name: /start recording/i }))

    await waitFor(() => {
      expect(notificationErrorMock).toHaveBeenCalledWith(
        expect.objectContaining({
          message: "Microphone access is blocked",
          description: expect.stringContaining("browser permission")
        })
      )
    })
  })
})
