import { describe, expect, it } from "vitest"

import { classifyAudioError } from "../audio-error-classification"

describe("audio error classification", () => {
  it.each([[503, "missing_model"], [401, "missing_credentials"], [500, "unknown"]])(
    "B05 unavailable structured model status respects HTTP %s",
    (status, category) => {
      const error = Object.assign(new Error(`Upload failed: ${status}`), {
        status,
        details: { detail: { status: "model_unavailable" } }
      })
      const result = classifyAudioError(error)
      expect(result.category).toBe(category)
      if (status === 503) expect(result.recovery).toContain("Audio Setup Guide")
    }
  )

  it("B05 invalid audio size offers smaller-input recovery for HTTP 413", () => {
    const error = Object.assign(new Error("Upload failed: 413"), {
      status: 413,
      details: { detail: "File size exceeds maximum of 1MB" }
    })
    const result = classifyAudioError(error)
    expect(result.category).toBe("input_too_large")
    expect(result.title).toBe("Audio input is too large")
    expect(result.recovery).toBe("Use a smaller audio file or shorter text, then try again.")
    expect(result.settingsHref).toBeUndefined()
  })

  it("B05 invalid audio uses file recovery instead of provider guidance", () => {
    const error=Object.assign(new Error("Upload failed: 400"), {status:400,details:{detail:{status:"invalid_audio"}}})
    const result=classifyAudioError(error)
    expect(result.category).toBe("invalid_audio")
    expect(result.title).toBe("Audio file cannot be read")
    expect(result.recovery).toBe("Choose a valid audio recording, then transcribe again.")
    expect(result.settingsHref).toBeUndefined()
  })

  it("B05 invalid audio status never masks authentication failure", () => {
    const error=Object.assign(new Error("Unauthorized"), {status:401,details:{detail:{status:"invalid_audio"}}})
    expect(classifyAudioError(error).category).toBe("missing_credentials")
  })

  it("maps missing credentials without exposing raw keys", () => {
    const result = classifyAudioError(
      new Error("Request failed for sk_secret_inline with status 401")
    )

    expect(result.category).toBe("missing_credentials")
    expect(result.title).toBe("Credentials need attention")
    expect(result.recovery).toContain("Settings -> Speech")
    expect(result.debugMessage).toContain("[redacted]")
    expect(result.debugMessage).not.toContain("sk_secret_inline")
  })

  it("maps missing models to setup guidance", () => {
    const result = classifyAudioError(new Error("Model whisper-large not found"))

    expect(result.category).toBe("missing_model")
    expect(result.title).toBe("Model is not available")
    expect(result.recovery).toContain("Audio Setup Guide")
  })

  it("maps microphone permission errors to browser recovery", () => {
    const result = classifyAudioError(
      new DOMException("Permission denied", "NotAllowedError")
    )

    expect(result.category).toBe("microphone_blocked")
    expect(result.title).toBe("Microphone access is blocked")
    expect(result.recovery).toContain("browser permission")
  })

  it("preserves audio capture busy owner guidance", () => {
    const result = classifyAudioError(
      new Error("Audio capture is already active for live_voice")
    )

    expect(result.category).toBe("capture_busy")
    expect(result.title).toBe("Audio capture is already active")
    expect(result.recovery).toBe(
      "Stop the active capture session in live_voice and try again."
    )
  })

  it("maps network and timeout errors separately", () => {
    expect(classifyAudioError(new Error("Failed to fetch")).category).toBe("network")
    expect(classifyAudioError(new Error("request timed out")).category).toBe("timeout")
  })

  it("maps engine, unsupported capability, and unknown failures", () => {
    expect(classifyAudioError(new Error("ffmpeg is not installed")).category).toBe(
      "engine_unavailable"
    )
    expect(classifyAudioError(new Error("opus output is not supported")).category).toBe(
      "unsupported_capability"
    )
    expect(classifyAudioError(new Error("unexpected audio failure")).category).toBe(
      "unknown"
    )
  })
})
