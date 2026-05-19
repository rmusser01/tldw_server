import { describe, expect, it } from "vitest"

import { classifyAudioError } from "../audio-error-classification"

describe("audio error classification", () => {
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
