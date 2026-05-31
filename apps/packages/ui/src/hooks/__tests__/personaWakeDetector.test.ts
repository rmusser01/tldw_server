import { afterEach, describe, expect, it, vi } from "vitest"

import {
  BrowserTranscriptWakeDetector,
  findCanonicalWakePhrase,
  normalizeWakePhraseText
} from "../personaWakeDetector"

type MockRecognitionInstance = {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: { results: Array<Array<{ transcript: string }>> }) => void) | null
  onerror: ((event: { error?: string; message?: string }) => void) | null
  onend: (() => void) | null
  start: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
}

const installMockRecognition = () => {
  const instances: MockRecognitionInstance[] = []

  class MockRecognition {
    continuous = false
    interimResults = false
    lang = ""
    onresult: MockRecognitionInstance["onresult"] = null
    onerror: MockRecognitionInstance["onerror"] = null
    onend: MockRecognitionInstance["onend"] = null
    start = vi.fn()
    stop = vi.fn()

    constructor() {
      instances.push(this)
    }
  }

  ;(window as any).SpeechRecognition = MockRecognition
  ;(window as any).webkitSpeechRecognition = undefined

  return {
    instances,
    get current() {
      const instance = instances.at(-1)
      if (!instance) {
        throw new Error("Expected SpeechRecognition to be constructed")
      }
      return instance
    }
  }
}

describe("personaWakeDetector", () => {
  afterEach(() => {
    ;(window as any).SpeechRecognition = undefined
    ;(window as any).webkitSpeechRecognition = undefined
  })

  it("normalizes phrase text for matching", () => {
    expect(normalizeWakePhraseText("  Hey,   Helper! ")).toBe("hey helper")
  })

  it("matches whole normalized phrase sequences", () => {
    expect(
      findCanonicalWakePhrase("um hey helper please wake up", ["Hey Helper"])
    ).toBe("Hey Helper")
    expect(findCanonicalWakePhrase("the helper is nearby", ["hey helper"])).toBeNull()
  })

  it("reports unavailable when SpeechRecognition is absent", async () => {
    ;(window as any).SpeechRecognition = undefined
    ;(window as any).webkitSpeechRecognition = undefined

    const detector = new BrowserTranscriptWakeDetector()
    await expect(detector.isAvailable()).resolves.toBe(false)
  })

  it("starts browser recognition and emits wake events for matched transcripts", async () => {
    const recognition = installMockRecognition()
    const onWake = vi.fn()
    const onStateChange = vi.fn()
    const detector = new BrowserTranscriptWakeDetector()

    await detector.start({
      phrases: ["Hey Helper"],
      locale: "fr-FR",
      onWake,
      onStateChange
    })

    expect(onStateChange).toHaveBeenNthCalledWith(1, "starting")
    expect(onStateChange).toHaveBeenNthCalledWith(2, "listening")
    expect(recognition.current.continuous).toBe(true)
    expect(recognition.current.interimResults).toBe(true)
    expect(recognition.current.lang).toBe("fr-FR")
    expect(recognition.current.start).toHaveBeenCalledTimes(1)

    recognition.current.onresult?.({
      results: [[{ transcript: "noise hey helper do the thing" }]]
    })

    expect(onStateChange).toHaveBeenLastCalledWith("detected")
    expect(onWake).toHaveBeenCalledWith(
      expect.objectContaining({
        canonicalPhrase: "Hey Helper",
        transcript: "noise hey helper do the thing",
        detectorKind: "browser_transcript"
      })
    )

    await detector.stop()
    expect(recognition.current.stop).toHaveBeenCalledTimes(1)
    expect(recognition.current.onresult).toBeNull()
    expect(recognition.current.onerror).toBeNull()
    expect(recognition.current.onend).toBeNull()
  })

  it("restarts browser recognition when it ends while still active", async () => {
    const recognition = installMockRecognition()
    const onStateChange = vi.fn()
    const detector = new BrowserTranscriptWakeDetector()

    await detector.start({
      phrases: ["Hey Helper"],
      onWake: vi.fn(),
      onStateChange
    })

    recognition.current.onend?.()

    expect(recognition.current.start).toHaveBeenCalledTimes(2)
    expect(onStateChange).toHaveBeenLastCalledWith("listening")

    await detector.stop()
  })

  it("does not restart browser recognition after fatal recognition errors", async () => {
    const recognition = installMockRecognition()
    const onError = vi.fn()
    const onStateChange = vi.fn()
    const detector = new BrowserTranscriptWakeDetector()

    await detector.start({
      phrases: ["Hey Helper"],
      onWake: vi.fn(),
      onError,
      onStateChange
    })

    recognition.current.onerror?.({
      error: "not-allowed",
      message: "permission denied"
    })

    expect(onStateChange).toHaveBeenLastCalledWith("error")
    expect(onError).toHaveBeenCalledWith({
      code: "not-allowed",
      message: "permission denied"
    })
    expect(recognition.current.onend).toBeNull()
    expect(recognition.current.start).toHaveBeenCalledTimes(1)
  })

  it("does not start recognition when no wake phrases are configured", async () => {
    const recognition = installMockRecognition()
    const onStateChange = vi.fn()
    const detector = new BrowserTranscriptWakeDetector()

    await detector.start({
      phrases: [" ", ""],
      onWake: vi.fn(),
      onStateChange
    })

    expect(onStateChange).toHaveBeenCalledWith("unavailable")
    expect(recognition.instances).toHaveLength(0)
  })
})
