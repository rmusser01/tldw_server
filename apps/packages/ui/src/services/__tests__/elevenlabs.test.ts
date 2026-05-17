import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { generateSpeech, getModels, getVoices } from "../elevenlabs"

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers)
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  return new Response(JSON.stringify(body), {
    ...init,
    headers
  })
}

describe("ElevenLabs fetch service", () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("fetches voices with the ElevenLabs API key and timeout signal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        voices: [{ voice_id: "voice-1", name: "Narrator" }]
      })
    )
    vi.stubGlobal("fetch", fetchMock)

    await expect(getVoices("eleven-key", { timeoutMs: 2500 })).resolves.toEqual([
      { voice_id: "voice-1", name: "Narrator" }
    ])

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("https://api.elevenlabs.io/v1/voices")
    expect(init.method).toBe("GET")
    expect(new Headers(init.headers).get("xi-api-key")).toBe("eleven-key")
    expect(init.signal).toBeInstanceOf(AbortSignal)
  })

  it("fetches models and parses the returned JSON list", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse([{ model_id: "model-1", name: "Fast" }])
    )
    vi.stubGlobal("fetch", fetchMock)

    await expect(getModels("eleven-key")).resolves.toEqual([
      { model_id: "model-1", name: "Fast" }
    ])

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("https://api.elevenlabs.io/v1/models")
    expect(init.method).toBe("GET")
    expect(new Headers(init.headers).get("xi-api-key")).toBe("eleven-key")
  })

  it("generates speech with a JSON payload and returns an ArrayBuffer", async () => {
    const bytes = new Uint8Array([7, 8, 9])
    const fetchMock = vi.fn().mockResolvedValue(new Response(bytes))
    vi.stubGlobal("fetch", fetchMock)

    const audio = await generateSpeech(
      "eleven-key",
      "hello",
      "voice-1",
      "model-1",
      1.2
    )

    expect(Array.from(new Uint8Array(audio))).toEqual([7, 8, 9])
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("https://api.elevenlabs.io/v1/text-to-speech/voice-1")
    expect(init.method).toBe("POST")
    const headers = new Headers(init.headers)
    expect(headers.get("xi-api-key")).toBe("eleven-key")
    expect(headers.get("Content-Type")).toBe("application/json")
    expect(JSON.parse(init.body as string)).toEqual({
      text: "hello",
      model_id: "model-1",
      voice_settings: { speed: 1.2 }
    })
  })

  it("aborts generated speech with the caller-provided signal without reporting a timeout", async () => {
    vi.useFakeTimers()
    const callerAbort = new AbortController()
    let fetchSignal: AbortSignal | undefined
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      fetchSignal = init?.signal ?? undefined
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"))
        })
      })
    })
    vi.stubGlobal("fetch", fetchMock)

    const request = generateSpeech(
      "eleven-key",
      "hello",
      "voice-1",
      "model-1",
      undefined,
      { signal: callerAbort.signal, timeoutMs: 10_000 }
    )
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    callerAbort.abort()

    expect(fetchSignal?.aborted).toBe(true)
    await expect(request).rejects.toThrow("Aborted")
  })

  it("keeps caller abort active while generated speech response bodies are read", async () => {
    const callerAbort = new AbortController()
    let resolveBody!: (value: ArrayBuffer) => void
    let bodyStarted!: () => void
    const bodyStartedPromise = new Promise<void>((resolve) => {
      bodyStarted = resolve
    })
    const fetchMock = vi.fn((_url: string, init?: RequestInit) =>
      Promise.resolve({
        ok: true,
        arrayBuffer: vi.fn(() => {
          bodyStarted()
          return new Promise<ArrayBuffer>((resolve, reject) => {
            resolveBody = resolve
            init?.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("Aborted", "AbortError")),
              { once: true }
            )
          })
        })
      } as Response)
    )
    vi.stubGlobal("fetch", fetchMock)

    const request = generateSpeech(
      "eleven-key",
      "hello",
      "voice-1",
      "model-1",
      undefined,
      { signal: callerAbort.signal, timeoutMs: 10_000 }
    )
    await bodyStartedPromise

    callerAbort.abort()
    resolveBody(new ArrayBuffer(1))

    await expect(request).rejects.toThrow("Aborted")
  })

  it("rejects failed ElevenLabs responses with a descriptive status error", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ detail: "bad key" }, { status: 401 }))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getVoices("bad-key")).rejects.toThrow(
      "ElevenLabs request failed with status 401"
    )
  })

  it("aborts ElevenLabs requests after the configured timeout", async () => {
    vi.useFakeTimers()
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"))
        })
      })
    })
    vi.stubGlobal("fetch", fetchMock)

    const request = getModels("eleven-key", { timeoutMs: 5 })
    const requestAssertion = expect(request).rejects.toThrow(
      "ElevenLabs request timed out"
    )
    await vi.advanceTimersByTimeAsync(5)

    await requestAssertion
  })

  it("keeps timeouts active while generated speech response bodies are read", async () => {
    vi.useFakeTimers()
    let resolveBody!: (value: ArrayBuffer) => void
    let bodyStarted!: () => void
    const bodyStartedPromise = new Promise<void>((resolve) => {
      bodyStarted = resolve
    })
    const fetchMock = vi.fn((_url: string, init?: RequestInit) =>
      Promise.resolve({
        ok: true,
        arrayBuffer: vi.fn(() => {
          bodyStarted()
          return new Promise<ArrayBuffer>((resolve, reject) => {
            resolveBody = resolve
            init?.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("Aborted", "AbortError")),
              { once: true }
            )
          })
        })
      } as Response)
    )
    vi.stubGlobal("fetch", fetchMock)

    const request = generateSpeech(
      "eleven-key",
      "hello",
      "voice-1",
      "model-1",
      undefined,
      { timeoutMs: 5 }
    )
    await bodyStartedPromise
    const requestAssertion = expect(request).rejects.toThrow(
      "ElevenLabs request timed out"
    )
    await vi.advanceTimersByTimeAsync(5)
    resolveBody(new ArrayBuffer(1))

    await requestAssertion
  })

  it("normalizes browser fetch timeout failures for retry UX", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))
    vi.stubGlobal("fetch", fetchMock)

    await expect(getVoices("eleven-key")).rejects.toThrow(
      "ElevenLabs request timed out"
    )
  })
})
