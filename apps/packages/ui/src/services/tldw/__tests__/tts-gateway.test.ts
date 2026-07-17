import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { modelsAudioMethods } from "../domains/models-audio"

type TestConfig = {
  serverUrl: string
  authMode: "single-user" | "multi-user"
  apiKey?: string
  accessToken?: string
  orgId?: number
}

const audioBytes = (values = [1, 2, 3]): ArrayBuffer =>
  Uint8Array.from(values).buffer

const makeClient = (options?: {
  config?: TestConfig
  capability?: unknown
  capabilityError?: Error
  speechResponse?: unknown
}) => {
  let config: TestConfig = options?.config ?? {
    serverUrl: "https://tts.example.test",
    authMode: "single-user",
    apiKey: "test-api-key"
  }
  const request = vi.fn(async (init: { path: string }) => {
    if (init.path === "/api/v1/audio/providers") {
      if (options?.capabilityError) throw options.capabilityError
      return options?.capability ?? { supports_explicit_backend: true }
    }
    return options?.speechResponse ?? {
      ok: true,
      status: 200,
      data: audioBytes(),
      headers: {}
    }
  })
  const client = {
    ensureConfigForRequest: vi.fn(async () => config),
    request
  }
  return {
    client,
    request,
    setConfig(next: TestConfig) {
      config = next
    }
  }
}

const detailed = (client: unknown, text: string, options?: Record<string, unknown>) =>
  (modelsAudioMethods as any).synthesizeSpeechDetailed.call(client, text, options)

const compatible = (client: unknown, text: string, options?: Record<string, unknown>) =>
  (modelsAudioMethods as any).synthesizeSpeech.call(client, text, options)

const speechCalls = (request: ReturnType<typeof vi.fn>) =>
  request.mock.calls
    .map(([init]) => init)
    .filter((init) => init.path === "/api/v1/audio/speech")

const capabilityCalls = (request: ReturnType<typeof vi.fn>) =>
  request.mock.calls
    .map(([init]) => init)
    .filter((init) => init.path === "/api/v1/audio/providers")

describe("tldw gateway speech client", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-16T12:00:00Z"))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("maps the detailed speech body exactly and preserves model case", async () => {
    const controller = new AbortController()
    const { client, request } = makeClient({
      config: {
        serverUrl: "https://body.example.test",
        authMode: "single-user",
        apiKey: "body-secret-key"
      }
    })

    await detailed(client, "Speak this", {
      voice: "Narrator",
      model: "Vendor/Expressive-TTS",
      responseFormat: "wav",
      speed: 1.25,
      language: "en-US",
      normalizationOptions: { normalize_numbers: true },
      extraParams: { provider: { style: "warm" } },
      backend: "gateway:company-proxy",
      allowFallback: false,
      stream: false,
      signal: controller.signal
    })

    expect(capabilityCalls(request)).toHaveLength(1)
    expect(speechCalls(request)).toEqual([
      {
        path: "/api/v1/audio/speech",
        method: "POST",
        headers: { Accept: "audio/wav" },
        body: {
          input: "Speak this",
          text: "Speak this",
          voice: "Narrator",
          model: "Vendor/Expressive-TTS",
          response_format: "wav",
          speed: 1.25,
          lang_code: "en-US",
          normalization_options: { normalize_numbers: true },
          extra_params: { provider: { style: "warm" } },
          stream: false,
          backend: "gateway:company-proxy",
          allow_fallback: false
        },
        responseType: "arrayBuffer",
        abortSignal: controller.signal,
        returnResponse: true
      }
    ])
  })

  it("sends a concrete true fallback choice when backend support is advertised", async () => {
    const { client, request } = makeClient({
      config: {
        serverUrl: "https://fallback-default.example.test",
        authMode: "single-user",
        apiKey: "fallback-secret"
      }
    })

    await detailed(client, "Default fallback", {
      model: "Case/Sensitive-Model",
      backend: "openrouter"
    })

    expect(speechCalls(request)[0].body).toEqual({
      input: "Default fallback",
      text: "Default fallback",
      model: "Case/Sensitive-Model",
      backend: "openrouter",
      allow_fallback: true
    })
  })

  it("reads backend response metadata case-insensitively from header records", async () => {
    const { client } = makeClient({
      config: {
        serverUrl: "https://record-headers.example.test",
        authMode: "single-user",
        apiKey: "record-secret"
      },
      speechResponse: {
        ok: true,
        status: 200,
        data: audioBytes([7, 8]),
        headers: {
          "x-tldw-tts-BACKEND": "gateway:company-proxy",
          "X-tLdW-TtS-FaLlBaCk-UsEd": "TrUe"
        }
      }
    })

    const result = await detailed(client, "Metadata")

    expect(Array.from(new Uint8Array(result.buffer))).toEqual([7, 8])
    expect(result.actualBackend).toBe("gateway:company-proxy")
    expect(result.fallbackUsed).toBe(true)
  })

  it("reads Headers metadata and defaults missing or invalid values safely", async () => {
    const withHeaders = makeClient({
      config: {
        serverUrl: "https://headers-object.example.test",
        authMode: "single-user",
        apiKey: "headers-secret"
      },
      speechResponse: {
        ok: true,
        status: 200,
        data: audioBytes(),
        headers: new Headers({
          "X-TLDW-TTS-Backend": "openrouter",
          "X-TLDW-TTS-Fallback-Used": "false"
        })
      }
    })
    const missingHeaders = makeClient({
      config: {
        serverUrl: "https://missing-headers.example.test",
        authMode: "single-user",
        apiKey: "missing-secret"
      },
      speechResponse: {
        ok: true,
        status: 200,
        data: audioBytes(),
        headers: { "x-tldw-tts-fallback-used": "sometimes" }
      }
    })

    await expect(detailed(withHeaders.client, "Headers")).resolves.toMatchObject({
      actualBackend: "openrouter",
      fallbackUsed: false
    })
    await expect(detailed(missingHeaders.client, "Missing")).resolves.toMatchObject({
      actualBackend: undefined,
      fallbackUsed: false
    })
  })

  it.each([
    ["ArrayBuffer", () => audioBytes([1, 2])],
    ["view", () => Uint8Array.from([0, 3, 4, 0]).subarray(1, 3)],
    ["Blob", () => new Blob([Uint8Array.from([5, 6])])]
  ])("normalizes %s detailed response data", async (_name, makeData) => {
    const { client } = makeClient({
      config: {
        serverUrl: `https://buffer-${_name}.example.test`,
        authMode: "single-user",
        apiKey: "buffer-secret"
      },
      speechResponse: {
        ok: true,
        status: 200,
        data: makeData(),
        headers: {}
      }
    })

    const result = await detailed(client, "Buffer")

    expect(result.buffer).toBeInstanceOf(ArrayBuffer)
    expect(result.buffer.byteLength).toBe(2)
  })

  it.runIf(typeof SharedArrayBuffer !== "undefined")(
    "copies SharedArrayBuffer response data into an ArrayBuffer",
    async () => {
      const shared = new SharedArrayBuffer(2)
      new Uint8Array(shared).set([9, 10])
      const { client } = makeClient({
        config: {
          serverUrl: "https://shared-buffer.example.test",
          authMode: "single-user",
          apiKey: "shared-secret"
        },
        speechResponse: { ok: true, status: 200, data: shared, headers: {} }
      })

      const result = await detailed(client, "Shared")

      expect(result.buffer).toBeInstanceOf(ArrayBuffer)
      expect(Array.from(new Uint8Array(result.buffer))).toEqual([9, 10])
    }
  )

  it("keeps the compatibility method returning only a legacy raw ArrayBuffer", async () => {
    const raw = audioBytes([11, 12])
    const { client, request } = makeClient({
      config: {
        serverUrl: "https://legacy-buffer.example.test",
        authMode: "single-user",
        apiKey: "legacy-secret"
      },
      speechResponse: raw
    })

    const result = await compatible(client, "Legacy", { model: "Model/Case" })

    expect(result).toBe(raw)
    expect(capabilityCalls(request)).toHaveLength(0)
  })

  it("does not negotiate or add extension fields without a backend option", async () => {
    const { client, request } = makeClient({
      config: {
        serverUrl: "https://no-backend.example.test",
        authMode: "single-user",
        apiKey: "no-backend-secret"
      }
    })

    await detailed(client, "Legacy body", {
      model: "Keep/Exact-Case",
      allowFallback: false
    })

    expect(capabilityCalls(request)).toHaveLength(0)
    expect(speechCalls(request)[0].body).toEqual({
      input: "Legacy body",
      text: "Legacy body",
      model: "Keep/Exact-Case"
    })
  })

  it.each([
    ["missing", {}],
    ["false", { supports_explicit_backend: false }],
    ["malformed", { supports_explicit_backend: "true" }],
    ["failed", new Error("old server")]
  ])("uses the exact legacy body when capability negotiation is %s", async (name, capability) => {
    const failed = capability instanceof Error
    const { client, request } = makeClient({
      config: {
        serverUrl: `https://old-${name}.example.test`,
        authMode: "single-user",
        apiKey: `old-${name}-secret`
      },
      capability: failed ? undefined : capability,
      capabilityError: failed ? capability : undefined
    })

    await detailed(client, "Old server", {
      model: "Vendor/Exact-Case",
      backend: "openrouter",
      allowFallback: false
    })

    expect(capabilityCalls(request)).toHaveLength(1)
    expect(speechCalls(request)[0].body).toEqual({
      input: "Old server",
      text: "Old server",
      model: "Vendor/Exact-Case"
    })
  })

  it.each([
    ["positive", { supports_explicit_backend: true }],
    ["negative", { supports_explicit_backend: false }]
  ])("caches %s capability results for 30 seconds", async (name, capability) => {
    const { client, request } = makeClient({
      config: {
        serverUrl: `https://cache-${name}.example.test`,
        authMode: "single-user",
        apiKey: `cache-${name}-secret`
      },
      capability
    })
    const options = { model: "Model", backend: "openrouter" }

    await detailed(client, "First", options)
    await detailed(client, "Second", options)
    vi.advanceTimersByTime(29_999)
    await detailed(client, "Third", options)

    expect(capabilityCalls(request)).toHaveLength(1)

    vi.advanceTimersByTime(2)
    await detailed(client, "Fourth", options)

    expect(capabilityCalls(request)).toHaveLength(2)
  })

  it("partitions capability support by sanitized server, auth mode, and organization", async () => {
    const { client, request, setConfig } = makeClient({
      config: {
        serverUrl: "https://scope-a.example.test",
        authMode: "single-user",
        apiKey: "scope-key"
      }
    })
    const options = { backend: "openrouter" }

    await detailed(client, "Server A", options)
    setConfig({
      serverUrl: "https://scope-b.example.test",
      authMode: "single-user",
      apiKey: "scope-key"
    })
    await detailed(client, "Server B", options)
    setConfig({
      serverUrl: "https://scope-b.example.test",
      authMode: "multi-user",
      accessToken: "scope-token",
      orgId: 1
    })
    await detailed(client, "Auth mode", options)
    setConfig({
      serverUrl: "https://scope-b.example.test",
      authMode: "multi-user",
      accessToken: "scope-token",
      orgId: 2
    })
    await detailed(client, "Organization", options)
    setConfig({
      serverUrl: "https://scope-b.example.test",
      authMode: "multi-user",
      accessToken: "rotated-token",
      orgId: 1
    })
    await detailed(client, "Cached scope", options)

    expect(capabilityCalls(request)).toHaveLength(4)
  })

  it("does not include raw URL credentials, API keys, or access tokens in cache identity or logs", async () => {
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const firstSecret = "first-raw-secret"
    const secondSecret = "second-raw-secret"
    const { client, request, setConfig } = makeClient({
      config: {
        serverUrl: `https://user:${firstSecret}@secure.example.test/api?token=${firstSecret}`,
        authMode: "multi-user",
        accessToken: firstSecret,
        orgId: 7
      }
    })

    await detailed(client, "First", { backend: "openrouter" })
    setConfig({
      serverUrl: `https://other:${secondSecret}@secure.example.test/api?token=${secondSecret}`,
      authMode: "multi-user",
      accessToken: secondSecret,
      orgId: 7
    })
    await detailed(client, "Second", { backend: "openrouter" })

    expect(capabilityCalls(request)).toHaveLength(1)
    const visibleState = JSON.stringify({
      requestCalls: request.mock.calls,
      warnings: consoleWarn.mock.calls,
      errors: consoleError.mock.calls
    })
    expect(visibleState).not.toContain(firstSecret)
    expect(visibleState).not.toContain(secondSecret)
  })
})
