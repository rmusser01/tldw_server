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
  const requestWithCurrentConfig = vi.fn(
    async (initOrFactory: any) => {
      const init =
        typeof initOrFactory === "function"
          ? initOrFactory(config)
          : initOrFactory
      return request(init)
    }
  )
  const client = {
    ensureConfigForRequest: vi.fn(async () => config),
    request,
    requestWithCurrentConfig
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
    if (_name === "Blob") vi.useRealTimers()
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

  it("preserves AbortError transport metadata without exposing response secrets", async () => {
    const rawSecret = "abort-token-secret"
    const { client } = makeClient({
      config: {
        serverUrl: "https://abort-error.example.test",
        authMode: "single-user",
        apiKey: "abort-key"
      },
      speechResponse: {
        ok: false,
        status: 0,
        error: `Request aborted token=${rawSecret}`,
        name: "AbortError",
        code: "REQUEST_ABORTED",
        details: {
          detail: "The speech request was cancelled",
          access_token: rawSecret
        }
      }
    })

    const error = await detailed(client, "Cancelled").catch((reason) => reason)

    expect(error).toBeInstanceOf(Error)
    expect(error).toMatchObject({
      name: "AbortError",
      code: "REQUEST_ABORTED",
      status: 0,
      details: {
        detail: "The speech request was cancelled",
        access_token: "[REDACTED]"
      }
    })
    expect(JSON.stringify(error)).not.toContain(rawSecret)
    expect(error.message).not.toContain(rawSecret)
  })

  it("preserves sanitized HTTP transport status, details, code, and name", async () => {
    const rawSecret = "http-api-secret"
    const { client } = makeClient({
      config: {
        serverUrl: "https://http-error.example.test",
        authMode: "multi-user",
        accessToken: "client-token",
        orgId: 22
      },
      speechResponse: {
        ok: false,
        status: 429,
        error: `Rate limited api_key=${rawSecret}`,
        name: "TldwRequestError",
        code: "UPSTREAM_RATE_LIMITED",
        details: {
          detail: "Try again later",
          api_key: rawSecret,
          retry_after: 3,
          stack: rawSecret,
          trace: rawSecret,
          sql: rawSecret,
          query: rawSecret,
          path: rawSecret,
          headers: { Authorization: rawSecret },
          internalid: rawSecret,
          session: rawSecret,
          private: rawSecret,
          access_key: rawSecret,
          refresh_token: rawSecret
        }
      }
    })

    const error = await detailed(client, "Rate limit").catch((reason) => reason)

    expect(error).toMatchObject({
      name: "TldwRequestError",
      code: "UPSTREAM_RATE_LIMITED",
      status: 429,
      details: {
        detail: "Try again later",
        api_key: "[REDACTED]",
        retry_after: 3,
        stack: "[REDACTED]",
        trace: "[REDACTED]",
        sql: "[REDACTED]",
        query: "[REDACTED]",
        path: "[REDACTED]",
        headers: "[REDACTED]",
        internalid: "[REDACTED]",
        session: "[REDACTED]",
        private: "[REDACTED]",
        access_key: "[REDACTED]",
        refresh_token: "[REDACTED]"
      }
    })
    expect(error.message).toContain("Rate limited")
    expect(error.message).not.toContain(rawSecret)
    expect(JSON.stringify(error)).not.toContain(rawSecret)
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

  it("rejects explicit-backend support from a failed capability envelope", async () => {
    const { client, request } = makeClient({
      config: {
        serverUrl: "https://failed-envelope.example.test",
        authMode: "single-user",
        apiKey: "failed-envelope-secret"
      },
      capability: {
        ok: false,
        status: 503,
        data: { supports_explicit_backend: true }
      }
    })

    await detailed(client, "Failed envelope", {
      backend: "openrouter",
      allowFallback: false
    })

    expect(speechCalls(request)[0].body).toEqual({
      input: "Failed envelope",
      text: "Failed envelope"
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

  it("uses monotonic expiry when the wall clock moves backward", async () => {
    const now = vi.spyOn(performance, "now").mockReturnValue(1_000)
    const { client, request } = makeClient({
      config: {
        serverUrl: "https://monotonic-cache.example.test",
        authMode: "single-user",
        apiKey: "monotonic-secret"
      }
    })

    await detailed(client, "First", { backend: "openrouter" })
    vi.setSystemTime(new Date("2020-01-01T00:00:00Z"))
    now.mockReturnValue(31_001)
    await detailed(client, "Expired", { backend: "openrouter" })

    expect(capabilityCalls(request)).toHaveLength(2)
  })

  it("bounds cached capability scopes", async () => {
    const { client, request, setConfig } = makeClient()
    const firstUrl = "https://bounded-0.example.test"

    for (let index = 0; index < 65; index += 1) {
      setConfig({
        serverUrl: `https://bounded-${index}.example.test`,
        authMode: "single-user",
        apiKey: `bounded-secret-${index}`
      })
      await detailed(client, `Scope ${index}`, { backend: "openrouter" })
    }
    setConfig({
      serverUrl: firstUrl,
      authMode: "single-user",
      apiKey: "rotated-bounded-secret"
    })
    await detailed(client, "Evicted scope", { backend: "openrouter" })

    expect(capabilityCalls(request)).toHaveLength(66)
  })

  it("coalesces same-scope capability requests through the scope-local in-flight map", async () => {
    let releaseCapability: ((value: unknown) => void) | undefined
    const capability = new Promise((resolve) => {
      releaseCapability = resolve
    })
    const speechRequests: Array<Record<string, any>> = []
    const request = vi.fn(async (init: Record<string, any>) => {
      if (init.path === "/api/v1/audio/speech") {
        speechRequests.push(init)
        return { ok: true, status: 200, data: audioBytes(), headers: {} }
      }
      throw new Error("generic GET transport must not be used")
    })
    const scopeConfig: TestConfig = {
      serverUrl: "https://same-scope.example.test",
      authMode: "multi-user",
      accessToken: "same-scope-secret",
      orgId: 3
    }
    const requestWithCurrentConfig = vi.fn(async (initOrFactory: any) => {
      const init =
        typeof initOrFactory === "function"
          ? initOrFactory(scopeConfig)
          : initOrFactory
      return init.path === "/api/v1/audio/providers" ? capability : request(init)
    })
    const client = {
      ensureConfigForRequest: vi.fn(async () => scopeConfig),
      request,
      requestWithCurrentConfig
    }

    const first = detailed(client, "First", { backend: "openrouter" })
    const second = detailed(client, "Second", { backend: "openrouter" })
    await vi.waitFor(() => expect(requestWithCurrentConfig).toHaveBeenCalledTimes(1))
    releaseCapability?.({
      ok: true,
      status: 200,
      data: { supports_explicit_backend: true }
    })
    await Promise.all([first, second])

    expect(requestWithCurrentConfig).toHaveBeenCalledWith({
      path: "/api/v1/audio/providers",
      method: "GET",
      returnResponse: true
    })
    expect(speechRequests.map((entry) => entry.body.backend)).toEqual([
      "openrouter",
      "openrouter"
    ])
  })

  it("never shares concurrent capability results across server or organization scopes", async () => {
    let releaseA: ((value: unknown) => void) | undefined
    let releaseB: ((value: unknown) => void) | undefined
    const capabilityA = new Promise((resolve) => {
      releaseA = resolve
    })
    const capabilityB = new Promise((resolve) => {
      releaseB = resolve
    })
    const globallyCoalesced = Promise.resolve({
      supports_explicit_backend: true
    })
    const makeScopedClient = (
      serverUrl: string,
      orgId: number,
      capability: Promise<unknown>
    ) => {
      const speechRequests: Array<Record<string, any>> = []
      const scopeConfig: TestConfig = {
        serverUrl,
        authMode: "multi-user",
        accessToken: `scope-${orgId}-secret`,
        orgId
      }
      return {
        speechRequests,
        client: {
          ensureConfigForRequest: vi.fn(async () => scopeConfig),
          request: vi.fn(async (init: Record<string, any>) => {
            if (init.path === "/api/v1/audio/providers") return globallyCoalesced
            speechRequests.push(init)
            return { ok: true, status: 200, data: audioBytes(), headers: {} }
          }),
          requestWithCurrentConfig: vi.fn(
            async (initOrFactory: any) => {
              const init =
                typeof initOrFactory === "function"
                  ? initOrFactory(scopeConfig)
                  : initOrFactory
              if (init.path === "/api/v1/audio/providers") return capability
              speechRequests.push(init)
              return { ok: true, status: 200, data: audioBytes(), headers: {} }
            }
          )
        }
      }
    }
    const scopeA = makeScopedClient(
      "https://concurrent-a.example.test",
      10,
      capabilityA
    )
    const scopeB = makeScopedClient(
      "https://concurrent-b.example.test",
      11,
      capabilityB
    )

    const first = detailed(scopeA.client, "A", { backend: "openrouter" })
    const second = detailed(scopeB.client, "B", { backend: "openrouter" })
    await vi.waitFor(() => {
      expect(scopeA.client.requestWithCurrentConfig).toHaveBeenCalledTimes(1)
      expect(scopeB.client.requestWithCurrentConfig).toHaveBeenCalledTimes(1)
    })
    releaseA?.({
      ok: true,
      status: 200,
      data: { supports_explicit_backend: false }
    })
    releaseB?.({
      ok: true,
      status: 200,
      data: { supports_explicit_backend: true }
    })
    await Promise.all([first, second])

    expect(scopeA.speechRequests[0].body).not.toHaveProperty("backend")
    expect(scopeB.speechRequests[0].body.backend).toBe("openrouter")
    expect(scopeA.client.request).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/audio/providers" })
    )
    expect(scopeB.client.request).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/audio/providers" })
    )
  })

  it("discards a capability result when the active config scope changes mid-request", async () => {
    let config: TestConfig = {
      serverUrl: "https://race-a.example.test",
      authMode: "multi-user",
      accessToken: "race-a-secret",
      orgId: 41
    }
    let releaseCapability: ((value: unknown) => void) | undefined
    const firstCapability = new Promise((resolve) => {
      releaseCapability = resolve
    })
    const speechRequests: Array<Record<string, any>> = []
    let capabilityCallsForRace = 0
    const request = vi.fn(async (init: Record<string, any>) => {
      speechRequests.push(init)
      return { ok: true, status: 200, data: audioBytes(), headers: {} }
    })
    const requestWithCurrentConfig = vi.fn(
      async (initOrFactory: any) => {
        const init =
          typeof initOrFactory === "function"
            ? initOrFactory(config)
            : initOrFactory
        if (init.path !== "/api/v1/audio/providers") return request(init)
        capabilityCallsForRace += 1
        if (capabilityCallsForRace === 1) return firstCapability
        return {
          ok: true,
          status: 200,
          data: { supports_explicit_backend: false }
        }
      }
    )
    const client = {
      ensureConfigForRequest: vi.fn(async () => config),
      request,
      requestWithCurrentConfig
    }

    const first = detailed(client, "Changing scope", { backend: "openrouter" })
    await vi.waitFor(() => expect(requestWithCurrentConfig).toHaveBeenCalledTimes(1))
    config = {
      serverUrl: "https://race-b.example.test",
      authMode: "multi-user",
      accessToken: "race-b-secret",
      orgId: 42
    }
    releaseCapability?.({
      ok: true,
      status: 200,
      data: { supports_explicit_backend: true }
    })
    await first
    await detailed(client, "Stable new scope", { backend: "openrouter" })

    expect(capabilityCallsForRace).toBe(2)
    expect(speechRequests[0].body).not.toHaveProperty("backend")
    expect(speechRequests[1].body).not.toHaveProperty("backend")
  })

  it("omits negotiated fields when dispatch scope changes after fresh discovery", async () => {
    const scopeA: TestConfig = {
      serverUrl: "https://dispatch-fresh-a.example.test",
      authMode: "multi-user",
      accessToken: "dispatch-fresh-a-secret",
      orgId: 51
    }
    const scopeB: TestConfig = {
      serverUrl: "https://dispatch-fresh-b.example.test",
      authMode: "multi-user",
      accessToken: "dispatch-fresh-b-secret",
      orgId: 52
    }
    const ensureConfigForRequest = vi
      .fn()
      .mockResolvedValueOnce(scopeA)
      .mockResolvedValueOnce(scopeA)
      .mockResolvedValueOnce(scopeA)
      .mockResolvedValue(scopeB)
    const speechRequests: Array<Record<string, any>> = []
    const sendSpeech = async (init: Record<string, any>) => {
      speechRequests.push(init)
      return { ok: true, status: 200, data: audioBytes(), headers: {} }
    }
    const request = vi.fn(sendSpeech)
    const requestWithCurrentConfig = vi.fn(
      async (initOrFactory: any) => {
        const init =
          typeof initOrFactory === "function"
            ? initOrFactory(scopeB)
            : initOrFactory
        return (
        init.path === "/api/v1/audio/providers"
          ? {
              ok: true,
              status: 200,
              data: { supports_explicit_backend: true }
            }
          : sendSpeech(init)
        )
      }
    )
    const client = {
      ensureConfigForRequest,
      request,
      requestWithCurrentConfig
    }

    await detailed(client, "Fresh dispatch race", { backend: "openrouter" })

    expect(speechRequests[0].body).not.toHaveProperty("backend")
    expect(requestWithCurrentConfig).toHaveBeenCalledWith(
      expect.any(Function),
      true
    )
  })

  it("omits cached negotiated fields when dispatch resolves a different scope", async () => {
    const scopeA: TestConfig = {
      serverUrl: "https://dispatch-cache-a.example.test",
      authMode: "multi-user",
      accessToken: "dispatch-cache-a-secret",
      orgId: 61
    }
    const scopeB: TestConfig = {
      serverUrl: "https://dispatch-cache-b.example.test",
      authMode: "multi-user",
      accessToken: "dispatch-cache-b-secret",
      orgId: 62
    }
    let config = scopeA
    const speechRequests: Array<Record<string, any>> = []
    const sendSpeech = async (init: Record<string, any>) => {
      speechRequests.push(init)
      return { ok: true, status: 200, data: audioBytes(), headers: {} }
    }
    const request = vi.fn(sendSpeech)
    const requestWithCurrentConfig = vi.fn(
      async (initOrFactory: any) => {
        const init =
          typeof initOrFactory === "function"
            ? initOrFactory(config)
            : initOrFactory
        return (
        init.path === "/api/v1/audio/providers"
          ? {
              ok: true,
              status: 200,
              data: { supports_explicit_backend: true }
            }
          : sendSpeech(init)
        )
      }
    )
    const client = {
      ensureConfigForRequest: vi.fn(async () => config),
      request,
      requestWithCurrentConfig
    }

    await detailed(client, "Prime scope A", { backend: "openrouter" })
    const second = detailed(client, "Cached dispatch race", {
      backend: "openrouter"
    })
    config = scopeB
    await second

    expect(
      requestWithCurrentConfig.mock.calls.filter(
        ([init]) => init.path === "/api/v1/audio/providers"
      )
    ).toHaveLength(1)
    expect(speechRequests[0].body.backend).toBe("openrouter")
    expect(speechRequests[1].body).not.toHaveProperty("backend")
    expect(requestWithCurrentConfig).toHaveBeenLastCalledWith(
      expect.any(Function),
      true
    )
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
