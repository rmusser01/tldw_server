import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  connect: vi.fn(),
  tldwRequest: vi.fn(),
  storageGet: vi.fn(async (_key?: string) => null),
  storageSet: vi.fn(async () => undefined)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      id: "test-extension",
      sendMessage: (...args: unknown[]) =>
        (mocks.sendMessage as (...args: unknown[]) => unknown)(...args),
      connect: (...args: unknown[]) =>
        (mocks.connect as (...args: unknown[]) => unknown)(...args)
    }
  }
}))

vi.mock("@/services/tldw/request-core", async () => {
  const actual = await vi.importActual<typeof import("@/services/tldw/request-core")>(
    "@/services/tldw/request-core"
  )
  return {
    ...actual,
    tldwRequest: (...args: unknown[]) =>
      (mocks.tldwRequest as (...args: unknown[]) => unknown)(...args)
  }
})

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (...args: unknown[]) =>
      (mocks.storageGet as (...args: unknown[]) => unknown)(...args),
    set: (...args: unknown[]) =>
      (mocks.storageSet as (...args: unknown[]) => unknown)(...args)
  })
}))

const importProxy = async () => import("@/services/background-proxy")

describe("background proxy fallback safety", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useRealTimers()
    mocks.sendMessage.mockReset()
    mocks.connect.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.storageGet.mockResolvedValue(null)
    mocks.storageSet.mockResolvedValue(undefined)
  })

  it("does not fall back to direct request when background returns non-2xx", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false, status: 500, error: "boom" })
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: { fallback: true } })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({ path: "/api/v1/health", method: "GET" })
    ).rejects.toMatchObject({ status: 500 })
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("keeps auth enabled for same-origin absolute URLs in background requests", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "https://api.example.com/api/v1/health",
        method: "GET"
      })
    ).resolves.toEqual({ ok: true })

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          noAuth: false
        })
      })
    )
  })

  it("skips auth for cross-origin absolute URLs in background requests", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "https://other.example.com/api/v1/health",
        method: "GET"
      })
    ).resolves.toEqual({ ok: true })

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          noAuth: true
        })
      })
    )
  })

  it("normalizes legacy media listing paths before forwarding request", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "/api/v1/media/?page=1&results_per_page=20&include_keywords=true",
        method: "GET"
      })
    ).resolves.toEqual({ ok: true })

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          path: "/api/v1/media?page=1&results_per_page=20&include_keywords=true"
        })
      })
    )
  })

  it("emits backend-unreachable event when API request fails with network status 0", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "NetworkError when attempting to fetch resource."
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/llm/models/metadata",
          method: "GET"
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).toHaveBeenCalledTimes(1)
    const detail = (eventSpy.mock.calls[0]?.[0] as CustomEvent | undefined)
      ?.detail as
      | {
          path?: string
          method?: string
          status?: number
          source?: string
        }
      | undefined
    expect(detail?.path).toBe("/api/v1/llm/models/metadata")
    expect(detail?.method).toBe("GET")
    expect(detail?.status).toBe(0)
    expect(detail?.source).toBe("direct")
  })

  it("classifies aborted direct fallback requests as AbortError", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "The operation was aborted."
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "/api/v1/chats/?limit=200&offset=0&ordering=-updated_at",
        method: "GET"
      })
    ).rejects.toMatchObject({
      name: "AbortError",
      status: 0,
      code: "REQUEST_ABORTED"
    })
  })

  it("falls back to direct request on GET extension timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: { via: "direct" } })

    const { bgRequest } = await importProxy()
    const pending = bgRequest<{ via: string }>({
      path: "/api/v1/health",
      method: "GET"
    })

    await vi.advanceTimersByTimeAsync(3001)

    await expect(pending).resolves.toEqual({ via: "direct" })
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("bypasses extension messaging for direct-preferred requests", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { via: "runtime" } })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { via: "direct" }
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "/api/v1/media/ingest/jobs/101",
        method: "GET",
        preferDirect: true
      })
    ).resolves.toEqual({ via: "direct" })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("does not fall back to direct request on POST extension timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: { via: "direct" } })

    const { bgRequest } = await importProxy()
    const pending = bgRequest({
      path: "/api/v1/notes/search/",
      method: "POST",
      body: { q: "hello" }
    })
    const assertion = expect(pending).rejects.toThrow("Extension messaging timeout")

    await vi.advanceTimersByTimeAsync(3001)

    await assertion
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("does not fall back to direct upload when background returns non-2xx", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: false,
      status: 400,
      error: "bad request",
      data: { detail: "invalid" }
    })

    const { bgUpload } = await importProxy()

    await expect(
      bgUpload({
        path: "/api/v1/media/add",
        method: "POST",
        fields: { title: "example" }
      })
    ).rejects.toMatchObject({ status: 400 })
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("bypasses extension messaging for direct-preferred uploads", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: { via: "runtime" }
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { via: "direct" }
    })

    const { bgUpload } = await importProxy()

    await expect(
      bgUpload({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        fields: { media_type: "document" },
        preferDirect: true
      })
    ).resolves.toEqual({ via: "direct" })

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("does not fall back to direct upload on POST extension timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))

    const { bgUpload } = await importProxy()
    const pending = bgUpload({
      path: "/api/v1/media/add",
      method: "POST",
      fields: { title: "example" },
      timeoutMs: 100
    })
    const assertion = expect(pending).rejects.toThrow("Extension messaging timeout")

    await vi.advanceTimersByTimeAsync(5001)

    await assertion
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("falls back to direct stream when port errors before first data chunk", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const onDisconnectListeners = new Set<() => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) => onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) => onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: (listener: () => void) => onDisconnectListeners.add(listener),
        removeListener: (listener: () => void) => onDisconnectListeners.delete(listener)
      },
      postMessage: vi.fn(() => {
        onMessageListeners.forEach((listener) =>
          listener({
            event: "error",
            message: "Could not establish connection. Receiving end does not exist."
          })
        )
      }),
      disconnect: vi.fn(() => {
        onDisconnectListeners.forEach((listener) => listener())
      })
    }
    mocks.connect.mockReturnValue(port as any)
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "not-a-real-key"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"event":"run_started","run_id":"run_1","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []

    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(mocks.connect).toHaveBeenCalledTimes(1)
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("classifies direct stream aborts as AbortError", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })

    let activeSignal: AbortSignal | null = null
    let resolveReadStarted: (() => void) | null = null
    const readStarted = new Promise<void>((resolve) => {
      resolveReadStarted = resolve
    })
    const reader = {
      read: vi.fn(() => {
        resolveReadStarted?.()
        return new Promise<never>((_, reject) => {
          const signal = activeSignal
          if (!signal) {
            reject(new Error("Missing abort signal"))
            return
          }
          const onAbort = () => {
            signal.removeEventListener("abort", onAbort)
            const abortError = new Error("The operation was aborted.")
            abortError.name = "AbortError"
            reject(abortError)
          }
          signal.addEventListener("abort", onAbort, { once: true })
        })
      }),
      cancel: vi.fn()
    }
    const fetchSpy = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      activeSignal = (init?.signal as AbortSignal | undefined) || null
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => reader
        }
      } as Response
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const controller = new AbortController()
    const consume = async () => {
      for await (const _chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] },
        abortSignal: controller.signal
      })) {
        // no-op
      }
    }

    const pending = consume()

    try {
      await readStarted
      controller.abort()

      await expect(pending).rejects.toMatchObject({
        name: "AbortError",
        status: 0,
        code: "REQUEST_ABORTED"
      })
      expect(fetchSpy).toHaveBeenCalledTimes(1)
      expect(reader.cancel).toHaveBeenCalledTimes(1)
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("treats post-first-chunk transport disconnect as graceful end", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const onDisconnectListeners = new Set<() => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) => onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) => onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: (listener: () => void) => onDisconnectListeners.add(listener),
        removeListener: (listener: () => void) => onDisconnectListeners.delete(listener)
      },
      postMessage: vi.fn(() => {
        onMessageListeners.forEach((listener) =>
          listener({
            event: "data",
            data: '{"choices":[{"delta":{"content":"H"}}]}'
          })
        )
        onMessageListeners.forEach((listener) =>
          listener({
            event: "error",
            message: "Could not establish connection. Receiving end does not exist."
          })
        )
      }),
      disconnect: vi.fn(() => {
        onDisconnectListeners.forEach((listener) => listener())
      })
    }
    mocks.connect.mockReturnValue(port as any)
    const fetchSpy = vi.fn(async () =>
      new Response("data: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []

    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chats/abc/complete-v2",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    expect(chunks).toContain('{"choices":[{"delta":{"content":"H"}}]}')
    expect(
      chunks.some((chunk) =>
        chunk.includes('"event":"stream_transport_interrupted"')
      )
    ).toBe(true)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it("falls back to direct stream when runtime.connect throws", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    mocks.connect.mockImplementation(() => {
      throw new Error("Could not establish connection. Receiving end does not exist.")
    })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"event":"run_started","run_id":"run_2","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []
    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(mocks.connect).toHaveBeenCalledTimes(1)
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("does not fall back to direct stream on HTTP status errors from port transport", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const onDisconnectListeners = new Set<() => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) => onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) => onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: (listener: () => void) => onDisconnectListeners.add(listener),
        removeListener: (listener: () => void) => onDisconnectListeners.delete(listener)
      },
      postMessage: vi.fn(() => {
        onMessageListeners.forEach((listener) =>
          listener({
            event: "error",
            status: 401,
            message: "Unauthorized"
          })
        )
      }),
      disconnect: vi.fn(() => {
        onDisconnectListeners.forEach((listener) => listener())
      })
    }
    mocks.connect.mockReturnValue(port as any)
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"event":"run_started","run_id":"run_fallback","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const consume = async () => {
      for await (const _chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        // no-op
      }
    }

    try {
      await expect(consume()).rejects.toMatchObject({
        message: "Unauthorized",
        status: 401
      })
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("blocks unallowlisted absolute URLs in direct stream fallback", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response("data: [DONE]\n\n", {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const consume = async () => {
      for await (const _chunk of bgStream({
        path:
          "https://evil.example.net/api/v1/chat/completions" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        // no-op
      }
    }

    try {
      await expect(consume()).rejects.toThrow("allowlisted")
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("preserves auth headers for same-origin absolute URLs in direct stream fallback", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"event":"run_started","run_id":"run_auth","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []

    try {
      for await (const chunk of bgStream({
        path:
          "https://api.example.com/api/v1/chat/completions" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    const fetchCalls = fetchSpy.mock.calls as unknown as Array<[RequestInfo | URL, RequestInit?]>
    const requestInit = fetchCalls[0]?.[1]
    const requestHeaders = new Headers(requestInit?.headers)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(requestHeaders.get("X-API-KEY")).toBe("test-key-not-placeholder")
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("uses hosted WebUI stream transport without browser auth headers", async () => {
    const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
    mocks.sendMessage.mockResolvedValue(null)
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://ignored-hosted.example.com",
          authMode: "multi-user",
          accessToken: "stale-browser-token",
          orgId: 17
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"event":"run_started","run_id":"run_hosted","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []

    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      if (originalDeploymentMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
      }
      vi.unstubAllGlobals()
    }

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const [url, init] = fetchSpy.mock.calls[0] as [RequestInfo | URL, RequestInit?]
    const requestHeaders = new Headers(init?.headers)
    expect(url).toBe("/api/proxy/chat/completions")
    expect(requestHeaders.get("Authorization")).toBeNull()
    expect(requestHeaders.get("X-TLDW-Org-Id")).toBe("17")
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("uses resolved advanced transport origin for stream and refresh when serverUrl is unset", async () => {
    const originalApiUrl = process.env.NEXT_PUBLIC_API_URL
    const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    const originalWindow = globalThis.window
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    process.env.NEXT_PUBLIC_API_URL = "https://api.example.test"
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "https://webui.example.test",
          protocol: "https:"
        }
      },
      configurable: true
    })
    mocks.sendMessage.mockResolvedValue(null)
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          authMode: "multi-user",
          accessToken: "expired-access",
          refreshToken: "refresh-token"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const authHeader = new Headers(init?.headers).get("Authorization") ?? ""
      if (url === "https://api.example.test/api/v1/auth/refresh") {
        return new Response(
          JSON.stringify({
            access_token: "fresh-access",
            refresh_token: "fresh-refresh",
            token_type: "bearer"
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" }
          }
        )
      }
      if (
        url === "https://api.example.test/api/v1/chat/completions" &&
        authHeader === "Bearer expired-access"
      ) {
        return new Response("Could not validate credentials", {
          status: 401,
          headers: { "content-type": "text/plain" }
        })
      }
      return new Response(
        'data: {"event":"run_started","run_id":"run_advanced","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []

    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      if (originalApiUrl === undefined) delete process.env.NEXT_PUBLIC_API_URL
      else process.env.NEXT_PUBLIC_API_URL = originalApiUrl
      if (originalDeploymentMode === undefined) {
        delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      } else {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
      }
      Object.defineProperty(globalThis, "window", {
        value: originalWindow,
        configurable: true
      })
      vi.unstubAllGlobals()
    }

    expect(fetchSpy.mock.calls[0]?.[0]).toBe(
      "https://api.example.test/api/v1/chat/completions"
    )
    expect(fetchSpy.mock.calls[1]?.[0]).toBe(
      "https://api.example.test/api/v1/auth/refresh"
    )
    expect(fetchSpy.mock.calls[2]?.[0]).toBe(
      "https://api.example.test/api/v1/chat/completions"
    )
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })
  it("does not refresh or re-add auth for cross-origin absolute stream URLs", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "multi-user",
          accessToken: "secret-access-token",
          refreshToken: "secret-refresh-token",
          absoluteUrlAllowlist: ["https://other.example.com"]
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes("/api/v1/auth/refresh")) {
        return new Response(JSON.stringify({ access_token: "new-token" }), {
          status: 200,
          headers: { "content-type": "application/json" }
        })
      }
      return new Response("Unauthorized", {
        status: 401,
        headers: { "content-type": "text/plain" }
      })
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const consume = async () => {
      for await (const _chunk of bgStream({
        path:
          "https://other.example.com/api/v1/chat/completions" as unknown as `/${string}`,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        // no-op
      }
    }

    try {
      await expect(consume()).rejects.toThrow("Unauthorized")
      expect(fetchSpy).toHaveBeenCalledTimes(1)
      expect(String(fetchSpy.mock.calls[0]?.[0] || "")).toContain(
        "https://other.example.com/api/v1/chat/completions"
      )
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("persists rotated refresh token during direct stream refresh retry", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    let storageReadCount = 0
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        storageReadCount += 1
        if (storageReadCount === 1) {
          return {
            serverUrl: "http://127.0.0.1:8000",
            authMode: "multi-user",
            accessToken: "expired-access",
            refreshToken: "old-refresh",
            orgId: 1
          }
        }
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "multi-user",
          accessToken: "expired-access",
          refreshToken: "old-refresh",
          orgId: 99,
          customFlag: true
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.endsWith("/api/v1/auth/refresh")) {
        return new Response(
          JSON.stringify({
            access_token: "new-access",
            refresh_token: "new-refresh",
            token_type: "bearer"
          }),
          {
            status: 200,
            headers: { "content-type": "application/json" }
          }
        )
      }
      const authHeader = new Headers(init?.headers).get("Authorization") ?? ""
      if (authHeader === "Bearer expired-access") {
        return new Response("Could not validate credentials", {
          status: 401,
          headers: { "content-type": "text/plain" }
        })
      }
      return new Response(
        'data: {"event":"run_started","run_id":"run_refresh","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    })
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []

    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    expect(mocks.storageSet).toHaveBeenCalledWith(
      "tldwConfig",
      expect.objectContaining({
        accessToken: "new-access",
        refreshToken: "new-refresh",
        orgId: 99,
        customFlag: true
      })
    )
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("falls back directly when runtime ping preflight times out", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"event":"run_started","run_id":"run_3","seq":1,"data":{}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []
    const streamTask = (async () => {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    })()
    try {
      await vi.advanceTimersByTimeAsync(401)
      await streamTask
    } finally {
      vi.unstubAllGlobals()
    }

    expect(mocks.connect).not.toHaveBeenCalled()
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("cooperatively yields while draining large stream queues", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) =>
          onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) =>
          onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: vi.fn(),
        removeListener: vi.fn()
      },
      postMessage: vi.fn(() => {
        for (let i = 0; i < 180; i += 1) {
          onMessageListeners.forEach((listener) =>
            listener({
              event: "data",
              data: JSON.stringify({
                choices: [{ delta: { content: String(i % 10) } }]
              })
            })
          )
        }
        onMessageListeners.forEach((listener) => listener({ event: "done" }))
      }),
      disconnect: vi.fn()
    }
    mocks.connect.mockReturnValue(port as any)
    const rafSpy = vi.fn((cb: FrameRequestCallback) => {
      cb(0)
      return 1
    })
    vi.stubGlobal("requestAnimationFrame", rafSpy as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []
    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    expect(chunks).toHaveLength(180)
    expect(rafSpy).toHaveBeenCalled()
  })

  it("preserves chunk ordering when draining queued stream data", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) =>
          onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) =>
          onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: vi.fn(),
        removeListener: vi.fn()
      },
      postMessage: vi.fn(() => {
        const ordered = ["A", "B", "C", "D", "E"]
        for (const token of ordered) {
          onMessageListeners.forEach((listener) =>
            listener({
              event: "data",
              data: JSON.stringify({
                choices: [{ delta: { content: token } }]
              })
            })
          )
        }
        onMessageListeners.forEach((listener) => listener({ event: "done" }))
      }),
      disconnect: vi.fn()
    }
    mocks.connect.mockReturnValue(port as any)
    vi.stubGlobal("requestAnimationFrame", ((cb: FrameRequestCallback) => {
      cb(0)
      return 1
    }) as any)

    const { bgStream } = await importProxy()
    const chunks: string[] = []
    try {
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    expect(chunks).toEqual([
      '{"choices":[{"delta":{"content":"A"}}]}',
      '{"choices":[{"delta":{"content":"B"}}]}',
      '{"choices":[{"delta":{"content":"C"}}]}',
      '{"choices":[{"delta":{"content":"D"}}]}',
      '{"choices":[{"delta":{"content":"E"}}]}'
    ])
  })
})
