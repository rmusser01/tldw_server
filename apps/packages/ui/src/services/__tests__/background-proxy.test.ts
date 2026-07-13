import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  connect: vi.fn(),
  tldwRequest: vi.fn(),
  getRuntimeSingleUserApiKeyOverride: vi.fn(),
  storageGet: vi.fn(async (_key?: string) => null),
  sessionStorageGet: vi.fn(async (_key?: string) => null),
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
  createSafeStorage: (options?: { area?: string }) => ({
    get: async (...args: unknown[]) =>
      await (options?.area === "session"
        ? (mocks.sessionStorageGet as (...args: unknown[]) => unknown)(...args)
        : (mocks.storageGet as (...args: unknown[]) => unknown)(...args)),
    set: (...args: unknown[]) =>
      (mocks.storageSet as (...args: unknown[]) => unknown)(...args)
  })
}))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: (...args: unknown[]) =>
    (mocks.getRuntimeSingleUserApiKeyOverride as (...args: unknown[]) => unknown)(...args),
  isCookieSessionConfigInvalidated: () => false
}))

const importProxy = async () => import("@/services/background-proxy")

describe("background proxy fallback safety", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useRealTimers()
    mocks.sendMessage.mockReset()
    mocks.connect.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.getRuntimeSingleUserApiKeyOverride.mockReset()
    mocks.storageGet.mockReset()
    mocks.sessionStorageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue(null)
    mocks.storageGet.mockResolvedValue(null)
    mocks.sessionStorageGet.mockResolvedValue(null)
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

  it("does not warn for expected response statuses", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    mocks.sendMessage.mockResolvedValue({
      ok: false,
      status: 404,
      error: "Chat settings not found"
    })

    try {
      const { bgRequest } = await importProxy()

      await expect(
        bgRequest({
          path: "/api/v1/chats/chat-1/settings",
          method: "GET",
          expectedStatuses: [404]
        })
      ).rejects.toMatchObject({ status: 404 })
      expect(warnSpy).not.toHaveBeenCalled()
    } finally {
      warnSpy.mockRestore()
    }
  })

  it("keeps auth enabled for same-origin absolute URLs in background requests", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://api.example.com"
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
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://api.example.com"
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

  it("keeps workspace migration chunk network failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/workspaces/migrations/mig-1/chunks/chunk-1",
          method: "PUT",
          body: { sha256: "a".repeat(64), byte_count: 1 }
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
  })

  it("keeps workspace source refresh network failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/workspaces/ws-1/sources",
          method: "GET"
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
  })

  it("keeps workspace upsert reconciliation failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/workspaces/ws-1",
          method: "PUT",
          body: { name: "Recovered workspace" }
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
  })

  it("keeps Research Workspace chat command bootstrap failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/chat/commands",
          method: "GET"
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
  })

  it("keeps optional audio voice bootstrap failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/audio/voices/catalog?provider=kitten_tts",
          method: "GET"
        })
      ).rejects.toMatchObject({ status: 0 })
      await expect(
        bgRequest({
          path: "/api/v1/audio/voices",
          method: "GET"
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
  })

  it("keeps optional ingestion-source capability bootstrap failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/ingestion-sources/capabilities",
          method: "GET"
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
  })

  it("keeps caller-handled best-effort failures scoped out of the global backend-unreachable modal", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("Could not establish connection. Receiving end does not exist.")
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })

    const eventSpy = vi.fn()
    const eventName = "tldw:backend-unreachable"
    window.addEventListener(eventName, eventSpy as EventListener)

    try {
      const { bgRequest } = await importProxy()
      await expect(
        bgRequest({
          path: "/api/v1/media/3/keywords",
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: { keywords: ["workspace:test"], mode: "add" },
          suppressBackendUnavailableEvent: true
        })
      ).rejects.toMatchObject({ status: 0 })
    } finally {
      window.removeEventListener(eventName, eventSpy as EventListener)
    }

    expect(eventSpy).not.toHaveBeenCalled()
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

  it("bypasses extension messaging for audio-studio artifact media arrayBuffer requests", async () => {
    const buffer = new Uint8Array([1, 2, 3]).buffer
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { via: "runtime" } })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: buffer
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "/api/v1/audio-studio/projects/p1/artifacts/a1/media",
        method: "GET",
        responseType: "arrayBuffer"
      })
    ).resolves.toBe(buffer)

    expect(mocks.sendMessage).not.toHaveBeenCalled()
    expect(mocks.tldwRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/audio-studio/projects/p1/artifacts/a1/media",
        method: "GET",
        responseType: "arrayBuffer"
      }),
      expect.any(Object)
    )
  })

  it("does not bypass extension messaging for unrelated audio-studio arrayBuffer requests", async () => {
    const buffer = new Uint8Array([4, 5, 6]).buffer
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: buffer
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { via: "direct" }
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({
        path: "/api/v1/audio-studio/projects/p1/artifacts",
        method: "GET",
        responseType: "arrayBuffer"
      })
    ).resolves.toBe(buffer)

    expect(mocks.sendMessage).toHaveBeenCalledTimes(1)
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("falls back to direct arrayBuffer requests while preserving response metadata", async () => {
    const buffer = new Uint8Array([7, 8, 9]).buffer
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: {},
      headers: {
        "content-disposition": 'attachment; filename="runtime.zip"'
      }
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: buffer,
      headers: {
        "content-disposition": 'attachment; filename="direct.zip"'
      }
    })

    const { bgRequest } = await importProxy()

    await expect(
      bgRequest<{
        ok: boolean
        status: number
        data: ArrayBuffer
        headers: Record<string, string>
      }>({
        path: "/api/v1/skills/client-skill/export",
        method: "GET",
        responseType: "arrayBuffer",
        returnResponse: true
      })
    ).resolves.toEqual({
      ok: true,
      status: 200,
      data: buffer,
      headers: {
        "content-disposition": 'attachment; filename="direct.zip"'
      }
    })

    expect(mocks.sendMessage).toHaveBeenCalledTimes(1)
    expect(mocks.tldwRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/skills/client-skill/export",
        method: "GET",
        responseType: "arrayBuffer"
      }),
      expect.any(Object)
    )
  })

  it("does not fall back to direct request on POST extension timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: { via: "direct" } })

    const { bgRequest } = await importProxy()
    const pending = bgRequest({
      path: "/api/v1/notes/search/",
      method: "POST",
      body: { q: "hello" },
      timeoutMs: 100
    })
    const assertion = expect(pending).rejects.toThrow("Extension messaging timeout")

    await vi.advanceTimersByTimeAsync(5001)

    await assertion
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("falls back to direct request for idempotent Web Clipper saves on extension timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { via: "direct" }
    })

    const { bgRequest } = await importProxy()
    const pending = bgRequest<{ via: string }>({
      path: "/api/v1/web-clipper/save",
      method: "POST",
      body: { clip_id: "clip-1" },
      timeoutMs: 100
    })

    await vi.advanceTimersByTimeAsync(5001)

    await expect(pending).resolves.toEqual({ via: "direct" })
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("waits beyond the short safe-method timeout for unsafe background writes", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(
            () => resolve({ ok: true, status: 200, data: { via: "runtime" } }),
            3500
          )
        })
    )
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200, data: { via: "direct" } })

    const { bgRequest } = await importProxy()
    const pending = bgRequest<{ via: string }>({
      path: "/api/v1/web-clipper/save",
      method: "POST",
      body: { clip_id: "clip-1" }
    })

    await vi.advanceTimersByTimeAsync(3001)
    await vi.advanceTimersByTimeAsync(500)

    await expect(pending).resolves.toEqual({ via: "runtime" })
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

  it("hydrates an origin-bound session key for direct uploads without persisting it", async () => {
    const persistentConfig = {
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    }
    mocks.storageGet.mockImplementation(async (key: string) =>
      key === "tldwConfig" ? persistentConfig : null
    )
    mocks.sessionStorageGet.mockImplementation(async (key: string) =>
      key === "tldwManualSessionApiKey"
        ? {
            apiKey: "session-upload-key",
            credentialSource: "manual",
            apiKeyPersistence: "session",
            apiKeyServerOrigin: "https://api.example.test"
          }
        : null
    )
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { bgUpload } = await importProxy()
    await bgUpload({
      path: "/api/v1/media/ingest/jobs",
      method: "POST",
      fields: { media_type: "document" },
      preferDirect: true
    })

    const runtime = mocks.tldwRequest.mock.calls[0]?.[1] as {
      getConfig: () => Promise<Record<string, unknown>>
    }
    await expect(runtime.getConfig()).resolves.toMatchObject({
      apiKey: "session-upload-key"
    })
    expect(persistentConfig).not.toHaveProperty("apiKey")
    expect(mocks.storageSet).not.toHaveBeenCalledWith(
      "tldwConfig",
      expect.objectContaining({ apiKey: "session-upload-key" })
    )
  })

  it("hydrates an origin-bound session key for direct HTTP streams without persisting it", async () => {
    const persistentConfig = {
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      authSource: "manual",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    }
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.storageGet.mockImplementation(async (key: string) =>
      key === "tldwConfig" ? persistentConfig : null
    )
    mocks.sessionStorageGet.mockImplementation(async (key: string) =>
      key === "tldwManualSessionApiKey"
        ? {
            apiKey: "session-stream-key",
            credentialSource: "manual",
            apiKeyPersistence: "session",
            apiKeyServerOrigin: "https://api.example.test"
          }
        : null
    )
    const fetchSpy = vi.fn(async () =>
      new Response('data: {"ok":true}\n\ndata: [DONE]\n\n', {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      })
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    try {
      const { bgStream } = await importProxy()
      const chunks: string[] = []
      for await (const chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        body: { stream: true }
      })) {
        chunks.push(chunk)
      }

      expect(chunks).toContain('{"ok":true}')
      const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
      expect(headers.get("X-API-KEY")).toBe("session-stream-key")
      expect(persistentConfig).not.toHaveProperty("apiKey")
      expect(mocks.storageSet).not.toHaveBeenCalledWith(
        "tldwConfig",
        expect.objectContaining({ apiKey: "session-stream-key" })
      )
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it.each(["POST", "PATCH"])(
    "uses cookie auth with current CSRF for direct %s streams",
    async (method) => {
      const previousMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      document.cookie = "csrf_token=stream-csrf; Path=/"
      mocks.sendMessage.mockResolvedValue({ ok: false })
      mocks.storageGet.mockImplementation(async (key: string) => {
        if (key === "tldwConfig") {
          return {
            serverUrl: "https://remote.example.test",
            authMode: "single-user",
            authSource: "manual",
            apiKey: "preserved-remote-key",
            credentialSource: "manual",
            apiKeyPersistence: "device",
            apiKeyServerOrigin: "https://remote.example.test"
          }
        }
        if (key === "tldwCookieSessionConfig") {
          return {
            serverUrl: window.location.origin,
            authMode: "single-user",
            authSource: "cookie-session"
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

      try {
        const { bgStream } = await importProxy()
        for await (const _chunk of bgStream({
          path: "/api/v1/chat/completions",
          method: method as "POST",
          headers: {
            "X-API-KEY": "stale-key",
            Authorization: "Bearer stale-token",
            "X-CSRF-Token": "stale-csrf"
          },
          body: { stream: true }
        })) {
          // no-op
        }

        const headers = new Headers(fetchSpy.mock.calls[0]?.[1]?.headers)
        expect(headers.get("X-CSRF-Token")).toBe("stream-csrf")
        expect(headers.has("X-API-KEY")).toBe(false)
        expect(headers.has("Authorization")).toBe(false)
      } finally {
        vi.unstubAllGlobals()
        document.cookie = "csrf_token=; Max-Age=0; Path=/"
        if (previousMode === undefined) {
          delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
        } else {
          process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = previousMode
        }
      }
    }
  )

  it("appends multiple named files for direct-preferred uploads", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { bgUpload } = await importProxy()
    const source = Uint8Array.from([1, 2, 3])
    const target = Uint8Array.from([4, 5, 6])

    await bgUpload({
      path: "/api/v1/audio/voice-conversion",
      method: "POST",
      fields: { response_format: "wav", stream: false },
      files: [
        {
          fieldName: "source_audio",
          name: "source.wav",
          type: "audio/wav",
          data: source
        },
        {
          fieldName: "target_voice",
          name: "target.wav",
          type: "audio/wav",
          data: target
        }
      ],
      preferDirect: true,
      responseType: "arrayBuffer"
    })

    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
    const requestPayload = mocks.tldwRequest.mock.calls[0][0] as { body?: FormData; responseType?: string }
    expect(requestPayload.responseType).toBe("arrayBuffer")
    const body = requestPayload.body as FormData
    expect(body.get("response_format")).toBe("wav")
    expect(body.get("stream")).toBe("false")
    expect(body.get("source_audio")).toBeInstanceOf(Blob)
    expect(body.get("target_voice")).toBeInstanceOf(Blob)
  })

  it("appends single direct-fallback uploads to files and the legacy file alias", async () => {
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { bgUpload } = await importProxy()

    await bgUpload({
      path: "/api/v1/media/add",
      method: "POST",
      file: {
        name: "clip.wav",
        type: "audio/wav",
        data: Uint8Array.from([7, 8, 9])
      },
      preferDirect: true
    })

    const requestPayload = mocks.tldwRequest.mock.calls[0][0] as { body?: FormData }
    const body = requestPayload.body as FormData
    expect(body.get("files")).toBeInstanceOf(Blob)
    expect(body.get("file")).toBeInstanceOf(Blob)
  })

  it("forwards multiple files through extension upload messaging", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { bgUpload } = await importProxy()
    await bgUpload({
      path: "/api/v1/audio/voice-conversion",
      method: "POST",
      files: [
        {
          fieldName: "source_audio",
          name: "source.wav",
          type: "audio/wav",
          data: Uint8Array.from([1])
        },
        {
          fieldName: "target_voice",
          name: "target.wav",
          type: "audio/wav",
          data: Uint8Array.from([2])
        }
      ],
      responseType: "arrayBuffer"
    })

    expect(mocks.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "tldw:upload",
        payload: expect.objectContaining({
          responseType: "arrayBuffer",
          files: [
            expect.objectContaining({ fieldName: "source_audio", name: "source.wav" }),
            expect.objectContaining({ fieldName: "target_voice", name: "target.wav" })
          ]
        })
      })
    )
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
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

  it("does not replay a non-idempotent POST when port errors before first data chunk", async () => {
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
          apiKey: "not-a-real-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000"
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
      // The POST must NOT be re-sent (no duplicate generation); a transport
      // interruption is surfaced instead.
      await expect(consume()).rejects.toMatchObject({ code: "STREAM_INTERRUPTED" })
      expect(fetchSpy).not.toHaveBeenCalled()
      expect(mocks.connect).toHaveBeenCalledTimes(1)
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("replays an idempotent GET stream via direct fetch when port errors before first data chunk", async () => {
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
          apiKey: "not-a-real-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000"
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
        path: "/api/v1/chat/completions" as unknown as `/${string}`,
        method: "GET",
        headers: { "Content-Type": "application/json" }
      })) {
        chunks.push(chunk)
      }
    } finally {
      vi.unstubAllGlobals()
    }

    // GET is idempotent, so a direct-fetch replay is safe.
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(mocks.connect).toHaveBeenCalledTimes(1)
    expect(chunks.some((chunk) => chunk.includes('"event":"run_started"'))).toBe(true)
  })

  it("does not replay POST streams after an ambiguous postMessage failure", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) => onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) => onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: vi.fn(),
        removeListener: vi.fn()
      },
      postMessage: vi.fn(() => {
        throw new Error("Message port closed during postMessage")
      }),
      disconnect: vi.fn()
    }
    mocks.connect.mockReturnValue(port as any)
    mocks.storageGet.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key-not-placeholder"
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
        path: "/api/v1/chat/completions",
        method: "POST",
        body: { stream: true, messages: [] }
      })) {
        // no-op
      }
    }

    try {
      await expect(consume()).rejects.toThrow("Message port closed")
      expect(port.postMessage).toHaveBeenCalledTimes(1)
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("does not replay a non-idempotent POST after a response-acquisition timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockResolvedValue({ ok: true })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "not-a-real-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000",
          // Small idle timeout so the connection window is easy to advance past.
          streamIdleTimeoutMs: 1000
        }
      }
      return null
    })
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
      // Never emit an open event: simulate a stalled response acquisition.
      postMessage: vi.fn(),
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
    const consume = async () => {
      for await (const _chunk of bgStream({
        path: "/api/v1/chats/abc/complete-v2",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true }
      })) {
        // no-op
      }
    }

    try {
      const pending = consume()
      const assertion = expect(pending).rejects.toMatchObject({
        code: "STREAM_INTERRUPTED"
      })
      // Advance past the derived connection timeout to fire the disconnect.
      await vi.advanceTimersByTimeAsync(1001)
      // Let the drain loop's 10ms poll observe done + throw.
      await vi.advanceTimersByTimeAsync(20)
      await assertion
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
      vi.unstubAllGlobals()
    }
  })

  it("classifies direct stream aborts as AbortError", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000"
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
      } as unknown as Response
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
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000"
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
            status: 429,
            message: "Too Many Requests",
            retryAfter: "40"
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
        message: "Too Many Requests",
        status: 429,
        retryAfter: 40
      })
      expect(fetchSpy).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it("preserves Retry-After on direct stream fallback errors", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    mocks.connect.mockImplementation(() => {
      throw new Error("runtime port unavailable")
    })
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "http://127.0.0.1:8000",
          authMode: "single-user",
          apiKey: "not-a-real-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000"
        }
      }
      return null
    })
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(null, {
          status: 503,
          statusText: "Service Unavailable",
          headers: { "Retry-After": "40" }
        })
      ) as any
    )

    const { bgStream } = await importProxy()
    const consume = async () => {
      for await (const _chunk of bgStream({
        path: "/api/v1/notifications/stream" as unknown as `/${string}`,
        method: "GET"
      })) {
        // no-op
      }
    }

    try {
      await expect(consume()).rejects.toMatchObject({
        status: 503,
        retryAfter: 40
      })
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
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://api.example.com"
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
          authSource: "manual",
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://api.example.com"
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

  it("uses the runtime single-user key for WebUI direct stream fallback", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue("runtime-stream-key")
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
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

    const fetchCalls = fetchSpy.mock.calls as unknown as Array<[RequestInfo | URL, RequestInit?]>
    const requestHeaders = new Headers(fetchCalls[0]?.[1]?.headers)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(requestHeaders.get("X-API-KEY")).toBe("runtime-stream-key")
    expect(chunks.some((chunk) => chunk.includes('"content":"ok"'))).toBe(true)
  })

  it("ignores whitespace runtime single-user keys for direct stream auth", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: false })
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue("   ")
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://api.example.com",
          authMode: "single-user",
          apiKey: "persisted-stream-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://api.example.com"
        }
      }
      return null
    })
    const fetchSpy = vi.fn(async () =>
      new Response(
        'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { "content-type": "text/event-stream" }
        }
      )
    )
    vi.stubGlobal("fetch", fetchSpy as any)

    const { bgStream } = await importProxy()

    try {
      for await (const _chunk of bgStream({
        path: "/api/v1/chat/completions",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { stream: true, messages: [] }
      })) {
        // drain stream
      }
    } finally {
      vi.unstubAllGlobals()
    }

    const fetchCalls = fetchSpy.mock.calls as unknown as Array<[RequestInfo | URL, RequestInit?]>
    const requestHeaders = new Headers(fetchCalls[0]?.[1]?.headers)
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(requestHeaders.get("X-API-KEY")).toBe("persisted-stream-key")
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
    const [url, init] = fetchSpy.mock.calls[0] as unknown as [
      RequestInfo | URL,
      RequestInit?
    ]
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
          apiKey: "test-key-not-placeholder",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "http://127.0.0.1:8000"
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

  it("signals response acquisition before the first stream data item", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true })
    const onMessageListeners = new Set<(msg: any) => void>()
    const port = {
      onMessage: {
        addListener: (listener: (msg: any) => void) => onMessageListeners.add(listener),
        removeListener: (listener: (msg: any) => void) => onMessageListeners.delete(listener)
      },
      onDisconnect: {
        addListener: vi.fn(),
        removeListener: vi.fn()
      },
      postMessage: vi.fn(() => {
        onMessageListeners.forEach((listener) => listener({ event: "open" }))
        onMessageListeners.forEach((listener) => listener({ event: "done" }))
      }),
      disconnect: vi.fn()
    }
    mocks.connect.mockReturnValue(port as any)
    const onOpen = vi.fn()

    const { bgStream } = await importProxy()
    for await (const _chunk of bgStream({
      path: "/api/v1/notifications/stream" as unknown as `/${string}`,
      method: "GET",
      onOpen
    })) {
      // The open signal is deliberately independent of stream data.
    }

    expect(onOpen).toHaveBeenCalledTimes(1)
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

describe("background proxy GET coalescing", () => {
  beforeEach(() => {
    vi.resetModules()
    mocks.sendMessage.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.storageGet.mockResolvedValue(null)
    mocks.storageSet.mockResolvedValue(undefined)
  })

  it("coalesces concurrent identical GETs into a single underlying request", async () => {
    let resolveSend: (value: unknown) => void = () => {}
    const pending = new Promise((resolve) => {
      resolveSend = resolve
    })
    mocks.sendMessage.mockReturnValue(pending)

    const { bgRequest } = await importProxy()

    // Two identical concurrent GETs + one different concurrent GET.
    const a1 = bgRequest({ path: "/api/v1/users/me/profile?sections=preferences", method: "GET" })
    const a2 = bgRequest({ path: "/api/v1/users/me/profile?sections=preferences", method: "GET" })
    const b1 = bgRequest({ path: "/api/v1/config/providers", method: "GET" })

    resolveSend({ ok: true, status: 200, data: { ok: true } })
    const [ra1, ra2] = await Promise.all([a1, a2])
    await b1

    // Identical pair shares one underlying call; the different path makes its own.
    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
    expect(ra1).toBe(ra2)
  })

  it("coalesces serialized returnResponse GETs", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: [{ id: "persona-1" }]
    })
    const { bgRequest } = await importProxy()

    const [first, second] = await Promise.all([
      bgRequest({
        path: "/api/v1/persona/profiles",
        method: "GET",
        returnResponse: true
      }),
      bgRequest({
        path: "/api/v1/persona/profiles",
        method: "GET",
        returnResponse: true
      })
    ])

    expect(mocks.sendMessage).toHaveBeenCalledTimes(1)
    expect(first).toBe(second)
  })

  it("does not coalesce returnResponse GETs with data-only GETs", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })
    const { bgRequest } = await importProxy()

    await Promise.all([
      bgRequest({
        path: "/api/v1/persona/profiles",
        method: "GET",
        returnResponse: true
      }),
      bgRequest({ path: "/api/v1/persona/profiles", method: "GET" })
    ])

    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
  })

  it("reuses a recent rate-limited GET failure instead of bursting", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: false,
      status: 429,
      error: "rate_limited"
    })
    const { bgRequest } = await importProxy()

    await expect(
      bgRequest({ path: "/api/v1/persona/profiles", method: "GET" })
    ).rejects.toMatchObject({ status: 429 })
    await expect(
      bgRequest({ path: "/api/v1/persona/profiles", method: "GET" })
    ).rejects.toMatchObject({ status: 429 })

    expect(mocks.sendMessage).toHaveBeenCalledTimes(1)
  })

  it("does not coalesce POST requests", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })
    const { bgRequest } = await importProxy()

    await Promise.all([
      bgRequest({ path: "/api/v1/users/me/profile", method: "POST", body: { a: 1 } }),
      bgRequest({ path: "/api/v1/users/me/profile", method: "POST", body: { a: 1 } })
    ])

    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
  })

  it("does not coalesce GETs with different timeoutMs", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })
    const { bgRequest } = await importProxy()

    await Promise.all([
      bgRequest({ path: "/api/v1/config/providers", method: "GET", timeoutMs: 5000 }),
      bgRequest({ path: "/api/v1/config/providers", method: "GET", timeoutMs: 30000 })
    ])

    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
  })

  it("does not coalesce absolute-URL GETs that differ only by noAuth omitted vs false", async () => {
    mocks.sendMessage.mockResolvedValue({ ok: true, status: 200, data: { ok: true } })
    const { bgRequest } = await importProxy()

    await Promise.all([
      bgRequest({ path: "https://api.example.com/api/v1/health", method: "GET" }),
      bgRequest({ path: "https://api.example.com/api/v1/health", method: "GET", noAuth: false })
    ])

    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
  })
})
