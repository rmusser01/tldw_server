import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  sendMessage: vi.fn(),
  tldwRequest: vi.fn(),
  runtimeId: "test-extension" as string | null,
  storageGet: vi.fn(async () => ({ serverUrl: "http://127.0.0.1:8000" })),
  sessionStorageGet: vi.fn(async () => null)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return mocks.runtimeId
      },
      sendMessage: (...args: unknown[]) =>
        (mocks.sendMessage as (...args: unknown[]) => unknown)(...args)
    }
  }
}))

vi.mock("@/services/tldw/request-core", () => ({
  tldwRequest: (...args: unknown[]) =>
    (mocks.tldwRequest as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options?: { area?: string }) => ({
    get: (...args: unknown[]) =>
      options?.area === "session"
        ? (mocks.sessionStorageGet as (...args: unknown[]) => unknown)(...args)
        : (mocks.storageGet as (...args: unknown[]) => unknown)(...args)
  })
}))

const importApiSend = async () => import("@/services/api-send")

describe("apiSend timeout fallback policy", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useRealTimers()
    mocks.sendMessage.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.runtimeId = "test-extension"
    mocks.storageGet.mockReset()
    mocks.sessionStorageGet.mockReset()
    mocks.storageGet.mockResolvedValue({ serverUrl: "http://127.0.0.1:8000" })
    mocks.sessionStorageGet.mockResolvedValue(null)
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  })

  it("does not replay a captured write after ambiguous extension rejection", async () => {
    mocks.sendMessage.mockRejectedValue(new Error("message channel closed"))
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200 })
    const { apiSend } = await importApiSend()
    const result = await apiSend({
      path: "/api/v1/prompt-studio/prompts/create",
      method: "POST",
      recipePersistence: { mode: "capture" }
    })
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
    expect(result).toMatchObject({
      ok: false,
      recipePersistence: { state: "unknown", actualOwnerId: null }
    })
  })

  it("does not replace unavailable extension reconciliation with page-local authority", async () => {
    mocks.sendMessage.mockRejectedValue(
      new Error("receiving end does not exist")
    )
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200 })
    const { apiSend } = await importApiSend()
    const result = await apiSend({
      path: "/api/v1/prompts/123",
      method: "GET",
      recipePersistence: { mode: "capture" }
    })
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "unknown",
      actualOwnerId: null
    })
  })

  it("does not coalesce captured reconciliation with an unowned GET", async () => {
    let finish!: (value: unknown) => void
    mocks.sendMessage.mockReturnValue(
      new Promise((resolve) => {
        finish = resolve
      })
    )
    const { apiSend } = await importApiSend()
    const first = apiSend({ path: "/api/v1/health", method: "GET" })
    const second = apiSend({
      path: "/api/v1/health",
      method: "GET",
      recipePersistence: { mode: "capture" }
    })
    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
    finish({ ok: true, status: 200 })
    await Promise.all([first, second])
  })

  it("falls back to direct request for GET timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { apiSend } = await importApiSend()
    const pending = apiSend({ path: "/api/v1/health", method: "GET" })

    await vi.advanceTimersByTimeAsync(10001)

    await expect(pending).resolves.toMatchObject({ ok: true, status: 200 })
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("coalesces concurrent identical GET requests through direct fallback", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: [{ id: "research_assistant" }]
    })

    const { apiSend } = await importApiSend()
    const first = apiSend({ path: "/api/v1/persona/profiles", method: "GET" })
    const second = apiSend({ path: "/api/v1/persona/profiles", method: "GET" })

    await vi.advanceTimersByTimeAsync(10001)

    await expect(Promise.all([first, second])).resolves.toEqual([
      { ok: true, status: 200, data: [{ id: "research_assistant" }] },
      { ok: true, status: 200, data: [{ id: "research_assistant" }] }
    ])
    expect(mocks.tldwRequest).toHaveBeenCalledTimes(1)
  })

  it("does not coalesce unsafe POST requests", async () => {
    mocks.sendMessage.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { apiSend } = await importApiSend()
    await Promise.all([
      apiSend({
        path: "/api/v1/notes/search/",
        method: "POST",
        body: { q: "hello" }
      }),
      apiSend({
        path: "/api/v1/notes/search/",
        method: "POST",
        body: { q: "hello" }
      })
    ])

    expect(mocks.sendMessage).toHaveBeenCalledTimes(2)
  })

  it.each(["extension", "direct"])(
    "bypasses GET coalescing only for client-local fresh calls via %s",
    async (transport) => {
      mocks.runtimeId = transport === "extension" ? "test-extension" : null
      const requests: unknown[] = []
      const finish: Array<(value: unknown) => void> = []
      const boundary = vi.fn((request: unknown) => {
        requests.push(request)
        return new Promise((resolve) => {
          finish.push(resolve)
        })
      })
      if (transport === "extension")
        mocks.sendMessage.mockImplementation(boundary)
      else mocks.tldwRequest.mockImplementation(boundary)
      const { apiSend } = await importApiSend()
      const payload = {
        path: "/api/v1/prompts/capabilities",
        method: "GET"
      } as const
      const first = apiSend(payload)
      const fresh = apiSend(payload, { coalesce: false })
      const secondFresh = apiSend(payload, { coalesce: false })
      const ordinary = apiSend(payload)
      try {
        expect(requests).toHaveLength(3)
        expect(requests).toEqual(
          Array.from({ length: 3 }, () =>
            transport === "extension"
              ? {
                  type: "tldw:request",
                  payload: {
                    path: "/api/v1/prompts/capabilities",
                    method: "GET"
                  }
                }
              : { path: "/api/v1/prompts/capabilities", method: "GET" }
          )
        )
      } finally {
        finish.forEach((resolve, index) =>
          resolve({ ok: true, status: 200, data: index })
        )
        await Promise.all([first, fresh, secondFresh, ordinary])
      }
      expect((await fresh).data).toBe(1)
      expect((await secondFresh).data).toBe(2)
      expect((await ordinary).data).toBe(0)
    }
  )

  it("does not fall back to direct request for POST timeout", async () => {
    vi.useFakeTimers()
    mocks.sendMessage.mockImplementation(() => new Promise(() => undefined))
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { apiSend } = await importApiSend()
    const pending = apiSend({
      path: "/api/v1/notes/search/",
      method: "POST",
      body: { q: "hello" }
    })
    const assertion = expect(pending).rejects.toThrow(
      "Extension messaging timeout"
    )

    await vi.advanceTimersByTimeAsync(10001)

    await assertion
    expect(mocks.tldwRequest).not.toHaveBeenCalled()
  })

  it("uses an eligible WebUI cookie marker in the direct fallback without stored auth", async () => {
    mocks.runtimeId = null
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    mocks.storageGet.mockImplementation(async (key: string) => {
      if (key === "tldwCookieSessionConfig") {
        return {
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "must-not-leak",
          accessToken: "must-not-leak"
        }
      }
      if (key === "tldwConfig") {
        return {
          serverUrl: "https://remote.example.test",
          authMode: "single-user",
          authSource: "manual",
          apiKey: "preserved-device-key",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: "https://remote.example.test"
        }
      }
      return null
    })
    mocks.tldwRequest.mockResolvedValue({
      ok: true,
      status: 200,
      data: { ok: true }
    })

    const { apiSend } = await importApiSend()
    await apiSend({
      path: "/api/v1/notes/search/",
      method: "POST",
      body: { q: "cookie" }
    })

    const runtime = mocks.tldwRequest.mock.calls[0]?.[1] as {
      getConfig: () => Promise<Record<string, unknown>>
    }
    const config = await runtime.getConfig()
    expect(config).toMatchObject({
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    })
    expect(config).not.toHaveProperty("apiKey")
    expect(config).not.toHaveProperty("accessToken")
  })

  it("hydrates an exact-origin session credential in the direct fallback", async () => {
    mocks.runtimeId = null
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
            apiKey: "api-send-session-key",
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

    const { apiSend } = await importApiSend()
    await apiSend({ path: "/api/v1/health", method: "GET" })

    const runtime = mocks.tldwRequest.mock.calls[0]?.[1] as {
      getConfig: () => Promise<Record<string, unknown>>
    }
    await expect(runtime.getConfig()).resolves.toMatchObject({
      apiKey: "api-send-session-key"
    })
    expect(persistentConfig).not.toHaveProperty("apiKey")
  })

  it("forwards Prompt Studio recipe policy only as client-local metadata", async () => {
    mocks.runtimeId = null
    mocks.tldwRequest.mockResolvedValue({ ok: true, status: 200 })
    const { createPrompt, getPrompt, updatePrompt } = await import(
      "@/services/prompt-studio"
    )
    const required = {
      recipePersistence: {
        mode: "require" as const,
        expectedOwnerId: `recipe-owner:sha256:${"a".repeat(64)}`,
        localId: "local-recipe-1"
      }
    }

    await createPrompt({ project_id: 1, name: "Recipe" }, required)
    await updatePrompt(
      2,
      { prompt_format: "legacy", change_description: "Update" },
      required
    )
    await getPrompt(2, { recipePersistence: { mode: "capture" } })

    expect(mocks.tldwRequest.mock.calls.map(([request]) => request)).toEqual([
      expect.objectContaining(required),
      expect.objectContaining(required),
      expect.objectContaining({
        recipePersistence: { mode: "capture" }
      })
    ])
    for (const [request] of mocks.tldwRequest.mock.calls) {
      expect(request).not.toHaveProperty("capturePersistenceScope")
      expect(request).not.toHaveProperty("requirePersistenceScope")
    }
  })
})
