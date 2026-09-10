import { afterEach, describe, expect, it, vi } from "vitest"
import { deriveRequestTimeout, tldwRequest } from "@/services/tldw/request-core"

const jsonResponse = (data: unknown, status = 200): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" }
  })

describe("deriveRequestTimeout generation defaults", () => {
  it("defaults chat completions to a generation-appropriate timeout (not 10s)", () => {
    const timeout = deriveRequestTimeout(null, "/api/v1/chat/completions")
    expect(timeout).toBeGreaterThanOrEqual(120000)
  })

  it("defaults rag endpoints to a generation-appropriate timeout (not 10s)", () => {
    const timeout = deriveRequestTimeout(null, "/api/v1/rag/search")
    expect(timeout).toBeGreaterThanOrEqual(120000)
  })

  it("keeps the model metadata catalog above the generic 10-second timeout", () => {
    expect(
      deriveRequestTimeout(
        { requestTimeoutMs: 10000 },
        "/api/v1/llm/models/metadata"
      )
    ).toBeGreaterThanOrEqual(60000)
    expect(
      deriveRequestTimeout(
        { requestTimeoutMs: 90000 },
        "/api/v1/llm/models/metadata"
      )
    ).toBe(90000)
  })

  it("still honors an explicit chatRequestTimeoutMs override", () => {
    const timeout = deriveRequestTimeout(
      { chatRequestTimeoutMs: 20000 },
      "/api/v1/chat/completions"
    )
    expect(timeout).toBe(20000)
  })
})

describe("tldwRequest post-refresh retry", () => {
  it("reuses the binary body (FormData) on the post-refresh retry instead of JSON.stringify", async () => {
    const bodies: unknown[] = []
    let refreshCalls = 0
    const fetchFn = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      bodies.push(init?.body)
      if (bodies.length === 1) {
        return new Response("unauthorized", { status: 401 })
      }
      return jsonResponse({ ok: true })
    }) as unknown as typeof fetch

    const runtime = {
      getConfig: async () => ({
        serverUrl: "https://api.example.com",
        authMode: "multi-user",
        accessToken: refreshCalls > 0 ? "fresh-access" : "stale-access",
        refreshToken: "refresh-token"
      }),
      refreshAuth: async () => {
        refreshCalls += 1
      },
      fetchFn
    }

    const form = new FormData()
    form.append("title", "example")

    const resp = await tldwRequest(
      {
        path: "https://api.example.com/api/v1/media/add",
        method: "POST",
        body: form
      },
      runtime
    )

    expect(resp.ok).toBe(true)
    expect(refreshCalls).toBe(1)
    expect(bodies).toHaveLength(2)
    // The retried request must send the SAME FormData instance, not "{}".
    expect(bodies[0]).toBe(form)
    expect(bodies[1]).toBe(form)
  })

  it("does not dispatch the retry when the request aborts during refresh", async () => {
    let releaseRefresh!: () => void
    let signalRefreshStarted!: () => void
    const refreshStarted = new Promise<void>((resolve) => {
      signalRefreshStarted = resolve
    })
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve
    })
    const fetchFn = vi.fn(async () =>
      new Response("unauthorized", { status: 401 })) as unknown as typeof fetch
    const abort = new AbortController()
    const runtime = {
      getConfig: async () => ({
        serverUrl: "https://api.example.com",
        authMode: "multi-user",
        accessToken: "stale-access",
        refreshToken: "refresh-token"
      }),
      refreshAuth: async () => {
        signalRefreshStarted()
        await refreshGate
      },
      fetchFn
    }

    const pending = tldwRequest(
      {
        path: "https://api.example.com/api/v1/chat/completions",
        method: "POST",
        body: { messages: [] },
        abortSignal: abort.signal
      },
      runtime
    )
    await refreshStarted
    abort.abort()
    releaseRefresh()

    await expect(pending).resolves.toMatchObject({ ok: false, status: 0 })
    expect(fetchFn).toHaveBeenCalledTimes(1)
  })
})

describe("tldwRequest timeout bounds", () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it("does not abort a >10s non-stream chat completion (uses the generation default)", async () => {
    vi.useFakeTimers()
    const fetchFn = vi.fn((_url: RequestInfo | URL, init?: RequestInit) => {
      return new Promise<Response>((resolve, reject) => {
        const signal = init?.signal
        const timer = setTimeout(() => resolve(jsonResponse({ ok: true })), 15000)
        signal?.addEventListener("abort", () => {
          clearTimeout(timer)
          const abortError = new Error("aborted")
          abortError.name = "AbortError"
          reject(abortError)
        })
      })
    }) as unknown as typeof fetch

    const runtime = {
      getConfig: async () => ({ serverUrl: "https://api.example.com" }),
      fetchFn
    }

    const pending = tldwRequest(
      {
        path: "https://api.example.com/api/v1/chat/completions",
        method: "POST",
        body: { messages: [] }
      },
      runtime
    )

    await vi.advanceTimersByTimeAsync(15001)
    const resp = await pending
    expect(resp.ok).toBe(true)
    expect(resp.status).toBe(200)
  })

  it("bounds the body read so a stalled body does not hang forever", async () => {
    vi.useFakeTimers()
    const fetchFn = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      const signal = init?.signal
      return {
        ok: true,
        status: 200,
        headers: new Headers({ "content-type": "application/json" }),
        // Body read never resolves on its own — only the request timeout can
        // unblock it by aborting the shared controller.
        json: () =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => {
              const abortError = new Error("aborted")
              abortError.name = "AbortError"
              reject(abortError)
            })
          }),
        text: () => Promise.resolve("")
      } as unknown as Response
    }) as unknown as typeof fetch

    const runtime = {
      getConfig: async () => ({ serverUrl: "https://api.example.com" }),
      fetchFn
    }

    const pending = tldwRequest(
      {
        path: "https://api.example.com/api/v1/chat/completions",
        method: "POST",
        body: { messages: [] }
      },
      runtime
    )

    // Advance past the derived (generation) timeout; the body read must abort
    // and resolve rather than hang.
    await vi.advanceTimersByTimeAsync(120001)
    const resp = await pending
    expect(resp.status).toBe(200)
    expect(resp.data).toBeNull()
  })
})
