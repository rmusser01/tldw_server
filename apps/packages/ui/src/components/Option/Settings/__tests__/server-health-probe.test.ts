import { describe, expect, it, vi } from "vitest"

import { probeServerHealth } from "@/components/Option/Settings/server-health-probe"

describe("probeServerHealth", () => {
  it("calls /api/v1/health on the provided server origin", async () => {
    const fetchFn = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 200 })
    )

    const resp = await probeServerHealth({
      serverUrl: "http://127.0.0.1:8000/",
      fetchFn
    })

    expect(resp).toEqual({ ok: true, status: 200, error: undefined })
    expect(fetchFn).toHaveBeenCalledTimes(1)
    const [calledUrl] = fetchFn.mock.calls[0] ?? []
    expect(calledUrl).toBe("http://127.0.0.1:8000/api/v1/health")
  })

  it("sends API key header in single-user mode when provided", async () => {
    const fetchFn = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(null, { status: 200 })
    )

    await probeServerHealth({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "abc123",
      fetchFn
    })

    const [, options] = fetchFn.mock.calls[0] ?? []
    expect(options?.headers).toMatchObject({ "X-API-KEY": "abc123" })
  })

  it("fails single-user readiness when an authenticated workspace storage endpoint rejects the API key", async () => {
    const fetchFn = vi.fn(
      async (input: RequestInfo | URL, _init?: RequestInit) => {
        const url = String(input)
        if (url.endsWith("/api/v1/health")) {
          return new Response(null, { status: 200 })
        }
        if (url.endsWith("/api/v1/users/storage")) {
          return new Response(
            JSON.stringify({ detail: "Invalid or missing API Key" }),
            {
              status: 401,
              statusText: "Unauthorized",
              headers: { "content-type": "application/json" }
            }
          )
        }
        return new Response("unexpected endpoint", { status: 404 })
      }
    )

    const resp = await probeServerHealth({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "bad-key",
      fetchFn
    })

    expect(resp.ok).toBe(false)
    expect(resp.status).toBe(401)
    expect(resp.error).toContain("Invalid or missing API Key")
    expect(fetchFn).toHaveBeenCalledTimes(2)
    expect(fetchFn.mock.calls[1]?.[0]).toBe(
      "http://127.0.0.1:8000/api/v1/users/storage"
    )
    expect(fetchFn.mock.calls[1]?.[1]?.headers).toMatchObject({
      "X-API-KEY": "bad-key"
    })
  })

  it("returns backend error text for non-2xx responses", async () => {
    const fetchFn = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response("forbidden", { status: 403, statusText: "Forbidden" })
    )

    const resp = await probeServerHealth({
      serverUrl: "http://127.0.0.1:8000",
      fetchFn
    })

    expect(resp.ok).toBe(false)
    expect(resp.status).toBe(403)
    expect(resp.error).toContain("forbidden")
  })

  it("returns a configuration error when server URL is empty", async () => {
    const resp = await probeServerHealth({
      serverUrl: ""
    })

    expect(resp).toEqual({
      ok: false,
      status: 400,
      error: "tldw server not configured"
    })
  })

  it("maps network failures to status 0", async () => {
    const fetchFn = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
      throw new Error("Failed to fetch")
    })

    const resp = await probeServerHealth({
      serverUrl: "http://127.0.0.1:8000",
      fetchFn
    })

    expect(resp).toEqual({
      ok: false,
      status: 0,
      error: "Failed to fetch"
    })
  })
})
