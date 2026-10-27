import { describe, expect, it, vi } from "vitest"

import { resolveTldwRequestPlan, tldwRequest } from "../request-core"

/**
 * TASK-13526 transport Stage B1: the pre-flight plan resolution extracted
 * from performTldwRequest, and the optional runtime credential strategy.
 */
const jsonFetch = (capture?: Array<Record<string, unknown>>) =>
  vi.fn(async (_url: unknown, init?: unknown) => {
    capture?.push((init as Record<string, unknown>) ?? {})
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  }) as unknown as typeof fetch

describe("resolveTldwRequestPlan (Stage B1)", () => {
  it("resolves a same-origin quickstart URL and seeds JSON content type", async () => {
    const plan = await resolveTldwRequestPlan(
      { path: "/api/v1/health", method: "POST", body: { a: 1 } },
      {
        getConfig: async () => ({ serverUrl: "https://cfg.example.test" }),
      }
    )
    expect(plan.ok).toBe(true)
    if (!plan.ok) return
    expect(plan.plan.url).toContain("/api/v1/health")
    expect(plan.plan.headers["Content-Type"]).toBe("application/json")
    expect(plan.plan.shouldSkipAuth).toBe(false)
    expect(plan.plan.cookieSession).toBe(false)
  })

  it("blocks a non-allowlisted absolute URL with the guard envelope", async () => {
    const plan = await resolveTldwRequestPlan(
      { path: "https://evil.example.test/x" },
      { getConfig: async () => ({ serverUrl: "https://cfg.example.test" }) }
    )
    expect(plan).toEqual({
      ok: false,
      status: 400,
      error: expect.stringContaining("Absolute URL"),
    })
  })

  it("flags cross-origin allowlisted URLs as auth-skipping", async () => {
    const dest = "http://127.0.0.1:59999"
    const plan = await resolveTldwRequestPlan(
      { path: `${dest}/x` },
      {
        getConfig: async () => ({
          serverUrl: "https://cfg.example.test",
          absoluteUrlAllowlist: [dest],
        }),
      }
    )
    expect(plan.ok).toBe(true)
    if (plan.ok) {
      expect(plan.plan.isAbsolute).toBe(true)
      expect(plan.plan.shouldSkipAuth).toBe(true)
    }
  })
})

describe("runtime credential strategy (Stage B1)", () => {
  const cfg = { serverUrl: "https://cfg.example.test", authMode: "multi-user" as const }

  it("replaces default auth headers with the strategy's headers", async () => {
    const captured: Array<Record<string, unknown>> = []
    const result = await tldwRequest(
      { path: "/api/v1/health", headers: { Authorization: "Bearer stale" } },
      {
        getConfig: async () => cfg,
        fetchFn: jsonFetch(captured),
        resolveCredentials: ({ existingHeaders }) => {
          expect(existingHeaders["Authorization"]).toBe("Bearer stale")
          return { headers: { Authorization: "Bearer web-token", "X-Web": "1" } }
        },
      }
    )
    expect(result.ok).toBe(true)
    const headers = captured[0]?.headers as Record<string, string>
    expect(headers["Authorization"]).toBe("Bearer web-token")
    expect(headers["X-Web"]).toBe("1")
  })

  it("propagates the strategy's error envelope", async () => {
    const fetchFn = vi.fn()
    const result = await tldwRequest(
      { path: "/api/v1/health" },
      {
        getConfig: async () => cfg,
        fetchFn: fetchFn as unknown as typeof fetch,
        resolveCredentials: () => ({
          error: { status: 401, error: "web session expired" },
        }),
      }
    )
    expect(result).toMatchObject({ ok: false, status: 401 })
    expect(fetchFn).not.toHaveBeenCalled()
  })

  it("leaves default resolution untouched when no strategy is provided", async () => {
    const captured: Array<Record<string, unknown>> = []
    const result = await tldwRequest(
      { path: "/api/v1/health" },
      {
        getConfig: async () => ({ ...cfg, accessToken: "cfg-token" }),
        fetchFn: jsonFetch(captured),
      }
    )
    expect(result.ok).toBe(true)
    const headers = captured[0]?.headers as Record<string, string>
    expect(headers["Authorization"]).toBe("Bearer cfg-token")
  })
})
