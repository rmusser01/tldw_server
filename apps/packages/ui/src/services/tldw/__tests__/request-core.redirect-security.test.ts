// @vitest-environment node
import http from "node:http"
import type { AddressInfo } from "node:net"
import { afterEach, describe, expect, it, vi } from "vitest"
import { tldwRequest } from "../request-core"
import type { PathOrUrl } from "../openapi-guard"

const servers: http.Server[] = []
const listen = async (handler: http.RequestListener) => {
  const server = http.createServer(handler)
  servers.push(server)
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", resolve)
  })
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`
}

afterEach(async () => {
  vi.unstubAllEnvs()
  await Promise.all(servers.splice(0).map((server) => new Promise<void>((resolve) => {
    server.close(() => resolve())
    server.closeAllConnections()
  })))
})

describe("configured server redirect boundary", () => {
  it.each([
    { serverUrl: "https://new.example.test" },
    { authMode: "single-user" },
    { authSource: "cookie-session" },
    { orgId: 2 },
    { accessToken: "header.eyJzdWIiOiJvdGhlci11c2VyIn0.signature" }
  ])("does not retry across a configuration scope change: %j", async (change) => {
    const config = {
      serverUrl: "https://old.example.test",
      authMode: "multi-user",
      authSource: "manual",
      orgId: 1,
      accessToken: "header.eyJzdWIiOiJvcmlnaW5hbC11c2VyIn0.signature",
      refreshToken: "synthetic-refresh"
    }
    const sent: RequestInit[] = []
    const fetchFn = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      sent.push(init!)
      return new Response("{}", { status: sent.length === 1 ? 401 : 200 })
    })
    await expect(tldwRequest({ path: "/api/v1/notes", method: "POST", body: "private draft" }, {
      getConfig: async () => config,
      refreshAuth: async () => { Object.assign(config, change) },
      fetchFn
    })).rejects.toMatchObject({ status: 412 })
    expect(sent).toHaveLength(1)
  })

  it.each([
    { path: ["https://attacker.example/collect"] },
    { path: { toString: () => "https://attacker.example/collect" } },
    { path: null },
    { path: undefined },
    { path: 123 }
  ])("rejects non-string paths before reading credentials: $path", async ({ path }) => {
    const getConfig = vi.fn(async () => ({
      serverUrl: "https://configured.example.test",
      authMode: "single-user",
      apiKey: "redirect-regression-key"
    }))
    const fetchFn = vi.fn(async () => new Response("{}"))
    const result = await tldwRequest({ path: path as unknown as PathOrUrl }, {
      getConfig,
      fetchFn,
      useRuntimeAuthOverride: false
    })
    expect(result.status).toBe(400)
    expect(getConfig).not.toHaveBeenCalled()
    expect(fetchFn).not.toHaveBeenCalled()
  })

  it("allows an explicitly selected cross-origin URL without server credentials", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    const credentials: Array<string | undefined> = []
    const destination = await listen((request, response) => {
      credentials.push(request.headers["x-api-key"] as string | undefined)
      response.setHeader("content-type", "application/json")
      response.end(JSON.stringify({ selected: true }))
    })
    const result = await tldwRequest({ path: `${destination}/selected` }, {
      getConfig: async () => ({
        serverUrl: "https://configured.example.test",
        authMode: "single-user",
        apiKey: "redirect-regression-key",
        absoluteUrlAllowlist: [destination]
      }),
      useRuntimeAuthOverride: false
    })
    expect(result.ok).toBe(true)
    expect(result.data).toEqual({ selected: true })
    expect(credentials).toEqual([undefined])
  })

  it.each(["single-user", "multi-user"] as const)(
    "does not send a %s request to a redirect destination",
    async (authMode) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
      const received: string[] = []
      const destination = await listen((request, response) => {
        received.push(String(request.headers["x-api-key"] || request.headers.authorization || "request"))
        response.setHeader("content-type", "application/json")
        response.end("{}")
      })
      let requests = 0
      const configured = await listen((_request, response) => {
        requests += 1
        if (authMode === "multi-user" && requests === 1) {
          response.writeHead(401)
        } else {
          response.writeHead(307, { Location: `${destination}/collect` })
        }
        response.end()
      })
      const result = await tldwRequest({ path: "/api/v1/notes", method: "GET" }, {
        getConfig: async () => ({
          serverUrl: configured,
          authMode,
          apiKey: "redirect-regression-key",
          accessToken: "redirect-regression-token",
          refreshToken: "redirect-regression-refresh"
        }),
        refreshAuth: async () => undefined,
        useRuntimeAuthOverride: false
      })
      expect(received).toEqual([])
      expect(result.ok).toBe(false)
      expect(requests).toBe(authMode === "multi-user" ? 2 : 1)
    }
  )
})
