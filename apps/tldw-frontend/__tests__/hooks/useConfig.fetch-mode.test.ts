import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { tldwRequest, type TldwRequestPayload } from "@/services/tldw/request-core"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  bgRequest: vi.fn(),
  fetch: vi.fn<typeof fetch>(),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: () => mocks.getConfig(),
    getOpenAPISpec: async () => ({ info: { version: "bootstrap" }, paths: {} }),
  },
  isActiveCookieSessionConfig: () => false,
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (request: TldwRequestPayload) => mocks.bgRequest(request),
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: async () => undefined, set: async () => undefined }),
}))

// The retired useConfig provider was removed in 661b76c144. Public docs discovery
// now lives in server-capabilities and distinguishes public from configured discovery.
describe("docs-info discovery fetch mode", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", undefined)
    mocks.getConfig.mockReset()
    mocks.fetch.mockReset().mockResolvedValue(new Response(JSON.stringify({}), {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
    mocks.bgRequest.mockReset().mockImplementation(async (request: TldwRequestPayload) => {
      const response = await tldwRequest(request, {
        getConfig: () => mocks.getConfig(),
        fetchFn: mocks.fetch,
      })
      if (!response.ok) throw new Error(response.error)
      return response.data
    })
  })

  afterEach(() => vi.unstubAllEnvs())

  const expectPublicFetch = () => {
    expect(mocks.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = mocks.fetch.mock.calls[0]
    expect(url).toBe("https://api.example.test/api/v1/config/docs-info")
    expect(init?.credentials ?? "same-origin").not.toBe("include")
    const headers = new Headers(init?.headers)
    expect(headers.has("authorization")).toBe(false)
    expect(headers.has("x-api-key")).toBe(false)
  }

  it("bootstraps public docs without usable credentials", async () => {
    const config = { serverUrl: "https://api.example.test", authMode: "single-user", apiKey: "" }
    mocks.getConfig.mockResolvedValue(config)
    await getServerCapabilities({ forceRefresh: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/config/docs-info", method: "GET", noAuth: true, configSnapshot: config,
    })
    expectPublicFetch()
  })

  it.each([
    { authMode: "single-user", apiKey: "private-api-key", header: "x-api-key", value: "private-api-key" },
    { authMode: "multi-user", accessToken: "private-access-token", header: "authorization", value: "Bearer private-access-token" },
  ])("authenticates configured $authMode discovery", async ({ header, value, ...auth }) => {
    const config = { serverUrl: "https://api.example.test", ...auth }
    mocks.getConfig.mockResolvedValue(config)
    await getServerCapabilities({ forceRefresh: true })
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/config/docs-info", method: "GET", noAuth: false, configSnapshot: config,
    })
    expect(mocks.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = mocks.fetch.mock.calls[0]
    expect(url).toBe("https://api.example.test/api/v1/config/docs-info")
    expect(new Headers(init?.headers).get(header)).toBe(value)
    expect(init?.credentials ?? "same-origin").not.toBe("include")
  })

  it.each([
    { authMode: "single-user", apiKey: "private-api-key" },
    { authMode: "multi-user", accessToken: "private-access-token" },
  ])("omits configured $authMode credentials for explicitly public requests", async (auth) => {
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://api.example.test", ...auth })
    const result = await tldwRequest({ path: "/api/v1/config/docs-info", method: "GET", noAuth: true }, {
      getConfig: () => mocks.getConfig(), fetchFn: mocks.fetch,
    })
    expect(result.ok).toBe(true)
    expectPublicFetch()
  })
})
